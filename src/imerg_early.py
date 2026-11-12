"""Chuva HORÁRIA do IMERG Early (GPM_3IMERGHHE V07, meia-hora, 0,1°) sobre a bacia e sub-bacias do Taquari.

Por que o Early: o IMERG Final diário chega com ~3,5 meses de atraso e, dentro do modelo, com até 1 dia de
defasagem; não serve para previsão de 6–12 h nem para operação ao vivo. O Early é publicado ~4 h depois da
observação, com passo de 30 min. No treino aplicamos a mesma defasagem (config.IMERG_LATENCIA_H), então o
modelo só "vê" o que estaria disponível em tempo real.

Download: só o recorte da caixa BBOX de cada arquivo, via OPeNDAP (alguns KB por arquivo). Retomável: cada
mês vira um NetCDF em data/processed/imerg_early/; meses completos são pulados.

Credenciais do Earthdata em ~/.netrc:
    machine urs.earthdata.nasa.gov login SEU_USUARIO password SUA_SENHA
(e o aplicativo "NASA GESDISC DATA ARCHIVE" autorizado no perfil do Earthdata).

Uso:
    python -m src.imerg_early baixar --start 2018-05-01 --end 2026-06-30 [--workers 12]
    python -m src.imerg_early series          # gera data/processed/chuva_imerg_early_horaria.csv
"""
from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import xarray as xr

from . import config

COLLECTION = "C2723758340-GES_DISC"          # GPM_3IMERGHHE V07
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
OPENDAP = "https://opendap.earthdata.nasa.gov/collections/{coll}/granules/{gid}"
BBOX = (-52.7, -30.0, -49.9, -28.1)          # mesma caixa do IMERG Final (src/imerg.py)
GRID_DIR = config.PROCESSED / "imerg_early"
OUT_CSV = config.PROCESSED / "chuva_imerg_early_horaria.csv"
BASINS = config.PROCESSED / "bacias_estacoes.gpkg"


def _idx(bbox=BBOX):
    """Índices da grade global 0,1° (lon -179,95..179,95; lat -89,95..89,95) que cobrem a caixa."""
    w, s, e, n = bbox
    i0, i1 = int(np.floor((w + 180) / 0.1)), int(np.ceil((e + 180) / 0.1)) - 1
    j0, j1 = int(np.floor((s + 90) / 0.1)), int(np.ceil((n + 90) / 0.1)) - 1
    return i0, i1, j0, j1


def _ce(bbox=BBOX) -> str:
    i0, i1, j0, j1 = _idx(bbox)
    return ";".join([f"/Grid/precipitation[0:1:0][{i0}:1:{i1}][{j0}:1:{j1}]",
                     f"/Grid/lon[{i0}:1:{i1}]", f"/Grid/lat[{j0}:1:{j1}]", "/Grid/time"])


# ----------------------------------------------------------------------------- catálogo
def list_granules(start, end, session=None) -> pd.DataFrame:
    """Granulos (título, início em UTC) entre start e end, via CMR (não exige login)."""
    s = session or requests.Session()
    rows, page = [], 1
    while True:
        r = s.get(CMR, params={"collection_concept_id": COLLECTION, "page_size": 2000, "page_num": page,
                               "temporal": f"{pd.Timestamp(start):%Y-%m-%dT%H:%M:%SZ},{pd.Timestamp(end):%Y-%m-%dT%H:%M:%SZ}",
                               "sort_key": "start_date"}, timeout=60)
        r.raise_for_status()
        ent = r.json()["feed"]["entry"]
        rows += [{"gid": e["title"], "t": pd.Timestamp(e["time_start"]).tz_localize(None)} for e in ent]
        if len(ent) < 2000:
            break
        page += 1
    return pd.DataFrame(rows).drop_duplicates("t").sort_values("t").reset_index(drop=True)


# ----------------------------------------------------------------------------- download
def fetch_subset(gid: str, session: requests.Session, retries: int = 4) -> xr.DataArray:
    """Recorte (lat, lon) de um granulo, em mm/h."""
    url = OPENDAP.format(coll=COLLECTION, gid=quote(gid, safe="")) + ".dap.nc4?dap4.ce=" + quote(_ce(), safe="/[]:;")
    last = None
    for k in range(retries):
        try:
            r = session.get(url, timeout=90)
            if r.status_code == 401:
                raise PermissionError("Earthdata recusou o login (HTTP 401). Confira ~/.netrc e a autorização do GES DISC.")
            r.raise_for_status()
            import h5py

            with h5py.File(io.BytesIO(r.content), "r") as f:          # variáveis ficam no grupo /Grid
                p = np.asarray(f["Grid/precipitation"][...], dtype="float32")[0]   # (lon, lat)
                lon = np.round(np.asarray(f["Grid/lon"][...], dtype="float64"), 2)
                lat = np.round(np.asarray(f["Grid/lat"][...], dtype="float64"), 2)
            da = xr.DataArray(p.T, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
            return da.where(da >= 0)
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"falhou {gid}: {last}")


def _session() -> requests.Session:
    s = requests.Session()              # usa ~/.netrc automaticamente (requests lê o netrc)
    s.headers["User-Agent"] = "capstone-taquari/1.0"
    return s


def download(start, end, workers: int = 12, grid_dir=GRID_DIR) -> None:
    grid_dir = Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    i0, i1, j0, j1 = _idx()
    lon = np.round(-179.95 + 0.1 * np.arange(i0, i1 + 1), 2)
    lat = np.round(-89.95 + 0.1 * np.arange(j0, j1 + 1), 2)
    for per in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        out = grid_dir / f"imerg_early_{per}.nc"
        t0 = max(per.start_time, pd.Timestamp(start))
        t1 = min(per.end_time, pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59))
        gr = list_granules(t0, t1)
        have = {}
        if out.exists():                      # retoma: só baixa o que falta
            with xr.open_dataset(out) as ds:
                old = ds["precipitation"].load()
            have = {pd.Timestamp(t): old.sel(time=t) for t in old["time"].values
                    if np.isfinite(old.sel(time=t).values).any()}
        todo = gr[~gr["t"].isin(list(have))]
        if todo.empty:
            continue
        tic, fails = time.time(), []
        sess = _session()
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_subset, row.gid, sess): row.t for row in todo.itertuples()}
            for f in as_completed(futs):
                try:
                    da = f.result()
                    have[futs[f]] = xr.DataArray(da.values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
                except PermissionError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    fails.append((futs[f], str(exc)[:120]))
        times = sorted(have)
        cube = xr.concat([have[t] for t in times], dim=pd.DatetimeIndex(times, name="time"))
        ds = cube.to_dataset(name="precipitation")
        ds["precipitation"].attrs.update(units="mm/h", produto="GPM_3IMERGHHE V07 (Early)", tempo="início do intervalo de 30 min, UTC")
        tmp = out.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp, encoding={"precipitation": {"zlib": True, "complevel": 4, "dtype": "float32"}})
        tmp.replace(out)
        print(f"{per}: {len(times)}/{len(gr)} meias-horas em {time.time() - tic:.0f}s"
              + (f" | {len(fails)} falhas (rode de novo para tentar)" if fails else ""), flush=True)


# ----------------------------------------------------------------------------- séries por polígono
def subbasin_geoms(path=BASINS) -> dict:
    """Bacia inteira (Encantado) e três sub-bacias sem sobreposição:
    antas = a montante de Linha José Júlio; taquari_sup = Muçum − Antas; guapore = Encantado − Muçum."""
    import geopandas as gpd

    g = gpd.read_file(path).set_index("codigo").to_crs(4326).geometry
    enc, muc, ljj = g["86720000"], g["86510000"], g["86472000"]
    return {"bacia": enc, "antas": ljj, "taquari_sup": muc.difference(ljj), "guapore": enc.difference(muc)}


def load_cube(grid_dir=GRID_DIR) -> xr.DataArray:
    files = sorted(Path(grid_dir).glob("imerg_early_*.nc"))
    if not files:
        raise FileNotFoundError(f"Sem arquivos em {grid_dir}. Rode: python -m src.imerg_early baixar ...")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds["precipitation"].load())
    da = xr.concat(parts, dim="time").sortby("time")
    return da.sel(time=~da.indexes["time"].duplicated())


def cell_weights(geoms: dict, lon, lat) -> dict:
    """Peso de cada célula (lat, lon) por polígono: cos(lat) se o centro está dentro, 0 fora."""
    from shapely import contains_xy

    lon2d, lat2d = np.meshgrid(np.asarray(lon), np.asarray(lat))
    return {name: np.cos(np.deg2rad(lat2d)) * contains_xy(geom, lon2d, lat2d) for name, geom in geoms.items()}


def hourly_series(da: xr.DataArray, geoms: dict | None = None, weights: dict | None = None) -> pd.DataFrame:
    """Chuva horária (mm) por polígono, média ponderada por cos(lat) das células com centro dentro.
    Índice = FIM da hora em UTC (a hora [h, h+1) fica em h+1), soma das duas meias-horas (mm/h × 0,5 h).
    `weights` (de cell_weights) dispensa o geopandas/shapely — usado na rotina ao vivo."""
    if weights is None:
        weights = cell_weights(geoms, da["lon"].values, da["lat"].values)
    vals = da.transpose("time", "lat", "lon").values.astype("float64")
    ok = np.isfinite(vals)
    out = {}
    for name, w in weights.items():
        w = np.asarray(w, dtype="float64")
        wt = w[None] * ok
        den = wt.sum(axis=(1, 2))
        num = (np.where(ok, vals, 0.0) * wt).sum(axis=(1, 2))
        frac = den / max(w.sum(), 1e-9)
        out[name] = np.where(frac >= 0.8, num / np.where(den > 0, den, 1), np.nan) * 0.5    # mm na meia-hora
    hh = pd.DataFrame(out, index=pd.DatetimeIndex(da["time"].values))
    hourly = hh.resample("1h").sum(min_count=2)                     # exige as duas meias-horas
    hourly.index = hourly.index + pd.Timedelta(hours=1)             # rótulo = fim da hora
    hourly.index.name = "fim_hora_utc"
    return hourly


def build_series(out_csv=OUT_CSV) -> pd.DataFrame:
    s = hourly_series(load_cube(), subbasin_geoms())
    s.round(3).to_csv(out_csv)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baixar")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--workers", type=int, default=12)
    sub.add_parser("series")
    a = ap.parse_args()
    if a.cmd == "baixar":
        download(a.start, a.end, a.workers)
    else:
        s = build_series()
        print(s.describe().round(2).to_string())
        print(f"Pronto: {OUT_CSV}")
