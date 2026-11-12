"""Chuva diária da NASA GPM IMERG Final (GPM_3IMERGDF, V07) para a bacia do Taquari-Antas.

Cada arquivo do IMERG é GLOBAL (~117 MB em memória). O fluxo é:
  baixar um mês -> RECORTAR a caixa da região e SALVAR a grade (arquivo pequeno) -> apagar o bruto.
Como a grade recortada fica guardada, o polígono da bacia (ou das áreas a montante de cada estação)
pode ser aplicado DEPOIS, sem baixar nada de novo.

Uso (no terminal, na raiz do projeto, com o ambiente `capstone` ativo):
    # 1) baixar e guardar as grades (retomável: se interromper, rode de novo)
    python -m src.imerg --start 2015-01-01 --end 2026-06-30

    # 2) gerar o CSV com a média sobre a área a montante de uma estação (sem baixar nada)
    python -m src.basin          # cria data/processed/bacias_estacoes.gpkg (HydroBASINS)
    python -m src.imerg --series-only --basin data/processed/bacias_estacoes.gpkg --station 86720000

Saídas:
    data/processed/imerg_caixa/imerg_AAAA-MM.nc   grades recortadas (mm/dia)
    data/processed/chuva_bacia_diaria.csv         média diária (colunas: data, chuva_bacia_mm)

Observações:
- O IMERG usa dia em UTC; os níveis do HidroWeb usam dia local (UTC-3). Registre isso na metodologia.
- Sem o polígono, a média é sobre a caixa retangular BBOX (diluída; só para testes).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

SHORT_NAME = "GPM_3IMERGDF"
VERSION = "07"
VAR = "precipitation"                    # mm/dia (média diária da taxa de chuva)
BBOX = (-52.7, -30.0, -49.9, -28.1)      # (oeste, sul, leste, norte) com folga sobre o Taquari-Antas
GRID_DIR = config.PROCESSED / "imerg_caixa"
OUT_CSV = config.PROCESSED / "chuva_bacia_diaria.csv"


# ----------------------------------------------------------------------------- leitura / recorte
def load_basin_geometry(path, station: str | None = None):
    """Lê um polígono (gpkg/geojson/shp) e devolve uma geometria única em EPSG:4326.

    Com `station`, usa só a linha cuja coluna `codigo` é essa estação (arquivo gerado por src.basin)."""
    import geopandas as gpd

    g = gpd.read_file(path).to_crs(4326)
    if station is not None:
        g = g[g["codigo"].astype(str) == str(station)]
        if g.empty:
            raise ValueError(f"Estação {station} não encontrada em {path}")
    return g.union_all() if hasattr(g, "union_all") else g.unary_union


def _read_clip(path, bbox=BBOX, var=VAR):
    """Abre um arquivo global do IMERG e devolve (dia, grade recortada (lat, lon) com NaN nos vazios)."""
    w, s, e, n = bbox
    with xr.open_dataset(path) as ds:
        da = ds[var].isel(time=0).sel(lon=slice(w, e), lat=slice(s, n)).transpose("lat", "lon").load()
        day = pd.Timestamp(ds["time"].values[0]).normalize()
    da = da.drop_vars("time", errors="ignore").where(da >= 0)      # -9999.9 (preenchimento) vira NaN
    return day, da


def save_month_grid(files, out_nc, bbox=BBOX):
    """Junta os dias de um mês (grade recortada) num NetCDF compacto."""
    parts = {}
    for f in files:
        day, da = _read_clip(f, bbox)
        parts[day] = da
    days = sorted(parts)
    da = xr.concat([parts[d] for d in days], dim=pd.DatetimeIndex(days, name="time"))
    ds = da.to_dataset(name="precipitation")
    ds["precipitation"].attrs["units"] = "mm/day"
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc, encoding={"precipitation": {"zlib": True, "complevel": 4}})
    return days


def _month_complete(nc: Path, days) -> bool:
    if not nc.exists():
        return False
    try:
        with xr.open_dataset(nc) as ds:
            have = set(pd.DatetimeIndex(ds["time"].values).normalize())
        return all(d in have for d in days)
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- médias sobre polígonos
def load_grids(grid_dir=GRID_DIR) -> xr.DataArray:
    """Todas as grades mensais guardadas, concatenadas no tempo: DataArray (time, lat, lon)."""
    files = sorted(Path(grid_dir).glob("imerg_*.nc"))
    if not files:
        raise FileNotFoundError(f"Sem grades em {grid_dir}. Rode: python -m src.imerg --start ... --end ...")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds["precipitation"].load())
    da = xr.concat(parts, dim="time").sortby("time")
    return da.sel(time=~da.indexes["time"].duplicated())


def mean_over(da: xr.DataArray, geom=None) -> pd.Series:
    """Média diária (mm/dia) sobre um polígono (EPSG:4326), ponderada por cos(latitude).

    Usa as células cujo CENTRO cai dentro do polígono. geom=None usa a caixa inteira."""
    lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values)          # (lat, lon)
    if geom is not None:
        from shapely import contains_xy

        mask = contains_xy(geom, lon2d, lat2d)
    else:
        mask = np.ones(lon2d.shape, dtype=bool)
    vals = da.transpose("time", "lat", "lon").values.astype("float64")
    wgt = np.cos(np.deg2rad(lat2d))[None] * mask[None] * np.isfinite(vals)
    den = wgt.sum(axis=(1, 2))
    num = (np.where(np.isfinite(vals), vals, 0.0) * wgt).sum(axis=(1, 2))
    mean = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return pd.Series(mean, index=pd.DatetimeIndex(da["time"].values), name="chuva_bacia_mm")


def mean_over_polygons(geoms: dict, grid_dir=GRID_DIR) -> pd.DataFrame:
    """Uma coluna por polígono (ex.: {'a_montante_mucum': geom1, 'a_montante_encantado': geom2})."""
    da = load_grids(grid_dir)
    return pd.DataFrame({name: mean_over(da, g) for name, g in geoms.items()})


def basin_mean(path, bbox=BBOX, geom=None, var=VAR):
    """Média sobre a bacia em UM arquivo global (útil para testes rápidos). Devolve (dia, média, nº de células)."""
    day, da = _read_clip(path, bbox, var)
    da = da.expand_dims(time=[day.to_datetime64()])
    val = float(mean_over(da, geom).iloc[0])
    n = int(np.isfinite(da.values[0]).sum()) if geom is None else None
    if geom is not None:
        from shapely import contains_xy

        lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values)
        n = int(contains_xy(geom, lon2d, lat2d).sum())
    return day, val, n


# ----------------------------------------------------------------------------- download mensal
def _write_csv(series: pd.Series, out_csv: Path) -> None:
    series = series.rename("chuva_bacia_mm")
    series.index.name = "data"
    series.to_csv(out_csv, header=True)


def build_series(start, end, out_csv=OUT_CSV, bbox=BBOX, geom=None, keep_raw=False, workdir=None,
                 grid_dir=GRID_DIR, retries=3):
    """Baixa mês a mês, guarda a grade recortada, atualiza o CSV e apaga os brutos. Retomável."""
    import earthaccess

    out_csv, grid_dir = Path(out_csv), Path(grid_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir or config.RAW / "imerg")
    workdir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()

    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    for per in pd.period_range(start, end, freq="M"):
        d0, d1 = max(per.start_time.normalize(), start), min(per.end_time.normalize(), end)
        days = pd.date_range(d0, d1, freq="D")
        nc = grid_dir / f"imerg_{per}.nc"
        if _month_complete(nc, days):
            continue
        t = time.time()
        files = []
        for attempt in range(1, retries + 1):
            try:
                res = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION,
                                              temporal=(d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")),
                                              bounding_box=bbox)
                files = earthaccess.download(res, str(workdir))
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  tentativa {attempt}/{retries} falhou: {exc}")
                time.sleep(5 * attempt)
        got = []
        if files:
            try:
                got = save_month_grid(files, nc, bbox)
            except Exception as exc:  # noqa: BLE001
                print(f"  erro ao processar {per}: {exc}")
        if not keep_raw:
            for f in files:
                Path(str(f)).unlink(missing_ok=True)
        falt = [d.date().isoformat() for d in days if d not in set(got)]
        print(f"{per}: {len(got)}/{len(days)} dias em {time.time() - t:.0f}s" + (f" | faltando {falt}" if falt else ""), flush=True)

    # CSV com a média sobre o polígono (ou a caixa) a partir de todas as grades guardadas
    s = mean_over(load_grids(grid_dir), geom)
    _write_csv(s, out_csv)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--basin", default=None, help="polígono da bacia (gpkg/geojson/shp)")
    ap.add_argument("--station", default=None, help="código da estação dentro do --basin (coluna 'codigo')")
    ap.add_argument("--series-only", action="store_true", help="não baixa; só recalcula o CSV a partir das grades")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    geom = load_basin_geometry(a.basin, a.station) if a.basin else None
    if geom is None:
        print("AVISO: sem --basin; usando a caixa retangular BBOX (média diluída, só para testes).")
    if a.series_only:
        s = mean_over(load_grids(), geom)
        _write_csv(s, OUT_CSV)
    else:
        if not (a.start and a.end):
            ap.error("informe --start e --end (ou use --series-only)")
        s = build_series(a.start, a.end, geom=geom, keep_raw=a.keep_raw)
    print(f"\nPronto: {len(s)} dias em {OUT_CSV}")
