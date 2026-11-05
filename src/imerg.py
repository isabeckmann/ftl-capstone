"""Chuva média diária sobre a bacia, a partir do GPM IMERG Final diário (GPM_3IMERGDF, V07).

Cada arquivo do IMERG é GLOBAL (~117 MB em memória). Por isso o fluxo é:
baixar um mês -> recortar pela bacia e tirar a média -> apagar os brutos -> salvar no CSV.
O CSV é retomável: se o processo for interrompido, rode de novo e ele continua de onde parou.

Uso (no terminal, na raiz do projeto, com o ambiente `capstone` ativo):
    python -m src.imerg --start 2015-01-01 --end 2026-06-30
    python -m src.imerg --start 2015-01-01 --end 2026-06-30 --basin data/raw/bacia_taquari_antas.gpkg

Saída: data/processed/chuva_bacia_diaria.csv  (colunas: data, chuva_bacia_mm  -> mm/dia)

Observações:
- O IMERG usa dia em UTC; os níveis do HidroWeb usam dia local (UTC-3). Registre isso na metodologia.
- Sem o polígono da bacia (--basin), a média é feita sobre a caixa retangular BBOX (diluída; só para testes).
"""
from __future__ import annotations

import argparse
import shutil
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
OUT_CSV = config.PROCESSED / "chuva_bacia_diaria.csv"


def load_basin_geometry(path):
    """Lê um polígono da bacia (gpkg/geojson/shp) e devolve uma geometria única em EPSG:4326."""
    import geopandas as gpd

    g = gpd.read_file(path).to_crs(4326)
    return g.union_all() if hasattr(g, "union_all") else g.unary_union


def basin_mean(path, bbox=BBOX, geom=None, var=VAR):
    """Média da chuva diária (mm/dia) sobre a bacia, num arquivo. Pondera por cos(latitude).

    Devolve (dia, média, nº de células usadas)."""
    w, s, e, n = bbox
    with xr.open_dataset(path) as ds:
        da = ds[var].isel(time=0).sel(lon=slice(w, e), lat=slice(s, n)).transpose("lon", "lat").load()
        day = pd.Timestamp(ds["time"].values[0]).normalize()
    vals = da.values.astype("float64")
    vals[vals < 0] = np.nan 
    lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values, indexing="ij")
    if geom is not None:
        from shapely import contains_xy

        mask = contains_xy(geom, lon2d, lat2d)
    else:
        mask = np.ones(vals.shape, dtype=bool)
    wgt = np.cos(np.deg2rad(lat2d)) * mask * np.isfinite(vals)
    if wgt.sum() == 0:
        return day, np.nan, 0
    return day, float(np.nansum(np.where(np.isfinite(vals), vals, 0.0) * wgt) / wgt.sum()), int(mask.sum())


def _read_done(out_csv: Path) -> pd.Series:
    if out_csv.exists():
        return pd.read_csv(out_csv, parse_dates=["data"], index_col="data")["chuva_bacia_mm"]
    return pd.Series(dtype="float64", name="chuva_bacia_mm")


def build_series(start, end, out_csv=OUT_CSV, bbox=BBOX, geom=None, keep_raw=False, workdir=None, retries=3):
    """Baixa mês a mês, calcula a média da bacia e apaga os brutos. Retomável."""
    import earthaccess

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir or config.RAW / "imerg")
    workdir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()

    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    done = _read_done(out_csv)
    for per in pd.period_range(start, end, freq="M"):
        d0, d1 = max(per.start_time.normalize(), start), min(per.end_time.normalize(), end)
        days = pd.date_range(d0, d1, freq="D")
        if all(d in done.index for d in days):
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
        rows = {}
        for f in files:
            try:
                day, val, ncell = basin_mean(f, bbox=bbox, geom=geom)
                rows[day] = val
            except Exception as exc:  # noqa: BLE001
                print(f"  erro lendo {Path(str(f)).name}: {exc}")
        if rows:
            done = pd.concat([done[~done.index.isin(rows)], pd.Series(rows, name="chuva_bacia_mm")]).sort_index()
            done.index.name = "data"
            done.to_csv(out_csv, header=True)
        if not keep_raw:
            for f in files:
                Path(str(f)).unlink(missing_ok=True)
        falt = [d.date().isoformat() for d in days if d not in done.index]
        print(f"{per}: {len(rows)}/{len(days)} dias em {time.time() - t:.0f}s" + (f" | faltando {falt}" if falt else ""), flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--basin", default=None, help="polígono da bacia (gpkg/geojson/shp)")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    geom = load_basin_geometry(a.basin) if a.basin else None
    if geom is None:
        print("AVISO: sem --basin; usando a caixa retangular BBOX (média diluída, só para testes).")
    s = build_series(a.start, a.end, geom=geom, keep_raw=a.keep_raw)
    print(f"\nPronto: {len(s)} dias em {OUT_CSV}")
