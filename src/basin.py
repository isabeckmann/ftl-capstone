"""Polígonos das áreas de drenagem a montante de cada estação, a partir do HydroBASINS (HydroSHEDS v1c).

O IMERG deve ser promediado sobre a área que de fato drena para a estação, e não sobre uma caixa retangular
(que mistura chuva de bacias vizinhas e dilui o sinal).

Método: entre as sub-bacias do HydroBASINS (nível 12) num raio de `SEARCH_KM` em volta da estação, escolhe a que
tem a área a montante (UP_AREA) mais próxima da área de drenagem informada pela ANA. Depois junta
essa sub-bacia e todas as que escoam para ela (topologia NEXT_DOWN). A diferença de área em relação à ANA
fica registrada na saída e deve ser citada na metodologia.

Dados de entrada (não versionados):
    data/raw/hydrobasins/hybas_sa_lev12_v1c.shp
    https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_sa_lev12_v1c.zip
    Lehner, B., Grill, G. (2013). Hydrological Processes, 27(15): 2171-2186.

Uso:
    python -m src.basin                     # gera data/processed/bacias_estacoes.gpkg
    python -m src.imerg --series-only --basin data/processed/bacias_estacoes.gpkg --station 86720000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config

HYBAS = config.RAW / "hydrobasins" / "hybas_sa_lev12_v1c.shp"
OUT_GPKG = config.PROCESSED / "bacias_estacoes.gpkg"
SEARCH_KM = 5.0
READ_BBOX = (-53.5, -30.5, -49.5, -27.5)      # lê só a região do Taquari-Antas (o arquivo cobre a América do Sul)

# Coordenadas e áreas de drenagem do inventário da ANA (HidroInventarioEstacoes, consultado em 11/2026).
STATION_META = {
    "86472000": {"lat": -29.0978, "lon": -51.6997, "area_ana_km2": 13000.0},
    "86472600": {"lat": -29.1781, "lon": -51.7322, "area_ana_km2": 15775.0},
    "86510000": {"lat": -29.1672, "lon": -51.8686, "area_ana_km2": 16000.0},
    "86720000": {"lat": -29.2344, "lon": -51.8550, "area_ana_km2": 19100.0},
}


def load_hybas(path=HYBAS, bbox=READ_BBOX):
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sem {path}. Baixe e descompacte o HydroBASINS nível 12 da América do Sul "
                                "(veja o cabeçalho de src/basin.py).")
    return gpd.read_file(path, bbox=bbox)


def _upstream_ids(hb, outlet_id: int) -> list:
    """IDs da sub-bacia `outlet_id` e de todas que drenam para ela (busca em largura sobre NEXT_DOWN)."""
    children = hb.groupby("NEXT_DOWN")["HYBAS_ID"].apply(list).to_dict()
    out, stack = [], [outlet_id]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(children.get(i, []))
    return out


def upstream_polygon(hb, lon: float, lat: float, area_ref_km2: float | None, search_km: float = SEARCH_KM) -> dict:
    """Área a montante de um ponto. Devolve dict com geometria, sub-bacia exutória e áreas (km²)."""
    import geopandas as gpd
    from shapely.geometry import Point

    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326)
    utm = pt.estimate_utm_crs()
    hb_m = hb.to_crs(utm)
    buf = pt.to_crs(utm).buffer(search_km * 1000).iloc[0]
    cand = hb_m[hb_m.intersects(buf)].copy()
    if cand.empty:
        raise ValueError(f"Nenhuma sub-bacia a {search_km} km de ({lat}, {lon}).")
    if area_ref_km2:
        cand["dif"] = (cand["UP_AREA"] - area_ref_km2).abs()
    else:                                        # sem área de referência: a sub-bacia que contém o ponto
        cand["dif"] = (~cand.contains(pt.to_crs(utm).iloc[0])).astype(float)
    outlet = cand.sort_values("dif").iloc[0]
    ids = _upstream_ids(hb, int(outlet["HYBAS_ID"]))
    sel = hb_m[hb_m["HYBAS_ID"].isin(ids)]
    geom_m = sel.union_all() if hasattr(sel, "union_all") else sel.unary_union
    geom = gpd.GeoSeries([geom_m], crs=utm).to_crs(4326).iloc[0]
    return {"geometry": geom, "hybas_exutorio": int(outlet["HYBAS_ID"]), "n_sub_bacias": len(ids),
            "area_km2_poligono": float(geom_m.area / 1e6), "area_km2_hybas": float(outlet["UP_AREA"]),
            "area_km2_ana": area_ref_km2,
            "dist_exutorio_km": float(cand.loc[outlet.name].geometry.distance(pt.to_crs(utm).iloc[0]) / 1000)}


def build_station_basins(stations=None, hybas_path=HYBAS, out=OUT_GPKG):
    import geopandas as gpd

    stations = stations or list(STATION_META)
    hb = load_hybas(hybas_path)
    rows = []
    for code in stations:
        m = STATION_META[code]
        r = upstream_polygon(hb, m["lon"], m["lat"], m.get("area_ana_km2"))
        r.update(codigo=code, nome=config.STATIONS.get(code, {}).get("nome", code))
        rows.append(r)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    gdf["dif_area_%"] = 100 * (gdf["area_km2_poligono"] - gdf["area_km2_ana"]) / gdf["area_km2_ana"]
    cols = ["codigo", "nome", "area_km2_ana", "area_km2_poligono", "dif_area_%", "area_km2_hybas",
            "hybas_exutorio", "n_sub_bacias", "dist_exutorio_km", "geometry"]
    gdf = gdf[cols]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")
    return gdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_GPKG))
    a = ap.parse_args()
    g = build_station_basins(a.stations, out=a.out)
    with pd.option_context("display.width", 160):
        print(g.drop(columns="geometry").round(1).to_string(index=False))
    print(f"\nlimites (oeste, sul, leste, norte): {tuple(round(v, 2) for v in g.total_bounds)}")
    print(f"Pronto: {a.out}")
