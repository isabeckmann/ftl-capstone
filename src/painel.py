"""Gera o painel interativo (HTML único, sem servidor) com bacias, chuva IMERG, eventos e métricas.

Uso (na raiz do projeto, depois dos notebooks 03-05):
    python -m src.painel                      # cenário com_imerg
    python -m src.painel --tag sem_imerg

Saídas:
    web/index.html                         site estático (deploy no Vercel, pasta web/)
    outputs/painel/painel_taquari.html     mesma página, sem o cabeçalho HTML (para visualizadores embutidos)
O template fica em src/painel_template.html.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, imerg, metrics, pipeline

TEMPLATE = Path(__file__).with_name("painel_template.html")
OUT = config.OUTPUTS / "painel" / "painel_taquari.html"
WEB = config.ROOT / "web" / "index.html"
_HEAD = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Previsão de curto prazo da cota do rio Taquari (Muçum e Encantado) com dados da ANA e chuva de satélite IMERG.">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""
MODELOS = ["persistencia", "ridge", "hgb", "mlp", "gru"]


def _r(x, nd=1):
    """Lista JSON-friendly: NaN -> None, arredondada."""
    a = np.asarray(x, dtype="float64")
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def _basins(tol=0.004):
    import geopandas as gpd

    g = gpd.read_file(config.PROCESSED / "bacias_estacoes.gpkg")
    g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    feats = json.loads(g.to_json(drop_id=True))["features"]

    def rnd(c):
        return [rnd(v) for v in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]

    for f in feats:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
        f["properties"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in f["properties"].items()
                           if k in ("codigo", "nome", "area_km2_ana", "area_km2_poligono", "dif_area_%")}
    from .basin import STATION_META
    stations = [{"codigo": c, "nome": config.STATIONS[c]["nome"], "papel": config.STATIONS[c]["papel"],
                 "lat": m["lat"], "lon": m["lon"], "area_ana": m["area_ana_km2"]} for c, m in STATION_META.items()]
    return {"type": "FeatureCollection", "features": feats}, stations


def _rain_grid(geom):
    from shapely import contains_xy

    da = imerg.load_grids()
    lon, lat = da["lon"].values, da["lat"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    inside = contains_xy(geom, lon2d, lat2d)
    t = pd.DatetimeIndex(da["time"].values)
    layers = {}
    full_years = (t >= "2015-01-01") & (t < "2025-01-01")
    layers["media_anual"] = {"rotulo": "Média anual 2015–2024", "unidade": "mm/ano",
                             "v": _r(da.isel(time=np.where(full_years)[0]).mean("time").values.ravel() * 365.25, 0)}
    for ev, (a, b) in config.EVENTS.items():
        a0 = pd.Timestamp(a) - pd.Timedelta(days=4)            # inclui a chuva que antecede a subida
        sel = (t >= a0) & (t <= pd.Timestamp(b))
        layers[ev] = {"rotulo": f"Acumulado {a0:%d/%m} a {pd.Timestamp(b):%d/%m/%Y}", "unidade": "mm",
                      "v": _r(da.isel(time=np.where(sel)[0]).sum("time").values.ravel(), 0)}
    return {"lon": _r(lon, 2), "lat": _r(lat, 2), "dentro": inside.ravel().astype(int).tolist(), "camadas": layers}


def _events(tag, rain_daily, rain_hourly=None):
    out = {}
    for alvo in config.TARGETS:
        preds = {m: pipeline.load_predictions(m, alvo, tag) for m in MODELOS}
        for ev, (a, b) in config.EVENTS.items():
            ini = pd.Timestamp(a) - pd.Timedelta(days=2)
            fim = pd.Timestamp(b) + pd.Timedelta(hours=23)
            idx = pd.date_range(ini, fim, freq="h")
            hz = {}
            for h in config.HORIZONS:
                ref = preds["persistencia"]
                obs = metrics.as_series(ref.index, ref[f"obs_{h}h"], h)
                obs = obs[~obs.index.duplicated()].reindex(idx)
                mods = {}
                for m, df in preds.items():
                    p = metrics.as_series(df.index, df[f"pred_{h}h"], h)
                    mods[m] = _r(p[~p.index.duplicated()].reindex(idx), 0)
                    col = f"pred_sem_limite_{h}h"
                    if m != "persistencia" and col in df:
                        q = metrics.as_series(df.index, df[col], h)
                        mods[m + "__livre"] = _r(q[~q.index.duplicated()].reindex(idx), 0)
                hz[str(h)] = {"obs": _r(obs, 0), "modelos": mods}
            if rain_hourly is not None:          # IMERG Early horária, já em hora local (início da hora)
                r = rain_hourly.reindex(idx)
                chuva = {"inicio": idx[0].strftime("%Y-%m-%dT%H:00"), "passo_h": 1, "v": _r(r.values, 1)}
            else:
                r = rain_daily.reindex(pd.date_range(ini.normalize(), fim.normalize(), freq="D"))
                chuva = {"inicio": r.index[0].strftime("%Y-%m-%d"), "passo_h": 24, "v": _r(r.values, 1)}
            out[f"{alvo}|{ev}"] = {"inicio": idx[0].strftime("%Y-%m-%dT%H:00"), "n": len(idx),
                                   "janela": [a, b], "horizontes": hz, "chuva": chuva}
    return out


def _rain_hourly_local():
    """IMERG Early na bacia inteira, reindexada para o relógio do painel (hora local, rótulo = início da hora).
    Sem defasagem de disponibilidade: aqui é para mostrar quando choveu, não o que o modelo via."""
    from . import imerg_early
    if not imerg_early.OUT_CSV.exists():
        return None
    s = pd.read_csv(imerg_early.OUT_CSV, parse_dates=["fim_hora_utc"], index_col="fim_hora_utc")["bacia"]
    s.index = s.index - pd.Timedelta(hours=1) + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS)
    return s


def _long_series():
    panel = pd.read_parquet(config.PROCESSED / "panel_hourly.parquet").loc[getattr(config, "DATA_START", "2018-05-01"):]
    d = panel[[f"cota_{c}" for c in config.TARGETS]].resample("D").max()
    cov = panel[[f"cota_{c}" for c in config.TARGETS]].notna().resample("D").mean()
    d = d.where(cov >= 0.25)
    return {"inicio": d.index[0].strftime("%Y-%m-%d"),
            "series": {c: _r(d[f"cota_{c}"].values, 0) for c in config.TARGETS},
            "treino_fim": config.TRAIN_END[:10], "val_fim": config.VAL_END[:10], "teste_ini": config.TEST_START[:10]}


def build(tag="com_imerg", out=OUT, web=WEB):
    basins, stations = _basins()
    geom = imerg.load_basin_geometry(config.PROCESSED / "bacias_estacoes.gpkg", config.TARGETS[-1])
    rain_daily = pd.read_csv(config.PROCESSED / "chuva_bacia_diaria.csv", parse_dates=["data"],
                             index_col="data")["chuva_bacia_mm"]
    rain_hourly = _rain_hourly_local()
    ev_stats = pd.read_csv(config.OUTPUTS / f"resultados_eventos_{tag}.csv")
    teste = pd.read_csv(config.OUTPUTS / f"resultados_teste_{tag}.csv")
    efeito = pd.read_csv(config.OUTPUTS / f"efeito_limites_{tag}.csv")
    payload = {
        "cenario": tag, "margem": getattr(config, "CLIP_MARGIN", None),
        "estacoes": stations, "bacias": basins, "chuva_grade": _rain_grid(geom),
        "eventos": _events(tag, rain_daily, rain_hourly),
        "eventos_stats": json.loads(ev_stats.round(1).to_json(orient="records")),
        "teste": json.loads(teste.round(3).to_json(orient="records")),
        "efeito": json.loads(efeito.round(1).to_json(orient="records")),
        "serie_longa": _long_series(),
        "eventos_janelas": config.EVENTS, "horizontes": config.HORIZONS,
        "ao_vivo": {"urls": [config.AO_VIVO_URL] if getattr(config, "AO_VIVO_URL", None) else [],
                    "utc_offset_h": config.LOCAL_UTC_OFFSET_HOURS},
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DADOS__*/null",
                                                         json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if web:                                    # documento completo, sem metadados de visualizador embutido
        body = "\n".join(l for l in html.splitlines() if 'name="claude:' not in l)
        web = Path(web)
        web.parent.mkdir(parents=True, exist_ok=True)
        web.write_text(_HEAD + body.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1)
                       + "\n</body>\n</html>\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="com_imerg")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    p = build(a.tag, a.out)
    print(f"Pronto: {p} ({p.stat().st_size / 1e6:.1f} MB) e {WEB}")
