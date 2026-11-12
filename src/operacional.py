"""Modelo operacional (usado pela rotina ao vivo): gradient boosting com IMERG Early, treinado com TODO o histórico.

Os notebooks treinam até 2021 para medir o desempenho em 2023–2026. Para operar, o modelo deve ver também as cheias
de 2023–2024 e o período depois da mudança de régua de Encantado (2024). Antes de salvar, um teste de checagem
treina até --corte e avalia depois dele, para conferir que a receita continua boa com dados recentes.

Tudo o que a rotina ao vivo precisa fica em operacional/ (versionado, sem dados brutos):
    modelo_hgb.joblib     modelos por estação e horizonte, lista de variáveis, limites de entrada/saída
    pesos_imerg.json      grade IMERG da caixa e o peso de cada célula em cada sub-bacia (dispensa geopandas)

Uso:
    python -m src.operacional                 # checagem (corte 2025-06-30) + treino final + pesos
    python -m src.operacional --corte 2025-01-01
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features, metrics, pipeline

DIR = config.ROOT / "operacional"
MODELO = DIR / "modelo_hgb.joblib"
PESOS = DIR / "pesos_imerg.json"
HGB_PARAMS = dict(max_depth=6, learning_rate=0.05, max_iter=300, early_stopping=False,
                  random_state=config.RANDOM_STATE)


def _fit(P, rows):
    from sklearn.ensemble import HistGradientBoostingRegressor

    gbs = []
    for j in range(len(P.horizons)):
        gb = HistGradientBoostingRegressor(**HGB_PARAMS)
        gb.fit(P.X[rows], P.Y[rows][:, j])
        gbs.append(gb)
    return gbs, features.Bounds.fit(P.X[rows]), features.Bounds.fit(P.Y[rows])


def predict(gbs, bin_, bout, X):
    """Previsão da VARIAÇÃO da cota (N, H), com entradas e saída limitadas à faixa do treino."""
    Xc = bin_.clip(np.asarray(X, dtype="float32"))
    return bout.clip(np.column_stack([gb.predict(Xc) for gb in gbs]).astype("float32"))


def checagem(panel, corte):
    """Treina até `corte` e avalia depois (com a mesma folga de max(horizonte) entre os dois)."""
    corte = pd.Timestamp(corte)
    gap = pd.Timedelta(hours=max(config.HORIZONS))
    linhas = []
    for alvo in config.TARGETS:
        P = features.build_tabular(panel, alvo)
        full = ~np.isnan(P.Y).any(axis=1)
        tr = np.asarray(P.index <= corte - gap) & full
        te = np.asarray(P.index > corte)
        gbs, bi, bo = _fit(P, tr)
        pred = P.base[:, None] + predict(gbs, bi, bo, P.X)
        tab = metrics.summary_table(P.Yabs[te], pred[te], P.base[te], P.horizons).reset_index()
        tab.insert(0, "estacao", alvo)
        linhas.append(tab)
    return pd.concat(linhas, ignore_index=True)


def treinar(panel):
    import sklearn

    out = {"horizontes": list(config.HORIZONS), "latencia_imerg_h": config.IMERG_LATENCIA_H,
           "margem": config.CLIP_MARGIN, "sklearn": sklearn.__version__, "estacoes": {}}
    for alvo in config.TARGETS:
        P = features.build_tabular(panel, alvo)
        rows = ~np.isnan(P.Y).any(axis=1)
        gbs, bi, bo = _fit(P, rows)
        out["estacoes"][alvo] = {"hgb": gbs, "features": P.feature_names, "limites_entrada": bi, "limites_saida": bo,
                                 "treino_ini": str(P.index[rows][0]), "treino_fim": str(P.index[rows][-1]),
                                 "n": int(rows.sum())}
        print(f"[operacional] {alvo}: {rows.sum()} amostras, {P.index[rows][0]:%Y-%m-%d} a {P.index[rows][-1]:%Y-%m-%d}, "
              f"{len(P.feature_names)} variáveis")
    return out


def salvar_pesos():
    from . import imerg_early as ie

    i0, i1, j0, j1 = ie._idx()
    lon = np.round(-179.95 + 0.1 * np.arange(i0, i1 + 1), 2)
    lat = np.round(-89.95 + 0.1 * np.arange(j0, j1 + 1), 2)
    w = ie.cell_weights(ie.subbasin_geoms(), lon, lat)
    PESOS.write_text(json.dumps({"bbox": ie.BBOX, "lon": lon.tolist(), "lat": lat.tolist(),
                                 "pesos": {k: np.round(v, 6).tolist() for k, v in w.items()}}), encoding="utf-8")


if __name__ == "__main__":
    import joblib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corte", default="2025-06-30")
    a = ap.parse_args()
    os.environ["CAPSTONE_IMERG"] = "1"
    DIR.mkdir(exist_ok=True)
    panel = pipeline.load_panel()
    chk = checagem(panel, a.corte)
    print(f"\nChecagem: treino até {a.corte}, teste depois\n" + chk.round(3).to_string(index=False))
    chk.round(4).to_csv(DIR / "checagem.csv", index=False)
    joblib.dump(treinar(panel), MODELO, compress=3)
    salvar_pesos()
    print(f"\nSalvo: {MODELO} ({MODELO.stat().st_size / 1e6:.1f} MB) e {PESOS}")
