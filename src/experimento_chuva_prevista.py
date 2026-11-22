"""Experimento: a chuva PREVISTA melhora a previsão da cota em 6, 8 e 12 h?

Mesmo modelo da rotina ao vivo (gradient boosting, src.operacional), mesma divisão para todos os cenários:
treino até 2022-12-31 (menos a folga de 12 h), teste de 2023-01-01 em diante (cheias de set/2023, nov/2023, mai/2024).

Variáveis novas (src.chuva_prevista.variaveis), por sub-bacia: chuva nas horas que o IMERG ainda não mostra ('lacuna',
~7 h), nas próximas 6 h e nas 6 h seguintes.

Cenários
    atual        só o que o modelo usa hoje (ANA + IMERG Early com 6 h de atraso)
    oraculo      + chuva FUTURA observada pelo IMERG no treino e no teste: teto teórico (previsão de chuva perfeita)
    imerg>gfs    treina com a chuva futura observada (IMERG) e testa com a previsão do GFS ("prognóstico perfeito")
    gfs          treina e testa com a previsão do GFS (só existe a partir de abr/2021: menos exemplos com GFS)
    pluvio       atual + 5 pluviômetros a mais nas cabeceiras (sem atraso; históricos curtos e com falhas)
    pluvio+gfs   pluvio + previsão do GFS

Uso:
    python -m src.experimento_chuva_prevista
Saídas: outputs/experimento_chuva_prevista/{metricas.csv, eventos.csv}
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ["CAPSTONE_IMERG"] = "1"
from . import chuva_prevista as cp, config, features, metrics, pipeline  # noqa: E402
from .operacional import _fit, predict  # noqa: E402

CORTE = pd.Timestamp("2022-12-31 23:00")
TESTE = pd.Timestamp("2023-01-01 00:00")
OUT = config.OUTPUTS / "experimento_chuva_prevista"


def _imerg_utc() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / "chuva_imerg_early_horaria.csv", parse_dates=[0], index_col=0)


def _gfs_utc() -> pd.DataFrame:
    if not cp.CSV_HIST.exists():
        raise FileNotFoundError("Rode antes: python -m src.chuva_prevista historico")
    return pd.read_csv(cp.CSV_HIST, parse_dates=[0], index_col=0)


PLUVIO_EXTRA = {"86160000": "Passo Tainhas", "2851072": "Ibiraiaras", "2851044": "Guaporé",
                "86471000": "UHE 14 de Julho", "86450000": "UHE Monte Claro"}


def pluvio_extra(index) -> pd.DataFrame:
    """Chuva horária dos pluviômetros extras, com controle de qualidade: > 60 mm/h vira NaN e janelas de 30 dias em que
    o pluviômetro soma menos de 20% (sensor parado) ou mais de 4x (sensor com defeito) do IMERG na bacia viram NaN."""
    from . import data

    im = _imerg_utc()["bacia"]
    im.index = im.index - pd.Timedelta(hours=1) + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS)
    im30 = im.reindex(index).rolling(24 * 30, min_periods=24 * 20, center=True).sum()
    cols = {}
    for code in PLUVIO_EXTRA:
        raw = pd.read_csv(config.RAW / "ana_pluvio" / f"{code}_telemetrica_api.csv", parse_dates=["Data_Hora_Medicao"])
        df = data.standardize_export(raw, cota_scale=config.COTA_SCALE, utc_offset_hours=config.UTC_OFFSET_HOURS)
        h = data.to_hourly(df[["chuva_mm"]])["chuva_mm"].reindex(index)
        h[h > 60] = np.nan
        g30 = h.rolling(24 * 30, min_periods=24 * 20, center=True).sum()
        ruim = (im30 > 40) & ((g30 < 0.2 * im30) | (g30 > 4 * im30))
        h[ruim | g30.isna()] = np.nan
        cols[f"chuva_{code}"] = h
        print(f"[experimento] pluviômetro {PLUVIO_EXTRA[code]}: {h.notna().mean():.0%} das horas válidas "
              f"({h.loc[:CORTE].notna().mean():.0%} no treino)")
    return pd.DataFrame(cols, index=index)


def _janelas_chuva(chuva: pd.DataFrame) -> pd.DataFrame:
    """Mesmos acumulados que features.tabular_features aplica às colunas chuva_*."""
    F = {}
    for c in chuva.columns:
        for w in config.RAIN_WINDOWS:
            F[f"{c}_s{w}h"] = chuva[c].rolling(w, min_periods=max(1, int(0.8 * w))).sum()
    return pd.DataFrame(F, index=chuva.index)


def cenarios(index) -> dict:
    """{nome: (variáveis de treino, variáveis de teste)}; None = sem variáveis novas."""
    im = cp.variaveis(_imerg_utc(), index)
    gf = cp.variaveis(_gfs_utc(), index)
    pl = _janelas_chuva(pluvio_extra(index))
    return {"atual": (None, None), "oraculo": (im, im), "imerg>gfs": (im, gf), "gfs": (gf, gf),
            "pluvio": (pl, pl), "pluvio+gfs": (pl.join(gf), pl.join(gf))}


def rodar():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pipeline.load_panel()
    cen = cenarios(panel.index)
    gap = pd.Timedelta(hours=max(config.HORIZONS))
    linhas, evs = [], []
    for alvo in config.TARGETS:
        P = features.build_tabular(panel, alvo)
        full = ~np.isnan(P.Y).any(axis=1)
        tr = np.asarray(P.index <= CORTE - gap) & full
        te = np.asarray(P.index >= TESTE)
        for nome, (ftr, fte) in cen.items():
            if ftr is None:
                Xtr = Xte = P.X
            else:
                Xtr = np.hstack([P.X, ftr.reindex(P.index).to_numpy("float32")])
                Xte = np.hstack([P.X, fte.reindex(P.index).to_numpy("float32")])
            # no cenário 'gfs', o treino antes de abr/2021 fica com NaN nas variáveis novas (o HGB trata como falta)
            P_ =features.Prepared(P.index, Xtr, P.Y, P.base, P.horizons, P.feature_names, P.masks)
            gbs, bi, bo = _fit(P_, tr)
            pred = P.base[:, None] + predict(gbs, bi, bo, Xte)
            tab = metrics.summary_table(P.Yabs[te], pred[te], P.base[te], P.horizons).reset_index()
            tab.insert(0, "cenario", nome)
            tab.insert(0, "estacao", alvo)
            linhas.append(tab)
            obs = panel[f"cota_{alvo}"]
            for j, h in enumerate(P.horizons):
                ps = metrics.as_series(P.index[te], pred[te][:, j], h)
                for ev, (a, b) in config.EVENTS.items():
                    st = metrics.event_stats(obs, ps, a, b)
                    if st:
                        evs.append({"estacao": alvo, "cenario": nome, "evento": ev, "horizonte_h": h, **st})
            print(f"[experimento] {config.STATIONS[alvo]['nome']} {nome}: skill "
                  + " / ".join(f"{v:.3f}" for v in tab["skill_vs_persist"]), flush=True)
    met = pd.concat(linhas, ignore_index=True)
    ev = pd.DataFrame(evs)
    met.round(4).to_csv(OUT / "metricas.csv", index=False)
    ev.round(2).to_csv(OUT / "eventos.csv", index=False)
    return met, ev


if __name__ == "__main__":
    met, ev = rodar()
    print("\nMétricas no teste (2023+):")
    print(met.pivot_table(index=["estacao", "cenario"], columns="horizonte_h",
                          values=["RMSE_cm", "skill_vs_persist"]).round(3).to_string())
    print("\nErro de pico (cm):")
    print(ev.pivot_table(index=["estacao", "evento", "cenario"], columns="horizonte_h",
                         values="erro_pico_cm").round(0).to_string())
