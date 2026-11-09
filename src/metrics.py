"""Métricas de avaliação (hidrologia + eventos extremos)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(obs, pred) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs, pred) -> float:
    return float(np.mean(np.abs(np.asarray(obs, float) - np.asarray(pred, float))))


def nse(obs, pred) -> float:
    """Eficiência de Nash-Sutcliffe: 1 = perfeito; 0 = tão bom quanto a média observada."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    den = np.sum((obs - obs.mean()) ** 2)
    return float("nan") if den == 0 else float(1 - np.sum((obs - pred) ** 2) / den)


def skill_vs_persistence(obs, pred, persist) -> float:
    """1 - MSE(modelo)/MSE(persistência). > 0 = melhor que repetir a cota atual."""
    mse_m = np.mean((np.asarray(obs, float) - np.asarray(pred, float)) ** 2)
    mse_p = np.mean((np.asarray(obs, float) - np.asarray(persist, float)) ** 2)
    return float("nan") if mse_p == 0 else float(1 - mse_m / mse_p)


def summary_table(obs_abs: np.ndarray, pred_abs: np.ndarray, base: np.ndarray, horizons) -> pd.DataFrame:
    """Uma linha por horizonte. obs_abs/pred_abs: (N, H) em cm; base = cota em t (persistência)."""
    rows = []
    for j, h in enumerate(horizons):
        o, p = obs_abs[:, j], pred_abs[:, j]
        rows.append({"horizonte_h": h, "RMSE_cm": rmse(o, p), "MAE_cm": mae(o, p), "NSE": nse(o, p),
                     "skill_vs_persist": skill_vs_persistence(o, p, base)})
    return pd.DataFrame(rows).set_index("horizonte_h").round(3)


def as_series(index_t: pd.DatetimeIndex, values: np.ndarray, h: int) -> pd.Series:
    """Alinha uma previsão feita em t ao instante previsto (t+h), para comparar com a observação."""
    return pd.Series(np.asarray(values), index=index_t + pd.Timedelta(hours=h))


def event_stats(obs: pd.Series, pred: pd.Series, start, end) -> dict:
    """Erro de pico (cm e %) e de horário do pico (h; > 0 = previsto atrasado) numa janela de evento.

    Também informa a cobertura de dados na janela e se o pico OBSERVADO fica ao lado de uma lacuna
    (sensor que falha na cheia faz o 'pico observado' ser menor que o pico real)."""
    o = obs.loc[start:end].dropna()
    p = pred.reindex(o.index).dropna()
    o = o.reindex(p.index)
    if len(o) == 0:
        return {}
    t_o, t_p = o.idxmax(), p.idxmax()
    horas = (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timestamp(start)) / pd.Timedelta(hours=1)
    viz = pd.date_range(t_o - pd.Timedelta(hours=3), t_o + pd.Timedelta(hours=3), freq="h")
    return {"pico_obs_cm": float(o.max()), "pico_prev_cm": float(p.max()),
            "erro_pico_cm": float(p.max() - o.max()),
            "erro_pico_%": float(100 * (p.max() - o.max()) / o.max()),
            "erro_horario_h": float((t_p - t_o) / pd.Timedelta(hours=1)),
            "RMSE_evento_cm": rmse(o, p),
            "cobertura_%": float(100 * len(o) / horas),
            "pico_perto_de_lacuna": bool(not all(t in o.index for t in viz))}


def alert_scores(obs: pd.Series, pred: pd.Series, threshold: float) -> dict:
    """Acerto na passagem da cota de alerta: POD (acertos), FAR (alarmes falsos), CSI."""
    both = pd.concat([obs, pred], axis=1, keys=["o", "p"]).dropna()
    o, p = both["o"] >= threshold, both["p"] >= threshold
    hits, miss, fa = int((o & p).sum()), int((o & ~p).sum()), int((~o & p).sum())
    pod = hits / (hits + miss) if hits + miss else float("nan")
    far = fa / (hits + fa) if hits + fa else float("nan")
    csi = hits / (hits + miss + fa) if hits + miss + fa else float("nan")
    return {"POD": pod, "FAR": far, "CSI": csi, "horas_acima_obs": int(o.sum())}
