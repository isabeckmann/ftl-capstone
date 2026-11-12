"""Funções compartilhadas pelos notebooks 02-05."""
from __future__ import annotations

import os

import pandas as pd

from . import config, data
from .features import Prepared


def use_imerg() -> bool:
    """True se o IMERG entra como feature. A variável CAPSTONE_IMERG (1/0) tem prioridade sobre config.USE_IMERG."""
    env = os.environ.get("CAPSTONE_IMERG")
    if env in ("0", "1"):
        return env == "1"
    return bool(getattr(config, "USE_IMERG", False))


def current_tag() -> str:
    return "com_imerg" if use_imerg() else "sem_imerg"


def pred_dir(tag: str | None = None):
    d = config.OUTPUTS / "preds" / (tag or current_tag())
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_panel() -> pd.DataFrame:
    """Painel horário salvo pelo notebook 01. Com USE_IMERG, acrescenta os acumulados diários do IMERG."""
    panel = data.clean_panel(pd.read_parquet(config.PROCESSED / "panel_hourly.parquet"))
    panel = data.add_basin_rain(panel)                 # chuva horária da bacia, se existir (opcional)
    if use_imerg():
        if getattr(config, "IMERG_PRODUTO", "early") == "early":
            panel = data.add_imerg_hourly(panel)
        else:
            panel = data.add_imerg_daily(panel, local_utc_offset=getattr(config, "LOCAL_UTC_OFFSET_HOURS", -3.0))
    print(f"[pipeline] cenário: {current_tag()} | colunas: {len(panel.columns)}")
    return panel


def save_predictions(model_name: str, target: str, P: Prepared, pred_abs, pred_abs_sem_limite=None) -> None:
    """Salva as previsões absolutas (cm) de validação+teste: índice = instante t da previsão.

    `pred_abs` é a previsão final (com limites de entrada/saída). Se `pred_abs_sem_limite` for dado,
    também é salvo (colunas pred_sem_limite_<h>h) para medir o efeito dos limites nos mesmos instantes."""
    keep = P.masks["val"] | P.masks["test"]
    df = pd.DataFrame(pred_abs[keep], index=P.index[keep], columns=[f"pred_{h}h" for h in P.horizons])
    if pred_abs_sem_limite is not None:
        for j, h in enumerate(P.horizons):
            df[f"pred_sem_limite_{h}h"] = pred_abs_sem_limite[keep][:, j]
    df["cota_t"] = P.base[keep]
    for j, h in enumerate(P.horizons):
        df[f"obs_{h}h"] = P.Yabs[keep][:, j]
    df["conjunto"] = ["teste" if t else "validacao" for t in P.masks["test"][keep]]
    df.to_parquet(pred_dir() / f"{model_name}__{target}.parquet")


def load_predictions(model_name: str, target: str, tag: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(pred_dir(tag) / f"{model_name}__{target}.parquet")
