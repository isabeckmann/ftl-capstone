"""Dados SINTÉTICOS só para testar o pipeline antes de ter os exports reais da ANA.

Simula chuva na bacia, resposta lenta dos rios (montante -> jusante) e três cheias extremas
(set/2023, nov/2023, mai/2024) maiores do que qualquer evento do período de treino.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def _storms(n: int, rng: np.random.Generator, extreme_days: dict) -> np.ndarray:
    rain = np.zeros(n)
    n_events = int(n / 24 / 6)                      # ~1 evento a cada 6 dias
    for s in rng.integers(0, n - 100, n_events):
        dur = int(rng.integers(6, 48))
        rain[s:s + dur] += rng.gamma(2.0, 1.2, dur) * rng.uniform(0.5, 2.5)
    for day, amp in extreme_days.items():           # eventos extremos
        s = int((pd.Timestamp(day) - pd.Timestamp("2015-01-01")) / pd.Timedelta(hours=1))
        if 0 <= s < n - 80:
            rain[s:s + 60] += rng.gamma(3.0, 2.0, 60) * amp
    return rain


def _respond(rain: np.ndarray, tau: float, delay: int, gain: float) -> np.ndarray:
    a = np.exp(-1.0 / tau)
    q = lfilter([1 - a], [1, -a], rain) * gain
    return np.roll(q, delay)


def write_synthetic_exports(out_dir, start="2015-01-01", end="2024-06-30", seed=7) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h")
    n = len(idx)
    rain = _storms(n, rng, {"2023-09-03": 3.0, "2023-11-16": 2.4, "2024-04-30": 3.6})

    up1 = 150 + _respond(rain, 10, 0, 110) + rng.normal(0, 2, n)
    up2 = 140 + _respond(rain, 14, 2, 105) + rng.normal(0, 2, n)
    mucum = 180 + _respond(rain, 20, 6, 75) + 0.30 * (np.roll(up1, 5) - 150) + 0.25 * (np.roll(up2, 4) - 140) + rng.normal(0, 3, n)
    enc = 200 + 0.85 * (np.roll(mucum, 4) - 180) + _respond(rain, 16, 2, 30) + rng.normal(0, 3, n)

    series = {"86472000": (up1, 0.9), "86472600": (up2, 0.8), "86510000": (mucum, 0.6), "86720000": (enc, 0.5)}
    q15 = pd.date_range(start, end, freq="15min")
    for code, (lvl, rain_scale) in series.items():
        lvl15 = np.interp(np.arange(len(q15)) / 4, np.arange(n), lvl)
        r15 = np.interp(np.arange(len(q15)) / 4, np.arange(n), rain * rain_scale) / 4
        df = pd.DataFrame({"DataHora": q15.strftime("%d/%m/%Y %H:%M:%S"),
                           "Nivel": [f"{v:.1f}".replace(".", ",") for v in lvl15],
                           "Chuva": [f"{v:.2f}".replace(".", ",") for v in r15]})
        # simula lacunas curtas
        drop = rng.random(len(df)) < 0.002
        df.loc[drop, ["Nivel", "Chuva"]] = ""
        df.to_csv(out_dir / f"{code}_sintetico.csv", sep=";", index=False, encoding="utf-8")
