"""Construção de features/alvos e divisão temporal.

Convenção: o alvo é a VARIAÇÃO da cota em h horas (cota[t+h] - cota[t]).
Isso permite que o modelo "extrapole" para níveis nunca vistos no treino, o que importa
nas cheias extremas de 2023-2024. A previsão absoluta é cota[t] + variação prevista.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config


def drop_sparse_columns(panel: pd.DataFrame, protect=(), verbose: bool = True) -> pd.DataFrame:
    """Remove colunas com pouco dado no período de TREINO (ex.: estação que só começou a transmitir em 2023).

    Sem isso, uma única coluna esparsa elimina todas as linhas (todas as features precisam estar válidas).
    Limiar: config.MIN_COVERAGE_TRAIN (padrão 0.5)."""
    min_frac = getattr(config, "MIN_COVERAGE_TRAIN", 0.5)
    ref = panel.loc[: pd.Timestamp(config.TRAIN_END)]
    if len(ref) == 0:
        return panel
    cov = ref.notna().mean()
    drop = [c for c in panel.columns if cov[c] < min_frac]
    for c in protect:
        if c in drop:
            raise ValueError(f"A coluna-alvo {c} tem só {100 * cov[c]:.0f}% de dados até TRAIN_END ({config.TRAIN_END}). "
                             "Ajuste TRAIN_END/DATA_START em config.py.")
    if drop and verbose:
        print("[features] colunas removidas por pouca cobertura no treino:",
              {c: f"{100 * cov[c]:.0f}%" for c in drop})
    return panel.drop(columns=drop)


@dataclass
class Prepared:
    index: pd.DatetimeIndex          # instante t de cada amostra
    X: np.ndarray                    # (N, F) tabular  ou  (N, L, F) sequência
    Y: np.ndarray                    # (N, H) variação da cota em cada horizonte (pode ter NaN no teste)
    base: np.ndarray                 # (N,) cota em t
    horizons: list
    feature_names: list = field(default_factory=list)
    masks: dict = field(default_factory=dict)

    @property
    def Yabs(self) -> np.ndarray:    # cota observada em t+h
        return self.base[:, None] + self.Y


def split_masks(index: pd.DatetimeIndex, horizons, train_end=None, val_end=None, test_start=None) -> dict:
    """Máscaras temporais sem sorteio. Remove `max(horizons)` horas no fim de treino/validação (evita vazamento)."""
    train_end = pd.Timestamp(train_end or config.TRAIN_END)
    val_end = pd.Timestamp(val_end or config.VAL_END)
    test_start = pd.Timestamp(test_start or config.TEST_START)
    gap = pd.Timedelta(hours=max(horizons))
    return {
        "train": np.asarray(index <= train_end - gap),
        "val": np.asarray((index > train_end) & (index <= val_end - gap)),
        "test": np.asarray(index >= test_start),
    }


def _masks_with_targets(index, horizons, Y: np.ndarray) -> dict:
    """Treino e validação exigem o alvo em TODOS os horizontes (a perda é calculada nos três de uma vez).

    No teste basta um horizonte válido: cada horizonte é avaliado nos instantes em que a cota em t+h existe.
    Sem isso, uma lacuna de poucas horas apagava também as amostras de 6 h e 12 h ao redor (até 24 h antes),
    e o "pico observado" dos eventos mudava conforme o horizonte."""
    m = split_masks(index, horizons)
    full = ~np.isnan(Y).any(axis=1)
    m["train"] &= full
    m["val"] &= full
    return m


def tabular_features(panel: pd.DataFrame, lags=None, rain_windows=None) -> pd.DataFrame:
    """Variáveis tabulares de cada instante t (só passado). Usada no treino e na previsão ao vivo (src.ao_vivo)."""
    lags = lags or config.LAGS
    rain_windows = rain_windows or config.RAIN_WINDOWS
    cota_cols = [c for c in panel.columns if c.startswith("cota_")]
    rain_cols = [c for c in panel.columns if c.startswith("chuva_")]
    F = {}
    for c in cota_cols:
        F[f"{c}_t"] = panel[c]
        for l in lags:
            F[f"{c}_d{l}h"] = panel[c] - panel[c].shift(l)
    for c in rain_cols:
        for w in rain_windows:
            F[f"{c}_s{w}h"] = panel[c].rolling(w, min_periods=max(1, int(0.8 * w))).sum()
    for c in [c for c in panel.columns if c.startswith("imerg_")]:      # acumulados diários do IMERG (já alinhados)
        F[c] = panel[c]
    return pd.DataFrame(F, index=panel.index)


def build_tabular(panel: pd.DataFrame, target: str, horizons=None, lags=None, rain_windows=None,
                  allow_missing: bool = True) -> Prepared:
    horizons = list(horizons or config.HORIZONS)
    panel = drop_sparse_columns(panel.dropna(axis=1, how="all"), protect=[f"cota_{target}"])
    tcol = f"cota_{target}"
    F = tabular_features(panel, lags, rain_windows)

    Y = pd.DataFrame({f"y_{h}h": panel[tcol].shift(-h) - panel[tcol] for h in horizons})
    if allow_missing:       # só a própria estação-alvo precisa estar válida; as demais podem ter buracos (NaN)
        req = [c for c in F.columns if c.startswith(f"{tcol}_")]
        ok = F[req].notna().all(axis=1) & Y.notna().any(axis=1)
    else:
        ok = F.notna().all(axis=1) & Y.notna().any(axis=1)
    F, Y, base = F[ok], Y[ok], panel.loc[ok, tcol]
    Yv = Y.to_numpy("float32")
    return Prepared(F.index, F.to_numpy("float32"), Yv, base.to_numpy("float32"),
                    horizons, list(F.columns), _masks_with_targets(F.index, horizons, Yv))


def build_sequences(panel: pd.DataFrame, target: str, horizons=None, seq_len=None,
                    allow_missing: bool = True) -> Prepared:
    horizons = list(horizons or config.HORIZONS)
    seq_len = seq_len or config.SEQ_LEN
    panel = drop_sparse_columns(panel.dropna(axis=1, how="all"), protect=[f"cota_{target}"], verbose=False)
    cols = list(panel.columns)
    A = panel[cols].to_numpy("float32")
    T = len(A)
    tgt = panel[f"cota_{target}"].to_numpy("float32")

    W = np.swapaxes(sliding_window_view(A, seq_len, axis=0), 1, 2)   # (T-L+1, L, F)
    end = np.arange(seq_len - 1, T)
    Y = np.full((len(end), len(horizons)), np.nan, dtype="float32")
    for j, h in enumerate(horizons):
        ok = end + h < T
        Y[ok, j] = tgt[end[ok] + h] - tgt[end[ok]]
    if allow_missing:       # só a cota da estação-alvo no instante t é obrigatória
        valid = ~np.isnan(W[:, -1, cols.index(f"cota_{target}")]) & ~np.isnan(Y).all(axis=1)
    else:
        valid = ~np.isnan(W).any(axis=(1, 2)) & ~np.isnan(Y).all(axis=1)
    idx = panel.index[end][valid]
    return Prepared(idx, np.ascontiguousarray(W[valid]), Y[valid], tgt[end][valid], horizons, cols,
                    _masks_with_targets(idx, horizons, Y[valid]))


@dataclass
class Scaler:
    """Padroniza com média/desvio do treino. NaN é ignorado no ajuste e preservado na transformação."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray):
        axes = tuple(range(X.ndim - 1))
        mean = np.nan_to_num(np.nanmean(X, axis=axes), nan=0.0)
        std = np.nan_to_num(np.nanstd(X, axis=axes), nan=1.0)
        return cls(mean, std + 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype("float32")

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean


@dataclass
class Bounds:
    """Faixa [mín, máx] de cada variável no TREINO, alargada por uma margem (fração da amplitude).

    Usada para (a) limitar as entradas à faixa vista no treino e (b) limitar a variação prevista.
    Redes e modelos lineares extrapolam sem freio fora da faixa do treino (ex.: MLP prevendo 4676 cm
    em mai/2024); árvores já são limitadas por construção, então para elas o efeito é quase nulo.
    NaN é ignorado no ajuste e preservado no recorte."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, A: np.ndarray, margin: float | None = None):
        margin = getattr(config, "CLIP_MARGIN", 0.10) if margin is None else margin
        axes = tuple(range(A.ndim - 1))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # colunas só com NaN
            lo, hi = np.nanmin(A, axis=axes), np.nanmax(A, axis=axes)
        pad = margin * (hi - lo)
        lo = np.where(np.isnan(lo), -np.inf, lo - pad)
        hi = np.where(np.isnan(hi), np.inf, hi + pad)
        return cls(lo.astype("float32"), hi.astype("float32"))

    def clip(self, A: np.ndarray) -> np.ndarray:
        return np.clip(A, self.lo, self.hi).astype(A.dtype)

    def frac_outside(self, A: np.ndarray) -> np.ndarray:
        """Fração das amostras fora da faixa, por variável (diagnóstico)."""
        axes = tuple(range(A.ndim - 1))
        return ((A < self.lo) | (A > self.hi)).mean(axis=axes)


def input_bounds(P: Prepared, margin: float | None = None) -> Bounds:
    return Bounds.fit(P.X[P.masks["train"]], margin)


def output_bounds(P: Prepared, margin: float | None = None) -> Bounds:
    return Bounds.fit(P.Y[P.masks["train"]], margin)


def limit_delta(P: Prepared, delta: np.ndarray, margin: float | None = None) -> np.ndarray:
    """Recorta a variação prevista (N, H) à faixa vista no treino em cada horizonte (+ margem)."""
    return output_bounds(P, margin).clip(np.asarray(delta, dtype="float32"))


def with_missing_flags(Xs: np.ndarray, cols=None):
    """Troca NaN por 0 (= média do treino, após padronizar) e acrescenta colunas 0/1 'estava faltando'.

    Redes neurais não aceitam NaN, e o fato de uma estação estar fora do ar também é informação."""
    miss = np.isnan(Xs)
    if cols is None:
        cols = np.where(miss.any(axis=tuple(range(Xs.ndim - 1))))[0]
    flags = miss[..., cols].astype("float32")
    filled = np.where(miss, 0.0, Xs).astype("float32")
    return np.concatenate([filled, flags], axis=-1), cols


def clipped_X(P: Prepared, clip: bool = True) -> np.ndarray:
    """P.X com cada variável limitada à faixa do treino (+ margem). No treino não muda nada."""
    return input_bounds(P).clip(P.X) if clip else P.X


def nn_inputs(P: Prepared, clip: bool = True):
    """Entradas prontas para Ridge/MLP/GRU: limita à faixa do treino (clip=True), padroniza (ajuste só no
    treino), imputa e acrescenta indicadores de falta.

    Os indicadores são escolhidos pelas colunas com falta em P.X inteiro (não muda com clip), então
    `nn_inputs(P, True)` e `nn_inputs(P, False)` têm as mesmas colunas e servem ao mesmo modelo."""
    X = clipped_X(P, clip)
    sc = Scaler.fit(X[P.masks["train"]])
    Xn, cols = with_missing_flags(sc.transform(X))
    return Xn, sc
