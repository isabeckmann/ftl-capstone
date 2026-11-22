"""Chuva PREVISTA por modelo meteorológico (GFS, ECMWF) na bacia e sub-bacias, via Open-Meteo (grátis, sem cadastro).

Por que: o IMERG Early chega com ~6 h de atraso e só mostra chuva que já caiu. Na cheia, boa parte da chuva que faz o
rio subir nas próximas 12 h ainda não caiu na hora da previsão. A previsão do modelo meteorológico cobre as duas coisas:
as horas recentes que o satélite ainda não publicou e as próximas horas.

Grade: pontos de 0,25° (resolução do GFS/ECMWF) dentro da bacia. O peso de cada ponto em cada sub-bacia é a soma dos
pesos das células IMERG (0,1°, operacional/pesos_imerg.json) mais próximas dele, então não precisa de shapely/geopandas.

Convenção de tempo igual à do IMERG (src.imerg_early): índice = FIM da hora em UTC, valor = mm caídos naquela hora.

Uso:
    python -m src.chuva_prevista historico --inicio 2021-04-01 --fim 2026-07-01   # GFS arquivado (treino/teste)
    python -m src.chuva_prevista agora                                          # previsão atual (teste rápido)

Atenção (para o artigo): o arquivo histórico do Open-Meteo junta as primeiras horas de cada rodada do modelo; ao vivo, a
janela de +6 a +12 h vem de uma rodada mais antiga (antecedência maior). O teste com o histórico é, portanto, um pouco
otimista para as janelas futuras.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from . import config

URL_PREVISAO = "https://api.open-meteo.com/v1/forecast"
URL_HISTORICO = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PASSO_GRADE = 0.25
MODELO_TREINO = "gfs_seamless"                 # único com arquivo desde 2021
MODELOS_AO_VIVO = ("gfs_seamless", "ecmwf_ifs025")
DIR_HIST = config.RAW / "gfs"
CSV_HIST = config.PROCESSED / "chuva_gfs_horaria.csv"
AREAS = ("bacia", "antas", "taquari_sup", "guapore")


# ----------------------------------------------------------------------------- grade e pesos
@lru_cache(maxsize=1)
def grade() -> tuple[np.ndarray, np.ndarray, dict]:
    """(lat, lon, pesos) dos pontos de 0,25° com peso > 0 na bacia; pesos[area] tem o mesmo tamanho de lat."""
    pes = json.loads((config.ROOT / "operacional" / "pesos_imerg.json").read_text(encoding="utf-8"))
    lon, lat = np.array(pes["lon"]), np.array(pes["lat"])
    glon = np.round(np.round(lon / PASSO_GRADE) * PASSO_GRADE, 3)
    glat = np.round(np.round(lat / PASSO_GRADE) * PASSO_GRADE, 3)
    W = {k: np.asarray(v, dtype=float) for k, v in pes["pesos"].items()}
    acc: dict[tuple, dict] = {}
    for j in range(len(lat)):
        for i in range(len(lon)):
            if W["bacia"][j, i] <= 0:
                continue
            d = acc.setdefault((glat[j], glon[i]), {k: 0.0 for k in W})
            for k in W:
                d[k] += W[k][j, i]
    pts = sorted(acc)
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
            {k: np.array([acc[p][k] for p in pts]) for k in W})


def _media_por_area(M: np.ndarray, pesos: dict) -> dict:
    """M (tempo, pontos) -> média ponderada por área; hora sem nenhum ponto válido vira NaN."""
    out = {}
    for k, w in pesos.items():
        ok = ~np.isnan(M)
        num = np.nansum(M * w, axis=1)
        den = (ok * w).sum(axis=1)
        out[k] = np.where(den > 0.5 * w.sum(), num / np.where(den > 0, den, 1), np.nan)
    return out


# ----------------------------------------------------------------------------- requisições
def _get(url: str, params: dict, tentativas: int = 4) -> list:
    last = None
    for n in range(1, tentativas + 1):
        try:
            r = requests.get(url, params=params, timeout=(15, 180))
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(30 * n)
                continue
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, list) else [js]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last = type(exc).__name__
            time.sleep(10 * n)
    raise RuntimeError(f"Open-Meteo falhou após {tentativas} tentativas ({last}).")


def _series(js: list, var: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Lista de respostas (uma por ponto, na ordem da grade) -> (índice FIM da hora UTC, matriz tempo x pontos)."""
    t = pd.to_datetime(js[0]["hourly"]["time"])
    M = np.array([[np.nan if v is None else v for v in r["hourly"][var]] for r in js], dtype=float).T
    return pd.DatetimeIndex(t), M


def _params(lat, lon, modelos) -> dict:
    return {"latitude": ",".join(f"{x:.3f}" for x in lat), "longitude": ",".join(f"{x:.3f}" for x in lon),
            "hourly": "precipitation", "models": ",".join(modelos), "timezone": "GMT"}


# ----------------------------------------------------------------------------- ao vivo
def prevista_agora(modelos=MODELOS_AO_VIVO, horas_passadas: int = 24, horas_futuras: int = 48) -> dict:
    """{modelo: DataFrame} com a chuva horária por área (mm), índice = fim da hora em UTC.

    Inclui as horas recentes (que o IMERG ainda não publicou) e as próximas `horas_futuras`."""
    lat, lon, pesos = grade()
    p = _params(lat, lon, modelos) | {"past_hours": horas_passadas, "forecast_hours": horas_futuras}
    js = _get(URL_PREVISAO, p)
    out = {}
    for m in modelos:
        t, M = _series(js, f"precipitation_{m}" if len(modelos) > 1 else "precipitation")
        df = pd.DataFrame(_media_por_area(M, pesos), index=t)
        if df.notna().any().any():
            out[m] = df
    return out


# ----------------------------------------------------------------------------- histórico
def baixar_historico(inicio, fim, modelo: str = MODELO_TREINO, pausa: float = 20.0) -> pd.DataFrame:
    """GFS arquivado, um mês por requisição (40 pontos ~ 80 'chamadas' do limite gratuito). Retomável."""
    lat, lon, pesos = grade()
    DIR_HIST.mkdir(parents=True, exist_ok=True)
    meses = pd.date_range(pd.Timestamp(inicio).to_period("M").to_timestamp(), pd.Timestamp(fim), freq="MS")
    for ini in meses:
        arq = DIR_HIST / f"{modelo}_{ini:%Y-%m}.csv"
        fim_mes = min(ini + pd.offsets.MonthEnd(0), pd.Timestamp(fim))
        if arq.exists() and fim_mes < pd.Timestamp.now() - pd.Timedelta(days=3):
            continue
        js = _get(URL_HISTORICO, _params(lat, lon, [modelo]) | {"start_date": f"{ini:%Y-%m-%d}",
                                                                 "end_date": f"{fim_mes:%Y-%m-%d}"})
        t, M = _series(js, "precipitation")
        df = pd.DataFrame(_media_por_area(M, pesos), index=t)
        df.index.name = "fim_hora_utc"
        df.round(3).to_csv(arq)
        print(f"[chuva_prevista] {modelo} {ini:%Y-%m}: {df['bacia'].notna().mean():.0%} das horas, "
              f"{df['bacia'].sum():.0f} mm na bacia", flush=True)
        time.sleep(pausa)
    partes = [pd.read_csv(f, parse_dates=[0], index_col=0) for f in sorted(DIR_HIST.glob(f"{modelo}_*.csv"))]
    hist = pd.concat(partes).sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    hist.to_csv(CSV_HIST)
    return hist


# ----------------------------------------------------------------------------- variáveis do modelo
JANELAS = {"lacuna": (-6, 1), "prox6h": (1, 7), "prox12h": (7, 13)}
"""Para o instante t do painel (média da hora [t, t+1), em hora local), soma da chuva das horas que TERMINAM em
(t+a, t+b]. 'lacuna' cobre as horas que o IMERG Early (6 h de atraso) ainda não mostra; 'prox6h' e 'prox12h' são as
próximas 6 h e as 6 h seguintes."""


def variaveis(chuva_utc: pd.DataFrame, index: pd.DatetimeIndex, prefixo: str = "prev",
              local_utc_offset: float | None = None) -> pd.DataFrame:
    """Colunas <prefixo>_<area>_<janela> alinhadas ao índice horário local do painel."""
    off = config.LOCAL_UTC_OFFSET_HOURS if local_utc_offset is None else local_utc_offset
    s = chuva_utc.copy()
    s.index = s.index + pd.Timedelta(hours=off)                  # fim da hora, em hora local
    s = s[~s.index.duplicated()]
    full = pd.date_range(min(s.index.min(), index.min()) - pd.Timedelta(hours=1),
                         max(s.index.max(), index.max()) + pd.Timedelta(hours=14), freq="h")
    s = s.reindex(full)
    out = {}
    for area in [c for c in AREAS if c in s.columns]:
        for nome, (a, b) in JANELAS.items():
            w = b - a
            soma = s[area].rolling(w, min_periods=w).sum()        # soma das horas que terminam em (y-w, y]
            out[f"{prefixo}_{area}_{nome}"] = soma.shift(-b).reindex(index)
    return pd.DataFrame(out, index=index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("historico")
    h.add_argument("--inicio", default="2021-04-01")
    h.add_argument("--fim", default="2026-07-01")
    h.add_argument("--modelo", default=MODELO_TREINO)
    h.add_argument("--pausa", type=float, default=20.0)
    sub.add_parser("agora")
    a = ap.parse_args()
    if a.cmd == "historico":
        hist = baixar_historico(a.inicio, a.fim, a.modelo, a.pausa)
        print(f"Salvo {CSV_HIST}: {hist.index.min()} a {hist.index.max()}")
    else:
        for m, df in prevista_agora().items():
            agora = pd.Timestamp.now(tz="UTC").tz_localize(None)
            fut = df[df.index > agora]
            print(f"{m}: próximas 12 h {fut.iloc[:12].sum().round(1).to_dict()} | 24 h {fut.iloc[:24].sum().round(1).to_dict()}")
