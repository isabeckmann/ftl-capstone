"""Cliente da API HidroWebService (ANA) para séries telemétricas (nível e chuva a cada 15 min).

Credenciais: crie um arquivo `.env` na raiz do projeto (NUNCA versione; já está no .gitignore):
    ANA_ID=seu_cpf_somente_numeros
    ANA_SENHA=sua_senha

Uso:
    python -m src.ana_api --start 2015-01-01 --end 2026-06-30 --stations 86510000 86720000 86472000 86472600
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

from . import config

BASE = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas"
ROUTE_SERIE = "HidroinfoanaSerieTelemetricaAdotada/v1"
ROUTE_INVENTARIO = "HidroInventarioEstacoes/v1"
DATE_PARAM = None
PROBE_STATION = "86510000"
PARAM_SETS = {
    "swagger": {"codigo": "Código da Estação", "filtro": "Tipo Filtro Data",
                "data": "Data de Busca (yyyy-MM-dd)", "range": "Range Intervalo de busca"},
    "manual": {"codigo": "CodigoDaEstacao", "filtro": "TipoFiltroData",
               "data": "DataDeBusca", "range": "RangeIntervaloDeBusca"},
}
RANGE = "DIAS_30"
COLS = ["Data_Hora_Medicao", "Cota_Adotada", "Cota_Adotada_Status", "Chuva_Adotada", "Chuva_Adotada_Status",
        "Vazao_Adotada", "Vazao_Adotada_Status"]


def _items_to_df(items) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=COLS)
    raw = pd.DataFrame(items)
    out = pd.DataFrame({"Data_Hora_Medicao": pd.to_datetime(raw.get("Data_Hora_Medicao"), errors="coerce")})
    for c in COLS[1:]:
        out[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else np.nan
    return out.dropna(subset=["Data_Hora_Medicao"])


class AnaClient:
    def __init__(self, ident=None, senha=None, base=BASE, session=None, pause=0.5, max_token_age=50 * 60,
                 range_=RANGE, verbose=True):
        try:
            from dotenv import load_dotenv

            load_dotenv(config.ROOT / ".env")
        except ImportError:
            pass
        self.ident = ident or os.environ.get("ANA_ID")
        self.senha = senha or os.environ.get("ANA_SENHA")
        if not self.ident or not self.senha:
            raise RuntimeError("Defina ANA_ID e ANA_SENHA no arquivo .env (raiz do projeto).")
        self.base, self.pause, self.max_token_age = base, pause, max_token_age
        self.session = session or requests.Session()
        self._token, self._t0 = None, 0.0
        self.names, self.names_key = None, None
        self.range_, self.verbose = range_, verbose

    # --- autenticação
    def _auth(self) -> str:
        if self.verbose:
            print("[ana_api] autenticando...", flush=True)
        r = self.session.get(f"{self.base}/OAUth/v1", headers={"Identificador": self.ident, "Senha": self.senha}, timeout=(15, 120))
        r.raise_for_status()
        js = r.json()
        tok = (js.get("items") or {}).get("tokenautenticacao")
        if not tok:
            raise RuntimeError(f"Autenticação sem token. Resposta: {str(js)[:300]}")
        self._token, self._t0 = tok, time.time()
        return tok

    def _tok(self) -> str:
        if self._token is None or time.time() - self._t0 > self.max_token_age:
            return self._auth()
        return self._token

    # --- requisição genérica
    def get(self, route: str, params: dict | None = None, retries: int = 4) -> dict:
        reauth_done = False
        last = None
        for attempt in range(1, retries + 1):
            url = f"{self.base}/{route}" + (("?" + urlencode(params, quote_via=quote)) if params else "")
            t0 = time.time()
            try:
                r = self.session.get(url, headers={"Authorization": f"Bearer {self._tok()}"}, timeout=(15, 180))
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last = type(exc).__name__
                if self.verbose:
                    print(f"[ana_api] {route.split('/')[0]}: {last} após {time.time() - t0:.0f}s (tentativa {attempt}/{retries})", flush=True)
                time.sleep(5 * attempt)
                continue
            if self.verbose:
                print(f"[ana_api] {route.split('/')[0]}: HTTP {r.status_code} em {time.time() - t0:.1f}s", flush=True)
            if r.status_code in (401, 403) and not reauth_done:
                self._token, reauth_done = None, True          # renova UMA vez
                continue
            last = r.status_code
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            time.sleep(self.pause)
            return r.json()
        raise RuntimeError(f"Falha ao consultar {route} após {retries} tentativas (último problema: {last}).")

    # --- séries
    def _fetch(self, params: dict) -> pd.DataFrame:
        js = self.get(ROUTE_SERIE, params)
        return _items_to_df(js.get("items"))

    def _param_names(self) -> dict:
        """Descobre qual formato de nomes de parâmetros a API aceita (usa Muçum, que está ativa)."""
        if self.names:
            return self.names
        for key, n in PARAM_SETS.items():
            try:
                df = self._fetch({n["codigo"]: PROBE_STATION, n["filtro"]: "DATA_LEITURA", n["range"]: "HORA_24"})
            except Exception:  # noqa: BLE001
                continue
            if len(df):
                self.names, self.names_key = n, key
                print(f"[ana_api] formato de parâmetros aceito: '{key}'")
                return n
        raise RuntimeError("A API não devolveu dados com nenhum dos formatos de parâmetros. Confira as credenciais, "
                           "se a estação de teste está ativa e me envie a Request URL do Swagger.")

    def series(self, code: str, date=None, range_: str | None = None, filtro: str = "DATA_LEITURA") -> pd.DataFrame:
        n = self._param_names()
        params = {n["codigo"]: code, n["filtro"]: filtro, n["range"]: range_ or self.range_}
        if date is not None:
            params[DATE_PARAM or n["data"]] = pd.Timestamp(date).strftime("%Y-%m-%d")
        return self._fetch(params)

    def inventario(self, **params) -> pd.DataFrame:
        """Metadados da estação (inclui Data_Periodo_Telemetrica_Inicio). Os nomes dos parâmetros: veja o Swagger."""
        js = self.get(ROUTE_INVENTARIO, params or None)
        return pd.DataFrame(js.get("items") or [])

    # --- histórico
    def _window_days(self) -> int:
        m = re.fullmatch(r"DIAS_(\d+)", self.range_)
        if not m:
            raise ValueError(f"Para baixar histórico use um intervalo DIAS_n (ex.: DIAS_30); recebi {self.range_}.")
        return int(m.group(1))

    def _detect_direction(self, code: str, probe) -> str:
        """Descobre se a janela termina na data informada ('before') ou começa nela ('after')."""
        df = self.series(code, date=probe)
        if df.empty:
            raise RuntimeError(f"Sem dados na data de teste {pd.Timestamp(probe).date()} para {code}. "
                               "Tente outra data em que a estação já operava (ver inventário).")
        p = pd.Timestamp(probe)
        lo, hi = df["Data_Hora_Medicao"].min(), df["Data_Hora_Medicao"].max()
        margin = pd.Timedelta(days=max(1, self._window_days() / 6))
        if hi <= p + pd.Timedelta(days=1) and lo < p - margin:
            return "before"
        if lo >= p - pd.Timedelta(days=1) and hi > p + margin:
            return "after"
        raise RuntimeError(f"Não consegui deduzir a direção da janela (data {p.date()}, dados {lo} -> {hi}). "
                           "Me envie esse resultado.")

    def download_history(self, code: str, start, end, out_dir=None, direction=None, verbose=True) -> Path:
        """Baixa janelas de ~30 dias e grava {code}_telemetrica_api.csv (retomável)."""
        out_dir = Path(out_dir or config.ANA_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{code}_telemetrica_api.csv"
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()

        have = pd.read_csv(out, parse_dates=["Data_Hora_Medicao"]) if out.exists() else pd.DataFrame(columns=COLS)
        if len(have):
            start = max(start, have["Data_Hora_Medicao"].max().normalize() - pd.Timedelta(days=1))

        direction = direction or self._detect_direction(code, min(end, start + pd.Timedelta(days=90)))
        step = pd.Timedelta(days=self._window_days() - 1)      # 1 dia de sobreposição
        d = start + (step if direction == "before" else pd.Timedelta(0))
        n = 0
        while (d - step if direction == "before" else d) <= end:
            df = self.series(code, date=d)
            n += 1
            if len(df):
                have = pd.concat([have, df]).drop_duplicates("Data_Hora_Medicao").sort_values("Data_Hora_Medicao")
                have.to_csv(out, index=False)
            if verbose:
                got = f"{len(df)} leituras ({df['Data_Hora_Medicao'].min():%Y-%m-%d} a {df['Data_Hora_Medicao'].max():%Y-%m-%d})" if len(df) else "sem dados"
                print(f"{code} janela {d:%Y-%m-%d} [{direction}]: {got}", flush=True)
            d += step
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--stations", nargs="+", default=list(config.STATIONS))
    ap.add_argument("--direction", choices=["before", "after"], default=None)
    ap.add_argument("--range", default=RANGE, help="janela por requisição: DIAS_30 (padrão), DIAS_21, DIAS_14, DIAS_7...")
    ap.add_argument("--date-param", default=None, help="(avançado) força o nome do parâmetro de data")
    a = ap.parse_args()
    if a.date_param:
        DATE_PARAM = a.date_param
    cli = AnaClient(range_=a.range)
    for code in a.stations:
        path = cli.download_history(code, a.start, a.end, direction=a.direction)
        print("salvo:", path)
