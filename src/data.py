"""Leitura tolerante dos exports da ANA e montagem do painel horário."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _to_num(s: pd.Series) -> pd.Series:
    """Converte texto em número aceitando '1234,5', '1234.5' e '1.234,5'."""
    s = s.astype(str).str.strip()
    both = s.str.contains(r"\.", regex=True) & s.str.contains(",", regex=False)
    s = s.where(~both, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def read_table(path: Path) -> pd.DataFrame:
    """Lê CSV/TXT/XLSX detectando separador e encoding. Tudo vem como texto (conversão depois)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        except Exception as e:  # noqa: BLE001
            last = e
    raise ValueError(f"Não consegui ler {path.name}: {last}")


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    cols = {_norm(c): c for c in df.columns}
    s = None
    for key in ("datahora", "datahoramedicao", "dataleitura", "datetime", "dthr", "dataehora", "datahoraleitura"):
        if key in cols:
            s = df[cols[key]]
            break
    if s is None:
        if "data" in cols and "hora" in cols:
            s = df[cols["data"]].astype(str).str.strip() + " " + df[cols["hora"]].astype(str).str.strip()
        elif "data" in cols:
            s = df[cols["data"]]
        else:
            cand = [c for c in df.columns if "data" in _norm(c) or "hora" in _norm(c)]
            if not cand:
                raise KeyError(f"Não achei coluna de data/hora. Colunas: {list(df.columns)}")
            s = df[cand[0]]
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5
    return pd.to_datetime(s, dayfirst=not iso, errors="coerce")


def _find_col(df: pd.DataFrame, exact: tuple, contains: tuple):
    norm = {_norm(c): c for c in df.columns}
    for k in exact:
        if k in norm:
            return norm[k]
    for c in df.columns:
        n = _norm(c)
        if any(k in n for k in contains) and not any(b in n for b in ("status", "consist", "qualid")):
            return c
    return None


def standardize_export(df: pd.DataFrame, cota_scale: float = 1.0, utc_offset_hours: float = 0.0,
                       drop_bad: bool = True) -> pd.DataFrame:
    """Converte um export bruto em DataFrame indexado por data/hora com colunas cota_cm e/ou chuva_mm.

    Se o arquivo tiver colunas de qualidade (ex.: Cota_Adotada_Status da API: 0 ok, 1 suspeito, 2 ruim),
    valores com status >= 2 viram NaN (drop_bad=True)."""
    idx = _parse_datetime(df)
    out = pd.DataFrame(index=pd.DatetimeIndex(idx))
    c_cota = _find_col(df, ("cota", "nivel", "cotaadotada", "niveladotado", "cotacm", "nivelcm"), ("cota", "nivel"))
    c_chuva = _find_col(df, ("chuva", "precipitacao", "chuvaadotada", "chuvamm"), ("chuva", "precip"))
    norm = {_norm(c): c for c in df.columns}

    def _clean(col, status_key):
        v = _to_num(df[col]).to_numpy(dtype="float64")
        st = norm.get(status_key)
        if drop_bad and st is not None:
            v = np.where(_to_num(df[st]).to_numpy(dtype="float64") >= 2, np.nan, v)
        return v

    if c_cota is not None:
        out["cota_cm"] = _clean(c_cota, "cotaadotadastatus") * cota_scale
    if c_chuva is not None:
        out["chuva_mm"] = _clean(c_chuva, "chuvaadotadastatus")
    if out.shape[1] == 0:
        raise KeyError(f"Não achei colunas de cota/chuva. Colunas: {list(df.columns)}")
    out = out[~out.index.isna()]
    if utc_offset_hours:
        out.index = out.index + pd.Timedelta(hours=utc_offset_hours)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


_CONVENCIONAIS = ("cotas", "vazoes", "chuvas", "resumodescarga", "qualagua", "sedimentos", "perfiltransversal")


def list_exports(raw_dir=None, code: str | None = None, warn: bool = True) -> list:
    """Arquivos de export válidos: nome começa com o código de uma estação do config, extensão csv/txt/xlsx.

    Ignora (com aviso) os arquivos do HidroWeb CONVENCIONAL/diário (..._Cotas.csv etc.), que têm outro formato."""
    raw_dir = Path(raw_dir or config.ANA_DIR)
    if not raw_dir.exists():
        return []
    out = []
    for p in sorted(raw_dir.glob("*")):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".txt", ".xlsx", ".xls"):
            continue
        if p.name[:8] not in config.STATIONS or (code and not p.name.startswith(code)):
            continue
        if any(k in _norm(p.stem) for k in _CONVENCIONAIS):
            if warn:
                print(f"[aviso] ignorando {p.name}: parece dado convencional/diário do HidroWeb (outro formato). "
                      "Guarde esses arquivos em outra pasta, ex.: data/raw/ana_diario/")
            continue
        out.append(p)
    return out


def load_station(code: str, raw_dir=None, cota_scale=None, utc_offset_hours=None) -> pd.DataFrame:
    """Lê e concatena todos os arquivos `{code}*` do diretório de exports."""
    raw_dir = Path(raw_dir or config.ANA_DIR)
    files = list_exports(raw_dir, code)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo '{code}*' em {raw_dir}")
    parts = [
        standardize_export(
            read_table(f),
            cota_scale=config.COTA_SCALE if cota_scale is None else cota_scale,
            utc_offset_hours=config.UTC_OFFSET_HOURS if utc_offset_hours is None else utc_offset_hours,
            drop_bad=getattr(config, "DROP_BAD_STATUS", True),
        )
        for f in files
    ]
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Passo horário: cota = média da hora; chuva = soma da hora (assume mm por leitura, não acumulado)."""
    agg = {}
    if "cota_cm" in df:
        agg["cota_cm"] = df["cota_cm"].resample("1h").mean()
    if "chuva_mm" in df:
        agg["chuva_mm"] = df["chuva_mm"].resample("1h").sum(min_count=1)
    return pd.DataFrame(agg)


def build_panel(raw_dir=None, stations=None, **kw) -> pd.DataFrame:
    """Painel horário com colunas cota_<codigo> e chuva_<codigo> (estações sem arquivo são ignoradas)."""
    stations = stations or config.STATIONS
    cols = []
    for code in stations:
        try:
            h = to_hourly(load_station(code, raw_dir=raw_dir, **kw))
        except FileNotFoundError:
            print(f"[aviso] sem arquivo para a estação {code} ({stations[code]['nome']})")
            continue
        h.columns = [f"{c.split('_')[0]}_{code}" for c in h.columns]
        cols.append(h)
    if not cols:
        raise FileNotFoundError("Nenhuma estação carregada.")
    panel = pd.concat(cols, axis=1).sort_index()
    full = pd.date_range(panel.index.min().floor("h"), panel.index.max().ceil("h"), freq="1h")
    return panel.reindex(full)


def fill_short_gaps(panel: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Interpola a cota só em lacunas curtas (<= max_gap horas). Chuva não é preenchida.

    Atenção: `interpolate(limit=n)` do pandas preenche as n PRIMEIRAS horas de qualquer lacuna, inclusive das
    longas (inventando valores no início de cada falha de sensor). Por isso as lacunas longas são deixadas intactas."""
    out = panel.copy()
    for c in out.columns:
        if c.startswith("cota_"):
            s = out[c]
            miss = s.isna()
            run_id = (miss != miss.shift()).cumsum()
            run_len = miss.groupby(run_id).transform("size")
            short = miss & (run_len <= max_gap)
            filled = s.interpolate(limit_area="inside")
            out[c] = s.where(~short, filled)
    return out


def add_basin_rain(panel: pd.DataFrame, path=None, col: str = "chuva_bacia") -> pd.DataFrame:
    """Junta a chuva média sobre a bacia (ex.: IMERG), se o arquivo existir: CSV com datetime e <col>."""
    path = Path(path or config.PROCESSED / "chuva_bacia.csv")
    if not path.exists():
        return panel
    s = pd.read_csv(path, parse_dates=[0], index_col=0)[col].resample("1h").mean()
    return panel.join(s.rename(col), how="left")


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in panel.columns:
        ok = panel[c].notna().to_numpy()
        if ok.any():
            first = panel.index[ok.argmax()]
            last = panel.index[len(ok) - 1 - ok[::-1].argmax()]
            # maior sequência de NaN
            best = cur = 0
            for v in ok:
                cur = 0 if v else cur + 1
                best = max(best, cur)
        else:
            first = last = pd.NaT
            best = len(ok)
        rows.append({"coluna": c, "primeiro": first, "ultimo": last,
                     "% válido": round(100 * ok.mean(), 1), "maior lacuna (h)": int(best)})
    return pd.DataFrame(rows).set_index("coluna")


def add_imerg_daily(panel: pd.DataFrame, csv=None, col: str = "chuva_bacia_mm", windows=(1, 3, 7),
                    local_utc_offset: float = -3.0) -> pd.DataFrame:
    """Acrescenta imerg_{w}d: chuva da bacia acumulada nos últimos w dias UTC COMPLETOS, para cada hora t do painel.

    O IMERG diário usa dia UTC (00h-24h UTC) e o painel está em horário local (UTC-3 => `local_utc_offset=-3`).
    Em cada instante t só entram dias que já terminaram (o dia UTC D termina às 21h locais), então não há vazamento
    do futuro. Consequência: a informação do IMERG diário chega ao modelo com até ~1 dia de atraso."""
    csv = Path(csv or config.PROCESSED / "chuva_bacia_diaria.csv")
    if not csv.exists():
        raise FileNotFoundError(f"Sem {csv}. Gere com: python -m src.imerg --series-only --basin ...")
    daily = pd.read_csv(csv, parse_dates=["data"], index_col="data")[col].sort_index().asfreq("D")
    t_utc = panel.index - pd.Timedelta(hours=local_utc_offset)          # local = UTC + offset
    last_complete = t_utc.floor("D") - pd.Timedelta(days=1)             # último dia UTC já encerrado em t
    out = panel.copy()
    for w in windows:
        roll = daily.rolling(w, min_periods=w).sum()
        out[f"imerg_{w}d"] = roll.reindex(last_complete).to_numpy()
    return out


def add_imerg_hourly(panel: pd.DataFrame, csv=None, latency_h: float | None = None,
                     local_utc_offset: float | None = None) -> pd.DataFrame:
    """Acrescenta chuva_imerg_<area> (mm/h, IMERG Early) com a defasagem de disponibilidade.

    O CSV (src.imerg_early) é indexado pelo FIM da hora em UTC. A hora que termina em T (local) só fica
    disponível em T + latency_h; é nesse instante que o valor entra no painel. Assim o treino usa exatamente
    a informação que existiria em tempo real. As somas móveis (RAIN_WINDOWS) são feitas em features.py."""
    if not isinstance(csv, pd.DataFrame):          # também aceita a série já carregada (uso ao vivo)
        csv = Path(csv or config.PROCESSED / "chuva_imerg_early_horaria.csv")
    if not isinstance(csv, pd.DataFrame) and not csv.exists():
        raise FileNotFoundError(f"Sem {csv}. Gere com: python -m src.imerg_early baixar ... && python -m src.imerg_early series")
    latency_h = getattr(config, "IMERG_LATENCIA_H", 6) if latency_h is None else latency_h
    off = getattr(config, "LOCAL_UTC_OFFSET_HOURS", -3.0) if local_utc_offset is None else local_utc_offset
    s = csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv, parse_dates=[0], index_col=0)
    s = s.copy()
    s.index = s.index + pd.Timedelta(hours=off) + pd.Timedelta(hours=latency_h)
    s = s[~s.index.duplicated()].reindex(panel.index)
    out = panel.copy()
    for c in s.columns:
        out[f"chuva_imerg_{c}"] = s[c].to_numpy()
    return out


def clean_panel(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Limpeza do painel horário (idempotente). Parâmetros opcionais em config.py (com estes padrões):
    DATA_START="2018-05-01"  descarta os anos iniciais, quase sem dados e com valores absurdos;
    MAX_CHUVA_MM_H=100       chuva horária acima disso vira NaN (erro de sensor);
    MIN_COTA_CM=0            cota abaixo disso vira NaN."""
    out = panel.copy()
    start = getattr(config, "DATA_START", "2018-05-01")
    rain_max = getattr(config, "MAX_CHUVA_MM_H", 100.0)
    cota_min = getattr(config, "MIN_COTA_CM", 0.0)
    out = out.loc[start:]
    rep = {}
    for c in out.columns:
        if c.startswith("chuva_") and c != "chuva_bacia" and not c.startswith("chuva_imerg_"):
            bad = out[c] > rain_max
            if bad.any():
                rep[c] = f"{int(bad.sum())} h > {rain_max:g} mm/h -> NaN"
                out.loc[bad, c] = np.nan
        elif c.startswith("cota_"):
            bad = out[c] < cota_min
            if bad.any():
                rep[c] = f"{int(bad.sum())} h < {cota_min:g} cm -> NaN"
                out.loc[bad, c] = np.nan
    if verbose:
        print(f"[clean_panel] dados a partir de {start}; ajustes: {rep or 'nenhum'}")
    return out
