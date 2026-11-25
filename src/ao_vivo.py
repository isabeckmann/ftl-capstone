"""Rotina ao vivo: coleta ANA + pluviômetros da bacia + IMERG Early + chuva prevista (GFS/ECMWF), prevê a cota em +6, +8 e +12 h e grava um JSON para o site.

Roda de hora em hora no GitHub Actions (.github/workflows/ao_vivo.yml), mas também funciona localmente:
    python -m src.ao_vivo --saida web/agora.json [--historico previsoes.csv]

Credenciais:
    ANA        ANA_ID e ANA_SENHA (variáveis de ambiente ou .env)
    Earthdata  ~/.netrc (machine urs.earthdata.nasa.gov ...). No Actions, o workflow escreve o .netrc a partir dos secrets.

Mesmas regras do treino: painel horário em hora local (UTC−3), lacunas de até 3 h interpoladas, IMERG entra com
config.IMERG_LATENCIA_H horas de atraso, entradas e variação prevista limitadas à faixa do treino.
Se o IMERG falhar, o modelo prevê assim mesmo (o gradient boosting aceita falta) e o JSON registra o aviso.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CAPSTONE_IMERG", "1")
from . import config, data, features  # noqa: E402

DIR = config.ROOT / "operacional"
HORAS_GRAFICO = 72


# ----------------------------------------------------------------------------- coleta
def coletar_ana(dias: str = "DIAS_7") -> tuple[dict, dict]:
    from .ana_api import AnaClient

    cli = AnaClient(verbose=False)
    hoje = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None)
    frames, ultimas = {}, {}
    for code in config.STATIONS:
        raw = None
        for tentativa in (1, 2):                 # a API às vezes demora a autenticar; uma segunda tentativa resolve
            try:
                raw = cli.series(code, date=hoje, range_=dias)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[ao_vivo] ANA {code}: falhou na tentativa {tentativa} ({str(exc)[:100]})")
                cli._token = None
        if raw is None:
            continue
        if raw.empty:
            print(f"[ao_vivo] ANA {code}: sem leituras")
            continue
        df = data.standardize_export(raw, cota_scale=config.COTA_SCALE, utc_offset_hours=config.UTC_OFFSET_HOURS)
        frames[code] = df
        ok = df["cota_cm"].dropna() if "cota_cm" in df else pd.Series(dtype=float)
        ultimas[code] = ok.index.max() if len(ok) else None
    return frames, ultimas


def coletar_imerg(horas: int = 84, workers: int = 8) -> pd.DataFrame:
    """Chuva horária (mm) por sub-bacia das últimas `horas`, índice = fim da hora em UTC."""
    import xarray as xr

    from . import imerg_early as ie

    pes = json.loads((DIR / "pesos_imerg.json").read_text(encoding="utf-8"))
    fim = pd.Timestamp.now(tz="UTC").tz_localize(None)
    gr = ie.list_granules(fim - pd.Timedelta(hours=horas), fim)
    if gr.empty:
        return pd.DataFrame()
    sess = ie._session()

    def um(row):
        try:
            return row.t, ie.fetch_subset(row.gid, sess, retries=3).values
        except PermissionError:
            raise
        except Exception:  # noqa: BLE001
            return row.t, None

    with ThreadPoolExecutor(workers) as ex:
        res = [r for r in ex.map(um, gr.itertuples()) if r[1] is not None]
    if not res:
        return pd.DataFrame()
    res.sort(key=lambda r: r[0])
    da = xr.DataArray(np.stack([v for _, v in res]), dims=("time", "lat", "lon"),
                      coords={"time": [t for t, _ in res], "lat": pes["lat"], "lon": pes["lon"]})
    s = ie.hourly_series(da, weights=pes["pesos"])
    return s.dropna(how="all")


def coletar_prevista() -> dict:
    """Chuva prevista (GFS e ECMWF via Open-Meteo) por sub-bacia: {modelo: DataFrame, índice = fim da hora UTC}."""
    from .chuva_prevista import prevista_agora

    return prevista_agora()


# ----------------------------------------------------------------------------- previsão
def _hora_ref(panel: pd.DataFrame, alvo: str, ultima_leitura) -> pd.Timestamp | None:
    """Última hora COMPLETA com cota válida (a hora h vai de h a h+1; a média de uma hora pela metade é enviesada)."""
    s = panel[f"cota_{alvo}"].dropna()
    if ultima_leitura is not None:
        s = s[s.index + pd.Timedelta(hours=1) <= ultima_leitura + pd.Timedelta(minutes=15)]
    return s.index.max() if len(s) else None


def prever(panel: pd.DataFrame, modelo: dict, ultimas: dict) -> dict:
    from .operacional import predict

    F = features.tabular_features(panel)
    out = {}
    for alvo, m in modelo["estacoes"].items():
        t = _hora_ref(panel, alvo, ultimas.get(alvo))
        if t is None:
            out[alvo] = {"erro": "sem leitura recente da cota"}
            continue
        x = F.reindex(columns=m["features"]).loc[[t]].to_numpy("float32")
        req = [i for i, c in enumerate(m["features"]) if c.startswith(f"cota_{alvo}_")]
        if np.isnan(x[0, req]).any():
            out[alvo] = {"erro": "histórico de 24 h da cota incompleto", "t_ref": t}
            continue
        bi = m["limites_entrada"]
        fora = [m["features"][i] for i in np.where((x[0] < bi.lo) | (x[0] > bi.hi))[0]]
        delta = predict(m["hgb"], bi, m["limites_saida"], x)[0]
        base = float(panel.at[t, f"cota_{alvo}"])
        faltando = [c for c, v in zip(m["features"], x[0]) if np.isnan(v)]
        out[alvo] = {"t_ref": t, "cota_t": base,
                     "previsoes": [{"h": int(h), "t": t + pd.Timedelta(hours=int(h)), "cota": base + float(d), "delta": float(d)}
                                   for h, d in zip(modelo["horizontes"], delta)],
                     "fora_da_faixa": fora, "faltando": faltando}
    return out


# ----------------------------------------------------------------------------- saída
def _r(v, nd=0):
    return [None if v is None or not np.isfinite(x) else round(float(x), nd) for x in np.asarray(v, dtype="float64")]


def _iso(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")


NOMES_MODELOS = {"gfs_seamless": "GFS (NOAA)", "ecmwf_ifs025": "ECMWF"}
LIMIARES_24H = (40, 80)
"""mm previstos em 24 h na bacia. No histórico (IMERG 2018–2026), com >= 40 mm em 24 h Muçum subiu 2 m ou mais nas
24 h seguintes em ~46% das vezes; com >= 80 mm, em ~62%."""


def resumo_prevista(prevista: dict, imerg_ultima_utc, idx_local) -> dict | None:
    """Acumulados previstos e série horária da bacia (para o gráfico), em hora local (rótulo = início da hora)."""
    if not prevista:
        return None
    agora = pd.Timestamp.now(tz="UTC").tz_localize(None)
    off = pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS)
    modelos = {}
    for m, df in prevista.items():
        fut = df[df.index > agora]                       # horas que terminam depois de agora
        rec = df[(df.index <= agora) & (df.index > (imerg_ultima_utc if imerg_ultima_utc is not None
                                                     else agora - pd.Timedelta(hours=6)))]
        loc = df.copy()
        loc.index = loc.index - pd.Timedelta(hours=1) + off
        ini = idx_local[0]
        serie = loc["bacia"].loc[ini:ini + pd.Timedelta(hours=len(idx_local) + 47)]
        modelos[m] = {"nome": NOMES_MODELOS.get(m, m), "inicio": _iso(serie.index.min()), "bacia": _r(serie.values, 1),
                      "acumulado_12h": {k: round(float(fut[k].iloc[:12].sum()), 1) for k in df.columns},
                      "acumulado_24h": {k: round(float(fut[k].iloc[:24].sum()), 1) for k in df.columns},
                      "recente_sem_imerg": {k: round(float(rec[k].sum()), 1) for k in df.columns},
                      "primeira_hora_futura": _iso(fut.index.min() - pd.Timedelta(hours=1) + off) if len(fut) else None}
    p24 = max(v["acumulado_24h"]["bacia"] for v in modelos.values())
    nivel = "alto" if p24 >= LIMIARES_24H[1] else "atencao" if p24 >= LIMIARES_24H[0] else "normal"
    return {"modelos": modelos, "nivel": nivel, "max_24h_bacia": p24, "limiares_24h": list(LIMIARES_24H),
            "fonte": "Open-Meteo (GFS 0,25° e ECMWF IFS 0,25°), média ponderada dos pontos de grade na bacia"}


def montar_json(panel, chuva_utc, prev, ultimas, modelo, avisos, prevista=None, pluvio=None) -> dict:
    agora = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None)
    fim = panel.index.max()
    idx = pd.date_range(fim - pd.Timedelta(hours=HORAS_GRAFICO - 1), fim, freq="h")
    est = {}
    for code, meta in config.STATIONS.items():
        col = f"cota_{code}"
        e = {"nome": meta["nome"], "papel": meta["papel"], "ultima_leitura": _iso(ultimas.get(code)),
             "obs": _r(panel[col].reindex(idx)) if col in panel else None,
             "chuva_24h_mm": (round(float(panel[f"chuva_{code}"].iloc[-24:].sum(min_count=1)), 1)
                              if f"chuva_{code}" in panel and panel[f"chuva_{code}"].iloc[-24:].notna().any() else None)}
        if code in prev:
            p = prev[code]
            e.update({"t_ref": _iso(p.get("t_ref")), "cota_t": None if "cota_t" not in p else round(p["cota_t"]),
                      "erro": p.get("erro"), "fora_da_faixa": p.get("fora_da_faixa", []),
                      "previsoes": [{"h": q["h"], "t": _iso(q["t"]), "cota": round(q["cota"]), "delta": round(q["delta"])}
                                    for q in p.get("previsoes", [])],
                      "cotas_referencia": (config.COTAS_REFERENCIA or {}).get(code)})
        est[code] = e
    chuva = None
    if chuva_utc is not None and len(chuva_utc):
        loc = chuva_utc.copy()       # hora de verdade (sem a defasagem do modelo), rótulo = início da hora local
        loc.index = loc.index - pd.Timedelta(hours=1) + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS)
        ult = loc.index.max()
        chuva = {"inicio": _iso(idx[0]), "bacia": _r(loc["bacia"].reindex(idx), 1),
                 "ultima_hora": _iso(ult),
                 "acumulado_24h": {k: round(float(loc[k].loc[ult - pd.Timedelta(hours=23):].sum()), 1) for k in loc.columns},
                 "acumulado_72h": {k: round(float(loc[k].loc[ult - pd.Timedelta(hours=71):].sum()), 1) for k in loc.columns}}
    imerg_ult = chuva_utc.index.max() if chuva_utc is not None and len(chuva_utc) else None
    chuva_prev = resumo_prevista(prevista, imerg_ult, idx)
    if pluvio and (pluvio["acumulado_24h"].get("bacia") or 0) >= LIMIARES_24H[0]:
        avisos = avisos + [f"Os pluviômetros registraram {pluvio['acumulado_24h']['bacia']:.0f} mm em 24 h na bacia "
                           f"(até {pluvio['ultima_hora'][11:16]}). Parte dessa chuva ainda não aparece no satélite."]
    if chuva_prev and chuva_prev["nivel"] != "normal":
        avisos = avisos + [f"Chuva prevista de até {chuva_prev['max_24h_bacia']:.0f} mm em 24 h na bacia. Com 40 mm ou "
                           "mais, no histórico o rio subiu 2 m ou mais em Muçum em cerca de metade das vezes — e a "
                           "previsão de cota acima ainda NÃO considera essa chuva futura."]
    return {"gerado_em": _iso(agora), "fuso": "UTC-3", "inicio": _iso(idx[0]), "n": len(idx),
            "horizontes": modelo["horizontes"], "estacoes": est, "chuva": chuva, "chuva_prevista": chuva_prev,
            "pluviometros": pluvio,
            "avisos": avisos,
            "modelo": {"tipo": "gradient boosting (HGB) + IMERG Early",
                       "treino_fim": min(v["treino_fim"] for v in modelo["estacoes"].values())[:10],
                       "latencia_imerg_h": modelo["latencia_imerg_h"]}}


COLS_HIST = ["t_ref", "estacao", "h", "t_alvo", "cota_t", "cota_prev", "gerado_em", "cota_obs"]
DIAS_VERIFICACAO = 7


def gravar_historico(path, prev, gerado_em, panel=None, ultimas=None) -> pd.DataFrame:
    """Acrescenta as previsões desta rodada ao CSV e preenche cota_obs das previsões cujo horário-alvo já passou.

    cota_obs é a média horária da cota em t_alvo, a mesma definição de cota_t e do alvo do treino. Só entram horas
    completas (como em _hora_ref) e o valor, uma vez gravado, não muda: o CSV guarda a conferência para sempre, mesmo
    depois que a hora sai da janela de 7 dias que a API da ANA devolve.
    """
    rows = [{"t_ref": _iso(p["t_ref"]), "estacao": code, "h": q["h"], "t_alvo": _iso(q["t"]),
             "cota_t": round(p["cota_t"], 1), "cota_prev": round(q["cota"], 1), "gerado_em": gerado_em}
            for code, p in prev.items() if "previsoes" in p for q in p["previsoes"]]
    path = Path(path)
    df = pd.DataFrame(rows, columns=COLS_HIST[:-1])
    if path.exists():
        df = pd.concat([pd.read_csv(path, dtype={"estacao": str}), df])
    if df.empty:
        return df
    df = df.drop_duplicates(["t_ref", "estacao", "h"], keep="first").sort_values(["t_ref", "estacao", "h"])
    df = df.reindex(columns=COLS_HIST).reset_index(drop=True)
    if panel is not None:
        t_alvo = pd.to_datetime(df["t_alvo"])
        for code in df["estacao"].unique():
            col = f"cota_{code}"
            if col not in panel:
                continue
            s = panel[col].dropna()
            ult = (ultimas or {}).get(code)
            if ult is not None:
                s = s[s.index + pd.Timedelta(hours=1) <= ult + pd.Timedelta(minutes=15)]
            sel = (df["estacao"] == code) & df["cota_obs"].isna()
            df.loc[sel, "cota_obs"] = t_alvo[sel].map(s).round(1)
    df.to_csv(path, index=False)
    return df


def verificacao(hist: pd.DataFrame | None, panel: pd.DataFrame, dias: int = DIAS_VERIFICACAO) -> dict | None:
    """Previsões ao vivo já conferidas: métricas por estação e horizonte (todo o histórico) e pares dos últimos dias."""
    if hist is None or hist.empty or "cota_obs" not in hist:
        return None
    v = hist.dropna(subset=["cota_obs", "cota_prev", "cota_t"]).copy()
    if v.empty:
        return None
    v["t_alvo"] = pd.to_datetime(v["t_alvo"])
    fim = panel.index.max()
    idx = pd.date_range(fim - pd.Timedelta(hours=24 * dias - 1), fim, freq="h")
    est = {}
    for code, g in v.groupby("estacao"):
        col = f"cota_{code}"
        metricas, pares = [], []
        for h, gh in g.groupby("h"):
            e, ep = gh["cota_prev"] - gh["cota_obs"], gh["cota_t"] - gh["cota_obs"]
            mse, msep = float((e ** 2).mean()), float((ep ** 2).mean())
            metricas.append({"h": int(h), "n": int(len(gh)), "mae": round(float(e.abs().mean()), 1),
                             "vies": round(float(e.mean()), 1), "rmse": round(mse ** .5, 1),
                             "rmse_persist": round(msep ** .5, 1),
                             "skill": round(1 - mse / msep, 2) if msep > 0 else None})
            r = gh[gh["t_alvo"] >= idx[0]]
            pares.append({"h": int(h),
                          "i": [int((t - idx[0]) / pd.Timedelta(hours=1)) for t in r["t_alvo"]],
                          "prev": _r(r["cota_prev"]), "obs": _r(r["cota_obs"]), "base": _r(r["cota_t"])})
        est[code] = {"metricas": metricas, "pares": pares,
                     "obs": _r(panel[col].reindex(idx)) if col in panel else None}
    return {"inicio": _iso(idx[0]), "n": len(idx), "dias": dias,
            "desde": _iso(pd.to_datetime(v["t_ref"]).min()), "estacoes": est}


def rodar(saida, historico=None):
    import joblib

    modelo = joblib.load(DIR / "modelo_hgb.joblib")
    avisos = []
    frames, ultimas = coletar_ana()
    if not frames:
        raise RuntimeError("A ANA não devolveu nenhuma estação.")
    panel = data.clean_panel(data.fill_short_gaps(data.panel_from_frames(frames), max_gap=3), verbose=False)
    usadas = {f.split("_")[1] for m in modelo["estacoes"].values() for f in m["features"] if f.startswith("cota_")}
    for code in config.STATIONS:                       # colunas de estações fora do ar continuam existindo (NaN)
        for k in ("cota", "chuva"):
            if f"{k}_{code}" not in panel:
                panel[f"{k}_{code}"] = np.nan
        recente = panel[f"cota_{code}"].iloc[-6:].notna().any()
        if code in usadas and not recente:
            avisos.append(f"{config.STATIONS[code]['nome']} sem leituras nas últimas 6 h: a previsão perde a "
                          "informação dessa estação e fica menos confiável.")
    try:
        chuva = coletar_imerg()
    except Exception as exc:  # noqa: BLE001
        chuva = pd.DataFrame()
        avisos.append(f"IMERG Early indisponível ({type(exc).__name__}); previsão feita sem a chuva de satélite.")
    if len(chuva):
        panel = data.add_imerg_hourly(panel, csv=chuva)
    else:
        for k in ("bacia", "antas", "taquari_sup", "guapore"):
            panel[f"chuva_imerg_{k}"] = np.nan
    try:
        prevista = coletar_prevista()
    except Exception as exc:  # noqa: BLE001
        prevista = {}
        avisos.append(f"Previsão de chuva (Open-Meteo) indisponível ({type(exc).__name__}).")
    try:
        from . import pluviometros

        pluvio = pluviometros.resumo(pluviometros.coletar())
    except Exception as exc:  # noqa: BLE001
        pluvio = None
        avisos.append(f"Pluviômetros da bacia indisponíveis ({type(exc).__name__}).")
    prev = prever(panel, modelo, ultimas)
    js = montar_json(panel, chuva, prev, ultimas, modelo, avisos, prevista, pluvio)
    if historico:
        hist = gravar_historico(historico, prev, js["gerado_em"], panel, ultimas)
        try:
            js["verificacao"] = verificacao(hist, panel)
        except Exception as exc:  # noqa: BLE001  (a conferência nunca deve derrubar a previsão)
            print(f"[ao_vivo] verificação falhou: {exc!r}")
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(json.dumps(js, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return js


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default=str(config.ROOT / "web" / "agora.json"))
    ap.add_argument("--historico", default=None)
    a = ap.parse_args()
    js = rodar(a.saida, a.historico)
    for code, e in js["estacoes"].items():
        if e.get("previsoes"):
            pv = ", ".join(f"+{q['h']}h {q['cota']} cm" for q in e["previsoes"])
            print(f"{e['nome']}: {e['cota_t']} cm às {e['t_ref']} -> {pv}" + (f" | fora da faixa: {e['fora_da_faixa']}" if e['fora_da_faixa'] else ""))
        elif e.get("erro"):
            print(f"{e['nome']}: {e['erro']}")
    print("chuva IMERG até:", js["chuva"] and js["chuva"]["ultima_hora"], "| avisos:", js["avisos"])
    vf = js.get("verificacao")
    if vf:
        for code, e in vf["estacoes"].items():
            print(f"conferência {code} desde {vf['desde']}: " + ", ".join(
                f"+{m['h']}h n={m['n']} RMSE {m['rmse']} cm (persist. {m['rmse_persist']})" for m in e["metricas"]))
    pl = js.get("pluviometros")
    if pl:
        print(f"pluviômetros ({pl['n_estacoes']}/{pl['n_rede']} estações, até {pl['ultima_hora']}): bacia 6 h "
              f"{pl['acumulado_6h']['bacia']} mm, 24 h {pl['acumulado_24h']['bacia']} mm")
    cp = js.get("chuva_prevista")
    if cp:
        for m, v in cp["modelos"].items():
            print(f"chuva prevista {v['nome']}: 12 h {v['acumulado_12h']['bacia']} mm, 24 h {v['acumulado_24h']['bacia']} mm"
                  f" (recente sem IMERG: {v['recente_sem_imerg']['bacia']} mm) | nível: {cp['nivel']}")
    print("Pronto:", a.saida)
