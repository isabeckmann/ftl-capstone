"""Rotina ao vivo: coleta ANA + IMERG Early, prevê a cota em +6, +8 e +12 h e grava um JSON para o site.

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


def montar_json(panel, chuva_utc, prev, ultimas, modelo, avisos) -> dict:
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
    return {"gerado_em": _iso(agora), "fuso": "UTC-3", "inicio": _iso(idx[0]), "n": len(idx),
            "horizontes": modelo["horizontes"], "estacoes": est, "chuva": chuva, "avisos": avisos,
            "modelo": {"tipo": "gradient boosting (HGB) + IMERG Early",
                       "treino_fim": min(v["treino_fim"] for v in modelo["estacoes"].values())[:10],
                       "latencia_imerg_h": modelo["latencia_imerg_h"]}}


def gravar_historico(path, prev, gerado_em):
    rows = [{"t_ref": _iso(p["t_ref"]), "estacao": code, "h": q["h"], "t_alvo": _iso(q["t"]),
             "cota_t": round(p["cota_t"], 1), "cota_prev": round(q["cota"], 1), "gerado_em": gerado_em}
            for code, p in prev.items() if "previsoes" in p for q in p["previsoes"]]
    if not rows:
        return
    new = pd.DataFrame(rows)
    path = Path(path)
    if path.exists():
        new = pd.concat([pd.read_csv(path, dtype={"estacao": str}), new])
    new.drop_duplicates(["t_ref", "estacao", "h"], keep="first").sort_values(["t_ref", "estacao", "h"]).to_csv(path, index=False)


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
    prev = prever(panel, modelo, ultimas)
    js = montar_json(panel, chuva, prev, ultimas, modelo, avisos)
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(json.dumps(js, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if historico:
        gravar_historico(historico, prev, js["gerado_em"])
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
    print("Pronto:", a.saida)
