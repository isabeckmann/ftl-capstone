"""Rede de pluviômetros telemétricos da ANA na bacia (SGB-CPRM, Defesa Civil RS, CERAN): chuva observada com ~1 h de
atraso, que cobre as horas que o satélite (IMERG Early, ~6 h de atraso) ainda não mostra.

Cada estação recebe um peso por sub-bacia pelo método de Thiessen na grade de 0,1° do IMERG: cada célula da bacia é
atribuída à estação mais próxima (distância em km, com cos(lat)), e o peso da estação é a soma dos pesos de área das
células dela. Numa hora em que só parte das estações reportou, a média usa só as presentes (pesos renormalizados).

A maior parte da rede (Defesa Civil RS) só tem telemetria desde out/2025, então por enquanto ela é usada para
MONITORAR (painel ao vivo), não para treinar o modelo.

Uso:
    python -m src.pluviometros rede       # (re)calcula operacional/pluviometros.json a partir de REDE
    python -m src.pluviometros agora      # chuva das últimas horas por sub-bacia
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config, data

ARQ = config.ROOT / "operacional" / "pluviometros.json"
# código, nome, operadora — inventário da ANA (HidroInventarioEstacoes), estações telemétricas dentro da bacia
# de Encantado que reportaram chuva em 22/11/2026; uma por estação física.
REDE = [
    ("86095100", "Bom Jesus/Jaquirana", "DCRS"),
    ("86507220", "Cotiporã/Dois Lajeados", "DCRS"),
    ("2951010", "Encantado", "SGB-CPRM"),
    ("2851159", "Flores da Cunha-Antônio Prado", "DCRS"),
    ("2851044", "Guaporé", "SGB-CPRM"),
    ("2851160", "Guaporé-Anta Gorda", "DCRS"),
    ("2851072", "Ibiraiaras", "SGB-CPRM"),
    ("86560000", "Linha Colombo", "SGB-CPRM"),
    ("2951092", "Linha José Júlio", "SGB-CPRM"),
    ("2852082", "Marau", "DCRS"),
    ("2951070", "Muçum", "SGB-CPRM"),
    ("86581000", "Muçum/Encantado", "DCRS"),
    ("2951164", "Nova Roma/Farroupilha", "DCRS"),
    ("86500000", "Passo Carreiro", "SGB-CPRM"),
    ("2850047", "Passo Tainhas", "SGB-CPRM"),
    ("2851158", "Protásio Alves/Ipê", "DCRS"),
    ("2952069", "Putinga", "DCRS"),
    ("2951150", "Santa Tereza", "SGB-CPRM"),
    ("2950127", "São Francisco de Paula-Cambará do Sul", "DCRS"),
    ("2951083", "UHE 14 de Julho (jusante)", "CERAN"),
    ("2951082", "UHE Monte Claro (jusante)", "CERAN"),
    ("2852084", "União da Serra", "DCRS"),
]


def calcular_rede(coords: dict) -> dict:
    """coords = {codigo: (lat, lon)} -> estrutura salva em operacional/pluviometros.json."""
    pes = json.loads((config.ROOT / "operacional" / "pesos_imerg.json").read_text(encoding="utf-8"))
    lon, lat = np.array(pes["lon"]), np.array(pes["lat"])
    W = {k: np.asarray(v, dtype=float) for k, v in pes["pesos"].items()}
    est = [r for r in REDE if r[0] in coords]
    slat = np.array([coords[r[0]][0] for r in est])
    slon = np.array([coords[r[0]][1] for r in est])
    LON, LAT = np.meshgrid(lon, lat)
    kx = 111.32 * np.cos(np.deg2rad(LAT))[..., None]
    d = np.hypot((LON[..., None] - slon) * kx, (LAT[..., None] - slat) * 110.57)
    viz = d.argmin(axis=-1)                                       # estação mais próxima de cada célula
    pesos = {k: [float(w[viz == i].sum()) for i in range(len(est))] for k, w in W.items()}
    return {"estacoes": [{"codigo": c, "nome": n, "operadora": o, "lat": float(a), "lon": float(b)}
                         for (c, n, o), a, b in zip(est, slat, slon)],
            "pesos": {k: [round(x / sum(v), 4) if sum(v) else 0.0 for x in v] for k, v in pesos.items()}}


def carregar() -> dict:
    return json.loads(ARQ.read_text(encoding="utf-8"))


def coletar(dias: str = "DIAS_2", workers: int = 4) -> pd.DataFrame:
    """Chuva horária (mm) por estação, índice = início da hora local. Estações que falharem ficam de fora."""
    from .ana_api import AnaClient

    rede = carregar()
    cli = AnaClient(verbose=False, pause=0.1)
    cli._param_names()                                            # autentica uma vez antes das threads
    hoje = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None)

    def um(e):
        try:
            raw = cli.series(e["codigo"], date=hoje, range_=dias)
            df = data.standardize_export(raw, cota_scale=config.COTA_SCALE, utc_offset_hours=config.UTC_OFFSET_HOURS)
            if "chuva_mm" not in df or df["chuva_mm"].notna().sum() == 0:
                return e["codigo"], None
            df = df[df["chuva_mm"] <= getattr(config, "MAX_CHUVA_MM_H", 100.0)]
            return e["codigo"], data.to_hourly(df[["chuva_mm"]])["chuva_mm"]
        except Exception as exc:  # noqa: BLE001
            print(f"[pluviometros] {e['codigo']} {e['nome']}: falhou ({str(exc)[:80]})")
            return e["codigo"], None

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(um, rede["estacoes"]))
    cols = {c: s for c, s in res.items() if s is not None}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def por_area(horaria: pd.DataFrame, ultima_hora_completa: pd.Timestamp | None = None) -> pd.DataFrame:
    """Média ponderada (Thiessen) por sub-bacia; hora com menos de metade do peso presente vira NaN."""
    rede = carregar()
    cods = [e["codigo"] for e in rede["estacoes"]]
    H = horaria.reindex(columns=cods)
    if ultima_hora_completa is not None:
        H = H.loc[:ultima_hora_completa]
    out = {}
    for k, w in rede["pesos"].items():
        w = np.asarray(w)
        M = H.to_numpy(float)
        ok = ~np.isnan(M)
        den = (ok * w).sum(axis=1)
        out[k] = np.where(den >= 0.5 * w.sum(), np.nansum(M * w, axis=1) / np.where(den > 0, den, 1), np.nan)
    return pd.DataFrame(out, index=H.index)


def resumo(horaria: pd.DataFrame) -> dict | None:
    """Acumulados de 1, 6 e 24 h por sub-bacia e por estação (para o painel)."""
    if horaria is None or horaria.empty:
        return None
    agora = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None)
    ult = agora.floor("h") - pd.Timedelta(hours=1)                # última hora completa (início da hora)
    A = por_area(horaria, ult)
    rede = carregar()

    def soma(s, h):
        x = s.loc[ult - pd.Timedelta(hours=h - 1):ult]
        return None if x.notna().sum() < 0.5 * h else round(float(x.sum()), 1)

    est = []
    for e in rede["estacoes"]:
        s = horaria.get(e["codigo"])
        est.append({**{k: e[k] for k in ("codigo", "nome", "operadora", "lat", "lon")},
                    "chuva_6h": None if s is None else soma(s, 6), "chuva_24h": None if s is None else soma(s, 24),
                    "ultima": None if s is None or s.dropna().empty else s.dropna().index.max().strftime("%Y-%m-%dT%H:%M")})
    return {"ultima_hora": ult.strftime("%Y-%m-%dT%H:%M"), "n_estacoes": int(sum(e["chuva_24h"] is not None for e in est)),
            "n_rede": len(est),
            "acumulado_1h": {k: soma(A[k], 1) for k in A}, "acumulado_6h": {k: soma(A[k], 6) for k in A},
            "acumulado_24h": {k: soma(A[k], 24) for k in A},
            "bacia_horaria": {"inicio": (ult - pd.Timedelta(hours=47)).strftime("%Y-%m-%dT%H:%M"),
                              "valores": [None if not np.isfinite(v) else round(float(v), 1)
                                          for v in A["bacia"].reindex(pd.date_range(ult - pd.Timedelta(hours=47), ult,
                                                                                    freq="h")).to_numpy(float)]},
            "estacoes": est}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rede", "agora"])
    ap.add_argument("--inventario", help="CSV com codigoestacao, Latitude, Longitude (para 'rede')")
    a = ap.parse_args()
    if a.cmd == "rede":
        inv = pd.read_csv(a.inventario, dtype={"codigoestacao": str})
        coords = {r.codigoestacao: (float(r.Latitude), float(r.Longitude)) for r in inv.itertuples()}
        rede = calcular_rede(coords)
        ARQ.write_text(json.dumps(rede, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Salvo {ARQ}: {len(rede['estacoes'])} estações")
        for k, v in rede["pesos"].items():
            top = sorted(zip(v, [e["nome"] for e in rede["estacoes"]]), reverse=True)[:4]
            print(f"  {k}: " + ", ".join(f"{n} {w:.0%}" for w, n in top))
    else:
        r = resumo(coletar())
        print(json.dumps({k: v for k, v in r.items() if k not in ("estacoes", "bacia_horaria")}, ensure_ascii=False, indent=1))
        print("sem dado nas últimas 24 h:", [e["nome"] for e in r["estacoes"] if e["chuva_24h"] is None])
