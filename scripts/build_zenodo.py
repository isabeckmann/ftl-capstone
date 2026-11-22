"""Gera os arquivos de dados do registro no Zenodo (pasta zenodo/, fora do Git).

    painel_horario.parquet / .csv   painel horário de treino, validação e teste, com o IMERG Early já defasado
    previsoes_teste.csv.gz          previsões de todos os modelos no teste (formato longo, cenários com/sem IMERG)
    modelo_hgb.joblib               cópia de operacional/modelo_hgb.joblib
    README.md                       cópia de scripts/zenodo_README.md (descrição do registro, em inglês)

Pré-requisitos: data/processed/panel_hourly.parquet e chuva_imerg_early_horaria.csv (notebook 01 e
src.imerg_early) e as previsões dos notebooks 03–04 nos dois cenários (outputs/preds/com_imerg e sem_imerg).
Se algum SHA-256 mudar, atualize scripts/zenodo_README.md e rode de novo.

Uso, na raiz do repositório:
    python scripts/build_zenodo.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CAPSTONE_IMERG"] = "1"          # painel com as colunas do IMERG Early

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, pipeline  # noqa: E402

OUT = ROOT / "zenodo"
CENARIOS = ("com_imerg", "sem_imerg")


def painel() -> pd.DataFrame:
    p = pipeline.load_panel().astype("float64")
    p.index.name = "instante"
    p.to_parquet(OUT / "painel_horario.parquet")
    p.to_csv(OUT / "painel_horario.csv", float_format="%.3f", date_format="%Y-%m-%d %H:%M")
    print(f"painel: {p.shape}, {p.index.min()} a {p.index.max()}")
    return p


def previsoes() -> pd.DataFrame:
    partes = []
    for cen in CENARIOS:
        arqs = sorted((config.OUTPUTS / "preds" / cen).glob("*.parquet"))
        if not arqs:
            raise FileNotFoundError(f"Sem previsões em outputs/preds/{cen}. Rode os notebooks 03–04 nesse cenário.")
        for f in arqs:
            modelo, est = f.stem.split("__")
            d = pd.read_parquet(f)
            d = d[d["conjunto"] == "teste"]
            for h in config.HORIZONS:
                partes.append(pd.DataFrame({
                    "estacao": est, "horizonte_h": h,
                    "instante": d.index, "instante_alvo": d.index + pd.Timedelta(hours=h),
                    "cota_atual_cm": d["cota_t"].to_numpy(),
                    "observado_cm": d[f"obs_{h}h"].to_numpy(), "previsto_cm": d[f"pred_{h}h"].to_numpy(),
                    "modelo": modelo, "cenario": cen}))
    pr = pd.concat(partes, ignore_index=True)
    pr = pr[pr["previsto_cm"].notna()].sort_values(["cenario", "modelo", "estacao", "horizonte_h", "instante"])
    pr.to_csv(OUT / "previsoes_teste.csv.gz", index=False, float_format="%.2f", date_format="%Y-%m-%d %H:%M",
              compression={"method": "gzip", "mtime": 0})          # mtime fixo: o .gz sai idêntico a cada execução
    print(f"previsões: {len(pr)} linhas, {pr['instante'].min()} a {pr['instante'].max()}")

    # conferência: RMSE do gradient boosting com IMERG (deve bater com outputs/resultados_teste_com_imerg.csv)
    r = pr[(pr["modelo"] == "hgb") & (pr["cenario"] == "com_imerg")].dropna()
    rmse = r.groupby(["estacao", "horizonte_h"]).apply(
        lambda g: np.sqrt(((g["previsto_cm"] - g["observado_cm"]) ** 2).mean()))
    print("RMSE hgb com_imerg (cm):\n" + rmse.round(1).to_string())
    return pr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    painel()
    previsoes()
    shutil.copy2(config.ROOT / "operacional" / "modelo_hgb.joblib", OUT / "modelo_hgb.joblib")
    shutil.copy2(ROOT / "scripts" / "zenodo_README.md", OUT / "README.md")
    print("\nSHA-256 (devem bater com scripts/zenodo_README.md):")
    for nome in ("painel_horario.parquet", "painel_horario.csv", "previsoes_teste.csv.gz", "modelo_hgb.joblib"):
        print(f"{sha256(OUT / nome)}  {nome}")
