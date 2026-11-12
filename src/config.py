"""Configuração central do projeto: caminhos, estações, horizontes e divisão temporal."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANA_DIR = RAW / "ana" 
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
for _p in (ANA_DIR, PROCESSED, OUTPUTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

STATIONS = {
    "86472000": {"nome": "Linha José Júlio", "papel": "montante"},
    "86472600": {"nome": "Santa Tereza", "papel": "montante"},
    "86510000": {"nome": "Muçum", "papel": "alvo"},
    "86720000": {"nome": "Encantado", "papel": "alvo"},
}
TARGETS = [c for c, v in STATIONS.items() if v["papel"] == "alvo"]

HORIZONS = [6, 12, 24]

UTC_OFFSET_HOURS = 0
COTA_SCALE = 1.0

TRAIN_END = "2021-12-31 23:00"
VAL_END = "2022-12-31 23:00"
TEST_START = "2023-01-01 00:00"

EVENTS = {
    "set/2023": ("2023-09-01", "2023-09-12"),
    "nov/2023": ("2023-11-10", "2023-11-25"),
    "mai/2024": ("2024-04-27", "2024-05-15"),
}

ALERT_LEVEL_CM = {"86510000": None, "86720000": None}

# Features / sequências
LAGS = (1, 3, 6, 12, 24)          # variações da cota em t - lag
RAIN_WINDOWS = (6, 12, 24, 48)    # acumulados de chuva (h)
SEQ_LEN = 48                      # horas de histórico para a GRU

# Limites contra extrapolação: entradas e variação prevista ficam na faixa do TREINO,
# alargada por CLIP_MARGIN x amplitude (0.10 = 10% da amplitude para cada lado).
CLIP_MARGIN = 0.10
USE_IMERG = True

