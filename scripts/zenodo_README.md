# Taquari River (RS, Brazil) short-term stage forecasting: hourly dataset, test-period forecasts and operational model

Data and model accompanying the capstone project *"Short-term forecasting of the Taquari River stage (Rio Grande do Sul, Brazil) with machine learning"*. The project forecasts the river stage at **Muçum (ANA 86510000)** and **Encantado (ANA 86720000)** with **6, 8 and 12 h lead times**. It uses ANA telemetry and hourly NASA GPM IMERG Early rainfall averaged over each sub-basin.

- **Code repository:** https://github.com/isabeckmann/ftl-capstone (archived on Zenodo: doi:10.5281/zenodo.24446019)
- **Author:** Isadora Beckmann
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

> Experimental forecasts from an academic project. **They are not official warnings.** For flood warnings in the Taquari basin, see the Brazilian Geological Survey (SGB) SACE Taquari system and the state Civil Defense.

## Files

| File | Size | Description |
|---|---|---|
| `painel_horario.parquet` | 1.3 MB | Full hourly panel used for training, validation and testing (Apache Parquet, full precision) |
| `painel_horario.csv` | 6.1 MB | Same panel as CSV (values rounded to 3 decimals) |
| `previsoes_teste.csv.gz` | 21 MB (134 MB uncompressed) | Forecasts from every model in the test period, in long format (gzip-compressed CSV) |
| `modelo_hgb.joblib` | 2.6 MB | Operational gradient-boosting model (scikit-learn 1.9.1), trained on the full record |
| `README.md` | | This file |

SHA-256 checksums:

```
57f981d944053feef87af3c839d06b6c551801e5b2d8bc854cd54c2edeb3307d  painel_horario.parquet
3c0317d412493bb5c215ab1c0364bfa0a88c8f8542b248193937fc392ee3fff3  painel_horario.csv
8cbaa67549b6efacfcdad13b64fa9d1c66b9ee5a3bab0a4eeef5fbe1c6d6f86f  previsoes_teste.csv.gz
9c4a5efd9653533d6c572a5b5d3ec955329480409853ea276e503d6571e10da9  modelo_hgb.joblib
```

## General conventions

- **Time zone:** all timestamps are **local time, UTC−3** (Brasília time, no daylight saving), with no time-zone suffix. ANA telemetry is reported in local time; IMERG (UTC) was converted to UTC−3.
- **Timestamp format:** `YYYY-MM-DD HH:MM`.
- **Missing values:** empty cells in the CSV files, `NaN` in Parquet.
- **CSV:** comma-separated, decimal point, UTF-8, header on the first row.
- **Stage** (`cota`) is in **centimetres (cm)** relative to each gauge's own staff datum (not a geodetic altitude). **Rainfall** (`chuva`) is in **millimetres (mm) accumulated over one hour**.

## Stations and sub-basins

| ANA code | Station | Role in the model | Drainage area (ANA) |
|---|---|---|---|
| 86472000 | Linha José Júlio (Antas River) | upstream predictor | 13,000 km² |
| 86472600 | Santa Tereza | upstream; **not used by the models** (data only from Oct 2023, too little coverage in training) | 15,775 km² |
| 86510000 | Muçum (Taquari River) | target and predictor | 16,000 km² |
| 86720000 | Encantado (Taquari River) | target and predictor | 19,100 km² |

IMERG Early rainfall is averaged over four polygons built from the drainage areas (HydroBASINS v1c, level 12):

| Name | Definition | Approx. area |
|---|---|---|
| `bacia` | whole basin upstream of Encantado | 19,193 km² |
| `antas` | upstream of Linha José Júlio | 12,891 km² |
| `taquari_sup` | Muçum area minus Antas | 3,059 km² |
| `guapore` | Encantado area minus Muçum | 3,244 km² |

The polygon areas differ from the ANA areas by less than 1.2%.

## 1. `painel_horario.parquet` / `painel_horario.csv`: hourly panel

- **Period:** 2018-05-01 00:00 to 2026-06-08 23:00 (UTC−3), 71,064 hourly rows, regular 1 h step with no missing rows.
- **Splits used in the evaluation** (no random shuffling; 12 h gap at the end of training and validation):
  - training: 2018-05-01 to 2021-12-31;
  - validation: 2022-01-01 to 2022-12-31;
  - test: 2023-01-01 to 2026-06-08 (includes the floods of Sep 2023, Nov 2023 and May 2024).
- Both files contain the same data. The Parquet file keeps full precision (`float64`, index `instante` as `datetime64`); the CSV rounds values to 3 decimals.

| Column | Unit | Description |
|---|---|---|
| `instante` | local time (UTC−3) | Start of the hour. For ANA columns, the row labelled `T` covers readings in `[T, T+1 h)`. |
| `cota_86472000` | cm | Stage at Linha José Júlio, mean of the 15-min readings in the hour |
| `chuva_86472000` | mm | Rain gauge at Linha José Júlio, sum of the 15-min readings in the hour |
| `cota_86472600` | cm | Stage at Santa Tereza (hourly mean) |
| `chuva_86472600` | mm | Rain gauge at Santa Tereza (hourly sum) |
| `cota_86510000` | cm | Stage at Muçum (hourly mean). Forecast target. |
| `chuva_86510000` | mm | Rain gauge at Muçum (hourly sum) |
| `cota_86720000` | cm | Stage at Encantado (hourly mean). Forecast target. |
| `chuva_86720000` | mm | Rain gauge at Encantado (hourly sum) |
| `chuva_imerg_bacia` | mm | IMERG Early areal mean over the whole basin, **lagged 6 h** (see below) |
| `chuva_imerg_antas` | mm | IMERG Early areal mean over the Antas sub-basin, lagged 6 h |
| `chuva_imerg_taquari_sup` | mm | IMERG Early areal mean over the upper Taquari sub-basin, lagged 6 h |
| `chuva_imerg_guapore` | mm | IMERG Early areal mean over the Guaporé sub-basin, lagged 6 h |

### IMERG Early columns (6 h lag)

The IMERG columns are **already shifted by 6 h** to mimic real-time availability: the IMERG Early latency is about 4 h, and 6 h was used as a safe margin. **The value in row `T` is the rainfall of the hour that ended at `T − 6 h`**, i.e. the interval `[T − 7 h, T − 6 h)` in local time. A model that uses row `T` therefore only sees satellite rainfall that would already have been published at `T`. Do not shift these columns again. To recover the rainfall at its true time, shift them back by 6 h (the hour ending at local time `E` is in row `E + 6 h`).

How the IMERG values were computed:
- product `GPM_3IMERGHHE` V07 (half-hourly, 0.1°), subset over the box 52.7–49.9° W, 30.0–28.1° S;
- each polygon value is the mean of the grid cells whose centre falls inside it, weighted by cos(latitude);
- a half-hour is valid only if cells with at least 80% of the polygon's weight have data;
- hourly value = sum of the two half-hours (mm/h × 0.5 h), and both half-hours are required.

### Processing of the ANA data

- Source: telemetric 15-min series from the ANA HidroWebService API. Readings flagged with status "bad" (≥ 2) were removed.
- Stage: hourly mean; gaps of up to 3 h were linearly interpolated, and longer gaps were left empty.
- Rainfall: hourly sum, no gap filling. Values above 100 mm/h were set to missing.
- Negative stage values were set to missing (2 h at Muçum, 303 h at Encantado).
- Coverage over the full period: Muçum 97.5%, Encantado 93.1%, Linha José Júlio 85.1%, Santa Tereza 32.7% (from Oct 2023 only); IMERG ~100%.

### Known data-quality issues

- **Sep 2023:** the Muçum and Encantado sensors stopped during the rise (2023-09-04) and resumed days later, so the flood peak was not measured.
- **May 2024, Encantado:** only about 17% of the event window has data, and the peak was not measured. The Muçum peak (2,552.5 cm hourly mean) was recorded and is above the training maximum.
- **Encantado changes level from mid-2024 onwards**, probably because the staff gauge was changed after the flood. This has not been confirmed with ANA/SGB. Stage values before and after may not be directly comparable.

## 2. `previsoes_teste.csv.gz`: test-period forecasts

Gzip-compressed CSV; most tools read it directly (e.g. `pandas.read_csv("previsoes_teste.csv.gz")`), or decompress it with `gunzip`. Forecasts of every model for the **test period** (forecast issue times from 2023-01-01 00:00 to 2026-06-08 17:00, UTC−3), for both stations, three horizons and two input scenarios. Long format, one row per (scenario, model, station, horizon, issue time); 1,636,884 rows. Only rows with a forecast are included. Rows whose observation at the target time is missing are kept, with an empty `observado_cm`.

| Column | Unit | Description |
|---|---|---|
| `estacao` | | ANA code of the forecast station: `86510000` (Muçum) or `86720000` (Encantado) |
| `horizonte_h` | hours | Lead time: 6, 8 or 12 |
| `instante` | local time (UTC−3) | Issue time `t`: the forecast uses only data available up to `t` |
| `instante_alvo` | local time (UTC−3) | Target time, `t + horizonte_h` |
| `cota_atual_cm` | cm | Observed stage at the station at `t` (the persistence forecast) |
| `observado_cm` | cm | Observed stage at `instante_alvo` (empty if missing) |
| `previsto_cm` | cm | Forecast stage at `instante_alvo` |
| `modelo` | | Model (see below) |
| `cenario` | | `com_imerg`: inputs include the IMERG Early columns; `sem_imerg`: ANA data only |

**Models (`modelo`)**. All models predict the stage *change* `cota[t+h] − cota[t]`; the absolute forecast is `cota[t] +` predicted change. All were trained on 2018-05 to 2021-12, with 2022 used for validation and early stopping.

| Value | Model |
|---|---|
| `persistencia` | Persistence: `previsto_cm = cota_atual_cm` |
| `ridge` | Ridge regression, α = 1 |
| `hgb` | scikit-learn `HistGradientBoostingRegressor` (max_depth 6, learning rate 0.05, 300 iterations), one model per horizon |
| `mlp` | PyTorch MLP (hidden layers 128 and 64), mean of 3 seeds |
| `gru` | PyTorch GRU (64 units) over the last 48 h, mean of 3 seeds |

**Inputs** (tabular models), for each station used:
- current stage and its changes over 1, 3, 6, 12 and 24 h;
- rainfall sums over 6, 12, 24 and 48 h of each rain gauge and, in `com_imerg`, of each IMERG sub-basin column.

Inputs and the predicted change are clipped to the training range widened by 10% on each side. The GRU uses the raw hourly panel columns over a 48 h window.

**Notes**
- The forecasts depend on the station only through the `estacao` column. All five models are in the same file; filter by `modelo` and `cenario`.
- The GRU has slightly more rows than the other models: it builds samples from 48 h windows, so its valid issue times differ a little. To compare models on identical samples, take the inner join on (`cenario`, `estacao`, `horizonte_h`, `instante`).
- Persistence is identical in both scenarios.
- As a check, the RMSE of `hgb` / `com_imerg` from this file is 29.2, 38.9 and 57.0 cm at Muçum and 16.3, 21.5 and 33.9 cm at Encantado (6, 8, 12 h), matching the results reported in the repository.

## 3. `modelo_hgb.joblib`: operational model

This is the gradient-boosting model with IMERG Early that the live system in the repository uses. **It is not the model that produced the `hgb` forecasts in `previsoes_teste.csv.gz`.** It uses the same recipe but was **trained on the full record** (2018-05-02 00:00 to 2026-06-08 11:00, UTC−3), including the 2023–2024 floods, so it must not be evaluated on the test period above.

- **Software:** scikit-learn **1.9.1** (Python 3.11, NumPy 2.4.6, pandas 3.1.1). Load it with the same scikit-learn version; other versions may fail or give different results.
- **Format:** a Python `dict` saved with `joblib.dump(..., compress=3)`:

```
{
  "horizontes": [6, 8, 12],          # lead times (h)
  "latencia_imerg_h": 6,             # IMERG lag (h) assumed in the features
  "margem": 0.1,                     # clipping margin (fraction of the training range)
  "sklearn": "1.9.1",
  "estacoes": {
    "86510000" | "86720000": {
      "hgb": [HistGradientBoostingRegressor x 3],   # one per horizon, same order as "horizontes"
      "features": [46 feature names],               # column order of X
      "limites_entrada": Bounds(lo, hi),            # input clipping (per feature)
      "limites_saida": Bounds(lo, hi),              # clipping of the predicted change (per horizon)
      "treino_ini": "...", "treino_fim": "...", "n": <number of training samples>
    }
  }
}
```

- **Feature names** follow the pattern `cota_<code>_t` (stage at t), `cota_<code>_d<L>h` (stage change over L hours: `cota[t] − cota[t−L]`), `chuva_<code>_s<W>h` and `chuva_imerg_<area>_s<W>h` (rainfall sum over the last W hours, valid if at least 80% of the hours have data). The stations used are 86472000, 86510000 and 86720000. All features can be computed from `painel_horario.*`.
- **Prediction:** `cota[t+h] = cota[t] + clip_out(hgb[h].predict(clip_in(X)))`, where `clip_in` / `clip_out` are `np.clip(A, lo, hi)` with the stored bounds.
- **Dependency:** loading the model requires the project code (doi:10.5281/zenodo.24446019). `Bounds` is a small dataclass from the repository (`src/features.py`). To unpickle the file, run Python from the repository root (so `src.features` can be imported). The bounds are also available as plain arrays in `.lo` and `.hi`.

```python
import joblib                     # run from the root of the GitHub repository
m = joblib.load("modelo_hgb.joblib")
st = m["estacoes"]["86510000"]
# X: float32 array (n, 46) with columns in the order of st["features"]
dX = [gb.predict(st["limites_entrada"].clip(X)) for gb in st["hgb"]]
```

The file contains only model objects and metadata: no data, credentials or personal information.

## Sources

- **ANA (Agência Nacional de Águas e Saneamento Básico)**, HidroWebService API: telemetric stage and rainfall series for stations 86472000, 86472600, 86510000 and 86720000. Public data; credentials are required only to query the API and are not included here.
- **NASA GPM IMERG Early V07:** Huffman, G. J., Stocker, E. F., Bolvin, D. T., Nelkin, E. J., Tan, J. (2023). *GPM IMERG Early Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07 (GPM_3IMERGHHE)*. Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC). doi:10.5067/GPM/IMERG/3B-HH-E/07
- **HydroBASINS v1c** (HydroSHEDS), level 12, used to build the drainage-area polygons: Lehner, B., Grill, G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. *Hydrological Processes*, 27(15), 2171–2186. doi:10.1002/hyp.9740
- **scikit-learn:** Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR*, 12, 2825–2830.

## License and citation

The files in this record are released under **CC BY 4.0**. You may share and adapt them for any purpose, provided you give appropriate credit. When using them, please cite this record and the original data sources listed above (ANA, NASA GPM IMERG, HydroBASINS).

Code, processing scripts and the full methodology: https://github.com/isabeckmann/ftl-capstone (doi:10.5281/zenodo.24446019). The data files in this record are generated by `scripts/build_zenodo.py` in that repository.
