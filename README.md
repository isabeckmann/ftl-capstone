# Previsão de curto prazo do nível do Rio Taquari (RS) com aprendizado de máquina

Projeto final (capstone) do **Future Tech Leaders Brasil — Space Tech Bootcamp 2026**.

Este repositório reúne o código, os notebooks e os resultados preliminares de um estudo que prevê o **nível do Rio Taquari** em Muçum e Encantado (RS) com **6, 12 e 24 horas de antecedência**, a partir de séries telemétricas da ANA e de chuva por satélite (NASA GPM IMERG). A motivação são as cheias de setembro e novembro de 2023 e de maio de 2024 no Vale do Taquari.

> **Status:** pesquisa em andamento. Os resultados abaixo são **preliminares** e podem mudar.

---

## 1. Problema e objetivo

O sistema oficial de alerta da bacia (SAH-Taquari, do Serviço Geológico do Brasil) divulga previsões de nível com poucas horas de antecedência. Este projeto avalia, com dados abertos e código reprodutível, **até que horizonte** modelos de aprendizado de máquina conseguem antecipar o nível do rio, e **onde eles falham** (principalmente nos picos das cheias extremas).

- **Estações-alvo:** Muçum (`86510000`) e Encantado (`86720000`).
- **Horizontes:** 6, 12 e 24 horas.
- **Produto:** séries de previsão, métricas por horizonte e por evento, e código aberto.

Não é um sistema de alerta em produção.

## 2. Dados

| Dado | Fonte | Uso |
|---|---|---|
| Nível (cota, cm) e chuva (mm), a cada 15 min | ANA/SGB, via API HidroWebService (Hidrotelemetria) | alvo e variáveis de entrada |
| Chuva diária estimada por satélite (`GPM_3IMERGDF`, V07, Final) | NASA GES DISC | variável de entrada opcional (cenário *com IMERG*) |
| Série diária de Muçum desde 1940 | ANA, HidroWeb (convencional) | contexto histórico; não entra no modelo |

**Estações usadas:** Muçum (86510000), Encantado (86720000) e, a montante, Linha José Júlio (86472000). Santa Tereza (86472600) só tem dados desde out/2023 e é descartada automaticamente por falta de cobertura no período de treino.

**Os dados não são versionados** (`data/` está no `.gitignore`). Para reproduzir, baixe-os conforme a seção 5 e registre a data do download:

- ANA (telemetria)
- NASA IMERG

### Pontos de qualidade dos dados

- **Os sensores falham nas maiores cheias.** Em Muçum há uma lacuna de 87 h a partir de 4/9/2023 23h, bem no pico, e Encantado também perde horas nos eventos. Por isso **set/2023 não é avaliável de forma confiável** (o notebook 05 marca `pico_perto_de_lacuna`). Nov/2023 e mai/2024 em Muçum têm cobertura de 100%.
- Leituras que a ANA marca como "ruins" (status 2) são descartadas (`DROP_BAD_STATUS` em `config.py`).
- Limpeza do painel: dados a partir de 2018-05-01; chuva acima de 100 mm/h e cota negativa viram `NaN` (erros de sensor).

## 3. Metodologia

**Painel horário.** Cota = média da hora; chuva = soma da hora; lacunas de até 3 h na cota são interpoladas.

**Alvo.** A *variação* da cota em *h* horas (`cota[t+h] − cota[t]`). A previsão absoluta é `cota[t] + variação`.

**Variáveis de entrada.** Cota atual e variações (1, 3, 6, 12 e 24 h) de todas as estações; chuva acumulada (6, 12, 24 e 48 h); no cenário *com IMERG*, acumulados de 1, 3 e 7 dias da chuva da bacia (só dias UTC já encerrados no instante da previsão, sem olhar o futuro).

**Divisão temporal (sem sorteio).**

| Conjunto | Período |
|---|---|
| Treino | 2018-05 a 2021-12 |
| Validação | 2022 |
| Teste | 2023-01 em diante (inclui as 3 cheias) |

Há uma folga de 24 h entre os blocos para evitar vazamento.

**Valores ausentes.** Só a estação-alvo precisa estar válida. As demais podem ter buracos: o gradient boosting os aceita nativamente, e as redes recebem a média do treino mais um indicador "estava faltando".

**Modelos.**

1. Persistência (repetir a cota atual), a referência mínima.
2. Regressão Ridge.
3. Gradient boosting (`HistGradientBoostingRegressor`).
4. MLP (PyTorch).
5. GRU (PyTorch), sobre as últimas 48 h.

As redes usam Adam, `ReduceLROnPlateau` e *early stopping* na validação, e a previsão final é a média de 3 sementes.

**Métricas.** RMSE, MAE e NSE; `skill_vs_persist` (1 − MSE do modelo / MSE da persistência; só vale a pena se > 0); e, por evento, erro de **pico** (cm), erro de **horário do pico** (h) e cobertura da janela. Se as cotas de alerta forem preenchidas em `config.py`, também POD, FAR e CSI.

## 4. Resultados preliminares (teste, cenário sem IMERG)

RMSE em cm no período de teste; entre parênteses, o ganho sobre a persistência (`skill_vs_persist`).

| Estação | Horizonte | Persistência | Gradient boosting | MLP | GRU |
|---|---|---|---|---|---|
| Muçum | 6 h | 52,1 | **29,4** (0,68) | 31,8 | 33,2 |
| Muçum | 12 h | 85,3 | **58,5** (0,53) | 65,3 | 60,4 |
| Muçum | 24 h | 131,8 | **102,2** (0,40) | 113,7 | 105,3 |
| Encantado | 6 h | 36,7 | 16,2 | **15,3** (0,83) | 26,3 |
| Encantado | 12 h | 61,7 | 33,6 | **32,7** (0,72) | 40,8 |
| Encantado | 24 h | 100,0 | 72,0 | **70,4** (0,50) | 74,8 |

**O que se observa**

- Os modelos superam a persistência em todos os horizontes; o ganho cai com o horizonte. O Ridge piora em 24 h (pior que a persistência).
- **Nas cheias o desempenho é pior.** Nas horas com cota ≥ 1000 cm o erro é cerca de 5 vezes maior que o geral. Em Muçum, mai/2024, o pico previsto supera o observado (gradient boosting: +253, +583 e +606 cm em 6, 12 e 24 h; as redes, mais ainda: o MLP chega a prever 4676 cm, quando o máximo observado foi 2552 cm). Em 24 h os modelos praticamente não antecipam o horário do pico.
- **Chuva do IMERG diário:** efeito pequeno e de sinal misto (entre −5 e +8 cm de RMSE), sem evidência de melhora neste desenho. A média usada ainda é da caixa retangular da região, não do polígono da bacia.
- Há poucas horas de cheia no teste (três eventos), então conclusões sobre extremos são frágeis.

## 5. Como reproduzir

**Ambiente**

```bash
conda env create -f environment.yml        # ou: conda env update -f environment.yml --prune
conda activate capstone
```

Se aparecer `CUDA error` ao treinar as redes, desative a GPU antes de importar o torch (a CPU basta para estes dados):

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

**Credenciais da ANA.** Solicite acesso à API HidroWebService à ANA e crie um arquivo `.env` na raiz (já está no `.gitignore`; nunca o envie ao GitHub):

```
ANA_ID=seu_cpf_somente_numeros
ANA_SENHA=sua_senha
```

**Baixar os dados**

```bash
# níveis e chuva (retomável); grava data/raw/ana/<codigo>_telemetrica_api.csv
python -m src.ana_api --start 2015-01-01 --end 2026-06-30

# chuva IMERG (precisa de conta no NASA Earthdata); retomável
python -m src.imerg --start 2015-01-01 --end 2026-06-30

# média sobre o polígono da bacia, sem baixar nada de novo
python -m src.imerg --series-only --basin data/raw/bacia_taquari_antas.gpkg
```

Guarde arquivos do HidroWeb diário em outra pasta (por exemplo `data/raw/ana_diario/`), para não se misturarem com a telemetria.

**Notebooks, em ordem** (`notebooks/`)

| Notebook | O que faz |
|---|---|
| `01_dados_brutos` | lê os exports da ANA, confere unidade, fuso e lacunas, salva o painel horário |
| `02_features_e_split` | variáveis, alvo e divisão temporal |
| `03_baselines` | persistência, Ridge e gradient boosting |
| `04_modelos_pytorch` | MLP e GRU |
| `05_avaliacao_eventos` | métricas, erro de pico e de horário, figuras, comparação com × sem IMERG |
| `06_chuva_bacia_imerg` | teste e conferência do IMERG |
| `07_api_ana` | teste e uso da API da ANA |

Para ligar o cenário com IMERG, coloque `USE_IMERG = True` em `src/config.py`, reinicie o kernel e rode os notebooks 03 a 05 de novo. Os dois cenários gravam previsões em pastas separadas (`outputs/preds/sem_imerg` e `outputs/preds/com_imerg`). Para um teste rápido das redes, use `CAPSTONE_FAST=1`.

Parâmetros ajustáveis em `src/config.py`: estações, horizontes, datas da divisão, janelas dos eventos, `UTC_OFFSET_HOURS`, `COTA_SCALE`, `DATA_START`, `MAX_CHUVA_MM_H`, `MIN_COTA_CM`, `MIN_COVERAGE_TRAIN`, `DROP_BAD_STATUS` e `USE_IMERG`.

## 6. Estrutura do repositório

```
.
├── environment.yml
├── src/
│   ├── config.py        # estações, horizontes, datas e parâmetros
│   ├── ana_api.py       # cliente da API da ANA (séries telemétricas)
│   ├── data.py          # leitura dos exports, painel horário, limpeza
│   ├── imerg.py         # chuva IMERG: grades recortadas e média sobre polígono
│   ├── features.py      # variáveis, alvo, divisão temporal, valores ausentes
│   ├── models.py        # MLP e GRU em PyTorch e laço de treino
│   ├── metrics.py       # RMSE, NSE, skill, erro de pico e de horário
│   ├── pipeline.py      # funções compartilhadas pelos notebooks
│   └── synthetic.py     # dados sintéticos só para testar o código
├── notebooks/           # 01 a 07
├── outputs/             # tabelas (csv) e figuras; modelos e previsões ficam fora do Git
└── data/                # não versionado (raw/ e processed/)
```

## 7. Fontes e referências

- ANA — Hidrotelemetria e HidroWeb (SNIRH) e API HidroWebService.
- SGB — Sistema de Alerta Hidrológico do Rio Taquari (SAH-Taquari).
- NASA GES DISC — GPM IMERG Final Precipitation L3 1 day 0.1° × 0.1° V07 (copie a citação oficial da página do produto).
- Marengo, J. A. et al. (2025). *Early warning services for disaster risk reduction in Brazil: the experience of CEMADEN during the floods of Rio Grande do Sul of May 2024.* International Journal of Disaster Risk Reduction.
- Nearing, G. et al. (2024). *Global prediction of extreme floods in ungauged watersheds.* Nature, 627, 559–563.
- Kratzert, F. et al. (2024). *HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin.* Hydrology and Earth System Sciences.

## 8. Autoria

- **Autoria:** `Isadora Beckmann` — `beckmannisadora@gmail.com`.
- Projeto desenvolvido no **Future Tech Leaders Brasil**, iniciativa do PNUD (UNDP) com a Agência Espacial Brasileira (AEB) e parceiros.
