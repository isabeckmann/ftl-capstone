# Previsão de curto prazo do nível do Rio Taquari (RS) com aprendizado de máquina

Projeto final (capstone) do **Future Tech Leaders Brasil — Space Tech Bootcamp 2026**.

O projeto prevê o **nível (cota) do Rio Taquari** em Muçum e Encantado (RS) com **6, 8 e 12 horas de antecedência**. Usa as séries telemétricas da ANA e a chuva horária de satélite da NASA (GPM IMERG Early) em cada sub-bacia. A motivação são as cheias de setembro e novembro de 2023 e de maio de 2024 no Vale do Taquari.

O repositório tem quatro partes:
1. a avaliação histórica (notebooks e tabelas);
2. um modelo operacional;
3. uma rotina **ao vivo**, que roda de hora em hora no GitHub Actions;
4. um painel interativo, que mostra a previsão atual, a chuva observada e prevista na bacia e os resultados da avaliação.

> **Previsão experimental, de um projeto acadêmico. Não é alerta oficial.** Para decisões, siga o [SACE Taquari (SGB)](https://sace.sgb.gov.br/index_bacias_monitoradas.php?getbacia=btaquari) e a Defesa Civil (telefone 199).

---

## 1. Problema e objetivo

O sistema oficial de alerta da bacia (SACE Taquari, do Serviço Geológico do Brasil) divulga previsões de nível com poucas horas de antecedência. Com dados abertos e código reprodutível, este projeto avalia:

- quanto modelos de aprendizado de máquina melhoram a **persistência** (repetir o nível atual) em 6, 8 e 12 h;
- quanto a **chuva de satélite horária** acrescenta;
- **onde os modelos falham**, principalmente nos picos das cheias extremas;
- se dados **mais imediatos** (previsão de chuva do GFS, mais pluviômetros) melhoram a previsão.

**Estações-alvo:** Muçum (`86510000`) e Encantado (`86720000`).

## 2. Principais resultados

Teste de jan/2023 a jun/2026, com as três cheias. O skill é 1 − MSE do modelo / MSE da persistência: acima de 0, o modelo é melhor que a persistência.

| Estação | Horizonte | RMSE da persistência | RMSE do gradient boosting | Skill |
|---|---|---|---|---|
| Muçum | 6 h | 54 cm | **29 cm** | 0,71 |
| Muçum | 8 h | 67 cm | **39 cm** | 0,66 |
| Muçum | 12 h | 88 cm | **57 cm** | 0,58 |
| Encantado | 6 h | 37 cm | **16 cm** | 0,81 |
| Encantado | 8 h | 47 cm | **22 cm** | 0,79 |
| Encantado | 12 h | 63 cm | **34 cm** | 0,71 |

- **O gradient boosting é o melhor modelo** na maioria dos casos. O MLP empata em Encantado. A Ridge quase empata com a persistência em Muçum, em 12 h.
- **O IMERG Early horário, por sub-bacia, reduz o RMSE do gradient boosting em 2% a 5%.** Também reduz a superestimação do pico. Em Muçum, nov/2023, em 12 h, o erro cai de +350 para +277 cm. O IMERG Final diário não tinha efeito consistente.
- **O pico das cheias extremas não é confiável.** Os modelos superestimam o pico em 1 a 5 m em 12 h e erram o horário por várias horas. Em mai/2024, o pico de Muçum (2553 cm) ficou acima do máximo visto no treino (2205 cm).
- **Dados mais imediatos não melhoraram o modelo.** Nem a chuva futura perfeita reduziu o erro global em até 12 h, porque a cota das estações a montante já carrega esse sinal. A previsão de chuva e os pluviômetros extras entram, então, como **monitoramento** no painel.

Tabelas completas: `outputs/resultados_teste_com_imerg.csv`, `outputs/resultados_eventos_com_imerg.csv`, `outputs/comparacao_com_sem_imerg.csv` e `outputs/experimento_chuva_prevista/`.

## 3. Dados

| Dado | Fonte | Uso |
|---|---|---|
| Cota (cm) e chuva (mm) a cada 15 min | ANA, API HidroWebService (telemetria) | alvo e variáveis; ao vivo |
| Chuva de satélite meia-horária, 0,1° (`GPM_3IMERGHHE` V07, Early) | NASA GES DISC, via OPeNDAP do Earthdata | variáveis (com 6 h de defasagem); ao vivo |
| Chuva de satélite diária, 0,1° (`GPM_3IMERGDF` V07, Final) | NASA GES DISC | cenário antigo; mapa do painel |
| Áreas de drenagem | HydroSHEDS, HydroBASINS v1c, nível 12 | polígonos da bacia e das sub-bacias |
| Previsão de chuva (GFS e ECMWF IFS, 0,25°) | Open-Meteo | experimento; monitoramento ao vivo |
| 22 pluviômetros telemétricos na bacia (SGB-CPRM, Defesa Civil RS, CERAN) | ANA, HidroWebService | experimento; monitoramento ao vivo |
| Cotas de atenção, alerta e inundação (Muçum) | SGB, SACE Taquari | painel ao vivo |

**Estações do modelo:**
- Muçum (`86510000`) e Encantado (`86720000`), as estações-alvo;
- Linha José Júlio (`86472000`), a montante, no rio das Antas;
- Santa Tereza (`86472600`) só tem dados desde out/2023 e é descartada automaticamente por falta de cobertura no treino.

**Sub-bacias da chuva:**
- **Antas:** a montante de Linha José Júlio;
- **Taquari superior:** área de Muçum menos a de Antas;
- **Guaporé:** área de Encantado menos a de Muçum;
- **bacia:** Encantado inteira (19.193 km²).

Os polígonos diferem menos de 1,2% das áreas informadas pela ANA.

**Os dados não são versionados** (`data/` está no `.gitignore`). Para reproduzir, baixe-os conforme a seção 6.

### Pontos de qualidade dos dados

- **set/2023: o pico não foi medido.** Os sensores de Muçum e de Encantado param na subida (04/09) e só voltam dias depois. O erro de pico desse evento não é avaliável.
- **mai/2024 em Encantado:** cobertura de ~17% da janela do evento. Os picos avaliáveis são os de nov/2023 (as duas estações) e mai/2024 (só Muçum).
- **Encantado muda de patamar a partir de meados de 2024**, provavelmente por mudança de régua depois da cheia. É preciso confirmar com a ANA ou o SGB.
- **Limpeza:**
  - dados a partir de 2018-05-01;
  - chuva acima de 100 mm/h e cota negativa viram `NaN`;
  - leituras com status "ruim" da ANA são descartadas;
  - só lacunas de até 3 h são interpoladas.
- **Fuso:** a ANA usa horário local (UTC−3), confirmado por correlação cruzada com o IMERG. O IMERG usa UTC.

## 4. Metodologia

- **Alvo:** a *variação* da cota em *h* horas (`cota[t+h] − cota[t]`). A previsão absoluta é `cota[t] + variação`.
- **Variáveis:**
  - cota atual e variações em 1, 3, 6, 12 e 24 h de cada estação;
  - chuva acumulada em 6, 12, 24 e 48 h de cada pluviômetro;
  - o mesmo acumulado para o IMERG Early de cada sub-bacia. Cada hora de chuva só entra **6 h depois** de terminar (latência real: ~4 h), então o treino só usa o que existiria em tempo real.
- **Limites contra extrapolação:** entradas e variação prevista ficam limitadas à faixa do treino, com 10% de margem.
- **Divisão temporal, sem sorteio:** treino de 2018-05 a 2021-12, validação em 2022, teste de 2023-01 a 2026-06. Há uma folga igual ao maior horizonte entre os blocos.
- **Modelos:**

  | Modelo | Configuração |
  |---|---|
  | Persistência | referência mínima |
  | Ridge | α = 1 |
  | Gradient boosting (`HistGradientBoostingRegressor`) | profundidade 6, taxa 0,05, 300 iterações |
  | MLP (PyTorch) | camadas de 128 e 64 |
  | GRU (PyTorch) | 64 unidades, sobre as últimas 48 h |

  As redes usam Adam, `ReduceLROnPlateau` e early stopping, com a média de 3 sementes.
- **Métricas:**
  - RMSE, MAE, NSE e skill contra a persistência;
  - por evento: erro de pico (cm), erro do horário do pico (h) e cobertura da janela.

O registro completo, com tabelas, decisões e limitações, fica em `docs/enquadramento_artigo.md` (local).

## 5. Sistema ao vivo (experimental)

```
GitHub Actions (de hora em hora)
  ├─ ANA: cota e chuva das 4 estações (últimos 7 dias)
  ├─ NASA IMERG Early: chuva por sub-bacia (últimas 84 h)
  ├─ Open-Meteo: chuva prevista GFS e ECMWF (próximas 48 h)
  ├─ ANA: 22 pluviômetros da bacia, com média por polígonos de Thiessen
  └─ modelo operacional (operacional/modelo_hgb.joblib) → previsão de +6, +8 e +12 h
        ↓
  agora.json e previsoes.csv no branch "dados"
        ↓
  site estático (web/index.html, Cloudflare Workers) lê o agora.json a cada 10 min
```

- **Modelo operacional** (`src/operacional.py`): o gradient boosting com IMERG Early, treinado com todo o histórico (até jun/2026). Antes de salvar, uma checagem treina até jun/2025 e testa depois disso. O skill ficou em 0,89, 0,83 e 0,67 em Muçum e em 0,90, 0,86 e 0,72 em Encantado (6, 8 e 12 h).
- **Avisos:**
  - quando uma estação usada pelo modelo fica 6 h sem leituras;
  - quando os dados passam de 3 h sem atualização;
  - quando a chuva prevista ou observada passa de **40 mm em 24 h** na bacia. No histórico, em ~46% dessas situações Muçum subiu 2 m ou mais nas 24 h seguintes.
- **Secrets do workflow** (`.github/workflows/ao_vivo.yml`): `ANA_ID`, `ANA_SENHA`, `EARTHDATA_USER` e `EARTHDATA_PASS`.
- **Frequência:** o cron do GitHub atrasa e pula rodadas; sozinho, ele roda só 3 ou 4 vezes por dia. Por isso, quem dispara o workflow de hora em hora (minuto 17, UTC) é um **Cron Trigger do Cloudflare Workers** (`worker/index.js`). Ele chama o endpoint `workflow_dispatch` da API do GitHub. O `schedule` do workflow continua como reserva.
  - **Token:** fine-grained PAT só deste repositório, com a permissão **Actions: Read and write**. Fica como secret `GITHUB_TOKEN` do Worker (painel do Cloudflare, em *Settings > Variables and Secrets*, ou `npx wrangler secret put GITHUB_TOKEN`). Nunca vai para o código.
  - **Conferência:** os disparos aparecem em *Actions* com o evento `workflow_dispatch`, e as falhas do Cron Trigger aparecem nos logs do Worker.

## 6. Como reproduzir

**Ambiente**

```bash
conda env create -f environment.yml        # ou: conda env update -f environment.yml --prune
conda activate capstone
```

Só a rotina ao vivo: `pip install -r requirements-ao-vivo.txt`.

**Credenciais.** Nunca envie nenhuma delas ao GitHub.
- **ANA:** solicite acesso à API HidroWebService e crie um `.env` na raiz (já está no `.gitignore`):

  ```
  ANA_ID=seu_cpf_somente_numeros
  ANA_SENHA=sua_senha
  ```

- **NASA Earthdata:** crie uma conta, autorize o aplicativo "NASA GESDISC DATA ARCHIVE" e grave em `~/.netrc`:

  ```
  machine urs.earthdata.nasa.gov login SEU_USUARIO password SUA_SENHA
  ```

**Dados e modelos**

```bash
python -m src.ana_api --start 2015-01-01 --end 2026-06-30               # telemetria ANA (retomável)
python -m src.basin                                                    # polígonos das áreas de drenagem
python -m src.imerg_early baixar --start 2018-05-01 --end 2026-06-30   # IMERG Early (retomável)
python -m src.imerg_early series                                       # chuva horária por sub-bacia
# notebooks 01 a 05, em ordem
python -m src.operacional                                              # checagem e modelo operacional
```

**Experimento de chuva prevista e pluviômetros**

```bash
python -m src.chuva_prevista historico --inicio 2021-04-01 --fim 2026-07-01   # GFS arquivado (Open-Meteo)
python -m src.experimento_chuva_prevista
```

**Rodada ao vivo e painel, localmente**

```bash
python -m src.ao_vivo --saida web/agora.json
python -m src.painel                       # gera web/index.html e outputs/painel/painel_taquari.html
```

**Pasta do registro de dados no Zenodo**

Depois dos notebooks 01 a 04 nos dois cenários (com e sem IMERG) e do `src.operacional`:

```bash
python scripts/build_zenodo.py
```

O script grava em `zenodo/` (fora do Git):
- `painel_horario.parquet` e `painel_horario.csv`, o painel horário com o IMERG Early já defasado em 6 h;
- `previsoes_teste.csv.gz`, as previsões de todos os modelos no teste;
- `modelo_hgb.joblib`, uma cópia do modelo operacional;
- `README.md`, uma cópia de `scripts/zenodo_README.md`, a descrição do registro em inglês, com as colunas, unidades e fontes.

No fim, o script mostra o SHA-256 de cada arquivo e confere o RMSE do gradient boosting. Se algum SHA-256 mudar, atualize `scripts/zenodo_README.md`.

**Notebooks** (`notebooks/`)

| Notebook | O que faz |
|---|---|
| `00_teste_imerg`, `00_chuva_bacia_imerg` | teste e conferência da chuva IMERG na bacia |
| `01_dados_brutos` | lê os dados da ANA, confere unidade, fuso e lacunas e salva o painel horário |
| `02_features_e_split` | variáveis, alvo e divisão temporal |
| `03_baselines` | persistência, Ridge e gradient boosting |
| `04_modelos_pytorch` | MLP e GRU |
| `05_avaliacao_eventos` | métricas, erro de pico e de horário, figuras e comparação com e sem IMERG |
| `07_api_ana` | teste e uso da API da ANA |

O cenário com IMERG vem ligado (`USE_IMERG = True` em `src/config.py`). Os dois cenários gravam em pastas separadas (`com_imerg` e `sem_imerg`). Para um teste rápido das redes, use `CAPSTONE_FAST=1`.

## 7. Estrutura do repositório

```
.
├── environment.yml, requirements-ao-vivo.txt
├── .github/workflows/ao_vivo.yml   # rotina horária
├── wrangler.jsonc                  # Cloudflare Workers: site (pasta web/) e disparo horário do workflow
├── worker/index.js                 # Worker: serve web/ e dispara ao_vivo.yml de hora em hora
├── src/
│   ├── config.py                   # estações, horizontes, datas, cotas de referência
│   ├── ana_api.py                  # cliente da API da ANA
│   ├── data.py                     # painel horário, limpeza, junção do IMERG
│   ├── basin.py                    # áreas de drenagem (HydroBASINS)
│   ├── imerg.py, imerg_early.py    # IMERG Final diário e Early horário por sub-bacia
│   ├── features.py                 # variáveis, alvo, divisão, limites
│   ├── models.py                   # MLP e GRU (PyTorch)
│   ├── metrics.py                  # RMSE, NSE, skill, erro de pico e de horário
│   ├── pipeline.py                 # funções compartilhadas pelos notebooks
│   ├── operacional.py              # modelo operacional e checagem
│   ├── ao_vivo.py                  # rotina ao vivo
│   ├── chuva_prevista.py           # previsão de chuva GFS/ECMWF (Open-Meteo)
│   ├── pluviometros.py             # rede de pluviômetros e médias de Thiessen
│   ├── experimento_chuva_prevista.py
│   ├── painel.py, painel_template.html   # painel interativo (D3.js)
│   └── synthetic.py                # dados sintéticos para testar o código
├── operacional/                    # modelo salvo, pesos da grade, rede de pluviômetros, checagem
├── notebooks/
├── scripts/                        # build_zenodo.py gera a pasta zenodo/; zenodo_README.md é o README do registro
├── outputs/                        # tabelas e figuras; modelos e previsões ficam fora do Git
├── web/                            # site estático
└── data/                           # não versionado
```

## 8. Fontes e referências

**Dados**
- ANA, API HidroWebService (séries telemétricas e inventário de estações).
- Huffman, G. J. et al. (2023). *GPM IMERG Early Precipitation L3 Half Hourly 0.1° × 0.1° V07 (GPM_3IMERGHHE).* GES DISC. doi:10.5067/GPM/IMERG/3B-HH-E/07
- Huffman, G. J. et al. (2023). *GPM IMERG Final Precipitation L3 1 day 0.1° × 0.1° V07 (GPM_3IMERGDF).* GES DISC. doi:10.5067/GPM/IMERGDF/DAY/07
- Lehner, B.; Grill, G. (2013). Global river hydrography and network routing. *Hydrological Processes*, 27(15), 2171–2186. doi:10.1002/hyp.9740 (HydroBASINS)
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API.* Zenodo. doi:10.5281/zenodo.7970649 (previsões NOAA GFS e ECMWF IFS)
- SGB, SACE Taquari: boletins e cotas de referência.

**Métodos e software**
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR*, 12, 2825–2830.
- Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS 32*.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014* (GRU).
- Nash, J. E.; Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I. *Journal of Hydrology*, 10(3), 282–290.
- Murphy, A. H. (1988). Skill scores based on the mean square error and their relationships to the correlation coefficient. *Monthly Weather Review*, 116(12), 2417–2424.
- Thiessen, A. H. (1911). Precipitation averages for large areas. *Monthly Weather Review*, 39(7), 1082–1084.
- Bostock, M.; Ogievetsky, V.; Heer, J. (2011). D3: data-driven documents. *IEEE TVCG*, 17(12), 2301–2309.

**Contexto**
- Marengo, J. A. et al. (2025). *Early warning services for disaster risk reduction in Brazil: the experience of CEMADEN during the floods of Rio Grande do Sul of May 2024.* International Journal of Disaster Risk Reduction.
- Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature*, 627, 559–563.
- Kratzert, F. et al. (2024). HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin. *Hydrology and Earth System Sciences*.

## 9. Autoria

- **Autoria:** `Isadora Beckmann` — `beckmannisadora@gmail.com`.
- Projeto desenvolvido no **Future Tech Leaders Brasil**, iniciativa do PNUD (UNDP) com a Agência Espacial Brasileira (AEB) e parceiros.
