# AeroCast-AI: Day-Ahead Flight Delay Risk

**AML Project 21 · Phase 1 (machine learning only).** Predict whether a flight arrives ≥ 15 min late using only information available at **20:00 NYC on the evening before departure**. Data: [`nycflights13`](https://www.kaggle.com/datasets/mahoora00135/flights) (336,776 departures from EWR, JFK, LGA in 2013). Models: logistic regression and XGBoost, against two references (the base-rate *prior* and a *route-history* heuristic). **No weather** is used (observed weather would leak at the decision clock; see [limitations](docs/limitations_and_assumptions.md)).

* Split: train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec (chronological); rolling-origin CV inside train.
* **Final model: LogReg (Platt, rule-selected)**, chosen by the CV rule declared before test. LogReg and XGBoost are **not statistically distinguished** (CV 0.4705 vs 0.4450, fold sd about 0.07; test bootstrap CI of the difference [-0.010, +0.041] includes 0).
* The chronological split is retained; a single year and three origins cannot establish cross-year generalisation.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
# place flights.csv in data/raw/ ; register the kernel:
.venv/bin/python -m ipykernel install --sys-prefix --name aerocast
```

## How to reproduce

```bash
bash scripts/reproduce.sh     # clean state -> pipeline -> 4 notebooks -> report export -> pytest
```

Details, runtime and the artefact list: [docs/reproducibility.md](docs/reproducibility.md).

## Headline result (test = Nov–Dec 2013, 53,991 flights, base rate 0.258)

PR-AUC is average precision. Every learned row names its calibration variant; the *prior* and *route history* are reference rows.

| model | calibration | pr_auc | roc_auc | brier | ece | ece_equal_mass | oracle_recall@p0.40 |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.258 | 0.500 | 0.192 | 0.007 | 0.084 | 0.000 |
| Route history | none (smoothed rate) | 0.353 | 0.621 | 0.187 | 0.037 | 0.038 | 0.244 |
| LogReg (raw) | raw | 0.394 | 0.654 | 0.181 | 0.019 | 0.022 | 0.430 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | 0.394 | 0.654 | 0.187 | 0.082 | 0.082 | 0.430 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | 0.392 | 0.657 | 0.181 | 0.027 | 0.026 | 0.432 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | 0.407 | 0.662 | 0.180 | 0.027 | 0.027 | 0.474 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | 0.396 | 0.659 | 0.180 | 0.024 | 0.023 | 0.416 |

* **Static rule vs raw vs rolling (LogReg):** test ECE 0.082 (Platt, rule-selected) vs 0.019 (raw) vs 0.027 (rolling Platt 28d). **The static rule fails in December** (ECE 0.142, December base rate 33.5%).
* **Primary policy (daily top 10%, `LogReg (Platt, rule-selected)`):** precision 0.444, recall 0.173, lift 1.72× (prior: 0.94×).
* `oracle_recall@p` tunes the threshold on the evaluated labels: a model-comparison metric, not an operating result.

## Documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | index and reading order |
| [01 data & EDA](docs/01_data_and_eda.md) · [02 features](docs/02_feature_engineering.md) · [03 training](docs/03_model_training.md) · [04 evaluation](docs/04_evaluation.md) | one doc per notebook |
| [development protocol](docs/development_protocol.md) | what was seen when (test not perfectly clean) |
| [limitations & assumptions](docs/limitations_and_assumptions.md) | weather, single year, assumptions |
| [reproducibility](docs/reproducibility.md) | environment, `reproduce.sh`, artefacts |
| [report/](report/README.md) | IEEE report source (`main.tex`, `references.bib`), generated figures/tables, build steps |

## Layout

```text
src/aerocast/   all logic (config, data, features, leakage, models, evaluate, drift, pipeline, plots, eda, artefacts, report)
scripts/        run_phase1.py (single source of truth) · reproduce.sh · export_report_assets.py
notebooks/      01_eda · 02_feature_engineering · 03_model_training · 04_evaluation (import only)
tests/          leakage, decision clock, time zones, evaluation metrics, report export
results/        every table + predictions.csv.gz + summary.json (committed; predictions.csv.gz is the audit trail)
figures/        PNG figures    docs/  documentation    report/  report assets
```
