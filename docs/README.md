# AeroCast-AI Phase 1: documentation index

Phase 1 (machine learning only): day-ahead prediction of `arr_delay ≥ 15 min` for 2013 NYC departures, decided at **20:00 NYC on D-1**. Models: logistic regression and XGBoost; references: the base-rate prior and a route-history heuristic. No weather and no deep learning in this phase; the IEEE report with related work is in `report/`.

## Reading order

| # | Document | Summarises |
|---|---|---|
| 1 | [01_data_and_eda.md](01_data_and_eda.md) | notebook 01: data quality, leakage audit, patterns |
| 2 | [02_feature_engineering.md](02_feature_engineering.md) | notebook 02: decision clock, time zones, smoothing, leakage tests |
| 3 | [03_model_training.md](03_model_training.md) | notebook 03: split, baselines, rolling CV, final-model rule |
| 4 | [04_evaluation.md](04_evaluation.md) | notebook 04: calibration, test metrics, policies, ablation, shift |
| 5 | [development_protocol.md](development_protocol.md) | what was seen when; what each decision rests on |
| 6 | [limitations_and_assumptions.md](limitations_and_assumptions.md) | weather, single year, assumptions |
| 7 | [reproducibility.md](reproducibility.md) | environment, `reproduce.sh`, artefacts |

## The result in one table (test = Nov–Dec 2013)

| model | calibration | pr_auc | roc_auc | brier | ece | ece_equal_mass | oracle_recall@p0.40 |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.258 | 0.500 | 0.192 | 0.007 | 0.084 | 0.000 |
| Route history | none (smoothed rate) | 0.353 | 0.621 | 0.187 | 0.037 | 0.038 | 0.244 |
| LogReg (raw) | raw | 0.394 | 0.654 | 0.181 | 0.019 | 0.022 | 0.431 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | 0.394 | 0.654 | 0.187 | 0.082 | 0.082 | 0.431 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | 0.392 | 0.657 | 0.181 | 0.027 | 0.027 | 0.431 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | 0.404 | 0.662 | 0.180 | 0.027 | 0.026 | 0.462 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | 0.394 | 0.659 | 0.181 | 0.026 | 0.025 | 0.417 |

Final model: **LogReg (Platt, rule-selected)**. LogReg has the higher development CV mean (0.4715 versus 0.4474), so it is selected by the declared rule. The test AP-difference interval (XGBoost − LogReg, [-0.013, +0.037]) includes zero: it does not establish a test-performance difference, equivalence, or noninferiority. Test exposure during development remains disclosed.
