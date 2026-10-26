# AeroCast-AI: Day-Ahead Flight Delay Risk

**Project 21: Flight Delay Prediction and Rescheduling**

**Project brief title:** Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

**Team T49 · AML Project 21 · Phase 1 (machine learning only).** Predict whether a flight arrives ≥ 15 min late using only information available at **20:00 NYC on the evening before departure**. Data: [`nycflights13`](https://www.kaggle.com/datasets/mahoora00135/flights) (336,776 departures from EWR, JFK, LGA in 2013). Models: logistic regression and XGBoost, against two references (the base-rate *prior* and a *route-history* heuristic). **No weather** is used (observed weather would leak at the decision clock; see [limitations](docs/limitations_and_assumptions.md)).

* Split: train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec (chronological); rolling-origin CV inside train.
* **Final model: LogReg (Platt, rule-selected)**, selected by higher development CV AP (0.4714 vs fixed-tree XGBoost 0.4500). The 95% day-block bootstrap interval for test AP, XGBoost − LogReg, [−0.010, +0.040] includes zero; it does not establish a difference, equivalence, or noninferiority. Development-time test exposure is disclosed below.
* The chronological split is retained; a single year and three origins cannot establish cross-year generalisation.

## Phase 1 scope and ML workflow

Phase 1 evaluates the ML component: dataset quality and EDA, feature engineering, baseline models, chronological validation, calibration, interpretation and reproducible code. The 7-page [IEEE report](report/main.pdf) contains the related work, the theory behind each method and the results. A compact MLP with learned airline, origin and destination embeddings is planned for a later phase to investigate interactions missed by manual route and congestion summaries. A temporal deep-learning model is optional and would require correctly ordered flight history.

1. **Data and target:** construct `is_delayed = arr_delay >= 15`. Cancelled/diverted flights remain in the schedule used for congestion counts but are excluded from the main target; a separate sensitivity analysis treats them as delayed.
2. **Features:** build 43 categorical, schedule, cyclic-time, calendar, congestion and historical-delay features. The flight's own actual departure/arrival measurements are excluded from predictors. Earlier flights' outcomes enter history only after they are known at the decision clock.
3. **Model fitting and selection:** use four rolling-origin validation folds (May–August), tune Logistic Regression and XGBoost, and compare them using the same available features. Fitting, CV scoring used for selection, calibration and threshold selection all enforce outcome-availability cutoffs. Of 211,684 training scoring rows, **211,517 labels** are eligible for final fitting; 167 outcomes are unavailable before the first September prediction. XGBoost's in-fold early-stopping score is reported as optimistic; model-family selection uses its fixed-tree CV score.
4. **Calibration and evaluation:** fit candidate calibrators on September, check them on October, and freeze the static calibrator and precision threshold before the first November prediction. The final static fit uses **55,377 of 55,628 calibration labels**, excluding 251 unavailable outcomes. Evaluate on November–December using average precision, ROC AUC, Brier score, calibration error and operating-policy precision/recall. Causal rolling Platt calibration is a separate comparison.
5. **Offline decision support:** flag the highest-risk 5–20% of flights each day under explicit review-capacity assumptions. The current analysis measures delayed-flight capture. A full what-if rescheduling simulation, including feasible alternative schedules and estimated delay reductions, remains future work.

Development-time test exposure and other limitations are disclosed in the [development protocol](docs/development_protocol.md) and [limitations](docs/limitations_and_assumptions.md).

## Team and roles

Team number: **T49**.

| Member | GitHub ID | Role |
|---|---|---|
| **Nipun Patlori** | [nipun1803](https://github.com/nipun1803) | XGBoost model development; IEEE report |
| **Anushka Tyagi** | [anu-ushka](https://github.com/anu-ushka) | Logistic regression model development |
| **Omkar Hadole** | [omkar-hadole](https://github.com/omkar-hadole) | Leakage removal and feature engineering |
| **Mohan Kumar C R** | [mohan101mk](https://github.com/mohan101mk) | Evaluation metrics, probability calibration and confusion-matrix analysis; IEEE report |
| **Krish Dabas** | [krish-2509](https://github.com/krish-2509) | Data loading and exploratory data analysis (EDA) |

## Setup

Run from the repository root with Python 3.12. On macOS, XGBoost needs the OpenMP runtime; install it with `brew install libomp` if it is missing.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --sys-prefix --name aerocast
```

Download the [Kaggle Flights dataset](https://www.kaggle.com/datasets/mahoora00135/flights) and place `flights.csv` in `data/raw/`. The raw CSV is excluded from Git by `.gitignore`; it is required to rerun the full pipeline. Saved results are tracked in the repository.

## How to reproduce

With `.venv` activated, run the ML pipeline and execute all four notebooks:

```bash
python -B -m pytest -q -p no:cacheprovider
python -B scripts/run_phase1.py
python -B -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=aerocast \
  notebooks/01_eda.ipynb notebooks/02_feature_engineering.ipynb \
  notebooks/03_model_training.ipynb notebooks/04_evaluation.ipynb
```

The full run on **2026-10-23** passed **36 tests**, completed the pipeline in about **22 minutes** on a MacBook Pro, and executed all four notebooks successfully. After the equal-mass ECE correction on **2026-10-24**, the expanded suite passes **42 tests**. On **2026-10-26** a realistic-cancellation leakage test showed that the aircraft-rotation features leaked same-day cancellations; after removing them (45 → 43 features), the pipeline (about 9 minutes) and all four notebooks were rerun. With the report-alignment and threshold-metric tests added the same day, the suite passes **51 tests**. Runtime varies by machine; history tuning can run for several minutes between progress messages. Notebooks 03 and 04 check their computed tables against the pipeline's saved outputs.

Rerunning overwrites generated files in `results/` and `figures/`, and notebook outputs. These commands leave report assets unchanged. `scripts/reproduce.sh` additionally deletes generated outputs and report assets, exports report tables/figures, checks report numbers and attempts a PDF build.

For another dataset, matching column names alone is insufficient: [config.py](src/aerocast/config.py) contains the 2013 split/warm-up dates, NYC departure-clock assumptions, destination time zones, daylight-saving rules and expected CV selections. Review these settings and dataset-specific checks before rerunning with different data.

Environment and artefact details: [docs/reproducibility.md](docs/reproducibility.md). Report instructions: [report/README.md](report/README.md).

## Headline result (test = Nov–Dec 2013, 53,991 flights, base rate 0.258)

PR-AUC is average precision (AP). Values below are rounded from [results/headline_test.csv](results/headline_test.csv); selection and fitting-clock metadata are in [results/summary.json](results/summary.json). Every learned row names its calibration variant; the *prior* and *route history* are reference rows. Lower Brier and ECE values indicate better probability quality.

| model | calibration | pr_auc | roc_auc | brier | ece | ece_equal_mass | oracle_recall@p0.40 |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.258 | 0.500 | 0.192 | 0.007 | 0.007 | 0.000 |
| Route history | none (smoothed rate) | 0.353 | 0.621 | 0.187 | 0.037 | 0.038 | 0.244 |
| LogReg (raw) | raw | 0.393 | 0.653 | 0.181 | 0.019 | 0.022 | 0.421 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | 0.393 | 0.653 | 0.187 | 0.082 | 0.082 | 0.421 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | 0.392 | 0.657 | 0.181 | 0.027 | 0.027 | 0.428 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | 0.407 | 0.665 | 0.179 | 0.027 | 0.027 | 0.483 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | 0.396 | 0.664 | 0.180 | 0.023 | 0.022 | 0.438 |

* **Static rule vs raw vs rolling (LogReg):** test ECE 0.082 (Platt, rule-selected) vs 0.019 (raw) vs 0.027 (rolling Platt 28d). **The static rule fails in December** (ECE 0.143, December base rate 33.5%).
* **Primary policy (daily top 10%, `LogReg (Platt, rule-selected)`):** precision 0.448, recall 0.174, lift 1.73× (prior: 0.94×).
* **Secondary policy:** the LogReg threshold frozen on eligible September–October labels to target precision ≥ 0.40 is 0.329; on test it achieves precision 0.491 and recall 0.141. Test precision is an observed result, not a guarantee for future flights.
* **Accuracy, F1 and confusion matrix** (same two operating points): LogReg accuracy 0.740 / 0.731 and F1 0.220 / 0.251 (frozen threshold / daily top 10%). Predicting every flight on time is already 74.2% accurate, so accuracy alone hides the ranking skill; see [evaluation notes](docs/04_evaluation.md) and `figures/confusion_logreg_test.png`.
* `oracle_recall@p` tunes the threshold on the evaluated labels: a model-comparison metric, not an operating result.
* `ece` uses 15 fixed-width bins. `ece_equal_mass` starts with 15 balanced rank bins, moves boundaries to keep identical probabilities together, and collapses duplicate boundaries. Bins can therefore have unequal counts or number fewer than 15; constant predictions use one bin. Use fixed-width ECE and Brier for the main calibration discussion; low ECE alone does not imply ranking skill.

## Documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | index and reading order |
| [01 data & EDA](docs/01_data_and_eda.md) · [02 features](docs/02_feature_engineering.md) · [03 training](docs/03_model_training.md) · [04 evaluation](docs/04_evaluation.md) | one doc per notebook |
| [development protocol](docs/development_protocol.md) | what was seen when (test not perfectly clean) |
| [limitations & assumptions](docs/limitations_and_assumptions.md) | weather, single year, assumptions |
| [reproducibility](docs/reproducibility.md) | environment, `reproduce.sh`, artefacts |
| [report/](report/README.md) | 7-page IEEE report (`main.tex`, `main.pdf`, `references.bib`), generated figures/tables, build steps |

## Layout

```text
data/raw/       flights.csv (local input, excluded from Git)
src/aerocast/   all logic (config, data, features, leakage, models, evaluate, drift, pipeline, plots, eda, artefacts, report)
scripts/        run_phase1.py · reproduce.sh · export_report_assets.py · check_report_numbers.py
notebooks/      01_eda · 02_feature_engineering · 03_model_training · 04_evaluation (import only)
tests/          leakage, feature/fitting clocks, time zones, evaluation metrics, report export
results/        every table + predictions.csv.gz + summary.json (committed; predictions.csv.gz is the audit trail)
figures/        generated PNG figures
docs/           stage explanations, reproducibility and limitations
report/         IEEE LaTeX source, bibliography, PDF and generated report assets
```
