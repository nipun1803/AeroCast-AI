# AeroCast-AI Phase 2: Deep Learning and Hybrid

**Team T49 · AML Project 21 · Phase 2 (deep learning)**. Status: **complete** (models, ablations, the test evaluation, what-if rescheduling, report). Slides are pending.

Phase 2 asks whether learned representations recover signal that the hand-built Phase 1 features ([`../phase1/`](../phase1/)) miss. Everything else stays identical, so the comparison is fair:

- **Same inputs:** decision clock 20:00 NYC on D−1, target `arr_delay ≥ 15`, the 43 Phase 1 features, the chronological split (train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec) and the four rolling-origin CV folds (May–Aug). They are imported from the `aerocast` package, not copied.
- **Models:** an MLP with learned carrier, origin and destination embeddings; the MLP plus a GRU over causal 28-day delay histories of the origin, carrier and route; a **Wide & Deep hybrid** whose wide part starts as the Phase 1 LogReg; and MLP embeddings fed to the Phase 1 XGBoost.
- **Rules first:** the selection and calibration rules and the test lock were committed before any model was trained ([docs/06](docs/06_rules_and_limitations.md)). Test labels are scored only in the locked final stage. It ran twice: a code audit after the first run removed two leaks (Wide & Deep's early-stopping holdout; the rescheduling and flag-budget cohort), and the stage was rerun. Only Wide & Deep and the top-k and rescheduling numbers changed; the first run stays in git history.

**Start with the [documentation](docs/README.md)**: six short documents explaining the inputs, models, results, test evaluation, rescheduling and rules, in reading order.

## Results (test set, Nov–Dec 2013)

| Model | PR AUC | ROC AUC | Recall @ precision 0.5 | Brier |
|---|---|---|---|---|
| Random guess | 0.258 | 0.500 | – | – |
| Logistic Regression (Phase 1) | 0.392 | 0.657 | 0.119 | 0.181 |
| XGBoost (Phase 1) | 0.407 | 0.665 | 0.170 | 0.179 |
| MLP with embeddings | 0.395 | 0.668 | 0.105 | 0.180 |
| MLP + GRU history | 0.398 | 0.668 | 0.090 | 0.180 |
| Wide & Deep | 0.394 | 0.656 | 0.139 | 0.181 |
| Embeddings → XGBoost | 0.400 | 0.664 | 0.146 | 0.179 |

*Note: Models evaluated on test (Nov–Dec 2013, 53,991 flights, base rate 0.258) using pre-registered calibration rules (rolling Platt 28d for LogReg, Embedding MLP, GRU, and Wide & Deep; raw for XGBoost; rolling temperature for Embeddings → XGBoost). Raw uncalibrated test metrics: LogReg PR AUC 0.393 / ROC AUC 0.653 / Recall@p0.5 0.128 / Brier 0.181; Embedding MLP 0.404 / 0.666 / 0.153 / 0.180; MLP + GRU 0.398 / 0.657 / 0.139 / 0.181; Wide & Deep 0.388 / 0.642 / 0.127 / 0.184; Embeddings → XGBoost 0.401 / 0.663 / 0.145 / 0.180.*

### Model Selection (Development CV vs Test)

| Model | CV AP (May–Aug, 5 seeds) | Test AP (Nov–Dec) | Test ECE | Test AP − LogReg, 95% CI |
|---|---|---|---|---|
| **LogReg (Phase 1), final model** | **0.471** | 0.392 | 0.027 | – |
| Wide & Deep | 0.468 | 0.394 | 0.036 | +0.003 [−0.016, 0.022] |
| MLP + GRU history | 0.467 | 0.398 | 0.035 | +0.006 [−0.013, 0.028] |
| Embedding MLP | 0.465 | 0.395 | 0.027 | +0.003 [−0.015, 0.027] |
| Embeddings → XGBoost | 0.450 | 0.400 | 0.017 | +0.009 [−0.014, 0.033] |
| XGBoost (Phase 1) | 0.450 | 0.407 | 0.027 | +0.016 [−0.012, 0.043] |

- **No network beats LogReg on cross-validation.** By the pre-registered rule (best CV AP; within 0.005, fewer parameters) LogReg stays the final model. On test every Phase 2 model scores slightly higher, but no day-block bootstrap interval excludes zero.
- **Why:** the hidden layers overfit after one or two epochs. The Phase 1 history features are already smoothed rates, so the remaining signal is close to linear. A CV procedure error (scoring the early-stopped network instead of the deployed refit) was found and corrected before testing.
- **Ablations:** the GRU cannot replace the hand-built history (−0.015 CV AP). Embeddings vs one-hot makes no material difference in the MLP (−0.001) and a small one in Wide & Deep (−0.005 with one-hot). Early stopping is the regulariser that matters (−0.035 without it).
- **What-if rescheduling** (tuned on Sep–Oct, offline): 49 moves a day avoid 1.9 **predicted** delays a day on Nov–Dec. A naive "everything 1 h earlier" rule avoids 0.1.

Report: [report/main.pdf](report/main.pdf) (8 pages).

## Team and roles

Team number: **T49**. Phase 1 roles are in [../phase1/README.md](../phase1/README.md#team-and-roles).

### Team (Phase 2 contributions)

| Member | GitHub ID | Role | Work Summary (2–3 words) | Contribution (Simple Terminology) |
|---|---|---|---|---|
| **Nipun Patlori** | [nipun1803](https://github.com/nipun1803) | Hybrids | Temporal Models & Reporting | Combined neural networks with tree models, evaluated delay patterns across time, and co-created the report and presentation |
| **Mohan Kumar C R** | [mohan101mk](https://github.com/mohan101mk) | Architecture | Neural Network Design | Designed deep neural network architectures with flight embeddings, trained models with early stopping, and co-created the report and presentation |
| **Anushka Tyagi** | [anu-ushka](https://github.com/anu-ushka) | Calibration | Calibration & Testing | Calibrated delay prediction probabilities, locked test evaluations, and audited reliability across airport segments |
| **Omkar Hadole** | [omkar-hadole](https://github.com/omkar-hadole) | Rescheduling | Flight Rescheduling Simulation | Designed and tested the flight rescheduling simulation to minimize delays and evaluated operational flight savings |
| **Krish Dabas** | [krish-2509](https://github.com/krish-2509) | Preprocessing | Data Pipeline Preparation | Cleaned raw data into neural network inputs, generated 28-day history sequences, and prevented data leakage |

### Notebook and Module Ownership

| # | Notebook | Owner | Phase 2 rubric | Module(s) it writes |
|---|---|---|---|---|
| 05 | [`05_network_inputs.ipynb`](notebooks/05_network_inputs.ipynb) | Krish Dabas | DL dataset, leakage validation | `tensors.py`, `sequences.py`, `guards.py` |
| 06, 08 | [`06_embedding_mlp.ipynb`](notebooks/06_embedding_mlp.ipynb), [`08_wide_deep_hybrid.ipynb`](notebooks/08_wide_deep_hybrid.ipynb) | Mohan Kumar C R | Architecture, regularization | `models.py`, `train.py` |
| 07, 09 | [`07_gru_history_model.ipynb`](notebooks/07_gru_history_model.ipynb), [`09_ablation_studies.ipynb`](notebooks/09_ablation_studies.ipynb) | Nipun Patlori | Temporal model, hybrid, model selection | `pipeline.py`, `report.py` |
| 10 | [`10_calibration_and_test.ipynb`](notebooks/10_calibration_and_test.ipynb) | Anushka Tyagi | Ablation, calibration, test lock | `calibrate.py`, `evaluation.py`, `testlock.py` |
| 11 | [`11_whatif_rescheduling.ipynb`](notebooks/11_whatif_rescheduling.ipynb) | Omkar Hadole | Final test, operational metrics, what-if | `reschedule.py` |

## Run

The venv is shared with Phase 1 at the repository root. From `phase2/`:

```bash
../.venv/bin/pip install -r requirements.txt && ../.venv/bin/pip install -e ../phase1 -e .
python -B -m pytest -q -p no:cacheprovider        # 60 tests: leakage, alignment, training, calibration, lock, rescheduling, report
python -B scripts/run_phase2.py                    # every stage up to the test (about 2 hours on an M3 Pro CPU)
python -B scripts/run_phase2.py --final            # the one test evaluation (refused while the rules docs have uncommitted changes)
python -m ipykernel install --sys-prefix --name aerocast   # once, for the notebooks
```

- **Full rebuild.** `bash scripts/reproduce.sh` runs the tests, a clean pipeline run, `--final`, notebooks 05–11, report export, the number check and the PDF build. It takes about 3 hours and has not been run end to end in one go; every stage was run individually.
- **Data.** `flights.csv` goes in `../phase1/data/raw/`.
- **Determinism.** Seeds 42–46, deterministic torch algorithms, CPU.
- **Stage times** (M3 Pro). GRU search 28 min, Wide & Deep 26 min, ablations 39 min; every other stage takes a few minutes.
- **macOS OpenMP.** `import aerocast_dl` loads torch before XGBoost and refuses the reverse order, which crashes on macOS. XGBoost fits run in a separate process (`scripts/xgb_worker.py`).
- **iCloud.** If the repository is in an iCloud folder, evicted files hang on read. Keep the venv outside iCloud, and fetch evicted files with `brctl download` (`find . -flags +dataless` lists them).
- **Model weights.** The final weights are cached in the git-ignored `data/cache/`. On a fresh clone, run `final_fit` before `reschedule` or `--final`.

## Contents

| | |
|---|---|
| [05_network_inputs](notebooks/05_network_inputs.ipynb) | Step 2: network inputs, fitting clock, cold-start coverage |
| [06_embedding_mlp](notebooks/06_embedding_mlp.ipynb) | Step 3: embedding MLP search, 5-seed CV, learning curves, learned embeddings, why it trails LogReg |
| [07_gru_history_model](notebooks/07_gru_history_model.ipynb) | Step 4: causal history windows, decision clock, GRU model |
| [08_wide_deep_hybrid](notebooks/08_wide_deep_hybrid.ipynb) | Step 5: Wide & Deep on the Phase 1 LogReg; embeddings into XGBoost |
| [09_ablation_studies](notebooks/09_ablation_studies.ipynb) | Step 6: 17 ablations on rolling CV |
| [10_calibration_and_test](notebooks/10_calibration_and_test.ipynb) | Step 7: Sep→Oct calibration, final-model rule, the test run, slices, drift |
| [11_whatif_rescheduling](notebooks/11_whatif_rescheduling.ipynb) | Step 8: what-if rescheduling under explicit assumptions |
| [docs/](docs/README.md) | project documentation: overview and the six explanatory documents |

```text
src/aerocast_dl/  config · guards (leakage assertions) · tensors (train-only encoders) · sequences (causal windows)
                  models (DelayNet, GRU encoders) · train (fit, CV as deployed, search) · pipeline (CV stages)
                  calibrate · evaluation (final fits, calibration, final-model rule, test) · testlock
                  reschedule (what-if rule) · plots · report (asset export)
scripts/          run_phase2.py · xgb_worker.py · reproduce.sh · export_report_assets.py · check_report_numbers.py
results/          CV, search, ablation, diagnostics, calibration, selection, test and rescheduling tables
figures/          9 PNGs; report/ holds the 8 the report uses and its 8 tables
```
