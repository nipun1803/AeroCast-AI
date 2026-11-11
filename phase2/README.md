# AeroCast-AI Phase 2: Deep Learning and Hybrid

**Team T49 · AML Project 21 · Phase 2 (deep learning)**. Status: **complete** (models, ablations, the one test run, what-if rescheduling, report). Slides are pending.

Phase 2 asks whether learned representations recover signal that the hand-built Phase 1 features ([`../phase1/`](../phase1/)) miss. Everything else stays identical, so the comparison is fair:

- **Same inputs:** decision clock 20:00 NYC on D−1, target `arr_delay ≥ 15`, the 43 Phase 1 features, the chronological split (train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec) and the four rolling-origin CV folds (May–Aug). They are imported from the `aerocast` package, not copied.
- **Models:** an MLP with learned carrier, origin and destination embeddings; the MLP plus a GRU over causal 28-day delay histories of the origin, carrier and route; a **Wide & Deep hybrid** whose wide part starts as the Phase 1 LogReg; and MLP embeddings fed to the Phase 1 XGBoost.
- **Rules first:** the selection and calibration rules and the test lock were committed before any model was trained ([docs/06](docs/06_rules_and_limitations.md)); the test labels were read once.

**Start with the [documentation](docs/README.md)**: six short documents explaining the inputs, models, results, test evaluation, rescheduling and rules, in reading order.

## Results

| Model | CV AP (May–Aug, 5 seeds) | Test AP (Nov–Dec) | Test ECE | Test AP − LogReg, 95% CI |
|---|---|---|---|---|
| **LogReg (Phase 1), final model** | **0.471** | 0.392 | 0.027 | – |
| Wide & Deep | 0.469 | 0.399 | 0.028 | +0.007 [−0.005, 0.023] |
| MLP + GRU history | 0.467 | 0.398 | 0.035 | +0.006 [−0.013, 0.028] |
| Embedding MLP | 0.465 | 0.395 | 0.027 | +0.003 [−0.015, 0.027] |
| Embeddings → XGBoost | 0.450 | 0.400 | 0.017 | +0.009 [−0.014, 0.033] |
| XGBoost (Phase 1) | 0.450 | 0.407 | 0.027 | +0.016 [−0.012, 0.043] |

- **No network beats LogReg on cross-validation.** By the pre-registered rule (best CV AP; within 0.005, fewer parameters) LogReg stays the final model. On test every Phase 2 model scores slightly higher, but no day-block bootstrap interval excludes zero.
- **Why:** the hidden layers overfit after one or two epochs. The Phase 1 history features are already smoothed rates, so the remaining signal is close to linear. A CV procedure error (scoring the early-stopped network instead of the deployed refit) was found and corrected before testing.
- **Ablations:** the GRU cannot replace the hand-built history (−0.015 CV AP). Embeddings vs one-hot makes no material difference (−0.001). Early stopping is the regulariser that matters (−0.035 without it).
- **What-if rescheduling** (tuned on Sep–Oct, offline): 48 moves a day avoid 1.9 **predicted** delays a day on Nov–Dec. A naive "everything 1 h earlier" rule avoids 0.1.

Report: [report/main.pdf](report/main.pdf) (7 pages).

## Team and roles

Team number: **T49**. Phase 1 roles are in [../phase1/README.md](../phase1/README.md#team-and-roles).

| Member | GitHub ID | Phase 2 role |
|---|---|---|
| **Nipun Patlori** | [nipun1803](https://github.com/nipun1803) | Hyperparameter searches, Wide & Deep and embeddings → XGBoost hybrids, ablation results; IEEE report |
| **Anushka Tyagi** | [anu-ushka](https://github.com/anu-ushka) | Embedding MLP, training loop and CV-as-deployed procedure; diagnostics of why the MLP trails LogReg |
| **Omkar Hadole** | [omkar-hadole](https://github.com/omkar-hadole) | What-if rescheduling rule: tuning on Sep–Oct and the Nov–Dec analysis |
| **Mohan Kumar C R** | [mohan101mk](https://github.com/mohan101mk) | Calibration, final-model rule, test lock and the one test evaluation (slices, drift, confusion matrices); protocol |
| **Krish Dabas** | [krish-2509](https://github.com/krish-2509) | Infrastructure and reproducibility (torch/XGBoost process split, `reproduce.sh`, report checks); notebooks and documentation |

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
| [09_ablation_studies](notebooks/09_ablation_studies.ipynb) | Step 6: 16 ablations on rolling CV |
| [10_calibration_and_test](notebooks/10_calibration_and_test.ipynb) | Step 7: Sep→Oct calibration, final-model rule, the one test run, slices, drift |
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
