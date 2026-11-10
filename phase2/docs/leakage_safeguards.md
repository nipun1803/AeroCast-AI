# Phase 2 leakage safeguards

> Companion to [development_protocol.md](development_protocol.md). Every rule here has a test or a code assertion; a step may not start training until its tests pass.

## Global rules

| # | Rule | Enforcement |
|---|---|---|
| G1 | **Decision clock.** An outcome may feed an input for day D only if it was known by 20:00 NYC on D−1 | Phase 2 imports `aerocast.data.available_date` and `features.rolling_past`; no reimplementation |
| G2 | **Label availability.** Every fit (network, vocabulary, imputer, scaler, sequence normaliser, calibrator, threshold) uses only rows whose labels were available at that fit's date | `assert_labels_available(rows, fit_date)` called inside every fit function |
| G3 | **Selection on CV only.** Hyperparameters, architectures and ablations are chosen on the May–Aug rolling folds | Search functions never receive Sep–Dec labels |
| G4 | **Calibration months stay calibration months.** Sep–Oct is used only for calibration, the threshold and rescheduling parameters | No training or early-stopping call receives Sep–Oct rows |
| G5 | **Test lock.** Nov–Dec labels load only in `run_phase2.py --final`, after the protocol is committed | Clean-tree check on the protocol files before the final stage |
| G6 | **Same inputs as Phase 1.** Same 43 features, split and folds; any new input also obeys G1 | Row-alignment test against `aerocast.pipeline.prepare()` |
| G7 | **No same-day aircraft rotation.** No input counts a tail number's day-D legs (the Phase 1 leak: cancelled flights have blank tail numbers) | Test that sequence and ID inputs contain no day-D tail counts |

## Risks specific to deep learning

| Risk | Why it leaks | Safeguard |
|---|---|---|
| BatchNorm in training mode at evaluation | One test flight's score would depend on other test flights in its batch | LayerNorm by default; if BatchNorm is ablated, a test asserts `model.eval()` before every validation/test forward pass |
| Early stopping on the validation month | The stopping epoch is chosen on the labels being scored (Phase 1 XGB: 0.4749 in-fold vs 0.4500 as deployed) | Inner holdout = last 14 days of fold-train |
| Vocabulary, rare-bucket cutoff, scaler fitted on all rows | Validation/test category frequencies and feature distributions shape the inputs | Fitted per fold on fold-train rows only, as Phase 1's sklearn pipelines are |
| Embeddings of rare IDs reused downstream | A tail number with few flights gets an embedding that memorises its training labels, acting as target encoding | Only low-cardinality embeddings (carrier, origin, destination, frequent routes) are passed to XGBoost; tail and flight embeddings are never reused |
| Frozen in-sample LogReg logit as an offset | The deep part learns residuals of in-sample logits on train but sees out-of-sample logits on test | Wide part trained jointly (initialised from the fold's LogReg), not a frozen offset |
| Sequences built by flight date instead of availability date | A D−1 evening arrival would reach day D | Sequences reuse the daily-by-availability-date table of `rolling_past` |
| Missing tail number | Occurs only for cancelled flights, so it reveals cancellation | Main target excludes cancelled flights; the cancellation-inclusive sensitivity target drops tail inputs, as in Phase 1 |
| Rescheduling scored with realised outcomes | Moving a flight into a slot that, in hindsight, had good weather would credit the rule with information unavailable at 20:00 D−1 | Benefit = reduction in calibrated predicted risk only; realised day-D outcomes of other flights are never used |

Row shuffling inside a training window is allowed: every feature is already computed causally, so batch order carries no future information. Only split boundaries must be chronological.

## Required tests (by step)

| Step | Test | Asserts |
|---|---|---|
| 2 | `test_tensor_alignment` | Tensor rows match the Phase 1 feature frame 1:1 (index, label, split) |
| 2 | `test_fit_stats_train_only` | Changing every Sep–Dec row leaves vocabulary, imputer and scaler statistics unchanged |
| 2 | `test_no_missing_tail_in_target` | No main-target row has a missing tail number |
| 3 | `test_eval_mode` | Validation/test forward passes run in `eval()` mode |
| 3 | `test_early_stopping_rows` | Early-stopping rows lie inside fold-train and before the validation month |
| 4 | `test_sequence_matches_history` | 14-day sums of each sequence equal Phase 1 `hist_n` / `hist_s` exactly |
| 4 | `test_sequence_future_perturbation` | Randomising all outcomes known after 20:00 D−1 leaves day-D sequences unchanged |
| 4 | `test_sequence_late_arrival` | A D−1 arrival after 20:00 first appears in the D+1 sequence |
| 4 | `test_sequence_cancellation` | Phase 1 cancellation check extended to the sequences |
| 4 | `test_no_tail_day_counts` | G7 |
| 5 | `test_wide_init_fold_only` | The wide part's initial weights come from a LogReg fitted on that fold's training rows |
| 5 | `test_embedding_transfer_low_card` | Only allowed embeddings reach XGBoost |
| 7 | `test_calibrator_clock` | Rolling calibrators for day D use only outcomes known by D's decision clock |
| 7 | `test_test_lock` | Test labels cannot be loaded without `--final` |
| 8 | `test_reschedule_no_outcomes` | The what-if scorer runs with all day-D outcomes set to NaN and gives identical results |
| 8 | `test_reschedule_params_sepoct` | Rescheduling parameters are fitted on Sep–Oct only |

The Phase 1 leakage tests (`../phase1/tests/`) keep running unchanged; Phase 2 does not modify Phase 1 code.
