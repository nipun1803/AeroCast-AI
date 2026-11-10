# Phase 2 development protocol (pre-registered)

> Written and committed **before** any Phase 2 model is trained or scored. The git commit that adds this file is the evidence of when these rules were fixed. Leakage rules and their tests are in [leakage_safeguards.md](leakage_safeguards.md).

## 1. What stays identical to Phase 1

Phase 2 is compared with Phase 1 on the same inputs. Everything below is imported from the `aerocast` package (`../phase1/src/aerocast`), not copied or re-estimated.

| Item | Value (source) |
|---|---|
| Decision clock | 20:00 NYC on D−1; an outcome enters any input only once it is known (`config.DECISION_HOUR`, `data.available_date`) |
| Target | `is_delayed = arr_delay ≥ 15`; cancelled and diverted flights excluded (`config.DELAY_THRESHOLD_MIN`) |
| Features | the 43 Phase 1 features from `pipeline.prepare()`, with m = 5, long window = 14 days |
| Split | train 2013-01-08 to 08-31 · calibration 09-01 to 10-31 · test 11-01 to 12-31 (`config.SPLITS`, `config.WARMUP_END`) |
| Model-selection CV | rolling origin, validation months May, Jun, Jul, Aug (`models.rolling_folds`, label-availability filtered) |
| Calibrator selection | Phase 1 rule as implemented in `evaluate.select_calibrator`: fit on September, check ECE on October, labels filtered by availability |
| Operating points | daily top 10% (primary); threshold frozen on Sep–Oct for precision ≥ 0.40 (secondary) |
| Metrics | AP (PR-AUC), ROC AUC, Brier, log loss, ECE (15 fixed-width bins and equal-mass), recall at the frozen threshold, policy precision/recall/F1/confusion counts, 500-replicate day-block bootstrap |
| Seed base | 42 |

The Phase 1 results (`../phase1/results/`, git tag `phase1-final`) are the reference and are not rerun or changed.

## 2. Models in scope

1. **MLP:** learned embeddings for carrier, origin, destination (high-cardinality IDs only as an ablation) plus the numeric Phase 1 features.
2. **GRU model:** the MLP plus a GRU encoder over causal daily delay-history sequences for origin, carrier and route.
3. **Hybrid (main):** Wide & Deep. The wide part is a linear layer on the Phase 1 LogReg inputs, initialised from that fold's LogReg (C = 0.1) and trained jointly; the deep part is the embedding + GRU network with a zero-initialised output layer.
4. **Hybrid (secondary):** low-cardinality entity embeddings learned by the MLP, added as features to the Phase 1 XGBoost configuration.

## 3. Training rules (fixed now)

* **Loss:** binary cross-entropy, no class weighting and no resampling (calibrated probability is a deliverable).
* **Early stopping:** on an inner holdout made of the **last 14 days of each fold's training window**. Never on the validation month, the calibration months or the test months.
* **Final refit:** on all training rows whose labels are available by 2013-09-01, for a **fixed number of epochs = mean of the fold-wise best epochs** (rounded up). No early stopping in the final refit.
* **Seeds:** 5 seeds (42–46). The deployed model is the **mean logit of the 5 seeds**; CV scores that ensemble per fold. Single-seed spread is reported alongside.
* **Search budget:** at most 20 random configurations for the MLP, 12 for the GRU, 12 for the Wide & Deep hybrid, fixed before the search starts. Grids are written into `aerocast_dl/config.py` before the search runs.
* **Hardware:** final runs on CPU with deterministic algorithms enabled, so results are reproducible.

## 4. Selection rules (fixed now)

* **Final-model rule:** among LogReg, XGBoost (Phase 1, as deployed), MLP, GRU and the two hybrids, the model with the **highest mean rolling-CV AP, scored as deployed**. If another model is within **0.005 AP** of the best, the one with fewer trainable parameters is chosen.
* **Calibrator rule:** the Phase 1 Sep→Oct rule, extended with more candidates. Candidates are raw, static Platt, static isotonic, static temperature scaling (each fitted on September labels available by Oct 1), plus causal rolling Platt 28 days and rolling temperature 28 days (each refitted daily on outcomes known by that day's decision clock). All are scored by ECE on October (labels available by Nov 1). The lowest October ECE wins, and it is adopted only if it beats raw by more than 0.005.
* For a fair comparison, the same extended rule is also applied to the saved Phase 1 LogReg and XGBoost scores (`../phase1/results/predictions.csv.gz`), without changing Phase 1 code or results.
* **Threshold:** frozen on Sep–Oct for precision ≥ 0.40, using labels available by 2013-11-01.
* **Rescheduling parameters** (risk reduction δ, shift window, slot-capacity cap): chosen on Sep–Oct only, then frozen for Nov–Dec.
* **Ablations:** CV only. No ablation decision uses calibration or test outcomes.
* A model that loses to LogReg is reported as such. The comparison is part of the result.

## 5. Test lock

* Nov–Dec labels are read only by the final evaluation stage (`run_phase2.py --final`), which refuses to run if this file or `leakage_safeguards.md` has uncommitted changes.
* The final stage is run **once**. If a bug is found afterwards, the fix and rerun are logged in §7 with the reason; the reason may be a correctness bug, never a test metric.
* Disclosure: the Nov–Dec window was already seen during Phase 1 development (`../phase1/docs/development_protocol.md`). Phase 2 rules are fixed here without reference to Phase 1 test results beyond those already published.

## 6. Known, disclosed optimism

* The wide part reuses Phase 1's C = 0.1, which was selected on the same four folds. This applies equally to every model that uses it and is reported in the Phase 2 report.

## 7. Change log

Amendments made before the freeze must cite CV or correctness evidence and are committed with the reason. Amendments after the test run are disclosed here and in the report.

| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | Protocol created | — |
| 2026-11-10 | **Search seed:** the hyperparameter search scores each configuration with one seed (42); only the best configuration of each family is re-scored with the 5-seed ensemble, which is what the selection rule uses | Compute: a 5-seed search costs 5× for no change in what is selected (one seed already ranks configurations; the ensemble is what is compared) |
| 2026-11-10 | **Early-stopping criterion:** inner-holdout log loss (a proper scoring rule), at most 40 epochs, patience 4 epochs | The protocol named the holdout but not the criterion |
| 2026-11-10 | **"As deployed" for networks:** in CV, each fold's network is early-stopped on its inner holdout and then scored once on the validation month; the final refit uses the fixed epoch count of §3 | Clarifies §4 for networks; the validation month never chooses an epoch |
| 2026-11-10 | **Hybrid deep part:** the Wide & Deep search keeps the deep architecture of the better of MLP and GRU model (5-seed CV) and searches learning rate, weight decay and the wide part's learning-rate scale | Fixes the search space before the search |
| 2026-11-10 | **Secondary hybrid:** XGBoost with the Phase 1 configuration (`../phase1/results/summary.json`, `xgb_params`) plus the MLP's carrier, origin and destination embeddings; trained in a separate process | Torch and XGBoost OpenMP runtimes crash in one process on macOS |
| 2026-11-10 | **Search grids** fixed in `src/aerocast_dl/config.py` (`MLP_GRID`, `GRU_GRID`, `HYBRID_GRID`) | Written before the search runs |
| 2026-11-10 | **Correction, supersedes the "as deployed" entry above.** CV now scores the deployed procedure exactly: early stopping on the inner holdout only chooses the epoch count; the network is then refitted on all of the fold's rows for that many epochs and scored once on the validation month. All Phase 2 CV numbers are recomputed. The one completed run under the old definition (MLP stage only: 5-seed CV AP 0.4590) is discarded and reported here | Correctness, found by CV-only diagnostics: the early-stopped model never sees the fold's last 14 days, which the deployed refit does see (LogReg without those days: 0.4714 → 0.4668; MLP refitted: 0.4588 → 0.4644, single seed). Phase 1 applied the same principle to XGBoost (scored at its fixed tree count). No test or calibration data were used |
