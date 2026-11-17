# 06 · Rules fixed in advance, changes and limitations

> The test lock (`src/aerocast_dl/testlock.py`) refuses to read test labels while this file or [01_inputs_and_leakage.md](01_inputs_and_leakage.md) has uncommitted changes. The rules below were first committed on 2026-11-10 (commit `e6baa42`), before any Phase 2 model was trained. Git history shows every later edit.

## Why fix rules in advance

With four CV folds and one test window, it is easy to keep adjusting a model until it wins. Phase 2 therefore fixed, before training, how models are chosen, calibrated and tested, and logged every later change with its reason.

## The rules

**Kept identical to Phase 1** (imported from `aerocast`, not copied):
- the 20:00 D−1 decision clock;
- the target (`arr_delay ≥ 15`, cancelled and diverted flights excluded);
- the 43 features;
- the split (train 8 Jan – 31 Aug, calibration Sep–Oct, test Nov–Dec);
- the four rolling CV folds (validation May–Aug);
- the operating points (daily top 10%; threshold frozen on Sep–Oct for precision ≥ 0.40);
- all metrics, including the 500-replicate day-block bootstrap.

**Training:**
- **Loss:** binary cross-entropy with no class weights.
- **Early stopping:** only on the last 14 days of each fold's training window.
- **Final refit:** on training rows known by 1 September, for the mean CV best epoch (rounded up).
- **Seeds:** 5 seeds, mean logit.
- **Search budgets:** at most 20 MLP, 12 GRU and 12 Wide & Deep configurations, with grids fixed in `config.py` before searching.

**Selection:**
- **Final model:** the highest mean CV AP, scored as deployed, among the six models. Within 0.005 of it, the one with fewer trainable parameters.
- **Calibrator:** fit candidates on September (or rolling over 28 days), score each by October ECE, and adopt the best only if it beats raw by more than 0.005.
- **Thresholds and rescheduling parameters:** chosen on Sep–Oct only.
- **Ablations:** CV only.
- **Losing models:** a model that loses to LogReg is reported as losing.

**Test lock:**
- Nov–Dec labels are scored only through `run_phase2.py --final`. The lock is an environment flag that only `--final` sets, plus a check that this file and `01` are committed. Earlier stages still load the labelled table (the Phase 1 `prepare()` does), and `write_predictions` uses the test labels only to check that rows line up; no earlier stage scores, fits or selects on them.
- That evaluation was meant to run once. It ran twice: the second run followed the 2026-11-17 audit fixes below.
- Any later fix is logged below and disclosed in the report.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | Search scores each configuration with one seed; only each family's best is re-scored with 5 seeds | compute; the 5-seed ensemble is what gets compared |
| 2026-11-10 | Early-stopping criterion fixed: holdout log loss, patience 4, at most 40 epochs | the original rules named the holdout but not the criterion |
| 2026-11-10 | Wide & Deep's deep part = the better of MLP and GRU model; its search covers learning rate, weight decay and the wide part's learning-rate scale | fixes the search space before searching |
| 2026-11-10 | Embeddings → XGBoost uses the Phase 1 XGBoost settings, in a separate process | PyTorch and XGBoost crash together on macOS (OpenMP) |
| 2026-11-10 | Search grids written into `config.py` | fixed before the searches ran |
| 2026-11-10 | **Correction.** CV now retrains each network on all of the fold's rows for the early-stopped epoch count before scoring it. The one MLP run under the old procedure (CV AP 0.4590) was discarded and every number recomputed | correctness, found by CV-only diagnostics: the old procedure scored a model that never saw the fold's last 14 days ([03](03_cv_results_and_ablations.md)) |
| 2026-11-11 | Descriptive test breakdowns (slices, PR curves, confusion matrices, drift) added to the test stage before it ran | they choose nothing |
| 2026-11-11 | **After the test run** (no rerun): (a) XGBoost's rows were written twice to `test_metrics.csv` because its calibrator is `raw`; duplicates removed and loop fixed. (b) Three figures redrawn for legibility. (c) `torch.use_deterministic_algorithms(True)` added, as the training rule required; a full Wide & Deep fit gives bit-identical output with and without it. (d) The rescheduling grid (shift ±1/±2 h, δ ∈ {0.01, 0.02, 0.05}), cost 0.02 and capacity cap were fixed in code at commit `97e5d43`, before tuning and testing | bookkeeping and disclosure; no model, calibrator, threshold, rescheduling parameter or test metric changed |
| 2026-11-11 | Documentation restructured: the separate protocol, leakage and reproducibility files were folded into these six documents; the test lock now watches this file and `01` | readability; the rules' content is unchanged |
| 2026-11-17 | **Audit fixes, then a second test run.** (a) For Wide & Deep's early-stopping run, the wide LogReg is fitted on the inner training rows only; it had been fitted on all of the fold's rows, so it had seen the 14-day holdout labels that choose the epoch count. (b) Rescheduling candidates, the daily flag budgets and the origin-day risk sums use the whole timetable; they had left out flights later cancelled or diverted, which is only known after day D. (c) The test flights without an outcome are scored by every model and join the daily top-k budgets; precision and recall are still measured on flights with an outcome. Rerun: Wide & Deep search and ablations, figures, final fits, calibration, rescheduling tuning, then the test stage a second time. MLP, GRU, embeddings → XGBoost and diagnostics are unaffected (no wide part) and keep their numbers | correctness, found by a code audit of the inputs and cohorts, not from test results. No search grid, rule, calibrator candidate or rescheduling parameter grid changed. Nov–Dec is read a second time; the first run's numbers stay in git history (commit `020bf24`) and the report states the difference |
| 2026-11-17 | **Second test run, outcome.** Wide & Deep CV AP 0.4691 → 0.4684 and its selected learning rate 3e-4 → 1e-3; it stays within 0.005 of LogReg, so the final model (LogReg) and the rescheduling rule (±2 h, δ = 0.02) are unchanged. Test: Wide & Deep AP 0.399 → 0.394; the other five models' metrics are identical; top-k and rescheduling numbers move slightly with the whole-timetable budget. Both Wide & Deep ablations reversed sign (they had seemed to help). Report and docs updated, with the audit's text corrections (drift claim, theory section, test-lock wording, disclosures) | reporting |

## Limitations

- **One year, three airports.** Nothing here shows the result holds in another year or at other airports.
- **December.** The late rate jumps to 33.5%. Calibration error rises to 0.050 and frozen thresholds stop flagging. Only rankings within a day stay reliable.
- **Small samples.** Four CV folds and two test months are few. The CV ranking was fixed before testing and is kept even though the test order differs. Slices are descriptive, with no intervals.
- **Rescheduling** is a counterfactual on predicted risk ([05](05_whatif_rescheduling.md)), not a measured saving.
- **No weather.** Observed weather would leak at the decision clock, and forecasts are not in the data (as in Phase 1).

## Disclosed optimism

- **Reused penalty.** The wide part of Wide & Deep reuses Phase 1's LogReg penalty (C = 0.1), which was chosen on the same four folds. This slightly flatters LogReg and Wide & Deep on CV.
- **Test months already seen.** The Nov–Dec window was seen during Phase 1 development (see Phase 1's documentation). No Phase 2 choice used it.
- **Searches on the reported folds.** The hyperparameter searches used the same four folds whose CV is reported, so the networks' CV AP is slightly optimistic. This makes LogReg's win safer, not weaker.
- **Tail-number history** assumes the aircraft assigned to a flight is known at 20:00 on D−1.
- **Bootstrap.** The day-block bootstrap treats days as independent; neighbouring days are correlated, so the intervals are, if anything, too narrow.
- **Rescheduling gain** is the same predicted gain that chose the moves, so it is biased upward (the optimiser's curse).
- **Static calibrators** are fitted on Sep–Oct when applied there, which would be in-sample; no reported number uses this path (every model uses a rolling calibrator or raw scores).
- **Early-September calibration.** The rolling calibrator's window for the first days of September includes late-August training rows, which the final models have seen. This affects only the frozen thresholds and the rescheduling tuning, not test calibration or model choice. Phase 1 has the same property.
