# Development protocol (disclosure)

> Mirrors [notebook 04 §0](../notebooks/04_evaluation.ipynb). The Nov–Dec window was reserved for final evaluation but was inspected repeatedly during development. This disclosure records that exposure and the current selection evidence; it does not restore an untouched test set.

## Diagnostics that printed test metrics before the design was frozen

1. **First run** (single-September tuning): test PR-AUC and calibration of all models were printed; the design was abandoned because of September evidence (XGBoost stopped at 13 trees).
2. **Month features:** a with/without comparison printed test PR-AUC; that was the original justification for removal. **Withdrawn**, re-justified by rolling CV ([02](02_feature_engineering.md)).
3. **Feature-subset variants** printed calibration and test PR-AUC and test recall@precision; they shaped the "day-level features memorise days" hypothesis, now re-justified by distinct-value analysis and the CV ablation.
4. **Precision target** 0.50 → 0.40: decided because 0.50 is unreachable on Sep–Oct (base rate 15.7%), but test recall at 0.35/0.40/0.50 was printed by the same diagnostic.
5. **Calibration:** test ECE of raw/Platt/isotonic was seen before any selection rule existed; the Sep→Oct rule, the shift-aware Platt variant (scored on test ECE) and the 0.005 margin were introduced afterwards.
6. **First review round:** the full pipeline ran once while the final-model comparison still used XGBoost's optimistic in-fold early-stopping score; corrected for a methodological reason found from CV alone, which reversed the choice to LogReg.
7. **Baselines in the policy table:** a constant score showed an apparent lift (Sep–Oct and test). Cause: order-based tie-breaking in the daily policy (rows are in departure order). Now a seeded random tie-break; model scores, policy definition and the 10% share unchanged.
8. **Second review round (items 1–11):** labels, baselines, assertions, documentation. No modelling decision changed; joint re-tuning of (m, window, C) re-selected the same values.
9. **Leakage audit (2026-10-26):** a realistic-cancellation test showed that `tail_leg_idx` / `tail_legs_day` changed when a sibling leg was cancelled on day D (the data blank the tail number of cancelled flights). Both were removed on leakage grounds; the decision used the test and CV evidence. An earlier sensitivity refit without all four tail features had already printed a test AP, and the full rerun's test metrics were seen afterwards.

## Current selection evidence (retain the exposure disclosure above)

| Decision | Evidence |
|---|---|
| Decision clock 20:00 (vs 2-day lag) | rolling CV: 0.4714 vs 0.4590; fitting and history clocks both move |
| m = 5, window = 14 days, C = 0.1 | rolling CV, tuned jointly |
| No month feature | rolling CV of the final model: LogReg 0.4714 vs 0.4691 (XGBoost 0.4500 vs 0.4548, but three of four folds worse) |
| Day-level history group kept (LogReg) | CV ablation: +0.0114, 3/4 folds |
| XGBoost configuration | rolling CV |
| Final model = LogReg | fixed-tree development CV: 0.4714 vs 0.4500 |
| Calibration rule outcome (Platt for LogReg, raw for XGBoost) | Sep → Oct |
| Rolling-Platt window N = 28 | pre-specified 28-day adaptation window; 14/56 sensitivity only; selected long history window is 14 days |
| Daily top-10% policy | a-priori capacity assumption |
| Precision target 0.40 | Sep–Oct reachability |
| No aircraft-rotation features | leakage: realistic-cancellation test (`leakage.cancellation_check`) |

**Implication:** test numbers are exploratory because of the exposure above. Selecting configurations on the same development folds also makes their reported CV scores selection scores rather than independent nested-CV estimates.

## Fitting-time cutoff correction

The D-1 20:00 clock applies to fitting and selection as well as historical
features. Each CV fold fits only labels available before its first validation
prediction; CV scoring excludes labels unavailable when the selected model
must be ready for September 1. Final classifiers, the prior and diagnostic
refits use only training labels known by August 31 at 20:00. September
calibrator candidates use labels known before October 1 predictions; candidate
selection, the static refit and the frozen threshold use only calibration
labels known by October 31 at 20:00. Unknown availability is excluded.

This removes 167 labels from final model fitting and 251 labels from static
calibration/threshold selection, while retaining their flights for subsequent
offline scoring. `results/summary.json` records the fitting counts. Tests in
`tests/test_fitting_clock.py` check clock equality, late/unknown outcomes and
invariance to changing labels that were unavailable at a fitting deadline.
