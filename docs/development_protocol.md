# Development protocol (disclosure)

> Mirrors [notebook 04 §0](../notebooks/04_evaluation.ipynb). The test window (Nov–Dec) was meant to be touched once; here is what was seen and what each decision rests on.

## Diagnostics that printed test metrics before the design was frozen

1. **First run** (single-September tuning): test PR-AUC and calibration of all models were printed; the design was abandoned because of September evidence (XGBoost stopped at 13 trees).
2. **Month features:** a with/without comparison printed test PR-AUC; that was the original justification for removal. **Withdrawn**, re-justified by rolling CV ([02](02_feature_engineering.md)).
3. **Feature-subset variants** printed calibration and test PR-AUC and test recall@precision; they shaped the "day-level features memorise days" hypothesis, now re-justified by distinct-value analysis and the CV ablation.
4. **Precision target** 0.50 → 0.40: decided because 0.50 is unreachable on Sep–Oct (base rate 15.7%), but test recall at 0.35/0.40/0.50 was printed by the same diagnostic.
5. **Calibration:** test ECE of raw/Platt/isotonic was seen before any selection rule existed; the Sep→Oct rule, the shift-aware Platt variant (scored on test ECE) and the 0.005 margin were introduced afterwards.
6. **First review round:** the full pipeline ran once while the final-model comparison still used XGBoost's optimistic in-fold early-stopping score; corrected for a methodological reason found from CV alone, which reversed the choice to LogReg.
7. **Baselines in the policy table:** a constant score showed an apparent lift (Sep–Oct and test). Cause: order-based tie-breaking in the daily policy (rows are in departure order). Now a seeded random tie-break; model scores, policy definition and the 10% share unchanged.
8. **Second review round (items 1–11):** labels, baselines, assertions, documentation. No modelling decision changed; joint re-tuning of (m, window, C) re-selected the same values.

## Decided without test evidence

| Decision | Evidence |
|---|---|
| Decision clock 20:00 (vs 2-day lag) | rolling CV: 0.4705 vs 0.4587 |
| m = 5, window = 14 days, C = 0.1 | rolling CV, tuned jointly |
| No month feature | rolling CV: LogReg 0.4705 vs 0.4686 |
| Day-level history group kept (LogReg) | CV ablation: +0.011, 3/4 folds |
| XGBoost configuration | rolling CV |
| Final model = LogReg | CV as deployed 0.4705 vs 0.4450 (rule declared before test) |
| Calibration rule outcome (Platt for LogReg, raw for XGBoost) | Sep → Oct |
| Rolling-Platt window N = 28 | a priori (= entity-history window scale); 14/56 sensitivity only |
| Daily top-10% policy | a-priori capacity assumption |
| Precision target 0.40 | Sep–Oct reachability |

**Implication:** test numbers are not perfectly clean confirmations; read small test differences with caution.
