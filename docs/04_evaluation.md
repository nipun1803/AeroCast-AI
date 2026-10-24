# 04 · Evaluation

> Summarises [04_evaluation.ipynb](../notebooks/04_evaluation.ipynb) (Parts A and B). All numbers come from `results/` or the executed notebook outputs.

Evidence is kept apart: **Part A** is what any decision rests on (CV / Sep–Oct); **Part B** is exploratory (test). See the [development protocol](development_protocol.md).

## Metric theory (notebook §1)

* **PR-AUC = average precision** (step-wise Σ (R_k − R_{k−1}) P_k, not the trapezoid); baseline = base rate. ROC-AUC ignores the base rate; PR matches what an ops desk sees under imbalance.
* **ECE** = Σ_b |B_b|/n · |ȳ_b − p̄_b|: a *biased* estimator (noise adds, grows with bins) and binning-dependent, so both **equal-width** (15 bins) and **approximately equal-mass** ECE are reported next to the Brier score. Equal-mass ECE starts with 15 balanced rank bins, moves each boundary to the end of its tied-probability group, and collapses duplicate boundaries. Identical probabilities stay together regardless of row order; ties can produce fewer bins or unequal counts. Constant predictions use one bin. It is a marginal check: the *prior* has a small ECE when the test base rate is near the training one (0.007 here), which says nothing about skill.
* **Platt / temperature scaling vs isotonic:** Platt = σ(a·logit p + b), 2 parameters, smooth; temperature scaling is its 1-parameter case; isotonic is a monotone step function (flexible, data-hungry). All assume calibration data resemble deployment data.
* **`oracle_recall@p`:** best recall at precision ≥ p with the threshold tuned **on the evaluated labels**. A point on the PR curve for **comparing models only**, never an operating result.
* **Day-block bootstrap:** whole days are resampled (flights within a day are correlated); model differences are paired on the same days.

# Part A · Evidence used for selection (CV / Sep–Oct)

### A.1 Final model: rolling-CV PR-AUC as deployed (with references)

| model | calibration | cv_pr_auc_mean | cv_pr_auc_std |
|---|---|---|---|
| Prior (base rate) | none | 0.2777 | 0.0481 |
| Route history | none | 0.3477 | 0.0601 |
| LogReg (as deployed) | raw | 0.4715 | 0.0710 |
| XGBoost (as deployed, fixed trees) | raw | 0.4474 | 0.0686 |
| XGBoost (in-fold early stopping, optimistic; not used) | raw | 0.4761 | 0.0703 |

LogReg has the higher development CV mean (0.4715 versus 0.4474), so it is selected by the declared rule. The test AP-difference interval (XGBoost − LogReg, [-0.013, +0.037]) includes zero: it does not establish a test-performance difference, equivalence, or noninferiority. Test exposure during development remains disclosed. The development CV reference scores are route history 0.3477 and prior 0.2777.

### A.2 History settings, decision clock, month (CV)

The selected settings remain **m = 5, window = 14, C = 0.1**. The 20:00 clock scores 0.4715, compared with 0.4589 at 00:00 (approximately a two-day lag) and 0.4814 at 24:00. The 24:00 scenario permits outcomes unavailable at the chosen 20:00 deadline. These scenarios move fitting and selection cutoffs as well as feature-history cutoffs, so their score differences do not isolate a feature-leakage effect.

Adding cyclic month features lowers mean CV AP for both models: LogReg 0.4715 → 0.4691 and fixed-tree XGBoost 0.4474 → 0.4405. For each model, two folds improve and two worsen. Training contains only months 1–8 in one year; these checks do not establish a reusable seasonal effect. **Decision: no month feature**, supported by development CV. The earlier test-based justification remains withdrawn. Details: [02](02_feature_engineering.md).

### A.3 Calibration rule, decided on Sep–Oct

Rule (fixed before test): fit Platt and isotonic on **September**, check ECE on **October**, keep **raw** unless the best recalibrator wins by more than **0.005**; then refit on eligible Sep–Oct outcomes known before the first November prediction (55,377 labels; 251 excluded). September fitting and October checking also enforce their own availability deadlines.

| model | calibrator | check_ece | check_ece_equal_mass | check_brier | gain_vs_raw |
|---|---|---|---|---|---|
| logreg | raw | 0.0625 | 0.0628 | 0.1398 | 0.0000 |
| logreg | platt | 0.0329 | 0.0329 | 0.1368 | 0.0296 |
| logreg | isotonic | 0.0342 | 0.0340 | 0.1376 | 0.0283 |
| xgboost | raw | 0.0199 | 0.0224 | 0.1372 | 0.0000 |
| xgboost | platt | 0.0422 | 0.0422 | 0.1393 | -0.0223 |
| xgboost | isotonic | 0.0421 | 0.0421 | 0.1396 | -0.0223 |

Outcome: **LogReg → Platt (rule-selected)** (October ECE gain 0.0296); **XGBoost → raw (rule-selected)**. Causal **rolling Platt** (refit daily on the trailing N = 28 days of known outcomes; N fixed a priori, 14/56 reported as sensitivity only) is the adaptive alternative, evaluated in Part B.

### A.4 Ablation: CV evidence

| model | calibration | step | cv_gain_mean | cv_folds_improved |
|---|---|---|---|---|
| LogReg | raw | + congestion | 0.0204 | 4/4 |
| LogReg | raw | + history_entity | 0.0203 | 4/4 |
| LogReg | raw | + history_day | 0.0114 | 3/4 |
| XGBoost | raw | + congestion | 0.0105 | 3/4 |
| XGBoost | raw | + history_entity | -0.0004 | 2/4 |
| XGBoost | raw | + history_day | -0.0007 | 3/4 |

At the selected hyperparameters, LogReg mean CV AP increases with congestion (+0.0204, 4/4 folds), entity history (+0.0203, 4/4), and day-level history (+0.0114, 3/4). For XGBoost, the corresponding mean gains are +0.0105 (3/4), −0.0004 (2/4), and −0.0007 (3/4). The last result has a slightly negative mean despite improvement in three folds. These are development comparisons at the chosen settings, not independent estimates after nested selection.

| model | calibration | features | n_features | cv_pr_auc_mean | cv_pr_auc_std |
|---|---|---|---|---|---|
| Prior (base rate) | none | (baseline) | 0 | 0.2777 | 0.0481 |
| Route history | none (smoothed rate) | (baseline) | 0 | 0.3477 | 0.0601 |
| LogReg | raw | schedule + ids | 13 | 0.4193 | 0.0634 |
| LogReg | raw | + congestion | 20 | 0.4398 | 0.0714 |
| LogReg | raw | + history_entity | 38 | 0.4601 | 0.0727 |
| LogReg | raw | + history_day | 45 | 0.4715 | 0.0710 |
| LogReg | raw | history only | 25 | 0.4480 | 0.0760 |
| XGBoost | raw | schedule + ids | 13 | 0.4380 | 0.0770 |
| XGBoost | raw | + congestion | 20 | 0.4485 | 0.0853 |
| XGBoost | raw | + history_entity | 38 | 0.4481 | 0.0807 |
| XGBoost | raw | + history_day | 45 | 0.4474 | 0.0686 |
| XGBoost | raw | history only | 25 | 0.4266 | 0.0831 |

![ablation_cv](../figures/ablation_cv.png)

### A.5 Grouped permutation importance (Sep–Oct)

| group | LogReg | XGBoost |
|---|---|---|
| categorical | 0.0376 | 0.0148 |
| congestion | 0.0468 | 0.0112 |
| history_day | 0.0264 | 0.0174 |
| history_entity | 0.0287 | 0.0195 |
| schedule | 0.0424 | 0.0562 |

Grouped permutation gives LogReg AP drops of congestion 0.047, schedule 0.042, categorical 0.038, entity history 0.029, and day-level history 0.026. For XGBoost: schedule 0.056, entity history 0.019, congestion 0.011, categorical 0.015, and day-level history 0.017. These describe sensitivity to shuffled feature groups; they do not identify causal effects. For XGBoost, the ordering by drop is schedule, entity history, day-level history, categorical, then congestion.

![perm_importance_groups](../figures/perm_importance_groups.png)

# Part B · Exploratory test-set findings (ablation test diffs, calibration study, target sensitivity)

Test = Nov–Dec 2013. The current rules use the development evidence described in Part A. Test numbers were visible during development, so these findings are exploratory and do not become an untouched confirmation merely by re-justifying the rules.

## B.1 Pre-specified rules on test

### Headline test metrics: references, calibration variants, final model

| model | calibration | pr_auc | roc_auc | brier | ece | ece_equal_mass | oracle_recall@p0.40 |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.258 | 0.500 | 0.192 | 0.007 | 0.007 | 0.000 |
| Route history | none (smoothed rate) | 0.353 | 0.621 | 0.187 | 0.037 | 0.038 | 0.244 |
| LogReg (raw) | raw | 0.394 | 0.654 | 0.181 | 0.019 | 0.022 | 0.431 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | 0.394 | 0.654 | 0.187 | 0.082 | 0.082 | 0.431 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | 0.392 | 0.657 | 0.181 | 0.027 | 0.027 | 0.431 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | 0.404 | 0.662 | 0.180 | 0.027 | 0.026 | 0.462 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | 0.394 | 0.659 | 0.181 | 0.026 | 0.025 | 0.417 |

`LogReg (Platt, rule-selected)` is the model as the pre-declared rule deploys it. Its ranking equals `LogReg (raw)` (Platt is monotone), but its calibration does not: **test ECE 0.082 (Platt, rule-selected) vs 0.019 (raw) vs 0.027 (rolling Platt 28d)**. **The static rule fails in December** (below). Both learned models beat both references on PR-AUC (0.394 LogReg, 0.404 XGBoost, 0.353 route history, 0.258 prior); the prior's low fixed-width and equal-mass ECE (both 0.007) reflects close overall means, not ranking skill. `oracle_recall@p0.40` is a comparison metric only.

### November and December

| model | calibration | split | base_rate | pr_auc | roc_auc | brier | ece |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | test_m11 | 0.181 | 0.181 | 0.500 | 0.155 | 0.084 |
| Route history | none (smoothed rate) | test_m11 | 0.181 | 0.221 | 0.560 | 0.149 | 0.035 |
| LogReg (raw) | raw | test_m11 | 0.181 | 0.292 | 0.655 | 0.145 | 0.056 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | test_m11 | 0.181 | 0.292 | 0.655 | 0.142 | 0.022 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | test_m11 | 0.181 | 0.286 | 0.650 | 0.142 | 0.006 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | test_m11 | 0.181 | 0.267 | 0.634 | 0.144 | 0.017 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | test_m11 | 0.181 | 0.259 | 0.627 | 0.144 | 0.011 |
| Prior (base rate) | none | test_m12 | 0.335 | 0.335 | 0.500 | 0.228 | 0.070 |
| Route history | none (smoothed rate) | test_m12 | 0.335 | 0.394 | 0.574 | 0.225 | 0.060 |
| LogReg (raw) | raw | test_m12 | 0.335 | 0.456 | 0.632 | 0.217 | 0.059 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | test_m12 | 0.335 | 0.456 | 0.632 | 0.233 | 0.143 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | test_m12 | 0.335 | 0.432 | 0.603 | 0.220 | 0.050 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | test_m12 | 0.335 | 0.463 | 0.634 | 0.216 | 0.061 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | test_m12 | 0.335 | 0.438 | 0.612 | 0.217 | 0.042 |

### LogReg vs XGBoost: day-block bootstrap (rule-selected variants)

| metric | XGBoost 95% CI | LogReg 95% CI | XGB − LR 95% CI |
|---|---|---|---|
| pr_auc | [0.346, 0.466] | [0.340, 0.450] | [-0.013, +0.037] |
| roc_auc | [0.631, 0.692] | [0.620, 0.682] | [-0.013, +0.027] |

XGBoost's test AP is higher by 0.0105, but the paired 95% interval [−0.013, +0.037] includes zero. This does not establish a difference, equivalence, or noninferiority; it does not replace the development CV selection rule.

![pr_roc_test](../figures/pr_roc_test.png)

![reliability_logreg_test](../figures/reliability_logreg_test.png)

![reliability_xgboost_test](../figures/reliability_xgboost_test.png)

### Operating policies (fixed before test)

* **Primary:** each day flag the top **10%** riskiest flights (a-priori capacity assumption; rank-based, so a monotone within-day recalibration cannot change it; ties broken by a seeded random draw).
* **Secondary:** a score threshold frozen on Sep–Oct to reach precision ≥ 0.4 (0.50 was unreachable on Sep–Oct, so it was lowered to 0.40).

| model | calibration | role | policy | flag_rate | precision | recall | lift | threshold |
|---|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | primary | daily top 10% | 0.101 | 0.243 | 0.095 | 0.941 | – |
| Prior (base rate) | none | secondary | frozen thr for P>=0.4 (from Sep-Oct) | 0.000 | – | 0.000 | – | ∞ |
| Route history | none (smoothed rate) | primary | daily top 10% | 0.101 | 0.358 | 0.139 | 1.387 | – |
| Route history | none (smoothed rate) | secondary | frozen thr for P>=0.4 (from Sep-Oct) | 0.059 | 0.425 | 0.097 | 1.646 | 0.468907 |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | primary | daily top 10% | 0.101 | 0.445 | 0.173 | 1.722 | – |
| LogReg (Platt, rule-selected) | Platt (rule-selected) | secondary | frozen thr for P>=0.4 (from Sep-Oct) | 0.074 | 0.495 | 0.142 | 1.917 | 0.330087 |
| XGBoost (raw, rule-selected) | raw (rule-selected) | primary | daily top 10% | 0.101 | 0.431 | 0.168 | 1.667 | – |
| XGBoost (raw, rule-selected) | raw (rule-selected) | secondary | frozen thr for P>=0.4 (from Sep-Oct) | 0.088 | 0.489 | 0.167 | 1.892 | 0.416729 |

**Primary result (LogReg, static Platt rule-selected, test): precision 0.445, recall 0.173, lift 1.72×.** The prior lift is 0.94, route history 1.39, and XGBoost 1.67. The LogReg threshold frozen on eligible Sep–Oct outcomes is 0.330087; it flags 3.8% of calibration flights and 7.4% of test flights. Test precision is 0.495 and recall 0.142. Ranking-based selection keeps the chosen daily share; its precision and recall can still change across periods.

## B.2 Exploratory findings

### Calibration study: every variant, Nov and Dec separately

| model | calibration | ece_test | ece_test_m11 | ece_test_m12 | ece_equal_mass_test | brier_test | brier_test_m11 | brier_test_m12 |
|---|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.0067 | 0.0840 | 0.0704 | 0.0067 | 0.1916 | 0.1553 | 0.2279 |
| Route history | none (smoothed rate) | 0.0373 | 0.0352 | 0.0602 | 0.0380 | 0.1869 | 0.1492 | 0.2246 |
| LogReg (raw) | raw | 0.0191 | 0.0557 | 0.0594 | 0.0218 | 0.1811 | 0.1455 | 0.2168 |
| LogReg (Platt, static) | Platt (static, fit Sep-Oct) | 0.0824 | 0.0217 | 0.1432 | 0.0824 | 0.1874 | 0.1418 | 0.2329 |
| LogReg (isotonic, static) | isotonic (static, fit Sep-Oct) | 0.0824 | 0.0234 | 0.1425 | 0.0810 | 0.1874 | 0.1419 | 0.2329 |
| LogReg (rolling Platt 14d) | rolling Platt 14d (causal) | 0.0216 | 0.0110 | 0.0382 | 0.0230 | 0.1807 | 0.1431 | 0.2181 |
| LogReg (rolling Platt 28d) | rolling Platt 28d (causal) | 0.0270 | 0.0061 | 0.0501 | 0.0266 | 0.1809 | 0.1419 | 0.2199 |
| LogReg (rolling Platt 56d) | rolling Platt 56d (causal) | 0.0510 | 0.0135 | 0.0888 | 0.0510 | 0.1820 | 0.1416 | 0.2223 |
| XGBoost (raw) | raw | 0.0272 | 0.0167 | 0.0608 | 0.0263 | 0.1798 | 0.1438 | 0.2158 |
| XGBoost (Platt, static) | Platt (static, fit Sep-Oct) | 0.0729 | 0.0314 | 0.1168 | 0.0729 | 0.1847 | 0.1442 | 0.2252 |
| XGBoost (isotonic, static) | isotonic (static, fit Sep-Oct) | 0.0708 | 0.0325 | 0.1126 | 0.0708 | 0.1845 | 0.1444 | 0.2246 |
| XGBoost (rolling Platt 14d) | rolling Platt 14d (causal) | 0.0282 | 0.0128 | 0.0502 | 0.0267 | 0.1802 | 0.1443 | 0.2160 |
| XGBoost (rolling Platt 28d) | rolling Platt 28d (causal) | 0.0261 | 0.0106 | 0.0422 | 0.0252 | 0.1806 | 0.1440 | 0.2172 |
| XGBoost (rolling Platt 56d) | rolling Platt 56d (causal) | 0.0362 | 0.0215 | 0.0527 | 0.0359 | 0.1804 | 0.1436 | 0.2172 |

Static Platt has LogReg ECE 0.022 in November and 0.143 in December (December base rate 33.5%): **the static rule fails in December**. Raw LogReg has test ECE 0.019 overall, but its monthly errors are larger; a close overall mean prediction and base rate do not establish reliable probabilities in every month. Rolling Platt with the pre-specified 28-day window has test ECE 0.027 for LogReg (November 0.006, December 0.050) and 0.026 for XGBoost. LogReg sensitivity windows of 14 and 56 days give test ECE 0.022 and 0.051. These test diagnostics do not select a new final variant or prove a cause of the errors.

![monthly_drift_final](../figures/monthly_drift_final.png)

### Ablation: exploratory test differences

| model | calibration | step | test_diff | test_diff_ci_lo | test_diff_ci_hi |
|---|---|---|---|---|---|
| LogReg | raw | + congestion | 0.0073 | -0.0093 | 0.0276 |
| LogReg | raw | + history_entity | 0.0321 | 0.0152 | 0.0487 |
| LogReg | raw | + history_day | 0.0095 | -0.0089 | 0.0272 |
| XGBoost | raw | + congestion | -0.0025 | -0.0166 | 0.0128 |
| XGBoost | raw | + history_entity | 0.0548 | 0.0301 | 0.0764 |
| XGBoost | raw | + history_day | 0.0054 | -0.0290 | 0.0428 |

LogReg test AP differences are: congestion +0.0073 [−0.0093, +0.0276], entity history +0.0321 [+0.0152, +0.0487], and day-level history +0.0095 [−0.0089, +0.0272]. XGBoost differences are −0.0025 [−0.0166, +0.0128], +0.0548 [+0.0301, +0.0764], and +0.0054 [−0.0290, +0.0428], respectively. Only the entity-history intervals exclude zero in these exploratory comparisons. These results do not establish causal contributions.

### Target sensitivity (cancelled and diverted = delayed)

| target | model | calibration | n_test | base_rate | pr_auc | roc_auc | brier | ece | top10_precision | top10_recall | top10_lift | xgb_minus_lr_pr_auc_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arr_delay>=15 (main) | Prior (base rate) | none | 53991 | 0.258 | 0.258 | 0.500 | 0.192 | 0.007 | 0.243 | 0.095 | 0.941 | – |
| arr_delay>=15 (main) | Route history | none (smoothed rate) | 53991 | 0.258 | 0.353 | 0.621 | 0.187 | 0.037 | 0.358 | 0.139 | 1.387 | – |
| arr_delay>=15 (main) | LogReg (raw) | raw | 53991 | 0.258 | 0.394 | 0.654 | 0.181 | 0.019 | 0.445 | 0.173 | 1.722 | [-0.013, +0.037] |
| arr_delay>=15 (main) | XGBoost (raw) | raw | 53991 | 0.258 | 0.404 | 0.662 | 0.180 | 0.027 | 0.431 | 0.168 | 1.667 | [-0.013, +0.037] |
| arr_delay>=15 OR cancelled/diverted | Prior (base rate) | none | 55403 | 0.277 | 0.277 | 0.500 | 0.200 | 0.011 | 0.262 | 0.095 | 0.946 | – |
| arr_delay>=15 OR cancelled/diverted | Route history | none (smoothed rate) | 55403 | 0.277 | 0.374 | 0.621 | 0.196 | 0.047 | 0.377 | 0.137 | 1.359 | – |
| arr_delay>=15 OR cancelled/diverted | LogReg (raw) | raw | 55403 | 0.277 | 0.446 | 0.661 | 0.186 | 0.024 | 0.479 | 0.174 | 1.729 | [-0.013, +0.037] |
| arr_delay>=15 OR cancelled/diverted | XGBoost (raw) | raw | 55403 | 0.277 | 0.455 | 0.672 | 0.185 | 0.029 | 0.474 | 0.172 | 1.708 | [-0.013, +0.037] |

Counting cancellation/diversion proxies as delayed raises the test base rate from 25.8% to 27.7%. In that sensitivity run, raw LogReg AP is 0.446 and raw XGBoost AP is 0.455; both exceed the reference scores. Their test AP-difference interval is [−0.013, +0.037], which does not establish a difference or equivalence. Top-10% lifts are 1.73 for LogReg, 1.71 for XGBoost, and 0.95 for the prior. This is one alternative target check, not proof that findings are independent of all missing-outcome definitions.

### Covariate shift: PSI and adversarial validation

| feature | group | psi_calib | psi_test |
|---|---|---|---|
| days_to_holiday | schedule | 0.261 | 4.769 |
| hist_logn_origin_7d | history_entity | 2.269 | 1.265 |
| hist_rate_network_7d | history_day | 5.147 | 1.058 |
| hist_rate_origin_7d | history_entity | 3.433 | 0.755 |
| sched_origin_day | congestion | 2.056 | 0.610 |
| hist_rate_carrier_14d | history_entity | 2.631 | 0.383 |
| hist_rate_dest_14d | history_entity | 2.143 | 0.292 |
| hist_logn_carrier_14d | history_entity | 0.472 | 0.289 |
| hist_cancel_network_1d | history_day | 1.229 | 0.222 |
| hist_rate_carrier_7d | history_entity | 1.699 | 0.209 |

| comparison | features | n_features | adversarial_auc |
|---|---|---|---|
| train vs calib | all features | 45 | 1.000 |
| train vs calib | categorical | 3 | 0.562 |
| train vs calib | schedule | 10 | 0.785 |
| train vs calib | congestion | 7 | 0.983 |
| train vs calib | history_entity | 18 | 1.000 |
| train vs calib | history_day | 7 | 1.000 |
| train vs test | all features | 45 | 1.000 |
| train vs test | categorical | 3 | 0.562 |
| train vs test | schedule | 10 | 0.921 |
| train vs test | congestion | 7 | 0.963 |
| train vs test | history_entity | 18 | 0.998 |
| train vs test | history_day | 7 | 1.000 |

PSI: the largest shifts into Sep–Oct are hist_rate_network_7d, hist_rate_origin_7d, hist_rate_carrier_14d (history rates follow the season by design), and into Nov–Dec days_to_holiday, hist_logn_origin_7d, hist_rate_network_7d; `days_to_holiday` is the largest (PSI 4.8) because Thanksgiving, Christmas and New Year all fall in the test window. Adversarial AUC: **categorical** 0.56 (vs calib) / 0.56 (vs test); **schedule** 0.78 (vs calib) / 0.92 (vs test); **congestion** 0.98 (vs calib) / 0.96 (vs test); **history_entity** 1.00 (vs calib) / 1.00 (vs test); **history_day** 1.00 (vs calib) / 1.00 (vs test). The IDs (carrier, origin, destination) are nearly stable; schedule and congestion shift through season and holiday proximity; history gives high row-level period separability in this diagnostic.

PSI describes marginal feature-distribution changes; the row-split adversarial classifier describes separability of sampled rows from different periods. Daily feature values are repeated across flights, so near-perfect separability should not be interpreted as verified generalisation to unseen days. The diagnostics are consistent with distribution change, but do not establish that it caused the calibration errors or that the delay models memorised dates. The underlying diagnostic implementation is unchanged.

![psi_top15](../figures/psi_top15.png)

![xgb_gain_top20](../figures/xgb_gain_top20.png)


## Decision support: rescheduling / add-buffer candidates (test)

Each test day the final model's ranking selects the top-k% flights as candidates for a capacity-limited review desk (k = 5, 10, 15, 20), against the prior (random) and route-history rankings. Source: `results/rescheduling_candidates.csv`.

| model | top-k% | flagged/day | precision | recall | lift | delayed per 100 flagged |
|---|---|---|---|---|---|---|
| Prior (base rate) | 5% | 45 | 0.251 | 0.049 | 0.97 | 25.1 |
| Prior (base rate) | 10% | 89 | 0.243 | 0.095 | 0.94 | 24.3 |
| Prior (base rate) | 15% | 133 | 0.250 | 0.145 | 0.97 | 25.0 |
| Prior (base rate) | 20% | 177 | 0.252 | 0.195 | 0.97 | 25.2 |
| Route history | 5% | 45 | 0.386 | 0.076 | 1.49 | 38.6 |
| Route history | 10% | 89 | 0.358 | 0.139 | 1.39 | 35.8 |
| Route history | 15% | 133 | 0.346 | 0.201 | 1.34 | 34.6 |
| Route history | 20% | 177 | 0.339 | 0.263 | 1.31 | 33.9 |
| LogReg (Platt, rule-selected) | 5% | 45 | 0.470 | 0.092 | 1.82 | 47.0 |
| LogReg (Platt, rule-selected) | 10% | 89 | 0.445 | 0.173 | 1.72 | 44.5 |
| LogReg (Platt, rule-selected) | 15% | 133 | 0.422 | 0.246 | 1.64 | 42.2 |
| LogReg (Platt, rule-selected) | 20% | 177 | 0.406 | 0.315 | 1.57 | 40.6 |
| XGBoost (raw, rule-selected) | 5% | 45 | 0.466 | 0.091 | 1.80 | 46.6 |
| XGBoost (raw, rule-selected) | 10% | 89 | 0.431 | 0.168 | 1.67 | 43.1 |
| XGBoost (raw, rule-selected) | 15% | 133 | 0.413 | 0.240 | 1.60 | 41.3 |
| XGBoost (raw, rule-selected) | 20% | 177 | 0.399 | 0.310 | 1.55 | 39.9 |

* **This shows ranking quality, not realised operational savings.** The data contain no cost, aircraft-rotation or schedule-change model, so the benefit of moving or buffering a flight cannot be estimated.
* The brief's exact definition of rescheduling was not supplied; anything beyond this ranking analysis (cost model, feasibility, intervention effect) is missing, not invented.
