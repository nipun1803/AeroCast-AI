# 02 · Feature engineering

> Summarises [02_feature_engineering.ipynb](../notebooks/02_feature_engineering.ipynb) (§1–§9). All numbers come from `results/` or the executed notebook outputs.

## Decision clock (notebook §1)

Predictions for all flights on day **D** are issued at **20:00 NYC time on D-1**. An outcome may feed a feature for D only if it was **known by then**: an arrival delay at the actual gate arrival (converted to the NYC clock), a cancellation at the scheduled/actual departure. `data.available_date(t) = ceil_day(t − 20h) + 1 day`, and history windows roll over these **availability dates**, not flight dates. Cancelled flights stay in the schedule (congestion) counts.

| usable_after_days | flights | delay_rate | share |
|---|---|---|---|
| 1 | 225400 | 0.183 | 0.689 |
| 2 | 101946 | 0.381 | 0.311 |

**31% of D-1 arrivals land after 20:00** (`usable_after_days = 2`) and are twice as often late (38% vs 18%).

## Time-zone table and DST (notebook §1.1)

`arr_time` is destination-local. Offsets are a **fixed lookup table** (`config.DEST_OFFSET_MIN`) plus the 2013 DST rule (2013-03-10 02:00 → 2013-11-03 02:00; HNL, PHX, SJU, BQN, PSE, STT do not observe DST). It must not be estimated from the flights being featurised, or past features would depend on future data (this was an actual bug caught by the future-dependence test). The table is re-derived from the data in the notebook and checked by a test.

| offset vs NYC (min) | destinations |
|---|---|
| -360 | 1 |
| -240 | 1 |
| -180 | 14 |
| -120 | 8 |
| -60 | 21 |
| 0 | 60 |

## Features (45) and smoothing (notebook §2–§4)

| group | n_features |
|---|---|
| categorical | 3 |
| schedule | 10 |
| congestion | 7 |
| history_entity | 18 |
| history_day | 7 |

Entity history rates use beta-binomial shrinkage toward a past-only network prior: `(late + m·prior)/(flights + m)`, with **m = 5** and a **@L@-day** long window (7-day short window). Worked example (EWR→CAE) in the notebook.

![feat_history_day_dist](../figures/feat_history_day_dist.png)

## Leakage checks (notebook §5; also `tests/test_no_leakage.py`)

**(a) Future outcomes.** Overwrite all post-departure columns from 2013-06-15 onward with random values, rebuild features: cells changed up to the cutoff = **0**, after the cutoff = 4,558,344 (non-vacuous).

**(b) Decision clock.** Randomise the arrival outcomes of D-1 flights that landed after 20:00: D's features must not change; randomising those before 20:00 must change them (positive control).

| target_date | changed_cells_on_D_after_scrambling_late | changed_cells_on_D_after_scrambling_early |
|---|---|---|
| 2013-02-14 | 0 | 12428 |
| 2013-07-11 | 0 | 13078 |
| 2013-12-20 | 0 | 12740 |

## Clock vs a 2-day lag vs the leaky variant (notebook §6)

Rolling-CV PR-AUC of LogReg at the configured (m, window), C tuned per row:

| decision_hour | best_C | cv_pr_auc_mean | cv_pr_auc_std | meaning |
|---|---|---|---|---|
| 20 | 0.1000 | 0.4705 | 0.0706 | 20:00 on D-1 (used) |
| 0 | 0.0030 | 0.4587 | 0.0730 | 00:00 on D-1 (about a 2-day lag) |
| 24 | 0.0100 | 0.4812 | 0.0723 | 24:00 on D-1 (every D-1 arrival counts: leaky) |

The 20:00 clock beats a 2-day lag by 0.012 (better in 4/4 folds). Counting every D-1 arrival would be 0.011 better in CV but uses outcomes not yet known: **that is the size of the leak the clock removes.**

## Smoothing m and window, tuned jointly with C (notebook §6)

| smoothing_m | long_window | best_C | cv_pr_auc_mean | cv_pr_auc_std | cv_pr_auc_mean_C0.001 |
|---|---|---|---|---|---|
| 5 | 14 | 0.1000 | 0.4705 | 0.0706 | 0.4684 |
| 5 | 28 | 0.0010 | 0.4667 | 0.0643 | 0.4667 |
| 5 | 56 | 0.0100 | 0.4678 | 0.0686 | 0.4662 |
| 20 | 14 | 0.0100 | 0.4689 | 0.0709 | 0.4669 |
| 20 | 28 | 0.0010 | 0.4657 | 0.0640 | 0.4657 |
| 20 | 56 | 0.0100 | 0.4665 | 0.0686 | 0.4648 |
| 50 | 14 | 0.0100 | 0.4648 | 0.0719 | 0.4631 |
| 50 | 28 | 0.0010 | 0.4631 | 0.0641 | 0.4631 |
| 50 | 56 | 0.0100 | 0.4641 | 0.0686 | 0.4626 |

The CV-best pair is **m = 5, window = @L@, C = @C@**; re-tuning jointly with C (it was first scored at a fixed C = 0.001) selected the same pair, so nothing changed. The pipeline fails if `config` disagrees with this table.

## Month features removed (notebook §7)

| features | model | fold_m5 | fold_m6 | fold_m7 | fold_m8 | cv_pr_auc_mean |
|---|---|---|---|---|---|---|
| without month | logreg | 0.4354 | 0.5029 | 0.5515 | 0.3922 | 0.4705 |
| without month | xgboost | 0.4044 | 0.4284 | 0.5523 | 0.3950 | 0.4450 |
| with month | logreg | 0.4265 | 0.5050 | 0.5490 | 0.3938 | 0.4686 |
| with month | xgboost | 0.4112 | 0.4257 | 0.5633 | 0.3882 | 0.4471 |

Adding cyclic month features does not help the final model (LogReg 0.4705 → 0.4686, worse in 2/4 folds); XGBoost +0.0021 (0.4450 → 0.4471). Training covers months 1–8 only, so months 9–12 would be an extrapolation. **Decision: no month feature** (evidence: CV only).

## Day-level history as its own group (notebook §8)

`hist_rate_network_1d` takes one value per day (shared by hundreds of flights), so a flexible model could use it as a date identifier. It is kept separate so the ablation can test it. **CV evidence** ([04](04_evaluation.md), Part A.4): LogReg +0.011 in 3/4 folds, so it stays. For XGBoost, CV shows no gain (-0.004, 3/4 folds), so no claim is made for it there.

![feat_history_corr](../figures/feat_history_corr.png)
