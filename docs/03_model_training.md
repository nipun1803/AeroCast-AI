# 03 · Model training

> Summarises [03_model_training.ipynb](../notebooks/03_model_training.ipynb) (§1–§9). All numbers come from `results/` or the executed notebook outputs.

## Split (notebook §1)

| split | rows | delay_rate | from | to |
|---|---|---|---|---|
| train | 211684 | 0.265 | 2013-01-08 | 2013-08-31 |
| calib | 55628 | 0.157 | 2013-09-01 | 2013-10-31 |
| test | 53991 | 0.258 | 2013-11-01 | 2013-12-31 |

Jan 1–7 is excluded (warm-up). Base rate shifts: 26.5% train, 15.7% calibration, 25.8% test (Nov 18.1%, Dec 33.5%).

## Baselines (notebook §3)

*Prior* (training base rate) and *Route history* (smoothed 14-day carrier-route rate) sit beside every learned model in every comparison table.

## Rolling-origin CV (notebook §4)

Folds: train Jan–Apr → validate May; Jan–May → Jun; Jan–Jun → Jul; Jan–Jul → Aug. A single validation month (Sep) was abandoned because it is the calmest month and its best tree count differed from October's on nearly flat curves.

![train_single_month_tree_curve](../figures/train_single_month_tree_curve.png)

## LogReg: C grid, CV mean ± sd (notebook §5)

| C | cv_mean | cv_std |
|---|---|---|
| 0.0003 | 0.4660 | 0.0683 |
| 0.0010 | 0.4684 | 0.0695 |
| 0.0030 | 0.4698 | 0.0699 |
| 0.0100 | 0.4705 | 0.0702 |
| 0.1000 | 0.4705 | 0.0706 |
| 1.0000 | 0.4703 | 0.0708 |

Chosen **C = @C@** (the curve is flat above C ≈ 0.003; the fold spread is far larger than any difference between C values).

![train_cv_logreg](../figures/train_cv_logreg.png)

## XGBoost: grid and top configurations (notebook §6)

Grid (48 combinations, 16 sampled): depth {2, 3, 5}, min_child_weight {20, 100}, colsample {0.5, 0.8}, λ {1, 10}, learning rate {0.03, 0.1}, subsample 0.8; up to 3,000 trees with early stopping (100) on each validation fold; tree count = mean of the four best iterations.

| config | mean_pr_auc | std_pr_auc | n_trees |
|---|---|---|---|
| 0 | 0.4754 | 0.0673 | 143 |
| 15 | 0.4725 | 0.0741 | 146 |
| 1 | 0.4711 | 0.0694 | 340 |
| 4 | 0.4708 | 0.0764 | 76 |
| 5 | 0.4704 | 0.0688 | 231 |
| 10 | 0.4692 | 0.0684 | 314 |
| 11 | 0.4692 | 0.0685 | 374 |
| 14 | 0.4683 | 0.0683 | 64 |

Chosen: depth 2, learning rate 0.1, min_child_weight 100, colsample 0.8, λ 10.0, **143 trees** (CV mean ± sd 0.4754 ± 0.0673 with in-fold early stopping). The top configurations differ by less than one fold sd.

## Final-model rule (notebook §7)

**Rule (fixed before test):** the model with the higher mean rolling-CV PR-AUC, **each scored as deployed**. XGBoost's search score early-stops on the validation fold itself (optimistic), so it is re-scored with its fixed, fold-averaged tree count.

| model | calibration | cv_pr_auc_mean | cv_pr_auc_std | fold_m5 | fold_m6 | fold_m7 | fold_m8 |
|---|---|---|---|---|---|---|---|
| Prior (base rate) | none | 0.2775 | 0.0483 | 0.2267 | 0.3152 | 0.3221 | 0.2460 |
| Route history | none | 0.3476 | 0.0601 | 0.2766 | 0.3911 | 0.4034 | 0.3194 |
| LogReg (as deployed) | raw | 0.4705 | 0.0706 | 0.4354 | 0.5029 | 0.5515 | 0.3922 |
| XGBoost (as deployed, fixed trees) | raw | 0.4450 | 0.0729 | 0.4044 | 0.4284 | 0.5523 | 0.3950 |
| XGBoost (in-fold early stopping, optimistic; not used) | raw | 0.4754 | 0.0673 | 0.4490 | 0.4956 | 0.5574 | 0.3995 |

Paired per-fold difference LogReg − XGBoost: +0.0310, +0.0746, -0.0007, -0.0027 (mean +0.0255, sd 0.0362; LogReg better in 2/4 folds).

**LogReg and XGBoost are not statistically distinguished**: the CV gap (0.0255) is much smaller than the fold sd (about 0.07), and the test bootstrap CI includes 0 ([04](04_evaluation.md)). **The choice of LogReg rests on the rule declared before test** (`config.FINAL_MODEL`, enforced by the pipeline), not on demonstrated superiority. In-fold early stopping would have inflated XGBoost to 0.4754 and reversed the choice; this is disclosed in the [development protocol](development_protocol.md).

## Theory notes

* **Logistic regression** minimises the L2-regularised log-loss `−(1/n)Σ[y log p + (1−y) log(1−p)] + ‖w‖²/(2Cn)`; convex; smaller C shrinks harder (needed for correlated history features).
* **XGBoost** fits trees stage-wise on a second-order Taylor expansion of the loss (gradients g = p − y, Hessians h = p(1−p)) plus a complexity penalty; optimal leaf weight −G/(H+λ).
* **Log-loss is a strictly proper scoring rule**: minimised only by the true probability, so training pushes toward calibration on the training distribution (the Brier score is also proper; accuracy and AUC are not).
