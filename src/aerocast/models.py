"""Model factories and rolling-origin model selection (LogReg, XGBoost)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config


# ---------------------------------------------------------------------------
# Rolling-origin folds
# ---------------------------------------------------------------------------
def rolling_folds(months: np.ndarray, val_months=config.CV_VAL_MONTHS):
    """Yield (train_idx, val_idx): train on months < m, validate on month m."""
    for m in val_months:
        yield np.flatnonzero(months < m), np.flatnonzero(months == m)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
def make_logreg(cat_cols: list[str], num_cols: list[str], C: float = 1.0) -> Pipeline:
    """L2-regularised logistic regression: minimises
    mean log-loss + ||w||^2 / (2 C n). Smaller C = stronger shrinkage."""
    cat = ([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
             cat_cols)] if cat_cols else [])
    pre = ColumnTransformer(
        [
            *cat,
            (
                "num",
                Pipeline(
                    [
                        # missing-indicator lets the model treat "no history" as a
                        # signal rather than silently imputing it away
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )
    # No class_weight: re-weighting distorts predicted probabilities, and
    # calibrated risk is a deliverable. Imbalance is handled at the threshold.
    clf = LogisticRegression(C=C, max_iter=3000)
    return Pipeline([("pre", pre), ("clf", clf)])


LOGREG_CS = (0.0003, 0.001, 0.003, 0.01, 0.1, 1.0)


def cv_logreg(X, y, months, cat_cols, num_cols, Cs=LOGREG_CS):
    rows = []
    for C in Cs:
        for k, (tr, va) in enumerate(rolling_folds(months)):
            model = make_logreg(cat_cols, num_cols, C).fit(X.iloc[tr], y[tr])
            ap = average_precision_score(y[va], model.predict_proba(X.iloc[va])[:, 1])
            rows.append({"C": C, "fold": k, "val_pr_auc": ap})
    res = pd.DataFrame(rows)
    summary = res.groupby("C")["val_pr_auc"].agg(["mean", "std"]).reset_index()
    best_C = float(summary.loc[summary["mean"].idxmax(), "C"])
    return best_C, res


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
# Grid rationale: depth 2-5 (interaction order), min_child_weight 20/100
# (leaf-size regularisation for a weak, noisy signal), column subsampling and
# L2 leaf penalty; learning rate 0.03 vs 0.1 (shrinkage; the tree count is
# set by early stopping, so the two trade off). 48 combinations, 16 sampled.
XGB_GRID = {
    "max_depth": [2, 3, 5],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [20, 100],
    "subsample": [0.8],
    "colsample_bytree": [0.5, 0.8],
    "reg_lambda": [1.0, 10.0],
}


def make_xgb(n_estimators: int = 3000, early_stopping: bool = True, **params) -> XGBClassifier:
    base = dict(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,  # native split on carrier/origin/dest
        max_cat_to_onehot=1,
        eval_metric="logloss",
        early_stopping_rounds=100 if early_stopping else None,
        random_state=config.SEED,
        n_jobs=-1,
    )
    base.update(params)
    return XGBClassifier(**base)


def cv_xgb(X, y, months, grid=XGB_GRID, n_configs: int | None = 16):
    """Rolling-origin search. For each config and fold, early-stop on that
    fold's validation month; score = mean fold PR-AUC; n_trees = mean of the
    fold-wise best iterations (used to refit on the full training window).
    """
    keys = list(grid)
    configs = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    if n_configs and n_configs < len(configs):
        rng = np.random.default_rng(config.SEED)
        configs = [configs[i] for i in rng.choice(len(configs), n_configs, replace=False)]

    rows = []
    for ci, params in enumerate(configs):
        for k, (tr, va) in enumerate(rolling_folds(months)):
            model = make_xgb(**params).fit(
                X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False
            )
            ap = average_precision_score(y[va], model.predict_proba(X.iloc[va])[:, 1])
            rows.append({"config": ci, **params, "fold": k,
                         "best_iteration": model.best_iteration, "val_pr_auc": ap})
    res = pd.DataFrame(rows)
    agg = res.groupby("config").agg(
        mean_pr_auc=("val_pr_auc", "mean"), std_pr_auc=("val_pr_auc", "std"),
        n_trees=("best_iteration", lambda s: int(np.mean(s)) + 1),
    )
    best = agg["mean_pr_auc"].idxmax()
    return {**configs[best], "n_estimators": int(agg.loc[best, "n_trees"])}, res, agg


def fit_xgb_final(X, y, params: dict) -> XGBClassifier:
    """Refit on the whole training window with the CV-chosen tree count."""
    return make_xgb(early_stopping=False, **params).fit(X, y, verbose=False)


def cv_score_features_logreg(X, y, months, cols, cat_cols, C: float) -> pd.Series:
    """Fold-wise PR-AUC for a fixed LogReg C on a feature subset (ablation)."""
    cats = [c for c in cols if c in cat_cols]
    nums = [c for c in cols if c not in cat_cols]
    scores = []
    for tr, va in rolling_folds(months):
        model = make_logreg(cats, nums, C).fit(X.iloc[tr][cols], y[tr])
        scores.append(average_precision_score(y[va], model.predict_proba(X.iloc[va][cols])[:, 1]))
    return pd.Series(scores, index=config.CV_VAL_MONTHS)


def cv_score_features(X, y, months, cols, params: dict) -> pd.Series:
    """Fold-wise PR-AUC for a fixed XGB config on a feature subset (ablation)."""
    scores = []
    for tr, va in rolling_folds(months):
        model = fit_xgb_final(X.iloc[tr][cols], y[tr], params)
        scores.append(average_precision_score(y[va], model.predict_proba(X.iloc[va][cols])[:, 1]))
    return pd.Series(scores, index=config.CV_VAL_MONTHS)
