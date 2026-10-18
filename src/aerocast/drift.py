"""Covariate-shift diagnostics: PSI per feature and adversarial validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config, models

_EPS = 1e-4


def _shares(codes: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_bins).astype(float)
    return np.clip(counts / counts.sum(), _EPS, None)


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    PSI = sum_b (c_b - r_b) * ln(c_b / r_b) over bins defined on the reference
    (deciles for numeric features, categories for categorical ones; missing
    values form their own bin). Rule of thumb: < 0.1 stable, 0.1-0.25
    moderate shift, > 0.25 large shift.
    """
    if isinstance(reference.dtype, pd.CategoricalDtype):
        cats = list(reference.cat.categories)
        to_code = lambda s: s.cat.codes.to_numpy() + 1  # -1 (NaN) -> 0  # noqa: E731
        n = len(cats) + 1
    else:
        edges = np.unique(np.nanquantile(reference.to_numpy(dtype=float),
                                         np.linspace(0, 1, n_bins + 1)[1:-1]))
        n = len(edges) + 2

        def to_code(s):
            v = s.to_numpy(dtype=float)
            return np.where(np.isnan(v), 0, np.searchsorted(edges, v, side="right") + 1)
    r, c = _shares(to_code(reference), n), _shares(to_code(current), n)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(Xs: dict[str, pd.DataFrame], groups: dict[str, list[str]]) -> pd.DataFrame:
    """PSI of every feature, train as reference, for calib and test."""
    group_of = {c: g for g, cols in groups.items() for c in cols}
    rows = [{"feature": c, "group": group_of[c],
             "psi_calib": psi(Xs["train"][c], Xs["calib"][c]),
             "psi_test": psi(Xs["train"][c], Xs["test"][c])}
            for c in Xs["train"].columns]
    return pd.DataFrame(rows).sort_values("psi_test", ascending=False).reset_index(drop=True)


def adversarial_validation(X_ref: pd.DataFrame, X_cur: pd.DataFrame, n_splits: int = 5,
                           max_rows: int = 60_000) -> tuple[float, pd.Series]:
    """Out-of-fold ROC-AUC of a classifier separating ``X_ref`` (label 0) from
    ``X_cur`` (label 1), plus its gain importances.

    AUC ~ 0.5: the periods are indistinguishable from the features alone;
    AUC -> 1: strong covariate shift, and the top features say where.
    Both sides are subsampled to ``max_rows`` for speed.
    """
    rng = np.random.default_rng(config.SEED)
    take = lambda X: X.iloc[np.sort(rng.choice(len(X), min(max_rows, len(X)), replace=False))]  # noqa: E731
    a, b = take(X_ref), take(X_cur)
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    oof = np.zeros(len(y))
    gains = []
    params = dict(max_depth=4, learning_rate=0.1, min_child_weight=20, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, n_estimators=200)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=config.SEED).split(X, y):
        mdl = models.fit_xgb_final(X.iloc[tr], y[tr], params)
        oof[va] = mdl.predict_proba(X.iloc[va])[:, 1]
        gains.append(pd.Series(mdl.get_booster().get_score(importance_type="gain")))
    imp = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    return float(roc_auc_score(y, oof)), imp.rename("gain")
