"""Metrics, calibration, operating threshold and uncertainty estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from . import config


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y, p, n_bins: int = config.N_CALIBRATION_BINS) -> float:
    """Equal-width ECE (Guo et al., 2017): sum_b |B_b|/N * |acc_b - conf_b|."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def expected_calibration_error_equal_mass(y, p, n_bins: int = config.N_CALIBRATION_BINS) -> float:
    """Equal-mass ECE: bins are score quantiles, so every bin holds ~N/n_bins
    flights. Equal-width bins leave the sparse high-risk tail with few
    samples (noisy) and lump most mass into a few bins; equal-mass binning
    trades that for bins whose edges move with the score distribution."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def reliability_table(y, p, n_bins: int = config.N_CALIBRATION_BINS) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    return df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))


# ---------------------------------------------------------------------------
# Post-hoc calibrators (fit on the calib month only)
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Logistic regression on the logit of the raw score."""

    def fit(self, p, y):
        self._lr = LogisticRegression(C=1e6).fit(_logit(p)[:, None], y)
        return self

    def predict(self, p):
        return self._lr.predict_proba(_logit(p)[:, None])[:, 1]


class IsotonicCalibrator:
    def fit(self, p, y):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p):
        return self._iso.predict(p)


class IdentityCalibrator:
    """'No recalibration' as an explicit candidate."""

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p)


CALIBRATORS = {"raw": IdentityCalibrator, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def select_calibrator(p, y, months, fit_month: int, check_month: int,
                      min_gain: float = config.CALIB_MIN_ECE_GAIN) -> tuple[str, pd.DataFrame]:
    """A-priori recalibration rule (no test data involved).

    Fit Platt and isotonic on ``fit_month``, score ECE on the later
    ``check_month`` (a mini forward test). Keep the raw scores unless the best
    recalibrator lowers ECE by more than ``min_gain``: a recalibrator must
    earn its place, a near-tie defaults to "do nothing".
    """
    p, y, months = map(np.asarray, (p, y, months))
    fit, chk = months == fit_month, months == check_month
    rows = []
    for name, cls in CALIBRATORS.items():
        q = cls().fit(p[fit], y[fit]).predict(p[chk])
        rows.append({"calibrator": name, "check_ece": expected_calibration_error(y[chk], q),
                     "check_ece_equal_mass": expected_calibration_error_equal_mass(y[chk], q),
                     "check_brier": float(brier_score_loss(y[chk], q))})
    table = pd.DataFrame(rows).set_index("calibrator")
    raw = table.loc["raw", "check_ece"]
    best = table.drop(index="raw")["check_ece"].idxmin()
    table["gain_vs_raw"] = raw - table["check_ece"]
    choice = best if table.loc[best, "gain_vs_raw"] > min_gain else "raw"
    return choice, table.reset_index()


def rolling_platt(scores, y, row_dates, avail_dates, target_dates,
                  window: int = config.ROLLING_CALIB_DAYS) -> np.ndarray:
    """Causal rolling recalibration.

    For every target day D, refit Platt on the flights whose outcome was known
    by D's decision clock and became available within the last ``window``
    days (availability date in [D - window + 1, D]), then apply it to the
    flights dated D. Uses the same availability dates as the history
    features, so it never sees an outcome before the decision time.

    ``scores``, ``y``, ``row_dates``, ``avail_dates`` cover the pool of all
    flights (any split); returns calibrated scores for rows whose date is in
    ``target_dates`` (NaN elsewhere).
    """
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    row_dates = pd.to_datetime(pd.Series(row_dates)).to_numpy()
    avail = pd.to_datetime(pd.Series(avail_dates)).to_numpy()
    out = np.full(len(scores), np.nan)
    win = np.timedelta64(window, "D")
    for D in pd.to_datetime(pd.Series(target_dates)).unique():
        D = np.datetime64(D)
        fit = (avail <= D) & (avail > D - win)
        rows = row_dates == D
        if rows.any():
            out[rows] = PlattCalibrator().fit(scores[fit], y[fit]).predict(scores[rows])
    return out


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
def threshold_for_precision(y, p, target: float = config.TARGET_PRECISION) -> float:
    """Lowest threshold whose precision on (y, p) reaches ``target``.

    The lowest such threshold maximises recall subject to the precision
    constraint. Chosen on the calib month, then frozen for the test period.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) == 0:
        return float("inf")  # target unreachable: flag nothing
    return float(thr[ok[np.argmax(rec[ok])]])


def oracle_recall_at_precision(y, p, target: float = config.TARGET_PRECISION) -> float:
    """Best achievable recall at precision >= target, with the threshold tuned
    on the evaluated labels themselves ("oracle").

    A point on the precision-recall curve: valid for COMPARING MODELS on the
    same data, NOT an operational result (no deployable threshold produces it).
    """
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def flag_daily_top_share(dates, p, share: float = config.DAILY_FLAG_SHARE,
                         seed: int = config.SEED) -> np.ndarray:
    """Capacity policy: flag the ``share`` highest-risk flights of each day.

    Exact ties are broken at random (seeded), not by row order: rows are in
    departure order, so an order-based tie-break would let an uninformative
    (constant) score pick the day's latest flights and look skilful.
    """
    p, dates = np.asarray(p), np.asarray(dates)
    order = np.random.default_rng(seed).permutation(len(p))
    pct = pd.Series(p[order]).groupby(pd.Series(dates[order])).rank(pct=True, method="first")
    flagged = np.empty(len(p), dtype=bool)
    flagged[order] = (pct > 1 - share).to_numpy()
    return flagged


def policy_report(y, flagged) -> dict:
    y, flagged = np.asarray(y), np.asarray(flagged, dtype=bool)
    tp = int((flagged & (y == 1)).sum())
    return {
        "flag_rate": float(flagged.mean()),
        "precision": float(tp / flagged.sum()) if flagged.any() else float("nan"),
        "recall": float(tp / (y == 1).sum()),
        "lift": float((tp / flagged.sum()) / y.mean()) if flagged.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Metric bundle
# ---------------------------------------------------------------------------
def metric_bundle(y, p, threshold: float | None = None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
        "ece_equal_mass": expected_calibration_error_equal_mass(y, p),
        "mean_pred": float(p.mean()),
        **{f"oracle_recall@p{t:.2f}": oracle_recall_at_precision(y, p, t)
           for t in config.PRECISION_SWEEP},
    }
    if threshold is not None:
        flag = p >= threshold
        tp = int((flag & (y == 1)).sum())
        out.update(
            threshold=float(threshold),
            flag_rate=float(flag.mean()),
            precision_at_thr=float(tp / flag.sum()) if flag.any() else float("nan"),
            recall_at_thr=float(tp / (y == 1).sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Day-block bootstrap
# ---------------------------------------------------------------------------
def day_block_bootstrap(
    dates: pd.Series,
    y,
    preds: dict[str, np.ndarray],
    metric=average_precision_score,
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Resample whole days (flights on one day share weather / ATC shocks, so
    they are not independent). Returns one row per replicate, one column per
    model.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    codes, uniq = pd.factorize(pd.Series(dates).to_numpy())
    rows_by_day = [np.flatnonzero(codes == d) for d in range(len(uniq))]
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_day[d] for d in pick])
        reps.append({k: metric(y[idx], np.asarray(v)[idx]) for k, v in preds.items()})
    return pd.DataFrame(reps)


def bootstrap_diff_ci(dates, y, p_a, p_b, metric=average_precision_score,
                      n_boot: int = config.N_BOOTSTRAP, seed: int = config.SEED) -> dict:
    """Paired day-block bootstrap of metric(a) - metric(b)."""
    boot = day_block_bootstrap(dates, y, {"a": p_a, "b": p_b}, metric, n_boot, seed)
    diff = boot["a"] - boot["b"]
    return {"diff": float(metric(np.asarray(y), np.asarray(p_a)) - metric(np.asarray(y), np.asarray(p_b))),
            "ci_lo": float(diff.quantile(0.025)), "ci_hi": float(diff.quantile(0.975)),
            "p_not_positive": float((diff <= 0).mean())}


def summarize_bootstrap(boot: pd.DataFrame, a: str, b: str) -> dict:
    diff = boot[a] - boot[b]
    ci = lambda s: [float(s.quantile(0.025)), float(s.quantile(0.975))]  # noqa: E731
    return {
        f"{a}_ci95": ci(boot[a]),
        f"{b}_ci95": ci(boot[b]),
        f"diff_{a}_minus_{b}_ci95": ci(diff),
        f"p_{a}_not_better": float((diff <= 0).mean()),
    }
