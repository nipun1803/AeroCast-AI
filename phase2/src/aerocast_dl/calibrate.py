"""Calibration (Step 7): the Phase 1 Sep -> Oct rule with more candidates.

Candidates (docs/development_protocol.md §4):

* static, fitted on September labels available by 1 Oct: raw, Platt,
  isotonic, temperature scaling;
* causal rolling, refitted for every day D on the outcomes that became
  available in the 28 days up to D's decision clock: rolling Platt (Phase 1
  ``evaluate.rolling_platt``) and rolling temperature.

All are scored by ECE on October (labels available by 1 Nov). The lowest
October ECE wins, and it is adopted only if it beats raw by more than
``CALIB_MIN_ECE_GAIN``. No test label is read here.

Temperature scaling (Guo et al., 2017) divides the logit by one fitted
T > 0. It has no intercept, so unlike Platt it cannot move the mean
prediction to a new base rate; a seasonal base-rate shift is exactly what
the Sep -> Dec period has, which is why Platt-type candidates are kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from aerocast import config as p1
from aerocast import data, evaluate

ROLL = p1.ROLLING_CALIB_DAYS
STATIC = ["raw", "platt", "isotonic", "temperature"]
ROLLING = [f"rolling_platt{ROLL}", f"rolling_temperature{ROLL}"]
CANDIDATES = STATIC + ROLLING


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class TemperatureCalibrator:
    """p -> sigmoid(logit(p) / T), T fitted by maximum likelihood."""

    def fit(self, p, y):
        z, y = _logit(p), np.asarray(y, dtype=float)

        def nll(log_t):
            q = 1 / (1 + np.exp(-z / np.exp(log_t)))
            q = np.clip(q, 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.T_ = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-_logit(p) / self.T_))


STATIC_CLS = {**evaluate.CALIBRATORS, "temperature": TemperatureCalibrator}


def rolling(kind: str, scores, y, row_dates, avail_dates, target_dates,
            window: int = ROLL) -> np.ndarray:
    """Causal rolling recalibration (as Phase 1 ``evaluate.rolling_platt``):
    for each target day D, fit on rows with availability date in
    (D - window, D], apply to the rows dated D. NaN elsewhere."""
    if kind == "platt":
        return evaluate.rolling_platt(scores, y, row_dates, avail_dates, target_dates, window)
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
            out[rows] = TemperatureCalibrator().fit(scores[fit], y[fit]).predict(scores[rows])
    return out


def _rolling_kind(name: str) -> str:
    return "platt" if name.startswith("rolling_platt") else "temperature"


def select(pool: dict, min_gain: float = p1.CALIB_MIN_ECE_GAIN) -> tuple[str, pd.DataFrame]:
    """Sep -> Oct rule on one model's scores.

    ``pool`` has arrays ``p, y, date, avail, split, month`` for the train and
    calibration rows (no test rows). Static candidates fit on September
    labels known by the first October day; every candidate is scored on the
    October labels known by 1 Nov.
    """
    split, month = np.asarray(pool["split"]), np.asarray(pool["month"])
    if (split == "test").any():
        raise ValueError("calibrator selection must not see test rows")
    dates = pd.to_datetime(pd.Series(pool["date"])).to_numpy()
    cal = split == "calib"
    chk = cal & (month == 10)
    fit = cal & (month == 9) & data.outcomes_available_by(pool["avail"], dates[chk].min())
    chk &= data.outcomes_available_by(pool["avail"], p1.SPLITS["test"][0])
    p, y = np.asarray(pool["p"], dtype=float), np.asarray(pool["y"])
    rows = []
    for name in CANDIDATES:
        if name in STATIC:
            q = STATIC_CLS[name]().fit(p[fit], y[fit]).predict(p[chk])
        else:
            q = rolling(_rolling_kind(name), p, y, pool["date"], pool["avail"],
                        np.unique(dates[chk]))[chk]
        rows.append({"calibrator": name,
                     "check_ece": evaluate.expected_calibration_error(y[chk], q),
                     "check_ece_equal_mass": evaluate.expected_calibration_error_equal_mass(y[chk], q),
                     "check_brier": float(np.mean((q - y[chk]) ** 2))})
    table = pd.DataFrame(rows).set_index("calibrator")
    table["gain_vs_raw"] = table.loc["raw", "check_ece"] - table["check_ece"]
    best = table.drop(index="raw")["check_ece"].idxmin()
    choice = best if table.loc[best, "gain_vs_raw"] > min_gain else "raw"
    return choice, table.reset_index()


def apply(name: str, pool: dict) -> np.ndarray:
    """Deploy calibrator ``name`` on the calibration and test rows of ``pool``
    (all splits). Static calibrators are refitted on every September-October
    label available by 1 Nov; rolling ones use the outcomes available by each
    day's decision clock. Train rows are returned as NaN."""
    split = np.asarray(pool["split"])
    p, y = np.asarray(pool["p"], dtype=float), np.asarray(pool["y"])
    out = np.full(len(p), np.nan)
    target = split != "train"
    if name in STATIC:
        fit = (split == "calib") & data.outcomes_available_by(pool["avail"], p1.SPLITS["test"][0])
        out[target] = STATIC_CLS[name]().fit(p[fit], y[fit]).predict(p[target])
    else:
        dates = pd.to_datetime(pd.Series(pool["date"])).to_numpy()
        q = rolling(_rolling_kind(name), p, y, pool["date"], pool["avail"], np.unique(dates[target]))
        out[target] = q[target]
    return out


def day_maps(name: str, pool: dict, target_dates) -> dict:
    """Fitted calibrator per target date (rolling) or one shared calibrator
    (static), so that NEW scores of the same day can be calibrated with
    exactly the map used for that day's real predictions (rescheduling)."""
    split = np.asarray(pool["split"])
    p, y = np.asarray(pool["p"], dtype=float), np.asarray(pool["y"])
    dates = [pd.Timestamp(d) for d in pd.to_datetime(pd.Series(target_dates)).unique()]
    if name in STATIC:
        fit = (split == "calib") & data.outcomes_available_by(pool["avail"], p1.SPLITS["test"][0])
        cal = STATIC_CLS[name]().fit(p[fit], y[fit])
        return {d: cal for d in dates}
    avail = pd.to_datetime(pd.Series(pool["avail"])).to_numpy()
    cls = evaluate.PlattCalibrator if _rolling_kind(name) == "platt" else TemperatureCalibrator
    out, win = {}, np.timedelta64(ROLL, "D")
    for d in dates:
        D = np.datetime64(d)
        fit = (avail <= D) & (avail > D - win)
        out[d] = cls().fit(p[fit], y[fit])
    return out
