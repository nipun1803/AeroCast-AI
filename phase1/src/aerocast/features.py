"""Day-ahead feature engineering.

Three families, each tied to what is legitimately known at the decision time
(``config.DECISION_HOUR``:00 on day D-1):

1. schedule / calendar  -- from the published timetable of day D
2. congestion           -- counts over the *scheduled* timetable of day D
   (cancelled flights included: at decision time nobody knows they will be
   cancelled)
3. history              -- outcomes that were already KNOWN at the decision time

The history builder dates every outcome by when it became known (actual
arrival / departure on the NYC clock), maps that to the first target date it
may serve (``data.available_date``), and rolls a trailing window over those
availability dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config, data

CATEGORICAL = ["carrier", "origin", "dest"]


# ---------------------------------------------------------------------------
# 1. Schedule & calendar
# ---------------------------------------------------------------------------
def _cyclic(values: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    angle = 2 * np.pi * values / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(angle), f"{prefix}_cos": np.cos(angle)})


def schedule_features(df: pd.DataFrame, include_month: bool = False) -> pd.DataFrame:
    dep_min = df["hour"] * 60 + df["minute"]
    arr_min = (df["sched_arr_time"] // 100) * 60 + df["sched_arr_time"] % 100
    dow = df["date"].dt.dayofweek

    holidays = USFederalHolidayCalendar().holidays("2012-12-01", "2014-01-31")
    # distance (days) from each date to the nearest federal holiday
    uniq = pd.Series(df["date"].unique())
    dist = uniq.apply(lambda d: np.abs((holidays - d).days).min())
    days_to_holiday = df["date"].map(dict(zip(uniq, dist))).clip(upper=14)

    out = pd.concat(
        [
            _cyclic(dep_min, 1440, "dep_tod"),
            _cyclic(arr_min, 1440, "arr_tod"),
            _cyclic(dow, 7, "dow"),
            # No month feature by default: with a single year each month is one
            # draw of weather/ops conditions and Sep-Dec never occur in
            # training. Rolling CV confirms it does not help (notebook 02).
            *([_cyclic(df["month"] - 1, 12, "month")] if include_month else []),
        ],
        axis=1,
    ).set_index(df.index)
    out["dep_hour"] = dep_min / 60
    out["is_weekend"] = (dow >= 5).astype(int)
    out["days_to_holiday"] = days_to_holiday
    out["log_distance"] = np.log1p(df["distance"])
    return out


# ---------------------------------------------------------------------------
# 2. Scheduled congestion
# ---------------------------------------------------------------------------
def congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    size = lambda keys: df.groupby(keys)["id"].transform("size")  # noqa: E731

    out["sched_origin_hour"] = size(["origin", "date", "hour"])
    out["sched_origin_day"] = size(["origin", "date"])
    out["sched_dest_day"] = size(["dest", "date"])
    out["sched_carrier_origin_day"] = size(["carrier", "origin", "date"])

    # +/- 1 hour bank around the scheduled departure at the same airport
    per_hour = df.groupby(["origin", "date", "hour"]).size()
    bank = sum(
        per_hour.reindex(
            pd.MultiIndex.from_arrays([df["origin"], df["date"], df["hour"] + k])
        ).fillna(0).to_numpy()
        for k in (-1, 0, 1)
    )
    out["sched_origin_3h"] = bank

    # No aircraft-rotation features (leg index, legs per day). They would count
    # an aircraft's day-D departures by tail number, but the data blank the
    # tail number of many cancelled flights, so the count would depend on
    # day-D cancellations (see leakage.cancellation_check).
    return out


# ---------------------------------------------------------------------------
# 3. Past-only historical outcomes (decision clock)
# ---------------------------------------------------------------------------
# Outcome column -> column holding the time that outcome became known.
KNOWN_AT = {"is_delayed": "arr_known_dt", "_clip": "arr_known_dt",
            "is_cancelled": "dep_known_dt"}


def history_specs(long_window: int = config.LONG_WINDOW) -> list[tuple[tuple[str, ...], int]]:
    """(keys, window_days) for the delay-rate features. Empty keys = network.

    1- and 7-day windows capture recent shocks; ``long_window`` (tuned by
    rolling CV) captures stable carrier / route / aircraft propensities.
    """
    L = long_window
    return [
        ((), 1), ((), 7),
        (("origin",), 1), (("origin",), 7),
        (("carrier",), 7), (("carrier",), L),
        (("dest",), L), (("origin", "dest"), L), (("carrier", "origin", "dest"), L),
        (("origin", "hour"), L), (("carrier", "flight"), L), (("tailnum",), L),
    ]


CANCEL_SPECS: list[tuple[tuple[str, ...], int]] = [((), 1), (("origin",), 1)]
PRIOR_WINDOW = 28


def rolling_past(
    df: pd.DataFrame, keys: tuple[str, ...], window: int, col: str,
    decision_hour: int = config.DECISION_HOUR,
) -> pd.DataFrame:
    """Trailing ``window``-day (count, sum) of ``col`` per key, as known at the
    decision clock of each row's date.

    Each outcome is assigned the first target date whose decision time
    (D-1, ``config.DECISION_HOUR``:00) follows the moment it became known
    (``data.available_date``). A row dated D sums outcomes whose availability
    date lies in [D - window + 1, D]. Hence nothing from day D, and nothing
    from D-1 that landed after the decision time, can reach day D.

    Returns a frame aligned to ``df.index`` with columns ``n`` and ``s``.
    """
    keys = list(keys)
    known = df[KNOWN_AT[col]]
    obs = df[df[col].notna() & known.notna()]
    avail = data.available_date(known[obs.index], decision_hour).rename("avail")
    daily = obs.groupby(keys + [avail])[col].agg(["count", "sum"])

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    result = {}
    for stat in ("count", "sum"):
        wide = daily[stat].unstack(keys) if keys else daily[stat].to_frame()
        # union with all_dates so availability dates beyond the last flight
        # date do not break the rolling window; then keep the flight dates
        wide = wide.reindex(wide.index.union(all_dates)).fillna(0)
        rolled = wide.rolling(window, min_periods=1).sum().reindex(all_dates)
        result[stat] = rolled

    if not keys:
        lookup = df["date"]
        n = result["count"].iloc[:, 0].reindex(lookup).to_numpy()
        s = result["sum"].iloc[:, 0].reindex(lookup).to_numpy()
    else:
        idx = pd.MultiIndex.from_frame(df[["date"] + keys].rename(columns={"date": None}))
        n = result["count"].stack(list(range(len(keys)))).reindex(idx).to_numpy()
        s = result["sum"].stack(list(range(len(keys)))).reindex(idx).to_numpy()
    # Unseen keys (new route, missing tailnum) have no history: count 0, so the
    # smoothed rate falls back to the prior.
    return pd.DataFrame({"n": n, "s": s}, index=df.index).fillna(0)


def _name(keys: tuple[str, ...], window: int) -> str:
    return f"{'_'.join(keys) if keys else 'network'}_{window}d"


def network_prior(df: pd.DataFrame, decision_hour: int = config.DECISION_HOUR) -> np.ndarray:
    """Network-wide trailing 28-day delay rate (past-only): the shrinkage target.
    NaN when nothing is known yet (the first days of January)."""
    net = rolling_past(df, (), PRIOR_WINDOW, "is_delayed", decision_hour)
    with np.errstate(invalid="ignore"):
        return (net["s"] / net["n"].replace(0, np.nan)).to_numpy()


def smoothed_rate(s, n, prior, m: float = config.SMOOTHING_M):
    """Bayesian (beta-binomial) shrinkage: (late + m * prior) / (flights + m).

    With no history (n = 0) the rate equals the prior; with n >> m it tends to
    the raw rate s / n.
    """
    return (s + m * prior) / (n + m)


def history_trace(df: pd.DataFrame, keys: tuple[str, ...], key_values: tuple, window: int,
                  m: float = config.SMOOTHING_M) -> pd.DataFrame:
    """Day-by-day trace of one history feature for one key, for explanation.

    Shows the same-day outcome next to the values that ``rolling_past`` /
    ``smoothed_rate`` actually produce for that day (decision clock applied).
    """
    h = rolling_past(df, keys, window, "is_delayed")
    prior = network_prior(df)
    mask = np.logical_and.reduce([df[k].eq(v).to_numpy() for k, v in zip(keys, key_values)])
    sub = df.loc[mask].assign(
        hist_n=h.loc[mask, "n"], hist_late=h.loc[mask, "s"], prior=prior[mask],
        usable_from=data.available_date(df.loc[mask, "arr_known_dt"]),
    )
    sub["hist_rate"] = smoothed_rate(sub["hist_late"], sub["hist_n"], sub["prior"], m)
    return sub.groupby("date").agg(
        flights_today=("is_delayed", "count"), late_today=("is_delayed", "sum"),
        arrival_nyc=("arr_known_dt", "first"), outcome_usable_from=("usable_from", "first"),
        hist_flights=("hist_n", "first"), hist_late=("hist_late", "first"),
        network_prior=("prior", "first"), hist_rate=("hist_rate", "first"),
    )


def history_features(df: pd.DataFrame, m: float = config.SMOOTHING_M,
                     long_window: int = config.LONG_WINDOW,
                     decision_hour: int = config.DECISION_HOUR) -> pd.DataFrame:
    """Smoothed past delay rates + log counts, using only outcomes known by
    the decision clock.

    Rate = (delayed + m * prior) / (flights + m), where the prior is the
    network-wide trailing 28-day rate (itself past-only). Sparse keys (a new
    tail number, a rare route) therefore shrink toward the network level
    instead of producing noisy 0/1 rates.
    """
    out = pd.DataFrame(index=df.index)
    prior = network_prior(df, decision_hour)
    past = lambda *a: rolling_past(*a, decision_hour=decision_hour)  # noqa: E731

    for keys, window in history_specs(long_window):
        h = past(df, keys, window, "is_delayed")
        name = _name(keys, window)
        out[f"hist_rate_{name}"] = smoothed_rate(h["s"], h["n"], prior, m)
        if keys:  # network count is ~constant, not informative
            out[f"hist_logn_{name}"] = np.log1p(h["n"].fillna(0))

    for keys, window in CANCEL_SPECS:
        h = past(df, keys, window, "is_cancelled")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"hist_cancel_{_name(keys, window)}"] = h["s"] / h["n"].replace(0, np.nan)

    # Average minutes of arrival delay at the origin over the last 24 h before
    # the decision (clipped to limit the influence of a few multi-hour delays).
    tmp = df.assign(_clip=df["arr_delay"].clip(-60, 240))
    h = past(tmp, ("origin",), 1, "_clip")
    out["hist_meandelay_origin_1d"] = h["s"] / h["n"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, m: float = config.SMOOTHING_M,
                   long_window: int = config.LONG_WINDOW, include_month: bool = False,
                   decision_hour: int = config.DECISION_HOUR,
                   ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (feature frame, feature-group mapping) aligned to ``df.index``.

    ``df`` must be the *full* schedule (including cancelled flights) so that
    congestion counts and cancellation history are correct.
    ``include_month`` exists only for the CV check that justifies leaving the
    month out (pipeline.month_feature_cv); ``decision_hour`` other than the
    config value only for the decision-clock comparison (pipeline.tune_history).
    """
    sched = schedule_features(df, include_month)
    cong = congestion_features(df)
    hist = history_features(df, m, long_window, decision_hour)
    cats = df[CATEGORICAL].astype("category")

    X = pd.concat([cats, sched, cong, hist], axis=1)
    # Day-level history takes one value per (day[, origin]) -- shared by ~300-900
    # flights -- so its effective sample size is the number of training days,
    # and trees can use it to memorise individual days. Kept as its own group so
    # its contribution can be tested separately.
    day_level = [c for c in hist.columns if "network" in c or c.endswith("origin_1d")]
    groups = {
        "categorical": CATEGORICAL,
        "schedule": list(sched.columns),
        "congestion": list(cong.columns),
        "history_entity": [c for c in hist.columns if c not in day_level],
        "history_day": day_level,
    }
    return X, groups
