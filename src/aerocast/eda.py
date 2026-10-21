"""Data-quality audit and exploratory aggregations (notebooks 01-02)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().to_frame("n_missing")
    miss["pct"] = 100 * miss["n_missing"] / len(df)
    return miss[miss["n_missing"] > 0].sort_values("n_missing", ascending=False)


def quality_audit(df: pd.DataFrame) -> pd.Series:
    """Row-level integrity counts. ``df`` must come from ``data.add_target``."""
    return pd.Series({
        "rows": len(df),
        "duplicate id": df["id"].duplicated().sum(),
        "duplicate (date, carrier, flight, origin, sched_dep)":
            df.duplicated(["date", "carrier", "flight", "origin", "sched_dep_time"]).sum(),
        "cancelled (no dep_time)": df["is_cancelled"].sum(),
        "diverted (departed, no arr_delay)": df["is_diverted"].sum(),
        "missing tailnum": df["tailnum"].isna().sum(),
        "missing tailnum AND cancelled":
            (df["tailnum"].isna() & df["is_cancelled"].eq(1)).sum(),
    }, name="value")


def _hhmm_to_min(hhmm):
    return (hhmm // 100) * 60 + hhmm % 100


def consistency_checks(df: pd.DataFrame) -> pd.Series:
    """Internal-consistency checks between redundant columns."""
    flown = df[df["dep_time"].notna()]
    clock = (_hhmm_to_min(flown["dep_time"]) - _hhmm_to_min(flown["sched_dep_time"])
             - flown["dep_delay"]) % 1440
    return pd.Series({
        "hour/minute match sched_dep_time (share)":
            (df["hour"].eq(df["sched_dep_time"] // 100)
             & df["minute"].eq(df["sched_dep_time"] % 100)).mean(),
        "dep_delay = dep_time - sched_dep_time, mod 24h (share)": (clock == 0).mean(),
        "routes with >1 distinct distance":
            (df.groupby(["origin", "dest"])["distance"].nunique() > 1).sum(),
        "air_time <= 0": (df["air_time"] <= 0).sum(),
    }, name="value")


def leakage_audit() -> pd.DataFrame:
    """Which raw columns are known at the decision time (end of D-1)."""
    return pd.DataFrame([
        ("year, month, day, hour, minute, sched_dep_time, sched_arr_time", "schedule",
         "yes", "calendar & time-of-day features"),
        ("carrier, flight, origin, dest, distance", "schedule", "yes",
         "identity / route features"),
        ("tailnum", "aircraft assignment", "yes (assumed)",
         "rotation features; assumes routing is published day-ahead"),
        ("dep_time, dep_delay", "operations", "NO - after departure",
         "never a feature; past outcomes known by 20:00 on D-1 only"),
        ("arr_time, arr_delay, air_time", "operations", "NO - after arrival",
         "target; past outcomes known by 20:00 on D-1 only (arrival on NYC clock)"),
        ("id, name, time_hour", "bookkeeping", "-", "dropped (row id / redundant)"),
    ], columns=["columns", "source", "known at D-1?", "use"])


def delay_shares(df: pd.DataFrame, thresholds=(15, 30, 60)) -> pd.Series:
    done = df["arr_delay"].notna()
    return pd.Series({f">= {t} min": (df.loc[done, "arr_delay"] >= t).mean()
                      for t in thresholds}, name="share of completed flights")


def rate_by(df: pd.DataFrame, key) -> pd.DataFrame:
    """Delay rate and flight count per value of ``key`` (column or Series)."""
    return df.groupby(key).agg(rate=("is_delayed", "mean"), n=("is_delayed", "count"))


def daily_rate(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["is_delayed"].mean()


def autocorrelation(series: pd.Series, lags=range(1, 15)) -> pd.Series:
    return pd.Series({k: series.autocorr(k) for k in lags}, name="autocorr")


def congestion_curve(df: pd.DataFrame, n_bins: int = 8) -> pd.Series:
    """Delay rate by octile of scheduled departures in the same origin-hour.

    The bank size is counted on the full schedule (cancelled flights included),
    the rate on completed flights.
    """
    bank = df.groupby(["origin", "date", "hour"])["id"].transform("size")
    done = df["is_delayed"].notna()
    bins = pd.qcut(bank[done], n_bins, duplicates="drop")
    return df.loc[done, "is_delayed"].groupby(bins, observed=True).mean()


def distinct_values(X: pd.DataFrame, cols: list[str], dates: pd.Series) -> pd.DataFrame:
    """How many distinct values a feature takes vs the number of days/rows.

    A feature with ~one value per day has an effective sample size of the
    number of days, not the number of flights.
    """
    return pd.DataFrame({
        "distinct_values": [X[c].nunique() for c in cols],
        "values_per_day": [X.groupby(dates)[c].nunique().mean() for c in cols],
    }, index=cols).assign(rows=len(X), days=dates.nunique())


def design_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        ("Missingness = cancellations / diversions",
         "drop from target, keep in schedule (congestion) counts"),
        ("Strong time-of-day gradient (10% -> 37%)", "cyclic dep/arr time + raw hour"),
        ("One year; seasonal months; Sep-Oct calm, Dec extreme",
         "no month feature; chronological split; calibration studied under shift"),
        ("Short-memory daily shocks (lag-1 autocorr ~0.4)",
         "1-day / 7-day lagged rates (day-level group)"),
        ("Stable carrier / route / aircraft propensities; some tiny groups",
         "28-day entity rates with Bayesian shrinkage; one-hot min_frequency=50"),
        ("Congestion correlates with delay but is confounded with hour",
         "scheduled-bank counts, kept and tested by ablation"),
        ("Leakage risk from post-departure columns; 31% of D-1 arrivals land after 20:00",
         "decision clock 20:00 on D-1; outcomes enter history only once known; two leakage tests"),
        ("History rates strongly inter-correlated (same keys, nested windows)",
         "L2-regularised LogReg; grouped (not per-feature) permutation importance"),
    ], columns=["EDA observation", "design decision"])


def distance_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Routes recorded with more than one distance, with flight counts."""
    multi = df.groupby(["origin", "dest"])["distance"].nunique()
    multi = multi[multi > 1].index
    sub = df.set_index(["origin", "dest"]).loc[multi].reset_index()
    return sub.groupby(["origin", "dest", "distance"]).agg(
        flights=("id", "size"), first=("date", "min"), last=("date", "max"))


def congestion_within_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Delay rate by departure hour x bank-size tercile, where terciles are
    computed WITHIN each hour (hour held fixed, removing the confounding of
    busy banks with the evening delay peak). Completed flights, 6-21 h."""
    d = df.assign(bank=df.groupby(["origin", "date", "hour"])["id"].transform("size"))
    d = d[d["is_delayed"].notna() & d["hour"].between(6, 21)]
    pct = d.groupby("hour")["bank"].rank(pct=True, method="average")
    d = d.assign(tercile=pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"]))
    return d.pivot_table(index="hour", columns="tercile", values="is_delayed",
                         aggfunc="mean", observed=True)


def within_hour_gap(tbl: pd.DataFrame) -> pd.Series:
    """High-minus-low tercile delay rate per hour, and its average."""
    gap = tbl["high"] - tbl["low"]
    return pd.Series({"mean high-low gap": gap.mean(), "hours with high > low": int((gap > 0).sum()),
                      "hours": len(gap)})


def arrival_timing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For completed flights: when does the outcome become usable (days after
    the flight date, under the decision clock), and how late are those
    flights? Lag 1 = known by DECISION_HOUR on the flight date; lag 2 = landed
    after it, so it first counts the day after tomorrow."""
    from . import data
    done = df[df["is_delayed"].notna()]
    lag = (data.available_date(done["arr_known_dt"]) - done["date"]).dt.days
    return done.groupby(lag.rename("usable_after_days")).agg(
        flights=("id", "size"), delay_rate=("is_delayed", "mean")).assign(
        share=lambda t: t["flights"] / t["flights"].sum())
