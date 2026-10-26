"""Leakage checks: features for date D must not depend on outcomes from D onward.

Method: overwrite every post-departure column for dates >= cutoff with random
values, rebuild all features, and compare. Features for dates <= cutoff must be
bit-for-bit identical (the cutoff day itself included: same-day outcomes are
not known at the D-1 decision time). Features after the cutoff must change,
otherwise the check would be vacuous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features


def scramble_future(df: pd.DataFrame, cutoff, seed: int = 0) -> pd.DataFrame:
    """Copy of ``df`` (with target) whose outcomes from ``cutoff`` on are random."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    future = df["date"] >= pd.Timestamp(cutoff)
    n = int(future.sum())
    df.loc[future, "arr_delay"] = rng.normal(100, 80, n)
    df.loc[future, "dep_delay"] = rng.normal(100, 80, n)
    cancel = rng.random(n) < 0.3
    df.loc[future, "dep_time"] = np.where(cancel, np.nan, 1200.0)
    return data.add_target(df.drop(columns=["is_delayed", "is_cancelled", "is_diverted"]))


def future_dependence_check(df: pd.DataFrame, cutoff, X: pd.DataFrame | None = None,
                            seed: int = 0) -> dict:
    """Rebuild features on scrambled outcomes and compare with the original.

    Returns counts of changed numeric cells on/before and after the cutoff.
    A leak-free pipeline has ``changed_up_to_cutoff == 0`` and
    ``changed_after_cutoff > 0``.
    """
    if X is None:
        X, _ = features.build_features(df)
    X2, _ = features.build_features(scramble_future(df, cutoff, seed))
    num = X.select_dtypes("number").columns
    a, b = X[num].to_numpy(), X2[num].to_numpy()
    changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    upto = (df["date"] <= pd.Timestamp(cutoff)).to_numpy()
    return {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "rows_up_to_cutoff": int(upto.sum()),
        "changed_up_to_cutoff": int(changed[upto].sum()),
        "rows_after_cutoff": int((~upto).sum()),
        "changed_after_cutoff": int(changed[~upto].sum()),
        "changed_columns_after": sorted(num[changed[~upto].any(axis=0)].tolist()),
    }


def _scramble_outcomes(df: pd.DataFrame, rows: pd.Series, seed: int) -> pd.DataFrame:
    """Randomise the arrival delay of ``rows`` only (timing columns untouched,
    so *when* each outcome becomes known does not move)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    rows = rows & df["arr_delay"].notna()
    df.loc[rows, "arr_delay"] = rng.normal(100, 80, int(rows.sum()))
    df["is_delayed"] = (df["arr_delay"] >= config.DELAY_THRESHOLD_MIN).astype(float)
    df.loc[df["arr_delay"].isna(), "is_delayed"] = float("nan")
    return df


def _cancel_rows(df: pd.DataFrame, rows: pd.Index) -> pd.DataFrame:
    """Cancel ``rows`` the way the raw data records a cancellation: no
    departure or arrival record and no tail number."""
    df = df.copy()
    df.loc[rows, ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time", "tailnum"]] = np.nan
    return data.add_target(df)


def cancellation_check(df: pd.DataFrame, target_date, X: pd.DataFrame | None = None,
                       share: float = 0.3, seed: int = 0) -> dict:
    """Does a day-D cancellation reach day D's features?

    ``scramble_future`` randomises outcomes but keeps every tail number, so it
    cannot see a leak through tail missingness. Here a random ``share`` of the
    day's operated flights is cancelled realistically (tail number blanked
    too), features are rebuilt, and the changed numeric columns are returned
    separately for

    * the OTHER flights of day D: must be empty, otherwise a same-day
      cancellation reaches the features of flights that are still predicted;
    * the cancelled flights themselves: they lose their tail number, so their
      tail-history features move. Only features that the cancellation-inclusive
      sensitivity target excludes may appear (this is also the positive control).
    """
    D = pd.Timestamp(target_date)
    if X is None:
        X, _ = features.build_features(df)
    rng = np.random.default_rng(seed)
    on_d = (df["date"] == D).to_numpy()
    pick = on_d & df["dep_time"].notna().to_numpy() & (rng.random(len(df)) < share)
    X2, _ = features.build_features(_cancel_rows(df, df.index[pick]))
    num = X.select_dtypes("number").columns

    def changed_columns(rows):
        a, b = X.loc[rows, num].to_numpy(), X2.loc[rows, num].to_numpy()
        changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        return sorted(num[changed.any(axis=0)].tolist())

    others = on_d & ~pick
    return {
        "target_date": str(D.date()),
        "flights_cancelled": int(pick.sum()),
        "other_flights_on_D": int(others.sum()),
        "changed_columns_other_flights": changed_columns(others),
        "changed_columns_cancelled_flights": changed_columns(pick),
    }


def decision_clock_check(df: pd.DataFrame, target_date, X: pd.DataFrame | None = None,
                         seed: int = 0) -> dict:
    """Do late-evening D-1 arrivals leak into day D's features?

    * scramble arrival outcomes of D-1 flights that arrived AFTER the decision
      time (D-1, DECISION_HOUR:00) -> every feature of day D must be unchanged;
    * positive control: scramble D-1 flights that arrived BEFORE it -> day D's
      history features must change (otherwise the check is vacuous).
    """
    D = pd.Timestamp(target_date)
    clock = D - pd.Timedelta(days=1) + pd.Timedelta(hours=config.DECISION_HOUR)
    if X is None:
        X, _ = features.build_features(df)
    prev = df["date"] == D - pd.Timedelta(days=1)
    late = prev & (df["arr_known_dt"] > clock)
    early = prev & (df["arr_known_dt"] <= clock)
    on_d = (df["date"] == D).to_numpy()
    num = X.select_dtypes("number").columns

    def changed(rows):
        X2, _ = features.build_features(_scramble_outcomes(df, rows, seed))
        a, b = X.loc[on_d, num].to_numpy(), X2.loc[on_d, num].to_numpy()
        return int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())

    return {
        "target_date": str(D.date()),
        "decision_time": str(clock),
        "d_minus_1_flights_after_clock": int(late.sum()),
        "d_minus_1_flights_before_clock": int(early.sum()),
        "changed_cells_on_D_after_scrambling_late": changed(late),
        "changed_cells_on_D_after_scrambling_early": changed(early),
    }
