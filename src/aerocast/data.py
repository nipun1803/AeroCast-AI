"""Loading, cleaning, target construction, outcome timing and the split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns only observable once the flight has departed / arrived. They may be
# used to build the *target* and *historical* aggregates of earlier dates, but
# never as features of the flight being predicted.
POST_DEPARTURE_COLS = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]

ID_COLS = ["id", "name", "time_hour", "year", "day"]

# Typical taxi-out + taxi-in minutes; only used to round the residual
# (gate-to-gate minus air time) to a whole-hour time-zone offset.
_TYPICAL_TAXI_MIN = 25


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    # Scheduled departure timestamp (local NYC time). sched_dep_time is HHMM.
    df["sched_dep_dt"] = df["date"] + pd.to_timedelta(
        df["hour"] * 60 + df["minute"], unit="min"
    )
    return df.sort_values(["sched_dep_dt", "id"]).reset_index(drop=True)


def add_target(df: pd.DataFrame, threshold: int = config.DELAY_THRESHOLD_MIN) -> pd.DataFrame:
    """Binary target is_delayed = arr_delay >= threshold, plus outcome timing.

    Cancelled (no dep_time) and diverted (departed, no arr_delay) flights get
    NaN: they stay in the frame so the *schedule* features (congestion) still
    count them, but they are dropped before modelling.
    """
    df = df.copy()
    df["is_cancelled"] = df["dep_time"].isna().astype(int)
    df["is_diverted"] = (df["dep_time"].notna() & df["arr_delay"].isna()).astype(int)
    df["is_delayed"] = (df["arr_delay"] >= threshold).astype(float)
    df.loc[df["arr_delay"].isna(), "is_delayed"] = float("nan")
    return add_outcome_times(df)


def target_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    """Alternative target definitions for the sensitivity analysis.

    ``"arr15"``            : the main target (NaN for cancelled/diverted)
    ``"arr15_or_cancel"``  : cancelled and diverted flights count as delayed
                             (a passenger-disruption view)
    """
    if variant == "arr15":
        return df["is_delayed"]
    if variant == "arr15_or_cancel":
        disrupted = (df["is_cancelled"] == 1) | (df["is_diverted"] == 1)
        return df["is_delayed"].where(~disrupted, 1.0)
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# Outcome timing (decision clock)
# ---------------------------------------------------------------------------
def _hhmm_to_min(hhmm):
    return (hhmm // 100) * 60 + hhmm % 100


def _in_dst(ts: pd.Series) -> np.ndarray:
    lo, hi = (pd.Timestamp(t) for t in config.DST_2013)
    return ((ts >= lo) & (ts < hi)).to_numpy()


def dest_offsets(df: pd.DataFrame) -> pd.Series:
    """Re-derive destination clock minus NYC clock (minutes) during DST.

    VERIFICATION ONLY -- featurisation uses the fixed table
    ``config.DEST_OFFSET_MIN``. The dataset has no airports table, so the
    offset can be recovered: (local arrival - local departure) - air time =
    taxi time + offset, and taxi time (~25 min) is far smaller than an hour,
    so rounding to a whole hour is unambiguous.
    """
    ok = df["arr_time"].notna() & df["dep_time"].notna() & df["air_time"].notna()
    ok &= _in_dst(df["sched_dep_dt"])
    f = df[ok]
    resid = (_hhmm_to_min(f["arr_time"]) - _hhmm_to_min(f["dep_time"])) % 1440 - f["air_time"]
    med = resid.groupby(f["dest"]).median()
    return (np.round((med - _TYPICAL_TAXI_MIN) / 60) * 60).astype(int).rename("offset_min")


def arrival_nyc(df: pd.DataFrame) -> pd.Series:
    """Actual gate-arrival timestamp on the NYC clock (NaT if not arrived)."""
    tz = df["dest"].map(config.DEST_OFFSET_MIN).astype(float)
    # Non-DST destinations sit one hour further ahead of NYC in winter.
    winter_no_dst = df["dest"].isin(config.NO_DST_AIRPORTS) & ~_in_dst(df["sched_dep_dt"])
    tz = tz + 60 * winter_no_dst
    actual_dep = df["sched_dep_dt"] + pd.to_timedelta(df["dep_delay"], unit="min")
    gate_to_gate = (_hhmm_to_min(df["arr_time"]) - _hhmm_to_min(df["dep_time"]) - tz) % 1440
    return actual_dep + pd.to_timedelta(gate_to_gate, unit="min")


def add_outcome_times(df: pd.DataFrame) -> pd.DataFrame:
    """When each outcome becomes known, on the NYC clock.

    * ``arr_known_dt``: actual gate arrival (delay outcome); NaT if none.
    * ``dep_known_dt``: actual departure, or scheduled departure for cancelled
      flights (cancellation outcome).
    """
    df = df.copy()
    df["arr_known_dt"] = arrival_nyc(df)
    # cancelled flights have no dep_delay -> known at the scheduled departure
    df["dep_known_dt"] = df["sched_dep_dt"] + pd.to_timedelta(
        df["dep_delay"].fillna(0), unit="min")
    return df


def available_date(known_dt: pd.Series, decision_hour: int = config.DECISION_HOUR) -> pd.Series:
    """First target date D whose decision time (D-1, decision_hour:00) is at
    or after ``known_dt``. An outcome may feed features of D only if
    D >= available_date.

    Example (decision_hour=20): known 14:00 on day X -> usable from X+1;
    known 21:30 on day X -> usable from X+2.
    """
    shifted = known_dt - pd.Timedelta(hours=decision_hour)
    return shifted.dt.ceil("D") + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def assign_split(df: pd.DataFrame, splits: dict = config.SPLITS) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (lo, hi) in splits.items():
        out[(df["date"] >= lo) & (df["date"] <= hi)] = name
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for training/evaluation: known outcome, past warm-up."""
    keep = df["is_delayed"].notna() & (df["date"] > config.WARMUP_END)
    return df.loc[keep].reset_index(drop=True)
