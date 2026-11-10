"""Causal daily delay-history sequences for the GRU (Step 4).

For each entity (an origin, a carrier, a route) and each target date D, the
input is the last ``SEQ_LEN`` *availability days* of daily outcome counts:
slot ``-1`` (the newest) holds the outcomes that first became usable for D,
i.e. known between the decision clocks of D-1 and D (20:00 on D-2 and on
D-1); slot ``-k`` holds those first usable for D-k+1. This is the same
bucketing as ``aerocast.features.rolling_past``: a Phase 1 ``w``-day history
feature is exactly the sum of the last ``w`` slots
(tests/test_sequences.py::test_sequence_matches_history).

Each outcome uses its own clock (``features.KNOWN_AT``): delay outcomes become
known at gate arrival, cancellations at the (scheduled or actual) departure.
No entity is keyed by tail number: counting an aircraft's day-D legs is the
Phase 1 cancellation leak (rule G7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from aerocast import config as p1
from aerocast import data, features
from aerocast.pipeline import Prepared

from . import guards

SEQ_LEN = 28
ENTITY_SPECS: dict[str, tuple[str, ...]] = {
    "origin": ("origin",),
    "carrier": ("carrier",),
    "route": ("origin", "dest"),
}
# (channel, outcome column, statistic); the outcome column decides the clock.
RAW_CHANNELS = [
    ("n_arr", "is_delayed", "count"),      # arrivals with a known delay outcome
    ("n_late", "is_delayed", "sum"),       # of which >= 15 min late
    ("delay_sum", "_clip", "sum"),         # sum of arrival delay clipped to [-60, 240]
    ("n_dep", "is_cancelled", "count"),    # departures/cancellations known
    ("n_cancel", "is_cancelled", "sum"),   # of which cancelled
]
CHANNELS = [c for c, _, _ in RAW_CHANNELS]
CLIP_MIN = (-60, 240)                      # as Phase 1 hist_meandelay_origin_1d


@dataclass
class EntitySequences:
    """Daily outcome counts of one entity type, padded with ``seq_len - 1``
    empty days before the first date so every date has a full window."""

    name: str
    keys: tuple[str, ...]
    dates: pd.DatetimeIndex        # target dates (one per calendar day)
    entities: pd.Index             # entity values (MultiIndex for routes)
    daily: np.ndarray              # float64 [seq_len - 1 + n_dates, n_entities, n_channels]
    seq_len: int

    def windows(self) -> np.ndarray:
        """View [n_dates, n_entities, seq_len, n_channels], oldest slot first."""
        w = sliding_window_view(self.daily, self.seq_len, axis=0)   # [D, E, C, L]
        return w.transpose(0, 1, 3, 2)

    def lookup(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(date position, entity position) of each row; -1 if not covered."""
        day = self.dates.get_indexer(pd.to_datetime(frame["date"]))
        if len(self.keys) == 1:
            ent = self.entities.get_indexer(frame[self.keys[0]])
        else:
            ent = self.entities.get_indexer(pd.MultiIndex.from_frame(frame[list(self.keys)]))
        return day, ent


def _daily_wide(df: pd.DataFrame, keys: tuple[str, ...], col: str, stat: str,
                dates: pd.DatetimeIndex, entities: pd.Index, decision_hour: int) -> np.ndarray:
    """[n_dates, n_entities] daily ``stat`` of ``col`` by availability date
    (the per-day table that ``features.rolling_past`` rolls up)."""
    known = df[features.KNOWN_AT[col]]
    obs = df[df[col].notna() & known.notna()]
    avail = data.available_date(known[obs.index], decision_hour).rename("avail")
    daily = obs.groupby(list(keys) + [avail])[col].agg(stat)
    wide = daily.unstack(list(range(len(keys))))
    # availability dates after the last flight date are dropped (no target uses them)
    return wide.reindex(index=dates, columns=entities).fillna(0).to_numpy(dtype=np.float64)


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN,
                    decision_hour: int = p1.DECISION_HOUR,
                    specs: dict[str, tuple[str, ...]] = ENTITY_SPECS) -> dict[str, EntitySequences]:
    """Sequences for every entity type in ``specs``.

    ``df`` must be the full schedule with target (``data.add_target``),
    cancelled flights included, as for the Phase 1 features. Entities are all
    key values of the timetable (schedule information, not outcomes).
    """
    if any("tailnum" in keys for keys in specs.values()):
        raise guards.LeakageError("tail-number sequences would count day-D legs (rule G7)")
    df = df.assign(_clip=df["arr_delay"].clip(*CLIP_MIN))
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out = {}
    for name, keys in specs.items():
        if len(keys) == 1:
            entities = pd.Index(sorted(df[keys[0]].dropna().unique()), name=keys[0])
        else:
            entities = pd.MultiIndex.from_frame(
                df[list(keys)].dropna().drop_duplicates().sort_values(list(keys)))
        chans = [_daily_wide(df, keys, col, stat, dates, entities, decision_hour)
                 for _, col, stat in RAW_CHANNELS]
        daily = np.stack(chans, axis=-1)
        pad = np.zeros((seq_len - 1, *daily.shape[1:]))
        out[name] = EntitySequences(name, keys, dates, entities,
                                    np.concatenate([pad, daily]), seq_len)
    return out


# ---------------------------------------------------------------------------
# Model inputs: derived channels, standardised on the fitting rows only
# ---------------------------------------------------------------------------
DERIVED = ["log_n_arr", "late_rate", "mean_delay", "log_n_dep", "cancel_rate", "has_arrivals"]
_STANDARDISED = DERIVED[:-1]           # the 0/1 indicator stays as is


def derive(w: np.ndarray) -> np.ndarray:
    """Raw count windows [..., 5] -> model channels [..., 6]. Rates of empty
    days are 0, flagged by ``has_arrivals`` = 0."""
    n_arr, n_late, dsum, n_dep, n_can = (w[..., k] for k in range(len(CHANNELS)))
    with np.errstate(invalid="ignore", divide="ignore"):
        late = np.where(n_arr > 0, n_late / n_arr, 0.0)
        mean_d = np.where(n_arr > 0, dsum / n_arr, 0.0)
        canc = np.where(n_dep > 0, n_can / n_dep, 0.0)
    return np.stack([np.log1p(n_arr), late, mean_d, np.log1p(n_dep), canc,
                     (n_arr > 0).astype(float)], axis=-1)


class SequenceNormalizer:
    """Per entity type: mean / sd of each derived channel over the windows
    that the fitting rows use (one per distinct (date, entity) pair)."""

    def fit(self, seqs: dict[str, EntitySequences], frame: pd.DataFrame) -> "SequenceNormalizer":
        self.stats_ = {}
        for name, s in seqs.items():
            day, ent = s.lookup(frame)
            if (day < 0).any() or (ent < 0).any():
                raise ValueError(f"{name}: fitting rows outside the sequence table")
            pairs = np.unique(np.stack([day, ent], axis=1), axis=0)
            x = derive(s.windows()[pairs[:, 0], pairs[:, 1]])[..., : len(_STANDARDISED)]
            x = x.reshape(-1, x.shape[-1])
            sd = x.std(axis=0)
            self.stats_[name] = (x.mean(axis=0), np.where(sd > 0, sd, 1.0))
        return self

    def transform(self, seqs: dict[str, EntitySequences]) -> dict[str, np.ndarray]:
        """Normalised windows per entity type, float32 [n_dates, n_entities, L, 6]."""
        out = {}
        for name, s in seqs.items():
            x = derive(s.windows())
            mu, sd = self.stats_[name]
            x[..., : len(_STANDARDISED)] = (x[..., : len(_STANDARDISED)] - mu) / sd
            out[name] = x.astype(np.float32)
        return out


def fit_sequence_normalizer(P: Prepared, seqs: dict[str, EntitySequences], rows: np.ndarray,
                            fit_date) -> SequenceNormalizer:
    """Fit on modelling-row positions ``rows`` for a model fitted at the clock
    of ``fit_date``; same guards as ``tensors.fit_encoder``."""
    from .tensors import row_info

    rows = np.asarray(rows)
    info = row_info(P).iloc[rows]
    guards.assert_rows_in_splits(info["split"], {"train"}, "sequence normaliser fit")
    guards.assert_labels_available(info["avail"], fit_date, "sequence normaliser fit")
    return SequenceNormalizer().fit(seqs, P.meta.iloc[rows])


def row_index(P: Prepared, seqs: dict[str, EntitySequences]) -> dict[str, np.ndarray]:
    """Per entity type, [n_rows, 2] (date position, entity position) of every
    modelling row, used to gather its window at batch time."""
    out = {}
    for name, s in seqs.items():
        day, ent = s.lookup(P.meta)
        if (day < 0).any() or (ent < 0).any():
            raise ValueError(f"{name}: modelling rows outside the sequence table")
        out[name] = np.stack([day, ent], axis=1)
    return out


def default_sequences(P: Prepared) -> dict[str, EntitySequences]:
    """Sequences for the Phase 1 modelling frame (full schedule in ``P.raw``)."""
    return build_sequences(P.raw, SEQ_LEN, p1.DECISION_HOUR)

