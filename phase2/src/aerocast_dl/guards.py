"""Leakage guards called inside every Phase 2 fit (docs/01_inputs_and_leakage.md).

They raise instead of warning: a fit that would use an unavailable label or a
forbidden split must not run at all.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from aerocast import data


class LeakageError(AssertionError):
    """A fit was asked to use information it may not see."""


def assert_labels_available(avail_dates, fit_date, what: str = "fit") -> None:
    """G2: every row used by a fit for ``fit_date`` must have its label known
    by the decision clock of ``fit_date`` (unknown availability counts as
    unavailable)."""
    avail_dates = np.asarray(avail_dates)
    if not len(avail_dates):
        raise LeakageError(f"{what}: no rows")
    known = data.outcomes_available_by(avail_dates, fit_date)
    if not known.all():
        raise LeakageError(
            f"{what}: {int((~known).sum())} of {len(known)} rows have labels "
            f"not yet available at the decision clock for {fit_date}")


def assert_rows_in_splits(row_splits, allowed: Iterable[str], what: str = "fit") -> None:
    """G3/G4: a fit may only use rows from the ``allowed`` splits (networks and
    encoders: train only; calibrators and thresholds: calib only)."""
    allowed = set(allowed)
    bad = sorted(set(np.asarray(row_splits, dtype=object)) - allowed)
    if bad:
        raise LeakageError(f"{what}: uses rows from {bad}; allowed {sorted(allowed)}")
