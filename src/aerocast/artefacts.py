"""Checks that a table recomputed in a notebook equals the artefact that
``scripts/run_phase1.py`` saved in ``results/`` (the single source of truth)."""
from __future__ import annotations

import io

import pandas as pd

from . import config


def assert_matches(df: pd.DataFrame, name: str, key: str | None = None) -> str:
    """Raise unless ``df`` equals ``results/<name>`` after a CSV round trip.

    ``key`` restricts the saved table to the rows whose ``key`` value appears
    in ``df`` (in ``df``'s order), for tables that show a subset of an artefact.
    """
    a = pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip")
    b = pd.read_csv(config.RESULTS_DIR / name, float_precision="round_trip")
    if key is not None:
        b = b.set_index(key).loc[a[key]].reset_index()
        a = a.reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    return f"OK: identical to results/{name}"
