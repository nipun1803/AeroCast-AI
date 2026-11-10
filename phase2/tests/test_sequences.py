"""Step 4 leakage tests for the GRU input sequences (docs/leakage_safeguards.md).

Perturbations reuse the Phase 1 helpers in ``aerocast.leakage`` so that both
phases are tested against the same definition of leakage. Every negative test
has a positive control, so it cannot pass vacuously.
"""
import numpy as np
import pandas as pd
import pytest

from aerocast import config as p1
from aerocast import features, leakage
from aerocast_dl import config, guards, sequences, tensors

DATES = ["2013-02-14", "2013-07-11", "2013-12-20"]      # as the Phase 1 leakage tests


@pytest.fixture(scope="module")
def seqs(P):
    return sequences.build_sequences(P.raw)


def _windows(s):
    return {n: e.windows() for n, e in s.items()}


def _same(a, b, mask):
    return all(np.array_equal(a[n][mask], b[n][mask]) for n in a)


# ---------------------------------------------------------------------------
# 1. Equality with the audited Phase 1 history features
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("window", [1, 7, 14, 28])
def test_sequence_matches_history(P, seqs, window):
    """Summing the last ``window`` slots reproduces ``features.rolling_past``
    exactly, for every flight of the full schedule and every entity type:
    the sequences use the same decision clock as Phase 1."""
    raw = P.raw.assign(_clip=P.raw["arr_delay"].clip(*sequences.CLIP_MIN))
    checks = [("is_delayed", "count", 0), ("is_delayed", "sum", 1), ("_clip", "sum", 2),
              ("is_cancelled", "count", 3), ("is_cancelled", "sum", 4)]
    for name, keys in sequences.ENTITY_SPECS.items():
        s = seqs[name]
        day, ent = s.lookup(raw)
        assert (day >= 0).all() and (ent >= 0).all()
        W = s.windows()[day, ent, -window:]
        for col, stat, ch in checks:
            ref = features.rolling_past(raw, keys, window, col)["n" if stat == "count" else "s"]
            np.testing.assert_array_equal(W[..., ch].sum(axis=1), ref.to_numpy(),
                                          err_msg=f"{name} {col} {stat} {window}d")


# ---------------------------------------------------------------------------
# 2. Future outcomes
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("cutoff", DATES)
def test_sequence_future_perturbation(P, seqs, cutoff):
    """Randomising every outcome from ``cutoff`` on (Phase 1
    ``scramble_future``) leaves all windows up to and including the cutoff
    date unchanged; later windows change (positive control)."""
    s2 = sequences.build_sequences(leakage.scramble_future(P.raw, cutoff))
    a, b = _windows(seqs), _windows(s2)
    upto = seqs["origin"].dates <= pd.Timestamp(cutoff)
    assert _same(a, b, upto)
    assert not _same(a, b, ~upto)


# ---------------------------------------------------------------------------
# 3. Decision clock: late-evening D-1 arrivals
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("target_date", DATES)
def test_sequence_late_arrival(P, seqs, target_date):
    """D-1 flights that arrive after 20:00 on D-1 must not reach day D's
    window, but must reach D+1's; flights that arrived before 20:00 reach D."""
    raw = P.raw
    D = pd.Timestamp(target_date)
    clock = D - pd.Timedelta(days=1) + pd.Timedelta(hours=p1.DECISION_HOUR)
    prev = raw["date"] == D - pd.Timedelta(days=1)
    late = prev & (raw["arr_known_dt"] > clock)
    early = prev & (raw["arr_known_dt"] <= clock)
    assert late.sum() > 0 and early.sum() > 0
    dates = seqs["origin"].dates
    a = _windows(seqs)

    b = _windows(sequences.build_sequences(leakage._scramble_outcomes(raw, late, seed=0)))
    assert _same(a, b, dates <= D)
    assert not _same(a, b, dates == D + pd.Timedelta(days=1))

    c = _windows(sequences.build_sequences(leakage._scramble_outcomes(raw, early, seed=0)))
    assert _same(a, c, dates < D)
    assert not _same(a, c, dates == D)


# ---------------------------------------------------------------------------
# 4. Same-day cancellations
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("target_date", DATES)
def test_sequence_cancellation(P, seqs, target_date):
    """Cancelling 30% of day D's flights the way the raw data records it
    (Phase 1 ``_cancel_rows``: no departure, arrival or tail number) leaves
    every window up to day D unchanged; D+1's window changes (control)."""
    raw = P.raw
    D = pd.Timestamp(target_date)
    rng = np.random.default_rng(0)
    pick = (raw["date"].eq(D) & raw["dep_time"].notna()).to_numpy() & (rng.random(len(raw)) < 0.3)
    assert pick.sum() > 0
    b = _windows(sequences.build_sequences(leakage._cancel_rows(raw, raw.index[pick])))
    a, dates = _windows(seqs), seqs["origin"].dates
    assert _same(a, b, dates <= D)
    assert not _same(a, b, dates == D + pd.Timedelta(days=1))


# ---------------------------------------------------------------------------
# 5. No same-day aircraft rotation (G7)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_no_tail_day_counts(P, seqs):
    """Tail-number entities are refused, and the sequences do not read the
    tail number at all: blanking every tail number changes nothing."""
    assert not any("tailnum" in keys for keys in sequences.ENTITY_SPECS.values())
    with pytest.raises(guards.LeakageError):
        sequences.build_sequences(P.raw, specs={"aircraft": ("tailnum",)})
    blank = P.raw.assign(tailnum=np.nan)
    a, b = _windows(seqs), _windows(sequences.build_sequences(blank))
    assert _same(a, b, np.ones(len(seqs["origin"].dates), dtype=bool))


# ---------------------------------------------------------------------------
# Normaliser and lookup
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_sequence_normalizer_train_only(P, seqs):
    """Statistics fitted on the final training rows do not change when every
    outcome from 1 Sep on is randomised; calibration rows are refused."""
    rows = tensors.final_fit_rows(P)
    a = sequences.fit_sequence_normalizer(P, seqs, rows, config.FINAL_FIT_DATE)
    s2 = sequences.build_sequences(leakage.scramble_future(P.raw, config.FINAL_FIT_DATE))
    b = sequences.fit_sequence_normalizer(P, s2, rows, config.FINAL_FIT_DATE)
    for n in a.stats_:
        for x, y in zip(a.stats_[n], b.stats_[n]):
            np.testing.assert_array_equal(x, y)
    with pytest.raises(guards.LeakageError):
        sequences.fit_sequence_normalizer(P, seqs, P.idx["calib"][:100], config.FINAL_FIT_DATE)


@pytest.mark.slow
def test_row_index_covers_modelling_rows(P, seqs):
    idx = sequences.row_index(P, seqs)
    for n, s in seqs.items():
        assert idx[n].shape == (len(P.meta), 2)
        np.testing.assert_array_equal(s.dates[idx[n][:, 0]], pd.to_datetime(P.meta["date"]))


def test_derive_empty_days():
    w = np.zeros((2, 5))
    w[1] = [4, 1, 40, 5, 1]
    d = sequences.derive(w)
    np.testing.assert_allclose(d[0], [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(d[1], [np.log1p(4), 0.25, 10, np.log1p(5), 0.2, 1])

