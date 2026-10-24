"""Unit tests for aerocast.evaluate (synthetic data, no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from aerocast import evaluate

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def calibrated():
    """Labels drawn from the predicted probabilities: perfectly calibrated."""
    p = RNG.uniform(0, 1, 200_000)
    y = (RNG.uniform(0, 1, p.size) < p).astype(int)
    return y, p


def test_ece_near_zero_when_calibrated(calibrated):
    y, p = calibrated
    assert evaluate.expected_calibration_error(y, p) < 0.01
    assert evaluate.expected_calibration_error_equal_mass(y, p) < 0.01


def test_ece_detects_miscalibration(calibrated):
    y, p = calibrated
    assert evaluate.expected_calibration_error(y, p * 0.5) > 0.2   # systematic under-prediction


@pytest.mark.parametrize("n_bins", [1, 10, 15, 200])
def test_equal_mass_ece_constant_scores_ignore_label_order(n_bins):
    """Identical probabilities form one bin, even when labels are grouped."""
    p = np.full(100, 0.5)
    grouped = np.r_[np.zeros(50), np.ones(50)]
    alternating = np.tile([0, 1], 50)
    assert evaluate.expected_calibration_error_equal_mass(grouped, p, n_bins) == pytest.approx(0)
    assert evaluate.expected_calibration_error_equal_mass(alternating, p, n_bins) == pytest.approx(0)
    assert evaluate.expected_calibration_error_equal_mass(grouped, p + 0.2, n_bins) == pytest.approx(0.2)


def test_equal_mass_ece_tied_groups_is_permutation_invariant():
    """Preserve score groups rather than cancelling their opposite errors."""
    p = np.repeat([0.1, 0.5, 0.9], [60, 30, 10])
    y = np.r_[np.ones(12), np.zeros(48), np.ones(15), np.zeros(15),
              np.ones(6), np.zeros(4)]
    # Score-group errors: 60% * |0.2-0.1| + 30% * 0 + 10% * |0.6-0.9|.
    expected = 0.09
    rng = np.random.default_rng(42)
    for _ in range(20):
        order = rng.permutation(len(p))
        assert evaluate.expected_calibration_error_equal_mass(y[order], p[order], 10) == pytest.approx(expected)


def test_equal_mass_ece_unique_scores_preserve_balanced_bins():
    p = np.linspace(0.01, 0.99, 37)
    y = (np.arange(len(p)) % 3 == 0).astype(int)
    bins = np.array_split(np.argsort(p), 15)
    expected = sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins)
    assert evaluate.expected_calibration_error_equal_mass(y, p) == pytest.approx(expected)


def test_threshold_for_precision_reaches_target():
    y = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    p = np.linspace(0.1, 0.8, 8)
    thr = evaluate.threshold_for_precision(y, p, target=0.8)
    flagged = p >= thr
    assert (y[flagged] == 1).mean() >= 0.8
    # lowest threshold reaching 0.8 (flags 0.3..0.8: 5/6 late) -> maximal recall
    assert thr == pytest.approx(p[2])
    assert (y[flagged] == 1).sum() == (y == 1).sum()


def test_threshold_for_precision_unreachable_is_inf():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.8, 0.2])  # the best-ranked flights are negatives
    assert evaluate.threshold_for_precision(y, p, target=0.99) == float("inf")


def test_oracle_recall_at_precision_bounds():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])  # perfect ranking
    assert evaluate.oracle_recall_at_precision(y, p, 0.99) == 1.0


def test_day_block_bootstrap_reproducible_and_shaped():
    dates = np.repeat(pd.date_range("2013-11-01", periods=20).to_numpy(), 50)
    y = RNG.integers(0, 2, dates.size)
    preds = {"a": RNG.uniform(size=dates.size), "b": RNG.uniform(size=dates.size)}
    b1 = evaluate.day_block_bootstrap(dates, y, preds, n_boot=30, seed=7)
    b2 = evaluate.day_block_bootstrap(dates, y, preds, n_boot=30, seed=7)
    pd.testing.assert_frame_equal(b1, b2)
    assert b1.shape == (30, 2)
    assert not b1.equals(evaluate.day_block_bootstrap(dates, y, preds, n_boot=30, seed=8))


@pytest.mark.parametrize("name", list(evaluate.CALIBRATORS))
def test_calibrators_fit_predict_shapes(name, calibrated):
    y, p = calibrated
    q = evaluate.CALIBRATORS[name]().fit(p[:5000], y[:5000]).predict(p[5000:6000])
    assert q.shape == (1000,)
    assert np.all((q >= 0) & (q <= 1))


def test_platt_fixes_a_monotone_distortion(calibrated):
    y, p = calibrated
    distorted = p ** 2  # still a perfect ranking, badly calibrated
    q = evaluate.PlattCalibrator().fit(distorted, y).predict(distorted)
    assert evaluate.expected_calibration_error(y, q) < evaluate.expected_calibration_error(y, distorted)


def test_select_calibrator_keeps_raw_on_near_tie(calibrated):
    y, p = calibrated
    months = np.where(np.arange(p.size) < p.size // 2, 9, 10)
    dates = pd.to_datetime(np.where(months == 9, "2013-09-01", "2013-10-01")).to_numpy()
    choice, table = evaluate.select_calibrator(
        p, y, months, 9, 10, min_gain=0.005, dates=dates,
        avail_dates=dates + np.timedelta64(1, "D"), selection_date="2013-11-01")
    assert choice == "raw"  # raw is already calibrated: no recalibrator earns its place
    assert {"raw", "platt", "isotonic"} == set(table["calibrator"])


def test_flag_daily_top_share_flag_rate():
    dates = np.repeat(pd.date_range("2013-11-01", periods=10).to_numpy(), 200)
    p = RNG.uniform(size=dates.size)
    flagged = evaluate.flag_daily_top_share(dates, p, share=0.1)
    per_day = pd.Series(flagged).groupby(dates).mean()
    assert np.allclose(per_day, 0.1)
    # flagged flights are each day's highest scores
    s = pd.Series(p)
    assert all(s[flagged & (dates == d)].min() >= s[~flagged & (dates == d)].max()
               for d in np.unique(dates))


def test_rolling_platt_is_causal():
    """Outcomes not yet available at a day's decision time must not change
    that day's recalibrated scores."""
    days = pd.date_range("2013-10-01", periods=40)
    dates = np.repeat(days.to_numpy(), 100)
    avail = dates + np.timedelta64(1, "D")
    s = RNG.uniform(size=dates.size)
    y = (RNG.uniform(size=dates.size) < s).astype(int)
    target = days[-5:].to_numpy()
    q1 = evaluate.rolling_platt(s, y, dates, avail, target, window=14)
    y2 = y.copy()
    y2[dates >= target[0]] = 1 - y2[dates >= target[0]]   # flip outcomes from day 36 on
    q2 = evaluate.rolling_platt(s, y2, dates, avail, target, window=14)
    first = dates == target[0]   # its fit window ends with outcomes available on target[0]
    assert np.allclose(q1[first], q2[first])
    assert np.isnan(q1[dates < target[0]]).all()


def test_daily_policy_ties_are_not_broken_by_row_order():
    """A constant score must behave like random flagging: later rows of each day
    are more often positive here, but an order-based tie-break would pick them."""
    n_days, per_day = 200, 50
    dates = np.repeat(np.arange(n_days), per_day)
    position = np.tile(np.arange(per_day), n_days)
    y = (np.random.default_rng(0).random(len(dates)) < 0.1 + 0.5 * position / per_day).astype(int)
    flagged = evaluate.flag_daily_top_share(dates, np.full(len(dates), 0.3), share=0.1)
    assert abs(flagged.mean() - 0.1) < 0.005
    lift = y[flagged].mean() / y.mean()
    assert abs(lift - 1.0) < 0.1          # order-based ties would give about 1.6
    assert np.array_equal(flagged, evaluate.flag_daily_top_share(
        dates, np.full(len(dates), 0.3), share=0.1))   # reproducible
