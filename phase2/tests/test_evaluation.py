"""Step 7: calibration rule, calibrator clock and the test lock."""
import numpy as np
import pandas as pd
import pytest

from aerocast_dl import calibrate, evaluation, guards, testlock


def _pool(n_days=120, per_day=200, seed=0):
    """Synthetic train/calib/test rows; label known the day after the flight."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2013-08-03", periods=n_days, freq="D"), per_day)
    z = rng.normal(-1, 1.2, len(dates))
    y = (rng.random(len(dates)) < 1 / (1 + np.exp(-0.6 * z))).astype(int)   # true T = 1/0.6
    split = np.where(dates < "2013-09-01", "train", np.where(dates < "2013-11-01", "calib", "test"))
    return {"p": 1 / (1 + np.exp(-z)), "y": y, "date": dates.to_numpy(),
            "avail": (dates + pd.Timedelta(days=1)).to_numpy(), "split": split,
            "month": dates.month.to_numpy()}


def test_temperature_recovers_scale():
    pool = _pool()
    cal = calibrate.TemperatureCalibrator().fit(pool["p"], pool["y"])
    assert cal.T_ == pytest.approx(1 / 0.6, rel=0.08)


def test_select_refuses_test_rows():
    with pytest.raises(ValueError):
        calibrate.select(_pool())


def test_select_picks_a_recalibrator_when_raw_is_miscalibrated():
    pool = {k: v[_pool()["split"] != "test"] for k, v in _pool().items()}
    choice, tbl = calibrate.select(pool)
    assert set(tbl["calibrator"]) == set(calibrate.CANDIDATES)
    assert choice != "raw" and tbl.set_index("calibrator").loc[choice, "gain_vs_raw"] > 0.005


@pytest.mark.parametrize("kind", ["platt", "temperature"])
def test_calibrator_clock(kind):
    """Rolling calibration of day D ignores every label that becomes available
    after D's decision clock: flipping them leaves D's scores unchanged."""
    pool = _pool()
    D = np.datetime64("2013-10-10")
    a = calibrate.rolling(kind, pool["p"], pool["y"], pool["date"], pool["avail"], [D])
    y2 = pool["y"].copy()
    later = pool["avail"] > D
    y2[later] = 1 - y2[later]
    b = calibrate.rolling(kind, pool["p"], y2, pool["date"], pool["avail"], [D])
    day = pool["date"] == D
    np.testing.assert_array_equal(a[day], b[day])
    y3 = pool["y"].copy()                       # positive control: recent known labels matter
    recent = (pool["avail"] <= D) & (pool["avail"] > D - np.timedelta64(28, "D"))
    y3[recent] = 1 - y3[recent]
    c = calibrate.rolling(kind, pool["p"], y3, pool["date"], pool["avail"], [D])
    assert not np.array_equal(a[day], c[day])


def test_day_maps_match_rolling():
    pool = _pool()
    days = pd.to_datetime(["2013-10-05", "2013-11-20"])
    maps = calibrate.day_maps(f"rolling_platt{calibrate.ROLL}", pool, days)
    ref = calibrate.rolling("platt", pool["p"], pool["y"], pool["date"], pool["avail"], days.to_numpy())
    for d in days:
        rows = pool["date"] == np.datetime64(d)
        np.testing.assert_allclose(maps[d].predict(pool["p"][rows]), ref[rows])


def test_test_lock(monkeypatch):
    """Test labels are unreachable without --final."""
    monkeypatch.delenv(testlock.ENV, raising=False)
    with pytest.raises(guards.LeakageError):
        testlock.require_final()
    with pytest.raises(guards.LeakageError):
        evaluation.load_predictions(include_test=True)


def test_test_lock_needs_committed_protocol(monkeypatch):
    monkeypatch.setenv(testlock.ENV, "1")
    monkeypatch.setattr(testlock, "protocol_is_committed", lambda: False)
    with pytest.raises(guards.LeakageError):
        testlock.require_final()
