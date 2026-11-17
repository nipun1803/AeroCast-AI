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
    with pytest.raises(guards.LeakageError):
        evaluation.load_no_outcome()          # which test flights lack an outcome is test information


def _policy_day(n_lab=180, n_ext=20, seed=0):
    rng = np.random.default_rng(seed)
    d = np.datetime64("2013-11-05")
    return (np.full(n_lab, d), rng.random(n_lab), (rng.random(n_lab) < 0.3).astype(int),
            np.full(n_ext, d), rng.random(n_ext))


def test_daily_policy_budget_is_the_whole_timetable():
    """The daily top-10% is 10% of every scheduled flight, the ones that will
    lack an outcome included; dropping them (the old cohort) changes who is
    flagged among the flights with an outcome (positive control)."""
    dates, q, y, de, qe = _policy_day()
    r = evaluation.daily_policy(dates, q, y, de, qe, 0.10)
    assert r["scheduled"] == 200 and r["flagged_total"] == 20
    assert r["tp"] + r["fp"] == 20 - r["flagged_no_outcome"]
    assert r["flagged_no_outcome"] > 0
    old = evaluation.daily_policy(dates, q, y, de[:0], qe[:0], 0.10)
    assert old["tp"] + old["fp"] == 18 != r["tp"] + r["fp"]


def test_daily_policy_flags_do_not_depend_on_labels():
    """Who is flagged depends on scores only: permuting the labels of the
    flights with an outcome leaves the flagged count and the flags unchanged."""
    dates, q, y, de, qe = _policy_day(seed=3)
    a = evaluation.daily_policy(dates, q, y, de, qe, 0.10)
    b = evaluation.daily_policy(dates, q, np.random.default_rng(9).permutation(y), de, qe, 0.10)
    assert a["tp"] + a["fp"] == b["tp"] + b["fp"] and a["flagged_no_outcome"] == b["flagged_no_outcome"]


@pytest.mark.parametrize("name", ["raw", "platt", f"rolling_platt{calibrate.ROLL}",
                                  f"rolling_temperature{calibrate.ROLL}"])
def test_calibrate_on_days_matches_apply(name):
    """Flights without an outcome get exactly the per-day map the calibrated
    flights with an outcome got."""
    pool = _pool()
    target = pool["split"] == "test"
    ref = calibrate.apply(name, pool) if name != "raw" else pool["p"]
    out = evaluation.calibrate_on_days(name, pool, pool["date"][target], pool["p"][target])
    np.testing.assert_allclose(out, ref[target], rtol=1e-7, atol=1e-9)


@pytest.mark.slow
def test_no_outcome_flights_and_xgb_worker(P):
    """The extra flights are exactly the test flights without an arrival
    outcome, and the XGBoost worker scores extra flights through the same
    feature path as modelling rows (checked on flights that have one)."""
    from aerocast_dl import pipeline
    S, ex = evaluation.no_outcome_test_flights(P)
    raw_test = P.raw[(P.raw["date"] >= "2013-11-01")]
    assert ex.sum() == raw_test["is_delayed"].isna().sum() > 0
    lab = np.flatnonzero(S.model_pos >= 0)[-5:]
    n = len(P.meta)
    probs, extra = pipeline.run_xgb_jobs(
        [(np.zeros((n, 0), np.float32), P.idx["train"][:4000], S.model_pos[lab],
          np.zeros((len(lab), 0), np.float32))], extra_index=S.raw_index[lab])
    np.testing.assert_allclose(extra[0], probs[0], rtol=1e-6)


def test_test_lock_watches_existing_rules_files():
    assert all((testlock.config.ROOT / f).is_file() for f in testlock.PROTOCOL_FILES)


def test_test_lock_needs_committed_protocol(monkeypatch):
    monkeypatch.setenv(testlock.ENV, "1")
    monkeypatch.setattr(testlock, "protocol_is_committed", lambda: False)
    with pytest.raises(guards.LeakageError):
        testlock.require_final()


def test_slice_table_gain_and_stability():
    rng = np.random.default_rng(1)
    n = 4000
    keys = pd.DataFrame({"carrier": rng.choice(["AA", "B6"], n), "dep_hour": rng.choice(["05-09", "18-23"], n)})
    y = (rng.random(n) < 0.2).astype(int)
    good = np.clip(0.2 + 0.5 * (y - 0.2) + rng.normal(0, 0.1, n), 0.01, 0.99)
    t = evaluation.slice_table(keys, y, {"logreg": rng.random(n), "mlp": good})
    assert len(t) == 4 and set(t["dimension"]) == {"carrier", "dep_hour"}
    assert (t["ap_gain_logreg"] == 0).all() and (t["ap_gain_mlp"] > 0.2).all()
    assert t["n"].groupby(t["dimension"]).sum().eq(n).all() and t["stable"].all()


def test_score_drift_is_label_free_and_detects_shift():
    rng = np.random.default_rng(2)
    n = 3000
    split = np.repeat(["train", "calib", "test"], n)
    df = pd.DataFrame({"split": split, "y": rng.integers(0, 2, 3 * n)})
    for m in evaluation.MODELS:
        df[m] = rng.random(3 * n)
    df.loc[df.split == "test", "mlp"] = rng.random(n) ** 3            # shifted test scores
    keys = pd.DataFrame({"carrier": rng.choice(list("ABC"), 3 * n), "origin": rng.choice(list("XY"), 3 * n)})
    a = evaluation.score_drift(df, keys).set_index("input")
    df["y"] = 1 - df["y"]
    b = evaluation.score_drift(df, keys).set_index("input")
    pd.testing.assert_frame_equal(a, b)                                # labels never used
    assert a.loc["score: mlp", "test"] > 0.25 and a.loc["score: logreg", "test"] < 0.05


def test_new_figures_render():
    from aerocast_dl import plots
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.4 * (y - 0.2) + rng.normal(0, 0.15, 2000), 0, 1)
    plots.pr_curves({"A": (y, p), "B": (y, rng.random(2000))})
    pol = pd.DataFrame([{"policy": "daily top 10%", "tp": 10, "fp": 5, "fn": 20, "tn": 65,
                         "precision": 0.667, "recall": 0.333},
                        {"policy": "frozen threshold", "tp": 15, "fp": 10, "fn": 15, "tn": 60,
                         "precision": 0.6, "recall": 0.5}])
    plots.confusion(pol, "A")
    plots.plt.close("all")
