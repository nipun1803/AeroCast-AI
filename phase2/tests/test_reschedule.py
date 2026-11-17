"""Step 8: what-if rescheduling mechanics and leakage (docs/01_inputs_and_leakage.md)."""
import numpy as np
import pandas as pd
import pytest

from aerocast import evaluate, leakage
from aerocast_dl import reschedule

D = pd.Timestamp("2013-09-17")


@pytest.fixture(scope="module")
def ctx(P):
    return reschedule.build_context(P.raw)


@pytest.fixture(scope="module")
def S(P):
    return reschedule.build_schedule(P.raw, P)       # also checks alignment with P


def _stub_score(X, meta):
    """Later and busier departures look riskier, as in the data."""
    return (0.15 * X["dep_hour"].to_numpy(float) + 0.01 * X["sched_origin_hour"].to_numpy(float)
            + 3 * X[reschedule.OH_RATE].to_numpy(float) - 3)


def _day(F, origin="JFK", date=D):
    """Row positions of one origin-day in ``F`` (P or a Schedule)."""
    m = (F.meta["origin"] == origin) & (F.meta["date"] == date)
    return np.flatnonzero(m.to_numpy())


@pytest.mark.slow
def test_context_reproduces_phase1_features(P, ctx):
    """Timetable counts and the origin-hour history table give back the Phase 1
    feature values of every modelling row (so a move edits them consistently)."""
    meta, X = P.meta, P.X
    key = pd.MultiIndex.from_arrays([meta["origin"], meta["date"], meta["hour"]])
    np.testing.assert_array_equal(ctx.hour_counts.reindex(key).to_numpy(), X["sched_origin_hour"])
    rng = np.random.default_rng(0)
    for i in rng.choice(len(meta), 3000, replace=False):
        rate, logn = reschedule.oh_history(ctx, meta["date"].iloc[i], meta["origin"].iloc[i],
                                           int(meta["hour"].iloc[i]))
        assert rate == pytest.approx(X[reschedule.OH_RATE].iloc[i], abs=1e-12)
        assert logn == pytest.approx(X[reschedule.OH_LOGN].iloc[i], abs=1e-12)


@pytest.mark.slow
def test_congestion_identity_and_round_trip(P, ctx):
    """Recomputing congestion from the day's counts reproduces Phase 1; moving
    a flight and moving it back restores every feature exactly."""
    pos = _day(P)
    X0 = P.X.iloc[pos].copy()
    hours = P.meta["hour"].to_numpy()[pos].astype(int)
    counts = np.zeros(24, dtype=int)
    c = ctx.hour_counts.loc[("JFK", D)]
    counts[c.index.to_numpy(dtype=int)] = c.to_numpy()
    X1 = X0.copy()
    reschedule._set_congestion(X1, hours, counts)
    pd.testing.assert_frame_equal(X1, X0, check_dtype=False)

    i = int(np.argmax(hours == 15))
    minute = int(P.meta["minute"].iloc[pos[i]])
    sat = int(P.meta["sched_arr_time"].iloc[pos[i]])
    arr = (sat // 100) * 60 + sat % 100
    X2, h2, c2 = X0.copy(), hours.copy(), counts.copy()
    c2[15] -= 1; c2[13] += 1; h2[i] = 13
    reschedule._set_congestion(X2, h2, c2)
    reschedule._move_row(X2, i, 13, -2, minute, arr, ctx, D, "JFK")
    assert not X2.equals(X0)
    c2[13] -= 1; c2[15] += 1; h2[i] = 15
    reschedule._set_congestion(X2, h2, c2)
    # back by +2 h from the moved arrival time (arr - 120) restores the original
    reschedule._move_row(X2, i, 15, 2, minute, (arr - 120) % 1440, ctx, D, "JFK")
    pd.testing.assert_frame_equal(X2, X0, check_dtype=False, atol=1e-12)


@pytest.mark.slow
def test_reschedule_no_outcomes(P, S, ctx):
    """Randomising every outcome from day D on (Phase 1 scramble_future, which
    also cancels about 30% of those flights) gives the same schedule rows,
    features, candidates and plan for day D: the rule never uses realised
    day-D outcomes, including which flights will lack one."""
    raw2 = leakage.scramble_future(P.raw, D)
    S2 = reschedule.build_schedule(raw2)
    ctx2 = reschedule.build_context(raw2)
    pos, pos2 = _day(S), _day(S2)
    # positive control: the outcome-filtered cohort of day D does change
    assert (raw2.loc[raw2["date"] == D, "is_delayed"].notna().sum()
            != P.raw.loc[P.raw["date"] == D, "is_delayed"].notna().sum())
    np.testing.assert_array_equal(pos, pos2)
    pd.testing.assert_frame_equal(S.X.iloc[pos], S2.X.iloc[pos2])
    risk = _stub_score(S.X.iloc[pos], None)
    risk2 = _stub_score(S2.X.iloc[pos2], None)
    dates = np.full(len(pos), np.datetime64(D))
    cand = reschedule.candidates_by_day(dates, risk, pos)
    np.testing.assert_array_equal(cand, reschedule.candidates_by_day(dates, risk2, pos2))
    cand = cand[np.argsort(-risk[np.searchsorted(pos, cand)], kind="stable")]
    ident = evaluate.IdentityCalibrator()
    a, ga = reschedule.plan_origin_day(S, ctx, _stub_score, ident, "JFK", D, cand, 2, 0.0)
    b, gb = reschedule.plan_origin_day(S2, ctx2, _stub_score, ident, "JFK", D, cand, 2, 0.0)
    assert len(a) > 0                          # positive control: the plan is not empty
    assert a == b and ga == gb


@pytest.mark.slow
def test_schedule_is_full_timetable(P, S):
    """Candidates come from every scheduled flight: on each calibration and
    test day the schedule has exactly the timetable's flights, more than the
    modelling rows wherever a flight was later cancelled or diverted, so the
    daily top-10% budget is a share of the whole timetable."""
    for split in ("calib", "test"):
        pos = np.flatnonzero(S.split == split)
        sched = S.meta.iloc[pos].groupby("date").size()
        full = P.raw.groupby("date").size().reindex(sched.index)
        model = P.meta.iloc[P.idx[split]].groupby("date").size().reindex(sched.index, fill_value=0)
        np.testing.assert_array_equal(sched.to_numpy(), full.to_numpy())
        assert (sched > model).any() and (sched >= model).all()
        risk = np.random.default_rng(0).random(len(pos))
        cand = reschedule.candidates_by_day(S.meta["date"].to_numpy()[pos], risk, pos)
        per_day = S.meta.iloc[cand].groupby("date").size().reindex(sched.index, fill_value=0)
        assert (np.abs(per_day - 0.1 * sched) <= 1).all()            # 10% of the whole timetable


def test_plan_refuses_partial_origin_day(S, ctx):
    """A plan built on a subset of the origin-day (e.g. only flights with an
    outcome) would sum risk over the wrong cohort: it must refuse to run."""
    pos = _day(S)
    sub = reschedule.Schedule(S.meta.drop(index=pos[:1]).reset_index(drop=True),
                              S.X.drop(index=pos[:1]).reset_index(drop=True),
                              np.delete(S.split, pos[:1]), np.delete(S.model_pos, pos[:1]))
    with pytest.raises(AssertionError, match="full schedule"):
        reschedule.plan_origin_day(sub, ctx, _stub_score, evaluate.IdentityCalibrator(),
                                   "JFK", D, _day(sub)[:5], 2, 0.0)


@pytest.mark.slow
def test_capacity_and_hour_range(S, ctx):
    pos = _day(S)
    cand = pos[np.argsort(-_stub_score(S.X.iloc[pos], None))][:60]
    moves, _ = reschedule.plan_origin_day(S, ctx, _stub_score, evaluate.IdentityCalibrator(),
                                          "JFK", D, cand, 2, 0.0)
    lo, hi = ctx.hour_range["JFK"]
    counts = ctx.hour_counts.loc[("JFK", D)].reindex(range(24), fill_value=0).to_numpy().copy()
    for m in moves:
        assert lo <= m["new_hour"] <= hi and abs(m["shift"]) <= 2
        counts[m["old_hour"]] -= 1
        counts[m["new_hour"]] += 1
        assert counts.max() <= ctx.cap["JFK"]


def test_reschedule_params_sepoct():
    """Parameters are tuned without test rows."""
    df = pd.DataFrame({"split": ["calib", "test"]})
    with pytest.raises(ValueError):
        reschedule.tune(None, None, df, "logreg", "raw", None, print)
