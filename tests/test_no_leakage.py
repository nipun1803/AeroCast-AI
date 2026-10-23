"""Leakage guards: features for date D must not depend on outcomes from D onward.

The check itself lives in ``aerocast.leakage`` (also run inline in notebook 02).
"""
import numpy as np
import pandas as pd
import pytest

from aerocast import config, data, features, leakage, models

CUTOFF = pd.Timestamp("2013-06-15")


@pytest.fixture(scope="module")
def raw():
    return data.add_target(data.load_raw())


@pytest.fixture(scope="module")
def built(raw):
    return features.build_features(raw)


@pytest.fixture(scope="module")
def check(raw, built):
    return leakage.future_dependence_check(raw, CUTOFF, X=built[0])


def test_features_up_to_cutoff_ignore_same_day_and_future(check):
    assert check["changed_up_to_cutoff"] == 0


def test_history_does_change_after_cutoff(check):
    """Sanity check the test itself: scrambling must affect later dates."""
    assert check["changed_after_cutoff"] > 0
    assert "hist_rate_origin_1d" in check["changed_columns_after"]


@pytest.mark.parametrize("target_date", ["2013-02-14", "2013-07-11", "2013-12-20"])
def test_late_evening_d_minus_1_arrivals_do_not_reach_day_d(raw, built, target_date):
    """Decision clock: scrambling D-1 flights that arrived after 20:00 on D-1
    leaves every feature of day D unchanged; scrambling the ones that arrived
    before it (positive control) changes day D's history features."""
    c = leakage.decision_clock_check(raw, target_date, X=built[0])
    assert c["d_minus_1_flights_after_clock"] > 0
    assert c["changed_cells_on_D_after_scrambling_late"] == 0
    assert c["changed_cells_on_D_after_scrambling_early"] > 0


def test_timezone_table_matches_data(raw):
    derived = data.dest_offsets(raw)
    assert all(config.DEST_OFFSET_MIN[d] == o for d, o in derived.items())
    assert set(raw["dest"]) <= set(config.DEST_OFFSET_MIN)


def test_arrival_times_are_physically_plausible(raw):
    # completed flights (diverted ones have arr_time but no air_time / arr_delay
    # and never feed the delay history)
    done = raw["arr_known_dt"].notna() & raw["air_time"].notna()
    actual_dep = raw.loc[done, "sched_dep_dt"] + pd.to_timedelta(raw.loc[done, "dep_delay"], "min")
    taxi = (raw.loc[done, "arr_known_dt"] - actual_dep).dt.total_seconds() / 60 - raw.loc[done, "air_time"]
    assert taxi.between(0, 300).all()   # gate-to-gate minus air time = taxi time


def test_no_post_departure_columns_in_features(built):
    X, _ = built
    assert not set(data.POST_DEPARTURE_COLS) & set(X.columns)


def test_splits_are_chronological_and_disjoint(raw):
    split = data.assign_split(raw)
    assert split.notna().all()
    last = raw.groupby(split)["date"].agg(["min", "max"]).loc[list(config.SPLITS)]
    assert (last["max"].iloc[:-1].to_numpy() < last["min"].iloc[1:].to_numpy()).all()


def test_rolling_folds_never_train_on_or_after_val_month():
    months = np.repeat(np.arange(1, 9), 10)
    dates = pd.to_datetime([f"2013-{m:02d}-01" for m in months]).to_numpy()
    avail = dates + np.timedelta64(1, "D")
    for tr, va in models.rolling_folds(months, dates=dates, avail_dates=avail):
        assert months[tr].max() < months[va].min()


def test_smoothed_rate_limits():
    assert features.smoothed_rate(0, 0, 0.25, m=20) == 0.25         # no history -> prior
    assert abs(features.smoothed_rate(5000, 10000, 0.25, m=20) - 0.5) < 1e-3  # n >> m -> raw
