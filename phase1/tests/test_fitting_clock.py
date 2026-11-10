"""Regression tests: fitting/selection must respect the same clock as features."""
import numpy as np
import pandas as pd
import pytest

from aerocast import config, data, evaluate, models, pipeline


def test_labels_at_clock_are_included_but_later_or_unknown_are_excluded():
    known = pd.Series(pd.to_datetime([
        "2013-10-31 19:59:59", "2013-10-31 20:00:00",
        "2013-10-31 20:00:01", None,
    ]))
    avail = data.available_date(known)
    assert data.outcomes_available_by(avail, "2013-11-01").tolist() == [True, True, False, False]


def test_cv_excludes_unavailable_fit_labels_and_late_selection_labels():
    dates = pd.to_datetime([
        "2013-04-29", "2013-04-30", "2013-04-30", "2013-04-30",
        "2013-05-01", "2013-08-30", "2013-08-31",
    ]).to_numpy()
    avail = pd.to_datetime([
        "2013-04-30", "2013-05-01", "2013-05-02", None,
        "2013-05-02", "2013-08-31", "2013-09-02",
    ]).to_numpy()
    months = pd.DatetimeIndex(dates).month.to_numpy()
    folds = list(models.rolling_folds(months, [5, 8], dates=dates, avail_dates=avail))
    assert folds[0][0].tolist() == [0, 1]
    assert folds[0][1].tolist() == [4]
    assert folds[1][1].tolist() == [5]  # August's last label cannot select September's model


@pytest.fixture
def prepared():
    dates = {
        "train": pd.to_datetime([
            "2013-01-01", "2013-01-02", "2013-03-01", "2013-04-01",
            "2013-05-01", "2013-05-02", "2013-06-01", "2013-06-02",
            "2013-07-01", "2013-07-02", "2013-08-01", "2013-08-02", "2013-08-31",
        ]).to_numpy(),
        "calib": pd.to_datetime(["2013-09-01", "2013-10-01"]).to_numpy(),
        "test": pd.to_datetime(["2013-11-01", "2013-12-01"]).to_numpy(),
    }
    avail = {s: d + np.timedelta64(1, "D") for s, d in dates.items()}
    avail["train"][-1] = np.datetime64("2013-09-02")
    y = {s: np.arange(len(d)) % 2 for s, d in dates.items()}
    Xs = {s: pd.DataFrame({"x": np.linspace(-1, 1, len(d))}) for s, d in dates.items()}
    meta = pd.DataFrame({"date": np.concatenate(list(dates.values()))})
    offsets = np.cumsum([0] + [len(d) for d in dates.values()])
    idx = {s: np.arange(offsets[i], offsets[i + 1]) for i, s in enumerate(dates)}
    labels = pipeline.Labels(y, {s: pd.DatetimeIndex(d).month.to_numpy() for s, d in dates.items()},
                             dates, avail)
    return pipeline.Prepared(pd.DataFrame(), meta, pd.concat(list(Xs.values()), ignore_index=True),
                             {"categorical": [], "schedule": ["x"]}, idx, Xs, labels)


@pytest.mark.parametrize("family", ["logreg", "xgboost"])
def test_final_model_predictions_ignore_unavailable_training_labels(prepared, monkeypatch, family):
    # Keep the real estimators and CV paths, using a compact XGB search for this test.
    if family == "xgboost":
        original = models.cv_xgb
        monkeypatch.setattr(models, "cv_xgb", lambda *args, **kw: original(
            *args, **{**kw, "n_configs": 1}))
    train = pipeline.train_logreg if family == "logreg" else pipeline.train_xgb
    first = train(prepared).model.predict_proba(prepared.Xs["test"])[:, 1]
    prepared.labels.y["train"][-1] = 1 - prepared.labels.y["train"][-1]
    second = train(prepared).model.predict_proba(prepared.Xs["test"])[:, 1]
    np.testing.assert_array_equal(first, second)
    assert not prepared.train_fit_mask[-1]


def test_prior_ignores_unavailable_training_labels(prepared):
    # The fixture has no route feature; add the shared baseline's required column.
    for X in prepared.Xs.values():
        X[pipeline.heuristic_col()] = 0.25
    first = pipeline.baseline_preds(prepared)["prior"]["test"]
    prepared.labels.y["train"][-1] = 1 - prepared.labels.y["train"][-1]
    np.testing.assert_array_equal(first, pipeline.baseline_preds(prepared)["prior"]["test"])


@pytest.fixture
def calibration_data():
    dates = pd.to_datetime([
        "2013-09-01", "2013-09-02", "2013-09-03", "2013-09-04", "2013-09-30",
        "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04", "2013-10-31",
    ]).to_numpy()
    avail = dates + np.timedelta64(1, "D")
    avail[4], avail[9] = np.datetime64("2013-10-02"), np.datetime64("2013-11-02")
    months = pd.DatetimeIndex(dates).month.to_numpy()
    p = np.array([.1, .3, .7, .9, .99, .1, .3, .7, .9, .99])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    return dates, avail, months, p, y


def test_calibrator_selection_ignores_late_fit_and_check_labels(calibration_data):
    dates, avail, months, p, y = calibration_data
    kwargs = dict(dates=dates, avail_dates=avail, selection_date="2013-11-01")
    choice1, table1 = evaluate.select_calibrator(p, y, months, 9, 10, **kwargs)
    changed = y.copy()
    changed[[4, 9]] = 1 - changed[[4, 9]]
    choice2, table2 = evaluate.select_calibrator(p, changed, months, 9, 10, **kwargs)
    assert choice1 == choice2
    pd.testing.assert_frame_equal(table1, table2)


def test_static_calibration_and_thresholds_ignore_labels_after_first_test_clock(
        calibration_data, monkeypatch):
    dates, avail, months, p, y = calibration_data
    # Isolate static calibration from the separately tested causal rolling path.
    monkeypatch.setattr(config, "ROLLING_CALIB_SENSITIVITY", [])
    test_dates = np.array(["2013-11-01", "2013-11-02"], dtype="datetime64[ns]")
    labels = pipeline.Labels(
        {"train": np.array([0, 1]), "calib": y.copy(), "test": np.array([0, 1])},
        {"train": np.array([8, 8]), "calib": months, "test": np.array([11, 11])},
        {"train": dates[:2], "calib": dates, "test": test_dates},
        {"train": avail[:2], "calib": avail, "test": test_dates + np.timedelta64(1, "D")})
    raw = {name: {"train": np.array([.2, .8]), "calib": p,
                  "test": np.array([.4, .6])}
           for name in pipeline.RAW_SCORES}
    final1, choice1, _ = pipeline.calibrate(labels, raw)
    thresholds1 = pipeline.frozen_thresholds(labels, final1)
    labels.y["calib"][-1] = 1 - labels.y["calib"][-1]
    final2, choice2, _ = pipeline.calibrate(labels, raw)
    assert choice1 == choice2
    assert thresholds1 == pipeline.frozen_thresholds(labels, final2)
    for name in final1:
        np.testing.assert_array_equal(final1[name]["test"], final2[name]["test"])


def test_sensitivity_target_uses_cancellation_notification_time():
    df = pd.DataFrame({
        "arr_known_dt": pd.to_datetime([None, "2013-08-31 21:00", None]),
        "dep_known_dt": pd.to_datetime(["2013-08-31 19:00", "2013-08-31 18:00", "2013-08-31 18:00"]),
        "is_cancelled": [1, 0, 0],
    })
    avail = data.target_available_date(df, "arr15_or_cancel")
    assert data.outcomes_available_by(avail, "2013-09-01").tolist() == [True, False, False]


def test_real_dataset_fitting_excludes_october_late_evening_outcomes():
    df = data.model_frame(data.add_target(data.load_raw()))
    cal = df[df.date.between("2013-09-01", "2013-10-31")]
    mask = data.outcomes_available_by(data.target_available_date(cal), "2013-11-01")
    clock = pd.Timestamp("2013-10-31 20:00")
    assert (~mask).sum() > 0  # positive control: boundary really excludes flights
    assert cal.loc[mask, "arr_known_dt"].le(clock).all()
    assert cal.loc[~mask, "arr_known_dt"].gt(clock).all()
