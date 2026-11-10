"""Step 2 leakage and alignment tests (docs/leakage_safeguards.md)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from aerocast import data as p1_data
from aerocast_dl import config, guards, tensors


# ---------------------------------------------------------------------------
# Unit tests (synthetic, fast)
# ---------------------------------------------------------------------------
def test_vocab_rare_unseen_missing_map_to_unk():
    fit = pd.Series(["A"] * 5 + ["B"] * 2)
    vocab = tensors.CategoryVocab(min_freq=3).fit(fit)
    assert vocab.cardinality == 2                      # A + UNK
    out = vocab.transform(pd.Series(["A", "B", "C", None]))
    assert out.tolist() == [1, tensors.UNK, tensors.UNK, tensors.UNK]


def test_guard_rejects_unavailable_labels():
    avail = pd.to_datetime(["2013-08-30", "2013-09-02"]).to_numpy()
    with pytest.raises(guards.LeakageError):
        guards.assert_labels_available(avail, "2013-09-01")
    guards.assert_labels_available(avail[:1], "2013-09-01")


def test_guard_treats_unknown_availability_as_unavailable():
    with pytest.raises(guards.LeakageError):
        guards.assert_labels_available(np.array(["NaT"], dtype="datetime64[ns]"), "2013-09-01")


def test_guard_rejects_forbidden_split():
    with pytest.raises(guards.LeakageError):
        guards.assert_rows_in_splits(["train", "calib"], {"train"})
    guards.assert_rows_in_splits(["train", "train"], {"train"})


# ---------------------------------------------------------------------------
# Real-data tests (build the Phase 1 feature frame once per session)
# ---------------------------------------------------------------------------
pytestmark_slow = pytest.mark.slow


@pytestmark_slow
def test_tensor_alignment(P):
    """Encoded rows match the Phase 1 frame 1:1 (row order, labels, values)."""
    rows = tensors.final_fit_rows(P)
    enc = tensors.fit_encoder(P, rows, config.FINAL_FIT_DATE)
    cats, nums = tensors.model_inputs(P)
    E = enc.transform(cats, nums)
    assert len(E) == len(P.X) == len(P.meta)

    info = tensors.row_info(P)
    for s, i in P.idx.items():
        assert (info["split"].iloc[i] == s).all()
        np.testing.assert_array_equal(info["y"].iloc[i].to_numpy(), P.labels.y[s])
    assert (info["y"] >= 0).all()

    # numeric: undoing the scaler gives back the Phase 1 values where present
    unscaled = enc.num_.named_steps["scale"].inverse_transform(E.num.astype(np.float64))
    orig = nums.to_numpy(dtype=float)
    present = ~np.isnan(orig)
    np.testing.assert_allclose(unscaled[:, : orig.shape[1]][present], orig[present],
                               rtol=1e-4, atol=1e-4)

    # categorical: every embedded index decodes to the Phase 1 value
    for k, c in enumerate(E.cat_cols):
        inv = {v: s for s, v in enc.vocabs_[c].index_.items()}
        known = E.cat[:, k] != tensors.UNK
        decoded = pd.Series(E.cat[known, k]).map(inv).to_numpy()
        np.testing.assert_array_equal(decoded, cats[c].astype(str).to_numpy()[known])


@pytestmark_slow
def test_final_fit_rows_respect_clock(P):
    """Final rows = training rows with labels known for 2013-09-01 (the
    Phase 1 fitting clock excludes 167 late-August rows)."""
    rows = tensors.final_fit_rows(P)
    info = tensors.row_info(P).iloc[rows]
    assert (info["split"] == "train").all()
    assert p1_data.outcomes_available_by(info["avail"], config.FINAL_FIT_DATE).all()
    assert int(P.train_fit_mask.sum()) == len(rows)
    assert len(P.idx["train"]) - len(rows) == 167


@pytestmark_slow
def test_fit_stats_train_only(P):
    """Scrambling every calibration/test row leaves the fitted vocabularies,
    imputation medians and scaler unchanged."""
    rows = tensors.final_fit_rows(P)
    ids = config.ID_COLS
    a = tensors.fit_encoder(P, rows, config.FINAL_FIT_DATE, ids=ids)

    later = np.concatenate([P.idx["calib"], P.idx["test"]])
    rng = np.random.default_rng(0)
    X2, meta2 = P.X.copy(), P.meta.copy()
    for c in P.num_cols:
        X2[c] = X2[c].astype(float)
        X2.loc[later, c] = rng.normal(1e3, 1e3, len(later))
    for c in P.cat_cols:
        X2[c] = X2[c].astype(object)
        X2.loc[later, c] = "ZZZ"
    for c in ["tailnum", "carrier", "origin"]:
        meta2.loc[later, c] = "ZZZ"
    meta2.loc[later, "flight"] = 99999
    b = tensors.fit_encoder(replace(P, X=X2, meta=meta2), rows, config.FINAL_FIT_DATE, ids=ids)

    for c in a.vocabs_:
        assert a.vocabs_[c].index_ == b.vocabs_[c].index_, c
    for step, attrs in {"impute": ["statistics_"], "scale": ["mean_", "scale_"]}.items():
        for attr in attrs:
            np.testing.assert_array_equal(getattr(a.num_.named_steps[step], attr),
                                          getattr(b.num_.named_steps[step], attr))


@pytestmark_slow
def test_fit_encoder_rejects_forbidden_rows(P):
    with pytest.raises(guards.LeakageError):     # calibration rows (G4)
        tensors.fit_encoder(P, np.concatenate([P.idx["train"][:10], P.idx["calib"][:10]]),
                            config.FINAL_FIT_DATE)
    late = np.setdiff1d(P.idx["train"], tensors.final_fit_rows(P))
    with pytest.raises(guards.LeakageError):     # labels unknown at the clock (G2)
        tensors.fit_encoder(P, late, config.FINAL_FIT_DATE)


@pytestmark_slow
def test_no_missing_tail_in_target(P):
    """A blank tail number occurs only for cancelled flights, which the main
    target excludes, so 'missing tail' can never reveal the label."""
    assert P.meta["tailnum"].notna().all()
    blank = P.raw["tailnum"].isna()
    assert (P.raw.loc[blank, "is_cancelled"] == 1).all()


@pytestmark_slow
def test_ids_ignore_outcome_columns(P):
    """ID inputs are built from schedule fields only: scrambling every
    post-departure column leaves them unchanged."""
    meta2 = P.meta.copy()
    for c in p1_data.POST_DEPARTURE_COLS:
        meta2[c] = np.random.default_rng(1).permutation(meta2[c].to_numpy())
    pd.testing.assert_frame_equal(tensors.id_columns(P.meta), tensors.id_columns(meta2))


@pytestmark_slow
def test_dataset_shapes(P):
    rows = tensors.final_fit_rows(P)[:256]
    enc = tensors.fit_encoder(P, tensors.final_fit_rows(P), config.FINAL_FIT_DATE)
    cats, nums = tensors.model_inputs(P)
    E = enc.transform(cats.iloc[rows], nums.iloc[rows])
    ds = E.dataset(tensors.row_info(P)["y"].iloc[rows])
    cat, num, y = ds[0]
    assert cat.shape == (len(E.cat_cols),) and num.shape == (len(E.num_cols),)
    assert len(ds) == 256 and set(np.unique(ds.tensors[2].numpy())) <= {0.0, 1.0}
