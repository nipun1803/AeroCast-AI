"""Training-loop sanity and leakage tests (Steps 3 and 5)."""
import numpy as np
import pandas as pd
import pytest
import torch

from aerocast_dl import config, guards, train


def _toy(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    cat = torch.stack([torch.randint(0, 5, (n,), generator=g),
                       torch.randint(0, 3, (n,), generator=g)], 1)
    num = torch.randn(n, 4, generator=g)
    y = ((num[:, 0] + (cat[:, 0] == 2).float() + 0.3 * torch.randn(n, generator=g)) > 0.5).float()
    return train.Inputs(cat, num, y, [5, 3])


# ---------------------------------------------------------------------------
# Training loop (synthetic, fast)
# ---------------------------------------------------------------------------
def test_overfit_one_batch():
    """A network that cannot overfit 64 rows has a bug (Karpathy's check)."""
    inp = _toy()
    spec = train.Spec(hidden=64, dropout=0.0, lr=3e-3, weight_decay=0.0)
    rows = np.arange(64)
    net, info = train.fit(spec, inp, rows, seed=0, epochs=300)
    assert info["history"][-1]["train_logloss"] < 0.05


def test_same_seed_same_model():
    inp, spec, rows = _toy(), train.Spec(hidden=32), np.arange(1500)
    a = train.predict_logits(train.fit(spec, inp, rows, seed=7, epochs=2)[0], inp, rows)
    b = train.predict_logits(train.fit(spec, inp, rows, seed=7, epochs=2)[0], inp, rows)
    np.testing.assert_array_equal(a, b)


def test_eval_mode():
    """Validation/test passes run in eval mode: dropout off, so repeated
    predictions are identical and the model is left in eval mode."""
    inp = _toy()
    net, _ = train.fit(train.Spec(hidden=32, dropout=0.5), inp, np.arange(1000), seed=0, epochs=1)
    net.train()
    a = train.predict_logits(net, inp, np.arange(1000, 1500))
    assert not net.training
    np.testing.assert_array_equal(a, train.predict_logits(net, inp, np.arange(1000, 1500)))


def test_early_stopping_restores_best_epoch():
    inp = _toy()
    net, info = train.fit(train.Spec(hidden=64, dropout=0.0, lr=1e-2), inp, np.arange(1500),
                          seed=0, es_rows=np.arange(1500, 2000))
    h = info["history"]
    best = min(range(len(h)), key=lambda i: h[i]["es_logloss"])
    assert info["best_epoch"] == best + 1
    assert train._logloss(net, inp, np.arange(1500, 2000)) == pytest.approx(h[best]["es_logloss"], abs=1e-6)


def test_fit_needs_one_stopping_rule():
    with pytest.raises(ValueError):
        train.fit(train.Spec(), _toy(), np.arange(10), seed=0)


def test_sample_configs_fixed_by_seed():
    a = train.sample_configs(config.MLP_GRID, config.MLP_N_CONFIGS)
    assert a == train.sample_configs(config.MLP_GRID, config.MLP_N_CONFIGS)
    assert len(a) == config.MLP_N_CONFIGS == len({tuple(c.items()) for c in a})


# ---------------------------------------------------------------------------
# Folds, inner holdout and the wide part (real data)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_early_stopping_rows(P):
    """Early-stopping rows are the last INNER_HOLDOUT_DAYS days of fold-train:
    inside the fold's fitting rows, before the validation month, never in it."""
    for k, fit_rows, va, fit_date in train.folds(P):
        tr, es = train.inner_split(P, fit_rows, fit_date)
        assert len(tr) and len(es)
        assert set(tr).isdisjoint(es) and set(tr) | set(es) == set(fit_rows)
        assert set(es).isdisjoint(va) and set(tr).isdisjoint(va)
        d_es = pd.to_datetime(P.meta["date"].iloc[es])
        assert d_es.max() < fit_date <= pd.to_datetime(P.meta["date"].iloc[va]).min()
        assert d_es.min() >= fit_date - pd.Timedelta(days=config.INNER_HOLDOUT_DAYS)
        assert pd.to_datetime(P.meta["date"].iloc[tr]).max() < d_es.min()


@pytest.mark.slow
def test_wide_init_fold_only(P):
    """The wide part starts as the LogReg fitted on the fold's rows: before
    training, the hybrid's logit equals that LogReg's decision function, and
    a LogReg fitted on calibration rows is refused."""
    from aerocast import models as p1_models
    k, fit_rows, va, fit_date = next(train.folds(P))
    spec = train.Spec(wide=True)
    inp = train.build_inputs(P, {}, fit_rows, fit_date, spec)
    net = train.make_model(spec, inp)
    lr = p1_models.make_logreg(P.cat_cols, P.num_cols, config.LOGREG_C)
    lr.fit(P.X.iloc[fit_rows][P.feat_cols], inp.y[torch.as_tensor(fit_rows)].numpy().astype(int))
    np.testing.assert_allclose(train.predict_logits(net, inp, va[:2000]),
                               lr.decision_function(P.X.iloc[va[:2000]][P.feat_cols]),
                               rtol=1e-4, atol=1e-4)
    with pytest.raises(guards.LeakageError):
        train.build_inputs(P, {}, P.idx["calib"][:500], config.FINAL_FIT_DATE, spec)


@pytest.mark.slow
def test_wide_never_sees_the_inner_holdout(P):
    """Flipping every inner-holdout label leaves the early-stopping run's wide
    part unchanged (it is fitted on the inner training rows only), while the
    refit's wide part, fitted on all fold rows, does change (positive
    control). Early stopping with a wide part that saw the holdout is refused."""
    k, fit_rows, va, fit_date = next(train.folds(P))
    spec = train.Spec(wide=True)
    tr, es = train.inner_split(P, fit_rows, fit_date)
    pos = {int(r): i for i, r in enumerate(P.idx["train"])}
    flip = np.array([pos[int(r)] for r in es])

    def wide_parts():
        inp = train.build_inputs(P, {}, fit_rows, fit_date, spec)
        es_inp = train.early_stopping_inputs(P, inp, tr, fit_date)
        return inp, es_inp

    inp_a, es_a = wide_parts()
    P.labels.y["train"][flip] = 1 - P.labels.y["train"][flip]
    try:
        inp_b, es_b = wide_parts()
    finally:
        P.labels.y["train"][flip] = 1 - P.labels.y["train"][flip]
    np.testing.assert_array_equal(es_a.wide_coef, es_b.wide_coef)
    assert es_a.wide_intercept == es_b.wide_intercept
    assert np.abs(inp_a.wide_coef - inp_b.wide_coef).max() > 1e-3
    assert set(es_a.wide_rows.tolist()) == set(tr.tolist())
    assert es_a.cat is inp_a.cat and es_a.num is inp_a.num      # only the wide part is refitted
    with pytest.raises(guards.LeakageError):
        train.fit(spec, inp_a, tr, seed=0, es_rows=es)


def test_embedding_transfer_low_card():
    """Only carrier / origin / destination embeddings reach XGBoost; tail and
    flight embeddings (label-memorising for rare IDs) are never transferred."""
    from aerocast_dl import models, pipeline
    cols = ["carrier", "origin", "dest", "tailnum", "carrier_flight"]
    cards = [16, 4, 94, 1569, 1394]
    net = models.DelayNet(cards, n_num=3)
    n = 50
    inp = train.Inputs(torch.stack([torch.randint(0, c, (n,)) for c in cards], 1),
                       torch.randn(n, 3), torch.zeros(n), cards)
    emb = pipeline.embedding_features(net, inp, cols)
    dims = [net.embs[k].embedding_dim for k in range(5)]
    assert emb.shape == (n, sum(dims[:3]))
    np.testing.assert_allclose(emb[:, :dims[0]], net.embs[0].weight[inp.cat[:, 0]].detach().numpy())


@pytest.mark.slow
def test_cv_scores_the_deployed_refit(P, monkeypatch):
    """CV fits twice per fold and seed: early stopping on the inner holdout,
    then a refit on ALL fold rows for the chosen epochs; the validation month
    is never a fitting or stopping row."""
    calls = []

    class Zero(torch.nn.Module):
        def forward(self, cat, num, **kw):
            return torch.zeros(len(cat))

    def fake_fit(spec, inp, rows, seed, es_rows=None, epochs=None):
        calls.append((set(np.asarray(rows).tolist()), None if es_rows is None else set(es_rows.tolist()), epochs))
        return Zero(), {"best_epoch": 3, "history": [{"es_logloss": 0.5}]}

    monkeypatch.setattr(train, "fit", fake_fit)
    cache = train.InputCache(P, {})
    train.cv(P, train.Spec(), [42], cache)
    folds = list(train.folds(P))
    assert len(calls) == 2 * len(folds)
    for (k, fit_rows, va, fd), es_call, refit_call in zip(folds, calls[::2], calls[1::2]):
        tr, es = train.inner_split(P, fit_rows, fd)
        assert es_call[0] == set(tr) and es_call[1] == set(es)
        assert refit_call[0] == set(fit_rows) and refit_call[1] is None and refit_call[2] == 3
        assert refit_call[0].isdisjoint(va)
