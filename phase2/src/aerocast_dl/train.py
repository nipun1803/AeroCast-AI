"""Training, early stopping, rolling-origin CV and seed ensembles.

Every network is fitted by the same procedure (docs/02_models_and_training.md):

1. inputs (encoder, sequence normaliser, wide LogReg) are fitted on the
   model's own fitting rows, through the leakage guards;
2. CV, exactly as deployed: train on the fold's rows minus their last
   ``INNER_HOLDOUT_DAYS`` days with early stopping on those days to choose
   the epoch count, then refit on ALL the fold's rows for that many epochs
   and score the validation month once. (Scoring the early-stopped model
   instead would score a model that never saw the freshest 14 days, which
   the deployed refit does see.) The only input fitted on labels, the wide
   LogReg, is refitted on the inner training rows for the early-stopping
   run (``early_stopping_inputs``), so no holdout label is seen before the
   holdout scores an epoch; ``fit`` refuses a holdout the wide part saw.
   The label-free transforms (vocabularies, medians, scalers) keep their fit
   on all the fold's rows;
3. final model: all training rows available at ``FINAL_FIT_DATE``, for a
   fixed number of epochs (the mean of the CV best epochs);
4. the deployed prediction is the mean logit over ``config.SEEDS``.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from aerocast import models as p1_models
from aerocast.pipeline import Prepared

from . import config, guards, models, sequences, tensors

torch.set_num_threads(config.N_THREADS)
# Training rule (docs/06). Added after the runs; checked to leave a Wide & Deep fit (GRU, embeddings,
# wide part) bit-identical on CPU, so no reported number depends on it (change log).
torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    """Everything that defines one trainable model (inputs + network + optimiser)."""

    name: str = "mlp"
    # inputs
    ids: tuple[str, ...] = ()               # high-cardinality IDs (ablation)
    drop_groups: tuple[str, ...] = ()       # Phase 1 feature groups removed (ablation)
    seq_names: tuple[str, ...] = ()         # GRU entity types; () = no GRU
    seq_len: int = sequences.SEQ_LEN
    wide: bool = False                      # Wide & Deep: add the LogReg as wide part
    # network
    cat_mode: str = "embed"
    emb_dim: int | None = None              # None = rule of thumb per table
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.2
    emb_dropout: float = 0.0
    seq_hidden: int = 16
    seq_dropout: float = 0.0
    # optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-4
    wide_lr_scale: float = 1.0
    early_stopping: bool = True             # False only for the regularisation ablation

    def input_key(self) -> tuple:
        return (self.ids, self.drop_groups, self.seq_names, self.seq_len, self.wide)

    def with_params(self, **kw) -> "Spec":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Inputs for one fitting set
# ---------------------------------------------------------------------------
@dataclass
class Inputs:
    cat: torch.Tensor                       # [n_rows, n_cat] long, every modelling row
    num: torch.Tensor                       # [n_rows, n_num] float
    y: torch.Tensor                         # [n_rows] float
    cardinalities: list[int]
    seq: dict[str, torch.Tensor] = field(default_factory=dict)       # [days, ents, L, C]
    seq_idx: dict[str, torch.Tensor] = field(default_factory=dict)   # [n_rows, 2]
    wide: torch.Tensor | None = None        # [n_rows, wide_dim]
    wide_coef: np.ndarray | None = None
    wide_intercept: float | None = None
    wide_rows: np.ndarray | None = None     # rows whose labels fitted the wide LogReg
    # fitted transforms, kept so that modified feature rows can be re-encoded
    # (what-if rescheduling); all were fitted on the same fitting rows
    encoder: tensors.TabularEncoder | None = None
    num_cols: list[str] | None = None
    wide_pre: object | None = None

    def batch(self, idx: torch.Tensor) -> dict:
        b = {"cat": self.cat[idx], "num": self.num[idx]}
        if self.seq:
            b["seq"] = {n: w[self.seq_idx[n][idx, 0], self.seq_idx[n][idx, 1]]
                        for n, w in self.seq.items()}
        if self.wide is not None:
            b["wide"] = self.wide[idx]
        return b


def num_cols_for(P: Prepared, spec: Spec) -> list[str]:
    drop = {c for g in spec.drop_groups for c in P.groups[g]}
    return [c for c in P.num_cols if c not in drop]


def build_inputs(P: Prepared, seqs, fit_rows: np.ndarray, fit_date, spec: Spec) -> Inputs:
    """Fit every input transform on ``fit_rows`` (guarded) and encode all rows."""
    fit_rows = np.asarray(fit_rows)
    num_cols = num_cols_for(P, spec)
    enc = tensors.fit_encoder(P, fit_rows, fit_date, ids=spec.ids, num_cols=num_cols)
    E = enc.transform(*tensors.model_inputs(P, spec.ids, num_cols))
    y = tensors.row_info(P)["y"].to_numpy(dtype=np.float32)
    inp = Inputs(torch.from_numpy(E.cat), torch.from_numpy(E.num), torch.from_numpy(y),
                 E.cardinalities, encoder=enc, num_cols=num_cols)
    if spec.seq_names:
        sub = {n: seqs[n] for n in spec.seq_names}
        norm = sequences.fit_sequence_normalizer(P, sub, fit_rows, fit_date)
        win = norm.transform(sub)
        inp.seq = {n: torch.from_numpy(np.ascontiguousarray(w[:, :, -spec.seq_len:]))
                   for n, w in win.items()}
        inp.seq_idx = {n: torch.from_numpy(v) for n, v in sequences.row_index(P, sub).items()}
    if spec.wide:
        _fit_wide(P, inp, fit_rows, fit_date, y)
    return inp


def _fit_wide(P: Prepared, inp: Inputs, rows: np.ndarray, fit_date, y: np.ndarray) -> None:
    """Fit the wide LogReg on ``rows`` (guarded) and set the wide part of ``inp``."""
    rows = np.asarray(rows)
    info = tensors.row_info(P).iloc[rows]
    guards.assert_rows_in_splits(info["split"], {"train"}, "wide LogReg fit")
    guards.assert_labels_available(info["avail"], fit_date, "wide LogReg fit")
    cols = P.cat_cols + inp.num_cols
    lr = p1_models.make_logreg(P.cat_cols, inp.num_cols, config.LOGREG_C)
    lr.fit(P.X.iloc[rows][cols], y[rows].astype(int))
    Z = lr.named_steps["pre"].transform(P.X[cols])
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    inp.wide = torch.from_numpy(Z.astype(np.float32))
    inp.wide_coef = lr.named_steps["clf"].coef_.ravel()
    inp.wide_intercept = float(lr.named_steps["clf"].intercept_[0])
    inp.wide_pre = lr.named_steps["pre"]
    inp.wide_rows = rows


def early_stopping_inputs(P: Prepared, inp: Inputs, tr: np.ndarray, fit_date) -> Inputs:
    """Inputs for the early-stopping run on the inner training rows ``tr``:
    ``inp`` with the wide LogReg refitted on ``tr`` only (the other tensors are
    shared, not copied). Without a wide part, ``inp`` itself."""
    if inp.wide is None:
        return inp
    es_inp = replace(inp)
    _fit_wide(P, es_inp, tr, fit_date, inp.y.numpy())
    return es_inp


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def make_model(spec: Spec, inp: Inputs) -> models.DelayNet:
    dims = ([spec.emb_dim] * len(inp.cardinalities) if spec.emb_dim else None)
    net = models.DelayNet(
        inp.cardinalities, inp.num.shape[1], hidden=spec.hidden, depth=spec.depth,
        dropout=spec.dropout, emb_dropout=spec.emb_dropout, emb_dims=dims,
        cat_mode=spec.cat_mode, seq_names=list(spec.seq_names) or None,
        seq_channels=len(sequences.DERIVED), seq_hidden=spec.seq_hidden,
        seq_dropout=spec.seq_dropout,
        wide_dim=inp.wide.shape[1] if inp.wide is not None else 0)
    if inp.wide is not None:
        net.init_wide(inp.wide_coef, inp.wide_intercept)
    return net


def _optimiser(net: models.DelayNet, spec: Spec) -> torch.optim.Optimizer:
    wide = [p for n, p in net.named_parameters() if n.startswith("wide.")]
    deep = [p for n, p in net.named_parameters() if not n.startswith("wide.")]
    groups = [{"params": deep, "lr": spec.lr}]
    if wide:
        groups.append({"params": wide, "lr": spec.lr * spec.wide_lr_scale})
    return torch.optim.AdamW(groups, weight_decay=spec.weight_decay)


def _seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)


def predict_logits(net: models.DelayNet, inp: Inputs, rows, batch: int = 8192) -> np.ndarray:
    net.eval()                              # dropout off; never batch statistics
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), batch):
            out.append(net(**inp.batch(rows[k:k + batch])))
    return torch.cat(out).numpy().astype(np.float64)


def _logloss(net, inp, rows) -> float:
    z = torch.from_numpy(predict_logits(net, inp, rows))
    y = inp.y[torch.as_tensor(rows)].double()
    return float(torch.nn.functional.binary_cross_entropy_with_logits(z, y))


def fit(spec: Spec, inp: Inputs, train_rows, seed: int, es_rows=None,
        epochs: int | None = None) -> tuple[models.DelayNet, dict]:
    """Train on ``train_rows``. With ``es_rows``: early stopping on their log
    loss (best epoch restored). Without: exactly ``epochs`` epochs."""
    if (es_rows is None) == (epochs is None):
        raise ValueError("give either es_rows (early stopping) or epochs (fixed)")
    if es_rows is not None and inp.wide_rows is not None and \
            np.intersect1d(inp.wide_rows, np.asarray(es_rows)).size:
        raise guards.LeakageError("early-stopping rows were used to fit the wide LogReg "
                                  "(use early_stopping_inputs)")
    gen = _seed(seed)
    net = make_model(spec, inp)
    opt = _optimiser(net, spec)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tr = torch.as_tensor(np.asarray(train_rows), dtype=torch.long)
    n_epochs = epochs or config.MAX_EPOCHS
    best, best_state, best_epoch, stall, hist = np.inf, None, 0, 0, []
    for epoch in range(1, n_epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen)]   # row order only; splits stay chronological
        total = 0.0
        for k in range(0, len(perm), config.BATCH_SIZE):
            idx = perm[k:k + config.BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(**inp.batch(idx)), inp.y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        rec = {"epoch": epoch, "train_logloss": total / len(perm)}
        if es_rows is not None:
            rec["es_logloss"] = _logloss(net, inp, es_rows)
            if rec["es_logloss"] < best - 1e-5:
                best, best_epoch, stall = rec["es_logloss"], epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stall += 1
        hist.append(rec)
        if es_rows is not None and spec.early_stopping and stall >= config.PATIENCE:
            break
    if es_rows is not None and spec.early_stopping:
        net.load_state_dict(best_state)
    else:
        best_epoch = len(hist)
    return net, {"best_epoch": best_epoch, "history": hist}


# ---------------------------------------------------------------------------
# Rolling-origin CV (Phase 1 folds)
# ---------------------------------------------------------------------------
def folds(P: Prepared):
    """(fold, fit rows, validation rows, fit date) on modelling-row positions,
    from the Phase 1 label-availability-filtered rolling folds."""
    tr_pos = P.idx["train"]
    dates = P.labels.dates["train"]
    for k, (tr, va) in enumerate(p1_models.rolling_folds(P.labels.months["train"], **P.cv_timing)):
        fit_date = pd.Timestamp(dates[va].min()).normalize()
        yield k, tr_pos[tr], tr_pos[va], fit_date


def inner_split(P: Prepared, rows: np.ndarray, fit_date) -> tuple[np.ndarray, np.ndarray]:
    """(training part, early-stopping part): the last INNER_HOLDOUT_DAYS days
    before ``fit_date`` stop the training; the validation month is never used."""
    d = pd.to_datetime(P.meta["date"].iloc[rows]).to_numpy()
    cut = np.datetime64(pd.Timestamp(fit_date) - pd.Timedelta(days=config.INNER_HOLDOUT_DAYS))
    es = d >= cut
    return rows[~es], rows[es]


class InputCache:
    """Inputs depend on (fold, input options) only, not on network settings."""

    def __init__(self, P: Prepared, seqs):
        self.P, self.seqs, self._c = P, seqs, {}

    def get(self, key, fit_rows, fit_date, spec: Spec) -> Inputs:
        k = (key, spec.input_key())
        if k not in self._c:
            self._c[k] = build_inputs(self.P, self.seqs, fit_rows, fit_date, spec)
        return self._c[k]

    def get_es(self, key, fit_rows, fit_date, spec: Spec) -> tuple[Inputs, Inputs, np.ndarray, np.ndarray]:
        """(inputs, early-stopping inputs, inner training rows, inner holdout
        rows) for one fold: see ``early_stopping_inputs``."""
        inp = self.get(key, fit_rows, fit_date, spec)
        tr, es = inner_split(self.P, np.asarray(fit_rows), fit_date)
        k = (key, spec.input_key(), "early_stopping")
        if k not in self._c:
            self._c[k] = early_stopping_inputs(self.P, inp, tr, fit_date)
        return inp, self._c[k], tr, es


def refit_as_deployed(spec: Spec, inp: Inputs, fit_rows, tr, es, seed: int,
                      es_inp: Inputs | None = None):
    """Choose the epoch count on the inner holdout (with ``es_inp``, whose
    wide part never saw it), then refit on all ``fit_rows`` for that many
    epochs with ``inp`` (the deployed procedure)."""
    _, info = fit(spec, es_inp if es_inp is not None else inp, tr, seed, es_rows=es)
    net, _ = fit(spec, inp, fit_rows, seed, epochs=info["best_epoch"])
    return net, info


def cv(P: Prepared, spec: Spec, seeds, cache: InputCache, keep_logits: bool = False) -> dict:
    """Fold x seed results and the seed-ensemble ("as deployed") AP per fold."""
    rows, ens, logits = [], [], {}
    for k, fit_rows, va, fit_date in folds(P):
        inp, es_inp, tr, es = cache.get_es(("fold", k), fit_rows, fit_date, spec)
        y_va = inp.y[torch.as_tensor(va)].numpy()
        zs = []
        for s in seeds:
            net, info = refit_as_deployed(spec, inp, fit_rows, tr, es, s, es_inp)
            z = predict_logits(net, inp, va)
            zs.append(z)
            rows.append({"fold": k, "val_month": int(P.meta["month"].iloc[va[0]]), "seed": s,
                         "val_pr_auc": average_precision_score(y_va, z),
                         "best_epoch": info["best_epoch"],
                         "es_logloss": min(h["es_logloss"] for h in info["history"])})
        zbar = np.mean(zs, axis=0)
        ens.append(average_precision_score(y_va, zbar))
        if keep_logits:
            logits[k] = (va, zbar)
    per_seed = pd.DataFrame(rows)
    return {"spec": asdict(spec), "per_seed": per_seed, "ensemble_folds": ens,
            "ensemble_mean": float(np.mean(ens)), "seed_mean": float(per_seed["val_pr_auc"].mean()),
            "mean_best_epoch": float(per_seed["best_epoch"].mean()), "logits": logits}


def sample_configs(grid: dict, n: int, seed: int = config.SEARCH_SEED) -> list[dict]:
    """``n`` distinct random configurations of ``grid`` (all if fewer exist)."""
    keys = list(grid)
    allc = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    if n >= len(allc):
        return allc
    pick = np.random.default_rng(seed).choice(len(allc), n, replace=False)
    return [allc[i] for i in sorted(pick)]


def search(P: Prepared, base: Spec, grid: dict, n: int, cache: InputCache, log=print) -> pd.DataFrame:
    """Single-seed CV over ``n`` sampled configurations of ``grid``."""
    out = []
    for i, params in enumerate(sample_configs(grid, n)):
        r = cv(P, base.with_params(**params), [config.SEARCH_SEED], cache)
        out.append({"config": i, **params, "cv_pr_auc": r["seed_mean"],
                    **{f"fold_m{m}": v for m, v in zip([5, 6, 7, 8], r["ensemble_folds"])},
                    "mean_best_epoch": r["mean_best_epoch"]})
        log(f"  {base.name} config {i + 1}/{n}: {params} -> CV AP {r['seed_mean']:.4f}")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------
def fit_final(P: Prepared, spec: Spec, epochs: int, seeds, cache: InputCache):
    """Seed ensemble fitted on all training rows available at FINAL_FIT_DATE."""
    rows = tensors.final_fit_rows(P)
    inp = cache.get("final", rows, config.FINAL_FIT_DATE, spec)
    nets = [fit(spec, inp, rows, s, epochs=epochs)[0] for s in seeds]
    return nets, inp


def ensemble_logits(nets, inp: Inputs, rows) -> np.ndarray:
    return np.mean([predict_logits(n, inp, rows) for n in nets], axis=0)
