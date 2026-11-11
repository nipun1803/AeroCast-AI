"""Phase 2 pipeline stages (model search, seed ensembles, hybrids, ablations).

Each function is one stage of ``scripts/run_phase2.py``. Every score here is
rolling-origin CV inside the training window (May-Aug folds); no stage in this
module reads calibration or test labels.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from aerocast import config as p1
from aerocast.pipeline import Prepared

from . import config, sequences, train

ALL_SEQ = tuple(sequences.ENTITY_SPECS)
PHASE1_RESULTS = config.ROOT.parent / "phase1" / "results"
FOLD_COLS = [f"fold_m{m}" for m in p1.CV_VAL_MONTHS]


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def n_params(P: Prepared, spec: train.Spec, cache: train.InputCache) -> int:
    k, fit_rows, _, fit_date = list(train.folds(P))[-1]
    net = train.make_model(spec, cache.get(("fold", k), fit_rows, fit_date, spec))
    return int(sum(p.numel() for p in net.parameters()))


def cv_row(name: str, res: dict, extra: dict | None = None) -> dict:
    ps = res["per_seed"]
    return {"model": name, "cv_pr_auc_mean": res["ensemble_mean"],
            "cv_pr_auc_std": float(np.std(res["ensemble_folds"], ddof=1)),
            **dict(zip(FOLD_COLS, res["ensemble_folds"])),
            "seed_mean": res["seed_mean"],
            "seed_sd_mean_over_folds": float(ps.groupby("fold")["val_pr_auc"].std().mean()),
            "mean_best_epoch": res["mean_best_epoch"], **(extra or {})}


def save_spec(name: str, spec: train.Spec) -> None:
    path = config.RESULTS_DIR / "specs.json"
    specs = json.loads(path.read_text()) if path.exists() else {}
    specs[name] = asdict(spec)
    path.write_text(json.dumps(specs, indent=2))


def load_spec(name: str) -> train.Spec:
    d = json.loads((config.RESULTS_DIR / "specs.json").read_text())[name]
    return train.Spec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def append_cv(row: dict, per_seed: pd.DataFrame) -> None:
    """Upsert one model's CV summary (cv_models.csv) and its fold x seed rows."""
    path = config.RESULTS_DIR / "cv_models.csv"
    tbl = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(tbl):
        tbl = tbl[tbl["model"] != row["model"]]
    pd.concat([tbl, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    ps_path = config.RESULTS_DIR / "cv_seeds.csv"
    ps = pd.read_csv(ps_path) if ps_path.exists() else pd.DataFrame()
    if len(ps):
        ps = ps[ps["model"] != row["model"]]
    pd.concat([ps, per_seed.assign(model=row["model"])], ignore_index=True).to_csv(ps_path, index=False)


def phase1_cv() -> pd.DataFrame:
    """Phase 1 rolling-CV AP as deployed (LogReg; XGBoost at fixed trees)."""
    t = pd.read_csv(PHASE1_RESULTS / "cv_summary.csv").set_index("variant")
    out = []
    for key, name in [("logreg", "Phase 1 LogReg"), ("xgboost", "Phase 1 XGBoost")]:
        out.append({"model": name, "cv_pr_auc_mean": t.loc[key, "cv_pr_auc_mean"],
                    "cv_pr_auc_std": t.loc[key, "cv_pr_auc_std"],
                    **{c: t.loc[key, c] for c in FOLD_COLS}})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Families: search (one seed) -> best configuration -> 5-seed ensemble
# ---------------------------------------------------------------------------
def run_family(P, cache, name: str, base: train.Spec, grid: dict, n: int, log) -> train.Spec:
    log(f"search {name}: {n} configurations, seed {config.SEARCH_SEED}")
    tbl = train.search(P, base, grid, n, cache, log)
    tbl.to_csv(config.RESULTS_DIR / f"search_{name}.csv", index=False)
    best = tbl.loc[tbl["cv_pr_auc"].idxmax()]
    spec = base.with_params(name=name, **{k: _cast(best[k], grid[k]) for k in grid})
    log(f"best {name}: {[(k, getattr(spec, k)) for k in grid]} (single-seed CV {best['cv_pr_auc']:.4f})")
    score_seeds(P, cache, name, spec, log)
    save_spec(name, spec)
    return spec


def score_seeds(P, cache, name, spec, log, extra=None) -> dict:
    res = train.cv(P, spec, config.SEEDS, cache)
    row = cv_row(name, res, {"n_params": n_params(P, spec, cache), **(extra or {})})
    append_cv(row, res["per_seed"])
    log(f"  {name}: 5-seed ensemble CV AP {res['ensemble_mean']:.4f} "
        f"(folds {np.round(res['ensemble_folds'], 4).tolist()})")
    return res


def _cast(v, options):
    t = type(options[0])
    return t(v)


def stage_mlp(P, cache, log) -> train.Spec:
    return run_family(P, cache, "mlp", train.Spec(name="mlp"), config.MLP_GRID,
                      config.MLP_N_CONFIGS, log)


def stage_gru(P, cache, log) -> train.Spec:
    base = load_spec("mlp").with_params(name="gru", seq_names=ALL_SEQ)
    return run_family(P, cache, "gru", base, config.GRU_GRID, config.GRU_N_CONFIGS, log)


def better_deep() -> str:
    t = pd.read_csv(config.RESULTS_DIR / "cv_models.csv").set_index("model")
    return "gru" if t.loc["gru", "cv_pr_auc_mean"] > t.loc["mlp", "cv_pr_auc_mean"] else "mlp"


def stage_hybrid(P, cache, log) -> train.Spec:
    deep = better_deep()
    log(f"Wide & Deep: deep part = {deep} (higher 5-seed CV AP)")
    base = load_spec(deep).with_params(name="wide_deep", wide=True)
    return run_family(P, cache, "wide_deep", base, config.HYBRID_GRID, config.HYBRID_N_CONFIGS, log)


# ---------------------------------------------------------------------------
# Secondary hybrid: MLP embeddings -> Phase 1 XGBoost (separate process)
# ---------------------------------------------------------------------------
EMB_COLS = ("carrier", "origin", "dest")      # low-cardinality only (docs/01_inputs_and_leakage.md)


def embedding_features(net, inp: train.Inputs, cat_cols: list[str]) -> np.ndarray:
    """Learned carrier / origin / destination vectors for every row."""
    parts = []
    with torch.no_grad():
        for k, c in enumerate(cat_cols):
            if c in EMB_COLS:
                parts.append(net.embs[k].weight[inp.cat[:, k]].numpy())
    return np.concatenate(parts, axis=1).astype(np.float32)


def xgb_phase1_params() -> dict:
    return json.loads((PHASE1_RESULTS / "summary.json").read_text())["xgb_params"]


def run_xgb_jobs(jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """[(embedding features, fit rows, predict rows)] -> XGBoost probabilities."""
    with tempfile.TemporaryDirectory(dir=config.ROOT) as d:
        payload = {"n_jobs": len(jobs), "params": json.dumps(xgb_phase1_params())}
        for j, (emb, fit_rows, pred_rows) in enumerate(jobs):
            payload |= {f"emb_{j}": emb, f"fit_{j}": fit_rows, f"pred_{j}": pred_rows}
        np.savez(Path(d) / "jobs.npz", **payload)
        subprocess.run([sys.executable, "-B", str(config.ROOT / "scripts" / "xgb_worker.py"),
                        str(Path(d) / "jobs.npz"), str(Path(d) / "out.npz")], check=True)
        out = np.load(Path(d) / "out.npz")
        return [out[f"p_{j}"] for j in range(len(jobs))]


def stage_emb_xgb(P, cache, log) -> None:
    """Per fold and seed: train the MLP as deployed (epochs from the inner
    holdout, refit on all fold rows), take its embeddings, fit Phase 1 XGBoost on fold-train with
    them, score the validation month; ensemble = mean probability over seeds."""
    spec = load_spec("mlp")
    jobs, meta = [], []
    for k, fit_rows, va, fit_date in train.folds(P):
        inp = cache.get(("fold", k), fit_rows, fit_date, spec)
        tr, es = train.inner_split(P, fit_rows, fit_date)
        for s in config.SEEDS:
            net, _ = train.refit_as_deployed(spec, inp, fit_rows, tr, es, s)
            jobs.append((embedding_features(net, inp, P.cat_cols), fit_rows, va))
            meta.append((k, s, va))
    log(f"emb->XGBoost: {len(jobs)} XGBoost fits in a separate process")
    probs = run_xgb_jobs(jobs)
    y = train.tensors.row_info(P)["y"].to_numpy()
    rows, ens = [], []
    for k in sorted({m[0] for m in meta}):
        ps = [(s, va, p) for (kk, s, va), p in zip(meta, probs) if kk == k]
        va = ps[0][1]
        for s, _, p in ps:
            rows.append({"fold": k, "val_month": int(P.meta["month"].iloc[va[0]]), "seed": s,
                         "val_pr_auc": average_precision_score(y[va], p),
                         "best_epoch": np.nan, "es_logloss": np.nan})
        ens.append(average_precision_score(y[va], np.mean([p for _, _, p in ps], axis=0)))
    per_seed = pd.DataFrame(rows)
    res = {"per_seed": per_seed, "ensemble_folds": ens, "ensemble_mean": float(np.mean(ens)),
           "seed_mean": float(per_seed["val_pr_auc"].mean()), "mean_best_epoch": np.nan}
    append_cv(cv_row("emb_xgb", res, {"n_params": np.nan}), per_seed)
    log(f"  emb_xgb: 5-seed ensemble CV AP {res['ensemble_mean']:.4f} "
        f"(folds {np.round(ens, 4).tolist()})")


# ---------------------------------------------------------------------------
# Ablations (CV only, 5 seeds)
# ---------------------------------------------------------------------------
def ablation_specs() -> list[tuple[str, str, train.Spec]]:
    """(ablation name, question, spec) built from the selected configurations."""
    mlp, gru, wd = load_spec("mlp"), load_spec("gru"), load_spec("wide_deep")
    out = [
        ("mlp_onehot", "embeddings vs one-hot", mlp.with_params(cat_mode="onehot")),
        *[(f"mlp_emb{d}", "embedding width", mlp.with_params(emb_dim=d)) for d in (2, 4, 8, 16)],
        ("mlp_ids", "+ tail, flight and origin-hour IDs", mlp.with_params(ids=tuple(config.ID_COLS))),
        ("mlp_no_reg", "no dropout or weight decay",
         mlp.with_params(dropout=0.0, emb_dropout=0.0, weight_decay=0.0)),
        ("mlp_no_es", "no early stopping (all epochs)", mlp.with_params(early_stopping=False)),
        *[(f"mlp_minus_{g}", "feature group removed", mlp.with_params(drop_groups=(g,)))
          for g in ("schedule", "congestion", "history_entity", "history_day")],
        ("gru_instead_of_history", "GRU replaces the hand-crafted history",
         gru.with_params(drop_groups=("history_entity", "history_day"))),
        *[(f"gru_L{L}", "sequence length", gru.with_params(seq_len=L)) for L in (7, 14)],
        ("wd_no_gru", "Wide & Deep without the GRU", wd.with_params(seq_names=())),
        ("wd_onehot", "Wide & Deep without embeddings", wd.with_params(cat_mode="onehot")),
    ]
    if not wd.seq_names:                     # deep part was the MLP: nothing to remove
        out = [o for o in out if o[0] != "wd_no_gru"]
    return out


def stage_ablation(P, cache, log, only: list[str] | None = None) -> None:
    for name, question, spec in ablation_specs():
        if only and name not in only:
            continue
        score_seeds(P, cache, f"ablation:{name}", spec.with_params(name=name), log,
                    {"question": question})


# ---------------------------------------------------------------------------
# Summaries and figures (read results/, no new scores except the curve run)
# ---------------------------------------------------------------------------
ABLATION_BASE = {"mlp": "mlp", "gru": "gru", "wd": "wide_deep"}
ABLATION_LABEL = {
    "mlp_onehot": "MLP: one-hot instead of embeddings",
    "mlp_emb2": "MLP: embedding width 2", "mlp_emb4": "MLP: embedding width 4",
    "mlp_emb8": "MLP: embedding width 8", "mlp_emb16": "MLP: embedding width 16",
    "mlp_ids": "MLP: + tail / flight / origin-hour IDs",
    "mlp_no_reg": "MLP: no dropout or weight decay", "mlp_no_es": "MLP: no early stopping",
    "mlp_minus_schedule": "MLP: − schedule features", "mlp_minus_congestion": "MLP: − congestion features",
    "mlp_minus_history_entity": "MLP: − entity history", "mlp_minus_history_day": "MLP: − day-level history",
    "gru_instead_of_history": "GRU: replaces hand-crafted history", "gru_L7": "GRU: window 7 days",
    "gru_L14": "GRU: window 14 days", "wd_no_gru": "Wide & Deep: without GRU",
    "wd_onehot": "Wide & Deep: without embeddings",
}


def ablation_table() -> pd.DataFrame:
    """Each ablation against the configuration it changes: Δ mean CV AP,
    folds improved, and the mean seed SD (differences inside it are noise)."""
    cv = pd.read_csv(config.RESULTS_DIR / "cv_models.csv").set_index("model")
    rows = []
    for m in [m for m in cv.index if m.startswith("ablation:")]:
        name = m.split(":", 1)[1]
        base = ABLATION_BASE[name.split("_", 1)[0]]
        d = cv.loc[m, FOLD_COLS].astype(float).to_numpy() - cv.loc[base, FOLD_COLS].astype(float).to_numpy()
        rows.append({"ablation": name, "label": ABLATION_LABEL.get(name, name),
                     "question": cv.loc[m, "question"], "base": base,
                     "base_cv": cv.loc[base, "cv_pr_auc_mean"], "cv": cv.loc[m, "cv_pr_auc_mean"],
                     "delta": cv.loc[m, "cv_pr_auc_mean"] - cv.loc[base, "cv_pr_auc_mean"],
                     "folds_improved": int((d > 0).sum()),
                     "seed_sd": cv.loc[[m, base], "seed_sd_mean_over_folds"].mean()})
    return pd.DataFrame(rows)


def stage_figures(P, cache, log) -> None:
    """CV-stage figures and the tables behind them (no calibration/test data)."""
    from . import plots
    config.FIGURES_DIR.mkdir(exist_ok=True)
    R, F = config.RESULTS_DIR, config.FIGURES_DIR
    cv = pd.concat([phase1_cv(), pd.read_csv(R / "cv_models.csv")], ignore_index=True)
    main = cv[~cv["model"].str.startswith("ablation:")]
    labels = {"mlp": "Embedding MLP", "gru": "MLP + GRU history", "wide_deep": "Wide & Deep",
              "emb_xgb": "Embeddings → XGBoost"}
    ref = float(cv.set_index("model").loc["Phase 1 LogReg", "cv_pr_auc_mean"])
    plots.cv_models(main, labels, ref).savefig(F / "cv_models.png", dpi=200)
    abl = ablation_table()
    abl.to_csv(R / "ablation.csv", index=False)
    plots.ablation(abl).savefig(F / "ablation_cv.png", dpi=200)
    # learning curves: largest fold, search seed, all epochs (no early stopping)
    k, fit_rows, va, fit_date = list(train.folds(P))[-1]
    curves = {}
    for name in ("mlp", "gru"):
        spec = load_spec(name).with_params(early_stopping=False)
        tr, es = train.inner_split(P, fit_rows, fit_date)
        _, info = train.fit(spec, cache.get(("fold", k), fit_rows, fit_date, spec), tr,
                            config.SEARCH_SEED, es_rows=es, epochs=None)
        curves[{"mlp": "Embedding MLP", "gru": "MLP + GRU history"}[name]] = info["history"][:15]
    (R / "learning_curves.json").write_text(json.dumps(curves, indent=2))
    plots.learning_curves(curves).savefig(F / "learning_curves.png", dpi=200)
    log(f"figures: {sorted(p.name for p in F.glob('*.png'))}")


# ---------------------------------------------------------------------------
# Diagnostics: why does the network trail LogReg? (CV only, search seed)
# ---------------------------------------------------------------------------
def stage_diagnostics(P, cache, log) -> None:
    """Decompose the MLP-LogReg gap on the rolling folds (single seed):

    D0/D1  LogReg on all fold rows vs without the inner-holdout days
           (what an early-stopped network trains on)
    D2/D3  linear network (one-hot, no hidden layer = a logistic regression)
           early-stopped vs refitted: does our training loop reach sklearn's LogReg?
    D4/D5  selected MLP early-stopped vs refitted: what the hidden layers add
    D6     selected MLP without the day-level history group (memorisation check)
    """
    from aerocast import models as p1_models
    y = train.tensors.row_info(P)["y"].to_numpy()
    folds = list(train.folds(P))
    s = config.SEARCH_SEED

    def logreg(drop_last: bool):
        out = []
        for k, fit_rows, va, fd in folds:
            r = train.inner_split(P, fit_rows, fd)[0] if drop_last else fit_rows
            m = p1_models.make_logreg(P.cat_cols, P.num_cols, config.LOGREG_C)
            m.fit(P.X.iloc[r][P.feat_cols], y[r])
            out.append((average_precision_score(y[va], m.decision_function(P.X.iloc[va][P.feat_cols])), np.nan))
        return out

    def net(spec, refit: bool):
        out = []
        for k, fit_rows, va, fd in folds:
            inp = cache.get(("fold", k), fit_rows, fd, spec)
            tr, es = train.inner_split(P, fit_rows, fd)
            n, info = (train.refit_as_deployed(spec, inp, fit_rows, tr, es, s) if refit
                       else train.fit(spec, inp, tr, s, es_rows=es))
            out.append((average_precision_score(y[va], train.predict_logits(n, inp, va)), info["best_epoch"]))
        return out

    linear = train.Spec(name="linear", cat_mode="onehot", depth=0, dropout=0.0)
    mlp = load_spec("mlp")
    runs = [
        ("D0", "LogReg, all fold rows (Phase 1)", lambda: logreg(False)),
        ("D1", "LogReg without the last 14 days", lambda: logreg(True)),
        ("D2", "linear network, early-stopped (no last 14 days)", lambda: net(linear, False)),
        ("D3", "linear network, refitted on all fold rows", lambda: net(linear, True)),
        ("D4", "selected MLP, early-stopped (no last 14 days)", lambda: net(mlp, False)),
        ("D5", "selected MLP, refitted on all fold rows", lambda: net(mlp, True)),
        ("D6", "selected MLP refitted, without day-level history",
         lambda: net(mlp.with_params(drop_groups=("history_day",)), True)),
    ]
    rows = []
    for key, label, fn in runs:
        res = fn()
        rows.append({"diagnostic": key, "label": label, "cv_pr_auc": float(np.mean([a for a, _ in res])),
                     **{c: a for c, (a, _) in zip(FOLD_COLS, res)},
                     "best_epochs": "/".join("" if np.isnan(e) else str(int(e)) for _, e in res)})
        log(f"  {key} {label}: {rows[-1]['cv_pr_auc']:.4f}")
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "diagnostics.csv", index=False)
