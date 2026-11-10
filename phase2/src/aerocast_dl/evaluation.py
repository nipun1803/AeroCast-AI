"""Final refits, calibration, model selection and the one-time test evaluation (Step 7).

Stages (``scripts/run_phase2.py``):

* ``final_fit``  refit every Phase 2 family on all training rows available at
  2013-09-01 (fixed epochs = mean CV best epoch, 5 seeds) and save scores for
  every modelling row. Scores need no labels.
* ``calibrate``  Sep -> Oct calibrator rule for every model (Phase 1 LogReg
  and XGBoost scores included), frozen thresholds, and the final-model rule
  on CV. Reads train and calibration labels only.
* ``final``      the test lock (``testlock.require_final``), then test metrics,
  policies and the day-block bootstrap. The only stage that reads Nov-Dec labels.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aerocast import config as p1
from aerocast import evaluate
from aerocast.pipeline import Prepared

from . import calibrate, config, pipeline, tensors, testlock, train

FAMILIES = ["mlp", "gru", "wide_deep"]
MODELS = ["logreg", "xgboost", *FAMILIES, "emb_xgb"]          # score columns
LABEL = {"logreg": "Phase 1 LogReg", "xgboost": "Phase 1 XGBoost", "mlp": "Embedding MLP",
         "gru": "MLP + GRU history", "wide_deep": "Wide & Deep (LogReg + deep)",
         "emb_xgb": "MLP embeddings -> XGBoost"}
CV_NAME = {"logreg": "Phase 1 LogReg", "xgboost": "Phase 1 XGBoost", "mlp": "mlp", "gru": "gru",
           "wide_deep": "wide_deep", "emb_xgb": "emb_xgb"}
PRED_PATH = config.RESULTS_DIR / "predictions.csv.gz"


# ---------------------------------------------------------------------------
# Final refits
# ---------------------------------------------------------------------------
def final_epochs(name: str) -> int:
    ps = pd.read_csv(config.RESULTS_DIR / "cv_seeds.csv")
    return int(math.ceil(ps.loc[ps["model"] == name, "best_epoch"].mean()))


def stage_final_fit(P: Prepared, cache: train.InputCache, log) -> None:
    all_rows = np.arange(len(P.meta))
    scores, nets_mlp, inp_mlp = {}, None, None
    for fam in FAMILIES:
        spec, ep = pipeline.load_spec(fam), final_epochs(fam)
        log(f"final {fam}: {len(config.SEEDS)} seeds x {ep} epochs on the training rows "
            f"available at {config.FINAL_FIT_DATE}")
        nets, inp = train.fit_final(P, spec, ep, config.SEEDS, cache)
        z = train.ensemble_logits(nets, inp, all_rows)
        scores[fam] = 1 / (1 + np.exp(-z))
        for s, net in zip(config.SEEDS, nets):
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            train.torch.save(net.state_dict(), config.CACHE_DIR / f"final_{fam}_seed{s}.pt")
        if fam == "mlp":
            nets_mlp, inp_mlp = nets, inp
    rows = tensors.final_fit_rows(P)
    jobs = [(pipeline.embedding_features(n, inp_mlp, P.cat_cols), rows, all_rows) for n in nets_mlp]
    log("final emb_xgb: Phase 1 XGBoost + final MLP embeddings, one per seed (separate process)")
    scores["emb_xgb"] = np.mean(pipeline.run_xgb_jobs(jobs), axis=0)
    write_predictions(P, scores)
    embedding_figures(P, nets_mlp[0], inp_mlp)


def embedding_figures(P: Prepared, net, inp: train.Inputs) -> None:
    """PCA maps of the final MLP's carrier and destination embeddings (seed 42).
    Destinations: only the 20 busiest are labelled. Descriptive only."""
    from . import plots
    config.FIGURES_DIR.mkdir(exist_ok=True)
    rows = tensors.final_fit_rows(P)
    for k, col in enumerate(P.cat_cols):
        if col not in ("carrier", "dest"):
            continue
        vocab = inp.encoder.vocabs_[col]
        names = sorted(vocab.index_, key=vocab.index_.get)            # index 1..n
        vecs = net.embs[k].weight.detach().numpy()[1:]               # drop the shared UNK row
        busy = P.meta[col].iloc[rows].value_counts()
        label = None if col == "carrier" else [n in set(busy.index[:20]) for n in names]
        title = {"carrier": "Learned carrier embeddings", "dest": "Learned destination embeddings"}[col]
        plots.embedding_map(vecs, names, title, label).savefig(
            config.FIGURES_DIR / f"embedding_{col}.png", dpi=200)


def write_predictions(P: Prepared, dl_scores: dict) -> None:
    """Phase 1 predictions file + Phase 2 score columns, same rows and order."""
    p1_pred = pd.read_csv(pipeline.PHASE1_RESULTS / "predictions.csv.gz", float_precision="round_trip")
    order = np.concatenate([P.idx[s] for s in p1.SPLITS])        # Phase 1 file order
    assert len(order) == len(p1_pred)
    assert (pd.to_datetime(p1_pred["date"]).to_numpy()
            == pd.to_datetime(P.meta["date"].iloc[order]).to_numpy()).all()
    assert (p1_pred["y"].to_numpy() == tensors.row_info(P)["y"].to_numpy()[order]).all()
    out = p1_pred[["split", "date", "avail_date", "month", "y", "logreg", "xgboost"]].copy()
    for m, s in dl_scores.items():
        out[m] = np.asarray(s)[order]
    out.to_csv(PRED_PATH, index=False, compression="gzip")


def load_predictions(include_test: bool) -> pd.DataFrame:
    """Saved scores and labels; test rows only behind the test lock (checked
    before the file is read)."""
    if include_test:
        testlock.require_final()
    df = pd.read_csv(PRED_PATH, float_precision="round_trip")
    return df if include_test else df[df["split"] != "test"].reset_index(drop=True)


def _pool(df: pd.DataFrame, m: str) -> dict:
    return {"p": df[m].to_numpy(dtype=float), "y": df["y"].to_numpy(), "date": df["date"].to_numpy(),
            "avail": df["avail_date"].to_numpy(), "split": df["split"].to_numpy(),
            "month": df["month"].to_numpy()}


# ---------------------------------------------------------------------------
# Calibration, thresholds and the final-model rule (no test labels)
# ---------------------------------------------------------------------------
def model_complexity(P: Prepared, cache: train.InputCache) -> dict:
    """Trainable parameters for the parsimony tie-break. LogReg: coefficients
    + intercept on its design matrix; XGBoost: an upper bound of 7 fitted
    values (3 splits + 4 leaves) per depth-2 tree."""
    cv = pd.read_csv(config.RESULTS_DIR / "cv_models.csv").set_index("model")
    k, fit_rows, _, fit_date = list(train.folds(P))[-1]
    wide = cache.get(("fold", k), fit_rows, fit_date, train.Spec(wide=True)).wide.shape[1]
    xgb = pipeline.xgb_phase1_params()
    out = {"logreg": wide + 1, "xgboost": 7 * xgb["n_estimators"]}
    for m in FAMILIES:
        out[m] = int(cv.loc[m, "n_params"])
    out["emb_xgb"] = out["mlp"] + out["xgboost"]
    return out


def select_final_model(complexity: dict) -> dict:
    """Protocol §4: highest mean rolling-CV AP as deployed; within 0.005 of the
    best, the model with fewer trainable parameters."""
    cv = pd.concat([pipeline.phase1_cv(), pd.read_csv(config.RESULTS_DIR / "cv_models.csv")],
                   ignore_index=True).set_index("model")
    ap = {m: float(cv.loc[CV_NAME[m], "cv_pr_auc_mean"]) for m in MODELS}
    best = max(ap, key=ap.get)
    near = [m for m in MODELS if ap[best] - ap[m] <= 0.005]
    final = min(near, key=lambda m: complexity[m])
    return {"cv_pr_auc": ap, "best_by_cv": best, "within_0.005": near,
            "complexity": complexity, "final_model": final,
            "rule": "highest mean rolling-CV AP as deployed; within 0.005, fewer parameters"}


def stage_calibrate(P: Prepared, cache: train.InputCache, log) -> None:
    df = load_predictions(include_test=False)
    tables, choice, thresholds = [], {}, {}
    fit_thr = (df["split"].eq("calib").to_numpy()
               & (pd.to_datetime(df["avail_date"]) <= pd.Timestamp(p1.SPLITS["test"][0])).to_numpy())
    for m in MODELS:
        pool = _pool(df, m)
        choice[m], tbl = calibrate.select(pool)
        tables.append(tbl.assign(model=m))
        q = calibrate.apply(choice[m], pool)
        thresholds[m] = evaluate.threshold_for_precision(df["y"].to_numpy()[fit_thr], q[fit_thr])
        log(f"  {m}: calibrator {choice[m]}, frozen threshold {thresholds[m]:.3f}")
    pd.concat(tables, ignore_index=True).to_csv(config.RESULTS_DIR / "calibrator_selection.csv",
                                                index=False)
    sel = select_final_model(model_complexity(P, cache))
    sel |= {"calibrator": choice, "frozen_threshold": thresholds}
    (config.RESULTS_DIR / "selection.json").write_text(json.dumps(sel, indent=2))
    log(f"final model by the CV rule: {sel['final_model']} (best CV: {sel['best_by_cv']}, "
        f"within 0.005: {sel['within_0.005']})")


# ---------------------------------------------------------------------------
# The one-time test evaluation
# ---------------------------------------------------------------------------
def stage_final(P: Prepared, cache: train.InputCache, log) -> None:
    df = load_predictions(include_test=True)            # raises unless --final and clean protocol
    sel = json.loads((config.RESULTS_DIR / "selection.json").read_text())
    test = df["split"].eq("test").to_numpy()
    y, dates, month = df["y"].to_numpy(), df["date"].to_numpy(), df["month"].to_numpy()
    cal = {}
    rows, pol = [], []
    for m in MODELS:
        for variant in ("raw", sel["calibrator"][m]):
            q = calibrate.apply(variant, _pool(df, m)) if variant != "raw" else df[m].to_numpy(float)
            key = f"{m}:{variant}"
            cal[key] = q
            for name, mask in [("test", test), ("test_m11", test & (month == 11)),
                               ("test_m12", test & (month == 12))]:
                rows.append({"model": m, "label": LABEL[m], "calibration": variant,
                             "rule_selected": variant == sel["calibrator"][m], "split": name,
                             **evaluate.metric_bundle(y[mask], q[mask], sel["frozen_threshold"][m]
                                                      if variant == sel["calibrator"][m] else None)})
        q = cal[f"{m}:{sel['calibrator'][m]}"]
        for share in p1.RESCHEDULE_SHARES:
            flagged = evaluate.flag_daily_top_share(dates[test], q[test], share)
            pol.append({"model": m, "label": LABEL[m], "policy": f"daily top {share:.0%}",
                        **evaluate.policy_report(y[test], flagged)})
        pol.append({"model": m, "label": LABEL[m], "policy": "frozen threshold (P>=0.40 on Sep-Oct)",
                    "threshold": sel["frozen_threshold"][m],
                    **evaluate.policy_report(y[test], q[test] >= sel["frozen_threshold"][m])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(pol).to_csv(config.RESULTS_DIR / "test_policies.csv", index=False)
    # paired day-block bootstrap of every Phase 2 model against Phase 1 LogReg
    ref = cal[f"logreg:{sel['calibrator']['logreg']}"][test]
    boot = {}
    for m in MODELS[1:]:
        q = cal[f"{m}:{sel['calibrator'][m]}"][test]
        boot[m] = {fn: evaluate.bootstrap_diff_ci(dates[test], y[test], q, ref, f)
                   for fn, f in [("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)]}
    (config.RESULTS_DIR / "test_bootstrap.json").write_text(json.dumps(boot, indent=2))
    calibrated = pd.DataFrame({k: v[test] for k, v in cal.items()})
    calibrated.insert(0, "y", y[test])
    calibrated.insert(0, "date", dates[test])
    calibrated.to_csv(config.RESULTS_DIR / "test_calibrated_scores.csv.gz", index=False,
                      compression="gzip")
    from . import plots
    config.FIGURES_DIR.mkdir(exist_ok=True)
    final = sel["final_model"]
    show = list(dict.fromkeys([final, "logreg", "wide_deep"]))
    curves = {LABEL[m]: (y[test], cal[f"{m}:{sel['calibrator'][m]}"][test]) for m in show}
    plots.reliability(curves).savefig(config.FIGURES_DIR / "reliability_test.png", dpi=200)
    head = metrics[(metrics.split == "test") & metrics.rule_selected]
    log(head[["label", "calibration", "pr_auc", "roc_auc", "brier", "ece"]].round(4).to_string(index=False))
    final_reschedule(P, cache, df, log)


# ---------------------------------------------------------------------------
# Step 8: what-if rescheduling (tuned on Sep-Oct, applied to test in `final`)
# ---------------------------------------------------------------------------
RESCHEDULABLE = ["logreg", "mlp", "gru", "wide_deep"]     # scorers that can re-score rows


def reschedule_model(sel: dict) -> str:
    """The final model if it can re-score modified rows, else the best such
    model by CV (tree models run in a separate process and are not used)."""
    if sel["final_model"] in RESCHEDULABLE:
        return sel["final_model"]
    return max(RESCHEDULABLE, key=lambda m: sel["cv_pr_auc"][m])


def stage_reschedule(P: Prepared, cache: train.InputCache, log) -> None:
    from . import reschedule
    path = config.RESULTS_DIR / "selection.json"
    sel = json.loads(path.read_text())
    model = reschedule_model(sel)
    log(f"rescheduling with {model} ({sel['calibrator'][model]}); tuning on Sep-Oct")
    ctx = reschedule.build_context(P.raw)
    params, tbl = reschedule.tune(P, cache, load_predictions(include_test=False), model,
                                  sel["calibrator"][model], ctx, log)
    tbl.to_csv(config.RESULTS_DIR / "reschedule_tuning.csv", index=False)
    sel["reschedule"] = {"model": model, "params": params, "move_cost": reschedule.MOVE_COST}
    path.write_text(json.dumps(sel, indent=2))
    log(f"frozen rescheduling rule: {params}")


def final_reschedule(P: Prepared, cache: train.InputCache, df: pd.DataFrame, log) -> None:
    from . import reschedule
    sel = json.loads((config.RESULTS_DIR / "selection.json").read_text())
    r = sel["reschedule"]
    moves, summary = reschedule.apply_test(P, cache, df, r["model"], sel["calibrator"][r["model"]],
                                           reschedule.build_context(P.raw), r["params"])
    moves.to_csv(config.RESULTS_DIR / "reschedule_test_moves.csv", index=False)
    if len(moves):
        from . import plots
        plots.reschedule_shifts(moves).savefig(config.FIGURES_DIR / "reschedule_shifts.png", dpi=200)
    summary.to_csv(config.RESULTS_DIR / "reschedule_test.csv", index=False)
    log(summary[["rule", "cost", "moves_per_day", "predicted_delays_avoided_per_day",
                 "avoided_per_move", "net_benefit"]].round(3).to_string(index=False))
