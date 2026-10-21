"""Phase 1 pipeline stages.

Each function is one stage of ``scripts/run_phase1.py``; the notebooks call the
same functions, so there is exactly one implementation of every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, data, drift, evaluate, features, models, plots

MODELS = ["logreg", "xgboost"]
HEADLINE = ["prior", "route_history", "logreg_cal", "xgboost_cal"]
ORACLE_COLS = [f"oracle_recall@p{t:.2f}" for t in config.PRECISION_SWEEP]
SHOW_COLS = ["pr_auc", "roc_auc", "brier", "ece", "ece_equal_mass", "mean_pred", "base_rate",
             *ORACLE_COLS]
RAW_SCORES = ["prior", "route_history", "logreg", "xgboost"]
# XGBoost emits float32 scores; they are restored to float32 on load so that
# everything recomputed from the predictions artefact is bit-identical.
SCORE_DTYPES = {"xgboost": np.float32}


def heuristic_col() -> str:
    return f"hist_rate_carrier_origin_dest_{config.LONG_WINDOW}d"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Labels:
    """Per-split outcomes, months, flight dates and outcome-availability dates."""

    y: dict[str, np.ndarray]
    months: dict[str, np.ndarray]
    dates: dict[str, np.ndarray]
    avail: dict[str, np.ndarray]

    @classmethod
    def from_predictions(cls, preds: pd.DataFrame) -> "Labels":
        parts = {s: preds[preds["split"] == s] for s in config.SPLITS}
        return cls(
            y={s: p["y"].to_numpy() for s, p in parts.items()},
            months={s: p["month"].to_numpy() for s, p in parts.items()},
            dates={s: pd.to_datetime(p["date"]).to_numpy() for s, p in parts.items()},
            avail={s: pd.to_datetime(p["avail_date"]).to_numpy() for s, p in parts.items()},
        )

    def pooled(self, attr: str) -> np.ndarray:
        """Concatenate an attribute over train, calib, test (chronological)."""
        return np.concatenate([getattr(self, attr)[s] for s in config.SPLITS])


@dataclass
class Prepared:
    raw: pd.DataFrame            # full schedule incl. cancelled flights
    meta: pd.DataFrame           # modelling rows (known outcome, past warm-up)
    X: pd.DataFrame              # features for the modelling rows
    groups: dict[str, list[str]]
    idx: dict[str, np.ndarray]   # row positions per split
    Xs: dict[str, pd.DataFrame] = field(repr=False)
    labels: Labels = field(repr=False)

    @property
    def cat_cols(self) -> list[str]:
        return self.groups["categorical"]

    @property
    def num_cols(self) -> list[str]:
        return [c for g, cols in self.groups.items() if g != "categorical" for c in cols]

    @property
    def feat_cols(self) -> list[str]:
        return self.cat_cols + self.num_cols

    @property
    def ys(self) -> dict[str, np.ndarray]:
        return self.labels.y

    def split_summary(self) -> dict:
        return {
            s: {"rows": int(len(i)), "delay_rate": float(self.ys[s].mean()),
                "from": str(self.meta["date"].iloc[i].min().date()),
                "to": str(self.meta["date"].iloc[i].max().date())}
            for s, i in self.idx.items()
        }


def prepare(target: str = "arr15", raw: pd.DataFrame | None = None,
            features_kw: dict | None = None) -> Prepared:
    """Load, build features on the full schedule, keep modelling rows, split.

    ``target`` selects the label definition (``data.target_variant``); the
    features are identical for every target (history always uses the main
    arr_delay >= 15 outcome).
    """
    raw = data.add_target(data.load_raw()) if raw is None else raw
    X_all, groups = features.build_features(raw, **(features_kw or {}))
    y_all = data.target_variant(raw, target)
    keep = y_all.notna() & (raw["date"] > config.WARMUP_END)
    meta, X = raw.loc[keep].reset_index(drop=True), X_all.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).to_numpy()
    split = data.assign_split(meta).to_numpy()
    idx = {s: np.flatnonzero(split == s) for s in config.SPLITS}

    feat_cols = groups["categorical"] + [c for g, cols in groups.items()
                                         if g != "categorical" for c in cols]
    avail = data.available_date(meta["arr_known_dt"]).to_numpy()
    labels = Labels(
        y={s: y[i] for s, i in idx.items()},
        months={s: meta["month"].iloc[i].to_numpy() for s, i in idx.items()},
        dates={s: meta["date"].iloc[i].to_numpy() for s, i in idx.items()},
        avail={s: avail[i] for s, i in idx.items()},
    )
    Xs = {s: X.iloc[i][feat_cols].reset_index(drop=True) for s, i in idx.items()}
    return Prepared(raw, meta, X, groups, idx, Xs, labels)


# ---------------------------------------------------------------------------
# Feature-design checks (rolling CV on the training window only)
# ---------------------------------------------------------------------------
def tune_history(raw: pd.DataFrame, C: float = 0.001) -> pd.DataFrame:
    """Rolling-CV PR-AUC of LogReg (fixed C) for every (SMOOTHING_M,
    LONG_WINDOW) pair, plus a decision-clock comparison: the 20:00 clock vs
    a clock at 00:00 on D-1 (= nothing from D-1 is used, i.e. a ~2-day lag)."""
    rows = []
    combos = [(m, L, config.DECISION_HOUR) for m in config.SMOOTHING_M_GRID
              for L in config.LONG_WINDOW_GRID]
    combos.append((config.SMOOTHING_M, config.LONG_WINDOW, 0))
    for m, L, hour in combos:
        P = prepare(raw=raw, features_kw={"m": m, "long_window": L, "decision_hour": hour})
        cv = models.cv_score_features_logreg(P.Xs["train"], P.ys["train"],
                                             P.labels.months["train"], P.feat_cols,
                                             P.cat_cols, C)
        rows.append({"smoothing_m": m, "long_window": L, "decision_hour": hour,
                     **{f"fold_m{k}": v for k, v in cv.items()},
                     "cv_pr_auc_mean": cv.mean(), "cv_pr_auc_std": cv.std()})
    return pd.DataFrame(rows)


def check_history_choice(tuning: pd.DataFrame) -> dict:
    """The config values must be the CV-best (m, L) at the 20:00 clock."""
    grid = tuning[tuning.decision_hour == config.DECISION_HOUR]
    best = grid.loc[grid["cv_pr_auc_mean"].idxmax()]
    chosen = {"smoothing_m": int(best.smoothing_m), "long_window": int(best.long_window)}
    if (chosen["smoothing_m"], chosen["long_window"]) != (config.SMOOTHING_M, config.LONG_WINDOW):
        raise RuntimeError(f"CV-best history settings {chosen} differ from config "
                           f"(SMOOTHING_M={config.SMOOTHING_M}, LONG_WINDOW={config.LONG_WINDOW})")
    return chosen


def month_feature_cv(P: Prepared, C: float, xgb_params: dict) -> pd.DataFrame:
    """Rolling-CV PR-AUC with vs without cyclic month features (both models)."""
    angle = 2 * np.pi * (P.meta["month"].iloc[P.idx["train"]].to_numpy() - 1) / 12
    Xm = P.Xs["train"].assign(month_sin=np.sin(angle), month_cos=np.cos(angle))
    rows = []
    for label, cols in [("without month", P.feat_cols),
                        ("with month", P.feat_cols + ["month_sin", "month_cos"])]:
        for model in MODELS:
            if model == "logreg":
                cv = models.cv_score_features_logreg(Xm, P.ys["train"], P.labels.months["train"],
                                                     cols, P.cat_cols, C)
            else:
                cv = models.cv_score_features(Xm, P.ys["train"], P.labels.months["train"],
                                              cols, xgb_params)
            rows.append({"features": label, "model": model,
                         **{f"fold_m{k}": v for k, v in cv.items()},
                         "cv_pr_auc_mean": cv.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def baseline_preds(P: Prepared) -> dict[str, dict[str, np.ndarray]]:
    """Constant train base rate, and the long-window carrier-route history rate."""
    base_rate = P.ys["train"].mean()
    return {
        "prior": {s: np.full(len(P.ys[s]), base_rate) for s in P.idx},
        "route_history": {s: P.Xs[s][heuristic_col()].fillna(base_rate).to_numpy()
                          for s in P.idx},
    }


def predict_splits(model, P: Prepared) -> dict[str, np.ndarray]:
    return {s: model.predict_proba(P.Xs[s])[:, 1] for s in P.idx}


@dataclass
class LogRegResult:
    model: object
    C: float
    cv: pd.DataFrame  # one row per (C, fold)


def train_logreg(P: Prepared) -> LogRegResult:
    """Rolling-origin CV over C, then refit on all of Jan-Aug."""
    best_C, cv = models.cv_logreg(P.Xs["train"], P.ys["train"], P.labels.months["train"],
                                  P.cat_cols, P.num_cols)
    model = models.make_logreg(P.cat_cols, P.num_cols, best_C).fit(P.Xs["train"], P.ys["train"])
    return LogRegResult(model, best_C, cv)


@dataclass
class XGBResult:
    model: object
    params: dict
    cv_folds: pd.DataFrame    # one row per (config, fold)
    cv_configs: pd.DataFrame  # one row per config, sorted best first


def train_xgb(P: Prepared, n_configs: int = 16) -> XGBResult:
    """Rolling-origin config search (tree count from fold early stopping),
    then refit on all of Jan-Aug with the averaged tree count."""
    params, folds, agg = models.cv_xgb(P.Xs["train"], P.ys["train"], P.labels.months["train"],
                                       n_configs=n_configs)
    model = models.fit_xgb_final(P.Xs["train"], P.ys["train"], params)
    return XGBResult(model, params, folds, agg.sort_values("mean_pr_auc", ascending=False))


def cv_summary(lr: LogRegResult, xgb: XGBResult, P: Prepared) -> dict:
    """Rolling-CV PR-AUC of each model AS DEPLOYED.

    XGBoost's search score early-stops on each validation fold, i.e. the
    tree count is tuned on the data it is scored on -> optimistic. The
    deployed model uses the fold-averaged tree count, so it is re-scored on
    the same folds with that fixed count (no in-fold tuning). Both numbers are
    reported; the fixed-count one is used for model selection.
    """
    lr_s = lr.cv[lr.cv.C == lr.C].set_index("fold")["val_pr_auc"]
    fixed = models.cv_score_features(P.Xs["train"], P.ys["train"], P.labels.months["train"],
                                     P.feat_cols, xgb.params)
    best = xgb.cv_configs.index[0]
    es = xgb.cv_folds[xgb.cv_folds.config == best]["val_pr_auc"]
    return {
        "logreg": {"mean": float(lr_s.mean()), "std": float(lr_s.std()),
                   "folds": [float(v) for v in lr_s]},
        "xgboost": {"mean": float(fixed.mean()), "std": float(fixed.std()),
                    "folds": [float(v) for v in fixed]},
        "xgboost_infold_early_stopping": {"mean": float(es.mean()), "std": float(es.std()),
                                          "folds": [float(v) for v in es]},
    }


def select_final_model(cv: dict) -> str:
    """Final model = higher mean rolling-CV PR-AUC (rule fixed before test).
    Fails loudly if the CV evidence disagrees with ``config.FINAL_MODEL``."""
    best = max(MODELS, key=lambda m: cv[m]["mean"])
    if best != config.FINAL_MODEL:
        raise RuntimeError(f"CV selects {best!r} but config.FINAL_MODEL={config.FINAL_MODEL!r}")
    return best


def single_month_tree_curve(P: Prepared, params: dict, max_trees: int = 800,
                            step: int = 10) -> pd.DataFrame:
    """Diagnostic for the abandoned design: PR-AUC on Sep and on Oct as a
    function of the number of boosting rounds (model fit on Jan-Aug)."""
    p = {k: v for k, v in params.items() if k != "n_estimators"}
    model = models.make_xgb(n_estimators=max_trees, early_stopping=False, **p).fit(
        P.Xs["train"], P.ys["train"], verbose=False)
    cal = P.Xs["calib"]
    months = P.labels.months["calib"]
    rows = []
    for n in range(step, max_trees + 1, step):
        s = model.predict_proba(cal, iteration_range=(0, n))[:, 1]
        rows.append({"n_trees": n,
                     **{f"pr_auc_m{m}": average_precision_score(P.ys["calib"][months == m],
                                                                s[months == m])
                        for m in (9, 10)}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration & evaluation (inputs: raw scores + Labels only)
# ---------------------------------------------------------------------------
def calibrate(labels: Labels, raw_preds: dict) -> tuple[dict, dict, pd.DataFrame]:
    """Static variants raw / Platt / isotonic (fit on all of Sep-Oct), causal
    rolling Platt for N in ROLLING_CALIB_SENSITIVITY (calib + test only), and
    ``<model>_cal`` = the variant chosen by the a-priori margin rule
    (``evaluate.select_calibrator``: fit Sep -> check Oct)."""
    final = {m: raw_preds[m] for m in ["prior", "route_history"]}
    y_cal, m_cal = labels.y["calib"], labels.months["calib"]
    pool_dates, pool_avail, pool_y = (labels.pooled("dates"), labels.pooled("avail"),
                                      labels.pooled("y"))
    eval_dates = np.concatenate([labels.dates["calib"], labels.dates["test"]])
    choice, tables = {}, []
    for m in MODELS:
        p_cal = raw_preds[m]["calib"]
        choice[m], tbl = evaluate.select_calibrator(p_cal, y_cal, m_cal, 9, 10)
        tables.append(tbl.assign(model=m))
        for tag, cls in evaluate.CALIBRATORS.items():
            fitted = cls().fit(p_cal, y_cal)
            final[f"{m}_{tag}"] = {s: fitted.predict(raw_preds[m][s]) for s in raw_preds[m]}
        final[f"{m}_cal"] = final[f"{m}_{choice[m]}"]
        pool_scores = np.concatenate([raw_preds[m][s] for s in config.SPLITS])
        for n in config.ROLLING_CALIB_SENSITIVITY:
            q = evaluate.rolling_platt(pool_scores, pool_y, pool_dates, pool_avail,
                                       eval_dates, n)
            bounds = np.cumsum([0] + [len(labels.y[s]) for s in config.SPLITS])
            final[f"{m}_rolling{n}"] = {s: q[bounds[i]:bounds[i + 1]]
                                        for i, s in enumerate(config.SPLITS) if s != "train"}
    return final, choice, pd.concat(tables, ignore_index=True)


def frozen_thresholds(labels: Labels, final_preds: dict) -> dict[str, float]:
    """Precision-target threshold chosen on Sep-Oct (inf if unreachable)."""
    return {n: evaluate.threshold_for_precision(labels.y["calib"], p["calib"])
            for n, p in final_preds.items()}


def metrics_table(labels: Labels, final_preds: dict, thresholds: dict) -> pd.DataFrame:
    rows = []
    test_month = labels.months["test"]
    for name, p in final_preds.items():
        for s in ["train", "calib", "test"]:
            if s in p:
                rows.append({"model": name, "split": s,
                             **evaluate.metric_bundle(labels.y[s], p[s], thresholds[name])})
        for mo in (11, 12):
            mask = test_month == mo
            rows.append({"model": name, "split": f"test_m{mo}",
                         **evaluate.metric_bundle(labels.y["test"][mask], p["test"][mask],
                                                  thresholds[name])})
    return pd.DataFrame(rows)


def headline_table(results: pd.DataFrame, names=HEADLINE) -> pd.DataFrame:
    tbl = results[(results.split == "test") & results.model.isin(names)].set_index("model")
    return tbl.loc[list(names)]


def operating_policies(labels: Labels, final_preds: dict, thresholds: dict,
                       names=HEADLINE[1:]) -> pd.DataFrame:
    """PRIMARY: daily top-k% capacity policy. SECONDARY: score threshold frozen
    on Sep-Oct for precision >= TARGET_PRECISION. Both fixed before test."""
    rows = []
    for name in names:
        for s in ["calib", "test"]:
            flagged = evaluate.flag_daily_top_share(labels.dates[s], final_preds[name][s])
            rows.append({"model": name, "split": s, "role": "primary",
                         "policy": f"daily top {config.DAILY_FLAG_SHARE:.0%}",
                         **evaluate.policy_report(labels.y[s], flagged)})
            thr = thresholds[name]
            rows.append({"model": name, "split": s, "role": "secondary",
                         "policy": f"frozen thr for P>={config.TARGET_PRECISION} (from Sep-Oct)",
                         "threshold": thr,
                         **evaluate.policy_report(labels.y[s], final_preds[name][s] >= thr)})
    return pd.DataFrame(rows)


def calibration_study(results: pd.DataFrame) -> pd.DataFrame:
    """Every calibration variant of both models on test, Nov and Dec."""
    variants = [f"{m}_{v}" for m in MODELS
                for v in ["raw", "platt", "isotonic",
                          *[f"rolling{n}" for n in config.ROLLING_CALIB_SENSITIVITY]]]
    sub = results[results.model.isin(variants) & results.split.isin(["test", "test_m11", "test_m12"])]
    return sub.pivot_table(index="model", columns="split",
                           values=["ece", "ece_equal_mass", "brier", "mean_pred"]).loc[variants]


def monthly_drift(labels: Labels, final_preds: dict,
                  names=("logreg_raw", "xgboost_raw", "logreg_platt", "logreg_rolling28")
                  ) -> pd.DataFrame:
    """Mean predicted risk vs observed delay rate per calendar month."""
    splits = list(config.SPLITS)
    month = labels.pooled("months")
    y = labels.pooled("y")
    out = []
    for name in names:
        # float64: XGBoost scores are float32, average them at full precision;
        # variants without train-period scores (rolling) are NaN there
        p = np.concatenate([np.asarray(final_preds[name].get(s, np.full(len(labels.y[s]), np.nan)),
                                       dtype=np.float64) for s in splits])
        g = pd.DataFrame({"month": month, "y": y, "p": p}).groupby("month").agg(
            observed=("y", "mean"), predicted=("p", "mean"))
        out.append(g.assign(model=name).reset_index())
    return pd.concat(out, ignore_index=True)


def bootstrap_compare(labels: Labels, final_preds: dict,
                      n_boot: int = config.N_BOOTSTRAP) -> dict:
    """Day-block bootstrap of test PR-AUC and ROC-AUC, XGBoost vs LogReg."""
    pair = {"xgboost": final_preds["xgboost_cal"]["test"],
            "logreg": final_preds["logreg_cal"]["test"]}
    return {
        name: evaluate.summarize_bootstrap(
            evaluate.day_block_bootstrap(labels.dates["test"], labels.y["test"], pair, fn,
                                         n_boot), "xgboost", "logreg")
        for name, fn in [("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)]
    }


# ---------------------------------------------------------------------------
# Ablation & importance (need the fitted models / features)
# ---------------------------------------------------------------------------
ABLATION_STEPS = ["schedule + ids", "+ congestion", "+ history_entity", "+ history_day"]


def ablation_sets(P: Prepared) -> dict[str, list[str]]:
    """Cumulative feature families, plus history alone. ("+ history_day" is
    the full model; "+ history_entity" equals "all minus history_day".)"""
    g = P.groups
    cumulative = P.cat_cols + g["schedule"]
    sets = {"schedule + ids": list(cumulative)}
    for name in ["congestion", "history_entity", "history_day"]:
        cumulative = cumulative + g[name]
        sets[f"+ {name}"] = list(cumulative)
    sets["history only"] = g["history_entity"] + g["history_day"]
    return sets


def ablation(P: Prepared, C: float, xgb_params: dict,
             n_boot: int = config.N_BOOTSTRAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both models on nested feature sets.

    Returns (per-set table, per-step table). Claims rest on the rolling-CV
    gain of each step (mean and number of folds improved); the test
    difference with a paired day-block bootstrap 95% CI is confirmation.
    """
    sets = ablation_sets(P)
    rows, preds, folds = [], {}, {}
    for model in MODELS:
        for label, cols in sets.items():
            if model == "logreg":
                cv = models.cv_score_features_logreg(P.Xs["train"], P.ys["train"],
                                                     P.labels.months["train"], cols, P.cat_cols, C)
                cats = [c for c in cols if c in P.cat_cols]
                mdl = models.make_logreg(cats, [c for c in cols if c not in cats], C).fit(
                    P.Xs["train"][cols], P.ys["train"])
            else:
                cv = models.cv_score_features(P.Xs["train"], P.ys["train"],
                                              P.labels.months["train"], cols, xgb_params)
                mdl = models.fit_xgb_final(P.Xs["train"][cols], P.ys["train"], xgb_params)
            p = mdl.predict_proba(P.Xs["test"][cols])[:, 1]
            preds[model, label], folds[model, label] = p, cv
            rows.append({"model": model, "features": label, "n_features": len(cols),
                         "cv_pr_auc_mean": cv.mean(), "cv_pr_auc_std": cv.std(),
                         "test_pr_auc": average_precision_score(P.ys["test"], p),
                         "test_roc_auc": roc_auc_score(P.ys["test"], p)})
    steps = []
    for model in MODELS:
        for prev, cur in zip(ABLATION_STEPS[:-1], ABLATION_STEPS[1:]):
            gain = folds[model, cur] - folds[model, prev]
            ci = evaluate.bootstrap_diff_ci(P.labels.dates["test"], P.ys["test"],
                                            preds[model, cur], preds[model, prev], n_boot=n_boot)
            steps.append({"model": model, "step": cur, "cv_gain_mean": gain.mean(),
                          "cv_folds_improved": f"{int((gain > 0).sum())}/{len(gain)}",
                          "test_diff": ci["diff"], "test_diff_ci_lo": ci["ci_lo"],
                          "test_diff_ci_hi": ci["ci_hi"]})
    return pd.DataFrame(rows), pd.DataFrame(steps)


def grouped_permutation_importance(P: Prepared, model, n_repeats: int = 5) -> pd.DataFrame:
    """Shuffle a whole feature family with one permutation (so correlated
    features can't cover for each other) and record the PR-AUC drop on
    Sep-Oct (not test)."""
    rng = np.random.default_rng(config.SEED)
    Xc, yc = P.Xs["calib"], P.ys["calib"]
    base_ap = average_precision_score(yc, model.predict_proba(Xc)[:, 1])
    rows = []
    for g, cols in P.groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(Xc))
            Xp = Xc.copy()
            for c in cols:
                Xp[c] = Xc[c].iloc[perm].to_numpy()
                if isinstance(Xc[c].dtype, pd.CategoricalDtype):
                    Xp[c] = pd.Categorical(Xp[c], categories=Xc[c].cat.categories)
            drops.append(base_ap - average_precision_score(yc, model.predict_proba(Xp)[:, 1]))
        rows.append({"group": g, "pr_auc_drop": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows)


def xgb_gain(model) -> pd.Series:
    return pd.Series(model.get_booster().get_score(importance_type="gain"),
                     name="gain").sort_values(ascending=False)


def logreg_coefficients(model) -> pd.Series:
    coefs = pd.Series(model.named_steps["clf"].coef_[0],
                      index=model.named_steps["pre"].get_feature_names_out(), name="coef")
    return coefs.sort_values(key=np.abs, ascending=False)


# ---------------------------------------------------------------------------
# Target sensitivity & distribution shift
# ---------------------------------------------------------------------------
def target_sensitivity(main: Prepared, main_raw: dict, C: float, xgb_params: dict,
                       n_boot: int = config.N_BOOTSTRAP) -> pd.DataFrame:
    """Refit both models (same hyper-parameters) with cancelled AND diverted
    flights counted as delayed, and compare raw-score test metrics and the
    primary policy with the main target. Features are unchanged."""
    alt = prepare(target="arr15_or_cancel", raw=main.raw)
    lr = models.make_logreg(alt.cat_cols, alt.num_cols, C).fit(alt.Xs["train"], alt.ys["train"])
    xg = models.fit_xgb_final(alt.Xs["train"], alt.ys["train"], xgb_params)
    alt_raw = {"logreg": lr.predict_proba(alt.Xs["test"])[:, 1],
               "xgboost": xg.predict_proba(alt.Xs["test"])[:, 1]}
    rows = []
    for target, P, scores in [("arr_delay>=15 (main)", main,
                               {m: main_raw[m]["test"] for m in MODELS}),
                              ("arr_delay>=15 OR cancelled/diverted", alt, alt_raw)]:
        y, d = P.ys["test"], P.labels.dates["test"]
        diff = evaluate.bootstrap_diff_ci(d, y, scores["xgboost"], scores["logreg"],
                                          n_boot=n_boot)
        for m in MODELS:
            b = evaluate.metric_bundle(y, scores[m])
            pol = evaluate.policy_report(y, evaluate.flag_daily_top_share(d, scores[m]))
            rows.append({"target": target, "model": m, "n_test": b["n"],
                         "base_rate": b["base_rate"], "pr_auc": b["pr_auc"],
                         "roc_auc": b["roc_auc"], "brier": b["brier"], "ece": b["ece"],
                         "top10_precision": pol["precision"], "top10_recall": pol["recall"],
                         "top10_lift": pol["lift"],
                         "xgb_minus_lr_pr_auc_ci": f"[{diff['ci_lo']:+.3f}, {diff['ci_hi']:+.3f}]"})
    return pd.DataFrame(rows)


def distribution_shift(P: Prepared) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PSI per feature (train reference) and adversarial-validation AUC for
    train-vs-calib and train-vs-test: with all features, and per feature
    family (to locate the shift), plus the features driving the full model."""
    psi = drift.psi_table(P.Xs, P.groups)
    sets = {"all features": P.feat_cols, **P.groups}
    adv_rows, imps = [], {}
    for cur in ["calib", "test"]:
        for label, cols in sets.items():
            auc, imp = drift.adversarial_validation(P.Xs["train"][cols], P.Xs[cur][cols])
            adv_rows.append({"comparison": f"train vs {cur}", "features": label,
                             "n_features": len(cols), "adversarial_auc": auc,
                             "top_features": ", ".join(imp.head(4).index)})
            if label == "all features":
                imps[cur] = imp
    return psi, pd.DataFrame(adv_rows), pd.DataFrame(imps).fillna(0)


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------
def predictions_frame(P: Prepared, raw_preds: dict) -> pd.DataFrame:
    """Uncalibrated scores for every modelling row: everything downstream
    (calibration, metrics, policies, drift, bootstrap) is a function of this."""
    parts = []
    for s, i in P.idx.items():
        parts.append(pd.DataFrame({
            "split": s, "date": P.meta["date"].iloc[i].to_numpy(),
            "avail_date": P.labels.avail[s],
            "month": P.labels.months[s], "y": P.ys[s],
            **{m: raw_preds[m][s] for m in RAW_SCORES},
        }))
    return pd.concat(parts, ignore_index=True)


def load_predictions(path=None) -> tuple[Labels, dict]:
    path = path or config.RESULTS_DIR / "predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/run_phase1.py` first")
    df = pd.read_csv(path, float_precision="round_trip")
    raw = {m: {s: df.loc[df.split == s, m].to_numpy(SCORE_DTYPES.get(m, np.float64))
               for s in config.SPLITS}
           for m in RAW_SCORES}
    return Labels.from_predictions(df), raw


def evaluation_figures(labels: Labels, final_preds: dict, drift_tbl: pd.DataFrame,
                       gain: pd.Series, perm: pd.DataFrame, abl: pd.DataFrame,
                       psi: pd.DataFrame) -> dict:
    """All report figures for the evaluation stage, keyed by file name."""
    yt = labels.y["test"]
    final = config.FINAL_MODEL
    figs = {"pr_roc_test": plots.pr_roc_curves(yt, {
        "LogReg": final_preds["logreg_cal"]["test"],
        "XGBoost": final_preds["xgboost_cal"]["test"],
        "Route history": final_preds["route_history"]["test"]}, " - test Nov-Dec")}
    for m, label in [("xgboost", "XGBoost"), ("logreg", "LogReg")]:
        figs[f"reliability_{m}_test"] = plots.reliability_diagram(yt, {
            f"{label} raw": final_preds[f"{m}_raw"]["test"],
            f"{label} Platt (static)": final_preds[f"{m}_platt"]["test"],
            f"{label} rolling Platt {config.ROLLING_CALIB_DAYS}d":
                final_preds[f"{m}_rolling{config.ROLLING_CALIB_DAYS}"]["test"]},
            f"{label} reliability - test")
    figs["monthly_drift_final"] = plots.monthly_drift(drift_tbl[drift_tbl.model == f"{final}_raw"])
    top = gain.head(20)
    figs["xgb_gain_top20"] = plots.hbar(top.to_numpy(), top.index.tolist(),
                                        "XGBoost top-20 features (gain)", "Average gain")
    pf = perm[perm.model == final]
    figs["perm_importance_groups"] = plots.hbar(
        pf["pr_auc_drop"].to_numpy(), pf["group"].tolist(),
        f"Grouped permutation importance, {final} (Sep-Oct)", "Drop in PR-AUC")
    figs["ablation_cv"] = plots.ablation_bars(abl)
    top_psi = psi.head(15)
    figs["psi_top15"] = plots.hbar(top_psi["psi_test"].to_numpy(), top_psi["feature"].tolist(),
                                   "Covariate drift: PSI train -> test (top 15)", "PSI")
    return figs
