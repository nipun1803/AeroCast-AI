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
# Variant keys: <model>_raw | _platt | _isotonic (static, fit on Sep-Oct) | _rolling<N> (causal
# rolling Platt) | _rule_selected (= the static variant chosen by the a-priori margin rule).
# Internally the pipeline works with these keys; every saved table and figure
# carries the explicit labels produced by ``label`` instead.
ROLL = config.ROLLING_CALIB_DAYS
HEADLINE = ["prior", "route_history", "logreg_raw", "logreg_rule_selected", f"logreg_rolling{ROLL}",
            "xgboost_rule_selected", f"xgboost_rolling{ROLL}"]
# Models compared by the operating policies (the daily top-k% policy only uses
# within-day ranks, which every monotone recalibration leaves unchanged).
POLICY_VARIANTS = ["prior", "route_history", "logreg_rule_selected", "xgboost_rule_selected"]
ORACLE_COLS = [f"oracle_recall@p{t:.2f}" for t in config.PRECISION_SWEEP]
SHOW_COLS = ["pr_auc", "roc_auc", "brier", "ece", "ece_equal_mass", "mean_pred", "base_rate",
             *ORACLE_COLS]
RAW_SCORES = ["prior", "route_history", "logreg", "xgboost"]
# XGBoost emits float32 scores; they are restored to float32 on load so that
# everything recomputed from the predictions artefact is bit-identical.
SCORE_DTYPES = {"xgboost": np.float32}


FAMILY = {"logreg": "LogReg", "xgboost": "XGBoost"}
CAL_NAME = {"raw": "raw", "platt": "Platt", "isotonic": "isotonic"}


def variant_label(key: str, choice: dict | None = None) -> tuple[str, str]:
    """(display label, calibration description) of a variant key, e.g.
    ``logreg_rule_selected`` -> ("LogReg (Platt, rule-selected)", "Platt (rule-selected)")."""
    if key == "prior":
        return "Prior (base rate)", "none"
    if key == "route_history":
        return "Route history", "none (smoothed rate)"
    fam, tag = key.split("_", 1)
    name = FAMILY[fam]
    if tag == "rule_selected":
        if choice is None:
            raise ValueError("the rule-selected variant needs the calibration choice")
        cal = CAL_NAME[choice[fam]]
        return f"{name} ({cal}, rule-selected)", f"{cal} (rule-selected)"
    if tag.startswith("rolling"):
        n = tag.removeprefix("rolling")
        return f"{name} (rolling Platt {n}d)", f"rolling Platt {n}d (causal)"
    if tag == "raw":
        return f"{name} (raw)", "raw"
    return f"{name} ({CAL_NAME[tag]}, static)", f"{CAL_NAME[tag]} (static, fit Sep-Oct)"


def label(df: pd.DataFrame, choice: dict, col: str = "model") -> pd.DataFrame:
    """Replace the variant keys in ``col`` by explicit labels and add the
    ``calibration`` column (key kept as ``variant``). Applied to every table
    before it is saved or displayed."""
    pairs = [variant_label(k, choice) for k in df[col]]
    out = df.copy()
    out.insert(0, "variant", out[col])
    out[col] = [p[0] for p in pairs]
    out.insert(1, "calibration", [p[1] for p in pairs])
    return out


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
def tune_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Rolling-CV PR-AUC of LogReg for every (SMOOTHING_M, LONG_WINDOW) pair,
    tuned JOINTLY with the regularisation strength: each pair is scored at the
    best C of ``models.LOGREG_CS`` (chosen by the same rolling CV), so the
    history settings are not conditioned on an arbitrary fixed C. The column
    ``cv_pr_auc_mean_C0.001`` keeps the earlier fixed-C score for comparison.
    The last two rows repeat the configured pair with the clock at 00:00 on D-1
    (= nothing from D-1 is used, i.e. a ~2-day lag) and at 24:00 on D-1 (= every
    D-1 arrival counts for D, i.e. the leaky variant the clock replaces)."""
    rows = []
    combos = [(m, L, config.DECISION_HOUR) for m in config.SMOOTHING_M_GRID
              for L in config.LONG_WINDOW_GRID]
    # clock comparison at the configured (m, window): 00:00 on D-1 (~2-day lag) and
    # 24:00 on D-1, i.e. every D-1 arrival counts for D (the leaky variant)
    combos += [(config.SMOOTHING_M, config.LONG_WINDOW, 0),
               (config.SMOOTHING_M, config.LONG_WINDOW, 24)]
    for m, L, hour in combos:
        P = prepare(raw=raw, features_kw={"m": m, "long_window": L, "decision_hour": hour})
        best_C, cv = models.cv_logreg(P.Xs["train"], P.ys["train"], P.labels.months["train"],
                                      P.cat_cols, P.num_cols)
        by_C = cv.groupby("C")["val_pr_auc"].agg(["mean", "std"])
        folds = cv[cv.C == best_C].set_index("fold")["val_pr_auc"]
        rows.append({"smoothing_m": m, "long_window": L, "decision_hour": hour,
                     "best_C": best_C,
                     **{f"fold_m{k}": v for k, v in zip(config.CV_VAL_MONTHS, folds)},
                     "cv_pr_auc_mean": by_C.loc[best_C, "mean"],
                     "cv_pr_auc_std": by_C.loc[best_C, "std"],
                     "cv_pr_auc_mean_C0.001": by_C.loc[0.001, "mean"]})
    return pd.DataFrame(rows)


def check_history_choice(tuning: pd.DataFrame) -> dict:
    """The config values must be the CV-best (m, L) at the 20:00 clock, with
    C tuned jointly."""
    grid = tuning[tuning.decision_hour == config.DECISION_HOUR]
    best = grid.loc[grid["cv_pr_auc_mean"].idxmax()]
    chosen = {"smoothing_m": int(best.smoothing_m), "long_window": int(best.long_window),
              "C": float(best.best_C)}
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
    base = baseline_cv(P)
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
        **{k: {"mean": float(v.mean()), "std": float(v.std()), "folds": [float(x) for x in v]}
           for k, v in base.items()},
    }


def baseline_cv(P: Prepared) -> dict[str, pd.Series]:
    """Rolling-CV PR-AUC of the two reference models on the same folds (the
    prior has PR-AUC = the validation month's base rate)."""
    heur = P.Xs["train"][heuristic_col()]
    y, months = P.ys["train"], P.labels.months["train"]
    out = {"prior": [], "route_history": []}
    for tr, va in models.rolling_folds(months):
        rate = y[tr].mean()
        out["prior"].append(average_precision_score(y[va], np.full(len(va), rate)))
        out["route_history"].append(average_precision_score(y[va], heur.iloc[va].fillna(rate)))
    return {k: pd.Series(v, index=config.CV_VAL_MONTHS) for k, v in out.items()}


CV_LABELS = {
    "prior": "Prior (base rate)", "route_history": "Route history",
    "logreg": "LogReg (as deployed)",
    "xgboost": "XGBoost (as deployed, fixed trees)",
    "xgboost_infold_early_stopping": "XGBoost (in-fold early stopping, optimistic; not used)",
}


def cv_table(cv: dict) -> pd.DataFrame:
    """Rolling-CV PR-AUC per model and fold (raw scores: CV never recalibrates)."""
    rows = [{"model": CV_LABELS[k], "calibration": "none" if k in ("prior", "route_history")
             else "raw", "variant": k, "cv_pr_auc_mean": cv[k]["mean"],
             "cv_pr_auc_std": cv[k]["std"],
             **{f"fold_m{m}": v for m, v in zip(config.CV_VAL_MONTHS, cv[k]["folds"])}}
            for k in CV_LABELS]
    return pd.DataFrame(rows)


def paired_cv_difference(cv: dict, a: str = "logreg", b: str = "xgboost") -> dict:
    """Per-fold PR-AUC difference a - b (same folds, so the fold-to-fold month
    effect cancels), with its mean, sd and the number of folds a wins."""
    d = np.asarray(cv[a]["folds"]) - np.asarray(cv[b]["folds"])
    return {"diffs": [float(x) for x in d], "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "folds_a_better": int((d > 0).sum()), "n_folds": int(len(d))}


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
    ``<model>_rule_selected`` = the variant chosen by the a-priori margin rule
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
        final[f"{m}_rule_selected"] = final[f"{m}_{choice[m]}"]
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


def metrics_table(labels: Labels, final_preds: dict, thresholds: dict,
                  choice: dict) -> pd.DataFrame:
    """Metrics of every variant on train/calib/test and test Nov / Dec, with
    explicit variant labels and a ``calibration`` column."""
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
    return label(pd.DataFrame(rows), choice)


def headline_table(results: pd.DataFrame, names=HEADLINE, split: str = "test") -> pd.DataFrame:
    """Baselines + learned models, each with its calibration variant (rows in
    ``names`` order). Input: the labelled ``metrics_table``."""
    tbl = results[(results.split == split) & results.variant.isin(names)]
    return tbl.set_index("variant").loc[list(names)].reset_index()


def headline_by_month(results: pd.DataFrame, names=HEADLINE) -> pd.DataFrame:
    """The headline table for test November and December separately."""
    return pd.concat([headline_table(results, names, split=sp)
                      for sp in ("test_m11", "test_m12")], ignore_index=True)


def mean_scores(raw_preds: dict) -> pd.DataFrame:
    """Mean raw score per model and split (a prediction-level artefact)."""
    return pd.DataFrame({m: {s: float(np.mean(p, dtype=np.float64)) for s, p in d.items()}
                         for m, d in raw_preds.items()}).T.rename_axis("model").reset_index()


def operating_policies(labels: Labels, final_preds: dict, thresholds: dict, choice: dict,
                       names=POLICY_VARIANTS) -> pd.DataFrame:
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
    return label(pd.DataFrame(rows), choice)


def rescheduling_candidates(labels: Labels, final_preds: dict, choice: dict,
                            names=("prior", "route_history", "logreg_rule_selected",
                                   "xgboost_rule_selected"),
                            shares=config.RESCHEDULE_SHARES, split: str = "test") -> pd.DataFrame:
    """Decision-support view: each day, the top-k% flights by risk become
    reschedule / add-buffer candidates for a capacity-limited review desk.

    Reports ranking quality only (precision, recall, lift, delayed flights
    captured per 100 flagged). It is NOT realised operational savings: the data
    holds no cost or schedule-change model.
    """
    rows = []
    n_days = len(np.unique(labels.dates[split]))
    for name in names:
        for share in shares:
            flagged = evaluate.flag_daily_top_share(labels.dates[split], final_preds[name][split], share)
            rep = evaluate.policy_report(labels.y[split], flagged)
            rows.append({"model": name, "split": split, "top_share": share,
                         "flagged_per_day": float(flagged.sum() / n_days),
                         **rep, "delayed_per_100_flagged": 100 * rep["precision"],
                         "delayed_captured": int((flagged & (labels.y[split] == 1)).sum())})
    return label(pd.DataFrame(rows), choice)


CALIB_STUDY_SPLITS = ["test", "test_m11", "test_m12"]


def calibration_study(results: pd.DataFrame) -> pd.DataFrame:
    """Every calibration variant of both models on test, Nov and Dec, with the
    two reference models first. One row per variant, columns ``<metric>_<split>``.
    Input: the labelled ``metrics_table``."""
    variants = ["prior", "route_history"] + [
        f"{m}_{v}" for m in MODELS
        for v in ["raw", "platt", "isotonic", *[f"rolling{n}" for n in config.ROLLING_CALIB_SENSITIVITY]]]
    metrics = ["ece", "ece_equal_mass", "brier", "mean_pred", "base_rate"]
    sub = results[results.variant.isin(variants) & results.split.isin(CALIB_STUDY_SPLITS)]
    wide = sub.pivot_table(index="variant", columns="split", values=metrics).loc[variants]
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    keep = [f"{m}_{s}" for m in metrics for s in CALIB_STUDY_SPLITS]
    lab = sub.drop_duplicates("variant").set_index("variant").loc[variants, ["model", "calibration"]]
    return pd.concat([lab, wide[keep]], axis=1).rename_axis("variant").reset_index()


def monthly_drift(labels: Labels, final_preds: dict, choice: dict,
                  names=("logreg_raw", "xgboost_raw", "logreg_platt", f"logreg_rolling{ROLL}")
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
    return label(pd.concat(out, ignore_index=True), choice)


def bootstrap_compare(labels: Labels, final_preds: dict,
                      n_boot: int = config.N_BOOTSTRAP) -> dict:
    """Day-block bootstrap of test PR-AUC and ROC-AUC, XGBoost vs LogReg."""
    pair = {"xgboost": final_preds["xgboost_rule_selected"]["test"],
            "logreg": final_preds["logreg_rule_selected"]["test"]}
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
    # reference rows: the two baselines scored on the same folds / test set
    base_cv = baseline_cv(P)
    base_raw = baseline_preds(P)
    for key in ("prior", "route_history"):
        lab, cal = variant_label(key)
        rows.append({"model": lab, "calibration": cal, "features": "(baseline)",
                     "n_features": 0, "cv_pr_auc_mean": base_cv[key].mean(),
                     "cv_pr_auc_std": base_cv[key].std(),
                     "test_pr_auc": average_precision_score(P.ys["test"], base_raw[key]["test"]),
                     "test_roc_auc": roc_auc_score(P.ys["test"], base_raw[key]["test"])})
    for model in MODELS:
        for set_name, cols in sets.items():
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
            preds[model, set_name], folds[model, set_name] = p, cv
            rows.append({"model": FAMILY[model], "calibration": "raw", "features": set_name,
                         "n_features": len(cols),
                         "cv_pr_auc_mean": cv.mean(), "cv_pr_auc_std": cv.std(),
                         "test_pr_auc": average_precision_score(P.ys["test"], p),
                         "test_roc_auc": roc_auc_score(P.ys["test"], p)})
    steps = []
    for model in MODELS:
        for prev, cur in zip(ABLATION_STEPS[:-1], ABLATION_STEPS[1:]):
            gain = folds[model, cur] - folds[model, prev]
            ci = evaluate.bootstrap_diff_ci(P.labels.dates["test"], P.ys["test"],
                                            preds[model, cur], preds[model, prev], n_boot=n_boot)
            steps.append({"model": FAMILY[model], "calibration": "raw", "step": cur,
                          "cv_gain_mean": gain.mean(),
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
        base_scores = baseline_preds(P)   # prior = this target's train base rate
        entries = [(k, base_scores[k]["test"]) for k in ("prior", "route_history")]
        entries += [(f"{m}_raw", scores[m]) for m in MODELS]
        for key, sc in entries:
            b = evaluate.metric_bundle(y, sc)
            pol = evaluate.policy_report(y, evaluate.flag_daily_top_share(d, sc))
            lab, cal = variant_label(key)
            learned = key.endswith("_raw")
            rows.append({"target": target, "model": lab, "calibration": cal, "n_test": b["n"],
                         "base_rate": b["base_rate"], "pr_auc": b["pr_auc"],
                         "roc_auc": b["roc_auc"], "brier": b["brier"], "ece": b["ece"],
                         "top10_precision": pol["precision"], "top10_recall": pol["recall"],
                         "top10_lift": pol["lift"],
                         "xgb_minus_lr_pr_auc_ci":
                             f"[{diff['ci_lo']:+.3f}, {diff['ci_hi']:+.3f}]" if learned else ""})
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


def evaluation_figures(labels: Labels, final_preds: dict, choice: dict, drift_tbl: pd.DataFrame,
                       gain: pd.Series, perm: pd.DataFrame, abl: pd.DataFrame,
                       psi: pd.DataFrame) -> dict:
    """All report figures for the evaluation stage, keyed by file name."""
    yt = labels.y["test"]
    final = config.FINAL_MODEL
    curves = ["logreg_rule_selected", "xgboost_rule_selected", "route_history"]
    figs = {"pr_roc_test": plots.pr_roc_curves(
        yt, {variant_label(k, choice)[0]: final_preds[k]["test"] for k in curves},
        " - test Nov-Dec")}
    for m in MODELS:
        keys = [f"{m}_raw", f"{m}_platt", f"{m}_rolling{ROLL}", "route_history"]
        figs[f"reliability_{m}_test"] = plots.reliability_diagram(
            yt, {variant_label(k, choice)[0]: final_preds[k]["test"] for k in keys},
            f"{FAMILY[m]} reliability - test")
    figs["monthly_drift_final"] = plots.monthly_drift(
        drift_tbl[drift_tbl.variant == f"{final}_raw"],
        f"{FAMILY[final]} (raw): predicted vs observed delay rate by month")
    top = gain.head(20)
    figs["xgb_gain_top20"] = plots.hbar(top.to_numpy(), top.index.tolist(),
                                        "XGBoost top-20 features (gain)", "Average gain")
    pf = perm[perm.model == FAMILY[final]]
    figs["perm_importance_groups"] = plots.hbar(
        pf["pr_auc_drop"].to_numpy(), pf["group"].tolist(),
        f"Grouped permutation importance, {final} (Sep-Oct)", "Drop in PR-AUC")
    figs["ablation_cv"] = plots.ablation_bars(abl)
    top_psi = psi.head(15)
    figs["psi_top15"] = plots.hbar(top_psi["psi_test"].to_numpy(), top_psi["feature"].tolist(),
                                   "Covariate drift: PSI train -> test (top 15)", "PSI")
    return figs
