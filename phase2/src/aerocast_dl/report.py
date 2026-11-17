"""Export the Phase 2 report's figures and tables (prose is written by hand).

Same contract as Phase 1 (``aerocast.report``): ``report/figures`` and
``report/tables`` are cleared and rewritten with exactly the files
``report/main.tex`` uses; every table is read from ``results/``, never
recomputed. ``tests/test_report.py`` checks main.tex against these lists.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from aerocast.report import to_booktabs

from . import config, evaluation, pipeline

REPORT_DIR = config.ROOT / "report"
REPORT_FIGURES = ("cv_models", "learning_curves", "ablation_cv", "embedding_carrier",
                  "reliability_test", "pr_curves_test", "confusion_test", "reschedule_shifts")
R = config.RESULTS_DIR
CALIB = {"raw": "none (raw)", "platt": "Platt (Sep)", "isotonic": "isotonic (Sep)",
         "temperature": "temperature (Sep)", "rolling_platt28": "rolling Platt, 28 d",
         "rolling_temperature28": "rolling temperature, 28 d"}
NAME = {"Phase 1 LogReg": "LogReg (Phase 1)", "Phase 1 XGBoost": "XGBoost (Phase 1)",
        "mlp": "Embedding MLP", "gru": "MLP + GRU history", "wide_deep": "Wide & Deep",
        "emb_xgb": "Embeddings → XGBoost"}   # cells are escaped by to_booktabs


def _cv() -> pd.DataFrame:
    cv = pd.concat([pipeline.phase1_cv(), pd.read_csv(R / "cv_models.csv")], ignore_index=True)
    cv = cv[cv["model"].isin(NAME)].copy()
    order = {m: i for i, m in enumerate(NAME)}            # fixed row order, not the CSV's
    cv = cv.sort_values("model", key=lambda s: s.map(order), kind="stable")
    sel = json.loads((R / "selection.json").read_text()) if (R / "selection.json").exists() else None
    key = {v: k for k, v in evaluation.CV_NAME.items()}
    cv["Params"] = [sel["complexity"][key[m]] if sel else float("nan") for m in cv["model"]]
    cv["Model"] = cv["model"].map(NAME)
    cv = cv.rename(columns={"cv_pr_auc_mean": "CV AP", "cv_pr_auc_std": "SD",
                            **{f"fold_m{m}": mo for m, mo in zip([5, 6, 7, 8], ["May", "Jun", "Jul", "Aug"])}})
    return cv[["Model", "Params", "CV AP", "SD", "May", "Jun", "Jul", "Aug"]]


def report_tables() -> dict[str, tuple[pd.DataFrame, int]]:
    """name -> (table, decimals). Tables whose results do not exist yet are skipped."""
    out = {}
    if (R / "cv_models.csv").exists():
        out["cv_models"] = (_cv(), 3)
    if (R / "diagnostics.csv").exists():
        d = pd.read_csv(R / "diagnostics.csv")
        out["diagnostics"] = (d.rename(columns={"diagnostic": "", "label": "Run", "cv_pr_auc": "CV AP",
                                                "best_epochs": "Best epochs"}).replace({"Best epochs": {"///": "--"}})
                           [["", "Run", "CV AP", "Best epochs"]], 4)
    if (R / "ablation.csv").exists():
        a = pd.read_csv(R / "ablation.csv").sort_values("delta")
        out["ablation"] = (a.rename(columns={"label": "Ablation", "cv": "CV AP", "delta": "$\\Delta$",
                                             "folds_improved": "Folds up", "seed_sd": "Seed SD"})
                           [["Ablation", "CV AP", "$\\Delta$", "Folds up", "Seed SD"]], 4)
    if (R / "calibrator_selection.csv").exists():
        c = pd.read_csv(R / "calibrator_selection.csv")
        piv = c.pivot(index="calibrator", columns="model", values="check_ece")
        piv = piv[[m for m in evaluation.MODELS if m in piv.columns]]
        short = {"logreg": "LogReg", "xgboost": "XGBoost", "mlp": "MLP", "gru": "MLP + GRU",
                 "wide_deep": "W\\&D", "emb_xgb": "Emb.$\\rightarrow$XGB"}
        piv.columns = [short[m] for m in piv.columns]
        piv = piv.reindex([c for c in CALIB if c in piv.index])
        piv.index = [CALIB[c] for c in piv.index]
        out["calibrator_selection"] = (piv.rename_axis("Oct ECE").reset_index(), 3)
    if (R / "test_metrics.csv").exists():
        m = pd.read_csv(R / "test_metrics.csv")
        h = m[(m.split == "test") & m.rule_selected].copy()
        h["Model"] = h["model"].map(lambda k: NAME[evaluation.CV_NAME[k]])
        h["calibration"] = h["calibration"].map(CALIB)
        out["results_test"] = (h.rename(columns={"calibration": "Calibration", "pr_auc": "AP", "roc_auc": "ROC",
                                                 "brier": "Brier", "ece": "ECE", "precision_at_thr": "P@thr",
                                                 "recall_at_thr": "R@thr"})
                               [["Model", "Calibration", "AP", "ROC", "Brier", "ECE", "P@thr", "R@thr"]], 3)
        pol = pd.read_csv(R / "test_policies.csv")
        p = pol[pol.policy == "daily top 10%"].copy()
        p["Model"] = p["model"].map(lambda k: NAME[evaluation.CV_NAME[k]])
        out["policy_top10"] = (p.rename(columns={"precision": "Precision", "recall": "Recall", "lift": "Lift",
                                                 "f1": "F1"})[["Model", "Precision", "Recall", "Lift", "F1"]], 3)
    if (R / "test_slices.csv").exists():
        s = pd.read_csv(R / "test_slices.csv")
        dims = [("dep_hour", "Dep. hour", ["05-09", "10-13", "14-17", "18-23"]),
                ("dest_train_rows", "Dest. rows", ["< 50", "50-499", "500-1,999", ">= 2,000"]),
                ("route_train_rows", "Route rows", ["< 50", "50-499", "500-1,999", ">= 2,000"])]
        parts = []
        for d, lab, order in dims:
            t = s[s.dimension == d].set_index("slice").loc[order].reset_index()
            t["Slice"] = [f"{lab} {v.replace('>= ', '≥ ').replace('-', '--')}" + ("" if k else "$^*$")
                          for v, k in zip(t["slice"], t["stable"])]
            parts.append(t)
        s = pd.concat(parts, ignore_index=True)
        out["test_slices"] = (s.rename(columns={"n": "Flights", "ap_logreg": "AP LogReg",
                                                "ap_gain_mlp": "$\\Delta$ MLP", "ap_gain_gru": "$\\Delta$ GRU",
                                                "ap_gain_wide_deep": "$\\Delta$ W\\&D",
                                                "ap_gain_xgboost": "$\\Delta$ XGB"})
                              [["Slice", "Flights", "AP LogReg", "$\\Delta$ MLP", "$\\Delta$ GRU",
                                "$\\Delta$ W\\&D", "$\\Delta$ XGB"]], 3)
    if (R / "reschedule_test.csv").exists():
        r = pd.read_csv(R / "reschedule_test.csv")
        r["rule"] = r["rule"].map(lambda x: "frozen rule" if x.startswith("frozen") else "naive, −1 h")
        out["reschedule"] = (r.rename(columns={"rule": "Rule", "cost": "Cost", "moves_per_day": "Moves/d",
                                               "predicted_delays_avoided_per_day": "Avoided/d",
                                               "avoided_per_move": "Per move", "net_benefit": "Net"})
                             [["Rule", "Cost", "Moves/d", "Avoided/d", "Per move", "Net"]], 2)
    return out


def export() -> dict:
    figs, tabs = REPORT_DIR / "figures", REPORT_DIR / "tables"
    for d in (figs, tabs):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    copied = []
    for f in REPORT_FIGURES:
        src = config.FIGURES_DIR / f"{f}.png"
        if src.exists():
            shutil.copy(src, figs / src.name)
            copied.append(f)
    tables = report_tables()
    for name, (df, dec) in tables.items():
        num = df.select_dtypes("float").columns
        df = df.copy()
        df[num] = df[num].round(dec) + 0.0                 # no "-0.000" cells
        df.to_csv(tabs / f"{name}.csv", index=False)
        (tabs / f"{name}.tex").write_text(to_booktabs(df, dec))
    return {"figures": copied, "tables": sorted(tables)}
