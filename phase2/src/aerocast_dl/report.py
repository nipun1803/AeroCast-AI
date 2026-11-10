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
                  "reliability_test", "reschedule_shifts")
R = config.RESULTS_DIR
NAME = {"Phase 1 LogReg": "LogReg (Phase 1)", "Phase 1 XGBoost": "XGBoost (Phase 1)",
        "mlp": "Embedding MLP", "gru": "MLP + GRU history", "wide_deep": "Wide \\& Deep",
        "emb_xgb": "Embeddings $\\rightarrow$ XGBoost"}


def _cv() -> pd.DataFrame:
    cv = pd.concat([pipeline.phase1_cv(), pd.read_csv(R / "cv_models.csv")], ignore_index=True)
    cv = cv[cv["model"].isin(NAME)].copy()
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
                                                "best_epochs": "Best epochs"})[["", "Run", "CV AP", "Best epochs"]], 4)
    if (R / "ablation.csv").exists():
        a = pd.read_csv(R / "ablation.csv").sort_values("delta")
        out["ablation"] = (a.rename(columns={"label": "Ablation", "cv": "CV AP", "delta": "$\\Delta$",
                                             "folds_improved": "Folds up", "seed_sd": "Seed SD"})
                           [["Ablation", "CV AP", "$\\Delta$", "Folds up", "Seed SD"]], 4)
    if (R / "calibrator_selection.csv").exists():
        c = pd.read_csv(R / "calibrator_selection.csv")
        piv = c.pivot(index="calibrator", columns="model", values="check_ece")
        piv = piv[[m for m in evaluation.MODELS if m in piv.columns]]
        piv.columns = [NAME[evaluation.CV_NAME[m]] for m in piv.columns]
        out["calibrator_selection"] = (piv.reset_index().rename(columns={"calibrator": "Oct ECE"}), 3)
    if (R / "test_metrics.csv").exists():
        m = pd.read_csv(R / "test_metrics.csv")
        h = m[(m.split == "test") & m.rule_selected].copy()
        h["Model"] = h["model"].map(lambda k: NAME[evaluation.CV_NAME[k]])
        out["results_test"] = (h.rename(columns={"calibration": "Calibration", "pr_auc": "AP", "roc_auc": "ROC",
                                                 "brier": "Brier", "ece": "ECE", "precision_at_thr": "P@thr",
                                                 "recall_at_thr": "R@thr"})
                               [["Model", "Calibration", "AP", "ROC", "Brier", "ECE", "P@thr", "R@thr"]], 3)
        pol = pd.read_csv(R / "test_policies.csv")
        p = pol[pol.policy == "daily top 10%"].copy()
        p["Model"] = p["model"].map(lambda k: NAME[evaluation.CV_NAME[k]])
        out["policy_top10"] = (p.rename(columns={"precision": "Precision", "recall": "Recall", "lift": "Lift",
                                                 "f1": "F1"})[["Model", "Precision", "Recall", "Lift", "F1"]], 3)
    if (R / "reschedule_test.csv").exists():
        r = pd.read_csv(R / "reschedule_test.csv")
        out["reschedule"] = (r.rename(columns={"rule": "Rule", "cost": "Cost/move", "moves_per_day": "Moves/day",
                                               "predicted_delays_avoided_per_day": "Avoided/day",
                                               "avoided_per_move": "Avoided/move", "net_benefit": "Net"})
                             [["Rule", "Cost/move", "Moves/day", "Avoided/day", "Avoided/move", "Net"]], 3)
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
        df.to_csv(tabs / f"{name}.csv", index=False)
        (tabs / f"{name}.tex").write_text(to_booktabs(df, dec))
    return {"figures": copied, "tables": sorted(tables)}
