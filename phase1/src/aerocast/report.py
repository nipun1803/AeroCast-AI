"""Export figures and tables for the written report (IEEE LaTeX, written by hand).

Copies the figures that ``report/main.tex`` uses into ``report/figures/`` and
writes its tables to ``report/tables/`` as CSV plus a LaTeX-ready ``booktabs``
``tabular`` snippet (requires ``\\usepackage{booktabs}``). Both folders are
cleared first, so they hold exactly what the report uses. No document body is
generated. Every table is read from ``results/``, never recomputed.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from . import config

REPORT_DIR = config.ROOT / "report"
# figures/ files included by report/main.tex (tests check the two lists agree)
REPORT_FIGURES = ("eda_rate_by_time", "eda_congestion_within_hour", "eda_history_corr_heatmap",
                  "pr_roc_test", "confusion_logreg_test", "reliability_logreg_test",
                  "monthly_drift_final", "perm_importance_groups")

_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
                "−": "$-$", "→": r"$\rightarrow$", "±": r"$\pm$"}


def tex_escape(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in str(text))


def _cell(v, decimals: int) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return "--"
        if v in (float("inf"), float("-inf")):
            return r"$\infty$" if v > 0 else r"$-\infty$"
        return f"{v:.{decimals}f}".replace("-", "$-$")
    if pd.isna(v):
        return "--"
    return tex_escape(str(v).replace(">=", "≥").replace("<=", "≤"))


def _head(c) -> str:
    """Headers containing math ($) or an explicit backslash are already LaTeX."""
    c = str(c)
    return c if ("$" in c or "\\" in c) else tex_escape(c)


def to_booktabs(df: pd.DataFrame, decimals: int = 3) -> str:
    """A ``tabular`` with booktabs rules: text columns left, numbers right."""
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_head(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_cell(v, decimals) for v in row) + r" \\"
              for row in df.itertuples(index=False, name=None)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# friendlier column headers for the .tex snippets (the CSVs keep the raw names)
TEX_HEADERS = {
    "model": "Model", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "brier": "Brier", "ece": "ECE",
    "ece_nov": "ECE Nov", "ece_dec": "ECE Dec", "threshold": "Thr.",
    "precision_at_thr": "Prec.@thr", "recall_at_thr": "Rec.@thr",
    "cv_pr_auc_mean": "Mean", "cv_pr_auc_std": "SD", "fold_m5": "May", "fold_m6": "Jun",
    "fold_m7": "Jul", "fold_m8": "Aug", "group": "Group", "n": "$n$", "content": "Content",
    "step": "Step", "cv_gain_mean": "CV gain", "cv_folds_improved": "Folds",
    "test_diff_ci": "Test diff. [95\\% CI]", "top_share": "$k$", "flagged_per_day": "Per day",
    "lr_precision": "LR P", "lr_recall": "LR R", "lr_lift": "LR lift", "xgb_precision": "XGB P",
    "route_precision": "Route P", "prior_precision": "Prior P", "calibrator": "Calibrator",
    "check_ece": "Oct ECE", "check_brier": "Oct Brier", "gain_vs_raw": "ECE gain",
    "policy": "Policy", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1",
}

FEATURE_GROUPS = {   # group -> (report name, content); counts come from results/summary.json
    "categorical": ("Identifiers", "carrier, origin, destination"),
    "schedule": ("Schedule", "cyclic departure and arrival time of day, cyclic weekday, departure "
                             "hour, weekend flag, days to holiday (cap 14), log distance"),
    "congestion": ("Congestion", "scheduled flights at the origin in the hour, the ±1 h window and "
                                 "the day; destination and carrier daily volumes"),
    "history_entity": ("Entity history", "smoothed delay rate and log-volume of origin (7 d), "
                                         "carrier (7, 14 d), destination, route, carrier-route, "
                                         "origin-hour, flight number and tail number (14 d)"),
    "history_day": ("Day history", "network delay rate (1, 7 d); origin delay rate, volume and "
                                   "mean delay (1 d); network and origin cancellation rates (1 d)"),
}
STEP_NAMES = {"+ congestion": "+ congestion", "+ history_entity": "+ entity history",
              "+ history_day": "+ day history"}


def _tex_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: TEX_HEADERS.get(c, c))


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS_DIR / name)


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def report_tables() -> dict[str, pd.DataFrame]:
    """name -> table, selected and ordered from the saved results (one per \\input in main.tex)."""
    summary = json.loads((config.RESULTS_DIR / "summary.json").read_text())
    cv = _read("cv_summary.csv")
    cv["model"] = cv["variant"].map({"prior": "Prior (base rate)", "route_history": "Route history",
                                     "logreg": "LogReg", "xgboost": "XGBoost (fixed trees)",
                                     "xgboost_infold_early_stopping": "XGBoost (in-fold, not used)"})
    head = _read("headline_test.csv")
    month = _read("headline_by_month.csv").pivot(index="variant", columns="split", values="ece")
    head["ece_nov"] = head["variant"].map(month["test_m11"])
    head["ece_dec"] = head["variant"].map(month["test_m12"])
    steps = _read("ablation_steps.csv")
    steps["step"] = steps["step"].map(STEP_NAMES)
    steps["test_diff_ci"] = [f"{_signed(d)} [{_signed(lo)}, {_signed(hi)}]" for d, lo, hi in
                             zip(steps.test_diff, steps.test_diff_ci_lo, steps.test_diff_ci_hi)]
    pol = _read("operating_policies.csv")
    pol = pol[pol.split == "test"].assign(
        model=lambda d: d["variant"].map({"prior": "Prior (base rate)", "route_history": "Route history",
                                          "logreg_rule_selected": "LogReg (Platt)",
                                          "xgboost_rule_selected": "XGBoost (raw)"}),
        policy=lambda d: d["role"].map({"secondary": "Frozen threshold", "primary": "Daily top 10%"}))
    res = _read("rescheduling_candidates.csv")
    by = {v: res[res.variant == v].set_index("top_share") for v in res.variant.unique()}
    lr = by["logreg_rule_selected"]
    resched = pd.DataFrame({
        "top_share": [f"{round(100 * k)}%" for k in lr.index],
        "flagged_per_day": lr["flagged_per_day"].round().astype(int).to_numpy(),
        "lr_precision": lr["precision"].to_numpy(), "lr_recall": lr["recall"].to_numpy(),
        "lr_lift": lr["lift"].to_numpy(),
        "xgb_precision": by["xgboost_rule_selected"]["precision"].to_numpy(),
        "route_precision": by["route_history"]["precision"].to_numpy(),
        "prior_precision": by["prior"]["precision"].to_numpy()})
    return {
        "feature_groups": pd.DataFrame(
            [{"group": FEATURE_GROUPS[g][0], "n": n, "content": FEATURE_GROUPS[g][1]}
             for g, n in summary["feature_groups"].items()]),
        "calibrator_selection": _read("calibrator_selection.csv").assign(
            model=lambda d: d["model"].map({"logreg": "LogReg", "xgboost": "XGBoost"}),
            calibrator=lambda d: d["calibrator"].map({"raw": "raw", "platt": "Platt",
                                                      "isotonic": "isotonic"}))[
            ["model", "calibrator", "check_ece", "check_brier", "gain_vs_raw"]],
        "cv_summary": cv[["model", "cv_pr_auc_mean", "cv_pr_auc_std",
                          "fold_m5", "fold_m6", "fold_m7", "fold_m8"]],
        "results_test": head[["model", "pr_auc", "roc_auc", "brier", "ece", "ece_nov", "ece_dec",
                              "threshold", "precision_at_thr", "recall_at_thr"]],
        "threshold_metrics": pol[["model", "policy", "accuracy", "precision", "recall", "f1"]],
        "ablation_steps": steps[["model", "step", "cv_gain_mean", "cv_folds_improved",
                                 "test_diff_ci"]],
        "rescheduling": resched,
    }


def export(decimals: int = 3) -> dict[str, list[str]]:
    """Copy the report's figures and write its tables, after clearing both
    output folders; raises if a source artefact is missing."""
    figs_out, tabs_out = REPORT_DIR / "figures", REPORT_DIR / "tables"
    for folder, patterns in [(figs_out, ["*.png"]), (tabs_out, ["*.csv", "*.tex"])]:
        folder.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for old in folder.glob(pattern):
                old.unlink()
    for name in REPORT_FIGURES:
        src = config.FIGURES_DIR / f"{name}.png"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found - run the pipeline and notebooks first")
        shutil.copy2(src, figs_out / src.name)
    tables = report_tables()
    for name, df in tables.items():
        df.to_csv(tabs_out / f"{name}.csv", index=False)
        tex = to_booktabs(_tex_df(df), decimals)
        if "content" in df.columns:   # long text column: let it wrap
            tex = tex.replace(r"\begin{tabular}{lrl}", r"\begin{tabular}{@{}lrp{5.3cm}@{}}", 1)
        (tabs_out / f"{name}.tex").write_text(
            f"% generated by scripts/export_report_assets.py from results/ -- do not edit\n"
            f"% needs \\usepackage{{booktabs}}\n{tex}")
    return {"figures": [f"{n}.png" for n in REPORT_FIGURES], "tables": list(tables)}
