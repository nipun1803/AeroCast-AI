"""Export figures and tables for the written report (IEEE LaTeX, written by hand).

Copies the final PNG figures into ``report/figures/`` and writes the headline,
policy, calibration, ablation, target-sensitivity and CV tables to
``report/tables/`` as CSV plus a LaTeX-ready ``booktabs`` ``tabular`` snippet
(requires ``\\usepackage{booktabs}``). No document body is generated.
Every table is read from ``results/``, never recomputed.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from . import config

REPORT_DIR = config.ROOT / "report"

_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
                "−": "-", "→": r"$\rightarrow$"}


def tex_escape(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in str(text))


def _cell(v, decimals: int) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return "--"
        if v in (float("inf"), float("-inf")):
            return r"$\infty$" if v > 0 else r"$-\infty$"
        return f"{v:.{decimals}f}"
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
    "model": "Model", "calibration": "Calibration", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
    "brier": "Brier", "ece": "ECE", "ece_equal_mass": "ECE (eq.-mass)",
    "oracle_recall@p0.40": "Oracle rec.@0.40", "split": "Split", "role": "Role",
    "policy": "Policy", "flag_rate": "Flag rate", "precision": "Precision", "recall": "Recall",
    "lift": "Lift", "threshold": "Threshold", "features": "Features", "n_features": "#feat.",
    "cv_pr_auc_mean": "CV PR-AUC", "cv_pr_auc_std": "CV sd", "test_pr_auc": "Test PR-AUC",
    "test_roc_auc": "Test ROC-AUC", "step": "Step", "cv_gain_mean": "CV gain",
    "cv_folds_improved": "Folds up", "test_diff": "Test diff.", "test_diff_ci_lo": "CI low",
    "test_diff_ci_hi": "CI high", "target": "Target", "base_rate": "Base rate",
    "top10_lift": "Top-10% lift", "ece_test": "ECE test", "ece_test_m11": "ECE Nov",
    "ece_test_m12": "ECE Dec", "brier_test": "Brier test", "fold_m5": "May", "fold_m6": "Jun",
    "fold_m7": "Jul", "fold_m8": "Aug", "rows": "Flights", "delay_rate": "Delay rate",
    "from": "From", "to": "To", "smoothing_m": "$m$", "long_window": "Window (d)",
    "decision_hour": "Clock (h)", "best_C": "Best $C$", "cv_pr_auc_mean_C0.001": "CV ($C{=}0.001$)",
    "C": "$C$", "cv_mean": "CV PR-AUC", "cv_std": "CV sd", "calibrator": "Calibrator",
    "check_ece": "Oct ECE", "check_ece_equal_mass": "Oct ECE (eq.-mass)",
    "check_brier": "Oct Brier", "gain_vs_raw": "ECE gain vs raw", "comparison": "Comparison",
    "adversarial_auc": "Adversarial AUC", "feature": "Feature", "group": "Group",
    "psi_calib": "PSI (Sep--Oct)", "psi_test": "PSI (test)", "metric": "Metric",
    "xgb_ci": "XGBoost 95\\% CI", "lr_ci": "LogReg 95\\% CI", "diff_ci": "XGB$-$LR 95\\% CI",
    "p_not_better": "P(XGB not better)", "n": "Flights", "top_share": "Top share",
    "flagged_per_day": "Flagged/day", "delayed_per_100_flagged": "Delayed per 100",
    "delayed_captured": "Delayed caught",
}


def _tex_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: TEX_HEADERS.get(c, c))


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS_DIR / name)


def report_tables() -> dict[str, pd.DataFrame]:
    """name -> table, selected and ordered from the saved results."""
    headline = _read("headline_test.csv")
    pol = _read("operating_policies.csv")
    pol = pol[pol.split == "test"].drop(columns=["split", "variant", "calibration", "role"])
    pol["policy"] = pol["policy"].str.replace(r"frozen thr.*", "frozen thr. (P>=0.4)", regex=True)
    abl = _read("ablation.csv")
    steps = _read("ablation_steps.csv")
    sens = _read("target_sensitivity.csv")
    splits = json.loads((config.RESULTS_DIR / "summary.json").read_text())
    boot = splits["bootstrap_test"]
    tuning = _read("history_tuning.csv")
    lr_cv = _read("cv_logreg.csv").groupby("C")["val_pr_auc"].agg(["mean", "std"]).reset_index()
    mcv = _read("month_feature_cv.csv")
    perm = _read("permutation_importance_groups.csv")
    resched = _read("rescheduling_candidates.csv")
    return {
        "split_summary": pd.DataFrame(splits["splits"]).T.rename_axis("split").reset_index()
        .rename(columns={"rows": "rows"}),
        "history_tuning": tuning[["smoothing_m", "long_window", "decision_hour", "best_C",
                                  "cv_pr_auc_mean", "cv_pr_auc_std", "cv_pr_auc_mean_C0.001"]],
        "logreg_c_grid": lr_cv.rename(columns={"mean": "cv_mean", "std": "cv_std"}),
        "calibrator_selection": _read("calibrator_selection.csv")[
            ["model", "calibrator", "check_ece", "check_ece_equal_mass", "check_brier",
             "gain_vs_raw"]],
        "month_feature_cv": mcv[["features", "model", "fold_m5", "fold_m6", "fold_m7", "fold_m8",
                                 "cv_pr_auc_mean"]],
        "permutation_importance": perm.pivot(index="group", columns="model",
                                             values="pr_auc_drop").reset_index(),
        "adversarial": _read("adversarial_validation.csv")[
            ["comparison", "features", "n_features", "adversarial_auc"]],
        "psi_top10": _read("psi.csv").head(10)[["feature", "group", "psi_calib", "psi_test"]],
        "bootstrap": pd.DataFrame([
            {"metric": {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}[k], "xgb_ci": str([round(x, 3) for x in v["xgboost_ci95"]]),
             "lr_ci": str([round(x, 3) for x in v["logreg_ci95"]]),
             "diff_ci": str([round(x, 3) for x in v["diff_xgboost_minus_logreg_ci95"]]),
             "p_not_better": v["p_xgboost_not_better"]} for k, v in boot.items()]),
        "headline_test": headline[["model", "calibration", "pr_auc", "roc_auc", "brier", "ece",
                                   "ece_equal_mass", "oracle_recall@p0.40"]],
        "headline_by_month": _read("headline_by_month.csv").assign(
            split=lambda d: d["split"].map({"test_m11": "Nov", "test_m12": "Dec"}))[
            ["model", "split", "pr_auc", "roc_auc", "brier", "ece"]],
        "operating_policies_test": pol,
        "calibration_study": _read("calibration_study.csv").drop(columns=["variant"]),
        "calibration_study_compact": _read("calibration_study.csv")[
            ["model", "ece_test", "ece_test_m11", "ece_test_m12", "brier_test"]],
        "ablation": abl,
        "ablation_steps": steps,
        "target_sensitivity": sens,
        "target_sensitivity_compact": sens[["target", "model", "base_rate", "pr_auc", "roc_auc",
                                            "top10_lift"]],
        "cv_summary": _read("cv_summary.csv").drop(columns=["variant"]),
        "rescheduling": resched[["model", "top_share", "flagged_per_day", "precision", "recall",
                                 "lift", "delayed_per_100_flagged"]].assign(
            top_share=lambda d: (d["top_share"] * 100).round().astype(int).astype(str) + "%"),
    }


def export(decimals: int = 3) -> dict[str, list[str]]:
    """Copy figures and write tables; raises if a source artefact is missing."""
    figs_out, tabs_out = REPORT_DIR / "figures", REPORT_DIR / "tables"
    figs_out.mkdir(parents=True, exist_ok=True)
    tabs_out.mkdir(parents=True, exist_ok=True)
    figures = sorted(config.FIGURES_DIR.glob("*.png"))
    if not figures:
        raise FileNotFoundError(f"no PNG figures in {config.FIGURES_DIR}")
    for f in figures:
        shutil.copy2(f, figs_out / f.name)
    tables = report_tables()
    for name, df in tables.items():
        df.to_csv(tabs_out / f"{name}.csv", index=False)
        (tabs_out / f"{name}.tex").write_text(
            f"% generated by scripts/export_report_assets.py from results/ -- do not edit\n"
            f"% needs \\usepackage{{booktabs}}\n{to_booktabs(_tex_df(df), decimals)}")
    return {"figures": [f.name for f in figures], "tables": list(tables)}
