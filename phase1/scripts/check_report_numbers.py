"""Assert that the numbers quoted in report/main.tex prose match results/.

Each check formats a value computed from results/ and requires that string to appear in the
report text. A failure means the prose drifted from the saved results (or vice versa).
Numbers that come from the raw data (EDA counts, leakage-test cell counts) are not in results/;
notebooks 01 and 02 compute and show them.
Usage: python scripts/check_report_numbers.py
"""
from __future__ import annotations

import json
import math
import sys

import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from aerocast import config  # noqa: E402

R = config.RESULTS_DIR
TEX = (config.ROOT / "report" / "main.tex").read_text()
TEXT = TEX.replace("{,}", "").replace("\\%", "%")


def csv(name):
    return pd.read_csv(R / name)


def pick(df, **where):
    for k, v in where.items():
        df = df[df[k] == v]
    assert len(df) == 1, (where, len(df))
    return df.iloc[0]


head = csv("headline_test.csv")
month = csv("headline_by_month.csv")
pol = csv("operating_policies.csv")
cv = csv("cv_summary.csv")
summ = json.loads((R / "summary.json").read_text())
cal = csv("calibration_study.csv")
calsel = csv("calibrator_selection.csv")
tuning = csv("history_tuning.csv")
mcv = csv("month_feature_cv.csv")
steps = csv("ablation_steps.csv")
perm = csv("permutation_importance_groups.csv")
sens = csv("target_sensitivity.csv")
resched = csv("rescheduling_candidates.csv")
psi = csv("psi.csv")

LR = "logreg_rule_selected"
XG = "xgboost_rule_selected"
checks: list[tuple[str, float, int, bool]] = []  # (name, value, decimals, as percentage)


def add(name, value, decimals=3):
    checks.append((name, float(value), decimals, False))


def pct(name, value):
    checks.append((name, float(value), 1, True))


# --- test metrics (Table IV prose)
for v, tag in [(LR, "LogReg"), (XG, "XGBoost"), ("route_history", "route"), ("prior", "prior")]:
    add(f"test PR-AUC {tag}", pick(head, variant=v).pr_auc)
    add(f"test ROC-AUC {tag}", pick(head, variant=v).roc_auc)
for v in ["logreg_raw", LR, "logreg_rolling28", "xgboost_rolling28"]:
    add(f"test ECE {v}", pick(head, variant=v).ece)
for v in ["logreg_raw", LR, "logreg_rolling28"]:
    for sp in ("test_m11", "test_m12"):
        add(f"ECE {v} {sp}", pick(month, variant=v, split=sp).ece)
for n in (14, 56):
    add(f"test ECE logreg rolling {n}d", pick(cal, variant=f"logreg_rolling{n}").ece_test)
lr = pick(head, variant=LR)
add("frozen threshold LogReg", lr.threshold)
add("frozen threshold test precision", lr.precision_at_thr)
add("frozen threshold test recall", lr.recall_at_thr)
pct("frozen threshold test flag rate", lr.flag_rate)
pct("frozen threshold calib flag rate",
    pick(pol, variant=LR, split="calib", role="secondary").flag_rate)

# --- accuracy, precision, recall, F1 and confusion counts at the two operating points
test_pol = pol[pol.split == "test"]
frozen, top = test_pol[test_pol.role == "secondary"], test_pol[test_pol.role == "primary"]
pct("accuracy of predicting every flight on time", pick(frozen, variant="prior").accuracy)
for policy, rows in [("frozen", frozen), ("top10", top)]:
    r = pick(rows, variant=LR)
    add(f"LogReg {policy} accuracy", r.accuracy)
    add(f"LogReg {policy} F1", r.f1)
    add(f"LogReg {policy} TP", r.tp, 0)
add("LogReg frozen FN", pick(frozen, variant=LR).fn, 0)
add("LogReg frozen FP", pick(frozen, variant=LR).fp, 0)
add("late test flights", pick(frozen, variant=LR).tp + pick(frozen, variant=LR).fn, 0)
add("XGBoost frozen precision", pick(frozen, variant=XG).precision)
add("XGBoost frozen recall", pick(frozen, variant=XG).recall)
for v in [XG, "route_history", "prior"]:
    add(f"{v} top10 F1", pick(top, variant=v).f1)
boot = summ["bootstrap_test"]["pr_auc"]["diff_xgboost_minus_logreg_ci95"]
add("bootstrap CI low", boot[0])
add("bootstrap CI high", boot[1])

# --- primary policy and review desk
prim = pol[(pol.split == "test") & (pol.role == "primary")]
add("primary precision LogReg", pick(prim, variant=LR).precision)
add("primary recall LogReg", pick(prim, variant=LR).recall)
add("primary lift LogReg", pick(prim, variant=LR).lift, 2)
add("primary lift route", pick(prim, variant="route_history").lift, 2)
for share in (0.05, 0.20):
    r = pick(resched, variant=LR, top_share=share)
    add(f"desk precision {share:.0%}", r.precision)
    add(f"desk lift {share:.0%}", r.lift, 2)
    add(f"desk flagged/day {share:.0%}", r.flagged_per_day, 0)
add("desk recall 20%", pick(resched, variant=LR, top_share=0.20).recall)
n_days = 61  # Nov + Dec test days
for v, who in [(LR, "LogReg"), ("route_history", "route"), ("prior", "random")]:
    add(f"desk delayed/day {who} 10%", pick(resched, variant=v, top_share=0.10).delayed_captured / n_days, 0)
add("test flights per day", summ["splits"]["test"]["rows"] / n_days, 0)

# --- cross-validation and selection
for v in ["logreg", "xgboost", "xgboost_infold_early_stopping", "route_history", "prior"]:
    add(f"CV PR-AUC {v}", pick(cv, variant=v).cv_pr_auc_mean)
for i, d in enumerate(summ["cv_paired_logreg_minus_xgboost"]["diffs"]):
    add(f"paired CV difference fold {i}", d)
clock = tuning[(tuning.smoothing_m == config.SMOOTHING_M) & (tuning.long_window == config.LONG_WINDOW)]
add("CV clock 24:00", pick(clock, decision_hour=24).cv_pr_auc_mean)
add("CV clock 00:00", pick(clock, decision_hour=0).cv_pr_auc_mean)
add("CV gain of the 24:00 clock",
    pick(clock, decision_hour=24).cv_pr_auc_mean - pick(clock, decision_hour=20).cv_pr_auc_mean)
add("CV with month features LogReg", pick(mcv, features="with month", model="logreg").cv_pr_auc_mean)
add("calibration gain LogReg Platt", pick(calsel, model="logreg", calibrator="platt").gain_vs_raw)
add("calibration gain XGBoost (best)", calsel[calsel.model == "xgboost"].gain_vs_raw.max())
add("XGBoost trees", summ["xgb_params"]["n_estimators"], 0)
add("XGBoost min child weight", summ["xgb_params"]["min_child_weight"], 0)
add("number of features", summ["n_features"], 0)

# --- feature groups, ablation, importance
for step in ["+ congestion", "+ history_entity", "+ history_day"]:
    add(f"LogReg CV gain {step}", pick(steps, model="LogReg", step=step).cv_gain_mean)
ent = pick(steps, model="LogReg", step="+ history_entity")
for k in ("test_diff", "test_diff_ci_lo", "test_diff_ci_hi"):
    add(f"LogReg entity-history {k}", ent[k])
for g in ["schedule", "congestion", "categorical", "history_entity", "history_day"]:
    add(f"permutation drop LogReg {g}", pick(perm, model="LogReg", group=g).pr_auc_drop)

# --- splits, fitting clock, calibration shift, robustness
for s in ("train", "calib", "test"):
    add(f"{s} rows", summ["splits"][s]["rows"], 0)
    pct(f"{s} base rate", summ["splits"][s]["delay_rate"])
for sp in ("test_m11", "test_m12"):
    pct(f"{sp} base rate", pick(month, variant="prior", split=sp).base_rate)
add("train labels excluded", summ["fitting_clock"]["train_labels_excluded"], 0)
add("calibration labels excluded", summ["fitting_clock"]["calibration_labels_excluded"], 0)
logit = lambda p: math.log(p / (1 - p))  # noqa: E731
add("label-shift logit Sep-Oct to Dec",
    logit(pick(month, variant="prior", split="test_m12").base_rate)
    - logit(summ["splits"]["calib"]["delay_rate"]), 1)
alt = sens[sens.target.str.contains("OR")]
pct("sensitivity base rate", alt.base_rate.iloc[0])
add("sensitivity PR-AUC LogReg", pick(alt, model="LogReg (raw)").pr_auc)
add("sensitivity PR-AUC XGBoost", pick(alt, model="XGBoost (raw)").pr_auc)
add("sensitivity lift LogReg", pick(alt, model="LogReg (raw)").top10_lift, 2)
add("sensitivity lift prior", pick(alt, model="Prior (base rate)").top10_lift, 2)
add("PSI days_to_holiday", pick(psi, feature="days_to_holiday").psi_test, 1)

bad = []
for name, value, dec, as_pct in checks:
    s = f"{100 * value:.{dec}f}%" if as_pct else f"{value:.{dec}f}"
    candidates = {s, s.replace("-", "$-$"), s.lstrip("0") if dec else s}
    if value < 0 and not as_pct:
        candidates |= {f"-{abs(value):.{dec}f}", f"$-${abs(value):.{dec}f}"}
    if not any(c in TEXT or c in TEX for c in candidates):
        bad.append(f"  {name}: results say {s}, not found in report/main.tex")
print(f"{len(checks) - len(bad)}/{len(checks)} quoted numbers match results/")
if bad:
    print("\n".join(bad))
    sys.exit(1)
