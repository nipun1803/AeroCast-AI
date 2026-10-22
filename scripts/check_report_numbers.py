"""Assert that the numbers quoted in report/main.tex prose match results/.

Each check formats a value computed from results/ and requires that string to appear in the
report text. A failure means the prose drifted from the saved results (or vice versa).
Usage: python scripts/check_report_numbers.py
"""
from __future__ import annotations

import json
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
pol = csv("operating_policies.csv")
cv = csv("cv_summary.csv")
summ = json.loads((R / "summary.json").read_text())
cal = csv("calibration_study.csv")
resched = csv("rescheduling_candidates.csv")

LR = "LogReg (Platt, rule-selected)"
XG = "XGBoost (raw, rule-selected)"
LRR = "LogReg (raw)"
checks: list[tuple[str, float, int]] = []  # (name, value, decimals)


def add(name, value, decimals=3):
    checks.append((name, float(value), decimals))


for model, tag in [(LR, "LogReg"), (XG, "XGBoost"), ("Route history", "route"), ("Prior (base rate)", "prior")]:
    add(f"test PR-AUC {tag}", pick(head, model=model).pr_auc)
add("test ROC-AUC LogReg", pick(head, model=LR).roc_auc)
add("test ECE LogReg Platt", pick(head, model=LR).ece)
add("test ECE LogReg raw", pick(head, model=LRR).ece)
add("test ECE LogReg rolling", pick(head, model="LogReg (rolling Platt 28d)").ece)
add("test ECE XGBoost rolling", pick(head, model="XGBoost (rolling Platt 28d)").ece)

prim = pol[(pol.split == "test") & (pol.role == "primary")]
add("primary precision LogReg", pick(prim, model=LR).precision)
add("primary recall LogReg", pick(prim, model=LR).recall)
add("primary lift LogReg", pick(prim, model=LR).lift, 2)
add("primary lift XGBoost", pick(prim, model=XG).lift, 2)
add("primary lift route", pick(prim, model="Route history").lift, 2)
add("primary lift prior", pick(prim, model="Prior (base rate)").lift, 2)

cv_lr = cv[cv.model.str.startswith("LogReg")].iloc[0]
cv_xg = cv[cv.model.str.startswith("XGBoost")].iloc[0]
add("CV PR-AUC LogReg", cv_lr.cv_pr_auc_mean, 4)
add("CV PR-AUC XGBoost", cv_xg.cv_pr_auc_mean, 4)
add("CV PR-AUC route", pick(cv, model="Route history").cv_pr_auc_mean, 4)
add("CV PR-AUC prior", pick(cv, model="Prior (base rate)").cv_pr_auc_mean, 4)

boot = summ["bootstrap_test"]["pr_auc"]["diff_xgboost_minus_logreg_ci95"]
add("bootstrap CI low", boot[0])
add("bootstrap CI high", boot[1])
add("split base rate test", summ["splits"]["test"]["delay_rate"])

add("Dec ECE static Platt", pick(cal, variant="logreg_platt").ece_test_m12)
add("Nov ECE static Platt", pick(cal, variant="logreg_platt").ece_test_m11)

for share in (0.05, 0.10, 0.20):  # the k values quoted in the prose
    r = pick(resched, model=LR, top_share=share)
    add(f"resched precision LogReg {share:.0%}", r.precision)
    add(f"resched lift LogReg {share:.0%}", r.lift, 2)

n_days = 61  # Nov + Dec test days
for share in (0.05, 0.10, 0.20):
    for name, who in [(LR, "LogReg"), ("Route history", "route"), ("Prior (base rate)", "random")]:
        r = pick(resched, model=name, top_share=share)
        add(f"resched delayed/day {who} {share:.0%}", r.delayed_captured / n_days, 0)
add("test flights per day", summ["splits"]["test"]["rows"] / n_days, 0)

bad = []
for name, value, dec in checks:
    s = f"{value:.{dec}f}"
    candidates = {s, s.replace("-", "$-$"), s.lstrip("0") if dec else s}
    if value < 0:
        candidates |= {f"-{abs(value):.{dec}f}", f"$-${abs(value):.{dec}f}"}
    if not any(c in TEXT or c in TEX for c in candidates):
        bad.append(f"  {name}: results say {s}, not found in report/main.tex")
print(f"{len(checks) - len(bad)}/{len(checks)} quoted numbers match results/")
if bad:
    print("\n".join(bad))
    sys.exit(1)
