"""Assert that the numbers quoted in report/main.tex prose match results/.

Each check formats a value read from results/ and requires that string to appear in the
report text. A failure means the prose drifted from the saved results (or vice versa).
Usage: python scripts/check_report_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "phase1" / "src"):
    sys.path.insert(0, str(p))

import aerocast_dl  # noqa: E402,F401  (torch before XGBoost)
import pandas as pd  # noqa: E402

from aerocast_dl import config, pipeline  # noqa: E402

R = config.RESULTS_DIR
TEX = (ROOT / "report" / "main.tex").read_text()
TEXT = TEX.replace("{,}", "").replace("\\%", "%")


def csv(name):
    return pd.read_csv(R / name)


def pick(df, **where):
    for k, v in where.items():
        df = df[df[k] == v]
    assert len(df) == 1, (where, len(df))
    return df.iloc[0]


checks: list[tuple[str, float, int, bool]] = []  # (name, value, decimals, as percentage)


def add(name, value, decimals=3):
    checks.append((name, float(value), decimals, False))


def pct(name, value):
    checks.append((name, float(value), 1, True))


# --- cross-validation (Phase 1 references and the four Phase 2 families)
cv = pd.concat([pipeline.phase1_cv(), csv("cv_models.csv")], ignore_index=True)
for m in ["Phase 1 LogReg", "Phase 1 XGBoost", "mlp", "gru", "wide_deep", "emb_xgb"]:
    add(f"CV AP {m}", pick(cv, model=m).cv_pr_auc_mean)
for m in ["gru", "wide_deep"]:
    add(f"parameters {m}", pick(cv, model=m).n_params, 0)
add("seed SD wide_deep", pick(cv, model="wide_deep").seed_sd_mean_over_folds)
sel = json.loads((R / "selection.json").read_text())
add("LogReg parameters", sel["complexity"]["logreg"], 0)

# --- diagnostics (why the networks trail LogReg)
diag = csv("diagnostics.csv")
for d in ["D0", "D1", "D3", "D4", "D5", "D6"]:
    add(f"diagnostic {d}", pick(diag, diagnostic=d).cv_pr_auc, 4)

# --- ablations (delta vs base, 5-seed ensembles)
abl = csv("ablation.csv")
for a in ["mlp_no_es", "mlp_no_reg", "mlp_minus_history_entity", "mlp_minus_congestion",
          "mlp_minus_history_day", "gru_instead_of_history", "mlp_onehot", "wd_no_gru", "wd_onehot"]:
    add(f"ablation delta {a}", pick(abl, ablation=a).delta)
for a in ["wd_no_gru", "gru_instead_of_history"]:
    add(f"ablation CV {a}", pick(abl, ablation=a).cv)

# --- calibration (selected on October)
calsel = csv("calibrator_selection.csv")
add("Oct ECE LogReg raw", pick(calsel, model="logreg", calibrator="raw").check_ece)
p1h = pd.read_csv(ROOT.parent / "phase1" / "results" / "headline_test.csv").set_index("variant")
add("Phase 1 deployed LogReg test AP", p1h.loc["logreg_rule_selected", "pr_auc"])
add("Phase 1 deployed LogReg test ECE", p1h.loc["logreg_rule_selected", "ece"])
assert abs(p1h.loc["logreg_rolling28", "pr_auc"] - pd.read_csv(R / "test_metrics.csv").query(
    "model == 'logreg' and split == 'test' and rule_selected").pr_auc.iloc[0]) < 1e-9   # same scores
add("Oct ECE LogReg chosen", pick(calsel, model="logreg", calibrator=sel["calibrator"]["logreg"]).check_ece)
assert sel["final_model"] == "logreg" and sel["reschedule"]["params"] == {"window": 2, "delta": 0.02}
p1sum = json.loads((ROOT.parent / "phase1" / "results" / "summary.json").read_text())
for s_ in ("train", "calib", "test"):
    pct(f"{s_} base rate", p1sum["splits"][s_]["delay_rate"])
add("fit rows", json.loads((R / "data_summary.json").read_text())["fit_rows"], 0)
add("rows excluded by the clock", json.loads((R / "data_summary.json").read_text())["fit_rows_excluded_by_clock"], 0)

# --- the one test evaluation
m = csv("test_metrics.csv")
t = m[(m.split == "test") & m.rule_selected]
for k in ["logreg", "xgboost", "mlp", "gru", "wide_deep", "emb_xgb"]:
    add(f"test AP {k}", pick(t, model=k).pr_auc)
    add(f"test ECE {k}", pick(t, model=k).ece)
for sp in ("test_m11", "test_m12"):
    add(f"LogReg ECE {sp}", pick(m[m.rule_selected], model="logreg", split=sp).ece)
pct("December base rate", pick(m[m.rule_selected], model="logreg", split="test_m12").base_rate)
add("LogReg precision at frozen threshold", pick(t, model="logreg").precision_at_thr)
boot = json.loads((R / "test_bootstrap.json").read_text())
for k in ("mlp", "gru", "wide_deep"):
    b = boot[k]["pr_auc"]
    add(f"bootstrap {k} AP diff", b["diff"])
    add(f"bootstrap {k} CI low", b["ci_lo"])
    add(f"bootstrap {k} CI high", b["ci_hi"])
pol = csv("test_policies.csv")
p = pol[pol.policy == "daily top 10%"]
for k in p["model"]:
    add(f"top10 precision {k}", pick(p, model=k).precision)

# --- slices and drift
sl = csv("test_slices.csv")
add("GRU gain 05-09", pick(sl, dimension="dep_hour", slice="05-09").ap_gain_gru)
add("W&D gain rare destinations", pick(sl, dimension="dest_train_rows", slice="< 50").ap_gain_wide_deep)
add("rare-destination test flights", pick(sl, dimension="dest_train_rows", slice="< 50").n, 0)
car = sl[sl.dimension == "carrier"]
assert (car.ap_gain_wide_deep > 0).sum() == 11 and (car.ap_gain_gru > 0).sum() == 13
assert (car.ap_gain_xgboost > 0).sum() == 15 and len(car) == 16
dr = csv("test_drift.csv").set_index("input")
add("max network/LogReg score PSI on test",
    dr.loc[[f"score: {k}" for k in ("logreg", "mlp", "gru", "wide_deep")], "test"].max())

# --- rescheduling
tun = csv("reschedule_tuning.csv")
rs = csv("reschedule_test.csv")
fr, nv = rs[rs.rule.str.startswith("frozen")].iloc[0], rs[rs.rule.str.startswith("naive")].iloc[0]
add("frozen moves/day", fr.moves_per_day, 1)
add("frozen avoided/day", fr.predicted_delays_avoided_per_day, 1)
add("naive moves/day", nv.moves_per_day, 1)
add("naive avoided/day", nv.predicted_delays_avoided_per_day, 1)
assert (rs[rs.rule.str.startswith("frozen")].set_index("cost").net_benefit.loc[[0.02, 0.05]] * [1, -1] > 0).all()
assert (rs[rs.rule.str.startswith("naive")].set_index("cost").net_benefit.loc[[0.02, 0.05]] < 0).all()

bad = []
for name, value, dec, as_pct in checks:
    s = f"{100 * value:.{dec}f}%" if as_pct else f"{value:.{dec}f}"
    candidates = {s, s.replace("-", "$-$"), f"+{s}", f"$+${s}"}
    if value < 0 and not as_pct:
        candidates |= {f"-{abs(value):.{dec}f}", f"$-${abs(value):.{dec}f}", f"−{abs(value):.{dec}f}"}
    if dec == 0 and abs(value) >= 1000:
        candidates.add(f"{value:,.0f}")
    if not any(c in TEXT or c in TEX for c in candidates):
        bad.append(f"  {name}: results say {s}, not found in report/main.tex")
print(f"{len(checks) - len(bad)}/{len(checks)} quoted numbers match results/")
if bad:
    print("\n".join(bad))
    sys.exit(1)
