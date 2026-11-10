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
for m in ["mlp", "gru", "wide_deep"]:
    add(f"mean best epoch {m}", pick(cv, model=m).mean_best_epoch, 1)
    add(f"parameters {m}", pick(cv, model=m).n_params, 0)
add("seed SD wide_deep", pick(cv, model="wide_deep").seed_sd_mean_over_folds)

# --- ablations (delta vs base, 5-seed ensembles)
abl = csv("ablation.csv")
for a in ["mlp_onehot", "mlp_no_es", "mlp_minus_congestion", "mlp_minus_history_entity",
          "mlp_minus_history_day", "gru_instead_of_history", "wd_no_gru", "wd_onehot", "mlp_ids"]:
    add(f"ablation delta {a}", pick(abl, ablation=a).delta)
for a in ["wd_no_gru", "gru_instead_of_history"]:
    add(f"ablation CV {a}", pick(abl, ablation=a).cv)

# --- diagnostics (why the MLP trails LogReg)
diag = csv("diagnostics.csv")
for d in diag["diagnostic"]:
    add(f"diagnostic {d}", pick(diag, diagnostic=d).cv_pr_auc, 4)

# --- final-model rule and calibration (pre-test)
sel = json.loads((R / "selection.json").read_text())
calsel = csv("calibrator_selection.csv")
fin = sel["final_model"]
raw = pick(calsel, model=fin, calibrator="raw").check_ece
add("Oct ECE final model raw", raw)
add("Oct ECE final model chosen", pick(calsel, model=fin, calibrator=sel["calibrator"][fin]).check_ece)

# --- the one test evaluation
if (R / "test_metrics.csv").exists():
    m = csv("test_metrics.csv")
    t = m[(m.split == "test") & m.rule_selected]
    for k in t["model"]:
        r = pick(t, model=k)
        add(f"test AP {k}", r.pr_auc)
        add(f"test ECE {k}", r.ece)
    boot = json.loads((R / "test_bootstrap.json").read_text())
    for k in ("mlp", "gru", "wide_deep"):  # paired AP difference vs Phase 1 LogReg
        b = boot[k]["pr_auc"]
        add(f"bootstrap {k} AP diff", b["diff"])
        add(f"bootstrap {k} CI low", b["ci_lo"])
        add(f"bootstrap {k} CI high", b["ci_hi"])
    pol = csv("test_policies.csv")
    p = pol[pol.policy == "daily top 10%"]
    for k in p["model"]:
        add(f"top10 precision {k}", pick(p, model=k).precision)
if (R / "reschedule_test.csv").exists():
    rs = csv("reschedule_test.csv")
    for rule in rs["rule"]:
        r = pick(rs, rule=rule)
        add(f"reschedule {rule} moves/day", r.moves_per_day, 1)
        add(f"reschedule {rule} avoided/day", r.predicted_delays_avoided_per_day, 1)

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
