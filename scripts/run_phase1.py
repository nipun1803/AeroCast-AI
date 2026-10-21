"""Phase 1 end-to-end pipeline (single source of truth for reported numbers).

history tuning (CV) -> features -> baselines -> rolling-origin CV (LogReg,
XGBoost) -> final-model rule -> calibrate on Sep-Oct (static rule + causal
rolling) -> Nov-Dec test -> policies -> bootstrap -> month CV -> ablation
(both models) -> importance -> target sensitivity -> covariate drift ->
figures.

Every stage lives in ``aerocast.pipeline``; this script only orchestrates and
writes artefacts. Notebooks 01-04 call the same functions.

Usage:  python scripts/run_phase1.py [--quick]
Outputs: results/*.json|csv|csv.gz, figures/*.png|pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `aerocast` importable without relying on the editable-install .pth file
# (macOS may flag it hidden in iCloud-synced folders, and Python then skips it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aerocast import config, data, pipeline, plots  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(quick: bool = False):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    R = config.RESULTS_DIR
    n_boot = 100 if quick else config.N_BOOTSTRAP

    raw = data.add_target(data.load_raw())

    # ------------------------------------------------- feature design (CV)
    if not quick:
        log("history smoothing / window tuning + decision-clock comparison (rolling CV)")
        tuning = pipeline.tune_history(raw)
        tuning.to_csv(R / "history_tuning.csv", index=False)
        history_choice = pipeline.check_history_choice(tuning)
    else:
        history_choice = {"smoothing_m": config.SMOOTHING_M, "long_window": config.LONG_WINDOW}

    log("features (full schedule incl. cancelled flights; decision clock "
        f"{config.DECISION_HOUR}:00 on D-1)")
    P = pipeline.prepare(raw=raw)
    log(f"splits: {json.dumps(P.split_summary())}")
    raw_preds = pipeline.baseline_preds(P)

    # --------------------------------------------------------------- models
    log("rolling-origin CV: logistic regression")
    lr = pipeline.train_logreg(P)
    lr.cv.to_csv(R / "cv_logreg.csv", index=False)
    raw_preds["logreg"] = pipeline.predict_splits(lr.model, P)
    log(f"best C={lr.C}")

    log("rolling-origin CV: XGBoost")
    xgb = pipeline.train_xgb(P, n_configs=4 if quick else 16)
    xgb.cv_folds.to_csv(R / "cv_xgb_folds.csv", index=False)
    xgb.cv_configs.to_csv(R / "cv_xgb_configs.csv")
    raw_preds["xgboost"] = pipeline.predict_splits(xgb.model, P)
    log(f"best xgb params {xgb.params}")

    cv = pipeline.cv_summary(lr, xgb, P)
    final_model = pipeline.select_final_model(cv)
    log(f"final model (CV rule, fixed before test): {final_model}  {cv}")

    pipeline.predictions_frame(P, raw_preds).to_csv(R / "predictions.csv.gz", index=False)

    # ---------------------------------------------- calibration & evaluation
    log("calibration on Sep-Oct (static rule) + causal rolling Platt")
    final_preds, calib_choice, calib_tbl = pipeline.calibrate(P.labels, raw_preds)
    calib_tbl.to_csv(R / "calibrator_selection.csv", index=False)
    log(f"static-rule calibrators: {calib_choice}")
    thresholds = pipeline.frozen_thresholds(P.labels, final_preds)

    log("evaluating on Nov-Dec test")
    results = pipeline.metrics_table(P.labels, final_preds, thresholds)
    results.to_csv(R / "metrics_all.csv", index=False)
    test_tbl = pipeline.headline_table(results)
    log("\n" + test_tbl[pipeline.SHOW_COLS].round(4).to_string())
    pipeline.calibration_study(results).to_csv(R / "calibration_study.csv")

    policies = pipeline.operating_policies(P.labels, final_preds, thresholds)
    policies.to_csv(R / "operating_policies.csv", index=False)
    log("\n" + policies.round(3).to_string())

    drift_tbl = pipeline.monthly_drift(P.labels, final_preds)
    drift_tbl.to_csv(R / "monthly_calibration_drift.csv", index=False)

    log("day-block bootstrap on test")
    boot = pipeline.bootstrap_compare(P.labels, final_preds, n_boot)

    # ----------------------------------------------------- design evidence
    log("month-feature check (rolling CV)")
    month_cv = pipeline.month_feature_cv(P, lr.C, xgb.params)
    month_cv.to_csv(R / "month_feature_cv.csv", index=False)

    log("feature-family ablation (both models, CV + test bootstrap)")
    abl, abl_steps = pipeline.ablation(P, lr.C, xgb.params, n_boot)
    abl.to_csv(R / "ablation.csv", index=False)
    abl_steps.to_csv(R / "ablation_steps.csv", index=False)
    log("\n" + abl_steps.round(4).to_string())

    log("grouped permutation importance (Sep-Oct)")
    reps = 3 if quick else 5
    perm = pd.concat([pipeline.grouped_permutation_importance(P, mdl, reps).assign(model=name)
                      for name, mdl in [("logreg", lr.model), ("xgboost", xgb.model)]],
                     ignore_index=True)
    perm.to_csv(R / "permutation_importance_groups.csv", index=False)
    gain = pipeline.xgb_gain(xgb.model)
    gain.to_csv(R / "xgb_gain_importance.csv")
    pipeline.logreg_coefficients(lr.model).to_csv(R / "logreg_coefficients.csv")

    log("target sensitivity (cancelled/diverted = delayed)")
    sens = pipeline.target_sensitivity(P, raw_preds, lr.C, xgb.params, n_boot)
    sens.to_csv(R / "target_sensitivity.csv", index=False)

    log("covariate drift (PSI, adversarial validation)")
    psi, adv, adv_imp = pipeline.distribution_shift(P)
    psi.to_csv(R / "psi.csv", index=False)
    adv.to_csv(R / "adversarial_validation.csv", index=False)
    adv_imp.to_csv(R / "adversarial_importance.csv")

    log("figures")
    for name, fig in pipeline.evaluation_figures(P.labels, final_preds, drift_tbl, gain, perm,
                                                 abl, psi).items():
        plots.save(fig, name)

    primary = policies[(policies.role == "primary") & (policies.split == "test")
                       & (policies.model == f"{final_model}_cal")].iloc[0]
    summary = {
        "decision_time": f"{config.DECISION_HOUR}:00 NYC on D-1; outcomes enter history once known",
        "target": f"arr_delay >= {config.DELAY_THRESHOLD_MIN} min; cancelled/diverted dropped",
        "splits": P.split_summary(),
        "cv_val_months": config.CV_VAL_MONTHS,
        "history_settings": history_choice,
        "n_features": len(P.feat_cols),
        "feature_groups": {k: len(v) for k, v in P.groups.items()},
        "final_model": {"model": final_model, "rule": "higher mean rolling-CV PR-AUC",
                        "fixed_before_test": True},
        "xgb_params": xgb.params,
        "xgb_cv_std": cv["xgboost"]["std"],
        "logreg_C": lr.C,
        "cv_pr_auc": cv,
        "calibration_rule": (f"raw unless Platt/isotonic (fit Sep) lowers Oct ECE by "
                             f"> {config.CALIB_MIN_ECE_GAIN}"),
        "calibrator_selected": calib_choice,
        "rolling_calibration_days": config.ROLLING_CALIB_DAYS,
        "primary_policy_test": {"policy": primary["policy"], "model": primary["model"],
                                "precision": primary["precision"], "recall": primary["recall"],
                                "lift": primary["lift"], "flag_rate": primary["flag_rate"]},
        "test": test_tbl[pipeline.SHOW_COLS].reset_index().to_dict(orient="records"),
        "oracle_metric_note": ("oracle_recall@p tunes the threshold on the test labels: "
                               "curve metric for model comparison only"),
        "bootstrap_test": boot,
        "operating_policies": policies.to_dict(orient="records"),
    }
    (R / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("done -> results/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip history tuning; fewer configs / bootstrap reps")
    main(ap.parse_args().quick)
