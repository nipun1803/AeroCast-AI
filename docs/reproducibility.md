# Reproducibility

## Environment

Python 3.12; pinned in `requirements.txt`:

```text
# Tested with Python 3.12
pandas==3.0.6
numpy==2.5.4
scikit-learn==1.9.1
xgboost==3.4.1
matplotlib==3.11.2
seaborn==0.13.2
pytest==9.1.1
jupyter==1.1.1
nbformat==5.11.1
nbconvert==7.17.2
ipykernel==7.4.0
```

Install: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`, place `flights.csv` in `data/raw/`, and register the kernel `.venv/bin/python -m ipykernel install --sys-prefix --name aerocast`. All randomness is seeded (`config.SEED = 42`).

## One command

```bash
bash scripts/reproduce.sh
```

It (0) deletes `results/` and `figures/`, (1) runs `scripts/run_phase1.py`, (2) executes the four notebooks in place (kernel `aerocast`; notebooks 03 and 04 assert every table they show equals its saved artefact in `results/`), (3) runs `scripts/export_report_assets.py`, (4) runs `pytest`. It stops with a non-zero status on any failure (`set -euo pipefail`).

Runtime of the last clean run: pipeline 9 min, notebooks 3 min, export + pytest 0 min; total 13 min.

## Artefacts

* `results/` (26 files): `ablation.csv`, `ablation_steps.csv`, `adversarial_importance.csv`, `adversarial_validation.csv`, `calibration_study.csv`, `calibrator_selection.csv`, `cv_logreg.csv`, `cv_summary.csv`, `cv_xgb_configs.csv`, `cv_xgb_folds.csv`, `headline_by_month.csv`, `headline_test.csv`, `history_tuning.csv`, `logreg_coefficients.csv`, `mean_scores.csv`, `metrics_all.csv`, `month_feature_cv.csv`, `monthly_calibration_drift.csv`, `operating_policies.csv`, `permutation_importance_groups.csv`, `predictions.csv.gz`, `psi.csv`, `single_month_tree_curve.csv`, `summary.json`, `target_sensitivity.csv`, `xgb_gain_importance.csv`
* `figures/*.png` (PNG only), `report/figures/` (copies) and `report/tables/` (CSV + booktabs `.tex`).

## macOS note

If `import aerocast` fails inside the venv, macOS may have flagged the `.pth` files as hidden (common in iCloud-synced folders) and Python skips hidden `.pth` files: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. The script and notebooks add `src/` to `sys.path` themselves, so they work either way.
