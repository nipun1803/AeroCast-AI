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

## Verified staged reproduction

With `.venv` activated, run from the repository root:

```bash
python -B -m pytest -q -p no:cacheprovider
python -B scripts/run_phase1.py
python -B -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=aerocast \
  notebooks/01_eda.ipynb notebooks/02_feature_engineering.ipynb \
  notebooks/03_model_training.ipynb notebooks/04_evaluation.ipynb
```

The verified MacBook Pro run on **2026-10-23** passed **36 tests**, completed the pipeline in about **22 minutes**, and executed all four notebooks successfully. Notebook execution time varies; the former 13-minute total estimate described an older experiment and is not a current benchmark.

The equal-mass ECE correction on **2026-10-24** passed the expanded **42-test** suite. Evaluation was recomputed from unchanged saved predictions, notebook 04 was executed again, and affected result/report tables and the PDF were refreshed. No flight-classifier training was rerun; model choices, other metrics, policies and figures remain unchanged. Tied-probability checks cover synthetic label orderings and 50 permutations of five real saved-score variants.

The pipeline overwrites generated `results/` and `figures/`; notebook execution updates stored outputs. For updated report assets, run:

```bash
python -B scripts/export_report_assets.py
python -B scripts/check_report_numbers.py
```

Then build the PDF as described in [report/README.md](../report/README.md).

`scripts/reproduce.sh` is the full clean-rebuild helper: it deletes generated results, figures and report assets; runs the pipeline and notebooks; exports report assets; runs tests and the report-number check; and attempts a PDF build when a LaTeX engine is installed. It currently dispatches notebooks through `python -m jupyter nbconvert`, which failed in the reproduced local setup. Use the direct `python -m nbconvert` commands above; this documentation update does not modify the helper.

For explanation-only changes, model training and notebook execution are unnecessary. Recompile the PDF only if its LaTeX source changes. The notebook table assertions compare calculations with saved tables; they do not automatically check handwritten Markdown explanations.

## Artefacts

* `results/` (27 files): `ablation.csv`, `ablation_steps.csv`, `adversarial_importance.csv`, `adversarial_validation.csv`, `calibration_study.csv`, `calibrator_selection.csv`, `cv_logreg.csv`, `cv_summary.csv`, `cv_xgb_configs.csv`, `cv_xgb_folds.csv`, `headline_by_month.csv`, `headline_test.csv`, `history_tuning.csv`, `logreg_coefficients.csv`, `mean_scores.csv`, `metrics_all.csv`, `month_feature_cv.csv`, `monthly_calibration_drift.csv`, `operating_policies.csv`, `permutation_importance_groups.csv`, `predictions.csv.gz`, `psi.csv`, `rescheduling_candidates.csv`, `single_month_tree_curve.csv`, `summary.json`, `target_sensitivity.csv`, `xgb_gain_importance.csv`
* `figures/`: 22 PNGs; `report/figures/`: 22 matching copies; `report/tables/`: 20 CSV tables and 20 booktabs `.tex` tables; `report/main.pdf`: the existing 20-page report. Counts describe this saved experiment, not a universal requirement for other datasets.

## macOS note

If `import aerocast` fails inside the venv, macOS may have flagged the `.pth` files as hidden (common in iCloud-synced folders) and Python skips hidden `.pth` files: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. The script and notebooks add `src/` to `sys.path` themselves, so they work either way.
