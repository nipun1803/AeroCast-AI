# report/: assets for the IEEE report

The report itself is written separately (by hand). This folder only holds generated assets; **no LaTeX document body is generated here.**

| Folder | Content | Produced by |
|---|---|---|
| `figures/` | copies of the final PNG figures from `figures/` | `scripts/export_report_assets.py` |
| `tables/` | headline, by-month, operating-policy, calibration-study, ablation, ablation-step, target-sensitivity and CV tables as `.csv` and LaTeX-ready booktabs `.tex` snippets | `scripts/export_report_assets.py` |

* Tables are read from `results/` (never recomputed). Include a snippet with `\input{tables/headline_test.tex}`; each needs `\usepackage{booktabs}`.
* Model rows carry explicit calibration labels (e.g. "LogReg (Platt, rule-selected)") and the references *Prior (base rate)* and *Route history*.
* Regenerate everything with `bash scripts/reproduce.sh` (or just `python scripts/export_report_assets.py` after the pipeline and notebooks have run). The folders are overwritten each time.
