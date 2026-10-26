# report/: IEEE report package

| File | What it is |
|---|---|
| `main.tex` | 7-page IEEEtran conference report (hand-written prose, bullets) |
| `references.bib` | bibliography (BibTeX, `IEEEtran` style) |
| `figures/` | the 8 PNG figures the report uses, **generated** (copied from `figures/`) |
| `tables/` | the report's 7 tables as `.csv` + `.tex` (booktabs), **generated** from `results/` |
| `main.pdf` | compiled report |
| `long_version/` | earlier 22-page version with its own figures, tables and bibliography; kept locally as a viva supplement and **git-ignored** |

## Build

```bash
python -B scripts/export_report_assets.py   # from the repository root, after the pipeline/notebooks
cd report
tectonic -X compile main.tex                # runs bibtex automatically
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs `IEEEtran`, `cite`, `amsmath`, `graphicx`, `booktabs`, `array`, `enumitem`, `hyperref`. The export clears `figures/` and `tables/` here and rewrites exactly the files `main.tex` uses (`aerocast.report.REPORT_FIGURES` and `report_tables()`); never edit them by hand. Explanation-only edits do not require retraining; changes to `main.tex` require recompiling `main.pdf`.

## Keeping the report aligned

* **Numbers.** Prose numbers are typed by hand. `python -B scripts/check_report_numbers.py` (from the repository root) checks 103 quoted values against `results/`: test, CV, calibration, policy, accuracy/F1/confusion, ablation, importance, split and robustness numbers. Values that come from the raw data (EDA counts, leakage-test cell counts) are not in `results/`; they are computed and shown in notebooks 01 and 02, and the leakage tests assert their pass/fail conditions.
* **Tables, figures, citations.** `tests/test_report.py` fails if `main.tex` includes a table or figure the export does not produce (or the reverse), if a figure is missing from `figures/`, or if the citations and `references.bib` differ.
