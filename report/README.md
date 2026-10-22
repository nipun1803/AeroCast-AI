# report/: self-contained IEEE report package

| File | What it is |
|---|---|
| `main.tex` | IEEEtran conference report (hand-written prose, bullets) |
| `references.bib` | bibliography (BibTeX, `IEEEtran` style) |
| `figures/` | PNG figures, **generated** from `results/` |
| `tables/` | `.csv` + `.tex` (booktabs) tables, **generated** from `results/` |
| `main.pdf` | compiled report |

## Build

```bash
cd report
tectonic -X compile main.tex            # runs bibtex automatically
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs `IEEEtran`, `cite`, `booktabs`, `graphicx`, `amsmath`, `enumitem`, `hyperref`. `figures/` and `tables/` are rebuilt by `python scripts/export_report_assets.py`; `bash scripts/reproduce.sh` does everything (pipeline, notebooks, export, tests, number check, PDF).

## Numbers

Prose numbers are typed by hand. `python scripts/check_report_numbers.py` asserts the quoted ones against `results/` and fails on drift. It covers the headline, policy, CV, calibration, bootstrap and rescheduling numbers only, not every number in the text.

## References to VERIFY

* `yu2019` (Yu et al., *Flight delay prediction for commercial air transport: a deep learning approach*, Transp. Res. E 2019): not confirmed by web search; author list and pages from memory.
* `belcastro2016`: existence confirmed from a bibliographic record (ACM TIST 8(1), 2016); abstract not read, so its matrix cells say `VERIFY`.
* Other entries were confirmed by web search on 2026-10-22; fields not confirmed (some pages and volumes) are omitted rather than guessed.
* The matrix cell value `not assessed` means the item was not read in the source.
* Author names, course line and email in `main.tex` are placeholders.
