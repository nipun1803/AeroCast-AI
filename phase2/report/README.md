# report/: Phase 2 IEEE report package

| File | What it is |
|---|---|
| `main.tex` | 8-page IEEEtran conference report (hand-written prose) |
| `references.bib` | 12 references (BibTeX, `IEEEtran` style) |
| `figures/` | the 8 PNG figures the report uses, **generated** (copied from `../figures/`) |
| `tables/` | the report's 8 tables as `.csv` + `.tex` (booktabs), **generated** from `../results/` |
| `main.pdf` | compiled report |

## Build

```bash
python -B scripts/export_report_assets.py   # from phase2/, after the pipeline and the --final stage
cd report
tectonic -X compile main.tex                # runs bibtex automatically
```

The export clears `figures/` and `tables/` here and rewrites exactly the files `main.tex` uses (`aerocast_dl.report.REPORT_FIGURES` and `report_tables()`). Never edit them by hand.

## Keeping the report aligned

* **Numbers.** Prose numbers are typed by hand. Run `python -B scripts/check_report_numbers.py` from `phase2/`. It checks every quoted CV, ablation, diagnostic, calibration, test, slice, drift and rescheduling number against `results/`. It also asserts the claims in the text that are not numbers: the final model, the frozen rescheduling rule, the carrier counts and the sign of each net benefit.
* **Tables, figures, citations.** `tests/test_report.py` fails in four cases:
  * a placeholder is left in `main.tex`;
  * `main.tex` includes a table or figure the export does not produce, or the reverse;
  * a figure is missing;
  * the citations and `references.bib` differ.
