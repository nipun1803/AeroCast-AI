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

Needs `IEEEtran`, `cite`, `booktabs`, `graphicx`, `amsmath`, `enumitem`, `hyperref`. The current saved report is 20 pages, with 22 figure copies and 20 CSV/LaTeX table pairs. Run `python -B scripts/export_report_assets.py` from the repository root only when saved results or figures have changed. For the working staged pipeline/notebook commands and the clean-rebuild helper's dispatch limitation, see [reproducibility.md](../docs/reproducibility.md). Explanation-only edits do not require model retraining; changes to `main.tex` require recompiling `main.pdf`.

## Numbers

Prose numbers are typed by hand. From the repository root, `python -B scripts/check_report_numbers.py` checks selected numeric strings against `results/`. The current report passed 40/40 checks. These cover selected headline, policy, CV, calibration, bootstrap and candidate-ranking values; they do not validate every number or its surrounding interpretation. Notebook outputs and supporting prose must also agree with the saved run.

Equal-mass ECE preserves identical probabilities in the same bin; ties can produce fewer bins or unequal counts. The corrected prior test value is 0.007 when rounded to three decimals. Its low calibration error does not imply flight-specific ranking skill. See the metric definition in `main.tex` and [evaluation notes](../docs/04_evaluation.md).

## Reference and manuscript status

* `yu2019`: authors, venue, volume, pages and DOI were verified against Crossref on 2026-10-23, as recorded in `references.bib`. Bibliographic verification does not mean that the paper's full method was assessed.
* `belcastro2016`: existence was confirmed from a bibliographic record; its abstract and full experimental protocol were not assessed.
* Literature-matrix cells marked `not assessed` disclose unreviewed methodological details. They are not instructions to insert guessed values.
* The current manuscript names the project team; the earlier README description of author placeholders is obsolete.
* Source, bibliography and PDF are at their existing paths. Raw `data/raw/flights.csv` remains a local, Git-ignored input; obtain it using the root README instructions.
