#!/usr/bin/env bash
# Rebuild every artefact from a clean state and verify it. Exits non-zero on any failure.
#
#   results/ and figures/ are deleted  ->  pipeline  ->  4 notebooks executed in place
#   (each asserts its tables equal results/)  ->  report assets  ->  pytest  ->  number check  ->  report PDF
#
# Usage:  bash scripts/reproduce.sh            (full run, about 25-30 min)
#         PY=/path/to/python bash scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
NB_TIMEOUT=3600
t0=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - t0 ))s] $*"; }

step "0. clean state: delete results/ and figures/ (and generated report assets)"
rm -rf results figures report/figures report/tables
mkdir -p results figures

step "1. pipeline (single source of truth)"
"$PY" scripts/run_phase1.py

step "2. execute the four notebooks in place (kernel: aerocast)"
for nb in 01_eda 02_feature_engineering 03_model_training 04_evaluation; do
  echo "--- $nb"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout="$NB_TIMEOUT" \
        --ExecutePreprocessor.kernel_name=aerocast "notebooks/$nb.ipynb"
done

step "3. export report assets"
"$PY" scripts/export_report_assets.py

step "4. pytest"
"$PY" -m pytest -q

step "4b. check the numbers quoted in report/main.tex against results/"
"$PY" scripts/check_report_numbers.py

step "5. build report/main.pdf (if a LaTeX engine is installed)"
if command -v tectonic >/dev/null 2>&1; then
  (cd report && tectonic -X compile main.tex)
elif command -v pdflatex >/dev/null 2>&1; then
  (cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
else
  echo "no LaTeX engine found (tectonic / pdflatex): skipping the PDF build"
fi

step "done: pipeline, 4 notebooks, report export and tests all succeeded"
