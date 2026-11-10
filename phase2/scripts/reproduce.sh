#!/usr/bin/env bash
# Rebuild every Phase 2 artefact from a clean state and verify it. Exits non-zero on any failure.
#
#   tests  ->  results/ and figures/ deleted  ->  pipeline stages (CV, final fits, calibration,
#   rescheduling)  ->  the test stage (--final)  ->  notebooks 05-11 executed in place
#   ->  report assets  ->  number check  ->  report PDF
#
# The --final stage reads the Nov-Dec labels. In development it was run once, after the
# protocol was frozen (docs/development_protocol.md); rerunning it here reproduces that
# evaluation (all seeds are fixed), it does not choose anything.
#
# Usage:  bash scripts/reproduce.sh            (about 3 hours on an M3 Pro, CPU)
#         PY=/path/to/python bash scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# The venv is shared by phase1/ and phase2/ and lives at the repository root.
PY="${PY:-../.venv/bin/python}"
t0=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - t0 ))s] $*"; }

step "0. tests (leakage, alignment, training loop, calibration, test lock, rescheduling)"
"$PY" -B -m pytest -q -p no:cacheprovider

step "1. clean state: delete results/, figures/, generated report assets and model cache"
rm -rf results figures report/figures report/tables data/cache
mkdir -p results figures

step "2. pipeline: inputs, CV searches, hybrids, ablations, figures, diagnostics, final fits, calibration, rescheduling"
"$PY" -B scripts/run_phase2.py

step "3. the one test evaluation"
"$PY" -B scripts/run_phase2.py --final

step "4. execute the notebooks in place (kernel: aerocast)"
for nb in notebooks/0*.ipynb notebooks/1*.ipynb; do
  echo "--- $nb"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=aerocast "$nb"
done

step "5. report assets, number check, PDF"
"$PY" -B scripts/export_report_assets.py
"$PY" -B scripts/check_report_numbers.py
if command -v tectonic >/dev/null 2>&1; then
  (cd report && tectonic -X compile main.tex)
else
  echo "tectonic not found: skipping the PDF build"
fi

step "done"
