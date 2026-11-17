#!/usr/bin/env bash
# Rebuild every Phase 2 artefact from a clean state and verify it. Exits non-zero on any failure.
#
#   tests  ->  old outputs moved aside  ->  pipeline stages (CV, final fits, calibration,
#   rescheduling)  ->  the test stage (--final)  ->  notebooks 05-11 executed in place
#   ->  report assets  ->  number check  ->  report tests  ->  report PDF
#
# The old outputs are moved to a temporary folder, not deleted. If any step fails they are
# put back, so an interrupted or failed run never leaves the folder half-built.
#
# The --final stage reads the Nov-Dec labels. In development it ran twice, after the rules
# were frozen and again after the 2026-11-17 audit fixes (docs/06_rules_and_limitations.md);
# rerunning it here reproduces that evaluation (all seeds are fixed), it does not choose anything.
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
# The report tests check the generated assets, so they run at the end, after the rebuild.
"$PY" -B -m pytest -q -p no:cacheprovider --ignore=tests/test_report.py

step "1. clean state: move results/, figures/, generated report assets and model cache aside"
OUTPUTS=(results figures report/figures report/tables data/cache)
BACKUP="$(mktemp -d "${TMPDIR:-/tmp}/aerocast_phase2_outputs.XXXXXX")"
for d in "${OUTPUTS[@]}"; do
  if [ -e "$d" ]; then mkdir -p "$BACKUP/$(dirname "$d")"; mv "$d" "$BACKUP/$d"; fi
done
restore() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "reproduce.sh failed (exit $status): restoring the previous outputs from $BACKUP"
    for d in "${OUTPUTS[@]}"; do
      rm -rf "$d"
      if [ -e "$BACKUP/$d" ]; then mkdir -p "$(dirname "$d")"; mv "$BACKUP/$d" "$d"; fi
    done
  fi
}
trap restore EXIT
mkdir -p results figures

step "2. pipeline: inputs, CV searches, hybrids, ablations, figures, diagnostics, final fits, calibration, rescheduling"
"$PY" -B scripts/run_phase2.py

step "3. the test evaluation (--final)"
"$PY" -B scripts/run_phase2.py --final

step "4. execute the notebooks in place (kernel: aerocast)"
for nb in notebooks/0*.ipynb notebooks/1*.ipynb; do
  echo "--- $nb"
  "$PY" -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=aerocast "$nb"
done

step "5. report assets, number check, PDF"
"$PY" -B scripts/export_report_assets.py
"$PY" -B scripts/check_report_numbers.py
"$PY" -B -m pytest -q -p no:cacheprovider tests/test_report.py
if command -v tectonic >/dev/null 2>&1; then
  (cd report && tectonic -X compile main.tex)
else
  echo "tectonic not found: skipping the PDF build"
fi

trap - EXIT
rm -rf "$BACKUP"
step "done"
