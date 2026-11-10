"""Write report/figures and report/tables from results/ and figures/.

Usage: python scripts/export_report_assets.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "phase1" / "src"):
    sys.path.insert(0, str(p))

import aerocast_dl  # noqa: E402,F401  (torch before XGBoost)
from aerocast_dl import report  # noqa: E402

if __name__ == "__main__":
    print(report.export())
