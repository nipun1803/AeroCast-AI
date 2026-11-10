"""Copy the final figures and export the report tables (CSV + booktabs .tex).

Run after ``scripts/run_phase1.py`` and the notebooks (so figures/ is complete).
Writes report/figures/*.png and report/tables/*.csv|tex; no LaTeX body.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerocast import report  # noqa: E402

if __name__ == "__main__":
    out = report.export()
    print(f"report/figures: {len(out['figures'])} PNG files")
    print("report/tables: " + ", ".join(out["tables"]))
