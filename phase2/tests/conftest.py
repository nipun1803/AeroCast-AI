import sys
from pathlib import Path

import pytest

# Same safeguard as Phase 1: do not depend on the editable-install .pth files,
# which macOS may hide in iCloud-synced folders.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "phase1" / "src"):
    sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def P():
    """The Phase 1 modelling frame (features on the full schedule, splits)."""
    from aerocast import pipeline
    return pipeline.prepare()
