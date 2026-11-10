"""AeroCast-AI Phase 2: deep-learning and hybrid models.

PyTorch and XGBoost each ship an OpenMP runtime (libomp). On macOS, loading
XGBoost's first and then running multi-threaded torch code crashes the
process (segfault). Importing this package therefore loads torch first and
refuses to continue if XGBoost is already loaded. XGBoost models are never
trained in a process that trains torch models (see ``hybrid``).
"""
import sys

if "xgboost" in sys.modules and "torch" not in sys.modules:
    raise ImportError(
        "import aerocast_dl (or torch) before aerocast.pipeline / xgboost: "
        "the two OpenMP runtimes crash in the other order")

import torch  # noqa: E402,F401
