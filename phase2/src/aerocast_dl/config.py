"""Phase 2 configuration.

Everything shared with Phase 1 (decision clock, target, splits, CV folds,
feature settings) is read from ``aerocast.config`` and never redefined here,
so the two phases cannot drift apart. Rules fixed in
docs/development_protocol.md are marked "(protocol)".
"""
from pathlib import Path

from aerocast import config as p1

ROOT = Path(__file__).resolve().parents[2]          # phase2/
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "data" / "cache"                 # git-ignored

# --- Seeds (protocol): 5 seeds; the deployed model is their mean logit.
SEEDS = tuple(p1.SEED + k for k in range(5))       # 42..46

# --- Categorical inputs -----------------------------------------------------
# Embedded in every model: the Phase 1 categorical features.
CAT_COLS = ["carrier", "origin", "dest"]
# High-cardinality identifiers, used only in the ablation that asks whether
# embeddings can exploit IDs a one-hot LogReg cannot. Each is derived from
# schedule fields known at the decision clock (tailnum: Phase 1 assumption
# that the aircraft assignment is known by 20:00 on D-1).
ID_COLS = ["tailnum", "carrier_flight", "origin_hour"]
# A category needs at least this many fitting rows to get its own embedding;
# rarer and unseen values share index 0. Equal to the min_frequency of the
# Phase 1 LogReg one-hot encoder (aerocast.models.make_logreg), so the two
# phases treat rare categories the same way.
MIN_CAT_FREQ = 50

# --- Fitting clock ------------------------------------------------------------
# The final network is fitted for the first calibration date, as in Phase 1.
FINAL_FIT_DATE = p1.SPLITS["calib"][0]
# Early stopping (protocol): the last 14 days of each fold's training window.
INNER_HOLDOUT_DAYS = 14
