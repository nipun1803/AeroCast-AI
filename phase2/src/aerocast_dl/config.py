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

# --- Training (protocol, amended before any training; see the change log) ----
MAX_EPOCHS = 40
PATIENCE = 4                 # epochs without inner-holdout log-loss improvement
BATCH_SIZE = 1024
SEARCH_SEED = SEEDS[0]       # hyperparameter search uses one seed; the best
                             # configuration of each family is re-scored with all 5
N_THREADS = 8

# Search spaces, fixed before the search runs (protocol budget in brackets).
MLP_GRID = {                                  # [20 random configurations]
    "hidden": [64, 128, 256],
    "depth": [2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "emb_dropout": [0.0, 0.1],
    "lr": [1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}
MLP_N_CONFIGS = 20
GRU_GRID = {                                  # [12], MLP part fixed at the best MLP config
    "seq_hidden": [8, 16, 32],
    "seq_dropout": [0.0, 0.2],
    "lr": [1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4],
}
GRU_N_CONFIGS = 12
HYBRID_GRID = {                               # [12], deep part = best GRU model
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4],
    "wide_lr_scale": [0.1, 1.0],              # wide (LogReg) part learns slower or equally
}
HYBRID_N_CONFIGS = 12
# The wide part reuses the Phase 1 LogReg regularisation (selected on the same folds).
LOGREG_C = 0.1
