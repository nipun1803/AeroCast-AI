"""Central configuration: paths, target definition, decision clock, splits,
selection rules and seeds.

Every experiment-level choice lives here so the report, notebooks and scripts
all refer to one source of truth. Rules marked "fixed before test" were set
using CV / Sep-Oct evidence only; see the Development protocol in the README.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "flights.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEED = 42

# --- Target -----------------------------------------------------------------
# BTS definition of an "on-time" arrival: < 15 minutes after schedule.
DELAY_THRESHOLD_MIN = 15

# --- Decision clock ---------------------------------------------------------
# Predictions for all flights on date D are issued at DECISION_HOUR:00 (NYC
# local time) on D-1. An outcome may feed a feature for D only if it was known
# by then:
#   * arrival delay  -> known at the flight's actual gate arrival, converted to
#                       the NYC clock (destination time zone + 2013 DST rule)
#   * cancellation   -> known at the scheduled departure (cancelled flights) or
#                       actual departure (flights that left)
# Late-evening D-1 arrivals therefore first count for D+1, not D.
DECISION_HOUR = 20
# 2013 US daylight-saving period (NYC observes it; HNL, PHX, Puerto Rico and
# the US Virgin Islands do not), used to convert local arrival times.
DST_2013 = ("2013-03-10 02:00", "2013-11-03 02:00")
NO_DST_AIRPORTS = {"HNL", "PHX", "SJU", "BQN", "PSE", "STT"}
# Destination clock minus NYC clock (minutes) during DST. Static geography (an
# airports lookup table): it must NOT be re-estimated from the flights being
# featurised, or past features would depend on future outcomes. Values were
# derived once from the data and are re-verified in notebook 02
# (data.dest_offsets reproduces them); LEX and LGA have no usable arrival
# and are Eastern.
_DEST_BY_OFFSET = {
    0: "ACK ALB ATL AVL BDL BGR BOS BQN BTV BUF BWI CAE CAK CHO CHS CLE CLT CMH CRW CVG "
       "DAY DCA DTW EYW FLL GRR GSO GSP IAD ILM IND JAX LEX LGA MCO MHT MIA MVY MYR ORF "
       "PBI PHL PIT PSE PVD PWM RDU RIC ROC RSW SAV SBN SDF SJU SRQ STT SYR TPA TVC TYS",
    -60: "AUS BHM BNA DFW DSM HOU IAH MCI MDW MEM MKE MSN MSP MSY OKC OMA ORD SAT STL TUL XNA",
    -120: "ABQ BZN DEN EGE HDN JAC MTJ SLC",
    -180: "BUR LAS LAX LGB OAK PDX PHX PSP SAN SEA SFO SJC SMF SNA",
    -240: "ANC",
    -360: "HNL",
}
DEST_OFFSET_MIN = {d: o for o, ds in _DEST_BY_OFFSET.items() for d in ds.split()}

# Rows in the first days have no/very thin history; they are excluded from
# training so the model does not learn a "cold-start" artefact.
WARMUP_END = "2013-01-07"

# --- Chronological split (inclusive bounds) ---------------------------------
SPLITS = {
    "train": ("2013-01-01", "2013-08-31"),   # model fitting + rolling-origin CV
    "calib": ("2013-09-01", "2013-10-31"),   # calibration + threshold choice
    "test": ("2013-11-01", "2013-12-31"),    # touched once, at the end
}

# Rolling-origin CV inside the training window: fold k trains on months
# < m_k and validates on month m_k.
CV_VAL_MONTHS = [5, 6, 7, 8]

# --- Feature hyper-parameters (tuned by rolling CV, see pipeline.tune_history)
# Pseudo-count strength of the shrinkage toward the network prior, and the
# long trailing window of the entity-level history rates. Rule: the
# (m, window) pair with the highest mean rolling-CV PR-AUC of LogReg, tuned
# JOINTLY with C (each pair is scored at its own best C of models.LOGREG_CS);
# the pipeline fails loudly if these values are not that pair.
# Was m = 20 / 28 days (hand-picked); CV picked m = 5 / 14 days -- a small gain
# relative to the fold spread. An earlier version scored the grid at a fixed
# C = 0.001; re-tuning jointly with C selects the same pair (and C = 0.1), so
# the decision is unchanged (results/history_tuning.csv).
SMOOTHING_M = 5
LONG_WINDOW = 14
SMOOTHING_M_GRID = [5, 20, 50]
LONG_WINDOW_GRID = [14, 28, 56]

# --- Final model (fixed before test) -----------------------------------------
# Rule: the model family with the higher mean rolling-CV PR-AUC, each scored
# AS DEPLOYED (XGBoost at its fixed, fold-averaged tree count; its search score
# early-stops on the validation fold itself and is optimistic). The pipeline
# re-derives this from CV and fails loudly if it disagrees with the value here.
# Decision-clock features: LogReg 0.4705 vs XGBoost 0.4450 (fixed trees);
# XGBoost's in-fold early-stopping score (0.4754) is reported but not used.
FINAL_MODEL = "logreg"

# --- Calibration (fixed before test) -----------------------------------------
# Static recalibration rule: fit candidates on Sep, check ECE on Oct; adopt a
# recalibrator only if it beats raw scores by more than this margin, else keep
# raw. (Replaces the earlier "lowest Oct ECE wins" rule, which chose Platt for
# LogReg on a 0.0007 near-tie.)
CALIB_MIN_ECE_GAIN = 0.005
# Causal rolling recalibration: for each day D, refit Platt on the trailing N
# days of outcomes known by the decision clock. N fixed a priori to the entity
# history window; the other values are reported as sensitivity only.
ROLLING_CALIB_DAYS = 28
ROLLING_CALIB_SENSITIVITY = [14, 28, 56]

# --- Evaluation -------------------------------------------------------------
# PRIMARY operating policy (fixed before test): an ops desk reviews the top-k%
# riskiest flights of each day. Rank-based, so it is robust to the base-rate
# shift between seasons. Chosen a priori as a capacity assumption.
DAILY_FLAG_SHARE = 0.10
# Secondary policy: a score threshold frozen on Sep-Oct to reach this
# precision. 0.50 was tried first and is unreachable on Sep-Oct (base rate
# 15.7%), so the target was lowered to 0.40 -- a Sep-Oct-based decision.
TARGET_PRECISION = 0.40
# Curve metric for MODEL COMPARISON ONLY: best recall at precision >= p with
# the threshold tuned on the evaluated labels themselves ("oracle").
PRECISION_SWEEP = [0.30, 0.40, 0.50]
N_CALIBRATION_BINS = 15
N_BOOTSTRAP = 500
