# 01 · Inputs, the decision clock and leakage

> Covers notebooks [05_network_inputs](../notebooks/05_network_inputs.ipynb) and [07_gru_history_model](../notebooks/07_gru_history_model.ipynb) Part A. Code: `src/aerocast_dl/tensors.py`, `sequences.py`, `guards.py`.

## The decision clock

A prediction for a flight on day D is made at **20:00 NYC on D−1**. Any outcome may feed it only if it was known by then. For example:

- a delay counts once the flight reached its gate;
- a cancellation counts once the scheduled departure has passed.

Each outcome therefore carries an **availability date**: the first day whose 20:00-on-the-eve deadline falls after the outcome became known. Phase 2 does not re-implement this. It imports `aerocast.data.available_date` and the Phase 1 history code, so both phases obey the same clock by construction.

## Data and split (identical to Phase 1)

| Split | Dates | Flights scored | Late rate | Used for |
|---|---|---|---|---|
| Train | 8 Jan – 31 Aug | 211,684 | 26.5% | fitting and cross-validation |
| Calibration | 1 Sep – 31 Oct | 55,628 | 15.7% | calibrators, thresholds, rescheduling tuning |
| Test | 1 Nov – 31 Dec | 53,991 | 25.8% | the one final evaluation |

- **Target:** arrival delay of at least 15 minutes. Cancelled and diverted flights are excluded.
- **Seasonal shift:** the late rate moves a lot (Nov 18.1%, Dec 33.5%), which matters for calibration ([04](04_calibration_and_test.md)).
- **Fitting clock for the final models:** they are fitted "on 1 September". They use the **211,517** training flights whose outcome was known by 20:00 on 31 August. The other 167 late-August flights are excluded.

## Tabular inputs

The networks use the same 43 Phase 1 features as LogReg: 3 categorical (carrier, origin, destination) and 40 numeric.

- **Numeric:** median imputation, then standardisation.
- **Categorical:** each value gets a learned embedding vector. Values seen in fewer than 50 fitting rows share one "unknown" (UNK) vector. That cut-off is the same one Phase 1's one-hot encoder uses.
- **Fitted per fit:** the vocabulary, the medians and the scaler are fitted **only on the rows of the fit they serve**: each CV fold's training rows, or the final training rows. A calibration or test row never shapes an input statistic. `fit_encoder` refuses such rows outright.

**Cold start** (`results/cold_start.csv`): how often a test flight falls into the UNK bucket of a vocabulary fitted on the final training rows.

| Column | Values with own embedding | Test rows in UNK |
|---|---|---|
| carrier | 15 | 0.01% |
| origin | 3 | 0% |
| destination | 93 | 0.17% |
| tail number (ablation only) | 1,568 | 26.6% |
| carrier + flight number (ablation only) | 1,393 | 43.9% |

The three main categoricals are almost fully covered. The high-cardinality identifiers are tried only as an ablation ([03](03_cv_results_and_ablations.md)), and a large share of their test values are unseen or rare.

## Causal history windows (for the GRU)

The brief allows a temporal model **only if the flight history is correctly ordered**. Here the order is the decision clock, not the flight date.

**What a window holds.** For every target day D and every origin, carrier and route, a window holds the last **28 availability days**. Each daily slot contains:
- known arrivals and late arrivals;
- total delay minutes, clipped to −60…240;
- known departures and cancellations.

**What the newest slot means.** It holds the outcomes that first became usable for D: those known between 20:00 on D−2 and 20:00 on D−1. Nothing from day D can enter D's window, and nothing from D−1 after 20:00 can either.

| Entity | Entities | Mean known arrivals per window | Phase 1 mismatches |
|---|---|---|---|
| origin | 3 | 8,234 | 0 |
| carrier | 16 | 3,140 | 0 |
| route | 224 | 281 | 0 |

**Proof of correctness.** The last 7 or 14 slots of every window sum **exactly** to the Phase 1 history features of the same flight, so the GRU and LogReg see the same past. The network reads per-slot channels:
- log arrival count, late rate, mean delay, log departure count, cancellation rate (each standardised with statistics from the final training rows only);
- a 0/1 flag for days with no arrivals.

## How leakage is prevented

Seven rules, each enforced by code or by a test:

| # | Rule | Enforced by |
|---|---|---|
| G1 | **Decision clock.** An outcome feeds day D only if known by 20:00 on D−1 | Phase 1's `available_date` and history code are imported, not rewritten |
| G2 | **Label availability.** Every fit (network, vocabulary, scaler, sequence normaliser, calibrator, threshold) uses only labels available at its own date | `guards.assert_labels_available` inside every fit |
| G3 | **Selection on CV only.** Architectures, hyperparameters and ablations are chosen on the May–Aug folds | search functions never receive Sep–Dec labels |
| G4 | **Calibration months stay calibration months.** Sep–Oct serve only calibration, thresholds and rescheduling tuning | no training or early-stopping call receives Sep–Oct rows |
| G5 | **Test lock.** Nov–Dec labels load only in `run_phase2.py --final`, with the rules document committed | `testlock.require_final` (see [06](06_rules_and_limitations.md)) |
| G6 | **Same inputs as Phase 1.** Same features, split and folds | row-alignment test against `aerocast.pipeline.prepare()` |
| G7 | **No same-day aircraft rotation.** No input counts a tail number's day-D legs (Phase 1 found this leak: cancelled flights have blank tail numbers) | tail-number windows are refused; blanking every tail number changes nothing |

### Risks specific to deep learning

| Risk | Why it would leak | What Phase 2 does |
|---|---|---|
| Batch normalisation | a test flight's score would depend on the other test flights in its batch | LayerNorm only; a test checks that a row scores the same alone or in a batch |
| Early stopping on the validation month | the stopping epoch would be chosen on the labels being scored | early stopping uses the last 14 days of each fold's training window instead |
| Rare-ID embeddings reused downstream | an embedding of a tail number with few flights memorises its labels, like target encoding | only carrier, origin and destination embeddings are passed to XGBoost |
| A frozen, in-sample LogReg logit as a fixed input | the network would learn residuals of over-fitted training logits | Wide & Deep trains the LogReg part jointly, starting from the fold's own LogReg |
| History bucketed by flight date | an evening arrival on D−1 would reach day D | windows reuse Phase 1's availability-date table |
| Rescheduling scored with what really happened | the rule would get credit for hindsight about weather | the benefit is computed from predicted risk only |

Shuffling rows inside a training window is safe: every feature is already computed causally, so the order of rows in a batch carries no information about the future.

### The tests

The 60 tests in `tests/` include these leakage checks:

- **Inputs:**
  - tensor rows match Phase 1 one-to-one;
  - changing every Sep–Dec row leaves all fitted statistics unchanged;
  - no target row has a missing tail number.
- **Sequences:**
  - windows equal the Phase 1 history sums for every flight;
  - scrambling all outcomes from a cutoff date leaves earlier windows unchanged, with a positive control showing later ones do change;
  - a D−1 arrival after 20:00 reaches D+1's window, not D's;
  - a realistic same-day cancellation changes nothing;
  - the sequence normaliser ignores every outcome from 1 September on.
- **Training:**
  - early-stopping rows lie inside the fold and before the validation month;
  - evaluation runs in eval mode;
  - the wide part starts from a LogReg fitted on that fold only;
  - only allowed embeddings reach XGBoost.
- **Calibration and test:**
  - a rolling calibrator for day D uses only outcomes known by D's decision clock;
  - test labels cannot be loaded without `--final`, or while the rules document has uncommitted changes.
- **Rescheduling:**
  - the scorer gives identical results with every day-D outcome set to missing;
  - its parameters are tuned on Sep–Oct only.

**Do the tests catch real leaks?** We introduced two leaks on purpose by temporarily patching `available_date`:
- **Outcomes bucketed by flight date:** fails **11** tests.
- **A 23:00 clock instead of 20:00:** fails **4**.
