# AeroCast-AI — Day-Ahead Flight Delay Risk

**AML Project 21 · Phase 1 (Machine Learning)**: predicting whether a flight will arrive ≥ 15 min late, using only information available at **20:00 on the evening before departure**.

Dataset: [`nycflights13`](https://www.kaggle.com/datasets/mahoora00135/flights): 336,776 departures from EWR, JFK and LGA in 2013.

## Problem framing

| Choice | Decision | Why |
|---|---|---|
| Decision time | **20:00 NYC on D-1** (`config.DECISION_HOUR`) | An outcome feeds day D's features only if it was **known** by then: arrival delay at the actual gate arrival (converted to the NYC clock), cancellation at the scheduled departure |
| Target | `arr_delay ≥ 15` (BTS on-time definition) | 24.5 % positive rate |
| Cancelled / diverted | Dropped from the target (2.8 %) | Kept in *schedule* counts. Counting them as delayed is a reported sensitivity analysis |
| Split | Chronological: **train** Jan 8–Aug · **calib** Sep–Oct · **test** Nov–Dec | No shuffling |
| Model selection | **Rolling-origin CV** inside train (validate May, Jun, Jul, Aug) | A single validation month (Sep, the calmest of the year) proved unreliable |
| Final model | **Logistic regression** (`config.FINAL_MODEL`), fixed before test | Higher rolling-CV PR-AUC *as deployed*: 0.4705 vs XGBoost 0.4450 |

## Features (45, `src/aerocast/features.py`)

| Family | Examples | Known at the decision time because… |
|---|---|---|
| IDs | carrier, origin, dest | timetable |
| Schedule | cyclic dep/arr time-of-day and day-of-week, days to the nearest US federal holiday, log distance | timetable |
| Congestion | scheduled departures in the same origin-hour / ±1 h bank / day; aircraft leg index | timetable (incl. later-cancelled flights) |
| Entity history | smoothed delay rate and volume for route, carrier-route, carrier, dest, origin × hour, flight number, tail number (7/14-day windows) | outcomes known by 20:00 on D-1 |
| Day-level history | last-24 h network / origin delay rate, cancellation rate, mean delay | outcomes known by 20:00 on D-1 |

* **Decision clock:** each outcome gets an *availability date* (`data.available_date`), and history windows roll over availability dates. 31 % of D-1 arrivals land after 20:00 (and are twice as often late), so they first count for D+1.
* **Shrinkage:** rates are pulled toward a past-only network prior, `(late + m·prior)/(flights + m)`. m = 5 and the 14-day window were chosen by rolling CV (`results/history_tuning.csv`).
* **Leakage tests** (`tests/test_no_leakage.py`): (a) scrambling all outcomes from a cutoff date leaves earlier features bit-for-bit unchanged; (b) scrambling D-1 arrivals after 20:00 leaves day D unchanged, while scrambling those before 20:00 changes it (positive control).

## Results (test = Nov–Dec 2013, 53,991 flights, base rate 0.258)

`_cal` = the calibration chosen by the static rule fixed in advance (below). PR-AUC is average precision.

| Model | PR-AUC | ROC-AUC | Brier | ECE | ECE (equal-mass) |
|---|---|---|---|---|---|
| Prior (base rate) | 0.258 | 0.500 | 0.192 | 0.007 | 0.084 |
| Route-history heuristic | 0.353 | 0.621 | 0.187 | 0.037 | 0.038 |
| **Logistic regression (final)**: C = 0.1, Platt by rule | 0.394 | 0.654 | 0.187 | 0.082 | 0.082 |
| ↳ same model, causal rolling Platt (28 d) | 0.394 | 0.654 | 0.181 | **0.027** | 0.027 |
| XGBoost: depth 2, 143 trees, raw by rule | 0.407 | 0.662 | 0.180 | 0.027 | 0.027 |

**Primary operating policy (fixed before test): flag the riskiest 10 % of each day's flights.** Final model on test: **precision 0.444, recall 0.173, lift 1.72×** over random flagging.

* Rolling-CV PR-AUC as deployed: LogReg 0.4705 ± 0.071, XGBoost 0.4450 ± 0.073. XGBoost's search score with in-fold early stopping (0.4754) is optimistic and not used for selection.
* Test difference XGB − LR in PR-AUC: day-block bootstrap 95 % CI **[−0.010, +0.041]**, so not significant.
* `oracle_recall@p` (in `results/`) tunes the threshold on the test labels. It's a curve metric **for model comparison only**, not an operating result.

### Key findings

1. **The decision-time leak was real.** Late-evening D-1 arrivals leaked into "yesterday" features. Fixing it costs about 0.018 rolling-CV PR-AUC. The 20:00 clock beats a 2-day lag by about 0.01 in all 4 folds.
2. **LogReg ≈ XGBoost.** CV favours LogReg as deployed, and the test difference is not significant. Hand-built day-ahead summaries leave little interaction structure for trees. That's the gap the deep-learning phase targets.
3. **Every feature family adds CV value for the final model** (congestion +0.021, entity history +0.020, day-level history +0.011). Entity history is confirmed on test (CI excludes 0). Month features don't help (CV).
4. **Static calibration breaks under seasonal shift.** The rule adopts Platt for LogReg (Oct ECE −0.026), which then under-predicts December (33.5 % late). Causal rolling recalibration tracks the shift (test ECE about 0.025 for both models).
5. **Robust to the target definition** (cancellations as delays): same conclusions, same policy lift.
6. **Strong covariate shift**: adversarial AUC is 1.0 with history features and 0.56 with IDs only. This explains the calibration behaviour.

## Development protocol (honest disclosure)

The test window was meant to be touched once. These earlier diagnostics **displayed test metrics** before the design was frozen:
1. the first single-September run;
2. the month-feature comparison (the original justification, since withdrawn and re-justified by CV);
3. feature-subset variants that printed test PR-AUC, which shaped the day-level hypothesis (now re-justified by CV and distinct-value analysis);
4. the 0.50 → 0.40 precision-target change, decided on Sep–Oct but with test recall visible;
5. calibration: test ECE of raw/Platt/isotonic was seen before any selection rule existed. The shift-aware Platt variant was scored on test, and the Sep→Oct rule and the 0.005 margin rule were introduced afterwards;
6. in this review round, the full pipeline (including test) ran once while the final-model comparison still used XGBoost's optimistic in-fold early-stopping score. That was corrected for a CV-methodological reason, which reversed the choice to LogReg.

Everything decided **without** test evidence: decision clock vs lag (CV), smoothing m and window (CV), C and the XGBoost configuration (CV), final model (CV as deployed), calibration rule outcome (Sep→Oct), rolling window N = 28 (a priori), daily top-10 % policy (a priori), precision target (Sep–Oct). The full list, with each re-justification, is in `notebooks/04_evaluation.ipynb` §0. **Small test differences should be read with caution.**

## Assumptions

- Tail-number (aircraft) assignment is known by 20:00 on D-1.
- Holidays come from the **US federal** holiday calendar (`days_to_holiday`, capped at 14).
- A cancellation is known by the flight's scheduled departure; an arrival outcome at the actual gate arrival.
- Destination time zones are a fixed lookup table (`config.DEST_OFFSET_MIN`, re-derivable from the data and checked by a test), plus the 2013 DST rule (HNL, PHX, Puerto Rico and USVI don't observe DST).
- **Single year**: each season and airport effect is observed once, and no out-of-year validation is possible.

## Repository layout

```text
data/raw/flights.csv          dataset (git-ignored; download from Kaggle)
src/aerocast/                 ALL logic lives here; script and notebooks only call it
  config.py                   paths, target, decision clock, time-zone table, splits, CV folds,
                              selection rules (final model, calibration margin, rolling window), policy
  data.py                     loading, target (+ variants), arrival on NYC clock, availability date, split
  eda.py                      data-quality audit, leakage audit, EDA aggregations
  features.py                 schedule / congestion / decision-clock history features
  leakage.py                  future-scramble and decision-clock leakage checks
  models.py                   LogReg & XGBoost factories, rolling-origin CV
  evaluate.py                 PR/ROC, ECE (equal-width & equal-mass), calibrators + margin rule,
                              rolling Platt, thresholds, policies, day-block bootstrap
  drift.py                    PSI and adversarial validation
  pipeline.py                 one function per stage (prepare -> tune -> train -> calibrate -> evaluate)
  plots.py                    report figures (PNG + vector PDF for LaTeX)
scripts/run_phase1.py         runs every stage, writes results/ + figures/ (source of truth for numbers)
notebooks/                    one notebook per stage, kernel "Python (aerocast)"
  01_eda.ipynb                data quality, leakage audit, delay patterns, congestion within hour,
                              history multicollinearity -> design decisions
  02_feature_engineering.ipynb  decision clock, feature groups, worked example, both leakage checks,
                              CV tuning of m / window / clock, month and day-level evidence
  03_model_training.ipynb     theory, rolling-origin CV for both models, final-model rule;
                              recomputes training and asserts it matches results/
  04_evaluation.ipynb         development protocol, metric theory, calibration study, test metrics,
                              bootstrap, policies, ablation (both models), importance, target
                              sensitivity, covariate shift; recomputed from results/predictions.csv.gz
tests/test_no_leakage.py      leakage, decision clock, time zones, splits, CV folds, smoothing
tests/test_evaluate.py        ECE, thresholds, bootstrap, calibrators, daily policy, rolling Platt
results/                      every table + predictions.csv.gz + summary.json (committed)
figures/                      all figures used in the report
```

`results/predictions.csv.gz` (~6 MB) **is committed**: it's the audit trail behind every evaluation number, and it lets notebook 04 run without re-training.

## Setup & reproduce

```bash
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
# place flights.csv in data/raw/

.venv/bin/python -m pytest                  # 24 tests (~30 s)
.venv/bin/python scripts/run_phase1.py      # full pipeline (~10 min; --quick for a smoke run)
.venv/bin/python -m ipykernel install --sys-prefix --name aerocast
.venv/bin/jupyter lab notebooks/            # select kernel "Python (aerocast)"
```

Run order: the script first (it writes `results/`), then notebooks 01 → 04. Notebooks 01–03 run standalone; 04 reads the script's artefacts. All randomness is seeded (`config.SEED = 42`).

> **macOS note:** if `import aerocast` fails inside the venv, macOS may have flagged the `.pth` files as hidden (common in iCloud-synced folders), and Python skips hidden `.pth` files. Fix with `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. The script and notebooks add `src/` to `sys.path` themselves, so they work either way.

## Limitations

- One year and three origins (see Assumptions).
- No weather data, the main missing driver of day-to-day delays.
- Only outbound NYC legs are visible, so aircraft rotation is partial (about 1.3 NYC departures per aircraft per day).
- The test window was not perfectly clean (see Development protocol).
