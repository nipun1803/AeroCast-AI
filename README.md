# AeroCast-AI — Day-Ahead Flight Delay Risk

**AML Project 21 · Phase 1 (Machine Learning)**: predicting whether a flight will arrive ≥ 15 min late, using only information available **the evening before departure**.

Dataset: [`nycflights13`](https://www.kaggle.com/datasets/mahoora00135/flights): 336,776 departures from EWR, JFK and LGA in 2013.

## Problem framing

| Choice | Decision | Why |
|---|---|---|
| Decision time | **End of day D-1** (day-ahead) | Clean leakage boundary: features may use the published schedule for D and outcomes from dates < D only |
| Target | `arr_delay ≥ 15` (BTS on-time definition) | 24.5 % positive rate |
| Cancelled / diverted | Dropped from the target (2.8 %) | Kept in *schedule* counts, since at D-1 nobody knows a flight will be cancelled |
| Split | Chronological: **train** Jan 8–Aug · **calib** Sep–Oct · **test** Nov–Dec | No shuffling; test touched once |
| Model selection | **Rolling-origin CV** inside train (validate May, Jun, Jul, Aug) | A single validation month (Sep, the calmest month of the year) proved unreliable |

## Features (45, `src/aerocast/features.py`)

| Family | Examples | Known at D-1 because… |
|---|---|---|
| IDs | carrier, origin, dest (native categorical / one-hot) | schedule |
| Schedule | cyclic dep/arr time-of-day, cyclic day-of-week, days-to-holiday, log distance | schedule |
| Congestion | scheduled departures in the same origin-hour / ±1 h bank / day; aircraft leg index | schedule (incl. later-cancelled flights) |
| Entity history | smoothed delay rate & volume for route, carrier-route, carrier, dest, origin×hour, flight number, tail number (7/28-day windows) | outcomes of dates ≤ D-1 |
| Day-level history | yesterday's network / origin delay rate, cancellation rate, mean delay | outcomes of D-1 |

History rates are computed on daily totals, then rolled over the window, then **shifted by one day**, and shrunk toward a past-only network prior: `(late + m·prior) / (flights + m)`.
`tests/test_no_leakage.py` scrambles all outcomes from a cutoff date onward and asserts every feature up to that date is **bit-for-bit unchanged**.

## Results (test = Nov–Dec 2013, 53,991 flights, base rate 0.258)

| Model | PR-AUC | ROC-AUC | Brier | ECE | Recall @ precision ≥ 0.40 |
|---|---|---|---|---|---|
| Prior (base rate) | 0.258 | 0.500 | 0.192 | 0.007 | — |
| Route-history heuristic | 0.340 | 0.615 | 0.188 | 0.040 | 0.095 |
| **Logistic regression** (C = 0.001, Platt) | **0.405** | **0.666** | 0.184 | 0.072 | **0.484** |
| **XGBoost** (depth 3, 294 trees, raw) | 0.392 | 0.664 | **0.181** | **0.036** | 0.433 |

Rolling-CV PR-AUC: LogReg 0.484 ± 0.070, XGBoost 0.470 ± 0.074.

### Key findings

1. **XGBoost ≈ Logistic Regression.** The day-block bootstrap 95 % CI for the PR-AUC difference (XGB − LR) is [−0.030, +0.008]. With hand-built day-ahead summaries, trees find little interaction structure to exploit. This is the gap motivating the learned-embedding model in Phase 2.
2. **Time of day + entity history carry the signal.** In the ablation, entity history adds +0.05 test PR-AUC. Scheduled congestion adds nothing beyond time of day.
3. **Day-level features act as a date identifier.** They take one value per day, so trees can memorise individual training days. They help in CV (bursty summer storms) but not on test.
4. **Calibration does not survive the seasonal shift.** Post-hoc calibrators fit on calm Sep–Oct (15.7 % late) under-predict December (33.5 % late). Raw log-loss models are closer to calibrated. Calibration is a property of the model *and* the distribution (Guo et al. 2017; Ovadia et al. 2019).
5. **Operating point.** A precision threshold frozen on Sep–Oct does not transfer, because precision moves with the base rate. A **daily top-10 % capacity policy** is rank-based and shift-robust: 44 % precision, 1.7× lift on test.

## Repository layout

```text
data/raw/flights.csv          dataset (git-ignored; see Setup)
src/aerocast/                 ALL logic lives here; script and notebooks only call it
  config.py                   paths, target, splits, CV folds, operating policy
  data.py                     loading, target, chronological split
  eda.py                      data-quality audit, leakage audit table, EDA aggregations
  features.py                 schedule / congestion / past-only history features
  leakage.py                  scramble-future-outcomes leakage check
  models.py                   LogReg & XGBoost factories, rolling-origin CV
  evaluate.py                 PR/ROC, ECE, calibrators, thresholds, day-block bootstrap
  pipeline.py                 one function per pipeline stage (prepare -> train -> calibrate -> evaluate)
  plots.py                    report figures (PNG + vector PDF for LaTeX)
scripts/run_phase1.py         runs every stage, writes results/ + figures/ (source of truth for numbers)
notebooks/                    one notebook per stage, kernel "Python (aerocast)"
  01_eda.ipynb                data quality, leakage audit, target, delay patterns -> design decisions
  02_feature_engineering.ipynb  past-only rule, feature groups, worked shift+smoothing example,
                              inline leakage check, why no month feature / separate day-level group
  03_model_training.ipynb     split, baselines, rolling-origin CV (LogReg, XGBoost), final fits;
                              recomputes training and asserts it matches results/
  04_evaluation.ipynb         calibration, test metrics, bootstrap, curves, drift, policies,
                              ablation, importance; recomputed from results/predictions.csv.gz
tests/test_no_leakage.py      leakage, split, CV-fold and smoothing guards
results/                      metrics tables, CV logs, predictions.csv.gz, summary.json
figures/                      all figures used in the report
```

Run order: `scripts/run_phase1.py` first (writes `results/`), then the notebooks in order. Notebooks 01–03 run standalone; 04 reads the script's artefacts.

## Setup & reproduce

```bash
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
# place flights.csv in data/raw/

.venv/bin/python -m pytest                  # leakage tests (~12 s)
.venv/bin/python scripts/run_phase1.py      # full pipeline (~2 min; --quick for a smoke run)
.venv/bin/python -m ipykernel install --sys-prefix --name aerocast
.venv/bin/jupyter lab notebooks/            # select kernel "Python (aerocast)"
```

All randomness is seeded (`config.SEED = 42`).

> **macOS note:** if `import aerocast` fails inside the venv, macOS may have flagged the `.pth` files as hidden (common in iCloud-synced folders), and Python skips hidden `.pth` files. Fix with `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. The notebooks add `src/` to `sys.path` themselves, so they work either way.

## Limitations

- One year and three origins: seasonality and airport effects are each observed only once.
- No weather. Only outbound NYC legs are visible, so aircraft rotation is partial (1.3 NYC departures per tail per day).
- Assumes `tailnum` assignment is known day-ahead.
