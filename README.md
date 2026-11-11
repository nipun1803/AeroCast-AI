# AeroCast-AI: Day-Ahead Flight Delay Risk

**Team T49 · AML Project 21: Flight Delay Prediction and Rescheduling**\
*Project brief: Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning*

Predict whether a flight arrives **≥ 15 min late** using only information available at **20:00 NYC on the evening before departure**, on the [`nycflights13`](https://www.kaggle.com/datasets/mahoora00135/flights) data (336,776 departures from EWR, JFK and LGA in 2013), and explore an offline what-if rescheduling rule.

| Phase | Folder | Scope | Status |
|---|---|---|---|
| **1: Machine learning** | [`phase1/`](phase1/) | Leakage-safe day-ahead features, logistic regression and XGBoost, rolling-origin CV, calibration, IEEE report and slides | Submitted, frozen at git tag `phase1-final` |
| **2: Deep learning and hybrid** | [`phase2/`](phase2/) | Embedding MLP, GRU over causally ordered delay history, Wide & Deep hybrid with the Phase 1 model, ablations, what-if rescheduling, new IEEE report | Complete (slides pending): no network beats the Phase 1 LogReg on CV; see [phase2/README.md](phase2/README.md#results) |

Phase 2 reuses Phase 1's decision clock, target, features, chronological split and CV folds by importing the `aerocast` package from `phase1/src/`, so the two phases are compared on identical, leakage-tested inputs.

## Team

| Member | GitHub ID |
|---|---|
| **Nipun Patlori** | [nipun1803](https://github.com/nipun1803) |
| **Anushka Tyagi** | [anu-ushka](https://github.com/anu-ushka) |
| **Omkar Hadole** | [omkar-hadole](https://github.com/omkar-hadole) |
| **Mohan Kumar C R** | [mohan101mk](https://github.com/mohan101mk) |
| **Krish Dabas** | [krish-2509](https://github.com/krish-2509) |

Roles per phase: [Phase 1](phase1/README.md#team-and-roles) · [Phase 2](phase2/README.md#team-and-roles).

## Setup

One virtual environment at the repository root serves both phases (Python 3.12; on macOS, XGBoost needs `brew install libomp`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r phase1/requirements.txt
python -m pip install -e phase1
```

Place the Kaggle `flights.csv` in `phase1/data/raw/` (excluded from Git). To reproduce Phase 1, follow [phase1/README.md](phase1/README.md#how-to-reproduce), running commands from `phase1/`. Phase 2 additionally needs `python -m pip install -r phase2/requirements.txt -e phase2` (adds PyTorch); see [phase2/README.md](phase2/README.md#run).

## Layout

```text
phase1/   Phase 1 (ML): src/aerocast, scripts, notebooks 01-04, tests, results, figures, docs, report, presentation
phase2/   Phase 2 (DL + hybrid): src/aerocast_dl, scripts, notebooks 05-11, tests, results, figures, docs, report
```
