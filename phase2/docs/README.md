# AeroCast-AI Phase 2: documentation

## What this project does

At **20:00 New York time on the evening before a flight**, AeroCast-AI estimates the probability that the flight will arrive **at least 15 minutes late**. The data are the 2013 departures from EWR, JFK and LGA (`nycflights13`). A daily review desk would use the estimates to look at the riskiest flights of the next day, and possibly re-time a few of them.

Phase 1 built this pipeline with classical machine learning. A regularised logistic regression (LogReg) on 43 hand-built features won on cross-validation, ahead of XGBoost.

**Phase 2 asks one question:** do deep-learning models find signal those hand-built features miss? The candidates are learned embeddings for airlines and airports, a recurrent network over each airport's, airline's and route's recent delay history, and a hybrid that starts from LogReg and lets a network add to it.

For a fair answer, Phase 2 changes **only the model**. Everything below is imported unchanged from Phase 1:

- the decision clock;
- the target;
- the 43 features;
- the train / calibration / test split;
- the four cross-validation folds;
- the evaluation code.

## The answer

| Model | CV AP (May–Aug) | Test AP (Nov–Dec) | Test AP − LogReg [95% CI] |
|---|---|---|---|
| **LogReg (Phase 1), final model** | **0.471** | 0.392 | – |
| Wide & Deep (LogReg + MLP + GRU) | 0.469 | 0.399 | +0.007 [−0.005, +0.023] |
| MLP + GRU history | 0.467 | 0.398 | +0.006 [−0.013, +0.028] |
| Embedding MLP | 0.465 | 0.395 | +0.003 [−0.015, +0.027] |
| Embeddings → XGBoost | 0.450 | 0.400 | +0.009 [−0.014, +0.033] |
| XGBoost (Phase 1) | 0.450 | 0.407 | +0.016 [−0.012, +0.043] |

- **On cross-validation, no network beats LogReg.** The rule fixed before training picks the best CV AP, and among models within 0.005 of it, the one with fewer parameters. So LogReg stays the final model.
- **On the test months, every Phase 2 model is slightly better**, but no confidence interval excludes zero.
- **Why the networks don't win:**
  - Their hidden layers overfit after one or two epochs.
  - The Phase 1 history features are already smoothed delay rates, so what remains to learn is close to linear in them.
  - The GRU cannot replace those features (−0.015 CV AP), and learned embeddings are no better than one-hot encoding.
- **What-if rescheduling** (offline, predicted effects only) avoids 1.9 predicted delays a day with 48 moves a day.

## Reading order

| # | Document | Explains | Notebook |
|---|---|---|---|
| 1 | [01_inputs_and_leakage.md](01_inputs_and_leakage.md) | the decision clock, network inputs, causal history windows, how leakage is prevented and tested | 05, 07 (Part A) |
| 2 | [02_models_and_training.md](02_models_and_training.md) | the four architectures, how they are trained, searched and scored | 06, 07 (Part B), 08 |
| 3 | [03_cv_results_and_ablations.md](03_cv_results_and_ablations.md) | the CV comparison, why the networks trail LogReg, 16 ablations | 06, 08, 09 |
| 4 | [04_calibration_and_test.md](04_calibration_and_test.md) | calibration, the final-model rule, the one test evaluation, where the models differ | 10 |
| 5 | [05_whatif_rescheduling.md](05_whatif_rescheduling.md) | the offline re-timing analysis and its assumptions | 11 |
| 6 | [06_rules_and_limitations.md](06_rules_and_limitations.md) | rules fixed before training, the test lock, every later change, limitations | – |

The 7-page IEEE report is [../report/main.pdf](../report/main.pdf). How to run everything is in [../README.md](../README.md#run).

## Terms used throughout

| Term | Meaning |
|---|---|
| **D, D−1** | the day of the flight, and the evening before it, when the prediction is made |
| **Decision clock** | 20:00 on D−1. An outcome (a delay, a cancellation) can feed a prediction only once it was known by then |
| **AP** | average precision, the area under the precision-recall curve. The main metric; its no-skill value is the late rate (0.258 on test) |
| **CV** | rolling-origin cross-validation inside the training months: train Jan–Apr → validate May, Jan–May → Jun, Jan–Jun → Jul, Jan–Jul → Aug |
| **As deployed** | a model scored in CV exactly as it would be trained for real, including the rows it would really see |
| **ECE** | expected calibration error: the average gap between predicted and observed late rates (15 bins) |
| **Seed ensemble** | 5 copies of a network trained from different random seeds; the deployed prediction is their mean logit |
