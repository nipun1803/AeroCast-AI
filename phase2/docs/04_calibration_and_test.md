# 04 · Calibration, the final model and the test evaluation

> Covers notebook [10_calibration_and_test](../notebooks/10_calibration_and_test.ipynb). Code: `src/aerocast_dl/evaluation.py`, `calibrate.py`, `testlock.py`.

## Order of operations

1. **Final fits (no labels scored).** Every network family is retrained on the 211,517 training flights known by 1 September, with 5 seeds and the fixed epoch count. All of them then score every flight; the scores are saved in `results/predictions.csv.gz`, committed before any test label was read.
2. **Calibration and selection (train and Sep–Oct labels only).** Choose a calibrator per model, freeze the decision thresholds, apply the final-model rule.
3. **The test evaluation (Nov–Dec labels, once).** Run only by `run_phase2.py --final`, with the rules committed.

## Calibration

A model's raw probabilities can be systematically too high or too low. Calibration maps them to honest probabilities. Six candidates are compared for every model, Phase 1's included:

| Candidate | How it works |
|---|---|
| none (raw) | the model's own probability |
| Platt / isotonic / temperature (static) | fitted once on September |
| rolling Platt, 28 days | refitted every day on the outcomes known in the last 28 days |
| rolling temperature, 28 days | the same, with temperature scaling |

**Rule.** Each candidate is scored by its October ECE. The lowest wins, but only if it beats raw by more than 0.005.

| October ECE | LogReg | XGBoost | MLP | MLP + GRU | Wide & Deep | Emb. → XGB |
|---|---|---|---|---|---|---|
| none (raw) | 0.063 | **0.023** | 0.038 | 0.064 | 0.058 | 0.031 |
| rolling Platt, 28 d | **0.021** | 0.031 | **0.026** | **0.020** | **0.023** | 0.032 |
| rolling temperature, 28 d | 0.032 | 0.024 | 0.029 | 0.030 | 0.035 | **0.025** |
| best static | 0.033 | 0.032 | 0.037 | 0.029 | 0.035 | 0.031 |

**Rolling Platt wins for LogReg and every network.** The reason is that the late rate moves between seasons: 26.5% in training, 15.7% in Sep–Oct, 33.5% in December.
- **Temperature scaling** fixes over- or under-confidence but has no intercept, so it cannot follow a new base rate.
- **Platt scaling** has an intercept.
- **The rolling version** re-learns that intercept every day from outcomes already known at the decision clock.

**Difference from Phase 1.** Phase 1's rule compared only static calibrators, so its deployed LogReg used static Platt (test ECE 0.082). Phase 2 applies the larger candidate set to the same saved LogReg scores. The resulting test AP of 0.392 reproduces Phase 1's own rolling-Platt result exactly.

## The final-model rule

Fixed before training:

1. Take the **highest mean CV AP** among LogReg, XGBoost, MLP, GRU model, Wide & Deep and embeddings → XGBoost.
2. If other models are within 0.005 of it, choose the one with the **fewest trainable parameters**.

LogReg has the best CV AP (0.4714). GRU (0.4672) and Wide & Deep (0.4684) are within 0.005 but are far larger. **Final model: Phase 1 LogReg with rolling Platt.** The choice used CV only (`results/selection.json`).

## Test results (Nov–Dec 2013, 53,991 flights)

**The test stage ran twice.** After the first run, a code audit of the inputs and cohorts (not of test results) found two leaks: Wide & Deep's early-stopping wide LogReg had seen the 14-day inner holdout, and the rescheduling candidates and daily flag budgets left out flights later cancelled or diverted, which is only known after day D. Both were fixed with no change to any grid, rule or calibrator candidate, the affected stages were rerun, and the test stage ran again ([06](06_rules_and_limitations.md)). Only Wide & Deep (test AP 0.399 → 0.394) and the top-k and rescheduling numbers changed; the first run's numbers are in git history (commit `020bf24`).

Each model uses its rule-selected calibrator.

| Model | AP | ROC AUC | Brier | ECE | AP − LogReg [95% CI] |
|---|---|---|---|---|---|
| **LogReg (final)** | 0.392 | 0.657 | 0.181 | 0.027 | – |
| Embedding MLP | 0.395 | 0.668 | 0.180 | 0.027 | +0.003 [−0.015, +0.027] |
| MLP + GRU history | 0.398 | 0.668 | 0.180 | 0.035 | +0.006 [−0.013, +0.028] |
| Wide & Deep | 0.394 | 0.656 | 0.181 | 0.036 | +0.003 [−0.016, +0.022] |
| Embeddings → XGBoost | 0.400 | 0.664 | 0.179 | 0.017 | +0.009 [−0.014, +0.033] |
| XGBoost (Phase 1) | 0.407 | 0.665 | 0.179 | 0.027 | +0.016 [−0.012, +0.043] |

- **The order reverses CV, but nothing is significant.** Every Phase 2 model scores slightly above LogReg on test. The intervals come from a paired day-block bootstrap (days resampled, all models scored on the same days), and every one contains zero. Neighbouring days are correlated, which day blocks ignore, so if anything the intervals are too narrow. XGBoost showed the same CV-vs-test reversal in Phase 1. The CV-based choice stands, because the test is not used to choose.
- **Calibration holds on average, not in December.** LogReg's ECE is 0.005 in November but 0.050 in December: the late rate jumps to 33.5%, beyond anything the 28-day window has seen.

### Decisions at two operating points

| Model | Daily top 10%: precision | recall | lift | Frozen threshold: precision | recall |
|---|---|---|---|---|---|
| LogReg (final) | 0.443 | 0.170 | 1.71 | 0.547 | 0.040 |
| Embedding MLP | 0.446 | 0.171 | 1.73 | 0.522 | 0.005 |
| MLP + GRU history | 0.446 | 0.172 | 1.73 | – | 0.000 |
| Wide & Deep | 0.434 | 0.167 | 1.68 | 0.632 | 0.005 |
| Embeddings → XGBoost | 0.437 | 0.168 | 1.69 | 0.467 | 0.211 |
| XGBoost (Phase 1) | 0.435 | 0.168 | 1.68 | 0.549 | 0.070 |

- **Daily top 10% (primary policy).** Each day the desk reviews the 10% riskiest of **all** scheduled flights, including those that will later be cancelled or diverted, since that is unknown at 20:00 (they are about 4% of the flags). Precision and recall are measured on the flights with an outcome: about 44% of those flagged are late, 1.7 times the base rate. This is robust because it ranks within a day.
- **Frozen threshold (secondary policy).** A probability cut chosen on Sep–Oct to reach 40% precision. Its precision holds on test, but it flags far fewer flights. The MLP's and Wide & Deep's thresholds flag 0.3% and 0.2% of flights, and the GRU model's none at all, because probabilities shift with the season.
- **LogReg confusion matrix at top 10%:** 2,367 late flights caught, 2,981 false alarms, 11,579 late flights missed (220 further flags fell on flights without an outcome) (`figures/confusion_test.png`).

### Where the deep models differ (`results/test_slices.csv`)

These breakdowns were fixed before the test run and are descriptive only: they choose nothing.

- **By departure hour.** The GRU gains most for early flights (+0.017 AP for 05–09, +0.015 for 10–13). There, recent entity history is the main signal, because the day's congestion has not built up yet. Wide & Deep is above LogReg in every hour band, but only just (+0.001 to +0.006).
- **Rare destinations.** Destinations with fewer than 50 training flights share one UNK embedding; there all networks gain (Wide & Deep +0.019). But that slice has only 93 flights, and mid-frequency destinations and routes show no consistent gain. The evidence that embeddings help rare categories is weak.
- **By carrier.** The GRU model beats LogReg on 13 of 16 carriers, Wide & Deep on 11 and XGBoost on 15. No individual difference is tested.

### Drift (`results/test_drift.csv`)

PSI against the training rows: below 0.1 is stable, above 0.25 is a large shift.

- **Network scores are stable.** The scores of LogReg and every network move by at most 0.017 from training to Nov–Dec. Carrier and origin shares barely move (≤ 0.004).
- **Tree models look unstable, but it is an artefact.** Their PSI is 0.33–0.38 on Sep–Oct, because their scores on training rows are in-sample.
- **The test-time problem is mainly the outcome rate.** Some inputs, such as holiday distance and recent traffic and delay rates, necessarily shift into the holiday season (this table does not measure feature-level PSI), but the scores stay stable.
