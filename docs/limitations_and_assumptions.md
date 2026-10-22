# Limitations and assumptions

> Mirrors the closing section of [notebook 04](../notebooks/04_evaluation.ipynb).

## Limitations

* **No weather, by design.** Observed weather for day D would **leak at the 20:00 D-1 decision clock**: it is not known then. Weather could only be added from **archived forecasts issued before 20:00 on D-1**, never from observations. It is the main missing driver of day-to-day delays.
* **Single year, three origins.** The **chronological split is retained** (train on the past, test on the future), but a single year (2013) and three NYC origins **cannot establish cross-year or cross-airport generalisation**: each season and airport effect is observed once.
* **Strong covariate shift** between train and later months (adversarial AUC near 1.0 with history features; [04](04_evaluation.md)), which breaks a frozen calibrator.
* **Partial aircraft rotation:** only outbound NYC legs are visible.
* **LogReg and XGBoost are not statistically distinguished**; the final-model choice rests on a pre-declared rule.
* **Test was not perfectly clean** ([development protocol](development_protocol.md)).

## Assumptions

* Tail-number assignment is known by 20:00 on D-1.
* Holidays follow the US **federal** calendar (`days_to_holiday`, capped at 14).
* A cancellation is known by the scheduled departure; an arrival outcome at the actual gate arrival.
* Destination time zones: fixed table + the 2013 DST rule.
* Cancelled/diverted flights are dropped from the target (counted as delayed in the sensitivity analysis).
* Phase 1 is machine learning only: no deep-learning model. The IEEE report (`report/main.tex`, with a related-work section) is a separate deliverable.
