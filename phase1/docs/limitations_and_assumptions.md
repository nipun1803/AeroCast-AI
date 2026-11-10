# Limitations and assumptions

> Mirrors the closing section of [notebook 04](../notebooks/04_evaluation.ipynb).

## Limitations

* **No weather, by design.** Observed weather for day D would **leak at the 20:00 D-1 decision clock**: it is not known then. Weather could only be added from **archived forecasts issued before 20:00 on D-1**, never from observations. Forecast weather is a potentially useful missing predictor; these experiments do not measure its contribution.
* **Single year, three origins.** The **chronological split is retained** (train on the past, test on the future), but a single year (2013) and three NYC origins **cannot establish cross-year or cross-airport generalisation**: each season and airport effect is observed once.
* **Calibration deteriorates in December.** PSI and row-level adversarial validation describe period differences ([04](04_evaluation.md)); they do not demonstrate that those differences caused the error. Repeated daily features also limit the interpretation of row-split adversarial scores.
* **No aircraft-rotation features:** only outbound NYC legs are visible, and counting an aircraft's legs by tail number leaks same-day cancellations (the data blank the tail number of cancelled flights), so delay propagation along a rotation is not modelled.
* **The model comparison does not establish equivalence or noninferiority.** LogReg is selected by higher development CV AP; the test difference interval includes zero. Four development folds and prior test exposure limit the conclusion.
* **Test was not perfectly clean** ([development protocol](development_protocol.md)).

## Assumptions

* Tail-number assignment is known by 20:00 on D-1. The recorded `tailnum` is the aircraft that flew, so same-day swaps are not modelled. The two remaining tail-history features use past outcomes only; dropping them changes main-target LogReg test AP by less than 0.0001.
* A missing tail number occurs only for cancelled flights, so tail features are excluded from the cancellation-inclusive sensitivity target, where they would reveal the label.
* Holidays follow the US **federal** calendar (`days_to_holiday`, capped at 14).
* A cancellation is known by the scheduled departure; an arrival outcome at the actual gate arrival.
* Destination time zones: fixed table + the 2013 DST rule.
* Cancelled/diverted flights are dropped from the target (counted as delayed in the sensitivity analysis).
* Phase 1 is machine learning only: no deep-learning model. The IEEE report (`report/main.tex`, with a related-work section) is a separate deliverable.
