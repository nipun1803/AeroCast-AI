# Limitations and assumptions

> Mirrors the closing section of [notebook 04](../notebooks/04_evaluation.ipynb).

## Limitations

* **No weather, by design.** Observed weather for day D would **leak at the 20:00 D-1 decision clock**: it is not known then. Weather could only be added from **archived forecasts issued before 20:00 on D-1**, never from observations. Forecast weather is a potentially useful missing predictor; these experiments do not measure its contribution.
* **Single year, three origins.** The **chronological split is retained** (train on the past, test on the future), but a single year (2013) and three NYC origins **cannot establish cross-year or cross-airport generalisation**: each season and airport effect is observed once.
* **Calibration deteriorates in December.** PSI and row-level adversarial validation describe period differences ([04](04_evaluation.md)); they do not demonstrate that those differences caused the error. Repeated daily features also limit the interpretation of row-split adversarial scores.
* **Partial aircraft rotation:** only outbound NYC legs are visible.
* **The model comparison does not establish equivalence or noninferiority.** LogReg is selected by higher development CV AP; the test difference interval includes zero. Four development folds and prior test exposure limit the conclusion.
* **Test was not perfectly clean** ([development protocol](development_protocol.md)).

## Assumptions

* Tail-number assignment is known by 20:00 on D-1.
* Holidays follow the US **federal** calendar (`days_to_holiday`, capped at 14).
* A cancellation is known by the scheduled departure; an arrival outcome at the actual gate arrival.
* Destination time zones: fixed table + the 2013 DST rule.
* Cancelled/diverted flights are dropped from the target (counted as delayed in the sensitivity analysis).
* Phase 1 is machine learning only: no deep-learning model. The IEEE report (`report/main.tex`, with a related-work section) is a separate deliverable.
