# AeroCast-AI Phase 2: Deep Learning and Hybrid

**Team T49 · AML Project 21 · Phase 2 (deep learning)**. Status: **not started**.

Phase 2 adds deep-learning models to the Phase 1 setup ([`../phase1/`](../phase1/)) and keeps everything else identical, so that the comparison is fair:

- **Same inputs:** decision clock 20:00 NYC on D−1, target `arr_delay ≥ 15`, the 43 Phase 1 features, the chronological split (train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec) and the four rolling-origin CV folds (May–Aug). These are imported from the `aerocast` package, not copied.
- **Planned models:** an MLP with learned airline, origin and destination embeddings; a GRU over causally ordered daily delay history; and a Wide & Deep hybrid whose wide part is the Phase 1 logistic regression.
- **Planned analyses:** ablations on rolling CV, calibration, and an offline what-if rescheduling analysis with explicit assumptions.

The selection rules and leakage safeguards are written down and committed in `docs/development_protocol.md` before any model is trained or scored on test.
