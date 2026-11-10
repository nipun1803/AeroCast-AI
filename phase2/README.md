# AeroCast-AI Phase 2: Deep Learning and Hybrid

**Team T49 · AML Project 21 · Phase 2 (deep learning)**. Status: **protocol fixed; network inputs (Step 2) and causal GRU sequences (Step 4, inputs only) built; no model trained yet**.

Phase 2 adds deep-learning models to the Phase 1 setup ([`../phase1/`](../phase1/)) and keeps everything else identical, so that the comparison is fair:

- **Same inputs:** decision clock 20:00 NYC on D−1, target `arr_delay ≥ 15`, the 43 Phase 1 features, the chronological split (train Jan 8–Aug · calibration Sep–Oct · test Nov–Dec) and the four rolling-origin CV folds (May–Aug). These are imported from the `aerocast` package, not copied.
- **Planned models:** an MLP with learned airline, origin and destination embeddings; a GRU over causally ordered daily delay history; and a Wide & Deep hybrid whose wide part is the Phase 1 logistic regression.
- **Planned analyses:** ablations on rolling CV, calibration, and an offline what-if rescheduling analysis with explicit assumptions.

The selection rules and leakage safeguards were committed before any model was trained or scored on test:

| | |
|---|---|
| [docs/development_protocol.md](docs/development_protocol.md) | pre-registered training, selection and calibration rules, test lock, change log |
| [docs/leakage_safeguards.md](docs/leakage_safeguards.md) | global leakage rules, deep-learning-specific risks, and the tests each step must pass |

## Run

From `phase2/`, with the shared venv (`../.venv`) active and `aerocast` installed from `../phase1`:

```bash
python -B -m pytest -q -p no:cacheprovider        # leakage and alignment tests
python -B scripts/run_phase2.py                    # writes results/
python -B -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=aerocast \
  notebooks/05_dl_data.ipynb notebooks/07_temporal_hybrid.ipynb
```

| | |
|---|---|
| [notebooks/05_dl_data.ipynb](notebooks/05_dl_data.ipynb) | network inputs, fitting clock, cold-start coverage |
| [notebooks/07_temporal_hybrid.ipynb](notebooks/07_temporal_hybrid.ipynb) | Part A: causal history windows, decision clock, equality with Phase 1 |
| `src/aerocast_dl/` | `config` (Phase 2 settings), `guards` (leakage assertions), `tensors` (train-only encoders), `sequences` (causal daily history windows), `models` (GRU encoder) |
| `results/` | `data_summary.json`, `cold_start.csv`, `sequence_summary.json` |
