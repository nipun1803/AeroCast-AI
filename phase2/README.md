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
  notebooks/0*.ipynb notebooks/1*.ipynb
```

| | |
|---|---|
| [05_network_inputs](notebooks/05_network_inputs.ipynb) | Step 2: network inputs, fitting clock, cold-start coverage |
| [06_embedding_mlp](notebooks/06_embedding_mlp.ipynb) | Step 3: embedding MLP search, 5-seed CV, learning curves, learned embeddings |
| [07_gru_history_model](notebooks/07_gru_history_model.ipynb) | Step 4: causal history windows, decision clock, GRU model |
| [08_wide_deep_hybrid](notebooks/08_wide_deep_hybrid.ipynb) | Step 5: Wide & Deep on the Phase 1 LogReg; embeddings into XGBoost |
| [09_ablation_studies](notebooks/09_ablation_studies.ipynb) | Step 6: ablations on rolling CV |
| [10_calibration_and_test](notebooks/10_calibration_and_test.ipynb) | Step 7: Sep→Oct calibration, final-model rule, the one test run |
| [11_whatif_rescheduling](notebooks/11_whatif_rescheduling.ipynb) | Step 8: what-if rescheduling under explicit assumptions |
| `src/aerocast_dl/` | `config` (Phase 2 settings), `guards` (leakage assertions), `tensors` (train-only encoders), `sequences` (causal daily history windows), `models` (GRU encoder) |
| `results/` | `data_summary.json`, `cold_start.csv`, `sequence_summary.json` |
