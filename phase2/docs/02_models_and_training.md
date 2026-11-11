# 02 · Models and training

> Covers notebooks [06_embedding_mlp](../notebooks/06_embedding_mlp.ipynb), [07_gru_history_model](../notebooks/07_gru_history_model.ipynb) Part B and [08_wide_deep_hybrid](../notebooks/08_wide_deep_hybrid.ipynb). Code: `src/aerocast_dl/models.py`, `train.py`, `pipeline.py`; settings in `config.py`.

## The four models

All four predict the same thing from the same rows. Each builds on the previous one.

```text
carrier, origin, dest ──► embeddings ─┐
40 numeric features ──────────────────┼─► [Linear → LayerNorm → ReLU → Dropout] × 2 ─► z_deep
28-day windows (origin, carrier, route) ─► 3 GRUs ─┘   (GRU model and Wide & Deep only)

Wide & Deep only:  Phase 1 LogReg inputs ─► linear ─► z_wide;   P(late) = σ(z_deep + z_wide)
```

### 1. Embedding MLP

This is the brief's required deep-learning model. Each carrier, origin and destination gets a learned vector of width 1.6·n^0.56 (rounded, between 2 and 16) for a vocabulary of n values. The vectors and the 40 standardised numeric features feed a small multilayer perceptron.

- **Normalisation:** LayerNorm rather than BatchNorm, so a flight's score never depends on the other flights in its batch.
- **Selected configuration:** 2 hidden layers of 128 units, dropout 0.3, learning rate 1e-3, weight decay 1e-4.
- **Size:** 27,501 trainable parameters.
- **Why embeddings:** an embedding is a compressed one-hot layer. Similar airports can share directions, so a rare one borrows strength from the rest.

### 2. MLP + GRU history

This model adds the brief's optional temporal component. There is one gated recurrent unit (GRU) per entity type: origin, carrier and route. Each reads its 28-day window ([01](01_inputs_and_leakage.md)) from the oldest day to the newest. The final hidden states (16 units each) join the MLP input.

- **What it can learn that LogReg can't:** a Phase 1 history feature is a fixed sum over the last 7 or 14 days. A GRU can learn such sums, and also learn how much recent days should outweigh old ones.
- **Selected configuration:** the MLP part is fixed at the MLP's configuration; GRU width 16, no extra dropout.
- **Size:** 37,101 parameters.

### 3. Wide & Deep (main hybrid)

The output is a **wide** part plus a **deep** part. The wide part is a linear layer on the Phase 1 LogReg design matrix (one-hot categoricals and standardised numerics). The deep part is the better of models 1 and 2, which was the GRU model.

- **Starting point:** the wide part starts with the weights of the LogReg fitted on the **same fold's** rows, and the deep part's output layer starts at zero. So before training, the hybrid **is** Phase 1's LogReg, and training can only learn a correction to its logit.
- **Training:** both parts are trained together, with the wide part at a tenth of the learning rate.
- **Selected configuration:** learning rate 3e-4, weight decay 1e-5.
- **Size:** 37,255 parameters.
- **Why train them together:** freezing the in-sample LogReg logit as a fixed input would teach the network to correct over-fitted training logits, which then look different on new data.

### 4. Embeddings → XGBoost (secondary hybrid)

The MLP's learned carrier, origin and destination vectors are added as extra columns to the Phase 1 XGBoost: same 108 depth-2 trees, same settings.

- **Not used:** tail-number and flight embeddings, which would act like target encoding.
- **Separate process:** XGBoost runs in its own process (`scripts/xgb_worker.py`). PyTorch and XGBoost ship different OpenMP runtimes, and on macOS loading both in one process crashes.

## How every network is trained

| Setting | Value | Why |
|---|---|---|
| Loss | binary cross-entropy, **no class weights** | it is a proper scoring rule, so it targets the true probability. Weighting late flights would shift every logit and mis-calibrate them |
| Optimiser | AdamW, batch 1,024, at most 40 epochs | decoupled weight decay |
| Early stopping | on an **inner holdout**: the last 14 days of the fold's training window; log loss, patience 4 | the validation month never chooses anything |
| Final number of epochs | mean best epoch over the CV folds, rounded up (2 for all three networks) | the final fit has no holdout of its own |
| Seeds | 5 (42–46); the prediction is the mean logit | averages out initialisation noise |
| Determinism | every generator seeded; `torch.use_deterministic_algorithms(True)`; CPU | same seed, same weights (tested) |

### Cross-validation "as deployed"

A model is scored in CV the way it would actually be trained. In each fold:

1. Train on the fold's rows minus its last 14 days, with early stopping on those 14 days. This **only chooses the number of epochs**.
2. Retrain from scratch on **all** of the fold's rows for that many epochs.
3. Score the retrained model once on the validation month.

The first version of the pipeline skipped step 2: it scored the early-stopped model, which never saw the fold's last 14 days. The deployed model would see them, so that version under-rated the networks. The error was found by CV-only diagnostics and corrected before any test data were touched; every number was recomputed ([03](03_cv_results_and_ablations.md), [06](06_rules_and_limitations.md)).

### Search

| Family | Configurations (1 seed, CV AP range) | Grid |
|---|---|---|
| MLP | 20 random, 0.444–0.464 | hidden {64, 128, 256}, depth {2, 3}, dropout {0.1, 0.2, 0.3}, embedding dropout {0, 0.1}, learning rate {1e-3, 3e-3}, weight decay {1e-5, 1e-4, 1e-3} |
| GRU | 12, 0.447–0.464 | GRU width {8, 16, 32}, GRU dropout {0, 0.2}, learning rate {1e-3, 3e-3}, weight decay {1e-5, 1e-4} |
| Wide & Deep | 12, 0.453–0.471 | learning rate {3e-4, 1e-3, 3e-3}, weight decay {1e-5, 1e-4}, wide learning-rate scale {0.1, 1} |

The grids and budgets were fixed in `config.py` before the searches ran. Each configuration is scored with one seed. Only the best of each family is re-scored as a 5-seed ensemble, and that ensemble's CV AP is the number used to compare models.

## Where to see it

- **Search tables:** `results/search_{mlp,gru,wide_deep}.csv`.
- **Selected configurations:** `results/specs.json`.
- **Per-seed, per-fold scores:** `results/cv_seeds.csv`.
- **Learning curves** (training vs holdout loss): `figures/learning_curves.png`.
- **Learned carrier and destination maps:** `figures/embedding_{carrier,dest}.png`.
