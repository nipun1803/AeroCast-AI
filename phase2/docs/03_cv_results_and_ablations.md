# 03 · Cross-validation results, diagnostics and ablations

> Covers notebooks [06](../notebooks/06_embedding_mlp.ipynb), [08](../notebooks/08_wide_deep_hybrid.ipynb) and [09_ablation_studies](../notebooks/09_ablation_studies.ipynb). Numbers from `results/cv_models.csv`, `diagnostics.csv` and `ablation.csv`. Only training months are used here.

## Model comparison

Rolling-origin CV average precision. Networks are 5-seed ensembles; the Phase 1 models are scored as deployed.

| Model | Parameters | CV AP | May | Jun | Jul | Aug |
|---|---|---|---|---|---|---|
| **LogReg (Phase 1)** | 154 | **0.471** | 0.434 | 0.505 | 0.552 | 0.394 |
| Wide & Deep | 37,255 | 0.468 | 0.439 | 0.492 | 0.554 | 0.389 |
| MLP + GRU history | 37,101 | 0.467 | 0.434 | 0.503 | 0.550 | 0.381 |
| Embedding MLP | 27,501 | 0.465 | 0.432 | 0.504 | 0.548 | 0.378 |
| Embeddings → XGBoost | – | 0.450 | 0.419 | 0.438 | 0.549 | 0.395 |
| XGBoost (Phase 1) | – | 0.450 | 0.432 | 0.425 | 0.548 | 0.395 |

- **Differences between models are tiny next to differences between months.** August scores about 0.39 and July about 0.55, whatever the model. Seed-to-seed spread (about 0.004–0.005) is also small next to the fold spread (about 0.07).
- **No network beats LogReg.** The selection rule ([04](04_calibration_and_test.md)) therefore keeps LogReg. Wide & Deep and the GRU model are within 0.005 of it but have about 240 times as many parameters.

## Why the networks trail LogReg

Single-seed CV runs, each changing one thing:

| Run | CV AP | Best epochs per fold |
|---|---|---|
| D0 LogReg on all fold rows (Phase 1) | 0.4714 | – |
| D1 LogReg without the fold's last 14 days | 0.4668 | – |
| D2 linear network (no hidden layer), early-stopped | 0.4586 | 5 / 8 / 3 / 8 |
| D3 linear network, retrained on all fold rows | 0.4640 | 5 / 8 / 3 / 8 |
| D4 selected MLP, early-stopped | 0.4588 | 1 / 1 / 1 / 2 |
| D5 selected MLP, retrained on all fold rows | 0.4644 | 1 / 1 / 1 / 2 |
| D6 selected MLP retrained, without day-level history | 0.4515 | 1 / 1 / 3 / 1 |

What the rows show:

1. **The training code is sound.** A network with no hidden layer is a logistic regression. Trained our way, it lands close to scikit-learn's LogReg (D3 0.464 vs D0 0.471). The remaining gap is consistent with the different penalty and solver; we did not test that further.
2. **Losing the last 14 days costs about 0.005.** It costs LogReg too (D0 → D1). That is why CV now retrains on all fold rows (D4 → D5); see [02](02_models_and_training.md).
3. **The hidden layers overfit almost at once.** The MLP's best epoch is 1 or 2 in every fold, against 3–8 for the linear network. Training loss keeps falling while holdout loss rises (`figures/learning_curves.png`). The history features are already smoothed rates, so the remaining signal is close to linear in them. Grinsztajn et al. (2022) describe this as the setting where networks gain least on tabular data.
4. **Day-level history carries signal.** Removing the day-level history features *lowers* the MLP (D6). This is consistent with them carrying real signal rather than only helping the network memorise days, but it does not rule memorisation out.

## Ablations

Each ablation changes one thing in a selected model. It is re-scored as a 5-seed ensemble on the same folds. Differences smaller than the seed SD (about 0.004) are noise.

| Ablation | CV AP | Change | Folds improved (of 4) |
|---|---|---|---|
| MLP: no early stopping (all 40 epochs) | 0.431 | −0.035 | 0 |
| MLP: − entity-history features | 0.448 | −0.017 | 0 |
| GRU replaces the hand-built history features | 0.453 | −0.015 | 0 |
| MLP: − congestion features | 0.456 | −0.010 | 0 |
| MLP: − day-level history features | 0.456 | −0.010 | 1 |
| MLP: embedding width 2 | 0.462 | −0.004 | 0 |
| MLP: − schedule features | 0.463 | −0.002 | 2 |
| MLP: embedding width 16 / 4 / 8 | 0.463–0.464 | −0.002 | 0–1 |
| GRU: 14-day / 7-day window | 0.466–0.467 | −0.001 | 1 |
| MLP: one-hot instead of embeddings | 0.464 | −0.001 | 1 |
| MLP: + tail, flight and origin-hour IDs | 0.466 | +0.000 | 2 |
| MLP: no dropout or weight decay | 0.466 | +0.000 | 2 |
| Wide & Deep: one-hot instead of embeddings | 0.464 | −0.005 | 0 |
| Wide & Deep: without the GRU | 0.468 | −0.000 | 2 |

What they answer:

- **Is regularisation needed?** Early stopping is essential. Dropout and weight decay make no difference: with only one or two epochs, the network never trains long enough to overfit in a way they could prevent.
- **Which features matter to a network?** The same ones that mattered to LogReg in Phase 1: entity history, then congestion and day-level history.
- **Can learned sequences replace hand-built history (the brief's hypothesis)?** No. Swapping the history features for the GRU costs 0.015. On top of them, the GRU adds little, and the window length doesn't matter.
- **Do embeddings help?** Not measurably in the MLP (−0.001 with one-hot). In Wide & Deep, one-hot costs 0.005 in all four folds, a small but consistent effect. With 16 carriers and 3 airports there is little to compress, and high-cardinality IDs add nothing.
- **Does Wide & Deep need the GRU?** No: removing it changes CV AP by less than 0.001. Before the 2026-11-17 audit fix, both Wide & Deep ablations seemed to *improve* the model (+0.003 without the GRU, +0.001 with one-hot); that was an artefact of the leaked early-stopping holdout ([06](06_rules_and_limitations.md)). Ablations are not candidates: the candidate list was fixed before they ran.
