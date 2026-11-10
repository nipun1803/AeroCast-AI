"""Phase 2 report figures. Reuses the Phase 1 style (``aerocast.plots``: validated
categorical slots, light print surface, recessive grid) so both reports match.
Identity is never carried by colour alone: every figure is direct-labelled."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aerocast import plots as p1_plots  # sets the shared matplotlib style
from aerocast.evaluate import reliability_table

plt = p1_plots.plt
SERIES, REFERENCE, TEXT, TEXT_2 = (p1_plots.SERIES, p1_plots.REFERENCE, p1_plots.TEXT,
                                   p1_plots.TEXT_2)
PHASE1, PHASE2 = REFERENCE, SERIES[0]
GAIN, LOSS = SERIES[0], SERIES[1]


def cv_models(tbl: pd.DataFrame, labels: dict, ref: float):
    """Rolling-CV AP per model: fold values (small dots) and the mean (large
    dot, labelled), Phase 1 in grey, Phase 2 in blue, LogReg as a reference line."""
    tbl = tbl.sort_values("cv_pr_auc_mean")
    folds = [c for c in tbl.columns if c.startswith("fold_m")]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(tbl) + 1.0))
    for i, (_, r) in enumerate(tbl.iterrows()):
        col = PHASE1 if r["model"].startswith("Phase 1") else PHASE2
        ax.scatter(r[folds].astype(float), [i] * len(folds), s=14, color=col, alpha=0.45, lw=0)
        ax.scatter([r["cv_pr_auc_mean"]], [i], s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(f"{r['cv_pr_auc_mean']:.3f}", (r["cv_pr_auc_mean"], i), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5, color=TEXT)
    ax.axvline(ref, color=REFERENCE, lw=1, ls="--", zorder=0)
    ax.set_yticks(range(len(tbl)), [labels.get(m, m) for m in tbl["model"]])
    ax.set_xlabel("Rolling-CV average precision (May–Aug folds; large dot = mean)")
    ax.set_title("Model selection by cross-validation", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def ablation(tbl: pd.DataFrame):
    """Change in mean CV AP versus the configuration it ablates, with the mean
    seed SD as a whisker (differences inside it are noise)."""
    tbl = tbl.sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(tbl) + 1.0))
    y = np.arange(len(tbl))
    cols = [GAIN if d > 0 else LOSS for d in tbl["delta"]]
    ax.barh(y, tbl["delta"], color=cols, height=0.6)
    ax.errorbar(tbl["delta"], y, xerr=tbl["seed_sd"], fmt="none", ecolor=TEXT_2, lw=0.8, capsize=2)
    for yi, d in zip(y, tbl["delta"]):
        ax.annotate(f"{d:+.3f}", (d, yi), xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if d >= 0 else "right", va="center", fontsize=7, color=TEXT)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_yticks(y, tbl["label"])
    ax.set_xlabel("Δ mean CV AP vs base configuration (whisker: mean seed SD)")
    ax.set_title("Ablations (5-seed ensembles, rolling CV)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def learning_curves(curves: dict):
    """Training vs inner-holdout log loss per epoch; the dot marks the epoch
    early stopping keeps."""
    fig, axes = plt.subplots(1, len(curves), figsize=(3.3 * len(curves), 2.8), sharey=True)
    for ax, (name, h) in zip(np.atleast_1d(axes), curves.items()):
        e = [r["epoch"] for r in h]
        tr, es = [r["train_logloss"] for r in h], [r["es_logloss"] for r in h]
        ax.plot(e, tr, color=SERIES[0], lw=2)
        ax.plot(e, es, color=SERIES[1], lw=2)
        b = int(np.argmin(es))
        ax.scatter([e[b]], [es[b]], s=40, color=SERIES[1], edgecolor="white", lw=1.5, zorder=3)
        ax.annotate("training", (e[-1], tr[-1]), xytext=(-2, -9), textcoords="offset points",
                    ha="right", fontsize=7.5, color=TEXT_2)
        ax.annotate("inner holdout", (e[-1], es[-1]), xytext=(-2, 5), textcoords="offset points",
                    ha="right", fontsize=7.5, color=TEXT_2)
        ax.set_title(name, loc="left")
        ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel("log loss")
    fig.tight_layout()
    return fig


def embedding_map(vecs: np.ndarray, names: list[str], title: str, label=None):
    """2-D PCA of one embedding table; points labelled (all, or where ``label``)."""
    x = vecs - vecs.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    expl = s[:2] ** 2 / (s ** 2).sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(xy[:, 0], xy[:, 1], s=30, color=SERIES[0], edgecolor="white", lw=1)
    label = np.ones(len(names), bool) if label is None else np.asarray(label)
    for (a, b), n, show in zip(xy, names, label):
        if not show:
            continue
        ax.annotate(n, (a, b), xytext=(3, 3), textcoords="offset points", fontsize=7, color=TEXT)
    ax.set_xlabel(f"PC1 ({expl[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({expl[1]:.0%})")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def reliability(curves: dict, n_bins: int = 15):
    """Observed delay rate vs mean predicted risk per bin, one line per model."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--")
    for k, (name, (y, p)) in enumerate(curves.items()):
        t = reliability_table(y, p, n_bins)
        t = t[t["n"] >= 50]
        ax.plot(t["mean_pred"], t["frac_pos"], color=SERIES[k], lw=2, marker="o", ms=4)
        ax.annotate(name, (t["mean_pred"].iloc[-1], t["frac_pos"].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=TEXT, va="center")
    ax.set_xlabel("mean predicted risk")
    ax.set_ylabel("observed delay rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability on test (Nov–Dec)", loc="left")
    fig.tight_layout()
    return fig


def pr_curves(curves: dict):
    """Precision-recall on test, one line per model. The first entry (the final
    model) is drawn in the primary series colour; the others are recessive and
    identified by direct labels at recall 0.30, with AP in the label."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    base = None
    labelled = []
    for k, (name, (y, p)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        base = y.mean()
        first = k == 0
        ax.plot(rec, prec, color=SERIES[0] if first else REFERENCE, lw=2.2 if first else 1.0,
                alpha=1 if first else 0.8, zorder=3 if first else 2)
        i = int(np.argmin(np.abs(rec - 0.30)))
        labelled.append((prec[i], f"{name}  AP {ap:.3f}", first))
    # stack labels at the right margin in the order of their precision at recall 0.30
    for j, (py, text, first) in enumerate(sorted(labelled, reverse=True)):
        ax.annotate(text, (1.02, 0.92 - 0.075 * j), xycoords=("axes fraction", "axes fraction"),
                    fontsize=7.5, color=TEXT if first else TEXT_2,
                    fontweight="bold" if first else "normal", va="center")
    ax.axhline(base, color=REFERENCE, lw=1, ls="--")
    ax.annotate(f"base rate {base:.3f}", (0.98, base), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7, color=TEXT_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–recall on test (Nov–Dec)", loc="left")
    fig.tight_layout()
    return fig


def confusion(pol: pd.DataFrame, model: str):
    """2x2 confusion matrices of the final model at its two operating points.
    Cells are labelled with count and share of all test flights; shading is
    one sequential hue by count."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", SERIES[0]])
    fig, axes = plt.subplots(1, len(pol), figsize=(3.2 * len(pol), 3.0))
    for ax, (_, r) in zip(np.atleast_1d(axes), pol.iterrows()):
        m = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        n = m.sum()
        ax.imshow(m, cmap=cmap, vmin=0, vmax=n)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{int(v):,}\n{v / n:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * n else TEXT)
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["on time", "late"])
        ax.set_xlabel("prediction")
        ax.set_ylabel("actual")
        name = "daily top 10%" if r.policy.startswith("daily") else "frozen threshold"
        ax.set_title(f"{name}\nprecision {r.precision:.3f}, recall {r.recall:.3f}", loc="left", fontsize=8.5)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices on test: {model}", x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout()
    return fig


def reschedule_shifts(moves: pd.DataFrame):
    """Accepted moves by shift, labelled with their count and mean predicted gain."""
    g = moves.groupby("shift").agg(n=("net_gain", "size"), gain=("net_gain", "mean"))
    g = g.reindex(sorted(set(g.index) | {-2, -1, 1, 2}), fill_value=0)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar([f"{s:+d} h" for s in g.index], g["n"], color=SERIES[0], width=0.6)
    for i, (n, gain) in enumerate(zip(g["n"], g["gain"])):
        ax.annotate(f"{n}\n(+{gain:.3f}/move)" if n else "0", (i, n), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color=TEXT)
    ax.set_ylabel("accepted moves (Nov–Dec)")
    ax.set_title("What-if moves by shift", loc="left")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.25)
    fig.tight_layout()
    return fig
