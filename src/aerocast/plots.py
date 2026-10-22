"""Report figures (static, light background, print-friendly)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import config  # noqa: E402
from .evaluate import reliability_table  # noqa: E402

# Validated categorical slots (first three pass all-pairs CVD checks).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#8a8984"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    fig.savefig(config.FIGURES_DIR / f"{name}.pdf")  # vector for LaTeX
    plt.close(fig)


def pr_roc_curves(y, preds: dict[str, np.ndarray], title_suffix=""):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for (name, p), c in zip(preds.items(), SERIES):
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(rec, prec, color=c, label=name)
        fpr, tpr, _ = roc_curve(y, p)
        a2.plot(fpr, tpr, color=c, label=name)
    base = float(np.mean(y))
    a1.axhline(base, color=REFERENCE, lw=1, ls="--", label=f"base rate {base:.2f}")
    a1.axhline(config.TARGET_PRECISION, color=REFERENCE, lw=1, ls=":",
               label=f"target precision {config.TARGET_PRECISION:.2f}")
    a1.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall{title_suffix}",
           xlim=(0, 1), ylim=(0, 1))
    a2.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--")
    a2.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC{title_suffix}", xlim=(0, 1), ylim=(0, 1))
    a1.legend(loc="upper right")
    a2.legend(loc="lower right")
    fig.tight_layout()
    return fig


def reliability_diagram(y, preds: dict[str, np.ndarray], title="Reliability (test)"):
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(4.4, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="perfect")
    for (name, p), c in zip(preds.items(), SERIES):
        t = reliability_table(y, p)
        t = t[t["n"] >= 30]
        ax.plot(t["mean_pred"], t["frac_pos"], color=c, marker="o", ms=4, label=name)
        axh.hist(p, bins=30, range=(0, 1), histtype="step", color=c, lw=1.5)
    ax.set(ylabel="Observed delay rate", title=title, xlim=(0, 1), ylim=(0, 1))
    axh.set(xlabel="Predicted probability", ylabel="Count")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def monthly_drift(drift, title="Predicted vs observed delay rate by month"):
    """Mean predicted risk vs realised rate, with split regions shaded."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axvspan(8.5, 10.5, color=GRID, alpha=0.6, lw=0)
    ax.axvspan(10.5, 12.5, color=GRID, alpha=1.0, lw=0)
    ax.plot(drift["month"], drift["observed"], color=SERIES[1], marker="o", ms=5, label="observed")
    ax.plot(drift["month"], drift["predicted"], color=SERIES[0], marker="o", ms=5,
            label="mean predicted")
    ymax = max(drift["observed"].max(), drift["predicted"].max()) * 1.15
    for x, txt in [(4.5, "train (Jan-Aug)"), (9.5, "calib"), (11.5, "test")]:
        ax.text(x, ymax * 0.97, txt, ha="center", va="top", fontsize=8, color=TEXT_2)
    ax.set(xticks=range(1, 13), xlabel="Month (2013)", ylabel="Delay rate", title=title,
           ylim=(0, ymax), xlim=(0.5, 12.5))
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def hbar(values, labels, title, xlabel, color=SERIES[0]):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(values) + 0.8))
    ax.barh(np.array(labels)[order], np.array(values)[order], color=color, height=0.65)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EDA & feature-engineering figures (notebooks 01-02)
# ---------------------------------------------------------------------------
def _shade_splits(ax):
    """Shade the calib (Sep-Oct) and test (Nov-Dec) months on a month axis."""
    ax.axvspan(8.5, 10.5, color=GRID, alpha=0.6, lw=0)
    ax.axvspan(10.5, 12.5, color=GRID, alpha=1.0, lw=0)


def delay_histogram(arr_delay, threshold: int = config.DELAY_THRESHOLD_MIN):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(arr_delay.clip(-80, 240).dropna(), bins=160, color=SERIES[0])
    ax.axvline(threshold, color=SERIES[1], lw=2, label=f"{threshold} min threshold")
    ax.set(xlabel="Arrival delay (min, clipped to [-80, 240])", ylabel="Flights",
           title="Arrival delay distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def rate_by_time(by_month, by_hour, by_dow):
    """Three panels: delay rate by month (split-shaded), hour, day of week."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    _shade_splits(axes[0])
    axes[0].plot(by_month.index, by_month.values, color=SERIES[0], marker="o", ms=5)
    axes[0].set(title="By month (shaded: calib, test)", xticks=range(1, 13),
                ylabel="Delay rate", xlim=(0.5, 12.5))
    axes[1].plot(by_hour.index, by_hour.values, color=SERIES[0], marker="o", ms=5)
    axes[1].set(title="By scheduled departure hour", xlabel="Hour")
    axes[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow.values,
                color=SERIES[0], width=0.65)
    axes[2].set(title="By day of week")
    axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def rate_by_group(by_carrier, by_origin):
    """Carrier bars annotated with flight counts, plus origin bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    c = by_carrier.sort_values("rate")
    axes[0].barh(c.index, c["rate"], color=SERIES[0], height=0.65)
    for i, (r, n) in enumerate(zip(c["rate"], c["n"])):
        axes[0].text(r + 0.005, i, f"n={n:,}", va="center", fontsize=7, color=TEXT_2)
    axes[0].set(title="Delay rate by carrier", xlabel="Delay rate",
                xlim=(0, c["rate"].max() * 1.3))
    axes[0].grid(axis="y", visible=False)
    axes[1].bar(by_origin.index, by_origin["rate"], color=SERIES[0], width=0.6)
    axes[1].set(title="By origin")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def daily_rate_and_acf(daily, acf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [2.4, 1]})
    axes[0].plot(daily.index, daily.values, color=SERIES[0], lw=1, label="daily")
    axes[0].plot(daily.rolling(28).mean(), color=SERIES[1], lw=2, label="28-day trailing mean")
    axes[0].set(title="Daily network delay rate", ylabel="Delay rate")
    axes[0].legend()
    axes[1].bar(acf.index, acf.values, color=SERIES[0], width=0.65)
    axes[1].set(title="Autocorrelation of daily rate", xlabel="Lag (days)")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def congestion_curve(curve):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(len(curve)), curve.values, color=SERIES[0], marker="o", ms=5)
    ax.set(xticks=range(len(curve)), xticklabels=[str(i) for i in curve.index],
           title="Delay rate vs scheduled departures in the same origin-hour",
           xlabel="Scheduled departures (same airport, same hour), octiles",
           ylabel="Delay rate")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


def feature_histograms(X, cols, title: str, ncols: int = 4, bins: int = 40):
    """Small multiples: one histogram per feature (NaNs dropped)."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.1 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        ax.hist(X[c].dropna(), bins=bins, color=SERIES[0])
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training figures (notebook 03)
# ---------------------------------------------------------------------------
def cv_logreg_curve(cv):
    """Validation PR-AUC per fold (thin) and mean (thick) across the C grid."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    wide = cv.pivot(index="C", columns="fold", values="val_pr_auc")
    for fold in wide.columns:
        ax.plot(wide.index, wide[fold], color=REFERENCE, lw=1, marker="o", ms=3)
    ax.plot(wide.index, wide.mean(axis=1), color=SERIES[0], marker="o", ms=6,
            label="mean of 4 folds")
    ax.set(xscale="log", xlabel="C (inverse regularisation)", ylabel="Validation PR-AUC",
           title="LogReg rolling-origin CV (grey: individual folds)")
    ax.legend()
    fig.tight_layout()
    return fig


def tree_count_curve(curve, cv_trees: int):
    """PR-AUC on Sep and Oct vs boosting rounds, with the CV-chosen count."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(curve["n_trees"], curve["pr_auc_m9"], color=SERIES[0], label="Sep")
    ax.plot(curve["n_trees"], curve["pr_auc_m10"], color=SERIES[1], label="Oct")
    ax.axvline(cv_trees, color=REFERENCE, ls="--", lw=1, label=f"CV choice ({cv_trees})")
    for col, c in [("pr_auc_m9", SERIES[0]), ("pr_auc_m10", SERIES[1])]:
        best = curve.loc[curve[col].idxmax()]
        ax.plot(best["n_trees"], best[col], marker="o", ms=8, color=c)
    ax.set(xlabel="Boosting rounds", ylabel="PR-AUC",
           title="Where does a single month say to stop? (dots = per-month optimum)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Review additions
# ---------------------------------------------------------------------------
def ablation_bars(abl):
    """Rolling-CV PR-AUC per cumulative feature set, LogReg vs XGBoost."""
    sets = list(dict.fromkeys(abl["features"]))
    y = np.arange(len(sets))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(sets) + 1.2))
    for k, (model, label) in enumerate([("logreg", "LogReg"), ("xgboost", "XGBoost")]):
        sub = abl[abl.model == model].set_index("features").loc[sets]
        ax.barh(y + (k - 0.5) * 0.38, sub["cv_pr_auc_mean"], height=0.36, color=SERIES[k],
                xerr=sub["cv_pr_auc_std"] / 2, error_kw={"ecolor": REFERENCE, "lw": 1},
                label=label)
    ax.set(yticks=y, yticklabels=sets, xlabel="Rolling-CV PR-AUC (mean, bar = ±½ sd)",
           title="Feature-family ablation (CV, training window)", xlim=(0, 0.6))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", framealpha=1, frameon=True, edgecolor=GRID)
    fig.tight_layout()
    return fig


def corr_heatmap(corr, title: str):
    """Diverging heatmap of a correlation matrix (blue -1 .. grey 0 .. orange +1)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [SERIES[0], "#f1f0ec", SERIES[1]])
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.5, 0.42 * n + 1.5))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set(xticks=range(n), yticks=range(n), title=title)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            if abs(v) >= 0.5 and i != j:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
    fig.tight_layout()
    return fig


def congestion_within_hour(tbl):
    """Delay rate by departure hour for low / mid / high bank-size terciles
    (terciles computed within each hour, so hour is held fixed)."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, col in enumerate(tbl.columns):
        ax.plot(tbl.index, tbl[col], color=SERIES[k], marker="o", ms=4, label=str(col))
    ax.set(xlabel="Scheduled departure hour", ylabel="Delay rate",
           title="Congestion vs delay within each departure hour")
    ax.legend(title="bank size (within hour)")
    fig.tight_layout()
    return fig
