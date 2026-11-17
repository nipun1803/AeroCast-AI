"""Fit Phase 1 XGBoost + learned embeddings, in a process without torch.

The torch and XGBoost OpenMP runtimes crash in one process on macOS, so the
secondary hybrid (aerocast_dl.hybrid) writes its jobs to an .npz file and runs
this script. Each job j has ``emb_j`` (embedding features for every modelling
row), ``fit_j`` and ``pred_j`` (modelling-row positions). Fitting uses the
Phase 1 XGBoost configuration at its fixed tree count (as deployed).

Optionally, ``extra_index`` holds index labels in ``P.raw`` of flights that are
not modelling rows (cancelled or diverted ones, which still compete for daily
flag budgets); each job then has ``extra_emb_j`` for them and the output adds
``extra_p_j``. Their features are the Phase 1 features built on the full
schedule, exactly as for the modelling rows. A job with zero embedding
columns is the plain Phase 1 XGBoost.

Usage: python scripts/xgb_worker.py jobs.npz out.npz
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "phase1" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aerocast import features  # noqa: E402
from aerocast import models as p1_models  # noqa: E402
from aerocast import pipeline as p1_pipeline  # noqa: E402


def _with_emb(X: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    cols = [f"emb_{k}" for k in range(emb.shape[1])]
    return pd.concat([X, pd.DataFrame(emb, index=X.index, columns=cols)], axis=1)


def main(jobs_path: str, out_path: str) -> None:
    assert "torch" not in sys.modules
    jobs = np.load(jobs_path)
    params = json.loads(str(jobs["params"]))
    P = p1_pipeline.prepare()
    y = np.empty(len(P.meta), dtype=int)
    for s, i in P.idx.items():
        y[i] = P.labels.y[s]
    X = P.X[P.feat_cols]
    Xe = None
    if "extra_index" in jobs.files:
        X_all, _ = features.build_features(P.raw)
        Xe = X_all.loc[jobs["extra_index"], P.feat_cols].reset_index(drop=True)
    out = {}
    for j in range(int(jobs["n_jobs"])):
        Xj = _with_emb(X, jobs[f"emb_{j}"])
        fit, pred = jobs[f"fit_{j}"], jobs[f"pred_{j}"]
        model = p1_models.fit_xgb_final(Xj.iloc[fit], y[fit], params)
        out[f"p_{j}"] = model.predict_proba(Xj.iloc[pred])[:, 1]
        if Xe is not None:
            out[f"extra_p_{j}"] = model.predict_proba(_with_emb(Xe, jobs[f"extra_emb_{j}"]))[:, 1]
    np.savez(out_path, **out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
