"""Fit Phase 1 XGBoost + learned embeddings, in a process without torch.

The torch and XGBoost OpenMP runtimes crash in one process on macOS, so the
secondary hybrid (aerocast_dl.hybrid) writes its jobs to an .npz file and runs
this script. Each job j has ``emb_j`` (embedding features for every modelling
row), ``fit_j`` and ``pred_j`` (modelling-row positions). Fitting uses the
Phase 1 XGBoost configuration at its fixed tree count (as deployed).

Usage: python scripts/xgb_worker.py jobs.npz out.npz
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "phase1" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aerocast import models as p1_models  # noqa: E402
from aerocast import pipeline as p1_pipeline  # noqa: E402


def main(jobs_path: str, out_path: str) -> None:
    assert "torch" not in sys.modules
    jobs = np.load(jobs_path)
    params = json.loads(str(jobs["params"]))
    P = p1_pipeline.prepare()
    y = np.empty(len(P.meta), dtype=int)
    for s, i in P.idx.items():
        y[i] = P.labels.y[s]
    X = P.X[P.feat_cols]
    out = {}
    for j in range(int(jobs["n_jobs"])):
        emb = pd.DataFrame(jobs[f"emb_{j}"], index=X.index,
                           columns=[f"emb_{k}" for k in range(jobs[f"emb_{j}"].shape[1])])
        Xj = pd.concat([X, emb], axis=1)
        fit, pred = jobs[f"fit_{j}"], jobs[f"pred_{j}"]
        model = p1_models.fit_xgb_final(Xj.iloc[fit], y[fit], params)
        out[f"p_{j}"] = model.predict_proba(Xj.iloc[pred])[:, 1]
    np.savez(out_path, **out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
