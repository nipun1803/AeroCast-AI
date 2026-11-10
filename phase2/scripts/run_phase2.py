"""Phase 2 end-to-end pipeline (single source of truth for reported numbers).

Stages are added step by step (docs/development_protocol.md):
  data       Step 2: network inputs from the Phase 1 features, fitting clock,
             cold-start coverage of the category vocabularies
  sequences  Step 4: causal daily history windows for the GRU, checked
             against the Phase 1 history features

Test labels are not read by any stage before the final evaluation stage.

Usage:  python scripts/run_phase2.py
Outputs: results/*.csv|json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Do not rely on the editable-install .pth files (see phase1/docs/reproducibility.md).
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "phase1" / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from aerocast import config as p1  # noqa: E402
from aerocast import features  # noqa: E402
from aerocast import pipeline as p1_pipeline  # noqa: E402

from aerocast_dl import config, sequences, tensors  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_data(P) -> dict:
    rows = tensors.final_fit_rows(P)
    enc = tensors.fit_encoder(P, rows, config.FINAL_FIT_DATE, ids=config.ID_COLS)
    cats, nums = tensors.model_inputs(P, config.ID_COLS)
    E = enc.transform(cats.iloc[:1], nums.iloc[:1])
    summary = {
        "fit_date": config.FINAL_FIT_DATE,
        "train_rows": int(len(P.idx["train"])),
        "fit_rows": int(len(rows)),
        "fit_rows_excluded_by_clock": int(len(P.idx["train"]) - len(rows)),
        "min_cat_freq": config.MIN_CAT_FREQ,
        "numeric_inputs": len(P.num_cols),
        "numeric_inputs_with_missing_indicators": len(E.num_cols),
        "missing_indicator_columns": [c for c in E.num_cols if c.startswith("missingindicator_")],
        "embedding_tables": dict(zip(E.cat_cols, E.cardinalities)),
    }
    tensors.cold_start_table(P).to_csv(config.RESULTS_DIR / "cold_start.csv", index=False)
    (config.RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_sequences(P) -> dict:
    seqs = sequences.default_sequences(P)
    idx = sequences.row_index(P, seqs)
    norm = sequences.fit_sequence_normalizer(P, seqs, tensors.final_fit_rows(P),
                                             config.FINAL_FIT_DATE)
    summary = {"seq_len": sequences.SEQ_LEN, "raw_channels": sequences.CHANNELS,
               "model_channels": sequences.DERIVED, "entity_types": {}}
    for name, keys in sequences.ENTITY_SPECS.items():
        s = seqs[name]
        W = s.windows()
        # exact agreement with the Phase 1 long-window history counts
        ref = features.rolling_past(P.raw, keys, 14, "is_delayed")
        day, ent = s.lookup(P.raw)
        mismatch = int((W[day, ent, -14:, 0].sum(1) != ref["n"].to_numpy()).sum()
                       + (W[day, ent, -14:, 1].sum(1) != ref["s"].to_numpy()).sum())
        if mismatch:
            raise AssertionError(f"{name}: sequences disagree with Phase 1 history ({mismatch})")
        used = W[idx[name][:, 0], idx[name][:, 1]]
        summary["entity_types"][name] = {
            "keys": list(keys), "entities": int(len(s.entities)),
            "window_shape": list(W.shape),
            "modelling_rows_with_empty_window": float((used[..., 0].sum(1) == 0).mean()),
            "mean_known_arrivals_per_window": float(used[..., 0].sum(1).mean()),
            "mismatches_vs_phase1_14d": mismatch,
            "normaliser_mean": np.round(norm.stats_[name][0], 6).tolist(),
            "normaliser_sd": np.round(norm.stats_[name][1], 6).tolist(),
        }
    (config.RESULTS_DIR / "sequence_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    log(f"Phase 1 features and splits (decision clock {p1.DECISION_HOUR}:00 on D-1)")
    P = p1_pipeline.prepare()
    log("stage data: encoder on the final training rows, cold-start coverage")
    summary = stage_data(P)
    log(json.dumps({k: v for k, v in summary.items() if k != "missing_indicator_columns"}))
    log("stage sequences: causal history windows, equality with Phase 1 history")
    seq = stage_sequences(P)
    log(json.dumps({n: {k: v[k] for k in ("entities", "window_shape", "mismatches_vs_phase1_14d")}
                    for n, v in seq["entity_types"].items()}))
    log("done")


if __name__ == "__main__":
    main()
