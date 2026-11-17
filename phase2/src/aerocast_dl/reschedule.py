"""Offline what-if rescheduling (Step 8), with explicit assumptions.

At 20:00 on D-1 the desk sees the day-D risk of every flight. For the daily
top-10% riskiest flights (the primary operating policy), it tries moving the
scheduled departure by a whole number of hours on the same day and keeps a
move only if the model's *predicted* expected number of delayed flights at
that origin on D falls by at least ``delta``.

Assumptions (stated in the report):
  A1  The counterfactual is predictive, not causal: the model's association
      between departure hour / congestion and delay is taken as the effect of
      moving a flight. Weather and ATC programmes are unknown at 20:00 D-1.
  A2  Same day, whole hours, minute unchanged, block time unchanged (the
      scheduled arrival moves with the departure). Carrier, aircraft, route
      and flight number unchanged; aircraft rotations, crews and passenger
      connections are not modelled.
  A3  Capacity: an origin-hour may not exceed the busiest hour that origin had
      in the training timetable; new hours stay within that origin's observed
      range of departure hours.
  A4  Every move has a cost; the rule's parameters are chosen on Sep-Oct by
      net predicted benefit = delays avoided - MOVE_COST x moves.

Leakage: a move changes only schedule-derived inputs (time-of-day, hourly
congestion counts at the origin) and the moved flight's origin-hour history,
which is read from a table built with the Phase 1 decision clock. Realised
day-D outcomes are never used (tests/test_reschedule.py).

Cohort: the desk works on day D's whole timetable (``Schedule``), cancelled
and diverted flights included. The modelling rows (``P.meta``) drop the
flights without an arrival outcome, which is only known after day D; taking
candidates, daily budgets or origin-day risk sums from them would let the
future outcome choose who is a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerocast import config as p1
from aerocast import data, evaluate, features
from aerocast.pipeline import Prepared

from . import calibrate, sequences

SHIFTS = (-2, -1, 1, 2)
GRID = {"window": [1, 2], "delta": [0.01, 0.02, 0.05]}
MOVE_COST = 0.02                    # A4, in expected delayed flights per move
COST_SENSITIVITY = [0.0, 0.02, 0.05]
L = p1.LONG_WINDOW
OH_RATE, OH_LOGN = f"hist_rate_origin_hour_{L}d", f"hist_logn_origin_hour_{L}d"


# ---------------------------------------------------------------------------
# Context: timetable counts, origin-hour history, network prior
# ---------------------------------------------------------------------------
@dataclass
class Context:
    hour_counts: pd.Series          # (origin, date, hour) -> scheduled departures (incl. cancelled)
    oh_n: pd.DataFrame              # dates x (origin, hour): known arrivals in the long window
    oh_s: pd.DataFrame              # same, delayed
    prior: pd.Series                # date -> network prior (trailing 28 days, past-only)
    cap: dict                       # origin -> max departures in one hour (training timetable)
    hour_range: dict                # origin -> (min, max) departure hour (training timetable)


def build_context(raw: pd.DataFrame) -> Context:
    """Everything a move needs that is not a model input already. ``raw`` is
    the full schedule with targets (cancelled flights included)."""
    counts = raw.groupby(["origin", "date", "hour"]).size()
    dates = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")
    ents = pd.MultiIndex.from_product([sorted(raw["origin"].unique()), range(24)],
                                      names=["origin", "hour"])
    tabs = {}
    for stat in ("count", "sum"):
        w = sequences._daily_wide(raw, ("origin", "hour"), "is_delayed", stat, dates, ents,
                                  p1.DECISION_HOUR)
        tabs[stat] = pd.DataFrame(w, index=dates, columns=ents).rolling(L, min_periods=1).sum()
    prior = pd.Series(features.network_prior(raw), index=raw.index).groupby(raw["date"]).first()
    tr = raw[raw["date"] <= p1.SPLITS["train"][1]]
    cap = tr.groupby(["origin", "date", "hour"]).size().groupby(level="origin").max().to_dict()
    rng = tr.groupby("origin")["hour"].agg(["min", "max"])
    return Context(counts, tabs["count"], tabs["sum"], prior, cap,
                   {o: (int(r["min"]), int(r["max"])) for o, r in rng.iterrows()})


@dataclass
class Schedule:
    """Every scheduled flight after the warm-up, cancelled and diverted
    included, with its Phase 1 features: what the desk sees at 20:00 on D-1.

    ``model_pos`` is the flight's position in ``P.meta`` (-1 for a flight
    without an arrival outcome). It is used only to check that the scorer
    reproduces the saved predictions, never to choose flights."""

    meta: pd.DataFrame
    X: pd.DataFrame
    split: np.ndarray
    model_pos: np.ndarray
    raw_index: np.ndarray | None = None     # index labels in ``raw`` (for the XGBoost worker)


def build_schedule(raw: pd.DataFrame, P: Prepared | None = None) -> Schedule:
    """Full-schedule rows of ``raw`` (with targets, as ``P.raw``). Rows are
    selected by date only. With ``P``, check that the flights with an outcome
    are exactly the modelling rows, with the same features."""
    X_all, _ = features.build_features(raw)
    keep = (raw["date"] > p1.WARMUP_END).to_numpy()
    raw_index = raw.index.to_numpy()[keep]
    meta = raw.loc[keep].reset_index(drop=True)
    X = X_all.loc[keep].reset_index(drop=True)
    has_y = meta["is_delayed"].notna().to_numpy()
    model_pos = np.full(len(meta), -1, dtype=np.int64)
    model_pos[has_y] = np.arange(int(has_y.sum()))
    if P is not None:
        if int(has_y.sum()) != len(P.meta):
            raise AssertionError("schedule flights with an outcome differ from the modelling rows")
        cols = ["date", "carrier", "flight", "origin", "dest", "hour", "minute"]
        pd.testing.assert_frame_equal(meta.loc[has_y, cols].reset_index(drop=True), P.meta[cols])
        pd.testing.assert_frame_equal(X.loc[has_y].reset_index(drop=True), P.X)
    split = data.assign_split(meta).to_numpy()
    return Schedule(meta, X, split, model_pos, raw_index)


def oh_history(ctx: Context, date, origin: str, hour: int, m: float = p1.SMOOTHING_M):
    """(smoothed rate, log count) of the origin-hour history for day ``date``,
    exactly as ``features.history_features`` computes it."""
    n = float(ctx.oh_n.at[pd.Timestamp(date), (origin, hour)])
    s = float(ctx.oh_s.at[pd.Timestamp(date), (origin, hour)])
    return features.smoothed_rate(s, n, ctx.prior[pd.Timestamp(date)], m), float(np.log1p(n))


# ---------------------------------------------------------------------------
# Counterfactual feature rows
# ---------------------------------------------------------------------------
def _set_congestion(X: pd.DataFrame, hours: np.ndarray, counts: np.ndarray) -> None:
    pad = np.r_[0, counts, 0]                                   # hours -1 and 24 are empty
    X["sched_origin_hour"] = counts[hours]
    X["sched_origin_3h"] = pad[hours] + pad[hours + 1] + pad[hours + 2]


def _move_row(X: pd.DataFrame, i: int, new_hour: int, shift: int, minute: int, arr_min: int,
              ctx: Context, date, origin: str) -> None:
    dep = new_hour * 60 + minute
    arr = (arr_min + 60 * shift) % 1440
    X.iloc[i, X.columns.get_loc("dep_tod_sin")] = np.sin(2 * np.pi * dep / 1440)
    X.iloc[i, X.columns.get_loc("dep_tod_cos")] = np.cos(2 * np.pi * dep / 1440)
    X.iloc[i, X.columns.get_loc("arr_tod_sin")] = np.sin(2 * np.pi * arr / 1440)
    X.iloc[i, X.columns.get_loc("arr_tod_cos")] = np.cos(2 * np.pi * arr / 1440)
    X.iloc[i, X.columns.get_loc("dep_hour")] = dep / 60
    rate, logn = oh_history(ctx, date, origin, new_hour)
    X.iloc[i, X.columns.get_loc(OH_RATE)] = rate
    X.iloc[i, X.columns.get_loc(OH_LOGN)] = logn


# ---------------------------------------------------------------------------
# Greedy plan for one origin-day
# ---------------------------------------------------------------------------
def plan_origin_day(S: Schedule, ctx: Context, score, calib_map, origin: str, date,
                    candidates: np.ndarray, window: int, delta: float,
                    shifts=SHIFTS) -> tuple[list[dict], float]:
    """Moves for the ``candidates`` (schedule positions, riskiest first) of one
    origin-day; the predicted risk is summed over every scheduled flight of
    that origin-day. ``score(X, meta) -> logits``; ``calib_map`` maps scores to
    calibrated risk for this date. Returns (moves, total predicted gain)."""
    meta = S.meta
    pos = np.flatnonzero((meta["origin"].to_numpy() == origin)
                         & (meta["date"].to_numpy() == np.datetime64(pd.Timestamp(date))))
    X = S.X.iloc[pos].copy()
    rows_meta = meta.iloc[pos]
    hours = meta["hour"].to_numpy()[pos].astype(int)
    minute = meta["minute"].to_numpy()[pos].astype(int)
    sat = meta["sched_arr_time"].to_numpy()[pos].astype(int)
    arr_min = (sat // 100) * 60 + sat % 100
    counts = np.zeros(24, dtype=int)
    day_counts = ctx.hour_counts.loc[(origin, pd.Timestamp(date))]
    counts[day_counts.index.to_numpy(dtype=int)] = day_counts.to_numpy()
    # the plan covers the whole timetable the congestion counts come from
    if not np.array_equal(np.bincount(hours, minlength=24), counts):
        raise AssertionError(f"{origin} {pd.Timestamp(date).date()}: plan rows are not the full schedule")
    lo, hi = ctx.hour_range[origin]

    def risk(Xs, n_opts):
        z = score(Xs, pd.concat([rows_meta] * n_opts))
        return calib_map.predict(1 / (1 + np.exp(-z))).reshape(n_opts, len(pos))

    p_cur = risk(X, 1)[0]
    moves, gain_total = [], 0.0
    where = {p: k for k, p in enumerate(pos)}
    for c in candidates:
        i = where[c]
        opts = []
        for d in shifts:
            h2 = hours[i] + d
            if abs(d) > window or not (lo <= h2 <= hi) or counts[h2] + 1 > ctx.cap[origin]:
                continue
            cnt = counts.copy()
            cnt[hours[i]] -= 1
            cnt[h2] += 1
            hr = hours.copy()
            hr[i] = h2
            Xn = X.copy()
            _set_congestion(Xn, hr, cnt)
            _move_row(Xn, i, h2, d, minute[i], arr_min[i], ctx, date, origin)
            opts.append((d, h2, cnt, hr, Xn))
        if not opts:
            continue
        p_new = risk(pd.concat([o[4] for o in opts]), len(opts))
        gains = p_cur.sum() - p_new.sum(axis=1)
        b = int(np.argmax(gains))
        if gains[b] >= delta:
            d, h2, cnt, hr, Xn = opts[b]
            moves.append({"pos": int(c), "date": pd.Timestamp(date), "origin": origin,
                          "carrier": meta["carrier"].iloc[c], "flight": int(meta["flight"].iloc[c]),
                          "old_hour": int(hours[i]), "new_hour": int(h2), "shift": int(d),
                          "risk_before": float(p_cur[i]), "risk_after": float(p_new[b, i]),
                          "net_gain": float(gains[b]),
                          "spillover": float(gains[b] - (p_cur[i] - p_new[b, i]))})
            X, counts, hours, p_cur = Xn, cnt, hr, p_new[b]
            gain_total += float(gains[b])
    return moves, gain_total


def candidates_by_day(dates, risk, pos, share: float = p1.DAILY_FLAG_SHARE) -> np.ndarray:
    """Positions of the daily top-``share`` flights (the primary policy)."""
    flagged = evaluate.flag_daily_top_share(np.asarray(dates), np.asarray(risk), share)
    return np.asarray(pos)[flagged]


def run(S: Schedule, ctx: Context, score, maps: dict, pos: np.ndarray, risk: np.ndarray,
        window: int, delta: float, shifts=SHIFTS) -> pd.DataFrame:
    """Apply the rule to every origin-day of the schedule rows ``pos`` (one
    split, sorted)."""
    dates = S.meta["date"].to_numpy()[pos]
    cand = candidates_by_day(dates, risk, pos)
    order = np.argsort(-risk[np.searchsorted(pos, cand)], kind="stable")
    cand = cand[order]
    meta = S.meta
    out = []
    for (o, d), grp in pd.Series(cand).groupby([meta["origin"].to_numpy()[cand],
                                               meta["date"].to_numpy()[cand]], sort=True):
        mv, _ = plan_origin_day(S, ctx, score, maps[pd.Timestamp(d)], o, d, grp.to_numpy(),
                                window, delta, shifts)
        out.extend(mv)
    return pd.DataFrame(out)


def summarise(moves: pd.DataFrame, n_days: int, n_candidates: int, cost: float) -> dict:
    n = len(moves)
    g = float(moves["net_gain"].sum()) if n else 0.0
    return {"moves": n, "moves_per_day": n / n_days, "share_of_candidates_moved": n / n_candidates,
            "predicted_delays_avoided": g, "predicted_delays_avoided_per_day": g / n_days,
            "avoided_per_move": g / n if n else 0.0,
            "net_benefit": g - cost * n,
            "share_moved_earlier": float((moves["shift"] < 0).mean()) if n else float("nan"),
            "spillover_share": float(moves["spillover"].sum() / g) if n and g else float("nan")}


# ---------------------------------------------------------------------------
# Scorers: the deployed model applied to (possibly modified) feature rows
# ---------------------------------------------------------------------------
def make_scorer(P: Prepared, model: str, cache):
    """``score(X, meta) -> logits`` for the deployed version of ``model``
    (fitted on the training rows available at 2013-09-01). ``meta`` holds the
    rows' date, origin, carrier and destination, which locate the GRU history
    windows (a move does not change them: they are per origin / carrier /
    route and day). Rows need not be modelling rows."""
    import torch

    from aerocast import models as p1_models

    from . import config, pipeline, tensors, train

    rows = tensors.final_fit_rows(P)
    if model == "logreg":
        y = tensors.row_info(P)["y"].to_numpy()
        lr = p1_models.make_logreg(P.cat_cols, P.num_cols, config.LOGREG_C)
        lr.fit(P.X.iloc[rows][P.feat_cols], y[rows])
        return lambda X, meta: lr.decision_function(X[P.feat_cols])
    spec = pipeline.load_spec(model)
    if spec.ids:
        raise ValueError("rescheduling assumes no ID inputs (origin_hour would move)")
    inp = cache.get("final", rows, config.FINAL_FIT_DATE, spec)
    nets = []
    for s in config.SEEDS:
        net = train.make_model(spec, inp)
        net.load_state_dict(torch.load(config.CACHE_DIR / f"final_{model}_seed{s}.pt"))
        nets.append(net.eval())
    return network_scorer(P, nets, inp, cache.seqs)


def network_scorer(P: Prepared, nets, inp, seqs):
    """``score(X, meta) -> mean logit`` of the seed ensemble ``nets`` for any
    feature rows ``X`` (Phase 1 columns), re-encoded with the transforms of
    ``inp``; ``meta`` locates the GRU windows in ``seqs``."""
    import torch

    for net in nets:
        net.eval()                          # dropout off

    def score(X, meta):
        E = inp.encoder.transform(X[P.cat_cols].astype(object), X[inp.num_cols])
        b = {"cat": torch.from_numpy(E.cat), "num": torch.from_numpy(E.num)}
        if inp.seq:
            b["seq"] = {}
            for n, w in inp.seq.items():
                day, ent = seqs[n].lookup(meta)
                if (day < 0).any() or (ent < 0).any():
                    raise ValueError(f"{n}: rows outside the sequence table")
                b["seq"][n] = w[torch.as_tensor(day), torch.as_tensor(ent)]
        if inp.wide is not None:
            Z = inp.wide_pre.transform(X[P.cat_cols + inp.num_cols])
            Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
            b["wide"] = torch.from_numpy(Z.astype(np.float32))
        with torch.no_grad():
            return np.mean([n(**b).numpy() for n in nets], axis=0).astype(np.float64)
    return score


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _setup(P, cache, df, model, calibrator, split):
    """Schedule, scorer, per-day calibration maps, schedule rows and
    calibrated risk of ``split`` (every scheduled flight of its days)."""
    from . import evaluation
    S = build_schedule(P.raw, P)
    score = make_scorer(P, model, cache)
    pos = np.flatnonzero(S.split == split)
    dates = S.meta["date"].to_numpy()[pos]
    pool = evaluation._pool(df, model)
    maps = calibrate.day_maps(calibrator, pool, dates)
    p = 1 / (1 + np.exp(-score(S.X.iloc[pos], S.meta.iloc[pos])))
    # on the flights with an outcome the scorer reproduces the saved predictions
    # (same model, same rows); the others are scored the same way
    mp = S.model_pos[pos]
    has = mp >= 0
    if not np.array_equal(mp[has], P.idx[split]):
        raise AssertionError(f"{split}: schedule and modelling rows are not aligned")
    raw_saved = df.loc[df["split"].eq(split).to_numpy(), model].to_numpy(float)
    np.testing.assert_allclose(p[has], raw_saved, rtol=1e-5, atol=1e-6)
    risk = np.empty(len(pos))
    for d, ix in pd.Series(np.arange(len(pos))).groupby(dates):
        risk[ix.to_numpy()] = maps[pd.Timestamp(d)].predict(p[ix.to_numpy()])
    return S, score, maps, pos, risk


def tune(P, cache, df, model, calibrator, ctx, log) -> tuple[dict, pd.DataFrame]:
    """Choose (window, delta) on Sep-Oct by net predicted benefit (A4)."""
    if (df["split"] == "test").any():
        raise ValueError("rescheduling parameters are chosen without test rows")
    S, score, maps, pos, risk = _setup(P, cache, df, model, calibrator, "calib")
    n_days = S.meta["date"].iloc[pos].nunique()
    n_cand = len(candidates_by_day(S.meta["date"].to_numpy()[pos], risk, pos))
    rows = []
    for w in GRID["window"]:
        for d in GRID["delta"]:
            mv = run(S, ctx, score, maps, pos, risk, w, d)
            rows.append({"window": w, "delta": d, **summarise(mv, n_days, n_cand, MOVE_COST)})
            log(f"  Sep-Oct window={w} delta={d}: {rows[-1]['moves']} moves, net benefit "
                f"{rows[-1]['net_benefit']:.1f}")
    tbl = pd.DataFrame(rows)
    best = tbl.loc[tbl["net_benefit"].idxmax()]
    return {"window": int(best["window"]), "delta": float(best["delta"])}, tbl


def apply_test(P, cache, df, model, calibrator, ctx, params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen rule on Nov-Dec, with cost sensitivity and a naive comparison
    (move every candidate one hour earlier when capacity allows)."""
    S, score, maps, pos, risk = _setup(P, cache, df, model, calibrator, "test")
    n_days = S.meta["date"].iloc[pos].nunique()
    n_cand = len(candidates_by_day(S.meta["date"].to_numpy()[pos], risk, pos))
    mv = run(S, ctx, score, maps, pos, risk, params["window"], params["delta"])
    rows = [{"rule": "frozen (Sep-Oct)", "cost": c, **params, **summarise(mv, n_days, n_cand, c)}
            for c in COST_SENSITIVITY]
    naive = _naive_earlier(S, ctx, score, maps, pos, risk)
    rows += [{"rule": "naive: every candidate 1 h earlier", "cost": c, "window": 1, "delta": np.nan,
              **summarise(naive, n_days, n_cand, c)} for c in COST_SENSITIVITY]
    return mv, pd.DataFrame(rows)


def _naive_earlier(S, ctx, score, maps, pos, risk) -> pd.DataFrame:
    return run(S, ctx, score, maps, pos, risk, window=1, delta=-np.inf, shifts=(-1,))
