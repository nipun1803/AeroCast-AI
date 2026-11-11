# 05 · What-if rescheduling (offline)

> Covers notebook [11_whatif_rescheduling](../notebooks/11_whatif_rescheduling.ipynb). Code: `src/aerocast_dl/reschedule.py`. Results: `results/reschedule_tuning.csv`, `reschedule_test.csv`, `reschedule_test_moves.csv`.

## The question

At 20:00 on D−1 the desk sees tomorrow's risk for every flight. Suppose it could move a few of the riskiest departures by an hour or two on the same day. How much **predicted** delay would that avoid?

## How a move is decided

1. **Candidates.** The daily top 10% riskiest flights, ranked by the final model (LogReg with rolling Platt).
2. **Try moves.** For each candidate, riskiest first, try moving it by −2, −1, +1 or +2 hours at the same origin. For each option the model re-scores **all** of that origin's flights for the day, including:
   - the moved flight's new hour features;
   - the congestion it creates in the new hour and relieves in the old one.
3. **Accept.** Keep the best option if the origin-day's **summed predicted risk** falls by at least δ. Because the whole origin-day is summed, spill-over onto other flights counts. The next candidate is then tried against the updated schedule.

## Assumptions

| | Assumption |
|---|---|
| A1 | **Predictive, not causal.** The model's link between hour, congestion and delay is read as the effect of a move. Weather and air-traffic programmes are unknown at 20:00 D−1. |
| A2 | **Scope.** Same day, whole hours, block time unchanged. Carrier, aircraft, route and flight number unchanged. Crews, aircraft rotations and connections are not modelled. |
| A3 | **Capacity.** No origin-hour may get busier than that origin's busiest hour in training, and new hours stay within the origin's observed departure hours. |
| A4 | **Cost.** Each move costs 0.02 expected delayed flights (with 0 and 0.05 shown for sensitivity). |

**No leakage.** The benefit uses predicted risk only, never what actually happened on day D. A test confirms the plan is identical when every day-D outcome is set to missing.

## Tuned on Sep–Oct, then frozen

Net benefit = predicted delays avoided − 0.02 × moves, over Sep–Oct (61 days).

| Shift window | δ | Moves | Delays avoided | Net benefit |
|---|---|---|---|---|
| ±1 h | 0.01 | 2,844 | 72.9 | 16.0 |
| ±1 h | 0.02 | 1,565 | 53.7 | 22.3 |
| ±1 h | 0.05 | 199 | 12.4 | 8.4 |
| ±2 h | 0.01 | 3,561 | 123.7 | 52.5 |
| **±2 h** | **0.02** | **2,612** | **109.8** | **57.6** |
| ±2 h | 0.05 | 740 | 50.3 | 35.5 |

Frozen rule: **shifts up to ±2 hours, δ = 0.02**.

## Result on Nov–Dec (61 days)

| Rule | Moves/day | Predicted delays avoided/day | Per move | Net at cost 0.02 | Net at cost 0.05 |
|---|---|---|---|---|---|
| **Frozen rule** | 48.1 | 1.88 | 0.039 | +55.9 | −32.2 |
| Naive: every candidate 1 h earlier | 88.8 | 0.14 | 0.002 | −99.9 | −262.3 |

- **The rule avoids about 1.9 predicted delays a day.** That is with 48 moves a day, about half of the candidates. It pays at a cost of 0.02 per move but not at 0.05.
- **Most accepted moves are two hours:** 1,438 two hours earlier and 1,166 two hours later, against 331 one-hour moves.
- **"Earlier is better" is not the answer.** Moving everyone an hour earlier achieves almost nothing, because it shifts congestion onto other flights at the origin: its spill-over is negative. The model's flight-by-flight assessment is what makes the rule work.

## How to read this

These are **predicted** effects under A1–A4, not realised savings. The data contain no rescheduled flights, no costs and no aircraft rotations. The analysis answers a narrower question: *given this model, which re-timings would a capacity-limited desk consider, and how much predicted risk would they remove?*
