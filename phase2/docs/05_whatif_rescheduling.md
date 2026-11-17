# 05 · What-if rescheduling (offline)

> Covers notebook [11_whatif_rescheduling](../notebooks/11_whatif_rescheduling.ipynb). Code: `src/aerocast_dl/reschedule.py`. Results: `results/reschedule_tuning.csv`, `reschedule_test.csv`, `reschedule_test_moves.csv`.

## The question

At 20:00 on D−1 the desk sees tomorrow's risk for every flight. Suppose it could move a few of the riskiest departures by an hour or two on the same day. How much **predicted** delay would that avoid?

## How a move is decided

1. **Candidates.** The daily top 10% riskiest flights of the **whole timetable**, ranked by the final model (LogReg with rolling Platt). Flights that will later be cancelled or diverted are included, because at 20:00 on D−1 nobody knows which they are. (Before the 2026-11-17 audit fix they were left out; [06](06_rules_and_limitations.md).)
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

**No leakage.** The benefit uses predicted risk only, never what actually happened on day D. Tests confirm that the plan is identical when every day-D outcome is scrambled or set to missing, that the scored schedule is the full timetable, and that planning refuses a partial origin-day.

## Tuned on Sep–Oct, then frozen

Net benefit = predicted delays avoided − 0.02 × moves, over Sep–Oct (61 days).

| Shift window | δ | Moves | Delays avoided | Net benefit |
|---|---|---|---|---|
| ±1 h | 0.01 | 2,912 | 74.7 | 16.5 |
| ±1 h | 0.02 | 1,607 | 55.1 | 22.9 |
| ±1 h | 0.05 | 204 | 12.7 | 8.6 |
| ±2 h | 0.01 | 3,613 | 125.9 | 53.7 |
| **±2 h** | **0.02** | **2,643** | **111.8** | **59.0** |
| ±2 h | 0.05 | 762 | 51.8 | 36.6 |

Frozen rule: **shifts up to ±2 hours, δ = 0.02**.

## Result on Nov–Dec (61 days)

| Rule | Moves/day | Predicted delays avoided/day | Per move | Net at cost 0.02 | Net at cost 0.05 |
|---|---|---|---|---|---|
| **Frozen rule** | 49.1 | 1.92 | 0.039 | +57.4 | −32.4 |
| Naive: every candidate 1 h earlier | 90.9 | 0.13 | 0.001 | −102.9 | −269.3 |

- **The rule avoids about 1.9 predicted delays a day.** That is with 49 moves a day, about half of the candidates. It pays at a cost of 0.02 per move but not at 0.05.
- **Most accepted moves are two hours:** 1,438 two hours earlier and 1,220 two hours later, against 335 one-hour moves.
- **"Earlier is better" is not the answer.** Moving everyone an hour earlier achieves almost nothing (+8.0 at zero cost, a loss at any positive cost), because it shifts congestion onto other flights at the origin: its spill-over is negative. The model's flight-by-flight assessment is what makes the rule work.

## How to read this

These are **predicted** effects under A1–A4, not realised savings. They are also biased upward: moves are chosen to maximise the predicted gain, and that same predicted gain is reported (the optimiser's curse). At about 0.04 predicted delays per move, the gain is of the order of the model's calibration error (test ECE 0.027). The data contain no rescheduled flights, no costs and no aircraft rotations. The analysis answers a narrower question: *given this model, which re-timings would a capacity-limited desk consider, and how much predicted risk would they remove?*
