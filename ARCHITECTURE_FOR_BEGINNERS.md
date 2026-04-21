# Architecture for Total Beginners

> No jargon. No scary formulas without English translations. If something looks hard, the next sentence explains it simpler.

---

## The 30-second story

You open the app → you type "Shahbag to Banani" → you say "I'm walking, it's raining, I'm a woman alone at 11 PM" → the app draws a route on the map that avoids dark flooded alleys and sticks to lit main roads, even if that's longer.

That's it. Everything below is just *how* it does that.

---

## The big picture — 7 building blocks

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                         YOU (the user)                                │
 │                                                                       │
 │  Click buttons in Streamlit: source, destination, gender, vehicle... │
 └──────────┬───────────────────────────────────────────────────────────┘
            │  "Find me a route!"
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  1️⃣  THE MAP  —  osm_loader.py                                       │
 │                                                                       │
 │  Downloads real Dhaka road network from OpenStreetMap (28k places,    │
 │  70k roads). Only happens once; the second time it just loads from    │
 │  a saved file.                                                        │
 │                                                                       │
 │  Output: a "graph" = dots (intersections) connected by lines (roads) │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  2️⃣  THE DETAILS  —  synthetic_data.py                               │
 │                                                                       │
 │  OpenStreetMap tells us "this road exists". It does NOT tell us:     │
 │    · is it in good condition?                                         │
 │    · is it flooded?                                                   │
 │    · is it safe?                                                      │
 │    · how much traffic?                                                │
 │                                                                       │
 │  This file MAKES UP those numbers for every road — but realistically │
 │  (Old Dhaka is always rough; Gulshan is always nice).                 │
 │                                                                       │
 │  Output: every road now has 9 extra numbers tagged onto it.          │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  3️⃣  YOU  —  context.py                                              │
 │                                                                       │
 │  Collects what you said in the sidebar:                               │
 │    · gender (male/female/nonbinary)                                   │
 │    · social (alone/accompanied)                                       │
 │    · age (adult/child/elderly)                                        │
 │    · vehicle (walk/rickshaw/cng/car/bus/motorbike)                    │
 │    · time_bucket (midday/rush/late_night/...)                         │
 │    · weather (clear/rain/fog/storm/heat)                              │
 │                                                                       │
 │  Output: one TravelContext object describing YOU right now.          │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  4️⃣  THE COST CALCULATOR  —  cost_model.py                           │
 │                                                                       │
 │  For every road, combines:                                            │
 │    · the road's OWN numbers (from step 2)                             │
 │    · YOUR numbers (from step 3)                                       │
 │  and spits out a single "cost" for travelling that road right now.    │
 │                                                                       │
 │    cost = length_meters × (lots of multipliers)                       │
 │                                                                       │
 │  Output: a dictionary saying "road X → costs 4,237.1 for this        │
 │  traveller right now".                                                │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  5️⃣  THE "GUESS" FUNCTION  —  heuristics.py   ← THE h(n) FUNCTION   │
 │                                                                       │
 │  For any road intersection, guess "roughly how much more cost to     │
 │  reach the destination from here?"                                    │
 │                                                                       │
 │  The smart search algorithms use this guess to avoid wandering       │
 │  randomly. It's an ESTIMATE, not a real number.                       │
 │                                                                       │
 │  Output: a function h(node) → estimated remaining cost.               │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  6️⃣  THE SEARCH  —  algorithms/                                      │
 │                                                                       │
 │  Six different ways of walking through the graph looking for the     │
 │  cheapest path from source to destination.                            │
 │                                                                       │
 │  ┌─────────────────────────────────────────────────────────────┐    │
 │  │  BFS           — try every road one hop at a time           │    │
 │  │  DFS           — pick one road, keep going, backtrack       │    │
 │  │  UCS           — expand whichever path is cheapest so far   │    │
 │  │  Greedy        — expand whichever intersection LOOKS        │    │
 │  │                  closest to goal (uses h(n))                │    │
 │  │  A*            — combine "cost so far" + "guess" (g+h)      │    │
 │  │  Weighted A*   — same as A* but pushes harder toward goal   │    │
 │  └─────────────────────────────────────────────────────────────┘    │
 │                                                                       │
 │  Output: list of intersections [source, A, B, C, …, destination].    │
 └──────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  7️⃣  THE DRAWING  —  visualizer.py                                   │
 │                                                                       │
 │  Takes the list of intersections and draws the route on a real map.  │
 │  Colors the winning path green, losing paths dashed, etc.             │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## What happens when you click "Compute route"

Step by step, no hand-waving:

| # | What happens | Where in the code |
|---|---|---|
| 1 | You clicked — Streamlit wakes up the `run_button` handler | `app.py` |
| 2 | App reads your sidebar choices and builds a **TravelContext** | `app.py` → `context.py` |
| 3 | App looks up the Dhaka graph (already in memory, cached by Streamlit) | `engine.py::load_engine` |
| 4 | App finds the graph-node closest to your "Source" landmark's lat/lon | `engine.py::nearest` |
| 5 | Same for "Destination" | `engine.py::nearest` |
| 6 | Engine asks the cost model: "under this exact context, what does EVERY road cost?" | `engine.py::_weights_for` |
| 7 | Result is a dictionary with 70,197 numbers, one per road. Cached for this context. | `cost_model.py::precompute_edge_weights` |
| 8 | For informed algos (A*, Greedy, Weighted A*), engine builds the **heuristic** `h(n)` bound to this goal | `heuristics.py::make_network_relaxed` |
| 9 | Each algorithm walks the graph using those numbers + h(n), produces a path | `algorithms/*.py` |
| 10 | For each algorithm: "your winning route is nodes [1234, 5678, 9012, ..., 3456]" | returns a `SearchResult` |
| 11 | Visualizer converts node IDs → (lat, lon) pairs → Folium map | `visualizer.py::build_route_map` |
| 12 | Streamlit embeds the map + result table into the page | `app.py` |

---

## Where each value lives — the cheat sheet

### Per-road numbers (stored on every edge in the graph)

| Number | Means | Range | What it does when big |
|---|---|---|---|
| `length` | metres of actual road | 0 – hundreds | Longer road = more cost (obvious) |
| `condition` | surface quality | 0 – 1 | High = smooth. Low = potholes → cost goes up |
| `traffic_base` | baseline busyness | 0 – 1 | High = jammed → cost goes up |
| `risk` | accident probability | 0 – 1 | High = dangerous → cost goes up |
| `safety` | general safety | 0 – 1 | High = safe. Low → cost goes up |
| `lighting` | street lamps | 0 – 1 | High = bright. Low → cost goes up at night |
| `water_logging_prob` | flood chance | 0 – 1 | High = floods in rain → cost explodes in rain |
| `crime_index` | neighbourhood crime | 0 – 1 | High = bad area → cost goes up for women especially |
| `num_lanes` | number of lanes | 1 – 10 | More lanes = cheaper for cars, harder for walkers to cross |
| `highway_class` | road type tag | `residential` / `primary` / etc | Determines vehicle suitability |

*Stored in:* each edge in the `G` MultiDiGraph, added by `synthetic_data.py::augment_graph`.

### Per-traveller numbers (what you pick in the UI)

| Field | Possible values | Controls |
|---|---|---|
| `gender` | `male`, `female`, `nonbinary` | Safety multiplier — women alone at night cost more |
| `social` | `alone`, `accompanied` | Pairs with gender — accompanied is safer |
| `age` | `adult`, `child`, `elderly` | Risk amp, crime amp, vehicle restrictions |
| `vehicle` | `walk`, `rickshaw`, `cng`, `motorbike`, `car`, `bus` | Which roads are even usable |
| `time_bucket` | `midday`, `morning_rush`, `late_night`, ... | Traffic and risk amplification |
| `weather` | `clear`, `rain`, `fog`, `storm`, `heat` | Water-log, lighting, risk, traffic amplification |

*Stored in:* a `TravelContext` dataclass (`context.py`) — frozen, can't be changed once created.

### Cost-function weights (tuning knobs)

| Weight | Default | Range | Controls |
|---|---|---|---|
| `length` | 1.0 | 0 – 3 | How much raw distance matters |
| `safety` | 1.2 | 0 – 3 | How scared we are of unsafe roads |
| `risk` | 1.5 | 0 – 3 | How scared we are of accidents |
| `traffic` | 1.3 | 0 – 3 | How much we hate jams |
| `gender_safety` | 1.4 | 0 – 3 | How much gender matters |
| `water_logging` | 0.4 | 0 – 2 | How much we care about floods |
| `age` | 1.1 | 0 – 2 | How much age matters |
| `weather` | 1.0 | 0 – 2 | How much weather matters |
| ...(12 total) | ... | ... | ... |

Change one of these in `config.py::DEFAULT_WEIGHTS` and the whole system re-weights itself. Or pick a preset (`balanced`, `safety`, `speed`, `comfort`) to get a coherent bundle.

---

## Turning knobs — what happens if you change things?

| Change this | Effect |
|---|---|
| `age = child` instead of `adult` | Cost roughly 1.3–1.8× higher on every edge. Motorbike routes are 2.5× penalised (children aren't allowed). |
| `weather = rain` instead of `clear` | Water-logging cost × 2.2, lighting cost × 1.3, risk × 1.4, traffic × 1.3. Roads in flood-prone areas get very expensive. |
| `weather = storm` | Like rain but WORSE — water-log × 3, risk × 1.85. |
| `vehicle = walk` on a trip that defaults to `car` | Motorways become basically forbidden (suitability = 0.15). Algorithm reroutes through residential streets. |
| `time_bucket = late_night` | Traffic drops (×0.3), but risk amp goes up (×1.4) and lighting amp down (×0.4). Safety-conscious routing. |
| `time_bucket = evening_rush` | Traffic amp × 2.0. Everything slows down; cost shoots up on busy roads. |
| `weight_preset = safety` | Safety / risk / crime / gender weights doubled; traffic weight cut. Routes go through safer but longer paths. |
| `weight_preset = speed` | Traffic weight × 1.5; safety weights halved. Routes go through direct but potentially unsafe paths. |
| `DHAKA_BBOX` in config.py | Changes which part of Dhaka is loaded. Big bbox = bigger graph, slower queries. |
| `SYNTHETIC_SEED` | Changes the random synthetic numbers. Same seed = same graph every run. |

---

## The `g(n)` vs `h(n)` distinction (your friend's question)

Your friend asked:

> "In A*, did I put all evaluation metrics in h(n)?"

The answer is **NO — and this is the single most important misconception to clear up about A\***.

### A* has TWO numbers, not one

For every intersection `n` that A* is considering, it computes:

```
f(n) = g(n) + h(n)
```

- **`g(n)`** = **"the actual cost I've ALREADY paid to reach this intersection from the source."**
  It's a **real, measured number**. You sum up the costs of the roads you actually walked along.

- **`h(n)`** = **"an ESTIMATE of how much more cost I'll have to pay to reach the destination from here."**
  It's a **guess**. You don't actually know yet; you're predicting.

A* then picks the intersection with the **smallest `f(n)`** next — smallest "total known + guessed total cost to finish".

### Where do the 12 evaluation metrics go?

All 12 metrics (traffic, gender, safety, risk, lighting, etc.) live in the **cost function** — the thing that decides how expensive each *individual road* is:

```
cost_of_road(road, context) = length × Π(12 multipliers for that road + traveller)
```

Then:

- `g(n)` = sum of `cost_of_road` for every road the search actually travelled to reach `n`. **← All 12 metrics are BAKED INTO g(n).**
- `h(n)` = a FAST GUESS of what `g` will look like by the time we reach the destination. It doesn't need to redo all the metric math for every remaining road — that would be as slow as just solving the problem.

### A concrete tiny example

Imagine a tiny graph: `S → A → B → GOAL`.

| Road | `length` | `traffic` | `safety` | ... | `cost_of_road` (output of the cost model) |
|---|---|---|---|---|---|
| S→A | 500 m | 0.4 | 0.8 | ... | **3,200** |
| A→B | 300 m | 0.6 | 0.3 | ... | **5,100** |
| B→GOAL | 400 m | 0.5 | 0.7 | ... | **3,800** |

A\* walking from S toward GOAL:

```
At S:      g(S) = 0.            h(S) = guess ≈ 9,000.   f(S) = 9,000.
After S→A: g(A) = 3,200.        h(A) = guess ≈ 7,500.   f(A) = 10,700.
After A→B: g(B) = 3,200+5,100   h(B) = guess ≈ 3,900.   f(B) = 12,200.
           = 8,300.
At GOAL:   g(GOAL) = 8,300      h(GOAL) = 0.            f(GOAL) = 12,100.
           + 3,800 = 12,100.    (we're HERE)
```

Notice:

- The 12 metrics (traffic, safety, risk, etc.) shaped **every road cost**, which fed into `g`.
- `h` is just the estimate — it never recomputes the full metric math for roads not yet traversed.
- By the time we reach GOAL, `g(GOAL)` is the **actual total realistic path cost**, which is what the project reports.

### How to answer your friend

> "No. The metrics (traffic, safety, gender, age, weather, vehicle, etc.) live in the COST function — they decide the cost of each individual road. A\* sums up those costs as `g(n)` = actual cost so far. `h(n)` is just a fast ESTIMATE of remaining cost from `n` to the goal. `f(n) = g(n) + h(n)` is what A\* uses to decide which intersection to expand next. Metrics in `g`, guess in `h`, decision from `f`."

### The really short version

| Symbol | What | Where the 12 metrics go |
|---|---|---|
| `cost_of_road(r, ctx)` | Price of one road | **Every metric lives here** |
| `g(n)` | Actual cost paid so far | **Sum of those prices — so metrics live here too** |
| `h(n)` | Guess of remaining cost | Only a BOUND — uses metrics indirectly through `best_possible_cost_per_metre` but doesn't redo the per-road math |
| `f(n) = g(n) + h(n)` | A*'s decision score | — |

---

## Admissibility — one more time, gently

The `h(n)` function has one rule: **it must never say "remaining cost is 100" when the real remaining cost is 90**. It's allowed to UNDER-estimate (say 60 when the truth is 90); it's not allowed to OVER-estimate.

Why? Because if `h` over-estimates, A\* thinks "this direction is too expensive, skip it" — and might skip the actual best path. If `h` under-estimates, A\* might try that direction, realise later it's expensive (because `g` goes up), and give up on it. Either way, no optimal path is wrongly discarded.

A heuristic that never over-estimates is called **admissible**. Our `network_relaxed` heuristic is admissible — proved with a real test on the 28,000-node Dhaka graph.

---

## One-page summary (stick this on your wall)

1. **Map (OSMnx)** → 28,094 intersections, 70,197 roads.
2. **Details (synthetic)** → each road gets 9 extra numbers: condition, traffic, risk, safety, lighting, flood prob, crime, lanes, etc.
3. **Context (you)** → 6 fields: gender, social, age, vehicle, time_bucket, weather.
4. **Cost function** → takes a road + a context, multiplies 12 things together, outputs one number (the cost of that road for that traveller right now).
5. **Heuristic** → takes a node, outputs a guess of "remaining cost from here to goal".
6. **Algorithms (6)** → walk the graph to find the cheapest path. Informed ones use the heuristic.
7. **Visualizer** → paints the winning path on a real map.

**The metrics (safety, traffic, etc.) are in the COST FUNCTION.** They feed `g(n)` indirectly. The heuristic `h(n)` is a separate FAST guess — not a dumping ground for everything.
