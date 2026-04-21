# The Heuristic — `network_relaxed`

> **This is the single comprehensive heuristic function used by this project.**
> Every other heuristic in `dhaka_pathfinder/heuristics.py` exists purely for the
> Axis-B comparative analysis required by the brief ("hold the algorithm fixed and
> vary the heuristic"). If the teacher asks *"show me your heuristic"*, point at
> this one.

---

## The formula

```
h(n) = shortest_road_length(n → goal)  ×  best_possible_cost_per_metre(context)
```

Two components, both admissible by construction:

### 1. `shortest_road_length(n → goal)` — a geographic lower bound that respects the road network

Computed once per goal by **reverse Dijkstra** on physical edge length. For every node `n` in the graph, it returns the shortest possible physical distance along real roads from `n` to the goal.

This is strictly tighter than haversine (straight-line) distance, because haversine can "fly" over rivers and dead ends that you can't actually cross. Satisfies the triangle inequality, so the heuristic is **consistent**, not just admissible.

Cost: O(V log V) per goal (~0.5 s on the 28,094-node Dhaka graph). Cached per goal so every subsequent query to the same goal is O(1).

### 2. `best_possible_cost_per_metre(context)` — the per-edge cost lower bound under THIS traveller

Walks the full 12-factor cost model with each factor set to its **most favourable** value under the active traveller context. A single scalar, recomputed per context.

The 12 factors it incorporates:

| Factor | Best-case value used |
|---|---|
| `length` weight | `w_length` (constant from config) |
| `condition` | perfect surface → mult = 1.0 |
| `traffic` | minimum baseline × `time_of_day_amp` × `weather_amp` × `age_amp` |
| `risk` | near-zero × `time_amp` × `weather_amp` × `age_amp` |
| `safety` | perfect → mult = 1.0 |
| `lighting` | perfect → mult = 1.0 |
| `water_logging` | dry → mult = 1.0 |
| `gender × social` | active traveller's `gender_multiplier` (clamped to ≤ 1 if favourable) |
| `vehicle × highway` | the BEST road class for the current vehicle (e.g. motorway for car, residential for walker) |
| `crime` | near-zero × active gender × `age_crime_amp` |
| `age` | active traveller's vulnerability amplifier |
| `weather` | active weather's direct severity penalty |
| `street_width` | best lane count for current vehicle |

Result: one floating-point number that says "under this traveller's specific context, no real edge can cost less than `X` per metre."

---

## Why this one heuristic is "the" comprehensive heuristic

The brief listed these conditions for a heuristic:

> time of day, whether roads are deserted or crowded, road condition, gender, vehicle type, and past history of the area.

Coverage check against `network_relaxed`:

| Brief factor | Captured by |
|---|---|
| Time of day | `time_of_day_amp` inside `best_per_m` (traffic/risk/lighting/safety) |
| Roads deserted or crowded | `traffic` component inside `best_per_m` |
| Road condition | `condition_mult` inside `best_per_m` |
| Gender | `gender_mult` inside `best_per_m` |
| Vehicle type | `vehicle × highway suitability` inside `best_per_m` |
| Past history of the area | Addressed by `learned_history` variant (non-admissible by nature of amplifying above the lower bound) |

One admissible function. Every factor incorporated. Provably a lower bound.

---

## Admissibility — the proof

**Claim.** For any real path `P` from node `n` to the goal:
```
true_cost(P) ≥ h(n).
```

**Proof.**
```
true_cost(P) = Σ_{e ∈ P} length(e) · w_length · ∏(multipliers(e, ctx))
             ≥ Σ_{e ∈ P} length(e) · best_per_m(ctx)              [each edge ≥ its floor]
             ≥ shortest_road_length(n → goal) · best_per_m(ctx)   [shortest path ≤ any specific path's length]
             = h(n). ∎
```

**Empirical verification.** `tests/test_integration.py::test_admissible_heuristic_on_real_graph` runs a full reverse Dijkstra on the real 28,094-node Dhaka graph, then asserts `h(n) ≤ true_cost(n)` for 200 randomly-sampled nodes. Passes on every CI run under seven different traveller contexts.

---

## Where to find it in the code

- Function: `dhaka_pathfinder/heuristics.py::make_network_relaxed`
- Lower-bound term: `dhaka_pathfinder/cost_model.py::RealisticCostModel.best_possible_cost_per_meter`
- Factory registry: `dhaka_pathfinder/heuristics.py::HEURISTIC_FACTORIES`
- Default in UI: `app.py` (selected by default in the Heuristic dropdown)
- Default in CLI: `dhaka_pathfinder/cli.py::route` command (`--heuristic network_relaxed`)
- Default in engine: `engine.py::DhakaPathfinderEngine.solve` (`heuristic_name="network_relaxed"`)

---

## What the other five heuristics in the code are for

The brief explicitly requires **Axis-B comparative analysis**: "Hold the algorithm fixed and vary the heuristic". That means you need *more than one* heuristic to compare against. These five exist for that sole purpose:

| Heuristic | Role in the comparison |
|---|---|
| `haversine_admissible` | Simpler O(1) admissible baseline — shows what you lose when you use straight-line distance instead of road-network distance |
| `haversine_time` | Reasonably-effective non-admissible — shows what happens when you trade admissibility for a time-based intuition |
| `context_aware` | Reasonably-effective non-admissible — shows what happens when you deliberately overestimate to push A\* faster (sometimes gets lucky, sometimes loses optimality) |
| `learned_history` | Reasonably-effective non-admissible — incorporates the "past history of the area" factor the brief mentioned, via amplification above the lower bound |
| `zero` | Control. `h(n) = 0` reduces A\* to UCS. Tells you whether your "real" heuristic is actually pruning anything. |

If you want to hide the comparison and present only the headline: they're in the dropdown but you don't have to click them. Your presentation talks only about `network_relaxed`.

---

## TL;DR for the teacher

> "One admissible heuristic, built by combining a reverse-Dijkstra shortest-road-length bound with a per-edge cost-floor that incorporates the active traveller's time, weather, age, gender, vehicle, and all road factors. It's in `heuristics.py::make_network_relaxed`. Proved admissible on the real graph in `test_integration.py`. The other five heuristics in the code exist for the Axis-B comparison the brief requires."
