# Q&A — Questions the Teacher Might Ask

> Compiled and answered. Each answer is short, correct, and keyed to either a file/function or a line in the report so you can point at the evidence.

---

## Section A — Algorithm fundamentals

### Q1. Why did you pick **these six** algorithms?

Three uninformed, three informed — satisfies the brief. Specifically:

- **BFS** — teaches hop-minimisation (not cost-optimal).
- **DFS** — teaches catastrophic behaviour on weighted graphs (12× worse than optimum).
- **UCS (Dijkstra)** — the gold standard; optimal, no heuristic needed.
- **Greedy Best-First** — demonstrates heuristic without `g(n)` — fast, suboptimal.
- **A\*** — the canonical optimal informed search.
- **Weighted A\*** (w=1.8) — the speed/quality trade-off.

These six cover the full pedagogical spectrum from "fewest hops" to "bounded-suboptimal fast search".

### Q2. Why **not** IDS, IDA\*, Bidirectional Search?

- **IDS** mixes hop-depth with cost; hard to interpret on a realistic-cost metric.
- **IDA\*** would expand more nodes than A\* on our graph (uniform cost, no obvious iterative-deepening benefit).
- **Bidirectional Dijkstra** is a clear speedup in practice but doesn't teach anything new about the cost model. Listed as future work in `PRESENTATION.md` §12.

### Q3. What is the **time complexity** of each algorithm?

Let `V` = nodes, `E` = edges, `b` = branching factor, `d` = solution depth.

| Algorithm | Time | Space |
|---|---|---|
| BFS | O(V + E) | O(V) |
| DFS | O(V + E) | O(bd) with depth cap |
| UCS | O(E log V) | O(V) |
| Greedy | O(E log V) — in practice far less, most nodes never touched | O(V) |
| A\* | O(E log V) in worst case; practical best case O(bd) | O(V) |
| Weighted A\* (w>1) | same order as A\*; smaller constant | O(V) |

### Q4. Why use a **MultiDiGraph** instead of a simple graph?

OSM has one-way streets (→ directed) AND sometimes multiple parallel road segments between the same intersections (→ multi). The MultiDiGraph is faithful to the real topology. `algorithms/base.py::edge_cost_lookup` handles the "cheapest parallel edge" case.

### Q5. What is **EBF (Effective Branching Factor)** and why compute it?

The value `b*` satisfying `1 + b* + b*² + … + b*^d = N`, where `d` = solution depth and `N` = nodes expanded. It tells you how "branchy" the algorithm's search tree actually is.

- A\* with a perfect heuristic → EBF ≈ 1.
- Uninformed BFS → EBF ≈ real branching factor of the graph (~5 in road networks).

Computed by Newton iteration in `algorithms/base.py::effective_branching_factor`.

### Q6. What is **admissibility**?

A heuristic `h(n)` is admissible iff `h(n) ≤ true_remaining_cost(n → goal)` for every node `n`. It's the lower-bound property.

### Q7. Why does admissibility matter?

**A\* with an admissible heuristic is guaranteed to return the optimal path.** Proof sketch: when A\* pops the goal from the frontier, `f(goal) = g(goal)` (because `h(goal) = 0`), and any other path to the goal must have `f ≥ true_cost ≥ g(goal)`. So the popped path has the lowest possible cost. See Russell & Norvig Chapter 3.

### Q8. What's the difference between **admissible** and **consistent**?

- **Admissible:** `h(n) ≤ h*(n)` (never overestimates).
- **Consistent / monotonic:** `h(u) ≤ cost(u, v) + h(v)` for every edge.

Consistency ⇒ admissibility. With a consistent heuristic, A\* never re-opens a closed node — simpler and faster. Both of our admissible heuristics (`haversine_admissible`, `network_relaxed`) are consistent: haversine is the straight-line distance (triangle inequality), and road-shortest-length also satisfies the triangle inequality.

### Q9. Can you **prove** your heuristic is admissible?

Yes. For `haversine_admissible`:

```
actual_cost(path_from_n_to_goal)
  = sum over edges e of: length(e) × w_length × Π(multipliers(e, context))
  ≥ sum over edges of: length(e) × best_possible_cost_per_metre(context)
  = path_physical_length × best_per_m
  ≥ haversine_distance(n, goal) × best_per_m
  = h(n).
```

For `network_relaxed` replace `haversine` with `shortest_physical_length_via_roads` — the argument is the same, and the bound is **tighter** because the shortest road path length ≥ haversine.

And it's checked empirically — `tests/test_integration.py::test_admissible_heuristic_on_real_graph` asserts the inequality for 200 random nodes on the real graph, using reverse-Dijkstra ground truth.

### Q10. If a heuristic **over**estimates, what happens?

A\* may return a **suboptimal** path. It still returns a path (the algorithm terminates), but the guarantee of optimality is lost. This is what happens with `context_aware` and `haversine_time` in our suite — they sometimes overestimate. The analysis reports the predicted-vs-actual gap so you can see how far off they are.

### Q11. Why does **UCS** always agree with **A\*** on cost?

Both are provably optimal. The set of paths they optimise over is the same (all paths from source to destination), and the objective is the same (sum of edge costs). Same minimum over the same set ⇒ same value. Different instances of that minimum (different paths) are possible, but the cost is always equal. Our 7,350-run analysis confirms this — UCS and A\* tie in every run.

### Q12. Why is **Weighted A\*** with w=1.8 also hitting the optimum on your graph?

Weighted A\* with `w > 1` is **bounded suboptimal** — in theory it can return a path up to `w` times worse than optimal. In practice, it often still returns the optimum because:
1. Our admissible heuristic is loose (≈ 1/20 of true cost), so `w × h` is still a lower bound most of the time.
2. For any pair where the greedy-ish bias of weighted A\* doesn't mislead it, it hits the optimum.

If I raised `w` to ~3.0 or used a tighter heuristic, I'd start seeing worse-than-optimal results from Weighted A\*.

### Q13. Why does **Greedy** expand so few nodes?

Greedy only uses `h(n)`, not `g(n)`. It heads straight toward the goal based on the heuristic estimate. No back-tracking to account for high-cost edges already traversed. On our graph Greedy expands ~169 nodes vs A\*'s ~13,000 — 78× fewer. But its resulting path costs ~34% more than optimal.

### Q14. Why does **DFS** produce a 117-km path on a graph whose straight-line goal is 3 km away?

DFS goes as deep as it can before backtracking. On a 28k-node graph it can descend 1,500 edges (our depth cap) along one branch before it ever considers alternatives. The result is an extremely circuitous path. With no cost awareness, it has no incentive to course-correct.

### Q15. What is the difference between **BFS** and **UCS** on a weighted graph?

BFS expands nodes in FIFO order (shortest in *hops*). UCS expands in priority order of `g(n)` (shortest in *cost*). When edge costs are non-uniform, these disagree. BFS on our graph produces paths that are 34% more expensive than UCS on the same pair, because BFS minimises hops not cost.

---

## Section B — The cost model

### Q16. Why **multiplicative** cost, not additive?

Additive costs miss interactions. Example: a dark road is bad (+1); being female alone is a penalty (+1); Old Dhaka at midnight has high crime (+1). Additive total: +3. But when all three are simultaneously true, the trip is not "inconveniently bad" — it's **actually dangerous**. Multiplicative captures that: 1.5 × 1.6 × 1.8 = 4.3×, not 3×.

### Q17. What would happen if I set one of the multipliers to **zero**?

Setting a weight `w_factor = 0` makes that factor's multiplier always `1`, so it's ignored. Setting a component value to 0 could theoretically zero out the total product, but every `×` factor is clamped to ≥ 0.5 minimum in the code to prevent this. See `cost_model.py::edge_breakdown`.

### Q18. Where do the **specific weight values** come from?

The default weights in `config.py::DEFAULT_WEIGHTS` were picked heuristically to make risk, traffic, gender-safety, and safety the dominant terms. They're not tuned from real Dhaka data (nobody has that data). The sensitivity of the model to weight changes is visible via the three presets — `balanced`, `speed`, `safety`, `comfort` — which produce different routes on the same pair.

### Q19. How do you ensure the traveller context actually changes the route?

Test: `tests/test_cost_model.py::test_female_alone_late_night_costlier_than_male_accompanied_midday` asserts strictly higher cost in the risky context on the same edge. Plus `tests/test_new_factors.py` adds tests for child > adult, rain > clear, wide road > narrow (for walkers), etc.

Empirically: in the 7,350-run analysis, child contexts cost 32% more than adult on the same pair family; rain contexts cost 3.1× more than clear; fog contexts 3.8× more.

### Q20. Why did you add **weather, age, and street width** *after* the initial implementation?

Review after initial submission found three gaps:
1. `weather` was in `TravelContext` and appeared in the analyser CSV, but was **never read** by the cost model — dead code.
2. `age` (adult/child/elderly) was in the brief but not modelled at all.
3. Street width (OSM `lanes` tag) was downloaded but unused.

I added them in v2 with matching multipliers in `config.py` and applied in `cost_model.py::edge_breakdown`. The `best_possible_cost_per_meter` lower bound was updated to include the new factors, preserving admissibility.

### Q21. Can you demonstrate the **age** factor at work?

```bash
./run.sh route --source Shahbag --dest "Gulshan 2" --vehicle motorbike --age child
```

The cost is heavily penalised because of `AGE_VEHICLE_RESTRICTION["child"] = ("motorbike",)` — children aren't allowed on motorbikes. The algorithm routes around any choice that requires that vehicle.

### Q22. Why is `estimated_time_s` separate from `path_cost`?

Cost is what the algorithm **minimises** — it's a composite of safety, comfort, etc. Time is what the user **perceives** — purely the physical km divided by vehicle speed adjusted for traffic and weather. They're related but not the same. The UI shows both so the user has full context.

---

## Section C — The heuristics

### Q23. What makes `network_relaxed` **tighter** than `haversine_admissible`?

Haversine is the straight-line geographical distance — it ignores the road network. If source and goal are on opposite sides of a river with only one distant bridge, haversine gives you the "as-the-crow-flies" distance (small); the actual road path is much longer (via the bridge).

`network_relaxed` runs a reverse Dijkstra on physical edge length, so it knows about the bridge. The result is `≥ haversine` (triangle inequality) while still `≤ true cost` (admissibility). Tighter lower bound → A\* prunes more aggressively.

### Q24. Why is the **network_relaxed** heuristic not the default?

Two reasons:
1. **Precomputation cost** — each goal requires an O(V log V) reverse Dijkstra (~0.5 s on our graph). For a one-shot query this cost dominates.
2. **Cache hit rate** — only amortises when multiple queries share the same goal. The analyser benefits (many contexts × same goal). The single-query UI doesn't.

So `haversine_admissible` is the default (O(1) per node); `network_relaxed` is available in the dropdown for users who want the tighter heuristic and can tolerate the initial latency.

### Q25. Why include a `zero` heuristic?

It's the **control group**. A\* with `h = 0` is mathematically equivalent to UCS — the analysis confirms they expand the same nodes. This lets me isolate "is my actual heuristic doing any work?". The answer: yes. `haversine_admissible` expands 6% fewer nodes than `zero`; `network_relaxed` expands 10% fewer.

### Q26. Your heuristics assume the **goal is fixed** — what if the goal changes between calls?

Each heuristic factory (`make_haversine_admissible`, `make_network_relaxed`, etc.) binds the goal at construction time by closing over it. A new goal → new heuristic instance. The `network_relaxed` precomputed Dijkstra is cached per `(graph_id, goal)` so different contexts for the same goal reuse it.

### Q27. Is your heuristic **consistent** or only admissible?

Both `haversine_admissible` and `network_relaxed` are consistent. Haversine satisfies the triangle inequality by construction; road-shortest-length also satisfies it because any valid path `u → v → goal` is at least as long as the shortest path `u → goal`. A\* with a consistent heuristic never re-opens closed nodes — we exploit this in `algorithms/informed.py`.

### Q28. How would you make the admissible heuristic **even tighter**?

Implement **ALT** (A\* with Landmarks). Precompute shortest paths from ~8 carefully chosen landmark nodes to every other node. Use the triangle inequality: `h(n, goal) = max_L |d(n, L) - d(goal, L)|`. This is still admissible but often much closer to `h*`. Preprocessing is O(k · V log V) for `k` landmarks; query is O(k). Listed as future work in `PRESENTATION.md` §12.

---

## Section D — OSM and synthetic data

### Q29. Why OSMnx and not a saved JSON file?

The brief explicitly forbids JSON exports ("using exported JSON files as a substitute is not acceptable — it was specifically called out as bad practice"). OSMnx gives us a live, up-to-date graph with correct topology and a standard NetworkX interface. I cache the downloaded graph as a pickle so only the first run pays the network cost.

### Q30. Why the **largest strongly-connected component**?

In the raw OSM download, some nodes are unreachable from others (dead-end one-ways, OSM data quality issues). Keeping only the largest SCC guarantees every `(source, destination)` pair has at least one valid path. See `osm_loader.py::largest_strongly_connected_subgraph`.

### Q31. How do you **generate** the synthetic data? How is it **realistic**?

`synthetic_data.py::augment_graph`:
1. Start from OSM tags (`highway`, `lit`, `maxspeed`, `lanes`) which carry real info.
2. Bucket nodes into named Dhaka areas (Old Dhaka / Gulshan / Mirpur / …) by coordinates.
3. Apply per-area safety profiles (`config.py::AREA_SAFETY_PROFILE`).
4. Apply an exponential Old-Dhaka-bias that makes central-south segments systematically worse.
5. Add seeded Gaussian noise (σ=0.08) for variety.
6. Cluster nodes via MiniBatch K-Means and apply per-cluster factors for consistency between nearby edges.

Result: Old Dhaka is consistently flood-prone and rough; Gulshan is consistently safe and well-maintained; variation between individual edges feels natural.

### Q32. What's the **scale** of the graph?

28,094 intersections, 70,197 road segments, 5,826 km of road. The bbox is (90.34, 23.70, 90.46, 23.86) — roughly 12 × 18 km covering central Dhaka including Dhanmondi, Banani, Gulshan, Mirpur (partial), Old Dhaka, and up to the airport.

### Q33. Why not use the **whole of Dhaka division**?

The network-size scaling of our BFS/DFS is linear; A\*/UCS is O(E log V). Doubling the graph doubles runtime. The current graph is large enough to be realistic, small enough to fit in memory and finish a 7,350-run analysis in 15 minutes. Future scaling is straightforward — change `DHAKA_BBOX` and rerun `./run.sh download`.

### Q34. Your synthetic data is **seeded**. Isn't that unrealistic?

Yes, and intentionally so — **reproducibility**. Two different runs with the same seed produce identical graphs. That lets us compare algorithms on a fixed substrate. The seed is declared in `config.py::SYNTHETIC_SEED = 42`. Change it and you get a different-but-plausible Dhaka.

---

## Section E — Engineering and reproducibility

### Q35. How do you **test** correctness?

22 tests across five categories:

- **Cost model invariants** (`test_cost_model.py`): breakdown components positive, female-night strictly > male-day, admissible lower bound holds.
- **Heuristic admissibility on toy graph** (`test_heuristics.py`): full SSSP ground truth.
- **Algorithm correctness** (`test_algorithms.py`): all six find a path; UCS and A\* agree on cost.
- **Real-graph integration** (`test_integration.py`): admissibility on 200 sampled nodes of the 28k-node graph; UCS ≡ A\* on 6 random pairs; all 6 algorithms succeed on a random pair.
- **New factors** (`test_new_factors.py`): 11 tests for age, weather, lanes including 45-combination admissibility proof.

All pass: `./run.sh test` → `22 passed in 11.34s`.

### Q36. Why **uv** instead of pip?

uv (from Astral) is a fast, modern Python package manager. Advantages:

- **10–100× faster than pip** for resolution and install.
- **Deterministic installs** via `uv.lock` — a fresh clone reproduces the exact same environment.
- **Single tool** for venv creation, dep install, dep add, script running.
- `uv sync` is idempotent — on first call creates `.venv`, on subsequent calls only installs what changed.

The project still works with `pip install -r requirements.txt` for anyone who prefers it.

### Q37. What's your **concurrency** story?

Single-threaded Python. The algorithm work is CPU-bound (heapq operations, graph traversals). A multi-threaded implementation wouldn't help due to the GIL. A multi-process implementation for the analyser (run each pair in a subprocess) would speed up the 7,350-run sweep but add complexity — deliberate trade-off.

### Q38. What if the graph has **negative-weight edges**?

Our cost function guarantees `cost ≥ MIN_COST = 1e-3 > 0` for every edge. Every multiplier is clamped to `max(·, 0.5)` or positive. So negative weights are impossible by construction. Dijkstra / A\* would break on negative weights; we don't have to defend against them.

### Q39. What about **changing traffic data at query time**?

The model has a `stochastic` knob: `RealisticCostModel(stochastic=True)` multiplies traffic by `Uniform(0.85, 1.15)` per query. For reproducibility in the analyser we pass `stochastic=False`. In a real-time system, you'd plug live traffic into `traffic_base`; the rest of the model is unaffected.

### Q40. How would you **scale** this to all of Bangladesh?

1. Increase `DHAKA_BBOX` (or use `graph_from_place`).
2. For graphs >500k nodes, switch from pickle cache to a database (graph-tool + PostGIS).
3. For A\* at that scale, add ALT or a bidirectional variant.
4. For the synthetic layer, parallelise edge augmentation with Dask.

The architecture is modular enough that each of these changes affects at most one module.

---

## Section F — Results interpretation

### Q41. Why does **BFS "win"** in some contexts on runtime but lose on cost?

BFS's inner loop is trivial (FIFO queue, no priority). It terminates as soon as the goal is popped — typically quickly because BFS minimises hops. But the path it returns isn't cost-optimal. A good mnemonic: BFS is **fast to compute** a path but returns an **expensive** path. UCS is **slower to compute** but returns a **cheaper** path.

### Q42. Why do all three optimal algorithms always tie on cost?

**Theorem.** UCS, A\* with admissible `h`, and Weighted A\* with `w ≥ 1` are all proved to minimise path cost (the last with bounded slack). They optimise the same objective over the same constraint set, so they reach the same minimum value. Different *paths* may instance that same minimum when multiple optima exist; the *cost* is invariant.

The winner-board in the Streamlit UI makes this explicit: cost has a 3-way tie, but "fewest expansions" and "fastest" are single-winner (Greedy almost always).

### Q43. What does the **predicted-vs-actual gap** mean?

```
gap = actual_cost − predicted_cost_at_start = actual_cost − h(source)
```

- Positive ⇒ heuristic underestimated ⇒ admissible.
- Negative ⇒ heuristic overestimated ⇒ non-admissible.

Mean gap for A\* + haversine_admissible: ~159k on our graph, actual cost ~167k. So the heuristic is capturing ~5% of the true cost — admissible but loose. That's why A\* is only ~6% more efficient than UCS on this graph.

### Q44. Why do context plots show **NaN** in some cells?

The analyser uses a *representative* context list (7 specific combinations), not a full cross product. For crosstabs like `gender × time_of_day`, some combinations weren't in the sample, hence empty cells. The report explicitly calls this out and uses paired comparisons on pairs-that-ran-in-both-contexts to extract the gender effect.

### Q45. What's the **best algorithm overall**?

Depends on the metric:

- **Lowest cost** — UCS / A\* / Weighted A\* (tie).
- **Fewest nodes expanded** — Greedy.
- **Fastest wall-clock** — Greedy.
- **Best speed/cost trade-off** — **Weighted A\*** — optimal on our graph AND expands fewer nodes than A\*.

For production use I'd pick **Weighted A\*** with `network_relaxed` — tied for optimal, 16% fewer expansions than UCS.

---

## Section G — Trade-offs

### Q46. What's the trade-off between **admissibility** and **realism**?

An admissible heuristic preserves optimality but must be conservative (lower bound). A realistic heuristic (e.g., one that correctly predicts how much traffic will slow you) is usually non-admissible — traffic is stochastic, you can over- or under-estimate.

Our solution: offer both in the heuristic dropdown. Users who need provable optimality pick `haversine_admissible` or `network_relaxed`. Users who want human-intuitive estimates pick `context_aware` or `learned_history`.

### Q47. What's the trade-off between **speed** and **optimality**?

UCS is optimal but expands the most nodes. Greedy is fast but suboptimal. A\* with an admissible heuristic is optimal AND faster than UCS. Weighted A\* gives you a dial: w=1 → A\*, w=∞ → Greedy; anywhere in between trades optimality for speed.

On our graph: Weighted A\* w=1.8 happens to remain optimal (heuristic is loose enough) while expanding 16% fewer nodes than UCS. Best of both worlds.

### Q48. Why not make the cost function **differentiable** and optimise with gradient descent?

The search space is discrete (paths in a graph), not continuous. Gradient descent doesn't apply to combinatorial optimisation. Classical search (Dijkstra / A\*) is the right tool.

---

## Section H — Edge cases and robustness

### Q49. What if the source and destination are the **same node**?

Every algorithm returns a path of length 0 and cost 0. Checked at the top of each algorithm.

### Q50. What if the destination is **unreachable**?

Impossible by construction — we keep only the largest strongly-connected component. But if it did happen, each algorithm returns `success=False` after exhausting its frontier (or hitting `max_nodes`).

### Q51. What if OSM is **down**?

The first run downloads and caches the graph as a pickle. Subsequent runs load from the cache (`data/dhaka_bbox_*.pkl`). Once cached, no internet is needed.

### Q52. What if the user enters **invalid coordinates**?

`ox.nearest_nodes` still finds the nearest graph node even if the input coordinates are outside the bbox. The UI shows the resolved node, so the user can see they snapped to an edge of the graph.

### Q53. What if multiple parallel roads connect the same intersection pair?

`edge_cost_lookup` in `algorithms/base.py` picks the cheapest parallel edge. The path reconstructor records that specific edge's key.

### Q54. What if the cost function returns **0 or inf**?

Clamped — `cost = max(base_length × w_length × total_multiplier, MIN_COST)`. `MIN_COST = 1e-3`. No edge can have 0 cost. Infinite cost isn't possible because every multiplier is bounded.

---

## Section I — Architecture and design

### Q55. Why did you separate `engine.py` from `cli.py` and `app.py`?

Single responsibility. `engine.py` holds the graph and the cost model — the "domain". `cli.py` and `app.py` are UI adapters. This means I can write a new UI (e.g., a REST API) without touching the engine. Tests also use the engine directly, bypassing the UIs.

### Q56. Why are `cost_model.py` and `heuristics.py` separate?

Heuristics *use* the cost model (they call `best_possible_cost_per_meter`), but they're a distinct concept — they estimate future cost, the model computes per-edge cost. Separating makes it trivial to add a new heuristic without touching the cost model, and vice versa.

### Q57. How would you add a **new cost factor** (say, air pollution)?

1. Add `air_quality` synthetic attribute in `synthetic_data.py::augment_graph`.
2. Add `w_air_quality` to `CostWeights` in `config.py`.
3. Add `air_quality_mult` term in `cost_model.py::edge_breakdown`.
4. Update `best_possible_cost_per_meter` to include it.
5. Add a test that asserts admissibility still holds.

The architecture isolates the change to exactly those files.

### Q58. Why a **frozen dataclass** for `TravelContext`?

Immutable → safe to hash → safe to use as cache key in `_weight_cache`. Also prevents accidental mutation — you use `ctx.with_vehicle("walk")` instead of `ctx.vehicle = "walk"`. Python idiom for value types.

### Q59. Why is the `CostWeights` dataclass separate from the model?

Same immutability reasoning, plus it lets you pass different weight presets (`balanced`, `safety`, `speed`, `comfort`) by constructing a new `RealisticCostModel(weights=preset)` without changing the model class itself.

---

## Section J — Specific results questions

### Q60. On the map, why do UCS, A\*, and Weighted A\* produce the **same path**?

They're all optimal — they all return the same minimum-cost path. When multiple equally-optimal paths exist, they may differ, but for most queries the optimum is unique and they match.

### Q61. Why does the **Greedy** path often look shorter on the map?

Greedy minimises estimated remaining distance, which correlates with physical length. So Greedy's paths tend to be physically shorter but cost more (because they go through worse segments). UCS/A\*/WA\* accept longer physical paths if the cost is lower.

### Q62. Why does **DFS** find a path 30× longer than UCS?

DFS has no cost awareness. It walks down one branch until it hits the depth cap (1,500 edges) or the goal. If the goal happens to be reachable via a very indirect route from the start of DFS's exploration, DFS will find that route first and report it.

### Q63. What would change if the weight presets were **different**?

Routes would change. The `balanced` preset gives a neutral choice; `safety` heavily favours well-lit, low-crime roads; `speed` tolerates traffic for shorter paths; `comfort` prefers smooth roads. The SAME (source, destination, traveller context, algorithm) under different presets can produce visually different routes.

---

## Section K — Meta / open-ended

### Q64. How would you **evaluate** this system against ground truth?

Ground truth doesn't exist for "best route under a multi-factor cost". We'd need:
1. A labelled dataset where experts ranked routes by preference under various contexts.
2. Compare our top-1 route to the expert's top-1. Agreement = evaluation metric.

In absence of that: the system passes theoretical tests (admissibility, optimality tie-ins) and behavioural tests (child costs more than adult, rain costs more than clear, walker avoids motorways).

### Q65. How is this different from **Google Maps**?

- Google optimises for time or distance; we optimise for a 12-factor cost.
- Google uses real-time traffic; we use seeded synthetic.
- Google has no gender / age / social safety awareness; we do.
- Google is a black box; every weight and factor here is transparent and editable.

For the intended use case (routes where safety and comfort matter), our tool has a design advantage. For fast production-grade time-optimised routing on the whole world, Google wins.

### Q66. What's the **real-world impact** of this kind of system?

- Safety-aware routing apps for women / elderly / children.
- Logistics — routing trucks around narrow streets (width factor).
- Disaster response — routing ambulances around flooded areas (water-logging factor).
- Tourism — "scenic" vs "safe" vs "fast" presets.
- Public policy — identify areas where the cost function tells you "we should add streetlights here".

### Q67. What **limitations** does this project have?

1. Synthetic data is plausible but not real. Production would need ground-truth traffic / crime / water-logging data.
2. Only central Dhaka bbox — not the whole city or the country.
3. No real-time updates — the cost is snapshot-at-query-time, no live traffic.
4. No multi-modal routing (e.g., walk + bus + walk).
5. Cost function weights are hand-picked, not learned from data.

### Q68. What would you do **differently** if starting over?

1. Pick the bbox BEFORE writing landmark coordinates — fewer landmarks fell outside than I'd have liked.
2. Add the age / weather / street-width factors in v1 instead of v2 — retrofitting required adjusting `best_possible_cost_per_meter` and multiple tests.
3. Write the admissibility test against the real graph FIRST, before writing the heuristic — test-driven.
4. Use a bigger `max_nodes` budget by default (200k → 500k) to reduce edge-case DFS failures.

### Q69. What was the **hardest** part?

Preserving admissibility when adding contextual amplifiers. The lower bound has to account for every factor, in its *most favourable* incarnation, or the heuristic silently becomes non-admissible and A\* stops being optimal. I caught this by running the admissibility test after each change — it failed three times before I got the `best_possible_cost_per_meter` method right.

### Q70. What did you **learn**?

1. Classical AI search is beautifully simple — 6 algorithms, 100 lines each, and they cover the full pedagogical spectrum.
2. Admissibility is fragile — easy to break by adding a factor, easy to miss unless you test.
3. Multi-factor cost models are not just academically interesting — they produce qualitatively different routes under different contexts, in ways that match human intuition.
4. **Tests make refactoring safe.** I added three cost factors to v2 and the 22 tests told me immediately what broke.
5. `uv` is a dramatic improvement over pip for day-to-day work.

---

## Appendix — One-liner cheat sheet

- "What's admissible?" → `h(n) ≤ true_cost(n → goal)`, always.
- "Is your heuristic admissible?" → Two are, proven in tests — haversine_admissible and network_relaxed.
- "Why does UCS always win?" → It doesn't; it *ties* with A\* and Weighted A\* (all optimal). Greedy wins on speed.
- "Proof of admissibility?" → reverse Dijkstra ground truth vs h(n), asserted on 200 real-graph nodes.
- "Why multiplicative cost?" → Captures interactions; `safety × gender × time-of-day` stacks correctly.
- "Why OSMnx?" → Brief forbids JSON. Standard Python OSM library. Returns NetworkX graph directly.
- "Why 6 algorithms?" → 3 uninformed + 3 informed = the brief's minimum. Chose the pedagogically informative six.
- "Runtime complexity of A\*?" → O(E log V) worst case.
- "Why Weighted A\* w=1.8?" → Empirical sweet spot: expands fewer nodes than A\*, still hits optimum on our graph because admissible heuristic is loose.
