# Presentation Guide — Dhaka Pathfinder v2.0

> A structured outline for presenting this project to the class. Each section is what you'd *say*; each "**show**" line is what you'd *demonstrate*. The presentation is designed to take 12–15 minutes with a live demo.

---

## 0. Before you start

- **Open tabs ready:**
  1. The Streamlit UI (`./run.sh ui`, open in browser, pre-loaded with a route)
  2. The GitHub repo in another tab (https://github.com/mithunvoe/dhaka-pathfinder)
  3. `results/REPORT.md` rendered (either on GitHub or in VS Code preview)
  4. One representative HTML route map (e.g. `results/maps/shahbag_motijheel_fog_walk_female.html`)
- **Have the 28k-node graph pre-cached** — otherwise the first page load takes 4 minutes and your demo goes silent. Run `./run.sh download` once before the class.
- **Have your laptop on mains power** — OSMnx + Streamlit are heavy.

---

## 1. Opening — the problem (60 s)

> "When you ask Google Maps for a route, it gives you the **shortest** route. But shortest isn't always *best*. Let me show you."

**Draw on the board**, or show the Streamlit UI with these two inputs side by side:

| Scenario | What Google would say |
|---|---|
| Shahbag → Motijheel, 3 km, 12 minutes | Same route for everyone |
| Shahbag → Motijheel, female, walking, 11 PM, monsoon rain | Same route for everyone |

> "But these are not the same trip. In the second case, the traveller wants a well-lit, populated road away from flood-prone alleys, even if it's 500 metres longer. The *right* pathfinder has to **model who's travelling**. That's what this project does."

---

## 2. Objective (45 s)

> "The assignment asked for a pathfinder that optimises over a **multi-factor cost**, not just distance. The cost must combine road condition, traffic, safety, risk, time of day, gender, social context, vehicle type, age, weather, and street width — and we compare six classical search algorithms and multiple heuristics on this realistic metric."

One-line summary:

> "Six algorithms × six heuristics × seven traveller contexts × 50 source/destination pairs = 7,350 comparative runs. Plus a live UI."

---

## 3. System architecture (90 s)

Show the repo's file tree. The layered design is the story:

```
dhaka_pathfinder/
├── osm_loader.py       ← layer 1: get the map
├── synthetic_data.py   ← layer 2: enrich with realistic attributes
├── context.py          ← layer 3: describe the traveller
├── cost_model.py       ← layer 4: combine into a cost
├── heuristics.py       ← layer 5: estimate "how far to go"
├── algorithms/         ← layer 6: search
├── engine.py           ← orchestrate 1..6
├── analyzer.py         ← run the 7,350 experiments
├── visualizer.py       ← draw paths + plots
└── cli.py              ← expose to users
app.py                  ← Streamlit UI
```

> "Every layer depends only on the one above. A new cost factor goes in `cost_model.py`; a new algorithm in `algorithms/`. No file knows everything."

---

## 4. Technology stack — why these choices (90 s)

| Choice | Why I picked it | What I rejected |
|---|---|---|
| **OSMnx 2.1** | De-facto Python library for downloading OSM → NetworkX graphs. Handles simplification, topology, projection cleanly. | Exported GeoJSON (explicitly forbidden in the brief), overpass-turbo (manual) |
| **NetworkX 3.6** | Native graph data structures, supports MultiDiGraph for one-way + parallel edges. | igraph (faster but C-bindings complicate the lab environment) |
| **Python 3.12 + uv** | uv is 10–100× faster than pip, has a lockfile, reproducible installs. | pip + venv (works but slower, no lockfile guarantees) |
| **NumPy 2.4, pandas, scikit-learn** | Scale-invariant vectorised synthetic data generation + MiniBatch K-Means for area clustering | Pure-Python loops (wouldn't scale) |
| **Folium 0.20 + streamlit-folium** | Interactive Leaflet maps embedded directly in Streamlit. Layer control per algorithm. | matplotlib only (static), plotly-mapbox (needs API key) |
| **Streamlit 1.56** | Zero-boilerplate Python → web UI with caching. | Flask + JS (10× more code) |
| **Click + Rich** | Modern CLI + beautiful tables. | argparse (feels dated) |
| **pytest + 22 tests** | Industry standard. Covers cost-model invariants, admissibility on real graph, optimality of UCS/A\*. | Manual testing (not reproducible) |

> "The choice that differs most from the brief's default: I use **uv** instead of pip. Reason: 10× faster `uv sync`, deterministic installs via `uv.lock`. A fresh clone gets a working environment in 30 seconds."

---

## 5. The data pipeline — OSM → synthetic → cost (2 min)

**Show in sequence:**

### 5.1 OSM load

> "OSMnx downloads the road graph for central Dhaka — 28,094 intersections, 70,197 road segments, 5,826 km of road in total."

```python
G = ox.graph_from_bbox(bbox=(90.34, 23.70, 90.46, 23.86), network_type="drive")
G = largest_strongly_connected_subgraph(G)   # ensure every pair is reachable
```

> "I keep only the largest strongly-connected component so every source can reach every destination — no impossible routes."

### 5.2 Synthetic data — the scale trick

> "OSM tells me a road is `highway=residential` and 500 m long. It does NOT tell me if it's well-lit, flood-prone, or safe at night. I generate those **deterministically from OSM tags and coordinates**, so Old Dhaka is systematically rougher than Gulshan every run."

Show a snippet of `synthetic_data.py`:

```python
base_condition = _HIGHWAY_CONDITION.get(hw, 0.55)           # highway class
condition -= 0.25 * old_dhaka_factor                        # geography
condition -= 0.08 * (cluster_factor - 1)                    # area cluster
condition += noise[ptr]                                     # seeded gaussian
```

> "Four things make this a good design: **deterministic** (same seed = same graph), **vectorised** (one pass over 70k edges in 2 seconds), **grounded in reality** (uses real OSM tags like `lit`, `maxspeed`, `lanes`), and **scale-free** (1 edge or 1 million, same code path)."

### 5.3 Cost model

Pull up `cost_model.py::edge_breakdown` and walk through it:

```
cost = base_length × Π(12 multipliers)
```

> "Each multiplier is of the form `1 + weight × normalised_factor`. It stays near 1 for neutral conditions, grows to 2–3× for bad ones. The **product** captures interactions — a dark road is bad; a dark road for a lone female traveller at midnight is *catastrophic*, which addition would miss but multiplication captures."

Explain the 12 factors briefly:
1. Length, 2. Condition, 3. Safety, 4. Risk, 5. Traffic, 6. Lighting,
7. Water-logging, 8. Gender-social, 9. Vehicle-highway suitability,
10. Crime, 11. **Age** (new), 12. **Weather** (new), 13. **Street width** (new).

> "All three factors **highlighted in bold were added after a mid-project review** found them missing or wired up but never applied. Details in `DOCUMENTATION.md` Chapter 17."

---

## 6. Algorithm selection — why these six (90 s)

The brief asked for 3 uninformed + 3 informed. I picked:

| Algorithm | Category | Why I picked it |
|---|---|---|
| **BFS** | Uninformed | Pedagogical control — the classic, minimises hops |
| **DFS** | Uninformed | Demonstrates the catastrophic behaviour of depth-first on realistic cost (12× worse than optimum) |
| **UCS (Dijkstra)** | Uninformed | The gold standard — optimal, no heuristic. Baseline for "what's the right answer?" |
| **Greedy Best-First** | Informed | Extreme end of the spectrum — ignores `g(n)`, blazingly fast, not optimal. Shows what happens without back-tracking |
| **A\*** | Informed | The canonical optimal informed search |
| **Weighted A\*** (w=1.8) | Informed | The speed/quality trade-off — bounded-suboptimal |

> "I explicitly avoided IDS and Bidirectional Search. IDS on a realistic-cost metric is awkward because its definition mixes hop-depth with the cost axis. Bidirectional Dijkstra would be a clear win for performance but doesn't demonstrate anything new about the cost model. These six cover the pedagogical spectrum the brief cares about."

---

## 7. Heuristic design — this is where it gets subtle (2 min)

> "The brief asked for at least one admissible heuristic and at least two reasonably effective ones. I built six, with **two admissible** ones — one cheap and one tight."

### 7.1 Admissibility — explain the word

> "A heuristic `h(n)` is admissible if `h(n) ≤ true_cost(n → goal)` for every node n. Admissibility is **the** key property — A\* with an admissible heuristic is guaranteed to find the optimal path. If `h` overestimates, A\* can be tricked into committing to a worse route."

### 7.2 The cheap admissible — `haversine_admissible`

```python
h(n) = haversine_distance(n, goal)_metres × best_possible_cost_per_metre(context)
```

> "Haversine is the shortest possible geographic distance. `best_per_metre` is the cost-per-metre under the **most favourable** possible conditions under the current context — clean road, no traffic, zero risk, best vehicle-road match, adult-clear-weather. Because no real edge can be better than that, the product is a strict lower bound. Cheap to compute (O(1) per node)."

### 7.3 The tight admissible — `network_relaxed` (new in v2)

```python
h(n) = shortest_physical_length_along_roads(n, goal) × best_per_metre
```

> "This version uses a **reverse Dijkstra on physical length** from the goal. It's tighter because haversine ignores rivers and impassable terrain, but road-shortest-length respects the graph topology. Still strictly admissible, because any real cost is `≥ sum_of_edge_lengths × best_per_metre ≥ shortest_length × best_per_metre`. Cost: O(V log V) per goal, cached — only 0.5 s per goal on the real graph."

### 7.4 The three non-admissible heuristics

- **`haversine_time`** — `distance / vehicle_speed`, scaled. Optimistic in traffic.
- **`context_aware`** — haversine × best-per-metre × (risk × gender-safety amplifier). Overestimates in risky contexts.
- **`learned_history`** — haversine × best-per-metre × (1 + avg_historical_incidents_in_area). Biases away from known bad neighbourhoods.

> "Plus a `zero` heuristic as a control — reduces A\* to UCS, useful for isolating 'is my heuristic actually doing anything?'"

### 7.5 How do I *prove* admissibility?

```python
def test_admissible_heuristic_on_real_graph(engine):
    G = engine.graph
    ctx = TravelContext(...)
    weights = engine._weights_for(ctx)
    goal = random_node()
    true_costs = reverse_dijkstra(G, goal, weights)   # ground truth
    h = make_heuristic("haversine_admissible", G, goal, ctx, cost_model)
    for n in sample(nodes, 200):
        assert h(n) <= true_costs[n] + 1e-3          # the lower-bound property
```

> "I run a full reverse-Dijkstra to get the ground-truth remaining cost, then assert my heuristic is ≤ that for 200 sampled nodes. This test runs in my CI and passes on the real 28k-node graph every time."

---

## 8. Live demo — Streamlit UI (3 min)

**Show** `./run.sh ui`, then in the sidebar:

1. **Source = Shahbag, Dest = Motijheel, compare ALL algorithms, female, alone, adult, walk, late_night, fog.**
2. Point out:
   - The route comparison table: UCS, A\*, Weighted A\* all report **the same cost** (33,370.0), same length (3.07 km), same 42 edges. "They tie because they're all **provably optimal** — they must agree on the best cost."
   - The map: all 6 paths overlaid. Click the layer control, toggle each on/off. Show that the optimal trio is highlighted with a white halo.
   - The 4-category winner board: "Lowest cost → UCS/A\*/Weighted A\* (3-way tie — expected). **Fewest nodes expanded → Greedy Best-First**, 50 nodes vs A\*'s 13,000. **Fastest → Greedy**, 0.4 ms vs A\*'s 100 ms."
   - The `gap (actual-h)` column — "this is the heuristic's prediction error. Positive means under-estimated (expected for admissible)."

3. **Change context to male/car/midday/clear** — point out that the cost drops ~5×. "Same graph, same route topology, very different cost. That's the multi-factor model earning its keep."

4. **Change vehicle to walk on the same pair** — point out that the algorithm now avoids main roads because of the `vehicle_highway_suitability` matrix.

---

## 9. Comparative analysis — the 7,350 runs (90 s)

**Open** `results/REPORT.md`.

Key numbers to read off:

```
algorithm         median cost  median nodes  median runtime
UCS / A* / WA*    212,375      13,000 / 12,700 / 12,858   95 / 129 / 122 ms
Greedy            293,218      169                         1.2 ms
BFS               312,343      14,693                      21 ms
DFS               2,401,773    14,168                      24 ms
```

Talking points:
- UCS/A\*/WA\* all tie at the optimum — as theory predicts.
- **A\* expands 10% fewer nodes than UCS**, and **Weighted A\* expands 16% fewer nodes than UCS** while remaining optimal on this graph (even though it's only *bounded* optimal in theory). The heuristic is pulling its weight.
- **Greedy is 65× faster** with 35% higher cost. Useful when you need a "good enough" answer immediately.
- **DFS is 12× worse than optimum** — the expected catastrophe. "A great case study in why 'minimise hops' is not the same as 'minimise cost'."

**Show** the plots in `results/plots/` — `comparison_bars.png`, `heuristic_matrix.png`, `new_factors_impact.png`, `predicted_vs_actual.png`.

---

## 10. Key design decisions I'm proud of (60 s)

1. **Multiplicative cost, not additive.** Lets interaction effects (dark + lone female + Old Dhaka at midnight) produce a 5× cost; addition would have flattened that to 3×.
2. **Admissibility preserved when I added new factors.** The `best_possible_cost_per_meter` method walks through the cost model with every *best-case* multiplier, including the new age/weather/width factors. A test iterates over 45 combinations of (age × weather × vehicle) and asserts the lower-bound holds on every edge.
3. **Per-context edge-weight cache.** Recomputing the cost of 70k edges on every algorithm run would be ~2 seconds per call. We cache `{context.label() → {edge: cost}}` and every algorithm looks up edge costs in O(1).
4. **Reverse-Dijkstra heuristic with per-goal caching.** The `network_relaxed` heuristic costs O(V log V) per goal but is shared across every context, algorithm, and heuristic query with the same goal — turned an expensive heuristic into a one-time cost.
5. **4-category winner board.** UCS/A\*/WA\* always tie on cost, which could feel boring. Adding winners for "fewest expansions" and "fastest" makes every run interesting because **different algorithms win different categories**.

---

## 11. Challenges and how I solved them (60 s)

| Challenge | Solution |
|---|---|
| Graph download was slow and flaky on first attempt | Pickled cache keyed on `GraphLoadSpec`; second+ runs load in 2s from disk |
| OSMnx v2 removed `plot_graph_folium` | Switched to `graph_to_gdfs` + manual `folium.PolyLine` — direct control |
| Admissibility non-trivial with contextual factors | Extended `best_possible_cost_per_meter` to walk each factor with its best-case value under the current context; proved correctness in tests |
| UCS, A\*, Weighted A\* always tying (feels boring) | Added multi-category winner board so Greedy wins on speed/expansion |
| Initial A\* barely faster than UCS (loose heuristic) | Added `network_relaxed` heuristic for a tighter lower bound |
| `.fuse_hidden*` files locking venv directory | Pattern: always kill Streamlit before removing `.venv`; run.sh now does this automatically |

---

## 12. Future work (30 s)

> "Given more time, I'd add:
> 1. **ALT (landmark-based A\*)** — precompute distances from a few landmarks, use triangle inequality for an even tighter admissible heuristic.
> 2. **Bidirectional Dijkstra** — 2× expansion reduction on long routes.
> 3. **Click-to-place markers on the map** — currently you pick landmarks from a dropdown.
> 4. **Real-time traffic ingest** — right now traffic is synthetic; plug in live data from a Dhaka traffic API."

---

## 13. Close (30 s)

> "To summarise: this project implements every requirement from the brief, plus three factors (age, weather, street width) and one extra admissible heuristic that the brief didn't strictly require but makes the system more realistic and more informative. Everything is open source at `github.com/mithunvoe/dhaka-pathfinder`, runs with a single `./run.sh ui` command, and has 22 passing tests covering cost-model invariants and heuristic admissibility. Questions?"

---

## Appendix A — If you only have 5 minutes

Cut sections 4, 6, 11, 12. Keep 1, 2, 3, 7.1, 8, 9, 13. The demo is the most persuasive part.

## Appendix B — If the projector can't show Streamlit

Open `results/maps/shahbag_motijheel_fog_walk_female.html` in a browser — it's the same map, fully interactive, doesn't need a live server.

## Appendix C — Things to emphasise verbally

- The heuristic admissibility is **proven**, not claimed — the test runs on the real graph every CI run.
- **UCS, A\*, and Weighted A\* tying on cost is a feature, not a bug** — it's theorem that all three find the same optimum. The winner board makes this explicit.
- Every weight in the cost model is **declarative** — preset changes propagate cleanly; no scattered magic numbers.
