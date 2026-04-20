# Dhaka Pathfinder — Realistic Multi-Factor Routing Simulation

A context-aware pathfinding engine for Dhaka that goes beyond "shortest distance." It combines the real road graph (OpenStreetMap, via OSMnx) with a **multi-factor cost model** that captures the feel of actually moving around Dhaka — traffic, road condition, safety, risk, lighting, water-logging, crime, gender-specific safety, social context, vehicle-type suitability, and time-of-day.

It implements **six search algorithms** — three uninformed, three informed — all operating on the realistic cost, and **five heuristics** of varying admissibility and realism.

---

## Highlights

- **Real OSM graph:** 28,000+ intersections, 70,000+ road segments covering central Dhaka (via OSMnx v2.1).
- **Scale-safe synthetic data generator:** populates every edge with plausible attributes derived deterministically from OSM tags + coordinates. One node or a million — same code path, same runtime per edge.
- **Twelve-factor cost model** covering every dimension of the assignment — length, road condition, safety, risk, traffic, time-of-day, lighting, water-logging, crime, gender & social context, vehicle/highway suitability, **age (adult/child/elderly)**, **weather (clear/rain/fog/storm/heat)**, and **street width / lanes**.
- **Six search algorithms — all modified to use the realistic cost metric**:
  - Uninformed: **BFS**, **DFS**, **UCS (Dijkstra)**
  - Informed: **Greedy Best-First**, **A\\***, **Weighted A\\***
- **Six heuristics**, including **two provably admissible** variants (`haversine_admissible`, `network_relaxed`) — both proven by test to satisfy `h(n) ≤ true_cost(n → goal)` on the real 28k-node Dhaka graph — plus four reasonably-effective context-aware variants.
- **Interactive Streamlit UI** with a live Folium map, layered route overlays for algorithm comparison, and a clear cost breakdown — now exposes age, weather, and street-width controls.
- **CLI**: `download`, `route`, `compare`, `synth-stats`, `landmarks`.
- **Iteration comparative analyzer** producing a flat CSV matrix across algorithm × heuristic × context (7 traveller contexts including rain/fog/child/elderly) plus 5 analytic plots and a Markdown report.
- **22 unit + integration tests** covering cost-model invariants, admissibility of both admissible heuristics, optimality of UCS and A\*, and context-sensitivity (child > adult risk, rain > clear cost, wide lanes cheaper for car, wider lanes costlier for walkers, etc.). All passing.

---

## Quickstart

Requires [**uv**](https://docs.astral.sh/uv/) (Astral's fast Python package manager). Install once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
# or:  brew install uv
```

Then:

```bash
# 1. install all deps (creates .venv from pyproject.toml / uv.lock — ~30 s first run)
uv sync

# 2. pre-cache the Dhaka OSM graph (one-time download, ~2 min)
uv run python -m dhaka_pathfinder.cli download

# 3a. interactive web UI
uv run streamlit run app.py

# 3b. or the CLI — find one route
uv run python -m dhaka_pathfinder.cli route \
    --source "Shahbag" --dest "Gulshan 2" \
    --algorithm astar --heuristic network_relaxed \
    --gender female --social alone --age adult --vehicle cng \
    --time-of-day evening_rush --weather rain --compare-all

# 3c. comparative analysis (100 pairs, all algorithms, all heuristics, all contexts)
uv run python scripts/run_comparison.py
uv run python scripts/generate_report.py
```

Or use the convenience wrapper `./run.sh`:

```bash
./run.sh ui                                      # Streamlit UI
./run.sh route --source "Shahbag" --dest "Banani" --compare-all
./run.sh compare && ./run.sh report              # full analysis
./run.sh test                                    # run all 22 tests
./run.sh all                                     # end-to-end regeneration
```

The Streamlit app exposes:
- Source + destination as dropdown landmarks or free lat/lon entry
- Algorithm picker (or **compare all 6** overlaid on one map)
- Heuristic picker with live admissibility tag
- Gender / social / vehicle / time-of-day / weight-preset controls
- Live cost-vs-actual table, legend, map with fullscreen + mini-map

---

## Problem mapping (what satisfies what)

| Assignment requirement | Where it lives |
|---|---|
| Dhaka OSM map via a Python OSM package | `dhaka_pathfinder/osm_loader.py` — OSMnx 2.1, cached as pickle |
| Nodes = intersections / landmarks | Native OSMnx output; landmarks listed in `config.LANDMARKS` |
| Edges = road segments | Native OSMnx multigraph edges |
| Scalable synthetic data generator | `synthetic_data.py` — seeded NumPy, vectorised, scale-invariant; now also reads `lanes` OSM tag and emits `num_lanes` / `street_width_m` |
| Multi-factor cost function | `cost_model.py`, `context.py`, `config.py` — 12 factors: length / condition / safety / risk / traffic / time / lighting / water-logging / crime / gender-social / vehicle-suitability / **age** / **weather** / **street width** |
| Traffic factor | `traffic_base` (synthetic) × time-of-day amp × weather amp × age amp × stochastic noise |
| Gender factor | `GENDER_SAFETY_MULTIPLIER` (`config.py`) — alone vs accompanied, 3 genders |
| Adult / children / elderly factor | `AGE_PROFILE` (`config.py`) — risk_amp, traffic_amp, crime_amp, wide_road_penalty + vehicle restrictions (children cannot ride motorbikes) |
| Vehicle type factor | `VEHICLE_HIGHWAY_SUITABILITY` matrix — 6 vehicles × 12 highway classes |
| Weather factor | `WEATHER_PROFILE` (`config.py`) — clear / rain / fog / storm / heat, amplifies water-logging / lighting / risk / traffic / condition |
| Street width factor | `num_lanes` from OSM (with road-class fallback); cars prefer wide, walkers/rickshaws penalised on wide roads, child-walkers doubly penalised |
| 3 uninformed search algorithms | `algorithms/uninformed.py` — BFS, DFS, UCS |
| Uninformed algorithms use realistic cost, not raw hops | Every algorithm queries the precomputed realistic-cost cache |
| 3 informed search algorithms | `algorithms/informed.py` — Greedy, A\*, Weighted A\* |
| Multiple heuristics (≥ 1 admissible, ≥ 2 reasonable) | `heuristics.py` — **6 variants**. Two admissible: `haversine_admissible`, `network_relaxed` (reverse Dijkstra on length × best-per-metre). Three reasonable: `haversine_time`, `context_aware`, `learned_history`. One control: `zero`. |
| User input (source, destination, algorithm) | Streamlit sidebar + CLI `route` command |
| Predicted-vs-actual cost tracking | `SearchStats.predicted_cost_at_start` / `predicted_vs_actual_gap` |
| ≥ 100 iterations of comparison | `analyzer.py`, `scripts/run_comparison.py` (default 100 pairs) |
| Metrics: nodes expanded, cost, backtracking, EBF, depth | `SearchStats` fields; `effective_branching_factor` computed by Newton solve |
| Results matrix (algorithm × heuristic × context) | `results/comparison_matrix.csv` |
| Path visualization on Dhaka map | `visualizer.build_route_map` → Folium HTML with layer control |
| Written performance report | `scripts/generate_report.py` → `results/REPORT.md` |

---

## Multi-factor cost model

For each edge, actual cost is:

```
cost = base_length × w_length × Π(multipliers)
```

where each multiplier ≥ 1 for unfavourable conditions, ≈ 1 for neutral, < 1 only when the traveller + road combination is particularly favourable:

```
condition_mult    = 1 + w_rc × (1 − condition / weather_cond_amp)
traffic_mult      = 1 + w_tr × traffic_base × time_amp × weather_amp × age_amp
risk_mult         = 1 + w_rk × risk         × time_amp × weather_amp × age_amp
safety_mult       = 1 + w_sf × (1 − effective_safety)
lighting_mult     = 1 + w_lt × (1 − effective_lighting / weather_light_amp)
water_log_mult    = 1 + w_wl × water_logging_prob × weather_water_amp
gender_mult       = 1 + w_gs × (gender_social_multiplier − 1)
vehicle_mult      = 1 + w_vs × (1 − vehicle_highway_suitability)
crime_mult        = 1 + w_cr × crime_index × gender_factor × age_crime_amp
age_mult          = 1 + w_ag × (age_risk_bonus)   +  2.5× if vehicle forbidden for age
weather_mult      = 1 + w_we × weather_direct_penalty
street_width_mult = vehicle-dependent — cars get a discount on wide roads, walkers
                    get a penalty, child-walkers get an extra penalty
```

**Admissible lower bound:** `best_possible_cost_per_meter(context)` multiplies every term by its most favourable value achievable under the current context (best condition, zero traffic, no risk, best vehicle/road pairing, best age — i.e. adult, best weather — i.e. clear, best lane count for the current vehicle). Because it uses the active context's amplifiers everywhere they can only grow the lower bound, the result is STRICTLY `≤ true_cost(any_edge)`. Two heuristics use this quantity:

1. `haversine_admissible` = `haversine_distance(n, goal) × best_per_metre` — the classical admissible lower bound.
2. `network_relaxed` = `reverse_Dijkstra_on_length(n, goal) × best_per_metre` — tighter because it respects the road-network topology instead of ignoring it.

Both admissibility claims are verified in `tests/test_heuristics.py` and `tests/test_integration.py` against a full reverse-Dijkstra ground truth on the real 28k-node Dhaka graph.

Four more heuristics (`zero`, `haversine_time`, `context_aware`, `learned_history`) trade admissibility for speed or realism — the report quantifies the trade-off.

---

## Repository layout

```
dhaka_pathfinder/
├── __init__.py
├── config.py              # paths, weights, landmarks, registries
├── context.py             # TravelContext — gender / social / vehicle / time
├── cost_model.py          # RealisticCostModel, cost breakdown, haversine
├── osm_loader.py          # OSMnx wrapper with pickle cache
├── synthetic_data.py      # vectorised synthetic attribute generator
├── heuristics.py          # 5 heuristic factories
├── engine.py              # high-level orchestration + edge-weight cache
├── visualizer.py          # Folium maps, comparison/heuristic/context plots
├── analyzer.py            # 100+ iteration comparison runner
├── cli.py                 # click-based CLI
└── algorithms/
    ├── base.py            # SearchStats, timer, EBF, path_cost helpers
    ├── uninformed.py      # BFS, DFS, UCS (all use realistic cost)
    └── informed.py        # Greedy, A*, Weighted A*

app.py                     # Streamlit web UI
scripts/
├── run_comparison.py
└── generate_report.py

tests/
├── conftest.py            # toy graph fixture
├── test_cost_model.py
├── test_heuristics.py     # includes admissibility proof
└── test_algorithms.py     # optimality check UCS/A*

results/
├── comparison_matrix.csv  # flat run matrix (9k rows)
├── algorithm_summary.csv
├── heuristic_summary.csv
├── context_summary.csv
├── plots/*.png            # 5 plots
├── maps/*.html            # interactive route maps
└── REPORT.md              # final written report
```

---

## Tests

```bash
uv run pytest tests/ -v
# or: ./run.sh test
```

- Heuristic admissibility is asserted against ground-truth single-source Dijkstra.
- UCS and A\* are verified to produce equal path cost = ground truth on a controlled toy graph.
- Context switch (male/accompanied/midday vs female/alone/late-night) produces strictly higher cost on the same edge.

---

## Extensions already implemented (beyond the core spec)

- **Age factor** (adult / child / elderly) with vehicle restrictions (children cannot ride motorbikes) and a wide-road penalty multiplier.
- **Weather factor** (clear / rain / fog / storm / heat) that amplifies water-logging, lighting, risk, traffic, and condition terms.
- **Street width** from the OSM `lanes` tag — cars get a discount on wide multi-lane roads; walkers (especially children) get a penalty on wide crossings.
- **Lighting**, **water-logging**, **crime index** as first-class cost components (spec asked for "safety/risk" broadly).
- **Area-level safety profile** — Old Dhaka / Gulshan / Uttara / Mirpur / etc. are distinguished automatically from geographic bucketing.
- **Weight presets** (`balanced`, `safety`, `speed`, `comfort`) so the same route can be recomputed under different priorities.
- **Streamlit UI** with interactive map overlay and layer control per algorithm — now also age and weather controls.
- **Effective branching factor** is solved via Newton iteration against the real expansion/depth — not approximated.
- **Historical incident** synthesis on every edge, feeding the `learned_history` heuristic.
- **6 heuristics** instead of 3 — including TWO admissible variants (`haversine_admissible` is a fast O(1) lower bound; `network_relaxed` is a tighter O(V log V) reverse-Dijkstra lower bound).
- **Network-aware admissible heuristic** (`network_relaxed`) — provably admissible, strictly tighter than haversine, cached per goal so it amortises across contexts.

---

## License

Academic / educational use — part of the AI Lab course under Prof. Mosaddek.
