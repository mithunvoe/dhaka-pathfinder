# Problem Statement — Realistic Pathfinding Simulation for Dhaka City

**Course:** AI Lab
**Instructor:** Prof. Mosaddek
**Student:** Mithun Voe
**Project name:** Dhaka Pathfinder v2.0
**Repository:** https://github.com/mithunvoe/dhaka-pathfinder

---

## 1. Domain & Motivation

Commercial map services (Google Maps, Apple Maps, HERE, OpenStreetMap Routing) choose routes that minimise either **raw distance** or **a fixed estimated-time function**. These default choices hide real decisions that matter to a human traveller in Dhaka:

- Is this road flooded during the monsoon?
- Is this a safe area to walk through at 11 PM as a woman alone?
- Is a rickshaw even allowed on this overpass?
- Does the 6-lane road that's "fastest" become a bottleneck during evening rush?
- Can a child safely take this route, or is the traffic too dangerous?

"Best route" is not a property of the road network alone — it depends on **who** is travelling, **how**, **when**, and **under what conditions**. This project builds a pathfinder that takes all of those factors into account, and compares the behaviour of classical AI search algorithms when they optimise over this realistic cost metric.

---

## 2. Objective

Build a realistic pathfinding simulation over Dhaka's road network that:

1. Loads the live Dhaka road graph from OpenStreetMap via a proper Python OSM package (not exported JSON).
2. Augments every node and edge with plausible synthetic attributes (condition, traffic, safety, risk, lighting, water-logging, crime, lanes, historical incidents).
3. Defines a **multi-factor cost function** that combines road-intrinsic, dynamic (contextual), and traveller-specific factors.
4. Implements **three uninformed** and **three informed** search algorithms — all modified to use the realistic cost function, not raw distance or hop-count.
5. Designs **multiple heuristic functions**, including at least one **provably admissible** and at least two reasonably-effective non-admissible variants.
6. Accepts a user-chosen source, destination, and algorithm, then returns the most preferable, safe, and cost-optimised route.
7. Runs a rigorous comparative analysis over ≥ 100 source/destination pairs across multiple contexts, reporting nodes expanded, path cost, revisits, effective branching factor, search depth, and predicted-vs-actual heuristic gap.
8. Visualises the chosen routes on the Dhaka map.
9. Produces a written performance report summarising findings at both the algorithm level and the heuristic level, highlighting trade-offs between admissibility, speed, and realism.

---

## 3. Required Cost Factors

The realistic cost metric `C_edge` must combine:

**Road-intrinsic factors**
- Base physical length (metres)
- Road condition / surface quality
- Safety indicators
- Risk factors (accident probability, hazards)
- **Street width / number of lanes**

**Dynamic / contextual factors**
- Current traffic level
- Time of day (rush hour, late night, etc.)
- **Weather** (clear / rain / fog / storm / heat)

**Traveller-specific factors**
- Gender (male / female / nonbinary)
- Social context (alone vs accompanied)
- **Age group (adult / child / elderly)**
- Vehicle type (walk / rickshaw / CNG / motorbike / car / bus)

Both the **actual cost** (used as edge weight during traversal) and the **heuristic function** (used for informed search) must be constructed from these factors. The simulation must explicitly track the gap between the heuristic's predicted remaining cost and the actual cost the agent experiences.

---

## 4. Algorithmic Requirements

### 4.1 Algorithm Suite

**Three uninformed** algorithms from: BFS, DFS, UCS, DLS, IDS, Bidirectional Search, Dijkstra.
**Three informed** algorithms from: Best-First, Greedy Best-First, A\*, Weighted A\*, IDA\*, Bidirectional A\*, Beam Search.

**Critical constraint:** all uninformed algorithms must compute path cost under the realistic metric. Using raw hop-count or raw physical length is explicitly disallowed.

### 4.2 Heuristic Requirements

- At least **one heuristic must be admissible** — provably never overestimates the true remaining cost.
- At least **two additional heuristics** should be reasonably effective but may trade admissibility for speed or realism.
- All heuristics must be grounded in real Dhaka conditions: time of day, crowdedness, road condition, gender, vehicle type, past incidents.

---

## 5. User Interaction

- Accept two places as input: a source and a destination (from a named landmark list, or raw latitude/longitude).
- Accept a traveller context: gender, social, age, vehicle, time of day, weather.
- Accept a cost-weight preset: balanced / speed / safety / comfort.
- Accept a choice of algorithm (from the six implemented) OR allow comparing all six in one run.
- Return the resulting route(s) visualised on the Dhaka map.

---

## 6. Comparative Analysis

Run ≥ 100 queries with varied source/destination pairs and produce a results matrix across:

- **Axis A — same context, different algorithm.** Compare all six algorithms on: nodes expanded, path cost, revisits (backtracking proxy), effective branching factor, search depth, runtime.
- **Axis B — same algorithm, different settings.** Vary the heuristic and the traveller context. Report the impact on each metric.

---

## 7. Deliverables

1. Source code — structured Python package with unit + integration tests.
2. Cached Dhaka OSM graph (downloaded once, reused across runs).
3. Interactive web UI for ad-hoc queries.
4. Command-line interface for scripted queries and batch analysis.
5. Flat CSV matrix of all run results (≥ 100 pairs × 6 algorithms × k heuristics × m contexts).
6. Analytic plots (algorithm comparison, heuristic matrix, predicted-vs-actual, context sweep, etc.).
7. Folium HTML maps of example routes across multiple traveller contexts.
8. Written Markdown report summarising results and trade-offs.
9. Complete documentation: a formal problem statement, a teaching guide, and a presentation outline.

---

## 8. Quality Bars

- **Reproducibility:** synthetic data seeded, graph cached on disk, Python environment declared via `pyproject.toml` + `uv.lock`.
- **Testability:** admissibility of admissible heuristics proven against reverse-Dijkstra ground truth on the real graph; UCS and A\* proven to agree on optimum for every random pair tested.
- **Scalability:** synthetic attribute generation must be vectorised — adding the 1,000,001st edge must not cost more than adding the 1st.
- **Architecture:** clean separation of concerns — graph I/O, cost model, heuristics, algorithms, analysis, visualisation, UI each in its own module.

---

## 9. Non-goals

- Real-time traffic data from live APIs (synthetic is fine for the simulation).
- A production-quality routing backend (correctness and clarity > performance).
- Support for cities other than Dhaka (though the architecture admits it — only `config.DHAKA_BBOX` and the landmark list are city-specific).
- Mobile client or deployment (a local Streamlit app suffices).

---

## 10. Success Criteria

The project is considered successful if:

1. Every checkbox in the brief's "Quick Checklist" is implemented AND tested.
2. The admissibility claim holds on the real 28,000-node Dhaka graph for every sampled node.
3. UCS and A\* agree on the optimal cost for every random pair in the test suite.
4. At least 100 source/destination iterations are logged in the results matrix.
5. Changing the traveller context measurably changes the cost of the same edge in predictable ways (female-alone-late-night strictly more expensive than male-midday on every edge).
6. The full pipeline runs end-to-end with a single command (`./run.sh all`).
