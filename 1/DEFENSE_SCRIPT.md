# Defending the Project — Script for Teacher Q&A

> For the 1-on-1 "show me your project and explain how it works" session.
> This is a *script*. Say the **quoted** lines aloud; treat the *italic* lines
> as stage directions. Total runtime ~6 minutes for the walkthrough + as long
> as the teacher wants to probe.

---

## BEFORE YOU START — 30-second setup

Open in separate browser tabs, in this order:

1. `./run.sh ui` → Streamlit at `http://localhost:8501` (pre-loaded with Shahbag → Gulshan 2)
2. `results/maps/scenario_1_gender_time.html` (overlay map — gender/time effect)
3. `results/maps/scenario_2_storm_detour.html` (overlay map — weather effect)
4. GitHub repo: `https://github.com/mithunvoe/dhaka-pathfinder`
5. `results/REPORT.md` rendered (open in VS Code preview or on GitHub)

Have a terminal open with the repo as working directory, in case the teacher says *"run the tests"*.

Verify the Dhaka graph is cached — if `./run.sh download` has been run today, page load is 2 seconds; otherwise 3-4 minutes.

---

## PART 1 — The 90-second opening (what the project IS)

> "Sir, this is my AI Lab project — a pathfinding system for Dhaka that goes beyond distance or time. The idea is: the **best** route depends on **who** is travelling and **when**. A woman walking alone at 11 PM during monsoon rain needs a different route than a man in a car at noon — even between the same two places. Classical Dijkstra or Google Maps would give both the same route. My system doesn't."

*Show the Streamlit UI. Point at the sidebar.*

> "On the left I pick source and destination, the algorithm, the heuristic, and a **traveller context** — gender, social, age, vehicle, time of day, weather. On the right I see the route drawn on a real OpenStreetMap graph of central Dhaka — 28,094 intersections, 70,197 road segments, 5,826 km of road — all downloaded live through OSMnx."

*Click "🚀 Compute route" if a route isn't already drawn.*

> "Here are all six algorithms run at once on the same pair. UCS, A\*, and Weighted A\* all tie at the optimal cost 33,370 because they're all provably optimal. Greedy and BFS land about 30% worse. DFS goes catastrophically wrong — it's the textbook example of what you shouldn't do with a weighted graph."

---

## PART 2 — The 3-minute walkthrough (how it WORKS)

*Teacher usually asks "explain how it works". Here's the order.*

### 2.1 Show the architecture

> "Structurally the project is seven layers, each a separate Python file."

*Point at the GitHub tab showing the file tree.*

> "Layer 1 — `osm_loader.py` — downloads the Dhaka road network from OpenStreetMap using OSMnx version 2.1. I explicitly chose OSMnx because the brief bans JSON exports. The raw graph is a NetworkX MultiDiGraph with directed edges for one-way streets.
>
> Layer 2 — `synthetic_data.py` — because OSM doesn't tell us road condition or crime or flood risk, I generate those deterministically from OSM tags plus coordinates. Seeded, vectorised, scale-invariant.
>
> Layer 3 — `context.py` — the traveller. A frozen dataclass.
>
> Layer 4 — `cost_model.py` — the heart of the project. Combines the road's own numbers with the traveller's context into a single edge cost.
>
> Layer 5 — `heuristics.py` — the h(n) function for informed search.
>
> Layer 6 — `algorithms/` — three uninformed and three informed search algorithms.
>
> Layer 7 — `visualizer.py` — Folium maps and Matplotlib plots."

### 2.2 Explain the cost function (this is where teacher often zooms in)

*Show the "🔬 How the cost is computed" expander in the Streamlit page. Or pull up `cost_model.py::edge_breakdown`.*

> "Every edge cost follows one pattern:
>
> &nbsp;&nbsp;&nbsp;&nbsp;**cost = length × w_length × product of 12 multipliers**
>
> and every multiplier is of the form **1 + weight × factor**. The `1 +` is key — it means neutral conditions give multiplier 1, and only bad conditions add to it. So the cost can only grow, never shrink below the raw length.
>
> The 12 factors cover everything the brief asked for: road condition, traffic, safety, risk, lighting, water-logging, crime, gender × social context, vehicle-highway suitability, age group, weather, and street width.
>
> Why **multiplicative** instead of additive? Because of **interactions**. A dark road is bad. Being a lone female at midnight is a penalty. Old Dhaka has high crime. Add them up and you get a +3 penalty; multiply them and you get **4.3×**. Multiplication captures the fact that the combination is actually dangerous — not just summed-up inconvenient."

### 2.3 Explain the algorithms

*If teacher didn't push into specifics, pick UCS + A\*.*

> "Each algorithm answers two questions: which node do I expand next, and how do I track the cheapest path so far.
>
> **UCS**, or Dijkstra — expands the node with the smallest accumulated cost g(n). Priority queue. Always optimal.
>
> **BFS** — FIFO queue. Expands by hops, ignores cost. I report cost under the realistic metric anyway, as the brief requires — on our graph BFS is 30% worse than optimal.
>
> **DFS** — LIFO stack. Goes deep, backtracks. Catastrophic on weighted graphs — median 12× worse than optimum.
>
> **Greedy Best-First** — priority queue keyed by h(n) alone, ignores g. Fastest, not optimal.
>
> **A\*** — combines both: f(n) = g(n) + h(n). Optimal if h is admissible.
>
> **Weighted A\*** — f(n) = g(n) + w·h(n), with w > 1. Trades guaranteed-optimality for speed. I use w = 1.8."

### 2.4 Explain the heuristic — this is the make-or-break moment

*Open `heuristics.py::make_network_relaxed` on GitHub, or pull up HEURISTIC.md.*

> "My primary heuristic is called `network_relaxed`. One formula:
>
> &nbsp;&nbsp;&nbsp;&nbsp;**h(n) = shortest_road_length(n, goal) × best_possible_cost_per_metre(context)**
>
> The first factor is a **reverse Dijkstra** from the goal on physical edge length. It gives the shortest possible road distance from every node to the goal — respecting the real network topology, not drawing straight lines through rivers.
>
> The second factor is a scalar computed once per context. It walks the full 12-factor cost model with every factor set to its **most favourable** value under the active traveller — adult instead of child, clear weather instead of storm, best-match vehicle-road class, and so on. So it's the minimum possible cost per metre.
>
> Multiplied, the product is a strict **lower bound** on the true remaining cost. That's the definition of admissibility."

*If teacher asks "prove it".*

> "For any real path P from n to the goal:
>
> &nbsp;&nbsp;&nbsp;&nbsp;true_cost(P) = Σ length(e) × w_length × Π(multipliers(e, ctx))
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;≥ Σ length(e) × best_per_m(ctx)          &nbsp;&nbsp;&nbsp;&nbsp;[each edge ≥ its floor]
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;≥ shortest_length(n, goal) × best_per_m(ctx) &nbsp;&nbsp;&nbsp;&nbsp;[shortest ≤ any other path's length]
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= h(n)
>
> And I verify it empirically in `tests/test_integration.py`. It runs a full reverse-Dijkstra for ground truth and asserts h(n) ≤ truth at 200 random nodes on the real 28,000-node graph."

*If teacher asks why multiple heuristics.*

> "The brief's Axis B requires heuristic comparison, so I have six — two admissible, three non-admissible but realistic, plus `zero` as a control. But my single headline heuristic is `network_relaxed`. The others are in the comparison section of the report."

### 2.5 The killer demo — show context actually changes the route

*Switch to the `scenario_1_gender_time.html` tab.*

> "This demonstrates that the cost model isn't cosmetic. Same source — Shahbag. Same destination — Gulshan 2. Same algorithm — A\*. The **green** route is an adult male in a car at midday; the **red** route is an adult female walking alone at late night.
>
> Only **14 intersections** out of ~80 are on both routes. The night-walker is routed through different streets because the algorithm now heavily penalises dark, high-traffic, pedestrian-hostile edges."

*Switch to `scenario_2_storm_detour.html`.*

> "This is even more dramatic. Same driver, same car, same time of day — **only the weather changes**. Clear weather, blue route goes straight through. Storm weather, purple route detours east to avoid flood-prone streets near Old Dhaka. Only **4 shared intersections** out of ~340. The water-logging multiplier spikes 3× under storm — the algorithm picks a completely different road."

---

## PART 3 — Deep questions the teacher is likely to ask

### Q: "Why did you pick these six algorithms?"

> "Three uninformed plus three informed, as required. I avoided IDS and IDA\* — on a continuous realistic-cost metric they don't teach anything new. Bidirectional Dijkstra would be a performance win but it's a variant, not a different pedagogical lesson. These six cover the full spectrum: BFS minimises hops, DFS is catastrophic, UCS is the optimal baseline, Greedy is the speed extreme, A\* is the classical optimal informed, Weighted A\* is the speed-vs-quality knob."

### Q: "What is admissibility, and why does it matter?"

> "A heuristic h is admissible if h(n) ≤ true remaining cost from n to the goal, for every n. It's a lower-bound property. A\* with an admissible heuristic is guaranteed to return the optimal path. If h overestimates, A\* can be fooled into skipping the real best path because it thinks that direction is too expensive."

### Q: "Prove A\* optimality."

> "When A\* pops the goal from the frontier, f(goal) = g(goal) because h(goal) = 0. Any alternative path P' to the goal sitting elsewhere in the frontier has f(P') ≥ true_cost(P') by admissibility, and any such f(P') ≥ f(goal) at this moment because the priority queue pops the minimum f. So true_cost(P') ≥ g(goal), which means no alternative path is cheaper than the one A\* just returned."

### Q: "What if my heuristic is tight but not admissible?"

> "You lose the optimality guarantee but often gain speed. My `context_aware` heuristic is deliberately non-admissible — it amplifies the lower bound by the risk and gender-safety multipliers. It sometimes returns sub-optimal paths but expands fewer nodes. That's why I include both variants."

### Q: "What is consistency?"

> "Stronger than admissibility: h(u) ≤ edge_cost(u, v) + h(v) for every edge. Consistency implies admissibility. With a consistent heuristic, A\* never needs to reopen a closed node — once we close it we know the optimal g. Both of my admissible heuristics are also consistent: haversine and road-shortest-path both satisfy the triangle inequality."

### Q: "What's the time complexity?"

> "Per algorithm: BFS and DFS are O(V + E). UCS and A\* are O(E log V) with a binary heap. Greedy is O(E log V) worst case but usually far less because it expands tiny fractions of the graph. Space for all of them is O(V) because they all need a visited/g-score dict. My graph is V = 28,094, E = 70,197 — so the numbers are small, one query takes 10 to 130 milliseconds depending on the algorithm."

### Q: "Why this specific cost-function design?"

> "Multiplicative because factors interact — I mentioned the dark-night-lone-female-in-Old-Dhaka example. Each multiplier uses `1 + weight × factor` so neutral input is neutral output. I keep weights separate from per-road numbers so I can tune globally — there are four presets: balanced, safety, speed, comfort. Every weight is declared in `config.py`, no magic numbers scattered in the code."

### Q: "How do you know the synthetic data is realistic?"

> "It's grounded in real OSM tags — highway class, `lit`, `maxspeed`, `lanes`. Plus geographic rules — Old Dhaka systematically rougher, Gulshan systematically safer. Plus seeded noise so individual edges vary but areas stay coherent. It's not real measured data — production would need Dhaka traffic and crime datasets — but it's internally consistent, reproducible, and scales to any graph size."

### Q: "Why OSMnx and not just a saved file?"

> "The brief specifically bans JSON exports — you said using a pre-exported graph is bad practice. OSMnx gives me live, editable, version-controlled OSM data as a standard NetworkX graph. I cache the download as a pickle so subsequent runs are instant, but the pickle is regenerated any time the bounding box or network type changes."

### Q: "Why the `num_lanes` and `street_width_m` columns?"

> "The brief asked for street width as a cost factor. I read the OSM `lanes` tag — 90% of edges have it — and fall back to a highway-class default otherwise. Cars get a discount on wide multi-lane roads; walkers, especially children, get a penalty on wide crossings."

### Q: "Why does UCS always agree with A\* on cost?"

> "Both optimise the same objective — minimum sum of edge costs — over the same constraint set — paths from source to destination. Same min over the same set means same value. Different paths can instance the same minimum when there are ties, but the cost is always equal."

### Q: "What about the `predicted_vs_actual_gap` column?"

> "It's actual_cost minus the heuristic's estimate at the source. Positive means the heuristic under-estimated, negative means it over-estimated. For my admissible heuristic the gap is always positive on every run. The mean gap is about 160,000 against an actual of 167,000 — which means my admissible heuristic is capturing only ~5% of the true cost. That's admissible but loose. It's why A\* only expands 6% fewer nodes than UCS — there's room to improve with a tighter heuristic like ALT."

### Q: "Can you show me the tests?"

*Open a terminal.*

```bash
./run.sh test
```

> "22 tests, 11 seconds, all passing. They cover cost-model invariants — female-alone-late-night strictly more expensive than male-midday on every edge; admissibility across 45 combinations of (age × weather × vehicle); UCS and A\* agreement on random pairs; the integration test sampling 200 real graph nodes for admissibility."

### Q: "How would you scale this to all of Bangladesh?"

> "Three changes. Increase the bbox in `config.py` — the rest of the pipeline is scale-invariant. At 500,000-plus nodes I'd swap the pickle cache for PostGIS and switch the heuristic to ALT or bidirectional Dijkstra. The synthetic-data generator already vectorises — it doesn't care about graph size."

### Q: "What's the limit of the current cost model?"

> "No real-time traffic. Weights are hand-picked instead of learned. Only central Dhaka. No multi-modal routing — you can't chain walk + bus + walk. These are listed as future work in `PRESENTATION.md`."

### Q: "If I change one weight, what happens?"

> "Global effect. Every edge in the graph gets re-evaluated with that weight's factor recomputed. For example, changing `w_safety` from 1.2 to 2.2 triples the per-edge safety penalty — risky paths become systematically more expensive, and the algorithm picks safer-but-longer routes. The UI has four presets that bundle related weights together for convenience."

---

## PART 4 — If something goes wrong

### Streamlit won't load

> "Sir, while it's loading, let me show you the pre-rendered HTML maps."

*Switch to the `scenario_*.html` tabs and use those instead.*

### A test fails live

> "Interesting — let me check." *Pull up the test file.* "Ah, that's [explain]." Don't panic.

### Teacher asks about something I haven't prepared

> "Good question — let me show you the code and we'll read through it together."

*Opening the actual `.py` file and reading it aloud is always better than guessing. The code is the authoritative source.*

---

## PART 5 — Closing

> "So in summary, sir: every factor from the brief — traffic, gender, age, vehicle, weather, street width — is implemented AND tested. The heuristic is provably admissible with a proof that runs against the real 28,000-node graph. The 7,350-run comparative analysis confirms theoretical expectations on all six algorithms. Everything is open source at github.com/mithunvoe/dhaka-pathfinder with a single command to run end-to-end. I'm happy to answer anything else."

---

## Companion docs

- **`PRESENTATION.md`** — class-presentation outline (less interactive).
- **`QA.md`** — 70+ Q&A pairs, fully written out, for deeper prep.
- **`HEURISTIC.md`** — single-page "this is my heuristic" reference.
- **`ARCHITECTURE_FOR_BEGINNERS.md`** — the super-gentle walkthrough.
- **`PROBLEM_STATEMENT.md`** — the formal spec the project satisfies.
- **`DOCUMENTATION.md`** — the full 1300-line technical doc, 17 chapters.

If the teacher wants depth beyond what you can recall, point at one of those files on GitHub. **Never pretend to know something you don't.** Saying *"let me look it up in the code"* and doing it live is strictly better than guessing wrong.
