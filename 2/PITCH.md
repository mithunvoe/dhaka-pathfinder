# Pitch — Dhaka Fuel-Crisis Resource Allocator

A 5–10 minute walk-through to use when presenting to the teacher.
Each section is one "beat" of the pitch — pause, breathe, move on.

---

## 0. Opening hook (15 seconds)

> "Sir, imagine Dhaka during a fuel crisis. There are limited stations,
> limited pumps, limited stock, and a thousand vehicles — ambulances,
> buses, trucks, cars — all trying to refuel. The question is: who goes
> where, on which pump, in which time slot, so that emergency vehicles
> are served first and total wait time is minimised?
>
> We modelled this as a **Constraint Satisfaction Problem**, and then
> upgraded it to a **Constraint Optimisation Problem** because we
> don't just want *any* feasible schedule — we want the *cheapest* one."

---

## 1. Formal formulation (1 minute)

Write on the board or point to `assignment.md §2`:

- **Variables** `X = {x_1, ..., x_N}` — one per vehicle.
- **Domain** `D_i` — feasible `(station_id, pump_id, slot_id)` triples
  per vehicle, pre-filtered by:
  - fuel-type compatibility (a diesel truck can't use a petrol-only pump),
  - reachability (`distance ≤ vehicle.range_km`),
  - shared time-window of vehicle and station.
- **Hard constraints** `C`:
  - **Pump exclusivity** — no two vehicles on the same `(station, pump, slot)`.
  - **Supply capacity** — Σ demand at a `(station, fuel_type)` ≤ reserve.
- **Soft objective** `J(S)` (the COP cost):
  ```
  J(S) = w_dist · Σ distance(v_i, station(a_i))
       + w_wait · Σ slot_index(a_i)
       + w_prio · Σ priority_penalty(v_i, a_i)
       + w_unass · (#unassigned vehicles)
  ```
  `priority_penalty` is **quadratic in slot for ambulances**, linear
  otherwise — encodes "emergency served late is much worse than a car
  served late."

> "The strict CSP just asks: does a feasible schedule exist? The COP
> asks: of all feasible schedules, which one has the lowest J(S)? That's
> the version that's actually useful in a real crisis."

---

## 2. The five algorithms (3 minutes — 30 seconds each)

Open `ALGORITHMS.md` or point to `fuel_csp/algorithms/`.

### Algorithm 1 — Basic Backtracking
> "Pure recursive depth-first search. Picks the next variable in input
> order, tries values in domain order. Our baseline. It works on small
> instances but blows up exponentially — `O(d^N)` worst case."

Code pointer: `backtracking.py:254`.

### Algorithm 2 — BT + MRV (Minimum Remaining Values)
> "*Fail first.* We pick next the variable whose live domain is smallest,
> with vehicle priority as a tie-break — that's our degree-flavoured
> twist. If a variable has only one or two legal values left, we explore
> them first: either we succeed quickly or we prune a huge subtree."

Code pointer: `heuristics.py:13`.

### Algorithm 3 — BT + LCV (Least Constraining Value)
> "*Succeed first.* Once we've picked a variable, we order its values by
> how few options they cut from the other unassigned variables. This is
> the heuristic that improves J(S) the most, because it picks values
> that leave room for cheap pairings later."

Code pointer: `heuristics.py:44`.

### Algorithm 4 — BT + Forward Checking + MRV + Degree (the champion)
> "Whenever we bind a value, we immediately propagate the constraint
> forward — we delete every clashing value from the other variables'
> live domains. Plus we run **AC-3 arc consistency once at the start**
> to prune values that have no support anywhere. This is the strongest
> backtracking variant we implemented and it has the lowest J(S) on
> almost every problem size."

Two implementation details to mention:
- **`was_nonempty` guard** in forward checking — we discovered that
  on real Dhaka instances some ambulances have empty initial domains
  (unreachable from any compatible station). A naive FC would abort
  the whole search. Our COP-relaxed FC only fails when *we* caused the
  domain to empty, not when it was already empty.
- **AC-3 preprocess** — textbook Mackworth 1977. Adds an initial
  reduction pass that's especially powerful when singleton domains
  exist; values that clash with a singleton can be removed everywhere.

Code pointers: `backtracking.py:305`, `arc_consistency.py`.

### Algorithm 5 — Min-Conflicts Local Search
> "A completely different paradigm — no tree search. Start from a random
> complete assignment, find variables in conflict, pick one at random,
> reassign it to the value that minimises conflicts. Tie-broken by
> J(S). Random-restart every 800 steps. This has **bounded runtime** —
> it doesn't care about N exponentially — so it's the fastest on big
> instances even when it doesn't fully converge."

Code pointer: `min_conflicts.py:26`.

---

## 3. The Dhaka grounding (1 minute)

> "We didn't want a toy problem. So the entire system can run on real
> Dhaka roadmap data — OSMnx pulls the drivable road graph for central
> Dhaka, we snap stations and vehicles to the nearest OSM nodes, and
> distances come from **single-source Dijkstra on the actual road
> network**, not Euclidean straight lines."

Highlights to point at:
- `fuel_csp/osm_data.py` — graph loading + caching to `data/dhaka_drive.pkl`.
- `fuel_csp/dhaka.py` — `generate_dhaka_problem(cfg)` returns a real
  Dhaka-grounded `Problem`.
- The Streamlit UI draws actual road-following paths between vehicles
  and assigned stations using `shortest_path_latlon`.

> "The solvers don't know the difference — they just read
> `problem.distance_km(i, station)`, which transparently switches
> between the OSM matrix and Euclidean fallback. That separation is
> what let us build the algorithms on synthetic instances first and
> swap in real data without changing any solver code."

---

## 4. Experimental setup (45 seconds)

Open the comparative-plots tab in the UI.

> "We ran each of the five algorithms on problem sizes `N ∈ {6, 10, 14,
> 18, 22}`, three random seeds each, time budget 5 seconds per run. We
> measure six things per run: runtime, nodes expanded, backtracks,
> constraint checks, repair steps (min-conflicts only), and J(S)."

The six plots:
1. `objective_vs_n.png` — solution quality.
2. `runtime_vs_n.png` — wall-clock.
3. `nodes_vs_n.png` — search-tree size.
4. `backtracks_vs_n.png` — how often we had to unwind.
5. `failure_rate_vs_n.png` — fraction of vehicles unassigned.
6. `heuristic_bars.png` — per-algorithm aggregates, horizontal bars.

---

## 5. The story the plots tell (1 minute)

Walk through three observations the teacher will appreciate:

> "**First** — basic backtracking has the highest J(S) at every N. It
> commits to the first feasible value, not the cheapest one. So even
> when it finds a complete assignment, the schedule is bad.
>
> **Second** — MRV cuts the search tree by 10–100× compared to basic
> BT, but the J(S) it returns is similar. That's expected — MRV is a
> *time* heuristic, not a *quality* heuristic. LCV is what improves
> quality.
>
> **Third** — Forward Checking with AC-3 is the COP champion. It has
> the lowest J(S) and the fewest backtracks at almost every N. The
> trade-off is memory — we snapshot the live domains on every binding.
>
> **Fourth** — Min-Conflicts is the only algorithm whose runtime is
> *flat* in N. On N=22 it beats every backtracking variant. But its
> quality is worse than FC because it doesn't have a search tree to
> systematically explore. It's the right algorithm for *very large*
> instances where completeness doesn't matter."

---

## 6. What makes this project specifically interesting (45 seconds)

Three things to drop in if you have time:

1. **Both CSP and COP behaviour from one engine.** The backtracking
   recursion always records the best partial assignment seen so far,
   and returns it on budget timeout. That converts a strict CSP solver
   into a degraded-but-useful COP solver with no extra code.

2. **A real bug we found and fixed.** On synthetic data, all five
   algorithms returned full assignments. On real Dhaka data, FC
   returned `0/N` for `N ≥ 18`. Root cause: ambulances unreachable from
   their only fuel-compatible station had empty domains, and FC was
   treating that as a global failure. The `was_nonempty` guard is the
   one-line fix that unlocked the Dhaka mode. (`backtracking.py:244`)

3. **AC-3 as preprocess.** Singleton domains in real Dhaka instances
   propagate through the constraint graph and can prune dozens of
   values before search even starts. We added Mackworth's textbook
   AC-3 (`arc_consistency.py`) and enabled it on the FC solver.

---

## 7. Defense Q&A — anticipated questions

### Q1. "Why isn't this just an integer linear program?"

> "Because the structure here is discrete and combinatorial — pump
> exclusivity and supply ceilings are natural binary/global constraints,
> not linear inequalities you'd want to feed to a simplex solver. CSP
> formulation lets us use heuristics that exploit the discrete domain
> structure — MRV, LCV, forward checking — and those don't have clean
> ILP analogues. Also, the assignment specifically asked for CSP/COP
> techniques."

### Q2. "Why is your degree heuristic just vehicle priority?"

> "In our problem every variable is connected to every other through
> pump-exclusivity and supply-capacity — the raw graph degree is just
> `N - 1` for everyone, so it gives no signal. Vehicle priority is the
> domain-specific surrogate that actually correlates with downstream
> constraint pressure: ambulances have tighter slot windows, win pump
> conflicts, and block more cheap pairings, so they constrain more."

### Q3. "Why does min-conflicts sometimes lose to forward checking?"

> "Because local search is incomplete. It can get stuck in a basin
> where every conflicted variable's best swap re-introduces some other
> conflict. We mitigate that with random restarts every 800 steps, but
> it's still incomplete by design. Forward checking explores
> systematically. The trade-off is exponential time vs. flat time —
> use FC up to about N=20, use min-conflicts beyond that."

### Q4. "What's the role of `J(S)` inside the algorithms?"

> "Two roles. First — in all four backtracking variants, J(S) is
> computed at `_record_best` to break ties between partial assignments
> with the same number of variables placed. So if the search hits the
> budget mid-way, we return the cheapest of the best partials seen.
> Second — in min-conflicts, J(S) is the per-step tie-break against
> conflict count: the best snapshot is the one with `(min_conflicts,
> min_J(S))` lexicographically. No algorithm uses J(S) for picking the
> next variable or value — that's done by cheap proxies like
> `distance + slot_index` to keep the inner loop fast."

### Q5. "Why a Streamlit UI? Why folium maps?"

> "Two reasons. One — the assignment asks us to compare algorithms, and
> a side-by-side map view of the actual road routes makes the
> comparison legible to a non-coder. Two — the same UI doubles as our
> demo and our defence. The 'Solve' button runs all five algorithms on
> the same Dhaka instance and shows the assigned routes, the J(S)
> values, and the per-algorithm stats in one screen. Folium is just
> the cleanest way to put a Leaflet.js map inside Streamlit."

### Q6. "Can you guarantee optimality?"

> "Yes — but only under two conditions. First, the time budget must
> not elapse. Second, you must use a backtracking variant (basic, MRV,
> LCV, or FC) — those are complete. Min-conflicts is incomplete; it
> may return a suboptimal feasible schedule even when the optimum
> exists. In practice, for `N ≤ 20` with a 5-second budget, the FC
> solver completes and returns the optimum J(S) on every test instance
> we ran."

### Q7. "What would you improve if you had more time?"

> "Three things, in order. **AC-3** — already done, added today.
> Next: **Maintaining Arc Consistency (MAC)** — re-run AC-3 after each
> binding, not just at the start. Then: **trail-based undo** — right
> now we snapshot the entire live-domain on every binding, which is
> O(N·d) memory per node. A trail records only what changed, which
> would give us a 2–5× speedup on FC."

---

## 8. Closing line (10 seconds)

> "So to summarise: we formulated Dhaka's fuel-crisis allocation as a
> CSP, upgraded it to a COP, implemented five algorithms ranging from
> textbook backtracking to local search, grounded it on real OSM road
> data, and produced a side-by-side empirical comparison. The FC
> solver with MRV + AC-3 + the degree-flavoured priority tie-break is
> the strongest configuration, and min-conflicts is the right choice
> when N gets very large. Happy to take questions, sir."

---

## Visual aids checklist before the meeting

- [ ] Streamlit UI running locally (`./run.sh ui`).
- [ ] `ALGORITHMS.md` open in a tab.
- [ ] `results/plots/` PNGs on screen (esp. `objective_vs_n.png` and `heuristic_bars.png`).
- [ ] One real Dhaka solve already cached so demo doesn't wait for OSMnx download.
- [ ] One tab open on `backtracking.py` at the `was_nonempty` guard and at the AC-3 wire-in.
