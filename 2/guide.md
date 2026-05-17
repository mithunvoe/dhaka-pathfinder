# Fuel-CSP — Comprehensive Defense Guide

> Read this top-to-bottom and you will be able to defend every line of this
> assignment in front of the teacher. Each section is written like a tutor
> explaining the material to someone who hasn't seen the slides — no
> hand-waving, no "trust me", everything is grounded in the code in this
> repo and the experimental data in `results/`.

---

## Table of contents

1. The 30-second elevator pitch
2. AI-course vocabulary, defined precisely
3. The problem, in plain words
4. The formal CSP/COP definition (X, D, C, J)
5. The five algorithms — what each one does and **why** it works
6. What we measure and why
7. Reading the results — what the plots prove
8. Likely defense questions (with answers)
9. Mapping every requirement in `assignment.md` to the code that fulfills it

---

## 1. The 30-second elevator pitch

The city has a fuel shortage. We have N vehicles that all need fuel, and
only a handful of stations with limited reserves. We have to decide
*which vehicle goes to which station, at which pump, in which time slot*
so that every physical constraint (no two cars on the same pump at the
same time, no station running dry, no car running farther than its fuel
allows) is satisfied. When perfect assignment is impossible we still
want the best partial assignment we can find.

This is a textbook **Constraint Optimization Problem (COP)**. We solve
it with five different AI search algorithms and compare them on
execution time, search effort (nodes / backtracks), and solution
quality.

---

## 2. AI-course vocabulary, defined precisely

| Term                              | Meaning in this project                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Constraint Satisfaction Problem (CSP)** | A triple `(X, D, C)`: variables, their finite domains, and constraints. A *solution* assigns each variable a value such that **all** constraints hold. |
| **Constraint Optimization Problem (COP)** | A CSP plus an objective function `J(S)`. Goal: find the assignment with the lowest `J(S)`. If a perfect CSP solution doesn't exist, return the *least-bad* partial one. |
| **Variable**                       | One vehicle that needs fuel. There are `N` of them. We call them `x_1, ..., x_N`. |
| **Domain**                         | The legal values a variable can take. Here, each `D_i` is the set of `(station, pump, slot)` triples that *individually* satisfy compatibility, reachability and time-window rules for vehicle `i`. |
| **Constraint**                     | A rule that constrains *combinations* of values. The hard ones in this problem are pump-exclusivity and supply-capacity. |
| **Hard constraint**                | Must never be violated. Pump-exclusivity, supply-capacity, fuel-type-compatibility, reachability, time window. |
| **Soft constraint**                | Preference, not a rule. Encoded as a weighted term inside the objective `J(S)`. Examples: minimize total distance, prefer earlier slots for ambulances. |
| **Objective function `J(S)`**      | Numerical badness score for an assignment `S`. Lower is better. See § 4. |
| **Search tree**                    | The recursive structure of all partial assignments backtracking explores. The depth is `N`. The branching factor is `|D_i|`. |
| **Backtracking**                   | DFS that incrementally extends a partial assignment and undoes the last choice (a "backtrack") whenever it leads to inconsistency. |
| **Node expansion**                 | One recursive call attempting one value for one variable. Counts work done. |
| **Backtrack**                      | An unwind caused by a constraint violation. We count these separately because they're the visible cost of bad search choices. |
| **Constraint propagation**         | After committing to a value, eagerly removing values from neighbors' domains that the commitment has just made impossible. **Forward Checking** is the simplest form. |
| **Forward Checking (FC)**          | After assigning `x_i = v`, for every still-unassigned `x_j`, drop from `D_j` any value that would clash with `v`. If `D_j` empties out for an important variable, abandon this branch immediately. |
| **Arc Consistency (AC-3)**         | A stronger propagation: re-prune transitively until no more values can be removed. Not implemented as a standalone algorithm here — FC is the simpler form, and we got most of the benefit from it. |
| **Variable-ordering heuristic**    | The rule for picking *which* unassigned variable to try next. **MRV** and **degree heuristic** are examples. |
| **MRV (Minimum Remaining Values)** | Pick the variable with the *smallest* legal domain first. Intuition: variables most likely to fail should fail early, before we waste effort on others. Also called the *fail-first* principle. |
| **Degree heuristic**               | Tie-break for MRV: pick the variable involved in the most constraints with other unassigned variables. In this code we use vehicle *priority* (ambulance > bus > truck > car > motorbike) as a domain-specific proxy. |
| **Value-ordering heuristic**       | The rule for picking *which* value to try first for the chosen variable. **LCV** is the standard one. |
| **LCV (Least Constraining Value)** | Try the value that removes the *fewest* options from other variables' domains. Intuition: keep options open for the rest of the search. |
| **Local search**                   | A class of algorithms that start from a *complete* assignment (possibly bad) and iteratively *modify* it to reduce conflicts. Doesn't backtrack. |
| **Min-Conflicts**                  | The canonical local-search algorithm for CSPs. At each step: pick a variable currently in conflict; reassign it to the value that minimizes its conflicts. Repeat. |
| **Completeness**                   | An algorithm is *complete* if it will find a solution when one exists. Backtracking is complete. Min-Conflicts is **not** — it can get stuck at a local optimum. |
| **Optimality**                     | An algorithm is *optimal* if it finds the *best* solution (lowest J(S)) when one exists. Plain backtracking is not optimal for COPs — it returns the first solution it sees. We approximate optimality by tracking the best-so-far and running until the budget expires. |
| **Time budget**                    | A wall-clock cap (`time_budget_s`). When it expires we return the best-found partial. This is what makes the system *anytime* — useful even when it doesn't finish. |
| **Failure rate**                   | (`#vehicles left unassigned`) / N. Quantifies graceful failure. A pure CSP solver fails entirely or succeeds entirely; a COP solver degrades gracefully — this number measures how gracefully. |
| **Effective branching factor**     | An empirical measure of "how bushy" the search tree was. Not reported in this version. |

---

## 3. The problem, in plain words

There's a fuel crisis. Three things are scarce: fuel itself (stations have
limited reserves), pumps (each station has only 1–3 pumps), and time
(only 6 time slots before the simulation closes). Five kinds of
vehicles need fuel:

* **Ambulances** — diesel, low remaining range (always low on fuel),
  must be served in the first third of the day. **Highest priority.**
* **Buses** — diesel, high demand (1.5×), public transport.
* **Trucks** — mostly diesel (some octane), highest demand (1.8×).
* **Cars** — petrol or octane, mid demand.
* **Motorbikes** — petrol, tiny demand (0.25×), lowest priority.

We have to **assign every vehicle to a `(station, pump, time-slot)`
triple** such that:

1. The vehicle can physically reach the station (Euclidean distance ≤
   remaining range).
2. The station stocks the right fuel type.
3. The station has enough reserve to satisfy this vehicle (and every
   other vehicle going to the same station that day).
4. No two vehicles occupy the **same pump at the same slot**.
5. The slot is inside the station's operating hours and the vehicle's
   acceptable service window.

If a perfect assignment is impossible (often the case when N > 30), we
still want the **best partial assignment** — and we want emergency
vehicles to win their slots.

---

## 4. The formal CSP/COP definition

### 4.1 Variables — `X`

```
X = { x_1, x_2, ..., x_N }
```

Where `x_i` represents the refueling decision for the `i`-th vehicle.

* Code: `fuel_csp/problem.py::Vehicle` is the i-th entity. The "variable" is
  conceptually `i`; the assignment is `Problem.domains[i] -> Assignment`.

### 4.2 Domains — `D_i`

```
D_i = { (s, p, t)  |  station s, pump p, slot t }
        intersected with the per-variable hard constraints:
           - reserves[s][fuel_type_of_i] >= demand_i
           - euclidean(v_i, s) <= range_i
           - earliest_slot_i <= t <= latest_slot_i
           - station s open at slot t
           - 0 <= p < pumps_of(s)
```

Built once, up-front, by `Problem.build_domains()`.

### 4.3 Constraints — `C`

Hard constraints applied **between variables** during search:

1. **Pump exclusivity** — `x_i.station == x_j.station ∧ x_i.pump == x_j.pump ∧ x_i.slot == x_j.slot ⇒  i = j`.
   Code: `constraints.pump_clash`.
2. **Supply capacity** — for every (station, fuel-type):
   `sum( demand_k  for every assigned x_k drawing that fuel from that station ) ≤ station.reserves[fuel-type]`.
   Code: `constraints.supply_ok` (full) and the incremental version inside
   `constraints.is_consistent`.

Per-variable hard constraints (fuel-type, reachability, time-window) are
already baked into each `D_i`, so the search doesn't need to re-check them.

### 4.4 Objective — `J(S)` (the COP)

```
J(S) =   w_dist * total_distance
       + w_wait * total_wait_time          # sum of slot indices
       + w_prio * priority_penalty         # ambulances penalized heavily for late slots
       + w_unassigned * #unassigned        # the big stick that turns CSP into COP
```

Weights live in `Problem.weights` (defaults: distance=1, wait=0.5,
priority=10, unassigned=100). Code: `constraints.objective`.

The teacher specifically asked us to frame this as a COP so the system
returns a best-partial when full assignment is impossible. The
`#unassigned` term inside `J(S)` is what enforces that preference —
without it, the solver would happily return the empty assignment (which
trivially satisfies every constraint).

---

## 5. The five algorithms — what they do and **why** they work

All five live under `fuel_csp/algorithms/`. They all share the same
`SolverStats` schema (`base.py`) so we can compare them apples-to-apples.

### 5.1 Basic Backtracking — `BasicBacktracking`

```
def recurse(assignment):
    if assignment is complete:
        return True
    var = next unassigned variable in input order
    for val in var's domain:
        if val is consistent with assignment:
            assignment[var] = val
            if recurse(assignment):
                return True
            del assignment[var]
    return False
```

No heuristics, no propagation. The textbook formulation of DFS over the
search tree. **This is the baseline.** Code:
`algorithms/backtracking.py::BasicBacktracking`.

**Why it works:** it's complete — if a feasible assignment exists, it
will eventually find one by exhaustive search.

**Why it's slow:** the search tree has size up to `|D_i|^N`. At N=50
with a domain of ~50 values per variable, the worst-case tree has
~10⁸⁵ leaves. That's the explosion you see in the plots.

### 5.2 BT + MRV — `BacktrackingMRV`

Same recursion. The only change: `next unassigned variable` is no longer
"in input order" but rather the variable with the **smallest live
domain**. Ties are broken by vehicle priority (ambulance first).

**Why it works:** the fail-first principle. If a variable is going to
cause a backtrack, we want to discover that as close to the root as
possible — we don't want to commit to N-1 other variables only to learn
the last one is infeasible.

**Empirically (from `results/experiments_summary.csv`):** at N=30, MRV
expands ~50k nodes vs. basic BT's ~200k. At N=50, basic BT hits the 4 s
budget; MRV hits it at the same N but had explored more "useful"
variables along the way.

### 5.3 BT + LCV — `BacktrackingLCV`

Variable ordering picks the highest-priority vehicle first (so ambulances
go first). Value ordering uses LCV: for each candidate value, count how
many *other* unassigned variables' domains contain that value. Try the
value with the **fewest** clashes first — i.e. the value that constrains
the rest of the search the least.

**Why it works:** keeps options open for the remaining variables. If the
first value we try succeeds, we never backtrack — minimizing backtracks
is the whole game.

**Empirically:** at N=20, LCV does only ~12 backtracks (vs. basic BT's
~60k). The cost is the LCV sort itself, which is O(N × |D|), so per-node
overhead is higher.

### 5.4 BT + FC + MRV + Degree — `BacktrackingForwardChecking`

The composite, "everything we know how to do" backtracking variant.
After each assignment we **forward-check**: walk every unassigned
variable's live domain and remove values that clash with the just-made
assignment. We also pre-screen supply capacity: if assigning a candidate
to a station would cause supply to overflow given the current
commitments, prune it.

Variable order: MRV. Value order: LCV, with a cost-aware tie-break (we
prefer values closer to the vehicle and in earlier slots).

**Why it works:** propagation discovers infeasibility many levels before
plain BT would. A pruning step in FC turns a deep, expensive subtree
explosion into a single domain-size check at the current level.

**Empirically:** at N=50, FC expands only ~22k nodes — an order of
magnitude fewer than plain BT. The trade-off (which the report
discusses): FC's aggressive pruning sometimes leaves low-priority
vehicles with empty domains, which then get *skipped* in the COP sense.
So FC has slightly higher *failure rate* than plain BT at large N — it
commits to a smaller-but-cheaper-to-find partial. This is a classic
exploration-vs-exploitation trade-off.

### 5.5 Min-Conflicts — `MinConflictsSolver`

A completely different shape of algorithm.

```
assignment = random complete assignment from each domain
while there is at least one conflict and we have budget:
    pick a conflicted variable at random
    set it to the value in its domain that minimizes its conflicts
    (tie-break by COP cost)
```

After the loop, a clean-up pass drops the minimum number of low-priority
vehicles needed to make the assignment *hard-feasible* (no pump clashes,
no supply overflows). That's `_extract_feasible` in
`algorithms/min_conflicts.py`.

**Why it works:** local search exploits a deep property of CSPs called
the *"cluster"* structure — the solution space tends to have large
regions of near-feasible assignments, so random-walk-with-improvement
finds them fast. For the N-queens problem, min-conflicts famously
solves N = 1,000,000 in seconds.

**Empirically:** min-conflicts has the **lowest J(S) at every N** ≥ 20
and the **lowest failure rate** at every N. Its node count grows
**linearly** with N (1775 at N=40, 3160 at N=50) while basic BT grows
exponentially.

**Why isn't it always best?**

* Not complete: cannot prove "no solution exists". With a tight time
  budget on an over-constrained instance, it can return a partial when a
  fully feasible solution actually exists.
* Stochastic: results vary by seed. We average across 5 seeds to handle
  this.

---

## 6. What we measure and why

Per `assignment.md § 7`, every solver run records:

| Field                  | Why we care                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `runtime_seconds`      | The wall-clock comparison the assignment specifically asks for.                                     |
| `nodes_expanded`       | Recursive calls (or value attempts in MC). Search effort.                                           |
| `backtracks`           | Recursion unwinds. The "wasted" search effort. The assignment explicitly asks for this.             |
| `constraint_checks`    | How many times the consistency oracle ran. A finer-grained search-effort number.                    |
| `objective`            | COP solution quality. Lower is better.                                                              |
| `num_assigned` / `num_unassigned` / `failure_rate` | Graceful-failure evidence — the COP behavior the teacher highlighted. |
| `success`              | True iff the assignment is complete *and* hard-feasible.                                            |
| `runtime_seconds`      | If this equals the time budget exactly, the solver was killed early.                                |
| `repair_steps` (MC)    | How many Min-Conflicts iterations ran.                                                              |
| `cost_trace` (MC)      | The J(S) value after every repair step — used to plot the convergence curve.                        |

These get saved into `results/experiments_raw.csv` (one row per run)
and aggregated into `results/experiments_summary.csv` (means + stds).

---

## 7. Reading the results — what the plots prove

All plots live in `results/plots/`.

### `runtime_vs_n.png` (log-y axis)

Each algorithm's curve. Basic BT climbs into the time budget; FC and
LCV/MRV stay flat much longer; min-conflicts stays nearly flat
throughout. **This is the scalability picture.**

### `nodes_vs_n.png` (log-y axis)

The exponential-vs-polynomial story. Basic BT crosses into the hundreds
of thousands at N=30+. Heuristic variants stay 1–3 orders of magnitude
below. **This is the empirical evidence that heuristics work.**

### `backtracks_vs_n.png` (log-y axis)

Almost identical to nodes (because nearly every node attempt in plain BT
ends in a backtrack at large N). Min-Conflicts is identically zero —
local search never backtracks.

### `objective_vs_n.png`

Solution quality. **Min-Conflicts is the lowest** at every N ≥ 20. This
is the key headline: local search produces better *solutions*, not just
faster *search*.

### `failure_rate_vs_n.png`

Graceful failure. As N grows, every solver returns more partial
assignments. The shape of each curve tells you how aggressively that
solver gives up on hard variables.

### `heuristic_bars.png`

Three bar charts side-by-side: mean runtime, mean backtracks, mean
J(S). Lets the reader compare all five algorithms at a glance.

### `min_conflicts_convergence.png`

One line per seed, showing J(S) at every repair step. Convergence is
fast initially and asymptotes — the canonical shape of a local-search
curve.

### `sample_topology.png`

A real picture of one solved instance: stations as green squares,
vehicles as colored dots, edges showing the assignment. Lets the reader
verify visually that we're solving a real spatial problem.

---

## 8. Likely defense questions (with answers)

**Q: Why is this a COP and not a CSP?**
A: We added an objective function `J(S)` and an "unassigned penalty"
inside it, so when a perfect CSP solution doesn't exist (because the
problem is over-constrained at N=50 with only 6 stations and 6 slots and
limited pumps), the solver still returns the *best partial* instead of
just failing. The teacher specifically recommended COP for this reason.

**Q: How do you define a "Variable"?**
A: One per vehicle. The assignment of variable `i` is a triple
`(station, pump, slot)` that says where, on which pump, and when
vehicle `i` is served.

**Q: How is the domain built?**
A: For each vehicle we enumerate every `(s, p, t)` triple where station
`s` stocks the right fuel, has enough of it, is reachable within the
vehicle's range, and the slot lies in both the station's hours and the
vehicle's window. Code: `Problem._domain_for`. This is the
*per-variable* hard-constraint filter.

**Q: How are constraints checked during search?**
A: Pump exclusivity and supply capacity are *between-variable*
constraints, so we check them every time we extend a partial assignment.
That's `constraints.is_consistent`. The constraint-check counter inside
`SolverStats` lets us see how much work the oracle does.

**Q: What is Forward Checking really doing?**
A: After we set `x_i = v`, we walk every unassigned `x_j` and drop from
its live domain every value that would clash with `v`. If a higher-or-
equal priority variable's domain empties out, we abort the branch
immediately (no point continuing — we know we'd backtrack eventually).
This is what makes the FC variant explore so few nodes.

**Q: Why does Forward Checking sometimes have a *higher* failure rate
than basic BT?**
A: FC commits to a small-but-clean partial more quickly. Basic BT runs
until the time budget, which lets it stumble onto larger partials by
trying many configurations. FC's pruning is theoretically sound — every
value it drops genuinely would clash — but the order in which it
commits to high-priority placements can leave low-priority vehicles
without any legal value. In the COP, those vehicles are *skipped*. So
FC trades higher failure-rate-on-low-priority for lower
failure-rate-on-high-priority. Look at the assignment to see this: FC
almost never leaves an ambulance unassigned.

**Q: How does Min-Conflicts know when to stop?**
A: It stops when every variable is conflict-free, or when it hits
`max_steps` (4000 by default), or when the time budget expires. The
final partial is then cleaned up via `_extract_feasible` to remove any
remaining hard-constraint violations by dropping the smaller side of
each clash.

**Q: How do you handle the "no valid assignment exists" case?**
A: Two layers. First, the COP objective penalizes unassigned vehicles
with `w_unassigned * #unassigned`, so the solver *prefers* assigning
more vehicles. Second, in code, when an algorithm runs out of options
for a variable, it *skips* the variable and continues with the rest;
when the algorithm terminates, the best partial assignment seen along
the way is returned. The failure rate is recorded and plotted.

**Q: Why do you sample multiple seeds?**
A: The synthetic data generator is parameterized by `seed`. Different
seeds give different vehicle/station layouts, which exercise different
parts of the algorithms. Averaging across 5 seeds at each N
de-randomizes the comparison and gives error bars.

**Q: What's the time complexity?**
A:
* Basic BT — worst-case `O(|D|^N)`. Confirmed empirically: ~exponential
  in N.
* MRV / LCV — same worst case but smaller constant; empirically ~10×
  faster than basic BT.
* FC — same worst case but pruning cuts most branches; empirically
  ~100× faster than basic BT.
* Min-Conflicts — each step is `O(|D| × constraint-check-cost)`, and the
  number of steps is empirically near-linear in N.

**Q: Why don't you use Arc Consistency (AC-3)?**
A: AC-3 is a stronger propagation than FC. We chose FC because it's the
simplest propagation that still demonstrates the heuristic-search lesson
the assignment is about. The course slides cover both. Adding AC-3
would be straightforward — extend `_forward_check` to run to fixed
point.

**Q: Where is the "priority" rule visible?**
A: Three places:
1. `PRIORITY` dict in `problem.py` (ambulance=5, ..., motorbike=1).
2. `priority_order` heuristic in `heuristics.py` — used as the variable
   selector for BT+LCV.
3. The COP objective: `priority_penalty` in `constraints.objective`
   penalizes late slots quadratically for ambulances. Late ambulance
   placement is exponentially more expensive than late car placement.

**Q: How is the failure rate reported?**
A: `SolverStats.failure_rate = num_unassigned / N`. It's per-run; the
report shows the mean across seeds.

---

## 9. Mapping every requirement in `assignment.md` to code

| Requirement                                                  | Where                                                                                              |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Formal X, D, C definition                                    | `fuel_csp/problem.py` + § 4 of this guide                                                          |
| Synthetic data generator                                     | `fuel_csp/synthetic.py`                                                                            |
| Basic backtracking                                           | `algorithms/backtracking.py::BasicBacktracking`                                                    |
| Backtracking + heuristic 1 (MRV)                             | `algorithms/backtracking.py::BacktrackingMRV` + `heuristics.py::mrv`                               |
| Backtracking + heuristic 2 (LCV)                             | `algorithms/backtracking.py::BacktrackingLCV` + `heuristics.py::lcv_sort`                          |
| Backtracking + heuristic 3 (FC + MRV + Degree)               | `algorithms/backtracking.py::BacktrackingForwardChecking`                                          |
| Local search / Min-Conflicts                                 | `algorithms/min_conflicts.py`                                                                      |
| Resource overlap constraint (two vehicles, same pump, same time) | `constraints.py::pump_clash`                                                                   |
| Supply capacity constraint                                   | `constraints.py::supply_ok` + `is_consistent`                                                      |
| Priority handling                                            | `problem.PRIORITY`, `heuristics.priority_order`, the priority terms inside `constraints.objective` |
| Time-window constraint                                       | Baked into `Problem._domain_for`                                                                   |
| Scalability over multiple sizes                              | `ExperimentConfig.sizes = (10, 20, 30, 40, 50)` in `analyzer.py`                                  |
| Track #backtracks                                            | `SolverStats.backtracks` (incremented in `_recurse`)                                                |
| Track execution time                                         | `Timer` context manager in `algorithms/base.py`                                                    |
| Graceful failure                                             | Skip-on-no-values inside `_recurse`; `_extract_feasible` in min-conflicts; `failure_rate` reporting |
| Tables of metrics across sizes                               | `results/experiments_summary.csv` + the tables in `results/REPORT.md`                              |
| Plots comparing exec-time and node-expansions                | `results/plots/runtime_vs_n.png`, `nodes_vs_n.png`, `backtracks_vs_n.png` (all three on one log axis) |
| Plot comparing BT, heuristic-BT, local search                | `results/plots/heuristic_bars.png`                                                                 |
| Performance report                                           | `results/REPORT.md` + `notebooks/fuel_csp_analysis.ipynb`                                          |

---

## 10. How to run everything from scratch

```bash
# from inside  2/
./run.sh test         # 34 tests pass, 1 skipped (seed-conditional)
./run.sh experiments  # ~110 s on a laptop — runs all 125 algorithm × N × seed cells
./run.sh report       # rebuilds results/REPORT.md from the latest CSV
./run.sh notebook     # rebuilds notebooks/fuel_csp_analysis.ipynb
./run.sh solve --algo bt_fc_mrv_deg -n 30 --seed 42
```

Or, end-to-end:

```bash
./run.sh all          # tests + experiments + report + notebook
```

---

## 11. One-paragraph summary you can recite

> We modeled an urban fuel-crisis vehicle-to-station allocator as a
> Constraint Optimization Problem with N variables (one per vehicle),
> finite domains of `(station, pump, time-slot)` triples, hard
> constraints on pump-exclusivity and station supply, and a weighted
> soft objective `J(S)` that penalizes distance, slot wait, late
> ambulance service, and unassigned vehicles. We implemented five
> solvers — plain backtracking, BT+MRV, BT+LCV, BT+Forward-Checking
> with composite heuristics, and Min-Conflicts local search — and
> measured their nodes expanded, backtracks, runtime, and objective
> across `N ∈ {10, 20, 30, 40, 50}` and five random seeds each.
> Plain backtracking exhibits the predicted exponential blow-up in
> nodes and runtime; each heuristic produces an order-of-magnitude
> reduction in search effort; Min-Conflicts produces the best
> objective value and the lowest failure rate at every problem size
> ≥ 20. All algorithms gracefully return the best partial assignment
> when no perfect solution exists, satisfying the COP requirement.
