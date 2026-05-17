# AI Lab — Assignment 2

## Constraint Satisfaction / Optimization for Urban Fuel Crisis Resource Allocation

> Synthesized from four classmates' independent transcripts of the in-class
> briefing. Where they overlapped, the requirement is treated as binding.
> Where only some mentioned a detail, it is included as an explicit requirement
> when it is operationally necessary to deliver a working, comparable solution.

---

## 1. Objective

Model a real-world scheduling / resource-allocation scenario as either a
**Constraint Satisfaction Problem (CSP)** or a **Constraint Optimization
Problem (COP)** and then **implement, evaluate, and compare** the performance
of multiple search algorithms and heuristics on it.

The teacher explicitly **recommends framing the problem as a COP** so that the
system can still return a partial / "best-found" solution when a perfect
assignment is impossible, instead of simply failing.

## 2. Scenario — Urban Fuel Crisis

A metropolitan city is experiencing a fuel shortage. Each fuel station has
strictly limited reserves of different fuel types (Petrol, Diesel, Octane).
A heterogeneous fleet of vehicles — ambulances, public buses, logistics
trucks, motorbikes, private cars — is scattered across the city and urgently
needs refueling.

If every vehicle rushes to the nearest station, gridlock occurs, emergency
vehicles get blocked, and stations run dry on the wrong fuel type. We must
design an AI-driven allocator that assigns every vehicle to a station
(and a pump and a time slot) without violating any physical, supply, or
priority limit, while optimizing total cost when full satisfaction is
impossible.

## 3. Formal Problem Definition

The student has full creative freedom over **what** the variables and
domains are; the formulation chosen for this submission is:

* **Variables `X = {x_1, x_2, ..., x_N}`** — one variable per vehicle. The
  assignment of `x_i` represents the refueling decision for vehicle `i`.
* **Domain `D_i`** of variable `x_i` — the set of feasible
  `(station, pump_id, time_slot)` triples for that vehicle, after the
  per-variable feasibility filters below have been applied.
* **Constraints `C`** — see § 4.

The CSP/COP solver searches over assignments of values from `D_i` to each
`x_i` such that all hard constraints are satisfied and the soft-cost objective
is minimized.

## 4. Constraints

### 4.1 Hard constraints (must hold in any valid assignment)

1. **Fuel-type compatibility** — vehicle `i` may only be assigned to a station
   that stocks a fuel type the vehicle accepts.
2. **Reachability** — the Euclidean distance from vehicle `i` to its assigned
   station must be `≤` the vehicle's remaining driving range.
3. **Pump exclusivity (no resource overlap)** — no two vehicles may occupy
   the **same pump at the same time slot**. This is the "two users cannot be
   served by the same fuel pump simultaneously" rule.
4. **Supply capacity** — the sum of fuel demanded by all vehicles assigned to
   a station must not exceed that station's remaining reserve of the
   corresponding fuel type.
5. **Operating-hours / time-window** — each `(station, slot)` pair must lie
   within the station's operating hours; each vehicle has a latest-acceptable
   service slot.

### 4.2 Soft constraints (used to define the COP objective)

The objective `J(S)` to **minimize** is:

```
J(S) = w_dist * total_distance
     + w_wait * total_wait_time
     + w_prio * priority_penalty
     + w_unassigned * (#unassigned vehicles)
```

where `priority_penalty` is large when emergency vehicles (ambulances) are
served in a late time slot, and `w_unassigned` is the cost paid for each
vehicle the optimizer fails to place (this is what turns a CSP into a COP).

### 4.3 Priority handling

Ambulances > buses > trucks > cars > motorbikes. The priority weight enters
both the soft objective and the variable-ordering heuristic, so emergency
vehicles are placed first and tend to receive the closest / earliest slots.

## 5. Required Algorithms

Implement **five** solver configurations:

1. **Basic Backtracking** — pure recursive search, no heuristics, no
   propagation.
2. **Backtracking + Heuristic 1 — MRV** (Minimum Remaining Values for
   variable ordering).
3. **Backtracking + Heuristic 2 — LCV** (Least Constraining Value for
   value ordering).
4. **Backtracking + Heuristic 3 — Forward Checking + MRV + Degree
   tie-break** (composite, the strongest backtracking variant).
5. **Local Search — Min-Conflicts** (start from a random complete
   assignment, repeatedly reassign the most-conflicted variable to its
   least-conflicted value).

## 6. Functional requirements

* **Formal CSP/COP formulation** of `X`, `D`, `C` in code.
* **Synthetic data generator** — generating realistic vehicles and stations
  programmatically. Real-world OSM data is explicitly **not required**
  (multiple classmates confirmed this).
* **All five algorithms implemented** and runnable from a single entry point.
* **Scalability testing** at multiple problem sizes — at minimum
  `N ∈ {10, 20, 30}`; we extend to `{10, 20, 30, 40, 50}` for clearer
  scaling curves.
* **Graceful failure handling** — when no valid assignment exists, the
  COP returns the best partial assignment, and the failure rate / number
  of unassigned variables is reported.

## 7. Non-functional / measurement requirements

For every (algorithm, problem-size) pair the solver records:

* Number of **backtracks** (recursive unwinds caused by constraint
  violation).
* Number of **nodes expanded** (recursive calls / variable assignments
  attempted).
* Number of **constraint checks**.
* **Execution time** (wall clock, `perf_counter`).
* **Final objective `J(S)`** (the COP cost).
* **Failure rate** — fraction of vehicles left unassigned.
* For Min-Conflicts: also the number of repair steps and whether it hit
  the step budget.

## 8. Deliverables

* **Source code** — clean, runnable, with tests.
* **Performance report** — both
  * `results/REPORT.md` with embedded tables and plot links, and
  * a Jupyter notebook `notebooks/fuel_csp_analysis.ipynb` re-running the
    experiments and rendering the same plots inline.
* **Tables** — CSVs in `results/` summarizing metrics per
  (algorithm, problem-size).
* **Plots** — at minimum:
  1. Execution time vs. N (scalability).
  2. Nodes expanded vs. N.
  3. Backtracks vs. N.
  4. Final objective `J(S)` vs. N (solution quality).
  5. Failure rate vs. N (graceful-degradation evidence).
  6. Heuristic comparison bar chart (per-algorithm aggregates).
  7. Min-Conflicts convergence curve (cost vs. repair step).

The teacher explicitly said a **strict formal lab-report format is NOT
required** for this submission — a clear, well-organized document or
notebook that shows the graphs and tables is sufficient.

## 9. AI-course terminology this assignment exercises

CSP, COP, variables, domains, constraints, hard vs. soft constraints,
constraint propagation, **forward checking**, **arc consistency (AC-3
spirit)**, variable-ordering heuristics (**MRV**, **degree heuristic**),
value-ordering heuristics (**LCV**), **backtracking search**, **local
search**, **min-conflicts**, **objective function**, **scalability /
asymptotic behavior**, **failure rate**, **completeness vs. optimality**,
exponential-vs-polynomial empirical scaling.
