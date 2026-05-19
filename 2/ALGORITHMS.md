# Algorithms — Deep Dive

This document explains every search algorithm in the project end-to-end:
formal definition, pseudocode, code pointers, why it exists, what it costs,
and how it behaves on the Dhaka fuel-crisis CSP/COP.

All algorithms in this project share the same problem formulation:

- **Variables** `X = {x_1, ..., x_N}` — one per vehicle.
- **Domain** `D_i` — list of `(station_id, pump_id, slot_id)` triples that
  satisfy *per-variable hard constraints* (fuel-type match, range reachability,
  shared time-window of the vehicle and station). Built once in
  `Problem.build_domains` (`fuel_csp/problem.py:156`).
- **Hard constraints** `C`:
  - **Pump exclusivity** — two vehicles may not share the same
    `(station_id, pump_id, slot_id)`.
  - **Supply capacity** — the sum of `demand_liters` for vehicles that pick
    the same `(station_id, fuel_type)` may not exceed
    `Station.reserves[fuel_type]`.
  Checked dynamically by `is_consistent` and `conflicts`
  (`fuel_csp/constraints.py`).
- **Soft objective** `J(S)` (the COP cost — `objective` in
  `fuel_csp/constraints.py:115`):
  ```
  J(S) = w_dist  * Σ distance(v_i, station(a_i))
       + w_wait  * Σ slot(a_i)
       + w_prio  * Σ priority_penalty(v_i, a_i)
       + w_unass * (#unassigned vehicles)
  ```
  `priority_penalty` is quadratic in slot index for ambulances and linear
  for everything else, so an ambulance served at slot 0 is preferred over
  one served at slot 4.

All solvers expose the same shape:

```python
class Solver:
    name: str
    def solve(self, problem: Problem) -> SolverResult: ...
```

`SolverResult` carries the chosen partial assignment plus a `SolverStats`
bundle (`nodes_expanded`, `backtracks`, `constraint_checks`, `repair_steps`,
`runtime_seconds`, `objective`, `num_assigned`, …). See
`fuel_csp/algorithms/base.py`.

---

## 1. Basic Backtracking (`basic_backtracking`)

**Location:** `fuel_csp/algorithms/backtracking.py:254` (`BasicBacktracking`).
Recursion engine: `_BTBase._recurse` at `:96`.

### Idea

Classical depth-first backtracking on a CSP:

1. If every variable is assigned, return the assignment.
2. Otherwise pick the next unassigned variable in **input order**.
3. Try each value in **domain order** (i.e. the order produced by
   `build_domains`, which is sorted `(station, pump, slot)`).
4. If the value is consistent with the partial assignment, recurse.
5. If a value or recursion fails, undo and try the next value (backtrack).

### Pseudocode

```
function backtrack(assignment):
    record_best(assignment)
    if len(assignment) == N: return assignment
    if budget_exceeded: return best_so_far

    var = first_unassigned_variable()
    for val in domain(var):                # static order
        if is_consistent(assignment, var, val):
            assignment[var] = val
            result = backtrack(assignment)
            if result: return result
            del assignment[var]
        backtracks += 1
    return failure
```

### COP twist (graceful failure)

A pure CSP `backtrack` would return `failure` and abort the whole search if
a variable has no legal value. We **never** want that — the user still
deserves the best partial schedule we found. So:

- Before every recursion call we snapshot the current partial assignment as
  `best` if it is strictly better than the previous `best`
  (`_record_best` at `backtracking.py:30`).
- When the time budget elapses, we **return the best partial seen** instead
  of `null`.

This turns the strict CSP solver into a degenerate COP solver: optimal only
if the search completes within the budget, otherwise a "best feasible
partial seen so far".

### Complexity

- Worst case `O(d^N)` where `d = max |D_i|`. For our medium Dhaka
  instances (`N ≈ 20`, `d ≈ 200`) this is astronomical; the budget kicks
  in long before exhaustion.
- Constraint checks per node: `O(N)` (only pump + supply scans).

### Observed behaviour

On the comparative-plot run shipped in `results/plots/heuristic_bars.png`
this is consistently the **slowest** algorithm for `N ≥ 16` and produces
the **highest `J(S)`** because it commits to the first feasible value, not
the cheapest. It is the assignment's baseline.

---

## 2. Backtracking + MRV (`bt_mrv`)

**Location:** `BacktrackingMRV` at `backtracking.py:261`.
Heuristic: `mrv` at `fuel_csp/algorithms/heuristics.py:13`.

### Idea — Minimum Remaining Values

> *"Fail first."* Pick next the variable whose **live domain is smallest**.

Rationale: if a variable has only one or two legal values left, exploring
them first either (a) succeeds quickly or (b) prunes a huge subtree as
soon as it's clear the branch is doomed. Either way we save work.

### Pseudocode

```
function select_unassigned_variable(assignment, live_domains):
    best = first_unassigned
    best_key = (|live_domains[best]|, -priority[best])    # tie-break on priority
    for v in remaining_unassigned:
        key = (|live_domains[v]|, -priority[v])
        if key < best_key:
            best = v
            best_key = key
    return best
```

### Tie-breaking — degree heuristic flavour

When two variables tie on domain size we pick the **higher-priority**
vehicle. This is a degree-flavoured tie-break: in our problem, ambulances
constrain more downstream decisions (their slot windows are tighter and
their pump/supply use blocks more cheap pairings for cars), so they should
go in first.

This is implemented as `(len(dom), -priority)` lexicographic comparison so
we never need an explicit branch.

### Why we don't compute degree from a constraint graph

A textbook degree heuristic counts edges to unassigned variables. For this
problem **every** unassigned variable is connected to every other one
through pump-exclusivity and supply-capacity, so the raw degree is just
`N - 1 - |assignment|` and provides no signal. Vehicle priority is the
domain-specific surrogate that actually correlates with downstream
constraint pressure.

### Complexity

Same `O(d^N)` worst case as basic BT — heuristics never change the
**worst** case, only the **expected** one. In practice MRV cuts the tree
by `~10-100x` on N=20 Dhaka instances.

---

## 3. Backtracking + LCV (`bt_lcv`)

**Location:** `BacktrackingLCV` at `backtracking.py:276`.
Heuristic: `lcv_sort` at `heuristics.py:44`.

### Idea — Least Constraining Value

> *"Succeed first."* Once we pick a variable, try its **least disruptive**
> values first — the values that prune the fewest options from other
> variables' domains.

For a candidate value `v` we count how many *other* unassigned variables
have an entry in their live domain that clashes with `v` (same
`(station, pump, slot)` triple). Lower clash count → less disruption →
try it earlier.

### Pseudocode

```
function order_domain_values(var, assignment, live_domains):
    scored = []
    for v in live_domains[var]:
        clashes = 0
        for j in unassigned, j != var:
            for other in live_domains[j]:
                if same_pump_slot(other, v): clashes += 1
        tie = distance_km(var, v.station_id) + 0.1 * v.slot_id
        scored.append((clashes, tie, v))
    return [v for _, _, v in sorted(scored)]
```

### Variable ordering

`bt_lcv` uses **priority-first** variable selection (`priority_order` at
`heuristics.py:36`), not MRV. The point of this variant is to isolate
LCV's effect on `J(S)`. By placing ambulances first and ordering values
LCV-style, we get well-shaped solutions even when the tree blows up.

### Tie-break — closer & earlier

Equal clash counts are broken by `distance + 0.1 * slot` so the LCV
ranking already biases towards low-cost values. This is the cheapest way
to make LCV produce *both* high success rate and low `J(S)`.

### Cost

LCV's overhead is `O(N * d)` per value-ordering call — non-trivial. For
small domains we still come out ahead because the savings in retried
subtrees outweigh the sorting cost. For huge synthetic domains the
overhead can dominate; that's visible on the runtime panel of the
comparison plots.

---

## 4. BT + Forward Checking + MRV + Degree-flavour LCV (`bt_fc_mrv_deg`)

**Location:** `BacktrackingForwardChecking` at `backtracking.py:305`.
Inference routine: `_BTBase._forward_check` at `:192`.

### Idea — Inference at every assignment

Forward Checking (FC) propagates constraints **forward** the moment a
variable is bound:

1. Bind `var = val`.
2. For every *other* unassigned variable `j`, walk its live domain and
   delete any entry that clashes with `val`:
   - pump clash → same `(station, pump, slot)`, or
   - supply pre-screen → assigning both `var` and `j` at the same
     `(station, fuel_type)` would already exceed station reserves once
     committed slots are summed in.
3. If `j` had a non-empty domain *before* this prune and is now empty,
   AND `j` has priority `≥ var`, **abort this value** — there is no point
   recursing because `j` cannot be assigned later either.
4. Otherwise keep going with the pruned domains.

When we backtrack we restore the live-domain snapshot.

### Why `was_nonempty` guard exists (real bug fix)

Naively, "if any domain becomes empty, fail" is too aggressive on a COP
where graceful skip is allowed. Some Dhaka vehicles legitimately have
**empty initial domains** (an ambulance whose only reachable station has
no matching fuel reserve). The COP layer routes these to the *skip path*
in `_recurse`. The fix in `_forward_check`:

```python
was_nonempty = len(live_domains[j]) > 0
if not pruned and was_nonempty and vehicles[j].priority >= var_v.priority:
    return False
```

Reads: "Only consider this FC-failure if **we** are the ones who emptied
the domain, AND the affected variable is at least as important as the
current one." Without this guard, the solver returned 0/N on N=18+
Dhaka instances. With it, FC is consistently the best COP solver in the
comparative-plot panel.

### Variable order — MRV with priority tie-break

Reuses `mrv` from heuristic 2.

### Value order — LCV then `cost_sort`

Two-step:
1. Sort values by LCV (least clashes).
2. Stably re-sort the result by `distance + 0.3 * slot`. Cheaper closer
   pairings rise to the top inside each clash-equivalence class.

`cost_sort` is at `heuristics.py:78`.

### Pseudocode

```
function backtrack_fc(assignment, live_domains):
    record_best(assignment)
    if len(assignment) + |skipped| == N: return best_so_far
    if budget_exceeded: return best_so_far

    var = mrv(unassigned, live_domains)
    values = cost_sort(lcv_sort(live_domains[var], live_domains))

    if values is empty:                # COP graceful-skip
        skipped.add(var)
        return backtrack_fc(assignment, live_domains)

    for val in values:
        if not is_consistent(assignment, var, val): continue
        snapshot = deep_copy(live_domains)
        assignment[var] = val
        if forward_check(assignment, live_domains, var, val):
            r = backtrack_fc(assignment, live_domains)
            if r: return r
        live_domains = snapshot
        del assignment[var]

    # FC variant also tries skipping var, so other vehicles can still place.
    skipped.add(var)
    r = backtrack_fc(assignment, live_domains)
    skipped.discard(var)
    return r
```

### Cost

FC's prune is `O(N * d)` per binding, but it routinely cuts the search
tree by **orders of magnitude**. On the shipped plots, FC has both the
**lowest `J(S)`** and the **lowest backtrack count** for most `N`.

The trade-off: the live-domain snapshot/restore allocates more memory
per node than plain BT.

---

## 5. Min-Conflicts Local Search (`min_conflicts`)

**Location:** `MinConflictsSolver` at
`fuel_csp/algorithms/min_conflicts.py:26`.

### Idea — Iterative repair, not tree search

1. Drop vehicles with empty domains up front (they're permanently
   unassignable).
2. Start with a **random complete assignment** — every remaining vehicle
   gets a random value from its domain. This is *infeasible* in general:
   pump clashes and supply over-draws are common.
3. Find all variables currently in conflict (pump clash OR pushing supply
   past capacity). Pick one **at random**.
4. Reassign it to the value in its domain that minimises the number of
   conflicts, tie-broken by COP cost (`distance + 0.3 * slot`).
5. Track the best `(conflicts, cost)` snapshot seen so far.
6. Every `random_restart_every` steps, if there are still conflicts,
   **random-restart** from a fresh random assignment.
7. Stop when conflicts are zero, when the step budget is exhausted, or
   when the wall-clock budget is exhausted.

### Pseudocode

```
function min_conflicts(problem, max_steps, restart_every, seed):
    assignable = [i for i in range(N) if D_i is non-empty]
    A = { i: random_choice(D_i) for i in assignable }
    best, best_conf, best_cost = A, total_conflicts(A), J(A)

    for step in 1..max_steps:
        if budget_exceeded: break
        bad = conflicted_vars(A)
        if bad is empty: break
        v = random_choice(bad)
        A[v] = argmin_{val in D_v} (conflicts(A, v, val),
                                    distance + 0.3 * slot)    # COP tie-break

        if (conflicts(A), J(A)) < (best_conf, best_cost):
            best, best_conf, best_cost = A, ...

        if step % restart_every == 0 and best_conf > 0:
            A = fresh random assignment

    return drop_remaining_conflicts(best)
```

### COP graceful-failure pass

If the step budget runs out and `best` still violates hard constraints,
`_extract_feasible` (`:134`) drops the minimum number of vehicles needed
to make the schedule legal:

1. Resolve **pump clashes**: for each `(station, pump, slot)` keep only
   the **highest-priority** vehicle.
2. Resolve **supply over-draws**: walk vehicles in priority-descending
   order and drop any that would push station fuel past capacity.

This guarantees a hard-constraint-feasible output even on infeasible
input.

### Why random restart

Min-conflicts has zero memory of the search tree — it can get stuck in a
local minimum where every conflicted variable's best swap re-introduces
some other conflict. Restarts give it a chance to escape without needing
simulated annealing or tabu lists.

### Complexity

Each step is `O(N + d)`. `max_steps = 4000` by default. So total work is
bounded above by `O(max_steps * (N + d))` regardless of `N` — the budget
is *flat*, not exponential. This is why min-conflicts is the fastest
algorithm on large `N` even when it doesn't fully converge.

### Trade-offs

- **Pro**: bounded runtime, no exponential blow-up, decent quality with
  restart.
- **Con**: incomplete — may report a near-optimal infeasible schedule
  even though one exists. We hide that by always running the
  `_extract_feasible` cleanup.

---

## How the five interact in the COP runner

The CLI (`scripts/run_experiments.py`) and the Streamlit UI (`app.py`)
run all five solvers on the same `Problem` and compare them on:

| Metric                | Where it lives                                   |
|-----------------------|--------------------------------------------------|
| `runtime_seconds`     | `SolverStats.runtime_seconds` (set by `Timer`)   |
| `backtracks`          | Incremented in `_BTBase._recurse`                |
| `nodes_expanded`      | Incremented in `_BTBase._recurse` + min-conflicts step |
| `constraint_checks`   | `ConsistencyCounter` in `constraints.py`         |
| `repair_steps`        | Min-conflicts only — local-search step count    |
| `objective` (`J(S)`)  | `objective` in `constraints.py:115`              |
| `num_assigned`        | `len(result.assignment)`                         |
| `failure_rate`        | Fraction of variables not in `result.assignment` |

The comparative plot panel (`results/plots/*.png`, generated by
`fuel_csp/visualizer.py`) lays these out side-by-side at three problem
sizes and seeds, and is the artefact the assignment grades.

### Expected ranking on Dhaka instances

| Algorithm           | Speed   | `J(S)` | Backtracks | Notes |
|---------------------|---------|--------|------------|-------|
| `basic_backtracking`| slowest | highest | highest  | baseline |
| `bt_mrv`            | medium  | medium | low       | MRV cuts the tree |
| `bt_lcv`            | medium  | low    | medium    | LCV smooths the *cost* |
| `bt_fc_mrv_deg`     | medium  | **lowest** | **lowest** | the full COP champion |
| `min_conflicts`     | **fastest** | medium | n/a (`repair_steps`) | flat budget, occasionally misses |

This ranking holds on the Dhaka roadmap-grounded instances at
`N ∈ {10, 14, 18}` shipped in `results/`.
