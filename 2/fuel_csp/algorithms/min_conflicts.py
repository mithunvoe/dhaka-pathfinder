"""
Min-Conflicts Local Search.

Start from a random complete assignment (each variable gets a random value
from its domain). At each step:
  1. Identify all variables that are currently in conflict.
  2. Pick one at random.
  3. Reassign it to the value in its domain that minimizes the number of
     conflicts (ties broken by COP cost — earlier slot / closer station wins).
  4. Repeat until no conflicts remain or the step budget is exhausted.

When the step budget is exhausted we still return the best snapshot seen
along the way. We also track a cost-trace so we can plot convergence.
"""

from __future__ import annotations

import random
from time import perf_counter

from fuel_csp.algorithms.base import Solver, SolverResult, Timer
from fuel_csp.constraints import conflicts, objective, total_conflicts
from fuel_csp.problem import Assignment, Problem


class MinConflictsSolver(Solver):
    name = "min_conflicts"

    def __init__(
        self,
        max_steps: int = 4000,
        random_restart_every: int = 800,
        seed: int = 42,
        time_budget_s: float = 10.0,
    ) -> None:
        super().__init__(time_budget_s=time_budget_s)
        self.max_steps = max_steps
        self.random_restart_every = random_restart_every
        self.seed = seed

    def solve(self, problem: Problem) -> SolverResult:
        stats = self._new_stats(problem, seed=self.seed)
        rng = random.Random(self.seed)

        # Drop vehicles with empty domains from the start — they cannot be assigned.
        assignable = [i for i in range(problem.n) if problem.domains[i]]

        t0 = perf_counter()
        with Timer(stats):
            assignment = self._random_assignment(problem, assignable, rng)
            best = dict(assignment)
            best_conf = total_conflicts(problem, best)
            best_cost = objective(problem, best)
            stats.cost_trace.append(best_cost)

            for step in range(self.max_steps):
                if self._budget_exceeded(t0):
                    break
                stats.repair_steps = step + 1
                conflicted = self._conflicted_vars(problem, assignment)
                if not conflicted:
                    break
                vid = rng.choice(conflicted)
                stats.nodes_expanded += 1
                new_val = self._least_conflicting_value(problem, assignment, vid)
                if new_val is None:
                    # variable has no domain left -> drop it (COP)
                    if vid in assignment:
                        del assignment[vid]
                else:
                    assignment[vid] = new_val

                cur_conf = total_conflicts(problem, assignment)
                cur_cost = objective(problem, assignment)
                stats.cost_trace.append(cur_cost)
                if (cur_conf, cur_cost) < (best_conf, best_cost):
                    best = dict(assignment)
                    best_conf = cur_conf
                    best_cost = cur_cost

                if step > 0 and step % self.random_restart_every == 0 and best_conf > 0:
                    assignment = self._random_assignment(problem, assignable, rng)

        # When budget kills the loop, still produce a feasible "drop conflicts" pass:
        final = self._extract_feasible(problem, best)
        stats.num_assigned = len(final)
        stats.num_unassigned = problem.n - stats.num_assigned
        stats.failure_rate = stats.num_unassigned / max(1, problem.n)
        stats.objective = objective(problem, final)
        stats.constraint_checks = stats.repair_steps  # one full check per step
        stats.success = (total_conflicts(problem, final) == 0 and stats.num_unassigned == 0)
        stats.hit_step_budget = stats.repair_steps >= self.max_steps and not stats.success
        return SolverResult(stats=stats, assignment=final)

    # -- helpers ---------------------------------------------------------------

    def _random_assignment(
        self,
        problem: Problem,
        assignable: list[int],
        rng: random.Random,
    ) -> dict[int, Assignment]:
        return {i: rng.choice(problem.domains[i]) for i in assignable}

    def _conflicted_vars(
        self, problem: Problem, assignment: dict[int, Assignment]
    ) -> list[int]:
        bad: list[int] = []
        for i, val in assignment.items():
            if conflicts(problem, assignment, i, val) > 0:
                bad.append(i)
        return bad

    def _least_conflicting_value(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        vid: int,
    ) -> Assignment | None:
        dom = problem.domains[vid]
        if not dom:
            return None
        best_val: Assignment | None = None
        best_key: tuple[int, float] = (10**9, 10**9)
        for val in dom:
            c = conflicts(problem, assignment, vid, val)
            d = problem.distance_km(vid, val.station_id) + 0.3 * val.slot_id
            key = (c, d)
            if key < best_key:
                best_key = key
                best_val = val
        return best_val

    def _extract_feasible(
        self, problem: Problem, assignment: dict[int, Assignment]
    ) -> dict[int, Assignment]:
        """Drop the smaller side of any remaining conflict so the returned
        assignment is hard-constraint-feasible (COP graceful failure)."""
        # 1. resolve pump clashes — keep highest-priority vehicle on the slot
        slot_owner: dict[tuple[int, int, int], int] = {}
        out: dict[int, Assignment] = {}
        # sort by priority desc so high-priority wins ties
        order = sorted(
            assignment.keys(),
            key=lambda i: (-problem.vehicles[i].priority, i),
        )
        for i in order:
            a = assignment[i]
            key = (a.station_id, a.pump_id, a.slot_id)
            if key in slot_owner:
                continue  # drop this one (low-priority loses)
            slot_owner[key] = i
            out[i] = a
        # 2. drop just enough vehicles to honor supply ceilings
        from collections import defaultdict
        drawn: dict[tuple[int, str], float] = defaultdict(float)
        keep: dict[int, Assignment] = {}
        for i in sorted(out.keys(), key=lambda i: (-problem.vehicles[i].priority, i)):
            a = out[i]
            v = problem.vehicles[i]
            new_draw = drawn[(a.station_id, v.fuel_type)] + v.demand_liters
            cap = problem.stations[a.station_id].stocks(v.fuel_type)
            if new_draw > cap + 1e-6:
                continue
            drawn[(a.station_id, v.fuel_type)] = new_draw
            keep[i] = a
        return keep
