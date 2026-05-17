"""
Backtracking solvers for the fuel-CSP/COP.

Four configurations live in this file. They share a common recursive engine
and only differ in two policies:
  - select_unassigned_variable(...)
  - order_domain_values(...)
  - (optional) forward_checking(...)

The recursion has a *time budget*. When the budget runs out we return the
best partial assignment seen so far — that's the COP behavior the teacher
asked for: never just fail, always return a best-found solution.
"""

from __future__ import annotations

from time import perf_counter

from fuel_csp.algorithms.base import Solver, SolverResult, SolverStats, Timer
from fuel_csp.algorithms.heuristics import (
    cost_sort,
    lcv_sort,
    mrv,
    priority_order,
)
from fuel_csp.constraints import ConsistencyCounter, is_consistent, objective
from fuel_csp.problem import Assignment, Problem


def _record_best(
    problem: Problem,
    assignment: dict[int, Assignment],
    best: dict[str, object],
) -> None:
    """Update the best-so-far snapshot if the current partial assignment beats it."""
    if len(assignment) < int(best["count"]):  # type: ignore[arg-type]
        return
    cost = objective(problem, assignment)
    if (
        len(assignment) > int(best["count"])  # type: ignore[arg-type]
        or cost < float(best["objective"])  # type: ignore[arg-type]
    ):
        best["count"] = len(assignment)
        best["objective"] = cost
        best["assignment"] = dict(assignment)


class _BTBase(Solver):
    """Engine shared by all four backtracking configurations."""

    use_forward_checking: bool = False

    def solve(self, problem: Problem) -> SolverResult:
        stats = self._new_stats(problem)
        counter = ConsistencyCounter()
        best: dict[str, object] = {
            "count": -1,
            "objective": float("inf"),
            "assignment": {},
        }
        live_domains = [list(d) for d in problem.domains]
        t0 = perf_counter()

        with Timer(stats):
            self._recurse(problem, {}, set(), live_domains, stats, counter, best, t0)

        assignment = dict(best["assignment"])  # type: ignore[arg-type]
        stats.constraint_checks = counter.checks
        stats.num_assigned = len(assignment)
        stats.num_unassigned = problem.n - stats.num_assigned
        stats.failure_rate = stats.num_unassigned / max(1, problem.n)
        stats.objective = objective(problem, assignment)
        stats.success = stats.num_unassigned == 0
        return SolverResult(stats=stats, assignment=assignment)

    # --- hooks subclasses override -------------------------------------------------
    def _select_var(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        unassigned: list[int],
        live_domains: list[list[Assignment]],
    ) -> int:
        return unassigned[0]

    def _order_values(
        self,
        problem: Problem,
        i: int,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
    ) -> list[Assignment]:
        return live_domains[i]

    # --- recursion ----------------------------------------------------------------
    def _recurse(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        skipped: set[int],
        live_domains: list[list[Assignment]],
        stats: SolverStats,
        counter: ConsistencyCounter,
        best: dict[str, object],
        t0: float,
    ) -> bool:
        _record_best(problem, assignment, best)

        # Full, no-skip assignment is the only "early-exit" condition. If we
        # have skips we keep searching — that's the COP optimization the
        # spec demands: don't commit to the first feasible-but-incomplete
        # answer when budget remains.
        if len(assignment) == problem.n:
            return True

        if self._budget_exceeded(t0):
            return True  # propagate up; best partial is already saved

        unassigned = [
            i for i in range(problem.n)
            if i not in assignment and i not in skipped
        ]
        if not unassigned:
            # Every variable is either assigned or skipped — this branch is
            # done. The best-so-far is already recorded; signal "no full
            # solution from here" so the parent tries another value.
            return False
        var = self._select_var(problem, assignment, unassigned, live_domains)

        values = self._order_values(problem, var, assignment, live_domains)
        if not values:
            # COP graceful-failure: no legal value for this variable. Mark it
            # skipped and recurse on the remaining variables instead of
            # unwinding the whole branch.
            skipped.add(var)
            ret = self._recurse(
                problem, assignment, skipped, live_domains,
                stats, counter, best, t0,
            )
            skipped.discard(var)
            return ret

        for val in values:
            stats.nodes_expanded += 1
            if not is_consistent(problem, assignment, var, val, counter):
                stats.backtracks += 1
                continue

            assignment[var] = val

            if self.use_forward_checking:
                snapshot = [list(d) for d in live_domains]
                if not self._forward_check(problem, assignment, live_domains, var, val):
                    live_domains[:] = snapshot
                    del assignment[var]
                    stats.backtracks += 1
                    continue
                if self._recurse(
                    problem, assignment, skipped, live_domains,
                    stats, counter, best, t0,
                ):
                    return True
                live_domains[:] = snapshot
            else:
                if self._recurse(
                    problem, assignment, skipped, live_domains,
                    stats, counter, best, t0,
                ):
                    return True

            del assignment[var]
            stats.backtracks += 1

            if self._budget_exceeded(t0):
                return True

        # Tried every value. For *forward-checking* solvers we also try
        # skipping this variable so the COP can still place the remaining
        # vehicles. Plain backtracking does NOT take the skip path here, so
        # we still see the exponential blow-up that the lecture predicts.
        if self.use_forward_checking:
            skipped.add(var)
            ret = self._recurse(
                problem, assignment, skipped, live_domains,
                stats, counter, best, t0,
            )
            skipped.discard(var)
            return ret
        return False

    # --- forward checking -----------------------------------------------------
    def _forward_check(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
        var: int,
        val: Assignment,
    ) -> bool:
        """Prune values from neighbors' live domains that clash with val.

        Returns False if any unassigned variable's domain becomes empty
        AND that variable has higher priority (we still allow lower-priority
        variables to drop to empty domains; that just means they go
        unassigned in the final COP output).
        """
        var_v = problem.vehicles[var]
        for j in range(problem.n):
            if j == var or j in assignment:
                continue
            pruned: list[Assignment] = []
            for other in live_domains[j]:
                # pump-exclusivity clash
                if (
                    other.station_id == val.station_id
                    and other.pump_id == val.pump_id
                    and other.slot_id == val.slot_id
                ):
                    continue
                # supply pre-screen (cheap): assigning var to (station, fuel)
                # already takes var_v.demand_liters; if other would push past
                # the reserve when combined with already-committed draws, drop it.
                if other.station_id == val.station_id:
                    o_v = problem.vehicles[j]
                    if o_v.fuel_type == var_v.fuel_type:
                        used = var_v.demand_liters + o_v.demand_liters
                        for k, ka in assignment.items():
                            if k == var:
                                continue
                            if (
                                ka.station_id == val.station_id
                                and problem.vehicles[k].fuel_type == var_v.fuel_type
                            ):
                                used += problem.vehicles[k].demand_liters
                        if used > problem.stations[val.station_id].stocks(var_v.fuel_type) + 1e-6:
                            continue
                pruned.append(other)
            if not pruned and problem.vehicles[j].priority >= var_v.priority:
                return False
            live_domains[j] = pruned
        return True


# -------------------------- the four concrete configurations -------------------


class BasicBacktracking(_BTBase):
    """Pure recursive backtracking. Variable in input order, values in domain order
    (which is itself in (station, pump, slot) order from problem.build_domains)."""

    name = "basic_backtracking"


class BacktrackingMRV(_BTBase):
    """BT + Minimum Remaining Values for variable ordering."""

    name = "bt_mrv"

    def _select_var(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        unassigned: list[int],
        live_domains: list[list[Assignment]],
    ) -> int:
        return mrv(problem, assignment, unassigned, live_domains)


class BacktrackingLCV(_BTBase):
    """BT + Least Constraining Value for value ordering.

    Variable ordering still picks the highest-priority vehicle first (so
    emergency vehicles are placed before cars) — this is the simplest
    ordering that lets LCV show its strength.
    """

    name = "bt_lcv"

    def _select_var(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        unassigned: list[int],
        live_domains: list[list[Assignment]],
    ) -> int:
        return priority_order(problem, unassigned)

    def _order_values(
        self,
        problem: Problem,
        i: int,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
    ) -> list[Assignment]:
        return lcv_sort(problem, i, assignment, live_domains[i], live_domains)


class BacktrackingForwardChecking(_BTBase):
    """The strongest backtracking variant — Forward Checking + MRV + LCV-flavored
    value ordering. This is the composite Heuristic-3 the assignment asks for."""

    name = "bt_fc_mrv_deg"
    use_forward_checking = True

    def _select_var(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        unassigned: list[int],
        live_domains: list[list[Assignment]],
    ) -> int:
        return mrv(problem, assignment, unassigned, live_domains)

    def _order_values(
        self,
        problem: Problem,
        i: int,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
    ) -> list[Assignment]:
        # cost-aware ordering on top of LCV — best COP quality
        lcv = lcv_sort(problem, i, assignment, live_domains[i], live_domains)
        return cost_sort(problem, i, lcv)
