"""Hard-constraint checking and soft-objective evaluation (the COP J(S))."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from fuel_csp.problem import Assignment, Problem, euclid


@dataclass
class ConsistencyCounter:
    """Lightweight metric counter for constraint checks."""

    checks: int = 0

    def tick(self) -> None:
        self.checks += 1


def per_variable_feasible(problem: Problem, i: int, val: Assignment) -> bool:
    """Already encoded into the domain — kept here for explicitness/tests."""
    v = problem.vehicles[i]
    s = problem.stations[val.station_id]
    if s.stocks(v.fuel_type) < v.demand_liters:
        return False
    if euclid(v.x, v.y, s.x, s.y) > v.range_km:
        return False
    if not (s.open_slot <= val.slot_id < s.close_slot):
        return False
    if not (v.earliest_slot <= val.slot_id <= v.latest_slot):
        return False
    return True


def pump_clash(a: Assignment, b: Assignment) -> bool:
    """Two values clash if they pin the same (station, pump, slot)."""
    return (
        a.station_id == b.station_id
        and a.pump_id == b.pump_id
        and a.slot_id == b.slot_id
    )


def supply_ok(problem: Problem, assignment: dict[int, Assignment]) -> bool:
    """Verify the *cumulative* fuel demand at each station does not exceed reserve."""
    drawn: dict[tuple[int, str], float] = defaultdict(float)
    for i, val in assignment.items():
        v = problem.vehicles[i]
        drawn[(val.station_id, v.fuel_type)] += v.demand_liters
    for (sid, ft), used in drawn.items():
        if used > problem.stations[sid].stocks(ft) + 1e-6:
            return False
    return True


def is_consistent(
    problem: Problem,
    assignment: dict[int, Assignment],
    i: int,
    val: Assignment,
    counter: ConsistencyCounter | None = None,
) -> bool:
    """Check val for variable i against partial assignment (pump + supply)."""
    if counter is not None:
        counter.tick()
    for j, other in assignment.items():
        if j == i:
            continue
        if pump_clash(val, other):
            return False
    # supply check (incremental — cheap to compute on the fly)
    v = problem.vehicles[i]
    sid = val.station_id
    ft = v.fuel_type
    used = v.demand_liters
    for j, other in assignment.items():
        if j == i:
            continue
        if other.station_id == sid and problem.vehicles[j].fuel_type == ft:
            used += problem.vehicles[j].demand_liters
    if used > problem.stations[sid].stocks(ft) + 1e-6:
        return False
    return True


def conflicts(
    problem: Problem,
    assignment: dict[int, Assignment],
    i: int,
    val: Assignment,
) -> int:
    """Number of hard-constraint violations val would cause against the current
    full assignment. Used by Min-Conflicts. Does NOT touch the consistency
    counter (Min-Conflicts has its own step counter).
    """
    n = 0
    for j, other in assignment.items():
        if j == i:
            continue
        if pump_clash(val, other):
            n += 1
    v = problem.vehicles[i]
    used = v.demand_liters
    for j, other in assignment.items():
        if j == i:
            continue
        if other.station_id == val.station_id and problem.vehicles[j].fuel_type == v.fuel_type:
            used += problem.vehicles[j].demand_liters
    if used > problem.stations[val.station_id].stocks(v.fuel_type) + 1e-6:
        n += 1
    return n


def objective(problem: Problem, assignment: dict[int, Assignment]) -> float:
    """Soft-cost J(S) — what makes this a COP rather than a strict CSP.

    J(S) = w_dist * total_distance
         + w_wait * total_wait_time (slot index of each placement)
         + w_prio * priority_penalty (emergency served late is expensive)
         + w_unassigned * (#unassigned vehicles)
    """
    w = problem.weights
    total_dist = 0.0
    total_wait = 0.0
    prio_penalty = 0.0
    n_unassigned = 0
    assigned_ids = set(assignment.keys())
    for i, v in enumerate(problem.vehicles):
        if i not in assigned_ids:
            n_unassigned += 1
            prio_penalty += v.priority * 5.0
            continue
        a = assignment[i]
        s = problem.stations[a.station_id]
        d = euclid(v.x, v.y, s.x, s.y)
        total_dist += d
        total_wait += float(a.slot_id)
        if v.kind == "ambulance":
            prio_penalty += float(a.slot_id) ** 2 * v.priority
        else:
            prio_penalty += float(a.slot_id) * (v.priority - 1)
    return (
        w["distance"] * total_dist
        + w["wait"] * total_wait
        + w["priority"] * prio_penalty
        + w["unassigned"] * n_unassigned
    )


def total_conflicts(problem: Problem, assignment: dict[int, Assignment]) -> int:
    """Hard-constraint violations across the full assignment. 0 means feasible."""
    n = 0
    # pump clashes
    seen: dict[tuple[int, int, int], int] = {}
    for i, a in assignment.items():
        key = (a.station_id, a.pump_id, a.slot_id)
        if key in seen:
            n += 1
        else:
            seen[key] = i
    # supply
    drawn: dict[tuple[int, str], float] = defaultdict(float)
    for i, a in assignment.items():
        v = problem.vehicles[i]
        drawn[(a.station_id, v.fuel_type)] += v.demand_liters
    for (sid, ft), used in drawn.items():
        if used > problem.stations[sid].stocks(ft) + 1e-6:
            n += 1
    return n
