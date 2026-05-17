"""Pluggable variable-ordering / value-ordering heuristics."""

from __future__ import annotations

from fuel_csp.problem import Assignment, Problem, euclid


def static_order(problem: Problem, unassigned: list[int]) -> int:
    """Pick the next variable in input order (no heuristic)."""
    return unassigned[0]


def mrv(
    problem: Problem,
    assignment: dict[int, Assignment],
    unassigned: list[int],
    live_domains: list[list[Assignment]] | None = None,
) -> int:
    """Minimum Remaining Values: pick the variable with the smallest live domain.

    Ties broken by the degree heuristic (highest priority vehicle wins — they
    constrain more downstream because their slots are tightest and they win
    pump conflicts).
    """
    dom = live_domains if live_domains is not None else problem.domains
    best = unassigned[0]
    best_key = (len(dom[best]), -problem.vehicles[best].priority)
    for vid in unassigned[1:]:
        key = (len(dom[vid]), -problem.vehicles[vid].priority)
        if key < best_key:
            best = vid
            best_key = key
    return best


def priority_order(problem: Problem, unassigned: list[int]) -> int:
    """Highest-priority vehicle first (used as a fallback for ordering)."""
    return max(
        unassigned,
        key=lambda i: (problem.vehicles[i].priority, -i),
    )


def lcv_sort(
    problem: Problem,
    i: int,
    assignment: dict[int, Assignment],
    candidate_values: list[Assignment],
    live_domains: list[list[Assignment]] | None = None,
) -> list[Assignment]:
    """Least Constraining Value: sort values by how few choices they cut from
    the other unassigned variables' domains.

    We approximate this by counting, for each value, how many *other* variables
    have a value that would clash with it (same station/pump/slot).
    """
    dom = live_domains if live_domains is not None else problem.domains
    scored: list[tuple[int, float, Assignment]] = []
    for v in candidate_values:
        clash_count = 0
        for j in range(problem.n):
            if j == i or j in assignment:
                continue
            for other in dom[j]:
                if (
                    other.station_id == v.station_id
                    and other.pump_id == v.pump_id
                    and other.slot_id == v.slot_id
                ):
                    clash_count += 1
        # tie-break: prefer closer station and earlier slot
        veh = problem.vehicles[i]
        s = problem.stations[v.station_id]
        tie = euclid(veh.x, veh.y, s.x, s.y) + 0.1 * v.slot_id
        scored.append((clash_count, tie, v))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [v for _, _, v in scored]


def cost_sort(
    problem: Problem,
    i: int,
    candidate_values: list[Assignment],
) -> list[Assignment]:
    """Sort candidate values by single-step cost (distance + slot wait).

    Cheap proxy used by basic BT and as a tie-breaker elsewhere — produces
    far better COP solutions than raw input order at negligible runtime
    cost.
    """
    v = problem.vehicles[i]

    def key(a: Assignment) -> tuple[float, int]:
        s = problem.stations[a.station_id]
        return (euclid(v.x, v.y, s.x, s.y) + 0.3 * a.slot_id, a.slot_id)

    return sorted(candidate_values, key=key)
