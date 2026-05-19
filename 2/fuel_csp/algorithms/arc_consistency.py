"""AC-3 arc consistency preprocess (Mackworth, 1977).

Runs once before backtracking and prunes values from each variable's live
domain that lack support under the pump-exclusivity binary constraint
with at least one value in every other variable's live domain.

Binary constraint: a value v in D_i has support in D_j iff there exists
w in D_j such that

    (v.station, v.pump, v.slot) != (w.station, w.pump, w.slot)

i.e. v and w would not occupy the exact same pump/slot.

COP relaxation: if D_j is empty, j will be graceful-skipped by the
solver, so values in D_i are NOT removed just because D_j has no support
to offer. Mirrors the `was_nonempty` guard in forward checking.
"""

from __future__ import annotations

from collections import deque

from fuel_csp.problem import Assignment


def ac3(live_domains: list[list[Assignment]]) -> int:
    """Prune `live_domains` in place. Returns the total number of values removed.

    Operates on a copy of `problem.domains` — the original is left alone so
    other solvers running on the same Problem see untouched domains.
    """
    n = len(live_domains)
    if n <= 1:
        return 0

    queue: deque[tuple[int, int]] = deque(
        (i, j) for i in range(n) for j in range(n) if i != j
    )
    total_removed = 0

    while queue:
        i, j = queue.popleft()
        removed = _revise(live_domains, i, j)
        if removed == 0:
            continue
        total_removed += removed
        if not live_domains[i]:
            # COP: i will be graceful-skipped, don't propagate further from it.
            continue
        for k in range(n):
            if k != i and k != j:
                queue.append((k, i))

    return total_removed


def _revise(live_domains: list[list[Assignment]], i: int, j: int) -> int:
    """Drop values from D_i that have no support in D_j. Returns removal count."""
    dj = live_domains[j]
    if not dj:
        return 0  # COP relaxation — j will be skipped

    di = live_domains[i]
    if len(dj) > 1:
        # When D_j has >= 2 distinct values, any value in D_i has at least one
        # non-clashing partner — short-circuit the inner loop.
        return 0

    sole = dj[0]
    new_di: list[Assignment] = []
    removed = 0
    for v in di:
        if (
            v.station_id == sole.station_id
            and v.pump_id == sole.pump_id
            and v.slot_id == sole.slot_id
        ):
            removed += 1
        else:
            new_di.append(v)

    if removed:
        live_domains[i] = new_di
    return removed
