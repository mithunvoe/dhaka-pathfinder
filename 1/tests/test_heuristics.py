"""Heuristic tests — admissibility and signature correctness."""

from __future__ import annotations

import networkx as nx

from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES, make_heuristic


def _sssp_actual(G: nx.MultiDiGraph, source: int, cost_model: RealisticCostModel, ctx: TravelContext) -> dict[int, float]:
    """True single-source shortest path costs from `source` under the realistic metric."""
    import heapq
    best: dict[int, float] = {source: 0.0}
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        c, n = heapq.heappop(heap)
        if c > best.get(n, float("inf")):
            continue
        for nbr in G.successors(n):
            edge_cost = min(
                cost_model.edge_cost(G[n][nbr][k], ctx, deterministic=True)
                for k in G[n][nbr]
            )
            nc = c + edge_cost
            if nc < best.get(nbr, float("inf")):
                best[nbr] = nc
                heapq.heappush(heap, (nc, nbr))
    return best


def test_admissible_heuristic_never_overestimates(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext) -> None:
    goal = 5
    h = make_heuristic("haversine_admissible", toy_graph, goal, context, cost_model)

    true_costs = _sssp_actual(toy_graph, goal, cost_model, context)

    for node in toy_graph.nodes():
        if node == goal:
            assert h(node) == 0.0
            continue
        lb = h(node)
        truth_to_goal = true_costs.get(node, float("inf"))
        assert lb <= truth_to_goal + 1e-3, (
            f"admissible heuristic violated at node {node}: h={lb:.2f} > true={truth_to_goal:.2f}"
        )


def test_all_heuristics_return_nonnegative_floats(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext) -> None:
    goal = 2
    for name in HEURISTIC_FACTORIES:
        h = make_heuristic(name, toy_graph, goal, context, cost_model)
        for node in toy_graph.nodes():
            val = h(node)
            assert isinstance(val, float)
            assert val >= 0
