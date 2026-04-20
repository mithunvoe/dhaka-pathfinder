"""Algorithm correctness tests against a toy graph."""

from __future__ import annotations

import heapq
import networkx as nx

from dhaka_pathfinder.algorithms import ALGORITHMS
from dhaka_pathfinder.algorithms.base import edge_cost_lookup, path_cost
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel
from dhaka_pathfinder.heuristics import make_heuristic


def _true_cost(G: nx.MultiDiGraph, source: int, dest: int, weights: dict[tuple[int, int, int], float]) -> float:
    best: dict[int, float] = {source: 0.0}
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        c, n = heapq.heappop(heap)
        if c > best.get(n, float("inf")):
            continue
        if n == dest:
            return c
        for nbr in G.successors(n):
            ec, _ = edge_cost_lookup(G, n, nbr, weights)
            nc = c + ec
            if nc < best.get(nbr, float("inf")):
                best[nbr] = nc
                heapq.heappush(heap, (nc, nbr))
    return float("inf")


def test_all_algorithms_find_path(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext) -> None:
    weights = cost_model.precompute_edge_weights(toy_graph, context, deterministic=True)
    source, dest = 0, 5

    for name, fn in ALGORITHMS.items():
        if name in ("greedy", "astar", "weighted_astar"):
            h = make_heuristic("haversine_admissible", toy_graph, dest, context, cost_model)
            result = fn(toy_graph, source, dest, weights, heuristic=h)
        else:
            result = fn(toy_graph, source, dest, weights)
        assert result.stats.success, f"{name} failed to find a path"
        assert result.path[0] == source
        assert result.path[-1] == dest
        cost, _ = path_cost(toy_graph, result.path, weights)
        assert cost == result.stats.path_cost
        assert result.stats.nodes_expanded >= 1


def test_ucs_and_astar_return_optimal(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext) -> None:
    weights = cost_model.precompute_edge_weights(toy_graph, context, deterministic=True)
    source, dest = 0, 5
    true = _true_cost(toy_graph, source, dest, weights)

    ucs_res = ALGORITHMS["ucs"](toy_graph, source, dest, weights)
    assert ucs_res.stats.path_cost == true

    h = make_heuristic("haversine_admissible", toy_graph, dest, context, cost_model)
    astar_res = ALGORITHMS["astar"](toy_graph, source, dest, weights, heuristic=h)
    assert astar_res.stats.path_cost == true
