"""Informed search algorithms — Greedy Best-First, A*, Weighted A*.

All of them accept a precomputed heuristic callable `h(node) -> float`. The `weight_cache`
holds edge weights under the realistic cost model so repeated calls inside the inner
loops are O(1).
"""

from __future__ import annotations

import heapq
from typing import Callable

import networkx as nx

from dhaka_pathfinder.algorithms.base import (
    SearchResult,
    SearchStats,
    Timer,
    edge_cost_lookup,
    effective_branching_factor,
    path_cost,
    reconstruct_path,
)


HeuristicFn = Callable[[int], float]


def _finalize(
    algorithm: str,
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    came_from: dict[int, int],
    stats: SearchStats,
    weight_cache: dict[tuple[int, int, int], float],
    success: bool,
    h_at_start: float,
) -> SearchResult:
    stats.success = success
    stats.predicted_cost_at_start = h_at_start
    if success:
        path = reconstruct_path(came_from, destination)
        stats.path_length_edges = len(path) - 1
        stats.depth = stats.path_length_edges
        total_cost, total_m = path_cost(G, path, weight_cache)
        stats.path_cost = total_cost
        stats.path_length_meters = total_m
        stats.predicted_vs_actual_gap = total_cost - h_at_start
    else:
        path = []
    stats.effective_branching_factor = effective_branching_factor(stats.nodes_expanded, stats.depth)
    return SearchResult(
        algorithm=algorithm,
        source=source,
        destination=destination,
        path=path,
        stats=stats,
    )


def astar_search(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    heuristic: HeuristicFn,
    *,
    max_nodes: int = 500_000,
) -> SearchResult:
    """Standard A* with realistic edge costs. Weighted factor = 1."""
    return _weighted_astar(
        "astar", G, source, destination, weight_cache, heuristic,
        weight=1.0, max_nodes=max_nodes,
    )


def weighted_astar_search(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    heuristic: HeuristicFn,
    *,
    weight: float = 1.8,
    max_nodes: int = 500_000,
) -> SearchResult:
    """A* with weighted heuristic term — trades optimality for speed.

    Typical sweet spot: weight in [1.3, 2.5]. Reports the *actual* cost anyway.
    """
    return _weighted_astar(
        "weighted_astar", G, source, destination, weight_cache, heuristic,
        weight=weight, max_nodes=max_nodes,
    )


def _weighted_astar(
    name: str,
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    heuristic: HeuristicFn,
    *,
    weight: float,
    max_nodes: int,
) -> SearchResult:
    stats = SearchStats()
    came_from: dict[int, int] = {}
    h_start = float(heuristic(source))
    with Timer(stats):
        g_score: dict[int, float] = {source: 0.0}
        heap: list[tuple[float, int, int]] = [(h_start * weight, 0, source)]
        counter = 0
        stats.nodes_generated = 1

        while heap:
            stats.max_frontier_size = max(stats.max_frontier_size, len(heap))
            if stats.nodes_expanded >= max_nodes:
                break
            f_cost, _, node = heapq.heappop(heap)
            current_g = g_score.get(node, float("inf"))
            if f_cost > current_g + weight * float(heuristic(node)) + 1e-6:
                stats.revisits += 1
                continue
            stats.nodes_expanded += 1
            if node == destination:
                return _finalize(name, G, source, destination, came_from, stats, weight_cache, True, h_start)
            for nbr in G.successors(node):
                edge_c, _ = edge_cost_lookup(G, node, nbr, weight_cache)
                tentative = current_g + edge_c
                if tentative < g_score.get(nbr, float("inf")):
                    g_score[nbr] = tentative
                    came_from[nbr] = node
                    f = tentative + weight * float(heuristic(nbr))
                    counter += 1
                    heapq.heappush(heap, (f, counter, nbr))
                    stats.nodes_generated += 1
    return _finalize(name, G, source, destination, came_from, stats, weight_cache, False, h_start)


def greedy_best_first(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    heuristic: HeuristicFn,
    *,
    max_nodes: int = 500_000,
) -> SearchResult:
    """Greedy Best-First — expand node with smallest h(n). No g term, not optimal."""
    stats = SearchStats()
    came_from: dict[int, int] = {}
    h_start = float(heuristic(source))
    with Timer(stats):
        visited: set[int] = {source}
        heap: list[tuple[float, int, int]] = [(h_start, 0, source)]
        counter = 0
        stats.nodes_generated = 1

        while heap:
            stats.max_frontier_size = max(stats.max_frontier_size, len(heap))
            if stats.nodes_expanded >= max_nodes:
                break
            _, _, node = heapq.heappop(heap)
            stats.nodes_expanded += 1
            if node == destination:
                return _finalize("greedy", G, source, destination, came_from, stats, weight_cache, True, h_start)
            for nbr in G.successors(node):
                if nbr in visited:
                    stats.revisits += 1
                    continue
                visited.add(nbr)
                came_from[nbr] = node
                counter += 1
                heapq.heappush(heap, (float(heuristic(nbr)), counter, nbr))
                stats.nodes_generated += 1
    return _finalize("greedy", G, source, destination, came_from, stats, weight_cache, False, h_start)
