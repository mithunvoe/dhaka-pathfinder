"""Uninformed search algorithms — BFS, DFS, UCS.

All three use the precomputed realistic-cost edge weights (§3 of the spec). BFS and DFS
traverse by edge count but their *reported* path cost is the realistic metric — not
hop-count — as explicitly required by the assignment.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

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


def _finalize(
    algorithm: str,
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    came_from: dict[int, int],
    stats: SearchStats,
    weight_cache: dict[tuple[int, int, int], float],
    success: bool,
) -> SearchResult:
    stats.success = success
    if success:
        path = reconstruct_path(came_from, destination)
        stats.path_length_edges = len(path) - 1
        stats.depth = stats.path_length_edges
        total_cost, total_m = path_cost(G, path, weight_cache)
        stats.path_cost = total_cost
        stats.path_length_meters = total_m
    else:
        path = []
    stats.effective_branching_factor = effective_branching_factor(
        stats.nodes_expanded, stats.depth,
    )
    return SearchResult(
        algorithm=algorithm,
        source=source,
        destination=destination,
        path=path,
        stats=stats,
    )


def bfs_search(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    *,
    max_nodes: int = 200_000,
) -> SearchResult:
    """Breadth-first traversal; reports realistic cost on success."""
    stats = SearchStats()
    came_from: dict[int, int] = {}
    with Timer(stats):
        if source == destination:
            stats.success = True
            return _finalize("bfs", G, source, destination, came_from, stats, weight_cache, True)

        visited: set[int] = {source}
        queue: deque[int] = deque([source])
        stats.nodes_generated = 1

        while queue:
            stats.max_frontier_size = max(stats.max_frontier_size, len(queue))
            if stats.nodes_expanded >= max_nodes:
                break
            node = queue.popleft()
            stats.nodes_expanded += 1
            if node == destination:
                return _finalize("bfs", G, source, destination, came_from, stats, weight_cache, True)
            for nbr in G.successors(node):
                if nbr in visited:
                    stats.revisits += 1
                    continue
                visited.add(nbr)
                came_from[nbr] = node
                queue.append(nbr)
                stats.nodes_generated += 1
    return _finalize("bfs", G, source, destination, came_from, stats, weight_cache, False)


def dfs_search(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    *,
    max_depth: int = 2_500,
    max_nodes: int = 200_000,
) -> SearchResult:
    """Iterative depth-first — depth-limited to avoid falling into pathological loops."""
    stats = SearchStats()
    came_from: dict[int, int] = {}
    with Timer(stats):
        if source == destination:
            stats.success = True
            return _finalize("dfs", G, source, destination, came_from, stats, weight_cache, True)

        visited: set[int] = {source}
        stack: list[tuple[int, int]] = [(source, 0)]
        stats.nodes_generated = 1

        while stack:
            stats.max_frontier_size = max(stats.max_frontier_size, len(stack))
            if stats.nodes_expanded >= max_nodes:
                break
            node, depth = stack.pop()
            stats.nodes_expanded += 1
            if node == destination:
                return _finalize("dfs", G, source, destination, came_from, stats, weight_cache, True)
            if depth >= max_depth:
                continue
            for nbr in G.successors(node):
                if nbr in visited:
                    stats.revisits += 1
                    continue
                visited.add(nbr)
                came_from[nbr] = node
                stack.append((nbr, depth + 1))
                stats.nodes_generated += 1
    return _finalize("dfs", G, source, destination, came_from, stats, weight_cache, False)


def ucs_search(
    G: nx.MultiDiGraph,
    source: int,
    destination: int,
    weight_cache: dict[tuple[int, int, int], float],
    *,
    max_nodes: int = 500_000,
) -> SearchResult:
    """Uniform Cost Search over the realistic cost model — Dijkstra's algorithm."""
    stats = SearchStats()
    came_from: dict[int, int] = {}
    with Timer(stats):
        g_score: dict[int, float] = {source: 0.0}
        heap: list[tuple[float, int, int]] = [(0.0, 0, source)]
        counter = 0
        stats.nodes_generated = 1

        while heap:
            stats.max_frontier_size = max(stats.max_frontier_size, len(heap))
            if stats.nodes_expanded >= max_nodes:
                break
            cost, _, node = heapq.heappop(heap)
            if cost > g_score.get(node, float("inf")):
                stats.revisits += 1
                continue
            stats.nodes_expanded += 1
            if node == destination:
                return _finalize("ucs", G, source, destination, came_from, stats, weight_cache, True)
            for nbr in G.successors(node):
                edge_c, _ = edge_cost_lookup(G, node, nbr, weight_cache)
                tentative = cost + edge_c
                if tentative < g_score.get(nbr, float("inf")):
                    g_score[nbr] = tentative
                    came_from[nbr] = node
                    counter += 1
                    heapq.heappush(heap, (tentative, counter, nbr))
                    stats.nodes_generated += 1
    return _finalize("ucs", G, source, destination, came_from, stats, weight_cache, False)
