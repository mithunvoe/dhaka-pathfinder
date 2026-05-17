"""Shared types and helpers for search algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import networkx as nx


@dataclass
class SearchStats:
    """Metrics gathered during one search run — used by the comparative analyser."""

    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    revisits: int = 0
    path_length_edges: int = 0
    path_length_meters: float = 0.0
    path_cost: float = 0.0
    predicted_cost_at_start: float = 0.0
    predicted_vs_actual_gap: float = 0.0
    runtime_seconds: float = 0.0
    depth: int = 0
    effective_branching_factor: float = 0.0
    heuristic_name: str = "n/a"
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SearchResult:
    """Final output of a search — carries both the path and full metrics."""

    algorithm: str
    source: int
    destination: int
    path: list[int] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)
    cost_trace: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "source": self.source,
            "destination": self.destination,
            "path": self.path,
            "stats": self.stats.as_dict(),
        }


class Timer:
    """Context manager that records wall-clock seconds into SearchStats."""

    def __init__(self, stats: SearchStats) -> None:
        self._stats = stats
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stats.runtime_seconds = perf_counter() - self._start


def reconstruct_path(came_from: dict[int, int], node: int) -> list[int]:
    """Walk parent-pointer map to produce a path from source to `node`."""
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def edge_cost_lookup(G: nx.MultiDiGraph, u: int, v: int, weight_cache: dict[tuple[int, int, int], float]) -> tuple[float, int]:
    """Cheapest parallel edge (u, v) and its key — handles MultiDiGraph correctly."""
    best_cost = float("inf")
    best_key = 0
    for key in G[u][v]:
        cost = weight_cache[(u, v, key)]
        if cost < best_cost:
            best_cost = cost
            best_key = key
    return best_cost, best_key


def path_cost(
    G: nx.MultiDiGraph,
    path: list[int],
    weight_cache: dict[tuple[int, int, int], float],
) -> tuple[float, float]:
    """(total realistic cost, total physical length in meters) for a path."""
    if not path or len(path) < 2:
        return 0.0, 0.0
    total_cost = 0.0
    total_m = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            return float("inf"), float("inf")
        cost, key = edge_cost_lookup(G, u, v, weight_cache)
        total_cost += cost
        total_m += float(G[u][v][key].get("length", 0.0))
    return total_cost, total_m


def effective_branching_factor(num_expanded: int, depth: int, tol: float = 1e-4) -> float:
    """Solve N+1 = 1 + b + b² + … + b^d for b via Newton iteration.

    Returns 1.0 when the equation is degenerate (depth 0 or 1-node path).
    """
    if num_expanded <= 1 or depth <= 0:
        return 1.0
    N = num_expanded
    b = max(1.1, (N) ** (1.0 / max(depth, 1)))
    for _ in range(60):
        b_next = b - (_geometric_sum(b, depth) - N) / max(_geometric_sum_deriv(b, depth), 1e-8)
        if abs(b_next - b) < tol:
            return float(max(b_next, 1.0))
        b = max(b_next, 1.0001)
    return float(b)


def _geometric_sum(b: float, d: int) -> float:
    if abs(b - 1.0) < 1e-9:
        return float(d + 1)
    return (b ** (d + 1) - 1.0) / (b - 1.0)


def _geometric_sum_deriv(b: float, d: int) -> float:
    if abs(b - 1.0) < 1e-9:
        return float(d * (d + 1) / 2)
    return ((d + 1) * b ** d * (b - 1.0) - (b ** (d + 1) - 1.0)) / ((b - 1.0) ** 2)
