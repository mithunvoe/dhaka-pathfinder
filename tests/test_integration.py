"""Integration tests against the real Dhaka graph.

These are marked `slow` so regular test runs can skip them; run with
`pytest -m slow` when you want the full proof.
"""

from __future__ import annotations

import heapq
import random

import networkx as nx
import pytest

from dhaka_pathfinder.algorithms import ALGORITHMS
from dhaka_pathfinder.algorithms.base import edge_cost_lookup
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.engine import DhakaPathfinderEngine
from dhaka_pathfinder.heuristics import make_heuristic


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def loaded_engine() -> DhakaPathfinderEngine:
    engine = DhakaPathfinderEngine()
    engine.load()
    return engine


def _true_shortest(G: nx.MultiDiGraph, goal: int, weights: dict) -> dict[int, float]:
    """Reverse Dijkstra from goal — true realistic cost from each node to goal."""
    Gr = G.reverse(copy=False)
    best = {goal: 0.0}
    heap = [(0.0, goal)]
    while heap:
        c, n = heapq.heappop(heap)
        if c > best.get(n, float("inf")):
            continue
        for nbr in Gr.successors(n):
            for key in Gr[n][nbr]:
                ec = weights.get((nbr, n, key))
                if ec is None:
                    continue
                nc = c + ec
                if nc < best.get(nbr, float("inf")):
                    best[nbr] = nc
                    heapq.heappush(heap, (nc, nbr))
    return best


def test_admissible_heuristic_on_real_graph(loaded_engine: DhakaPathfinderEngine) -> None:
    """On the real Dhaka graph: admissible heuristic must never exceed true cost."""
    G = loaded_engine.graph
    ctx = TravelContext(gender="female", social="alone", vehicle="cng", time_bucket="evening_rush")
    weights = loaded_engine._weights_for(ctx)

    rng = random.Random(11)
    nodes = list(G.nodes())
    goal = rng.choice(nodes)
    true_costs = _true_shortest(G, goal, weights)

    h = make_heuristic("haversine_admissible", G, goal, ctx, loaded_engine.cost_model)

    sample = rng.sample([n for n in nodes if n in true_costs], 200)
    for n in sample:
        lb = h(n)
        truth = true_costs[n]
        assert lb <= truth + 1e-3, f"Admissibility violated at node {n}: h={lb:.2f} truth={truth:.2f}"


def test_ucs_and_astar_agree_on_real_graph(loaded_engine: DhakaPathfinderEngine) -> None:
    """UCS and A* (admissible) must agree on path cost for random pairs."""
    G = loaded_engine.graph
    ctx = TravelContext(gender="male", social="alone", vehicle="car", time_bucket="midday")
    weights = loaded_engine._weights_for(ctx)

    rng = random.Random(3)
    nodes = list(G.nodes())

    matched = 0
    trials = 0
    for _ in range(6):
        s, d = rng.sample(nodes, 2)
        ucs_r = ALGORITHMS["ucs"](G, s, d, weights)
        if not ucs_r.stats.success:
            continue
        h = make_heuristic("haversine_admissible", G, d, ctx, loaded_engine.cost_model)
        astar_r = ALGORITHMS["astar"](G, s, d, weights, heuristic=h)
        if astar_r.stats.success:
            trials += 1
            if abs(ucs_r.stats.path_cost - astar_r.stats.path_cost) < 1e-3:
                matched += 1
    assert trials >= 3, "Too few successful trials."
    assert matched == trials, f"UCS/A* cost mismatch: {matched}/{trials}"


def test_all_algorithms_reach_destination(loaded_engine: DhakaPathfinderEngine) -> None:
    """All 6 algorithms should find *some* path on a close pair."""
    G = loaded_engine.graph
    ctx = TravelContext()
    weights = loaded_engine._weights_for(ctx)

    rng = random.Random(77)
    nodes = list(G.nodes())

    for _ in range(5):
        s, d = rng.sample(nodes, 2)
        ucs_r = ALGORITHMS["ucs"](G, s, d, weights)
        if not ucs_r.stats.success:
            continue
        for algo in ALGORITHMS:
            if algo in ("greedy", "astar", "weighted_astar"):
                h = make_heuristic("haversine_admissible", G, d, ctx, loaded_engine.cost_model)
                r = ALGORITHMS[algo](G, s, d, weights, heuristic=h)
            else:
                r = ALGORITHMS[algo](G, s, d, weights)
            assert r.stats.success, f"{algo} failed on pair {s}->{d}"
        return
    pytest.skip("Could not find a reachable pair.")
