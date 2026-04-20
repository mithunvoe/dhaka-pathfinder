"""Pytest fixtures — build a tiny synthetic MultiDiGraph for algorithm tests."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel
from dhaka_pathfinder.synthetic_data import SyntheticConfig, augment_graph


@pytest.fixture
def toy_graph() -> nx.MultiDiGraph:
    """A 6-node grid-ish MultiDiGraph with Dhaka-like coordinates."""
    G = nx.MultiDiGraph()
    pts = {
        0: (23.80, 90.40),
        1: (23.80, 90.41),
        2: (23.80, 90.42),
        3: (23.81, 90.40),
        4: (23.81, 90.41),
        5: (23.81, 90.42),
    }
    for n, (lat, lon) in pts.items():
        G.add_node(n, x=lon, y=lat)

    def dist(a: int, b: int) -> float:
        lat1, lon1 = pts[a]
        lat2, lon2 = pts[b]
        return float(np.hypot(lat2 - lat1, lon2 - lon1) * 111_000)

    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (1, 4), (2, 5), (3, 1), (4, 2)]
    for u, v in edges:
        G.add_edge(u, v, highway="residential", length=dist(u, v))
        G.add_edge(v, u, highway="residential", length=dist(u, v))

    augment_graph(G, SyntheticConfig(seed=1, num_area_clusters=2))
    return G


@pytest.fixture
def context() -> TravelContext:
    return TravelContext(gender="female", social="alone", vehicle="cng", time_bucket="evening_rush")


@pytest.fixture
def cost_model() -> RealisticCostModel:
    return RealisticCostModel(weights="balanced", stochastic=False)
