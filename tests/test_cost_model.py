"""Tests for RealisticCostModel."""

from __future__ import annotations

import networkx as nx
import pytest

from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel, haversine_m


def test_cost_breakdown_components(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext) -> None:
    u, v, key = next(iter(toy_graph.edges(keys=True)))
    breakdown = cost_model.edge_breakdown(toy_graph[u][v][key], context)
    assert breakdown.cost > 0
    assert breakdown.base_length == pytest.approx(toy_graph[u][v][key]["length"])
    assert breakdown.total_multiplier > 0
    assert breakdown.estimated_time_s > 0


def test_female_alone_late_night_costlier_than_male_accompanied_midday(
    toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel,
) -> None:
    safe_ctx = TravelContext(gender="male", social="accompanied", vehicle="car", time_bucket="midday")
    risky_ctx = TravelContext(gender="female", social="alone", vehicle="walk", time_bucket="late_night")
    u, v, key = next(iter(toy_graph.edges(keys=True)))
    assert (
        cost_model.edge_cost(toy_graph[u][v][key], risky_ctx, deterministic=True)
        > cost_model.edge_cost(toy_graph[u][v][key], safe_ctx, deterministic=True)
    )


def test_best_possible_cost_per_meter_is_lower_bound(
    toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel, context: TravelContext,
) -> None:
    best = cost_model.best_possible_cost_per_meter(context)
    assert best > 0
    for u, v, key, data in toy_graph.edges(keys=True, data=True):
        actual = cost_model.edge_cost(data, context, deterministic=True)
        lower_bound = data["length"] * best
        assert lower_bound <= actual + 1e-3, (
            f"Lower bound {lower_bound} exceeded actual {actual} on edge ({u},{v},{key})"
        )


def test_haversine_symmetric() -> None:
    d1 = haversine_m(23.7, 90.4, 23.8, 90.5)
    d2 = haversine_m(23.8, 90.5, 23.7, 90.4)
    assert d1 == pytest.approx(d2, rel=1e-6)
    assert d1 > 14_000
