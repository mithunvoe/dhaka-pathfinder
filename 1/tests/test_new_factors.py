"""Tests for the newly-added factors: age, weather, street width.

Every test here is an independent invariant — the cost function must still behave
"sensibly" after those factors were added.
"""

from __future__ import annotations

import networkx as nx
import pytest

from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel


@pytest.fixture
def edge(toy_graph: nx.MultiDiGraph) -> dict:
    u, v, key = next(iter(toy_graph.edges(keys=True)))
    return toy_graph[u][v][key]


def test_child_costs_more_than_adult_in_risky_context(edge: dict, cost_model: RealisticCostModel) -> None:
    """Kids walking home alone at night should be routed with more caution than adults."""
    adult = TravelContext(gender="female", social="alone", age="adult",
                          vehicle="walk", time_bucket="late_night")
    child = adult.with_age("child")
    assert cost_model.edge_cost(edge, child, deterministic=True) > cost_model.edge_cost(edge, adult, deterministic=True)


def test_elderly_costs_more_than_adult(edge: dict, cost_model: RealisticCostModel) -> None:
    adult = TravelContext(gender="male", social="alone", age="adult",
                          vehicle="walk", time_bucket="evening")
    elderly = adult.with_age("elderly")
    assert cost_model.edge_cost(edge, elderly, deterministic=True) > cost_model.edge_cost(edge, adult, deterministic=True)


def test_child_on_motorbike_is_heavily_penalised(edge: dict, cost_model: RealisticCostModel) -> None:
    """Age-vehicle restriction fires and penalises the cost by 2.5x."""
    child_bike = TravelContext(age="child", vehicle="motorbike", gender="male")
    child_car = TravelContext(age="child", vehicle="car", gender="male")
    c1 = cost_model.edge_cost(edge, child_bike, deterministic=True)
    c2 = cost_model.edge_cost(edge, child_car, deterministic=True)
    assert c1 > c2 * 1.5
    assert child_bike.vehicle_is_allowed is False
    assert child_car.vehicle_is_allowed is True


@pytest.mark.parametrize("bad_weather", ["rain", "fog", "storm"])
def test_bad_weather_is_costlier_than_clear(edge: dict, cost_model: RealisticCostModel, bad_weather: str) -> None:
    clear = TravelContext(weather="clear", vehicle="walk", time_bucket="evening")
    bad = clear.with_weather(bad_weather)
    assert cost_model.edge_cost(edge, bad, deterministic=True) > cost_model.edge_cost(edge, clear, deterministic=True)


def test_storm_is_costlier_than_rain(edge: dict, cost_model: RealisticCostModel) -> None:
    rain = TravelContext(weather="rain", vehicle="walk")
    storm = rain.with_weather("storm")
    assert cost_model.edge_cost(edge, storm, deterministic=True) > cost_model.edge_cost(edge, rain, deterministic=True)


def test_more_lanes_is_cheaper_for_car(cost_model: RealisticCostModel) -> None:
    """Wider multi-lane road is preferable to a narrow one for a car (same everything else)."""
    narrow = {"length": 500.0, "highway_class": "primary", "condition": 0.8,
              "traffic_base": 0.3, "risk": 0.2, "safety": 0.8, "lighting": 0.7,
              "water_logging_prob": 0.2, "crime_index": 0.2, "free_flow_speed": 40.0,
              "num_lanes": 2}
    wide = dict(narrow); wide["num_lanes"] = 6
    ctx = TravelContext(vehicle="car", time_bucket="midday")
    assert cost_model.edge_cost(wide, ctx, deterministic=True) < cost_model.edge_cost(narrow, ctx, deterministic=True)


def test_wide_road_is_costlier_for_pedestrian(cost_model: RealisticCostModel) -> None:
    """A 6-lane road is harder to cross for a walker than a 2-lane road."""
    narrow = {"length": 500.0, "highway_class": "primary", "condition": 0.8,
              "traffic_base": 0.3, "risk": 0.2, "safety": 0.8, "lighting": 0.7,
              "water_logging_prob": 0.2, "crime_index": 0.2, "free_flow_speed": 40.0,
              "num_lanes": 2}
    wide = dict(narrow); wide["num_lanes"] = 6
    ctx = TravelContext(vehicle="walk")
    assert cost_model.edge_cost(wide, ctx, deterministic=True) > cost_model.edge_cost(narrow, ctx, deterministic=True)


def test_child_walker_wide_road_double_penalty(cost_model: RealisticCostModel) -> None:
    """Child + walking + wide road should stack via AGE_PROFILE['child']['wide_road_penalty']."""
    narrow = {"length": 500.0, "highway_class": "primary", "condition": 0.8,
              "traffic_base": 0.3, "risk": 0.2, "safety": 0.8, "lighting": 0.7,
              "water_logging_prob": 0.2, "crime_index": 0.2, "free_flow_speed": 40.0,
              "num_lanes": 2}
    wide = dict(narrow); wide["num_lanes"] = 6
    adult_ctx = TravelContext(age="adult", vehicle="walk")
    child_ctx = TravelContext(age="child", vehicle="walk")
    adult_delta = cost_model.edge_cost(wide, adult_ctx, deterministic=True) - cost_model.edge_cost(narrow, adult_ctx, deterministic=True)
    child_delta = cost_model.edge_cost(wide, child_ctx, deterministic=True) - cost_model.edge_cost(narrow, child_ctx, deterministic=True)
    assert child_delta > adult_delta


def test_best_per_meter_is_still_a_lower_bound_in_every_context(toy_graph: nx.MultiDiGraph, cost_model: RealisticCostModel) -> None:
    """Crucial admissibility test — new factors must NOT break the lower-bound guarantee."""
    for age in ("adult", "child", "elderly"):
        for weather in ("clear", "rain", "storm"):
            for vehicle in ("walk", "rickshaw", "cng", "car", "motorbike"):
                ctx = TravelContext(age=age, weather=weather, vehicle=vehicle)
                best = cost_model.best_possible_cost_per_meter(ctx)
                for u, v, key, data in toy_graph.edges(keys=True, data=True):
                    actual = cost_model.edge_cost(data, ctx, deterministic=True)
                    lower_bound = data["length"] * best
                    assert lower_bound <= actual + 1e-3, (
                        f"Lower bound violated under age={age} weather={weather} vehicle={vehicle}"
                    )
