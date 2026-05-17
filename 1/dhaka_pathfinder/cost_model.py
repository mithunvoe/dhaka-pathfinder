"""Multi-factor cost model — combines road, contextual, and user factors.

Design:
  * Actual edge cost = base_length * (product of normalized multipliers)
      so physically longer edges still cost more; multipliers scale in [~0.5, ~3.5].
  * The model exposes both `edge_cost` (used as edge weight during traversal)
    and `optimistic_time_seconds` (used by the admissible heuristic).
  * A `predict_edge_cost` variant samples without stochastic traffic so heuristics
    can query the model deterministically.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from dhaka_pathfinder.config import (
    AGE_PROFILE,
    DEFAULT_WEIGHTS,
    CostWeights,
    LANES_DEFAULT,
    VEHICLE_HIGHWAY_SUITABILITY,
    WEATHER_PROFILE,
)
from dhaka_pathfinder.context import TravelContext

logger = logging.getLogger(__name__)


MIN_COST = 1e-3


WEIGHT_PRESETS: dict[str, CostWeights] = {
    "balanced": DEFAULT_WEIGHTS,
    "speed": CostWeights(
        length=0.5, road_condition=0.4, safety=0.3, risk=0.4,
        traffic=2.0, time_of_day=1.4, lighting=0.2, water_logging=0.3,
        gender_safety=0.4, social_context=0.3, vehicle_suitability=1.2, crime=0.3,
        age=0.5, weather=0.8, street_width=1.1,
    ),
    "safety": CostWeights(
        length=0.6, road_condition=0.7, safety=2.2, risk=2.5,
        traffic=0.7, time_of_day=0.9, lighting=1.4, water_logging=0.6,
        gender_safety=2.4, social_context=1.6, vehicle_suitability=0.8, crime=2.0,
        age=2.0, weather=1.4, street_width=0.9,
    ),
    "comfort": CostWeights(
        length=0.7, road_condition=2.0, safety=1.0, risk=1.0,
        traffic=1.4, time_of_day=1.0, lighting=1.0, water_logging=1.8,
        gender_safety=1.0, social_context=0.7, vehicle_suitability=1.6, crime=0.8,
        age=1.2, weather=1.6, street_width=1.1,
    ),
}


@dataclass
class CostBreakdown:
    """Per-edge cost decomposition — useful for the "predicted vs actual" reports."""

    base_length: float
    condition_mult: float
    traffic_mult: float
    risk_mult: float
    safety_mult: float
    lighting_mult: float
    water_log_mult: float
    gender_mult: float
    vehicle_mult: float
    crime_mult: float
    age_mult: float
    weather_mult: float
    street_width_mult: float
    total_multiplier: float
    cost: float
    estimated_time_s: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class RealisticCostModel:
    """Compute edge costs combining road-intrinsic, dynamic, and user factors."""

    def __init__(
        self,
        weights: CostWeights | str | None = None,
        stochastic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if weights is None:
            weights = DEFAULT_WEIGHTS
        if isinstance(weights, str):
            weights = WEIGHT_PRESETS.get(weights, DEFAULT_WEIGHTS)
        self.weights: CostWeights = weights
        self.stochastic = stochastic
        self._rng = rng or np.random.default_rng(7)

    def edge_cost(
        self,
        edge_data: dict[str, Any],
        context: TravelContext,
        *,
        deterministic: bool = False,
    ) -> float:
        """Scalar cost for a single edge — cheap and vectorisable."""
        return self.edge_breakdown(edge_data, context, deterministic=deterministic).cost

    def edge_breakdown(
        self,
        edge_data: dict[str, Any],
        context: TravelContext,
        *,
        deterministic: bool = False,
    ) -> CostBreakdown:
        base_length = max(float(edge_data.get("length", 1.0)), MIN_COST)
        condition = float(edge_data.get("condition", 0.7))
        traffic_base = float(edge_data.get("traffic_base", 0.4))
        risk = float(edge_data.get("risk", 0.2))
        safety = float(edge_data.get("safety", 0.8))
        lighting = float(edge_data.get("lighting", 0.6))
        water_log = float(edge_data.get("water_logging_prob", 0.3))
        crime = float(edge_data.get("crime_index", 0.3))
        highway = edge_data.get("highway_class", edge_data.get("highway", "residential"))
        if isinstance(highway, list):
            highway = highway[0] if highway else "residential"
        free_flow = float(edge_data.get("free_flow_speed", 25.0))
        num_lanes = int(edge_data.get("num_lanes", LANES_DEFAULT))

        w = self.weights
        tmult = context.time_multipliers
        wmult = context.weather_multipliers
        amult = context.age_profile

        traffic_live = traffic_base * tmult["traffic"] * wmult["traffic_amp"] * amult["traffic_amp"]
        if self.stochastic and not deterministic:
            traffic_live *= float(self._rng.uniform(0.85, 1.15))
        traffic_live = float(np.clip(traffic_live, 0.02, 3.5))

        effective_risk = float(np.clip(risk * tmult["risk"] * wmult["risk_amp"] * amult["risk_amp"], 0.01, 2.5))
        effective_safety = float(np.clip(safety * tmult["safety"], 0.05, 1.6))
        effective_lighting_input = lighting * tmult["lighting"]
        effective_lighting = float(np.clip(effective_lighting_input / wmult["lighting_amp"], 0.02, 1.2))
        effective_condition = float(np.clip(condition / wmult["condition_amp"], 0.05, 1.0))
        effective_water_log = float(np.clip(water_log * wmult["water_log_amp"], 0.01, 1.5))

        condition_mult = 1.0 + w.road_condition * (1.0 - effective_condition)
        traffic_mult = 1.0 + w.traffic * traffic_live
        risk_mult = 1.0 + w.risk * effective_risk
        safety_mult = 1.0 + w.safety * (1.0 - min(effective_safety, 1.0))
        lighting_mult = 1.0 + w.lighting * (1.0 - min(effective_lighting, 1.0))
        water_log_mult = 1.0 + w.water_logging * effective_water_log

        gender_factor = context.gender_multiplier
        gender_mult = 1.0 + w.gender_safety * (gender_factor - 1.0)
        gender_mult = max(gender_mult, 0.5)

        veh_suitability = VEHICLE_HIGHWAY_SUITABILITY.get(
            context.vehicle, VEHICLE_HIGHWAY_SUITABILITY["car"]
        ).get(highway, 0.7)
        veh_suitability = max(veh_suitability, 0.05)
        vehicle_mult = 1.0 + w.vehicle_suitability * (1.0 - veh_suitability)

        crime_mult = 1.0 + w.crime * crime * gender_factor * amult["crime_amp"]

        age_risk_bonus = (amult["risk_amp"] - 1.0) + (amult["crime_amp"] - 1.0) * 0.5
        age_mult = 1.0 + w.age * max(age_risk_bonus, 0.0)
        if not context.vehicle_is_allowed:
            age_mult *= 2.5

        if context.vehicle in ("car", "bus", "cng"):
            lane_benefit = np.clip((num_lanes - 2) * 0.05, 0.0, 0.25)
            width_mult = 1.0 - w.street_width * lane_benefit * 0.5
            width_mult = max(width_mult, 0.75)
        elif context.vehicle == "motorbike":
            lane_benefit = np.clip((num_lanes - 2) * 0.03, 0.0, 0.15)
            width_mult = 1.0 - w.street_width * lane_benefit * 0.5
            width_mult = max(width_mult, 0.85)
        elif context.vehicle == "walk":
            wide_penalty = max(0, num_lanes - 3) * 0.1
            width_mult = 1.0 + w.street_width * wide_penalty * amult["wide_road_penalty"]
        elif context.vehicle == "rickshaw":
            rickshaw_penalty = max(0, num_lanes - 4) * 0.08
            width_mult = 1.0 + w.street_width * rickshaw_penalty
        else:
            width_mult = 1.0

        weather_direct_penalty = (
            (wmult["water_log_amp"] - 1.0) * 0.3
            + (wmult["risk_amp"] - 1.0) * 0.4
            + (wmult["lighting_amp"] - 1.0) * 0.2
        )
        weather_mult = 1.0 + w.weather * max(weather_direct_penalty, 0.0)

        total_multiplier = (
            condition_mult
            * traffic_mult
            * risk_mult
            * safety_mult
            * lighting_mult
            * water_log_mult
            * gender_mult
            * vehicle_mult
            * crime_mult
            * age_mult
            * weather_mult
            * width_mult
        )

        cost = max(base_length * w.length * total_multiplier, MIN_COST)

        weather_speed_factor = max(2.0 - wmult["traffic_amp"], 0.4)
        effective_speed_kmph = (
            min(context.vehicle_speed, free_flow) * weather_speed_factor
            / (1.0 + traffic_live)
        )
        effective_speed_kmph = max(effective_speed_kmph, 0.5)
        estimated_time_s = base_length / (effective_speed_kmph * 1000.0 / 3600.0)

        return CostBreakdown(
            base_length=base_length,
            condition_mult=condition_mult,
            traffic_mult=traffic_mult,
            risk_mult=risk_mult,
            safety_mult=safety_mult,
            lighting_mult=lighting_mult,
            water_log_mult=water_log_mult,
            gender_mult=gender_mult,
            vehicle_mult=vehicle_mult,
            crime_mult=crime_mult,
            age_mult=age_mult,
            weather_mult=weather_mult,
            street_width_mult=float(width_mult),
            total_multiplier=float(total_multiplier),
            cost=float(cost),
            estimated_time_s=float(estimated_time_s),
        )

    def precompute_edge_weights(
        self, G: nx.MultiDiGraph, context: TravelContext, *, deterministic: bool = True,
    ) -> dict[tuple[int, int, int], float]:
        """Snapshot of edge weights under `context` — used for deterministic reruns."""
        out: dict[tuple[int, int, int], float] = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            out[(u, v, key)] = self.edge_cost(data, context, deterministic=deterministic)
        return out

    def best_possible_cost_per_meter(self, context: TravelContext) -> float:
        """Lower bound on edge cost per meter — used by the admissible heuristic.

        Walks through the cost model with the MOST favourable attributes possible
        under the *active* context, guaranteeing the result never overestimates
        true remaining cost. Each contextual amplifier (time / weather / age) is
        applied here because it can only make the lower bound smaller, not larger.
        """
        tmult = context.time_multipliers
        wmult = context.weather_multipliers
        amult = context.age_profile
        w = self.weights

        condition_mult = 1.0
        traffic_mult = 1.0 + w.traffic * max(0.02 * tmult["traffic"] * wmult["traffic_amp"] * amult["traffic_amp"], 0.0)
        risk_mult = 1.0 + w.risk * 0.01 * wmult["risk_amp"] * amult["risk_amp"]
        safety_mult = 1.0
        lighting_mult = 1.0
        water_log_mult = 1.0

        gender_factor = min(context.gender_multiplier, 1.0)
        gender_mult = max(1.0 + w.gender_safety * (gender_factor - 1.0), 0.5)

        best_suitability = max(
            VEHICLE_HIGHWAY_SUITABILITY.get(context.vehicle, {}).values(),
            default=1.0,
        )
        vehicle_mult = 1.0 + w.vehicle_suitability * (1.0 - best_suitability)

        crime_mult = 1.0 + w.crime * 0.02 * gender_factor * amult["crime_amp"]

        age_risk_bonus = (amult["risk_amp"] - 1.0) + (amult["crime_amp"] - 1.0) * 0.5
        age_mult = 1.0 + w.age * max(age_risk_bonus, 0.0)

        weather_direct_penalty = (
            (wmult["water_log_amp"] - 1.0) * 0.3
            + (wmult["risk_amp"] - 1.0) * 0.4
            + (wmult["lighting_amp"] - 1.0) * 0.2
        )
        weather_mult = 1.0 + w.weather * max(weather_direct_penalty, 0.0)

        if context.vehicle in ("car", "bus", "cng"):
            width_mult = max(1.0 - w.street_width * 0.25 * 0.5, 0.75)
        elif context.vehicle == "motorbike":
            width_mult = max(1.0 - w.street_width * 0.15 * 0.5, 0.85)
        else:
            width_mult = 1.0

        total_multiplier = (
            condition_mult * traffic_mult * risk_mult * safety_mult
            * lighting_mult * water_log_mult * gender_mult * vehicle_mult * crime_mult
            * age_mult * weather_mult * width_mult
        )
        total_multiplier = max(total_multiplier, 0.02)
        return float(w.length * total_multiplier)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (standard Haversine)."""
    R = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))
