"""Heuristic functions h(n) for informed search.

Four families are implemented so the comparative analysis can sweep over them:

  1. zero              — h(n) = 0. Trivially admissible. Reduces A* to UCS (control case).
  2. haversine_admissible — great-circle distance × lowest-possible cost-per-meter under
                            the active context. Never overestimates, so admissible and
                            consistent. This is the admissible baseline required by §4.1.
  3. haversine_time    — great-circle distance divided by the vehicle's free-flow speed,
                         so h(n) approximates "seconds remaining" — a reasonably
                         effective but *not* admissible heuristic.
  4. context_aware     — haversine distance modulated by time-of-day and gender safety
                         multipliers. Non-admissible, meant to be realistic.
  5. learned_history   — uses aggregated `historical_incidents` on outgoing edges of the
                         candidate neighbourhood to bias h(n) — non-admissible, realistic.

Every heuristic exposes the same callable signature: `h(node) -> float`. They all get
bound to a specific goal node and the active TravelContext at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np

from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel, haversine_m


HeuristicFn = Callable[[int], float]


@dataclass(frozen=True)
class HeuristicInfo:
    name: str
    admissible: bool
    description: str


HEURISTIC_INFO: dict[str, HeuristicInfo] = {
    "zero": HeuristicInfo("zero", True, "h(n)=0, reduces A* to UCS (control)."),
    "haversine_admissible": HeuristicInfo(
        "haversine_admissible", True,
        "Great-circle distance × best-possible cost/meter under the active context."
    ),
    "network_relaxed": HeuristicInfo(
        "network_relaxed", True,
        "Reverse-Dijkstra shortest physical length from n to goal × best-per-metre — tighter than haversine."
    ),
    "haversine_time": HeuristicInfo(
        "haversine_time", False,
        "Distance divided by free-flow speed — optimistic in traffic, pessimistic on safety."
    ),
    "context_aware": HeuristicInfo(
        "context_aware", False,
        "Haversine, modulated by time-of-day and gender safety multipliers."
    ),
    "learned_history": HeuristicInfo(
        "learned_history", False,
        "Haversine scaled by average historical incidents in the neighbourhood."
    ),
}


def _coords(G: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    d = G.nodes[node]
    return float(d["y"]), float(d["x"])


def make_zero(*_: object, **__: object) -> HeuristicFn:
    def h(_node: int) -> float:
        return 0.0
    return h


def make_haversine_admissible(
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    cost_model: RealisticCostModel,
) -> HeuristicFn:
    """h(n) = haversine(n, goal) × best_possible_cost_per_meter."""
    lat_g, lon_g = _coords(G, goal)
    best = cost_model.best_possible_cost_per_meter(context)

    def h(node: int) -> float:
        lat, lon = _coords(G, node)
        return haversine_m(lat, lon, lat_g, lon_g) * best

    return h


def make_haversine_time(
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    _cost_model: RealisticCostModel,
) -> HeuristicFn:
    """h(n) ≈ distance / speed — optimistic on traffic, not admissible for realistic cost."""
    lat_g, lon_g = _coords(G, goal)
    speed_kmph = context.vehicle_speed
    speed_mps = max(speed_kmph * 1000.0 / 3600.0, 0.1)

    def h(node: int) -> float:
        lat, lon = _coords(G, node)
        return haversine_m(lat, lon, lat_g, lon_g) / speed_mps * 2.0

    return h


def make_context_aware(
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    cost_model: RealisticCostModel,
) -> HeuristicFn:
    """Haversine × best-per-meter × (time-risk × gender-safety) — usually overestimates."""
    lat_g, lon_g = _coords(G, goal)
    best = cost_model.best_possible_cost_per_meter(context)
    risk_mult = context.time_multipliers["risk"]
    gender_mult = context.gender_multiplier
    amp = max(risk_mult * gender_mult * 1.2, 1.0)

    def h(node: int) -> float:
        lat, lon = _coords(G, node)
        return haversine_m(lat, lon, lat_g, lon_g) * best * amp

    return h


def make_learned_history(
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    cost_model: RealisticCostModel,
) -> HeuristicFn:
    """Learn a per-cluster incident rate once, then bias h toward safer clusters."""
    cluster_incidents: dict[int, list[int]] = {}
    for _, _, d in G.edges(data=True):
        u = d.get("area_name", "default")
        cluster_incidents.setdefault(u, []).append(int(d.get("historical_incidents", 0)))
    cluster_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in cluster_incidents.items()}
    max_mean = max(cluster_mean.values()) if cluster_mean else 1.0

    lat_g, lon_g = _coords(G, goal)
    best = cost_model.best_possible_cost_per_meter(context)

    def h(node: int) -> float:
        lat, lon = _coords(G, node)
        base = haversine_m(lat, lon, lat_g, lon_g) * best
        area = G.nodes[node].get("area_name", "default")
        incident_factor = 1.0 + 0.5 * (cluster_mean.get(area, 0.0) / max(max_mean, 0.01))
        return base * incident_factor

    return h


_NETWORK_RELAXED_CACHE: dict[tuple[int, int], dict[int, float]] = {}


def _network_relaxed_distances(G: nx.MultiDiGraph, goal: int) -> dict[int, float]:
    """Precompute shortest physical distance (metres) from every node to `goal`.

    Cached per (graph id, goal) — the same physical topology can serve any number
    of contexts, since physical length does not depend on the traveller.
    """
    key = (id(G), int(goal))
    cached = _NETWORK_RELAXED_CACHE.get(key)
    if cached is not None:
        return cached
    G_rev = G.reverse(copy=False)
    shortest_m = nx.single_source_dijkstra_path_length(G_rev, goal, weight="length")
    _NETWORK_RELAXED_CACHE[key] = shortest_m
    return shortest_m


def make_network_relaxed(
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    cost_model: RealisticCostModel,
) -> HeuristicFn:
    """Reverse Dijkstra on PHYSICAL length only, then scale by best-per-metre.

    Because the true remaining cost on ANY path respects:
        true_cost(path) >= sum_of_lengths(path) × best_cost_per_metre(context)
    and the physical-length shortest path is the minimum of the first factor,
    the product is a strict lower bound on the true remaining cost. This
    heuristic is therefore admissible AND consistent (triangle inequality on
    physical length is preserved).

    The O(V log V) precomputation per *goal* is cached across contexts — only
    the best-per-metre scalar changes when the traveller context changes.
    """
    best = cost_model.best_possible_cost_per_meter(context)
    shortest_m = _network_relaxed_distances(G, goal)
    lat_g, lon_g = _coords(G, goal)

    def h(node: int) -> float:
        m = shortest_m.get(node)
        if m is None:
            lat, lon = _coords(G, node)
            m = haversine_m(lat, lon, lat_g, lon_g)
        return float(m) * best

    return h


HEURISTIC_FACTORIES: dict[str, Callable[..., HeuristicFn]] = {
    "zero": make_zero,
    "haversine_admissible": make_haversine_admissible,
    "network_relaxed": make_network_relaxed,
    "haversine_time": make_haversine_time,
    "context_aware": make_context_aware,
    "learned_history": make_learned_history,
}


def make_heuristic(
    name: str,
    G: nx.MultiDiGraph,
    goal: int,
    context: TravelContext,
    cost_model: RealisticCostModel,
) -> HeuristicFn:
    factory = HEURISTIC_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown heuristic {name!r}. Choose from {sorted(HEURISTIC_FACTORIES)}."
        )
    return factory(G, goal, context, cost_model)
