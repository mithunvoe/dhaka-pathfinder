"""
Dhaka-grounded problem generator.

Wires the OSM data layer (osm_data.py) into the CSP/COP problem
formulation (problem.py). The result is a fully populated ``Problem``
instance with real lat/lon coordinates, real OSM node ids, and a
precomputed road-distance matrix.

This is a drop-in alternative to ``synthetic.generate_problem``: both
produce the same ``Problem`` schema, so every solver, the analyzer, and
the CLI work without modification.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx

from fuel_csp.osm_data import (
    StationPOI,
    compute_distance_matrix,
    extract_fuel_stations,
    load_dhaka_graph,
    sample_vehicle_nodes,
)
from fuel_csp.problem import (
    FUEL_TYPES,
    Problem,
    Station,
    Vehicle,
)
from fuel_csp.synthetic import (
    GeneratorConfig,
    _demand_for,
    _fuel_for,
    _pick_kind,
    _range_for,
    _slot_window,
)

logger = logging.getLogger(__name__)


@dataclass
class DhakaConfig:
    """Knobs for the Dhaka-grounded generator. Parallels ``GeneratorConfig``."""

    num_vehicles: int
    max_stations: int = 8
    num_slots: int = 6
    pumps_min: int = 1
    pumps_max: int = 3
    reserve_min: float = 80.0
    reserve_max: float = 220.0
    demand_min: float = 15.0
    demand_max: float = 45.0
    range_min: float = 8.0
    range_max: float = 22.0
    ambulance_rate: float = 0.10
    bus_rate: float = 0.15
    truck_rate: float = 0.15
    seed: int = 42


def _gen_cfg(cfg: DhakaConfig) -> GeneratorConfig:
    """Shim so we can re-use synthetic.py's per-vehicle samplers verbatim."""
    return GeneratorConfig(
        num_vehicles=cfg.num_vehicles,
        num_stations=cfg.max_stations,
        num_slots=cfg.num_slots,
        pumps_min=cfg.pumps_min,
        pumps_max=cfg.pumps_max,
        reserve_min=cfg.reserve_min,
        reserve_max=cfg.reserve_max,
        demand_min=cfg.demand_min,
        demand_max=cfg.demand_max,
        range_min=cfg.range_min,
        range_max=cfg.range_max,
        ambulance_rate=cfg.ambulance_rate,
        bus_rate=cfg.bus_rate,
        truck_rate=cfg.truck_rate,
        seed=cfg.seed,
    )


def _build_station(
    poi: StationPOI,
    rng: random.Random,
    cfg: DhakaConfig,
) -> Station:
    pumps = rng.randint(cfg.pumps_min, cfg.pumps_max)
    reserves = {
        ft: round(rng.uniform(cfg.reserve_min, cfg.reserve_max), 1)
        for ft in FUEL_TYPES
    }
    if rng.random() < 0.25:
        reserves[rng.choice(FUEL_TYPES)] = 0.0  # 25 % chance one fuel is out
    return Station(
        sid=poi.sid,
        x=poi.lon, y=poi.lat,        # back-compat fields use lon/lat
        pumps=pumps,
        open_slot=0,
        close_slot=cfg.num_slots,
        reserves=reserves,
        name=poi.name,
        lat=poi.lat, lon=poi.lon,
        node_id=poi.node_id,
    )


def _build_vehicle(
    veh_poi,
    rng: random.Random,
    cfg: DhakaConfig,
) -> Vehicle:
    gcfg = _gen_cfg(cfg)
    kind = _pick_kind(rng, gcfg)
    fuel = _fuel_for(kind, rng)
    demand = _demand_for(kind, rng, gcfg)
    rng_km = _range_for(kind, rng, gcfg)
    lo, hi = _slot_window(kind, rng, cfg.num_slots)
    return Vehicle(
        vid=veh_poi.vid,
        kind=kind,
        fuel_type=fuel,
        x=veh_poi.lon, y=veh_poi.lat,  # back-compat fields use lon/lat
        range_km=rng_km,
        demand_liters=demand,
        earliest_slot=lo,
        latest_slot=hi,
        lat=veh_poi.lat, lon=veh_poi.lon,
        node_id=veh_poi.node_id,
    )


def generate_dhaka_problem(
    cfg: DhakaConfig,
    graph: nx.MultiDiGraph | None = None,
) -> tuple[Problem, nx.MultiDiGraph]:
    """Build a Dhaka-grounded ``Problem`` plus the underlying road graph.

    Returns ``(problem, graph)`` so the UI can reuse the graph for shortest-
    path polyline drawing without reloading it.
    """
    rng = random.Random(cfg.seed)
    G = graph if graph is not None else load_dhaka_graph()

    station_pois = extract_fuel_stations(G, max_stations=cfg.max_stations)
    if not station_pois:
        raise RuntimeError("No fuel stations could be loaded — OSM and fallbacks both failed.")

    vehicle_pois = sample_vehicle_nodes(
        G, cfg.num_vehicles, seed=cfg.seed,
        avoid_nodes={s.node_id for s in station_pois},
    )

    stations = [_build_station(p, rng, cfg) for p in station_pois]
    vehicles = [_build_vehicle(v, rng, cfg) for v in vehicle_pois]

    logger.info("Computing road-distance matrix for %d × %d ...",
                len(vehicles), len(stations))
    distance_matrix = compute_distance_matrix(G, vehicle_pois, stations_to_pois(stations))

    problem = Problem(
        vehicles=vehicles,
        stations=stations,
        num_slots=cfg.num_slots,
        distance_matrix=distance_matrix,
    )
    problem.build_domains()
    return problem, G


def stations_to_pois(stations: list[Station]) -> list[StationPOI]:
    """Map our ``Station`` rows back to ``StationPOI`` for matrix helpers."""
    return [
        StationPOI(sid=s.sid, name=s.name, lat=s.lat or 0.0,
                   lon=s.lon or 0.0, node_id=s.node_id or 0)
        for s in stations
    ]
