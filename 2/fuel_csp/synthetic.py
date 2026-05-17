"""Synthetic-data generator.

Produces a deterministic-but-realistic instance from a (seed, N) pair.
Stations are deliberately scarce so the harder instances really do force
backtracking and force the COP into partial assignments.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from fuel_csp.problem import FUEL_TYPES, Problem, Station, Vehicle


@dataclass
class GeneratorConfig:
    """Tunable knobs for the data generator."""

    num_vehicles: int
    num_stations: int = 6
    num_slots: int = 6
    grid_size_km: float = 20.0
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


_KINDS_RNG: tuple[str, ...] = ("ambulance", "bus", "truck", "car", "motorbike")


def _pick_kind(rng: random.Random, cfg: GeneratorConfig) -> str:
    """Sample vehicle class given the configured mix."""
    r = rng.random()
    if r < cfg.ambulance_rate:
        return "ambulance"
    if r < cfg.ambulance_rate + cfg.bus_rate:
        return "bus"
    if r < cfg.ambulance_rate + cfg.bus_rate + cfg.truck_rate:
        return "truck"
    # remaining split car/motorbike 50/50
    return "car" if rng.random() < 0.5 else "motorbike"


def _fuel_for(kind: str, rng: random.Random) -> str:
    if kind == "ambulance":
        return "diesel"
    if kind == "bus":
        return "diesel"
    if kind == "truck":
        return "diesel" if rng.random() < 0.7 else "octane"
    if kind == "car":
        return rng.choice(("petrol", "octane"))
    # motorbike
    return "petrol"


def _demand_for(kind: str, rng: random.Random, cfg: GeneratorConfig) -> float:
    base = rng.uniform(cfg.demand_min, cfg.demand_max)
    scale = {
        "ambulance": 0.6,
        "bus": 1.5,
        "truck": 1.8,
        "car": 0.7,
        "motorbike": 0.25,
    }[kind]
    return round(base * scale, 1)


def _range_for(kind: str, rng: random.Random, cfg: GeneratorConfig) -> float:
    base = rng.uniform(cfg.range_min, cfg.range_max)
    if kind == "ambulance":  # emergency vehicles tend to be lower on fuel
        base *= 0.7
    if kind == "motorbike":
        base *= 0.9
    return round(base, 1)


def _slot_window(kind: str, rng: random.Random, num_slots: int) -> tuple[int, int]:
    if kind == "ambulance":  # must be served early
        return 0, max(1, num_slots // 3)
    lo = rng.randrange(0, max(1, num_slots // 2))
    hi = rng.randrange(lo, num_slots)
    return lo, hi


def generate_problem(cfg: GeneratorConfig) -> Problem:
    """Build a Problem instance for the given config."""
    rng = random.Random(cfg.seed)

    stations: list[Station] = []
    for sid in range(cfg.num_stations):
        x = rng.uniform(0, cfg.grid_size_km)
        y = rng.uniform(0, cfg.grid_size_km)
        pumps = rng.randint(cfg.pumps_min, cfg.pumps_max)
        open_slot = 0
        close_slot = cfg.num_slots
        reserves = {
            ft: round(rng.uniform(cfg.reserve_min, cfg.reserve_max), 1)
            for ft in FUEL_TYPES
        }
        # 25% chance one fuel type is "out" at this station -- forces constraint
        if rng.random() < 0.25:
            empty = rng.choice(FUEL_TYPES)
            reserves[empty] = 0.0
        stations.append(
            Station(
                sid=sid, x=x, y=y, pumps=pumps,
                open_slot=open_slot, close_slot=close_slot,
                reserves=reserves,
            )
        )

    vehicles: list[Vehicle] = []
    for vid in range(cfg.num_vehicles):
        kind = _pick_kind(rng, cfg)
        fuel = _fuel_for(kind, rng)
        x = rng.uniform(0, cfg.grid_size_km)
        y = rng.uniform(0, cfg.grid_size_km)
        rng_km = _range_for(kind, rng, cfg)
        demand = _demand_for(kind, rng, cfg)
        lo, hi = _slot_window(kind, rng, cfg.num_slots)
        vehicles.append(
            Vehicle(
                vid=vid, kind=kind, fuel_type=fuel,
                x=x, y=y,
                range_km=rng_km, demand_liters=demand,
                earliest_slot=lo, latest_slot=hi,
            )
        )

    problem = Problem(vehicles=vehicles, stations=stations, num_slots=cfg.num_slots)
    problem.build_domains()
    return problem
