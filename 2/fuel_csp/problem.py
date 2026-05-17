"""
Formal CSP / COP definition for the urban fuel-crisis allocator.

Variables  X  : one variable per vehicle (X = {x_1, ..., x_N})
Domain    D_i : feasible (station_id, pump_id, slot_id) triples for vehicle i,
                pre-filtered by compatibility, reachability, and time-window
                hard-constraints (4.1.1, 4.1.2, 4.1.5).
Constraints C : pump-exclusivity and supply-capacity are checked dynamically
                during search (constraints.py).

Soft objective J(S) (COP) is computed in objective.py. Algorithms that solve
the CSP form ignore J(S); algorithms that solve the COP use it for ordering
and for measuring solution quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

FUEL_TYPES: tuple[str, ...] = ("petrol", "diesel", "octane")

VEHICLE_CLASSES: tuple[str, ...] = (
    "ambulance",
    "bus",
    "truck",
    "car",
    "motorbike",
)

PRIORITY: dict[str, int] = {
    "ambulance": 5,
    "bus": 3,
    "truck": 2,
    "car": 1,
    "motorbike": 1,
}


@dataclass(frozen=True)
class Vehicle:
    """A vehicle that needs refueling.

    Coordinates are in an abstract Euclidean km grid (synthetic city).
    """

    vid: int
    kind: str
    fuel_type: str
    x: float
    y: float
    range_km: float
    demand_liters: float
    earliest_slot: int
    latest_slot: int

    @property
    def priority(self) -> int:
        return PRIORITY[self.kind]


@dataclass(frozen=True)
class Station:
    """A fuel station with limited reserves per fuel type and a set of pumps."""

    sid: int
    x: float
    y: float
    pumps: int
    open_slot: int
    close_slot: int
    reserves: dict[str, float]  # liters available per fuel type

    def stocks(self, fuel: str) -> float:
        return self.reserves.get(fuel, 0.0)


@dataclass(frozen=True)
class Assignment:
    """A value in a variable's domain — one (station, pump, slot) triple."""

    station_id: int
    pump_id: int
    slot_id: int

    def __str__(self) -> str:
        return f"S{self.station_id}/P{self.pump_id}@T{self.slot_id}"


@dataclass
class Problem:
    """Container for a single CSP/COP instance.

    The instance is immutable after construction; solver state lives elsewhere.
    """

    vehicles: list[Vehicle]
    stations: list[Station]
    num_slots: int
    domains: list[list[Assignment]] = field(default_factory=list)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1.0,
            "wait": 0.5,
            "priority": 10.0,
            "unassigned": 100.0,
        }
    )

    @property
    def n(self) -> int:
        return len(self.vehicles)

    def build_domains(self) -> None:
        """Construct each variable's domain after applying per-variable hard constraints."""
        self.domains = [self._domain_for(i) for i in range(self.n)]

    def _domain_for(self, i: int) -> list[Assignment]:
        v = self.vehicles[i]
        out: list[Assignment] = []
        for s in self.stations:
            if s.stocks(v.fuel_type) < v.demand_liters:
                continue
            if euclid(v.x, v.y, s.x, s.y) > v.range_km:
                continue
            slot_lo = max(v.earliest_slot, s.open_slot)
            slot_hi = min(v.latest_slot, s.close_slot - 1)
            if slot_lo > slot_hi:
                continue
            for slot in range(slot_lo, slot_hi + 1):
                for pump in range(s.pumps):
                    out.append(Assignment(s.sid, pump, slot))
        return out


def euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def iter_pairs(seq: Iterable[int]) -> Iterable[tuple[int, int]]:
    """Yield (i, j) with i < j over a sequence."""
    items = list(seq)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
