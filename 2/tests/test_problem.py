"""Tests for the problem formulation + synthetic data generator."""

from __future__ import annotations

from fuel_csp.problem import FUEL_TYPES, PRIORITY, euclid
from fuel_csp.synthetic import GeneratorConfig, generate_problem


def test_problem_has_requested_size():
    p = generate_problem(GeneratorConfig(num_vehicles=15, seed=1))
    assert p.n == 15
    assert len(p.vehicles) == 15
    assert len(p.stations) == GeneratorConfig(num_vehicles=15).num_stations


def test_priorities_have_ambulance_top():
    assert PRIORITY["ambulance"] == max(PRIORITY.values())


def test_fuel_types_are_three():
    assert len(FUEL_TYPES) == 3


def test_domains_built_and_filtered(small_problem):
    p = small_problem
    assert len(p.domains) == p.n
    for i, v in enumerate(p.vehicles):
        for a in p.domains[i]:
            s = p.stations[a.station_id]
            assert s.stocks(v.fuel_type) >= v.demand_liters
            assert euclid(v.x, v.y, s.x, s.y) <= v.range_km
            assert v.earliest_slot <= a.slot_id <= v.latest_slot
            assert 0 <= a.pump_id < s.pumps


def test_problem_is_deterministic():
    p1 = generate_problem(GeneratorConfig(num_vehicles=20, seed=42))
    p2 = generate_problem(GeneratorConfig(num_vehicles=20, seed=42))
    assert [v.kind for v in p1.vehicles] == [v.kind for v in p2.vehicles]
    assert [s.reserves for s in p1.stations] == [s.reserves for s in p2.stations]


def test_ambulance_window_is_early():
    p = generate_problem(GeneratorConfig(num_vehicles=40, seed=11))
    for v in p.vehicles:
        if v.kind == "ambulance":
            assert v.earliest_slot == 0
            assert v.latest_slot <= p.num_slots
