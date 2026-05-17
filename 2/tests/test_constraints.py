"""Tests for the constraint checks + COP objective."""

from __future__ import annotations

import pytest

from fuel_csp.constraints import (
    conflicts,
    is_consistent,
    objective,
    per_variable_feasible,
    pump_clash,
    total_conflicts,
)
from fuel_csp.problem import Assignment
from fuel_csp.synthetic import GeneratorConfig, generate_problem


def test_pump_clash():
    a = Assignment(1, 0, 2)
    b = Assignment(1, 0, 2)
    c = Assignment(1, 1, 2)
    assert pump_clash(a, b)
    assert not pump_clash(a, c)


def test_is_consistent_rejects_pump_double_use(small_problem):
    p = small_problem
    # find two vehicles with at least one shared candidate value
    vid_a, vid_b = 0, 1
    shared = next(
        (a for a in p.domains[vid_a] if a in p.domains[vid_b]),
        None,
    )
    if shared is None:
        pytest.skip("No shared domain value in this seed.")
    assignment = {vid_a: shared}
    assert not is_consistent(p, assignment, vid_b, shared)


def test_per_variable_feasible_matches_domain(small_problem):
    p = small_problem
    for i, v in enumerate(p.vehicles):
        for a in p.domains[i]:
            assert per_variable_feasible(p, i, a)


def test_objective_empty_is_unassigned_penalty():
    p = generate_problem(GeneratorConfig(num_vehicles=5, seed=2))
    j = objective(p, {})
    # 5 unassigned * 100 baseline + priority penalties > 500
    assert j >= 500


def test_total_conflicts_is_zero_for_empty():
    p = generate_problem(GeneratorConfig(num_vehicles=4, seed=3))
    assert total_conflicts(p, {}) == 0


def test_conflicts_counts_clashes(small_problem):
    p = small_problem
    # two vehicles assigned to the same triple -> 1 pump clash
    v1, v2 = 0, 1
    a = p.domains[v1][0]
    if a not in p.domains[v2]:
        pytest.skip("seed-dependent setup")
    assignment = {v1: a, v2: a}
    assert conflicts(p, assignment, v2, a) >= 1
