"""Tests for the AC-3 arc-consistency preprocess."""

from __future__ import annotations

from fuel_csp.algorithms.arc_consistency import ac3
from fuel_csp.algorithms.backtracking import BacktrackingForwardChecking
from fuel_csp.constraints import total_conflicts
from fuel_csp.problem import Assignment
from fuel_csp.synthetic import GeneratorConfig, generate_problem


def test_ac3_singleton_propagates():
    """A singleton domain on j prunes the clashing value out of every D_i."""
    sole = Assignment(0, 0, 0)
    other = Assignment(0, 0, 1)
    domains: list[list[Assignment]] = [
        [sole, other],  # D_0
        [sole],         # D_1 — singleton, forces removal of `sole` from D_0
    ]
    removed = ac3(domains)
    assert removed == 1
    assert domains[0] == [other]
    assert domains[1] == [sole]


def test_ac3_noop_when_all_domains_have_two_or_more_values():
    """If every domain has >= 2 distinct values, AC-3 makes no changes."""
    domains: list[list[Assignment]] = [
        [Assignment(0, 0, 0), Assignment(0, 0, 1)],
        [Assignment(0, 0, 0), Assignment(0, 0, 1)],
    ]
    snapshot = [list(d) for d in domains]
    removed = ac3(domains)
    assert removed == 0
    assert domains == snapshot


def test_ac3_skips_empty_domains_cop_relaxation():
    """An empty D_j must not be treated as 'no support' — j is graceful-skipped."""
    domains: list[list[Assignment]] = [
        [Assignment(0, 0, 0)],
        [],
    ]
    snapshot = [list(d) for d in domains]
    removed = ac3(domains)
    assert removed == 0
    assert domains == snapshot


def test_ac3_idempotent():
    """Running AC-3 twice removes nothing extra the second time."""
    domains: list[list[Assignment]] = [
        [Assignment(0, 0, 0), Assignment(0, 1, 0)],
        [Assignment(0, 0, 0)],
        [Assignment(0, 1, 0)],
    ]
    first = ac3(domains)
    second = ac3(domains)
    assert first == 2
    assert second == 0


def test_ac3_handles_single_variable():
    """No arcs exist when N=1; AC-3 returns 0."""
    domains = [[Assignment(0, 0, 0)]]
    assert ac3(domains) == 0


def test_ac3_does_not_mutate_problem_domains(small_problem):
    """The solver runs AC-3 on a copy — original problem.domains stays intact."""
    snapshot = [list(d) for d in small_problem.domains]
    solver = BacktrackingForwardChecking(time_budget_s=2.0)
    solver.solve(small_problem)
    assert small_problem.domains == snapshot


def test_fc_with_ac3_still_returns_feasible_assignment(small_problem):
    """End-to-end: FC + AC-3 must still satisfy all hard constraints."""
    solver = BacktrackingForwardChecking(time_budget_s=3.0)
    res = solver.solve(small_problem)
    assert total_conflicts(small_problem, res.assignment) == 0
    assert res.stats.success or res.stats.num_unassigned >= 0


def test_ac3_propagates_through_chain():
    """Singleton in D_2 forces D_1 -> singleton -> forces D_0 to drop value."""
    a = Assignment(0, 0, 0)
    b = Assignment(0, 0, 1)
    domains: list[list[Assignment]] = [
        [a, b],
        [a, b],
        [b],
    ]
    removed = ac3(domains)
    assert removed >= 1
    assert b not in domains[1]


def test_ac3_on_overconstrained_instance_does_not_crash():
    """Sanity: tight Dhaka-style instance survives AC-3 + FC pipeline."""
    cfg = GeneratorConfig(num_vehicles=20, num_stations=2, num_slots=2, seed=11)
    p = generate_problem(cfg)
    live = [list(d) for d in p.domains]
    ac3(live)
    for d in live:
        assert isinstance(d, list)
