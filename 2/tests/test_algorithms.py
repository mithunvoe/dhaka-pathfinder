"""Smoke tests + property tests for all 5 algorithms."""

from __future__ import annotations

import pytest

from fuel_csp.algorithms import (
    ALL_SOLVERS,
    BacktrackingForwardChecking,
    BacktrackingLCV,
    BacktrackingMRV,
    BasicBacktracking,
    MinConflictsSolver,
)
from fuel_csp.constraints import total_conflicts
from fuel_csp.synthetic import GeneratorConfig, generate_problem


@pytest.mark.parametrize("name", list(ALL_SOLVERS.keys()))
def test_all_solvers_register(name):
    assert name in ALL_SOLVERS


@pytest.mark.parametrize(
    "solver_cls",
    [BasicBacktracking, BacktrackingMRV, BacktrackingLCV,
     BacktrackingForwardChecking, MinConflictsSolver],
)
def test_returns_feasible_partial_assignment(solver_cls, small_problem):
    """Whatever subset of vehicles the solver assigns must violate zero hard constraints."""
    solver = solver_cls(time_budget_s=3.0)
    res = solver.solve(small_problem)
    assert total_conflicts(small_problem, res.assignment) == 0
    assert res.stats.runtime_seconds > 0


@pytest.mark.parametrize(
    "solver_cls",
    [BasicBacktracking, BacktrackingMRV, BacktrackingLCV,
     BacktrackingForwardChecking, MinConflictsSolver],
)
def test_stats_populated(solver_cls, small_problem):
    solver = solver_cls(time_budget_s=3.0)
    res = solver.solve(small_problem)
    s = res.stats
    assert s.n == small_problem.n
    assert s.num_assigned + s.num_unassigned == s.n
    assert s.algorithm == solver.name
    assert 0.0 <= s.failure_rate <= 1.0


def test_fc_at_least_matches_basic_quality(small_problem):
    """Forward-Checking + MRV+Deg should never produce a *worse* J(S) than basic BT
    on the same instance (the time budget is the same and FC strictly dominates).
    Allows equality."""
    a = BasicBacktracking(time_budget_s=3.0).solve(small_problem)
    b = BacktrackingForwardChecking(time_budget_s=3.0).solve(small_problem)
    assert b.stats.objective <= a.stats.objective + 1e-6


def test_min_conflicts_terminates_within_budget(small_problem):
    solver = MinConflictsSolver(max_steps=400, time_budget_s=2.0, seed=1)
    res = solver.solve(small_problem)
    assert res.stats.runtime_seconds <= 2.5
    assert res.stats.repair_steps >= 1


def test_min_conflicts_assignment_is_feasible(small_problem):
    solver = MinConflictsSolver(max_steps=600, time_budget_s=2.0, seed=1)
    res = solver.solve(small_problem)
    # _extract_feasible must guarantee no clashes/overcapacity
    assert total_conflicts(small_problem, res.assignment) == 0


def test_solver_handles_overconstrained_instance_without_crash():
    cfg = GeneratorConfig(num_vehicles=25, num_stations=2, num_slots=2, seed=99)
    p = generate_problem(cfg)
    for cls in (BasicBacktracking, BacktrackingForwardChecking, MinConflictsSolver):
        res = cls(time_budget_s=2.0).solve(p)
        assert 0 <= res.stats.num_assigned <= p.n
        assert total_conflicts(p, res.assignment) == 0  # COP graceful failure
