"""Cover the visualizer paths the main analyzer tests don't hit."""

from __future__ import annotations

from pathlib import Path

from fuel_csp.algorithms import MinConflictsSolver
from fuel_csp.algorithms.backtracking import BacktrackingForwardChecking
from fuel_csp.synthetic import GeneratorConfig, generate_problem
from fuel_csp.visualizer import (
    plot_min_conflicts_convergence,
    plot_problem_topology,
)


def test_plot_min_conflicts_convergence(tmp_path: Path):
    problem = generate_problem(GeneratorConfig(num_vehicles=8, seed=1))
    traces = []
    for s in (1, 2, 3):
        solver = MinConflictsSolver(max_steps=80, time_budget_s=1.0, seed=s)
        res = solver.solve(problem)
        traces.append(res.stats.cost_trace)
    out = plot_min_conflicts_convergence(traces, tmp_path / "convergence.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_topology(tmp_path: Path):
    problem = generate_problem(GeneratorConfig(num_vehicles=10, seed=1))
    solver = BacktrackingForwardChecking(time_budget_s=1.0)
    res = solver.solve(problem)
    out = plot_problem_topology(problem, res.assignment, tmp_path / "topology.png")
    assert out.exists()
    assert out.stat().st_size > 0
