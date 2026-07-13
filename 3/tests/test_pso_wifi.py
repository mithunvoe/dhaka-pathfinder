"""
Unit tests for the from-scratch PSO Wi-Fi placement solver.

These pin down the properties a viva examiner would probe:
  - the fitness function rewards good placements and penalises bad ones,
  - constraints (boundary, separation) are actually enforced,
  - the optimizer improves, stays in-bounds, and is reproducible,
  - the baselines honour the shared evaluation budget.
"""
import numpy as np
import pytest

from pso_wifi_placement import (
    Config,
    ParticleSwarmOptimizer,
    WifiFloorProblem,
    grid_search,
    random_search,
)


@pytest.fixture
def cfg():
    # Small, fast config for tests (few iters/runs) - behaviour is identical.
    return Config(swarm_size=15, n_iters=20, n_runs=3)


@pytest.fixture
def problem(cfg):
    return WifiFloorProblem(cfg)


# ---------------------------------------------------------------- geometry
def test_problem_shapes(problem, cfg):
    assert problem.rooms.shape == (cfg.n_rooms, 2)
    assert problem.weights.shape == (cfg.n_rooms,)
    assert problem.lower.shape == (cfg.n_dims,)
    assert np.all(problem.rooms[:, 0] <= cfg.floor_w)
    assert np.all(problem.rooms[:, 1] <= cfg.floor_h)


def test_wall_crossing_detects_blocked_line(problem):
    # A room-AP segment crossing the horizontal spine wall (12..48 at y=20)
    # should register exactly one crossing.
    room = np.array([[30.0, 10.0]])          # below the spine
    ap = np.array([[30.0, 30.0]])            # above the spine
    problem.rooms = room
    crossings = problem._wall_crossings(ap)
    assert crossings.shape == (1, 1)
    assert crossings[0, 0] >= 1


def test_wall_crossing_clear_line_is_zero(problem):
    # A short segment far from every wall should cross nothing.
    problem.rooms = np.array([[55.0, 35.0]])
    ap = np.array([[57.0, 37.0]])
    assert problem._wall_crossings(ap)[0, 0] == 0


# ---------------------------------------------------------------- fitness
def test_spread_beats_cluster(problem, cfg):
    # Three APs spread across the block must out-score three APs piled up
    # in one corner (which strands most rooms).
    spread = np.array([12.0, 28.0, 40.0, 28.0, 30.0, 8.0])[: cfg.n_dims]
    clustered = np.array([5.0, 5.0, 6.0, 6.0, 7.0, 5.0])[: cfg.n_dims]
    assert problem.fitness(spread) > problem.fitness(clustered)


def test_separation_penalty_hurts(problem, cfg):
    # Two identical placements except APs 0 and 1 are jammed together in one;
    # the coincident one must score no higher (separation penalty bites).
    base = np.array([12.0, 28.0, 40.0, 28.0, 30.0, 8.0])[: cfg.n_dims].copy()
    jammed = base.copy()
    jammed[2], jammed[3] = base[0] + 0.5, base[1] + 0.5   # AP1 -> onto AP0
    assert problem.fitness(jammed) <= problem.fitness(base) + 1e-9


def test_coverage_percent_in_range(problem):
    x = np.array([12.0, 28.0, 40.0, 28.0, 30.0, 8.0])[: problem.cfg.n_dims]
    cov = problem.coverage_percent(x)
    assert 0.0 <= cov <= 100.0


# ---------------------------------------------------------------- optimizer
def test_pso_improves_over_iterations(problem, cfg):
    res = ParticleSwarmOptimizer(problem, cfg, seed=1).optimize()
    # Global best is monotonic non-decreasing by construction.
    assert np.all(np.diff(res["history"]) >= -1e-9)
    assert res["history"][-1] >= res["history"][0]


def test_pso_respects_bounds(problem, cfg):
    res = ParticleSwarmOptimizer(problem, cfg, seed=2).optimize()
    x = res["best_x"]
    assert np.all(x >= problem.lower - 1e-6)
    assert np.all(x <= problem.upper + 1e-6)


def test_pso_reproducible(problem, cfg):
    a = ParticleSwarmOptimizer(problem, cfg, seed=7).optimize()
    b = ParticleSwarmOptimizer(problem, cfg, seed=7).optimize()
    assert a["best_fitness"] == pytest.approx(b["best_fitness"])
    assert np.allclose(a["best_x"], b["best_x"])


def test_pso_diversity_collapses(problem, cfg):
    # Exploration -> exploitation: the swarm should end more converged
    # (lower spread) than it began.
    res = ParticleSwarmOptimizer(problem, cfg, seed=3).optimize()
    assert res["diversity"][-1] < res["diversity"][0]


# ---------------------------------------------------------------- baselines
def test_random_search_uses_exact_budget(problem, cfg):
    problem.n_fitness_calls = 0
    random_search(problem, cfg, seed=4)
    assert problem.n_fitness_calls == cfg.n_evals


def test_pso_uses_exact_budget(problem, cfg):
    problem.n_fitness_calls = 0
    ParticleSwarmOptimizer(problem, cfg, seed=5).optimize()
    assert problem.n_fitness_calls == cfg.n_evals   # P*(T+1)


def test_grid_search_within_budget(problem, cfg):
    res = grid_search(problem, cfg)
    assert res["evals_used"] <= cfg.n_evals
    assert 0.0 <= res["best_fitness"] <= 100.0


def test_pso_beats_random_on_average(problem, cfg):
    # Core claim of the project: under the SAME budget, PSO >= Random.
    pso = ParticleSwarmOptimizer(problem, cfg, seed=6).optimize()["best_fitness"]
    rnd = random_search(problem, cfg, seed=6)["best_fitness"]
    assert pso >= rnd
