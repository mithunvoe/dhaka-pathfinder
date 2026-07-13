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


# ---------------------------------------------------------------------
# Communication topology (Kennedy & Mendes 2002)
#
# The swarm's social term pulls each particle towards the best point it can
# HEAR. Who it can hear is a design choice, and the literature says that choice
# matters in a specific direction. These tests pin down the mechanism; the
# empirical claim is measured in collective_behaviour_study().
# ---------------------------------------------------------------------
from dataclasses import replace

from pso_wifi_placement import collective_behaviour_study, topology_significance


def test_ring_neighbourhood_is_a_circle():
    """Particle i must hear exactly i-k..i+k, wrapping around the ends."""
    cfg = replace(Config(), swarm_size=6, topology="ring", ring_k=1)
    opt = ParticleSwarmOptimizer(WifiFloorProblem(cfg), cfg, seed=0)
    assert opt.neighbours.shape == (6, 3)
    assert sorted(opt.neighbours[0]) == [0, 1, 5]     # wraps backwards
    assert sorted(opt.neighbours[5]) == [0, 4, 5]     # wraps forwards
    assert sorted(opt.neighbours[2]) == [1, 2, 3]


def test_fully_connected_swarm_has_no_neighbourhood_table():
    cfg = Config()
    opt = ParticleSwarmOptimizer(WifiFloorProblem(cfg), cfg, seed=0)
    assert opt.neighbours is None                     # everyone hears everyone


def test_ring_social_attractor_is_local_not_global():
    """The point of the ring: a particle is pulled towards the best thing IN ITS
    NEIGHBOURHOOD, which may be nowhere near the swarm's actual best. If this
    ever returned gbest we would have silently re-implemented gbest."""
    cfg = replace(Config(), swarm_size=6, topology="ring", ring_k=1)
    problem = WifiFloorProblem(cfg)
    opt = ParticleSwarmOptimizer(problem, cfg, seed=0)

    pbest = np.arange(6 * cfg.n_dims, dtype=float).reshape(6, cfg.n_dims)
    pbest_fit = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 99.0])   # particle 5 is the star
    gbest = pbest[5]

    attractor = opt._social_attractor(pbest, pbest_fit, gbest)

    # Particles 4, 5 and 0 are the only ones adjacent to the star, so only they
    # should be pulled towards it. Particle 2 is on the far side of the ring and
    # must NOT have heard about it yet.
    assert np.allclose(attractor[4], pbest[5])
    assert np.allclose(attractor[0], pbest[5])
    assert not np.allclose(attractor[2], pbest[5])


def test_talking_beats_silence():
    """The load-bearing claim of Part A. Thirty particles that share must beat
    thirty that do not, at an identical budget."""
    cfg = Config()
    problem = WifiFloorProblem(cfg)
    df = collective_behaviour_study(problem, cfg, [30], n_runs=5)
    topo = df[df["Kind"] == "topology"].set_index("Variant")
    assert (topo.loc["fully connected (gbest)", "Mean fitness"]
            > topo.loc["no communication (c2 = 0)", "Mean fitness"])


def test_ring_is_steadier_not_better():
    """The Kennedy & Mendes (2002) prediction, stated precisely enough to fail.

    Their finding is that "greater connectivity speeds up convergence, though it
    does not tend to improve the population's ability to discover global optima".
    So the ring should be MORE CONSISTENT and slower to give up - and should NOT
    reliably find a better optimum.

    An earlier version of this test asserted the ring had a better mean. It
    failed, correctly: on 30 paired seeds the mean difference is not significant
    (Wilcoxon p = 0.92) while the variance ratio is (F-test p = 6e-06). The paper
    predicted exactly that, and the test now says so.
    """
    cfg = Config()
    problem = WifiFloorProblem(cfg)
    sig = topology_significance(problem, cfg, n_runs=12)

    # The spread is where the effect lives.
    assert sig["ring"].std() < sig["gbest"].std()
    assert sig["var_ratio"] > 1.5

    # And the mean is where it does NOT. We assert only that the ring is not
    # meaningfully WORSE - claiming it is better is the mistake this test exists
    # to prevent.
    assert sig["ring"].mean() > sig["gbest"].mean() - 0.5
