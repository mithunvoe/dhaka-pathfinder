"""Tests for the Dhaka OSM-grounded mode.

These tests require:
* the osmnx-built graph cache (``data/dhaka_drive.pkl``) — built by running
  ``./run.sh ui`` or ``./run.sh experiments`` once with network access.

If the cache file is missing, every test in this file is skipped, so the
suite still runs cleanly offline. We do NOT attempt a fresh OSM download
from a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fuel_csp.algorithms import ALL_SOLVERS

CACHE = Path(__file__).resolve().parent.parent / "data" / "dhaka_drive.pkl"

pytestmark = pytest.mark.skipif(
    not CACHE.exists(),
    reason="Dhaka OSM cache not built yet — run `./run.sh ui` once with network.",
)


def _build_small_problem(n: int = 8, max_stations: int = 4, seed: int = 5):
    from fuel_csp.dhaka import DhakaConfig, generate_dhaka_problem
    return generate_dhaka_problem(
        DhakaConfig(num_vehicles=n, max_stations=max_stations, seed=seed)
    )


def test_dhaka_problem_has_real_coords():
    problem, _ = _build_small_problem()
    for s in problem.stations:
        assert 23.6 < s.lat < 23.9
        assert 90.3 < s.lon < 90.5
        assert s.node_id is not None
    for v in problem.vehicles:
        assert 23.6 < v.lat < 23.9
        assert 90.3 < v.lon < 90.5
        assert v.node_id is not None


def test_distance_matrix_is_positive_finite():
    problem, _ = _build_small_problem()
    assert problem.distance_matrix is not None
    finite = problem.distance_matrix[problem.distance_matrix < np.inf]
    # At least some pair must be reachable, and the road distance must be > 0
    assert finite.size > 0
    assert finite.min() > 0.0


def test_problem_mode_reports_dhaka():
    problem, _ = _build_small_problem()
    assert problem.mode == "dhaka"


def test_distance_km_uses_matrix_when_set():
    problem, _ = _build_small_problem()
    # vehicle 0 -> station 0: must equal matrix value / 1000
    d_km = problem.distance_km(0, 0)
    expected = problem.distance_matrix[0, 0] / 1000.0
    assert abs(d_km - expected) < 1e-9


@pytest.mark.parametrize(
    "solver_cls",
    [ALL_SOLVERS[n] for n in (
        "basic_backtracking", "bt_mrv", "bt_fc_mrv_deg", "min_conflicts",
    )],
)
def test_solvers_run_on_dhaka_instance(solver_cls):
    problem, _ = _build_small_problem()
    res = solver_cls(time_budget_s=2.0).solve(problem)
    # Either we placed something, or every vehicle was unreachable
    assert 0 <= res.stats.num_assigned <= problem.n
    # Distances on the assigned routes should match what the matrix says
    for vid, a in res.assignment.items():
        m = problem.distance_matrix[vid, a.station_id]
        assert m < np.inf  # no impossibly-far placements
