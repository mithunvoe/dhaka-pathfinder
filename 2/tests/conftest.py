"""Shared fixtures."""

from __future__ import annotations

import pytest

from fuel_csp.synthetic import GeneratorConfig, generate_problem


@pytest.fixture
def small_problem():
    """A reproducible 10-vehicle instance."""
    return generate_problem(GeneratorConfig(num_vehicles=10, seed=7))


@pytest.fixture
def medium_problem():
    """A reproducible 20-vehicle instance."""
    return generate_problem(GeneratorConfig(num_vehicles=20, seed=13))
