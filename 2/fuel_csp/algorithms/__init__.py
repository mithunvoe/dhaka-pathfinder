"""Solver implementations."""

from fuel_csp.algorithms.backtracking import (
    BacktrackingForwardChecking,
    BacktrackingLCV,
    BacktrackingMRV,
    BasicBacktracking,
)
from fuel_csp.algorithms.base import SolverResult, SolverStats
from fuel_csp.algorithms.min_conflicts import MinConflictsSolver

ALL_SOLVERS: dict[str, type] = {
    "basic_backtracking": BasicBacktracking,
    "bt_mrv": BacktrackingMRV,
    "bt_lcv": BacktrackingLCV,
    "bt_fc_mrv_deg": BacktrackingForwardChecking,
    "min_conflicts": MinConflictsSolver,
}

__all__ = [
    "BasicBacktracking",
    "BacktrackingMRV",
    "BacktrackingLCV",
    "BacktrackingForwardChecking",
    "MinConflictsSolver",
    "SolverResult",
    "SolverStats",
    "ALL_SOLVERS",
]
