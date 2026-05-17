"""Tests for analyzer + visualizer pipelines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuel_csp.analyzer import ExperimentConfig, run_matrix, save_csvs, summarise
from fuel_csp.visualizer import (
    plot_backtracks,
    plot_failure_rate,
    plot_heuristic_bars,
    plot_nodes,
    plot_objective,
    plot_runtime,
)


def test_run_matrix_returns_rows():
    cfg = ExperimentConfig(sizes=(8,), seeds=(1, 2), time_budget_s=1.5)
    df = run_matrix(cfg)
    assert not df.empty
    # 5 solvers * 1 size * 2 seeds
    assert len(df) == 5 * 1 * 2
    assert {"algorithm", "n", "objective", "runtime_seconds"} <= set(df.columns)


def test_summarise_groups_rows():
    cfg = ExperimentConfig(sizes=(8,), seeds=(1, 2), time_budget_s=1.5)
    df = run_matrix(cfg)
    summary = summarise(df)
    assert len(summary) == 5  # one row per algorithm


def test_save_csvs(tmp_path: Path):
    cfg = ExperimentConfig(sizes=(8,), seeds=(1,), time_budget_s=1.0)
    df = run_matrix(cfg)
    paths = save_csvs(df, tmp_path)
    assert paths["raw"].exists()
    assert paths["summary"].exists()
    loaded = pd.read_csv(paths["summary"])
    assert "algorithm" in loaded.columns


def test_plot_pipelines(tmp_path: Path):
    cfg = ExperimentConfig(sizes=(8,), seeds=(1, 2), time_budget_s=1.0)
    df = run_matrix(cfg)
    for fn, name in [
        (plot_runtime, "runtime.png"),
        (plot_nodes, "nodes.png"),
        (plot_backtracks, "backtracks.png"),
        (plot_objective, "objective.png"),
        (plot_failure_rate, "failure.png"),
        (plot_heuristic_bars, "bars.png"),
    ]:
        out = fn(df, tmp_path / name)
        assert out.exists()
        assert out.stat().st_size > 0
