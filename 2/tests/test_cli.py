"""Tests for the click CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from fuel_csp.cli import cli


def test_cli_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "solve" in res.output
    assert "experiments" in res.output


def test_cli_solve_minimal():
    runner = CliRunner()
    res = runner.invoke(cli, [
        "solve", "--algo", "min_conflicts",
        "-n", "8", "--seed", "1", "--time-budget", "1.0",
    ])
    assert res.exit_code == 0, res.output
    assert "min_conflicts" in res.output
    assert "Assignment" in res.output


def test_cli_solve_basic_backtracking():
    runner = CliRunner()
    res = runner.invoke(cli, [
        "solve", "--algo", "basic_backtracking",
        "-n", "6", "--seed", "1", "--time-budget", "1.0",
    ])
    assert res.exit_code == 0, res.output


def test_cli_experiments(tmp_path: Path):
    runner = CliRunner()
    res = runner.invoke(cli, [
        "experiments",
        "--sizes", "6",
        "--seeds", "1,2",
        "--time-budget", "0.8",
        "--results-dir", str(tmp_path),
    ])
    assert res.exit_code == 0, res.output
    raw = tmp_path / "experiments_raw.csv"
    summary = tmp_path / "experiments_summary.csv"
    assert raw.exists() and summary.exists()
    df = pd.read_csv(raw)
    # 5 solvers * 1 size * 2 seeds
    assert len(df) == 10
