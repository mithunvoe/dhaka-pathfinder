"""Run the comparative experiment matrix and produce summary DataFrames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fuel_csp.algorithms import ALL_SOLVERS
from fuel_csp.algorithms.base import SolverResult
from fuel_csp.synthetic import GeneratorConfig, generate_problem

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Defaults are tuned to finish a full sweep in well under a minute while
    still showing the exponential vs polynomial scaling story. Pass `--full`
    to scripts/run_experiments.py for the slow, paper-grade matrix.
    """

    sizes: tuple[int, ...] = (10, 20, 30, 40, 50)
    seeds: tuple[int, ...] = (7, 13, 42)
    time_budget_s: float = 1.5
    min_conflicts_steps: int = 2000
    num_stations: int = 6
    num_slots: int = 6


def _make_solver(name: str, cfg: ExperimentConfig, seed: int):
    cls = ALL_SOLVERS[name]
    if name == "min_conflicts":
        return cls(
            max_steps=cfg.min_conflicts_steps,
            seed=seed,
            time_budget_s=cfg.time_budget_s,
        )
    return cls(time_budget_s=cfg.time_budget_s)


def run_one(
    algo: str,
    n: int,
    seed: int,
    cfg: ExperimentConfig,
) -> SolverResult:
    gcfg = GeneratorConfig(
        num_vehicles=n,
        num_stations=cfg.num_stations,
        num_slots=cfg.num_slots,
        seed=seed,
    )
    problem = generate_problem(gcfg)
    solver = _make_solver(algo, cfg, seed)
    res = solver.solve(problem)
    res.stats.seed = seed
    return res


def run_matrix(cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    """Run every (algorithm × size × seed) combination and collect a DataFrame."""
    cfg = cfg or ExperimentConfig()
    rows: list[dict] = []
    total = len(ALL_SOLVERS) * len(cfg.sizes) * len(cfg.seeds)
    pbar = tqdm(total=total, desc="experiments", unit="run")
    for algo in ALL_SOLVERS:
        for n in cfg.sizes:
            for seed in cfg.seeds:
                try:
                    res = run_one(algo, n, seed, cfg)
                    rows.append(res.stats.as_dict())
                except Exception as exc:  # noqa: BLE001
                    log.warning("run failed: %s n=%d seed=%d: %s", algo, n, seed, exc)
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate (mean across seeds) per (algorithm, n)."""
    if df.empty:
        return df
    grouped = df.groupby(["algorithm", "n"]).agg(
        runtime_s_mean=("runtime_seconds", "mean"),
        runtime_s_std=("runtime_seconds", "std"),
        nodes_mean=("nodes_expanded", "mean"),
        backtracks_mean=("backtracks", "mean"),
        constraint_checks_mean=("constraint_checks", "mean"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        failure_rate_mean=("failure_rate", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()
    return grouped


def save_csvs(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write the raw matrix + the per-(algorithm, n) summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "experiments_raw.csv"
    summary_path = out_dir / "experiments_summary.csv"
    df.to_csv(raw_path, index=False)
    summarise(df).to_csv(summary_path, index=False)
    return {"raw": raw_path, "summary": summary_path}
