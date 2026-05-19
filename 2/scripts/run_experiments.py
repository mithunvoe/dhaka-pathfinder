"""End-to-end experiment driver — runs the matrix, saves CSVs, renders plots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from fuel_csp.analyzer import (  # noqa: E402
    ExperimentConfig,
    run_matrix,
    run_one,
    save_csvs,
)
from fuel_csp.visualizer import (  # noqa: E402
    plot_backtracks,
    plot_failure_rate,
    plot_heuristic_bars,
    plot_min_conflicts_convergence,
    plot_nodes,
    plot_objective,
    plot_problem_topology,
    plot_runtime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
console = Console()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="10,20,30,40,50",
                   help="comma-separated problem sizes (vehicles)")
    p.add_argument("--seeds", default="7,13,42",
                   help="comma-separated random seeds")
    p.add_argument("--time-budget", type=float, default=1.5,
                   help="per-run wall-clock cap in seconds")
    p.add_argument("--full", action="store_true",
                   help="paper-grade sweep: 5 seeds × 5 sizes × 6 s budget (~3 min)")
    args = p.parse_args()

    if args.full:
        cfg = ExperimentConfig(
            sizes=(10, 20, 30, 40, 50),
            seeds=(7, 13, 21, 42, 99),
            time_budget_s=6.0,
        )
    else:
        cfg = ExperimentConfig(
            sizes=tuple(int(x) for x in args.sizes.split(",")),
            seeds=tuple(int(x) for x in args.seeds.split(",")),
            time_budget_s=args.time_budget,
        )

    results = PROJECT_ROOT / "results"
    plots = results / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Running experiment matrix...")
    df = run_matrix(cfg)
    paths = save_csvs(df, results)
    console.print(f"[green]Saved[/green] {paths['raw']}  ({len(df)} rows)")
    console.print(f"[green]Saved[/green] {paths['summary']}")

    for fn, name in [
        (plot_runtime, "runtime_vs_n.png"),
        (plot_nodes, "nodes_vs_n.png"),
        (plot_backtracks, "backtracks_vs_n.png"),
        (plot_objective, "objective_vs_n.png"),
        (plot_failure_rate, "failure_rate_vs_n.png"),
        (plot_heuristic_bars, "heuristic_bars.png"),
    ]:
        try:
            out = fn(df, plots / name)
            console.print(f"[green]Saved plot:[/green] {out}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Plot failed {name}: {exc}[/yellow]")

    # Min-Conflicts convergence -- pick N = mid-range, all seeds
    mid_n = cfg.sizes[len(cfg.sizes) // 2]
    cost_traces: list[list[float]] = []
    for seed in cfg.seeds:
        res = run_one("min_conflicts", mid_n, seed, cfg)
        cost_traces.append(res.stats.cost_trace)
    plot_min_conflicts_convergence(
        cost_traces, plots / "min_conflicts_convergence.png"
    )
    console.print(f"[green]Saved plot:[/green] {plots / 'min_conflicts_convergence.png'}")

    # Topology of a sample assignment from the strongest BT solver
    from fuel_csp.algorithms import ALL_SOLVERS
    from fuel_csp.synthetic import GeneratorConfig, generate_problem
    sample_n = cfg.sizes[len(cfg.sizes) // 2]
    sample_seed = cfg.seeds[0]
    problem = generate_problem(
        GeneratorConfig(num_vehicles=sample_n, seed=sample_seed)
    )
    solver = ALL_SOLVERS["bt_fc_mrv_deg"](time_budget_s=cfg.time_budget_s)
    res = solver.solve(problem)
    plot_problem_topology(problem, res.assignment, plots / "sample_topology.png")
    console.print(f"[green]Saved plot:[/green] {plots / 'sample_topology.png'}")

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
