"""Click-based CLI: solve a single instance, or run the full experiment matrix."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from fuel_csp.algorithms import ALL_SOLVERS
from fuel_csp.analyzer import ExperimentConfig, run_matrix, run_one, save_csvs
from fuel_csp.synthetic import GeneratorConfig, generate_problem

console = Console()
RESULTS = Path(__file__).resolve().parent.parent / "results"


@click.group()
def cli() -> None:
    """fuel-csp — CSP/COP solver for the urban fuel-crisis assignment."""


@cli.command()
@click.option("--algo", type=click.Choice(list(ALL_SOLVERS.keys())),
              default="bt_fc_mrv_deg", show_default=True)
@click.option("-n", "--num-vehicles", default=20, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--time-budget", default=6.0, show_default=True,
              help="Seconds before falling back to best-partial.")
def solve(algo: str, num_vehicles: int, seed: int, time_budget: float) -> None:
    """Solve one instance and print the metrics + assignment."""
    cfg = ExperimentConfig(time_budget_s=time_budget)
    res = run_one(algo, num_vehicles, seed, cfg)
    console.rule(f"[bold cyan]{algo}  N={num_vehicles}  seed={seed}")
    table = Table(show_header=True, header_style="bold magenta")
    for k in ("algorithm", "n", "objective", "num_assigned", "num_unassigned",
              "failure_rate", "backtracks", "nodes_expanded",
              "constraint_checks", "runtime_seconds", "success"):
        table.add_column(k)
    s = res.stats
    table.add_row(
        s.algorithm, str(s.n), f"{s.objective:.2f}",
        str(s.num_assigned), str(s.num_unassigned),
        f"{s.failure_rate:.2%}", str(s.backtracks),
        str(s.nodes_expanded), str(s.constraint_checks),
        f"{s.runtime_seconds*1000:.2f}ms", str(s.success),
    )
    console.print(table)

    console.print("\n[bold]Assignment (vehicle -> station/pump@slot):[/bold]")
    problem = generate_problem(GeneratorConfig(num_vehicles=num_vehicles, seed=seed))
    for vid in sorted(res.assignment):
        a = res.assignment[vid]
        v = problem.vehicles[vid]
        console.print(
            f"  v{vid:02d} ({v.kind:<10s} {v.fuel_type:<7s}) -> {a}",
            highlight=False,
        )
    # full machine-readable dump
    payload = {
        "stats": s.as_dict(),
        "assignment": {
            str(k): {
                "station_id": v.station_id, "pump_id": v.pump_id, "slot_id": v.slot_id,
            }
            for k, v in res.assignment.items()
        },
    }
    console.print()
    console.print_json(json.dumps(payload))


@cli.command()
@click.option("--sizes", default="10,20,30,40,50", show_default=True)
@click.option("--seeds", default="7,13,21,42,99", show_default=True)
@click.option("--time-budget", default=6.0, show_default=True)
@click.option("--results-dir", default=str(RESULTS), show_default=True)
def experiments(sizes: str, seeds: str, time_budget: float, results_dir: str) -> None:
    """Run the full algorithm × size × seed matrix and save CSVs."""
    cfg = ExperimentConfig(
        sizes=tuple(int(x) for x in sizes.split(",")),
        seeds=tuple(int(x) for x in seeds.split(",")),
        time_budget_s=time_budget,
    )
    df = run_matrix(cfg)
    if df.empty:
        console.print("[red]No rows produced.")
        return
    paths = save_csvs(df, Path(results_dir))
    console.print(f"[green]Wrote[/green] {paths['raw']}  ({len(df)} rows)")
    console.print(f"[green]Wrote[/green] {paths['summary']}")
    summary = pd.read_csv(paths["summary"])
    console.print("\n[bold]Per-algorithm × N summary[/bold]")
    console.print(summary.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    cli()
