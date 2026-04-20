"""Run the 100+ iteration comparative analysis and generate all plots + summary CSVs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from dhaka_pathfinder.analyzer import (  # noqa: E402
    AnalyzerConfig,
    run_comparative_analysis,
    summarise,
    summarise_contexts,
    summarise_heuristics,
)
from dhaka_pathfinder.config import PLOTS_DIR, RESULTS_DIR  # noqa: E402
from dhaka_pathfinder.engine import DhakaPathfinderEngine  # noqa: E402
from dhaka_pathfinder.visualizer import (  # noqa: E402
    plot_comparison_bars,
    plot_context_sweep,
    plot_heuristic_matrix,
    plot_new_factors_impact,
    plot_predicted_vs_actual,
    plot_success_and_revisits,
)


logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, show_path=False)])
console = Console()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-pairs", type=int, default=100)
    p.add_argument("--min-m", type=float, default=1200.0)
    p.add_argument("--max-m", type=float, default=12000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-nodes", type=int, default=200_000)
    args = p.parse_args()

    engine = DhakaPathfinderEngine()
    console.print("[cyan]Loading graph…[/cyan]")
    engine.load()

    config = AnalyzerConfig(
        num_pairs=args.num_pairs,
        min_distance_m=args.min_m,
        max_distance_m=args.max_m,
        pair_seed=args.seed,
        max_nodes_per_algo=args.max_nodes,
    )

    csv_path = RESULTS_DIR / "comparison_matrix.csv"
    df = run_comparative_analysis(engine, config=config, save_csv=csv_path)

    console.print(f"[green]Wrote[/green] {csv_path}  ({len(df)} rows)")

    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "algorithm_summary.csv", index=False)
    console.print("\n[bold]Per-algorithm summary[/bold]")
    console.print(summary.to_string(index=False))

    heur_summary = summarise_heuristics(df)
    heur_summary.to_csv(RESULTS_DIR / "heuristic_summary.csv", index=False)
    console.print("\n[bold]Per-(algo, heuristic) summary[/bold]")
    console.print(heur_summary.to_string(index=False))

    ctx_summary = summarise_contexts(df)
    ctx_summary.to_csv(RESULTS_DIR / "context_summary.csv", index=False)
    console.print("\n[bold]Context summary rows:[/bold] %d" % len(ctx_summary))

    plots = {
        "comparison_bars.png": plot_comparison_bars,
        "heuristic_matrix.png": plot_heuristic_matrix,
        "predicted_vs_actual.png": plot_predicted_vs_actual,
        "success_and_revisits.png": plot_success_and_revisits,
        "context_sweep.png": plot_context_sweep,
        "new_factors_impact.png": plot_new_factors_impact,
    }
    for name, fn in plots.items():
        try:
            out = fn(df, PLOTS_DIR / name)
            console.print(f"[green]Saved plot:[/green] {out}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Plot failed {name}: {exc}[/yellow]")


if __name__ == "__main__":
    main()
