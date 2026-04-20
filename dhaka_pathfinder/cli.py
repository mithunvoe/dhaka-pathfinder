"""Command-line interface for the Dhaka pathfinder."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from dhaka_pathfinder.algorithms import ALGORITHMS, INFORMED
from dhaka_pathfinder.analyzer import (
    AnalyzerConfig,
    run_comparative_analysis,
    summarise,
    summarise_contexts,
    summarise_heuristics,
)
from dhaka_pathfinder.config import (
    DATA_DIR,
    LANDMARKS,
    MAPS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
)
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.engine import DhakaPathfinderEngine, EngineConfig
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES
from dhaka_pathfinder.osm_loader import (
    GraphLoadSpec,
    graph_summary,
    load_dhaka_graph,
)
from dhaka_pathfinder.synthetic_data import summarize_synthetic
from dhaka_pathfinder.visualizer import (
    plot_comparison_bars,
    plot_context_sweep,
    plot_heuristic_matrix,
    plot_predicted_vs_actual,
    plot_success_and_revisits,
    save_route_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
console = Console()


@click.group()
def cli() -> None:
    """Dhaka Pathfinder — realistic multi-factor pathfinding over OSM."""


@cli.command("download")
@click.option("--mode", type=click.Choice(["bbox", "place", "point"]), default="bbox")
@click.option("--force", is_flag=True, help="Re-download even if cache exists.")
def download(mode: str, force: bool) -> None:
    """Fetch the Dhaka road network from OSM and cache it."""
    console.print("[cyan]Downloading Dhaka road network from OSM…[/cyan]")
    spec = GraphLoadSpec(mode=mode)
    G = load_dhaka_graph(spec, force_refresh=force)
    summary = graph_summary(G)
    table = Table(title="Graph summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric"); table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, f"{v:,.4f}" if isinstance(v, float) else str(v))
    console.print(table)


def _print_landmarks() -> None:
    table = Table(title="Known Dhaka landmarks", show_header=True)
    table.add_column("Name"); table.add_column("Latitude"); table.add_column("Longitude")
    for name, (lat, lon) in LANDMARKS.items():
        table.add_row(name, f"{lat:.4f}", f"{lon:.4f}")
    console.print(table)


@cli.command("landmarks")
def landmarks_cmd() -> None:
    """List supported landmark names."""
    _print_landmarks()


def _resolve_point(arg: str) -> tuple[float, float]:
    if arg in LANDMARKS:
        return LANDMARKS[arg]
    if "," in arg:
        lat_s, lon_s = arg.split(",", 1)
        return float(lat_s.strip()), float(lon_s.strip())
    raise click.BadParameter(f"Could not resolve {arg!r}. Use a landmark name or 'lat,lon'.")


@cli.command("route")
@click.option("--source", required=True, help="Source: landmark name or 'lat,lon'.")
@click.option("--dest", required=True, help="Destination: landmark name or 'lat,lon'.")
@click.option("--algorithm", type=click.Choice(sorted(ALGORITHMS)), default="astar")
@click.option("--heuristic", type=click.Choice(sorted(HEURISTIC_FACTORIES)), default="haversine_admissible")
@click.option("--gender", type=click.Choice(["male", "female", "nonbinary"]), default="male")
@click.option("--social", type=click.Choice(["alone", "accompanied"]), default="alone")
@click.option("--age", type=click.Choice(["adult", "child", "elderly"]), default="adult")
@click.option("--vehicle", type=click.Choice(["walk", "rickshaw", "cng", "motorbike", "car", "bus"]), default="car")
@click.option("--time-of-day", "time_bucket",
              type=click.Choice(["early_morning", "morning_rush", "midday", "afternoon",
                                 "evening_rush", "evening", "late_night"]),
              default="midday")
@click.option("--weather", type=click.Choice(["clear", "rain", "fog", "storm", "heat"]), default="clear")
@click.option("--weight-preset", type=click.Choice(["balanced", "speed", "safety", "comfort"]), default="balanced")
@click.option("--save-map/--no-save-map", default=True)
@click.option("--compare-all/--no-compare-all", default=False, help="Run every algorithm and overlay them.")
def route(
    source: str, dest: str, algorithm: str, heuristic: str, gender: str, social: str,
    age: str, vehicle: str, time_bucket: str, weather: str, weight_preset: str,
    save_map: bool, compare_all: bool,
) -> None:
    """Compute one (or all) routes for a source/destination pair."""
    src_lat, src_lon = _resolve_point(source)
    dst_lat, dst_lon = _resolve_point(dest)
    ctx = TravelContext(gender=gender, social=social, age=age, vehicle=vehicle,
                        time_bucket=time_bucket, weather=weather, weight_preset=weight_preset)
    engine = DhakaPathfinderEngine(EngineConfig(weight_preset=weight_preset))
    console.print("[cyan]Loading Dhaka graph…[/cyan]")
    engine.load()

    s_node = engine.nearest(src_lat, src_lon)
    d_node = engine.nearest(dst_lat, dst_lon)
    console.print(f"[green]Source node:[/green] {s_node}  ({src_lat:.4f},{src_lon:.4f})")
    console.print(f"[green]Destination node:[/green] {d_node}  ({dst_lat:.4f},{dst_lon:.4f})")

    if compare_all:
        results = {algo: engine.solve(algo, s_node, d_node, ctx, heuristic) for algo in ALGORITHMS}
    else:
        results = {algorithm: engine.solve(algorithm, s_node, d_node, ctx, heuristic)}

    table = Table(title=f"Route(s) — {source} → {dest}  |  {ctx.label()}", show_header=True, header_style="bold magenta")
    for h in ("algorithm", "success", "cost", "km", "edges", "expanded", "revisits", "EBF", "runtime ms"):
        table.add_column(h)
    for name, r in results.items():
        s = r.stats
        table.add_row(
            name, str(s.success), f"{s.path_cost:,.1f}", f"{s.path_length_meters/1000:.2f}",
            str(s.path_length_edges), str(s.nodes_expanded), str(s.revisits),
            f"{s.effective_branching_factor:.2f}", f"{s.runtime_seconds*1000:.1f}",
        )
    console.print(table)

    if save_map and any(r.path for r in results.values()):
        fname = f"route_{source.replace(' ', '_')}_to_{dest.replace(' ', '_')}.html"
        path = save_route_map(engine.graph, results, s_node, d_node, fname,
                              title=f"{source} → {dest}  |  {ctx.label()}")
        console.print(f"[green]Saved map:[/green] {path}")


@cli.command("compare")
@click.option("--num-pairs", default=100, show_default=True)
@click.option("--min-m", default=1200.0, show_default=True)
@click.option("--max-m", default=12000.0, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--csv", type=click.Path(path_type=Path), default=RESULTS_DIR / "comparison_matrix.csv")
def compare(num_pairs: int, min_m: float, max_m: float, seed: int, csv: Path) -> None:
    """Run the 100+ iteration comparative analysis."""
    engine = DhakaPathfinderEngine()
    console.print("[cyan]Loading Dhaka graph…[/cyan]")
    engine.load()
    config = AnalyzerConfig(num_pairs=num_pairs, min_distance_m=min_m, max_distance_m=max_m, pair_seed=seed)
    df = run_comparative_analysis(engine, config=config, save_csv=csv)
    console.print(f"[green]Wrote[/green] {csv}  ({len(df)} rows)")

    summary = summarise(df)
    console.print(summary.to_string(index=False))

    fig_paths = {
        "comparison_bars.png": plot_comparison_bars,
        "heuristic_matrix.png": plot_heuristic_matrix,
        "predicted_vs_actual.png": plot_predicted_vs_actual,
        "success_and_revisits.png": plot_success_and_revisits,
        "context_sweep.png": plot_context_sweep,
    }
    for name, fn in fig_paths.items():
        try:
            out = fn(df, PLOTS_DIR / name)
            console.print(f"[green]Saved plot:[/green] {out}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Skipped {name}: {exc}[/yellow]")


@cli.command("synth-stats")
def synth_stats() -> None:
    """Print synthetic-layer statistics for the default graph."""
    engine = DhakaPathfinderEngine()
    engine.load()
    stats = summarize_synthetic(engine.graph)
    for k, v in stats.items():
        console.print(f"{k:>30s} = {v:.4f}")


if __name__ == "__main__":
    cli()
