"""Produce a gallery of sample HTML route maps for the report."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from dhaka_pathfinder.config import LANDMARKS  # noqa: E402
from dhaka_pathfinder.context import TravelContext  # noqa: E402
from dhaka_pathfinder.engine import DhakaPathfinderEngine  # noqa: E402
from dhaka_pathfinder.visualizer import save_route_map  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, show_path=False)])
console = Console()


SAMPLES: list[tuple[str, str, str, TravelContext]] = [
    ("Shahbag", "Gulshan 2",
     "shahbag_to_gulshan2_female_cng_evening_rush.html",
     TravelContext(gender="female", social="alone", vehicle="cng", time_bucket="evening_rush")),
    ("New Market", "Motijheel",
     "newmarket_to_motijheel_car_morning_rush.html",
     TravelContext(gender="male", social="alone", vehicle="car", time_bucket="morning_rush")),
    ("Old Dhaka (Chawkbazar)", "Banani",
     "olddhaka_to_banani_walk_late_night.html",
     TravelContext(gender="female", social="alone", vehicle="walk", time_bucket="late_night")),
    ("Mirpur 10", "Farmgate",
     "mirpur10_to_farmgate_rickshaw_midday.html",
     TravelContext(gender="male", social="accompanied", vehicle="rickshaw", time_bucket="midday")),
    ("Uttara Sector 7", "Airport (HSIA)",
     "uttara_to_airport_motorbike_afternoon.html",
     TravelContext(gender="male", social="alone", vehicle="motorbike", time_bucket="afternoon")),
    ("Dhanmondi 27", "Mohammadpur",
     "dhanmondi_to_mohammadpur_cng_evening.html",
     TravelContext(gender="female", social="accompanied", vehicle="cng", time_bucket="evening")),
]


def main() -> None:
    engine = DhakaPathfinderEngine()
    engine.load()

    for src_name, dst_name, fname, ctx in SAMPLES:
        if src_name not in LANDMARKS or dst_name not in LANDMARKS:
            console.print(f"[yellow]Skipping {src_name} → {dst_name} (not in bbox)[/yellow]")
            continue
        s_lat, s_lon = LANDMARKS[src_name]
        d_lat, d_lon = LANDMARKS[dst_name]
        s_node = engine.nearest(s_lat, s_lon)
        d_node = engine.nearest(d_lat, d_lon)

        results = engine.solve_all(s_node, d_node, ctx, "haversine_admissible")
        out = save_route_map(
            engine.graph, results, s_node, d_node, fname,
            title=f"{src_name} → {dst_name} | {ctx.label()}"
        )
        best = min(
            (r for r in results.values() if r.stats.success),
            key=lambda r: r.stats.path_cost, default=None,
        )
        if best:
            console.print(
                f"[green]{src_name:>24s} → {dst_name:<24s}[/green] "
                f"best cost={best.stats.path_cost:,.0f}  "
                f"({best.stats.path_length_meters/1000:.2f} km)  "
                f"saved: {out.name}"
            )


if __name__ == "__main__":
    main()
