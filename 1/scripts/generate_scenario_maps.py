"""Produce side-by-side comparison maps for the teacher-demo scenarios.

For each scenario we render ONE HTML map that overlays the two routes (context A
and context B) in different colors so the visual route difference is obvious.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import folium  # noqa: E402
from folium.plugins import Fullscreen, MiniMap  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from dhaka_pathfinder.config import LANDMARKS, MAPS_DIR  # noqa: E402
from dhaka_pathfinder.context import TravelContext  # noqa: E402
from dhaka_pathfinder.engine import DhakaPathfinderEngine  # noqa: E402
from dhaka_pathfinder.osm_loader import node_coords  # noqa: E402
from dhaka_pathfinder.visualizer import path_coords  # noqa: E402


logging.basicConfig(
    level=logging.WARNING, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
console = Console()


SCENARIOS = [
    {
        "id": "scenario_1_gender_time",
        "title": "Scenario 1 — Safe commute vs risky night walk",
        "source": "Shahbag",
        "dest": "Gulshan 2",
        "ctx_a": {
            "label": "☀️ Adult male in a car, midday, clear weather",
            "color": "#2ca02c",
            "context": TravelContext(
                gender="male", social="alone", age="adult",
                vehicle="car", time_bucket="midday", weather="clear",
            ),
        },
        "ctx_b": {
            "label": "🌙 Adult female walking alone, late night, clear weather",
            "color": "#d62728",
            "context": TravelContext(
                gender="female", social="alone", age="adult",
                vehicle="walk", time_bucket="late_night", weather="clear",
            ),
        },
        "what_to_notice": (
            "Same source, same destination, same graph — but the night-walker "
            "context adds heavy penalties for dark / unsafe / high-crime edges "
            "AND for the vehicle-highway suitability (walking on wide primary "
            "roads is penalised). The algorithm therefore picks a different "
            "sequence of intersections."
        ),
    },
    {
        "id": "scenario_2_storm_detour",
        "title": "Scenario 2 — Same driver, different weather",
        "source": "Mirpur 10",
        "dest": "Jatrabari",
        "ctx_a": {
            "label": "☀️ Car, afternoon, clear weather",
            "color": "#1f77b4",
            "context": TravelContext(
                gender="male", social="alone", age="adult",
                vehicle="car", time_bucket="afternoon", weather="clear",
            ),
        },
        "ctx_b": {
            "label": "⛈️ Car, afternoon, STORM weather",
            "color": "#9467bd",
            "context": TravelContext(
                gender="male", social="alone", age="adult",
                vehicle="car", time_bucket="afternoon", weather="storm",
            ),
        },
        "what_to_notice": (
            "Only the weather changes. A storm amplifies water-logging (×3.0), "
            "risk (×1.85), and traffic (×1.5). The algorithm reroutes AROUND "
            "flood-prone low-lying streets near Old Dhaka and picks higher-"
            "elevation, larger roads even at the cost of extra distance."
        ),
    },
]


def render_scenario(engine: DhakaPathfinderEngine, scenario: dict) -> Path:
    src_lat, src_lon = LANDMARKS[scenario["source"]]
    dst_lat, dst_lon = LANDMARKS[scenario["dest"]]
    s_node = engine.nearest(src_lat, src_lon)
    d_node = engine.nearest(dst_lat, dst_lon)

    r_a = engine.solve("astar", s_node, d_node, scenario["ctx_a"]["context"], "network_relaxed")
    r_b = engine.solve("astar", s_node, d_node, scenario["ctx_b"]["context"], "network_relaxed")

    G = engine.graph
    lat_s, lon_s = node_coords(G, s_node)
    lat_d, lon_d = node_coords(G, d_node)
    center = ((lat_s + lat_d) / 2, (lon_s + lon_d) / 2)

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    folium.Marker(
        [lat_s, lon_s],
        tooltip=f"Source: {scenario['source']}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        [lat_d, lon_d],
        tooltip=f"Destination: {scenario['dest']}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    for which in ("a", "b"):
        ctx_info = scenario[f"ctx_{which}"]
        result = r_a if which == "a" else r_b
        if not result.path:
            continue
        coords = path_coords(G, result.path)
        fg = folium.FeatureGroup(name=ctx_info["label"], show=True)
        folium.PolyLine(
            locations=coords,
            color="#ffffff",
            weight=9,
            opacity=0.85,
        ).add_to(fg)
        folium.PolyLine(
            locations=coords,
            color=ctx_info["color"],
            weight=6,
            opacity=0.95,
            popup=folium.Popup(
                f"<b>{ctx_info['label']}</b><br>"
                f"Length: {result.stats.path_length_meters/1000:.2f} km<br>"
                f"Cost: {result.stats.path_cost:,.1f}<br>"
                f"Nodes expanded: {result.stats.nodes_expanded:,}<br>"
                f"Runtime: {result.stats.runtime_seconds*1000:.1f} ms",
                max_width=350,
            ),
            tooltip=ctx_info["label"],
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    caption = (
        f"<div style='position: fixed; top: 10px; left: 50%; transform: translateX(-50%);"
        f" z-index: 9999; background: rgba(255,255,255,0.96); padding: 10px 18px;"
        f" border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);"
        f" font-family: sans-serif; max-width: 680px; line-height: 1.4;'>"
        f"<b style='font-size: 17px;'>{scenario['title']}</b>"
        f"<br><span style='font-size: 13px; color:#333;'>"
        f"<b>{scenario['source']} → {scenario['dest']}</b> &middot; algorithm: A\\*"
        f"</span><br>"
        f"<span style='color:{scenario['ctx_a']['color']}; font-weight:600;'>"
        f"━━  {scenario['ctx_a']['label']}  —  "
        f"{r_a.stats.path_length_meters/1000:.2f} km, cost {r_a.stats.path_cost:,.0f}"
        f"</span><br>"
        f"<span style='color:{scenario['ctx_b']['color']}; font-weight:600;'>"
        f"━━  {scenario['ctx_b']['label']}  —  "
        f"{r_b.stats.path_length_meters/1000:.2f} km, cost {r_b.stats.path_cost:,.0f}"
        f"</span><br>"
        f"<span style='font-size: 12px; color:#555; font-style:italic;'>"
        f"{scenario['what_to_notice']}"
        f"</span></div>"
    )
    m.get_root().html.add_child(folium.Element(caption))

    out_path = MAPS_DIR / f"{scenario['id']}.html"
    m.save(str(out_path))

    shared = set(r_a.path) & set(r_b.path)
    only_a = set(r_a.path) - set(r_b.path)
    only_b = set(r_b.path) - set(r_a.path)
    console.print(f"\n[bold]{scenario['title']}[/bold]")
    console.print(f"  {scenario['source']} → {scenario['dest']}")
    console.print(
        f"  Context A ({scenario['ctx_a']['label']})\n"
        f"    path: {r_a.stats.path_length_edges} segments, "
        f"{r_a.stats.path_length_meters/1000:.2f} km, cost {r_a.stats.path_cost:,.0f}"
    )
    console.print(
        f"  Context B ({scenario['ctx_b']['label']})\n"
        f"    path: {r_b.stats.path_length_edges} segments, "
        f"{r_b.stats.path_length_meters/1000:.2f} km, cost {r_b.stats.path_cost:,.0f}"
    )
    console.print(
        f"  Route difference: {len(only_a)} nodes only in A, {len(only_b)} only in B, "
        f"{len(shared)} shared."
    )
    console.print(f"  [green]Saved:[/green] {out_path}")
    return out_path


def main() -> None:
    engine = DhakaPathfinderEngine()
    console.print("[cyan]Loading Dhaka graph…[/cyan]")
    engine.load()

    for scenario in SCENARIOS:
        render_scenario(engine, scenario)


if __name__ == "__main__":
    main()
