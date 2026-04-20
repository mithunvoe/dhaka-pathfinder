"""Visualisation — Folium maps and Matplotlib/Seaborn plots."""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import Fullscreen, MiniMap

from dhaka_pathfinder.algorithms import ALGORITHMS, SearchResult
from dhaka_pathfinder.config import DHAKA_CENTER, MAPS_DIR, PLOTS_DIR
from dhaka_pathfinder.osm_loader import node_coords

logger = logging.getLogger(__name__)


ALGO_COLORS = {
    "bfs":             "#1f77b4",
    "dfs":             "#d62728",
    "ucs":             "#2ca02c",
    "greedy":          "#ff7f0e",
    "astar":           "#9467bd",
    "weighted_astar":  "#e377c2",
}

ALGO_LABELS = {
    "bfs":             "BFS",
    "dfs":             "DFS",
    "ucs":             "UCS (Dijkstra)",
    "greedy":          "Greedy Best-First",
    "astar":           "A*",
    "weighted_astar":  "Weighted A* (w=1.8)",
}


def path_coords(G: nx.MultiDiGraph, path: list[int]) -> list[tuple[float, float]]:
    """(lat, lon) list for a node-path."""
    return [node_coords(G, n) for n in path]


def build_route_map(
    G: nx.MultiDiGraph,
    results: dict[str, SearchResult],
    source: int,
    destination: int,
    title: str = "Dhaka Route Comparison",
) -> folium.Map:
    """One Folium map with a toggleable layer per algorithm.

    Layers are added in *worst-to-best* cost order so the winning (lowest-cost)
    path always ends up **on top** of the Z-stack. We also give the winner a
    thicker stroke + brighter outline so it stands out even under a layer pile.
    """
    src_lat, src_lon = node_coords(G, source)
    dst_lat, dst_lon = node_coords(G, destination)
    center = ((src_lat + dst_lat) / 2, (src_lon + dst_lon) / 2)

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    folium.Marker(
        location=[src_lat, src_lon],
        popup=folium.Popup(f"<b>Source</b><br>Node {source}", max_width=300),
        tooltip="Source",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=[dst_lat, dst_lon],
        popup=folium.Popup(f"<b>Destination</b><br>Node {destination}", max_width=300),
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    successful = [(algo, r) for algo, r in results.items() if r.path and r.stats.success]
    if successful:
        min_cost = min(r.stats.path_cost for _, r in successful)
        ordered = sorted(
            successful,
            key=lambda t: (-t[1].stats.path_cost, t[0]),
        )
    else:
        ordered = []
        min_cost = float("inf")

    for algo, result in ordered:
        coords = path_coords(G, result.path)
        stats = result.stats
        is_winner = abs(stats.path_cost - min_cost) < 1e-3
        popup_html = (
            f"<b>{ALGO_LABELS.get(algo, algo)}</b>{' 🏆' if is_winner else ''}<br>"
            f"Cost: {stats.path_cost:,.1f}<br>"
            f"Length: {stats.path_length_meters/1000:.2f} km<br>"
            f"Edges: {stats.path_length_edges}<br>"
            f"Nodes expanded: {stats.nodes_expanded}<br>"
            f"Runtime: {stats.runtime_seconds*1000:.1f} ms<br>"
            f"EBF: {stats.effective_branching_factor:.2f}"
        )
        fg = folium.FeatureGroup(
            name=f"{'🏆 ' if is_winner else ''}{ALGO_LABELS.get(algo, algo)}",
            show=True,
        )
        if is_winner:
            folium.PolyLine(
                locations=coords,
                color="#ffffff",
                weight=9,
                opacity=0.9,
            ).add_to(fg)
        folium.PolyLine(
            locations=coords,
            color=ALGO_COLORS.get(algo, "#000000"),
            weight=7 if is_winner else 4,
            opacity=0.95 if is_winner else 0.55,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{'🏆 ' if is_winner else ''}{ALGO_LABELS.get(algo, algo)}  |  {stats.path_cost:,.0f}",
            dash_array=None if is_winner else "6, 6",
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = (
        f'<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);'
        f' z-index: 9999; background: white; padding: 8px 16px; border-radius: 6px;'
        f' box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-family: sans-serif;">'
        f'<b>{title}</b></div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def save_route_map(
    G: nx.MultiDiGraph,
    results: dict[str, SearchResult],
    source: int,
    destination: int,
    filename: str,
    title: str = "Dhaka Route Comparison",
) -> Path:
    m = build_route_map(G, results, source, destination, title=title)
    out = MAPS_DIR / filename
    m.save(str(out))
    return out


def plot_comparison_bars(df: pd.DataFrame, output_path: Path) -> Path:
    """Grouped bar chart for nodes expanded, cost, runtime across algorithms."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    metrics = [
        ("nodes_expanded", "Nodes Expanded (median)", "Greens_d"),
        ("path_cost", "Path Cost (median)", "Blues_d"),
        ("runtime_seconds", "Runtime seconds (median)", "Oranges_d"),
        ("effective_branching_factor", "Effective Branching Factor (median)", "Purples_d"),
    ]
    for ax, (col, title, palette) in zip(axes.flat, metrics):
        agg = df.groupby("algorithm")[col].median().sort_values()
        sns.barplot(x=agg.index, y=agg.values, ax=ax, palette=palette)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=10)
    fig.suptitle("Algorithm comparison — aggregated over runs", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_heuristic_matrix(df: pd.DataFrame, output_path: Path) -> Path:
    """Heatmap: rows = algorithms, cols = heuristics, cells = median path_cost."""
    sns.set_theme(style="white", context="talk")
    pivot = df.pivot_table(
        index="algorithm", columns="heuristic", values="path_cost", aggfunc="median"
    )
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * pivot.shape[1] + 4), 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar_kws={"label": "median cost"})
    ax.set_title("Median path cost — algorithm × heuristic")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_context_sweep(df: pd.DataFrame, output_path: Path) -> Path:
    """Facet grid: one subplot per context, algorithms as hue, cost on y."""
    sns.set_theme(style="whitegrid", context="talk")
    facet = sns.catplot(
        data=df,
        x="algorithm", y="path_cost", hue="algorithm",
        col="time_bucket", row="vehicle",
        kind="box", height=3, aspect=1.2, sharey=False,
        legend=False,
    )
    facet.set_xticklabels(rotation=35)
    facet.figure.suptitle("Path cost across algorithm × (vehicle, time_of_day)", y=1.02, weight="bold")
    facet.figure.tight_layout()
    facet.figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(facet.figure)
    return output_path


def plot_predicted_vs_actual(df: pd.DataFrame, output_path: Path) -> Path:
    """Scatter: predicted initial heuristic vs realised path cost, colored by algorithm."""
    sns.set_theme(style="whitegrid", context="talk")
    data = df[df["predicted_cost_at_start"] > 0].copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No predicted-vs-actual data", ha="center")
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=data, x="predicted_cost_at_start", y="path_cost",
        hue="algorithm", style="heuristic", s=60, alpha=0.7, ax=ax,
    )
    lim = max(data["predicted_cost_at_start"].max(), data["path_cost"].max())
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1, label="perfect prediction")
    ax.set_xlabel("Heuristic predicted cost at source")
    ax.set_ylabel("Actual realised path cost")
    ax.set_title("Heuristic prediction vs. actual experience")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_new_factors_impact(df: pd.DataFrame, output_path: Path) -> Path:
    """Three-panel: cost by age, cost by weather, cost-density by gender."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sub = df[df["success"]].copy()
    if sub.empty:
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    if "age" in sub.columns:
        sns.boxplot(data=sub, x="age", y="path_cost", ax=axes[0],
                    order=[a for a in ("adult", "child", "elderly") if a in sub["age"].unique()],
                    palette="pastel")
        axes[0].set_title("Cost distribution by age")
        axes[0].set_yscale("log")
    else:
        axes[0].text(0.5, 0.5, "no age column", ha="center")
        axes[0].set_axis_off()

    if "weather" in sub.columns:
        sns.boxplot(data=sub, x="weather", y="path_cost", ax=axes[1],
                    order=[w for w in ("clear", "rain", "fog", "storm", "heat") if w in sub["weather"].unique()],
                    palette="crest")
        axes[1].set_title("Cost distribution by weather")
        axes[1].set_yscale("log")
    else:
        axes[1].text(0.5, 0.5, "no weather column", ha="center")
        axes[1].set_axis_off()

    sns.violinplot(data=sub, x="gender", y="path_cost", ax=axes[2], hue="social",
                   split=True, inner="quart", palette="flare")
    axes[2].set_title("Cost distribution by gender / social")
    axes[2].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_success_and_revisits(df: pd.DataFrame, output_path: Path) -> Path:
    """Two panels — success rate and revisit rate per algorithm."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    succ = df.groupby("algorithm")["success"].mean().sort_values()
    sns.barplot(x=succ.index, y=succ.values, ax=axes[0], palette="crest")
    axes[0].set_title("Success rate")
    axes[0].set_ylabel("Fraction of runs that found a path")
    axes[0].tick_params(axis="x", rotation=35)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height()*100:.0f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=10)
    rv = df.groupby("algorithm")["revisits"].median().sort_values()
    sns.barplot(x=rv.index, y=rv.values, ax=axes[1], palette="flare")
    axes[1].set_title("Median revisits (backtracking proxy)")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path
