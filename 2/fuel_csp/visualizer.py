"""Matplotlib plots for the comparative report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")

ALGO_ORDER = [
    "basic_backtracking",
    "bt_mrv",
    "bt_lcv",
    "bt_fc_mrv_deg",
    "min_conflicts",
]

PRETTY: dict[str, str] = {
    "basic_backtracking": "Basic BT",
    "bt_mrv": "BT + MRV",
    "bt_lcv": "BT + LCV",
    "bt_fc_mrv_deg": "BT + FC + MRV+Deg",
    "min_conflicts": "Min-Conflicts",
}

PALETTE = {
    "basic_backtracking": "#d62728",
    "bt_mrv":             "#ff7f0e",
    "bt_lcv":             "#9467bd",
    "bt_fc_mrv_deg":      "#2ca02c",
    "min_conflicts":      "#1f77b4",
}


def _line_panel(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out: Path,
    logy: bool = False,
) -> Path:
    summary = df.groupby(["algorithm", "n"])[y_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGO_ORDER:
        sub = summary[summary["algorithm"] == algo].sort_values("n")
        if sub.empty:
            continue
        ax.plot(
            sub["n"],
            sub[y_col],
            marker="o",
            linewidth=2.5,
            label=PRETTY.get(algo, algo),
            color=PALETTE.get(algo),
        )
    ax.set_xlabel("Problem size N (vehicles)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=11, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_runtime(df: pd.DataFrame, out: Path) -> Path:
    return _line_panel(
        df, "runtime_seconds", "Mean runtime (s)",
        "Scalability — execution time vs problem size", out, logy=True,
    )


def plot_nodes(df: pd.DataFrame, out: Path) -> Path:
    return _line_panel(
        df, "nodes_expanded", "Mean nodes expanded",
        "Search effort — nodes expanded vs problem size", out, logy=True,
    )


def plot_backtracks(df: pd.DataFrame, out: Path) -> Path:
    return _line_panel(
        df, "backtracks", "Mean backtracks",
        "Backtracks vs problem size", out, logy=True,
    )


def plot_objective(df: pd.DataFrame, out: Path) -> Path:
    return _line_panel(
        df, "objective", "Mean COP objective  J(S)  (lower is better)",
        "Solution quality vs problem size", out,
    )


def plot_failure_rate(df: pd.DataFrame, out: Path) -> Path:
    return _line_panel(
        df, "failure_rate", "Failure rate  (unassigned / N)",
        "Graceful-failure behavior vs problem size", out,
    )


def plot_heuristic_bars(df: pd.DataFrame, out: Path) -> Path:
    """One bar per algorithm: mean runtime, mean backtracks, mean J(S)."""
    summary = df.groupby("algorithm").agg(
        runtime=("runtime_seconds", "mean"),
        backtracks=("backtracks", "mean"),
        objective=("objective", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("runtime", "Mean runtime (s)", "Runtime"),
        ("backtracks", "Mean backtracks", "Backtracks"),
        ("objective", "Mean J(S)", "Objective"),
    ]
    for ax, (col, ylabel, title) in zip(axes, metrics):
        x = [PRETTY.get(a, a) for a in ALGO_ORDER if a in summary["algorithm"].values]
        y = [
            summary.loc[summary["algorithm"] == a, col].values[0]
            for a in ALGO_ORDER if a in summary["algorithm"].values
        ]
        colors = [PALETTE.get(a) for a in ALGO_ORDER if a in summary["algorithm"].values]
        bars = ax.bar(x, y, color=colors, edgecolor="black")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for b, val in zip(bars, y):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height(),
                f"{val:.2g}", ha="center", va="bottom", fontsize=10,
            )
    fig.suptitle("Per-algorithm aggregates (averaged across all N and seeds)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_min_conflicts_convergence(cost_traces: list[list[float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, trace in enumerate(cost_traces):
        if not trace:
            continue
        steps = np.arange(len(trace))
        ax.plot(steps, trace, alpha=0.6, linewidth=1.5, label=f"seed {i}")
    ax.set_xlabel("Repair step")
    ax.set_ylabel("J(S) at step")
    ax.set_title("Min-Conflicts convergence — J(S) over repair steps")
    if len(cost_traces) <= 8:
        ax.legend(fontsize=10, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_problem_topology(problem, assignment, out: Path) -> Path:
    """Render the synthetic city: stations, vehicles, and the chosen edges."""
    fig, ax = plt.subplots(figsize=(10, 9))
    sx = [s.x for s in problem.stations]
    sy = [s.y for s in problem.stations]
    ax.scatter(sx, sy, s=320, marker="s", color="#2ca02c", edgecolor="black",
               zorder=3, label="Fuel station")
    for s in problem.stations:
        ax.annotate(f"S{s.sid}\n{s.pumps}p", (s.x, s.y), color="black",
                    fontsize=9, ha="center", va="center")

    kind_color = {
        "ambulance": "#d62728",
        "bus": "#9467bd",
        "truck": "#8c564b",
        "car": "#1f77b4",
        "motorbike": "#7f7f7f",
    }
    for v in problem.vehicles:
        ax.scatter(v.x, v.y, s=80, color=kind_color.get(v.kind, "gray"),
                   edgecolor="black", zorder=2)
    # plot connections for assigned vehicles
    for vid, a in assignment.items():
        v = problem.vehicles[vid]
        s = problem.stations[a.station_id]
        ax.plot([v.x, s.x], [v.y, s.y], color=kind_color.get(v.kind, "gray"),
                alpha=0.5, linewidth=1.4, zorder=1)
    handles = [
        plt.Line2D([], [], marker="s", color="w", markerfacecolor="#2ca02c",
                   markeredgecolor="black", markersize=14, label="Fuel station"),
    ]
    for k, c in kind_color.items():
        handles.append(plt.Line2D([], [], marker="o", color="w",
                                  markerfacecolor=c, markeredgecolor="black",
                                  markersize=10, label=k))
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.set_xlabel("x  (km)")
    ax.set_ylabel("y  (km)")
    ax.set_title(f"Sample instance — N={problem.n}, |assigned|={len(assignment)}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
