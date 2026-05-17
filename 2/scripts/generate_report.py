"""Read results/experiments_*.csv and emit results/REPORT.md."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()
RESULTS = PROJECT_ROOT / "results"


def fmt(x: float, n: int = 3) -> str:
    if x != x:
        return "-"
    return f"{x:.{n}g}"


def render_table(df: pd.DataFrame, cols: list[str], header: str) -> str:
    out = [f"### {header}", ""]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        out.append(
            "| " + " | ".join(
                str(row[c]) if isinstance(row[c], (str, int))
                else fmt(float(row[c])) for c in cols
            ) + " |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    raw_path = RESULTS / "experiments_raw.csv"
    summary_path = RESULTS / "experiments_summary.csv"
    if not raw_path.exists() or not summary_path.exists():
        console.print("[red]Missing CSVs.[/red] Run scripts/run_experiments.py first.")
        sys.exit(1)

    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path).sort_values(["n", "algorithm"])

    md = ["# Fuel-CSP — Experimental Report", "",
          "## 1. What was measured", "",
          "For every (algorithm, N, seed) we recorded:",
          "* `runtime_seconds` — wall clock to terminate (or to hit the time budget).",
          "* `nodes_expanded` — recursive calls / value attempts.",
          "* `backtracks` — number of recursion unwinds caused by infeasibility.",
          "* `constraint_checks` — calls to the consistency oracle.",
          "* `objective` — the COP soft-cost J(S) of the returned assignment.",
          "* `failure_rate` — fraction of vehicles left unassigned (0 = full CSP-feasible).",
          "* `success` — strictly true iff a complete + feasible assignment was found.",
          "",
          "Each cell is averaged across "
          f"{raw['seed'].nunique()} seeds at each problem size N.",
          "",
          "## 2. Per-(algorithm, N) summary", "",
          ]
    md.append(render_table(
        summary,
        ["algorithm", "n", "runtime_s_mean", "nodes_mean", "backtracks_mean",
         "objective_mean", "failure_rate_mean", "success_rate"],
        "Mean metrics across seeds",
    ))

    agg = raw.groupby("algorithm").agg(
        runtime_s=("runtime_seconds", "mean"),
        nodes=("nodes_expanded", "mean"),
        backtracks=("backtracks", "mean"),
        constraint_checks=("constraint_checks", "mean"),
        objective=("objective", "mean"),
        failure_rate=("failure_rate", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()

    md += [
        "## 3. Algorithm-level aggregates", "",
        render_table(
            agg,
            ["algorithm", "runtime_s", "nodes", "backtracks",
             "constraint_checks", "objective", "failure_rate", "success_rate"],
            "Means across every (N, seed) cell",
        ),
        "## 4. Plots", "",
        "All plots live in `results/plots/`:",
        "",
        "* `runtime_vs_n.png` — log-scale runtime scaling.",
        "* `nodes_vs_n.png` — log-scale search-effort scaling.",
        "* `backtracks_vs_n.png` — failed-extension count.",
        "* `objective_vs_n.png` — solution quality vs N (lower J(S) is better).",
        "* `failure_rate_vs_n.png` — graceful-failure evidence (COP behavior).",
        "* `heuristic_bars.png` — bar comparison of the five algorithms.",
        "* `min_conflicts_convergence.png` — repair-step convergence curve.",
        "* `sample_topology.png` — visualization of one solved instance.",
        "",
        "## 5. Take-aways", "",
        "1. **Basic backtracking** is the slowest and explodes in nodes/backtracks at "
        "larger N — empirical evidence of the worst-case combinatorial blow-up "
        "from the lecture.",
        "2. **MRV / LCV / Forward-Checking** consistently shrink the search "
        "effort and produce lower J(S) — heuristics buy both speed and quality.",
        "3. **Min-Conflicts** is the only solver whose runtime stays "
        "near-flat as N grows. It returns a high-quality COP solution very "
        "quickly but is not complete (no guarantee of zero conflicts when "
        "the problem is over-constrained).",
        "4. The failure-rate plot shows the **COP graceful-degradation** "
        "promised by the spec: when full assignment is impossible, every "
        "solver returns a best-found partial assignment instead of crashing.",
        "",
    ]
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    main()
