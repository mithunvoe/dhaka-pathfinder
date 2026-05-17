"""Programmatically construct notebooks/fuel_csp_analysis.ipynb.

We build the notebook from code+markdown cells rather than maintaining a
checked-in .ipynb, so the analysis stays in sync with the actual code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf  # noqa: E402

OUT = PROJECT_ROOT / "notebooks" / "fuel_csp_analysis.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    cells = [
        md("# Fuel-CSP — Comparative Analysis Notebook\n\n"
           "This notebook re-runs the experiment matrix and renders every "
           "plot inline. It mirrors `scripts/run_experiments.py` and "
           "`scripts/generate_report.py`."),
        md("## 1. Setup"),
        code(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "%matplotlib inline\n"
            "from fuel_csp.analyzer import ExperimentConfig, run_matrix, save_csvs\n"
            "from fuel_csp.visualizer import (\n"
            "    plot_runtime, plot_nodes, plot_backtracks, plot_objective,\n"
            "    plot_failure_rate, plot_heuristic_bars,\n"
            "    plot_min_conflicts_convergence, plot_problem_topology,\n"
            ")\n"
            "from fuel_csp.synthetic import GeneratorConfig, generate_problem\n"
            "from fuel_csp.algorithms import ALL_SOLVERS\n"
        ),
        md("## 2. Run the experiment matrix\n\n"
           "We sweep N ∈ {10, 20, 30, 40, 50} and 5 random seeds per cell."),
        code(
            "cfg = ExperimentConfig(sizes=(10, 20, 30, 40, 50),\n"
            "                       seeds=(7, 13, 21, 42, 99),\n"
            "                       time_budget_s=6.0)\n"
            "df = run_matrix(cfg)\n"
            "df.head()\n"
        ),
        md("## 3. Per-(algorithm × N) summary"),
        code(
            "from fuel_csp.analyzer import summarise\n"
            "summary = summarise(df)\n"
            "summary\n"
        ),
        md("## 4. Plots — scalability"),
        code(
            "import pathlib\n"
            "out = pathlib.Path('../results/plots')\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "plot_runtime(df, out / 'runtime_vs_n.png');\n"
            "plot_nodes(df, out / 'nodes_vs_n.png');\n"
            "plot_backtracks(df, out / 'backtracks_vs_n.png');\n"
            "from IPython.display import Image, display\n"
            "for f in ['runtime_vs_n.png', 'nodes_vs_n.png', 'backtracks_vs_n.png']:\n"
            "    display(Image(out / f))\n"
        ),
        md("## 5. Plots — solution quality + failure rate"),
        code(
            "plot_objective(df, out / 'objective_vs_n.png');\n"
            "plot_failure_rate(df, out / 'failure_rate_vs_n.png');\n"
            "plot_heuristic_bars(df, out / 'heuristic_bars.png');\n"
            "for f in ['objective_vs_n.png', 'failure_rate_vs_n.png',\n"
            "          'heuristic_bars.png']:\n"
            "    display(Image(out / f))\n"
        ),
        md("## 6. Min-Conflicts convergence"),
        code(
            "from fuel_csp.analyzer import run_one\n"
            "traces = [run_one('min_conflicts', 30, s, cfg).stats.cost_trace\n"
            "          for s in cfg.seeds]\n"
            "plot_min_conflicts_convergence(traces, out / 'min_conflicts_convergence.png');\n"
            "display(Image(out / 'min_conflicts_convergence.png'))\n"
        ),
        md("## 7. Sample instance topology"),
        code(
            "problem = generate_problem(GeneratorConfig(num_vehicles=30, seed=7))\n"
            "solver = ALL_SOLVERS['bt_fc_mrv_deg'](time_budget_s=6.0)\n"
            "res = solver.solve(problem)\n"
            "plot_problem_topology(problem, res.assignment, out / 'sample_topology.png');\n"
            "display(Image(out / 'sample_topology.png'))\n"
        ),
        md("## 8. Take-aways\n\n"
           "* Basic backtracking blows up.\n"
           "* MRV / LCV / Forward-Checking each tame the explosion.\n"
           "* Min-Conflicts scales nearly flat but is not complete.\n"
           "* Every solver returns a best-partial assignment under load — the\n"
           "  graceful-failure behavior the COP formulation demands."),
    ]
    nb["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
