#!/usr/bin/env bash
# Convenience wrapper for the fuel-csp project. Uses uv for env management.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync --quiet

case "${1:-help}" in
    test)      exec uv run pytest tests/ "${@:2}" ;;
    cli)       exec uv run python -m fuel_csp.cli "${@:2}" ;;
    experiments) exec uv run python scripts/run_experiments.py "${@:2}" ;;
    report)    exec uv run python scripts/generate_report.py "${@:2}" ;;
    notebook)  exec uv run python scripts/build_notebook.py "${@:2}" ;;
    solve)     exec uv run python -m fuel_csp.cli solve "${@:2}" ;;
    all)       uv run pytest tests/ -q
               uv run python scripts/run_experiments.py
               uv run python scripts/generate_report.py
               uv run python scripts/build_notebook.py
               echo "Done. See results/ and notebooks/." ;;
    *)
        cat <<'HELP'
fuel-csp — usage

./run.sh test            Run unit tests
./run.sh cli <args>      Direct CLI passthrough
./run.sh solve <args>    Solve one instance (algorithm, N, seed)
./run.sh experiments     Run the full scalability sweep (writes CSVs + plots)
./run.sh report          Regenerate REPORT.md from the latest CSV
./run.sh notebook        Build notebooks/fuel_csp_analysis.ipynb
./run.sh all             tests + experiments + report + notebook
HELP
        ;;
esac
