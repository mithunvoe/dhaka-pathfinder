#!/usr/bin/env bash
# Convenience wrapper — uses uv (https://docs.astral.sh/uv/) for env + dependency management.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "or on macOS:"
    echo "    brew install uv"
    exit 1
fi

# `uv sync` is idempotent — on first call it creates .venv and installs deps from
# pyproject.toml / uv.lock; on subsequent calls it only updates what changed.
uv sync --quiet

case "${1:-help}" in
    ui)        exec uv run streamlit run app.py "${@:2}" ;;
    download)  exec uv run python -m dhaka_pathfinder.cli download "${@:2}" ;;
    route)     exec uv run python -m dhaka_pathfinder.cli route "${@:2}" ;;
    compare)   exec uv run python scripts/run_comparison.py "${@:2}" ;;
    report)    exec uv run python scripts/generate_report.py "${@:2}" ;;
    maps)      exec uv run python scripts/generate_sample_maps.py "${@:2}" ;;
    test)      exec uv run pytest tests/ "${@:2}" ;;
    test-slow) exec uv run pytest -m slow tests/test_integration.py "${@:2}" ;;
    cli)       exec uv run python -m dhaka_pathfinder.cli "${@:2}" ;;
    python)    exec uv run python "${@:2}" ;;
    sync)      exec uv sync ;;
    lock)      exec uv lock ;;
    add)       shift; exec uv add "$@" ;;
    all)       uv run pytest -q tests/ --ignore=tests/test_integration.py
               uv run python scripts/run_comparison.py
               uv run python scripts/generate_report.py
               uv run python scripts/generate_sample_maps.py
               echo "Done. See results/ for outputs." ;;
    *)
        cat <<'HELP'
Dhaka Pathfinder — usage (powered by uv)

./run.sh ui                 Launch Streamlit web UI
./run.sh download           Pre-cache Dhaka OSM graph
./run.sh route <args>       CLI route query (see --help)
./run.sh compare            Run comparative analysis
./run.sh report             Regenerate Markdown report from latest CSV
./run.sh maps               Produce sample route maps
./run.sh test               Run fast unit tests
./run.sh test-slow          Run integration tests on real Dhaka graph
./run.sh cli <args>         Direct CLI passthrough
./run.sh python <args>      Run arbitrary Python in the env
./run.sh sync               Re-sync the venv from pyproject.toml / uv.lock
./run.sh lock               Regenerate the lock file
./run.sh add <pkg>          Add a new dependency (`uv add …`)
./run.sh all                Tests + compare + report + maps (end-to-end)

First run creates the .venv via `uv sync` (~30 s the first time, cached after).
HELP
        ;;
esac
