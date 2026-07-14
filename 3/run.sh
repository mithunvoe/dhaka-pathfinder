#!/usr/bin/env bash
# =====================================================================
# Assignment 3 - task runner
#   Part A : Population-based search  (PSO, Wi-Fi access-point placement)
#   Part B : Decision making          (Value Iteration + Q-learning, water tank)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-help}"
shift || true

# Use the local venv if one exists (the UI needs streamlit); otherwise plain python3.
PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

case "$CMD" in
  ui)
    # Interactive demo: both parts of the assignment, in a browser.
    if [ ! -x .venv/bin/python ]; then
      echo "Creating .venv (one-off)..."
      uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
    fi
    exec .venv/bin/python -m streamlit run app.py "$@"
    ;;
  swarm)
    # Part A: from-scratch PSO + Random/Grid baselines + the collective ablation.
    python3 src/pso_wifi_placement.py
    ;;
  rl)
    # Part B: Value Iteration, Q-learning, certainty-equivalence, all experiments.
    # Pass --quick for a ~40 s sanity run instead of the full ~4 min one.
    python3 src/rl_experiments.py "$@"
    ;;
  all)
    python3 src/pso_wifi_placement.py
    echo
    python3 src/rl_experiments.py
    ;;
  test)
    python3 -m pytest tests/ -v
    ;;
  pdfs)
    # Turn every markdown doc into a PDF (pdf/), plus one combined dossier.
    exec scripts/build_pdfs.sh "$@"
    ;;
  report)
    # Build the LaTeX report (needs pdflatex + bibtex).
    cd report
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    bibtex main >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    rm -f main.aux main.log main.out main.bbl main.blg
    echo "report/main.pdf written"
    ;;
  help|*)
    cat <<'EOF'
Usage: ./run.sh <command>

  ui               Interactive demo of BOTH parts in a browser   <-- best for a live demo
  swarm            Part A: PSO Wi-Fi placement + baselines + collective ablation
  rl [--quick]     Part B: Value Iteration vs Q-learning (--quick = ~40 s)
  all              Both parts
  test             Unit tests (pytest)
  report           Compile report/main.pdf  (the actual submission)
  pdfs             Convert every markdown doc to PDF, into pdf/
  help             This message

Figures land in results/plots/ and tables in results/tables/.

  Part A                              Part B
  --------------------------------    ------------------------------------
  convergence.png                     rl_vi_convergence.png
  spatial_swarm.png                   rl_policy_maps.png
  diversity.png                       rl_learning_curves.png
  collective_behaviour.png            rl_model_mismatch.png
                                      rl_hyperparams.png
EOF
    ;;
esac
