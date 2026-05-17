# Fuel-CSP — Urban Fuel-Crisis Allocator (Assignment 2)

A Constraint Satisfaction / Optimization solver for assigning vehicles to fuel
stations during a citywide fuel shortage. Five search algorithms are
implemented and compared empirically:

1. **Basic Backtracking**
2. **Backtracking + MRV** (Minimum Remaining Values)
3. **Backtracking + LCV** (Least Constraining Value)
4. **Backtracking + Forward Checking + MRV + Degree tie-break**
5. **Min-Conflicts Local Search**

See [`assignment.md`](assignment.md) for the full problem statement and
[`guide.md`](guide.md) for the in-depth walkthrough.

## Quick start

```bash
./run.sh test            # run unit tests
./run.sh experiments     # full sweep -> results/*.csv + results/plots/*.png
./run.sh report          # build results/REPORT.md
./run.sh notebook        # build notebooks/fuel_csp_analysis.ipynb
./run.sh solve --algo bt_fc_mrv_deg -n 20 --seed 42
./run.sh all             # tests + experiments + report + notebook
```

## Layout

```
2/
├── assignment.md          formal problem statement
├── guide.md               defense-ready walkthrough
├── fuel_csp/              package
│   ├── problem.py         X, D, C formulation
│   ├── synthetic.py       data generator
│   ├── constraints.py     hard checks + COP objective J(S)
│   ├── algorithms/        BT (4 variants) + Min-Conflicts
│   ├── analyzer.py        experiment matrix runner
│   ├── visualizer.py      matplotlib/seaborn plots
│   └── cli.py             click CLI
├── scripts/               run_experiments / generate_report / build_notebook
├── tests/                 pytest suite
├── results/               CSVs + PNGs + REPORT.md
└── notebooks/             fuel_csp_analysis.ipynb
```
