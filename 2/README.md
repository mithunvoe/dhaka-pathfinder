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
./run.sh ui              # 🎯 interactive map UI (Streamlit) — primary entry point
./run.sh test            # run unit tests
./run.sh experiments     # fast comparative sweep -> results/*.csv + results/plots/*.png
./run.sh experiments --full   # paper-grade matrix (5 seeds × 5 sizes × 6 s budget)
./run.sh report          # build results/REPORT.md
./run.sh notebook        # build notebooks/fuel_csp_analysis.ipynb
./run.sh solve --algo bt_fc_mrv_deg -n 20 --seed 42
./run.sh all             # tests + experiments + report + notebook
```

The **UI** is the easiest way to see what this project does:

* **Real Dhaka OSM map** with real fuel-station POIs (queried from
  `amenity=fuel` via OSMnx, cached to `data/`).
* Vehicles are random OSM road nodes inside the central-Dhaka bounding box.
* **Routes follow the actual road network** — each assignment is drawn as
  a folium `PolyLine` that traces the shortest road path from the
  vehicle's OSM node to its assigned station.
* Sliders for `N`, number of stations, time slots, and seed.
* A dropdown to pick any of the 5 algorithms — or tick **"Compare ALL"**
  to run every solver on the same instance and see a side-by-side metric
  table.
* Hovering over a vehicle shows its kind, fuel, range, priority, and the
  station / pump / time-slot it was assigned to.
* For Min-Conflicts, a live convergence chart shows `J(S)` decreasing
  step-by-step.
* Second tab — **Comparative plots** — embeds every scaling chart from
  `results/plots/` inline.

The first launch downloads the Dhaka graph from OpenStreetMap
(~10 MB, ~30 s) and caches it as `data/dhaka_drive.pkl`. Every subsequent
launch loads from cache in under a second.

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
