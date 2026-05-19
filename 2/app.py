"""Streamlit UI for the fuel-CSP solver — Dhaka map edition.

Run:
    ./run.sh ui
"""

from __future__ import annotations

import time
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium

from fuel_csp.algorithms import ALL_SOLVERS
from fuel_csp.algorithms.base import SolverResult
from fuel_csp.dhaka import DhakaConfig, generate_dhaka_problem
from fuel_csp.osm_data import DHAKA_CENTER, shortest_path_latlon
from fuel_csp.problem import PRIORITY, Problem

PRETTY = {
    "basic_backtracking": "Basic Backtracking",
    "bt_mrv": "BT + MRV",
    "bt_lcv": "BT + LCV",
    "bt_fc_mrv_deg": "BT + Forward Checking (FC + MRV + Degree)",
    "min_conflicts": "Min-Conflicts (Local Search)",
}

KIND_COLOR = {
    "ambulance": "#d62728",
    "bus":       "#9467bd",
    "truck":     "#8c564b",
    "car":       "#1f77b4",
    "motorbike": "#7f7f7f",
}

KIND_ICON = {
    "ambulance": "ambulance",
    "bus":       "bus",
    "truck":     "truck",
    "car":       "car",
    "motorbike": "motorcycle",
}

FUEL_GLYPH = {"petrol": "P", "diesel": "D", "octane": "O"}

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


st.set_page_config(
    page_title="Fuel-CSP — Dhaka Fuel-Crisis Allocator",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        padding: 14px 18px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
    }
    [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; font-weight: 500 !important; }
    [data-testid="stMetricValue"] * { color: #ffffff !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- helpers -----------------------------------------

@st.cache_resource(show_spinner="Loading Dhaka OSM map (first time only)...")
def _load_problem_cached(n: int, num_stations: int, num_slots: int, seed: int):
    """Cached Problem builder. The graph itself is also cached on disk via osm_data."""
    return generate_dhaka_problem(
        DhakaConfig(
            num_vehicles=n,
            max_stations=num_stations,
            num_slots=num_slots,
            seed=seed,
        )
    )


def _solve(algo: str, problem: Problem, time_budget: float, seed: int) -> SolverResult:
    cls = ALL_SOLVERS[algo]
    if algo == "min_conflicts":
        solver = cls(time_budget_s=time_budget, max_steps=4000, seed=seed)
    else:
        solver = cls(time_budget_s=time_budget)
    return solver.solve(problem)


def _build_dhaka_map(
    problem: Problem,
    graph,
    assignment: dict | None,
    algo_name: str,
) -> folium.Map:
    m = folium.Map(
        location=list(DHAKA_CENTER),
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    # Station markers
    station_group = folium.FeatureGroup(name="⛽ Fuel stations", show=True)
    for s in problem.stations:
        popup = folium.Popup(
            html=(
                f"<b>{s.name}</b><br>"
                f"Pumps: {s.pumps}<br>"
                f"Open slots: {s.open_slot}–{s.close_slot - 1}<br>"
                f"Petrol: {s.stocks('petrol'):.0f} L &nbsp;"
                f"Diesel: {s.stocks('diesel'):.0f} L &nbsp;"
                f"Octane: {s.stocks('octane'):.0f} L"
            ),
            max_width=320,
        )
        folium.Marker(
            location=[s.lat, s.lon],
            popup=popup,
            tooltip=f"S{s.sid} — {s.name[:40]}",
            icon=folium.Icon(color="green", icon="tint", prefix="fa"),
        ).add_to(station_group)
    station_group.add_to(m)

    # Vehicle markers (grouped per kind, so the layer control toggles them as one)
    for kind, color in KIND_COLOR.items():
        kind_group = folium.FeatureGroup(name=f"🚗 {kind.title()}s", show=True)
        for v in problem.vehicles:
            if v.kind != kind:
                continue
            assigned = "—"
            if assignment and v.vid in assignment:
                a = assignment[v.vid]
                assigned = (
                    f"S{a.station_id} / pump {a.pump_id} / slot {a.slot_id}"
                )
            popup_html = (
                f"<b>v{v.vid:02d} — {kind}</b><br>"
                f"Fuel: {v.fuel_type} (need {v.demand_liters:.1f} L)<br>"
                f"Range: {v.range_km:.1f} km<br>"
                f"Window: slots {v.earliest_slot}–{v.latest_slot}<br>"
                f"Priority: {PRIORITY[kind]}<br>"
                f"<b>Assigned:</b> {assigned}"
            )
            folium.CircleMarker(
                location=[v.lat, v.lon],
                radius=6,
                color="black",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"v{v.vid:02d} {kind}",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(kind_group)
        kind_group.add_to(m)

    # Assignment polylines along the real road network
    if assignment:
        edges_group = folium.FeatureGroup(name="🛣️ Assigned routes", show=True)
        for vid, a in assignment.items():
            v = problem.vehicles[vid]
            s = problem.stations[a.station_id]
            try:
                path = shortest_path_latlon(graph, v.node_id, s.node_id)
            except Exception:  # noqa: BLE001
                path = [(v.lat, v.lon), (s.lat, s.lon)]
            color = KIND_COLOR.get(v.kind, "#1f77b4")
            folium.PolyLine(
                locations=path,
                color=color,
                weight=4,
                opacity=0.75,
                tooltip=(
                    f"v{vid:02d} ({v.kind}) → S{s.sid} "
                    f"[{problem.distance_km(vid, s.sid):.2f} km]"
                ),
            ).add_to(edges_group)
        edges_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = (
        '<div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);'
        ' z-index:9999; background:rgba(255,255,255,0.94); padding:8px 16px;'
        ' border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.2);'
        ' font-family:sans-serif; color:#0f172a; font-weight:600;">'
        f'{algo_name}'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))
    return m


# --------------------------- sidebar -----------------------------------------

with st.sidebar:
    st.header("Problem instance")
    n = st.slider("Number of vehicles (N)", 5, 60, 20, 1)
    num_stations = st.slider("Max fuel stations", 3, 24, 8)
    num_slots = st.slider("Number of time slots", 3, 10, 6)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    st.markdown("---")
    st.header("Algorithm")
    algo = st.selectbox(
        "Solver", options=list(ALL_SOLVERS.keys()), index=3,
        format_func=lambda a: PRETTY.get(a, a),
    )
    compare_all = st.checkbox(
        "Compare ALL 5 algorithms",
        value=False,
        help="Solve the same Dhaka instance with every solver and show metrics side-by-side.",
    )
    time_budget = st.slider("Time budget (s)", 0.2, 8.0, 1.5, 0.1)

    st.markdown("---")
    solve_btn = st.button("⚡  Solve", type="primary", width="stretch")


# --------------------------- header + tabs -----------------------------------

st.title("⛽  Fuel-CSP — Dhaka Urban Fuel-Crisis Allocator")
st.caption(
    "Assigns vehicles scattered across Dhaka to real OSM fuel stations during a "
    "citywide shortage. Routes follow the actual road network."
)

# Load (or fetch from cache) the problem and the underlying Dhaka graph.
with st.spinner("Building Dhaka problem instance..."):
    problem, graph = _load_problem_cached(n, num_stations, num_slots, seed)
st.session_state["graph"] = graph

tab_map, tab_plots, tab_info = st.tabs([
    "🗺️ Dhaka map", "📊 Comparative plots", "ℹ️ Instance details"
])


# --------------------------- solve state machine -----------------------------
# We persist the latest solve in st.session_state so the solved map STAYS on
# screen across reruns (e.g. when the user just drags the time-budget slider
# after a solve). The cached solve is invalidated whenever the underlying
# problem signature (N, stations, slots, seed) changes — at that point we
# revert to the unsolved view until the user presses Solve again.

ss = st.session_state
ss.setdefault("solve_count", 0)
ss.setdefault("last_solve", None)

problem_sig = (n, num_stations, num_slots, seed)
if ss.last_solve is not None and ss.last_solve["problem_sig"] != problem_sig:
    ss.last_solve = None

if solve_btn:
    ss.solve_count += 1
    if compare_all:
        results: dict[str, SolverResult] = {}
        progress = st.progress(0.0, text="Running 5 solvers...")
        for i, name in enumerate(ALL_SOLVERS):
            t0 = time.perf_counter()
            results[name] = _solve(name, problem, time_budget, seed)
            progress.progress(
                (i + 1) / len(ALL_SOLVERS),
                text=f"Finished {PRETTY[name]} in {time.perf_counter() - t0:.2f}s",
            )
        progress.empty()
        ss.last_solve = {
            "kind": "all",
            "problem_sig": problem_sig,
            "results": results,
        }
    else:
        t0 = time.perf_counter()
        res = _solve(algo, problem, time_budget, seed)
        ss.last_solve = {
            "kind": "single",
            "problem_sig": problem_sig,
            "algo": algo,
            "result": res,
            "elapsed": time.perf_counter() - t0,
        }


# --------------------------- tab: map ---------------------------------------

with tab_map:
    folium_key = f"folium_{ss.solve_count}_{problem_sig}"

    if ss.last_solve is None:
        m = _build_dhaka_map(
            problem, graph, None,
            f"Press “Solve” to assign vehicles (N={problem.n}, "
            f"{len(problem.stations)} stations, mode={problem.mode})",
        )
        st_folium(m, height=640, width=None, returned_objects=[], key=folium_key)

    elif ss.last_solve["kind"] == "all":
        results = ss.last_solve["results"]
        rows = []
        for name, r in results.items():
            s = r.stats
            rows.append({
                "Algorithm": PRETTY[name],
                "J(S)": round(s.objective, 1),
                "Assigned": f"{s.num_assigned}/{s.n}",
                "Failure %": f"{s.failure_rate * 100:.1f}",
                "Backtracks": s.backtracks,
                "Nodes": s.nodes_expanded,
                "Runtime (ms)": f"{s.runtime_seconds * 1000:.1f}",
                "Complete?": "✅" if s.success else "⚠️ partial",
            })
        st.subheader("Comparative metrics — same Dhaka instance, 5 algorithms")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        best_name = min(results, key=lambda k: results[k].stats.objective)
        best_res = results[best_name]
        st.success(
            f"🏆 Winner by J(S): **{PRETTY[best_name]}** — "
            f"J(S)={best_res.stats.objective:.1f}, "
            f"assigned={best_res.stats.num_assigned}/{best_res.stats.n}"
        )

        m = _build_dhaka_map(
            problem, graph, best_res.assignment,
            f"{PRETTY[best_name]} — J(S)={best_res.stats.objective:.1f}",
        )
        st_folium(m, height=640, width=None, returned_objects=[], key=folium_key)

    else:  # ss.last_solve["kind"] == "single"
        res = ss.last_solve["result"]
        elapsed = ss.last_solve["elapsed"]
        last_algo = ss.last_solve["algo"]
        s = res.stats

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("J(S)", f"{s.objective:.1f}")
        c2.metric("Assigned", f"{s.num_assigned}/{s.n}",
                  delta=f"{(1 - s.failure_rate) * 100:.0f}% placed")
        c3.metric("Backtracks", s.backtracks)
        c4.metric("Nodes", s.nodes_expanded)
        c5.metric("Runtime", f"{elapsed * 1000:.1f} ms")

        m = _build_dhaka_map(
            problem, graph, res.assignment,
            f"{PRETTY[last_algo]} — {s.num_assigned}/{s.n} assigned, "
            f"J(S)={s.objective:.1f}",
        )
        st_folium(m, height=640, width=None, returned_objects=[], key=folium_key)

        if last_algo == "min_conflicts" and res.stats.cost_trace:
            st.subheader("Min-Conflicts convergence — J(S) over repair steps")
            trace_df = pd.DataFrame({
                "step": list(range(len(res.stats.cost_trace))),
                "J(S)": res.stats.cost_trace,
            })
            st.line_chart(trace_df, x="step", y="J(S)", height=240)

        rows = []
        for vid in sorted(res.assignment):
            a = res.assignment[vid]
            v = problem.vehicles[vid]
            rows.append({
                "Vehicle": f"v{vid:02d}",
                "Kind": v.kind,
                "Fuel": f"{FUEL_GLYPH[v.fuel_type]} {v.fuel_type}",
                "→": f"S{a.station_id} / pump {a.pump_id} / slot {a.slot_id}",
                "Dist (km)": f"{problem.distance_km(vid, a.station_id):.2f}",
            })
        if rows:
            with st.expander(f"Assignment table ({len(rows)} vehicles placed)",
                             expanded=False):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        unassigned = [v for v in problem.vehicles if v.vid not in res.assignment]
        if unassigned:
            with st.expander(f"⚠️ Unassigned ({len(unassigned)})"):
                st.dataframe(
                    pd.DataFrame([
                        {"Vehicle": f"v{v.vid:02d}",
                         "Kind": v.kind,
                         "Fuel": v.fuel_type,
                         "Range (km)": v.range_km,
                         "Priority": PRIORITY[v.kind]}
                        for v in unassigned
                    ]),
                    width="stretch", hide_index=True,
                )


# --------------------------- tab: plots --------------------------------------

PLOT_SPECS = [
    ("runtime_vs_n.png",
     "Runtime scalability — execution time vs problem size (log-y)"),
    ("nodes_vs_n.png",
     "Search effort — nodes expanded vs problem size (log-y)"),
    ("backtracks_vs_n.png",
     "Backtracks vs problem size (log-y)"),
    ("objective_vs_n.png",
     "Solution quality — J(S) vs problem size"),
    ("failure_rate_vs_n.png",
     "Graceful failure rate vs problem size"),
    ("heuristic_bars.png",
     "Per-algorithm aggregates (runtime / backtracks / J(S))"),
]


def _rerun_experiments(sizes: tuple[int, ...], seeds: tuple[int, ...],
                       time_budget: float, progress_cb) -> None:
    """Run the full experiment matrix and regenerate every plot.

    Mirrors scripts/run_experiments.py but lives inline so we can stream
    progress into the Streamlit UI.
    """
    from fuel_csp.analyzer import ExperimentConfig, run_one, save_csvs
    from fuel_csp.visualizer import (
        plot_backtracks,
        plot_failure_rate,
        plot_heuristic_bars,
        plot_nodes,
        plot_objective,
        plot_runtime,
    )

    cfg = ExperimentConfig(sizes=sizes, seeds=seeds, time_budget_s=time_budget)
    total_runs = len(ALL_SOLVERS) * len(sizes) * len(seeds)

    rows = []
    done = 0
    for solver_name in ALL_SOLVERS:
        for n_ in sizes:
            for seed_ in seeds:
                res = run_one(solver_name, n_, seed_, cfg)
                rows.append(res.stats.as_dict())
                done += 1
                progress_cb(done / total_runs,
                            f"{PRETTY[solver_name]} — N={n_}, seed={seed_}")
    df = pd.DataFrame(rows)
    save_csvs(df, PROJECT_ROOT / "results")
    for fn, name in [
        (plot_runtime, "runtime_vs_n.png"),
        (plot_nodes, "nodes_vs_n.png"),
        (plot_backtracks, "backtracks_vs_n.png"),
        (plot_objective, "objective_vs_n.png"),
        (plot_failure_rate, "failure_rate_vs_n.png"),
        (plot_heuristic_bars, "heuristic_bars.png"),
    ]:
        fn(df, PLOTS_DIR / name)


with tab_plots:
    st.markdown(
        "Comparative plots are produced by sweeping the 5 solvers across "
        "`N ∈ {10, 20, 30, 40, 50}` and 3 random seeds each (synthetic mode "
        "— independent of the Dhaka instance shown on the map)."
    )

    head_cols = st.columns([2, 1, 2])
    with head_cols[0]:
        rerun_sizes = st.text_input("Problem sizes (N)", "10,20,30,40,50",
                                    help="comma-separated list")
    with head_cols[1]:
        rerun_seeds = st.text_input("Seeds", "7,13,42")
    with head_cols[2]:
        rerun_budget = st.slider("Time budget per cell (s)", 0.5, 6.0, 1.5, 0.1,
                                 key="rerun_budget")

    if st.button("🔄 Re-run experiments & refresh plots",
                 type="primary", width="stretch"):
        try:
            sizes = tuple(int(x.strip()) for x in rerun_sizes.split(",") if x.strip())
            seeds = tuple(int(x.strip()) for x in rerun_seeds.split(",") if x.strip())
        except ValueError:
            st.error("Sizes and Seeds must be comma-separated integers.")
            sizes = seeds = ()

        if sizes and seeds:
            total = len(ALL_SOLVERS) * len(sizes) * len(seeds)
            bar = st.progress(0.0, text=f"Running 0/{total} cells...")

            def _cb(frac: float, label: str) -> None:
                bar.progress(frac, text=f"{int(frac * total)}/{total}  —  {label}")

            t0 = time.perf_counter()
            _rerun_experiments(sizes, seeds, rerun_budget, _cb)
            bar.empty()
            st.success(
                f"Done in {time.perf_counter() - t0:.1f}s — "
                f"plots regenerated in `results/plots/`."
            )

    cols = st.columns(2)
    rendered = 0
    for i, (fname, caption) in enumerate(PLOT_SPECS):
        path = PLOTS_DIR / fname
        if not path.exists():
            continue
        with cols[i % 2]:
            st.image(str(path), caption=caption, width="stretch")
        rendered += 1

    if rendered == 0:
        st.info("No plots yet. Click **Re-run experiments** above to generate them.")


# --------------------------- tab: instance info ------------------------------

with tab_info:
    st.subheader("Stations")
    st.dataframe(
        pd.DataFrame([
            {
                "Station": f"S{s.sid}",
                "Name": s.name,
                "Pumps": s.pumps,
                "Petrol (L)": s.stocks("petrol"),
                "Diesel (L)": s.stocks("diesel"),
                "Octane (L)": s.stocks("octane"),
                "Lat": s.lat, "Lon": s.lon,
            }
            for s in problem.stations
        ]),
        width="stretch", hide_index=True,
    )

    st.subheader("Vehicle composition")
    veh_df = pd.DataFrame([
        {"kind": v.kind, "fuel_type": v.fuel_type, "priority": PRIORITY[v.kind]}
        for v in problem.vehicles
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(veh_df.groupby("kind").size().rename("count"),
                     width="stretch")
    with c2:
        st.dataframe(veh_df.groupby("fuel_type").size().rename("count"),
                     width="stretch")
