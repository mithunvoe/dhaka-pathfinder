"""Streamlit web UI for the Dhaka Pathfinder.

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from dhaka_pathfinder.algorithms import ALGORITHMS
from dhaka_pathfinder.config import LANDMARKS
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import haversine_m
from dhaka_pathfinder.engine import DhakaPathfinderEngine, EngineConfig
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES, HEURISTIC_INFO
from dhaka_pathfinder.visualizer import ALGO_COLORS, ALGO_LABELS, build_route_map


def _edge_road_name(graph, u: int, v: int) -> str:
    """Return the OSM street name for the edge (u, v), or a short fallback."""
    if not graph.has_edge(u, v):
        return "—"
    best_name = None
    for key in graph[u][v]:
        name = graph[u][v][key].get("name")
        highway = graph[u][v][key].get("highway", "")
        if name:
            if isinstance(name, list):
                name = name[0] if name else None
            if name:
                return str(name)
        if not best_name and highway:
            hw = highway[0] if isinstance(highway, list) else highway
            best_name = f"(unnamed {hw})"
    return best_name or "(unnamed road)"


def _nearest_landmark(lat: float, lon: float, max_km: float = 0.5) -> str | None:
    best_name = None
    best_d = max_km * 1000.0
    for name, (lmk_lat, lmk_lon) in LANDMARKS.items():
        d = haversine_m(lat, lon, lmk_lat, lmk_lon)
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def _pretty_area(area: str) -> str:
    return (area or "").replace("_", " ").title() or "—"

logging.basicConfig(level=logging.INFO)


st.set_page_config(
    page_title="Dhaka Pathfinder",
    page_icon="🛺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        padding: 14px 18px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetric"] label {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .algo-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        color: white; font-weight: 600; font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine(weight_preset: str) -> DhakaPathfinderEngine:
    engine = DhakaPathfinderEngine(EngineConfig(weight_preset=weight_preset))
    engine.load()
    return engine


st.title("🛺 Dhaka Pathfinder — Realistic Multi-Factor Routing")
st.caption(
    "A context-aware pathfinding simulation over Dhaka's OpenStreetMap road network. "
    "Chooses routes by road condition, traffic, safety, gender, vehicle, and time of day — "
    "not just distance."
)

with st.sidebar:
    st.header("Route Configuration")

    st.subheader("Source & Destination")
    landmark_names = sorted(LANDMARKS.keys())
    source_name = st.selectbox("Source landmark", landmark_names, index=landmark_names.index("Shahbag") if "Shahbag" in landmark_names else 0)
    dest_name = st.selectbox("Destination landmark", landmark_names, index=landmark_names.index("Motijheel") if "Motijheel" in landmark_names else 1)

    st.caption("Or enter coordinates manually:")
    use_manual = st.checkbox("Use manual coordinates", value=False)
    if use_manual:
        c1, c2 = st.columns(2)
        with c1:
            src_lat = st.number_input("Source lat", value=float(LANDMARKS[source_name][0]), format="%.6f")
            dst_lat = st.number_input("Dest lat", value=float(LANDMARKS[dest_name][0]), format="%.6f")
        with c2:
            src_lon = st.number_input("Source lon", value=float(LANDMARKS[source_name][1]), format="%.6f")
            dst_lon = st.number_input("Dest lon", value=float(LANDMARKS[dest_name][1]), format="%.6f")
    else:
        src_lat, src_lon = LANDMARKS[source_name]
        dst_lat, dst_lon = LANDMARKS[dest_name]

    st.subheader("Algorithm")
    algorithm = st.selectbox(
        "Search algorithm",
        options=list(ALGORITHMS.keys()),
        format_func=lambda a: ALGO_LABELS.get(a, a),
        index=list(ALGORITHMS.keys()).index("astar"),
    )
    compare_all = st.checkbox("Compare ALL algorithms", value=True)

    st.subheader("Heuristic")
    heuristic = st.selectbox(
        "Heuristic (informed algorithms only)",
        options=list(HEURISTIC_FACTORIES.keys()),
        index=list(HEURISTIC_FACTORIES.keys()).index("network_relaxed"),
        help=("`network_relaxed` is THE primary comprehensive heuristic — it takes every "
              "contextual factor (time, weather, age, gender, vehicle, road condition, "
              "safety, lighting, water-logging, crime, lanes, traffic) into account while "
              "remaining provably admissible. The other options exist for the Axis-B "
              "comparative analysis required by the brief."),
    )
    admissible_tag = "✅ admissible" if HEURISTIC_INFO[heuristic].admissible else "⚠️ non-admissible"
    st.caption(f"{admissible_tag} — {HEURISTIC_INFO[heuristic].description}")

    st.subheader("Traveler Context")
    gender = st.selectbox("Gender", ["male", "female", "nonbinary"], index=0)
    social = st.selectbox("Social", ["alone", "accompanied"], index=0)
    age = st.selectbox("Age group", ["adult", "child", "elderly"], index=0,
                       help="Children and the elderly face higher risk/crime amplification and may be forbidden from motorbikes.")
    vehicle = st.selectbox("Vehicle", ["walk", "rickshaw", "cng", "motorbike", "car", "bus"], index=4)
    time_bucket = st.selectbox(
        "Time of day",
        ["early_morning", "morning_rush", "midday", "afternoon",
         "evening_rush", "evening", "late_night"],
        index=2,
    )
    weather = st.selectbox("Weather", ["clear", "rain", "fog", "storm", "heat"], index=0,
                            help="Weather amplifies water-logging, lighting and risk multipliers.")

    st.subheader("Cost Preset")
    preset = st.selectbox("Weight preset", ["balanced", "safety", "speed", "comfort"], index=0)

    run_button = st.button("🚀 Compute route", type="primary", use_container_width=True)

engine = load_engine(preset)

s_node = engine.nearest(src_lat, src_lon)
d_node = engine.nearest(dst_lat, dst_lon)
ctx = TravelContext(gender=gender, social=social, age=age, vehicle=vehicle,
                    time_bucket=time_bucket, weather=weather, weight_preset=preset)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Source node", f"{s_node}")
col_info2.metric("Destination node", f"{d_node}")
col_info3.metric("Graph nodes", f"{engine.graph.number_of_nodes():,}")
col_info4.metric("Graph edges", f"{engine.graph.number_of_edges():,}")


if run_button or "results" not in st.session_state:
    with st.spinner("Running search…"):
        if compare_all:
            results = {algo: engine.solve(algo, s_node, d_node, ctx, heuristic) for algo in ALGORITHMS}
        else:
            results = {algorithm: engine.solve(algorithm, s_node, d_node, ctx, heuristic)}
        st.session_state["results"] = results
        st.session_state["ctx_label"] = ctx.label()

results: dict = st.session_state.get("results", {})

if results:
    st.markdown("### Route Comparison")
    rows = []
    for name, r in results.items():
        s = r.stats
        rows.append({
            "Algorithm": ALGO_LABELS.get(name, name),
            "Success": "✅" if s.success else "❌",
            "Cost": f"{s.path_cost:,.1f}",
            "Length (km)": f"{s.path_length_meters/1000:.2f}",
            "Edges": s.path_length_edges,
            "Nodes expanded": s.nodes_expanded,
            "Revisits": s.revisits,
            "EBF": f"{s.effective_branching_factor:.2f}",
            "Runtime (ms)": f"{s.runtime_seconds*1000:.1f}",
            "h(source)": f"{s.predicted_cost_at_start:,.1f}",
            "gap (actual-h)": f"{s.predicted_vs_actual_gap:,.1f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("📍 Show the exact node sequence for each route", expanded=False):
        st.caption(
            "Every intermediate intersection each algorithm passes through. "
            "**Road** is the OSM street name of the segment you follow to reach the next "
            "intersection. **Area** is the Dhaka neighbourhood. **Landmark** is filled in "
            "when a node is within 500 m of a known landmark (always set at source/dest)."
        )
        tab_labels = [ALGO_LABELS.get(a, a) for a, r in results.items() if r.path]
        tab_keys = [a for a, r in results.items() if r.path]
        if tab_keys:
            tabs = st.tabs(tab_labels)
            G = engine.graph
            for tab, algo in zip(tabs, tab_keys):
                with tab:
                    r = results[algo]
                    rows_path = []
                    cum_m = 0.0
                    for i, node in enumerate(r.path):
                        lat, lon = engine.coords(node)
                        if i == 0:
                            seg_m = 0.0
                            road_name = "(start)"
                            marker = "🟢 Source"
                        else:
                            prev = r.path[i - 1]
                            seg_m = 0.0
                            if G.has_edge(prev, node):
                                seg_m = min(
                                    float(G[prev][node][k].get("length", 0.0))
                                    for k in G[prev][node]
                                )
                            road_name = _edge_road_name(G, prev, node)
                            marker = "🔴 Destination" if i == len(r.path) - 1 else ""
                        cum_m += seg_m
                        node_data = G.nodes[node]
                        area = _pretty_area(node_data.get("area_name", ""))
                        landmark = _nearest_landmark(lat, lon, max_km=0.5) or ""
                        rows_path.append({
                            "Step": i + 1,
                            "Road (to get here)": road_name,
                            "Area": area,
                            "Landmark": landmark,
                            "Segment (m)": round(seg_m, 1),
                            "Cumulative (m)": round(cum_m, 1),
                            "Lat": round(lat, 5),
                            "Lon": round(lon, 5),
                            "Node ID": int(node),
                            "Marker": marker,
                        })
                    st.caption(
                        f"**{len(r.path)} intersections**, "
                        f"**{r.stats.path_length_edges} road segments**, "
                        f"**{r.stats.path_length_meters/1000:.2f} km total**, "
                        f"realistic cost **{r.stats.path_cost:,.1f}**."
                    )

                    df_path = pd.DataFrame(rows_path)
                    unique_roads = [
                        r for r in df_path["Road (to get here)"].unique()
                        if r not in ("(start)", "—") and not r.startswith("(unnamed")
                    ]
                    if unique_roads:
                        road_preview = ", ".join(unique_roads[:8])
                        if len(unique_roads) > 8:
                            road_preview += f", … ({len(unique_roads) - 8} more)"
                        st.markdown(f"**Roads used:** {road_preview}")

                    st.dataframe(
                        df_path,
                        use_container_width=True,
                        hide_index=True,
                        height=min(320, 38 + 35 * len(rows_path)),
                    )
        else:
            st.info("No successful paths to list.")

    col_map, col_panel = st.columns([3, 1])

    with col_map:
        st.markdown("### Map")
        m = build_route_map(
            engine.graph, results, s_node, d_node,
            title=f"{source_name} → {dest_name} | {ctx.label()}",
        )
        st_folium(m, use_container_width=True, height=620, returned_objects=[])

    with col_panel:
        st.markdown("### Legend")
        for algo in results:
            color = ALGO_COLORS.get(algo, "#666")
            label = ALGO_LABELS.get(algo, algo)
            st.markdown(
                f"<span class='algo-badge' style='background:{color};'>{label}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🏆 Category Winners")
        succ = [(a, r) for a, r in results.items() if r.stats.success]
        if not succ:
            st.error("No algorithm found a path.")
        else:
            def _winners_by(key, smaller_is_better=True, tol=1e-3):
                vals = [(a, r, key(r)) for a, r in succ]
                best_val = min(v for _, _, v in vals) if smaller_is_better else max(v for _, _, v in vals)
                winners = [(a, r, v) for a, r, v in vals if abs(v - best_val) <= tol]
                return winners, best_val

            cost_winners, best_cost = _winners_by(lambda r: r.stats.path_cost)
            expand_winners, best_expand = _winners_by(lambda r: r.stats.nodes_expanded)
            time_winners, best_time = _winners_by(lambda r: r.stats.runtime_seconds)
            length_winners, best_length = _winners_by(lambda r: r.stats.path_length_meters)

            def _names(winners: list) -> str:
                raw = " / ".join(ALGO_LABELS.get(a, a) for a, _, _ in winners)
                return raw.replace("*", "\\*")

            st.success(f"💰 **Lowest cost** — {_names(cost_winners)}  \n`{best_cost:,.1f}`")
            st.info(f"🎯 **Fewest nodes expanded** — {_names(expand_winners)}  \n`{int(best_expand):,}` nodes")
            st.info(f"⚡ **Fastest** — {_names(time_winners)}  \n`{best_time*1000:.2f} ms`")
            st.info(f"📏 **Shortest distance** — {_names(length_winners)}  \n`{best_length/1000:.3f} km`")

            if len(cost_winners) > 1:
                st.caption(
                    f"{len(cost_winners)}-way tie on cost is expected — UCS, A★, and "
                    "Weighted A★ are all provably optimal. What distinguishes them is "
                    "how efficiently they get there (see the other categories)."
                )


with st.expander("🔬 How the cost is computed", expanded=False):
    st.markdown(
        """
        **Actual edge cost** = base_length × Π multipliers, where each multiplier comes from:

        * **Road-intrinsic** — surface condition, highway class, water-logging propensity, lighting.
        * **Dynamic** — baseline traffic × time-of-day traffic amplifier.
        * **Safety** — accident risk, crime index, area-level safety profile.
        * **Traveler** — gender × social (alone vs accompanied), vehicle-highway suitability.

        Heuristics (for informed search) use haversine distance blended with these factors.
        The **haversine_admissible** heuristic uses the best-case cost-per-meter under the active
        context, so it never overestimates — guaranteeing A* optimality.
        """
    )


with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown(
        """
        - Data: **OpenStreetMap** via [OSMnx](https://osmnx.readthedocs.io/) (live, cached locally).
        - Synthetic attributes (condition, traffic, risk, safety, lighting, crime, water-logging)
          are generated deterministically from real OSM tags + coordinates. The same seed reproduces the same graph every run.
        - Six algorithms: BFS, DFS, UCS (uninformed); Greedy, A*, Weighted A* (informed).
          Uninformed algorithms report path cost under the realistic metric, as required.
        - Five heuristics: `zero`, `haversine_admissible`, `haversine_time`, `context_aware`, `learned_history`.
        """
    )
