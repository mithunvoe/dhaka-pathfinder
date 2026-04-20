"""Streamlit web UI for the Dhaka Pathfinder.

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from dhaka_pathfinder.algorithms import ALGORITHMS
from dhaka_pathfinder.config import DHAKA_CENTER, LANDMARKS
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.engine import DhakaPathfinderEngine, EngineConfig
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES, HEURISTIC_INFO
from dhaka_pathfinder.visualizer import ALGO_COLORS, ALGO_LABELS, build_route_map

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
    .stMetric { background: #fafafa; padding: 6px 10px; border-radius: 8px; }
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
        index=list(HEURISTIC_FACTORIES.keys()).index("haversine_admissible"),
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

        st.markdown("### Best path by cost")
        best_algo, best = min(
            ((a, r) for a, r in results.items() if r.stats.success),
            key=lambda t: t[1].stats.path_cost,
            default=(None, None),
        )
        if best_algo:
            st.success(f"{ALGO_LABELS.get(best_algo, best_algo)} — cost {best.stats.path_cost:,.1f}")
        else:
            st.error("No algorithm found a path.")


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
