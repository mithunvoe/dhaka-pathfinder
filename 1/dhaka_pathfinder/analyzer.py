"""Comparative analyzer — runs all algorithms × heuristics × contexts across many pairs."""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from dhaka_pathfinder.algorithms import ALGORITHMS, INFORMED
from dhaka_pathfinder.config import RESULTS_DIR
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel, haversine_m
from dhaka_pathfinder.engine import DhakaPathfinderEngine
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES, make_heuristic

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    num_pairs: int = 100
    min_distance_m: float = 1200.0
    max_distance_m: float = 12000.0
    pair_seed: int = 42
    heuristics: tuple[str, ...] = (
        "zero", "haversine_admissible", "network_relaxed",
        "haversine_time", "context_aware", "learned_history",
    )
    contexts: tuple[TravelContext, ...] = (
        TravelContext(gender="male",   social="alone",        age="adult",   vehicle="car",       time_bucket="midday",       weather="clear"),
        TravelContext(gender="male",   social="alone",        age="adult",   vehicle="motorbike", time_bucket="morning_rush", weather="clear"),
        TravelContext(gender="female", social="alone",        age="adult",   vehicle="rickshaw",  time_bucket="late_night",   weather="clear"),
        TravelContext(gender="female", social="accompanied",  age="adult",   vehicle="cng",       time_bucket="evening_rush", weather="rain"),
        TravelContext(gender="female", social="alone",        age="adult",   vehicle="walk",      time_bucket="evening",      weather="fog"),
        TravelContext(gender="male",   social="accompanied",  age="child",   vehicle="walk",      time_bucket="midday",       weather="clear"),
        TravelContext(gender="female", social="accompanied",  age="elderly", vehicle="car",       time_bucket="afternoon",    weather="clear"),
    )
    weighted_astar_weight: float = 1.8
    max_nodes_per_algo: int = 200_000
    dfs_max_depth: int = 1500


def _sample_pairs(
    G: nx.MultiDiGraph,
    num_pairs: int,
    min_m: float,
    max_m: float,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph too small to sample source/destination pairs.")
    coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes}

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 200:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        key = (u, v)
        if key in seen:
            continue
        d = haversine_m(*coords[u], *coords[v])
        if not (min_m <= d <= max_m):
            continue
        seen.add(key)
        pairs.append((u, v))
    if len(pairs) < num_pairs:
        logger.warning("Only sampled %d/%d pairs within distance band", len(pairs), num_pairs)
    return pairs


def run_comparative_analysis(
    engine: DhakaPathfinderEngine,
    config: AnalyzerConfig | None = None,
    save_csv: Path | None = None,
) -> pd.DataFrame:
    """Main entry point. Returns a flat DataFrame with one row per (pair, algo, heuristic, context)."""
    if engine.graph is None:
        raise RuntimeError("Engine not loaded.")
    config = config or AnalyzerConfig()
    G = engine.graph

    pairs = _sample_pairs(
        G, config.num_pairs, config.min_distance_m, config.max_distance_m, config.pair_seed,
    )
    logger.info("Running analysis over %d pairs × %d algos × %d heuristics × %d contexts",
                len(pairs), len(ALGORITHMS), len(config.heuristics), len(config.contexts))

    rows: list[dict[str, object]] = []
    total = len(pairs) * len(config.contexts) * (
        len(ALGORITHMS) - len(INFORMED) + len(INFORMED) * len(config.heuristics)
    )
    bar = tqdm(total=total, desc="analysis", unit="run", mininterval=0.5)

    for pair_idx, (src, dst) in enumerate(pairs):
        pair_dist = haversine_m(
            *engine.coords(src), *engine.coords(dst),
        )
        for ctx in config.contexts:
            weights = engine._weights_for(ctx)

            for algo_name in ALGORITHMS:
                if algo_name in INFORMED:
                    heuristic_names = config.heuristics
                else:
                    heuristic_names = ("n/a",)

                for h_name in heuristic_names:
                    try:
                        if algo_name in INFORMED:
                            h_fn = make_heuristic(h_name, G, dst, ctx, engine.cost_model)
                            if algo_name == "weighted_astar":
                                result = ALGORITHMS[algo_name](
                                    G, src, dst, weights, heuristic=h_fn,
                                    weight=config.weighted_astar_weight,
                                    max_nodes=config.max_nodes_per_algo,
                                )
                            else:
                                result = ALGORITHMS[algo_name](
                                    G, src, dst, weights, heuristic=h_fn,
                                    max_nodes=config.max_nodes_per_algo,
                                )
                        elif algo_name == "dfs":
                            result = ALGORITHMS[algo_name](
                                G, src, dst, weights,
                                max_depth=config.dfs_max_depth,
                                max_nodes=config.max_nodes_per_algo,
                            )
                        else:
                            result = ALGORITHMS[algo_name](
                                G, src, dst, weights, max_nodes=config.max_nodes_per_algo,
                            )
                        s = result.stats
                        rows.append({
                            "pair_id": pair_idx,
                            "source": src,
                            "destination": dst,
                            "euclid_m": pair_dist,
                            "algorithm": algo_name,
                            "heuristic": h_name,
                            "gender": ctx.gender,
                            "social": ctx.social,
                            "age": ctx.age,
                            "vehicle": ctx.vehicle,
                            "time_bucket": ctx.time_bucket,
                            "weather": ctx.weather,
                            "success": s.success,
                            "nodes_expanded": s.nodes_expanded,
                            "nodes_generated": s.nodes_generated,
                            "revisits": s.revisits,
                            "max_frontier_size": s.max_frontier_size,
                            "path_length_edges": s.path_length_edges,
                            "path_length_meters": s.path_length_meters,
                            "path_cost": s.path_cost,
                            "depth": s.depth,
                            "effective_branching_factor": s.effective_branching_factor,
                            "predicted_cost_at_start": s.predicted_cost_at_start,
                            "predicted_vs_actual_gap": s.predicted_vs_actual_gap,
                            "runtime_seconds": s.runtime_seconds,
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Run failed %s %s %s: %s", algo_name, h_name, ctx.label(), exc)
                        rows.append({
                            "pair_id": pair_idx,
                            "source": src,
                            "destination": dst,
                            "euclid_m": pair_dist,
                            "algorithm": algo_name,
                            "heuristic": h_name,
                            "gender": ctx.gender,
                            "social": ctx.social,
                            "age": ctx.age,
                            "vehicle": ctx.vehicle,
                            "time_bucket": ctx.time_bucket,
                            "weather": ctx.weather,
                            "success": False,
                            "error": str(exc),
                        })
                    finally:
                        bar.update(1)
    bar.close()

    df = pd.DataFrame(rows)
    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Saved comparison matrix to %s", save_csv)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Median-centric per-algorithm summary used in the report."""
    agg = (
        df[df["success"] == True]
        .groupby("algorithm")
        .agg(
            runs=("algorithm", "size"),
            success_rate=("success", "mean"),
            nodes_expanded_median=("nodes_expanded", "median"),
            nodes_expanded_mean=("nodes_expanded", "mean"),
            cost_median=("path_cost", "median"),
            cost_mean=("path_cost", "mean"),
            cost_std=("path_cost", "std"),
            length_km_median=("path_length_meters", lambda s: float(np.median(s) / 1000)),
            revisits_median=("revisits", "median"),
            ebf_median=("effective_branching_factor", "median"),
            depth_median=("depth", "median"),
            runtime_ms_median=("runtime_seconds", lambda s: float(np.median(s) * 1000)),
            predicted_gap_median=("predicted_vs_actual_gap", "median"),
        )
        .round(3)
    )
    return agg.reset_index()


def summarise_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["success"] == True) & (df["heuristic"] != "n/a")]
    if sub.empty:
        return pd.DataFrame()
    agg = (
        sub.groupby(["algorithm", "heuristic"])
        .agg(
            runs=("heuristic", "size"),
            cost_median=("path_cost", "median"),
            nodes_expanded_median=("nodes_expanded", "median"),
            runtime_ms_median=("runtime_seconds", lambda s: float(np.median(s) * 1000)),
            gap_mean=("predicted_vs_actual_gap", "mean"),
        )
        .round(3)
    )
    return agg.reset_index()


def summarise_contexts(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["success"] == True]
    agg = (
        sub.groupby(["algorithm", "gender", "vehicle", "time_bucket"])
        .agg(
            runs=("algorithm", "size"),
            cost_median=("path_cost", "median"),
            nodes_expanded_median=("nodes_expanded", "median"),
        )
        .round(3)
    )
    return agg.reset_index()
