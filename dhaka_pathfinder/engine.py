"""High-level orchestration — load graph, augment, solve, and report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx

from dhaka_pathfinder.algorithms import ALGORITHMS, INFORMED, SearchResult
from dhaka_pathfinder.config import DATA_DIR
from dhaka_pathfinder.context import TravelContext
from dhaka_pathfinder.cost_model import RealisticCostModel, WEIGHT_PRESETS
from dhaka_pathfinder.heuristics import HEURISTIC_FACTORIES, make_heuristic
from dhaka_pathfinder.osm_loader import (
    GraphLoadSpec,
    ensure_graph,
    largest_strongly_connected_subgraph,
    load_dhaka_graph,
    nearest_node,
    node_coords,
)
from dhaka_pathfinder.synthetic_data import SyntheticConfig, augment_graph

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    spec: GraphLoadSpec = GraphLoadSpec()
    synth: SyntheticConfig = SyntheticConfig()
    weight_preset: str = "balanced"
    deterministic_weights: bool = True


class DhakaPathfinderEngine:
    """Holds the graph, the cost model, and an edge-weight cache keyed by context."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.graph: nx.MultiDiGraph | None = None
        self._weight_cache: dict[str, dict[tuple[int, int, int], float]] = {}
        self._cost_model: RealisticCostModel | None = None

    def load(self, force_refresh: bool = False) -> nx.MultiDiGraph:
        G = load_dhaka_graph(self.config.spec, force_refresh=force_refresh)
        G = largest_strongly_connected_subgraph(G)
        augment_graph(G, self.config.synth)
        self.graph = G
        self._cost_model = RealisticCostModel(
            weights=self.config.weight_preset,
            stochastic=not self.config.deterministic_weights,
        )
        logger.info("Engine ready: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    @property
    def cost_model(self) -> RealisticCostModel:
        if self._cost_model is None:
            raise RuntimeError("Engine not loaded; call .load() first.")
        return self._cost_model

    def _weights_for(self, context: TravelContext) -> dict[tuple[int, int, int], float]:
        if self.graph is None:
            raise RuntimeError("Engine not loaded; call .load() first.")
        key = context.label()
        if key not in self._weight_cache:
            self._weight_cache[key] = self.cost_model.precompute_edge_weights(
                self.graph, context, deterministic=True,
            )
        return self._weight_cache[key]

    def nearest(self, lat: float, lon: float) -> int:
        if self.graph is None:
            raise RuntimeError("Engine not loaded; call .load() first.")
        return nearest_node(self.graph, lat, lon)

    def coords(self, node: int) -> tuple[float, float]:
        if self.graph is None:
            raise RuntimeError("Engine not loaded; call .load() first.")
        return node_coords(self.graph, node)

    def solve(
        self,
        algorithm: str,
        source: int,
        destination: int,
        context: TravelContext | None = None,
        heuristic_name: str = "network_relaxed",
        **kwargs: object,
    ) -> SearchResult:
        if self.graph is None:
            raise RuntimeError("Engine not loaded; call .load() first.")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from {sorted(ALGORITHMS)}.")
        context = context or TravelContext.default()
        weights = self._weights_for(context)
        fn = ALGORITHMS[algorithm]

        if algorithm in INFORMED:
            h = make_heuristic(heuristic_name, self.graph, destination, context, self.cost_model)
            result = fn(self.graph, source, destination, weights, heuristic=h, **kwargs)
            result.stats.heuristic_name = heuristic_name
        else:
            result = fn(self.graph, source, destination, weights, **kwargs)
            result.stats.heuristic_name = "n/a"
        return result

    def solve_all(
        self,
        source: int,
        destination: int,
        context: TravelContext | None = None,
        heuristic_name: str = "network_relaxed",
    ) -> dict[str, SearchResult]:
        out: dict[str, SearchResult] = {}
        for name in ALGORITHMS:
            out[name] = self.solve(name, source, destination, context, heuristic_name)
        return out
