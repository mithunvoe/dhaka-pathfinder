"""Load Dhaka's road network from OpenStreetMap via OSMnx with disk caching."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from dhaka_pathfinder.config import DATA_DIR, DEFAULT_NETWORK_TYPE, DHAKA_BBOX, DHAKA_CENTER

logger = logging.getLogger(__name__)

ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = str(DATA_DIR / "osmnx_cache")


@dataclass(frozen=True)
class GraphLoadSpec:
    """Describes how to fetch the Dhaka graph. Acts as a cache key."""

    mode: str = "bbox"
    place: str = "Dhaka, Bangladesh"
    bbox: tuple[float, float, float, float] = DHAKA_BBOX
    center: tuple[float, float] = DHAKA_CENTER
    radius_m: int = 6000
    network_type: str = DEFAULT_NETWORK_TYPE
    simplify: bool = True

    def cache_filename(self) -> str:
        if self.mode == "place":
            token = self.place.lower().replace(",", "").replace(" ", "_")
        elif self.mode == "bbox":
            token = "bbox_" + "_".join(f"{v:.4f}" for v in self.bbox)
        else:
            token = f"point_{self.center[0]:.4f}_{self.center[1]:.4f}_r{self.radius_m}"
        return f"dhaka_{token}_{self.network_type}.pkl"


def load_dhaka_graph(
    spec: GraphLoadSpec | None = None,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """Load the Dhaka road network, using a pickled cache when available.

    The returned graph has float coordinates on nodes (`x`=lon, `y`=lat) and
    `length` in metres on edges — standard OSMnx output.
    """
    spec = spec or GraphLoadSpec()
    cache_path = DATA_DIR / spec.cache_filename()

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached graph from %s", cache_path)
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        return G

    logger.info("Downloading Dhaka graph via OSMnx (mode=%s) — this may take a few minutes...", spec.mode)
    if spec.mode == "place":
        G = ox.graph_from_place(spec.place, network_type=spec.network_type, simplify=spec.simplify)
    elif spec.mode == "bbox":
        G = ox.graph_from_bbox(bbox=spec.bbox, network_type=spec.network_type, simplify=spec.simplify)
    elif spec.mode == "point":
        G = ox.graph_from_point(
            spec.center, dist=spec.radius_m,
            network_type=spec.network_type, simplify=spec.simplify,
        )
    else:
        raise ValueError(f"Unknown mode: {spec.mode!r}")

    with open(cache_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached graph to %s", cache_path)
    return G


def graph_summary(G: nx.MultiDiGraph) -> dict[str, float | int]:
    """Quick stats on a graph — useful for sanity checks in tests and UI."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    lengths = np.array([d.get("length", 0.0) for _, _, d in G.edges(data=True)])
    ys = np.array([d["y"] for _, d in G.nodes(data=True)])
    xs = np.array([d["x"] for _, d in G.nodes(data=True)])
    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_edge_length_m": float(lengths.mean()) if lengths.size else 0.0,
        "total_length_km": float(lengths.sum() / 1000.0),
        "lat_min": float(ys.min()),
        "lat_max": float(ys.max()),
        "lon_min": float(xs.min()),
        "lon_max": float(xs.max()),
        "mean_degree": float(2 * num_edges / max(num_nodes, 1)),
    }


def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the node id closest to a (lat, lon) pair. Wraps OSMnx."""
    return int(ox.nearest_nodes(G, X=float(lon), Y=float(lat)))


def node_coords(G: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    """(lat, lon) for a node id."""
    data = G.nodes[node]
    return float(data["y"]), float(data["x"])


def largest_strongly_connected_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the largest strongly-connected component so every source can reach every destination."""
    comp = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


def ensure_graph(cache_dir: Path | None = None) -> nx.MultiDiGraph:
    """Convenience — load the default Dhaka graph, restricted to its largest SCC."""
    G = load_dhaka_graph()
    return largest_strongly_connected_subgraph(G)
