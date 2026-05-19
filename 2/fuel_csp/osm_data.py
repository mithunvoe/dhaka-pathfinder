"""
OSM data layer — caches the Dhaka road graph and computes road distances.

Public API:
    load_dhaka_graph()        -> nx.MultiDiGraph
    extract_fuel_stations(G)  -> list[StationPOI]
    sample_vehicle_nodes(G)   -> list[VehiclePOI]
    compute_distance_matrix() -> numpy.ndarray of shape (n_vehicles, n_stations)
    shortest_path_latlon()    -> list[(lat, lon)] for one (src_node, dst_node) pair

The first call downloads ~10 MB and takes ~30 s; subsequent calls hit the
pickle cache and return in well under a second.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Central Dhaka bbox — matches Assignment 1's smaller default.
DHAKA_BBOX: tuple[float, float, float, float] = (90.34, 23.70, 90.46, 23.86)
DHAKA_CENTER: tuple[float, float] = (23.78, 90.40)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "dhaka_drive.pkl"
STATIONS_FILE = DATA_DIR / "dhaka_fuel_stations.pkl"

# Well-known Dhaka landmarks — used as deterministic fuel-station fall-backs
# when the OSM `amenity=fuel` query is unavailable. The lat/lon for each
# landmark is snapped to the nearest road node at load time.
LANDMARKS_FALLBACK: dict[str, tuple[float, float]] = {
    "Mohakhali Fuel Depot":   (23.7800, 90.4050),
    "Tejgaon Petrol Pump":    (23.7644, 90.3929),
    "Mirpur 10 Fuel Station": (23.8069, 90.3687),
    "Dhanmondi Pump":         (23.7560, 90.3730),
    "Motijheel Pump":         (23.7330, 90.4172),
    "Gulshan-2 Fuel Pump":    (23.7925, 90.4078),
    "Uttara Fuel Station":    (23.8759, 90.3795),
    "Old Dhaka Pump":         (23.7178, 90.3984),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationPOI:
    """A real fuel station / landmark with snapped OSM node + lat/lon."""

    sid: int
    name: str
    lat: float
    lon: float
    node_id: int


@dataclass(frozen=True)
class VehiclePOI:
    """A vehicle starting location — random OSM node + lat/lon."""

    vid: int
    lat: float
    lon: float
    node_id: int


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def load_dhaka_graph(force_refresh: bool = False) -> nx.MultiDiGraph:
    """Return Dhaka's road graph, using a pickle cache when present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force_refresh:
        logger.info("Loading cached Dhaka graph from %s", CACHE_FILE)
        with CACHE_FILE.open("rb") as f:
            return pickle.load(f)

    # Late import — osmnx pulls in geopandas which is heavy.
    import osmnx as ox

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA_DIR / "osmnx_cache")

    logger.info("Downloading Dhaka drive graph (bbox=%s) — first time, ~30 s",
                DHAKA_BBOX)
    G = ox.graph_from_bbox(bbox=DHAKA_BBOX, network_type="drive", simplify=True)
    with CACHE_FILE.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached graph (%d nodes, %d edges) -> %s",
                G.number_of_nodes(), G.number_of_edges(), CACHE_FILE)
    return G


# ---------------------------------------------------------------------------
# Station / vehicle extraction
# ---------------------------------------------------------------------------


def _snap(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Nearest node id to (lat, lon). OSMnx wants (longitude, latitude)."""
    import osmnx as ox
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))


def _build_full_station_pool(G: nx.MultiDiGraph) -> list[StationPOI]:
    """Snap every OSM fuel POI inside the bbox to a road node.

    This is the *full pool*; callers slice it down to ``max_stations``.
    """
    pois: list[StationPOI] = []
    try:
        import osmnx as ox
        tags = {"amenity": "fuel"}
        gdf = ox.features.features_from_bbox(bbox=DHAKA_BBOX, tags=tags)
        rng = random.Random(7)  # deterministic shuffle so slicing is stable
        rows = list(gdf.iterrows())
        rng.shuffle(rows)
        for i, (_, row) in enumerate(rows):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lon, lat = geom.centroid.x, geom.centroid.y
            if not (DHAKA_BBOX[0] <= lon <= DHAKA_BBOX[2]
                    and DHAKA_BBOX[1] <= lat <= DHAKA_BBOX[3]):
                continue
            raw_name = row.get("name")
            if raw_name is None or (isinstance(raw_name, float) and raw_name != raw_name):
                raw_name = f"Fuel station #{i}"
            try:
                node_id = _snap(G, lat, lon)
            except Exception:  # noqa: BLE001
                continue
            pois.append(StationPOI(
                sid=len(pois), name=str(raw_name),
                lat=lat, lon=lon, node_id=node_id,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("OSM fuel-station query failed (%s); using landmark fallbacks", exc)

    if not pois:
        for name, (lat, lon) in LANDMARKS_FALLBACK.items():
            try:
                node_id = _snap(G, lat, lon)
            except Exception:  # noqa: BLE001
                continue
            pois.append(StationPOI(
                sid=len(pois), name=name,
                lat=lat, lon=lon, node_id=node_id,
            ))
    return pois


def extract_fuel_stations(
    G: nx.MultiDiGraph,
    max_stations: int | None = None,
    force_refresh: bool = False,
) -> list[StationPOI]:
    """Return real OSM fuel stations inside the bbox, snapped to road nodes.

    The **full pool** of stations (all 100+ ``amenity=fuel`` POIs in the
    bbox) is cached to disk. Each call slices the pool to ``max_stations``,
    so the cache stays valid no matter how the UI slider moves.
    """
    if STATIONS_FILE.exists() and not force_refresh:
        with STATIONS_FILE.open("rb") as f:
            pool: list[StationPOI] = pickle.load(f)
    else:
        pool = _build_full_station_pool(G)
        STATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATIONS_FILE.open("wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

    # If a previously-cached pool was smaller than what's now being asked
    # for (older cache only stored 8 picks), rebuild once.
    if max_stations is not None and len(pool) < max_stations:
        pool = _build_full_station_pool(G)
        with STATIONS_FILE.open("wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

    if max_stations is not None:
        pool = pool[:max_stations]

    # Re-index sids so they're 0..len(pool)-1 (matters because Station.sid
    # is used as an array index by the solvers).
    return [
        StationPOI(sid=i, name=p.name, lat=p.lat, lon=p.lon, node_id=p.node_id)
        for i, p in enumerate(pool)
    ]


def sample_vehicle_nodes(
    G: nx.MultiDiGraph,
    n: int,
    seed: int = 42,
    avoid_nodes: Iterable[int] = (),
) -> list[VehiclePOI]:
    """Pick `n` random OSM nodes as vehicle origins."""
    rng = random.Random(seed)
    avoid = set(avoid_nodes)
    candidates = [
        node for node in G.nodes if node not in avoid and "x" in G.nodes[node]
    ]
    rng.shuffle(candidates)
    picks = candidates[:n]
    out: list[VehiclePOI] = []
    for vid, node_id in enumerate(picks):
        lat = float(G.nodes[node_id]["y"])
        lon = float(G.nodes[node_id]["x"])
        out.append(VehiclePOI(vid=vid, lat=lat, lon=lon, node_id=node_id))
    return out


# ---------------------------------------------------------------------------
# Road-distance matrix
# ---------------------------------------------------------------------------


def compute_distance_matrix(
    G: nx.MultiDiGraph,
    vehicles: list[VehiclePOI],
    stations: list[StationPOI],
    cutoff_m: float = 30_000.0,
) -> np.ndarray:
    """Return a `(len(vehicles), len(stations))` matrix of road metres.

    Strategy: run single-source Dijkstra from each station once with a
    distance cutoff (default 30 km — bigger than any vehicle range in this
    project), so Dijkstra prunes early. Unreachable pairs stay `+inf`.
    """
    n_v, n_s = len(vehicles), len(stations)
    mat = np.full((n_v, n_s), np.inf, dtype=float)
    veh_nodes = [v.node_id for v in vehicles]
    veh_index = {nd: i for i, nd in enumerate(veh_nodes)}

    # as_view=True is a zero-copy adapter — huge speedup on big graphs.
    UG = G.to_undirected(as_view=True)
    for s_idx, s in enumerate(stations):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                UG, s.node_id, weight="length", cutoff=cutoff_m,
            )
        except nx.NodeNotFound:
            continue
        for nd, dist in lengths.items():
            if nd in veh_index:
                mat[veh_index[nd], s_idx] = float(dist)
    return mat


def shortest_path_latlon(
    G: nx.MultiDiGraph,
    src_node: int,
    dst_node: int,
) -> list[tuple[float, float]]:
    """Shortest road path between two nodes as [(lat, lon), ...].

    Falls back to a straight line of the two endpoints if no path exists.
    """
    UG = G.to_undirected(as_view=True)
    try:
        path = nx.shortest_path(UG, src_node, dst_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [
            (float(G.nodes[src_node]["y"]), float(G.nodes[src_node]["x"])),
            (float(G.nodes[dst_node]["y"]), float(G.nodes[dst_node]["x"])),
        ]
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in path]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Used as the COP-objective short-circuit
    when we don't need a full graph traversal."""
    R = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))
