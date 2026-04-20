"""Generate synthetic-but-plausible attributes for every node and edge in a graph.

The generator is deterministic (seeded NumPy), vectorised, and scale-free: running it on
a million-edge graph uses the same code path as on a ten-edge test graph. The attributes
are *derived* from the node/edge geometry and real OSM tags, so they are internally
consistent (Old Dhaka is uniformly rougher than Gulshan, even if the OSM tags agree).

Attributes written per edge:
  - condition:          [0, 1]  surface quality (higher = better)
  - traffic_base:       [0, 1]  baseline congestion (higher = more congested)
  - risk:               [0, 1]  accident / hazard probability
  - safety:             [0, 1]  perceived safety (higher = safer)
  - lighting:           [0, 1]  street lighting quality
  - water_logging_prob: [0, 1]  probability of water-logging during rain
  - crime_index:        [0, 1]  area crime level (higher = worse)
  - free_flow_speed:    float   km/h best-case driving speed
  - highway_class:      str     simplified OSM highway tag
  - area_name:          str     fuzzy neighbourhood label
  - historical_incidents: int   count of synthesised past incidents on this segment

Attributes written per node:
  - area_name, area_safety_profile, cluster_id
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from dhaka_pathfinder.config import (
    AREA_SAFETY_PROFILE,
    LANES_DEFAULT,
    LANES_MAX,
    LANES_MIN,
    SYNTHETIC_SEED,
)

logger = logging.getLogger(__name__)


_HIGHWAY_SPEED_KMPH = {
    "motorway": 60, "trunk": 50, "primary": 45, "secondary": 40,
    "tertiary": 35, "residential": 25, "service": 20, "living_street": 20,
    "unclassified": 25, "footway": 5, "pedestrian": 5, "path": 5,
}

_HIGHWAY_CONDITION = {
    "motorway": 0.9, "trunk": 0.85, "primary": 0.8, "secondary": 0.72,
    "tertiary": 0.65, "residential": 0.55, "service": 0.5, "living_street": 0.5,
    "unclassified": 0.55, "footway": 0.45, "pedestrian": 0.55, "path": 0.4,
}

_HIGHWAY_BASE_TRAFFIC = {
    "motorway": 0.55, "trunk": 0.6, "primary": 0.7, "secondary": 0.6,
    "tertiary": 0.5, "residential": 0.35, "service": 0.25, "living_street": 0.2,
    "unclassified": 0.45, "footway": 0.15, "pedestrian": 0.3, "path": 0.1,
}

_HIGHWAY_RISK = {
    "motorway": 0.3, "trunk": 0.35, "primary": 0.4, "secondary": 0.35,
    "tertiary": 0.3, "residential": 0.25, "service": 0.2, "living_street": 0.2,
    "unclassified": 0.35, "footway": 0.2, "pedestrian": 0.2, "path": 0.3,
}

_HIGHWAY_DEFAULT_LANES = {
    "motorway": 4, "trunk": 4, "primary": 3, "secondary": 2,
    "tertiary": 2, "residential": 2, "service": 1, "living_street": 1,
    "unclassified": 2, "footway": 1, "pedestrian": 1, "path": 1,
}


def _parse_lanes(raw: object, highway: str) -> int:
    """Parse OSM's messy lanes tag into a sane integer in [LANES_MIN, LANES_MAX]."""
    fallback = _HIGHWAY_DEFAULT_LANES.get(highway, LANES_DEFAULT)
    if raw is None:
        return fallback
    if isinstance(raw, list):
        raw = raw[0] if raw else None
        if raw is None:
            return fallback
    try:
        v = int(float(str(raw).split(";")[0].strip()))
    except (TypeError, ValueError):
        return fallback
    return max(LANES_MIN, min(v, LANES_MAX))


def _simplify_highway(tag: str | list) -> str:
    """Return a single canonical highway class from OSM's messy data."""
    if isinstance(tag, list):
        tag = tag[0] if tag else "unclassified"
    if not isinstance(tag, str):
        return "unclassified"
    if "_link" in tag:
        tag = tag.replace("_link", "")
    return tag if tag in _HIGHWAY_SPEED_KMPH else "unclassified"


@dataclass(frozen=True)
class SyntheticConfig:
    """Tunables for the generator. Keep seeded so results are reproducible."""

    seed: int = SYNTHETIC_SEED
    num_area_clusters: int = 12
    noise_sigma: float = 0.08
    old_dhaka_bias_lat: float = 23.72
    old_dhaka_bias_lon: float = 90.40
    area_gradient_strength: float = 0.35


def _bucket_area(lat: float, lon: float) -> str:
    """Map a point to a named Dhaka neighbourhood (coarse — purely for flavour)."""
    if lat < 23.725:
        return "old_dhaka" if lon > 90.395 else "mohammadpur"
    if lat < 23.75 and lon < 90.38:
        return "dhanmondi"
    if lat < 23.75 and lon > 90.41:
        return "motijheel"
    if lat < 23.78 and lon > 90.395 and lon < 90.42:
        return "tejgaon"
    if lat < 23.78 and lon < 90.37:
        return "mohammadpur"
    if lat < 23.80 and lon > 90.42:
        return "bashundhara"
    if lat < 23.82 and lon > 90.39 and lon < 90.42:
        return "banani"
    if lat >= 23.78 and lat < 23.82 and lon >= 90.39 and lon < 90.43:
        return "gulshan"
    if lat >= 23.82 and lon > 90.37:
        return "uttara"
    if lat < 23.78 and lon < 90.38:
        return "mirpur"
    return "default"


def _cluster_nodes(G: nx.MultiDiGraph, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means on node coordinates, producing `cluster_id` per node (used as area proxy)."""
    coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes()])
    k = min(k, max(2, coords.shape[0] // 50))
    km = MiniBatchKMeans(n_clusters=k, random_state=int(rng.integers(0, 10**6)), n_init=10)
    return km.fit_predict(coords)


def augment_graph(G: nx.MultiDiGraph, config: SyntheticConfig | None = None) -> nx.MultiDiGraph:
    """Mutate `G` in place with synthetic context attributes and also return it."""
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)

    nodes = list(G.nodes())
    cluster_ids = _cluster_nodes(G, config.num_area_clusters, rng)
    cluster_profile = rng.uniform(0.75, 1.2, size=cluster_ids.max() + 1)

    for idx, n in enumerate(nodes):
        ndata = G.nodes[n]
        lat, lon = ndata["y"], ndata["x"]
        area = _bucket_area(lat, lon)
        ndata["area_name"] = area
        ndata["area_safety_profile"] = AREA_SAFETY_PROFILE.get(area, AREA_SAFETY_PROFILE["default"])
        ndata["cluster_id"] = int(cluster_ids[idx])
        ndata["cluster_factor"] = float(cluster_profile[cluster_ids[idx]])

    logger.info("Assigned area/cluster to %d nodes", len(nodes))

    num_edges = G.number_of_edges()
    noise = rng.normal(0.0, config.noise_sigma, size=num_edges)
    ptr = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        hw = _simplify_highway(data.get("highway", "unclassified"))
        data["highway_class"] = hw

        lat_u, lon_u = G.nodes[u]["y"], G.nodes[u]["x"]
        lat_v, lon_v = G.nodes[v]["y"], G.nodes[v]["x"]
        mid_lat, mid_lon = (lat_u + lat_v) / 2, (lon_u + lon_v) / 2
        area = _bucket_area(mid_lat, mid_lon)
        data["area_name"] = area

        area_factor = AREA_SAFETY_PROFILE.get(area, 1.0)
        cluster_u = G.nodes[u].get("cluster_factor", 1.0)
        cluster_v = G.nodes[v].get("cluster_factor", 1.0)
        cluster_factor = (cluster_u + cluster_v) / 2

        dist_to_old = float(np.hypot(mid_lat - config.old_dhaka_bias_lat, mid_lon - config.old_dhaka_bias_lon))
        old_dhaka_factor = float(np.exp(-dist_to_old * 40))

        base_condition = _HIGHWAY_CONDITION.get(hw, 0.55)
        condition = base_condition - 0.25 * old_dhaka_factor - 0.08 * (cluster_factor - 1) + noise[ptr]
        condition = float(np.clip(condition, 0.1, 1.0))

        base_traffic = _HIGHWAY_BASE_TRAFFIC.get(hw, 0.45)
        traffic = base_traffic + 0.12 * (cluster_factor - 1) + 0.15 * old_dhaka_factor + noise[ptr] * 0.5
        traffic = float(np.clip(traffic, 0.05, 0.98))

        base_risk = _HIGHWAY_RISK.get(hw, 0.3)
        risk = base_risk * area_factor + 0.1 * old_dhaka_factor + noise[ptr] * 0.4
        risk = float(np.clip(risk, 0.02, 0.95))

        safety = (1.0 - risk * 0.6) / area_factor - 0.1 * (1 - condition) + noise[ptr] * 0.2
        safety = float(np.clip(safety, 0.05, 1.0))

        lit_raw = data.get("lit")
        if isinstance(lit_raw, str) and lit_raw.lower() in ("yes", "24/7", "automatic"):
            lighting_base = 0.9
        elif isinstance(lit_raw, str) and lit_raw.lower() in ("no", "disused"):
            lighting_base = 0.2
        else:
            lighting_base = 0.6 if hw in ("primary", "secondary", "trunk", "motorway") else 0.4
        lighting = float(np.clip(lighting_base * (1.1 - 0.3 * old_dhaka_factor) + noise[ptr] * 0.3, 0.05, 1.0))

        water_log = float(np.clip(0.25 + 0.4 * old_dhaka_factor + 0.15 * (cluster_factor - 1) + noise[ptr] * 0.3, 0.02, 0.95))

        crime = float(np.clip(0.3 * area_factor + 0.15 * old_dhaka_factor + noise[ptr] * 0.3, 0.02, 0.95))

        free_flow_speed = _HIGHWAY_SPEED_KMPH.get(hw, 25)
        try:
            maxspeed = data.get("maxspeed")
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if maxspeed:
                free_flow_speed = float(str(maxspeed).split()[0])
        except (ValueError, IndexError):
            pass

        num_lanes = _parse_lanes(data.get("lanes"), hw)
        street_width_m = float(num_lanes) * 3.5

        data["condition"] = condition
        data["traffic_base"] = traffic
        data["risk"] = risk
        data["safety"] = safety
        data["lighting"] = lighting
        data["water_logging_prob"] = water_log
        data["crime_index"] = crime
        data["free_flow_speed"] = float(free_flow_speed)
        data["num_lanes"] = int(num_lanes)
        data["street_width_m"] = street_width_m

        historical_incidents = int(np.clip(rng.poisson(risk * 4), 0, 30))
        data["historical_incidents"] = historical_incidents
        ptr += 1

    logger.info("Augmented %d edges with synthetic attributes", num_edges)
    return G


def summarize_synthetic(G: nx.MultiDiGraph) -> dict[str, float]:
    """Mean / percentile stats over the synthetic layer — used in tests and reports."""
    attrs = [
        "condition", "traffic_base", "risk", "safety",
        "lighting", "water_logging_prob", "crime_index",
        "num_lanes", "street_width_m",
    ]
    out: dict[str, float] = {}
    for a in attrs:
        vals = np.array([d.get(a, np.nan) for _, _, d in G.edges(data=True)])
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        out[f"{a}_mean"] = float(vals.mean())
        out[f"{a}_p10"] = float(np.percentile(vals, 10))
        out[f"{a}_p90"] = float(np.percentile(vals, 90))
    return out
