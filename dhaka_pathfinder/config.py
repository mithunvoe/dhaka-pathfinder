"""Central configuration — paths, constants, and tunable defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MAPS_DIR: Path = RESULTS_DIR / "maps"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

for _p in (DATA_DIR, RESULTS_DIR, MAPS_DIR, PLOTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


DHAKA_CENTER = (23.7800, 90.4000)
DHAKA_BBOX = (90.34, 23.70, 90.46, 23.86)
DHAKA_BBOX_FULL = (90.33, 23.68, 90.48, 23.88)

DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_RADIUS_KM = 6.0
SYNTHETIC_SEED = 42

LANDMARKS: dict[str, tuple[float, float]] = {
    "Shahbag": (23.7385, 90.3958),
    "Motijheel": (23.7330, 90.4172),
    "Dhanmondi 27": (23.7560, 90.3730),
    "Gulshan 1": (23.7806, 90.4145),
    "Gulshan 2": (23.7925, 90.4078),
    "Banani": (23.7937, 90.4066),
    "Uttara Sector 7": (23.8759, 90.3795),
    "Mirpur 10": (23.8069, 90.3687),
    "Mirpur 1": (23.7956, 90.3537),
    "Farmgate": (23.7580, 90.3896),
    "New Market": (23.7333, 90.3847),
    "Sadarghat": (23.7080, 90.4125),
    "Old Dhaka (Chawkbazar)": (23.7178, 90.3984),
    "Mohammadpur": (23.7589, 90.3582),
    "Jatrabari": (23.7104, 90.4349),
    "Airport (HSIA)": (23.8433, 90.3978),
    "Tejgaon": (23.7644, 90.3929),
    "Bashundhara R/A": (23.8197, 90.4250),
    "Kakrail": (23.7388, 90.4114),
    "Badda": (23.7808, 90.4264),
    "Khilgaon": (23.7478, 90.4267),
    "Paltan": (23.7335, 90.4158),
    "Ramna": (23.7378, 90.4050),
    "Wari": (23.7181, 90.4202),
    "Science Lab": (23.7396, 90.3847),
    "Karwan Bazar": (23.7515, 90.3933),
    "Elephant Road": (23.7380, 90.3878),
    "Bijoy Sarani": (23.7604, 90.3875),
    "Agargaon": (23.7780, 90.3795),
    "Shyamoli": (23.7731, 90.3664),
}


@dataclass(frozen=True)
class CostWeights:
    """Weights for the multi-factor cost model. Must be non-negative."""

    length: float = 1.0
    road_condition: float = 0.8
    safety: float = 1.2
    risk: float = 1.5
    traffic: float = 1.3
    time_of_day: float = 1.0
    lighting: float = 0.6
    water_logging: float = 0.4
    gender_safety: float = 1.4
    social_context: float = 0.9
    vehicle_suitability: float = 1.0
    crime: float = 1.0
    age: float = 1.1
    weather: float = 1.0
    street_width: float = 0.6

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_WEIGHTS = CostWeights()


AGE_PROFILE: dict[str, dict[str, float]] = {
    "adult":    {"risk_amp": 1.00, "traffic_amp": 1.00, "crime_amp": 1.00, "wide_road_penalty": 1.00},
    "child":    {"risk_amp": 1.80, "traffic_amp": 1.50, "crime_amp": 1.60, "wide_road_penalty": 1.35},
    "elderly":  {"risk_amp": 1.35, "traffic_amp": 1.25, "crime_amp": 1.20, "wide_road_penalty": 1.20},
}

AGE_VEHICLE_RESTRICTION: dict[str, tuple[str, ...]] = {
    "adult": (),
    "child": ("motorbike",),
    "elderly": ("motorbike",),
}


WEATHER_PROFILE: dict[str, dict[str, float]] = {
    "clear": {"water_log_amp": 1.00, "lighting_amp": 1.00, "risk_amp": 1.00, "traffic_amp": 1.00, "condition_amp": 1.00},
    "rain":  {"water_log_amp": 2.20, "lighting_amp": 1.30, "risk_amp": 1.40, "traffic_amp": 1.30, "condition_amp": 1.20},
    "fog":   {"water_log_amp": 1.00, "lighting_amp": 2.50, "risk_amp": 1.55, "traffic_amp": 1.20, "condition_amp": 1.10},
    "storm": {"water_log_amp": 3.00, "lighting_amp": 1.90, "risk_amp": 1.85, "traffic_amp": 1.50, "condition_amp": 1.35},
    "heat":  {"water_log_amp": 1.00, "lighting_amp": 1.00, "risk_amp": 1.10, "traffic_amp": 1.05, "condition_amp": 1.10},
}


LANES_DEFAULT: int = 2
LANES_MIN: int = 1
LANES_MAX: int = 10


VEHICLE_SPEED_KMPH: dict[str, float] = {
    "walk": 5.0,
    "rickshaw": 12.0,
    "cng": 22.0,
    "motorbike": 30.0,
    "car": 28.0,
    "bus": 18.0,
}


VEHICLE_HIGHWAY_SUITABILITY: dict[str, dict[str, float]] = {
    "walk": {
        "motorway": 0.15, "trunk": 0.25, "primary": 0.45, "secondary": 0.65,
        "tertiary": 0.8, "residential": 1.0, "service": 1.0, "footway": 1.0,
        "pedestrian": 1.0, "path": 0.9, "living_street": 1.0, "unclassified": 0.75,
    },
    "rickshaw": {
        "motorway": 0.1, "trunk": 0.2, "primary": 0.55, "secondary": 0.8,
        "tertiary": 0.95, "residential": 1.0, "service": 0.9, "footway": 0.0,
        "pedestrian": 0.2, "living_street": 1.0, "unclassified": 0.85,
    },
    "cng": {
        "motorway": 0.7, "trunk": 0.85, "primary": 1.0, "secondary": 1.0,
        "tertiary": 0.95, "residential": 0.85, "service": 0.7, "footway": 0.0,
        "pedestrian": 0.0, "living_street": 0.5, "unclassified": 0.8,
    },
    "motorbike": {
        "motorway": 0.9, "trunk": 0.95, "primary": 1.0, "secondary": 1.0,
        "tertiary": 0.95, "residential": 0.9, "service": 0.8, "footway": 0.0,
        "pedestrian": 0.0, "living_street": 0.7, "unclassified": 0.85,
    },
    "car": {
        "motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 0.95,
        "tertiary": 0.85, "residential": 0.7, "service": 0.55, "footway": 0.0,
        "pedestrian": 0.0, "living_street": 0.4, "unclassified": 0.75,
    },
    "bus": {
        "motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 0.8,
        "tertiary": 0.5, "residential": 0.2, "service": 0.15, "footway": 0.0,
        "pedestrian": 0.0, "living_street": 0.1, "unclassified": 0.5,
    },
}


TIME_OF_DAY_MULTIPLIERS: dict[str, dict[str, float]] = {
    "early_morning": {"traffic": 0.35, "risk": 0.9, "safety": 0.75, "lighting": 0.6},
    "morning_rush": {"traffic": 1.8, "risk": 1.2, "safety": 1.0, "lighting": 1.0},
    "midday": {"traffic": 1.0, "risk": 0.9, "safety": 1.0, "lighting": 1.0},
    "afternoon": {"traffic": 1.1, "risk": 1.0, "safety": 1.0, "lighting": 1.0},
    "evening_rush": {"traffic": 2.0, "risk": 1.3, "safety": 0.95, "lighting": 0.9},
    "evening": {"traffic": 1.2, "risk": 1.0, "safety": 0.85, "lighting": 0.7},
    "late_night": {"traffic": 0.3, "risk": 1.4, "safety": 0.55, "lighting": 0.4},
}

TIME_BUCKET_HOURS: dict[str, tuple[int, int]] = {
    "early_morning": (4, 7),
    "morning_rush": (7, 10),
    "midday": (10, 13),
    "afternoon": (13, 16),
    "evening_rush": (16, 20),
    "evening": (20, 23),
    "late_night": (23, 28),
}


GENDER_SAFETY_MULTIPLIER: dict[str, dict[str, float]] = {
    "male": {"alone": 1.0, "accompanied": 0.85},
    "female": {"alone": 1.6, "accompanied": 1.15},
    "nonbinary": {"alone": 1.35, "accompanied": 1.05},
}


AREA_SAFETY_PROFILE: dict[str, float] = {
    "gulshan": 0.9,
    "banani": 0.9,
    "dhanmondi": 0.85,
    "uttara": 0.95,
    "bashundhara": 0.9,
    "old_dhaka": 1.25,
    "jatrabari": 1.2,
    "mirpur": 1.05,
    "mohammadpur": 1.1,
    "tejgaon": 1.0,
    "motijheel": 1.05,
    "default": 1.0,
}


ALGORITHM_REGISTRY = (
    "bfs", "dfs", "ucs",
    "greedy", "astar", "weighted_astar",
)


HEURISTIC_REGISTRY = (
    "haversine_admissible",
    "network_relaxed",
    "haversine_time",
    "context_aware",
    "learned_history",
    "zero",
)
