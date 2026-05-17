"""Traveler context and time-of-day utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace

from dhaka_pathfinder.config import (
    AGE_PROFILE,
    AGE_VEHICLE_RESTRICTION,
    GENDER_SAFETY_MULTIPLIER,
    TIME_BUCKET_HOURS,
    TIME_OF_DAY_MULTIPLIERS,
    VEHICLE_SPEED_KMPH,
    WEATHER_PROFILE,
)


VALID_GENDERS = frozenset(GENDER_SAFETY_MULTIPLIER.keys())
VALID_VEHICLES = frozenset(VEHICLE_SPEED_KMPH.keys())
VALID_TIME_BUCKETS = frozenset(TIME_BUCKET_HOURS.keys())
VALID_SOCIAL = frozenset({"alone", "accompanied"})
VALID_AGES = frozenset(AGE_PROFILE.keys())
VALID_WEATHER = frozenset(WEATHER_PROFILE.keys())


def hour_to_bucket(hour: int) -> str:
    """Map a 0-23 hour to a named time bucket."""
    hour = hour % 24
    for bucket, (start, end) in TIME_BUCKET_HOURS.items():
        if start <= hour < end:
            return bucket
        if end > 24 and (hour >= start or hour < (end - 24)):
            return bucket
    return "midday"


@dataclass(frozen=True)
class TravelContext:
    """Traveler-side state feeding the cost model and heuristics.

    All fields have sensible defaults so the tests and simpler flows can omit them.
    """

    gender: str = "male"
    social: str = "alone"
    age: str = "adult"
    vehicle: str = "car"
    time_bucket: str = "midday"
    weather: str = "clear"
    weight_preset: str = "balanced"

    def __post_init__(self) -> None:
        if self.gender not in VALID_GENDERS:
            raise ValueError(f"gender must be one of {sorted(VALID_GENDERS)}; got {self.gender!r}")
        if self.social not in VALID_SOCIAL:
            raise ValueError(f"social must be one of {sorted(VALID_SOCIAL)}; got {self.social!r}")
        if self.age not in VALID_AGES:
            raise ValueError(f"age must be one of {sorted(VALID_AGES)}; got {self.age!r}")
        if self.vehicle not in VALID_VEHICLES:
            raise ValueError(f"vehicle must be one of {sorted(VALID_VEHICLES)}; got {self.vehicle!r}")
        if self.time_bucket not in VALID_TIME_BUCKETS:
            raise ValueError(f"time_bucket must be one of {sorted(VALID_TIME_BUCKETS)}; got {self.time_bucket!r}")
        if self.weather not in VALID_WEATHER:
            raise ValueError(f"weather must be one of {sorted(VALID_WEATHER)}; got {self.weather!r}")

    @property
    def time_multipliers(self) -> dict[str, float]:
        return TIME_OF_DAY_MULTIPLIERS[self.time_bucket]

    @property
    def weather_multipliers(self) -> dict[str, float]:
        return WEATHER_PROFILE[self.weather]

    @property
    def age_profile(self) -> dict[str, float]:
        return AGE_PROFILE[self.age]

    @property
    def gender_multiplier(self) -> float:
        return GENDER_SAFETY_MULTIPLIER[self.gender][self.social]

    @property
    def vehicle_speed(self) -> float:
        return VEHICLE_SPEED_KMPH[self.vehicle]

    @property
    def vehicle_is_allowed(self) -> bool:
        """False if the active age forbids this vehicle (e.g. child on a motorbike)."""
        return self.vehicle not in AGE_VEHICLE_RESTRICTION.get(self.age, ())

    def with_time_bucket(self, bucket: str) -> "TravelContext":
        return replace(self, time_bucket=bucket)

    def with_vehicle(self, vehicle: str) -> "TravelContext":
        return replace(self, vehicle=vehicle)

    def with_gender(self, gender: str, social: str | None = None) -> "TravelContext":
        return replace(self, gender=gender, social=social or self.social)

    def with_weather(self, weather: str) -> "TravelContext":
        return replace(self, weather=weather)

    def with_age(self, age: str) -> "TravelContext":
        return replace(self, age=age)

    def label(self) -> str:
        return (
            f"{self.gender}-{self.social}-{self.age}|{self.vehicle}"
            f"|{self.time_bucket}|{self.weather}"
        )

    @classmethod
    def default(cls) -> "TravelContext":
        return cls()


CONTEXT_GRID: tuple[TravelContext, ...] = tuple(
    TravelContext(
        gender=g, social=s, age=a, vehicle=v, time_bucket=t, weather=w,
    )
    for g, s in (("male", "alone"), ("male", "accompanied"),
                 ("female", "alone"), ("female", "accompanied"))
    for a in ("adult", "child", "elderly")
    for v in ("walk", "rickshaw", "cng", "car", "motorbike")
    for t in ("morning_rush", "midday", "evening_rush", "late_night")
    for w in ("clear", "rain")
)
