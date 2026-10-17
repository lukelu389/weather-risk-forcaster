"""Deterministic synthetic integration data, never empirical investment evidence."""

from datetime import date, timedelta
import math
import random

from .schema import Observation, assumed_availability


def synthetic_observations(config: dict) -> list[Observation]:
    rows = []
    first, last = date.fromisoformat(config["start"]), date.fromisoformat(config["end"])
    retrieved_at = (last + timedelta(days=120)).isoformat() + "T23:00:00+00:00"
    for spec in config["series"]:
        rng = random.Random(f"{config['seed']}:{spec['series_id']}")
        level = 50.0
        day = first
        while day <= last:
            frequency = spec["frequency"]
            include = (frequency == "daily" or
                       frequency == "business_daily" and day.weekday() < 5 or
                       frequency == "weekly" and day.weekday() == 4 or
                       frequency == "monthly" and (day + timedelta(days=1)).month != day.month)
            if include:
                field = spec.get("source_field", "")
                temp = 8 + 18 * math.sin(2 * math.pi * (day.timetuple().tm_yday - 100) / 365.25)
                weather_rng = random.Random(f"{config['seed']}:{spec['source_series']}:{day}")
                temp += weather_rng.gauss(0, 3)
                if field == "Mean Temp (°C)":
                    value = temp
                elif field == "HDD18":
                    value = max(18 - temp, 0)
                elif field == "CDD18":
                    value = max(temp - 18, 0)
                elif field == "Total Precip (mm)":
                    value = max(rng.gauss(1, 3), 0)
                elif spec["unit"] == "CAD/USD":
                    value = 1.3 + rng.gauss(0, .01)
                elif spec["unit"] == "percent":
                    value = 3.5 + rng.gauss(0, .05)
                elif spec["unit"] == "index_1982_84_100":
                    value = 300 + (day - first).days / 365
                elif spec["unit"] == "Bcf":
                    value = 2500 + 600 * math.sin(day.timetuple().tm_yday / 365 * 2 * math.pi)
                elif spec["unit"] == "MMcf":
                    value = 2500000 + rng.gauss(0, 10000)
                else:
                    level *= math.exp(rng.gauss(0, .015))
                    value = level
                rows.append(Observation(
                    spec["series_id"], day.isoformat(), round(value, 8), spec["unit"],
                    "synthetic", assumed_availability(day.isoformat(), spec["timezone"],
                    spec["lag_days"], spec["available_hour"]), retrieved_at,
                    "synthetic", "synthetic", "SYNTHETIC_NOT_MARKET_DATA"))
            day += timedelta(days=1)
    return rows
