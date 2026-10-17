"""One explicit data contract shared by all sources."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Observation:
    series_id: str
    observation_date: str
    value: float | None
    unit: str
    source: str
    available_at: str
    retrieved_at: str
    availability_basis: str = "assumed_lag"
    vintage: str = "latest"
    quality_flag: str = ""


def assumed_availability(day: str, timezone_name: str, lag_days: int, hour: int) -> str:
    """Calendar-day delay in source local time; this is NOT verified release time."""
    if lag_days < 0 or not 0 <= hour <= 23:
        raise ValueError("lag_days must be nonnegative and hour must be 0..23")
    local_day = date.fromisoformat(day) + timedelta(days=lag_days)
    local = datetime.combine(local_day, time(hour), ZoneInfo(timezone_name))
    return local.astimezone(timezone.utc).isoformat()
