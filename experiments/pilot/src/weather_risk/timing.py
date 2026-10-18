"""Explicit as-of feature joins and chronological validation splits."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .schema import Observation
from .validation import _date, _timestamp


def build_feature_panel(
    rows: list[Observation],
    forecast_times: list[str],
    specs: list[dict],
    allow_latest_vintage: bool = False,
) -> list[dict]:
    """Select the newest observation known at each exact forecast cutoff.

    Strict mode needs point-in-time vintages and actual release timestamps.
    Explicitly labeled synthetic records are also accepted for pipeline tests.
    ``allow_latest_vintage`` permits a research approximation, not a claim that
    today's revised historical observations were actually known in the past.

    Staleness is measured in local calendar days from observation date. A null
    newest observation stays null, and is never replaced with an older value.
    """
    spec_map = {spec["series_id"]: spec for spec in specs}
    if len(spec_map) != len(specs):
        raise ValueError("Duplicate series specification")
    grouped: dict[str, list[tuple[Observation, datetime]]] = {key: [] for key in spec_map}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.series_id not in spec_map:
            raise ValueError(f"Unknown series_id: {row.series_id}")
        key = row.series_id, row.observation_date
        if key in seen:
            raise ValueError(f"Duplicate series/date: {key}")
        seen.add(key)
        _date(row.observation_date)
        available = _timestamp(row.available_at)
        if row.value is not None and (
            isinstance(row.value, bool)
            or not isinstance(row.value, (int, float))
            or not math.isfinite(row.value)
        ):
            raise ValueError(f"Invalid numeric value: {key}")
        grouped[row.series_id].append((row, available))
    for records in grouped.values():
        records.sort(key=lambda item: item[0].observation_date)

    panel: list[dict] = []
    for forecast_at in forecast_times:
        cutoff = _timestamp(forecast_at)
        for identifier, spec in spec_map.items():
            zone = ZoneInfo(spec.get("timezone", "UTC"))
            stale_days = spec.get("stale_days", 7)
            if isinstance(stale_days, bool) or not isinstance(stale_days, int) or stale_days < 0:
                raise ValueError(f"{identifier}: stale_days must be a nonnegative integer")
            eligible = [(row, available) for row, available in grouped[identifier] if available <= cutoff]
            feature = {
                "forecast_at": cutoff.isoformat(),
                "series_id": identifier,
                "value": None,
                "observation_date": None,
                "available_at": None,
                "age_days": None,
                "vintage": None,
                "quality_flag": "",
                "missing_reason": "absent",
            }
            if eligible:
                # Choose before checking provenance or nulls. This prevents a
                # rejected newest record silently reviving an older value.
                latest, available = eligible[-1]
                age = (cutoff.astimezone(zone).date() - _date(latest.observation_date)).days
                if age < 0:
                    raise ValueError(f"{identifier}: availability predates observation date")
                feature.update(
                    observation_date=latest.observation_date,
                    available_at=available.isoformat(),
                    age_days=age,
                    vintage=latest.vintage,
                    quality_flag=latest.quality_flag,
                )
                trustworthy = (
                    latest.vintage == "point_in_time" and latest.availability_basis == "release_timestamp"
                ) or (
                    latest.vintage == "synthetic" and latest.availability_basis == "synthetic"
                )
                if not trustworthy and not allow_latest_vintage:
                    feature["missing_reason"] = "strict_vintage"
                elif age > stale_days:
                    feature["missing_reason"] = "stale"
                elif latest.value is None:
                    feature["missing_reason"] = "source_missing"
                else:
                    feature["value"] = latest.value
                    feature["missing_reason"] = ""
            panel.append(feature)
    return panel


def walk_forward_splits(
    n: int,
    min_train: int,
    test_size: int = 1,
    gap: int = 0,
    train_window: int | None = None,
) -> list[dict[str, list[int]]]:
    """Produce chronological expanding or fixed-window train/test indices.

    ``gap`` excludes observations immediately before each test block and can
    purge overlapping forward targets. The caller must choose it from the
    target horizon. A final partial test block is included. No test data are
    used to choose hyperparameters; tuning needs a second walk-forward split
    inside the training data.
    """
    for name, value, lower in (("n", n, 0), ("min_train", min_train, 1), ("test_size", test_size, 1), ("gap", gap, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if train_window is not None and (
        isinstance(train_window, bool) or not isinstance(train_window, int) or train_window < min_train
    ):
        raise ValueError("train_window must be an integer >= min_train")
    splits = []
    for test_start in range(min_train + gap, n, test_size):
        train_end = test_start - gap
        train_start = max(0, train_end - train_window) if train_window is not None else 0
        splits.append({
            "train_indices": list(range(train_start, train_end)),
            "test_indices": list(range(test_start, min(test_start + test_size, n))),
        })
    return splits
