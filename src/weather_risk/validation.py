"""Deterministic data checks; never impute, clip, or remove observations."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .schema import Observation


def _date(value: str) -> date:
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("dates must use YYYY-MM-DD")
    return result


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return result.astimezone(timezone.utc)


def _period(day: date, frequency: str) -> str:
    if frequency == "monthly":
        return day.strftime("%Y-%m")
    if frequency == "weekly":
        year, week, _ = day.isocalendar()
        return f"{year}-W{week:02d}"
    return day.isoformat()


def validate_observations(
    rows: list[Observation], specs: list[dict], start: str, end: str
) -> dict:
    """Check a requested inclusive date range and return a JSON-safe report.

    Daily series have a full calendar coverage requirement. For other frequencies,
    absent weekdays/weeks/months are warnings: exchange holidays and release
    calendars are not inferred here. Their explicit null values still obey the
    missing-value threshold. Price jumps are warnings, never corrected data.
    """
    report: dict = {"errors": [], "warnings": [], "series": {}, "passed": False}
    errors, warnings = report["errors"], report["warnings"]
    try:
        first, last = _date(start), _date(end)
        if first > last:
            raise ValueError("start is after end")
    except (TypeError, ValueError) as exc:
        errors.append(f"Invalid requested date range: {exc}")
        return report

    spec_map: dict[str, dict] = {}
    for spec in specs:
        identifier = spec.get("series_id")
        if not isinstance(identifier, str) or not identifier:
            errors.append("Every series specification needs a nonempty series_id")
        elif identifier in spec_map:
            errors.append(f"Duplicate series specification: {identifier}")
        else:
            spec_map[identifier] = spec

    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        if row.series_id not in spec_map:
            errors.append(f"Unknown series_id: {row.series_id}")
        else:
            grouped[row.series_id].append(row)

    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    for identifier, spec in spec_map.items():
        frequency = spec.get("frequency", "daily")
        if frequency not in {"daily", "business_daily", "weekly", "monthly"}:
            errors.append(f"{identifier}: unsupported frequency {frequency!r}")
            continue
        try:
            zone = ZoneInfo(spec.get("timezone", "UTC"))
            maximum_missing = float(spec.get("max_missing_fraction", 0.1))
            if not 0 <= maximum_missing <= 1:
                raise ValueError("max_missing_fraction must be in [0, 1]")
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            errors.append(f"{identifier}: invalid configuration: {exc}")
            continue

        expected = {
            _period(day, frequency)
            for day in days
            if frequency != "business_daily" or day.weekday() < 5
        }
        entries = grouped[identifier]
        seen: set[date] = set()
        observed: set[str] = set()
        null_dates: set[date] = set()
        valid_dates: list[date] = []
        previous_day: date | None = None
        previous_value: float | None = None
        null_count, nonfinite_count, outlier_count = 0, 0, 0
        for row in entries:
            label = f"{identifier} {row.observation_date}"
            try:
                day = _date(row.observation_date)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}: invalid observation date: {exc}")
                continue
            if day in seen:
                errors.append(f"{label}: duplicate series/date")
            seen.add(day)
            if previous_day is not None and day < previous_day:
                errors.append(f"{label}: observations are unsorted")
            previous_day = day
            if not first <= day <= last:
                errors.append(f"{label}: outside requested date range")
            else:
                observed.add(_period(day, frequency))
                valid_dates.append(day)
            if row.unit != spec.get("unit"):
                errors.append(f"{label}: unit {row.unit!r} does not match {spec.get('unit')!r}")
            local_midnight = datetime.combine(day, time(), tzinfo=zone)
            for field in ("available_at", "retrieved_at"):
                try:
                    timestamp = _timestamp(getattr(row, field))
                    if timestamp < local_midnight:
                        errors.append(f"{label}: {field} predates observation's local midnight")
                except (TypeError, ValueError, AttributeError) as exc:
                    errors.append(f"{label}: invalid {field}: {exc}")
            # A conservative availability assumption can fall after retrieval.
            # Such recent records are withheld by the as-of join, not rejected.
            if row.value is None:
                null_count += 1
                if first <= day <= last:
                    null_dates.add(day)
                previous_value = None
                continue
            if isinstance(row.value, bool) or not isinstance(row.value, (int, float)):
                errors.append(f"{label}: value must be numeric or null")
                previous_value = None
                continue
            if not math.isfinite(row.value):
                nonfinite_count += 1
                errors.append(f"{label}: value is not finite")
                previous_value = None
                continue
            if spec.get("positive", False) and row.value <= 0:
                errors.append(f"{label}: value must be strictly positive")
            if spec.get("minimum") is not None and row.value < spec["minimum"]:
                errors.append(f"{label}: value below hard minimum {spec['minimum']}")
            if spec.get("maximum") is not None and row.value > spec["maximum"]:
                errors.append(f"{label}: value above hard maximum {spec['maximum']}")
            threshold = spec.get("jump_warning")
            if threshold is not None and previous_value is not None:
                if abs(row.value - previous_value) > threshold:
                    outlier_count += 1
                    warnings.append(f"{label}: absolute jump exceeds {threshold}; retained unchanged")
            previous_value = row.value

        missing_periods = sorted(expected - observed)
        # Multiple daily records cannot masquerade as complete monthly coverage.
        coverage_missing_fraction = len(missing_periods) / len(expected) if expected else 0.0
        if frequency == "daily":
            missing_fraction = (len(missing_periods) + len(null_dates)) / len(expected) if expected else 0.0
        else:
            missing_fraction = null_count / len(entries) if entries else 0.0
        report["series"][identifier] = {
            "frequency": frequency,
            "row_count": len(entries),
            "null_count": null_count,
            "nonfinite_count": nonfinite_count,
            "outlier_count": outlier_count,
            "expected_period_count": len(expected),
            "observed_period_count": len(observed & expected),
            "missing_period_count": len(missing_periods),
            "missing_periods_sample": missing_periods[:20],
            "coverage_missing_fraction": coverage_missing_fraction,
            "missing_fraction": missing_fraction,
            "max_missing_fraction": maximum_missing,
            "first_date": min(valid_dates).isoformat() if valid_dates else None,
            "last_date": max(valid_dates).isoformat() if valid_dates else None,
        }
        required = spec.get("required", True)
        if not entries and required:
            errors.append(f"{identifier}: required series has no observations")
        if missing_periods:
            qualifier = "; weekday calendar does not account for market holidays" if frequency == "business_daily" else ""
            warnings.append(f"{identifier}: {len(missing_periods)} missing {frequency} periods{qualifier}")
        if null_count:
            warnings.append(f"{identifier}: {null_count} explicit missing values; retained as null")
        if missing_fraction > maximum_missing:
            message = f"{identifier}: missing fraction {missing_fraction:.3f} exceeds {maximum_missing:.3f}"
            (errors if required else warnings).append(message)

    report["passed"] = not errors
    return report
