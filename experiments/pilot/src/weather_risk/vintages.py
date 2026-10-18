"""Versioned observations without silently upgrading date-only release evidence.

The existing single-snapshot contract remains unchanged. This archive stores one
record per revision, including explicit missing-value revisions. ``effective_at``
requires a documented actual release time; provider vintage dates belong in
``vintage_date`` instead. ``release_evidence`` identifies the document or cached
source supporting the time, but this module cannot verify that evidence itself.

Date-only revisions have an uncertainty interval covering their local calendar
day. Strict selection masks them. An explicit research override permits them
only after the entire day has elapsed, retaining the ``vintage_date`` label.
During that day even an older, precisely timed revision is masked: it might have
already been superseded. Dates refer to observed periods, not weather forecasts.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class VersionedObservation:
    series_id: str
    observation_date: str
    value: float | None
    unit: str
    source: str
    retrieved_at: str
    source_timezone: str
    availability_basis: str
    effective_at: str | None = None
    vintage_date: str | None = None
    valid_through_date: str | None = None
    release_evidence: str = ""
    quality_flag: str = ""


_FIELD_NAMES = tuple(field.name for field in fields(VersionedObservation))
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


def _day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("Dates must be YYYY-MM-DD strings")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def _instant(value: str) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise ValueError("Timestamps must include seconds and an explicit UTC offset")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _bounds(row: VersionedObservation) -> tuple[datetime, datetime]:
    if row.availability_basis == "release_timestamp":
        instant = _instant(row.effective_at)
        return instant, instant
    day = _day(row.vintage_date)
    zone = ZoneInfo(row.source_timezone)
    return (
        datetime.combine(day, time(), zone).astimezone(timezone.utc),
        datetime.combine(day + timedelta(days=1), time(), zone).astimezone(timezone.utc),
    )


def validate_archive(rows: list[VersionedObservation]) -> list[VersionedObservation]:
    """Return UTC-normalized, sorted records; reject ambiguous/conflicting keys.

    Unit, provider, and observation-calendar timezone are fixed within a series.
    A date-only revision cannot coexist with another revision of the same
    observation during its uncertain day; obtain better evidence to resolve it.
    Identical duplicate versions are rejected too, rather than silently deduped.
    """
    normalized = []
    series_metadata = {}
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, VersionedObservation):
            raise ValueError("Archive rows must be VersionedObservation records")
        for name in ("series_id", "unit", "source", "source_timezone"):
            value = getattr(row, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("release_evidence", "quality_flag"):
            if not isinstance(getattr(row, name), str):
                raise ValueError(f"{name} must be a string")
        day = _day(row.observation_date)
        retrieved = _instant(row.retrieved_at)
        try:
            zone = ZoneInfo(row.source_timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"Invalid source_timezone: {row.source_timezone}") from exc
        if row.value is not None and (
            isinstance(row.value, bool)
            or not isinstance(row.value, (int, float))
            or not math.isfinite(row.value)
        ):
            raise ValueError("Values must be finite numbers or null")
        if row.availability_basis == "release_timestamp":
            if row.vintage_date is not None or row.valid_through_date is not None or not row.release_evidence.strip():
                raise ValueError("Actual release times need evidence and no date-only vintage field")
            effective = _instant(row.effective_at)
            row = replace(row, effective_at=effective.isoformat())
        elif row.availability_basis == "vintage_date":
            if row.effective_at is not None:
                raise ValueError("Date-only vintages cannot claim an effective timestamp")
            _day(row.vintage_date)
            if row.valid_through_date is not None and _day(row.valid_through_date) < _day(row.vintage_date):
                raise ValueError("Vintage validity ends before it starts")
        else:
            raise ValueError("availability_basis must be release_timestamp or vintage_date")
        start, end = _bounds(row)
        if day > start.astimezone(zone).date():
            raise ValueError("Availability predates the observation date; forecasts need a separate contract")
        if retrieved < start:
            raise ValueError("Retrieval precedes the recorded release/vintage")
        metadata = row.unit, row.source, row.source_timezone
        if row.series_id in series_metadata and series_metadata[row.series_id] != metadata:
            raise ValueError(f"Conflicting series metadata: {row.series_id}")
        series_metadata[row.series_id] = metadata
        row = replace(row, retrieved_at=retrieved.isoformat())
        normalized.append(row)
        grouped[row.series_id, row.observation_date].append((start, end, row))
    for key, versions in grouped.items():
        versions.sort(key=lambda entry: entry[0])
        for previous, current in zip(versions, versions[1:]):
            if current[0] == previous[0] or current[0] < previous[1]:
                raise ValueError(f"Duplicate or overlapping revision versions: {key}")
            valid_through = previous[2].valid_through_date
            if valid_through is not None and current[0].astimezone(ZoneInfo(current[2].source_timezone)).date() <= _day(valid_through):
                raise ValueError(f"Overlapping declared vintage validity periods: {key}")
    return sorted(normalized, key=lambda row: (row.series_id, row.observation_date, _bounds(row)[0]))


def _select_validated(rows, cutoff, allow_date_only):
    chosen = {}
    for row in rows:
        start, end = _bounds(row)
        if start > cutoff:
            continue
        key = row.series_id, row.observation_date
        # Validated input is sorted by effective time within each observation.
        selected = asdict(row)
        selected.update(
            forecast_at=cutoff.isoformat(),
            available_at=end.isoformat(),
            vintage="point_in_time" if row.availability_basis == "release_timestamp" else "date_only",
            missing_reason="source_missing" if row.value is None else "",
        )
        if row.availability_basis == "vintage_date":
            if cutoff < end:
                selected.update(value=None, available_at=None, missing_reason="uncertain_vintage_day")
            elif not allow_date_only:
                selected.update(value=None, missing_reason="unverified_release_time")
            if row.valid_through_date is not None and cutoff.astimezone(ZoneInfo(row.source_timezone)).date() > _day(row.valid_through_date):
                selected.update(value=None, missing_reason="outside_vintage_period")
        chosen[key] = selected
    return list(chosen.values())


def select_vintages_as_of(
    rows: list[VersionedObservation], forecast_at: str, *, allow_date_only: bool = False
) -> list[dict]:
    """Select the last known revision for each observation, preserving nulls.

    Future revisions cannot affect earlier selected values. Date-only uncertainty
    can mask a value, never resurrect an earlier revision. ``available_at`` for
    an allowed date-only record is its conservative next-midnight bound, not an
    actual publication timestamp. The original vintage date/basis are retained.
    """
    if not isinstance(allow_date_only, bool):
        raise ValueError("allow_date_only must be a bool")
    return _select_validated(validate_archive(rows), _instant(forecast_at), allow_date_only)


def build_vintage_feature_panel(
    rows: list[VersionedObservation], forecast_times: list[str], specs: list[dict],
    *, allow_date_only: bool = False,
) -> list[dict]:
    """Select revisions, then newest observations, with the pilot's age rules."""
    if not isinstance(allow_date_only, bool):
        raise ValueError("allow_date_only must be a bool")
    records = validate_archive(rows)
    spec_map = {spec["series_id"]: spec for spec in specs}
    if len(spec_map) != len(specs):
        raise ValueError("Duplicate series specification")
    for row in records:
        if row.series_id not in spec_map:
            raise ValueError(f"Unknown series_id: {row.series_id}")
    for identifier, spec in spec_map.items():
        stale_days = spec.get("stale_days", 7)
        if isinstance(stale_days, bool) or not isinstance(stale_days, int) or stale_days < 0:
            raise ValueError(f"{identifier}: stale_days must be a nonnegative integer")
        ZoneInfo(spec.get("timezone", "UTC"))
    panel = []
    for forecast_at in forecast_times:
        cutoff = _instant(forecast_at)
        snapshots = _select_validated(records, cutoff, allow_date_only)
        latest = {entry["series_id"]: entry for entry in snapshots}
        for identifier, spec in spec_map.items():
            feature = {
                "forecast_at": cutoff.isoformat(), "series_id": identifier,
                "value": None, "observation_date": None, "available_at": None,
                "age_days": None, "vintage": None, "quality_flag": "",
                "missing_reason": "absent", "availability_basis": None,
                "effective_at": None, "vintage_date": None, "release_evidence": "",
                "valid_through_date": None,
            }
            if identifier in latest:
                selected = latest[identifier]
                feature.update({key: selected[key] for key in feature if key in selected})
                age = (cutoff.astimezone(ZoneInfo(spec.get("timezone", "UTC"))).date()
                       - _day(selected["observation_date"])).days
                if age < 0:
                    raise ValueError("Feature cutoff precedes observation date in the configured calendar")
                feature["age_days"] = age
                if not feature["missing_reason"] and age > spec.get("stale_days", 7):
                    feature.update(value=None, missing_reason="stale")
            panel.append(feature)
    return panel


def write_archive_json(path: str | Path, rows: list[VersionedObservation]) -> None:
    """Write a deterministic archive, refusing to overwrite an existing file."""
    records = [asdict(row) for row in validate_archive(rows)]
    content = json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def read_archive_json(path: str | Path) -> list[VersionedObservation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "records"} or type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported vintage archive JSON format")
    if not isinstance(payload["records"], list):
        raise ValueError("Archive records must be a list")
    try:
        records = [VersionedObservation(**record) for record in payload["records"]]
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid archive record fields") from exc
    return validate_archive(records)


def write_archive_csv(path: str | Path, rows: list[VersionedObservation]) -> None:
    records = validate_archive(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELD_NAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)


def read_archive_csv(path: str | Path) -> list[VersionedObservation]:
    records = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(_FIELD_NAMES) or set(reader.fieldnames) != set(_FIELD_NAMES):
            raise ValueError("Invalid archive CSV columns")
        for entry in reader:
            if set(entry) != set(_FIELD_NAMES) or any(value is None for value in entry.values()):
                raise ValueError("Malformed archive CSV row")
            entry["value"] = None if entry["value"] == "" else float(entry["value"])
            for key in ("effective_at", "vintage_date", "valid_through_date"):
                entry[key] = entry[key] or None
            records.append(VersionedObservation(**entry))
    return validate_archive(records)
