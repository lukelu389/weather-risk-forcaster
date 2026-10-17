"""ALFRED date-only archives and a standalone import/select/fetch command.

Official contracts:
https://fred.stlouisfed.org/docs/api/fred/series_observations.html
https://fred.stlouisfed.org/docs/api/fred/realtime_period.html

The API's inclusive real-time dates are not intraday release timestamps. A
requested range can truncate a version's start/end, so both bounds are retained
without claiming they identify the original publication. Cached payloads and
request metadata remain available for audit. Live access requires FRED_API_KEY;
offline cache replay does not. No credential is accepted on the command line.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from .cache import Cache
from .vintages import (
    VersionedObservation, _day, _instant, read_archive_csv, read_archive_json,
    select_vintages_as_of, validate_archive, write_archive_csv, write_archive_json,
)


_FREQUENCIES = {"daily": "D", "business_daily": "D", "weekly": "W", "monthly": "M"}


def _spec(spec: dict) -> None:
    for name in ("series_id", "source_series", "unit", "source_unit", "timezone"):
        if not isinstance(spec.get(name), str) or not spec[name].strip():
            raise ValueError(f"ALFRED specification needs {name}")
    if spec.get("frequency") not in _FREQUENCIES:
        raise ValueError("Supported ALFRED frequencies: daily, business_daily, weekly, monthly")
    ZoneInfo(spec["timezone"])


def parse_alfred_observations(payload: dict, spec: dict, retrieved_at: str) -> list[VersionedObservation]:
    """Parse output_type=1, units=lin; preserve null and finite revisions.

    ``source_unit`` is checked against series metadata by ``fetch_alfred``. For
    manual import, the caller is responsible for that metadata check. ``timezone``
    defines an explicit date-boundary assumption, not verified publication time.
    Monthly reference dates are normalized to month end as in the pilot schema.
    """
    _spec(spec)
    _instant(retrieved_at)
    if not isinstance(payload, dict) or type(payload.get("output_type")) is not int or payload["output_type"] != 1 or payload.get("units") != "lin":
        raise ValueError("Expected untransformed ALFRED output_type=1 JSON")
    if not isinstance(payload.get("observations"), list):
        raise ValueError("ALFRED response is missing observations")
    records = []
    for entry in payload["observations"]:
        if not isinstance(entry, dict) or not {"date", "value", "realtime_start", "realtime_end"}.issubset(entry):
            raise ValueError("Malformed ALFRED observation")
        observed = _day(entry["date"])
        if spec["frequency"] == "monthly":
            observed = observed.replace(day=calendar.monthrange(observed.year, observed.month)[1])
        value = entry["value"]
        if value == "." or value is None:
            value = None
        else:
            if isinstance(value, bool) or not isinstance(value, (str, float, int)):
                raise ValueError("Invalid ALFRED numeric value")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Nonfinite ALFRED numeric value")
        records.append(VersionedObservation(
            series_id=spec["series_id"], observation_date=observed.isoformat(), value=value,
            unit=spec["unit"], source=f"alfred:{spec['source_series']}", retrieved_at=retrieved_at,
            source_timezone=spec["timezone"], availability_basis="vintage_date",
            vintage_date=entry["realtime_start"], valid_through_date=entry["realtime_end"],
            quality_flag="ALFRED date-only validity interval; query bounds may truncate dates",
        ))
    return validate_archive(records)


def fetch_alfred(
    spec: dict, observation_start: str, observation_end: str,
    realtime_start: str, realtime_end: str, cache: Cache, *, page_size: int = 100000,
) -> list[VersionedObservation]:
    """Fetch all pages through the existing sanitized, content-addressed cache.

    Source metadata must match the explicitly expected unit and frequency.
    Repeated offsets, changing totals, wrong ranges, and truncated pages fail.
    The request never aggregates or transforms the source values.
    """
    _spec(spec)
    if _day(observation_start) > _day(observation_end) or _day(realtime_start) > _day(realtime_end):
        raise ValueError("ALFRED start dates must not follow end dates")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100000:
        raise ValueError("page_size must be an integer from 1 to 100000")
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key and not cache.offline:
        raise ValueError("FRED_API_KEY is required for live ALFRED acquisition; no request was sent")
    common = {
        "series_id": spec["source_series"], "api_key": api_key, "file_type": "json",
        "realtime_start": realtime_start, "realtime_end": realtime_end,
    }
    raw, _ = cache.get("https://api.stlouisfed.org/fred/series", common)
    metadata = json.loads(raw)
    variants = metadata.get("seriess") if isinstance(metadata, dict) else None
    if not isinstance(variants, list) or not variants:
        raise ValueError("ALFRED response missing series metadata")
    for variant in variants:
        if not isinstance(variant, dict) or variant.get("id") != spec["source_series"] or variant.get("units") != spec["source_unit"] or variant.get("frequency_short") != _FREQUENCIES[spec["frequency"]]:
            raise ValueError("ALFRED series identity, unit, or frequency differs from the specification")
    request_start = observation_start[:8] + "01" if spec["frequency"] == "monthly" else observation_start
    offset, expected_count = 0, None
    records = []
    while True:
        params = {
            **common, "observation_start": request_start, "observation_end": observation_end,
            "output_type": 1, "units": "lin", "sort_order": "asc",
            "offset": offset, "limit": page_size,
        }
        raw, record = cache.get("https://api.stlouisfed.org/fred/series/observations", params)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Malformed ALFRED observations response")
        count, returned_offset, limit = payload.get("count"), payload.get("offset"), payload.get("limit")
        if type(count) is not int or count < 0 or type(returned_offset) is not int or returned_offset != offset or type(limit) is not int or limit != page_size:
            raise ValueError("Invalid ALFRED pagination metadata")
        if expected_count is not None and count != expected_count:
            raise ValueError("ALFRED observation count changed during pagination; retry a fresh acquisition")
        expected_count = count
        if any(payload.get(key) != value for key, value in (
            ("realtime_start", realtime_start), ("realtime_end", realtime_end),
            ("observation_start", request_start), ("observation_end", observation_end),
            ("sort_order", "asc"),
        )):
            raise ValueError("ALFRED returned a different requested range or sort order")
        page = parse_alfred_observations(payload, spec, record["retrieved_at"])
        if len(page) != min(page_size, count - offset):
            raise ValueError("ALFRED returned a truncated or inconsistent page")
        for row in page:
            if not request_start <= row.observation_date <= observation_end or not realtime_start <= row.vintage_date <= row.valid_through_date <= realtime_end:
                raise ValueError("ALFRED row lies outside the requested observation/vintage range")
        records.extend(row for row in page if observation_start <= row.observation_date <= observation_end)
        offset += len(page)
        if offset >= count:
            break
    if not records:
        raise ValueError("ALFRED returned no observations in the requested normalized period")
    return validate_archive(records)


def _read(path: Path):
    if path.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("Archive paths must end in .csv or .json")
    return read_archive_csv(path) if path.suffix.lower() == ".csv" else read_archive_json(path)


def _write(path: Path, rows):
    if path.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("Archive paths must end in .csv or .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    (write_archive_csv if path.suffix.lower() == ".csv" else write_archive_json)(path, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("import", help="Validate and convert a revision archive between JSON and CSV")
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    select = commands.add_parser("select", help="Write as-of revision selections as JSON")
    select.add_argument("--archive", required=True, type=Path)
    select.add_argument("--forecast-at", required=True)
    select.add_argument("--allow-date-only", action="store_true")
    select.add_argument("--output", required=True, type=Path)
    fetch = commands.add_parser("fetch-alfred", help="Fetch one explicit series specification and write its archive")
    fetch.add_argument("--spec", required=True, type=Path, help="JSON object with series_id, source_series, unit, source_unit, timezone, frequency")
    for field in ("observation-start", "observation-end", "realtime-start", "realtime-end"):
        fetch.add_argument(f"--{field}", required=True)
    fetch.add_argument("--cache", type=Path, default=Path("data/cache"))
    fetch.add_argument("--output", required=True, type=Path)
    modes = fetch.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.command == "import":
        _write(args.output, _read(args.input))
    elif args.command == "select":
        selected = select_vintages_as_of(_read(args.archive), args.forecast_at, allow_date_only=args.allow_date_only)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {manifest}")
        cache = Cache(args.cache, offline=args.offline, refresh=args.refresh)
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        rows = fetch_alfred(spec, args.observation_start, args.observation_end, args.realtime_start, args.realtime_end, cache)
        _write(args.output, rows)
        code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ('vintage_sources.py', 'vintages.py', 'cache.py')}
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump({"spec": spec, "records": len(rows), "requests": cache.records,
                       "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                       "code_sha256": code,
                       "requested_ranges": {key: getattr(args, key) for key in
                                            ('observation_start', 'observation_end', 'realtime_start', 'realtime_end')},
                       "availability_basis": "vintage_date", "actual_release_times_verified": False},
                      handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
