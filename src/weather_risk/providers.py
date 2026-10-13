"""Public-source adapters. Observation dates are never release timestamps.

These endpoints return latest-vintage history, not point-in-time archives.
Configured calendar-day release lags are explicit assumptions, not evidence
of historical publication times. No adapter forward-fills or repairs values.
"""

from __future__ import annotations

import calendar
import csv
import io
import json
import math
import os
import re
from datetime import date, timedelta
from urllib.parse import quote

from .cache import Cache
from .schema import Observation, assumed_availability


class ProviderError(ValueError):
    """Unexpected payload, unsupported field, or inconsistent source units."""


def _number(value) -> float | None:
    # Documented source missing markers only; unexpected strings must fail.
    if value is None or str(value).strip() in {"", ".", "NA", "N/A", "NaN", "M"}:
        return None
    if isinstance(value, bool):
        raise ProviderError("Boolean is not a numeric observation")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ProviderError("Unexpected nonnumeric observation") from exc
    if not math.isfinite(result):
        raise ProviderError("Nonfinite observation")
    return result


def _period(value: str) -> str:
    """Anchor monthly/annual periods at period end before applying release lag."""
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        return date(year, month, calendar.monthrange(year, month)[1]).isoformat()
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-12-31"
    return date.fromisoformat(value).isoformat()


def _unit(value: str) -> str:
    value = value.lower().strip().replace("°", "").replace("·", " ")
    value = value.replace("_", " ").replace(" per ", "/")
    value = re.sub(r"\s+", "", value)
    aliases = {
        "celsius": "c", "degc": "c", "degcday": "cday", "c-day": "cday", "c-days": "cday",
        "c-days/day": "cday", "degreedays": "cday", "celsiusdays": "cday",
        "cad/share": "cad", "usd/share": "usd", "usd/bbl": "usd/barrel",
        "dollars/barrel": "usd/barrel", "$/barrel": "usd/barrel",
        "dollars/millionbtu": "usd/mmbtu", "dollars/mmbtu": "usd/mmbtu",
        "$/mmbtu": "usd/mmbtu", "billioncubicfeet": "bcf", "millioncubicfeet": "mmcf",
        "%": "percent",
    }
    return aliases.get(value, value)


def _check_unit(spec: dict, observed: str) -> None:
    if _unit(spec["unit"]) != _unit(observed):
        raise ProviderError(f"{spec['series_id']}: configured unit {spec['unit']!r} "
                            f"does not match source unit {observed!r}")


def _observation(spec: dict, day: str, value, metadata: dict, flag: str = "") -> Observation:
    return Observation(
        series_id=spec["series_id"], observation_date=day, value=_number(value),
        unit=spec["unit"], source=f"{spec['provider']}:{spec['source_series']}",
        available_at=assumed_availability(day, spec["timezone"], spec["lag_days"],
                                          spec["available_hour"]),
        retrieved_at=metadata["retrieved_at"], availability_basis="assumed_lag",
        vintage="latest", quality_flag=flag,
    )


def _csv(data: bytes, date_columns: tuple[str, ...]) -> csv.DictReader:
    text = data.decode("utf-8-sig")
    # ECCC exports sometimes prepend station metadata before the CSV header.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        header = next(csv.reader([line]))
        if any(column in header for column in date_columns):
            return csv.DictReader(io.StringIO("\n".join(lines[i:])))
    raise ProviderError("CSV missing the expected date header")


def _fred(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    query_start = start[:8] + "01" if spec.get("frequency") == "monthly" else start
    payload, metadata = cache.get("https://fred.stlouisfed.org/graph/fredgraph.csv", {
        "id": spec["source_series"], "cosd": query_start, "coed": end})
    reader = _csv(payload, ("DATE", "observation_date"))
    column = spec["source_series"]
    if column not in reader.fieldnames:
        raise ProviderError(f"FRED CSV missing requested series {column}")
    known_units = {"DHHNGSP": "USD/MMBtu", "DCOILWTICO": "USD/barrel",
                   "DFF": "percent", "FEDFUNDS": "percent", "DGS10": "percent",
                   "DGS3MO": "percent", "CPIAUCSL": "index_1982_84_100"}
    if column in known_units:
        _check_unit(spec, known_units[column])
    result = []
    for row in reader:
        day = row.get("DATE", row.get("observation_date"))
        if spec.get("frequency") == "monthly":
            day = _period(day[:7])
        else:
            day = _period(day)
        result.append(_observation(spec, day, row[column], metadata))
    return result


def _boc(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    column = spec["source_series"]
    query_start = start[:8] + "01" if spec.get("frequency") == "monthly" else start
    payload, metadata = cache.get(
        f"https://www.bankofcanada.ca/valet/observations/{quote(column, safe='')}/json",
        {"start_date": query_start, "end_date": end})
    document = json.loads(payload)
    details = document.get("seriesDetail", {}).get(column)
    if not isinstance(details, dict):
        raise ProviderError("BoC response missing requested series metadata")
    if column == "FXUSDCAD":
        _check_unit(spec, "CAD/USD")
    elif column == "FXCADUSD":
        _check_unit(spec, "USD/CAD")
    if details.get("units") or details.get("unit"):
        _check_unit(spec, details.get("units", details.get("unit")))
    if "observations" not in document:
        raise ProviderError("BoC response missing observations")
    result = []
    for row in document["observations"]:
        if column not in row or "v" not in row[column]:
            raise ProviderError("BoC observation missing requested value")
        day = _period(row["d"][:7] if spec.get("frequency") == "monthly" else row["d"])
        result.append(_observation(spec, day, row[column]["v"], metadata))
    return result


def _eccc(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    field = spec["source_field"]
    expected_units = {"Mean Temp (°C)": "°C", "Total Precip (mm)": "mm",
                      "HDD18": "°C·day", "CDD18": "°C·day"}
    if field not in expected_units:
        raise ProviderError(f"Unsupported ECCC field: {field}")
    _check_unit(spec, expected_units[field])
    input_field = "Mean Temp (°C)" if field in {"HDD18", "CDD18"} else field
    flag_field = input_field.split(" (")[0] + " Flag"
    result = []
    for year in range(date.fromisoformat(start).year, date.fromisoformat(end).year + 1):
        payload, metadata = cache.get("https://climate.weather.gc.ca/climate_data/bulk_data_e.html", {
            "format": "csv", "stationID": spec["source_series"], "Year": year,
            "Month": 1, "Day": 1, "timeframe": 2, "submit": "Download Data"})
        reader = _csv(payload, ("Date/Time",))
        if input_field not in reader.fieldnames:
            raise ProviderError(f"ECCC CSV missing {input_field}")
        for row in reader:
            if spec.get("station_name") and row.get("Station Name", "").strip().upper() != spec["station_name"].upper():
                raise ProviderError("ECCC station name does not match configured station")
            if spec.get("climate_id") and row.get("Climate ID", "").strip() != spec["climate_id"]:
                raise ProviderError("ECCC climate ID does not match configured station")
            day = _period(row["Date/Time"])
            if date.fromisoformat(day).year != year:
                raise ProviderError("ECCC response contains an unexpected year")
            value = _number(row[input_field])
            flag = row.get(flag_field, "").strip()
            if value is not None and field == "HDD18":
                value = max(18.0 - value, 0.0)
            elif value is not None and field == "CDD18":
                value = max(value - 18.0, 0.0)
            result.append(_observation(spec, day, value, metadata, flag))
    return result


def _eia(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    key = os.environ.get("EIA_API_KEY", "")
    if not key and not cache.offline:
        raise ProviderError("EIA_API_KEY is required for live EIA ingestion (free registration)")
    result = []
    offset = 0
    total = None
    while total is None or offset < total:
        query_start = start[:7] if spec.get("frequency") == "monthly" else start
        query_end = end[:7] if spec.get("frequency") == "monthly" else end
        params = {"api_key": key, "start": query_start, "end": query_end, "offset": offset,
                  "length": 5000, "sort[0][column]": "period", "sort[0][direction]": "asc"}
        payload, metadata = cache.get(
            f"https://api.eia.gov/v2/seriesid/{quote(spec['source_series'], safe='')}", params)
        document = json.loads(payload)
        response = document.get("response", {})
        if not isinstance(response.get("data"), list) or "total" not in response:
            raise ProviderError("EIA response missing data/total; check series ID and credentials")
        try:
            new_total = int(response["total"])
        except (ValueError, TypeError) as exc:
            raise ProviderError("Invalid EIA pagination total") from exc
        if new_total < 0 or (total is not None and new_total != total):
            raise ProviderError("EIA total changed during pagination; retry with a frozen date range")
        total = new_total
        rows = response["data"]
        if (not rows and offset < total) or offset + len(rows) > total:
            raise ProviderError("EIA response is truncated or pagination is inconsistent")
        for row in rows:
            field = spec.get("source_field", "value")
            units = row.get(f"{field}-units", row.get("units"))
            if not units or field not in row:
                raise ProviderError("EIA row missing value or units")
            _check_unit(spec, units)
            result.append(_observation(spec, _period(row["period"]), row[field], metadata))
        offset += len(rows)
    return result


def _yahoo(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    field = spec.get("source_field", "Adj Close")
    if field not in {"Close", "Adj Close"}:
        raise ProviderError("Yahoo source_field must be Close or Adj Close")
    exclusive_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    options = {"start": start, "end": exclusive_end, "interval": "1d",
               "auto_adjust": False, "back_adjust": False, "repair": False,
               "actions": True, "keepna": True, "rounding": False, "timeout": 30}

    def fetch() -> bytes:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderError("Install the optional market extra to fetch Yahoo data") from exc
        yf.set_tz_cache_location(str(cache.root / "yahoo-client"))
        ticker = yf.Ticker(spec["source_series"])
        frame = ticker.history(**options)
        if frame.empty or field not in frame.columns:
            raise ProviderError("Yahoo returned no data or omitted the requested price field")
        rows = []
        for timestamp, values in frame.iterrows():
            row = {"date": timestamp.date().isoformat()}
            for column, value in values.items():
                row[column] = None if value is None or math.isnan(float(value)) else float(value)
            rows.append(row)
        metadata = ticker.get_history_metadata()
        return json.dumps({"rows": rows, "currency": metadata.get("currency"),
                           "yfinance_version": yf.__version__}, allow_nan=False).encode()

    payload, metadata = cache.get_or_create(
        {"provider": "yahoo", "ticker": spec["source_series"], "options": options}, fetch)
    document = json.loads(payload)
    currency = document.get("currency")
    if not currency or _unit(spec["unit"]).split("/")[0] != _unit(currency):
        raise ProviderError("Yahoo quotation currency does not match configured unit")
    result = []
    for row in document["rows"]:
        if field not in row:
            raise ProviderError("Cached Yahoo response lacks requested price field")
        result.append(_observation(spec, _period(row["date"]), row[field], metadata))
    return result


def fetch_series(spec: dict, start: str, end: str, cache: Cache) -> list[Observation]:
    """Fetch one configured field. Both date bounds are inclusive."""
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must not be after end")
    adapters = {"fred": _fred, "boc": _boc, "eccc": _eccc, "eia": _eia, "yahoo": _yahoo}
    if spec["provider"] not in adapters:
        raise ProviderError(f"Unknown provider {spec['provider']}")
    result = adapters[spec["provider"]](spec, start, end, cache)
    result = sorted((row for row in result if start <= row.observation_date <= end),
                    key=lambda row: row.observation_date)
    if not result or not any(row.value is not None for row in result):
        raise ProviderError(f"No usable observations for {spec['series_id']} in requested range")
    if len({row.observation_date for row in result}) != len(result):
        raise ProviderError(f"Duplicate observation dates for {spec['series_id']}")
    return result
