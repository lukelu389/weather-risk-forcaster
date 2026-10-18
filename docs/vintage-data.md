# Revision-aware data archives

The new `vintages.py` and `vintage_sources.py` modules are independent of the
single-snapshot pilot. They store every revision, including revisions to missing
values, and select the version eligible at a specified forecast time. They do
not change the pilot's latest-vintage labels or certify its historical inputs.

## What is implemented

`VersionedObservation` adds release evidence and version validity to an
observation. The required fields are `series_id`, `observation_date`, `value`,
`unit`, `source`, `retrieved_at`, `source_timezone` and `availability_basis`.
Values must be finite numbers or null. Timestamps require seconds and an explicit
UTC offset. Observation dates use `YYYY-MM-DD`.

Use exactly one evidence type:

- `release_timestamp`: provide `effective_at` and a nonempty `release_evidence`
  reference to the archived source supporting that publication time. Leave
  `vintage_date` and `valid_through_date` null. The importer checks the contract,
  not the truth of the supplied evidence; a person must verify that evidence.
- `vintage_date`: provide the provider's vintage date and, when supplied, its
  inclusive `valid_through_date`. Leave `effective_at` null. This is a date-only
  information interval, not an intraday release timestamp.

Optional `quality_flag` preserves source qualifications. Unit, provider and
observation-calendar timezone must remain consistent within a series. Duplicate
versions, overlapping validity intervals, ambiguous revisions within a date-only
day, nonfinite values and releases before the observed period are rejected.
This schema describes observations; it does not represent weather forecasts with
distinct issue and valid times.

Strict selection masks date-only records. An explicit `--allow-date-only`
research override permits a value only after the entire vintage day has elapsed
in its declared timezone. During that day an older version is also masked: it
could already have been superseded. Null revisions and expired validity
intervals never revive an older finite value. Day boundaries account for daylight
saving changes. These conservative bounds remain labeled `date_only`.

The Python interfaces `select_vintages_as_of` and `build_vintage_feature_panel`
select versions, then the newest eligible observation, retaining missingness and
the specified staleness rule. They do not forward-fill across a missing revision.
JSON and CSV import/export preserve the contract and refuse to overwrite files.

## ALFRED acquisition

The implemented adapter uses the official [series metadata](https://fred.stlouisfed.org/docs/api/fred/series.html)
and [observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
with explicit observation and real-time ranges, untransformed values, and all
pages. Series identity, units, frequency, page totals, offsets and returned ranges
must match. Inclusive [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
remain date-only; query boundaries can truncate a version's original lifetime.
Monthly observations are anchored to month-end before selection.

Live access requires `FRED_API_KEY` in the environment. Do not paste a credential
into the command or commit it. No credential was available for this extension:
**ALFRED is implemented and fixture-tested, but not live-verified.** The example
CPI specification explicitly declares the expected source units; the adapter
rejects a mismatch instead of converting or silently relabeling values.

```sh
PYTHONPATH=src .venv/bin/python -m weather_risk.vintage_sources fetch-alfred \
  --spec config/alfred-cpi.json \
  --observation-start 2000-01-01 --observation-end 2025-12-31 \
  --realtime-start 2000-01-01 --realtime-end 2025-12-31 \
  --cache data/extensions/alfred --output runs/alfred-cpi/archive.json

PYTHONPATH=src .venv/bin/python -m weather_risk.vintage_sources select \
  --archive runs/alfred-cpi/archive.json --forecast-at 2008-10-01T18:00:00Z \
  --output runs/alfred-cpi/strict-selection.json
```

The second command deliberately uses strict selection, so ALFRED's date-only
values remain masked. Add `--allow-date-only` only for a disclosed approximation.
The archive and sidecar manifest record the saved bytes' hash, consuming code
hashes, requested ranges and sanitized source-request records. Raw response blobs
are content addressed. Offline acquisition uses `--offline` and a **new output
path**; it needs no key. It uses current cache pointers, so preserve the cache
alongside the archive and manifest. For pinned programmatic replay, instantiate
`Cache(cache_path, offline=True, pinned_records=manifest['requests'])` and pass it
to `fetch_alfred` with the recorded spec and ranges. Compare the serialized archive
hash to `archive_sha256`. Metadata flags describing a cache hit may differ.

For independently obtained release archives, import an exact-schema CSV or JSON:

```sh
PYTHONPATH=src .venv/bin/python -m weather_risk.vintage_sources import \
  --input path/to/release-archive.csv --output runs/releases/archive.json
```

No verified historical release archive has been supplied. Neither synthetic
tests nor ALFRED's date-only validity metadata establish verified intraday
availability. Archived weather forecast acquisition and a model runner consuming
these new versioned panels remain separate work requiring an agreed source and
a frozen new experiment.
