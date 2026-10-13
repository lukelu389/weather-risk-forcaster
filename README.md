# Weather-Aware Energy and Commodity Risk Forecaster

A quantitative research portfolio project focused on defensible volatility and tail-risk forecasts. **Phase 1 only:** ingestion, provenance, validation, and time alignment. No predictive results, VaR estimates, trading performance, or claims of bank model approval exist yet.

## Run the verified offline workflow

Requires Python 3.11+ and `make`; tested with Python 3.14.2. The core has **no third-party dependencies**. From this repository:

```sh
make reproduce
```

This runs the unit/integration tests and a deterministic, explicitly **synthetic** demonstration. It creates a new `runs/demo-*/` directory. Without `make`:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m weather_risk --mode demo --config config/demo.json
```

Each run saves:

| File | Purpose |
|---|---|
| `observations.csv` | Long-form observations, units, availability assumptions, retrieval time, flags |
| `feature_panel.csv` | As-of alignment diagnostics with explicit missing/stale/rejected reasons |
| `validation.json` | Errors, warnings, completeness, date ranges, and retained outlier flags |
| `config.json` | Exact resolved configuration, universe, dates, and seed |
| `source_snapshot.json` | Source code used for the run |
| `manifest.json` | Status, source checksums, code/config/output hashes, runtime and installed library versions |

`data/raw/` holds cached source responses. Failed runs retain diagnostics and exit nonzero. Raw data, runs, credentials, and virtual environments are ignored by Git.

## Fetch public data

```sh
make live-smoke
```

This requests January 2024 from FRED, Bank of Canada, and ECCC. Full pilot dates are **2018–2025**, chosen to overlap the initial source coverage, not as the final study period. For Canadian equities, install the optional Yahoo adapter in a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-market.lock.txt
make live PYTHON=.venv/bin/python
```

The complete tested Yahoo dependency set is version-pinned in `requirements-market.lock.txt` (Python 3.14.2, macOS ARM64). `requirements-market.txt` contains just the top-level dependency for intentionally resolving a new environment. The verified core workflow does not need Yahoo, NumPy, pandas, or API keys.

EIA requires a [free personal API key](https://www.eia.gov/opendata/). Set `EIA_API_KEY` in your process environment. Storage and consumption are explicitly optional while the key is absent, and every skipped source is logged. Change `required` to `true` when those features become prerequisites for an experiment. Do not commit a key.

Select a single source if diagnosing a provider:

```sh
PYTHONPATH=src python3 -m weather_risk --mode live --config config/smoke.json --series usd_cad
```

**Verification:** 55 tests pass. The live smoke test and full 2018–2025 pilot passed for FRED, Bank of Canada, ECCC and Yahoo: 39,760 observation rows across 17 series. Station identities were confirmed from returned ECCC records. EIA storage and consumption were explicitly skipped because no API key is configured; their adapters have fixture tests but no live verification. See `docs/verification.json` and the local run reports for evidence. The downloadable archive includes only a synthetic example; provider data remain in the local ignored data/run directories.

## Availability is a data requirement

An observation date is the period measured. `available_at` is when our research assumption allows it into a forecast. `retrieved_at` records when this particular response was downloaded. Monthly observations are normalized to month-end before applying a lag.

All current live adapters download **latest-vintage** history and assign **assumed** calendar-day delays. Those are not verified historical release times. Lagging revised history does not eliminate revision lookahead; FRED distinguishes historical information sets through its [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

Default alignment withholds these observations (`missing_reason=strict_vintage`). To inspect a deliberately approximate research panel:

```sh
PYTHONPATH=src python3 -m weather_risk --mode live --config config/live.json --allow-latest-vintage
```

That flag is recorded and never changes the provenance label. It does **not** certify a historically tradable backtest. Synthetic data use a separate provenance class. No automatic backfill, interpolation, winsorization, or current-value fallback is performed.

The alignment grid is weekdays at 18:00 America/Toronto. It is a diagnostic grid, not a TSX/NYMEX holiday calendar or an execution schedule. Tradeable targets and session calendars must be fixed before model fitting.

## Replay an exact run

Preserve the run directory and `data/raw/blobs/`, then substitute its manifest path:

```sh
PYTHONPATH=src python3 -m weather_risk --replay runs/demo-YYYYMMDDTHHMMSS.ffffffZ/manifest.json
```

Replay restores the saved config, source omissions and response hashes, even if newer downloads replaced the request pointers. An optional source skipped in the original run remains skipped on replay. It verifies output hashes. It rejects changed source code; `source_snapshot.json` preserves the original files for restoring that version. Run from the same repository/data root, or pass `--output-root` pointing to the preserved root. Ordinary `--offline` uses the currently cached requests; `--refresh` obtains a new snapshot, once per distinct request per run.

## Code map

```text
config/                 demo, live pilot, and small live smoke configurations
src/weather_risk/
  schema.py             observation contract and timezone-aware lag calculation
  providers.py          ECCC, FRED, BoC, EIA and optional Yahoo adapters
  cache.py              sanitized response blobs, integrity checks, pinned replay
  validation.py         missingness, units, physical limits, timestamps, jumps
  timing.py             as-of joins and expanding/rolling walk-forward splits
  demo.py               labeled synthetic integration data
  pipeline.py           CLI, local experiment manifests and failure reports
tests/                  offline unit and integration tests
docs/phase-1-review.md   design rationale, limitations and interview preparation
docs/data-contract.md   source definitions and information-time assumptions
```

Local JSON manifests are the experiment log for this phase. They avoid introducing an external service before models exist; model metrics and trained artifacts can later use the same run IDs or MLflow.

## Phase gates

1. **Data foundation:** implemented and verified offline and live for the 17-series pilot; EIA live verification requires a key.
2. **Exploration:** next, after reviewing the coverage audit and fixing a development/test protocol. Study only the development period when selecting features.
3. **Baselines:** historical simulation, GARCH/GJR-GARCH, weather regression; add VaR/ES implementations and unit tests together here.
4. **ML:** one sequence baseline and one state-space model, distributional/quantile outputs, fixed seeds, comparable budgets.
5. **Risk evaluation:** 95%/99% VaR and ES, coverage/independence, appropriate loss comparisons, calibration and uncertainty.
6. **Portfolio:** explicit currency, investable instrument, execution/roll, cost and regime conventions.
7. **Communication:** final README, two-page committee memo, dashboard and reproducible notebooks.

No random train/test splits are provided. `walk_forward_splits` supports expanding/rolling windows and a gap for overlapping target horizons. Later tuning, scaling and imputation must use training windows only; the final test period must be fixed before model selection. Previous test observations can enter later expanding training windows only after their labels have actually matured.

Phase 1 deliberately stops here. Read the [phase review](docs/phase-1-review.md) before beginning Phase 2.
