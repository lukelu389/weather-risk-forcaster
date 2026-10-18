# Post-pilot extension review — 2026-10-15

The implementable local components are complete and tested. The extended
empirical study is **not** complete: historical data validation failed, verified
release archives are absent, and no new experiment has been frozen. GPU work is
deferred at the user's request. Phases 1–7 and their original review records
remain intact. This is an extension record, not a relabeling of the pilot phases.

## Changes and their evidence

| Component | Exact change points | Verified result |
|---|---|---|
| Historical staging | `src/weather_risk/historical.py`, `config/history.json`, `tests/test_historical.py`, `Makefile` | Public downloads cached separately; seven artifacts and their manifests reproduce byte for byte with pinned source blobs. Parser, quality-flag, failure-report and replay tests pass. |
| Revision archives | `src/weather_risk/vintages.py`, `src/weather_risk/vintage_sources.py`, `config/alfred-cpi.json`, corresponding tests | Revision selection preserves null revisions and uncertainty; ALFRED pagination, units, bounds, credential handling and output hashes tested with fixtures. |
| Prospective gate | `src/weather_risk/prospective.py`, `tests/test_prospective.py` | Future-period, dependency-hash, bootstrap-resolution and first-access rules tested, including concurrent access, premature access and invalid numeric sentinels. |
| Futures/FX ledger | `src/weather_risk/execution.py`, `tests/test_execution.py` | Explicit individual contracts and roll legs; fill-based P&L, costs and foreign cash reconcile, including an independent randomized closed-book cashflow identity. |
| Documentation | README, this review, verification JSON, vintage/prospective/execution contracts | Actual live evidence is distinguished from synthetic tests and missing external inputs. |

No dependencies were added. New modules do not call the original final-evaluation
runner or modify its inputs. The existing Phase 5 seal and input hash were
verified after implementation. The recorded full suite has **304 passing tests**
(230 prior tests and 74 extension tests). Reproducibility applies to the recorded
local environment and retained source bytes, not arbitrary platforms or future
provider revisions. See [machine-readable evidence](extensions-verification.json).

## Historical data: acquired, with a failed modelling gate

The separate 2000–2025 snapshot has **225,819 observations in 29 series**, from 15
cached public requests. All values retain latest-vintage / assumed-lag labels.
The original pilot input and cache are separate. No model fit or new score used
this extension.

Three [NOAA GHCN-Daily](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)
airport stations provide six fields each: minimum and maximum temperature,
their midpoint, HDD18, CDD18 and precipitation. The midpoint is explicitly named
`midrange_temp`; it is not substituted for an observed daily mean. Nonblank
quality flags and missing source values become null, with flags and raw records
retained. Station identifiers and names are checked against the downloaded
inventory. Chicago and New York are candidate demand-region proxies, and Houston
is a candidate Gulf-region proxy. This geographic expansion does not establish
population weighting or predictive usefulness. All three have complete midpoint
coverage in 2008 and 2014–2016 in this snapshot.

Live retrieval and parsing succeeded for the official EIA
[weekly storage table](https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm)
and [monthly consumption table](https://www.eia.gov/dnav/ng/hist/n9140us2m.htm).
They do not require an API key. The saved tables provide 835 storage observations
from 2010-01-01 and 300 consumption observations from 2001-01-31 through 2025.
The retrieved storage table therefore supplies **no 2008 storage observations**.
It must not be described as full crisis-period storage coverage. These are
revised histories; page-level release dates are not historical release timestamps.

The separate FRED `DEXCAUS` CAD-per-USD series is not spliced into the pilot's
Bank of Canada daily-average FX series. Weekday/weekly/monthly coverage counts
are approximate diagnostics, not exchange or publication calendars.

The unchanged validation rules reject three conditions:

- Henry Hub has 1,561 explicit nulls, about 23.0% of its rows, above the 12% limit.
  The missingness is concentrated in 2000–2007. This describes the retrieved
  snapshot; it does not prove that complete historical source data do not exist.
- Yahoo ENB.TO adjusted close is zero on 2003-05-19.
- Yahoo SU.TO adjusted close is zero on 2001-09-12.

The raw values, source snapshots, failed earlier attempts, and diagnostic reports
are retained. No threshold was relaxed, no nonpositive price was converted into
a valid price, and no missing date was filled. The final reproducible manifest
has `status: validation_failed`; the command returns exit status 1 even when the
replay matches. Acquisition/parser errors separately save `status: failed`.
See the [complete coverage report](../reports/extensions/history_report.md).

## What still requires inputs or a research decision

1. **Data acceptance:** investigate the missing gas history and two zero equity
   observations against authoritative source records/calendars. Obtain older
   storage data if 2008 storage is required. Document any exclusion or source
   substitution before choosing features or training on the extension.
2. **Historical information sets:** live ALFRED verification needs a locally
   configured FRED API key. Its date-only metadata still does not certify
   intraday release times. Verified release evidence and archived weather
   forecasts require actual archives. See [vintage instructions](vintage-data.md).
3. **New experiment:** select the accepted dataset, target, model variants,
   complete dependencies, hypotheses and future period, then use the
   [new protocol gate](prospective-protocol.md). No real protocol was frozen;
   2024–2025 cannot become untouched again. A new runner must consume the new
   version-aware panel and enforce the frozen forecasting rules.
4. **Executable portfolio study:** supply verified expiry-specific contracts,
   settlements, execution fills, FX, session/roll calendars and measured costs.
   The [ledger](execution-input-contract.md) is implemented, but it is not a
   strategy, margin engine or empirical backtest.
5. **GPU:** wait for the user to resume this work with the school's actual
   cluster instructions. No scheduler, account, partition, CUDA environment,
   job script or resource allocation was guessed or submitted.

The earlier phase-change PDF is preserved as a record of the completed pilot
before these extensions. The README and this review carry the updated status.
The previous research follow-up remains paused.

## Three interview questions

1. **Why can an extension be reproducible while its data gate fails?**
   Reproducibility means the same source bytes and code produce the same outputs.
   It does not make missing or invalid data suitable for modelling. Here the
   rejection report itself reproduces, and the failure remains visible.
2. **Why isn't ALFRED's vintage date equivalent to a verified publication time?**
   A date does not establish which intraday forecast origins could have seen a
   release. Strict selection withholds it; an explicit approximation waits until
   the next local midnight and keeps the weaker provenance label.
3. **Why does rolling a futures position not earn the price difference between
   expiries?** Each expiry has its own cost basis and settlement cashflows. A
   roll closes one contract and opens another. P&L comes from those individual
   price changes and fills, with costs and FX accounted for separately.
