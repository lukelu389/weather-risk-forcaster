# Phase 1 data contract

## Initial universe and units

| Series | Provider / identifier | Unit / interpretation |
|---|---|---|
| Henry Hub | FRED `DHHNGSP` (EIA source) | USD/MMBtu, daily spot benchmark |
| WTI | FRED `DCOILWTICO` (EIA source) | USD/barrel, daily Cushing spot benchmark |
| US three-month Treasury yield | FRED `DGS3MO` | Percent per year; not a daily investment return |
| US CPI | FRED `CPIAUCSL` | Seasonally adjusted index, 1982–84 = 100; monthly, revision-prone |
| USD/CAD | BoC `FXUSDCAD` | CAD per USD; daily average, not executable closing FX |
| Toronto City | ECCC station `31688`, climate `6158355` | Mean temperature in °C, precipitation in mm |
| Calgary Intl A | ECCC station `50430`, climate `3031092` | Mean temperature in °C, precipitation in mm |
| HDD18 / CDD18 | Derived separately at both stations | °C-day: max(18 − mean temperature, 0), max(mean temperature − 18, 0) |
| Canadian equity basket candidates | Yahoo `ENB.TO`, `SU.TO`, `FTS.TO`, `EMA.TO` | Adjusted CAD/share; raw close and actions retained in cached history |
| Lower-48 working gas | EIA `NG.NW2_EPG0_SWO_R48_BCF.W` | Bcf inventory; weekly reference date, not release date |
| US total gas consumption | EIA `NG.N9140US2.M` | MMcf consumed during the month; not Bcf/day |

FRED source definitions: [Henry Hub](https://fred.stlouisfed.org/series/DHHNGSP), [WTI](https://fred.stlouisfed.org/series/DCOILWTICO), [Treasury yield](https://fred.stlouisfed.org/series/DGS3MO), [CPI](https://fred.stlouisfed.org/series/CPIAUCSL). Other definitions: [BoC FX metadata](https://www.bankofcanada.ca/valet/series/FXUSDCAD/json), [EIA storage](https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm), [EIA consumption](https://www.eia.gov/dnav/ng/hist/n9140us2m.htm), [ECCC downloads](https://climate.weather.gc.ca/FAQ_e.html), [yfinance documentation](https://ranaroussi.github.io/yfinance/index.html).

Station IDs are configured explicitly; adapters check returned station name and climate ID. Both identities were confirmed during live ingestion of the pilot period. Stations may move, change instruments, close, or have incomplete elements. Preserve source flags such as estimated, missing, and trace. A reported zero with a trace flag is distinguishable from an ordinary zero; blank values remain null.

## Assumed availability policy

Every delay below is a **research assumption**, not a claim about a verified release calendar or historical vintage. All delays are calendar days. The canonical timestamps are UTC with offsets.

| Data | Assumed availability | Maximum age at alignment |
|---|---|---|
| Henry Hub / WTI | Observation + 7 days, 18:00 Toronto | 14 days |
| Treasury yield | Observation + 2 days, 18:00 Toronto | 10 days |
| CPI | Reference month-end + 60 days, 18:00 Toronto | 110 days |
| USD/CAD | Observation + 1 day, 18:00 Toronto | 7 days |
| Weather and derived degree days | Observation + 2 days, 12:00 station standard time | 5 days |
| Equity adjusted close | Observation + 1 day, 18:00 Toronto | 7 days |
| Gas storage | Reference date + 8 days, 18:00 Toronto | 21 days |
| Gas consumption | Reference month-end + 100 days, 18:00 Toronto | 160 days |

These deliberately delayed features cost timeliness. They are a starting sensitivity scenario, not optimized parameters. Even a long delay cannot fix a later data revision or an unusually delayed publication.

BoC normally publishes daily FX by 16:30 ET ([schedule](https://www.bankofcanada.ca/rates/exchange/)). EIA storage normally follows a Thursday release schedule, with exceptions ([methodology](https://ir.eia.gov/ngs/methodology.html)). Historical CPI seasonal adjustments can change ([BLS](https://www.bls.gov/cpi/seasonal-adjustment/)). Verified release timestamps and original vintages must replace assumptions for a strict historical study.

ECCC defines local standard time independently of daylight saving ([glossary](https://climate.weather.gc.ca/glossary_e.html)). The configuration uses fixed UTC−5 for Toronto and UTC−7 for Calgary (`Etc/GMT+5` and `Etc/GMT+7`; the IANA sign convention is reversed). The forecast grid uses `America/Toronto` and therefore changes with DST. Different element definitions and historical observing-day conventions still require review.

## Canonical row

Unique key in this phase: `(series_id, observation_date)`. Fields:

- `observation_date`: ISO date; monthly observations normalized to the last date of the reference month.
- `value`, `unit`, `source`: unfilled numeric value or null, configured unit, provider and source identifier.
- `available_at`: timezone-aware assumed availability, or an actual release timestamp only if independently established.
- `retrieved_at`: timestamp of the preserved source response; cached replays keep it unchanged.
- `availability_basis`: `assumed_lag`, `release_timestamp`, or `synthetic`.
- `vintage`: `latest`, `point_in_time`, or `synthetic`.
- `quality_flag`: original field flag, or an unmistakable synthetic-data label.

This is a single-snapshot schema. A true revision archive requires a versioned key and an as-of revision selection rule; simply relabeling latest data as `point_in_time` is invalid.

The feature join selects the newest observation with availability at or before the forecast cutoff. It then checks provenance, age measured from observation date, and missingness. A missing newest value does not revive an older value. No value can enter from the future. Provenance and cutoff protection are separate checks.

## Validation decisions

- Duplicate dates, malformed timestamps, inconsistent units, nonfinite numbers, impossible weather bounds, and nonpositive adjusted equity prices fail.
- Missing daily-weather dates and explicit nulls obey a configured 12% maximum. This is an ingestion quality threshold, not permission to fit a model with 12% missing values.
- Business-day, weekly and monthly gaps are reported. Holiday/release calendars are not yet modeled, so absent periods warn; explicit nulls still obey the threshold. Inspect coverage fractions before research use even when a run passes.
- Large configured absolute price changes warn and remain unchanged. There is no full-sample outlier fitting or clipping.
- Negative WTI is allowed. EIA records negative spot WTI in April 2020 ([daily table](https://www.eia.gov/dnav/pet/hist/RWTCD.htm)); log returns are undefined across nonpositive prices. The later return/P&L definition must handle this explicitly.
- Standard errors, model calibration, return transforms, VaR/ES, exchange sessions, and portfolio weights are not part of Phase 1.

## Cache and logging

HTTPS requests use bounded retries and timeouts. Cached responses are content-addressed by SHA-256. API credentials are removed from request identities and redacted if echoed in response bytes; hashes therefore identify the **sanitized stored response**. Never publish cached provider data without checking applicable source terms.

`--refresh` fetches each distinct request once within a run. Old blobs remain available. `--replay` pins the exact hashes in an earlier manifest, checks source-code and config hashes, and compares generated artifact hashes. This is reproducibility of a snapshot, not proof of historical availability.
