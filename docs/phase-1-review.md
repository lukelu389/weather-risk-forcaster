# Phase 1 review and interview preparation

## (a) Short plan

1. Define a small, explicit daily research universe and a common observation schema.
2. Implement cached adapters with source identity, units, response hashes and retrieval timestamps.
3. Validate observations before creating any feature panel; preserve valid extremes and missingness.
4. Join by information time, enforce a strict vintage gate, and provide chronological split indices.
5. Verify deterministic offline runs and exact cached replay. Smoke-test live endpoints separately.

All five steps are implemented. The offline suite passes 55 tests; the full live pilot passes validation for 17 series. EIA live verification remains conditional on a user-provided API key, with both skipped series explicitly recorded. Phase 2 has not started.

## (b) Code and design choices

The implementation lives in `src/weather_risk/`; `tests/` exercises the same modules used by the CLI. There are no placeholder models or made-up financial results.

**Standard-library core.** This phase parses modest daily histories and checks contracts; it does not need a tensor framework, orchestration service or database. This makes the required offline workflow portable and inspectable. DataFrame/ML dependencies can be introduced when they solve actual modeling needs.

**Long-form observations.** Keeping one row per series and date exposes different currencies, frequencies, publication lags and missing periods. A wide table created by blindly joining dates would hide those differences.

**Local experiment manifests.** The seed, exact config, source snapshot, hashes, runtime, package versions, source responses, and quality report are attached to each run. A hosted tracker would add setup before there are model metrics to compare. MLflow can be added when training begins without changing this provenance contract.

**Explicit as-of joins.** A Friday storage measurement cannot be used on Friday merely because its row has Friday's date. The join uses its availability timestamp. Historical release lags and revision provenance are tested independently. A synthetic demo validates software behavior; it is never evidence that weather predicts risk.

**Chronological evaluation primitive.** `walk_forward_splits(n, min_train, test_size, gap, train_window)` creates expanding or rolling indices. It cannot randomly mix future dates into training. A later model still must respect label maturity, fit transforms inside each training window, and tune only on nested chronological validation folds.

## (c) Run and verify

Run `make reproduce` from the project root. Expect all tests to pass and a new synthetic run containing observations, an alignment panel, a quality report, a config, a source snapshot and a manifest.

The tests cover missing observations, nulls, nonfinite values, duplicates, units, physical limits, retained negative WTI, timestamp awareness, exact cutoffs, DST, stale data, future-data perturbation, chronological splits, parser failures, bounded retries, key redaction, corrupted cache entries, repeated shared downloads, old-snapshot replay, deterministic outputs and failed-live-run logging.

Run `make live-smoke` to check public HTTPS access and the ECCC, FRED and BoC adapters. This passed here, and the full pilot also verified all four Canadian tickers through Yahoo. Install the version-pinned optional environment to reproduce market downloads. Set the EIA key and verify storage/consumption before making them mandatory. Never replace failed live responses with synthetic values.

For exact replay, use the manifest command in the README. Equality is checked on actual observation, panel, config and validation bytes; a successful test suite alone is not treated as proof that real market data have been obtained.

## (d) What could go wrong

1. **Revision lookahead.** Current CPI and corrected weather values may differ from the values known at the historical forecast time. Ordinary lagging does not solve this. Strict alignment rejects all current live-adapter records; research mode requires an explicit flag and remains labeled approximate.
2. **Wrong geography.** Toronto and Calgary are initial Canadian proxies. They do not represent the full US demand/production exposure driving Henry Hub. Add justified US population-weighted demand and Gulf production weather before making a broad weather-risk claim. Lagged realized weather is not an archived weather forecast.
3. **Invalid return construction.** A negative WTI price makes logarithmic returns invalid. Spot-price changes are not a self-financing futures strategy. Futures roll, collateral, denomination and cash-flow conventions belong in the later target/portfolio design.
4. **Selection and coverage bias.** Four currently selected TSX names are a selected-survivor basket, not a historical sector index. The pilot period is 2018–2025 and therefore cannot support 2008 or 2014–16 stress results. Calgary's current station starts in 2012 ([station history](https://climate.weather.gc.ca/climate_normals/results_1991_2020_e.html?climate_id=3031092)); the selected storage page begins in 2010. Historical station mapping, older FX series and precrisis training coverage need explicit work. Do not report nonexistent regime coverage.
5. **Calendar and availability assumptions.** Weekdays include exchange holidays; fixed delays do not capture every exceptional publication. A passing ingestion report is not approval to trade. Exact market sessions, origin/target alignment and forecast evaluation eligibility must be fixed before baselines.
6. **Data service failure or drift.** Free services can block requests, revise formats or change adjusted prices. Raw snapshots, strict parser checks and explicit failures make that visible. Yahoo's optional dependencies are version-pinned to the tested environment; future updates still need a live smoke check.
7. **Too few tail observations.** A few years of daily data offer few 99% exceedances. Later tests require power/uncertainty discussion, not just a p-value ranking. VaR/ES functions and their unit tests are deferred together to Phase 3.

## (e) What you should be able to explain

You should be able to trace one raw provider response through its checksum, canonical units, availability assumption, validation result, and inclusion/exclusion at a particular forecast cutoff. You should also distinguish reproducible software from historically valid research data, and explain why unresolved provenance restricts the claims you can make.

### Interview question 1: Why isn't shifting every feature by one day enough?

**Answer:** Different sources become public on different schedules. A storage measurement may be released several days after its reference date, and monthly CPI arrives after the reference month. There is also a second problem: today's historical value may incorporate a later revision. I store observation, availability and retrieval time separately, join only eligible observations, and withhold latest-vintage data by default. Verified release calendars and vintages are needed to claim a point-in-time study.

### Interview question 2: How do you decide whether an extreme price is a data error?

**Answer:** I distinguish hard validity constraints from statistical surprises. Invalid units, duplicate records and impossible physical values fail. Large financial jumps are flagged and checked against the original source, not automatically clipped. Negative WTI is a real example that breaks a generic positive-price rule. I preserve it and choose a later P&L definition that remains meaningful across zero.

### Interview question 3: Could another researcher reproduce your result after a provider revises its history?

**Answer:** Yes, for a preserved run snapshot: its manifest records the exact config, source code, runtime and response/output hashes. Cached blobs retain old responses, and replay pins the earlier hashes rather than downloading today's history. The deterministic pipeline verifies reproduced artifact hashes. That proves what computation I performed; it does not prove the snapshot was available at the original historical forecast time.

## Stop before Phase 2

The next step is to review the coverage audit and fix a development/test protocol. Resolve the missing historical regimes before claiming full-study coverage; configure the EIA key if storage/consumption will enter the models. No exploration, model fitting, or performance comparison has been performed in this phase.
