# Phase 2 review and interview preparation

## Scope and protocol

Phase 2 is complete for the existing 2018–2025 pilot snapshot. Exploration uses **2018–2021** only; **2022–2023** is reserved for tuning and **2024–2025** for final evaluation. Held-out rows are filtered before numeric conversion; Phase 1 had already inspected their data quality. No forecasting models, VaR/ES calculations, or performance comparisons were added.

The implementation preserves the existing architecture: `eda_core.py` constructs targets and statistics, `eda_weather.py` aligns weather and estimates descriptive relationships, `eda_plots.py` draws figures, and `eda.py` validates the protocol and records artifacts. The configuration pins the source observations by SHA-256 and the random seed to 1729.

Henry Hub and adjusted equities use percentage log returns. WTI uses dollar changes to retain negative prices. The four-equity basket averages simple returns at fixed 25% weights before converting back to a log return. It assumes rebalancing before costs and is a selected-survivor basket, not a historical index. Missing constituents are never reweighted. Null-adjacent movements and quote gaps over four calendar days remain unavailable.

## Tests and reproduction

Run `make test PYTHON=.venv/bin/python` after installing `requirements-analysis.lock.txt`: **86 tests pass, no skips**. The 55 core tests remain intact; 31 Phase 2 tests cover target definitions, missing intervals, DST, basket synchronization, input integrity, development isolation, weather availability, past-only anomaly references, tied ranks, block resampling, protocol rejection, latest-vintage acknowledgement, and successful/failed run artifacts. Runner lifecycle tests mock weather/plot work; the separate full empirical reruns exercise the real stack.

Two complete runs of the preserved input are compared in [phase-2-verification.json](phase-2-verification.json). All output hashes and manifests match, including PNG/PDF figures. Existing published analytical outputs are also checked against the reruns. `pip check` reports no broken requirements. See the README for the repeat-and-compare command.

This verifies repeatability in the installed Python 3.14.2/macOS ARM64 environment. It does not verify a fresh dependency installation, cross-platform rendering, or reconstruction from today's provider responses. The ignored input snapshot must be preserved. A seed alone cannot preserve revised data or changed software.

## Review of all five existing plots

The PNGs were visually inspected for legible axes, legends, units, captions, missing-data treatment, and consistency with the saved tables. No layout changes were needed. PDF files are included in the byte-level reproduction check; they were not separately visually inspected.

| Figure | Finding and interpretation |
|---|---|
| Movements and dispersion | Large movements are retained. Basket dispersion rises sharply in 2020; gas has a prominent 2021 episode. Broken rolling-SD lines follow the requirement for 21 valid grid rows, not zero volatility. WTI dollar dispersion cannot be compared directly with percentage-return dispersion. |
| Distributions and Q–Q | All three primary series deviate strongly from the normal reference. Excess kurtosis is 66.3 for gas, 266.6 for WTI, and 35.6 for the basket. WTI's two extreme movements dominate its tail estimate; these moments do not establish a stable tail law. |
| Autocorrelation | Lag-one absolute-movement correlations are 0.576, 0.478, and 0.511 for gas, WTI, and basket respectively. Basket magnitude dependence persists over several lags. WTI signed lag-one correlation is −0.330, so “returns have no dependence” would overstate the evidence. Lags count observed rows, preserving explicit null positions; dates absent from the source are not inserted. |
| Seasonality | HDD/CDD proxies show clear seasonal structure; target magnitude patterns differ by asset. Monthly counts are displayed. Four development years cannot separate a stable seasonal effect from a few crisis episodes. |
| Lagged weather relationships | The heatmap shows 30 primary pairs; the CSV retains all 70 pairs. The largest primary absolute rho is about 0.102 (Toronto precipitation versus gas magnitude). Its pointwise interval is approximately [0.036, 0.177]. This is not a multiple-testing-adjusted discovery or a forecast result. |

The predefined two-half sensitivity reinforces the regime caveat: basket SD rises from 0.727% in 2018–2019 to 1.933% in 2020–2021; WTI SD rises from $1.252 to $3.608 per barrel. Full-sample moments conceal substantial variation.

## Limitations and Phase 3 gate

Weather must be available by 18:00 Toronto on the calendar day before the target interval's earlier price. This conservative origin precedes both endpoints, but differs from a conventional next-close horizon. Latest-vintage observations and assumed publication lags remain explicitly qualified; timing checks cannot remove revision lookahead.

Temperature anomalies use earlier-year, same-month available observations with at least 20 valid reference rows. This is a short expanding reference, not an official climate normal. The paired moving-block bootstrap uses 20 original grid rows and 500 seeded resamples, retaining missing positions. Its intervals are pointwise and exploratory; shared seasonality, geographic mismatch, nonstationarity, and multiple comparisons limit interpretation.

Before Phase 3, settle the forecast horizon and label maturity, exchange-calendar treatment, investable target conventions, older regime coverage, and vintage policy. The pilot cannot support 2008 or 2014–2016 analysis. EIA remains unverified live without a key. No decision on these issues is silently resolved by completing EDA.

## Three interview questions

1. **Why use dollar changes for WTI but log returns for gas and equities?**
   Log returns require positive endpoints. Dollar changes retain WTI's negative-price episode without clipping or inventing a denominator. They have different units and are not futures-strategy returns; a later risk comparison needs explicit instrument and exposure conventions.

2. **How did you prevent leakage, and what leakage remains possible?**
   I fixed chronological development/tuning/test periods, filtered held-out values before numeric analysis, used weather availability cutoffs before both target endpoints, and built anomaly references from eligible earlier years. Mutation tests verify these boundaries. Revised historical values and assumed release times still prevent a point-in-time claim.

3. **Does a nonzero weather correlation justify adding weather to the forecast model?**
   It motivates a hypothesis, not a performance claim. These 70 exploratory comparisons share seasonality and use pointwise intervals. After the Phase 3 gate, incremental value would need chronological comparisons against the same non-weather baseline, tuning confined to the tuning protocol, and an untouched final evaluation. Reproducible EDA proves the computation can be repeated, not that weather improves prediction.

## Stop before Phase 3

Phase 2 tests, reproducibility checks, plot review, and documentation are complete. The next sensible step is a decision on the unresolved target, calendar, coverage, and vintage requirements. Phase 3 has not started.
