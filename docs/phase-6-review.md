# Phase 6 review and interview preparation

## Outcome

Phase 6 translates the preserved basket forecasts into an explicit CAD sensitivity. Each interval starts independently with **CAD 1,000,000**, allocated 25% to each of ENB.TO, SU.TO, FTS.TO and EMA.TO. All nine fixed models and both confidence levels are retained. The calculation does not refit models, select a winner, compound a wealth path or optimize weights.

Two complete runs matched **six artifacts and their manifests** byte-for-byte. They produced **27,162 sensitivity rows** from **9,054 original basket forecast rows**, with **162 summary rows** across models, confidence levels, costs and calendar periods. There are **502 scored intervals** per full-period model/confidence combination, no skipped sensitivities, and 54 rows with missing realized losses. Those 54 rows represent one original missing realization repeated across nine models, two confidence levels and three cost budgets; they are not 54 distinct dates. The expanded full suite passed **230 tests with no skips**.

The [verification record](phase-6-verification.json), [portfolio report](../reports/phase6/portfolio_report.md), [complete summary](../reports/phase6/summary.csv) and [fixed configuration](../config/portfolio.json) preserve the evidence. Reproduce with `make portfolio` in the locked research environment and retain the pinned Phase 5 run. A fresh checkout without the ignored source run cannot reproduce the empirical calculation.

## Method and findings

For a log-percent return R and starting notional N, gross monetary loss is `N × (1 − exp(R / 100))`. Gaussian forecasts use the exact transformed normal quantile and tail expectation. Historical simulation reconstructs the original matured rolling sample and transforms every sample return before calculating empirical monetary VaR and fractional-tail ES. Applying the nonlinear transformation to the previous log-return ES would give the wrong expected monetary loss.

Across all nine models, full-period mean gross 99% ES ranges from **CAD 20,121.75 to CAD 25,535.01**. The worst observed interval loss is **CAD 39,679.05**. A realized loss exceeding a mean ES is not by itself a calibration test; the mean averages changing conditional forecasts, and ES is a tail expectation rather than a maximum-loss bound. The Phase 5 calibration findings remain material.

Fixed total interval budgets of **0, 5 and 10 basis points** add **CAD 0, 500 and 1,000** to both realized loss and forecast VaR/ES. The saved checks confirm that exceedance classifications remain unchanged. These are declared sensitivities, not estimates of spreads, market impact, turnover, financing or per-side execution costs. Calendar-year summaries for 2024 and 2025 are descriptive slices and add no new significance tests.

The runner checks the pinned Phase 5 manifest and consumed artifact hashes before parsing their contents. It validates the common model/date grid, target and maturity agreement, and reconstructed historical sample. Missing realizations remain explicit; risk averages, realized-loss maxima and exceedance rates each use their stated finite-value requirements.

## Scope and phase gate

The basket is CAD-denominated, so this calculation requires no inferred currency conversion. USD gas and WTI spot benchmarks are excluded from the monetary book: spot changes do not establish investable futures P&L, contract sizing, roll costs or an FX hedge. The fixed survivor basket, adjusted-price proxies, latest-vintage data with assumed lags, retrospective quote grid and variable interval lengths remain qualifications. The pilot covers neither 2008 nor 2014–2016.

The calculation and reproduction gate is complete for this qualified pilot. [Phase 7](phase-7-review.md) communicates these preserved results. Production limits, capital approval and a trading strategy remain outside the evidence. The 2024–2025 holdout is consumed; any new feature or model requires prospective evaluation or an explicit exploratory label.

## Three interview questions

1. **Why can VaR be transformed into CAD directly while ES needs a new tail expectation?**
   A monotone return-to-loss transformation maps the relevant quantile to monetary VaR. ES averages outcomes in the tail, and a nonlinear transformation does not commute with that average. We integrate the transformed normal tail or transform the full empirical sample before averaging its upper loss tail.

2. **Why do the cost scenarios change reported risk but leave exceedance counts unchanged?**
   Each scenario adds the same deterministic cost C to realized loss and its risk threshold: `L + C > VaR + C` is equivalent to `L > VaR`. This verifies the declared translation. Random, state-dependent or model-dependent costs could change calibration and require a different design.

3. **What would be needed to turn this sensitivity into an executable portfolio study?**
   Specify a prospective instrument universe, cash flows, currencies, exchange calendars, rebalancing and financing rules; obtain executable price and cost data and verified historical information sets. Then evaluate the fixed strategy on a new prospective period. Independent CAD 1 million intervals and adjusted-price proxies cannot establish compounded performance or deployment readiness.
