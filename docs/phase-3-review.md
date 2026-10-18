# Phase 3 review and interview preparation

## Scope and decisions

Phase 3 adds historical simulation, GARCH(1,1), GJR-GARCH(1,1), and a weather volatility regression alongside a matched seasonal control. It also adds loss-based 95%/99% VaR and expected shortfall with numerical tests. Forecasts and descriptive scores cover **2022–2023 only**. The **2024–2025 final test is filtered before numeric values or availability timestamps are parsed**; no final-test result or model selection is reported.

The Phase 2 gates are resolved for this pilot by retaining its existing target/origin definitions and explicitly limiting the claims. Gas and adjusted equities use percentage log returns; WTI uses USD/barrel changes, preserving negative prices. The four-name basket rebalances to equal weights before costs. An origin is 18:00 Toronto on the calendar day before the interval's earlier price. Quote intervals over four calendar days or adjacent to null quotes remain unavailable.

The calendar is the retrospective observed quote grid, not a prospective exchange schedule. It can contain explicit null rows and omit absent dates. Interval lengths vary and are not annualized. This is research on benchmark changes and a selected-survivor basket, not portfolio currency P&L or an executable strategy. The existing latest-vintage/assumed-release qualification remains; historical expansion and verified vintages are not claimed complete.

## Implementation and information boundaries

`baselines.py` reuses the existing snapshot validation, target construction and weather alignment. `baseline_models.py` provides numerical model fitting; `risk.py` provides standard-library risk functions. Only SciPy 1.17.1 was added to the analysis environment for constrained optimization. Existing Phase 1/2 code and reports are preserved.

A label becomes eligible at the latest recorded availability among both price endpoints, or all eight constituent endpoints for the basket. Training checks each timestamp separately, including nonmonotone release times; it does not assume every row before the last eligible date is available. Every historical feature is constructed at that row's original origin. Mutating an unreleased label cannot change an earlier forecast.

Monthly fits use up to 756 original grid rows ending at the latest matured finite label and require at least 252 finite labels. The window retains internal missing positions. Model parameters, variance initialization, imputation medians, scaling and regression retransformation remain frozen until the next origin month. States incorporate newly matured labels at subsequent origins. Origin-month scheduling produces 25 fit months here because an early January target has a late December origin. Historical simulation updates every origin using the same matured-window rule.

Forecasts are issued when origin/history are valid even if the eventual target is missing; realized validity is not a forecast-time filter. All statuses remain in the forecast file. Scores exclude missing realizations and use the intersection of valid forecasts across all five models for each asset/confidence.

## Model definitions

- **Historical simulation:** the empirical distribution of matured targets in the rolling grid window. VaR uses the inverted empirical loss CDF. ES integrates exactly the worst 5%/1% probability mass, using a fractional boundary observation when needed. Negative VaR/ES values are allowed.
- **GARCH/GJR:** zero-mean Gaussian quasi-likelihood with nonnegative coefficients and persistence `alpha + beta + gamma/2 <= 0.999`. GARCH has `gamma=0`; GJR adds a negative-shock term. Three deterministic SLSQP starts are compared only when converged. Scale and initial variance come from training history. Missing shocks advance expected variance without a likelihood contribution; this is an approximation to likelihood with latent shocks. There is no silent model substitution on failure.
- **Matched variance regressions:** ridge regression of `log(y² + floor)` with an unpenalized intercept and fixed penalty 1. Both use log mean-square of the last 21 finite matured labels, plus annual sine/cosine at the origin. These 21 labels can span more than 21 grid rows. Weather adds Toronto/Calgary HDD18 and CDD18. Training medians fill missing features and explicit missing indicators accompany them. An entirely missing training column uses zero with an indicator. Scaling, the floor (`max(training mean-square × 1e-6, smallest positive normal float)`, or `1e-12` for an all-zero sample), and residual smearing use training data only. Predictions remove the additive log floor and retain a positive variance floor. Smearing is a baseline retransformation assumption, not guaranteed conditional calibration.

Loss is `−target`. Gaussian VaR/ES uses zero mean and the predicted variance for GARCH and both regressions. When GARCH must propagate over unknown intervening shocks, a normal distribution at the expected variance is a moment-matched approximation, not the exact mixture distribution. Heavy tails and persistent variance therefore remain material model risks.

## Tuning evidence

The pilot produces **15,400 forecast rows**, counting both confidence levels, and **300 successful model fits** with no failed or skipped forecasts. Common scored dates are **478** each for gas and WTI and **498** for the basket. No parameter was changed based on these scores.

Selected comparisons from [the complete tuning report](../reports/phase3/baseline_report.md):

| Asset | Seasonal / weather 95% quantile loss | Seasonal / weather 99% quantile loss | Seasonal / weather QLIKE |
|---|---:|---:|---:|
| Henry Hub | 0.8459 / 0.8169 | 0.2294 / 0.2460 | 4.9499 / 4.8896 |
| WTI | 0.2710 / 0.2703 | 0.0871 / 0.0881 | 2.6254 / 2.6171 |
| Canadian basket | 0.1097 / 0.1111 | 0.0308 / 0.0318 | 0.9424 / 0.9903 |

Lower scores are better within an asset. Weather helps some tuning scores and worsens others; this does not establish incremental out-of-sample skill. QLIKE is `log(m2) + y²/m2`, using the forecast second moment and a noisy squared-target proxy; absolute scores are not comparable across target units. Expected shortfall is forecast but not formally backtested in this phase.

**112 of 150 GARCH/GJR fits are near the 0.999 persistence cap.** All optimizer starts converge, but repeated boundary estimates warrant caution about stationarity, crisis sensitivity and specification. Convergence is a numerical result, not evidence of model adequacy. No calibration p-values, significance ranking, or final holdout evaluation is included.

## Verification and reproduction

**131 tests pass with no skips** in the full baseline environment. The dependency-free workflow also passes 74 executed tests, skips five optional modules, and completes its synthetic demo.

Install `requirements-baselines.lock.txt`, then run `make test PYTHON=.venv/bin/python` and `make baselines`. The README includes a two-run artifact comparison; [phase-3-verification.json](phase-3-verification.json) records the verified runs, test results, dependency checks, and hashes.

Tests cover empirical tail mass, analytic normal risks, sign/scale/translation invariants, constrained fitting, observed/missing variance recurrences, training-only preprocessing, current-target isolation, delayed availability, basket maturity, monthly caching, failed fits, common-date scoring, holdout poisoning and artifact integrity. Runner integration tests use controlled fit mocks; the separate empirical reruns exercise all actual models.

Two complete runs matched all **eight artifacts and their manifests** byte-for-byte. Saved forecasts also pass timestamp ordering, positive variance, ES ≥ VaR, and 99% ≥ 95% risk checks.

Reproducibility is verified for the preserved input and installed Python 3.14.2/macOS ARM64 environment. Cross-platform identity and fresh provider reconstruction are not claimed. Preserve ignored input/run directories; hashes alone cannot recover revised source data. The Phase 1 snapshot is validated by its artifact hashes, not replayed against the now-expanded source tree.

## Three interview questions

1. **Why is shifting returns by one row insufficient for training eligibility?**
   A target requires both endpoint prices, and an earlier price may be published late. The seven-calendar-day spot lag differs from equity availability. I require the latest endpoint timestamp to precede the origin and test every row individually. A basket requires every constituent endpoint. This addresses timing, while revised historical values still prevent a strict point-in-time claim.

2. **Why retain missing grid positions in GARCH, and can you use a normal tail after several unknown shocks?**
   Dropping a missing row changes the time recursion and incorrectly makes separated observations adjacent. I propagate conditional expected variance through each unknown shock. That preserves grid timing, but the resulting marginal distribution is generally a variance mixture. The normal VaR/ES is an explicit moment approximation; variance recursion alone does not establish accurate tail risk.

3. **How would you tell whether weather adds value rather than repeating seasonality?**
   I compare the same volatility/seasonal regression with and without weather, using the same matured training labels and common scored dates. Fixed preprocessing and parameters never see future labels. The tuning evidence is mixed, so I make no skill claim; formal chronological evaluation, uncertainty, and untouched final-test assessment remain later gates.

## Stop before Phase 4

Phase 3 ends with implemented baselines, tested risk calculations, repeatable tuning forecasts, and documented limitations. Preserve the final holdout. Phase 4 sequence/state-space models and Phase 5 formal risk calibration have not started.
