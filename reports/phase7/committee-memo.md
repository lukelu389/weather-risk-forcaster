# Research committee memo

Weather-aware energy and commodity risk | 2024-2025 final evaluation

## Decision requested

Retain the project as a reproducible research pilot. Do not approve production risk limits or trading deployment. The final sample reveals calibration weaknesses, and the evidence does not establish a general weather-feature advantage.

## Experiment

The 2018-2025 snapshot supports development in 2018-2021, tuning in 2022-2023 and a locally sealed final evaluation in 2024-2025. Nine fixed models include historical simulation, GARCH/GJR, matched seasonal/weather regressions, and matched GRU/diagonal state-space variants. Monthly rolling fits use matured labels only; earlier final-period labels can enter later fits after their assumed release time.

## Evidence

There are 477 shared scored dates for gas, 473 for WTI and 502 for the CAD equity basket. The final run contains 27,882 forecast rows without failed or skipped forecasts. Separate Holm families reject 2 exact-coverage, 13 independence and 19 conditional-coverage hypotheses. Counts overlap and are not distinct failed-model counts. Sparse tails and asymptotic references limit interpretation.

## Weather result

Matched weather effects vary by asset and architecture. All three basket weather variants have higher final quantile-loss point estimates at both confidence levels. Gas GRU weather improves the point estimate, with pointwise intervals spanning zero. A richer architecture or an additional feature does not by itself establish calibrated tail risk.

## Disclosed evaluation correction

The primary 1,000-draw bootstrap could not attain p-values small enough for its planned Holm families. Its non-rejections are not evidence of equivalence or adequate ES. A separate post-exposure sensitivity repeats every hypothesis with 19,999 draws and the same forecasts, blocks and seeds. It yields 1 adjusted score-comparison rejections and 3 adjusted ES-residual rejections at 5%. This correction is supplementary, not a new untouched test.

## Portfolio interpretation

A separate sensitivity resets CAD 1 million independently at each interval start, equally split among ENB.TO, SU.TO, FTS.TO and EMA.TO. Across the nine models, full-period mean gross 99% ES ranges from CAD 20,122 to CAD 25,535. These are model-dependent risk forecasts, not a capital recommendation or a wealth path.

## Currency, execution and costs

The basket is CAD-denominated and uses adjusted-price return proxies. Monetary loss is N times (1 - exp(R/100)). Gaussian ES uses the exact transformed tail expectation; empirical ES transforms the entire matured sample before tail averaging. Fixed total interval budgets of 0/5/10 bp add CAD 0/500/1,000 to loss, VaR and ES, preserving exceedance classifications. They are scenarios, not measured spreads or turnover. USD spot benchmarks are not added to this book; no futures roll or FX hedge is inferred.

## Material limitations

The data are latest-vintage with assumed lags, not verified historical information sets. The quote grid is retrospective, horizons vary, and the fixed equity basket has survivor-selection risk. Missing realizations remain gaps. Normal tails, moment-matched ensembles and sparse 99% events limit model adequacy. Calendar-year slices are descriptive; the pilot does not cover 2008 or 2014-2016.

## Controls and next decision

Two complete final evaluations matched 617 artifacts, including 600 checkpoints and the plots. The local seal and first-access record preserve the boundary; they are not external preregistration. Preserve these records and the input snapshot. Before any deployment study, obtain verified vintages, prospective calendars and executable instrument/cost data, and reserve a new prospective evaluation period. Any new architecture or feature must not reuse this period as untouched.

Sources: reports/phase5, reports/phase5-sensitivity, reports/phase6; docs/phase-5-seal.json and phase-5-first-access.json. All figures are from the preserved local experiment.
