# Model limitations and research decision

The defensible conclusion is narrow: this pilot does not establish a general benefit from its lagged Toronto/Calgary weather features. It also finds material calibration weaknesses. A reproducible forecast can still be a poor risk forecast.

## Information set and target

- **Revised history:** live adapters use latest-vintage data and assumed release lags. Lagging revised values does not reconstruct what was known historically. Revision-aware extension tests do not retroactively certify the pilot.
- **Retrospective horizons:** targets use the observed quote grid and variable intervals of up to four calendar days. Missing quotes and their endpoints are known retrospectively; this is not a fixed next-session executable forecast contract.
- **Weather scope:** two cities' observed heating/cooling degree days are coarse proxies. The experiment does not test service-territory loads, demand-weighted national weather, forecast revisions, hurricanes or supply outages. Failure here cannot establish that weather has no economic value.
- **Universe and regimes:** four surviving Canadian equities are a selected basket. The primary experiment omits 2008 and 2014–2016. Later history staging failed validation and was not used to extend the model evaluation.

The [original protocol](phase-5-protocol.md) fixes these definitions. Current release fixes improve software validation and reviewability; they do not change the archived empirical findings.

## Tail risk and inference

The parametric models turn estimated variance into zero-mean Gaussian VaR/ES. Variance accuracy does not guarantee tail accuracy. Historical simulation avoids that distributional assumption but has very little effective tail mass: a fully observed 756-row window contains only 7.56 observations' worth of 1% tail probability.

| Final-period example | Observation | Interpretation |
|---|---|---|
| Gas weather SSM, 99% VaR | 19 / 477 exceptions; nominal expectation 4.77 | Material undercoverage in this model/asset case |
| WTI historical simulation, 95% VaR | 6 / 473 exceptions; nominal expectation 23.65 | Excessively conservative coverage also fails calibration |
| 99% VaR sample size | 4.73–5.02 expected exceptions per asset | Weak power and unstable tail estimates even before multiplicity adjustment |

Counts and tests are in [calibration_scores.csv](../reports/phase5/calibration_scores.csv). The primary diagnostics reject 2 exact-coverage, 13 independence and 19 conditional-coverage hypotheses after separate 54-test Holm adjustments. These overlap and are not counts of distinct bad models. Binomial references assume independent exceptions; sparse transition likelihood-ratio references and stationary-block bootstrap inference are approximate. Missing grid positions break transitions rather than joining nonadjacent observations.

The original bootstrap used 1,000 draws and `(1 + tail_count)/(B + 1)`. Its minimum p-value, 1/1001, exceeds both initial Holm thresholds, 0.05/165 and 0.05/54. Zero rejections in those two families were therefore mechanically constrained. The [preserved correction](phase-5-bootstrap-addendum.md) uses 19,999 draws without changing forecasts or hypotheses. It is post-exposure sensitivity, not a fresh holdout. Three adjusted ES-residual rejections and one adjusted score-comparison rejection remain; no matched-weather comparison rejects. The gas GARCH/GJR negative ES residual means indicate overforecasted tail loss under the joint diagnostic. Non-rejection neither proves equivalence nor establishes calibration.

GARCH persistence frequently approaches its imposed cap, including 36 of 150 final GARCH/GJR fits. Convergence therefore needs to be read alongside persistence, exception clustering and tail-residual diagnostics. The next model study should first test a single prespecified heavier-tail challenger, such as filtered historical simulation, under a new prospective protocol; it should not select a challenger on the consumed holdout.

## Economic use and next gate

The CAD analysis resets a CAD 1 million equal-weight basket each interval. Adding a fixed 0/5/10 bp charge to realized loss, VaR and ES leaves exceptions unchanged. This isolates monetary scale and an assumed cost shift; it supplies no turnover estimate, compounded NAV, Sharpe ratio, funding model or investable commodity book. The futures/FX ledger is tested on synthetic accounting examples only.

The [issuer bridge](financial-context.md) explains how weather can enter business diligence. Portfolio return VaR does not estimate issuer cash flow, covenant headroom or enterprise value.

Before a further empirical study: resolve input-quality failures; verify historical releases and instrument calendars; freeze a fixed prospective target and evaluation protocol; retain enough data for meaningful tail inference. GPU scale is not the binding limitation. The consumed 2024–2025 period must remain labelled as such.
