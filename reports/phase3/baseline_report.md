# Phase 3 baseline pilot

**2022–2023 tuning forecasts only. The 2024–2025 final test remains unread.**

Latest-vintage research on a retrospective observed quote grid with assumed availability. No executable strategy, official market calendar, or causal/predictive weather claim.

Targets and origins are inherited from Phase 2. Each training label requires both endpoint prices (all four constituents for the basket) to have matured. Parameters refit monthly on up to 756 original grid rows, requiring 252 finite labels. Historical simulation updates at every origin. All failures and skips remain in forecasts.csv.

GARCH(1,1) and GJR-GARCH(1,1) use zero-mean Gaussian quasi-likelihood. Missing/unreleased shocks propagate expected variance; the resulting likelihood is an approximation when shocks are missing. Normal VaR/ES at that variance are moment-matched approximations when intervening shocks are unobserved, not exact multi-step mixture quantiles.

Both ridge log-variance regressions use past-only recent squared movements and annual sine/cosine. The weather variant adds Toronto/Calgary HDD18 and CDD18. Medians, missing indicators, scaling, and retransformation are fitted within each training window. Fixed ridge=1; no score-driven selection. Recent history means the last 21 finite matured labels, potentially spanning more than 21 grid rows. Refit months follow the origin, so the first January target can use a December fit.

Loss = −target. VaR is the upper loss quantile; empirical ES integrates exactly the upper tail mass. Gaussian ES uses the analytic normal tail mean. Negative risk numbers are not clamped. Units are percentage log returns or USD/barrel changes, not currency portfolio P&L.

## Descriptive tuning scores

Scores use common valid dates across all five models for each asset and confidence. Lower quantile loss and QLIKE are better within an asset. QLIKE uses the forecast second moment and squared target as a noisy proxy; its absolute values are not comparable across units. No formal calibration tests, significance ranking, or final model selection is performed in Phase 3.

| asset | model | confidence | common_scored_rows | mean_quantile_loss | mean_qlike |
|---|---|---|---|---|---|
| canadian_basket | historical | 0.950 | 498 | 0.106 | 1.131 |
| canadian_basket | garch | 0.950 | 498 | 0.109 | 0.926 |
| canadian_basket | gjr_garch | 0.950 | 498 | 0.109 | 0.881 |
| canadian_basket | seasonal_variance | 0.950 | 498 | 0.110 | 0.942 |
| canadian_basket | weather_variance | 0.950 | 498 | 0.111 | 0.990 |
| canadian_basket | historical | 0.990 | 498 | 0.040 | 1.131 |
| canadian_basket | garch | 0.990 | 498 | 0.033 | 0.926 |
| canadian_basket | gjr_garch | 0.990 | 498 | 0.031 | 0.881 |
| canadian_basket | seasonal_variance | 0.990 | 498 | 0.031 | 0.942 |
| canadian_basket | weather_variance | 0.990 | 498 | 0.032 | 0.990 |
| henry_hub_spot | historical | 0.950 | 478 | 0.838 | 4.919 |
| henry_hub_spot | garch | 0.950 | 478 | 0.833 | 4.887 |
| henry_hub_spot | gjr_garch | 0.950 | 478 | 0.834 | 4.889 |
| henry_hub_spot | seasonal_variance | 0.950 | 478 | 0.846 | 4.950 |
| henry_hub_spot | weather_variance | 0.950 | 478 | 0.817 | 4.890 |
| henry_hub_spot | historical | 0.990 | 478 | 0.232 | 4.919 |
| henry_hub_spot | garch | 0.990 | 478 | 0.261 | 4.887 |
| henry_hub_spot | gjr_garch | 0.990 | 478 | 0.261 | 4.889 |
| henry_hub_spot | seasonal_variance | 0.990 | 478 | 0.229 | 4.950 |
| henry_hub_spot | weather_variance | 0.990 | 478 | 0.246 | 4.890 |
| wti_spot | historical | 0.950 | 478 | 0.321 | 2.743 |
| wti_spot | garch | 0.950 | 478 | 0.260 | 2.632 |
| wti_spot | gjr_garch | 0.950 | 478 | 0.275 | 2.771 |
| wti_spot | seasonal_variance | 0.950 | 478 | 0.271 | 2.625 |
| wti_spot | weather_variance | 0.950 | 478 | 0.270 | 2.617 |
| wti_spot | historical | 0.990 | 478 | 0.110 | 2.743 |
| wti_spot | garch | 0.990 | 478 | 0.087 | 2.632 |
| wti_spot | gjr_garch | 0.990 | 478 | 0.097 | 2.771 |
| wti_spot | seasonal_variance | 0.990 | 478 | 0.087 | 2.625 |
| wti_spot | weather_variance | 0.990 | 478 | 0.088 | 2.617 |

Fit attempts: 300; failed forecast rows: 0; skipped forecast rows: 0. GARCH/GJR fits near the 0.999 persistence cap: 112/150. Convergence is not evidence of model adequacy; boundary estimates warrant caution.

## Next gate

Stop before Phase 4. Retain the final holdout. Broader regime coverage, verified release/vintage history, a prospective market calendar, investable targets, and formal risk evaluation remain separate work.

## Method references

[GARCH/GJR recurrence](https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.GARCH.html); [conditional variance forecasting](https://arch.readthedocs.io/en/stable/univariate/forecasting.html); [SciPy constrained optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
