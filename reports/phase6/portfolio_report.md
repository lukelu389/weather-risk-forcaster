# Phase 6 fixed-notional CAD basket sensitivity

Each interval starts independently with CAD 1,000,000, allocated 25% to each of the four existing Canadian equity constituents. This is an adjusted-price basket proxy using latest-vintage data and assumed availability. It is not an executable strategy, compounded NAV, CAGR, or portfolio optimization.

All nine fixed Phase 5 models and both risk levels are retained. Gas and WTI are not aggregated into the CAD basket. No FX hedge, futures roll, financing, turnover or realized transaction-cost estimate is inferred.

A log-percent basket return R becomes gross CAD loss N × (1 − exp(R/100)). Gaussian log-return forecasts use the exact transformed normal quantile and tail expectation. Historical simulation rebuilds the original matured rolling sample and transforms each observation before calculating monetary ES; transforming the previous log-return ES would be incorrect.

## Full-period gross sensitivity

| model | confidence | scored_rows | mean_var_cad | mean_es_cad | worst_realized_loss_cad | exceedance_rate |
|---|---|---|---|---|---|---|
| garch | 0.950 | 502 | 13176.265 | 16490.903 | 39679.050 | 0.026 |
| garch | 0.990 | 502 | 18583.634 | 21258.055 | 39679.050 | 0.004 |
| gjr_garch | 0.950 | 502 | 12682.896 | 15873.653 | 39679.050 | 0.032 |
| gjr_garch | 0.990 | 502 | 17888.159 | 20462.729 | 39679.050 | 0.006 |
| gru_control | 0.950 | 502 | 12469.108 | 15607.485 | 39679.050 | 0.026 |
| gru_control | 0.990 | 502 | 17588.859 | 20121.745 | 39679.050 | 0.006 |
| gru_weather | 0.950 | 502 | 12901.976 | 16147.421 | 39679.050 | 0.028 |
| gru_weather | 0.990 | 502 | 18196.474 | 20814.987 | 39679.050 | 0.010 |
| historical | 0.950 | 502 | 14103.658 | 18567.565 | 39679.050 | 0.022 |
| historical | 0.990 | 502 | 21285.250 | 25535.014 | 39679.050 | 0.002 |
| seasonal_variance | 0.950 | 502 | 13567.339 | 16979.452 | 39679.050 | 0.026 |
| seasonal_variance | 0.990 | 502 | 19133.765 | 21886.455 | 39679.050 | 0.008 |
| ssm_control | 0.950 | 502 | 12920.329 | 16171.044 | 39679.050 | 0.026 |
| ssm_control | 0.990 | 502 | 18223.397 | 20846.434 | 39679.050 | 0.008 |
| ssm_weather | 0.950 | 502 | 13371.017 | 16733.069 | 39679.050 | 0.026 |
| ssm_weather | 0.990 | 502 | 18855.805 | 21567.827 | 39679.050 | 0.010 |
| weather_variance | 0.950 | 502 | 13854.611 | 17337.445 | 39679.050 | 0.024 |
| weather_variance | 0.990 | 502 | 19536.477 | 22345.588 | 39679.050 | 0.010 |

## Fixed cost budgets and calendar splits

The 0, 5 and 10 basis-point scenarios add CAD 0, 500 and 1,000 per interval to both realized loss and forecast VaR/ES. Each is the entire fixed interval cost budget, not a per-side or turnover estimate. Exceedance classifications therefore remain unchanged. summary.csv reports every model, confidence and cost for the full period and the prespecified calendar years 2024 and 2025. These calendar slices are descriptive, not market regime classifications or new significance tests.

Saved sensitivity rows: 27162; original basket forecast rows: 9054.

Missing realized intervals and skipped forecasts remain explicit; no interpolation or annualization is performed. Mean risks use finite risk forecasts, worst realized loss uses finite realizations, and exceedance rates require both. The fixed survivor basket, adjusted-price conventions, retrospective quote grid and variable interval lengths remain limitations. No 2008 or 2014–2016 coverage, trading recommendation, capital approval or model selection is claimed.

The source Phase 5 manifest and all consumed artifact hashes are verified before numeric parsing. Saved configuration, source snapshot and hashes support reproduction of this sensitivity calculation.
