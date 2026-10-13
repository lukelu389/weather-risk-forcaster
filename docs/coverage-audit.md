# Pilot data coverage audit

Validated source snapshot: `runs/live-20261013T120935.365011Z`. This is a Phase 1 data-quality audit, not an exploratory/modeling result.

39,760 observation rows across 17 series; no validation errors. All downloaded records are latest-vintage. The default alignment therefore withholds them from strict historical forecasts.

| Series | Rows | Nulls | First observation | Last observation |
|---|---:|---:|---|---|
| calgary_cdd18 | 2,922 | 74 | 2018-01-01 | 2025-12-31 |
| calgary_hdd18 | 2,922 | 74 | 2018-01-01 | 2025-12-31 |
| calgary_precip | 2,922 | 13 | 2018-01-01 | 2025-12-31 |
| calgary_temp | 2,922 | 74 | 2018-01-01 | 2025-12-31 |
| ema_to_adj_close | 2,008 | 0 | 2018-01-02 | 2025-12-31 |
| enb_to_adj_close | 2,008 | 0 | 2018-01-02 | 2025-12-31 |
| fts_to_adj_close | 2,008 | 0 | 2018-01-02 | 2025-12-31 |
| henry_hub_spot | 2,087 | 88 | 2018-01-02 | 2025-12-31 |
| su_to_adj_close | 2,008 | 0 | 2018-01-02 | 2025-12-31 |
| toronto_cdd18 | 2,922 | 39 | 2018-01-01 | 2025-12-31 |
| toronto_hdd18 | 2,922 | 39 | 2018-01-01 | 2025-12-31 |
| toronto_precip | 2,922 | 39 | 2018-01-01 | 2025-12-31 |
| toronto_temp | 2,922 | 39 | 2018-01-01 | 2025-12-31 |
| us_3m_yield | 2,087 | 88 | 2018-01-02 | 2025-12-31 |
| us_cpi | 96 | 1 | 2018-01-31 | 2025-12-31 |
| usd_cad | 1,995 | 0 | 2018-01-02 | 2025-12-31 |
| wti_spot | 2,087 | 88 | 2018-01-02 | 2025-12-31 |

Weather rows include every calendar date, with missing values preserved. Price, yield and FX counts differ because their source calendars differ. Missing weekdays are not automatically exchange trading days. Monthly CPI retains one source null; nothing was imputed.

EIA storage and consumption were skipped because a key is not configured. Their absence is preserved when replaying this snapshot.

This pilot does not cover 2008 or 2014–16. Extending the dates requires older weather-station/FX coverage and explicit source splicing decisions. No full-study or regime-performance claim is supported yet.

The next phase must decide how to treat these gaps using development data only, without revising the test protocol after seeing performance.
