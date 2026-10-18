# Phase 4 sequence and state-space pilot

**2022–2023 tuning only. Final-test 2024–2025 values remain unread.**

One small GRU and one stable diagonal state-space model each have matched weather/control variants. The state-space model is a simple learned diagonal recurrence, not Mamba. All use the same historical volatility/seasonality controls. Weather adds the same four HDD/CDD features as Phase 3.

Input histories contain 21 original-grid feature snapshots ending at the origin. Missing target rows remain sequence steps. Each historical snapshot was constructed at its own origin; context can extend before the rolling fit-label window. Weather ablation masks values and missingness before preprocessing.

Both architectures train for 40 full-batch Adam updates at learning rate 0.01, from fixed seeds [1729, 2718], with monthly refits. No early stopping, best-seed selection, or test-period tuning is used. Parameter counts and loss trajectories are saved per member; identical epoch budgets do not imply identical compute or convergence.

All fits use matured labels and training-only imputation, scaling and target normalization. CPU float64, one thread and deterministic PyTorch operations support same-environment repeatability. Checkpoints contain tensor state dictionaries and are hashed in the manifest.

The two seed variances are averaged into a zero-mean Gaussian forecast. VaR/ES use this moment-matched normal, not the exact Gaussian-mixture tails. All latest-vintage, calendar, geography, survivor-basket and tail assumptions remain.

## Common-date tuning comparison

All nine models share the same scored dates within each asset/confidence. Lower pinball loss and QLIKE are better within an asset; no significance test or final model selection is implied.

| asset | model | confidence | common_scored_rows | mean_quantile_loss | mean_qlike |
|---|---|---|---|---|---|
| canadian_basket | historical | 0.950 | 498 | 0.106 | 1.131 |
| canadian_basket | garch | 0.950 | 498 | 0.109 | 0.926 |
| canadian_basket | gjr_garch | 0.950 | 498 | 0.109 | 0.881 |
| canadian_basket | seasonal_variance | 0.950 | 498 | 0.110 | 0.942 |
| canadian_basket | weather_variance | 0.950 | 498 | 0.111 | 0.990 |
| canadian_basket | gru_control | 0.950 | 498 | 0.117 | 1.023 |
| canadian_basket | gru_weather | 0.950 | 498 | 0.115 | 1.052 |
| canadian_basket | ssm_control | 0.950 | 498 | 0.118 | 1.032 |
| canadian_basket | ssm_weather | 0.950 | 498 | 0.121 | 1.064 |
| canadian_basket | historical | 0.990 | 498 | 0.040 | 1.131 |
| canadian_basket | garch | 0.990 | 498 | 0.033 | 0.926 |
| canadian_basket | gjr_garch | 0.990 | 498 | 0.031 | 0.881 |
| canadian_basket | seasonal_variance | 0.990 | 498 | 0.031 | 0.942 |
| canadian_basket | weather_variance | 0.990 | 498 | 0.032 | 0.990 |
| canadian_basket | gru_control | 0.990 | 498 | 0.036 | 1.023 |
| canadian_basket | gru_weather | 0.990 | 498 | 0.037 | 1.052 |
| canadian_basket | ssm_control | 0.990 | 498 | 0.035 | 1.032 |
| canadian_basket | ssm_weather | 0.990 | 498 | 0.038 | 1.064 |
| henry_hub_spot | historical | 0.950 | 478 | 0.838 | 4.919 |
| henry_hub_spot | garch | 0.950 | 478 | 0.833 | 4.887 |
| henry_hub_spot | gjr_garch | 0.950 | 478 | 0.834 | 4.889 |
| henry_hub_spot | seasonal_variance | 0.950 | 478 | 0.846 | 4.950 |
| henry_hub_spot | weather_variance | 0.950 | 478 | 0.817 | 4.890 |
| henry_hub_spot | gru_control | 0.950 | 478 | 0.852 | 4.914 |
| henry_hub_spot | gru_weather | 0.950 | 478 | 0.784 | 4.925 |
| henry_hub_spot | ssm_control | 0.950 | 478 | 0.834 | 4.879 |
| henry_hub_spot | ssm_weather | 0.950 | 478 | 1.386 | 5.477 |
| henry_hub_spot | historical | 0.990 | 478 | 0.232 | 4.919 |
| henry_hub_spot | garch | 0.990 | 478 | 0.261 | 4.887 |
| henry_hub_spot | gjr_garch | 0.990 | 478 | 0.261 | 4.889 |
| henry_hub_spot | seasonal_variance | 0.990 | 478 | 0.229 | 4.950 |
| henry_hub_spot | weather_variance | 0.990 | 478 | 0.246 | 4.890 |
| henry_hub_spot | gru_control | 0.990 | 478 | 0.275 | 4.914 |
| henry_hub_spot | gru_weather | 0.990 | 478 | 0.258 | 4.925 |
| henry_hub_spot | ssm_control | 0.990 | 478 | 0.283 | 4.879 |
| henry_hub_spot | ssm_weather | 0.990 | 478 | 0.387 | 5.477 |
| wti_spot | historical | 0.950 | 478 | 0.321 | 2.743 |
| wti_spot | garch | 0.950 | 478 | 0.260 | 2.632 |
| wti_spot | gjr_garch | 0.950 | 478 | 0.275 | 2.771 |
| wti_spot | seasonal_variance | 0.950 | 478 | 0.271 | 2.625 |
| wti_spot | weather_variance | 0.950 | 478 | 0.270 | 2.617 |
| wti_spot | gru_control | 0.950 | 478 | 0.316 | 2.885 |
| wti_spot | gru_weather | 0.950 | 478 | 0.320 | 2.990 |
| wti_spot | ssm_control | 0.950 | 478 | 0.307 | 2.764 |
| wti_spot | ssm_weather | 0.950 | 478 | 0.315 | 2.874 |
| wti_spot | historical | 0.990 | 478 | 0.110 | 2.743 |
| wti_spot | garch | 0.990 | 478 | 0.087 | 2.632 |
| wti_spot | gjr_garch | 0.990 | 478 | 0.097 | 2.771 |
| wti_spot | seasonal_variance | 0.990 | 478 | 0.087 | 2.625 |
| wti_spot | weather_variance | 0.990 | 478 | 0.088 | 2.617 |
| wti_spot | gru_control | 0.990 | 478 | 0.114 | 2.885 |
| wti_spot | gru_weather | 0.990 | 478 | 0.128 | 2.990 |
| wti_spot | ssm_control | 0.990 | 478 | 0.098 | 2.764 |
| wti_spot | ssm_weather | 0.990 | 478 | 0.110 | 2.874 |

Ensemble fits: 300; constituent fits: 600; failed forecast rows: 0.

## Next phase

Phase 5 must freeze model/evaluation settings before reading the final period and report calibration, uncertainty, and forecast-comparison limitations. Keep a separate audit of the first final-test evaluation.

## Implementation references

[PyTorch GRU](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html), [Gaussian likelihood](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GaussianNLLLoss.html), [determinism limits](https://docs.pytorch.org/docs/2.10/notes/randomness.html).
