# Phase 5 frozen final-evaluation protocol

This protocol is written before numerical access to the 2024–2025 final period. The local seal records the protocol, model configuration, implementation, dependency lock, input hash, and UTC freeze time. A separate exclusive-create audit records the first authorized final-period access attempt, including failures. These are local provenance records, not an external preregistration or tamper-proof timestamps. A changed sealed file prevents replay; corrections after exposure must be disclosed and must not be represented as an untouched test.

## Fixed experiment

Evaluate all nine Phase 3/4 models on 2024-01-01 through 2025-12-31, retaining every model regardless of tuning results. Do not choose a winning seed, architecture, feature, epoch, confidence level, window, or threshold from final results. Model settings are exactly those in `config/ml.json` and `config/baselines.json`: original-grid rolling window 756, minimum 252 matured labels, monthly origin-month refits, historical simulation updated each origin, risk levels 95% and 99%, and the same targets, origins, controls, weather variables, two seeds and neural training budget. Only the forecast interval changes. Earlier final-period labels can enter a later refit once their recorded availability passes that origin; this is a rolling deployment simulation, not a single frozen fit.

Retain the latest-vintage/assumed-lag qualification and the retrospective quote grid. WTI is a USD/barrel difference; gas and the Canadian basket are percentage log returns. Scores are compared within each asset. The sample cannot establish point-in-time trading performance or coverage of 2008/2014–2016. No new data sources or model settings are introduced.

## Samples and failures

Save every forecast, status, target invalidity, fitted parameter, preprocessing record and checkpoint. A run with failed forecasts is not passed. Build one chronological grid per asset from all forecast target dates. For each confidence, score only dates with finite realized targets and valid forecasts from all nine models. Retain excluded dates as missing positions for dependence tests and block resampling; never bridge missing rows. Report total, common and excluded counts and reasons. Missing prices are not filled. All models must agree on the realized target and interval on a common date.

## Prespecified diagnostics

Loss is minus the target. An exceedance is strictly `loss > VaR`; equality is a nonexceedance. Rounded/discrete targets can affect nominal coverage, and no random tie-breaking is used.

- VaR: exceedance count/rate, nominal count, exact two-sided binomial p-value and 95% Clopper–Pearson interval; Kupiec unconditional-coverage likelihood ratio with chi-square(1) reference.
- Dependence: Christoffersen first-order transition likelihood ratio using only pairs adjacent on the original grid with both observations valid. If either prior-state transition row is unobserved, independence is unidentified and reported unavailable. Conditional coverage is the conventional sum of unconditional and independence LR statistics with chi-square(2) reference. The unconditional statistic includes all scored rows; the transition statistic excludes gaps and initial segment observations. With missingness and sparse tails these are approximate diagnostics, not exact size guarantees.
- ES: report the mean relative identification residual `[q + I(loss>q)*(loss-q)/(1-alpha) - e] / e`, where `q` is VaR and `e` ES. This is a joint VaR/ES calibration diagnostic, not a stand-alone ES validation. Positive ES is required; otherwise the ES score/diagnostic is explicitly unavailable without clamping.
- Proper forecast losses: quantile (pinball) loss, QLIKE `log(m2)+target²/m2`, and upper-loss FZ0 `I(loss>q)*(loss-q)/((1-alpha)*e)+q/e+log(e)-1` for positive ES. Lower is better within an asset. QLIKE uses a noisy squared-target proxy. FZ0 jointly evaluates VaR and ES. Positivity-domain failures are disclosed with score counts; comparisons use the intersection of finite scores for the paired models on the common grid.

For ES residuals and paired score differences use a noncircular moving-block bootstrap: overlapping candidate blocks of 20 original rows, sample blocks with replacement and truncate to the original grid length. Keep NaNs inside blocks. Use 1,000 replications, 95% percentile intervals, base seed 1729 with stable hashed per-test keys, and at least 100 finite observations. At least 95% of bootstrap replications must have a finite mean. The centered two-sided p-value is `(1 + count(abs(bootstrap_mean - sample_mean) >= abs(sample_mean))) / (valid_replicates + 1)`. These are approximate stationary-block uncertainty estimates; no bandwidth search or alternate-bootstrap selection follows results.

Compare each of the eight nonhistorical models against historical simulation, and compare weather versus control within the variance regression, GRU, and SSM. Save signed candidate-minus-reference loss differences: a negative value favors the candidate. Quantile and FZ0 comparisons use both risk levels; QLIKE appears once at 95% because it is identical across levels. This gives 165 prespecified comparisons (11 pairs × 3 assets × 5 score/level combinations). Apply Holm adjustment across this full comparison family. Apply Holm separately across the 54 tests in each of the exact-coverage, independence, conditional-coverage and ES-residual diagnostic families; unavailable p-values count as 1 in family size and remain unavailable in output. Kupiec raw p-values are supplementary. Use 5% as the fixed diagnostic threshold. Always report raw effects, intervals and sample/tail counts; lack of rejection is not evidence of calibration.

## Outputs and verification

Save the full design panel, forecasts, all model fits/checkpoints, score and calibration tables, paired comparisons, source snapshot, sealed protocol and first-access audit, dependency lock, deterministic plots, report and hashes. Repeat the full run twice in the installed environment and compare every artifact plus manifests. Review the plots. Verify tests on synthetic data before first access, then inspect empirical invariants and saved-model restoration. Reproduction uses the same sealed experiment; it is not a fresh holdout.

Do not optimize a portfolio, make a trading recommendation, or alter forecasts from these final scores. Later phases may communicate these findings and define a separately labelled portfolio sensitivity exercise subject to the existing currency, instrument, execution and history limitations.

## Sources

[Christoffersen (1998), Evaluating Interval Forecasts](https://users.ssc.wisc.edu/~bhansen/718/Christoffersen1998.pdf); [Fissler, Ziegel and Gneiting, joint VaR/ES evaluation](https://arxiv.org/abs/1507.00244); [SciPy binomial test and exact interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). Formula signs above use upper-tail positive losses.
