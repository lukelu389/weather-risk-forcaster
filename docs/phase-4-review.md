# Phase 4 review and interview preparation

## Scope and fixed choices

Phase 4 adds a small GRU and a stable diagonal state-space variance model, each with matched control/weather variants. Forecasts and comparisons cover **2022–2023 tuning only**. The loader filters **2024–2025 before parsing numeric values or availability timestamps**. Phase 5 evaluation rules must be frozen before the first final-test evaluation.

The same preserved Phase 1 snapshot, targets, origins and assumed availability lags remain in force. Gas and equities use percentage log returns; WTI uses USD/barrel changes. The basket contains four fixed survivors at equal weights before costs. Forecast origins precede both target endpoints. Training requires every endpoint to have matured, including all eight constituent endpoints for the basket.

These are latest-vintage data with assumed release delays. The retrospective observed quote grid, variable intervals up to four calendar days, limited weather geography and missing older crisis periods restrict the claims. Neither predictive trading performance nor verified historical point-in-time data is established.

## Models and information boundaries

| Choice | GRU | Diagonal state-space model |
|---|---|---|
| Hidden width | 8 | 32 |
| Trainable parameters per member | 585 | 545 |
| Recurrence | Standard GRU | `h[t] = decay * h[t-1] + B*x[t] + b` |
| Variants | Control / weather | Control / weather |

The state-space model learns a diagonal decay strictly between zero and one. It is a simple stable recurrence, not Mamba. Both architectures read the final state into a variance head. Normalized log variance is `12*tanh(raw/12)` to bound numerical extremes; the mean is fixed at zero. The training objective is Gaussian quasi-negative-log-likelihood, with targets divided by their training RMS.

Each sample contains **21 original-grid feature snapshots**, ending at the forecast row. The three controls are lagged log mean-square and annual sine/cosine; weather adds Toronto/Calgary HDD18 and CDD18. A missing target does not remove its sequence step. Historical features use their own original origins, and unavailable or future-origin snapshots remain missing. Context may precede the rolling fit-label window, but it never uses later information.

Training uses up to **756 original grid rows** and requires **252 finite, matured labels**. Parameter refits occur monthly by origin month. An early January target can have a December origin, giving 25 fit months in this pilot. Preprocessing and parameters stay fixed between refits while each forecast receives its current causal feature sequence.

Medians, missing indicators, centering, scaling and target RMS use only the training windows. An entirely missing training feature has median zero. Control variants mask all weather values before preprocessing, removing weather missingness as well. Both variants retain the same input shape, architecture and parameter count.

Each member receives **40 full-batch Adam updates**, learning rate **0.01**, and the final epoch is saved. Seeds **1729 and 2718** are both retained; there is no early stopping, checkpoint search or best-seed selection. Equal epoch budgets do not imply equal compute or convergence. CPU float64, one thread, isolated random seeds and deterministic PyTorch operations support same-environment repeatability.

The two predicted variances are averaged and converted to zero-mean Gaussian 95%/99% VaR and ES. This is a moment-matched normal approximation, not the exact Gaussian-mixture tail. A failed member fails its ensemble; there is no silent baseline substitution.

## Tuning evidence

The runner validates the pinned Phase 3 forecast artifacts and protocol before combining all **nine models**. Scores use common valid dates within each asset/confidence: **478** each for gas and WTI and **498** for the basket. Lower scores are better within an asset; target units prevent comparisons of absolute scores across assets.

Selected 95% quantile-loss results from the [complete report](../reports/phase4/ml_report.md):

| Asset | GRU control / weather | State-space control / weather |
|---|---:|---:|
| Henry Hub | 0.8521 / 0.7842 | 0.8342 / 1.3862 |
| WTI | 0.3158 / 0.3204 | 0.3071 / 0.3148 |
| Canadian basket | 0.1165 / 0.1155 | 0.1183 / 0.1209 |

The gas GRU benefits on quantile loss while its QLIKE worsens slightly. The weather state-space model performs poorly for gas. Both WTI neural weather variants worsen quantile loss and QLIKE. The basket's GRU weather improvement at 95% does not extend to 99% or QLIKE. This is mixed tuning evidence, not a general weather or neural-model advantage. The settings were retained without retuning to these outcomes. Formal calibration, uncertainty and final-test comparison remain Phase 5 work.

## Verification and reproduction

At this gate, **164 tests pass with no skips**. The 33 additional tests cover sequence fitting/restoration, deterministic seeds, training-only transforms, invalid checkpoint/input rejection, causal feature construction, upstream target mutation, weather-value and missingness ablation, monthly caching, failures, nine-model common-date scoring, baseline provenance, holdout poisoning and artifact integrity. Controlled runner tests complement actual empirical fits.

Two full runs produced **608 byte-identical artifacts and identical manifests**, including **600 checkpoints**. They contain **300 ensemble fits**, **12,320 neural forecast rows**, and **27,720 combined forecast rows**, with no failed or skipped forecasts. Every one of the 600 members reduced its training loss; this verifies optimization progress without establishing forecast skill.

All 600 checkpoints were restored to verify 300 ensemble predictions at fit origins. The maximum relative variance difference from the rounded saved design CSV was **6.13e-12**. Forecast checks confirm positive variance, ES at least VaR, and training availability no later than the forecast origin. The final period remained unread during this verification.

Install `requirements-ml.lock.txt`, run `make test PYTHON=.venv/bin/python`, then `make ml`. Preserve both the pinned Phase 1 input run and Phase 3 baseline run. The [README](../README.md#reproduce-phase-4-sequence-models) includes a two-run hash comparison; [phase-4-verification.json](phase-4-verification.json) records the artifacts, test hashes, checkpoint checks and environment. Reproducibility is established for the preserved input and installed Python 3.14.2/macOS ARM64 CPU environment, not across hardware, platforms or fresh provider downloads.

## Three interview questions

1. **How do you isolate weather's contribution in a sequence model?**
   Compare the same architecture, seeds, fit windows, controls and optimization budget with and without weather. Mask weather before preprocessing so its missingness cannot identify weather availability in the control. Score the variants on the same realized dates. The mixed results here justify reporting the ablation without claiming a general benefit.

2. **Why does a chronological split alone not prevent leakage?**
   Historical observations can arrive late, and historical features can accidentally be rebuilt using later information. Every label must mature by the fit origin, and every sequence snapshot must preserve its own information cutoff. Imputation, scaling and target normalization also use only training samples. These controls address timing; they cannot undo revisions in latest-vintage history.

3. **What do identical runs and lower training loss establish?**
   They establish reproducible computation and that the optimizer improved its stated objective. They do not establish out-of-sample calibration or superior tail forecasts. Averaging variances still imposes an approximate normal tail, so Phase 5 needs frozen evaluation rules, VaR/ES diagnostics and uncertainty before a comparative risk claim.

## Next gate

Phase 4 is complete. Freeze the Phase 5 model and evaluation protocol before accessing final-test values, retain all nine fixed model specifications, and audit the first holdout evaluation separately. Preserve the data-provenance and execution limitations in all later reporting.

Implementation references: [PyTorch GRU](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html), [Gaussian likelihood](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GaussianNLLLoss.html), and [determinism limits](https://docs.pytorch.org/docs/2.10/notes/randomness.html).
