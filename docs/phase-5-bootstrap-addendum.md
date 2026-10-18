# Post-exposure bootstrap resolution correction

This addendum was written after the sealed 2024–2025 evaluation completed and its scores were inspected. It is a disclosed computational sensitivity check, not a new untouched holdout or a replacement for the primary protocol. All original files, predictions, models, settings, hypotheses and results remain preserved.

## Issue

The frozen protocol used 1,000 resamples and the finite-sample correction `(1 + exceedance_count)/(B + 1)`. Its smallest attainable p-value is 1/1001. Holm's first rejection threshold is 0.05/165 for the paired-score family and 0.05/54 for the ES-residual family. Neither family can reject even its smallest p-value. Therefore zero rejections in those primary families is mechanically constrained and must not be interpreted as evidence of equivalent forecast skill or adequate ES calibration. Exact coverage and transition likelihood-ratio diagnostics do not share this bootstrap resolution issue. Pointwise intervals remain pointwise and cannot be interpreted as familywise intervals.

## Fixed correction

Repeat all bootstrap computations with **19,999 resamples**, retaining the base seed, hashed stream keys, 20-row blocks, NaN positions, minimum observations, confidence intervals, centering, two-sided alternative, all 54 ES tests and all 165 paired comparisons. The minimum p-value is now 1/20000, below both first-step thresholds. No model, asset, period, cost rule, weather variant or comparison is selected or removed in response to its score.

Read the already saved, checksummed forecasts; do not refit models. Saved CSVs use 12 significant digits, so recomputed arithmetic may differ from the original in-memory results by rounding. Require identical sample/tail counts and agreement of means and nonbootstrap diagnostics within a documented small numerical tolerance. Keep the sensitivity in separate run and report directories, with the original manifest and seal hashes, this addendum, and its configuration. Reproduce the sensitivity twice. Its p-values are approximate and retain the stationary-block, sparse-tail, latest-vintage and post-exposure limitations.

The scientific conclusion must consider effect sizes, pointwise intervals, calibration failures and this correction together. Non-rejection still cannot establish equivalence or model approval.
