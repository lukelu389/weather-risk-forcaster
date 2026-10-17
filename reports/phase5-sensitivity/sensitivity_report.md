# Phase 5 bootstrap resolution sensitivity

**Disclosed post-exposure computational correction; not an untouched confirmatory analysis.**

Post-exposure correction of insufficient p-value resolution under the prespecified Holm families; no forecast refitting or model selection.

All 54 ES-residual tests and all 165 paired-score comparisons are recomputed using 19,999 resamples. Models, forecasts, seeds, block length, missing positions and hypothesis families are preserved. No models were refitted. The primary output remains unchanged.

At B=1,000, the smallest attainable p-value was 1/1001, so neither Holm family could reject at 5%. The new minimum is 1/20000. Original zero-rejection counts therefore cannot establish adequate ES calibration or equivalent skill. Both sets of intervals are pointwise, not familywise.

| Family | Primary 5% Holm rejections | Sensitivity 5% Holm rejections | Unavailable |
|---|---:|---:|---:|
| ES residual | 0 | 3 | 0 |
| Paired scores | 0 | 1 | 0 |

The input forecasts were saved at 12 significant digits. Sample and tail counts match the primary results exactly, and all checked means and nonbootstrap diagnostics agree within 1e-8. See provenance.json for measured differences and hashes. Bootstrap uncertainty remains approximate under dependence, nonstationarity, sparse tails and latest-vintage data. Failure to reject does not establish equivalence or model approval.
