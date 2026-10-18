# Phase 5 review and interview preparation

## Outcome

All nine fixed models were evaluated on 2024–2025 after a local protocol seal and first-access audit. The two complete executions matched **617 artifacts and their manifests**, including 600 trained checkpoints and both PNG/PDF figure pairs. The pre-access suite passed **195 tests**. There were **27,882 forecast rows**, **300 baseline fits**, **300 neural ensemble fits**, and no failed or skipped forecasts. This is an auditable latest-vintage pilot, not a historically executable strategy or regulatory model approval.

The evaluation preserves the same target definitions, availability masks, original grid, 756-row fit window, monthly refits, neural architecture/training budget and fixed seeds. Prior final-period labels enter later fits only after recorded maturity. All nine models remain; no final-result-based model selection or retuning was performed. The local seal is not an external preregistration or tamper-proof timestamp.

## Findings

Common dates are 477 for Henry Hub, 473 for WTI and 502 for the Canadian basket. Compare scores within an asset because the target units differ. The complete [score table](../reports/phase5/calibration_scores.csv) and [paired comparisons](../reports/phase5/paired_comparisons.csv) retain all results.

- Henry Hub weather SSM at 99% has **19 exceedances in 477 dates** (3.98%, versus nominal 1%); the exact-coverage Holm p-value is about **0.000031**. Gaussian moment matching and additional weather features do not ensure tail calibration.
- WTI historical simulation at 95% has **6 exceedances in 473 dates** (1.27%, versus nominal 5%); its exact-coverage Holm p-value is about **0.00156**. Excessively conservative coverage can also be a calibration failure.
- The original diagnostic families reject 2 exact-coverage, 13 independence and 19 conditional-coverage hypotheses at 5% after their separate Holm adjustments. These counts overlap; they are not counts of distinct failed models. Missing rows break transition pairs, and asymptotic transition tests with sparse tails remain approximate.
- Weather effects are mixed. On the basket, weather increases quantile loss for all three matched architectures at both risk levels. Gas GRU weather lowers the point estimate at both levels, but its pointwise intervals include zero. WTI effects are small and mixed. These patterns do not justify a universal weather benefit or a neural-model advantage.

## Bootstrap limitation and disclosed correction

The sealed bootstrap used 1,000 replications. Its smallest possible corrected p-value, 1/1001, exceeds both Holm first-step thresholds for 165 score comparisons and 54 ES tests. Consequently neither family could reject at 5%, regardless of the data. The original zero-rejection counts are **not evidence of equivalence or ES calibration**. This design limitation was identified after exposure; the source, seal and original artifacts were preserved.

The [dated access record](phase-5-first-access.json) and [separate addendum](phase-5-bootstrap-addendum.md) distinguish the primary experiment from a post-exposure numerical sensitivity using 19,999 draws. The correction retains every model, comparison, date, block rule and seed; it does not refit or optimize forecasts. Its output must be labelled supplementary. Sparse 99% tails and stationary-block assumptions remain even with more resamples.

The completed [supplementary analysis](../reports/phase5-sensitivity/sensitivity_report.md) matches **14 artifacts and manifests** across two runs; [verification](phase-5-sensitivity-verification.json) confirms unchanged counts and non-bootstrap statistics within CSV-rounding tolerance. It rejects three ES-residual hypotheses after Holm adjustment: gas GJR at 95% (p=0.0442), and gas GARCH/GJR at 99% (p=0.0027 each). Their negative residual means indicate overforecasted tail losses under this joint diagnostic. Of 165 score comparisons, basket GARCH versus historical simulation on QLIKE is the only adjusted rejection (mean difference −0.05447, pointwise 95% interval [−0.08388, −0.03184], adjusted p=0.02475). No matched weather comparison rejects after adjustment. These supplementary results do not establish equivalence, reliable calibration, or a fresh confirmatory discovery.

## Plot review

Both PNGs were visually inspected. Labels, nominal rates, risk levels, model identities, units, error bars and captions are legible, with no overlapping or clipped content. PDF bytes matched across reproductions; the PDF plots were not separately visually inspected.

The coverage figure makes both undercoverage and overcoverage visible. Its exact binomial intervals assume independent exceedances and are not multiplicity-adjusted. The weather figure consistently displays candidate-minus-control quantile loss, with negative values favoring weather. Its 95% block intervals are pointwise; a visually excluded zero does not imply a Holm-adjusted discovery. Axis scales differ across assets and panels, so the line lengths should not be compared as absolute economic effects.

## Three interview questions

1. **Why can correct average VaR coverage still fail a backtest?**
   Exceedances can cluster even when their overall count is near nominal. Coverage, transition dependence and conditional coverage address different aspects. We retain missing grid positions so gaps cannot create artificial adjacent exceedances. Sparse tails limit the power and reference accuracy of these diagnostics.

2. **Why evaluate ES jointly with VaR, and what does a zero mean residual show?**
   The ES identification residual uses VaR to define the tail and is interpreted jointly with quantile calibration. FZ0 is a joint VaR/ES score on the positive-ES domain. A non-rejection is not proof of calibration, particularly with few tail observations. Our original bootstrap resolution also prevented familywise rejection, which is disclosed rather than hidden.

3. **How do you correct a methodological error discovered after opening the holdout?**
   Preserve the original experiment and its artifacts, state exactly when the error was found, and record a separate fixed correction. Here only the resample count changes; all models and comparisons remain. The corrected calculation is labelled post-exposure sensitivity, not retroactively claimed as a pristine confirmatory test.

## Remaining scope

The final holdout is consumed. Any new feature or architecture needs a new prospective evaluation period or an explicitly exploratory label. [Phase 6](phase-6-review.md) translates the existing basket forecasts into fixed-notional CAD risk under declared cost assumptions; [Phase 7](phase-7-review.md) communicates the preserved findings. Spot benchmarks do not by themselves support a combined investable commodity portfolio. The missing older regimes, verified vintages, official exchange calendars and execution data remain unresolved.
