# Phase 7 review and interview preparation

## Outcome and artifacts

The communication artifacts are built from verified, preserved Phase 5 primary results, the separately disclosed bootstrap sensitivity, and the Phase 6 CAD summary. They introduce no model fitting, forecast selection or trading-performance calculation.

- [Local dashboard](../reports/phase7/dashboard/index.html): coverage diagnostics, matched weather comparisons and CAD basket sensitivities, with asset, confidence, inference-version, cost and calendar-period controls where applicable. The data are embedded in the HTML; no external data service is needed to view the results.
- [Two-page committee memo](../output/pdf/committee-memo.pdf), with an [editable Markdown counterpart](../reports/phase7/committee-memo.md): the decision, evidence, correction, monetary interpretation and material limitations.
- [Final-evaluation audit notebook](../notebooks/01_final_evaluation.ipynb) and [portfolio-sensitivity notebook](../notebooks/02_portfolio_sensitivity.ipynb): inspect saved, hash-checked tables and reproduce their displays and declared invariants. These are audit notebooks, not another untouched evaluation.

The [communication manifest](../reports/phase7/manifest.json) records the five output hashes, the builder and dashboard-template hashes, and the consumed report manifests, seal, first-access record and correction addendum. The builder is `scripts/build_communication.py`; PDF authoring uses ReportLab in the document runtime, separately from the locked research environment. Numerical research dependencies and sealed source remain unchanged.

## Communication decisions

The central conclusion is a qualified research pilot with reproducible forecasts and material calibration limitations. Weather effects vary across assets and architectures; the final evidence does not support a general weather-feature advantage. All nine models remain visible. Lower risk estimates are not presented as better models or as approved risk limits.

The memo distinguishes the original sealed 1,000-draw inference from the **post-exposure 19,999-draw sensitivity**. The original bootstrap could not reach the first Holm rejection threshold for its planned families. The supplementary calculation retains every forecast and hypothesis and yields one adjusted score-comparison rejection and three adjusted ES-residual rejections at 5%; no matched weather comparison rejects after adjustment. Non-rejection does not establish equivalence, and the correction does not restore an untouched holdout.

The CAD presentation states the independent starting notional, nonlinear return-to-loss conversion and fixed cost budgets. Calendar-year filters are descriptive. The materials distinguish the original benchmark units, conditional risk estimates and realized monetary losses, and they avoid claims about NAV, returns from a trading strategy, capital approval or a combined commodity/equity book.

## Verification gate

The communication gate is complete for this qualified pilot. [Phase 7 verification](phase-7-verification.json) records two builds matching all five artifacts and their manifests, all 42 dashboard filter combinations, desktop and narrow-screen visual inspection, both memo pages reviewed, and all eight notebook code cells executed. The full research suite passed 230 tests without skips, dependency consistency passed, and the original Phase 5 seal and both runs' 617 primary, 14 supplementary and six portfolio artifact hashes were rechecked. The generated manifest's `passed` status alone is not used as evidence of those presentation checks.

Notebook code cells ran sequentially in a fresh Python namespace per notebook, using a noninteractive plotting backend; the Jupyter UI and kernel were not tested. The dashboard checks include 2024/2025 rows and both inference versions. Narrow-screen tables scroll inside their container without page overflow. A reproducible checksum does not establish numerical correctness, legibility or an honest interpretation; the earlier numerical tests and phase reviews remain part of the evidence. Cross-platform byte identity is not claimed.

## Scope and handoff

The materials preserve the latest-vintage/assumed-lag qualification, retrospective calendar and variable intervals, sparse tail events, fixed survivor basket, absent 2008 and 2014–2016 regimes, and unresolved execution data. The local seal records an evaluation boundary but is not external preregistration. The final 2024–2025 holdout is consumed.

The authorized pilot is complete through Phase 7, with local input snapshots, saved runs, dependency locks, manifests and reviews preserved. Any new architecture, feature, instrument or deployment study requires a separately defined protocol and prospective evidence. No publication, trading deployment or production approval is implied by these local artifacts.

## Three interview questions

1. **How would you explain the main result to a risk committee?**
   The pipeline produces reproducible risk forecasts, but calibration weaknesses and mixed weather effects prevent a general claim that extra features improve tail risk. Present the decision, uncertainty and limitations alongside the complete model comparison. A working research system and an adequate production model are different evidentiary achievements.

2. **How do you present an error discovered after final evaluation without overstating the correction?**
   Preserve the original protocol and results, disclose the p-value resolution limitation, and label the fixed resample-count correction as post-exposure sensitivity. Show both versions and retain all hypotheses. More resamples improve numerical resolution; they do not remove exposure, sparse tails or assumptions about dependence.

3. **What makes a dashboard or notebook an auditable research deliverable?**
   Trace displayed values to hash-checked source tables, preserve the code and environment, document units and sample counts, and verify controls and execution. Record numerical, reproduction and visual checks separately. A reader should be able to reproduce the evidence and understand its limits without inferring hidden model selection or unavailable historical data.
