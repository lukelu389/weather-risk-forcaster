# Recruiter and interview guide

This is an AI-assisted quantitative research portfolio project. Its strongest evidence is the handling of time, fair model comparisons, negative results and reproducibility. It is suitable for discussing research judgment and engineering discipline; it does not demonstrate profitable trading, a bank-approved risk model or an investment-banking valuation.

## A five-minute review

1. Read the [two-page committee memo](../output/pdf/committee-memo.pdf) for the decision and limitations.
2. Inspect the [coverage figure](../reports/phase5/01_var_coverage.png) and the [bootstrap correction](phase-5-bootstrap-addendum.md). The project preserves a methodological flaw and its disclosed sensitivity analysis.
3. Run `make demo` and `make verify-results` from the repository root. These require no credentials or research libraries.
4. Inspect [timing.py](../src/weather_risk/timing.py), [risk.py](../src/weather_risk/risk.py), [evaluation.py](../src/weather_risk/evaluation.py) and their tests. Review the original experiment separately in [experiments/pilot](../experiments/pilot/).
5. Use the [financial context](financial-context.md) to discuss what a risk analyst or banker would still need before making an economic decision.

## Evidence by role

| Role | Evidence to examine | Boundary to probe |
|---|---|---|
| Quantitative research | Matured labels, matched weather ablations, common-date scoring, uncertainty and multiplicity | Latest-vintage inputs, small tails, Gaussian distribution, retrospective horizons |
| Risk / model validation | Calibration, missing-grid transition handling, failure records, hashes and sealed source snapshot | Reproduction is distinct from validity; local seals are not external preregistration |
| Banking / equity research | Clear committee communication, monetary risk interpretation and issuer mechanism | No three-statement forecast, DCF, comparable-company valuation or deal model was built |
| Research engineering | Deterministic synthetic workflow, validation failures, optional-dependency handling and public result verification | Full empirical refits need an exact input bundle omitted from the public clone |

## Contribution disclosure

Implementation, tests, analysis support and documentation were developed with substantial OpenAI Codex assistance. The repository records outputs and verification; it does not by itself establish what a candidate implemented independently. An interview or application should identify the candidate's actual design, review and debugging contributions and explain any adapted methods. Do not claim sole unaided authorship or professional deployment.

An accurate project description is: “AI-assisted research comparing nine volatility and tail-risk models across energy benchmarks and a Canadian equity basket, with availability-aware evaluation, reproducible saved evidence, and disclosed calibration and inference limitations.” Adapt personal contribution verbs only to work actually performed and understood.

## Three interview questions

1. **Why is a lag insufficient to eliminate lookahead?** A lag can delay a revised historical value without restoring its earlier vintage. The forecast information set needs both release timing and vintage evidence; the pilot has assumed lags only.
2. **How can a working pipeline produce an unconvincing model result?** Tests and hashes establish aspects of implementation and reproducibility. Forecast scores, calibration, power, fair comparisons and information availability determine the strength of the scientific claim. This project's negative result is retained rather than replaced by selected winners.
3. **What would you change before presenting this to a bank?** Resolve data quality and prospective target definitions first; separate portfolio market risk from issuer cash-flow risk; then freeze one justified model extension and a new evaluation period. Explain why the present CAD sensitivities cannot be turned into approved limits or valuation inputs without those steps.
