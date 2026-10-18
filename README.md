# Weather-Aware Energy and Commodity Risk Forecaster

**Do lagged weather observations improve volatility and tail-risk forecasts beyond standard financial controls?** This research pilot compares nine models for Henry Hub spot, WTI spot changes and a four-stock Canadian energy/utility basket.

**Result:** the 2024–2025 evaluation does not establish a general weather-feature advantage. Material VaR/ES calibration weaknesses remain. No matched weather comparison rejects after multiplicity adjustment in the disclosed bootstrap sensitivity. Non-rejection is not evidence of equivalence.

The contribution is an auditable comparison: availability-aware features, matured training labels, matched weather ablations, a recorded evaluation boundary, and preserved negative results.

[Committee memo (PDF)](output/pdf/committee-memo.pdf) · [Recruiter guide](docs/recruiter-guide.md) · [Full results](reports/phase5/evaluation_report.md) · [Release review](docs/release-review.md) · [Financial context](docs/financial-context.md) · [CI checks](https://github.com/lukelu389/weather-risk-forcaster/actions/workflows/ci.yml)

![Final-period VaR coverage by asset, model and confidence level](reports/phase5/01_var_coverage.png)

Dots show observed exceedance rates; dashed lines show the nominal 5% or 1% rate. Bars are pointwise exact binomial intervals that assume independent exceedances. Henry Hub weather SSM has 19 exceptions in 477 dates at 99% VaR; coverage alone also misses exception clustering. Read the [limitations](docs/model-limitations.md) before treating any model as suitable for use.

## Try it locally

Python 3.11+ and `make`; the following commands use only the Python standard library and need no API keys, GPU or provider data:

```sh
make demo
make verify-results
```

`demo` creates a deterministic, explicitly synthetic ingestion run under `runs/demo-*/`, with observations, as-of alignment diagnostics and a manifest. `verify-results` checks the [public evidence](evidence/README.md) and saved-result claims; it does not train the nine models or reproduce private raw inputs. Without `make`, use:

```sh
PYTHONPATH=src python3 -m weather_risk --mode demo --config config/demo.json
python3 scripts/verify_results.py
```

Download or clone the repository, then open [reports/phase7/dashboard/index.html](reports/phase7/dashboard/index.html) in a browser for offline interactive views. GitHub's file viewer does not run the dashboard. The [memo](reports/phase7/committee-memo.md), [coverage figure](reports/phase5/01_var_coverage.png), [weather figure](reports/phase5/02_weather_comparisons.png) and saved tables provide static alternatives.

## Study design and interpretation

| Component | Recorded pilot |
|---|---|
| Periods | 2018–2021 development; 2022–2023 tuning; 2024–2025 final evaluation |
| Targets | Henry Hub percentage log returns; WTI USD/barrel changes; equal-weight ENB.TO/SU.TO/FTS.TO/EMA.TO basket percentage log returns |
| Models | Historical simulation, GARCH, GJR-GARCH; seasonal/weather variance; control/weather GRU; control/weather diagonal state-space model |
| Comparison | Same valid dates within each asset; 95%/99% VaR and ES; matched architecture and training budgets |
| Final sample | 477 gas, 473 WTI and 502 basket dates; 27,882 forecast rows including both confidence levels |
| Monetary illustration | Independent CAD 1 million basket intervals with 0/5/10 bp assumed costs; no compounded trading strategy |

Latest-vintage data with assumed availability lags and retrospective, variable quote intervals limit historical implementability. Weather means lagged observed Toronto/Calgary heating and cooling degree days; it does not mean archived weather forecasts. The small 99% tail sample, fixed survivor basket and Gaussian parametric tails further limit inference.

The original 1,000-draw bootstrap could not reach the first Holm rejection thresholds. A separately recorded, **post-exposure** 19,999-draw sensitivity changes only numerical resolution: three adjusted ES-residual rejections, one adjusted score-comparison rejection, and no matched-weather rejection. Both versions remain available. See the [correction](docs/phase-5-bootstrap-addendum.md) and [supplementary results](reports/phase5-sensitivity/sensitivity_report.md).

**The final period is consumed.** This is a completed research pilot, with no claim of tradable alpha, regulatory approval, issuer valuation or production risk limits. A different model, feature set or instrument requires a new study boundary.

## Reproduction and code

Public saved-result verification and synthetic tests work from a clean clone. **A full empirical refit additionally requires the exact, permissioned local provider snapshots and preserved runs**, which are not distributed. Re-downloading revised history creates a different input. The [data policy](docs/data-policy.md) explains that boundary; [experiments/pilot](experiments/pilot/) preserves the original sealed implementation, separately from release engineering fixes.

Run `make test` for the available environment. Core-only testing skips optional research modules. To exercise the full research suite, install `requirements-ml.lock.txt` in a virtual environment and use `make test-full PYTHON=.venv/bin/python`; the release environment uses Python 3.14.2/macOS ARM64 with PyTorch 2.13.0. The archived pilot retains PyTorch 2.10.0; see the [security review](docs/security-review.md). Cross-platform byte identity is not claimed.

| Location | Purpose |
|---|---|
| [src/weather_risk](src/weather_risk/) and [tests](tests/) | Ingestion, timing, models, scoring, accounting and offline tests |
| [config](config/) | Explicit ingestion and research configurations |
| [reports](reports/) and [notebooks](notebooks/) | Published derived results, plots and saved-result inspection |
| [docs/research-history.md](docs/research-history.md) | Phase chronology, historical checks and original private-input workflows |
| [docs/model-limitations.md](docs/model-limitations.md) | What the evidence supports and the next research gate |

Post-pilot vintage, prospective-protocol and futures-accounting modules have synthetic tests; they do not constitute an empirical futures backtest. The longer-history snapshot failed data validation. GPU work remains deferred. See the [extension review](docs/extensions-review.md).

Developed with substantial OpenAI Codex assistance; [contribution and interview guidance](docs/recruiter-guide.md) separates project evidence from personal authorship claims. Original project code is available under the [MIT license](LICENSE). Provider data and third-party materials retain their own rights.
