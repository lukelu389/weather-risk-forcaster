# Research history and original phase workflows

> Historical record of the original Phases 1–7 and subsequent extension work. Counts, hashes and commands below describe their recorded environments and private input snapshots. They are not promises that a public clone can refit the pilot. Start with the [current README](../README.md) for supported public commands. The original source/configuration snapshot is preserved in [experiments/pilot](../experiments/pilot/); corrected release code is separate. The 2024–2025 holdout has been consumed.

A quantitative research portfolio project focused on defensible volatility and tail-risk forecasts. **Phases 1–7 are complete for the qualified pilot:** auditable data, development-only exploration, tuning forecasts, a sealed 2024–2025 final evaluation of nine fixed models, CAD portfolio sensitivities, and communication artifacts. The final period has been evaluated and must not be reused as an untouched holdout. Results use latest-vintage data; no trading performance or bank model approval is claimed. Verified historical vintages, executable calendars and the missing older regimes remain outside this completed pilot.

Start with the [interactive local dashboard](../reports/phase7/dashboard/index.html), [two-page committee memo](../output/pdf/committee-memo.pdf), or [final phase review](phase-7-review.md). Pilot completion recorded 230 passing tests; the current suite including the [post-pilot extensions](extensions-review.md) passes **304 tests with no skips** in the locked environment. Repeated pilot runs match 617 primary evaluation artifacts, 14 supplementary artifacts and six portfolio artifacts; the five communication artifacts also reproduce exactly. Detailed evidence accompanies each phase below.

## Run the verified offline workflow

Requires Python 3.11+ and `make`; tested with Python 3.14.2. The core has **no third-party dependencies**. From this repository:

```sh
make reproduce
```

This runs the unit/integration tests and a deterministic, explicitly **synthetic** demonstration. It creates a new `runs/demo-*/` directory. Without `make`:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m weather_risk --mode demo --config config/demo.json
```

Each run saves:

| File | Purpose |
|---|---|
| `observations.csv` | Long-form observations, units, availability assumptions, retrieval time, flags |
| `feature_panel.csv` | As-of alignment diagnostics with explicit missing/stale/rejected reasons |
| `validation.json` | Errors, warnings, completeness, date ranges, and retained outlier flags |
| `config.json` | Exact resolved configuration, universe, dates, and seed |
| `source_snapshot.json` | Source code used for the run |
| `manifest.json` | Status, source checksums, code/config/output hashes, runtime and installed library versions |

`data/raw/` holds cached source responses. Failed runs retain diagnostics and exit nonzero. Raw data, runs, credentials, and virtual environments are ignored by Git.

## Fetch public data

```sh
make live-smoke
```

This requests January 2024 from FRED, Bank of Canada, and ECCC. Full pilot dates are **2018–2025**, chosen to overlap the initial source coverage, not as the final study period. For Canadian equities, install the optional Yahoo adapter in a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-market.lock.txt
make live PYTHON=.venv/bin/python
```

The complete tested Yahoo dependency set is version-pinned in `requirements-market.lock.txt` (Python 3.14.2, macOS ARM64). `requirements-market.txt` contains just the top-level dependency for intentionally resolving a new environment. The verified core workflow does not need Yahoo, NumPy, pandas, or API keys.

EIA requires a [free personal API key](https://www.eia.gov/opendata/). Set `EIA_API_KEY` in your process environment. Storage and consumption are explicitly optional while the key is absent, and every skipped source is logged. Change `required` to `true` when those features become prerequisites for an experiment. Do not commit a key.

Select a single source if diagnosing a provider:

```sh
PYTHONPATH=src python3 -m weather_risk --mode live --config config/smoke.json --series usd_cad
```

**Phase 1 verification:** 55 Phase 1 tests pass. The live smoke test and full 2018–2025 pilot passed for FRED, Bank of Canada, ECCC and Yahoo: 39,760 observation rows across 17 series. Station identities were confirmed from returned ECCC records. EIA storage and consumption were explicitly skipped because no API key is configured; their adapters have fixture tests but no live verification. See `docs/verification.json` and the local run reports for evidence. The original local workspace retained provider data in ignored data/run directories. For the current public package boundary, see [data policy](data-policy.md).

## Reproduce Phase 2 exploration

Use the preserved Phase 1 snapshot pinned in `config/analysis.json`. Provider data are ignored by Git: a fresh checkout alone cannot reproduce these empirical results. Preserve the input run's `observations.csv`, `config.json`, `validation.json`, and `manifest.json`. Phase 2 reads that snapshot without network access; it does not replay the Phase 1 source code.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.lock.txt
make test PYTHON=.venv/bin/python
make eda
```

At the Phase 2 gate, **86 tests** passed with no skips; the current expanded suite is described below. Without the optional analysis dependencies, Phase 2 test modules are skipped; a core-only pass does not verify Phase 2. The lock file records the tested Python 3.14.2/macOS ARM64 environment, including the earlier market dependencies. Cross-platform byte identity has not been verified.

`make eda` explicitly acknowledges latest-vintage data and publishes tables and five PNG/PDF pairs in [reports/phase2](../reports/phase2/analysis_report.md). Full targets, weather features, source snapshot, dependency lock, and checksummed outputs remain in a new `runs/eda-*/` directory. Use `make eda INPUT_RUN=runs/live-...` only when intentionally replacing the pinned snapshot; that establishes a new analysis input.

To independently repeat the exact analysis twice without replacing the published report:

```sh
PYTHONPATH=src .venv/bin/python - <<'PYCODE'
import json
from weather_risk.eda import run_analysis, sha
runs = [run_analysis(allow_latest_vintage=True, publish=False) for _ in range(2)]
manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
assert manifests[0] == manifests[1]
for name, expected in manifests[0]["artifacts"].items():
    assert all(sha(p / name) == expected for p in runs), name
print("All artifacts and manifests match:", *runs)
PYCODE
```

Two full local reruns matched byte-for-byte, including all eight CSVs and ten plot files; see [verification evidence](phase-2-verification.json). The reviewed plots show heavy tails and clustered movement magnitudes. The primary weather correlations are small (largest absolute rho about 0.102); these are exploratory associations, not evidence of incremental forecast skill. Units differ: WTI uses USD/barrel changes, while gas and equities use percentage log returns. Phase 2 itself fits no models; the separate Phase 3 workflow below adds tuning-period baselines.

## Reproduce Phase 3 baselines

The same preserved Phase 1 snapshot is pinned in `config/baselines.json`. SciPy is the only new dependency, needed for constrained GARCH estimation; the standard-library ingestion workflow remains independent.

```sh
.venv/bin/python -m pip install -r requirements-baselines.lock.txt
make test PYTHON=.venv/bin/python
make baselines
```

At the Phase 3 gate, **131 tests passed with no skips** in the baseline environment. The dependency-free `make reproduce` also passed (74 executed tests, five optional modules skipped) and completed the synthetic demo. The expanded Phase 4 suite is described below.

The five baselines are historical simulation, GARCH(1,1), GJR-GARCH(1,1), seasonal log-variance ridge regression, and its weather-augmented counterpart. The last two share the same historical volatility and annual seasonality controls; weather adds Toronto/Calgary HDD18 and CDD18. Parameters and preprocessing refit monthly using up to 756 original grid rows and at least 252 finite, matured labels. Historical simulation updates each origin. Fixed settings are not selected using the reported scores.

Training labels require both endpoint prices to be available, including all four basket constituents. Historical features use their original forecast cutoffs. Unavailable shocks remain missing in GARCH filtering and propagate expected variance. The calendar is the retrospective observed quote grid, not an official prospective trading calendar. The pilot uses the existing conservative origin before both price endpoints and variable quote intervals up to four calendar days. Broader history and verified vintages remain unresolved.

Outputs: [baseline report and scores](../reports/phase3/baseline_report.md), [phase review](phase-3-review.md), and [verification evidence](phase-3-verification.json). Full forecasts, the design panel, fitted parameters/preprocessing, source snapshot, lock file, and hashes remain in `runs/baselines-*/`. Failed or skipped forecasts are retained; a run with failed forecasts exits nonzero. Scores compare dates common to all five models, within each asset. Formal VaR/ES calibration and final-test evaluation are deferred.

For an independent byte-for-byte check without publishing:

```sh
PYTHONPATH=src .venv/bin/python - <<'PYCODE'
import json
from weather_risk.baselines import run_baselines
from weather_risk.eda import sha
runs = [run_baselines(allow_latest_vintage=True, publish=False) for _ in range(2)]
manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
assert all(m["status"] == "passed" for m in manifests)
assert manifests[0] == manifests[1]
for name, expected in manifests[0]["artifacts"].items():
    assert all(sha(p / name) == expected for p in runs), name
print("All artifacts and manifests match:", *runs)
PYCODE
```

Two complete reruns matched all **eight artifacts and their manifests** byte-for-byte in the installed environment.

The empirical run produced 15,400 forecast rows (including both risk levels), 300 successful monthly model fits, and no failed/skipped forecasts. Realization gaps leave 478 common scored dates for each spot series and 498 for the basket. Weather effects are mixed; 112 of 150 GARCH/GJR fits approach the 0.999 persistence constraint. Convergence alone does not establish adequate tail forecasts.

## Reproduce Phase 4 sequence models

`config/ml.json` pins the same Phase 1 snapshot and the verified Phase 3 forecast run. Preserve both ignored run directories; a fresh checkout alone cannot reproduce the empirical comparison. PyTorch 2.10.0 is the only new top-level dependency.

```sh
.venv/bin/python -m pip install -r requirements-ml.lock.txt
make test PYTHON=.venv/bin/python
make ml
```

At the Phase 4 gate, **164 tests passed with no skips**. A core-only environment skips optional model tests and does not verify this phase. The lock file records the tested Python 3.14.2/macOS ARM64 environment.

One GRU with 8 hidden units (585 parameters) and one stable diagonal state-space model with 32 states (545 parameters) each have matched control/weather variants. The latter is a simple learned recurrence, not Mamba. Each input contains 21 original-grid feature snapshots. Training uses at most 756 grid rows, at least 252 matured labels, monthly refits, training-only preprocessing, and 40 full-batch Adam updates at learning rate 0.01. Seeds 1729 and 2718 have their variances averaged into a zero-mean Gaussian risk forecast. Weather ablation removes both values and missingness; architecture and training budgets remain matched. No early stopping, best-seed selection, or score-driven retuning is used.

Outputs: [nine-model tuning report](../reports/phase4/ml_report.md), [phase review and three interview questions](phase-4-review.md), and [verification evidence](phase-4-verification.json). Full forecasts, design panel, fitted preprocessing, 600 tensor checkpoints, source snapshot, lock file, and hashes remain in `runs/ml-*/`.

Two complete runs matched all **608 artifacts and their manifests** byte-for-byte. All 600 saved checkpoints were restored and checked at 300 ensemble fit origins; the largest relative variance difference from CSV rounding was approximately `6.13e-12`. The runs produced 12,320 neural forecast rows, 27,720 combined rows, 300 ensemble fits, and no failed/skipped forecasts. Every member reduced its training loss; this is an optimization check, not evidence of predictive skill.

For an independent comparison without publishing:

```sh
PYTHONPATH=src .venv/bin/python - <<'PYCODE'
import json
from weather_risk.ml import run_ml
from weather_risk.eda import sha
runs = [run_ml(allow_latest_vintage=True, publish=False) for _ in range(2)]
manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
assert all(m["status"] == "passed" for m in manifests)
assert manifests[0] == manifests[1]
for name, expected in manifests[0]["artifacts"].items():
    assert all(sha(p / name) == expected for p in runs), name
print("All artifacts and manifests match:", *runs)
PYCODE
```

All nine models use the same scored dates within each asset/confidence. Tuning results are mixed: weather improves the gas GRU's quantile loss but worsens the gas state-space model and both WTI neural variants. No general neural or weather advantage is established. Reproducibility is verified only for the preserved inputs, source, and installed CPU environment. The Phase 5 workflow below freezes those settings before final evaluation and adds calibration and uncertainty diagnostics.

## Reproduce Phase 5 final evaluation

The [protocol](phase-5-protocol.md), [local seal](phase-5-seal.json), and [first-access audit](phase-5-first-access.json) record the final evaluation boundary. The seal includes consumed source metadata, model settings, implementation and lock file. Preserve the original Phase 1 snapshot and these files. Do not regenerate the seal after exposure.

```sh
make test PYTHON=.venv/bin/python
make evaluate
```

At this gate, **195 tests passed**. Two complete runs produced **617 byte-identical artifacts**, including 600 checkpoints and four plot files, with matching manifests. There are 27,882 forecast rows, 300 baseline fits and 300 neural ensemble fits; no forecasts failed or were skipped. Common scored dates are 477 for gas, 473 for WTI and 502 for the basket. See [verification](phase-5-verification.json), [results](../reports/phase5/evaluation_report.md), and [review with three interview questions](phase-5-review.md).

Coverage and independence diagnostics expose material calibration problems; all models remain in the comparison. The initial 1,000-draw bootstrap has insufficient p-value resolution for the 165-comparison and 54-ES Holm families. Its non-rejections cannot establish equivalence or calibration. A separately recorded [post-exposure resolution correction](phase-5-bootstrap-addendum.md) retains all forecasts and comparisons and increases only the resample count to 19,999. It is a sensitivity analysis, not a fresh holdout.

```sh
make sensitivity
```

The correction reproduces **14 artifacts and matching manifests** across two runs. It finds three adjusted ES-residual rejections (gas GJR at 95%, gas GARCH/GJR at 99%) and one of 165 adjusted score-comparison rejections (basket GARCH versus historical simulation, QLIKE). No matched weather comparison rejects after Holm adjustment. This does not establish equivalence or adequate calibration. See the [supplementary report](../reports/phase5-sensitivity/sensitivity_report.md) and [verification](phase-5-sensitivity-verification.json). The sealed original calculation remains unchanged.

## Reproduce Phase 6 CAD portfolio sensitivity

```sh
make portfolio
```

`config/portfolio.json` pins the preserved Phase 5 manifest. Each interval independently starts with CAD 1 million, split 25% each among ENB.TO, SU.TO, FTS.TO and EMA.TO. The calculation converts the basket's percentage log-return distribution into monetary losses. Gaussian ES uses the exact transformed tail expectation; historical simulation transforms every matured sample outcome before averaging its tail. Applying the loss transformation to log-return ES alone would be incorrect.

Fixed total interval cost budgets of 0/5/10 basis points add CAD 0/500/1,000 to realized loss, VaR and ES, leaving exceedance classifications unchanged. They are scenarios, not measured spreads or turnover. There is no compounded wealth path, strategy selection, annualization, inferred futures roll or combined USD commodity book. The adjusted equity prices remain research proxies.

Two complete runs match **six artifacts and their manifests**. There are 27,162 sensitivity rows, 162 summary rows, no skipped sensitivities and 502 valid realized intervals. One missing outcome produces 54 missing realized sensitivity rows across models, levels and costs. Across all nine models, full-period mean gross 99% ES ranges from **CAD 20,122 to CAD 25,535**; these model-dependent forecasts are not a capital recommendation. Calendar-year slices are descriptive. See the [report](../reports/phase6/portfolio_report.md), [review and three interview questions](phase-6-review.md), and [verification](phase-6-verification.json).

## Phase 7 communication and audit artifacts

- [Dashboard](../reports/phase7/dashboard/index.html): open the HTML file in a browser; embedded data work offline. Coverage and weather views retain both the sealed primary calculation and the disclosed correction. The CAD view exposes costs and calendar slices.
- [Committee memo](../output/pdf/committee-memo.pdf), also available as [Markdown](../reports/phase7/committee-memo.md): two pages covering evidence, interpretation, limitations and the next research decision.
- [Evaluation notebook](../notebooks/01_final_evaluation.ipynb) and [portfolio notebook](../notebooks/02_portfolio_sensitivity.ipynb): read checksummed saved outputs and construct audit tables and figures without refitting models.
- [Phase review and three interview questions](phase-7-review.md) and [verification record](phase-7-verification.json).

The dashboard's 42 filter combinations were checked in the browser; desktop and narrow-screen layouts were reviewed. Both memo pages were visually inspected. All eight notebook code cells executed in the research environment; Jupyter's interactive interface was not tested. Two communication builds match all five artifacts and the manifest. Reproducibility claims apply to the recorded local environment, not arbitrary platforms.

To rebuild these communication artifacts, use `python scripts/build_communication.py` in an authoring environment with ReportLab 4.4.9. The bundled document runtime supplied that package for the verified build; it is separate from the research dependency lock. To execute notebook cells, use the locked research environment. The builder verifies the consumed report hashes and records source, template and output hashes in [its manifest](../reports/phase7/manifest.json). A fresh checkout lacks the ignored provider snapshots and model runs required to repeat the empirical research.

## Post-pilot extensions — GPU work on hold

The local extensions are implemented separately from Phases 1–7. They add no
dependencies and leave the sealed pilot unchanged. See the [extension review](extensions-review.md)
and [verification record](extensions-verification.json) for completion status
and three interview questions. The earlier phase-by-phase PDF records the pilot
before these additions; this review records the subsequent changes.

| Extension | Implemented | Evidence and remaining gate |
|---|---|---|
| Longer history and broader weather | Separate 2000–2025 staging; Chicago, New York and Houston NOAA stations; EIA storage/consumption; separate FRED CAD/USD series | 225,819 rows, 29 series, seven reproduced artifacts and matching manifests. **Data validation failed**, so this snapshot is not approved for model fitting. |
| Revision-aware inputs | Versioned JSON/CSV archives, as-of selection and paginated ALFRED acquisition | Synthetic tests pass. Live ALFRED needs `FRED_API_KEY`; verified release evidence is still required. [Instructions](vintage-data.md). |
| New evaluation gate | Future-only protocol sealing, hypothesis families, bootstrap-resolution checks and audited first access | Synthetic tests pass. No real new experiment has been frozen or evaluated. [Protocol contract](prospective-protocol.md). |
| Futures and FX accounting | Explicit contract sizing, roll legs, fills, fees, slippage attribution and retained foreign cash | Synthetic accounting identities pass. Actual contract/execution/settlement/FX data and calendars remain necessary. [Input contract](execution-input-contract.md). |
| GPU cluster | Deferred at your request | No GPU code, submission script or cluster job added. |

Historical staging and exact replay use the existing research environment:

```sh
make history
# The current snapshot exits with status 1 because data validation fails.
PYTHONPATH=src .venv/bin/python -m weather_risk.historical --offline
PYTHONPATH=src .venv/bin/python -m weather_risk.historical \
  --replay runs/history-YYYYMMDDTHHMMSS.ffffffZ/manifest.json --no-publish
```

Read [coverage and validation findings](../reports/extensions/history_report.md)
before modelling. The current snapshot retains two zero adjusted equity prices
and excessive missing Henry Hub observations, mostly before 2008. EIA storage
starts in 2010 in the retrieved table, so it does not cover 2008. No values were
imputed, replaced or silently dropped. Source quality and missingness must be
resolved under a recorded policy before any extended model study. Date-only
vintages, assumed lags and airport proxies retain their limitations.

The full suite passes **304 tests**; the Phase 5 seal still verifies. No new model
fits or performance claims were produced, and 2024–2025 remains a consumed
holdout. The new modules are `historical.py`, `vintages.py`, `vintage_sources.py`,
`prospective.py` and `execution.py` under `src/weather_risk/`.

## Availability is a data requirement

An observation date is the period measured. `available_at` is when our research assumption allows it into a forecast. `retrieved_at` records when this particular response was downloaded. Monthly observations are normalized to month-end before applying a lag.

All current live adapters download **latest-vintage** history and assign **assumed** calendar-day delays. Those are not verified historical release times. Lagging revised history does not eliminate revision lookahead; FRED distinguishes historical information sets through its [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

Default alignment withholds these observations (`missing_reason=strict_vintage`). To inspect a deliberately approximate research panel:

```sh
PYTHONPATH=src python3 -m weather_risk --mode live --config config/live.json --allow-latest-vintage
```

That flag is recorded and never changes the provenance label. It does **not** certify a historically tradable backtest. Synthetic data use a separate provenance class. No automatic backfill, interpolation, winsorization, or current-value fallback is performed.

The alignment grid is weekdays at 18:00 America/Toronto. It is a diagnostic grid, not a TSX/NYMEX holiday calendar or an execution schedule. Tradeable targets and session calendars must be fixed before model fitting.

## Replay an exact run

Preserve the run directory and `data/raw/blobs/`, then substitute its manifest path:

```sh
PYTHONPATH=src python3 -m weather_risk --replay runs/demo-YYYYMMDDTHHMMSS.ffffffZ/manifest.json
```

Replay restores the saved config, source omissions and response hashes, even if newer downloads replaced the request pointers. An optional source skipped in the original run remains skipped on replay. It verifies output hashes. It rejects changed source code; `source_snapshot.json` preserves the original files for restoring that version. Run from the same repository/data root, or pass `--output-root` pointing to the preserved root. Ordinary `--offline` uses the currently cached requests; `--refresh` obtains a new snapshot, once per distinct request per run.

## Code map

```text
config/                 ingestion, exploration, baseline and sequence protocols
src/weather_risk/
  schema.py             observation contract and timezone-aware lag calculation
  providers.py          ECCC, FRED, BoC, EIA and optional Yahoo adapters
  cache.py              sanitized response blobs, integrity checks, pinned replay
  validation.py         missingness, units, physical limits, timestamps, jumps
  timing.py             as-of joins and expanding/rolling walk-forward splits
  demo.py               labeled synthetic integration data
  pipeline.py           CLI, local experiment manifests and failure reports
  eda.py                Phase 2 runner, protocol gates and artifact manifests
  eda_core.py           development filtering, targets and descriptive statistics
  eda_weather.py        availability-aware weather and seeded block intervals
  eda_plots.py          five PNG/PDF figure pairs
  baselines.py          Phase 3 availability-aware runner and tuning scores
  baseline_models.py    GARCH/GJR QMLE and log-variance ridge regressions
  risk.py               loss-based empirical/Gaussian VaR, ES and pinball loss
  ml.py                 Phase 4 monthly sequence fits and nine-model comparison
  evaluation.py         coverage, joint tail scores and block uncertainty
  final_evaluation.py   sealed Phase 5 runner and first-access audit
  bootstrap_sensitivity.py  disclosed bootstrap-resolution correction
  portfolio.py          Phase 6 fixed-notional CAD sensitivity runner
  portfolio_risk.py     exact monetary Gaussian/empirical VaR and ES
  sequence_models.py    deterministic GRU/diagonal-state-space variance models
scripts/                Phase 7 local dashboard, memo and notebook builder
notebooks/              checksummed evaluation and portfolio audit notebooks
tests/                  offline unit and integration tests
docs/phase-*-review.md   phase decisions, limitations and interview preparation
docs/data-contract.md   source definitions and information-time assumptions
```

Local JSON manifests record experiment inputs, metrics and trained artifacts without an external tracking service.
