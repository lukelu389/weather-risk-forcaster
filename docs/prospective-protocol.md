# A new prospective evaluation gate

The 2024–2025 holdout was consumed in Phase 5. It remains useful for descriptive analysis and later training, but cannot support another untouched-test claim. The standalone `weather_risk.prospective` module provides a gate for a **new, genuinely future evaluation period**. It does not alter the pilot's sealed code, outputs or conclusions.

No real prospective protocol has been frozen. The period, new hypotheses, complete model settings, data snapshot and nominated runner still require a concrete research decision. Adding this gate does not create a trained model, historical data, verified publication timestamps, or GPU support.

## What is enforced

- The evaluation start must be strictly after the UTC date on which the protocol is frozen. Any overlap with the consumed 2024–2025 holdout is rejected.
- Every hypothesis has a unique ID, an explicit statement and a declared Holm family. Enumerated hypothesis counts must equal declared family sizes. State each asset, confidence level, metric and comparison in the individual statements; the gate cannot infer omitted scientific hypotheses from prose.
- Bootstrap replications must provide sufficient p-value resolution for the first Holm threshold in every declared bootstrap family, including the declared minimum valid fraction. The check is `1 / (ceil(B × fraction) + 1) <= alpha / family_size`. This prevents the pilot's original resolution problem; it does not establish power or justify a particular block length.
- Model settings, refit rules, maturity rules, missing-value policy and local code/configuration/input files must be explicit. The protocol JSON itself and this gate's source are included in the hash manifest. Every dependency must be nominated; transitive dependencies are not discovered automatically. Nominate the runner, imported research modules and dependency lock file as well as the dataset, source manifest and source configuration.
- The seal contains the exact dependency paths, their SHA-256 hashes and a UTC timestamp. Verification rejects changed, missing or extraneous manifest entries. Unrelated new files do not invalidate it. Relative paths must remain inside the project; symlink files and symlink path components are rejected.
- The future holdout path must be absent when freezing. Its eventual bytes are bound to the first-access audit. Reproductions require precisely the same snapshot.
- This is a **completed-period evaluation gate**. `access_not_before` must be after the evaluation end and allow the declared target maturity delay in calendar days. Nothing opens the holdout CSV before that date. This avoids treating an unfinished prospective period as a completed evaluation.
- First-access JSON is published atomically and exclusively before CSV or numeric parsing. A failed parse still consumes the gate. Parallel reproductions share the original record.
- The loader filters dates and series before reading numeric cells, then excludes records whose availability timestamp exceeds the original first-access time. Reproductions keep that same time cutoff. It preserves blank numeric cells as `None` and rejects other non-finite numeric values. Excluded rows need no valid numeric values; their numeric cells are never parsed.

A local seal is an auditable record, not a digital signature or independent preregistration. Archive the frozen files and seal in version control or an external research record before the evaluation period begins. Hashes alone do not prevent someone deliberately rewriting both files and their manifest.

## Protocol format

This is a **format example**, not a proposed experiment. Its future dates and synthetic model are illustrative. The absent example files make it unsuitable for execution, and `status: "draft"` intentionally prevents freezing. Fill in the actual research decisions and all dependency paths before deliberately changing the status to `ready`.

```json
{
  "schema_version": 1,
  "status": "draft",
  "protocol_id": "example-future-evaluation",
  "evaluation": {
    "start": "2027-01-01",
    "end": "2027-12-31",
    "access_not_before": "2028-01-08"
  },
  "hypotheses": [
    {
      "id": "example-gas-95-quantile-loss",
      "family": "score_comparisons",
      "statement": "For the specified gas target at 95% confidence, mean candidate-minus-control quantile loss is zero."
    }
  ],
  "models": {
    "example_control": {"constant_variance": 1},
    "example_candidate": {"constant_variance": 2}
  },
  "refit": {
    "schedule": "monthly at the first eligible forecast origin",
    "window_rows": 756,
    "minimum_training_rows": 252
  },
  "maturity": {
    "target_delay_days": 8,
    "forecast_origin": "18:00 America/Toronto on the day before the earlier price endpoint",
    "availability_policy": "Use verified release timestamps and preserve original forecast-origin eligibility",
    "prior_holdout_labels_in_later_fits": true
  },
  "inference": {
    "alpha": 0.05,
    "adjustment": "holm",
    "family_sizes": {"score_comparisons": 1},
    "bootstrap_families": ["score_comparisons"],
    "bootstrap": {
      "replications": 19999,
      "minimum_valid_fraction": 0.95,
      "block_length": 20,
      "minimum_observations": 100,
      "seed": 1729
    },
    "missing_policy": "Retain missing dates on the original grid; use the frozen common-date scoring rule"
  },
  "files": {
    "code": ["src/weather_risk/prospective.py", "src/weather_risk/example_runner.py"],
    "config": ["config/example_models.json", "requirements-ml.lock.txt"],
    "input": ["runs/example_training/observations.csv", "runs/example_training/manifest.json"]
  },
  "holdout": {
    "path": "runs/example_future/observations.csv",
    "date_column": "observation_date",
    "availability_column": "available_at",
    "series_column": "series_id",
    "series_ids": ["gas"],
    "numeric_columns": ["value"]
  },
  "outputs": {
    "seal": "docs/example-prospective-seal.json",
    "first_access": "docs/example-prospective-first-access.json"
  }
}
```

All displayed keys are required; unknown keys are rejected to catch misspellings. Settings within each model remain model-specific JSON objects. Additional named models, hypotheses, families, files and selected series are supported. Each file may have only one nominated role. Do not list the protocol JSON under `files`: it is sealed automatically.

## Commands and integration

Run from the repository root using its existing Python environment:

```sh
# Only after the actual future experiment and ready configuration are finalized:
PYTHONPATH=src .venv/bin/python -m weather_risk.prospective freeze --config config/prospective.json

# Read-only verification; does not open or consume the holdout:
PYTHONPATH=src .venv/bin/python -m weather_risk.prospective verify --config config/prospective.json

# After access_not_before and acquisition of the final holdout snapshot:
# This explicitly consumes the gate, loads eligible values and prints counts only.
PYTHONPATH=src .venv/bin/python -m weather_risk.prospective read --config config/prospective.json
```

The module is standard-library only. Its public Python interfaces accept the config path and optional `root=Path(...)`:

- `freeze_protocol(...) -> Path`: publish the seal without replacing an existing one.
- `verify_protocol(...) -> (protocol, seal)`: verify without opening the holdout.
- `record_first_access(...) -> audit`: verify the protocol/calendar, pin the holdout bytes and record access without parsing CSV values.
- `load_holdout_csv(...) -> {"rows": [...], "excluded": {...}, "audit": {...}}`: verify, record and return eligible rows with selected numeric columns converted to floats or `None`. Other columns retain their strings.

A new evaluation runner must call this gate before accessing its holdout, use the returned records, and enforce the nominated model/refit/maturity/scoring rules. The gate does not run the new experiment or independently verify those scientific rules. It deliberately does not retrofit or unlock the consumed Phase 5 runner. Future prediction generation can be developed separately; releasing completed-period evaluation values remains blocked until the recorded date.

## Verification

`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_prospective.py' -v`

The 18 synthetic tests cover consumed-period rejection, freeze-date checks, maturity delay, inadequate bootstrap resolution, family-count mismatches, changed dependencies, exact manifests, unsafe paths, concurrent first access, snapshot-bound reproductions, parse failures and deliberately invalid numeric sentinels in excluded rows. The early-access test fails if the holdout's bytes are opened at all. No empirical future holdout is used by these tests.
