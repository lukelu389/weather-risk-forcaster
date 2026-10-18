# GitHub release checklist

This repository is a qualified research pilot. Publishing it does not certify its
historical data as point in time or its forecasts as suitable for trading.

## Checks supplied in this release

- `make demo`: synthetic ingestion, validation and timing integration; no credentials.
- `make verify-results`: public saved-evidence checks; no private input bundle.
- `make test`: standard-library tests, with explicit skips for absent optional packages.
- `make test-full`: complete research test suite; any skipped test fails this command.
- `.github/workflows/ci.yml`: core Python 3.11/3.14, NumPy/Pandas without Matplotlib,
  and full Python 3.14.2 CPU research jobs. All four jobs passed in the
  [hosted run for source commit 3f64cda](https://github.com/lukelu389/weather-risk-forcaster/actions/runs/39038470454),
  checked 2026-10-18. Recheck the latest PR head after further commits.
- `make package-source`: `git archive` of a clean reviewed commit, with a SHA-256 sidecar.
  It rejects committed private/transient paths and preserves existing archives.

The full job uses the macOS ARM runner and the recorded dependency lock because
that matches the pilot environment. See the [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [PyTorch wheel index](https://download.pytorch.org/whl/torch/).
The core and public-evidence paths also run on Linux. New platform claims require their own checks.

## Before the first public push

1. Review the staged diff and commit only source, tests, configuration, documentation,
   curated derived evidence, plots, and the committee memo. Keep the provider inputs,
   local environments and scratch output private. Review the [data policy](data-policy.md).
2. Run the core demo, public evidence verification and full research test suite.
   Record actual counts and environment in the release notes; do not treat skipped
   optional modules as a full research-suite pass.
3. Scan both the candidate working tree and its Git history for secrets using a
   dedicated tool (for example Gitleaks), and audit the recorded dependencies with
   an advisory service (for example pip-audit). Review findings and archive tool
   versions/results. A text search alone is not a comprehensive security scan.
4. Confirm original pilot seal provenance and the distinction between the archived
   implementation, corrected release code, and sanitized published result subsets.
5. Create the source archive from the reviewed commit, extract it into a fresh
   directory, and run `make demo` and `make verify-results` there.
6. Push when ready, inspect all hosted CI jobs, and link the passing workflow in the
   release. No hosted CI badge or external security pass is asserted before it runs.

## Recorded checks and remaining external verification

Dedicated secret and dependency audits have now run. See [security-review.md](security-review.md)
and [release-verification.json](release-verification.json) for actual versions, scope,
findings and the PyTorch correction. The historical lock is intentionally preserved
with its disclosed advisories; new work uses the active release lock.
Hosted CI has passed for the linked source commit. Provider redistribution
permission remains separate; the release excludes the private provider bundle.

The source archive deliberately cannot reproduce all pilot model fits by itself:
that route requires the exact permitted private input bundle referenced by the
archived pilot seal. The public evidence verifier documents its narrower scope.
