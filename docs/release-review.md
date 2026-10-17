# GitHub release review

Prepared 2026-10-17 from the supplied GitHub audit. Its suggestions were reviewed
against the code and evidence; the audit was not treated as permission to alter
the original study, publish private data, or claim an investment result.

The release is a recruiter-facing **qualified research pilot**. Its evidence is
the comparison design, tested timing and scoring rules, preserved negative
findings, and explicit limits. The 2024–2025 final period remains consumed.

## Changes and audit disposition

| Audit | Release action | Scientific boundary |
|---|---|---|
| F01: only Phase 1 in Git | Curated all completed source, tests, configurations, reports and notebooks in an isolated release branch | Original research workspace retained |
| F02: directory archive includes private state | Git-tree-only packaging, private-path rejection and SHA sidecar | Only committed public material is packaged |
| F03: private inputs needed for refit | Public categorical evidence and independent standard-library coverage verifier | Full empirical refit still needs the exact permitted input bundle |
| F04: partial dependencies fail discovery | EDA tests guard NumPy, pandas and Matplotlib; core, partial and full CI jobs | Full job fails on any skipped test |
| F05: latest-vintage history | Qualification on the first README screen and in limitations/data policy | Verified historical vintages require a new study |
| F06: retrospective horizons | Explicit quote-grid/horizon limitation and prospective next gate | Targets and published primary scores unchanged |
| F07: tail adequacy and persistence | Documented coverage failures, clustering, ES direction and 36/150 capped fits; one prespecified heavier-tail challenger proposed | No challenger selected or fitted on the consumed holdout |
| F08: sparse tails and bootstrap resolution | Retained original and 19,999-draw supplementary results, family sizes and non-equivalence warning | Sensitivity remains explicitly post-exposure |
| F09: narrow weather proxy | Precise Toronto/Calgary observed HDD/CDD scope | No claim that weather is generally irrelevant |
| F10: economic interpretation | Primary-source Fortis weather-to-cash-flow diligence bridge and clearly hypothetical arithmetic | No issuer valuation, alpha, executable futures backtest or compounded NAV claim |
| F11: absent-period validation | Independent configurable coverage ceiling, required failure/optional warning, finite-range validation | Current live/history use 12%; archived configuration untouched |
| F12: empty startup run folders | Metadata preflight before output-directory creation; initial artifact writes included in failure-recording lifecycle for EDA/baseline/ML/final runners | No empirical rerun or new final access |
| F13: reviewer navigation | Short README, visible result/figure, offline commands, recruiter guide, historical workflow appendix | Detailed phase evidence remains accessible |
| F14: rights, privacy and security | MIT code license, separate data policy, sanitized public manifests, actual secret/dependency audits; active PyTorch update | Original source seal and vulnerable historical dependency lock retained as provenance |

The coverage policy uses whole-range approximate weekdays, weeks or months. A
12% absent-period allowance tolerates ordinary calendar closures but is not an
exchange-calendar certification. Explicit nulls retain their separate threshold.
There are regression tests for a one-row month, optional-series warnings,
legitimate closure allowance and invalid configuration. New lifecycle tests cover
missing package metadata, initial-write failures and omitted private inputs.

## Verification and reproducibility

[release-verification.json](release-verification.json) records the actual local
test environments/counts, public-evidence verification and security findings.
The release passes all 329 tests with zero skips, adding 25 tests to the prior 304-test suite. Core discovery passes 173 entries with 12 optional-module skips; NumPy/pandas-only discovery passes 196 entries with 10 skips. Two independent synthetic runs match all five deterministic artifacts. No neural training result
or headline calibration table was regenerated to improve a finding.

The public verifier recomputes 54 coverage rows from 27,882 categorical events,
checks preserved source and public artifact hashes, and recomputes saved-p-value
Holm adjustments. It does not regenerate ES bootstrap draws or validate private
realized prices. [Evidence documentation](../evidence/README.md) specifies this
boundary and the publication transformations.

The original protocol seal and first-access record remain byte-identical.
`experiments/pilot/` preserves the sealed implementation. Corrected active source
and PyTorch 2.13.0 belong to the release, not the original experiment. The two
public notebook files, seven PNG figures, numerical tables and committee PDF
remain the original scientific/communication outputs.

## Remaining external actions

The package can be reviewed locally without credentials. Push the prepared branch
when ready and check all hosted CI jobs before advertising a passing CI badge.
No push or public hosting is part of this local preparation. Any private input
sharing requires the relevant provider rights. GPU work, new vintages, revised
targets and empirical futures testing belong to separately authorized future
research, not this release.

## Three interview questions

1. What does the public verifier establish, and what requires private input data?
2. Why were the archived implementation and post-exposure fixes kept separate?
3. Which result would change a financial decision, and what evidence is missing
   before using this pilot for a risk limit or issuer valuation?
