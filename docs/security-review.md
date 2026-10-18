# Release security review

Reviewed 2026-10-17. These are bounded checks, not a security certification.

## Dependency audit and correction

`pip-audit 2.10.1` queried PyPI advisories for the complete 41-package ML lock
with `--no-deps --disable-pip`. That lock also includes every market-lock pin.
The initial scan reported two advisories for the pilot's `torch==2.10.0`:
[CVE-2025-3000 / GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
and [CVE-2026-4538 / GHSA-33x2-ppm4-v46v](https://github.com/advisories/GHSA-33x2-ppm4-v46v).
The first concerns `torch.jit.script`; the second concerns PT2 loading. Neither
API is used by this project. The only deserialization test loads a checkpoint
created in that same test, from memory, with `weights_only=True`.

The active release pins PyTorch **2.13.0**. Re-auditing the entire active ML lock
reported **zero known advisories**, with all 41 packages checked and none skipped.
This is an advisory-database result at review time, not a guarantee that all
possible vulnerabilities are absent. It also does not audit native operating
system libraries or arbitrary third-party model files.

The archived pilot retains its exact original PyTorch 2.10.0 lock and its two
reported advisories. It is an historical provenance artifact, not the recommended
environment for new work. The release dependency change was tested separately;
the empirical models were not refitted, results were not relabelled as produced
by 2.13.0, and the original seal was not changed. Public demonstration and result
verification need no PyTorch. Follow
[PyTorch's security guidance](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
when handling model artifacts; this release does not require untrusted checkpoints.

## Secret scanning

Gitleaks **8.30.1** was downloaded from its official release and checked against
the published archive SHA-256. Default rules scanned the candidate source files
and all reachable Git refs with output redaction. The pre-release history scan
covered the original Phase 1 commit and found no matches.

The initial candidate-file scan reported seven `generic-api-key` matches. Every
match was the same public SHA-256 for `phase-5-first-access.json`, verified against
that JSON file. They were reviewed as checksum false positives, not suppressed
by a broad path or secret-rule exclusion. Final candidate and history scan counts
are recorded in [release-verification.json](release-verification.json).

Provider inputs, credentials, environments and private scratch files are excluded
from the source package. Scan coverage is limited to candidate tracked files and
reachable history; it is not a scan of the author's private workspace or ignored
provider cache. A clean scan does not confer data-redistribution rights.

## Repeating the checks

Pre-merge refresh on 2026-10-18: Gitleaks 8.30.1 scanned all three reachable
commits through source commit `3f64cda`. Its seven matches were checked against
the actual historical file lines and the SHA-256 of the public first-access
audit JSON; all seven are the same documented checksum false positive.
There were no confirmed credentials. A fresh pip-audit 2.10.1 PyPI query checked
all 41 active-lock packages, with zero known advisories and none skipped.
The archived pilot lock and its disclosed advisories remain unchanged.

Use a disposable tool environment and the official tools. From a reviewed checkout:

```sh
gitleaks git --redact --log-opts="--all" .
pip-audit --no-deps --disable-pip -r requirements-ml.lock.txt
```

For a directory scan, first export the intended Git tree and scan that directory;
do not scan or upload the private working directory as the release artifact.
Review checksum matches against the actual referenced file. Keep scanner reports
redacted and share only the review summary, tool versions, input hashes and scope.
The [hosted source-commit run](https://github.com/lukelu389/weather-risk-forcaster/actions/runs/39038470454)
passed all four jobs, checked 2026-10-18, including a fresh active-lock installation
and 329 research tests with zero skips. Confirm CI on the latest PR head before merging.
