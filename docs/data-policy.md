# Code, data and reproduction policy

The [MIT license](../LICENSE) applies to original project code and documentation. It does not grant rights to third-party source data, dependencies, issuer filings or other referenced materials. Follow each provider's current access and redistribution terms independently. Public accessibility is not a blanket redistribution license.

## What the public repository contains

- Original source, tests, configurations and dependency locks.
- Derived aggregate research tables, figures, committee materials and public verification evidence.
- A labelled synthetic demonstration generated locally by `make demo`.
- An original implementation snapshot in [experiments/pilot](../experiments/pilot/), including the frozen protocol and seal.

Raw provider responses, empirical observation panels, full local model runs, neural checkpoints, credentials and virtual environments are excluded. The repository's published results are research summaries; they should not be used to reconstruct or redistribute provider datasets. Source archives must be built from the reviewed Git tree, not by zipping an entire working directory.

## Three distinct reproduction claims

| Workflow | Public clone sufficient? | What it establishes |
|---|---|---|
| Synthetic demo and offline tests | Yes, subject to optional test dependencies | Deterministic core behavior and tested invariants on synthetic/fixture inputs |
| `make verify-results` | Yes; standard library only | Integrity and independently checked claims for the bundled derived evidence; the command reports its scope |
| Original empirical refit | No | Requires the exact original input bundle, original implementation and compatible locked environment |

The saved-result verifier is not a raw-data reconstruction, independent provider audit, full bootstrap regeneration or neural refit. Published-subset provenance records must be read as public copies, not as a claim that every original run artifact is distributed.

## Exact-input boundary

The original empirical input is pinned to `runs/live-20261013T183642.101447Z/`. Its `observations.csv`, `config.json`, `validation.json` and `manifest.json` are bound by the archived Phase 5 seal. Later workflows also pin earlier model-run artifacts through their configurations. These local inputs are absent from the release; obtain and share them only where provider terms and applicable permissions allow. A reviewer should not be expected to possess them merely to inspect the published findings.

The original seal and source are preserved in `experiments/pilot/`. Release engineering corrections are outside that original scientific snapshot. Restoring authorized data must match the recorded checksums; changing a hash or resealing after exposure would not reproduce the original experiment. Current `live` downloads can contain revisions and are new data snapshots, even for the same requested dates. Full refits belong in a separate private working directory restored from the archived implementation and exact bundle. Consult the historical [phase workflows](research-history.md) for their recorded requirements.

## Acquisition and credentials

FRED, Bank of Canada, ECCC, Yahoo and optional EIA adapters record their sources and timing assumptions; the later historical/vintage extensions add NOAA and ALFRED. Provider details are in the [data contract](data-contract.md) and [vintage guide](vintage-data.md). API keys are supplied through process environment variables and must not enter Git, public logs or archives. `.gitignore` is a convenience, not evidence that an arbitrary directory archive is safe to share.

The source package does not include the author's private workspace, downloaded audit document, caches or all local PDFs. The public committee memo and derived reports are included intentionally. No new redistribution rights for provider data are asserted by this release.
