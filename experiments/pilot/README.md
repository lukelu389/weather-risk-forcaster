# Original pilot implementation

This is a byte-preserved snapshot of the source, configurations, dependency
specifications and Phase 5 protocol records used before the GitHub release fixes.
It preserves the original implementation, including limitations and defects
subsequently corrected in the repository's active `src/` tree. Use active source
for software review and synthetic demonstrations; use this snapshot to identify
what produced the reported empirical results.

`docs/phase-5-seal.json` binds 24 files. Twenty source/configuration/protocol files
are included and checked by `make verify-results` from the repository root. Four
private provider-input files under `runs/live-20261013T183642.101447Z/` are omitted.
The seal SHA-256 is
`18a989f48d2f993e52f34c0e3c2d2840047ea59e214f54ff30a5e5cc52d8e261`.
Additional copied source/configuration files are historical context and are not
all covered by that original 24-file seal; the public evidence manifest hashes
the complete distributed snapshot.

This directory alone cannot reproduce empirical model fits. A permitted full
reproduction must restore the exact input bundle and any earlier run artifacts
referenced by the preserved configurations in a separate private working copy,
verify their original hashes, then use the compatible locked environment.
The [historical workflows](../../docs/research-history.md) record the commands.
Do not redirect the archived protocol at newly downloaded data or modify its
hashes. Current release fixes are disclosed post-exposure changes, not a new
sealed evaluation; the 2024–2025 final period remains consumed.
