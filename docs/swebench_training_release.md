# SWE-bench training release freeze interface

This producer layer is downstream of canonical Gold and the TaskBoard
projector. It freezes metadata bindings only; it does not call a provider,
modify Gold, read held-out case bodies, or start training.

Every produced artifact is a new directory containing exactly the canonical
entry files `manifest.json` and `manifest.json.sha256`. The sidecar form is
strictly:

```text
<lowercase-sha256>  manifest.json
```

All input identities are supplied as expected SHA-256 values. Files are read
through an inode/stat-bound bytes snapshot, inputs are rechecked before
publication, and the output directory is published atomically. Missing,
stale, overlapping, symlinked, or not-ready inputs fail without leaving a
ready artifact.

## Stages

`generic` freezes a sanitized `anchor.generic-train-execution-contract.v1`
from an offline coordinator preflight, execution lock, execution attestation,
coordinator config, and public source-bank manifest. It requires the generic
train gate to be ready while retaining the separate
`not_official_swebench_pass` boundary.

`source-disjoint` freezes
`anchor.swebench-source-disjoint-manifest.v1`. It binds the frozen training
snapshot and final projector manifest, verifies the three projected JSONL
files, and emits only split counts and hashes. The held-out count and canonical
cases digest come from a separately hash-pinned held-out metadata manifest;
body, content, path, file, record, prompt, and label fields are rejected.

`release` freezes `anchor.generic-train-release-lock.v1`. It binds the final
projector, source-disjoint artifact, generic execution artifact, external
consumer contract, execution lock, and the fixed files:

- `train/clean.jsonl`
- `train/noisy.jsonl`
- `calibration/clean.jsonl`

The release requires `task_bundle_sha256` split grouping, the inner
`training_record.task_board.task_id` cross-binding, all five role views,
outer-sidecar provenance, `calibration_is_heldout=false`, and
`claim_scope=research_proxy_only`.

Use `scripts/data/freeze_swebench_training_release.py --help` for the command
arguments. No command may infer or discover a missing expected SHA.
