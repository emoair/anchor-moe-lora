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
`anchor.swebench-source-disjoint-manifest.v2`. It binds the frozen training
snapshot, final `anchor.swebench-taskboard-projector-manifest.v2`, sidecar and
segment-plan schema byte hashes, verifies the three projected JSONL files, and
emits only split counts and hashes. Its Hierarchical Task-KV summary must match
the authenticated projector summary. The held-out count and canonical cases
digest come from a separately hash-pinned held-out metadata manifest; body,
content, path, file, record, prompt, and label fields are rejected.

Verification is semantic, not line-count-only. The freeze layer authenticates
the local v2 config and all three projector schema byte hashes, validates every
outer sidecar and segment plan, rechecks clean/noisy pairing, five-role bundle
membership and shared chains, and recomputes all manifest counts and unique
segment identities from the fixed files. It also cross-binds the projector's
logical snapshot SHA and mandatory sidecar-byte SHA to the loaded snapshot
artifact.

`release` freezes `anchor.generic-train-release-lock.v2`. It binds the final
projector, source-disjoint artifact, generic execution artifact, external
`anchor.swebench-training-consumer-interface.v2` contract, execution lock, and
the fixed files:

- `train/clean.jsonl`
- `train/noisy.jsonl`
- `calibration/clean.jsonl`

The release requires `task_bundle_sha256` split grouping, the inner
`training_record.task_board.task_id` cross-binding, all five role views,
outer-sidecar provenance, `calibration_is_heldout=false`, and
`claim_scope=research_proxy_only`.

## Hierarchical Task-KV release boundary

The v2 source-disjoint artifact, consumer contract, and release lock all bind
`anchor.hierarchical-task-kv-segment-plan.v1` plus its exact schema SHA-256.
They accept only `decoupled_frozen_prefix_producer_required`; a Q-only adapter
or naive in-stack Q-LoRA cannot claim exact shared KV. Exact reuse is limited
to `identical_ordered_prefix_lineage_only`, so independently encoded segment
KV may not be concatenated and treated as an exact prefix.

Cache identity is fail-closed. Model architecture, tokenizer, token order,
position IDs, RoPE configuration, KV-producing weights, and prefix lineage
must all match exactly. An unknown field or any mismatch yields
`cache_incompatible`. The release metadata fixes
`full_generation_kv_shared_claimed=false` and
`token_level_moe_claimed=false`; this pipeline describes hierarchical shared
prefixes plus private increments, not all-generation KV sharing or token-level
MoE.

The consumer, source-disjoint artifact, and release lock retain the complete
constraint set, including strict five-role shared-prefix intersection, ordered
lineage, no independent segment concatenation, no shared-then-mask insertion,
expert-private target deltas until explicit committed downstream promotion,
and `current_target_segment_emitted=false`.

These stages still emit metadata only. They do not serialize tensor/KV
payloads, read held-out bodies, authorize training, or turn a synthetic
projector manifest into a formal release. A v2 lock remains unavailable until
the final frozen projector manifest, authenticated v2 consumer contract, and
all other expected SHA-256 inputs exist and pass their exact schemas.

Use `scripts/data/freeze_swebench_training_release.py --help` for the command
arguments. No command may infer or discover a missing expected SHA.
