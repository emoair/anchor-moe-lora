# SWE-bench TaskBoard projector contract

[简体中文](swebench_taskboard_projector.zh-CN.md)

## Status and scope

The TaskBoard projector is a deterministic, post-Gold research transform. It
may run only after an `anchor.training-snapshot.v2` snapshot has been frozen
with its manifest and SHA-256 sidecar. Its output is a proxy dataset for the
query-specialization research track. It is not canonical Gold, is not a
distillation stage, and does not by itself promote any model or training run.

The projector reads only the formal `train` and `calibration` partitions. It
must not read or emit held-out record content. It must never modify the source
snapshot, its manifest, its SHA sidecar, or any canonical Gold file.

## Versioned contract

The fixed policy is
[`swebench_taskboard_projector_v1.yaml`](../configs/research/swebench_taskboard_projector_v1.yaml).
Every output JSONL row must validate against
[`taskboard_projector_sidecar.schema.json`](../configs/research/taskboard_projector_sidecar.schema.json),
and the output inventory must validate against
[`taskboard_projector_manifest.schema.json`](../configs/research/taskboard_projector_manifest.schema.json).

Each `anchor.swebench-taskboard-sidecar.v1` row is a provenance wrapper around
one strictly validated `anchor.query-specialization.v1` `training_record`.
The inner record retains its closed schema: `schema_version`, `id`, `pair_id`,
`variant`, `language`, `split`, `role`, `task_board`, `attention_targets`, and
`target`. Provenance is not added to that inner object. Instead, the wrapper
binds the source Gold record and file, frozen snapshot and manifest, source task
bundle, base TaskBoard, projector, config, and sidecar schema by identifiers and
lower-case SHA-256 values.

The wrapper's `id`, `pair_id`, `variant`, and `split` must equal the same fields
inside `training_record`; `stage` and `expert` must use this mapping:

| Canonical stage | Research expert |
| --- | --- |
| `planner` | `planner` |
| `tool_policy` | `tool_policy` |
| `domain_builder` | `frontend_gen` |
| `domain_review` | `frontend_review` |
| `security` | `security_gate` |

These cross-field equalities and the stage/expert mapping are semantic checks
performed by the projector in addition to JSON Schema validation.

## Split, visibility, and augmentation rules

The projector assigns the source task to `train` or `calibration` before it
creates any variant. All five stage views for a task stay in the source split.
The `train` split receives paired `clean` and `noisy` rows. The `calibration`
split receives `clean` rows only.

The checked-in policy and every output manifest explicitly fix
`split_group_key=task_bundle_sha256`,
`task_id_cross_binding_key=training_record.task_board.task_id`, and
`all_five_role_views_same_split=true`. A task-bundle digest is the canonical
SHA-256 of the source task ID plus the ordered five-stage source-record
bindings. Before publication, the projector rebuilds that digest from the
clean role views and verifies the forward and reverse bundle/task-ID mapping,
the unique split, the complete five-role set, and the train clean/noisy pair.

The only allowed noise policy is `stale_duplicate_overlay`. A noisy row copies
one or more blocks from the same source task into stale overlay blocks. It may
not import text, identifiers, or facts from another task. The wrapper records
the original block IDs and overlay block IDs in `augmentation`; the clean row
uses empty arrays. Augmentation never runs before splitting.

Causal visibility is fail-closed. A stage may consume only blocks visible to
its mapped expert and available at that stage. Future-stage or otherwise
forbidden blocks remain in the structured record only as negative supervision
and must not be rendered into the model-visible prompt. A block's
`commit_state` is never silently upgraded by projection.

## Implementation and scale boundary

Input and output files are authenticated from single in-memory byte snapshots:
SHA-256, byte length, record count, and JSON parsing all use the same bytes,
and the complete inventory is checked again immediately before atomic publish.
The current research implementation also materializes the authenticated Gold
and projected rows in memory. It is verified on the 15-record integration
fixture, but full-bank peak memory and output volume have not been qualified.
Run a measured shard/streaming pilot before projecting a large frozen snapshot;
do not treat this MVP as a full-bank throughput claim.

## Formal command

Use the frozen manifest SHA-256 as an explicit command argument. The snapshot
directory must contain the matching `manifest.json` and `manifest.json.sha256`;
it must not be a held-out source or output directory.

```powershell
python scripts/data/project_swebench_taskboard.py `
  --config configs/research/swebench_taskboard_projector_v1.yaml `
  --snapshot-dir <FROZEN_TRAINING_SNAPSHOT_V2_DIRECTORY> `
  --snapshot-manifest-sha256 <FROZEN_MANIFEST_SHA256> `
  --output-dir <NEW_RESEARCH_OUTPUT_DIRECTORY>
```

The run must fail closed if either frozen input binding is absent or mismatched,
if any source split is not `train` or `calibration`, if held-out content would
be read, or if an output row or manifest fails validation.

## Required manifest evidence

The output is exactly `train/clean.jsonl`, `train/noisy.jsonl`, and
`calibration/clean.jsonl`. The manifest binds the exact snapshot input,
producer identity and schema/config hashes (including the manifest schema's
own byte hash), all three files, the unique task-bundle count and task-ID
digest, and record counts by split, variant, canonical stage, expert, and
language. A valid manifest also proves these fixed invariants:

- `canonical_gold_written=false`
- `provider_requests=0`
- `heldout_content_read=false`
- `heldout_content_emitted=false`
- `split_preserved=true`
- `augmentation_applied_after_split=true`
- `claim_scope=research_proxy_only`

These facts establish only that the deterministic projection respected its
research boundary. Training promotion still requires a separate frozen
training manifest, unseen-task evaluation, causal deletion checks, distractor
invariance, strict-JSON metrics, cross-role separation, and the project's
normal review gates.

The subsequent hash-only generic execution, source-disjoint, and training
release freeze stages are specified in
[`swebench_training_release.md`](swebench_training_release.md). They cannot
publish a ready release lock until a real frozen projector manifest and a
bound consumer contract are available.
