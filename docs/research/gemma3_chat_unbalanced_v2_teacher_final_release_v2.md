# Teacher FINAL integrated controller and external release v2

## Purpose

This additive v2 closes a specific lifecycle gap in the Gemma 3 Chat
unbalanced-v2 distillation pipeline:

1. one process owns one `RuntimeSecretSlots` instance;
2. it runs `exact1 -> bounded15 -> bulk_c30`, with authenticated fallback to
   `bulk_c16` only for a signed provider rate-limit condition;
3. before the runtime HMAC slot closes, it authenticates the WAL, all three
   phase receipts, the 3,520 terminal job receipts, and the projected Teacher
   records;
4. it atomically creates an immutable Teacher FINAL *candidate*;
5. a different reviewer later authenticates that candidate and atomically
   publishes a separate release attestation, final manifest, and v2 consumer
   binding.

The frozen v1 controller, its config, and its schema remain byte-identical.
The v2 work is additive and does not change Producer FINAL source bytes.

## Why finalization is in the live process

The runtime receipt HMAC key exists only in the controller's memory slot. A
new process cannot recreate or authenticate that key without persistence,
which is intentionally forbidden. Therefore, candidate materialization must
finish before `RuntimeSecretSlots.close()`. Cross-process HMAC resume remains
unsupported. The external reviewer binds the already-authenticated receipt
bytes; it does not claim to recover or independently recompute the secret
HMAC.

## Candidate contract

The finalizer consumes the authenticated main-train projection and one
authenticated mixed router file:

- 3,440 five-expert records;
- 100 mixed planner/router rows are parsed, of which 80 train rows are emitted;
- the 20 router `eval_proxy` rows are parsed for split filtering and zero are
  emitted to Teacher train shards;
- 3,520 total accepted records;
- 20 identity bundles / 100 identity records.

The body-free Producer identity-probe inventory is bound by its authenticated
digest without reading its 50 eval-probe bodies
(`identity_eval_probe_body_reads=0`). This scoped statement does not claim that
the mixed router file's 20 eval rows were unread.

Every output record has exactly ten closed fields. Its joins use:

- `record_id_sha256 = sha256(UTF-8 source record ID)`;
- `source_content_sha256 = canonical_sha256(source messages)`;
- the authenticated source serialization identity;
- `task_bundle_sha256`;
- the original Producer asset;
- the validated Teacher target and its UTF-8 SHA-256.

The candidate contains ordered, bundle-boundary-preserving shards, mandatory
SHA-256 sidecars, a closed manifest, an output inventory, a same-process build
receipt, and a release request. Every payload is below 50 MiB. It remains:

- `candidate=true`;
- `final=false`;
- `training_authorized=false`;
- `formal_training_authorized=false`;
- `live_authorized=false`.

The candidate is not a consumer input.

## External release contract

The external v2 reviewer uses single-byte snapshots and terminal TOCTOU
rechecks to recompute:

- the complete candidate file/sidecar inventory and physical-tree digest;
- every record against the frozen Draft 2020-12 record schema;
- ordered record, target, source-join, shard, and logical-dataset digests;
- role counts and bundle boundaries;
- the identity subset in the projection's real domain: exactly 80 Air-sentence
  anchors across humor, serious, angry-style, and tool-call, grouped into 20
  bundles with one of each anchor role;
- bundle closure over those 20 IDs: exactly 100 physical Teacher rows, with the
  four anchors plus exactly one structured `review_audit` row per bundle. The
  review target is not required to repeat the Air sentence;
- Producer P/R parentage, release tree, and metadata-only physical-tree
  digest;
- integrated controller, base controller, batch, finalizer config,
  finalizer implementation, and all schema identities.

It then writes six files to an owned staging directory:

- `independent_release_attestation.json` and sidecar;
- `final_manifest.json` and sidecar;
- `teacher_alignment_binding.v2.json` and sidecar.

One OS-level no-replace directory rename publishes all six together. A
destination collision fails closed. Failed staging is preserved for forensic
inspection; recursive cleanup is not used.

The identity graph is deliberately acyclic. The attestation is produced
first. The final manifest binds the attestation and declares the binding
schema/path without embedding the binding hash. The terminal binding then
binds both preceding payloads. No document claims its own hash.

An external release means only that the data artifact is final and independently
reviewed. Consumer acceptance is still pending, so training, formal training,
live serving, quality, and generalization remain unauthorized.

The external reviewer does not claim to re-prove review identity correctness
from the structured review target alone. It binds that fact to the
same-process finalizer's closed `identity_audit`, the candidate manifest, its
build-receipt digest, and the authenticated body-free Producer identity-probe
inventory.

## Files and versions

- Integrated controller:
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_integrated_controller_v2.py`
- Same-process finalizer:
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_finalizer_v1.py`
- External reviewer:
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_v2.py`
- CLI wrapper:
  `scripts/data/review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py`
- Record candidate schemas:
  `configs/data/gemma3_chat_unbalanced_v2_teacher_final_*_v2.schema.json`
- External release schemas:
  `configs/data/gemma3_chat_unbalanced_v2_teacher_final_release_*_v2.schema.json`
  and
  `configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json`

All JSON, Python, tests, and docs in this contract are LF-bound through
`.gitattributes`. The final handoff must report physical SHA-256 values after
formatting and testing; transient hashes are not release identities.

## Reproduction

Model-free controller validation:

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  -m anchor_mvp.data.gemma3_chat_unbalanced_v2_integrated_controller_v2 `
  --validate-only
```

Independent candidate validation (no write):

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  scripts\data\review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py `
  --candidate data\gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1\candidate-<run-id> `
  --implementer-id producer.implementer `
  --reviewer-id independent.reviewer `
  --validate-only
```

Independent atomic release:

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  scripts\data\review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py `
  --candidate data\gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1\candidate-<run-id> `
  --implementer-id producer.implementer `
  --reviewer-id independent.reviewer `
  --publish
```

Focused verification:

```powershell
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe -m pytest -q `
  tests\test_gemma3_chat_unbalanced_v2_integrated_controller_v2.py `
  tests\test_gemma3_chat_unbalanced_v2_teacher_finalizer_v1.py `
  tests\test_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py
```

These commands do not load a model or GPU and do not make provider or network
requests. A real live run remains blocked until the independent consumer gates
and in-memory credential/HMAC slots are green.

## Compatibility and non-goals

The finalizer imports `datetime` and `timezone` and uses `timezone.utc`; it
does not import `datetime.UTC`, so the module parses and imports on Python
3.10. The integrated loader executes authenticated finalizer bytes under a
digest-qualified temporary module name, registers it only for the duration of
execution, verifies `__file__` and the raw digest, and removes the module cache
entry on success or failure.

This work does not:

- persist or expose credentials or HMAC keys;
- support cross-process HMAC resume;
- read identity eval-probe, heldout, Gold, or protected sample bodies; the
  authenticated mixed router file is explicitly parsed as 80 train plus 20
  `eval_proxy` rows, and those 20 rows are never emitted to train shards;
- authorize training or launch a model/GPU/provider request;
- claim quality, generalization, physical KV reuse, or a formal release.
