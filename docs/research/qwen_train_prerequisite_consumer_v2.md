# Qwen training prerequisite consumer v2

## Purpose

Consumer v2 is a content-free, low-memory conjunction gate. It authenticates
the frozen v1 prerequisite consumers and the independent companion v2 overlay
without rewriting either source:

```text
authenticated frozen v1 AND authenticated companion v2
```

This closes only the request-local trigger *diagnostic* gap. It does not make
the dataset, model, or formal training release ready.

## Current decision

| Input | Authenticated state | Meaning |
|---|---|---|
| Frozen training prerequisite v1 | blocked | Formal artifacts remain unavailable. |
| Frozen toy prerequisite v1 | `pending_request_local_materialization` | The frozen bytes remain unchanged by design. |
| Companion v2 | `ready_diagnostic_only` | The request-2 trigger receipt is independently authenticated. |
| Protected source-ID inventories | 2/6 ready | Only `swebench_source` and `heldout` are ready. |
| Consumer v2 result | blocked | `training_authorized=false` and `formal_training_authorized=false`. |

The four unavailable inventory classes are `gold_partition`,
`partial_gold_export`, `legacy_heldout_cases`, and `synthetic_scaffold`.
Calibration is not treated as held-out data.

The effective logic is deliberately asymmetric:

```text
metadata_conjunction_verified = verify(frozen_v1) AND verify(companion_v2)

training_authorized =
    metadata_conjunction_verified
    AND all_six_source_inventories_ready
    AND zero_intersection_proof_ready
    AND formal_v3_snapshot_ready
    AND final_projector_ready
    AND generic_execution_contract_ready
    AND source_disjoint_manifest_ready
    AND formal_release_lock_ready
```

Today the first line can pass while the second expression remains false.
`ready_diagnostic_only` must never be promoted to training readiness.

## What v2 authenticates

Consumer v2 binds and rechecks:

- the frozen training-prerequisite v1 config and implementation;
- the frozen toy-prerequisite v1 config and implementation;
- the companion v2 config, schema, implementation, canonical manifest, and
  mandatory SHA-256 sidecar;
- the Producer companion release commit and exact Git blobs;
- the copied request-local trigger receipt and its mandatory sidecar;
- the companion's 2/6 inventory status and prohibited-claim fields;
- the requirement that Planner request-1 private KV is not reused as Expert
  request-2 KV.

Unknown, missing, partially overridden, or hash-drifted inputs fail closed.
Frozen v1 is evaluated as frozen code; companion v2 is a mandatory second
input, not a mutation or replacement for v1.

## Claim boundary

The authenticated trigger receipt proves only that a trigger span was derived
from one exact, fully chat-templated request-2 serialization and one complete
tokenization. It preserves the two-request protocol: validate and commit the
Planner scaffold, then re-encode that committed scaffold as Expert input.

The gate does **not** claim:

- training or formal-training authorization;
- a complete six-source inventory or zero intersection;
- numeric equivalence, quality, or formal thresholds;
- physical KV sharing, zero-copy, multistream execution, or full-generation
  KV reuse;
- any provider, model, GPU, or network execution.

## Frozen companion identities

| Artifact | SHA-256 |
|---|---|
| Companion config | `21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f` |
| Companion schema | `596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119` |
| Companion implementation | `dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e` |
| Companion manifest | `7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115` |
| Manifest sidecar, physical | `f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095` |
| Request-local receipt | `ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e` |
| Receipt sidecar, physical | `ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a` |

Producer companion release commit:
`2648129d599a5041100278cb04b12291ffd8a482`.

## Low-memory verification

Run from the repository root in the project Python environment. These commands
read metadata and authenticated hash-ID inventories only. They perform zero
provider requests, network requests, model loads, GPU requests, or protected
body reads.

```powershell
$env:PYTHONPATH = "src"

python scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --artifact fixtures/research/qwen_toy_prerequisite_companion_v2

python -m anchor_mvp.research.qwen_train_prerequisite_consumer_v2 `
  --config configs/research/qwen_train_prerequisite_consumer_v2.yaml

python -m pytest -q `
  tests/test_qwen_train_prerequisite_consumer_v2.py `
  tests/test_qwen_toy_prerequisite_companion_v2.py
```

The consumer command intentionally exits with code `2` while printing a
machine-readable blocked decision. At the current 2/6 inventory state, exit
code `0` would be a regression rather than success.

## Remaining formal-v3 gates

Formal training remains blocked until independently frozen, authenticated
artifacts exist for all of the following:

- the four missing body-free per-ID inventories and the six-source
  namespaced zero-intersection proof;
- the formal-v3 training snapshot;
- the final TaskBoard projector manifest;
- the generic execution contract;
- the source-disjoint manifest;
- the formal release lock;
- the trainable-base snapshot and tokenizer/base compatibility attestation;
- formal train/calibration/held-out partition identities and the remaining
  release bindings.

No real training, provider distillation, large-model load, physical KV/CUDA
test, or formal quality/performance evaluation is performed by this gate.

## Release discipline

This consumer is a research preflight only. It creates no tag or release and
does not authorize either. A tag or release must not be published without an
explicit later instruction and a fully satisfied formal release lock.
