# Tokenizer-bound long-context token inventory

[简体中文](swebench_long_context_preflight.zh-CN.md)

## Status and scope

This producer creates a tokenizer-bound token inventory downstream of an
authenticated `anchor.swebench-taskboard-projector-manifest.v2`. It never emits
an estimated-token proxy. Local mode publishes exact counts for its bound local
tokenizer. Synthetic mode publishes only a deterministic contract fixture whose
counts are exact for the declared synthetic tokenizer. It is not a formal exact
Gemma or target-model inventory. If the selected tokenizer identity is
incomplete, the run fails before publication and emits no null inventory.

The checked-in contract is
[`swebench_long_context_preflight_v1.yaml`](../configs/research/swebench_long_context_preflight_v1.yaml).
Despite the historical filename, its schema identity is
`anchor.long-context-token-inventory-config.v1`. Output records validate
against
[`swebench_long_context_preflight_sidecar.schema.json`](../configs/research/swebench_long_context_preflight_sidecar.schema.json)
as `anchor.long-context-token-inventory.v1`; the inventory manifest validates
against
[`swebench_long_context_preflight_manifest.schema.json`](../configs/research/swebench_long_context_preflight_manifest.schema.json)
as `anchor.long-context-token-inventory-manifest.v1`. Both schemas are closed
and contain no remote references.

## Authenticated projector input

The producer accepts only a complete projector v2 directory with an exact
`manifest.json`, mandatory `manifest.json.sha256`, authenticated projector
manifest/config/sidecar/segment-plan schema hashes, and these three fixed
partitions in order:

- `train/clean.jsonl`
- `train/noisy.jsonl`
- `calibration/clean.jsonl`

Each source JSONL line is authenticated from its raw UTF-8 bytes excluding the
line-feed delimiter. The record stores `source_line_sha256`; it must not parse
and canonically reserialize the row to invent a different line identity. The
manifest binds the SHA-256, byte length, and record count of all three source
partitions and of all three output partitions. Publication also requires a new
mandatory `manifest.json.sha256` and a final input inventory recheck.

The split remains grouped by `task_bundle_sha256`; `task_id_sha256` is the
SHA-256 of the exact UTF-8 bytes of
`training_record.task_board.task_id`. All five roles remain in one split and
the source split precedes augmentation.

## Mandatory tokenizer binding

Exactly one backend must be declared:

- `local_offline_tokenizer`, backed by local authenticated assets; or
- `explicit_synthetic_tokenizer`, named as synthetic and used only for a
  deterministic fixture.

The manifest must bind all of the following non-null values:

- `tokenizer_id` and `tokenizer_revision`;
- `tokenizer_assets_sha256` and `tokenizer_runtime_sha256`;
- `chat_template_sha256`;
- `serialization_policy_sha256`;
- `special_token_policy_sha256`.

`tokenizer_label_source=caller_supplied_and_hash_bound` makes the provenance of
the human-readable tokenizer label explicit: the caller supplies the label,
while the manifest hash-binds it to the authenticated tokenizer assets and
runtime. The label alone is not a model-identity assertion.

Network access is forbidden. A tokenizer name without revision/assets/runtime
hashes is insufficient. A synthetic backend must never be presented as the
Gemma tokenizer or used as formal model evidence.

The backend determines the complete publication mode:

| Backend | `inventory_mode` | `status` | `claim_scope` | Target-model match |
| --- | --- | --- | --- | --- |
| `explicit_synthetic_tokenizer` | `synthetic_fixture` | `synthetic_fixture_inventory_ready` | `synthetic_fixture_contract_only` | `not_applicable` |
| `local_offline_tokenizer` | `local_exact_tokenizer` | `exact_token_inventory_ready` | `exact_bound_tokenizer_inventory_only` | `consumer_verification_required` |

The manifest also fixes `synthetic_fixture_only=true` for the synthetic backend
and `false` for the local backend. Even local exact mode proves counts only for
the bound tokenizer. A consumer must verify that tokenizer identity against the
actual target model before treating its buckets as target-model buckets.

## Exact record contract

Each inventory record carries only identifiers, hashes, integer counts,
bucket/gate state, and false authorization claims. It binds the task and role,
raw source line, exact segment plan, ordered segment-ID digest, terminal prefix
lineage, and the canonical hash of the global tokenizer binding. It emits no
segment array.

For all metadata-only object hashes, canonical JSON means UTF-8 JSON with
lexicographically sorted keys, no insignificant whitespace, and non-ASCII
characters left unescaped. `segment_plan_sha256` hashes the closed frozen plan,
`ordered_segment_ids_sha256` hashes the ordered segment-ID array, and
`tokenizer_binding_sha256` hashes the exact manifest tokenizer-binding object.

The producer serializes only the causally selected segment plan. Current,
future, and forbidden blocks are excluded before serialization. Inventory
records must not contain fields or values carrying source text, previews,
rendered inputs, targets, token ID arrays, or held-out material.

The semantic invariants are:

```text
shared_prefix_input_tokens + private_delta_input_tokens = input_tokens
reserved_output_tokens = 4096
total_tokens = input_tokens + reserved_output_tokens
```

The retained field name `shared_prefix_input_tokens` is a consumer-contract
label. Its value covers every ordered non-private segment in the visible chain:
both strict task-shared prefix segments and causally committed downstream
task-shared immutable segments. It must not be interpreted as saying that every
counted token belongs to the strict all-five-role shared-prefix intersection.

The authenticated serialization and special-token policies define how wrapper
and special tokens are attributed between the two scope counts. Token counts
come from the declared exact backend, not a UTF-8 byte heuristic.

The tokenizer identity does not establish a reusable model KV identity.
Therefore every record fixes `cache_identity_status=identity_unbound` and
`reuse_savings_tokens=0`.

## Buckets and gates

Buckets apply to `total_tokens` of the bound tokenizer, using inclusive upper
bounds. The manifest records this as
`bucket_basis=bound_tokenizer_total_tokens`:

| Bucket | Total tokens | Gate |
| --- | ---: | --- |
| `le_8k` | 1–8,192 | `measurement_candidate` |
| `le_16k` | 8,193–16,384 | `measurement_candidate` |
| `le_32k` | 16,385–32,768 | `measurement_candidate` |
| `le_64k` | 32,769–65,536 | `measurement_candidate` |
| `le_128k` | 65,537–131,072 | `measurement_candidate` |
| `le_256k` | 131,073–262,144 | `capability_only` |
| `le_1m` | 262,145–1,048,576 | `research_only_blocked` |
| `gt_1m` | greater than 1,048,576 | `reject` |

`measurement_candidate` is not an evaluation pass. `capability_only` requires
separate verified model/runtime support. `research_only_blocked` cannot enter
allocation or execution, and `reject` is terminal.

A synthetic bucket is only a bucket under the declared synthetic tokenizer; it
is not a Gemma or target-model bucket. A local-tokenizer bucket likewise remains
consumer-unverified until the consumer proves the bound tokenizer is the target
model tokenizer.

## Caller-supplied model metadata

The manifest repeats, but does not verify, the caller-supplied Gemma4 12B
description: 48 text layers, 40 sliding-attention layers, 8 global-attention
layers, sliding window 1,024, and
`max_position_embeddings=262144`. It must state
`architecture_verified_by_preflight=false`. Exact tokenization does not prove
model architecture, memory capacity, RoPE behavior, KV compatibility, or
long-context quality.

## Manifest evidence and non-authorization

The manifest binds the projector identity, the three source partitions, this
producer/config/record/manifest schema, the complete tokenizer identity, the
model metadata statement, all three inventory files, and counts by split,
variant, stage, expert, bucket, and gate. It also records total rows, unique
task bundles, the task-ID digest, segment references, and unique segment IDs.

Every record and manifest fixes `provider_requests=0`,
`evaluation_status=not_evaluated`, `quality_validated=false`,
`allocation_validated=false`, and `execution_authorized=false`. The manifest
also fixes `capability_validated=false`, `canonical_gold_written=false`,
`approximate_inventory_emitted=false`, and `null_inventory_emitted=false`.
Its backend-bound mode, status, claim scope, target-model match state, and
`synthetic_fixture_only` flag are mandatory machine-checkable evidence. Neither
mode authorizes evaluation, allocation, execution, or a target-model capability
claim.
