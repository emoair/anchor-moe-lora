# Gemma 3 unbalanced-v2 shared-prefix KV runtime v2

## Scope

This additive diagnostic implements the first executable boundary-switch probe
for the unbalanced-v2 planner/expert design. It does not authorize training,
formal evaluation, release, or live serving.

The runtime consumes only the frozen sharded-v3 Producer lineage through
`gemma3_chat_unbalanced_v2_runtime_source_v2`. That source module owns immutable
Git/blob authentication, the checked-in consumer preflight, and the single
authenticated SentencePiece snapshot. This runtime never invokes an HF
tokenizer and never performs a second `tokenizer.model` path read.

The only accepted training receipt is:

- schema:
  `anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-run-receipt.v2`
- status: `passed`
- mode: `full`
- phase/adapters: the exact nine trained arms in frozen order

Smoke receipts and the superseded multiarm-v1 receipt are rejected. The v2
receipt must bind one frozen base-file inventory, one identical Q8 SCB identity
across all nine phases, and nine physically authenticated adapter trees.

The same training receipt must also bind the one independently released
Teacher v2 identity: binding, manifest, record schema, release receipt,
external release attestation, shard inventory and public identity, together
with the frozen body-free counts. The KV runtime never rereads teacher rows.
Teacher v1 and temporary, mutually inconsistent v2 candidate names are not
consumable. The checked-in Teacher fields intentionally remain `pending` until
Producer returns one signed FINAL v2 handoff, so physical execution is
fail-closed.

## Boundary semantics

The handoff has three ordered stages:

1. one real, adapter-off frozen-base prefill for the authenticated ordered
   prefix;
2. one selected expert-planner private tail;
3. one selected output-expert private append-only tail.

The prior cache is the cache actually produced by the adapter-off prefill. It
is copied into a private branch and passed across the explicit adapter switch;
the new adapter does not recompute the preceding prefix. Source and sibling
caches must remain byte-value unchanged.

The boundary commit binds Producer P/R/tree, source blob and record, training
serialization, authenticated per-example serialization, tokenizer and model
config, token order, positions, mask, RoPE, model/base/Q8-SCB/adapter lineage,
derived layer topology, and route/plan commits.

The model topology is derived from the authenticated Gemma config at execution
time. A passing receipt requires exactly 26 layers, comprising 22 sliding
attention layers and 4 full-attention layers, and records a value digest for
each layer.

Exact handoff is verified after both adapter boundaries, not inferred from two
matching prefills. For every layer, the runtime slices K and V on sequence axis
`-2` from the planner cache after its one-token tail and from the output cache
after its two-token cumulative tail. Each branch slice must equal the matching
suffix of the original adapter-off source cache. Full-attention layers retain
the entire source prefix. Sliding-attention layers follow the authenticated
Transformers cache rule and retain a suffix capped at `sliding_window - 1`
(511 tokens for the frozen 512-token window). The receipt records the source,
branch and retained-prefix counts plus separate K/V digests at both boundaries.

## Claim limits

A physical pass may claim only:

- exact retained prefix K/V equality at the planner and output boundaries;
- avoided prefill recomputation at the adapter-switch boundary.

It never claims shared storage, zero-copy, RDMA, or full-generation KV sharing.
`DynamicCache` may concatenate and reallocate tensors, so pointer equality is
recorded only as an observation and is never proof.

Latent packets are disabled. If a future version enables a soft prefix, that
version must freeze dimensions, serialization, position and commit digests and
must not describe the packet as hidden chain of thought.

## Fail-closed execution

`--status`, `--validate`, and `--dry-run` are model-free. `--execute` remains
blocked until the config binds physical full-run receipt, schema, sidecar and
model identities and explicitly sets `execute_authorized=true`.

An execute run additionally requires:

- local-only model loading;
- an exclusive GPU lock;
- authenticated base files and nine adapter trees before and after execution;
- matching Q8 SCB state before and after inference;
- cold-recompute versus handoff logits/argmax comparison;
- source/sibling-cache isolation and private-tail checks;
- negative rejection for adapter, position and mask lineage drift;
- an atomic create-once receipt.

Existing receipt targets are never overwritten. Lock and staging cleanup is
conditioned on the file identity created by this process. Raw prompts, token
IDs, logits and cache values are never written; only hashes, booleans, counts,
latencies and peak-memory scalars are allowed in receipts.

The checked-in config is intentionally model-free and execution-blocked. No
GPU, model, provider or network action is performed by its validation tests.
