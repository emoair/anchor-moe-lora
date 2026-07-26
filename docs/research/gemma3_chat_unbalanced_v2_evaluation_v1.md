# Gemma 3 chat unbalanced-v2 evaluation v1

Status: model-free contract complete; physical KV execution, model generation,
GPU use, and evaluation are not performed by this change.

## Dependency boundary

Both contracts accept only
`anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1`
with `status=passed`. The earlier unsharded `.v1` receipt is rejected. The
checked-in configs contain no producer candidate hash, old 44-file identity,
single `train/chat.jsonl` assumption, or unpublished sharded hash.

The evaluator additionally requires:

- a body-free multi-arm training run receipt and adapter inventory identity;
- a passed physical shared-prefix KV receipt;
- the same artifact-tree and logical shard-inventory identities across all
  receipts.

Each dependency receipt and its Draft 2020-12 schema are accepted only as
physical regular-file snapshots independently bound to configured SHA-256
identities. Parsed mappings are not an authentication input. The loader checks
path and descriptor identity before reading, reads once through the
authenticated handle, verifies byte count, UTF-8/LF form and SHA-256 before
JSON parsing, then repeats the path identity check to reject replacement
races. Consumer, training, trusted-lineage-context, and KV dependency
projections use closed exact-key contracts: top-level or nested additions,
omissions, type changes, schema-version drift, non-finite constants, duplicate
keys, or schema weakening are rejected.

All checked-in dependency identities are `pending`, so status and preflight
remain fail-closed until a release-reviewed consumer handoff binds them.

## Shared-prefix KV claim

The first physical probe compares:

1. isolated re-prefill for each branch; and
2. one adapter-off frozen-base prefill followed by an explicit local copy at
   the `route_plan_commit` boundary.

The only positive claim pre-registered by v1 is exact ordered retained-prefix
values and prefill-compute handoff. The following remain false:

- `shared_storage`;
- `zero_copy`;
- `full_generation_kv_shared`;
- `rdma`.

Object identity or pointer equality cannot promote any of those claims. Expert
tails are private and append-only.

Gemma 3 is frozen as 26 attention layers: 22 sliding-window layers with window
512 and four full-attention layers at zero-based indices 5, 11, 17, and 23.
For a boundary length `n`, a full layer retains `[0,n)` and a sliding layer
retains `[max(0,n-512),n)`. Synthetic cases cover 127, 511, 512, 513, and
1025 positions.

The physical receipt must report:

- two actually executed experiments: isolated re-prefill and
  one-prefill-plus-copy;
- exact retained-value digests for all 26 layers;
- isolated re-prefill versus one-prefill-plus-copy;
- exact per-arm first-token distribution and argmax digest operands; the
  equality diagnostics are recomputed from those operands and cannot be
  supplied as trusted booleans;
- unchanged source cache, explicit cache-miss rejection, all 24 registered
  cross-mutation rejections, and non-aliasing private tails;
- latency, copy cost, peak device bytes, and repetition count.

Complete lineage binds model, tokenizer, serialization, ordered prefix, token
order, positions, attention mask, RoPE, adapter lineage, adapter-off state,
route commit, plan commit, boundary commit, training run, adapter inventory,
and cache presence. Both experiments must bind the same authenticated source
prefix and full lineage. That lineage comes from an independent physical
trusted-context snapshot bound by its own SHA-256, not from either experiment
or the receipt under test. It additionally freezes template, attention
implementation, dtype, quantization, adapter-off policy, and layer-layout
identities. Synchronized mutation of both experiment lineages remains invalid
even if their boundary commits are recomputed. No raw token identifiers are
persisted.

## Evaluation matrix

The aggregate evaluator freezes 4,990 slots:

| Group | Records | Arms / routes | Slots |
| --- | ---: | ---: | ---: |
| Tool comparison | 400 | 3 | 1,200 |
| Planner comparison | 240 | 4 | 960 |
| Router | 80 train proxy + 20 eval proxy | 1 | 100 |
| Identity probes | 50 | 3 | 150 |
| Wrong-route proxy | 860 | 3 derangements | 2,580 |

Tool arms are base, Q-only, and Q+micro-O. Planner arms are base, Q-only,
O-only, and Q+micro-O. Identity uses base, correct route, and wrong route.
Wrong-route evaluation uses three deterministic derangements derived from
authenticated role labels.

Tool strata are seen-template interpolation (160), argument-composition proxy
(120), and cross-language-transfer proxy (120). Planner strata are task
classification, expert selection, and commit boundary (80 each).

## Aggregate-only receipts

Receipts may contain only counts, rates, deltas, confusion aggregates, and
performance aggregates. They reject sample bodies, prompts, targets,
references, raw token identifiers, record identifiers, and per-record arrays.
Every rate and router macro-F1 is recomputed from integer aggregates. All eight
comparison fields are then recomputed from those authenticated counts using
`decimal-string-quantized-1e-12-v1`: exact decimal division, `ROUND_HALF_EVEN`,
and fixed 12-place strings. Supplied deltas are never trusted. The `eval_proxy`
split is not held out, wrong-route deltas are not causal proof, and neither
quality nor generalization is declared validated.

## Model-free commands

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1 --status
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1 --preflight `
  --consumer-receipt <consumer-receipt.json> `
  --consumer-receipt-schema <consumer-receipt.schema.json> `
  --training-receipt <training-run-receipt.json> `
  --training-receipt-schema <training-run-receipt.schema.json> `
  --trusted-lineage-context <trusted-lineage-context.json> `
  --trusted-lineage-context-schema <trusted-lineage-context.schema.json>
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_generation_eval_v1 --status
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_generation_eval_v1 --preflight `
  --consumer-receipt <consumer-receipt.json> `
  --consumer-receipt-schema <consumer-receipt.schema.json> `
  --training-receipt <training-run-receipt.json> `
  --training-receipt-schema <training-run-receipt.schema.json> `
  --kv-receipt <kv-probe-receipt.json> `
  --kv-receipt-schema <kv-probe-receipt.schema.json>
```

`--execute` is only a reserved lazy interface. With no authenticated physical
backend it fails closed and does not import an ML runtime.

## Non-claims

This contract does not prove training quality, generalization, causal routing,
zero-copy cache sharing, RDMA, or formal/release readiness. It performs zero
model loads, GPU requests, provider requests, network requests, protected
reads, Gold reads, held-out body reads, dataset-body reads, and raw-token-ID
reads.
