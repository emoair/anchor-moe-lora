# RFC: Producer-native hierarchical Task-KV metadata

Status: producer-v2 control-plane MVP
Branch: `research/neural-swarm-kv`
Outer sidecar: `anchor.swebench-taskboard-sidecar.v2`
Segment plan: `anchor.hierarchical-task-kv-segment-plan.v1`

## Scope and single source of truth

The SWE-bench TaskBoard projector v2 emits one authenticated `segment_plan`
inside every outer sidecar at `outer_sidecar.segment_plan`. That producer-native
plan is the only current Task-KV plan. Consumers must not derive, merge, or
silently translate it into the retired `anchor.taskboard-kv-segment-plan.v1`
shape.

The normative files are:

- [`swebench_taskboard_projector_v2.yaml`](../../configs/research/swebench_taskboard_projector_v2.yaml),
  the current producer policy;
- [`taskboard_projector_sidecar.schema.json`](../../configs/research/taskboard_projector_sidecar.schema.json),
  the closed outer-sidecar envelope;
- [`hierarchical_task_kv_segment_plan.schema.json`](../../configs/research/hierarchical_task_kv_segment_plan.schema.json),
  the independently authenticated nested plan schema;
- [`taskboard_projector_manifest.schema.json`](../../configs/research/taskboard_projector_manifest.schema.json),
  the producer manifest contract;
- [`swebench_source_disjoint_manifest.schema.json`](../../configs/research/swebench_source_disjoint_manifest.schema.json),
  the source-disjoint release evidence; and
- [`generic_train_execution_contract.schema.json`](../../configs/research/generic_train_execution_contract.schema.json),
  the generic execution-envelope contract; and
- [`generic_train_release_lock.schema.json`](../../configs/research/generic_train_release_lock.schema.json),
  the downstream training-release lock.

The older `swebench_taskboard_projector_v1.yaml` remains historical only. A v2
fixture or release must not cite it.

The frozen producer-v2 hashes are:

| Artifact | SHA-256 |
| --- | --- |
| Projector config | `b36945a2693183f0b213da403afcf8bb5611f46298bb849434e7b7d5854ba943` |
| Outer sidecar schema | `c1863bfab69ce2f2388ee37fadae951b14f3d5120706bab032cab3f9aab6bdc5` |
| Native segment-plan schema | `80f760497e0d21f7d4d532db758362a800e845e6919b18b23958caabc7f155bf` |
| Projector manifest schema | `2cd9dc98d2b2865ed0586abfe291e3f6d161686597fcd2a7884c5762d2195347` |
| Source-disjoint schema | `2a2aae532c25b324a96b929a6a396d55d051c765258a5da0ebb7547724c68f6b` |
| Generic execution schema | `63c699fdb7932b9fe1593b044d6a588bb4234b349816ce70f65568ab7b0f0b3a` |
| Training release-lock schema | `119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa` |
| Fixture manifest | `595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac` |

The consumer CLI pins
[`hierarchical_task_kv_mvp.yaml`](../../configs/research/hierarchical_task_kv_mvp.yaml)
at SHA-256 `f695e02cd2da8ca9c8d40fc99a0c33a4803b23dbc5e7f4cf296d40156315252d`.

Partition counts and hashes are authenticated by that fixture manifest and are
not duplicated here.

## What the plan is, and is not

The plan contains ordered segment IDs, content hashes, ownership, visibility,
commit state, dependencies, and lineage hashes. It contains no prompt body,
answer body, token IDs, tensor, device pointer, KV page, or model weight.

Every current plan fixes:

- `execution_mode: decoupled_frozen_prefix_producer_required`;
- `materialization: metadata_only_no_tensor_or_kv`;
- `full_generation_kv_shared_claimed: false`; and
- `token_level_moe_claimed: false`.

Its cache compatibility is deliberately `identity_unbound` with
`cache_reuse_allowed: false`. Therefore the logical plan does not claim exact
runtime reuse, CUDA overlap, reduced memory, lower latency, higher throughput,
preserved quality, or production readiness.

## The CPU-cache analogy is only a lifecycle analogy

Task-shared prefixes, downstream immutable snapshots, and expert-private
deltas resemble shared and private cache lifetimes. That analogy applies to
ownership, visibility, fork, publication, and copy-on-write. It does not make a
Transformer attention stack equivalent to a CPU cache hierarchy.

A cache address or lookup key is not the Transformer K tensor. Those terms must
not be interchanged in code, schemas, metrics, or reports.

## Why ordinary Q-LoRA cannot make full-stack KV sharing exact

In a normal decoder, an in-stack Q-LoRA changes attention output, which changes
the residual hidden state and the later layers' K/V inputs. K-LoRA or V-LoRA
changes the cached projections directly. Consequently:

- Q specialization alone is not sufficient for exact full-stack sharing;
- naive in-stack Q-LoRA exact reuse is forbidden; and
- K/V-adapted paths are incompatible with an unchanged shared K/V page.

An exact future design requires a frozen, expert-independent prefix producer
and a decoupled expert query/cross-attention/readout path. Expert generation
then forks private delta state. The producer plan records that architecture
requirement; it does not implement or benchmark it.

## Causal segment construction

The producer first splits source tasks, then creates clean/noisy views. It never
uses a noisy view to decide a split. Within one source task it builds five role
views for `planner`, `tool_policy`, `frontend_gen`, `frontend_review`, and
`security_gate`.

The shared prefix follows
`membership_rule: strict_all_five_role_visibility_intersection`. Blocks are
serialized in TaskBoard causal order and linked into one prefix chain.
Independent block encoding followed by arbitrary KV concatenation is forbidden.

The three cache scopes are:

- `task_shared_prefix`: committed task-source blocks visible to all five roles;
- `downstream_task_shared_immutable`: an earlier expert output that was
  explicitly committed and is causally visible to the downstream role; and
- `expert_private_delta`: candidate or verified role-private context visible
  only to its owning expert.

The current target answer is never emitted as an input segment. Current and
future forbidden blocks may physically remain in the TaskBoard source object,
but they may not be preinserted and masked later. `shared_then_mask_allowed` and
`forbidden_current_future_preinsert_allowed` are both false.

New expert output starts private. Promotion requires an explicit commit and
downstream causal visibility; a model output cannot grant itself shared scope.

## Content addressing and lineage

For one segment, the producer computes:

```text
segment_id = "task-kv-segment-v1:" + sha256(canonical_json({
  task_bundle_sha256,
  source_block_id,
  content_sha256,
  producer_role,
  cache_scope
}))
```

The root lineage binds the task bundle, execution mode, and
`ordered_prefix_genesis`. Each next prefix lineage binds the prior lineage,
segment ID, serialization order, and original TaskBoard causal order. A segment
also lists every preceding segment ID as a dependency. Equal unordered block
sets are therefore not equal prefixes.

Each plan additionally binds task/bundle/base-board identity, producer and
schema identities, source Gold/file/snapshot identities, split, stage, expert,
and variant. Consumers must cross-check those bindings against the outer
sidecar and authenticated producer manifest.

## Runtime compatibility remains fail-closed

Before any logical prefix can become a physical cache hit, a future runtime
must bind exact equality for:

1. `model_architecture_sha256`;
2. `tokenizer_sha256`;
3. `token_order_sha256`;
4. `position_ids_sha256`;
5. `rope_config_sha256`;
6. `kv_producing_weights_sha256`; and
7. `prefix_lineage_sha256`.

Missing or mismatched identity yields `cache_incompatible`. The current
producer intentionally has no bound variant and permits no physical reuse.

The model-free runtime prototype also requires a caller-configured
`TrustedExactKVBinding` that pins the task bundle, complete ordered page list,
terminal lineage, compatibility identity, and producer execution mode. A
provider cannot obtain an `exact` claim merely by returning those strings in a
runtime context. This trust binding is still control-plane evidence, not a
cryptographic proof that a future CUDA backend produced the tensors correctly.

All metadata registries have explicit page, prefix, branch, inline-byte, and
identity limits. Capability collisions fail before publication, stale but
authentic branch handles can still release references, and nested async
iterators receive bounded cooperative shutdown on completion, cancellation,
or early consumer close.

## Consumer and release requirements

Consumers must:

1. read each manifest/schema/partition from one immutable bytes snapshot;
2. authenticate the exact local schema bytes named by the producer;
3. validate the outer sidecar and nested plan separately, without remote `$ref`
   resolution;
4. cross-bind outer provenance to every nested `bindings` field;
5. reject unknown fields, unknown schema versions, stale v1 configuration, and
   any hash drift;
6. preserve `task_bundle_sha256` as the split group key so five role views
   cannot cross train/calibration boundaries;
7. materially exclude forbidden/current/future content rather than stringify
   the whole board; and
8. require source-disjoint evidence and a matching release lock before real
   training.

Dry-run validation remains CPU-only, bounded, and model-free. It must not send
provider requests, load model weights, allocate GPU tensors, read heldout
content, or rewrite canonical Gold.

## Related work and attribution

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  motivates binding parent lineage, ordered block tokens, and extra identities
  such as LoRA IDs. Its cache key is not an attention K tensor.
- [Punica](https://arxiv.org/abs/2310.18547) provides multi-LoRA batching and
  shared base-weight serving kernels; that is not cross-adapter KV sharing.
- [LRAgent](https://arxiv.org/abs/2602.01053) studies a shared base component,
  adapter-dependent low-rank cache components, and a specialized attention
  kernel. This metadata contract does not claim those kernels or results.
- [ForkKV](https://arxiv.org/abs/2604.06370) introduces DualRadixTree, CoW cache
  management, and ResidualAttention, while describing cross-layer base-cache
  reuse after adapter hidden-state divergence as lossy. This RFC does not
  relabel that approximation as exact.

This bilingual control-plane RFC was developed with coding and architecture
assistance from OpenAI GPT-5.6-sol. Prior work is credited for its actual
contribution; none of its measurements or quality results are presented as
Anchor-MoE-LoRA results.

## Explicit non-claims

This MVP does not claim decoder KV block concatenation, exact cross-adapter KV
reuse, full-generation cache sharing, token-level MoE routing, CUDA overlap,
memory/latency/throughput improvement, quality preservation, foundation-model
training, or production readiness. Evaluation groups remain undefined until a
runtime implementation and its controlled benchmark are separately approved.
