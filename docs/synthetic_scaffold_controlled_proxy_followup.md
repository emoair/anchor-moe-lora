# Synthetic Scaffold Controlled-Proxy Follow-up v1

## Status and non-goals

This Producer-side artifact records a metadata-only follow-up design for the
Consumer comparison at commit
`6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465`. It is an additive,
non-authorizing overlay. It does not modify canonical Gold, held-out data, the
frozen natural-language scaffold v1/v2 contracts, or any formal-v3 artifact.

Its only positive result claim is that `q_plus_o` had the lowest endpoint
eval-proxy loss under the exact single-seed, seq512, 80-step,
parameter-budget-matched comparison. It does not establish a formal winner,
statistical significance, sample efficiency, a causal Q/O mechanism,
long-context generalization, physical KV reuse, zero-copy execution, quality,
or training authorization. The 20 eval records derive from only two independent
source bundles. CUDA deterministic algorithms were not enabled. Equal
trainable parameters do not imply equal compute: the arms have 56, 112, and
224 trainable tensors.

## Authenticated report evidence

The upstream `comparison.json` and mandatory sidecar were local ignored
artifacts, not Git blobs in the Consumer commit. The Producer carries an exact
metadata-only byte copy under
`fixtures/research/synthetic_scaffold_controlled_proxy_followup_v1/source/`.
It contains no prompt, answer, token IDs, Gold body, held-out body, or protected
scaffold body.

The Producer authenticates the exact out-of-tree report bytes and a closed
semantic projection of that report. It does not reopen the referenced
preflights, receipts, adapters, model, or Consumer Git blobs at audit runtime.
The tracked schema/config/runner/auditor hashes were cross-checked against the
Consumer commit during this freeze, but that is not a transitive re-execution
of the training run. Schema validation alone is insufficient; the hash-bound
Producer auditor is mandatory.

| Identity | SHA-256 |
|---|---|
| Consumer commit | `6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465` |
| Consumer tree | `2e40aaa1685b3da914275f4f61f8161138ffe6b5` |
| Source comparison | `920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45` |
| Source sidecar file | `bbdeb5b3ac8f7890a93c736b19bcd920875f642174f99914b199d1d62ce06830` |
| Source comparison schema | `3b1c81cc888f0b56e013d3afc1317ed78ccd62f7c53606aa2257ed6d389161e2` |
| Source training config | `490b0e18fce004a44b97b4c4ab2a3d0f9d0809e1d1f20a2fdc8f93938490e9c2` |
| Source runner | `ea2b822f0aefc3f7deea9654849a2a5ea87b2e1c35356ad7615941bcb1cefd9b` |
| Source auditor | `c7328cdc3083d52ee8d40de94ee0f18169d920cc448b75f2860868911c522b71` |
| Producer schema | `fe3878cac9d3be773a676c23025e79cc7f64063da03a53c54eb7f4b59594e0b6` |
| Producer contract | `06ed121b23570546eb00088c891b273806dab1a1f764c9e40fba527cbf6447df` |
| Producer auditor implementation | `588c396febab5de75b772a4f46d58f21c8456247097fccc345cb1c738b0093a3` |

The upstream schema closes only its top level; nested comparison objects remain
open. Its JSON authentication also used separate reads for hashing and parsing.
The Producer instead uses a closed semantic projection, one immutable bytes
snapshot for hash/parse/reparse, exact mandatory sidecars, path and
reparse-point rejection, and a final fresh-byte identity check.

## Observed contract and interpretation

All arms used the same Qwen2.5-1.5B-Instruct base fingerprint, 80 training
records exactly once, 20 eval-proxy records, sequence length 512, 80 steps,
learning rate `5e-5`, seed 1337, BF16 plus TF32, and 1,376,256 trainable
parameters. The historical split key is `source_bundle_id`; it is not rewritten
as `task_bundle_sha256`.

The base digest `2e05af50...e15` is not the `model.safetensors` file digest. It
is a SHA-256 fingerprint over `named_parameters()` iteration order. Each leaf
binds the UTF-8 parameter name, NUL, the ASCII SHA-256 of raw CPU-contiguous
tensor bytes, and LF. Train-order and eval-view digests carry their explicit
seeded-record/token-view algorithm identities. LoRA ranks and alpha/scaling
values are both projected.

| Producer label | Consumer profile | Rank / alpha | Eval loss after | Throughput (tok/s) |
|---|---|---|---:|---:|
| `q_only` | `q_only` | Q16 / 32 | 2.0898047030 | 902.9 |
| `q_plus_o` | `q_plus_o` | Q8,O8 / 16,16 | 0.8586097196 | 814.5 |
| `wide_lora` | `wide_budget_matched` | Q5,O4,K6,V6 / 10,8,12,12 | 1.1850866765 | 621.9 |

The neutral planning label for Q+O is “attention-query intervention plus
attention-output-to-residual intervention.” The evidence only says it reached
lower endpoint proxy loss; it does not say why. Throughput is a single
descriptive run: Q+O was about 9.8% slower and Wide about 31.1% slower than
Q-only, without randomized repeated timing.

## Preregistered replication phase

The follow-up freezes five master seeds and separately derives and hashes the
adapter-initialization, record-order, and CUDA seed schedules. Each seed uses
the same examples and parameter budget across arms. Performance measurement is
separate from training: one warm-up per arm and all six arm-order permutations
within every seed, with CUDA synchronization and runtime/thermal/clock receipts.

The primary metric is bundle-macro eval-loss delta versus the shared base,
aggregated by equal-weight bundles within seed and then equal-weight seeds.
Uncertainty uses a preregistered paired two-level seed/bundle bootstrap with
10,000 resamples, seed 20260723, and 95% intervals. Every seed, source bundle,
language, role, scaffold variant, and 5/10/20/40/80-step checkpoint must be
reported. O-only and K+V budget-matched controls are required before any
mechanistic interpretation.

Readiness is an integrity-and-coverage gate independent of ranking. A valid run
may replicate Q+O, tie or be inconclusive, or overturn the ranking. Producer
still selects no winner, and a passing run remains controlled-proxy evidence.

## Planned confirmation fixture and split

The planned fixture has 60 source bundles, 40 train and 20 eval-proxy, balanced
English/Simplified Chinese, five roles, two paired variants, and 600 records.
The five preregistered information-flow strata each receive 12 bundles: eight
train and four eval-proxy, with four/four train and two/two eval-proxy per
language. Derived counts are therefore 400 train and 200 eval-proxy records.
The domain-separated blueprint-hash ordering and balanced round-robin
allocation algorithm is frozen before generation.
It uses `task_bundle_sha256` before role, variant, noise, length, or causal
augmentation. The task-bundle preimage binds a fixed domain, confirmation
source namespace, source-bundle identity, language, and
`source_task_blueprint_sha256`; it excludes role/variant/noise/length. All five
roles and both variants share the bundle, and inner `task_board.task_id` must
cross-bind it. `eval_proxy` is not held-out.

This artifact specifies strata only; the future fixture will contain synthetic
training records. It is not yet generated and is not yet proven independent of
the 10-bundle discovery fixture. The discovery source-bundle inventory hash is
bound, but namespaced bundle-ID non-overlap is explicitly insufficient to prove
semantic independence. A common-domain, namespace-neutral inventory of
`source_task_blueprint_sha256` is required for both discovery and confirmation.
The discovery blueprint inventory is currently unpublished, the confirmation
inventory is pending, zero overlap is unavailable, and
`independent_confirmation_claimed=false`.

## Length and KV gates

The tokenizer-bound length plan starts at 8K, 16K, and 32K. Every bucket means
`total_tokens=input_tokens+reserved_output_tokens` and must not exceed 32,768.
The recorded local Qwen candidate revision/config/tokenizer/chat-template/
serialization/special-token identities are diagnostic candidates, not a formal
binding. This contract permits zero provider requests, model loads, or GPU
requests for the preflight.

The candidate config reports `max_position_embeddings=32768`. Therefore 64K,
128K, and 256K are blocked for the current identity and require a newly
authenticated model or RoPE identity; 256K remains capability-gate-only even
then. One Mi token remains research-only/blocked. Existing Gemma or synthetic
byte-tokenizer inventories cannot satisfy this Qwen gate.

The runtime gate keeps every adapter off on the frozen shared prefix and permits
an expert adapter only after the route boundary. Exact reuse is restricted to
an identical ordered prefix lineage with identical positions, RoPE, model, and
KV-producing weights. Expert tail KV remains private. No ordinary in-stack
LoRA arm establishes exact full-stack reuse or a physical Q-reader by name.

## Fail-closed and unfinished work

The audit rejects schema or byte drift, duplicate JSON keys, non-finite numbers,
bad sidecars, absolute/traversing paths, symlink/junction/reparse components,
initial/final identity drift, arm/alias/rank/alpha drift, unequal budgets,
base/order/token-view drift, ranking drift, historical split rewriting,
post-hoc metric changes, seed/permutation drift, unsupported length promotion,
protected-body reads, and any blocked-claim promotion.

Zero provider/network/model/GPU and protected-body counters describe only
construction and auditing of this Producer metadata overlay. The upstream
controlled proxy did load and train a local model on a GPU.

Formal-v3 snapshot, projector, generic execution contract, source-disjoint
contract, and release lock remain 0/5. Protected body-free inventories remain
2/6, with no six-source zero-intersection proof. Multi-seed replication,
confirmation blueprint disjointness, long-context evidence, physical KV
correctness/performance, a real teacher-distillation dataset, and formal quality
evaluation remain unavailable. `training_authorized=false` and
`formal_training_authorized=false` are fixed.

## Metadata-only reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_synthetic_scaffold_controlled_proxy_followup_v1.py `
  --repo-root .
python -m pytest -q tests/test_synthetic_scaffold_controlled_proxy_followup_v1.py
python -m ruff check `
  src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_followup.py `
  scripts/data/audit_synthetic_scaffold_controlled_proxy_followup_v1.py `
  tests/test_synthetic_scaffold_controlled_proxy_followup_v1.py
```

These commands issue no provider request or model/GPU load. No tag or release is
created. The artifact cannot authorize training by itself.
