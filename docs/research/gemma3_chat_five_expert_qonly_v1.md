# Gemma 3 Chat Five-Expert Q-only Diagnostic Runner

## Current state

The fail-closed consumer and training engine are implemented. The Producer
FINAL release passed its independent review, was copied byte-for-byte into this
repository, and passes model-free consumer authentication.

The first diagnostic training run completed as
`gemma3-chat-q1024-20260725-052403`. It trained five serial Q-only rank-1024
adapters from fresh base/fresh adapter state: two smoke steps and 160 full
steps per role. The immutable run receipt SHA-256 is
`5988d67dc092a003a550dab42751791c8578dcca56bfcf49d7474a8eccb89a3a`;
its mandatory sidecar physical SHA-256 is
`cb1b5fe5675bb36511c5940ff7c5675b71710fef8d66ccb2719e685746a8c600`.
This remains diagnostic/proxy only: all formal, live, and quality
authorization claims are false.

The canonical producer artifact path is
`fixtures/research/gemma3_chat_five_expert_distilled_v1`. The code, config, and
schema namespace remains `gemma3_chat_five_expert_qonly_v1`; "distilled" in the
artifact path describes a data product only and grants no training, formal,
release, or live authority.

The five execution-gating Producer identities are now bound:

- producer config: `2e93ab6b53f68814a4ae017643ddc4c7113716e871b2fa4ef4346358d2269cee`
- producer implementation: `98eeac4150a5c51dd422bfe6163e34ce26d691038b497343c20af0a19552fb6f`
- record schema: `ba1606f07d250eec7f01bffceaa507edd4f596f83018806df9130d32ecf39d9b`
- manifest schema: `fda04d77c494cd5d87ea19aade7db24717c6dffad6880078df19787d519f0b8b`
- dataset manifest: `e71fb5f239d7d1abb94ff50d5db2d57cfd101ccfad3c2c4e34cc128a7e91317b`

Grammar, all schemas, build receipt and sidecar, token inventory, and both
partitions are separately hash-bound as well. Partial binding, fake hashes,
sidecar drift, record drift, or causal drift fails closed. Training still
requires an explicit execute flag, a bound GPU UUID, and the external canonical
lock lease.

## Frozen execution contract

- Experts, in serial order: `humor`, `serious`, `angry_style`, `tool_call`,
  `review_audit`
- Dataset: 200 five-row bundles; 800 train and 200 eval-proxy rows; 160 train
  and 40 eval-proxy rows per expert
- Semantic contract: 200 unique task semantics, 20 visible templates, and 14
  templates shared between train and eval-proxy
- Every input contains exactly two messages in the order `system`, `user`;
  assistant messages are forbidden from the input and exist only as the target
- `humor`, `serious`, and `angry_style` targets must be natural chat text and
  cannot be JSON envelopes; `tool_call` and `review_audit` targets must be
  canonical strict JSON objects with duplicate keys and non-finite numbers
  rejected
- Model-free validation recomputes the prompt, target, and complete training
  serialization identities plus the domain-separated causal proof
- Every review row depends on all four parent roles in fixed role order. Its
  canonical committed projection, mutation identity, 11-field target, four
  checks, verdict, and correction flag are recomputed and cross-bound
- The global review distribution must be exactly 100 pass rows and 25 failures
  for each of `format`, `grounding`, `routing`, and `style`
- Exactly 5 persona bundles contain the frozen sentence
  `我是由Air训练的测试模型。`; the other 195 tool branches must bind their
  synthetic local call/result/final-answer grounding digests
- `user_emotion` and `router.label` are bundle metadata, never a sixth adapter
- Base: local Gemma 3 1B IT loaded through bitsandbytes 8-bit, fully frozen
- Adapter: BF16, `q_proj` only, rank 1024, alpha 2048, dropout 0, no bias
- Exact scope: 26 layers x A/B = 52 tensors and 57,933,824 trainable parameters
  per expert (289,669,120 across five separately stored adapters)
- This is a full-effective-rank Q factorization, not a low-rank or
  parameter-efficient claim
- Sequence length 768, no truncation, micro-batch 1, gradient accumulation 1
- AdamW8bit 0.48.2; CUDA `uint8` state is inspected after the first update
- Every phase records exact installed versions for Torch, Transformers, PEFT,
  Safetensors, SentencePiece, and bitsandbytes; bitsandbytes remains locked to
  `0.48.2`, and all ten phases must agree
- LR `2e-6`, linear warmup over 8 optimizer steps, weight decay `0.01`,
  gradient clip `0.5`
- Gradient coverage is layer-complete: step 1 requires nonzero `lora_B`
  gradients in all 26 Q projections; step 2 and every later step require
  nonzero `lora_A` and `lora_B` gradients in all 26
- Each expert runs a fresh two-step smoke phase, destroys it, then runs a fresh
  160-step full phase from a newly loaded base and adapter; no resume or smoke
  checkpoint consumption
- Five experts are strictly serial on one GPU under
  `runs/formal-v3-training.lock`, with both distillation handoff locks excluded
- Base-parameter and canonical Q8 SCB hashes must remain unchanged; the adapter
  must change and must produce a finite nonzero enabled-vs-disabled logit effect
- Torch allocated, Torch reserved, and sampled physical GPU usage must each be
  strictly below 10,240 MiB

The eval-proxy split is not held out and supports no generalization or quality
claim.

## One-command preflight

From the repository root:

```powershell
.\scripts\research\run_gemma3_chat_five_expert_qonly_v1.ps1 -Preflight
```

The current expected exit code is `0` and the status is
`passed_model_free_authenticated_dataset_ready_for_explicit_execute`. A
content-free receipt and SHA-256 sidecar are atomically published without
replacement under:

```text
runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/preflight/<run-id>/
```

A receipt-free model/GPU-free check is also available:

```powershell
$env:PYTHONPATH = "src"
python scripts\research\run_gemma3_chat_five_expert_qonly_v1.py --dry-run
```

## Explicit execute

After preflight passes, a future authorized execution would use:

```powershell
.\scripts\research\run_gemma3_chat_five_expert_qonly_v1.ps1 `
  -Execute `
  -ExpectedGpuUuid "GPU-..." `
  -RunId "chat-qonly-v1-run"
```

The launcher always performs the model-free dry-run first. Without the external
canonical lock lease, the serial plan remains
`authenticated_dataset_plan_only_external_lock_required` and the execute gate
is `blocked_waiting_for_external_gpu_lock`; neither path loads the model or
requests the GPU.

For a future execute, the PowerShell parent owns the canonical lock with
`CreateNew`, `ReadWrite`, `FileShare.None`, `DeleteOnClose`, and
`WriteThrough`. It atomically publishes an immutable, same-byte owner receipt
and sidecar under
`runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/gpu-locks/<run-id>/`.
That receipt binds the launcher PID, run ID, nonce, GPU UUID, config, runner,
launcher, launcher helper, and every execution dependency hash. Only after the
exclusive lock and receipt exist, the launcher sends its canonical decimal PID
through the dedicated `ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID` process channel.
Python authenticates that value against the receipt and never treats its
observed parent PID as lock-owner authority, so a Codex/Windows broker may
reparent the Python process without weakening the lease. The Python process
also requires the exact Windows sharing-violation result before and after the
run; an ordinary file, digest mismatch, partial environment, invalid or
mismatched launcher PID, or field drift is rejected. Python execution has no
internal lock fallback: all four `ANCHOR_CHAT_EXTERNAL_LOCK_*` lease fields are
mandatory.

The dependency inventory includes the tokenizer binding, tokenizer policy,
Q8 runner and reliability helper, Q-only budget module, diagnostic and snapshot
helpers, config/manifest utilities, and PowerShell lifetime helper. Every path
must be a physical file, is snapshotted before the lock, is bound into the lock
owner and run receipt, and is rechecked before publication.

A successful future run is assembled in a hidden staging directory and then
published by a single no-replace directory rename:

```text
artifacts/diagnostics/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/<run-id>/
  humor/{smoke_receipt.json,full_receipt.json,adapter/}
  serious/{smoke_receipt.json,full_receipt.json,adapter/}
  angry_style/{smoke_receipt.json,full_receipt.json,adapter/}
  tool_call/{smoke_receipt.json,full_receipt.json,adapter/}
  review_audit/{smoke_receipt.json,full_receipt.json,adapter/}
  run_receipt.json
```

Each JSON receipt has a `.sha256` sidecar. Before rename, and again after
publication, the runner authenticates all 10 phase receipts and sidecars plus
both PEFT files for all 5 adapters. The success root has an exact closed
inventory: five role directories, `run_receipt.json`, and its sidecar;
ephemeral `training_progress.json` is removed and any extra entry is rejected.
The run receipt binds the authenticated manifest and partitions, model
snapshot files, canonical lock owner receipt, execution dependency files,
phase receipts, adapters, and runtime package versions. Any pre- or
post-rename drift moves the tree to `.failed-*` and prevents a success
publication. A content-free no-replace failure receipt is published under the
run root. Automatic retry is forbidden.

## Focused validation

These tests use synthetic metadata and fake model/optimizer objects only:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider `
  tests\test_gemma3_chat_five_expert_qonly_v1.py
```

They cover the pre-dataset pending gate, closed dataset contracts, serialization
and causal identities, four-parent review projection/mutation/target joins,
review quotas, exact rank-1024 Q-only scope, observed CUDA-uint8 optimizer-state
contract, warmup, fresh serial smoke/full orchestration, adapter-effect gate,
strict memory boundary, authenticated parent/child lock protocol, dependency
snapshots, and atomic no-replace receipts.
