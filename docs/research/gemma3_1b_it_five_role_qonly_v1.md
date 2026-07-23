# Gemma 3 1B IT five-role Q-only diagnostic runner

This is a controlled-proxy training runner for the 1,000-record bilingual
synthetic five-role dataset. It is deliberately separate from formal-v3:
`training_authorized=false`, `formal_training_authorized=false`, and
`eval_proxy_is_heldout=false` remain true boundaries even after a successful
diagnostic run.

## What it runs

The role order is fixed:

1. `planner`
2. `tool_policy`
3. `frontend_gen`
4. `frontend_review`
5. `security_gate`

Only one role is resident at a time. Each role executes:

1. a fresh Gemma base object plus a fresh rank-4 `q_proj` LoRA for 2 smoke
   optimizer steps;
2. destruction of that model, optimizer, and CUDA cache;
3. a second fresh Gemma base object plus a fresh adapter, restarted from step
   zero for 160 optimizer steps.

The full phase never reads smoke weights, optimizer state, or a smoke
checkpoint. The same seed intentionally makes the two initial adapter digests
equal; the runner fails if this fresh-initialization gate does not hold.

The fixed numerics are BF16 compute, TF32 matmul, SDPA, microbatch 1,
gradient accumulation 1, bitsandbytes 0.48.2 `AdamW8bit` at `2e-5`, and a
strict 768-token sequence length with no truncation. Both optimizer moments
must physically materialize as CUDA `uint8` state for every Q-LoRA tensor.
bitsandbytes 0.48.2 requires its compatibility `optim_bits` constructor
argument to remain `32`; the runner therefore verifies the actual 8-bit state
rather than interpreting that compatibility argument as the stored-state width.
The Torch allocated and reserved peak gates are both 11 GiB. The tokenizer
preflight observed 449–665 tokens
across all 1,000 examples; 514 examples exceed 512, which is why 512 is not an
admitted configuration.

## Adapter output-effect gate

Every smoke and full phase must prove more than a changed adapter file. After
training, the runner selects the first training record, finds its first
supervised next-token position, physically truncates the forward input
immediately before that target token, and evaluates the same prefix twice:
once with the Q-LoRA enabled and once inside PEFT's `disable_adapter()`
context. This excludes the future target suffix and avoids materializing its
logits. The phase passes only when both final-position logits vectors and their
absolute difference are finite and `max_abs > 0`.

Receipts contain only `max_abs`, `mean_abs`, vocabulary width, lengths, and
namespaced hashes of the record/view. They never contain sample text or token
IDs. This is an output-effect diagnostic, not a quality, generalization, or
formal-training claim.

## Independent-agent KV boundary

The intended runtime boundary is machine-bound in the config:

- the identical ordered shared prefix is immutable and computed adapter-off;
- after a Q-only expert activates, its post-activation prompt tokens and newly
  generated tokens append only to that expert's private tail KV;
- a private tail is never reused by a different expert;
- only committed text crosses a role boundary, and that text is re-encoded for
  the next stage's shared context.

The append-only private tail is required for experts to behave like independent
agents. It is not equivalent to sharing full-generation KV, ordinary in-stack
Q-LoRA exact KV reuse, or token-level MoE. This training runner binds the
contract but does not yet materialize the physical inference cache, so
`runtime_private_tail_materialized=false` remains explicit.

## Model-free preflight

The default launcher path never queries the GPU:

```powershell
Set-Location D:\LLM\anchor-moe-lora-neural-swarm
.\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.ps1
```

The Python entry point is also directly usable:

```powershell
python `
  .\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.py `
  --dry-run
```

The preflight authenticates:

- all five local Gemma export files;
- the tokenizer/template binding and mandatory sidecar;
- 1,000 official consumer records before any role selection;
- exact labels and the 768-token no-truncation gate;
- the rank-4 five-expert parameter budget.

It makes zero provider or network requests and does not load model tensors onto
the GPU.

## Explicit diagnostic execution

Execution requires the exact physical GPU UUID. The launcher rejects an empty
or `UNBOUND` value:

```powershell
$env:ANCHOR_GEMMA_GPU_UUID = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
.\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.ps1 -Execute
```

Alternatively pass `-ExpectedGpuUuid`. The launcher:

- takes three idle samples before locking and three after locking;
- requires the configured 12 GiB GPU identity and idle/temperature limits;
- rejects every foreign compute PID;
- takes `runs/formal-v3-training.lock` with `CreateNew` and `FileShare.None`;
- rejects both legacy and v3 handoff GPU locks;
- holds the canonical lock for all five roles;
- publishes authenticated lease and GPU-attestation receipts;
- invokes the Python runner with concurrency one.

The Python runner copies the five authenticated model files once into a
private, read-only-for-the-run snapshot. Ten fresh model objects load from that
same snapshot. The snapshot is rehashed before deletion. Full adapters and
receipts are atomically published under:

`artifacts/diagnostics/gemma3_1b_it_five_role_qonly_v1/<run-id>`

A failure publishes a content-free failure receipt and never retries,
resumes, silently lowers the configuration, or publishes a partial adapter.

## Bound files

- Config:
  `configs/training/gemma3_1b_it_five_role_qonly_v1.yaml`
- Core:
  `src/anchor_mvp/training/gemma3_five_role_qonly_v1.py`
- Python entry:
  `scripts/research/run_gemma3_1b_it_five_role_qonly_v1.py`
- PowerShell launcher:
  `scripts/research/run_gemma3_1b_it_five_role_qonly_v1.ps1`
- Tokenizer binding:
  `fixtures/research/gemma3_1b_it_tokenizer_binding_v1/manifest.json`

No tag or release is implied by this diagnostic.
