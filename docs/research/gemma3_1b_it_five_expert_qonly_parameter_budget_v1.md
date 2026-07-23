# Gemma 3 1B IT five-expert Q-only parameter budget

This is a metadata-only, diagnostic-only feasibility contract. It does not load
the model, open weight tensors, use a GPU, contact a provider, or authorize
training.

## Reproduce the audit

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.gemma3_qonly_parameter_budget --repo-root .
python -m pytest tests/test_gemma3_qonly_parameter_budget.py -q
```

The first command validates the Draft 2020-12 schema, the mandatory
`sha256sum`-style sidecar, all physical contract/schema hashes, and every
integer in the rank table from one bytes snapshot. A successful audit still
prints `training_authorized: false`.

## What is being counted

The local export metadata reports 26 transformer layers, hidden size 1152,
four query heads of width 256, and 999,885,952 base parameters. Therefore each
`q_proj` maps 1152 to 1024. A rank-\(r\) Q-only LoRA has:

```text
per expert = 26 × r × (1152 + 1024) = 56,576r
five stored experts = 282,880r
single routed request = 56,576r active parameters
```

The default runtime interpretation is one private expert adapter per request.
Adding five checkpoint sizes does not mean five adapters execute per token and
does not make the system an equivalent 2B model.

| rank | one expert | five stored experts | interpretation |
|---:|---:|---:|---|
| 4 | 226,304 | 1,131,520 | small MVP |
| 8 | 452,608 | 2,263,040 | small MVP |
| 16 | 905,216 | 4,526,080 | small MVP |
| 32 | 1,810,432 | 9,052,160 | medium |
| 64 | 3,620,864 | 18,104,320 | medium |
| 256 | 14,483,456 | 72,417,280 | aggregate stress |
| 512 | 28,966,912 | 144,834,560 | aggregate stress |
| 542 | 30,664,192 | 153,320,960 | approximately dense-Q storage boundary |
| 1024 | 57,933,824 | 289,669,120 | full effective rank, redundant factor storage |
| 3535 | 199,996,160 | 999,980,800 | infeasible for base-parameter parity |

A dense Q delta across all 26 layers contains 30,670,848 parameters per
expert. Factorized LoRA storage exceeds that dense-Q control above rank
542.1176. The effective rank cannot exceed 1024. Rank 3535 is therefore not a
large but meaningful LoRA: it is a heavily redundant factorization chosen only
because five raw parameter counts happen to land 94,848 above the base count.
The contract classifies this target as
`infeasible_for_base_parameter_parity`.

## Memory estimates

The table in the JSON contract uses explicit accounting:

- BF16 adapter checkpoint: 2 bytes/parameter.
- BF16 gradient: 2 bytes/parameter.
- FP32 master copy: 4 bytes/parameter.
- two FP32 Adam moments: 8 bytes/parameter.
- adapter training state: 16 bytes/trainable parameter.

The “single-route sequential” estimate is BF16 base + one active adapter's
16-byte training state + four inactive BF16 adapter checkpoints. The
“all-experts optimizer” estimate keeps training state for all five. Both omit
activations, KV cache, CUDA context, kernels, allocator fragmentation,
dataloader buffers, and framework workspaces; they are accounting baselines,
not peak-VRAM promises. TF32 changes eligible FP32 matmul execution, not these
parameter-storage counts.

## Fair comparisons

The pre-registered primary arms are:

1. frozen base;
2. monolithic LoRA;
3. five experts with correct routing;
4. five experts with deliberately wrong routing;
5. five experts with random routing.

O-only and Q+O remain diagnostic overlays. An active-budget comparison gives
the monolithic control the same parameters as one routed expert. An
aggregate-storage comparison gives the monolithic control the same stored
parameters as all five experts, but then all monolithic parameters are active.
These are different experiments; a fivefold aggregate budget must never be
presented as an active-compute-matched win.

Recommended proxy metrics are tool-call schema validity, search-result
grounding, micro-code test pass rate, routing accuracy, and wrong-route delta.
No quality conclusion is made by this contract.

## Why a real run is blocked

The local export is useful metadata, but it is not a complete training
identity:

- no exact official model ID or source revision is bound;
- `EXPORT_MANIFEST.json` says `chat_template_bound=false`;
- no official chat-template hash is present;
- `config.json` declares BOS=1 and EOS=2, while
  `tokenizer_config.json` maps token 1 to `<eos>` and token 2 to `<bos>`;
- tokenizer/base compatibility and runner preflight have not passed.

The config, tokenizer assets, weights, model ID, revision, and official chat
template must be frozen and cross-validated before any real training run.
