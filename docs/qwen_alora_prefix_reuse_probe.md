# Qwen aLoRA two-request Prefix-KV probe

Date: 2026-07-22. This is local mechanical feasibility evidence, not formal
training, a quality benchmark, a tokenizer binding, or a physical multi-stream
implementation. It used the local `Qwen/Qwen2.5-1.5B-Instruct`, PEFT 0.19.1,
and a temporary rank-4 `q_proj` aLoRA. A deterministic nonzero random B matrix
created an observable mechanical effect; this run performed no training, read
no project dataset, made no provider request, and saved no adapter.

## Causal boundary under test

Planner-private KV from request 1 is **not reused across requests**. The short
natural-language scaffold committed by the planner is serialized and encoded
again in request 2. Request 2 has three regions:

1. a base-compatible pre-invocation prefix encoded by the frozen base;
2. one unique invocation-token sequence;
3. an active continuation beginning at the invocation.

The probe compares:

- `full`: one aLoRA forward over the complete request 2;
- `reuse`: adapter-off frozen-base prefill for the prefix, followed by aLoRA
  continuation from the invocation using the same prefix KV.

A base-only full/reuse control estimates and calibrates the numerical floor of
chunked cache execution. This subtraction is not strict causal separation in a
nonlinear computation.

## Input identity

- complete request 2: 44 tokens;
- base-compatible pre-invocation prefix: 25 tokens;
- invocation covering span: 8 tokens; its text occurs exactly once in the
  serialized request;
- active continuation: 19 tokens, comprising the 8 invocation tokens and an
  11-token post-invocation tail, so `25 + 8 + 11 = 44`;
- the invocation token span is derived by tokenizing the complete
  chat-templated request 2 once, then locating and validating the invocation
  character span through offset mapping. The complete tokenizer encoding
  produces token IDs. Encoding the invocation in isolation and searching
  is invalid: it is 7 tokens in isolation in this case and has no exact match
  in the full input, while the authoritative 8-token span includes the `>\n`
  boundary. This boundary overhang belongs in the request-local
  materialization receipt;
- the invocation-span digest is
  `fca583f23dcd62a9d0531744a7c70366558768a6483398150d1d5e94c13fbf8d`;
  the complete ordered request-2 token digest is
  `c78e409a9ec5b772c0b56f9efbd0f179bfbf35a21c6e8cbb3f6c998d9d1f57f2`.
  Both are local probe identities, not a global tokenizer binding.

## Results

| Mode | aLoRA full/reuse max | base full/reuse max | argmax agreement | Observation |
|---|---:|---:|---:|---|
| BF16 eager | 0.890625 | 0.90625 | 100% / 100% | full and cached attention take materially different numerical paths; strict max-logit equality is an invalid gate |
| FP32 eager | 0.000282675 | 0.000255227 | 100% / 100% | aLoRA error is on the same scale as the base numerical floor |

In FP32, the maximum absolute difference between the aLoRA and base
full/reuse residuals was `0.000139236`. The observable adapter effect was
`0.0649891`, and peak allocated GPU memory was `6054.7 MiB`. In a separate BF16
negative control, omitting the invocation produced exactly `0.0` adapter
effect. This validates the PEFT 0.19.1 masking control in this run; it does not
validate a formal tokenizer binding, service routing, or hot swapping. The two
argmax values in the table are aLoRA and base respectively, and come from this
single 44-token synthetic case only, not a quality claim.

## Current conclusion

The result supports mechanical feasibility of feeding request-2 frozen-base
prefix KV into an aLoRA suffix. It does **not** establish that:

- planner-private KV can cross the request boundary;
- BF16/Q4 logits are element-wise identical;
- ordinary in-stack LoRA, arbitrary Q-LoRA, or complete expert generations can
  share full KV;
- the current single-slot runtime supports concurrency, multiple streams, or
  physical hot swapping;
- tokenizer binding, toy source-disjoint attestation, or formal-v3 is ready.

A formal diagnostic should use FP32 eager as a high-precision reference path
and require:

- eager FP32 with TF32 disabled and a unique invocation derived only from exact
  request-2 serialization;
- active output equal to adapter-off output without the invocation, and adapter
  effect of at least `1e-3` with it;
- with `E_base=base_split-base_full` and
  `E_alora=alora_split-alora_full`,
  `max(abs(E_alora-E_base)) <= 1e-3`;
- finite values and consistent argmax, shape, and cache lineage;
- layer-by-layer bit equality between active and adapter-off prefix KV before
  the invocation;
- BF16 only as a performance/semantic gate: finite values, full/split argmax
  agreement, and identical 4--8-token greedy continuations, never strict raw
  max-logit equality;
- continuation that preserves the aLoRA activation offset. PEFT `generate`
  maintains it; a manual token-by-token loop can silently fall back to the base
  if it drops the offset, and user-supplied offsets must not be accepted.

## Runtime version isolation

`NaturalLanguageScaffoldRuntime v1` is currently a single-slot serial state
machine and cannot host multiple R1/R2 pairs in one instance. A future
diagnostic mux must leave v1 unchanged, give each `(run_id, stream_id)` its own
v1 instance, use per-stream atomic transitions and tombstones, and only run
independent pairs concurrently. Missing adapter-registry, tokenizer-binding,
activation-receipt, physical-KV-lease, or formal-lock identity must keep the
path `diagnostic_only=true`, `formal=false`, and `training_authorized=false`.

The 44-token numbers above came from the original manual probe and had no
receipt. A subsequent standalone diagnostic now binds the model snapshot,
tokenizer/chat template, Torch/PEFT/Transformers/CUDA, attention
implementation, seed, and adapter recipe. It is still not wired into the
multi-stream runtime and has no formal authority. A prefix is base-compatible
only when ordered tokens, position/RoPE, mask, cache dtype/layout/device, and
all of those identities match.

## Reproducible execution

```powershell
# Contract only; does not import the ML runtime
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml

# Authenticates four local model assets without loading weights
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml --preflight

# The only path that loads the model/GPU
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml --execute
```

The 2026-07-22 diagnostic-only execution passed. Request 2 had 48 tokens, a
30-token pre-invocation prefix, a 7-token invocation, and an 18-token active
continuation. The paired differential was `4.86374e-5`, adapter effect was
`0.205176`, and missing-trigger effect was `0`. All 28 prefix-KV layers were
bit-equal, the 4-token greedy continuations matched, and peak allocated memory
was `5954.57 MiB`. The result is stored in
`artifacts/diagnostics/qwen_alora_prefix_kv_v1/diagnostic_receipt.json`.

This request happened not to have the earlier 44-token case's `>\n` overhang,
so its invocation covered 7 tokens. The difference is direct evidence that the
span belongs in a request-local receipt and cannot be frozen globally. The
tool trained nothing, saved no adapter, used no provider/network/dataset
input, and still declares `formal=false` and `training_authorized=false`.
