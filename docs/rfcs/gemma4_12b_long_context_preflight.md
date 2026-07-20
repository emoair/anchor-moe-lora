# Gemma 4 12B static long-context preflight

This is a small, model-free capacity check for the local Gemma 4 12B profile.
It reads one YAML file and performs integer KV-cache math. It does **not** load
weights, use a GPU or network, inspect JSONL/data bodies, or read held-out data.

Pinned local facts:

- native context metadata: 262,144 tokens;
- model identity: `google/gemma-4-12B` at revision
  `56820d7d8cbe8e47975a53325439ed272e91cff2` (source config SHA-256
  `14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098`);
- 48 text layers: 40 sliding-attention and 8 full-attention layers;
- sliding window 1,024; llama.cpp allocation 1,024 + ubatch 256 = 1,280 cells;
- full-attention config uses `rope_type=proportional`, theta 1,000,000, with
  `partial_rotary_factor=0.25`;
- sliding-attention RoPE: theta 10,000;
- conservative separate K/V storage: K `q8_0` (34 bytes / 32 elements),
  V `q4_0` (18 bytes / 32 elements), with Flash Attention pinned on;
- the 1,280-cell allocation rule is pinned to llama.cpp commit
  `33c718db1fbfe834f30eef28cf206f98736fe612`.

Run:

```powershell
python scripts/research/preflight_gemma4_12b_long_context.py `
  --config configs/research/gemma4_12b_long_context_preflight.yaml `
  --pretty
```

The report covers 8K, 16K, 32K, 64K, 128K, 256K, 512K, and 1Mi tokens. Up to
256K is only `native_metadata_only`: it is not a runtime, quality, retrieval,
or training result. The 512K and 1Mi rows are `research_only_blocked` because
they exceed native metadata and no extrapolation scaling or validation is
bound. Reported KV numbers are quantized tensor payload planning values, not
measured runtime allocation; they exclude allocator alignment, graph/workspace,
weights, outputs, and concurrency overhead.

The report also requests `anchor.long-context-token-inventory.v1`, a scalar-only
producer handoff. It is explicitly **pending producer freeze**, not a frozen
producer schema. It asks the distillation side for per-record token counts,
lineage hashes, reserved output, and bucket IDs after source-task splitting
while prohibiting prompt, completion, TaskBoard, block, or message bodies. This
preflight never consumes that inventory; it only publishes expected fields and
invariants. Counts must include the bound chat template, framing, and special
tokens. `total_tokens` is the observed prompt-plus-target sequence;
`required_context_tokens` reserves the larger of target and output allowance.

The staged, non-launch plan is: validate native 64K, 128K, then 256K without
position scaling; investigate 512K with a 2x YaRN or linear candidate; keep 1Mi
blocked behind a 4x YaRN feasibility study or a LongRoPE-style training route.
YaRN and LongRoPE refer to their primary papers: [YaRN](https://arxiv.org/abs/2309.00071)
and [LongRoPE](https://arxiv.org/abs/2402.13753).

Before any 1Mi claim, a later gate must bind an explicit RoPE/extrapolation
method, measured runtime memory, positional/retrieval quality, task quality,
and long-context training evidence. A rotary encoding mechanism alone is not
evidence that one-million-token behavior is stable.

Primary references: the [official Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4),
the pinned [12B configuration](https://huggingface.co/google/gemma-4-12B/blob/56820d7d8cbe8e47975a53325439ed272e91cff2/config.json),
the [Gemma 4 technical report](https://arxiv.org/html/2607.02770), and the
[llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
