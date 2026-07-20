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

That default command reads only the preflight config; it does not authenticate
or parse inventory JSONL. To explicitly authenticate the three frozen producer
contract files plus the body-free scalar fixture, add:

```powershell
python scripts/research/preflight_gemma4_12b_long_context_preflight.py `
  --config configs/research/gemma4_12b_long_context_preflight.yaml `
  --authenticate-producer-fixture `
  --pretty
```

The report covers 8K, 16K, 32K, 64K, 128K, 256K, 512K, and 1Mi tokens. Up to
256K is only `native_metadata_only`: it is not a runtime, quality, retrieval,
or training result. The 512K and 1Mi rows are `research_only_blocked` because
they exceed native metadata and no extrapolation scaling or validation is
bound. Reported KV numbers are quantized tensor payload planning values, not
measured runtime allocation; they exclude allocator alignment, graph/workspace,
weights, outputs, and concurrency overhead.

## Frozen synthetic token-inventory handoff

The distillation side has now frozen a scalar-only fixture contract:

- producer: `anchor.long-context-token-inventory-producer.v1`;
- record schema: `anchor.long-context-token-inventory.v1`, SHA-256
  `aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f`;
- manifest schema: `anchor.long-context-token-inventory-manifest.v1`, SHA-256
  `8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0`;
- producer config SHA-256
  `79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe`;
- source projector manifest SHA-256
  `595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac`;
- fixture manifest SHA-256
  `73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e`.

The fixture contains 3 authenticated partitions, 15 scalar records, 2 task
bundles, 3 complete five-role groups, 89 segment references, and 25 unique
segments; it made zero provider requests. The consumer fails closed when the
version, declared hash, physical `manifest.json.sha256`, partition hash, or
manifest count drifts. Source-task splitting remains keyed by
`task_bundle_sha256` and happens before augmentation. Prompt, completion,
TaskBoard, block, message, and held-out bodies remain outside this handoff.
The explicit authentication path parses only the closed scalar/hash records to
recompute token formulae, clean/noisy private-delta rules, role-stage binding,
bundle split isolation, complete five-role groups, bucket/gate assignment, and
the canonical tokenizer-binding hash. It reports that scalar JSONL was parsed
while keeping `content_bodies_materialized=false`.

This fixture uses the explicitly synthetic tokenizer binding
`anchor.synthetic-fixture-utf8-byte`. It proves only that the producer and
consumer can authenticate and exchange the scalar inventory contract. It does
**not** prove agreement with the local Gemma tokenizer, 256K capability, 1Mi
support or quality, long-context retrieval, runtime memory, KV correctness, or
training quality. Those claims remain false until a target-model tokenizer and
real measurements are independently bound.

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
