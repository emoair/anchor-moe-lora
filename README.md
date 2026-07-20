# Anchor-MoE-LoRA

Anchor-MoE-LoRA is a runnable research scaffold for **task-routed LoRA experts** on one
frozen base model. It routes planning, tool-policy advice, frontend generation,
paired frontend review, and a final defensive security gate through an observable
application-level DAG.

The repository and distribution name are `anchor-moe-lora`. The Python import
package remains `anchor_mvp`, while the prepared Conda environment remains
`anchor-mvp`. Commands below assume the repository root as the working directory.

中文端到端使用、密钥防泄漏、Provider、自动蒸馏、OpenCode execution gold 与
常见故障处理见 [简中快速上手](QUICKSTART.zh-CN.md)。当前 OpenCode live 门禁也以该文档为准。

This is not a neural Mixture-of-Experts layer. The claim under test is whether
specialized adapters plus explicit routing beat a mixed adapter **under matched
call, token, and wall-time budgets**.

## Neural Swarm research branch

The `research/neural-swarm-kv` branch explores a task-to-token Adapter-MoE with
correctness-aware KV sharing and rank-grouped residual execution. Start with the
[English RFC](docs/rfcs/neural_swarm_kv.md) or the
[简体中文 RFC](docs/rfcs/neural_swarm_kv.zh-CN.md). This branch labels
cross-adapter base-cache reuse as approximate after hidden states diverge; it
does not claim lossless O(1) multi-LoRA inference.

The transport-neutral multi-stream scaffold has its own
[English RFC](docs/rfcs/neural_swarm_multistream_pipeline.md) and
[简体中文 RFC](docs/rfcs/neural_swarm_multistream_pipeline.zh-CN.md). Run its
content-free smoke test without loading model weights or using provider quota:

```powershell
python scripts/research/demo_neural_swarm_streaming.py --max-concurrency 2
```

This command validates logical-ID routing and interleaved event delivery only.
An optional OpenAI-compatible Chat Completions SSE adapter is covered by
in-memory transport tests, but no real endpoint, model weights, or evaluation
group is connected in this milestone.

The hierarchical Task-KV control-plane contract is documented in the
[English RFC](docs/rfcs/neural_swarm_hierarchical_task_kv.md) and
[简体中文 RFC](docs/rfcs/neural_swarm_hierarchical_task_kv.zh-CN.md). Its current
source of truth is the producer-v2 `outer_sidecar.segment_plan`, validated by
[`hierarchical_task_kv_segment_plan.schema.json`](configs/research/hierarchical_task_kv_segment_plan.schema.json).
It is metadata-only and `identity_unbound`; physical cache reuse is disabled,
and no quality, memory, latency, or throughput claim is made. The v1 projector
configuration is historical and must not be used for a current fixture.

Run the bounded, model-free cache/stream smoke and TaskBoard projector dry-run:

```powershell
$env:PYTHONPATH = "src"
python scripts/research/demo_hierarchical_kv_swarm.py
python scripts/research/materialize_taskboard_kv_segments.py --dry-run
```

The first command proves one shared logical prefix plus two isolated private
branches are acquired and released. The second authenticates the producer-v2
fixture and its native nested plans, then emits only a content-free summary;
neither command loads a model, touches a GPU, or sends a provider request.

## What is implemented

- SOP-injected, asynchronous and resumable teacher-data generation.
- Strict `plan -> tool_policy -> frontend -> local benign mutation -> review -> security`
  data dependencies;
  downstream teachers never invent or echo their own candidate code.
- Tool-policy proposals are local inert data; model labels never grant runtime permission.
- Canonical JSONL, public `decision_trace`, provenance hashes, cleaning and dedupe.
- Kimi Code Anthropic-first/OpenAI-fallback client, honest client identity, secret
  redaction, request/token budgets, probe mode and deterministic offline mock.
- Gemma 4 12B base QLoRA configs for five specialists plus a mixed baseline.
- NF4 online quantization and compatible pre-quantized PEFT loading; inference-only
  GGUF and W4A16 artifacts are explicitly rejected by the trainer.
- vLLM/OpenAI-compatible client, `frontend -> review -> security` DAG, fail-closed
  handling and structured per-stage traces.
- A/B/C/D controls plus planned E/F adaptive-rank arms, latency/token/VRAM/error accounting and
  Pass@1/TPR/FPR metric hooks.

## Local environment

The prepared environment is isolated from Anaconda `base`:

```powershell
conda activate anchor-mvp
cd <repo>
python -m pip install -e .
python -m pytest
```

To reproduce it without modifying `base`, run
`scripts/environment/bootstrap_windows.ps1` from PowerShell.

The training path has been exercised on one RTX 3080 Ti with 12 GiB VRAM over an
OCuLink PCIe 4.0 x4 connection. A real rank-16 update completed forward, backward,
paged 8-bit optimizer step, adapter save, and adapter reload against the persistent
NF4 checkpoint. This is a resource-constrained feasibility result, not a claim that
the hardware or resulting model quality is optimal.

The verified low-memory profile uses sequence length 64, batch size 1, frozen NF4
base weights, BF16 primary compute, and TF32 for eligible FP32 matrix multiplies.
Normalization parameters remain FP32. Rank 32/64, longer contexts, and larger
batches are gated by measured peak VRAM; they are not assumed to fit.

## Safe first run

Run the whole data path without network or quota use:

```powershell
python -m anchor_mvp data --config configs/data/smoke.yaml --dry-run run
```

For Kimi Code, set the key only in the current shell. Do not put it in YAML, `.env`,
arguments, logs, or source control:

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
python -m anchor_mvp data --config configs/data/default.yaml probe
python -m anchor_mvp data --config configs/data/default.yaml --seed-count 3 --concurrency 1 run
Remove-Item Env:KIMI_API_KEY
```

The stable model ID is `kimi-for-coding`. Kimi's documentation requires the caller
to retain its real User-Agent; this project identifies as `anchor-moe-lora/0.1` and does
not impersonate Claude Code. Thinking is explicitly enabled/configured so the probe
can verify that the coding model path is selected. Hidden reasoning is discarded;
only short, auditable decision artifacts enter the dataset.

Anthropic live generation is non-streaming; all live transports default to a
600-second HTTP read timeout with one retry. The probe uses the same policy. Tune these explicitly with
`--timeout-seconds` and `--max-retries`; a provider may have consumed quota even when
a timed-out response produced no validated JSONL row, so avoid aggressive retries.

OpenAI-compatible Kimi calls use SSE (`stream_openai: true`) by default, matching the
documented third-party setup. The client buffers only final content deltas and discards
reasoning deltas; JSONL remains atomic after the complete response. Optional
`stream_options.include_usage` is off unless a probe confirms compatibility.

After the mock and one-seed gates pass, unattended distillation is available through
`scripts/data/start_automation.ps1`. Concurrency defaults to one; operators may configure
any non-empty sequence of positive integers, with no static code ceiling. The automation
persists provider cooldowns, enforces request/token/failure budgets, and exposes
atomic status plus append-only events. Every configured scale step also rechecks the
frozen held-out corpus against all five current training JSONLs and five SOPs; any
collision blocks the concurrency upgrade. Tool policy and security use independent
low-effort workers; other data workers use medium effort. The script is not started
automatically. See `docs/heldout_benchmark.md` for the frozen manifest and audit contract.

## Model and training

The non-instruction-tuned base is pinned to:

```text
google/gemma-4-12B
revision 56820d7d8cbe8e47975a53325439ed272e91cff2
```

The pinned base snapshot is exported once to a reloadable Transformers/bitsandbytes
NF4 training checkpoint. Subsequent jobs load that persistent Q4 artifact directly,
avoiding repeated online quantization and the associated host-memory and PCIe cost.
A W4A16 deployment copy remains a separate inference artifact.

Validate one adapter without downloading or training:

```powershell
python -m anchor_mvp train `
  --config configs/training/gemma4_12b_qlora_smoke.yaml `
  --adapter frontend_gen `
  --dry-run
```

Use `--execute --allow-model-download` only after the dataset exists and the dry-run
manifest reports a ready environment. See [training details](docs/training.md).
The current evidence, interrupted experiments, and remaining work are tracked in
[project status](docs/PROJECT_STATUS.md).

## Serving and benchmark

The primary A/B/C/D/E/F comparison has one non-negotiable controlled-variable contract:
all arms load the exact same locally serialized Q4/NF4 base artifact (same model
revision, quantization settings, tokenizer, and artifact digest), execute the same
five ordered stages with the same per-stage token caps, and differ only in adapter
assignment/rank allocation. A uses no adapter, B reuses one mixed adapter, C routes five
full rank-16 specialists, D manually matches B's total parameter budget, E searches a
variable-budget adaptive Pareto allocation, and F applies the same adaptive allocator under
a hard B-sized budget. Reports show A's absolute metrics and normalize A to index `100`.
The fair equal-budget comparison is B/D/F; C/E are capacity/Pareto comparisons. A different
Q4 artifact invalidates the run.

Official vLLM is Linux-only; the real server entry uses the installed WSL2 Ubuntu.
Start with the 3080 Ti safe profile: 1-2K context, one sequence, one active adapter,
CPU-cached inactive adapters, and eager execution. Enable CUDA graphs and throughput
features only after a VRAM probe passes.

The application talks to any OpenAI-compatible backend, so vLLM can be compared with
a lower-overhead engine rather than treated as mandatory. See
[serving and benchmark details](docs/serving_benchmark.md).

## Project map

```text
skills/                 versioned expert SOPs
src/anchor_mvp/data/    teacher, cleaning, schema, storage, pipeline
src/anchor_mvp/training QLoRA config, validation, runtime, manifests
src/anchor_mvp/serving/ backend client and routed adapter DAG
src/anchor_mvp/benchmark/ fair baselines, records and metrics
configs/                smoke and experiment configurations
scripts/                model, training and WSL serving entry points
tests/                  offline unit and integration tests
```

## License

Copyright (C) 2026 emoair.

Anchor-MoE-LoRA is licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE). Modified versions made
available over a network must provide the corresponding source as required by the
license. Bundled third-party Skills retain their original licenses and attribution;
see [THIRD_PARTY_SKILLS.md](THIRD_PARTY_SKILLS.md).

## Acknowledgements

This project was built with coding, testing, documentation, and architecture
assistance from OpenAI GPT-5.6-sol. All repository results remain subject to the
project's own reproducibility checks and declared evidence boundaries.

This project builds on openly shared Skills, adapter research, model/training stacks,
and serving tools. Directly vendored assets, research inspiration, and infrastructure
dependencies are credited separately in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md),
with pinned commits and original license locations where applicable.
The isolated SWE task-card importer records its SWE-bench/SWE-smith source policy and
official upstream acknowledgements in
[docs/swebench_metadata_import.md](docs/swebench_metadata_import.md).
