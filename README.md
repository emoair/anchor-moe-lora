# Anchor-MoE-LoRA

Anchor-MoE-LoRA is a runnable research scaffold for **task-routed LoRA experts** on one
frozen base model. It routes planning, tool-policy advice, frontend generation,
paired frontend review, and a final defensive security gate through an observable
application-level DAG.

The repository and distribution name are `anchor-moe-lora`. The Python import
package remains `anchor_mvp`, and the existing local Conda environment and checkout
directory remain `anchor-mvp`, so resumable runs and installed entry points are not
invalidated by the public rename.

This is not a neural Mixture-of-Experts layer. The claim under test is whether
specialized adapters plus explicit routing beat a mixed adapter **under matched
call, token, and wall-time budgets**.

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
- A/B/C plus matched-compute baselines, latency/token/VRAM/error accounting and
  Pass@1/TPR/FPR metric hooks.

## Local environment

The prepared environment is isolated from Anaconda `base`:

```powershell
conda activate anchor-mvp
cd C:\Users\Air\Documents\Codex\2026-07-10\x-b-x\outputs\anchor-mvp
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
`scripts/data/start_automation.ps1`. It ramps concurrency only through 1, 2, 4, and 8,
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

## Serving and benchmark

The primary A/B/C comparison has one non-negotiable controlled-variable contract:
all arms load the exact same locally serialized Q4/NF4 base artifact (same model
revision, quantization settings, tokenizer, and artifact digest), execute the same
five ordered stages with the same per-stage token caps, and differ only in adapter
assignment. A uses no adapter, B reuses one mixed adapter, and C routes five specialist
adapters. Reports show A's absolute metrics and normalize A to index `100`; B/C are
reported as deltas and ratios against A. A different Q4 artifact invalidates the run.

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
