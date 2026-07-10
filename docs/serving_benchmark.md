# Serving, DAG, and benchmark

## What this is

The primary `planner -> tool_policy -> frontend -> review -> security` path is an
application-level task DAG. The legacy `PipelineRouter.run` three-stage method remains
available only for compatibility; new routed experiments use `run_five_stage`. The
router changes the OpenAI-compatible request's `model` field so vLLM selects a
loaded LoRA adapter for that call. It is **not** a learned token router, sparse
neural MoE, or simultaneous expert activation inside the transformer.

Each pipeline stage emits a structured artifact containing its input, output,
adapter/model id, status, attempts, latency, token usage, and error. Stages have
timeouts and bounded retries. A failed or ambiguous stage returns `BLOCK` with
`fail_closed=true`; benchmark metrics keep these infrastructure failures separate
so they cannot silently inflate malicious-request TPR.

Artifacts distinguish application-level stage attempts from nested backend HTTP
attempts, since client retries and DAG retries can otherwise hide the real load.

## Start vLLM locally through WSL2

Official vLLM does not support native Windows. This project therefore runs vLLM
inside the existing WSL2 Ubuntu 22.04 environment. The Bash script is the real
entrypoint; the PowerShell wrapper only converts Windows paths and invokes it.
See the official [vLLM GPU installation page](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

The wrapper registers five specialists plus the mixed control at startup and binds to localhost:

```powershell
.\scripts\serve\start_vllm.ps1 `
  -PlannerAdapter "D:\anchor\lora-planner" `
  -ToolPolicyAdapter "D:\anchor\lora-tool-policy" `
  -FrontendAdapter "D:\anchor\lora-frontend-gen" `
  -ReviewAdapter "D:\anchor\lora-frontend-review" `
  -SecurityAdapter "D:\anchor\lora-final-security" `
  -MixedAdapter "D:\anchor\lora-mixed-all" `
  -Profile 3080ti-safe `
  -Quantization bitsandbytes `
  -LoadFormat bitsandbytes `
  -MaxModelLength 1024
```

The canonical experiment base is the non-instruction-tuned
`google/gemma-4-12B` at revision
`56820d7d8cbe8e47975a53325439ed272e91cff2`. Google currently publishes a ready
Gemma 4 QAT W4A16 compressed-tensors checkpoint for the `-it` model, not this base.
To avoid silently changing the experiment's alignment, the default served path is
the pinned local `models/google-gemma-4-12B-base`. Populate it from that exact
revision before launch. A Hub id can cause vLLM to download weights when invoked.

There is no aligned official Q4 checkpoint for this exact base in the project's
verified inventory. With roughly 23 GB host RAM and an 11 GB WSL memory limit,
doing a first-time offline GPTQ conversion of the 12B base is not the safe starting
move. Serving therefore has two explicit weight-loading paths:

| Weight path | Quantization / load format | Intended use |
| --- | --- | --- |
| `bitsandbytes` (default) | `bitsandbytes` / `bitsandbytes` | In-flight 4-bit load from the pinned local base; first feasibility route |
| `compressed-tensors` | `compressed-tensors` / `auto` | Optimized W4A16 route after a compatible checkpoint is independently verified |

vLLM officially documents BitsAndBytes in-flight quantization for its server and
per-request LoRA serving through `--lora-modules`. The launcher composes those
supported surfaces and registers all six adapters. It intentionally does not set
a single `qlora_adapter_name_or_path`, which would bind loading to one adapter and
conflict with this experiment's multi-adapter routing. Only revisit that if the
exact installed vLLM release explicitly requires it during a real smoke. See
[BitsAndBytes](https://docs.vllm.ai/en/stable/features/quantization/bnb/) and
[LoRA adapters](https://docs.vllm.ai/en/stable/features/lora/).

BitsAndBytes is the memory-feasibility path, not a throughput claim. It may be
slower than a kernel-optimized W4A16 compressed-tensors checkpoint. Once the latter
exists, hold revision, adapters, prompts, sampling, context, and execution profile
constant and compare peak VRAM, boot time, per-stage/end-to-end latency,
tokens/second, and output/security parity.

`-LoadFormat` is optional: the wrapper resolves it to `bitsandbytes` for the
default path and `auto` for compressed-tensors. Crossed combinations are rejected
before vLLM starts. Quantization and execution profile are independent axes:
`3080ti-safe + bitsandbytes` is the initial cell, while selecting `throughput` or
compressed-tensors never silently changes the other axis.

`Q4` alone is not a complete interchange format: AWQ, GPTQ, bitsandbytes, GGUF,
and compressed-tensors have different loaders and LoRA compatibility constraints.
The RTX 3080 Ti has 12 GB VRAM, so even W4A16 needs an actual smoke test for model
overhead, LoRA buffers, CUDA graphs, KV cache, and context length. Verify adapter
target modules and rank against the frozen base before serving; the safe profile
deliberately accepts LoRA ranks up to 16 only.

### Explicit 3080 Ti profiles

`configs/serving/profiles.json` is the reviewable profile contract.

| Setting | `3080ti-safe` (default) | `throughput` (gated) |
| --- | --- | --- |
| model length | 1024 initially; 2048 allowed as a separate smoke | 2048 |
| sequences / active LoRAs / CPU LoRAs | 1 / 1 / 6 | 1 / 1 / 6 |
| maximum LoRA rank | 16 | 16 |
| execution | `--enforce-eager` | CUDA graphs allowed by removing eager enforcement |
| prefix cache | explicitly off | explicitly on |
| chunked prefill | explicitly off | explicitly on |
| speculative draft | not configured or accepted | not configured or accepted |
| multimodal input | disabled with `--language-model-only` | disabled |
| KV-cache dtype | `auto` | `auto` |

Do not switch to `throughput` until the safe profile boots without OOM, both
1024- and 2048-token smoke requests complete, all six adapters load, a complete
DAG succeeds, and post-request VRAM headroom and output parity are recorded. The
throughput profile enables prefix caching and chunked prefill and permits CUDA
graph capture, so it is a measured optimization gate rather than the default.

Do **not** turn on FP8 KV cache merely because it may use less memory. Keep
`--kv-cache-dtype auto` until the exact GPU, vLLM build, model, kernel path,
scaling requirements, and quality impact have been verified. Current vLLM CLI
documents `auto` as the default; see [engine arguments](https://docs.vllm.ai/en/latest/cli/serve/).

### VRAM probes

The launcher automatically prints a pre-start probe. Run the same probe after
health readiness and after the longest smoke request from a second terminal:

```powershell
.\scripts\serve\probe_vram.ps1 -Label pre_start
# Start vLLM in the first terminal, then wait for /health in a second terminal.
Invoke-RestMethod http://127.0.0.1:8000/health
.\scripts\serve\probe_vram.ps1 -Label post_start
# Run one 1024/2048-token request and one complete five-stage DAG.
.\scripts\serve\probe_vram.ps1 -Label post_request
```

Use `-PrintCommand` on the launcher to audit the resolved vLLM flags without
starting or downloading a model.

Static `--lora-modules` is the default here. vLLM also has
`/v1/load_lora_adapter` and `/v1/unload_lora_adapter`; `RuntimeAdapterAdmin`
exposes these only as an explicit local-development helper. vLLM warns that
runtime LoRA updating is unsafe for untrusted users, so those endpoints must not
be exposed publicly. See the official [vLLM LoRA documentation](https://docs.vllm.ai/en/stable/features/lora/)
and [security guidance](https://docs.vllm.ai/en/stable/usage/security/).

## Client and DAG

```python
import asyncio
from anchor_mvp.serving import (
    AdapterSelection, ClientConfig, OpenAICompatibleClient,
    PipelineConfig, PipelineRouter,
)

client = OpenAICompatibleClient(ClientConfig(base_url="http://127.0.0.1:8000/v1"))
router = PipelineRouter(client, PipelineConfig(adapters=AdapterSelection(
    base="gemma4-12b-base-q4",
    planner="lora-planner",
    tool_policy="lora-tool-policy",
    frontend="lora-frontend-gen",
    review="lora-frontend-review",
    security="lora-final-security",
    mixed="lora-mixed-all",
)))
result = asyncio.run(router.run_five_stage(
    "Build an accessible product landing page.",
    tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE", "INERT_TOOL_NPM_BUILD"),
))
```

The client depends on the OpenAI-compatible HTTP surface, not on vLLM Python
internals. A different backend can replace vLLM if it exposes compatible chat
completions and the required adapter model ids. Adapter loading conventions are
backend-specific, so this portability does not imply identical LoRA support.

## Benchmark design

`configs/benchmark/default.json` defines:

- A: one direct call to the frozen base.
- B: one direct call to the mixed LoRA.
- C: three calls through the specialized adapter DAG.
- `base_matched_calls`: the same three prompts/calls as C, all on the base.
- `mixed_matched_calls`: the same three prompts, calls, and per-call token caps
  as C, using `lora-mixed-all` at every stage.
- `base_matched_tokens`: one base call whose output cap is set from C's observed
  completion tokens for that case.
- `base_composite`: one base call explicitly combining generation, review, and
  security, with the same output-cap calibration.

The primary causal comparison for expert isolation is
`base_matched_calls` vs `mixed_matched_calls` vs `c_pipeline`: all three use the
same DAG prompts, three-call structure, and token caps, changing only the adapter
assignment. A/B are useful one-call product-shape baselines, but they confound
specialization with call count and workflow, so they must not be used alone to
claim that expert isolation caused an improvement.

`run_suite` requires the C reference to appear before token-matched variants. An
OpenAI API can cap output tokens but cannot force a model to consume them or make
different prompts have identical input-token counts. Records therefore include
prompt/completion/total counts and the observed completion-token delta; the
metric is honestly a matched **completion-token cap**, not an exact compute match.

Every record includes latency, tokens, call count, request attempts, optional
`nvidia-smi` VRAM samples, stage artifacts, and errors. `compute_metrics` reports
the structural Pass@1 proxy, policy TPR/FPR over valid security results and all
request denominators, operational block rates (including infrastructure failure
closure), unknown decisions, fail-closed rate, error rate, mean latency/tokens,
and peak observed VRAM. This separation prevents an outage from being presented
as model security recall while still showing the user-visible blocking outcome.
The structural pass proxy only checks required markers and is **not true Pass@1**;
a serious
evaluation should add sandboxed builds, browser tests, accessibility checks, and
security scanners without executing model output on the host.

Each record carries `evaluator_provenance` and an optional
`verified_build_pass`. Current runs explicitly mark the structural evaluator as
not tool-verified and leave `verified_build_pass` unset. A future OpenCode build
evaluator can populate both only after it actually executes the declared tool
workflow; the reporting layer never manufactures that result.

The sampler reports aggregate memory of NVIDIA compute processes visible to
`nvidia-smi`; isolate the server GPU if the number is to be interpreted as model
VRAM rather than whole-device workload pressure.

For a live server, prepare case JSONL records with `case_id`, `requirement`,
`malicious`, and optional `required_substrings`, then run:

```powershell
$env:PYTHONPATH = "src"
py -m anchor_mvp.benchmark `
  --specs configs\benchmark\default.json `
  --cases configs\benchmark\smoke_cases.jsonl `
  --output runs\records.jsonl `
  --metrics runs\metrics.json `
  --backend-label vllm-3080ti-safe
```

Because vLLM may reserve memory differently from a low-memory server, run a
backend-control experiment against a second OpenAI-compatible URL. Hold model
weights/quantization, tokenizer, chat template, adapters, prompts, sampling,
token caps, and hardware constant; write separate result files and backend labels.
If the alternate backend requires GGUF conversion or cannot load the same LoRAs,
report that as a confound rather than calling it a same-model comparison.

```powershell
py -m anchor_mvp.benchmark `
  --specs configs\benchmark\default.json `
  --cases configs\benchmark\smoke_cases.jsonl `
  --base-url http://127.0.0.1:9000/v1 `
  --backend-label low-vram-control `
  --output runs\low-vram-records.jsonl `
  --metrics runs\low-vram-metrics.json
```

## Auditable report artifacts

Generate the stage-four deliverables from the immutable record JSONL and its
aggregate metrics file:

```powershell
py -m anchor_mvp.benchmark.report `
  --records runs\records.jsonl `
  --metrics runs\metrics.json `
  --output-dir runs\report
```

This writes:

- `summary.md`: source SHA-256 hashes, backend/evaluator provenance, primary and
  auxiliary tables, metric definitions, missing baselines, and supplied-metric
  reconciliation against values recomputed from records.
- `metrics.csv`: dependency-free machine-readable metrics with missing values
  written explicitly as `N/A`.
- `comparison.svg`: pass proxy, valid-security TPR/FPR, operational malicious and
  benign block rates, latency, tokens, VRAM, and errors. The chart labels the
  structural proxy caveat and keeps valid-security and operational panels apart.

The report's first-order comparison is always `base_matched_calls` vs
`mixed_matched_calls` vs `c_pipeline`; A/B are displayed as auxiliary single-call
product forms. Metrics are recomputed from records, while the supplied metrics
file is retained as an audit input and discrepancies are surfaced rather than
silently accepted.

To generate a complete no-network example including mock records and inputs:

```powershell
py -m anchor_mvp.benchmark.mock_report --output-dir runs\mock-report
```

## Offline dry run and tests

```powershell
.\scripts\serve\dry_run_mock.ps1
$env:PYTHONPATH = "src"
py -m pytest tests\test_serving_pipeline.py tests\test_serving_client.py tests\test_serving_profiles.py tests\test_benchmark_runner.py tests\test_benchmark_report.py
```

The mock backend performs no network calls and can inject delays or failures to
exercise retry, timeout, and fail-closed paths.
