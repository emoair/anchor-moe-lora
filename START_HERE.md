# Start here

[English](START_HERE.md) | [简体中文](START_HERE.zh-CN.md)

This is the shortest operator path for the current repository. It keeps four
different states separate: code present, offline checks passed, live evidence
captured, and training/evaluation completed.

## Three commands to remember

Run these from PowerShell. None of them reads an API key, starts a provider
request, starts GPU training, or opens heldout task bodies.

```powershell
Set-Location D:\LLM\anchor-moe-lora

# 1. Compact, read-only status.
.\anchor.ps1 -Action status

# 2. Start or reconnect to the local control panel.
.\anchor.ps1 -Action ui

# 3. Run the formal full-bank offline preflight only.
.\anchor.ps1 -Action distill-swebench
```

Open <http://127.0.0.1:8765/> after command 2. The page defaults to **Formal
SWE-bench + CC Switch + OpenCode** and exposes Start, safe Stop, checkpoint
Resume, concurrency, speed, tokens/cost, retry controls, connection state, and
the exact fail-closed reason. A process-only key may be entered in the password
field only after the formal live gate is ready. Starting the page by itself
consumes no quota.

## The formal data path

```text
public SWE-bench train bank: 19,008 task cards / 95,040 work orders
                               |
                               v
        planner -> tool_policy -> domain_builder -> domain_review -> security
                               |
                               v
     patched CC Switch + patched OpenCode + repo/base_commit train sandbox
                               |
                               v
      authenticated five-stage Gold export -> immutable formal-v3 snapshot
                               |
                               v
                 A / B / C / D / E / F training and evaluation
```

The source bank is pinned to `SWE-bench/SWE-bench` train revision
`7074ef12ea2a6f70a228943c1336553333c22786`. Its public derived bank contains
19,008 task cards and exactly five dependency-bound work orders per task. The
9,504 `en-US` and 9,504 `zh-CN` counts are deterministic routing assignments;
they do not claim that 9,504 Chinese bodies have already been generated.

The old c10, 384+128, and direct-API collectors remain useful synthetic
experiments. They are not the formal full-bank path and are never a fallback for
`distill-swebench`.

## Current evidence boundary

| Area | What exists now | What has **not** happened |
| --- | --- | --- |
| Public task bank | 19,008 train-only task cards and 95,040 work orders; deterministic train/calibration allowlists; files are sharded below 50 MiB | No task body from official dev/test/heldout is part of this bank |
| Web control plane | Formal mode is the default; lifecycle, provider profile, route, gate, speed, token/cost, retry, disconnect, and reason-code controls are implemented | The panel cannot override a failed formal gate |
| Patched CC Switch | Pinned route profiles, pricing metadata, component manifest, and offline attestation exist | A component attestation alone is not a successful container-to-router live probe |
| Patched OpenCode | The pinned source patch and v3 tool contract are recorded in `patches/opencode/patch-manifest.json`; patch `b61617124977d156f5702be23b46e7564325a4e796037e6faaa89ed42543106b` clean-applies and passes offline contract checks | A stale binary bundle must not be treated as the current v3 patch merely because a binary is present |
| Generic train sandbox | **READY** for all 19,008 `repository + base_commit` candidates: digest-pinned WSL2/Podman sandbox, terminal `anchor-validate compile|test`, final-state recomputation, cleanup, HMAC receipt, resume, and offline tests | No provider-backed one-task pilot or full-bank run is claimed |
| Official TestSpec / heldout | Independent evaluation track | **NOT READY**; it never gates train distillation and no official SWE-bench PASS is claimed |
| Formal Gold bridge | `scripts/data/export_swebench_formal_gold.py` implements fail-closed export from authenticated complete chains | `artifacts/swebench/full-bank-live-v1/training-export/` does not exist; no formal-v3 Gold snapshot has been published |
| A–F training | Snapshot-sized schedules and a 9 GiB/64-token low-memory profile are implemented | The low-memory profile is not full-context, and formal-v3 A–F training has not run |
| Formal-v3 evaluation | A=100 and A–F bindings are specified; evaluation integration may be checked offline | No formal-v3 A–F heldout/GPU evaluation result is claimed |

The generic train contract is READY in implementation and offline tests. That is
not live evidence: this repository update did **not** execute a provider-backed
pilot. Trust the WebUI/coordinator's current `live_start_allowed` result; a
failed gate cannot be overridden by pasting a key. Training also remains blocked
until the authenticated Gold exporter creates a training export and
`prepare_full_v3_snapshot.py` freezes it.

The next authorized live action is the concurrency-one, one-task pilot in
[Quickstart section 4](QUICKSTART.md#4-start-formal-live-only-after-all-gates-pass),
not an official TestSpec run and not the full bank. After its authenticated
receipt is verified, Resume the same checkpoint with cumulative caps
`c8/cap16 -> c16/cap48 -> c24/cap96 -> c30/cap156`. Authenticated successes are
skipped and remain usable; failed or incomplete tasks are retried and never
enter Gold.

## A–F means all six controls

| Arm | Controlled structure |
| --- | --- |
| A | Same frozen native Q4 base, no LoRA and no training; evaluation index baseline 100 |
| B | One mixed rank-16 LoRA reused across all five stages |
| C | Five independent full rank-16 specialists |
| D | Five fixed small specialists with ranks `3/3/4/3/3`, exactly budget-matched to B |
| E | Five calibration-adaptive specialists, each rank `<=16`, with variable total budget |
| F | The same adaptive mechanism as E, with total materialized parameters exactly matched to B |

B through F must use the same immutable snapshot and equal total/per-stage sample
exposure. E and F may derive ranks only from the calibration-from-train split.
Formal-v2 adapters, reports, and registries are not formal-v3 evidence.

## Verify OpenCode provenance without copying a stale hash

The manifest is authoritative. Read the current patch hash from it:

```powershell
$OpenCodePatch = Get-Content patches\opencode\patch-manifest.json -Raw |
  ConvertFrom-Json
$OpenCodePatch.baseline_commit
$OpenCodePatch.patch_sha256
$OpenCodePatch.tool_contract_version
```

Never copy a patch SHA from an old report. A current bundle is acceptable only
when its source binding matches this manifest and its verification/behavior
checks pass.

## Where to read next

| Need | English | 简体中文 |
| --- | --- | --- |
| Complete operator guide | [Quickstart](QUICKSTART.md) | [快速上手](QUICKSTART.zh-CN.md) |
| Formal contract | [Canonical contract](docs/CANONICAL_DISTILLATION_CONTRACT.md) | [正式契约](docs/CANONICAL_DISTILLATION_CONTRACT.zh-CN.md) |
| Full-bank coordinator | [Coordinator guide](docs/swebench_ccswitch_live.md) | [协调器说明](docs/swebench_ccswitch_live.zh-CN.md) |
| A–F training | [Formal-v3 training](docs/formal_v3_training.md) | [Formal-v3 训练](docs/formal_v3_training.zh-CN.md) |
| Dashboard | [Dashboard guide](docs/distillation_dashboard.md) | [仪表盘指南](docs/distillation_dashboard.zh-CN.md) |
| Exact current claims | [Project status](docs/PROJECT_STATUS.md) | [项目状态（英文事实表）](docs/PROJECT_STATUS.md) |
| All public documents | [Documentation index](docs/README.md) | [文档索引](docs/README.zh-CN.md) |

Credentials remain process-only. Do not put them in YAML, `.env`, arguments,
logs, Git, or examples. Do not create a Git tag or publish a Release without a
new explicit user instruction.

This project was built with assistance from OpenAI GPT-5.6 SOL.
