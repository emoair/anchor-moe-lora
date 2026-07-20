# Anchor-MoE-LoRA quickstart

[English](QUICKSTART.md) | [简体中文](QUICKSTART.zh-CN.md)

This guide describes the formal full-bank path. It does not promote historical
synthetic collectors to formal evidence.

## 0. Know the current boundary

The formal corpus is the pinned `SWE-bench/SWE-bench` **train** projection:

- 19,008 public task cards;
- 95,040 dependency-bound work orders, exactly five per task;
- `planner -> tool_policy -> domain_builder -> domain_review -> security`;
- 9,504 `en-US` and 9,504 `zh-CN` routing assignments.

Routing assignments are not completed translations, and task cards are not
accepted Gold. A task becomes formal Gold only after the real patched OpenCode
tool trace, review/security PASS, at least one successful nontrivial public
validation command, cleanup, and an HMAC-authenticated train receipt all agree
on the same final patch. This is `real_sandbox_self_verified` evidence, not an
official SWE-bench PASS claim.

At the 2026-07-18 evidence refresh:

- the public bank, formal WebUI, coordinator, Gold exporter, immutable-snapshot
  bridge, A-F schedule code, and formal-v3 evaluation integration are
  implemented and offline-tested;
- the generic `repository + base_commit` train runtime is READY for all 19,008
  candidates at the `real_sandbox_self_verified` evidence tier;
- the canonical OpenCode v3 patch clean-applies and passes its offline contract
  checks; read its identity from `patches/opencode/patch-manifest.json`;
- no provider-backed one-task pilot has been executed;
- no 19,008-task formal LIVE run has been completed;
- `artifacts/swebench/full-bank-live-v1/training-export/` is absent;
- official TestSpec/heldout evaluation is an independent NOT READY track;
- formal-v3 A-F training and heldout/GPU evaluation have not run.

READY here means implemented and offline-tested, not live-demonstrated. The
WebUI/coordinator's current `live_start_allowed` value remains authoritative;
supplying a key does not override a failed gate. Every downstream formal-v3
experiment remains blocked until authenticated live Gold is exported and frozen.

## 1. Open PowerShell in the repository

```powershell
Set-Location D:\LLM\anchor-moe-lora
```

`anchor.ps1` looks for Python 3.10+ in this order: `-PythonExe`,
`ANCHOR_MVP_PYTHON`, `.venv`, the `anchor-mvp` Conda environment, then
`py -3.11`. To pin the known local environment for this PowerShell window:

```powershell
$env:ANCHOR_MVP_PYTHON = "$HOME\.conda\envs\anchor-mvp\python.exe"
& $env:ANCHOR_MVP_PYTHON -c "import sys; print(sys.executable); print(sys.version)"
```

If that file does not exist, create/activate an equivalent Python 3.10+
environment and install the repository before continuing:

```powershell
conda create -n anchor-mvp python=3.11 -y
conda activate anchor-mvp
python -m pip install -e ".[teacher,dev]"
$env:ANCHOR_MVP_PYTHON = (Get-Command python).Source
```

Install the `training` and `serving` extras only on the machine that will run
those later stages. Do not put a provider credential in a YAML file, `.env`,
command argument, or Git file.

## 2. Run the three quota-free commands first

These commands do not read a provider key, start a provider request, start GPU
training, or open heldout task bodies.

```powershell
# Compact, read-only state.
.\anchor.ps1 -Action status

# Start or reconnect to the local control panel.
.\anchor.ps1 -Action ui

# Run only the formal coordinator's offline gates.
.\anchor.ps1 -Action distill-swebench
```

Open <http://127.0.0.1:8765/> after `-Action ui`. Formal SWE-bench is selected by
default. The panel shows component, bank, execution, image/route, localization,
and live-start gates plus the exact reason code. It also exposes concurrency,
retry/reconnect controls, safe Stop, checkpoint Resume, throughput, ETA,
request/token/cost totals, and backend-disconnect state.

The bank and generic train execution contract are expected to report READY once
their current content hashes match. This does not mean a live pilot ran. Start
stays disabled whenever any required train gate fails, before a credential is
retained or a child process is created. Official TestSpec/heldout readiness is
reported separately and does not block train distillation.

## 3. Verify the patched OpenCode and CC Switch identities

Never copy an OpenCode hash from an old report. Read the current source binding:

```powershell
$OpenCodePatch = Get-Content patches\opencode\patch-manifest.json -Raw |
  ConvertFrom-Json
$OpenCodePatch.baseline_commit
$OpenCodePatch.patch_sha256
$OpenCodePatch.tool_contract_version
```

The manifest currently binds upstream OpenCode 1.17.18 and tool contract
`anchor.execution-tool-contract.v3`. A binary is current only if its bundle
manifest binds this same source manifest and the verification/behavior checks
pass.

Patched CC Switch owns provider/model switching, exact MAX-tier routing,
pricing, and request/token/cost accounting. The formal profiles are GLM-5.2 MAX
and Kimi-K3 MAX. A component manifest marked ready is not proof that a
WSL/Podman sandbox reached that route during a real task.

For the longer offline component diagnostic:

```powershell
.\anchor.ps1 -Action preflight -AllowIncomplete
```

`-AllowIncomplete` is diagnostic only. It never authorizes LIVE.

## 4. Start formal LIVE only after all gates pass

The 19,008 public train rows are generic `repository + base_commit` work items.
They do not contain the private TestSpec/test patch required for an official
SWE-bench verdict. The train path is therefore the separately named
`real_sandbox_self_verified` tier: patched OpenCode works in a disposable,
digest-pinned WSL2/Podman train sandbox, the final tool call must be
`anchor-validate compile|test`, the supervisor independently recomputes the
final state, cleanup must succeed, and the resulting receipt is HMAC-bound. Its
manifest explicitly records `not_official_swebench_pass=true`.

This generic train contract is implemented and offline-tested. No provider-backed
one-task pilot was performed in this repository update. Official TestSpec and
heldout evaluation remain independent and NOT READY; they do not block train
distillation and never enter train prompts or Gold.

Do not continue while the recomputed offline report says
`live_start_allowed=false`. The WebUI at <http://127.0.0.1:8765/> is the
recommended operator surface: enter the key in its password field only after
Start is enabled. The key is passed only to the selected child environment,
cleared from the browser field, and is not written to config, status, logs, or
Git.

For CLI operation, put the credential only in the current PowerShell process.
The very first live launch is deliberately restricted to concurrency one and
one cumulative task:

```powershell
$env:ARK_CODING_API_KEY = '<paste for this PowerShell process only>'

.\anchor.ps1 -Action distill-swebench -ConfirmLive -Concurrency 1 -MaxTasks 1
```

Inspect the content-free status/checkpoint and verify the pilot produced a
complete five-stage chain, terminal validator result, successful cleanup, and a
valid train receipt. A provider response alone is not success. If the pilot
fails, Resume the same checkpoint after correcting the cause; do not delete
`status.json`, start a duplicate run, or export the failure to Gold.

After the pilot receipt is verified, raise concurrency through this exact
checkpoint-preserving ladder. `MaxTasks` is a **cumulative cap**, not the number
of new tasks in that wave:

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 8  -MaxTasks 16
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 16 -MaxTasks 48
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 24 -MaxTasks 96
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 30 -MaxTasks 156
```

Each cap ends as `stopped_checkpoint_resumable`. Resume hash-verifies and skips
already authenticated successes, so every successful prefix remains usable.
Failed or incomplete tasks remain retryable and never contaminate Gold. Stop
increasing concurrency on repeated systemic provider, transport, route, RAM,
disk, or sandbox errors and Resume at the last clean tier. There is no
product-wide hard maximum, but reaching provider concurrency 30 does not prove
the local machine can sustain 30 complete sandboxes.

When a clean tier is established, Resume the same checkpoint with the next
cumulative cap or leave the cap unset for the full 19,008-task bank. Remove the
process-local credential after the child exits:

```powershell
Remove-Item Env:ARK_CODING_API_KEY -ErrorAction SilentlyContinue
```

## 5. Export authenticated Gold, then freeze the snapshot

Run these commands only after the coordinator reaches a terminal state. They do
not call a provider or start GPU training.

```powershell
$Python = $env:ANCHOR_MVP_PYTHON

# Exports only complete, HMAC-authenticated self-verified train chains.
& $Python scripts\data\export_swebench_formal_gold.py

# Publishes the immutable formal-v3 dataset snapshot.
& $Python scripts\data\prepare_full_v3_snapshot.py `
  --config configs\orchestration\full_v3_snapshot.yaml

Test-Path artifacts\swebench\full-bank-live-v1\training-export\partitions\manifest.json
Test-Path artifacts\formal_v3\dataset\manifest.json
```

The exporter excludes a task when its complete five-stage chain, real tool
trace, qualifying validation result, diff, cleanup, review/security decision,
train-sandbox image binding, source-bank/candidate-shard binding, or private
train receipt is missing. A malformed or tampered authenticated artifact fails
the export closed. Capped
`stopped_checkpoint_resumable` runs are exportable, so already authenticated
successes remain usable while incomplete tasks remain retryable. The export
does not claim official SWE-bench PASS and does not retain hidden reasoning.
Never train directly from the growing live output directory.

For an intermediate capped checkpoint, pass a new versioned `--output-dir`;
exports are immutable. Keep the default canonical `training-export` path for
the snapshot you intend to freeze. A capped manifest is usable as an explicit
partial dataset but cannot support a full-bank completion claim.

If either `Test-Path` result is `False`, stop. Do not substitute historical
`data/automated_v3`, c10, 384+128, mock, or partial-Gold directories.

## 6. Materialize and train all A-F controls

The formal matrix is:

| Arm | Controlled structure |
| --- | --- |
| A | Frozen native Q4 base, no LoRA and no training; evaluation index baseline 100 |
| B | One mixed rank-16 LoRA reused across all five stages |
| C | Five independent full rank-16 specialists |
| D | Five fixed small specialists with ranks `3/3/4/3/3`, exactly budget-matched to B |
| E | Five calibration-adaptive specialists, each rank `<=16`, variable total budget |
| F | The E mechanism with total materialized parameters exactly matched to B |

B-F use the same immutable snapshot and equal total/per-stage sample exposure.
E/F ranks are frozen from calibration-from-train before heldout access.

First run the schedule-only preflight:

```powershell
.\scripts\train\formal_v3_preflight.ps1
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm A
```

Then use an explicit resource smoke/probe before any long run:

```powershell
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm smoke -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm probe -Execute
```

Only `-Execute` starts a GPU job. Run B-F one at a time on a single GPU:

```powershell
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm B -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm C -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm D -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm E -AllocationManifest <E.json> -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm F -AllocationManifest <F.json> -Execute
```

The local low-memory profile targets about 9 GiB but truncates sequences to 64
tokens. It is explicitly not full-context training. Its B-F control uses the
same conservative one-epoch `5e-5` learning rate; no arm receives private
hyperparameter tuning. A higher-precision cloud profile must remain a separate,
versioned experiment.

## 7. Evaluate A-F without reusing formal-v2 artifacts

After all formal-v3 registries are complete, use a new version ID. The first
command is an offline preflight and does not read heldout bodies:

```powershell
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001 -Finalize
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001
```

Only an explicitly authorized command may open heldout cases and run the GPU
evaluation:

```powershell
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 `
  -Execute `
  -AuthorizeHeldoutAccess
```

Use `-Resume` only with the exact same version/output checkpoint. A is normalized
to 100; B is the single mixed adapter; C-F are five-stage serial runtime-LoRA
swaps. Formal-v1/v2 adapters, reports, and registries are not formal-v3 evidence.

## 8. Fast troubleshooting

- **Start disabled / BLOCKED:** read the formal reason code. Do not paste a key
  or switch to the synthetic collector.
- **Web page shows disconnected:** the frontend lost its local backend. Run
  `.\anchor.ps1 -Action ui`, then reload `http://127.0.0.1:8765/`.
- **HTTP 400 `invalid_url`:** a tool received prose or a URL without
  `http://`/`https://`. Fix the actual URL field; do not retry malformed text.
- **HTTP 499 / context canceled:** the client, network, or timeout canceled the
  request before completion. Verify the checkpoint, route, and timeout, then use
  exact Resume rather than starting a duplicate run.
- **Unknown token/cost:** the panel reports exact provider usage only. Unknown is
  not silently estimated from text length.
- **Chinese count is 9,504 but localization is missing:** that number is the
  routing assignment, not proof of generated Chinese text.
- **Training preflight is blocked:** verify the authenticated training export
  and immutable snapshot exist. Never lower the gate to make a command run.

## 9. Historical collector and publication boundary

The explicit legacy entry point remains available for synthetic experiments:

```powershell
.\anchor.ps1 -Action distill-synthetic -ConfirmLegacySynthetic
```

It calls a compatible teacher directly and does not produce the real patched
OpenCode tool/evaluator evidence required by formal-v3. Its c10 and 384+128
configs are historical controls, not a fallback for the full bank.

Public Git scope is code, configs, documentation, audited public train
projection shards below 50 MiB, and content-free manifests/audits. Exclude
provider keys, private HMAC keys/receipts, model/session bodies outside an
approved public view, heldout bodies, weights, adapters, checkpoints, runs,
logs, and private evaluation records.

Do not create a Git tag, GitHub Release, or version package without a new,
explicit instruction.

This project was built with assistance from OpenAI GPT-5.6 SOL.
