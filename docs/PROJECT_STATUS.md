# Project status and remaining work

Last evidence refresh: 2026-07-18. This file is the conservative claim ledger
for the current formal path. It distinguishes implementation, offline evidence,
live evidence, and completed experiments.

## Executive status

| Layer | Current state | Claim boundary |
| --- | --- | --- |
| Public full bank | **Implemented and offline-validated** | 19,008 train-only candidate tasks and 95,040 dependency-bound work orders exist; this is not 19,008 accepted Gold chains |
| Formal WebUI | **Implemented and offline-tested** | Formal SWE-bench is the default target; Start remains disabled when the coordinator gate is blocked |
| Patched CC Switch | **Component attested** | Profiles, pricing metadata, launcher, and route manifest exist; container-to-router live reachability is separate evidence |
| Patched OpenCode v3 source | **Clean-apply and contract-tested** | The current patch identity comes only from `patches/opencode/patch-manifest.json`; an older binary bundle is not automatically current |
| Generic train runtime | **READY: implemented and offline-tested** | All 19,008 public train rows use `repository + base_commit` materialization, patched OpenCode, the digest-pinned train sandbox, terminal `anchor-validate compile|test`, cleanup, and an HMAC-bound `real_sandbox_self_verified` receipt contract; no provider pilot is claimed |
| Official TestSpec / heldout evaluation | **NOT READY; independent** | Official SWE-bench PASS/FAIL is not required for train distillation and no official TestSpec or heldout run is claimed |
| Formal Gold exporter | **Implemented and offline-tested; no live export yet** | The authenticated full-bank `training-export` directory does not exist |
| Formal-v3 snapshot | **Not published** | No A–F formal-v3 training may start from a growing or synthetic directory |
| Formal-v3 A–F training | **Configured; not run** | The 9 GiB low-memory profile is a 64-token truncated control, not full-context training |
| Formal-v3 A–F evaluation | **Contract/integration work only; not run** | No formal-v3 heldout/GPU score or A=100 report is claimed |

The generic train execution contract is READY for the required one-task live
pilot. That is implementation and offline-test evidence, not proof that a
provider-backed task has completed. The WebUI/coordinator remains authoritative
for the current machine's live-start gate; a blocked gate must never be bypassed.

## Canonical full-bank boundary

The only formal source bank is `SWE-bench/SWE-bench` train revision
`7074ef12ea2a6f70a228943c1336553333c22786`:

- source rows and unique instance IDs: 19,008;
- dependency-bound work orders: 95,040, exactly five per source task;
- repositories: 35;
- source parquet SHA-256:
  `0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69`;
- deterministic source split: at most 17,105 train tasks and 1,903
  calibration-from-train tasks;
- locale routing: 9,504 `en-US` and 9,504 `zh-CN` assignments.

Locale routing is not completed translation. Official dev/test/heldout task
bodies, gold patches, test patches, hints, and oracle labels are not formal
training inputs. External heldout is represented in the training contract only
by content-free manifest and leak-audit hashes.

The public derived bank is under
`datasets/public/swebench-full-bank-v1/`. It is split into files below 50 MiB and
is governed by the attribution, secret, heldout, license, and large-file gates.
The old c10, 384+128, `automated_v3`, and Kimi mock directories remain historical
synthetic evidence and cannot feed formal-v3.

## Five-stage generic train runtime

Every task must preserve one identity across:

```text
planner -> tool_policy -> domain_builder -> domain_review -> security
```

The train split does not provide the private official TestSpec/test patch needed
for an official benchmark verdict. Each candidate is therefore materialized as
its public `repository + base_commit` in a disposable WSL2/Podman workspace. The
builder runs the patched OpenCode tool path there, preserves real tool
calls/results and the cumulative final diff, and receives only planner-proposed
plus tool-policy-approved commands. A terminal `anchor-validate compile|test`
result must bind the final workspace state; validation followed by another edit
is rejected.

The generic train runtime, immutable train-sandbox image binding, validator
binding, fixed-target CC Switch relay, HMAC receipt binding, crash/resume logic,
cleanup checks, and fail-closed offline tests are implemented. The complete
19,008-row train bank is therefore READY for this
`real_sandbox_self_verified` evidence tier. This readiness does **not** mean a
provider pilot or full-bank run has happened. The first live launch must be
exactly one task at concurrency one. It becomes live evidence only after the
provider-backed five-stage chain, final validation, supervisor recomputation,
HMAC receipt, and cleanup all succeed.

Official TestSpec and heldout evaluation remain a separate NOT READY track.
They never gate train-trajectory generation, never enter train prompts or Gold,
and are the only route to a later official SWE-bench benchmark claim. The current
absence of `artifacts/swebench/full-bank-live-v1/training-export/` means no live
Gold export or immutable formal-v3 snapshot is claimed.

## Patched OpenCode and CC Switch provenance

The canonical OpenCode source manifest is
`patches/opencode/patch-manifest.json`. It pins upstream 1.17.18 commit
`b1fc8113948b518835c2a39ece49553cffe9b30c`, the current patch SHA-256, Bun
version, tool-contract version, and required behavior tests. The current patch
SHA-256 is
`b61617124977d156f5702be23b46e7564325a4e796037e6faaa89ed42543106b`;
it matches the manifest, passes a clean `git apply --check` against that
baseline, and passes the Python patch contract tests. Read the SHA dynamically
rather than copying this evidence snapshot:

```powershell
$OpenCodePatch = Get-Content patches\opencode\patch-manifest.json -Raw |
  ConvertFrom-Json
$OpenCodePatch.patch_sha256
```

Do not copy a SHA from an older report. A binary is current only when its bundle
manifest binds the same source manifest and all verification/behavior checks
pass.

Patched CC Switch owns provider/model selection, exact MAX-tier routing, pricing,
token/cost accounting, and request statistics. The project ships pinned GLM-5.2
MAX and Kimi-K3 MAX route profiles plus an attested local component manifest.
That manifest is a component gate, not proof that a WSL/Podman model container
has reached the route during a real task.

## Formal control plane

`http://127.0.0.1:8765/` is the standard operator surface. Formal SWE-bench is
selected by default. The page exposes:

- Start, cooperative safe Stop, and exact-checkpoint Resume;
- concurrency, reconnect delay, retry limit, and process-only credential input;
- provider URL/model/protocol/MAX route binding and pricing metadata;
- requests, exact known token/cost totals, throughput, ETA, errors, disconnect
  state, reconnect reason, and formal reason code;
- component, task-bank, generic execution-contract, image/route, localization,
  and live-start gates. Official TestSpec/heldout readiness is reported
  separately and is non-blocking for train distillation.

The formal target uses a fixed coordinator/config/argument shape and never falls
back to the legacy direct-API collector. A blocked gate disables Start before a
credential is stored or a provider request is sent.

## Authenticated Gold and immutable snapshot

`scripts/data/export_swebench_formal_gold.py` is the only formal bridge from a
terminal live coordinator run to the training projection. It requires:

- exact checkpoint/config/task/revision/execution-lock identity;
- all five stage artifacts and predecessor hashes;
- real builder tool calls/results, controlled session export, and exact diff;
- one final domain-review PASS and one security PASS;
- one terminal nontrivial zero-exit `anchor-validate compile|test` command whose
  parsed result matches an independent supervisor recomputation of final state;
- successful sandbox cleanup;
- an HMAC-authenticated `real_sandbox_self_verified` train receipt bound to the
  checkpoint, task, source-bank/candidate-shard identity, repository,
  train-sandbox image digest/ID, base commit, final patch, tool transcript,
  validation evidence, and complete stage lineage.

It excludes incomplete or unverified tasks while allowing a later authenticated
retry to recover a task that has an older failure event. A capped
`stopped_checkpoint_resumable` checkpoint is exportable, so successful prefixes
are usable without claiming the full bank completed. It removes explicit hidden
reasoning fields and never accepts a model-supplied PASS as a substitute for
the train receipt. The receipt explicitly declares
`not_official_swebench_pass=true`; independent heldout evaluation supplies any
later benchmark claim. The root-owned WSL train key is read only in-process.

The exporter code and offline tests are implementation evidence only. Because
the training export is absent, `configs/orchestration/full_v3_snapshot.yaml`
cannot publish `artifacts/formal_v3/dataset/manifest.json`; `training_ready`
remains false.

## Formal-v3 A–F training boundary

All six arms use the same frozen Q4/NF4 base and, when applicable, the same
immutable Gold snapshot:

| Arm | Structure | Budget |
| --- | --- | --- |
| A | frozen native Q4 base; no LoRA and no training | evaluation index baseline 100 |
| B | one mixed adapter reused at all five stages | rank 16; 10,387,456 trainable parameters |
| C | five independent full specialists | rank 16 each |
| D | five fixed small specialists | ranks `3/3/4/3/3`; exactly B-sized total |
| E | five calibration-adaptive specialists | each rank <=16; variable total budget |
| F | same adaptive mechanism as E | exact B-sized materialized total |

B–F use equal total and per-stage sample exposure derived from the actual frozen
train Gold count. E/F ranks are frozen from calibration-from-train before heldout
access. No checked-in `max_steps` placeholder can override the materialized
snapshot-sized schedule.

The local `formal_v3_lowmem_*` profile targets a 9 GiB peak and uses sequence
length 64 under `formal_v3_lowmem_truncated_v1`. It is deliberately marked
`full_trajectory_training=false`. Its one-epoch B–F control fixes the same
conservative `5e-5` learning rate for every arm; this is a fairness control and
not a convergence-optimality claim. No formal-v3 B–F schedule can materialize
while the snapshot is absent, and no formal-v3 A–F training run is currently
claimed.

Existing formal-v1/formal-v2 adapters and reports are historical feasibility
evidence only. They may not be copied into a formal-v3 registry or evaluation.

## Formal-v3 evaluation boundary

The evaluation contract fixes the same content-free heldout binding, A as the
frozen Q4 baseline normalized to index 100, B as one mixed adapter, and C/D/E/F
as five-stage serial runtime-LoRA swaps. Artifacts must be version-isolated and
bound to formal-v3 checkpoint/snapshot/schedule hashes. Implementation or offline
mock tests do not equal an executed heldout/GPU evaluation. There is currently no
formal-v3 A–F quality result to report.

## Safe operator sequence

These commands are read-only/offline:

```powershell
.\anchor.ps1 -Action status
.\anchor.ps1 -Action ui
.\anchor.ps1 -Action distill-swebench
```

Do not add `-ConfirmLive` until the recomputed coordinator report says
`live_start_allowed=true`. The first live command must be a one-task pilot:

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Concurrency 1 -MaxTasks 1
```

No provider pilot was executed in this evidence refresh. After that pilot's
authenticated receipt is verified, use the same exact checkpoint and increase
the **cumulative** cap through this Resume ladder:

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 8  -MaxTasks 16
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 16 -MaxTasks 48
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 24 -MaxTasks 96
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 30 -MaxTasks 156
```

Stop raising concurrency on repeated systemic provider/transport/resource
errors and Resume at the last clean tier. Authenticated successful chains are
hash-checked and skipped; failed or incomplete tasks remain retryable and never
enter Gold. After an eventual terminal live run, the correct order is Gold
export, immutable snapshot publication, A–F schedule materialization,
low-memory smoke/probe, then explicit A–F training and version-isolated
evaluation. Never train directly from the growing live output directory.

## Publication boundary

The public repository may contain code, configs, manifests, attribution, and
audited data shards below 50 MiB. It must not contain provider credentials,
private receipt keys, teacher/session bodies outside the approved public view,
heldout bodies, model weights, adapters, checkpoints, runtime logs, or private
evaluation records.

No Git tag, GitHub Release, or version package may be created without a new,
explicit user instruction.

This project was built with assistance from OpenAI GPT-5.6 SOL.
