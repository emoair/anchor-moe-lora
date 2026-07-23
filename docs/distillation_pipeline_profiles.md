# Versioned distillation pipeline profiles

## Status and scope

This document defines the first checked-in distillation pipeline profile:
`task-level-moe-lora-v1`.

The profile is a metadata-only routing boundary **after** authenticated
canonical five-stage Gold. It does not alter the SWE-bench source bank,
teacher prompts, provider routes, CC Switch, patched OpenCode, sandbox,
validator, HMAC receipt, checkpoint/resume logic, or canonical Gold bytes.
It also does not authorize live execution, training, a release, a provider
request, a model load, or GPU work.

The execution core remains
`anchor.swebench-five-stage-execution-core.v1`. A later
`frozen-prefix-qreader-v2` profile may consume the same immutable canonical
Gold through a separate TaskBoard/scaffold projection, but it must not mutate
this v1 profile or silently reinterpret its direct five-partition output.

## Why the boundary is after canonical Gold

The existing execution coordinator is deliberately strict and SWE-bench
specific. It fixes the five stages, task/work-order schemas, real repository
workspace, validator commands, and authenticated execution receipt. Treating
an arbitrary prompt template as a profile field would weaken that contract and
would prevent the two consumers from reusing the same expensive teacher and
sandbox execution.

The v1 profile therefore authenticates the common core and selects only this
post-Gold view:

| Canonical stage | Task-level expert | Partition file |
| --- | --- | --- |
| `planner` | `planner` | `data_plan.jsonl` |
| `tool_policy` | `tool_policy` | `data_tool_policy.jsonl` |
| `domain_builder` | `frontend_gen` | `data_frontend.jsonl` |
| `domain_review` | `frontend_review` | `data_review.jsonl` |
| `security` | `security_gate` | `data_security.jsonl` |

TaskBoard projection, natural-language scaffold production, frozen-prefix
Q-reader claims, and source-record rewriting are all false in this profile.

## Files and identities

The strict Draft 2020-12 schema is:

`configs/orchestration/distillation_pipeline_profile.schema.json`

The frozen metadata manifest has its own strict Draft 2020-12 schema:

`configs/orchestration/distillation_profile_freeze_manifest.schema.json`

The only accepted v1 profile is:

`configs/orchestration/profiles/task_level_moe_lora_v1.json`

The profile binds both published schemas and every dependency by
project-relative path, exact byte count, and SHA-256:

- `anchor.ps1`;
- the full-bank builder, implementation, and config;
- the coordinator implementation and config;
- the HMAC execution-contract and receipt-runtime implementations;
- the formal Gold exporter script and implementation;
- the profile implementation; and
- the profile CLI runner.

Dependency roles and order are closed. Unknown fields, missing bindings,
duplicate JSON keys, non-finite JSON values, path escapes, reused paths,
symlinks/reparse points, byte-count drift, hash drift, or terminal TOCTOU drift
are rejected.

## Offline preflight

Run the standalone profile authentication:

```powershell
py -3.10 scripts/data/run_distillation_profile.py preflight `
  --profile configs/orchestration/profiles/task_level_moe_lora_v1.json
```

The report is content-free. It reports hashes, paths, counts, and false
authorization flags only. It does not read canonical Gold or heldout bodies
and performs zero provider, network, model, and GPU requests.

The normal launcher remains unchanged when no profile is supplied:

```powershell
.\anchor.ps1 -Action distill-swebench
```

Select the authenticated v1 profile explicitly:

```powershell
.\anchor.ps1 -Action distill-swebench `
  -DistillationProfile task-level-moe-lora-v1
```

The launcher authenticates the profile first, then runs the existing
full-bank and coordinator offline gates. Supplying either `-SWEConfig` or
`-SWECoordinatorConfig` together with `-DistillationProfile` is rejected to
avoid ambiguous or partially authenticated configuration.

The profile never replaces the existing `-ConfirmLive` gate. A false
`live_authorized` value in the profile is intentional: only the existing
four-gate coordinator and an explicit operator action can attempt live work.
Likewise, profile authentication can never authorize training or formal
release.

## Freezing the profile identity

Freeze a content-free profile manifest below the local `artifacts` root:

```powershell
py -3.10 scripts/data/run_distillation_profile.py freeze `
  --profile configs/orchestration/profiles/task_level_moe_lora_v1.json `
  --output-dir artifacts/distillation-profiles/task-level-moe-lora-v1
```

Freeze is create-once and atomic. The manifest is validated against the
published freeze schema before and after serialization. It produces:

- `manifest.json`; and
- mandatory `manifest.json.sha256` in the exact
  `<sha256>  manifest.json\n` format.

Inputs are read as single byte snapshots and rechecked at the terminal
boundary. Existing output is never overwritten. The manifest repeats the
non-authorizing status and records no sample body.

## Fail-closed matrix

| Condition | Result |
| --- | --- |
| Profile/schema/dependency path, byte count, or SHA drift | Reject |
| Duplicate key, unknown field, non-finite value | Reject |
| Symlink, junction/reparse point, or project-root escape | Reject |
| Dependency changes between initial snapshot and terminal recheck | Reject |
| Profile used with direct SWE config overrides | Reject |
| Any live/training/formal/release authorization set true | Schema reject |
| TaskBoard, scaffold, or frozen-prefix claim enabled in v1 | Schema reject |
| Freeze destination already exists or lies outside `artifacts` | Reject |

## Migration discipline

This v1 profile is immutable once consumed. Changes to dependency identities
require a new profile version and a newly frozen manifest; consumers must not
relax or patch hashes locally. The future frozen-prefix profile must use its
own profile ID, schemas, materializer, manifest, and output root while binding
the same authenticated canonical execution by SHA. Existing 20-record
scaffold fixtures are not a large-scale producer and must not be promoted by a
profile toggle.

## Verification

```powershell
py -3.10 -m pytest tests/test_distillation_profiles.py -q
py -3.10 -m ruff check `
  src/anchor_mvp/swebench/distillation_profile.py `
  scripts/data/run_distillation_profile.py `
  tests/test_distillation_profiles.py
py -3.10 -m py_compile `
  src/anchor_mvp/swebench/distillation_profile.py `
  scripts/data/run_distillation_profile.py
```

The tests cover the real published Draft 2020-12 schema, physical dependency
authentication, mandatory sidecar, atomic create-once freeze, terminal drift,
duplicate keys, symlinks, authorization promotion, CLI metadata-only output,
launcher override conflict, and the unchanged no-profile docs action.
