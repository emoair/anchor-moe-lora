# Distillation handoff - 2026-07-20

This is a content-free handoff for the SWE-bench five-stage distillation
pipeline. It contains no credential, task body, model answer, reasoning trace,
tool input, held-out content, or execution record.

## Current boundary

- No distillation or training process is active. The loopback dashboard is
  running at `http://127.0.0.1:8765/`; its root endpoint returns HTTP 200.
  No provider credential is present in the current process, so no live run was
  launched.
- The current working tree passes the full zero-request launch preflight:
  `bank_ready=true`, `component_ready=true`,
  `execution_contract_ready=true`, `launch_ready=true`,
  `live_start_allowed=true`, and
  `reason_code=generic_train_execution_contract_ready`.
- The preflight observed 19,008 tasks and 95,040 five-stage work orders with
  `provider_requests=0`, `credentials_read=false`,
  `heldout_files_read=false`, and `sample_bodies_printed=false`.
- `training_ready=false` on the public-bank gate is expected before runtime
  Gold, real tool results, and localization manifests exist. It does not block
  a controlled first live distillation task.
- The content-free attestation remains `ready=false` for the separate official
  hidden-evaluation track. Its 14 remaining official gates do not block the
  generic training-repository distillation contract, which is ready.
- No formal-v3 Gold snapshot exists yet, and no training run has started.

## Latest pilot and identity break

- The latest live pilot used an older execution identity. It reached Planner
  and Tool Policy, then failed before Builder with
  `v3_builder_policy_proposal_binding_ambiguous`.
- That failed attempt was archived under
  `artifacts/swebench/full-bank-live-v1.failed-attempts/` and was not deleted.
- The coordinator now rejects duplicate canonical permission families before
  Tool Policy, allows one directed Planner correction, validates the full
  Planner/Policy authorization pair again before projecting the Builder trace,
  and keeps unauthorized or ambiguous calls fail-closed.
- The source-bank manifest, execution config, lock, coordinator, and adapter
  have all changed since the archived pilot. Never resume an old checkpoint.
  Start a new control-run/output identity at concurrency 1 and `MaxTasks=1`.
- A live c1 run has not yet been attempted under the current identity because
  no provider credential is present in the current process environment.

## Current reproducible identity

- Source-bank manifest SHA-256:
  `55c84236e42a803d029ce961fcce064b0b894b632e2789191c3ed1e106ebcf28`
- Coordinator SHA-256:
  `1ad9fa1ea7f87179fede9fed766f99784fd38a9937f5c171cf23ae6a3071e48c`
- Execution-v3 adapter SHA-256:
  `6967eef0b4009a0019cb23c9bfa88593d47db7fca51d4950d981609a877d349e`
- Route-diagnostics module SHA-256:
  `112b8ec3c08811cacb6f2bdba170b19876ad8b3a01e58cbe400e87c970925259`
- Execution lock SHA-256:
  `14463771cf16d1c88842e78b3415f4cdf349075e1be364fc09fd15650eec3084`
- Content-free execution attestation SHA-256:
  `ac8a234c5d9129dcdb251ecc7b98eea6895a10a6fcc1fb5ef425962377f1c2cd`
- All 16 lock-bound files match the lock, and the five-stage YAML binds the
  exact lock SHA above.

## Verification completed

- 264 coordinator, execution-v3, trace, runner, route-diagnostics,
  train-sandbox, wire/artifact, and patched-contract tests passed.
- The 25 projector and training-release contract tests also pass on the same
  final working-tree identity.
- 89 dashboard, control-plane, and PowerShell 5.1 launcher tests passed.
- 17 full-bank publication tests passed.
- Ruff and `git diff --check` passed.
- The full `anchor.ps1 -Action distill-swebench` offline preflight exited 0
  with no provider request.

## Publication scope

Remote commit `6befb99` is not by itself a clean-checkout reproduction of this
state. This handoff accompanies the targeted follow-up that includes the final
lock/config, coordinator and adapter, policy/models/runner/trace, route
diagnostics, train-sandbox validator and Containerfile, behavioral/wire/artifact
probes, their focused tests, the LF rule for `*.Containerfile`, and the
full-bank attribution/manifest hash fix. Unrelated historical training
artifacts remain outside that commit. Publication is branch-only: no tag or
release package is created.

## New-architecture training boundary

- The experimental Role-Conditioned Query Specialization work is not yet a
  promoted training design. Its first toy probe trained and evaluated on the
  same records, had role/task confounding, and could prove fitting rather than
  generalization. Treat its current result only as `proxy_signal_passed`.
- Keep canonical five-stage Gold unchanged: Planner, Tool Policy, Domain
  Builder, Domain Review, Security, plus the existing tool trace, diff,
  validator, cleanup, receipt, HMAC, attestation, held-out separation, and Gold
  selection contracts.
- Only after a complete canonical Gold record, run a separate deterministic
  TaskBoard projector. Its research sidecar must bind
  `source_gold_record_id`, `source_gold_sha256`, `task_bundle_sha256`, stage or
  expert, projector version, and projector-config SHA-256.
- Build all five role views from the same source task, split by source task
  before noise augmentation, and test causal evidence deletion, distractor
  invariance, JSON validity, cross-role separation, and unseen-task behavior.
  Never write LoRA ranks, target modules, KV modes, CUDA details, or attention
  losses back into canonical Gold.
- Logical TaskBoard sharing is not proof of full-layer physical KV-cache
  sharing. Q-only adapters preserve exact reuse only up to the first adapter
  frontier; downstream hidden states and KV diverge unless a separately tested
  approximation is quality-gated.

## Deterministic TaskBoard projector

> Status update: the v2 producer contract and synthetic fixture below are
> stable. The later v1 identities are retained only as historical context. The
> synthetic identities are integration fixtures, not a formal-v3 release.

- Current v2 policy/schema SHA-256 values are: projector config
  `b36945a2693183f0b213da403afcf8bb5611f46298bb849434e7b7d5854ba943`;
  `anchor.swebench-taskboard-sidecar.v2`
  `c1863bfab69ce2f2388ee37fadae951b14f3d5120706bab032cab3f9aab6bdc5`;
  `anchor.hierarchical-task-kv-segment-plan.v1`
  `80f760497e0d21f7d4d532db758362a800e845e6919b18b23958caabc7f155bf`;
  and `anchor.swebench-taskboard-projector-manifest.v2`
  `2cd9dc98d2b2865ed0586abfe291e3f6d161686597fcd2a7884c5762d2195347`.
- The final 15-record, two-bundle synthetic fixture manifest SHA-256 is
  `595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac`.
  Its fixed partitions are: `train/clean`, 5 records,
  `3b0f83991bdd5330de5261a922c1d0051c434418ff54b744957af02d7eb04927`;
  `train/noisy`, 5 records,
  `eb6eea5d13bb6416836b3463082398a8906ba85b09389c073cffc3ede12c613d`;
  and `calibration/clean`, 5 records,
  `f6d5534897d93ef2544023d7d5ac9bcac15d64e4fe14c2529a754e1253b16448`.
  It contains 89 segment references and 25 unique segments: 4 task-shared,
  16 downstream immutable, and 5 expert-private. Provider requests are zero.
- The v2 manifest fixes
  `segment_plan_location=outer_sidecar.segment_plan` and
  `execution_mode=decoupled_frozen_prefix_producer_required`. The producer
  derives relevant/current/future partitions from the fixed stage order rather
  than trusting row declarations. It emits one ordered prefix lineage and
  explicitly rejects full-generation KV sharing, token-level MoE, naive
  in-stack Q-LoRA exact reuse, independent segment concatenation, and
  shared-then-mask handling of forbidden content.
- Fresh-pycache focused validation passes 55 tests (37 projector and 18 freeze
  layer), Ruff, JSON/YAML parsing, exact fixture/physical-schema hash checks,
  and an independent static review with no remaining P0/P1 finding.

Historical v1 notes follow.

- A provider-free post-Gold projector is implemented in
  `src/anchor_mvp/swebench/taskboard_projector.py`, with the CLI at
  `scripts/data/project_swebench_taskboard.py` and configuration at
  `configs/research/swebench_taskboard_projector_v1.yaml`.
- It accepts only a pinned, frozen `anchor.training-snapshot.v2`, keeps the
  canonical Gold bytes unchanged, and emits research-only sidecars under
  `anchor.swebench-taskboard-sidecar.v1` plus a content-addressed
  `anchor.swebench-taskboard-projector-manifest.v1`.
- The projector emits five same-task role views, preserves the source split,
  applies deterministic noise only after splitting, enforces causal visibility,
  rejects input drift/cross-split reuse/secret material, and writes output
  atomically without provider requests.
- Its manifest and checked-in policy explicitly fix
  `split_group_key=task_bundle_sha256`,
  `task_id_cross_binding_key=training_record.task_board.task_id`, and
  `all_five_role_views_same_split=true`. The bundle hash includes the task ID
  and the ordered five-stage source bindings; publication revalidates the
  forward and reverse task/bundle mapping and the complete role set.
- Current projector identity SHA-256 values are:
  config `5d1207bf0a16f84c7cfa3448350b2a26f4127384664704b967c7c4280c9e63c9`,
  sidecar schema `654e9f7fddbe67885156c4e1fac9aa48c0b415c6fb52d3dcf501d53520b6f146`,
  manifest schema `75dc191849a0fba084dbc81064e7c5634c8727b41ad4b62522e3a28ecd727e53`,
  implementation `f92c7140ffc7f83ec19605b0e03aa774834b0020ecd3f413f7ae10a9075a6c39`,
  and CLI `19ff29a62ad22bf9bd39699cdfc69ae304636a9908facf1ef1cda5044c26af28`.
- Sixteen independent projector tests and 46 existing formal Gold/snapshot/training
  regression tests pass. They cover single-read byte/hash binding, source and
  output file-switch attacks, actual split-ID digest recomputation, structured
  credential rejection, causal visibility, grouping cross-binding, schema
  TOCTOU, and deterministic output. A regenerated 15-record synthetic fixture
  passes the training worktree's fixed three-partition outer-wrapper contract;
  the consumer's five focused suites pass 73 tests, its real
  `build_training_view` excludes current/future/forbidden content, and its
  materializer bridge reports `bridge_contract_passed=true`. The projector
  remains outside the live Gold namespace and does not make the experimental
  training architecture production-ready.
- The current research projector materializes authenticated inputs and output
  rows in memory. Its contract and 15-record bridge are verified, but full-bank
  peak memory/output volume are not yet qualified; run a measured shard or
  streaming pilot before large-scale projection.

## Generic execution and training release freeze

- `src/anchor_mvp/swebench/training_release.py` and
  `scripts/data/freeze_swebench_training_release.py` implement three metadata-
  only, fail-closed stages: `generic`, `source-disjoint`, and `release`.
- Every frozen stage uses explicit expected SHA-256 inputs, a single
  bytes/inode/stat snapshot, strict `manifest.json.sha256`, a final TOCTOU
  inventory check, and atomic directory publication.
- Schema identities are:
  `anchor.generic-train-execution-contract.v1` / SHA
  `63c699fdb7932b9fe1593b044d6a588bb4234b349816ce70f65568ab7b0f0b3a`,
  `anchor.swebench-source-disjoint-manifest.v2` / SHA
  `2a2aae532c25b324a96b929a6a396d55d051c765258a5da0ebb7547724c68f6b`,
  and `anchor.generic-train-release-lock.v2` / SHA
  `119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa`.
- The release lock binds the final projector, source-disjoint, generic,
  consumer, and execution-lock manifests plus the attestation, coordinator
  config, source-bank, fixed three projected files, outer-sidecar provenance,
  calibration-not-heldout claim, and the complete Hierarchical Task-KV
  identity/lineage/cache boundary. Eighteen freeze-layer tests pass; the
  current focused projector/freeze regression is 55 tests.
- No real formal-v3 snapshot or final projector manifest exists, so no real
  generic/source/release artifact or final release-manifest SHA has been
  published. The synthetic fixture SHA must never be used as that final SHA.
- This round is pipeline construction and small validation only. Do not launch
  a foundation model, high-memory training, or full-bank projection/training.

## Required live resume sequence

1. Publish or otherwise freeze the exact current working-tree identity before
   spending on a new run; never use the archived checkpoint identity.
2. Start the loopback dashboard and enter the provider key through the
   process-local credential path. Do not put it in a file, YAML, command line,
   log, or Git.
3. Launch exactly one fresh task at concurrency 1 and require all five stage
   counts to reach one, one completed task, zero failures, trusted
   identity/telemetry/process bindings, a real Builder tool trace and diff, a
   passing fresh-container validator receipt, and complete cleanup/HMAC data.
4. Freeze the first valid Gold record and its manifest before scaling.
5. Ramp cumulative work only after clean checkpoints: c8, c16, c24, c30, then
   the full checkpointed bank. Stop at the first new fixed error code.
6. Feed only completed canonical Gold through the separate versioned TaskBoard
   projector. Keep projector outputs and training manifests outside the Gold
   namespace.

Do not create a tag or release package. Do not store a credential in this file,
YAML, command arguments, logs, or Git.
