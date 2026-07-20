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
  `600e4d77c71813c40c37d98696ea9533a6a6ef721645cc71ff1ddaf229f27294`
- Execution lock SHA-256:
  `e40cd5b7daca350b813d8f30a587a5821f81f242a321779dba8bbacd1db0218b`
- Content-free execution attestation SHA-256:
  `1ecdbc7f45e3bb930ae1b35aa860f360c1d197e4c699617c87ee7358b52e23bf`
- All 15 lock-bound files match the lock, and the five-stage YAML binds the
  exact lock SHA above.

## Verification completed

- 212 coordinator, execution-v3, trace, runner, policy, route-diagnostics,
  train-sandbox, and attestation tests passed.
- 89 dashboard, control-plane, and PowerShell 5.1 launcher tests passed.
- 17 full-bank publication tests passed.
- Ruff and `git diff --check` passed.
- The full `anchor.ps1 -Action distill-swebench` offline preflight exited 0
  with no provider request.

## Publication caveat

Remote commit `6befb99` is not by itself a clean-checkout reproduction of this
ready state. The current working tree contains execution-bound changes and the
previously untracked `route_diagnostics.py` import required by the coordinator.
Before calling the stack published or reproducible on another machine, make a
targeted follow-up commit that includes the final lock/config, coordinator and
adapter, policy/models/runner/trace, route diagnostics, train-sandbox validator
and Containerfile, behavioral/wire/artifact probes, their focused tests, the
LF rule for `*.Containerfile`, and the full-bank attribution/manifest hash fix.
Do not mix unrelated historical training artifacts into that commit.

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
  `anchor.swebench-source-disjoint-manifest.v1` / SHA
  `4e7622d7a8ee07678963a8712ec50c2061223a91492b0b2cf004e5ae3caeeb72`,
  and `anchor.generic-train-release-lock.v1` / SHA
  `889787be1391aec2d59f91b1ba171588c82e455aaddc342b79d3680e0284210d`.
- The release lock binds the final projector, source-disjoint, generic,
  consumer, and execution-lock manifests plus the attestation, coordinator
  config, source-bank, fixed three projected files, outer-sidecar provenance,
  and the calibration-not-heldout claim. Nine freeze-layer tests pass; the
  combined projector/release/formal regression is 71 tests.
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
