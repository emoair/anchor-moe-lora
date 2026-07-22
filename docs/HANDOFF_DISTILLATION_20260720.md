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

## Long-context token inventory

- The post-projector producer emits the body-free record contract
  `anchor.long-context-token-inventory.v1` and manifest contract
  `anchor.long-context-token-inventory-manifest.v1`. It does not change the
  frozen projector v2, sidecar v2, or segment-plan v1 bytes.
- Policy/schema SHA-256 values are: config
  `79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe`,
  record schema
  `aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f`,
  and manifest schema
  `8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0`.
- The final synthetic integration inventory manifest is
  `73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e`.
  Its three partitions contain 5 records each and bind 15 records, 2 task
  bundles, 89 segment references, and 25 unique segment IDs. The source
  task-ID digest is identical to the authenticated projector manifest. The
  train-clean, train-noisy, and calibration-clean partition SHA-256 values are
  respectively
  `d58471790406130cfbbde0b473a296665227f920f4f338455302fde462167846`,
  `dfc3e5423ca4368a3974d9cdfc312af540b75c44aa41ff5c43ff940343c60bc1`,
  and `6fcc71a051cab56ab253ffe8cf23983c5e13f3515f8651a6eeb00bc27f7712e5`.
- The producer first proves bundle/split/five-role invariants, then sends only
  the ordered causal segment whitelist to the exact local counter. Current,
  future, and forbidden blocks are never selected, tokenized, serialized, or
  persisted. The inventory contains no body, preview, rendered input, target,
  token-ID array, or hidden-evaluation material.
- The checked-in fixture uses an explicitly synthetic exact tokenizer binding.
  It verifies authentication and deterministic accounting only and is not
  Gemma-compatible evidence. A real inventory requires offline local tokenizer
  assets plus tokenizer ID/revision/runtime, chat-template, serialization, and
  special-token-policy hashes. Unknown identity fails closed; cache-reuse
  savings remain zero.
- The fixed buckets are 8K, 16K, 32K, 64K, and 128K measurement candidates;
  256K is capability-only; 1 Mi is research-only and blocked; values above
  1 Mi are rejected. All buckets remain unevaluated and keep quality,
  allocation, execution, distillation, and training authorization false.
- Fresh-pycache projector/release/inventory validation collects 84 tests and
  passes 83 with one platform-permission skip for directory-symlink creation on
  Windows. Ruff, formatting, `py_compile`, Draft 2020-12 instance validation,
  physical hash/count checks, and body/absolute-path scans pass.

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

## Natural-language scaffold producer freeze (2026-07-22)

The original `0a47b8a8fcf460a3da50ccbf4d77436f7bd78a25` freeze is
superseded. Its config
`9c092c1f39e56b5e745515a0124715dd025aed5f78597edc2a1dfc894f0fdd40`,
record schema
`3892af891f3f3c0cf9462dd28713ece6c74bcaefffea20ad4d614a82898f85dd`,
manifest schema
`58acca23cf5f76187aadf40b95f6fbeb4cd667299e6c1183656bf296c21ee302`,
and fixture manifest
`ca95110a0062da50b1b01d273e478f5b781b7cb4ed820c324f97cffd1c1e5a9a`
must not be accepted.
That freeze's record schema omitted the authenticated TaskBoard v2 stale-noise
overlay ID form. The replacement below explicitly enumerates canonical
`tb-block-v1:<64-lowercase-hex>` and stale-overlay
`tb-stale-v1:<64-lowercase-hex>` identifiers and was revalidated against every
published JSONL instance with a real Draft 2020-12 validator. Consumers remain
fail-closed; no compatibility exception is permitted.

- The provider-free producer is implemented in
  `src/anchor_mvp/swebench/natural_language_scaffold.py`; its core physical
  SHA-256 is
  `09e7dae7f0fcafabbf2ea682504355d2c95c545764295c84eace7d16b3332330`.
  The build and audit entry points are
  `scripts/data/build_swebench_natural_language_scaffold.py` and
  `scripts/data/audit_swebench_natural_language_scaffold_fixture.py`.
- The immutable producer contracts are config
  `e81fc742ffb99d0f71ff3cc03ba68e82644ed7f539eb190eb2945bec7567fe38`,
  record schema
  `84efd818a52334e6b63a2132126d4a133ea3a143e13d11431bda3a242ba67d14`,
  manifest schema
  `8034b673798b0dc8b8a620b53a4a92e5565b5f9d936ad76ef8d30add50a98b16`,
  smoke schema
  `3944b28736ad1b6df9088ec69753c471d52ddb4f2753a974a23a29343c2cba5b`,
  and smoke contract
  `46bca04c358cc1e80f55c7eacff36fdf3f11a83efda52ad4386035ca5d614719`.
- The checked-in synthetic fixture manifest is
  `25e40da8fea46ba018ae0031fa8c37da38b59438bb92d9052a915fd256d822dc`.
  It contains 20 records: two source-disjoint task bundles, five roles per
  bundle, and paired `json_only` / `concise_rationale_plus_json` views. The
  four partition SHA-256 values are train JSON-only
  `6aad30ccc1aaaac432a76559e15879acac4f82f382c4020a8e7c1831b2ef2751`,
  train rationale+JSON
  `02e85be2477fd8937fe3dc3f6222c771d7725864232e11ae8338fcc061d02617`,
  calibration JSON-only
  `2e30596b078a47416aec51c80aaaa5fa946d7f0cbd2231f49974ba4e6f7ef065`,
  and calibration rationale+JSON
  `260f1b7d14f1c1563b7a235bc475992f628f70f1e5e67351d05b76aba2299168`.
- The architecture contract is
  `frozen_prefix_q_reader__prefix_branch_producer_consumer`. Expert adapters
  are off during the shared frozen prefix and may be expert-only after an
  explicit route boundary. aLoRA is optional and strictly
  `next_request_input_activation_only`: planner output is validated and
  committed, the frozen base with adapters off re-encodes the short committed
  scaffold into a new downstream immutable lineage, and only the next expert
  request may scan its input invocation. Same-request generated-trigger
  switching and Planner-private KV handoff are rejected.
- The fixture is body-filtered and split-before-augmentation. Current target,
  future, forbidden, held-out, and whole-TaskBoard bodies are not serialized.
  Tokenizer identity is unbound, so no token ID, position, offset, or boundary
  index is emitted. Q-only, Q+O, and wide-LoRA are labels for future controlled
  comparisons, not outcome claims.
- Focused scaffold tests pass 20/20. The combined projector, long-context,
  release-freeze, and scaffold set collects 104 tests and passes 103 with one
  Windows symlink-capability skip. Ruff, formatting, `py_compile`, both CLI
  helps, deterministic rebuild, physical manifest/sidecar audit, real Draft
  2020-12 validation of all 20 published records plus manifest and smoke,
  UTF-8/LF, body exclusion, and zero-request/resource assertions pass.
- The local Qwen GGUF smoke contract remains unexecuted and non-authorizing.
  It is not trainable weights and is not proof of physical Q-reader zero-copy
  or shared KV. A real tokenizer binding, aLoRA adapter, frozen formal-v3
  release, provider distillation, model training, physical KV backend, and
  correctness/performance/quality evaluation remain incomplete and fail
  closed.

The detailed construction and migration rules are in
`docs/swebench_natural_language_scaffold.md` and its Chinese counterpart.

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
