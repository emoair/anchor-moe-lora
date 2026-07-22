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

## Qwen diagnostic training prerequisite status (2026-07-22)

- The content-free, fail-closed status contract is
  `anchor.qwen-train-prerequisite-status.v1`, schema SHA-256
  `e8d09abc26effcedc642125b4d84185f0e5072a23f5611f068274bd963c4f577`.
  The published status manifest SHA-256 is
  `70c8f0a866c5fb41c4c3726638b55a66efab77f8b2ee31c27ad31ab55def67da`;
  its mandatory sidecar binds the same raw bytes. The manifest binds baseline
  producer commit `03ea0214567289e4f46378d4731b0177c18a1402`; this is deliberately a
  baseline commit, not a circular claim that a file contains the hash of the
  commit which first publishes that file.
- No usable formal-v3 frozen snapshot, final projector, frozen generic
  execution contract, source-disjoint manifest, or release lock exists at the
  schema-locked canonical paths. The repository schema-version scan also found
  zero matching frozen formal artifacts. Their contract schema SHA-256 values
  remain respectively the snapshot implementation
  `eba4263854301df75074609e9504d37cf6a7cf6d4204b05b6e3ebf26ee7476ba`,
  projector manifest
  `2cd9dc98d2b2865ed0586abfe291e3f6d161686597fcd2a7884c5762d2195347`,
  generic execution
  `63c699fdb7932b9fe1593b044d6a588bb4234b349816ce70f65568ab7b0f0b3a`,
  source-disjoint
  `2a2aae532c25b324a96b929a6a396d55d051c765258a5da0ebb7547724c68f6b`,
  and release lock
  `119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa`.
  The 15-record sibling TaskBoard fixture is explicitly `research_proxy_only`
  and is not a substitute.
- The current physical, unfrozen Gold observation binds partition manifest
  `4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7`,
  automation status
  `f51d4e813ec009ba865b88ff7d45be98a768ae5b2040be8d01b82f10f4b1d605`,
  and partial-export manifest
  `1b8e5b87957d7ec1e867813c95b8f7ab3bef55861e778b6ba9f197e6edf3f2ec`.
  The canonical counts digest is
  `b931dbeff6646fc2d1c210cfb98de1660cad1b18f9a0fbea064839ba9d9e3814`:
  Planner 384, Tool Policy 384, Frontend Gen 346, Frontend Review 203, and
  Security Gate 148, for 1465 records. The formal floor is 256 per expert;
  Review is short by 53, Security by 108, and only 85 of 256 strict complete
  chains exist. These are raw metadata observations, not frozen train,
  calibration, or held-out partition counts or hashes; all three formal
  partitions remain unavailable and calibration is not held-out.
- Producer owns the offline authenticated tokenizer/chat-template/trigger-text
  binding; Consumer owns exact artifact and release-lock verification; Runtime
  alone may materialize request-2 token coordinates after exact serialization.
  The binding schema is
  `anchor.scaffold-tokenizer-binding-manifest.v1`, SHA-256
  `5b2e7c2e8e6efc1c9b7251fde853631e65806aca0364d9bb092ee9a07d135b25`.
  Global artifacts never contain request-specific token IDs or indices.
  Before any `bound` artifact can be consumed, a real loader must recompute and
  enforce the cross-field equalities that JSON Schema alone cannot express:
  requested and nested tokenizer revisions, chat-template source SHA versus the
  authenticated asset entry, scaffold manifest SHA versus trigger source,
  partition inventory, and the nested tokenizer-binding digest. Each equality
  requires a negative drift test; the current candidate status is not that
  loader and cannot be promoted on schema validation alone.
- A real local ModelScope clone is recorded only as a candidate source:
  `Qwen/Qwen2.5-1.5B-Instruct` revision
  `3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34`. Producer rehashed the four
  tokenizer-only assets (combined inventory SHA-256
  `872b55391b4daff195a1caf9a05bb3305d7c84b759394ad5cd0d7ebc0b8192a8`)
  and the exact chat-template UTF-8 value (SHA-256
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`).
  The model weight identity
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
  is authenticated as the Git LFS pointer OID and size 3,087,467,144 at that
  revision; Producer intentionally did not reread and rehash the 3 GB physical
  file. Model weights are excluded from tokenizer inventory. A bound manifest
  still requires the source-scaffold partition inventory, tokenizer runtime
  descriptor, serialization and special-token policies, five-role trigger
  binding, and request-coordinate policy.
- The diagnostic-only toy attestation schema is
  `anchor.qwen-toy-source-disjoint-attestation.v1`, SHA-256
  `7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea`.
  A `ready` attestation must authenticate its generator, grammar, attester
  implementation/config, declared semantic generation-input set, protected
  source-ID inventories,
  namespace policy, and deterministic intersection proof. It proves only
  provenance and identifier disjointness, never semantic uniqueness. No toy
  attestation artifact exists yet, so the current status is
  `unverified_pending_authenticated_attestation`; toy data is diagnostic-only
  and cannot enter a formal release.
- Consumer must freeze the status schema/manifest, tokenizer binding
  schema/manifest, toy attestation schema/artifact, and external release-lock
  schema/artifact identities. A future release v3 (or separate compatibility
  lock) must also bind the formal scaffold and token inventory, trainable base
  snapshot, tokenizer/base compatibility attestation, formal snapshot,
  projector, source-disjoint, and generic execution identities. Do not mutate
  release-lock v2 silently, include diagnostic toy data in a formal lock, or
  place a release lock's own SHA inside its body.
- Validation is metadata-only and low-memory. It performs no provider or
  network request, model load, GPU request, full-bank projection, Gold write,
  held-out read/write, training, tag, or release. Reproduce the focused checks
  with:

  ```powershell
  python -m pytest -q tests/test_qwen_train_prerequisite_status.py
  python -m ruff check tests/test_qwen_train_prerequisite_status.py
  ```

  The tokenizer binding artifact, authenticated toy attestation, formal-v3
  five-artifact chain, complete Gold coverage, trainable base snapshot, true
  training, and correctness/quality/performance evaluation remain incomplete
  and fail closed.

## Consumer synthetic scaffold diagnostic checkpoint (2026-07-23; non-authorizing)

- The sibling Consumer/training repository published commit
  `9539fb56c236f08ffe9d7a8f56dfc28f14e1907c` on branch
  `research/neural-swarm-kv`. At handoff, local, upstream, and remote HEAD were
  equal; the commit contains exactly 22 files, its worktree was clean, and no
  tag or release points at HEAD. Its parent is
  `c2b9174930270352c7d25490afd57d02db790e0d`; its precise Producer cherry-pick
  `77b48d429608ec0c1ff3e478567598fa202beb12` has the same stable patch and 12
  artifact blobs as Producer release
  `2648129d599a5041100278cb04b12291ffd8a482`. This section is a downstream
  checkpoint record. It does not alter any frozen Producer v1/v2 artifact
  identity.
- The committed fixture is a closed-grammar synthetic diagnostic proxy with
  10 source bundles, five roles, and two paired scaffold variants, for 100
  records total. Splitting precedes role/variant expansion: eight bundles and
  80 records are `train`; two bundles and 20 records are `eval_proxy`. English
  and Simplified Chinese each contribute five bundles. `eval_proxy` is
  explicitly not held-out. The fixture manifest SHA-256 is
  `64b1ce813477deef48de16dbdc0d2561bbeaa0ef5d6248862e9f2bedc8acc0dd`;
  record-schema SHA-256 is
  `0137d73392f5ea471c166219eaca66b097e66178132ac8688569ae88b1d720fa`;
  manifest-schema SHA-256 is
  `0ece3e5ddf80b756b0a9613076f45f1df06bb2a487e6fc7537516a0f5285db12`.
  The four authenticated partitions are:

  | Split / variant | Records | SHA-256 |
  |---|---:|---|
  | `train/json_only` | 40 | `3ea3cf57e9990b2b07e98cd2bf27a620ed3d2eb2bfd495b0f89ae3d22cce60df` |
  | `train/concise_rationale_plus_json` | 40 | `ff656ff6d2b5303880e5a5ec8db05ffa33e7eca7ca3b1bbd78787a1bd28f1852` |
  | `eval_proxy/json_only` | 10 | `06fe504e33ef61b05df7f3aff5969025b299343d613441bbb04dea1599fbc680` |
  | `eval_proxy/concise_rationale_plus_json` | 10 | `ed0d60a609edbac6ba9db7cf8d818f85354da4f17e7409b0981d8d4abaf7cfaa` |

- The synthetic Producer config, config schema, closed grammar, grammar schema,
  and implementation SHA-256 values are respectively
  `b34d54fea39340ab2e1567f380241fc410276ab17035d69a2285c7c7c83cebaf`,
  `938313a1265187e1d030fdc9d808a8d669c64ad8e51fda26e93ad725e5451d58`,
  `e895defeb8f3a077b30fc7c9188274c208c54996c8bfc16c6b4594f92797cd90`,
  `5f2322945d8c7e0ca5fadd12086ac192dede4e27234ecaa2617057fc49fc2082`,
  and `4c2b7b70dffc215b3aa3b370db2e183d616084d530073a269a5964baa3b81525`.
  Its namespace is `anchor.synthetic-nl-scaffold-diagnostic.v1`. Generation
  reports zero provider/network/model/GPU/protected-body reads, but it consumes
  no authenticated protected-source inventory. Therefore formal source
  disjointness and zero intersection remain unproven. This new namespace does
  not close or replace the missing body-free `synthetic_scaffold` protected
  inventory from the six-source prerequisite contract.
- The separate training runner implementation SHA-256 is
  `8b5cf325eb6876493eb1ff2e617a716cb9e2f5d8644749826c275e2e216d8569`;
  its config SHA-256 is
  `40d97f2ac733924560ff333cf91066cde9d0a16c58f4961f9492bbf05398c33e`.
  The only exercised arm is `q_only`: rank 4, 344,064 trainable parameters in
  56 tensors, BF16 compute with TF32 enabled, sequence length 512, and
  non-reentrant gradient checkpointing. `q_plus_o` and `wide_lora` remain
  control labels, not measured results.
- Tokenizer-only preflights bind all 100 ordered record-local prompt, full-input,
  and masked-label digests without publishing raw token IDs. They bind
  Transformers 5.13.0, Tokenizers 0.22.2, maximum full length 394 of 512, and
  zero truncation. The step-2 and step-20 preflight SHA-256 values are
  `08d283a5521b79ff036d54cf82fa5590b36a280a50db742deb88f442be95c76f`
  and `b7679f6f75756c25ff0154b89fd41021411035f2b095f567ffae58a95b16e00d`.
  These preflights and the final adapters/receipts live under ignored local
  diagnostic paths; their hashes are local evidence, not published formal
  release artifacts.
- Two explicitly requested local Qwen2.5-1.5B-Instruct diagnostic executions
  passed mechanical integrity gates. The 2-step receipt SHA-256 is
  `b8a8dd026e4543f5c0e2c7d9e99815c699734970b20110d7ec234bd38553d0f1`:
  eval-proxy loss moved from 3.1020863533 to 3.1025008678, delta
  +0.0004145145; maximum adapter effect was 0.47265625. The 20-step receipt
  SHA-256 is
  `d0a31ef6d471a60213f6c306cf7fb3de0c359b3b889572c5620e0090fa1dee54`:
  eval-proxy loss moved from 3.1020863533 to 3.0703784108, delta
  -0.0317079425, approximately a 1.02% proxy decrease; maximum adapter effect
  was 0.6875. Both receipts report 56/56 finite nonzero LoRA gradients, zero
  reload-logit delta, and unchanged base hashes. Each real run intentionally
  records one GPU request and two model loads, while provider/network requests
  and protected-body/held-out reads remain zero. Neither loss trace is a
  formal quality, held-out, numerical-equivalence, shared-KV, or architecture
  result.
- Consumer reported 146 focused tests collected: 140 passed and six Windows
  symlink-capability skips, with zero failures. Dataset Producer tests passed
  28 with one capability skip; runner tests passed 26. Ruff, formatting,
  `py_compile`, audit, deterministic rebuild, mandatory-sidecar/TOCTOU,
  secret, size, and scoped-Git checks passed. Code plus the synthetic fixture
  are committed; adapters, preflights, and receipts remain local and ignored.
  The Producer-side independent rerun of the two new focused files collected
  55 tests and passed 54 with the one Windows capability skip; its metadata
  audit also passed with zero provider/network/model/GPU/protected-body counts.
- Reproduce the committed metadata-only audit and focused tests from the exact
  Consumer commit without starting a model or provider request:

  ```powershell
  $env:PYTHONPATH = "src"
  python scripts/research/build_synthetic_nl_scaffold_diagnostic_v1.py audit `
    --repo-root . `
    --config configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml `
    --artifact fixtures/research/synthetic_nl_scaffold_diagnostic_v1
  python -m pytest -q `
    tests/test_synthetic_nl_scaffold_diagnostic_v1.py `
    tests/test_qwen_synthetic_scaffold_diagnostic.py
  ```

  The Producer-side handoff audit copied no sample body and did not read Gold,
  held-out, or the earlier protected scaffold bodies. The 100-record fixture
  remains proxy-only; both `training_authorized` and
  `formal_training_authorized` remain false. Formal-v3 snapshot/projector/
  generic/source-disjoint/release-lock artifacts, four missing body-free
  inventories, six-source zero-intersection/v1 attestation, a large teacher
  dataset, full training, physical KV/multistream/Q-hijack measurements, and
  formal quality/performance evaluation remain incomplete and fail closed.

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
