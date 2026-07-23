# Frozen-prefix Q-reader distillation pipeline v2

Status: additive research profile; metadata materialization may be ready, but
execution, training, formal training, and release authorization remain blocked.

## 1. Goal and non-goals

This version reuses the authenticated SWE-bench execution base instead of
building a second orchestration stack. The reused base is the existing
five-stage scheduler, CC Switch routing, OpenCode sandbox, real tool trace
capture, validators, HMAC receipts, checkpoint/resume machinery, canonical
Gold exporter, TaskBoard projector, and generic release contracts. V2 adds a
versioned post-Gold profile for a **frozen-prefix Q-reader / prefix-branch
producer-consumer** experiment.

The producer boundary is deliberately narrow:

- preserve canonical five-stage Gold byte-for-byte;
- derive role-specific, causally filtered views only after authenticated Gold;
- use the five roles `planner`, `tool_policy`, `frontend_gen`,
  `frontend_review`, and `security_gate`;
- represent shared-prefix and expert-private-tail semantics as authenticated
  metadata, not as physical KV tensors;
- make `q_only` the only primary training label while keeping `o_only` and
  `q_plus_o` as non-authorizing diagnostic overlays.

This work does **not** claim hidden chain-of-thought, lossless inheritance of
reasoning, zero KV, O(1) attention, disappearance of base-model compute,
mid-request aLoRA switching from a generated sentinel, ordinary in-stack
Q-LoRA exact full-stack KV reuse, a physical cross-attention Q-reader,
token-level MoE, formal quality, or formal training readiness.

## 2. Two-version boundary and reuse

`task-level-moe-lora-v1` is the compatibility profile for the earlier
task-level MoE-LoRA branch. `frozen-prefix-qreader-v2` is additive. V2 does not
edit the v1 profile, v1 generic release schema, v1 scaffold, canonical Gold, or
held-out assets.

Both versions share the same execution core:

1. source-bank staging and five-stage work-order scheduling;
2. CC Switch provider/model routing;
3. controlled OpenCode sandbox execution and real tool calls/results;
4. stage-specific semantic validation;
5. HMAC execution receipts and checkpoint/resume;
6. canonical Gold export;
7. deterministic TaskBoard projection and release gates.

Only the authenticated post-Gold profile, bundle capability metadata, training
view shape, and downstream release overlay differ. This keeps expensive
provider-backed logic shared while preventing one version's output contract
from silently changing the other.

## 3. End-to-end data flow

The intended flow is:

```text
canonical Gold / TaskBoard
  -> visibility filter before serialization
  -> route directive
  -> one concise_rationale_plus_json primary scaffold
  -> validate and explicitly commit text
  -> frozen-base re-encode of the short committed scaffold
  -> next-request expert invocation
  -> append-only expert-private KV tail
```

The v2 large-batch materializer strictly emits one primary
`concise_rationale_plus_json` view per role and bundle, with `pair_count=0`.
The older frozen scaffold v1 keeps its paired `json_only` /
`concise_rationale_plus_json` ablation assets unchanged; that pairing is not
imported into v2. V2 adapter diagnostics are execution overlays over the same
record inventory, not a second data view and not a duplicated-row count.

Splitting happens on `task_bundle_sha256` before any view, role, noise, or
causal augmentation. All five roles of a bundle stay in one split.
`eval_proxy` is never called held-out.

Current, future, and forbidden block bodies are excluded before prompt
serialization, tokenization, shared-prefix construction, logs, or receipts.
They may not be inserted and hidden later with a mask. The producer never
stringifies an entire TaskBoard.

## 4. Why two requests and an explicit commit

An aLoRA-compatible runtime scans invocation tokens in the **input** of a new
request. A sentinel generated midway through request 1 does not cause a safe
mid-request adapter switch. Therefore:

1. Planner/base request 1 produces the concise rationale, route JSON, and
   optional sentinel.
2. A verifier validates the schema and commits the visible text.
3. The frozen base producer re-encodes the short committed scaffold as an
   immutable downstream segment.
4. Expert request 2 receives the scaffold as input and activates the expert
   adapter only after the invocation boundary.

Planner-private KV cannot be handed directly to an Expert. It was produced
under a different adapter/hidden-state lineage. Only committed text crosses
that boundary, and the frozen base re-encodes it. After activation, prompt and
generated tokens append to the selected expert's private tail. A private tail
never migrates between experts.

Even Q-only adaptation changes an attention layer's output and therefore later
hidden states and K/V. Q-only is a useful control, not proof that all later
layers' K/V are exactly reusable. Exact reuse is restricted to an identical
ordered prefix lineage with identical token order, positions, RoPE, tokenizer,
model architecture, and KV-producing weights.

## 5. Machine contracts and field meanings

The v2 profile freeze authenticates the shared execution core and the v2-only
profile boundary. The training-view materializer emits:

- `anchor.frozen-prefix-qreader-training-view.v2` records;
- `anchor.frozen-prefix-qreader-training-view-manifest.v2`;
- `train.jsonl` and `eval_proxy.jsonl` under a strict mandatory sidecar.

The body-free bundle producer emits:

- `anchor.frozen-prefix-qreader-bundle-profile.v2` records;
- `anchor.frozen-prefix-qreader-bundle-profile-manifest.v2`;
- one `bundle_profiles.jsonl` metadata inventory.

A record binds bundle, source, split, language, information-flow stratum,
role, capability labels, ordered segment and prefix lineage identities, and
the route-boundary architecture contract. Route JSON uses canonical UTF-8
JSON with deterministic key ordering. Every record carries
`training_view.routing_json_sha256=SHA256(canonical_route_json_bytes)` and the
same digest participates in the deterministic record-identity preimage; the
materializer recomputes both relations before publication. No task/source
namespace or language is accepted as a salt to evade semantic overlap.

The release overlay
`anchor.frozen-prefix-qreader-release-overlay.v1` is a conjunction of:

1. an authenticated old `anchor.generic-train-release-lock.v2`;
2. the authenticated v2 profile freeze manifest;
3. the authenticated v2 training-view manifest and mandatory sidecar;
4. the authenticated bundle-profile manifest and mandatory sidecar;
5. an authenticated consumer diagnostic reference and mandatory sidecar.

The CLI supplies runtime paths and expected manifest SHA-256 values. The
checked-in config locks schema versions and producer implementation identities,
not runtime manifest hashes. This avoids a hash cycle: the v2 profile may bind
the overlay schema/module/CLI, while it must not bind the overlay config. The
overlay config depends on those code identities; runtime manifests depend on
their own producers; all five manifests point only into the final overlay. A
runtime DAG assertion rejects cycles.

The freeze CLI is a dedicated new-process entrypoint with a standard-library
bootstrap. It ordinary-imports no `anchor_mvp` package. It first authenticates
the exact executing CLI path/bytes, config snapshot, profile manifest plus
strict sidecar, and implementation path/bytes against both config and profile
bindings. It then compiles and executes that single authenticated
implementation snapshot in a digest-qualified private module. Package
`__init__`, cached bytecode, a second implementation-file read, a modified
sibling CLI, and preloaded package state cannot select the code that runs.
The overlay still terminally rechecks the physical identities. Embedded use
must call an already-authenticated library API deliberately; it is not treated
as the CLI trust boundary.

The overlay does not trust producer hashes merely because each manifest is
internally self-consistent. It rebuilds a role-indexed map from the profile
freeze dependencies, authenticates the physical projector config,
implementation, CLI and schemas plus the materializer config,
implementation, schemas, builder and auditor, and terminally rechecks those
snapshots. Training-view and bundle-profile producer fields must then match
that map and the physically loaded manifest schemas. Projector manifest,
sidecar/record, and segment-plan schema identities must agree across the
generic lock, profile, training view, and hierarchical-KV contract.

For the generic lock, profile freeze, training view, and bundle profile, the
CLI path must equal the checked-in project-relative canonical runtime
directory; a byte-identical copy in a temporary directory is rejected. Output
bindings call these `logical_manifest_path` / `logical_sidecar_path` and mark
their location as `project_canonical_runtime_dir`. The consumer reference is
the sole external input: its absolute machine path is deliberately not
published, and the binding records the repository-logical path with
`source_location_kind=external_consumer_reference`.

The consumer reference is checked against a strict **minimum compatibility
subschema** embedded in the overlay schema, plus its exact frozen manifest and
sidecar hashes. That subschema authenticates only the fields the producer needs
for conjunction (counts, semantic identity, Q-only label, false gates, and
zero-request audit). It is not presented as, and does not replace, the
consumer repository's complete publication schema.

The overlay status is always
`profile_materialized_execution_blocked`. A base lock that reports `ready`
cannot promote `training_authorized`, `formal_training_authorized`, or
`release_authorized`; all remain `false`.

## 6. Producer, consumer, and runtime responsibilities

The producer authenticates source metadata, filters causal visibility, assigns
bundles before expansion, materializes deterministic views, publishes
create-once manifests/sidecars, and never embeds runtime-specific llama.cpp
state into a producer sidecar.

The consumer authenticates the exact bytes, schema, hashes, split/bundle/role
cross-bindings, and forbidden-content exclusions before deriving a training
view. It treats the producer record as opaque until authentication succeeds.
Diagnostic metadata never authorizes formal training.

The runtime owns request-local serialization, tokenizer/chat-template identity,
the exact request-2 tokenization, trigger covering span and boundary overhang,
adapter activation, private-tail allocation, physical KV behavior, GPU locking,
and execution receipts. Optional aLoRA capability means
`next_request_input_activation_only`; it is not a cross-attention Q-reader and
does not prove physical shared KV.

No global trigger token index or isolated trigger token IDs are authoritative.
If a token boundary is needed, the full request 2 is serialized once and the
zero-based, end-exclusive covering span is recorded in a request-local receipt
bound to the exact serialization SHA.

## 7. Diagnostic reference and Gemma boundary

The external 1000-record reference is exactly:

- manifest SHA-256
  `a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed`;
- sidecar physical SHA-256
  `7f238be47cc60af808421bbbdaefb6bbc5d5c0f617d976b66d5b2a87d767b0a0`;
- 200 unique task semantics x 5 roles x 1 primary view;
- 800 train / 200 eval-proxy records;
- 100 English / 100 zh-CN semantics and zero translation pairs;
- `q_only` as the only primary label.

It is a reference only. It neither becomes canonical Gold nor contributes
formal authority.

The model-free Gemma tokenizer observation uses sequence length 768 with
strict no truncation. Full-view maxima are 504 / 523 / 546 / 605 / 665 for
Planner / Tool Policy / Frontend Generator / Frontend Reviewer / Security
Gate; 514 of 1000 records exceed 512, and all fit below 768. The old Qwen
declarative length 512 does not apply to the Gemma run. Until the exact Gemma
tokenizer + official chat-template identity and runner binding are frozen,
execution remains blocked.

## 8. Fail-closed matrix

| Condition | Result |
| --- | --- |
| Missing manifest or mandatory sidecar | refuse |
| Manifest SHA differs from CLI expectation | refuse |
| Sidecar is not exactly `<sha>  manifest.json\n` | refuse |
| Schema path/hash/version drift | refuse |
| Duplicate JSON key, non-UTF-8, or non-finite number | refuse |
| Any metadata/config/schema/implementation input is 50 MB or larger | refuse |
| Symlink/reparse point in an authenticated path | refuse |
| A producer artifact is loaded from a non-canonical runtime directory | refuse |
| Input changes between snapshot and terminal recheck | refuse |
| Output exists, overlaps input, or publish is not create-once | refuse |
| Generic v2 lock is not self-ready research proxy | refuse |
| Projector/bundle/view producer, strict projector sidecar, or transitive schema cross-binding differs | refuse |
| Profile-bound projector/materializer/builder/auditor physical bytes drift | refuse |
| Consumer reference count, split, language, or Q-only identity differs | refuse |
| Any input claims Gold/held-out body reads or training authority | refuse |
| Tokenizer/template/runner identity is unknown | keep blocked |
| Execution decision, lease, or data-byte TOCTOU lease is missing | keep blocked |

Failures use content-free error codes. Parse errors never echo an offending
line or body. The read set contains five manifests, five sidecars, checked-in
schemas/config/overlay code, and the 17 profile-bound projector/materializer
dependency files needed for physical identity cross-checking. JSONL
partitions, Gold, and held-out bodies are never opened by the release overlay.

## 9. Reproduction and resource stages

Validate code and schemas without reading data:

```powershell
python -m pytest -q tests/test_frozen_prefix_qreader_release.py
python -m ruff check src/anchor_mvp/swebench/frozen_prefix_qreader_release.py scripts/data/freeze_frozen_prefix_qreader_release.py tests/test_frozen_prefix_qreader_release.py
python -m py_compile src/anchor_mvp/swebench/frozen_prefix_qreader_release.py scripts/data/freeze_frozen_prefix_qreader_release.py
```

Freeze the v2 profile and materialize a tiny, offline fixture with their
dedicated CLIs first. Then freeze the release overlay:

```powershell
python scripts/data/freeze_frozen_prefix_qreader_release.py `
  --config-sha256 <checked-in-config-sha256> `
  --generic-release-dir artifacts/formal_v3/training_release/release_lock `
  --generic-release-manifest-sha256 <sha256> `
  --profile-freeze-dir artifacts/distillation-profiles/frozen-prefix-qreader-v2 `
  --profile-freeze-manifest-sha256 <sha256> `
  --training-view-dir artifacts/swebench/frozen-prefix-qreader-view-v2 `
  --training-view-manifest-sha256 <sha256> `
  --bundle-profile-dir artifacts/swebench/frozen-prefix-qreader-bundle-profile-v2 `
  --bundle-profile-manifest-sha256 <sha256> `
  --consumer-reference-dir <consumer-repo>/fixtures/research/synthetic_five_role_qonly_diagnostic_v1 `
  --consumer-reference-manifest-sha256 a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed `
  --output-dir artifacts/swebench/frozen-prefix-qreader-release-overlay-v1
```

This phase performs zero provider, network, model, and GPU requests. A later
small provider pilot requires an explicit live flag, credentials, cost cap,
one work order, checkpoint/resume test, and receipt verification. Scaling is
allowed only after pilot evidence, streaming/peak-memory measurement, a
formal-v3 snapshot/projector/source-disjoint/generic/release set, protected
inventory completion, tokenizer/runner binding, and a separately authenticated
execution decision and lease. The overlay itself can never grant that
authority.

## 10. Version and Git discipline

Every schema, config, implementation, fixture manifest, sidecar, and runtime
input is bound by physical-byte SHA-256. JSON output is canonical
UTF-8/sorted-key/compact/LF. The producer snapshots each file once, validates
from those exact bytes, reparses only those bytes, and repeats identity checks
at the terminal boundary.

V1 remains a compatibility branch. V2 is a separate branch and additive
schema namespace. Changes are committed from an explicit whitelist after
staged-diff, credential, personal-path, file-size, JSON/YAML/UTF-8/LF, Ruff,
py_compile, and focused-test checks. No tag or release is created. Runtime
hashes are supplied only after final producer freeze; temporary identities are
not propagated to the consumer.

Migration is opt-in: a consumer must explicitly support the v2 profile and
overlay. Absence, older versions, or unknown fields fail closed; no v1 loader
is relaxed.

The V2 branch baseline is
`524ca359eff128221ef4fa9f5a9e665abf64c7c3` (`task-level-moe-lora-v1`).
The scoped commit whitelist is exactly:

```text
configs/orchestration/frozen_prefix_qreader_profile.schema.json
configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json
configs/orchestration/profiles/frozen_prefix_qreader_v2.json
configs/research/frozen_prefix_qreader_release_overlay_v1.json
configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json
configs/research/swebench_natural_language_scaffold_v2.yaml
configs/research/swebench_natural_language_scaffold_v2_bundle_profile.schema.json
configs/research/swebench_natural_language_scaffold_v2_bundle_profile_descriptor.schema.json
configs/research/swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json
configs/research/swebench_natural_language_scaffold_v2_config.schema.json
configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json
configs/research/swebench_natural_language_scaffold_v2_record.schema.json
docs/frozen_prefix_qreader_distillation_v2.md
docs/frozen_prefix_qreader_distillation_v2.zh-CN.md
scripts/data/audit_swebench_natural_language_scaffold_v2.py
scripts/data/build_swebench_natural_language_scaffold_v2.py
scripts/data/freeze_frozen_prefix_qreader_release.py
scripts/data/run_frozen_prefix_qreader_profile.py
src/anchor_mvp/swebench/frozen_prefix_qreader_profile.py
src/anchor_mvp/swebench/frozen_prefix_qreader_release.py
src/anchor_mvp/swebench/natural_language_scaffold_v2.py
tests/test_frozen_prefix_qreader_profile.py
tests/test_frozen_prefix_qreader_release.py
tests/test_swebench_natural_language_scaffold_v2.py
```

Any path outside this list must remain unstaged. Final physical SHA-256 values
for every schema/config/implementation and generated manifest/sidecar are
reported only from the final commit bytes and final runtime freeze; the Git
commit, local HEAD, upstream HEAD, and live remote HEAD must match before the
work is called complete.

The final pre-commit machine-identity snapshot is:

| Layer | Artifact | SHA-256 |
| --- | --- | --- |
| profile | profile schema | `5900f144c5aa25d359400b727fd5c8c31281b3a99792b3c27d783b357a2eb85a` |
| profile | freeze-manifest schema | `1310dd1c74f2f2f7c86bcfa3628925102a9c0b6f398df306dec99b446c60cfc5` |
| profile | checked-in profile | `f39ebde344d41ac29cf50d224795450d5d5da10534e382591d088e0b97224994` |
| profile | implementation | `973eea883e1e412083e3be8bc538428630d286fb2c09758493b3e2acaf18a944` |
| profile | runner | `1b1e88447ecfb53d6e91155ebbc0820d3d9d4df32ff441d8b31e455d01db3ccc` |
| materializer | config | `6fda2ff6bb6a92f8764daa8f68dc8226ae0839d2d9613dc240d9f0b75c9baee5` |
| materializer | config schema | `11f6a8555178f851a75341057750fadb415be7357e070275e4603453e03ec9e1` |
| materializer | descriptor / bundle-record / bundle-manifest schemas | `4614b0924ced82c483f6ead94e754e70426f7b12d45936eef40b43f18b21265a` / `42af26d742db8c06104f3955dfbd19c552c705c9237bd69455b42d132cb1ac5a` / `fbe7a543e0e8f19b27436ca882af629e44252ab5e7a5d246443709cadc39609f` |
| materializer | training-record / training-manifest schemas | `ac5176b53072a75439fd4f29f9e96e16416b6f690bed776df9e5509ae88b98c3` / `dae37cd63462ac6945547c32ca617d7a51d41700432292f989fc901491a2eb2c` |
| materializer | implementation | `3a3ff1eceed67f489e2edfb1df46f18615adf94128fe1c3ae499e1917e6228a3` |
| materializer | builder / auditor | `99a88e90d2419fca1cc6c35445058105fee7357962985063aefa01be7462b4d7` / `bf3f48476e0b174353c1363fcf70ab6aa518a504c9454ec634e02601d6fda08a` |
| release overlay | config / schema | `a6367f25654e7a5d2ea1d27cb56c50d19350a2af6cf8431664515983302dc611` / `03fd592a8fa98aee08a7193089dab8934663474b37de6ecc7b9cd5cf11eb5b91` |
| release overlay | implementation / authenticated-bootstrap CLI | `35af412ad992ab6e19e53267d98a2188ee20bfe931d4884ce674bfb3764315c1` / `1c700cd6863bfb1862a1b60363dc5226914b9cdacf5e2be14f4bb61c3a980bdc` |
| runtime freeze | profile manifest / sidecar physical bytes | `97f3fe1e8aa89bac107844413cd8a5da41ea6df474cf431687096a4e7972e255` / `72470ae4716733147ed654a27bc214151fe65816a54f1a1a93bdabbb7fd9c2eb` |

The runtime freeze is an ignored local artifact, not a committed fixture.
These identities do not grant live, training, formal, or release authority.

## 11. Verified facts, proxy signals, and unfinished work

Verified contract facts are the existing reusable five-stage execution core,
metadata-only v2 schemas, strict causal filtering contract, create-once
authentication behavior, diagnostic-reference counts/hashes, and the
model-free Gemma token-length observation. Earlier Qwen/Gemma aLoRA and
Q-only/Q+O experiments are proxy or smoke signals only.

Still unfinished:

- current-identity live teacher pilot and large provider-backed distillation;
- measured streaming projector peak memory at full-bank scale;
- real formal-v3 frozen snapshot, final projector, source-disjoint contract,
  generic execution contract, and final release lock;
- four missing protected source-ID inventories and zero-intersection proof;
- final Gemma tokenizer/chat-template/runner binding;
- authenticated execution decision, single-GPU lease, and data-byte TOCTOU
  lease;
- physical KV backend/CUDA, multi-stream correctness, and Q-reader zero-copy;
- multi-seed independent-bundle training plus quality, safety, and performance
  evaluation.

Until those are complete, metadata readiness must not be promoted to training
or formal readiness.
