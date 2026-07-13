# Anchor-MoE-LoRA data distillation pipeline

Teacher provider presets, model discovery/manual selection, secret handling, quota
capabilities, and legacy-config migration are documented in
[`teacher_providers.md`](teacher_providers.md).

This subsystem builds five SOP-injected corpora (`planner`, `tool_policy`,
`frontend_gen`, `frontend_review`, and `security_gate`) through a configurable
teacher endpoint. It is asynchronous,
append-only, deduplicated, resumable, and runnable offline with a deterministic mock.

It deliberately does **not** collect hidden chain-of-thought. `decision_trace` contains
only short public work products: the check performed, evidence observable in the input,
and the resulting action. Teacher responses containing `thinking`, `cot`, `reasoning`,
or `chain_of_thought` top-level fields fail validation.

## Canonical JSONL

Every `data_*.jsonl` row has the integration fields expected by the trainer:

```json
{
  "schema_version": "1.0",
  "id": "record_...",
  "expert": "frontend_gen",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "input": {"requirement": "..."},
  "provenance": {
    "seed_id": "seed_...",
    "sop": {"sop_id": "...", "sha256": "...", "source": "...", "task_type": "..."},
    "teacher": {
      "model": "kimi-for-coding",
      "base_url": "https://api.kimi.com/coding/",
      "protocol": "anthropic",
      "generation_params": {
        "temperature": 0.2,
        "max_tokens": 4096,
        "timeout_seconds": 600,
        "max_retries": 1,
        "thinking_enabled": true,
        "thinking_effort": "medium",
        "thinking_budget_tokens": 1024,
        "stream_openai": true,
        "stream_options_include_usage": false
      }
    },
    "template_sha256": "...",
    "created_at": "..."
  },
  "decision_trace": [{"check": "...", "evidence": "...", "action": "..."}],
  "output": {"code": "..."}
}
```

For `security_gate`, the assistant message is exactly one token-like label:
`[BLOCK]` or `[PASS]`. Findings and defensive rationale remain in `output`; this keeps
the trainer's classification target unambiguous.

Expert inputs are task-real, not requirement-only placeholders:

- `frontend_gen`: user content is the requirement; assistant content is complete code.
- `frontend_review`: the pipeline loads the same-seed successful frontend record and applies
  one deterministic benign-only mutation locally. The preferred mutation removes one
  literal `aria-label`; allowlisted fallbacks degrade a semantic `main` or `h1` while
  preserving balanced JSX/text. The canonical user turn is `REQUIREMENT`, `CANDIDATE
  CODE`, and `KNOWN_BENIGN_DEFECT`; the teacher returns only decision trace plus complete
  repaired `output.code`. A teacher-emitted `input` object is rejected.
- `security_gate`: the pipeline loads the same-seed successful review `output.code` as
  canonical `reviewed_code`. The user turn is `REQUIREMENT` plus `REVIEWED CODE`; the
  teacher returns only decision trace and BLOCK/PASS output and must not echo code or an
  input object. The assistant remains exactly `[BLOCK]` or `[PASS]`.

Record IDs hash the SOP and complete canonical user turn, including candidate/reviewed
code. Consequently retries for the same training input deduplicate even if teacher output
wording varies, while genuinely different candidate code receives a different ID.

Review provenance stores `source_frontend_record_id` and a mutation manifest containing
`mutation_id`, allowlisted rule, `path`, replacement count, and SHA-256 before/after.
Security provenance stores `source_review_record_id`. If the exact same-seed upstream
record is absent, ambiguous, or lacks successful code, downstream generation fails with
`UpstreamDependencyError`; it never asks the teacher to invent a substitute. This also
applies to direct CLI runs selecting only review or security.

## Kimi Code teacher

The checked-in defaults follow the Kimi Code documentation: Anthropic-compatible base
`https://api.kimi.com/coding/`, Messages endpoint `/v1/messages`, OpenAI-compatible
fallback base `https://api.kimi.com/coding/v1`, and model ID `kimi-for-coding`.
Anthropic requests use the standard `content-type`, `x-api-key`, and
`anthropic-version: 2023-06-01` headers. The client identifies itself honestly as
`anchor-moe-lora/0.1`; it does not impersonate Claude Code.

Kimi's release notes state that K2.7 Code takes effect only with Thinking enabled.
Consequently the default config sets `thinking_enabled: true` and
`thinking_effort: medium`. The OpenAI-compatible payload sends the documented
`reasoning_effort`. Both protocols omit `temperature` while Thinking is enabled so
the model can apply its required/default value; configured temperature is sent only
when Thinking is disabled. The Anthropic-compatible payload uses only the public extended
thinking shape, `thinking: {type: enabled, budget_tokens: ...}`; it does not invent
Kimi-private headers or parameters. `thinking_budget_tokens` is configurable and the
client enforces the public minimum of 1024 and refuses startup unless `max_tokens` is
greater than that budget. Anthropic
thinking and redacted-thinking response blocks are ignored: only final `text` blocks
are passed to JSON extraction, so hidden reasoning is never distilled or persisted.

- Kimi endpoint/model reference: <https://www.kimi.com/code/docs/en/>
- K2.7 Thinking requirement: <https://www.kimi.com/code/docs/en/kimi-code/whats-new.html>
- Anthropic Messages header reference: <https://docs.anthropic.com/en/api/messages>

The key is read only from `KIMI_API_KEY`. Never add it to YAML, command arguments,
JSONL, logs, or shell history. The client does not include response bodies in HTTP
errors and redacts common key forms from retry errors.

Run a minimal authentication/protocol probe before a paid batch:

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi API key"
python -m anchor_mvp.data --config configs/data/default.yaml probe
```

Model ID may be overridden with `--model` or `KIMI_MODEL_ID`. Protocol, endpoint,
API version, honest user agent, per-request token cap, total request cap, total output
token cap, HTTP read timeout, retry count, and concurrency are explicit settings.
The default `timeout_seconds: 600` accommodates long non-streaming code responses;
`max_retries: 1` limits accidental duplicate spend after a timeout. Probe and bulk
generation use the same values. Anthropic is attempted first;
OpenAI fallback is used only for compatibility statuses, not authentication or rate
limit failures. A successful compatibility fallback is latched for the remaining run,
so later requests and their provenance use `openai` directly.

Dataset persistence always waits for a complete final JSON response: a record is
parsed, safety validated, and appended atomically, and partial HTTP bodies are never
persisted as training data. The Anthropic transport remains non-streaming.

The OpenAI-compatible wire protocol defaults to SSE streaming because Kimi's
third-party setup recommends enabling streaming. SSE is consumed incrementally, but
the assembled final text is still parsed and persisted atomically only after `[DONE]`.
Only `choices[].delta.content` is accumulated; `reasoning_content`, `reasoning`, and
`reasoning_details` are ignored. Final usage is used when supplied, otherwise the
existing conservative text estimate is used. Optional
`stream_options: {include_usage: true}` is controlled separately by
`stream_options_include_usage` and defaults off for compatibility.

If a stream ends without final content, the exception reports only bounded structural
metadata: `finish_reason`, whether reasoning fields appeared, their aggregate character
count, final completion-token usage, and up to eight sanitized unknown delta key names.
Reasoning text and unknown delta values are never retained or emitted. This distinguishes
token exhaustion (`finish_reason=length`) from a parser/content-routing failure without
turning hidden reasoning into logs or provenance.

Every OpenAI SSE request also has `wall_clock_deadline_seconds` (default 900), which
is independent of socket inactivity timeout. A watchdog closes the response at the
absolute deadline, and the reader loop verifies the deadline before and after each
chunk. The resulting `ClientDeadlineExceeded` is classified as `client_deadline`, not
as a provider/server error, and partial content is discarded.

HTTP failures never log the raw provider body. Diagnostics allowlist only
`error.type`, `error.code`, and a short `error.message`, with API key and prompt
fragments redacted and strict length caps.

## Offline smoke run

From the repository root, with the package installed editable or `src` on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m anchor_mvp.data --config configs/data/smoke.yaml --dry-run run
```

Outputs are `seeds.jsonl` and one `data_<task>.jsonl` per expert. Repeating the same
command skips completed seed/task pairs. Each append is flushed and synced. A malformed
complete line fails loudly; an unterminated final line from an interrupted append is
ignored during resume.

## Safety boundary

Seed prompts permit only lawful product requirements and defensive descriptions.
Security records may name vulnerability classes but cannot contain active exploit,
malware, credential-theft, or mining payloads. Known active forms are replaced in
seed text and rejected in teacher output. This lexical boundary is a minimum safeguard,
not a substitute for human review before any corpus is used for training.

Review candidates are produced locally from frontend code and constrained to benign
accessibility/semantic mutations; the teacher no longer constructs candidate defects.
Security always audits the review stage's repaired code rather than synthesizing a second
candidate. Active-payload validation still applies to pipeline-supplied inputs and teacher
outputs, so dependency chaining cannot be used to smuggle active material into JSONL.

## Unattended gated ramp

The automation runner defaults to one serialized stage. Operators may configure an
optional non-empty list of positive concurrency values. The default `gated` policy opens
each subsequent configured stage only after the preceding stage passes all gates:

- requested-record success rate;
- canonical/training JSONL validation;
- duplicate rate;
- defensive payload scan and safety-violation budget;
- frozen held-out manifest/audit integrity plus an exact, normalized, case-family,
  seed, containment, and approximate-similarity scan of the current five JSONLs
  against the independent held-out corpus;
- request, output-token, and failure budgets.

Each domain has an independent worker/client configuration while sharing one persisted
wire budget. Seed, planner, frontend, and review default to medium Thinking effort;
tool policy and security default to low because their targets are concise defensive
classifications rather than long reasoning transcripts. Workers run one domain at a time
within the current configured stage concurrency, so five domain workers cannot silently
multiply the operator-selected limit.

When the four `heldout_*` paths are present in the automation config, every scale gate
first verifies the frozen manifest and pre-bulk audit sidecars, then locally scans the
current `data_plan`, `data_tool_policy`, `data_frontend`, `data_review`, and
`data_security` JSONLs plus all five SOP files. A collision or integrity error sets state
`gate_blocked`; concurrency does not advance. `status.json` and the
`heldout_leakage_gate` event record only the manifest/audit hashes, PASS/FAIL, counts,
threshold, and hashed collision metadata—never training or held-out text.

`data/automated_v2/automation/status.json` is atomically replaced with current stage,
budgets, cooldown, throughput, ETA, and latest gate. `events.jsonl` is append-only and
records stage starts, held-out leakage results, budget stops, client deadlines,
cooldowns, and completion. Dataset JSONL remains append-only; restarting skips completed
seed/expert pairs.

Wire usage uses a bounded group-commit checkpoint. The first request reservation is
flushed immediately; subsequent activity is flushed after eight reservations, 4,096
output tokens, five seconds, or any normal/terminal stage boundary. `status.json` records
the configured `worst_case_requests` as `max(concurrency_stages) - 1`, the tighter
`maximum_unpersisted_requests`, and the output-token threshold. Status writes use a
temporary file, file `fsync`, and atomic replacement. All role clients that share one
provider budget also share one logical pre-wire reservation tracker, so a long seed or
planner request cannot disappear merely because the security wrapper is the deduplicated
usage source.

For bulk teacher collection, `collection_policy: collect_then_partition` appends every
structurally valid, safe response before partitioning it. A stage advances only when the
partition manifest is `training_ready`. Missing raw/task rows are retried without calling
already-completed seed/expert pairs. Retry state is persisted in `status.json`; repeated
format failures converge through `max_failure_retries`, the failure budget, or
`max_stagnant_gate_rounds`. A safe but non-gold row is never overwritten: if no missing
work can improve a coverage or label-quota shortfall, the stage terminates `gate_blocked`
instead of falsely reporting `complete`.

For deterministic tool-policy and security tasks, the teacher decision remains explicit as
`provenance.teacher_observed_decision`; the local oracle supplies the authoritative target.
An unresolved disagreement remains a quality negative. A disagreement may enter gold only
when the assistant label and public trace are proven fully oracle-normalized, with no
contrary teacher rationale retained. Its partitioned provenance then explicitly records
`supervision_source: deterministic_oracle`, `oracle_normalized: true`,
`teacher_decision_agrees_with_oracle: false`, and the trace source. The manifest reports
observed, normalized, and unresolved disagreement counts; it never relabels disagreement as
teacher agreement. Malformed or unsafe classification structures are rejected before
normalization.

After collection, the runner atomically writes:

- `automation/quality_staging.jsonl`, with retained records and recomputed quality labels;
- `partitions/gold/data_<task>.jsonl`, the only training-eligible view;
- `partitions/negative.jsonl`, safe but non-gold responses retained for analysis;
- `partitions/oracle_label_only.jsonl`, trace-free deterministic labels for optional
  weak classification supervision;
- `partitions/task_bank.jsonl`, exactly one canonical card per complete strict chain;
- `partitions/reject.jsonl`, content-free hashes and reason codes for hard rejects; and
- `partitions/manifest.json`, including coverage, label quotas, hashes, and
  `training_ready`.

Malformed JSON/response structure and unclassified partition damage are corpus blockers.
Active/unsafe or credential-bearing individual rows are quarantined into content-free
rejects and never enter gold; an isolated reject does not permanently block a clean corpus
that still meets every gold floor. Frozen held-out collisions or manifest/audit drift remain
corpus blockers. Failed teacher responses are never stored verbatim;
`automation/attempts.jsonl` retains only content-free seed/task/error-class accounting.
Recompute the split without another API call. When a bound automation status exists, this
command atomically refreshes `status.partition` together with the v2 contract migration so
the snapshot gate cannot see a stale partition binding:

```powershell
py -3.10 -m anchor_mvp.data.automation `
  --config configs/data/automation.full_v3.fast.yaml `
  --partition-only
```

HTTP 429 uses short `Retry-After`/exponential retries in the client. Exhausted rate
limits persist the configured cooldown floor (or a longer server `Retry-After`) in
`status.json`. The visible runner can remain alive with `--wait-cooldown`, or exit with
state `cooldown` and resume later without repeating completed samples.

Kimi Code does not document a remaining-quota HTTP endpoint. Operators should use the
[Kimi Code Console](https://www.kimi.com/code/console) or the official CLI
[`/usage` command](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/slash-commands.html).
Do not call the Moonshot/Open Platform balance API with a Kimi Code key: it is a
different product, key namespace, and Base URL. Automation therefore classifies the
documented Code API error messages and persists state, rather than inventing a quota
probe or assuming Open Platform `X-RateLimit-*` headers exist on Kimi Code.

Offline automation E2E, which does not use a key or network:

```powershell
scripts/data/start_automation.ps1 `
  -Config configs/data/automation.mock.yaml `
  -DryRun `
  -NoWaitCooldown
```

Visible live entrypoint, only after inspecting the mock status and gate events:

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
scripts/data/start_automation.ps1 -Config configs/data/automation.yaml
scripts/data/show_automation_status.ps1 -Config configs/data/automation.yaml
```

`configs/data/automation.full_v3.yaml` is the isolated full-corpus profile. The legacy
stage marker remains 128 for an explicit status-binding migration, while the v2 contract
overcollects up to 192 raw same-seed records per expert and requires 128 strict-gold records
per expert in `data/automated_v3/`. It must start with its own quota epoch and
must not reuse an incompatible output/state directory. The top-level JSONLs are
append-only raw collection output. Both full-v3 profiles collect first: oracle
mismatches and ordinary model-quality failures remain in staging and are excluded
from gold offline. The partition manifest enforces the configured gold-label floors
(all three tool-policy labels and both security labels). Frontend/review records are checked as a same-seed DAG in
isolated copied workspaces: the trusted fixture runs `npm run build` and
`npm run test` against TSX stored as data, never imported or executed. Review
must exactly restore the deterministic frontend source after its benign mutation.
This is a bounded TSX-fragment build/test contract, not a claim that an
untrusted generated component was executed in a browser or a full React runtime.
Only a partition with `training_ready: true` may be copied to an immutable curated freeze
before training.

Prepare or inspect that freeze with the metadata-only full-v3 gate:

```powershell
py -3.10 scripts/data/prepare_full_v3_snapshot.py `
  --config configs/orchestration/full_v3_snapshot.yaml
```

The command always atomically writes `runs/full-v3-snapshot/readiness.json`. When the
partition says `training_ready: false`, it exits `3`, does not create
`artifacts/formal_v3/dataset`, and does not copy any training JSONL. The report contains
only counts, hashes, normalized blocker codes, and held-out gate metadata; it contains
neither training record bodies nor held-out text. The v2 partition contract separates
`raw_collection_target` from `minimum_gold_records_per_task`. The report computes, per
task, the maximum gold coverage still possible under the raw collection target. This
makes a mathematically unreachable target explicit instead of suggesting that a simple
resume can fill it.

When all gates pass, the command copies all five strict-gold files into a temporary
sibling directory, validates schema/secrets/cross-expert IDs, verifies that source files,
automation status, and partition manifest did not change during the copy, then publishes
the whole directory with one rename. Isolated rejects do not block a corpus merely by
existing: the v2 manifest must instead prove `partition_complete` and
`rejects_quarantined`, while the strict-gold files independently pass schema, secret,
coverage, label-quota, and held-out gates. `manifest.json` uses
`anchor.training-snapshot.v2`; `manifest.json.sha256` binds the manifest, and every
dataset has a record count, byte count, source hash, and frozen hash. An existing snapshot
is verified and reused only when it has the same source partition binding; it is never
overwritten.

This is the data-snapshot gate only. The strict OpenCode accepted-gold/session-candidate
execution gate remains independent and cannot be inferred or weakened by snapshot
readiness.

The default `automation.full_v3.yaml` stays serialized (`concurrency=1`).
`automation.full_v3.fast.yaml` is an explicit local operator profile
(`concurrency=10`) for the same 192-raw/128-gold contract. Use it only after the one-sample
OpenCode live gate is `PASS`. Provider/network stops remain resumable; soft model-quality
failures are handled by offline partitioning instead of immediate retries. Both profiles intentionally target
`data/automated_v3`, but a persisted status is bound to the ramp and quality
configuration hash, so attempting to mix serialized and fast profiles against
the same state directory fails closed. Its different quota epoch is for this
operator window only, not a bypass for the state binding. The one supported legacy v1 to
v2 migration preserves append-only rows, records old/new binding hashes and both targets,
then resumes only missing raw rows. Every other binding change still fails closed.

### Task bank, coverage matrix, and five-stage alignment

`configs/data/task_cards.v1.yaml` defines 16 balanced **sampling templates** over
nine axes: domain, interaction, layout, edge case, complexity, accessibility risk,
tool posture, review defect, and security class. These templates are moulds, not
final questions. Reusing a template at another global seed index never creates a
new card by adding a slot suffix. For `self_synthetic` data, the pipeline materializes
the final `card_id` from the accepted canonical requirement. For `swe_smith` and
`swebench_heldout`, identity is bound to an immutable source SHA-256. Gold/model
records never contain a gold patch or test oracle.

Every accepted seed carries one pipeline-owned `card_id`, `template_id`, global
`seed_index`, source binding, and canonical tags. The same `alignment_id =
hash(seed_id + card_id)` must survive planner, tool-policy, frontend, review, and
security. Failed or duplicate seed generation can retry only the same global slot;
it cannot consume a fresh question to make the count look complete. A partition is
eligible only when all of these counts are equal:

```text
unique final cards == unique alignment IDs == complete five-stage chains
                   == task_bank.jsonl rows == every strict-gold stage row count
```

Near-duplicate detection runs on canonical requirements after a base chain is
otherwise strict gold. It compares seeds, never the five per-stage prompts. The
stable lower `(seed_index, seed_id)` is retained and a loser is moved as one complete
five-record chain to negatives. `task_bank.jsonl` is then built only from the
remaining complete chains. The partition manifest exposes the content-free coverage
matrix, counts, and hashes; the immutable snapshot copies and binds the task bank as
well as all five datasets.

Previously accepted `variant-XX` and short-lived `-slot-XXXXXXXX` rows remain usable.
Each is mapped to a unique `legacy_collected` card derived from its accepted seed ID
and canonical requirement. They participate in the hard one-card/one-chain proof,
but receive no invented nine-axis labels and do not contribute to nine-axis coverage.
`swebench_heldout` cards are always excluded from training coverage and strict gold.

Finally, a teacher classification disagreement with a deterministic policy/security
oracle is not a distilled reasoning example. Such a row is excluded from strict gold,
the complete-chain count, and the task bank. A trace-free, label-only view is exported
to `partitions/oracle_label_only.jsonl` for optional weak classification supervision;
it must never be treated as planner/reviewer SFT reasoning.

### Ark GLM-5.2 384/256 monotonic expansion

`configs/data/automation.full_v3.ark_glm52.max384.c8.yaml` is the formal Ark
Responses profile for the expanded target. It keeps the append-only
`data/automated_v3` corpus, runs one concurrency-8 stage to 384 seed/complete-chain
attempts, and requires at least 256 strict-gold rows for each of the five task types.
Its 2,400-request ceiling covers the 2,304 nominal seed-plus-five-task calls with a
small bounded retry margin. The aggregate output budget is 307,200,000 tokens, exactly
`2,400 × 128,000`, so it adds no earlier local ceiling beyond the per-request and API
limits. Every worker uses `reasoning.effort=max`; the API key remains environment-only.

The profile contains `monotonic_expansion_from`, an exact description of the prior
concurrency-10, 192-raw, 128-gold contract. This declaration is not automatically
accepted by status inspection, live collection, or partition refresh. The operator
must stop every automation process and run the explicit, offline migration once:

```powershell
py -3.10 -m anchor_mvp.data.automation `
  --config configs/data/automation.full_v3.ark_glm52.max384.c8.yaml `
  --migrate-monotonic-expansion
```

The command makes no provider request and reads no credential. It succeeds only when
the persisted v2 binding equals the declared source hash, all seed/raw/gold targets and
budgets are non-decreasing, and every unlisted quality/workspace setting is unchanged.
It atomically records the old/new contracts, archives the prior run metadata, resets the
stage cursor to zero so existing append-only rows are reused, and removes the old
partition from active status with `partition_stale_reason`. A fresh directory needs no
migration; run the new profile directly. A source mismatch must be investigated rather
than bypassed or repaired by deleting an unknown state directory.

After migration, start or resume collection with the same immutable profile:

```powershell
$env:ARK_CODING_API_KEY = Read-Host -MaskInput "Ark Coding Plan key"
py -3.10 -m anchor_mvp.data.automation `
  --config configs/data/automation.full_v3.ark_glm52.max384.c8.yaml `
  --wait-cooldown
```

After a quota window ends or new rows are appended, stop the live runner and safely
recompute the partitions without a key or network call:

```powershell
Remove-Item Env:ARK_CODING_API_KEY -ErrorAction SilentlyContinue
py -3.10 -m anchor_mvp.data.automation `
  --config configs/data/automation.full_v3.ark_glm52.max384.c8.yaml `
  --partition-only
py -3.10 -m anchor_mvp.data.automation `
  --config configs/data/automation.full_v3.ark_glm52.max384.c8.yaml `
  --status-only
```

`--partition-only` rebuilds staging, gold/negative/reject files and the manifest, then
atomically refreshes `status.partition` and clears the stale marker. Exit code `3` means
the refresh was valid but the 256-per-task strict-gold gate is not yet satisfied; it is
not permission to train. Only `training_ready: true` may proceed to the immutable
snapshot gate.

The scripts contain ASCII only and read credentials solely from the current process
environment. No real unattended batch is launched by tests or repository setup.
