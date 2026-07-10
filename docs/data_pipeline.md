# Anchor-MoE-LoRA data distillation pipeline

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

The automation runner does not start a large live batch by itself. It advances through
the fixed concurrency sequence `1 -> 2 -> 4 -> 8` (hard ceiling 8) only after every
previous stage passes all gates:

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
within the current global concurrency ceiling, so five domain workers cannot silently
multiply concurrency past 8.

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

The scripts contain ASCII only and read credentials solely from the current process
environment. No real unattended batch is launched by tests or repository setup.
