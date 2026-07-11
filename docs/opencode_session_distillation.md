# OpenCode session distillation boundary

## Verified implementation

The local CLI is OpenCode `1.17.18`. Tag `v1.17.18` resolves to official commit
`b1fc8113948b518835c2a39ece49553cffe9b30c`.

At that commit:

- [`export.ts` lines 240-290](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L240-L290)
  loads `Session.Info` plus every message with parts and serializes
  `{info, messages}`. This is the useful raw shape.
- [`sql.ts` lines 22-98](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/core/src/session/sql.ts#L22-L98)
  stores session, message, and part records in SQLite. Message and part payloads are JSON
  columns; direct database parsing would couple us to internal schema migrations.
- [`database.ts` lines 43-54](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/core/src/database/database.ts#L43-L54)
  places the default database under OpenCode's global data directory. On this Windows
  installation, `opencode debug paths` plus `opencode db path` resolves it to
  `%USERPROFILE%\.local\share\opencode\opencode.db`.
- [`session.ts` lines 102-128 and 259-323](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/schema/src/v1/session.ts#L102-L128)
  define text/reasoning parts and completed tool state. A completed tool part carries
  `callID`, tool name, structured input, and string output, so call-result association is
  recoverable without parsing console prose.

No existing real session transcript was read during this verification.

## Why `opencode export --sanitize` is not a training export

Official sanitize is intentionally destructive:

- text and reasoning are replaced at
  [`export.ts` lines 69-82](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L69-L82);
- file paths and source text are replaced at
  [`export.ts` lines 35-66](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L35-L66);
- tool input, output, title, and metadata are replaced at
  [`export.ts` lines 92-124](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L92-L124);
- diff paths and patches are replaced at
  [`export.ts` lines 27-32 and 125-145](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L27-L32);
- session directory, user system text, assistant cwd/root, titles, and summaries are
  replaced at
  [`export.ts` lines 163-218](https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/cli/cmd/export.ts#L163-L218).

Therefore a sanitized export is useful for sharing or forensic structure, but cannot
preserve the task, public answer, tool calls/results, or final diff needed here. The
converter rejects `[redacted:...]` markers instead of silently training on placeholders.

An unsanitized export is also **not safe by itself**. It can contain reasoning, system
prompts, environment-derived output, credentials, arbitrary absolute paths, raw tool
output, and held-out text.

## Controlled capture format

Only sessions created inside a disposable, audited fixture are eligible. Capture has two
inputs:

1. raw `opencode export <controlled-session-id>` JSON;
2. an `anchor.controlled-session-capture.v1` sidecar produced by our harness, containing
   sample ID, matching session ID, OpenCode version, complete validator stdout/stderr,
   and `anchor.public-outcome.v1`.

The resulting `anchor.session-training-candidate.v1` record contains:

- complete retained user task input and public assistant text in sequence;
- ordered `tool_call` and `tool_result` events sharing normalized `call_0001` IDs, with
  structured call input and full result content for `read`, `edit`, `apply_patch`, and
  allowlisted build/test/lint commands;
- final file patches;
- full validator stdout/stderr and exit status;
- the public outcome.

Reasoning/thinking parts, system fields, provider metadata, environment fields, original
session/message IDs, and original absolute workspace paths are not emitted.

## Fail-closed gates

Before a candidate is appended, every retained field passes:

- workspace containment and `<workspace>/...` path normalization;
- secret/credential detection;
- held-out identifier and exact-requirement leakage detection;
- UTF-8/control/binary and per-field/record size limits;
- tool and validation command allowlists;
- completed validator and public-outcome requirements.

Any hit quarantines the **entire** capture. The quarantine row stores only sample ID when
safe, reason code, source SHA-256, and `content_retained=false`; it never stores redacted
fragments from the unsafe transcript.

```powershell
py scripts/tooling/convert_session_export.py `
  --export runs/capture/session.raw.json `
  --capture runs/capture/sidecar.json `
  --workspace runs/tooling-live/sample-workspace `
  --heldout-cases configs/benchmark/heldout_cases_v1.jsonl `
  --heldout-fixtures-root examples/benchmark/fixtures `
  --heldout-manifest artifacts/benchmark/heldout_v1/manifest.json `
  --output artifacts/tooling/session_candidates.jsonl `
  --quarantine artifacts/tooling/session_quarantine.jsonl
```

The official database and arbitrary historical sessions are deliberately outside this
pipeline. The preferred future integration is event-side capture in `ToolingHarness`,
followed by raw export only for the same controlled session as a completeness cross-check.
