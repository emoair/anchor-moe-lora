# OpenCode + external Skill distillation P0

This layer produces auditable tool-execution candidates. It is intentionally separate
from ordinary text distillation and never stores private reasoning, raw OpenCode event
streams, environment variables, or API keys. Controlled safe tool calls and their complete
results are retained because they are the training target.

The custom Kimi provider defines a named `medium` model variant with
`reasoningEffort: medium`, and every OpenCode execution pins `--variant medium`. OpenCode
also declares the model as reasoning-capable with interleaved `reasoning_content`, so
assistant tool-call messages retain Kimi's required protocol field on later turns. This
field remains protocol state only: the CLI is never passed `--thinking`, reasoning
events are ignored by both trace and public-outcome reducers, and only a final
`type: text` event may supply the bounded public outcome.

## Hard gates

Every candidate runs in its own copied fixture. A record is accepted only when the
agent exits cleanly, no denied tool event is observed, required local validation passes,
and the final event contains a valid `anchor.public-outcome.v1` object with status
`completed`. A modification task must also set `requires_changes: true`; an empty diff is
then recorded as `no_changes`. Missing or partial outcomes and failed validation remain
attempt records and cannot enter accepted gold.

External Skills are not trusted merely because a repository has many stars. Before a
Skill is injected, the registry verifies all of the following:

- the source repository is a literal HTTPS URL;
- the source commit is a full 40-character SHA-1;
- the SPDX license identifier and vendored license SHA-256 are present;
- every injected file matches its pinned SHA-256;
- a versioned malicious-instruction scanner finds no instruction override, credential
  exfiltration, external download command, or safety/test bypass directive.

Gold provenance carries the commit, license hash, path-bound bundle hash, and a
content-bound instruction-audit receipt. A deterministic local tool policy remains the
authority even after these checks.

## Candidate/held-out separation

`configs/tooling/execution_split_policy_v1.yaml` is the independent input inventory.
It lists candidate inputs separately from frozen execution held-out inputs and pins each
held-out file by SHA-256. Batch preflight rejects a changed held-out file, an unlisted
candidate manifest, overlapping paths, reused held-out identifiers, or an exact held-out
requirement copied into a candidate prompt.

The checked-in P0 pool currently exposes exactly one runnable task: the stable-sort task
whose fixture requirement, package scripts, and public tests agree. Its acceptance files
are hash-pinned and protected from agent edits; changing or deleting one makes the gold
record fail even if the modified scripts report success. Fourteen earlier task ideas remain
listed as deferred and are not loaded until each receives an independent fixture and
frozen acceptance contract.

The batch runner defaults to **one stage only**. Operators may configure additional stages
as any positive-integer list, provided every requested stage has audited candidates. A
stage that misses its success gate stops the requested slice, and a single sample exception
is reduced to a content-free failure record without cancelling siblings.

## Attempt ledger and accepted gold

Every live result is first merged into `artifacts/tooling/live_attempts.jsonl`. This is the
append-only audit ledger: it retains failed, partial, and successful attempts, and permits
multiple content-distinct attempts for one sample ID. Exact canonical replays are
idempotent.

Only records with `success=true` and a completed public outcome are then merged into
`artifacts/tooling/live_gold.accepted.jsonl`. The legacy mixed file remains at
`artifacts/tooling/live_gold.jsonl` for audit only. `merge_gold_jsonl` rejects ineligible new records and
also refuses to append when a legacy failure is already present. Existing accepted sample
IDs remain immutable: an identical replay is idempotent and a differing record is a hard
conflict. A failed attempt therefore does not reserve its sample ID in accepted gold.

## Collect first, filter offline

The first readiness sample still uses the default strict capture. Once it passes, formal
bulk collection adds `--capture-mode collect`. Safe complete sessions append to
`artifacts/tooling/session_staging.raw.jsonl` even when the task is blocked/partial,
validation fails, a tool returns an error, or policy rejects a tool call. Those outcomes are
quality labels, not collection-time deletion. A `rejected` tool state retains its safe input
and available error/description and can never enter strict gold.

Secrets, malformed/incomplete JSON, hidden reasoning in public text, workspace escape, and
held-out leakage remain content-dropping hard rejects. Then run:

```powershell
py -3.10 scripts/tooling/partition_session_staging.py
```

The offline pass recomputes quality and writes strict gold-compatible candidates, safe
negative trajectories, and metadata-only rejects. See
[OpenCode session distillation](opencode_session_distillation.md) for schemas and paths.

Older mixed files are not changed or deleted automatically. Review a dry-run first:

```powershell
py -3.10 scripts/tooling/migrate_legacy_tool_gold.py
```

After reviewing the three distinct paths, repeat with `--confirm`. The script writes new
`.migrated.jsonl` outputs and verifies that the legacy source remains byte-for-byte
unchanged. Replacing operational paths is a separate deliberate operator action.

## 400 and 499 handling

The Kimi base URL is validated before process launch and must be a literal official
`https://api.kimi.com/...` URL without whitespace, embedded credentials, query, or
fragment. Network tools are denied by policy, which prevents descriptive prose from
being passed as a URL. A structured HTTP 400 with code `invalid_url` is classified as
`invalid_url`, not a generic model failure and not blindly retried.

HTTP 499 or `context canceled` is classified as `client_cancelled`; it is not an upstream
5xx model failure. A wrapper deadline additionally records `wrapper_timeout`, preserving
the difference between local timeout and service failure.

## Offline preflight

### Attested OpenCode is mandatory

Live tool distillation is fail-closed until the repository-local attested OpenCode binary
exists at `artifacts/tooling/opencode-patched/opencode-anchor.exe`. The global `opencode`
installation is never accepted as a fallback. Before checking API credentials or starting
a session, the launcher runs the binary's local `debug agent anchor-distiller --pure` command
with the key removed from the child environment. The resolved agent must not contain
`requireInitialToolCall` at the top level or in provider options. Tool choice remains
automatic; the model is never forced to call a tool merely to satisfy the transport.

The reproducible patch is pinned to official OpenCode commit
`b1fc8113948b518835c2a39ece49553cffe9b30c` in
`patches/opencode/v1.17.18-anchor-distillation.patch`. Build it without touching the
global installation:

```powershell
$nodeGypRoot = "runs\opencode-build\tools\node-gyp-13"
npm install --prefix $nodeGypRoot --ignore-scripts node-gyp@13.0.1

scripts/tooling/build_patched_opencode.ps1 `
  -BunPath D:\path\to\bun-1.3.14\bun.exe `
  -NodeGypPath "$nodeGypRoot\node_modules\.bin\node-gyp.cmd"
```

The builders verify the repository origin, exact commit and patch SHA-256, require an
explicit Bun 1.3.14 executable, run the focused tests and typecheck, and emit separate
Windows x64 and Linux x64 artifacts plus a source-bound bundle manifest. On Windows the
builder also uses isolated node-gyp v13.0.1, follows upstream's hoisted Bun layout, and
uses a process-local FileTracker workaround; neither setting modifies the global
installation. The artifact manifest names four upstream Windows baseline timeouts
excluded from the focused test command. A failure leaves the globally installed
OpenCode unchanged. Exact dual-build commands and hashes are documented in
[OpenCode dual-platform build](opencode_dual_build.md).

The local binary still carries the historical opt-in patch for reproducibility, but the
Anchor configuration deliberately leaves that option unset. The offline behavioral
probe rejects `tool_choice: required` on either request and verifies automatic choice,
tool-result replay, and `reasoning_content` replay. Execution candidates enter gold only
when their observed session, file diff, public outcome, and validators pass; rejection
happens after observation rather than by constraining the model's first move.

Live execution uses the patched `opencode anchor run/export/cleanup` command family, with
one Podman job per copied fixture. This WSL 2 host uses root `systemd-run` for the outer
cgroup and rootful Podman with its own cgroup handling disabled; native Linux defaults to
direct rootless Podman. Its arguments, cleanup rule, operator
resource settings, attribution, and explicit local-only security boundary are documented
in [Anchor sandbox execution](anchor_sandbox_execution.md).

The command below validates config, all Skill hashes/audits, the frozen held-out input
inventory, candidate leakage, OpenCode availability, and key presence. It does not launch
OpenCode or call Kimi unless `--confirm-live` is explicitly added.

```powershell
py -3.10 scripts/tooling/run_live.py `
  --batch-config configs/tooling/opencode_distillation_ramp.yaml
```

After reviewing the first-stage record, an operator may request any configured stage count
with both `--confirm-live` and `--max-stages N`. Omitting `--max-stages` remains one stage.

Offline regression tests:

```powershell
py -3.10 -m pytest tests -k tooling -q
py -3.10 -m ruff check src/anchor_mvp/tooling scripts/tooling tests
```
