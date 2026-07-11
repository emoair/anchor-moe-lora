# OpenCode + external Skill distillation P0

This layer produces auditable tool-execution candidates. It is intentionally separate
from ordinary text distillation and never stores private reasoning, raw OpenCode event
streams, tool output, environment variables, or API keys.

The custom Kimi provider defines a named `thinking` model variant with
`reasoningEffort: high`, and every OpenCode execution pins `--variant thinking`. This is
request-side routing/compute configuration only: the CLI is never passed `--thinking`,
reasoning events are ignored by both trace and public-outcome reducers, and only a final
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

The ramp shape remains 1, 2, 4, and 8, but only stage 1 can run with the current trusted
pool. Requesting later stages fails preflight for insufficient audited candidates. The live
CLI and batch runner default to **one stage only**; `--confirm-live` alone can never widen
the run. A stage that misses its success gate stops the requested slice, and a single sample
exception is reduced to a content-free failure record without cancelling siblings.

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

The command below validates config, all Skill hashes/audits, the frozen held-out input
inventory, candidate leakage, OpenCode availability, and key presence. It does not launch
OpenCode or call Kimi unless `--confirm-live` is explicitly added.

```powershell
py -3.10 scripts/tooling/run_live.py `
  --batch-config configs/tooling/opencode_distillation_ramp.yaml
```

After reviewing the first-stage record, a deliberate two-stage live run would require
both `--confirm-live` and `--max-stages 2`. Omitting `--max-stages` remains capped at one.

Offline regression tests:

```powershell
py -3.10 -m pytest tests -k tooling -q
py -3.10 -m ruff check src/anchor_mvp/tooling scripts/tooling tests
```
