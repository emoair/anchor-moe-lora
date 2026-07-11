# OpenCode + external Skill distillation P0

This layer produces auditable tool-execution candidates. It is intentionally separate
from ordinary text distillation and never stores private reasoning, raw OpenCode event
streams, tool output, environment variables, or API keys.

## Hard gates

Every candidate runs in its own copied fixture. A record is successful only when the
agent exits cleanly, no denied tool event is observed, required local validation passes,
and the final event contains a valid `anchor.public-outcome.v1` object with status
`completed`. Missing or partial outcomes remain failure records and cannot enter gold.

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

The checked-in P0 candidate pool contains 15 isolated tasks. The ramp consumes fresh
tasks in stages of 1, 2, 4, and 8 concurrent sessions. A stage that misses its success
gate stops the ramp; a single sample exception is reduced to a content-free failure
record and does not cancel sibling samples.

## Append-only gold

Live stages merge through `merge_gold_jsonl`. Existing sample IDs are immutable:
replaying byte-equivalent canonical records is idempotent, while a differing record with
the same ID is rejected. The output is atomically replaced only after the merged set has
been validated, so later stages cannot silently overwrite earlier gold.

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

Offline regression tests:

```powershell
py -3.10 -m pytest tests -k tooling -q
py -3.10 -m ruff check src/anchor_mvp/tooling scripts/tooling tests
```
