[English](distillation_dashboard.md) | [简体中文](distillation_dashboard.zh-CN.md)

# Distillation dashboard and local control plane

This standalone page observes distillation JSONL metadata and can optionally
start one strictly-configured automation subprocess. It does not expose prompts,
messages, model output, generated code, absolute shard paths, or credentials.

## Start the page

Run from the repository root. The current external c10 collector can be attached
read-only at startup:

```powershell
python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10
```

Open `http://127.0.0.1:8765/`. The server accepts only the exact IPv4 loopback
bind. `localhost`, `::1`, `0.0.0.0`, and remote binds are rejected.

The page supports English and Simplified Chinese. Its first visit follows the
browser language; the language button stores only the non-sensitive language
preference in local storage.

Use monitor-only mode when process controls are not needed:

```powershell
python scripts/observability/distillation_dashboard.py `
  --monitor-only `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10
```

For a one-shot terminal summary or content-free JSON snapshot:

```powershell
python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10 `
  --once

python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10 `
  --once --json
```

`--shard` is repeatable. Prefer `LABEL=DIRECTORY`; only the operator label is
returned by the HTTP API. A shard passed with `--shard`, or added with **Attach
read-only**, is never granted process ownership and cannot be stopped by the
page.

## Live telemetry

The page polls `/api/snapshot` every two seconds and displays:

- seed rows and rows for `plan`, `tool_policy`, `frontend`, `review`, and
  `security`;
- complete chains, derived from the five stage seed-ID intersection;
- cumulative requests and output tokens from the status audit ledger;
- retained-stage input/output/total/cache token subtotals, clearly marked as
  lower bounds when any request or usage dimension is missing;
- rolling audit-ledger output tokens and requests, alongside separately named
  retained-row token and wire-attempt rates;
- rolling per-stage and total persisted rows per minute;
- accepted seed count plus quarantined seed-rejection count/rate and recent
  content-free rejection reason codes;
- retries, content-free error-class counts, budgets, ETA, and lifecycle events;
- whether a shard is a managed child or an external read-only attachment.

All rolling rates use observed counter deltas over a window of up to 60 seconds.
They are `unknown` until two usable observations exist. A counter reset also
returns `unknown` rather than a negative rate.

Token numbers are never estimated from text length. Cumulative request and
output-token counters come only from `status.json`'s `audit_ledger`; current
`quota_epoch` counters are used only for quota progress bars and are never
presented as lifetime totals. Retained-stage usage comes only from
`provenance.teacher.provider.completion.usage`. Input and total tokens are not
claimed globally exact when the audit ledger does not contain those dimensions.
Each value carries an `exact` flag and, where applicable, an `unknown_rows`
count. A non-exact value is a known subtotal plus an unknown remainder.

A status checkpoint is fresh only when it is no older than the newest retained
data beyond a grace period. The grace period is at least 30 seconds and grows to
`3 * usage_checkpoint_policy.maximum_seconds + 5` when that policy is larger.
This prevents a normal checkpoint cadence from looking stale while still
preventing old current-state counters from being paired with newer JSONL data.

`seed_rejections.jsonl` is scanned with a separate four-field whitelist. The
dashboard reads only `error_class`, `reason`, `content_retained`, and
`observed_at`; it does not materialize seed indices, response hashes, or any
other field. Free-form validation text is reduced to a fixed reason-code enum
such as `active_payload_material`, `credential_like_material`, or
`invalid_json_object`. Unknown text becomes `unclassified_validation` and is
never returned. A row that does not explicitly declare `content_retained:
false` is displayed only as `metadata_policy_violation`.

## Start a new managed shard

The form deliberately has no free-form command field. The base-config selector
lists only strict automation configs containing valid SOP/output, stage,
concurrency, seed-count, and budget structures; task cards and SWE-bench configs
are excluded. Supply:

- a strict base config and task-card config from `configs/data`;
- a new relative output directory below `data/`;
- a seed offset whose range does not overlap a registered config or prior
  control-plane manifest;
- concurrency, provider URL/protocol/model, and the API key;
- transport timeouts/retries, automation budgets, cooldown behavior, and
  supervisor reconnect settings.

The optional CC Switch catalog is read-only metadata derived from the bundled or
validated active snapshot pinned to v3.16.5. Provider/model presets fill only
the provider URL, protocol, and exact request model ID; every field remains
manually editable. **Check pinned diff** performs an offline read-only refresh
and never downloads or applies metadata. The dashboard never reads the CC
Switch database, OpenCode configuration, or provider keys.

Pinned cost is displayed only when the provider binding, exact alias, supported
protocol, all four usage dimensions (`input`, `output`, `cache_read`, and
`cache_write`), and reviewed price are known. Otherwise the result is explicitly
`UNKNOWN`; no missing dimension is treated as zero.

The default network route is **direct**. The child receives `NO_PROXY=*` and no
inherited proxy URL. **Inherit detected proxy** is an explicit opt-in that copies
the current process's proxy environment to the child. The API exposes only a
`proxy_detected` boolean, never a proxy URL or credential.

The key is accepted only for that action, copied into a best-effort zeroizable
RAM slot, and passed to the child as `ANCHOR_CONTROL_API_KEY`. It is not written
to YAML/JSON, returned by an API, placed in argv, or printed by the dashboard.
The browser password field is cleared after Start, Continue, or model discovery.
The RAM slot is cleared after exit, stop, discovery, explicit **Clear key**, or
server shutdown. Supplying it again is required after a dashboard restart.

Start generates immutable, secret-free files at:

```text
runs/control-plane/<run-id>/effective-config.yaml
runs/control-plane/<run-id>/control-manifest.json
```

They record hashes of the base config, task-card file, and SOP tree plus the
effective provider, transport, budget, concurrency, output, and invocation
settings. The only production child command is:

```text
<current-python> -m anchor_mvp.data.automation --config <effective-config>
```

`--wait-cooldown` is appended only when selected. It runs with `shell=False`, a
fixed repository working directory, a sanitized environment, null stdio, and a
new process group/session. The output directory receives an exclusive ownership
lock before spawn.

## Stop and Continue

**Stop** first requests a cooperative process-group interrupt. After a bounded
grace period it terminates the process tree, then kills it if necessary. The
collector writes accepted JSONL rows with flush/fsync, but its current core does
not install a dedicated cooperative signal checkpoint. Therefore a stop may
leave `status.json` looking mid-worker even though already-fsynced rows remain
durable.

**Continue exact run** reloads the original effective config and rejects every
override. It checks the config SHA-256, strict manifest/effective-config field
agreement, run/directory identity, automation corpus binding, current base
config and task-card hashes, current SOP-tree hash, completion state, and an
ownership-token lock. A control process may resume an active-looking state only
when that same in-memory controller observed its own child exit. After a
dashboard restart, an active-looking or otherwise externally-owned shard is
attach-only; this prevents accidentally launching a second writer.

The reconnect controls in the form supervise unexpected child-process exits.
They are separate from `max_retries`, which controls provider request retries
inside the automation process. Exponential supervisor backoff is bounded and is
cancelled by Stop.

## Model discovery

**Probe/load models** queries the provider's model-list endpoint using the
RAM-only key. Discovery is optional: selecting **Force model** permits an exact
manually-entered model ID if the provider does not support listing. Probe
responses are reduced to syntactically safe model IDs and a small status enum;
provider response bodies and credentials are never exposed. Each concurrent
probe owns its own request-local key, probes cannot race with Start, and HTTP
redirects are rejected so an authorization header cannot cross origins.

## Browser and HTTP boundary

The page has no external scripts, stylesheets, fonts, or telemetry. Mutating
POSTs require all of the following:

- exact `Host: 127.0.0.1:<bound-port>`;
- exact same-origin `Origin`;
- an HttpOnly, SameSite=Strict RAM session cookie;
- `X-Anchor-CSRF: 1`;
- one strict UTF-8 `application/json` body of at most 16 KiB.

Duplicate JSON keys, non-finite numbers, BOMs, query strings, absolute request
targets, chunked bodies, unexpected fields, and CORS preflights are rejected.
Control inputs are workspace-confined, symlink-checked, and range-validated.

JSONL readers retain byte offsets and partial-line buffers, so unchanged files
are not rescanned on each poll. They decode only whitelisted metadata paths.
Malformed JSONL is represented only by source, line number, and SHA-256; no line
fragment is returned. HTTP responses use `no-store`, a restrictive Content
Security Policy, `nosniff`, and same-origin resource policy, and default request
logging is disabled.
