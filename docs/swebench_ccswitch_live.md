# Full-bank SWE-bench train coordinator

[Simplified Chinese](swebench_ccswitch_live.zh-CN.md)

This is the formal patched CC Switch + patched OpenCode coordinator for the
19,008 public **train-only** candidates and 95,040 dependency-bound five-stage
work orders. The train path uses generic `repository + base_commit`
materialization and the explicit `real_sandbox_self_verified` evidence tier. It
does not read SWE-bench dev/test/heldout bodies, gold patches, test patches,
hints, oracle labels, or private TestSpec data.

## Claim boundary

Current train status is **READY in implementation and offline tests**:

- the public bank contains 19,008 candidates, exactly five work orders each;
- patched OpenCode owns real sessions, edits, tools/results, exports, and cleanup;
- patched CC Switch owns provider/model routing, exact MAX reasoning, pricing,
  token/cost accounting, and request statistics;
- every candidate is checked out at its public `repository + base_commit` in a
  disposable digest-pinned WSL2/Podman train sandbox;
- a terminal `anchor-validate compile|test` result, independently recomputed
  final state, exact patch, cleanup result, and complete stage lineage are bound
  into a supervisor-HMAC receipt;
- the receipt and exported manifest declare
  `not_official_swebench_pass=true`.

READY does not mean live-demonstrated. No provider-backed one-task pilot or
full-bank run is claimed by this documentation refresh. Official TestSpec and
heldout evaluation are an independent **NOT READY** track and never gate train
trajectory generation.

## Zero-request preflight

```powershell
Set-Location D:\LLM\anchor-moe-lora
$Python = "$HOME\.conda\envs\anchor-mvp\python.exe"

& $Python scripts\tooling\run_swebench_ccswitch.py
# equivalent entry point
.\anchor.ps1 -Action distill-swebench
```

The default command is read-only. It sends zero provider requests, does not read
credential environment variables, and does not start CC Switch, OpenCode, a
sandbox, or a GPU job. It reports separate gates:

- `component_ready`: pinned patched binaries, manifests, and route profiles;
- `bank_ready`: full task/work-order inventory, hashes, dependencies, and locale
  assignments;
- `execution_contract_ready`: the generic train-sandbox, validator, OpenCode,
  source-bank, and receipt-key bindings match the current execution lock;
- `live_start_allowed`: every train launch prerequisite is true.

Official evaluation readiness is reported separately and is non-blocking for
train distillation. `launch_ready` is compatibility metadata for components and
bank only; it is not LIVE authorization. A JSON file that merely says
`ready=true` is rejected.

### WSL route-listen discovery

`sandbox_route_visibility` is fixed to `wsl-probed-host`. LIVE startup does not
trust the first WSL default-route next hop: TUN software may place an address
there that Windows cannot bind. The coordinator tries `127.0.0.1` first. A NAT
fallback is eligible only when the address is both a WSL default gateway and a
Windows Preferred IPv4 address; arbitrary WLAN/private addresses are never
exposed. Every candidate must bind a temporary Windows TCP listener, accept a
TCP connection from the configured WSL distribution, and then pass the same
Windows-health plus WSL-TCP checks on each fixed route port. Failure is a
pre-provider, fail-closed startup error.

## Generic train sandbox contract

The train split does not provide a usable private TestSpec/test patch. The
coordinator therefore materializes the public repository at the exact base
commit, copies it into an isolated task workspace, and proves the validation
capability before sending any provider request. A task whose repository cannot
be materialized or whose changed code cannot be validated fails before provider
work and remains retryable.

The patched OpenCode agent may edit only the disposable workspace. Its final
qualifying tool call must be exactly one of:

```text
anchor-validate compile
anchor-validate test
```

The validator emits a structured final-state record covering changed paths,
tracked binary diff, untracked file identities, validator version/source hash,
and the immutable train-sandbox image digest/ID. The trusted supervisor parses
that final tool result, independently recomputes the same state before cleanup,
requires an exact match, then signs a post-cleanup HMAC receipt. A
validate-then-edit trajectory, arbitrary command, unsupported changed code,
failed cleanup, missing binding, or tampered artifact is not Gold-eligible.

The sandbox is not an official SWE-bench evaluator. Its successful receipt
means only `real_sandbox_self_verified`; every receipt and Gold manifest must set
`not_official_swebench_pass=true`.

## WebUI and process-only credential

Start the control plane and open <http://127.0.0.1:8765/>:

```powershell
.\anchor.ps1 -Action ui
```

The WebUI is the recommended surface. It shows train gates, exact reason codes,
progress, complete-chain concurrency, cumulative cap, throughput, ETA,
request/token/cost counters, retry/reconnect state, backend disconnects, safe
Stop, and exact-checkpoint Resume. Official TestSpec/heldout state is separate
from the generic train gate.

Enter a provider key only in the password field after Start is enabled. The key
exists only in the dashboard process and selected child environment; it is
cleared from the browser field and never written to YAML, `.env`, argv, status,
logs, Git, or examples.

For CLI use, set the key only in the current PowerShell process:

```powershell
$env:ARK_CODING_API_KEY = '<paste for this PowerShell process only>'
```

## Required one-task pilot

The first non-resume LIVE launch is intentionally restricted to one task at
concurrency one:

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Concurrency 1 -MaxTasks 1
```

This command has **not** been run as part of the current evidence refresh. A
provider response is not sufficient. Before raising concurrency, verify the
pilot's content-free status/checkpoint shows a complete five-stage chain,
terminal validator result, successful cleanup, and valid supervisor-HMAC train
receipt. On failure, correct the cause and Resume the same checkpoint; do not
delete the checkpoint or export the failed task.

## Exact cumulative concurrency ramp

After the pilot receipt is verified, use the same checkpoint and the following
exact Resume ladder:

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 8  -MaxTasks 16
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 16 -MaxTasks 48
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 24 -MaxTasks 96
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 30 -MaxTasks 156
```

`MaxTasks` is cumulative. Each cap ends as
`stopped_checkpoint_resumable`. Resume verifies and skips authenticated
successes instead of spending quota on them again. Failed or incomplete tasks
remain retryable and never enter Gold. Stop increasing concurrency on repeated
systemic provider, transport, route, RAM, disk, or sandbox errors and Resume at
the last clean tier. Provider support for 30 concurrent requests does not prove
the local host can sustain 30 complete task chains and sandboxes.

After a clean tier is established, Resume with a larger cumulative cap or leave
the cap unset to cover the complete bank. Remove the process-only key after the
child exits:

```powershell
Remove-Item Env:ARK_CODING_API_KEY -ErrorAction SilentlyContinue
```

## Five-stage and checkpoint contract

Every task is ordered and predecessor-hash-bound:

```text
planner -> tool_policy -> domain_builder -> domain_review -> security
```

The 9,504 `zh-CN` and 9,504 `en-US` values are deterministic routing
assignments. Localization changes natural language only and preserves code,
identifiers, paths, commands, JSON keys, tool arguments/results, URLs, and
hashes.

Content-bearing prompts, responses, tool calls/results, OpenCode exports, and
diffs stay under the private live content directory. Public status and
checkpoint metadata remain content-free and hash-bound. A security PASS alone
never completes a task. Gold requires the complete five-stage lineage, real
tool transcript, terminal validation bound to the final diff, review/security
PASS, successful cleanup, and authenticated train receipt.

An authenticated successful chain remains usable in a capped export and is
skipped on Resume. The partial manifest must set
`not_for_full_bank_completion_claim=true` until all tasks are covered. Failed or
incomplete tasks are retried; malformed or tampered authenticated evidence
fails closed rather than being silently regenerated into Gold.

## Official evaluation stays independent

Official TestSpec/heldout work is not part of this train coordinator and is NOT
READY. It must use a separately authorized, version-isolated evaluation path;
heldout bodies, private tests, official verdicts, and evaluation HMAC material
never enter teacher prompts, train transcripts, Gold, or public artifacts. A
later official benchmark claim requires that independent evaluation and cannot
be inferred from `real_sandbox_self_verified` receipts.

## Publication boundary

Public scope is limited to code, configs, documentation, attribution, audited
public train-bank shards below 50 MiB, and content-free manifests/audits. Do not
publish provider credentials, private HMAC keys/receipts, unapproved
teacher/session bodies, heldout bodies, weights, adapters, checkpoints, runs,
logs, or private evaluation records.

Do not create a Git tag, GitHub Release, or version package without a new,
explicit user instruction.

This project was built with assistance from OpenAI GPT-5.6 SOL.
