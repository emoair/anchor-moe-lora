# Canonical Distillation Contract

This is the single source of truth for the formal pipeline. Historical c10,
384+128, web-template generation, and direct-API collectors remain synthetic
evidence only; they do not replace this path.

## Corpus boundary

- Pin an immutable official SWE-bench/SWE-smith **train** revision and process
  every eligible train task, not a 512-task sample.
- The verified source lock is `SWE-bench/SWE-bench` revision
  `7074ef12ea2a6f70a228943c1336553333c22786`: train has 19,008 rows,
  19,008 unique instance IDs, and 35 repositories. The 106,492,326-byte parquet
  matches the official LFS SHA-256
  `0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69`.
- Record the upstream revision, hashes, exact counts, repository distribution,
  exclusions, and derived manifests.
- Create validation deterministically from train only. Never read, translate,
  route, train on, or publish official test/dev/heldout task bodies.
- Keep every public data file below 50 MiB and run secret, heldout, license, and
  large-file checks before Git staging.

## Dependency-bound stages

Every source task keeps one `task_id` and expands in order:

1. `planner`: strict JSON instructions for the selected builder, scope,
   dependencies, acceptance checks, and risks.
2. `tool_policy`: pre-build review of the plan and proposed tool calls.
3. `builder`: real checkout, editing, tool use, build, and tests in a disposable
   sandbox.
4. `reviewer`: domain review with structured revision loops back to the builder.
5. `security`: final auditable release/block verdict.

The stages must share predecessor artifacts and hashes. Five unrelated answers
are invalid.

## Teachers and MAX reasoning

- Formal teachers are GLM-5.2 and Kimi-K3.
- **Every formal request to either teacher uses the provider's exact `MAX`
  reasoning tier.** `high`, default, missing, or silently downgraded values fail
  preflight closed.
- GLM-5.2 provides broad coverage. Kimi-K3 is also a formal teacher and is
  preferentially routed to frontend, multimodal, and build-heavy work.
- The product must not hard-code a provider or model. Protocol, base URL, model
  ID, reasoning tier, route, and pricing metadata remain operator-configurable.
  Model discovery may use the URL plus the process-only credential; discovery
  failure permits explicit manual or forced-manual model selection. GLM-5.2 MAX
  and Kimi-K3 MAX are the current formal profile, not product limits.
- Credentials live only in process memory/environment and are inherited by child
  processes. They are never written to YAML, Git, examples, logs, or a repository
  secret store.

## Training view

Keep the model-visible input, structured decision/rationale trace, strict stage
output, tool calls, real tool results, build/test evidence, and final patch or
verdict. Never fabricate hidden reasoning or narrate an execution that did not
happen. For now, exclude OpenCode's mutable system/front prompt while retaining
model I/O and the actual execution trajectory.

The derived view is bilingual Chinese/English. Translate natural language only;
never translate code, identifiers, paths, commands, JSON keys, tool arguments or
results, hashes, or license text.

Train-bank acceptance uses the explicit
`real_sandbox_self_verified` evidence tier, not an official SWE-bench verdict.
Each accepted task binds the complete five-stage artifact lineage, exact
candidate task/work-order shard hashes, isolated tool transcript, one terminal
nontrivial zero-exit `anchor-validate compile|test` result, the parsed validator
JSON and independently recomputed final state, exact final patch, immutable
image digest/ID, validator source hash, repository/base identity, and successful
cleanup in a supervisor-HMAC receipt. The consumer independently verifies the
publication inventory and all of those bindings. The manifest must set
`not_official_swebench_pass=true`. Missing evidence remains excluded and
retryable; malformed or tampered authenticated evidence fails closed. A capped
`stopped_checkpoint_resumable` run may export authenticated successes but must
set `not_for_full_bank_completion_claim=true` until all tasks are covered.
Official heldout evaluation remains independent and never enters training.

## Dual modified runtime

- Patched OpenCode owns agent sessions, real tools/results, disposable WSL2/Podman
  sandboxes, build/test, exports, and cleanup.
- Patched CC Switch owns provider/model selection, routing, MAX-tier preservation,
  pricing, token/cost accounting, usage, and request statistics.
- The formal launcher fails closed when either attested component, route manifest,
  or sandbox preflight is missing. It never silently falls back to synthetic direct
  API collection.

The standard control surface is `http://127.0.0.1:8765/`: URL, process-memory key,
model, concurrency, reconnect delay, retry count, start/stop/resume, throughput,
tokens/cost, progress, disconnect state, and reconnect reason must be visible.
Concurrency defaults to one for portability and is explicitly raised after
preflight; the product does not impose a hard global ceiling.

Network routing is explicit. Domestic resources and bulk downloads, especially
assets above 10 GiB, default to the physical NIC route and bypass system/virtual
network proxies. When a proxy or virtual route is detected, the panel shows the
direct/proxy choice, expected size, and destination before transfer and allows an
override; it never silently spends proxy traffic on bulk assets.

Clearing proxy variables or setting `NO_PROXY=*` means proxy-environment bypass;
it must not be presented as a physical-NIC route when a TUN default route wins.
True physical mode requires an auditable, reversible provider-host route pin or
equivalent interface binding. If elevation is unavailable, fail closed or require
the operator to choose inherited routing explicitly. Preflight does not mutate the
system route table.

No Git tag, GitHub Release, or version package may be created without a new,
explicit user instruction.

Every operator-facing guide must have separate, synchronized Simplified Chinese
and English files. Commands, fields, evidence boundaries, status semantics, and
troubleshooting must match; a partial machine-translated summary is insufficient.
