# Anchor-MoE-LoRA five-stage routed-adapter MVP

[English](architecture_five_stage_mvp.md) | [简体中文](architecture_five_stage_mvp.zh-CN.md)

The MVP route is strictly ordered per seed:

`planner -> tool_policy -> (frontend_gen <-> frontend_review, bounded) -> security_gate`

All adapters use the same frozen and identically serialized Q4/NF4 base. The model
revision, quantization settings, tokenizer, local artifact digest, stage order, and
per-stage token cap are experiment invariants across A/B/C/D/E/F. Full-capacity LoRA rank
is 16 for the first experiment. A new domain coder is valid only when its paired domain reviewer is
registered and evaluated in the same change.

| Stage | Input | Output | Failure behavior |
| --- | --- | --- | --- |
| `planner` | requirement | summary, ordered steps, constraints | stop; no policy/coder call |
| `tool_policy` | plan + inert abstract proposals | exactly `APPROVE`, `BLOCK`, or `ESCALATE`, rationale and public trace | stop/fail closed; label is advisory only |
| `frontend_gen` | requirement + plan + policy advisory; on retry, current code + public issues | complete code | stop; no reviewer call |
| `frontend_review` | requirement + current candidate | strict public `anchor.domain-review-verdict.v2` JSON: `PASS` with no issues, or `REVISE` with concise issues | ambiguity/error/timeout or cycle exhaustion fails closed |
| `security_gate` | requirement + final review-passed code + public tool trace summary | exactly `[PASS]` or `[BLOCK]` | ambiguous/error/timeout becomes BLOCK |

The model cannot authorize itself. Runtime execution permission is determined by
a non-model allowlist plus workspace-boundary, side-effect, and explicit-approval
rules. A model `APPROVE` never overrides deterministic `BLOCK` or `ESCALATE`.

The primary runtime permits at most two review cycles by default. `REVISE` reuses
the same domain builder LoRA; the reviewer never writes repaired code and never
emits private reasoning. The trace can contain repeated `frontend` and `review`
attempts, but it still uses exactly five expert types. Security runs only after a
strict `PASS` and receives the final candidate plus the public proposal/policy/cycle
summary.

Legacy v1 data files are `data_plan.jsonl`, `data_tool_policy.jsonl`,
`data_frontend.jsonl`, `data_review.jsonl`, and `data_security.jsonl`. The v1
`data_review.jsonl` target is complete repaired code and remains valid only for
the compatibility `PipelineRouter.run` path and old benchmark records. It is not
silently treated as a v2 verdict adapter. Primary v2 training writes separate
`data_review_verdict_v2.jsonl` and `data_frontend_revision_v2.jsonl` targets under
schema `anchor.review-loop-data.v2`; `run_five_stage` fails closed when its
`review_verdict` adapter is absent. Every
downstream row records its same-seed source record IDs. Tool proposals are produced
locally by `anchor-inert-tool-proposals-v1`, contain no executable arguments or
URLs, and persist `executed: false`. Existing successful three-stage live rows are
recognized by seed provenance and are never rewritten during five-stage resume.

For benchmarks, the fair comparison uses five matched stages:

- A uses the Q4 base at every stage and is index 100.
- B reuses one mixed-data rank-16 LoRA at all stages (10,387,456 trainable parameters).
- C routes five full-capacity rank-16 specialists (51,937,280 stored trainable parameters).
- D routes five smaller specialists with ranks `3/3/4/3/3`; their rank sum and
  materialized trainable parameter count exactly match B.
- E is a complexity-adaptive, non-uniform routed arm. Each stage is capped at rank 16,
  but total rank/parameters are allowed to vary across a calibration-only budget ladder;
  E searches the capacity/performance Pareto frontier.
- F uses the same complexity-adaptive allocation algorithm and calibration split as E,
  but hard-constrains total rank and materialized adapter parameters to exactly match B.

C measures the maximum-capacity routed architecture. **B versus D versus F** is the
primary equal-budget comparison: B is mixed, D is manually allocated, and F is
algorithmically allocated under the same hard parameter budget. C and E are capacity/Pareto
comparisons and must not be used alone to claim an equal-budget routing win. D is fixed;
E and F are calibration-selected and frozen before held-out evaluation. Single-call A/B results remain
auxiliary only; changing the serialized Q4 artifact invalidates the comparison.

E and F are later allocation experiments, not part of `formal-v1`. Their shared initial complexity prior
is `frontend_gen >= frontend_review >= planner >= tool_policy/security_gate`. Candidate
allocations and the selection rule live in
`configs/training/complexity_adaptive_lora.yaml`. Rank selection may use only a
separate calibration split; the frozen held-out benchmark must never influence the
chosen ranks. Every stage rank is at most 16. E reports the calibration Pareto frontier
over quality, materialized parameters, routed latency, and peak VRAM. F uses the same
complexity evaluator and candidate mechanism, restricted to the exact B-sized budget.

For E, total rank is a search variable, so the output is a capacity/performance Pareto
point. For F, the same allocator is restricted to total rank 16 and exactly 10,387,456
materialized trainable parameters, matching B. This separates gains from adaptive
allocation under equal budget (B/D/F) from gains obtained by spending more or less
capacity (C/E).

The checked-in adaptive benchmark entries remain `calibration_pending`. The held-out
gate rejects them until each selected allocation has a calibration snapshot hash,
attempt ledger, frozen ranks, materialized parameter count, and immutable manifest hash.

## Roadmap after the MVP: Phase 2 context-directed expert routing

Phase 1 remains the current MVP and the canonical A/B/C/D/E/F control experiment. Its
route is deliberately fixed:

`planner -> tool_policy -> (frontend_gen <-> frontend_review, bounded) -> security_gate`

No Phase 2 implementation, sample, router decision, or metric may alter, relabel, append
to, or retroactively reinterpret the Phase 1 datasets, registries, held-out set, benchmark
records, or A--F claims. Phase 1 must finish its frozen evaluation and produce immutable
manifests before Phase 2 data generation begins. This ordering makes the fixed route a
real baseline instead of a moving target.

Immediately after that MVP evaluation is frozen, Phase 2 starts a separate experiment:
a planner/router observes the current public task context and chooses which expert LoRA
to activate, call, and unload. It may skip an unnecessary expert, revisit an earlier
expert, or create and join logical branches. The route is therefore a bounded state
machine or task graph, not a renamed `review -> execute` serial chain. On the 12 GB
single-GPU profile, branching is logically interleaved and still keeps at most one active
adapter in VRAM; Phase 2 does not imply simultaneous multi-LoRA activation.

The proposed router action contract is typed and auditable, for example `ACTIVATE`,
`CALL`, `UNLOAD`, `SKIP`, `BRANCH`, `JOIN`, `RETRY`, and `STOP`. Every action records the
public state digest, selected expert, public rationale, budget consumed, adapter lifecycle,
tool-result summary, and next-state digest. Deterministic runtime policy retains final
authority: a learned router cannot bypass workspace boundaries, tool allowlists, explicit
approval, loop/call budgets, or fail-closed security rules.

Phase 2 requires new distillation rather than repackaging the five fixed-stage targets.
The new data version must contain fresh tasks and validated router trajectories in which
the teacher receives the accumulated public context, chooses the next action/expert, sees
the public expert or tool result, and continues until a terminal state. It must include
positive examples of legitimate skips, retries, loops, branches and early stops, plus
negative or rejected trajectories for adapter thrashing, repeated no-progress calls,
unsafe transitions and budget exhaustion. Store public decisions and observable results;
do not require hidden chain-of-thought. The dataset must have a new schema/version,
snapshot hash, provenance graph, train/calibration split and immutable run ID. It must not
copy Phase 1 held-out prompts, oracle labels, solutions, sample text, or derived task IDs.

Phase 2 also requires a new held-out set and a new benchmark contract created only after
the Phase 1 evaluation is frozen. The new held-out inventory must be excluded from both
Phase 1 and Phase 2 training/distillation inputs and pass a fresh hash-based leakage audit.
Router tuning and stopping-rule selection may use training/calibration data only. For a
fair architecture comparison, rerun a frozen fixed-route reference on the new held-out set;
never present a Phase 2 score on new cases versus a Phase 1 score on old cases as a direct
gain.

In addition to task quality and safety, the Phase 2 benchmark must report routing/action
validity, expert-selection accuracy where an auditable label exists, unnecessary-call and
skip rates, loop termination, branch completion, adapter load/unload counts, adapter
thrashing, calls and tokens per task, end-to-end latency, peak VRAM, fail-closed rate, and
performance under matched call/token/parameter budgets. All route traces and evaluator
versions are frozen with the result. Phase 2 gets its own registry namespace and benchmark
name; the historical A--F table remains the fixed-route Phase 1 result.

## Two-call live smoke

With one existing seed in `data/live_smoke/seeds.jsonl`, the following command
makes at most two teacher calls and does not touch successful frontend/review/security
rows:

```powershell
python -m anchor_mvp data --config configs/data/default.yaml `
  --output-dir data/live_smoke --seed-count 1 --concurrency 1 `
  --tasks plan tool_policy --protocol openai --no-fallback `
  --thinking-enabled --thinking-effort low --stream-openai `
  --max-tokens 32768 --max-requests 2 --max-output-tokens-total 65536 `
  --max-retries 0 --wall-clock-deadline-seconds 900
```

Expected requests: one `plan`, then one `tool_policy`. If either fails validation,
the second/downstream dataset is not promoted and no bulk concurrency is enabled.
