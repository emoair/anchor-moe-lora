# Anchor-MoE-LoRA five-stage routed-adapter MVP

The MVP route is strictly ordered per seed:

`planner -> tool_policy -> frontend_gen -> frontend_review -> security_gate`

All adapters use the same frozen and identically serialized Q4/NF4 base. The model
revision, quantization settings, tokenizer, local artifact digest, stage order, and
per-stage token cap are experiment invariants across A/B/C/D. Full-capacity LoRA rank
is 16 for the first experiment. A new domain coder is valid only when its paired domain reviewer is
registered and evaluated in the same change.

| Stage | Input | Output | Failure behavior |
| --- | --- | --- | --- |
| `planner` | requirement | summary, ordered steps, constraints | stop; no policy/coder call |
| `tool_policy` | plan + inert abstract proposals | exactly `APPROVE`, `BLOCK`, or `ESCALATE`, rationale and public trace | stop/fail closed; label is advisory only |
| `frontend_gen` | requirement + plan + policy advisory | complete code | stop; no reviewer call |
| `frontend_review` | requirement + deterministically benign-mutated coder output | complete repaired code | stop; no security call |
| `security_gate` | requirement + reviewed code | exactly `[PASS]` or `[BLOCK]` | ambiguous/error/timeout becomes BLOCK |

The model cannot authorize itself. Runtime execution permission is determined by
a non-model allowlist plus workspace-boundary, side-effect, and explicit-approval
rules. A model `APPROVE` never overrides deterministic `BLOCK` or `ESCALATE`.

Data files are `data_plan.jsonl`, `data_tool_policy.jsonl`,
`data_frontend.jsonl`, `data_review.jsonl`, and `data_security.jsonl`. Every
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

C measures the maximum-capacity routed architecture. B versus D isolates routing and
task separation under an equal adapter-parameter budget. The D allocation is frozen
before held-out evaluation. Single-call A/B results remain auxiliary only; changing
the serialized Q4 artifact invalidates the comparison.

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
