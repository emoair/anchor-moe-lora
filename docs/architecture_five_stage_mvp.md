# Five-stage routed-adapter MVP

The MVP route is strictly ordered per seed:

`planner -> tool_policy -> frontend_gen -> frontend_review -> security_gate`

All adapters use the same frozen and identically serialized Q4/NF4 base. Initial
LoRA rank is at most 16. A new domain coder is valid only when its paired domain
reviewer is registered and evaluated in the same change.

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

For benchmarks, the primary fair comparison is five matched stages: A uses the
base model at every stage, B uses the same mixed adapter at every stage, and C uses
the five specialist adapters. Single-call A/B results remain auxiliary only.

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

