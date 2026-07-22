# Synthetic natural-language scaffold diagnostic v1

## Purpose

This artifact is a small, deterministic training proxy for the five-role
scaffold interface. It is independent of SWE-bench Gold, partial Gold,
held-out cases, and the earlier scaffold inventory. The producer reads none of
those bodies or paths.

It is deliberately **diagnostic only**:

- `formal=false`;
- `training_authorized=false`;
- `eval_proxy` is not held-out data;
- no source-disjoint or zero-intersection proof is claimed;
- no physical KV reuse, numeric equivalence, or quality result is claimed.

## Dataset shape

The closed grammar defines ten synthetic source bundles: five English and five
Simplified Chinese. Each bundle expands into five roles and two paired output
forms:

```text
10 bundles × 5 roles × 2 variants = 100 records
```

The roles are `planner`, `tool_policy`, `frontend_gen`, `frontend_review`, and
`security_gate`. The variants are `json_only` and
`concise_rationale_plus_json`. The concise rationale is an auditable decision
summary, not hidden chain of thought; both variants carry the exact same
canonical routing JSON.

Splitting happens at the source-bundle level before role expansion or any
augmentation. Eight bundles produce 80 training records, and two bundles
produce 20 `eval_proxy` records. Every role, language, and variant remains
balanced, and no bundle crosses the split.

## Compact causal view

The training materializer never stringifies the whole task board. A role sees:

- the synthetic task and three constraints;
- its role instruction;
- short committed summaries and local `S0`–`S4` references for prior stages.

Full prior targets stay outside the prompt. Current and future segment IDs,
references, and target bodies are forbidden. The record retains hashes that
bind every short summary back to its full synthetic target.

The dataset itself is tokenizer-unbound and emits no token counts. A later run
preflight must bind a tokenizer and measure the complete chat-templated prompt
plus target without truncation. As a non-persisted local check, the current
compact view was measured with the local Qwen 2.5 1.5B tokenizer and all 100
records were below 1024 tokens; this is not a formal dataset claim.

## Ablation controls

Every record is eligible for all three labels:

- `q_only`;
- `q_plus_o`;
- `wide_lora`.

The dataset does not assign target modules or choose a winner. A future
diagnostic run manifest must bind the actual module map and use the exact same
record inventory, split, and seed for all three arms.

## Build and audit

Run from the repository root in the project Python environment:

```powershell
$env:PYTHONPATH = "src"

python scripts/research/build_synthetic_nl_scaffold_diagnostic_v1.py build `
  --repo-root . `
  --config configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_nl_scaffold_diagnostic_v1

python scripts/research/build_synthetic_nl_scaffold_diagnostic_v1.py audit `
  --repo-root . `
  --config configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_nl_scaffold_diagnostic_v1

python -m pytest -q tests/test_synthetic_nl_scaffold_diagnostic_v1.py
```

Build refuses to replace an existing output. Publication uses a same-parent
temporary directory, validates the complete artifact, rechecks all input
snapshots, and then renames it into place. Audit uses one byte snapshot per
file for hashing, parsing, and counting, followed by a final TOCTOU recheck.
`manifest.json.sha256` is mandatory and must use the exact
`<64hex>  manifest.json\n` form.

## Read set and release discipline

The declared semantic read set contains only the versioned config, closed
grammar, schemas, and generator implementation in this namespace. Provider,
network, model, GPU, real-tool, and protected-body counters are all zero.

This work creates no tag or release and authorizes neither. Do not publish a
tag or release without a later explicit instruction and an independent formal
release lock.
