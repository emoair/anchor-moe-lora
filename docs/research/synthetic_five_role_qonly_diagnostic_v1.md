# Synthetic five-role Q-only diagnostic v1

## Scope

This is a deterministic, diagnostic-only corpus for the five-expert scaffold:
`planner`, `tool_policy`, `frontend_gen`, `frontend_review`, and
`security_gate`. OpenAI GPT-5.6-sol assisted the catalog authorship. The local
builder itself sends zero provider or network requests; those runtime counters
must not be misrepresented as zero-model content authorship.

The corpus is not formal training data, does not replace the frozen 100-record
fixture, and does not satisfy either future 600-record independent or
factorial materialization track.

## Exact shape

The catalog contains 200 unique task bundles. Each bundle emits one primary
view for each of the five roles:

```text
200 unique task bundles × 5 roles × 1 primary view = 1,000 records
```

Each role has exactly 200 records: 160 `train` and 40 `eval_proxy`. English
and Simplified Chinese each contribute 100 independent task semantics. The
five strata each contain 20 bundles per language; deterministic splitting is
performed within every language/stratum cell as 16 train plus 4 eval bundles.
All five role views of a task stay in the same split and share the same inner
task ID and chain-root hash.

There are no translated pairs: `translation_pair_count=0`, the English and
Chinese semantic-identity sets are disjoint, and every role covers the same
set of 200 language-neutral `task_semantic_sha256` identities.

## Output and causal boundary

The sole materialized view is `concise_rationale_plus_json`. Its rationale is
a short auditable decision summary, not hidden chain of thought. A record sees
only the task, constraints, its role instruction, and short summaries of
previous committed stages. Its current target and every future target remain
present only as hash-bound inventory entries and are listed in
`forbidden_segment_ids`; their IDs, references, bodies, summaries, and target
hashes are excluded from the materialized prompt and serialized training
input. Previous stages contribute only the short committed summary plus
non-secret role/stage labels—not target hashes or private KV.

The external roles keep the established API names. `canonical_stage` maps
them to the research stages:

| role | canonical stage |
| --- | --- |
| `planner` | `planner` |
| `tool_policy` | `tool_policy` |
| `frontend_gen` | `domain_builder` |
| `frontend_review` | `domain_review` |
| `security_gate` | `security` |

## Q-only primary

`q_only` is the only primary execution arm. `o_only` and `q_plus_o` are
diagnostic execution overlays. They do not produce duplicate dataset rows and
are never serialized into the prompt or target. The legacy `wide_lora`
control is not inherited. The dataset selects no winner and authorizes no
training or quality conclusion.

## Capability metadata

`simple_tool_search` and `micro_coding` are reserved capability names for a
later explicit catalog annotation pass. The frozen catalog has no authoritative
per-bundle capability label, so the manifest records both as
`unavailable_no_explicit_catalog_label` with null counts and no quota claim.
This makes the gap machine-auditable without guessing from task prose and does
not add rows, views, or arms.

## Build and audit

From the repository root:

```powershell
$env:PYTHONPATH = "src"

python scripts/research/build_synthetic_five_role_qonly_diagnostic_v1.py build `
  --repo-root . `
  --config configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_five_role_qonly_diagnostic_v1

python scripts/research/build_synthetic_five_role_qonly_diagnostic_v1.py audit `
  --repo-root . `
  --config configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_five_role_qonly_diagnostic_v1

python -m pytest -q tests/test_synthetic_five_role_qonly_diagnostic_v1.py
```

The builder refuses replacement, writes through a same-parent temporary
directory, validates the complete result, rechecks every input snapshot, and
publishes atomically. Audit uses one byte snapshot per file for hashing,
parsing, and counting, then performs a final TOCTOU recheck. Every file must
remain below 50 MB and `manifest.json.sha256` is mandatory.

No tag or release is created.
