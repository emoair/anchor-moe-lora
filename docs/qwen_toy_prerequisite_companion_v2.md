# Qwen toy prerequisite companion v2

## Purpose

This companion authenticates the Consumer's request-local Qwen trigger receipt
as a non-mutating overlay on the frozen Producer v1 prerequisite package. It
turns the trigger gate from `pending_request_local_materialization` into an
independently authenticated `ready_diagnostic_only` fact without rewriting v1.

The effective Consumer condition is conjunctive:

```text
frozen v1 prerequisite AND companion v2
```

The companion does not authorize training. Four protected source-ID inventories
remain unavailable, so coverage stays 2/6, the zero-intersection proof remains
blocked, and the v1 toy attestation remains absent.

## Non-goals

This package does not:

- alter canonical Gold, held-out data, scaffold bodies, or any frozen v1 byte;
- emit prompt, answer, preview, content, raw token IDs, or a global token index;
- run the Consumer implementation, tokenizer, model, provider, network, or GPU;
- rerun request-local materialization;
- authenticate a live remote, a signed commit, formal thresholds, numeric
  equivalence, quality, physical KV sharing, multistream execution, zero-copy,
  or full-generation KV sharing;
- bind the separate Gemma proxy experiment.

`proxy_signal_passed=true` is inherited only as diagnostic provenance. It is not
a formal or numeric-equivalence claim.

## Authenticated data flow

```text
Producer v1 metadata (fixed bytes at 744e23f)
        +
Consumer Git blobs (fixed commit 7cb1f745, tree 67ca22bd)
        |
        v
strict bytes/hash/size checks
        |
        +-- Draft 2020-12 validation of Consumer config and receipt
        +-- exact sidecar and canonical JSON checks
        +-- raw 7cb1f745 tree/parent and b0441e6 ancestry checks
        +-- 7cb1f745 -> local remote-tracking ref ancestry check
        +-- inherited GIT_* overrides cleared; replace/grafts rejected
        +-- request-2 projection and safety-claim checks
        |
        v
temporary four-file fixture
        |
        +-- same-bytes reparse
        +-- final local snapshots and 8 Git blobs rechecked
        +-- ancestry/tree/local-ref rechecked
        |
        v
atomic no-replace publication
```

The source bytes come from binary `git cat-file blob <commit>:<path>` calls.
No file is read from the sibling Consumer worktree. The local ref
`refs/remotes/origin/research/neural-swarm-kv` proves only that the fixed release
commit remains in the locally observed lineage; it is not a network fetch or a
live-remote claim. Commit `7cb1f745...` is content-addressed provenance and is
not claimed to carry a verified signature.

## Why the receipt is request-local

The authoritative boundary is derived from one complete serialization and one
tokenization of request 2. The trigger covers token span `[25,33)`, using
zero-based, end-exclusive coordinates, within 44 total tokens. The covering
span has no leading overhang and one UTF-8 byte / one code point of trailing
overhang. Isolated trigger encoding is explicitly non-authoritative.

This preserves the two-request rule: Planner output is validated and committed,
then the committed scaffold is passed as input to the expert request. Planner
request-1 private KV is never represented as reusable expert KV. The companion
copies only the authenticated metadata receipt; it never copies raw IDs or
request content.

## Manifest contract

The closed Draft 2020-12 schema binds these sections:

- `producer`: companion config, schema, implementation, baseline commit/tree,
  canonical JSON, snapshot, reparse, final-recheck, and atomic-publish rules;
- `v1_dependency`: exact v1 manifest schema, pending-trigger schema, manifest,
  sidecar, protected-inventory root, 2/6 coverage, and four missing classes;
- `consumer_dependency`: release commit/tree, baseline semantics, local
  remote-tracking ref, exact ordered eight-artifact inventory, and provenance
  limitations;
- `trigger_materialization`: the receipt copy, tokenizer/chat/R2/ordered-ID
  digests, digest algorithm, span, overhang, occurrence counts, and prohibited
  reuse/emission flags;
- `inventory_status` and `proof`: unchanged v1 inventory state and a blocked
  zero-intersection proof;
- `verification`: actual schema, sidecar, projection, span, reparse, ancestry,
  and final-recheck results;
- `execution`: zero provider/network/model/GPU/protected-body/worktree reads,
  plus 8 initial and 8 final Git-blob reads;
- `claims`: diagnostic trigger readiness only; all formal and training claims
  remain false.

The ordered Consumer artifact digest is:

```text
SHA256(canonical_json({
  "domain": "anchor.qwen-toy-prerequisite-companion.consumer-artifacts.v2",
  "artifacts": [the eight ordered path/SHA/byte entries]
}))
```

Canonical JSON is UTF-8, sorted keys, compact separators, no Unicode
normalization, and no non-finite numbers. JSON documents end with one LF.
Sidecars use lowercase SHA-256, two ASCII spaces, the basename, and one LF.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| Companion config | `21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f` |
| Companion manifest schema | `596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119` |
| Companion implementation | `dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e` |
| Fixture manifest | `7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115` |
| Manifest sidecar, physical | `f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095` |
| Ordered Consumer artifact inventory | `bf38c88fd993d4804f9a624bbf98330a9d4572a6803a80c2f57a6beb05b5f567` |
| Copied source receipt | `ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e` |
| Copied source sidecar, physical | `ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a` |

Producer baseline is commit `744e23f975b13923903f5fabe04c32e74ea25dc4`
and tree `90cb962f5341717501fcb16caef13db8922f1cb4`. Consumer materialization is
commit `7cb1f7454a76fa3c8c9f46d64da9f11244b51c54`, tree
`67ca22bd2f9d50642bf88e484408082abebe2126`, with required ancestor
`b0441e6beaa07b180d7fc69e462b4d2babf21792`.

The final Producer release commit is intentionally outside the artifact hash
DAG and is reported after commit/push. A commit cannot safely embed its own
identity. Consumers should bind the reported commit and the physical hashes.

## Fail-closed conditions

Publication or audit fails if any of these occur:

- config, schema, implementation, v1 byte, Git blob path/SHA/size, raw commit
  parent/tree, ancestry, or local-ref lineage drifts;
- a Git replacement ref, legacy graft file, or inherited `GIT_*` repository
  override is present; Git object reads always set `GIT_NO_REPLACE_OBJECTS=1`;
- a sidecar is missing, non-exact, CRLF/BOM encoded, has the wrong filename, or
  is merely recomputed around tampered content;
- JSON contains a duplicate key, non-finite number, extra schema field, or is
  not canonical where canonical bytes are required;
- request-2 digests, span, coordinate semantics, width, overhang, occurrence
  count, or tokenization count drifts;
- any forbidden formal/training/KV/quality claim is promoted;
- v1 coverage is not exactly 2/6 or a zero-intersection result appears before
  all six authenticated inventories exist;
- the output exists, a path escapes, a symlink/reparse point is encountered, an
  input changes after its first snapshot, or atomic publication fails.

Focused negative tests cover synchronized manifest/sidecar tampering,
synchronized receipt/sidecar tampering, malformed sidecars, identity drift,
schema validation, exact rebuild, and no-replace publication.

## Reproduction

Use a Python 3.10+ environment with the development dependencies installed.
These commands perform no provider, network, model, or GPU operation:

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --artifact fixtures/research/qwen_toy_prerequisite_companion_v2
python -m pytest tests/test_qwen_toy_prerequisite_companion_v2.py -q
python -m ruff check `
  src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py `
  scripts/data/build_qwen_toy_prerequisite_companion_v2.py `
  scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  tests/test_qwen_toy_prerequisite_companion_v2.py
```

For a deterministic rebuild, choose a nonexistent directory:

```powershell
$env:PYTHONPATH = "src"
python scripts/data/build_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --output tmp/qwen-toy-prerequisite-companion-v2-rebuild
```

The rebuilt four files must be byte-identical to the published fixture.

## Migration and remaining work

Consumers must not modify or relax v1. They may add companion v2 as a second,
mandatory authenticated input and project only the request-local trigger fields.
Older consumers that know only v1 remain safely blocked.

Still unavailable:

- body-free per-ID inventories for `gold_partition`, `partial_gold_export`,
  `legacy_heldout_cases`, and `synthetic_scaffold`;
- the six-source namespaced zero-intersection proof and v1 toy attestation;
- a frozen formal-v3 snapshot/projector/generic/source-disjoint/release lock;
- formal tokenizer/adapter/tensor/release identities, real training, physical KV
  backend/CUDA, and formal quality/performance evaluation.

Until every independent gate is frozen and consumed, both
`training_authorized` and `formal_training_authorized` remain false.
