# Qwen diagnostic toy prerequisite producer

## Goal and non-goals

This companion producer makes the prerequisites for a diagnostic-only Qwen
toy run machine-verifiable without changing the frozen
`anchor.qwen-toy-source-disjoint-attestation.v1` schema. It authenticates
metadata-only source-ID inventories, creates a deterministic toy partition
from a closed local grammar, and records exactly which prerequisite classes
remain unavailable.

It does not read or emit Gold, held-out, or scaffold sample bodies. It does
not call a provider, load a model, use a GPU, authorize formal training, prove
semantic/content uniqueness, or make a quality claim. An unavailable ID
inventory is never represented as a zero count.

## Data flow

1. Authenticate the producer config, schemas, implementations, closed grammar,
   and permitted source metadata from a single byte snapshot per file.
2. Reparse the same immutable bytes and reject any non-identical result.
3. Build one inventory manifest plus mandatory `manifest.json.sha256` sidecar
   for each of the six protected source classes.
4. Emit opaque, domain-separated ID leaves only when a complete body-free
   per-ID set is available.
5. Generate eight diagnostic toy records solely from the closed grammar and
   public deterministic seed.
6. Independently rebuild the expected toy partition in an auditor that does
   not import the generator.
7. Recheck every authenticated input at the end, write the audit and main
   manifest sidecars, then atomically publish the directory.

The generator's semantic read set is exactly its implementation, the pinned
config, and the closed grammar. Python/runtime I/O is not misrepresented as
application semantic input tracing.

## Six protected source classes

The fixed ordering is `swebench_source`, `gold_partition`,
`partial_gold_export`, `heldout`, `legacy_heldout_cases`, and
`synthetic_scaffold`.

- `swebench_source` is ready. Its 19,008 IDs are obtained from two
  manifest-authenticated, identifier-only allowlists; candidate task bodies
  are not opened.
- `heldout` is ready. Six already-published normalized case-ID digests are
  consumed from the held-out manifest; the case body file is not opened.
- `gold_partition` is unavailable because complete IDs exist only inside
  protected Gold JSONL bodies.
- `partial_gold_export` is unavailable because it has no independent body-free
  per-ID inventory. Aggregate record counts are not substituted.
- `legacy_heldout_cases` is unavailable because the current protected JSONL
  lacks an authenticated body-free witness for its exact bytes.
- `synthetic_scaffold` is unavailable because its aggregate task digest cannot
  be expanded into a recomputable per-ID leaf set without opening scaffold
  records.

Consequently coverage is 2/6. The artifact deliberately omits intersection
count, intersection digest, and proof-input digest, sets
`zero_intersection_claimed=false`, and emits no frozen v1 attestation.

## Identifier and hash rules

Raw file identity is SHA-256 of exact bytes. JSON documents use UTF-8 compact
JSON with sorted keys and no Unicode normalization; published documents end
with one LF. A manifest sidecar is the lowercase digest, two spaces, the
basename, and one LF.

An available source ID leaf is:

```text
SHA256(UTF8(namespace) || NUL || UTF8(native_identifier))
```

For held-out data, `native_identifier` is explicitly the previously published
normalized case-ID digest in lowercase hexadecimal form. It is not claimed to
be the raw case ID. An inventory digest is SHA-256 of the sorted unique leaf
hex strings joined by LF with no trailing LF. Domain policy and namespace
inventory are canonical-JSON hashed and embedded in each source manifest.

## Request-local trigger materialization

The companion contract reserves a request-local receipt for the verified
two-request aLoRA boundary. A future ready receipt must bind the exact complete
chat-templated request-2 serialization SHA, tokenizer/chat-template identity,
the digest of ordered input token IDs, a zero-based exclusive trigger span,
and boundary overhang. The full request must be serialized and tokenized once.

Isolated trigger encoding is explicitly non-authoritative. Raw token IDs and a
global token index are forbidden. Activation is
`next_request_input_activation_only`; a trigger generated during request 1
cannot hot-switch the same request, and Planner request-1 private KV is never
claimed reusable by the expert. The checked-in fixture remains pending and
contains no tokenizer-derived values.

## Fail-closed conditions

The producer or auditor rejects path traversal, symlinks/reparse points,
identity or byte-length drift, schema/config/implementation hash drift,
noncanonical JSONL, duplicate or malformed identifier leaves, source ordering
changes, missing/mismatched sidecars, toy record tampering, an open or
authorizing grammar, unavailable inventories with a count/digest, fewer than
six explicit source statuses represented as complete coverage, and any
attempt to claim zero intersection while coverage is incomplete.

The test suite validates every JSON instance with the published Draft 2020-12
schemas and also changes toy bytes together with their main manifest to prove
that the independent grammar rebuild still detects tampering.

## Reproduction

From the repository root, with the project on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='src'
python scripts/data/build_qwen_toy_prerequisite.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_v1.json `
  --output <new-empty-output-directory>
python scripts/data/audit_qwen_toy_prerequisite.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_v1.json `
  --artifact <new-empty-output-directory>
python -m pytest tests/test_qwen_toy_prerequisite.py -q
```

The build is intentionally small: 8 toy records, no provider/model/GPU/network
activity, and no protected sample-body reads. Existing output directories are
not overwritten.

## Version and migration discipline

The frozen `qwen_toy_source_disjoint_attestation.schema.json` remains byte-for-
byte at SHA-256 `7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea`.
This producer is a companion prerequisite version, not a migration of v1.
Consumers may authenticate and display its exact missing-source status, but
must not mint the frozen v1 attestation until all six per-ID inventories are
independently authenticated and a full namespaced intersection is recomputed.

Real Gold/partial/legacy/scaffold ID inventory producers, a completed
request-local tokenizer receipt, a formal-v3 release lock, real training, and
quality/performance evaluation remain unfinished. Nothing in this artifact
promotes a diagnostic signal into formal training authorization.
