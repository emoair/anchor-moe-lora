# Synthetic scaffold independent-confirmation identity producer v1

## Status and scope

This document specifies a metadata-only Producer artifact for two logically
separate 60-bundle tracks. It contains no training examples, prompts, answers,
rationales, tool traces, protected sample bodies, or token IDs. It does not
modify the frozen natural-language scaffold fixture, canonical Gold, held-out
data, or any formal-v3 artifact.

All final physical SHA-256 identities, byte counts, partition identities, proof
identities, and the ordered read set are bound by the published manifest. They
are intentionally not copied into this document before the final freeze.

The tracks are:

1. `independent_confirmation`: 60 bundles whose namespace-neutral task
   semantics are new relative to discovery and to one another.
2. `secondary_controlled_factorial`: 60 bundles that deliberately reuse task
   or template identities under a three-cell factorial design.

They must not be merged. The secondary track cannot satisfy the
Producer-required independent-confirmation gate.

## Authenticated discovery bridge

The discovery source is the frozen `synthetic_nl_scaffold_diagnostic_v1`
producer metadata. The bridge consumes only the exact config, schemas, closed
grammar, implementation identity, manifest, and mandatory manifest sidecar
bound by the new config. It never opens the four discovery JSONL partitions.

The historical fixture has ten localized source-bundle views: five English and
five Simplified Chinese. Its old `source_bundle_id` preimage includes language,
source-local keys, and localized metadata, so it is a grouping/provenance
identity rather than a namespace-neutral semantic identity.

The bridge publishes five curated, language-neutral task semantic descriptors.
Each cross-binds exactly one English view and one Simplified Chinese view. The
two localized views share one `task_semantic_sha256` while retaining distinct
historical source-bundle identities and localized-view digests.

This bilingual mapping is an explicit Producer-curated metadata assertion. It
is not an automated proof of natural-language equivalence, and a coarse
archetype alone is not a complete task semantic descriptor.

Recomputing the historical split in the common domain exposes a fact that must
not be hidden or rewritten:

```text
| discovery_train_task_semantic_ids
  intersection discovery_eval_proxy_task_semantic_ids | = 2
```

The historical language-specific split is therefore not semantic-disjoint. The
fixture remains a diagnostic proxy, but its eval-proxy partition cannot be
promoted to an independent or held-out semantic evaluation.

## Common-domain identities

Identity preimages use strict UTF-8 canonical JSON with recursively sorted
keys, compact separators, no Unicode normalization, no non-finite numbers, and
no trailing LF. Published JSON documents use that encoding plus one final LF.

### Task semantic identity

```text
task_semantic_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-task-identity.v1",
  "descriptor_schema_sha256": descriptor_schema_sha256,
  "ontology_sha256": ontology_sha256,
  "descriptor_atom_catalog_sha256": descriptor_atom_catalog_sha256,
  "descriptor": <complete language-neutral symbolic task descriptor>
}))
```

`source_task_blueprint_sha256` is an alias of `task_semantic_sha256`, not a
second independently reported hash.

The descriptor covers the task's information-flow behavior, state/evidence
topology, operation and transition requirements, constraint classes, and
acceptance invariants. Its preimage must exclude:

- language and localized prose;
- source namespace, source-bundle ID, and bundle key;
- split, factor, role, scaffold variant, noise, and length augmentation;
- ordinal, UUID, clock, random salt, and other uniqueness-only fields.

These exclusions prevent namespace or indexing from manufacturing false
zero-overlap. Every descriptor leaf must also belong to the frozen
`anchor.synthetic-scaffold-common-domain-atom-catalog.v1` catalog (241 atoms;
catalog SHA-256
`517f6b829bb78700b171a349d14541f75a9b76aa2a9267acb92a0e1a646d9545`).
Unknown language, locale, source, namespace, salt, nonce, seed, ordinal, or
digitized atoms fail closed. Task/template identities are comparable only when
their descriptor-schema, ontology, and atom-catalog SHA-256 context is exactly
the same.

### Template family identity

```text
template_family_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-template-identity.v1",
  "descriptor_schema_sha256": descriptor_schema_sha256,
  "ontology_sha256": ontology_sha256,
  "descriptor_atom_catalog_sha256": descriptor_atom_catalog_sha256,
  "descriptor": <complete language-neutral structural template descriptor>
}))
```

The descriptor represents scaffold interface and structural rendering
semantics, not localized wording. Localized template views may have separate
view digests, but language is not a common-family salt.

### Task-template pair identity

```text
task_template_pair_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-task-template-pair.v1",
  "task_semantic_sha256": task_semantic_sha256,
  "template_family_sha256": template_family_sha256
}))
```

The pair is always recomputed from authenticated task and template leaves. A
record cannot self-report or override it.

Each logical inventory contains sorted, unique common-domain leaves. Its
logical root is distinct from the physical file SHA-256 and binds the domain,
count, and sorted values. Duplicate, missing, reordered, malformed, or unbound
leaves fail closed.

## Source bundle and split identity

`task_bundle_sha256` remains a source/grouping identity. Its canonical preimage
binds the frozen task-bundle domain, track source namespace, source-bundle
identity, language, and `source_task_blueprint_sha256`. It excludes roles,
variants, noise, length, and later causal augmentation.

Splitting occurs on `task_bundle_sha256` before five-role or two-variant
expansion. A future materializer must bind all five roles and both scaffold
variants to the same bundle and cross-bind inner `task_board.task_id`. This v1
publishes identity metadata only; it does not materialize the expected 600
training records per track.

Neither `source_bundle_id` nor `task_bundle_sha256` proves namespace-neutral
semantic independence.

## Independent-confirmation track

The independent track freezes:

- 60 source bundles and 60 unique `task_semantic_sha256` values;
- 30 English and 30 Simplified Chinese bundles;
- zero cross-language translation pairs within the track;
- five information-flow strata;
- ten language/stratum cells, each with six bundles;
- four train and two eval-proxy bundles per cell;
- 40 train and 20 eval-proxy bundles in total.

Every task semantic leaf is recomputed in the same domain as discovery. The
Producer proof compares actual leaves one by one. Namespaced IDs, aggregate
counts, and source names are not substitutes for set intersection.

Its only positive result is authenticated identity construction and a
reproducible zero-overlap proof. It does not establish quality, generalization,
statistical significance, or training readiness. `eval_proxy` is not held-out.

## Secondary controlled-factorial track

The secondary track also has 60 bundles and the same language/stratum and
`4 train + 2 eval_proxy` cell totals. It has three factors with 20 bundles each:

| Factor | Task membership | Template membership | Required pair relation |
| --- | --- | --- | --- |
| `old_task_new_template` | In discovery task inventory | Outside discovery template inventory | Outside discovery pair inventory |
| `new_task_old_template` | Outside discovery task inventory | In discovery template inventory | Outside discovery pair inventory |
| `new_task_new_template` | Outside discovery task inventory | Outside discovery template inventory | Outside discovery pair inventory |

Global task and template zero-overlap are intentionally false. Global
task-template-pair zero-overlap is required and recomputed from leaves.

Each language/stratum cell has two bundles per factor. The frozen rotation gives
one factor `2 train + 0 eval_proxy`; each other factor gets
`1 train + 1 eval_proxy`, ordered within the pair by `task_bundle_sha256`.
Across ten cells, the totals are:

```text
factor order: old_task_new_template,
              new_task_old_template,
              new_task_new_template
train:        13 / 14 / 13
eval_proxy:    7 /  6 /  7
```

The proof publishes per-bundle membership truth values and recomputes them
against discovery inventories. Labels cannot stand in for membership. Passing
this proof remains secondary factorial plumbing and cannot satisfy independent
confirmation or bundle generalization.

## Metadata-only artifact and read boundary

The artifact contains closed-schema metadata records, IDs-only inventories,
proof metadata, a manifest, and `manifest.json.sha256`. It contains no prompt,
answer, rationale, task text, constraint prose, tool trace, token ID, Gold row,
held-out row, or scaffold record body.

The manifest binds an explicit ordered read set. On the Producer side it reads
the canonical config, four schemas, and the implementation. From the sibling
source repository it physically reads exactly three authenticated source
metadata files (diagnostic config, manifest, and mandatory sidecar) plus one
separately classified authenticated consumer-plan contract. Generator and
closed-grammar identities are transitive bindings read from that source
manifest; their physical files are not opened by this producer. It does not
recursively discover inputs or open discovery partitions, Gold, held-out,
protected scaffold JSONL, provider, model, tokenizer, or GPU resources.

That exact semantic-file read set is separate from local Git provenance reads.
The manifest also lists and hashes the fixed 11-operation Git metadata/object
read set: worktree/Git-dir resolution, graft/replace checks, both commit
objects, four exact blobs, and the source-to-plan ancestry check. These are
local object-database attestations, not a live-remote or signed provenance
claim.

For every structured JSON/YAML input and output (with raw-byte identity checks
for implementation and sidecars), the producer enforces:

1. repository-relative lexical paths without absolute paths, `..`, or
   backslashes;
2. rejection of symlinks, junctions, reparse points, and non-regular files at
   every component;
3. one immutable bytes snapshot for hash, parse, schema validation, and count;
4. duplicate-key and non-finite-number rejection;
5. same-bytes reparse;
6. final hash/size/identity recheck for TOCTOU detection;
7. exact artifact layout and output-size limits;
8. a temporary sibling directory and atomic no-replace publication;
9. an exact lowercase sidecar line:

```text
<manifest sha256><two spaces>manifest.json<LF>
```

An existing target, publication race, source drift, synchronized manifest
tamper, or sidecar drift fails closed.

## Claims and authorization boundary

The top-level `claims` object remains entirely false. This artifact does not
authorize training or formal release and does not claim held-out evaluation,
quality, statistical significance, bundle generalization, physical KV reuse,
Q-reader implementation, zero-copy reuse, or numeric equivalence.

Mechanical evidence such as schema validation or an authenticated zero-overlap
proof may pass without promoting a claim. Runtime counters remain zero for
provider, network, model, tokenizer, GPU, training, Gold-body, held-out-body,
scaffold-body, and dataset-body access.

## Reproduction

Run from the Producer repository. `PYTHONPATH=src` resolves the local
implementation. The source root is used only for exact authenticated metadata
files in the manifest read set.

The supported trust path is a fresh CLI process: `--config` must resolve to the
canonical config, schemas and implementation use compiled-in canonical paths,
the config and four schema SHA-256 identities are compiled into the
implementation, and the loaded implementation path/bytes are frozen at module
import and rechecked through publication. Alternate configs, synchronized
parallel schema roots, and stale imported-module execution fail closed.

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.swebench.synthetic_scaffold_independent_confirmation_identity build `
  --repo-root . `
  --source-root ../anchor-moe-lora-neural-swarm `
  --artifact fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1

python -m anchor_mvp.swebench.synthetic_scaffold_independent_confirmation_identity audit `
  --repo-root . `
  --source-root ../anchor-moe-lora-neural-swarm `
  --artifact fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1
```

`build` is no-replace and requires the target not to exist. `audit`
authenticates the frozen artifact. Deterministic rebuild comparison must use a
separate empty artifact path and compare exact bytes; it must not overwrite the
frozen fixture.

## Known limitations and unfinished work

- Historical discovery semantic split intersection remains two; this producer
  records it and does not rewrite history.
- The five bilingual discovery mappings are curated metadata assertions, not a
  general semantic-equivalence theorem.
- The secondary track cannot satisfy independent confirmation because two
  factor cells deliberately reuse discovery identities.
- Neither track has materialized its future 600 role/variant records.
- Provider requests, model loads, tokenizer binding, GPU execution, training,
  physical KV, multistream runtime, quality evaluation, and performance
  evaluation are outside this artifact.
- Protected-source inventories and formal-v3 snapshot/projector/generic
  execution/source-disjoint/release-lock gates remain separate prerequisites.
- Consumer integration remains fail-closed until it binds the final manifest,
  schemas, config, implementation, partitions, proofs, and sidecar identities.
- Final SHA-256 values and exact counts are authoritative only through the
  final manifest; provisional hashes must not circulate as frozen inputs.
