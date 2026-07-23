# Independent-confirmation identity consumer v2

This additive consumer authenticates and independently verifies the frozen
metadata artifact produced at Producer commit
`09a6829084f76790e3488cb999a6755cd4d5f95e`. It does not replace or modify any
v1 Producer contract, and it is not an authorization mechanism.

## What becomes ready

Only the metadata identity layer becomes ready:

- all 19 Producer files are externally pinned by physical SHA-256 and compared
  byte-for-byte with their local Git blobs at Producer commit `09a6829...`;
- `manifest.json.sha256` is mandatory and must be exactly
  `<manifest-sha256>  manifest.json\n`;
- five JSON Schemas are Draft 2020-12 validated with local fragment references
  only;
- 321 JSONL records and two proof documents are parsed, counted, and validated
  from the same byte snapshots used for hashing;
- the 241-entry namespace-neutral descriptor atom catalog is rebuilt;
- task, template, pair, task-bundle, inventory, membership, intersection, quota,
  and matched-factor identities are recomputed rather than copied from proofs;
- every snapshot is opened again at the end and compared by bytes, SHA-256, and
  file identity to close the parse/hash TOCTOU window.

The true independent track has 60 metadata bundles (40 train, 20 eval-proxy),
with 60 task identities, 20 template identities, 60 pair identities, and
discovery intersections of task/template/pair = `0/0/0`.

The separate controlled-factorial track has 60 metadata bundles and 20 matched
three-factor groups. Its discovery intersections are `5/1/0`; its train quotas
are `13/14/13`, and its eval-proxy quotas are `7/6/7`. This track is deliberately
not accepted as independent confirmation.

## What remains blocked

Metadata readiness does not imply that records exist or that any experiment ran.
The decision therefore fixes all of these to `false`:

- `records_materialized`
- `protected_source_disjoint`
- `independent_confirmation_executed`
- `controlled_factorial_executed`
- `quality_validated` and `generalization_validated`
- `training_authorized`, `formal_training_authorized`, and `formal`

The existing gates also remain unchanged: protected inventories are `2/6`,
formal-v3 artifacts are `0/5`, and no execution lease exists. The CLI exits with
status 2 even after a successful metadata audit.

## Run

From the repository root, using the project environment:

```powershell
$env:PYTHONPATH = "src"
python `
  -m anchor_mvp.research.independent_confirmation_identity_consumer `
  --config configs/research/independent_confirmation_identity_consumer_v2.yaml
```

Expected semantic result:

```text
status=metadata_identity_ready_execution_blocked
metadata_identity_ready=true
process exit code=2
```

No model, GPU, provider, network, protected dataset body, or training operation
is used.

## Frozen consumer entry points

- config: `configs/research/independent_confirmation_identity_consumer_v2.yaml`
  (`e69fa162dfc03e5c92168e1c529630a42b929027400870305829cad6877ea723`)
- blocked-only decision schema:
  `configs/research/independent_confirmation_identity_decision_v2.schema.json`
  (`84c9ebcbdf80b9ccb9e1a2f8c875642777bcef32b5ff6ead99a3059401ec609e`)
- implementation:
  `src/anchor_mvp/research/independent_confirmation_identity_consumer.py`
- tests: `tests/test_independent_confirmation_identity_consumer_v2.py`
