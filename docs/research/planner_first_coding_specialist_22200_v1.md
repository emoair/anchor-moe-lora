# Planner-first coding-specialist 22,200 v1

This is an additive migration contract, not a provider launch or training
authorization.

The count is intentionally semantic: 19,000 legacy coding candidate objects
plus 3,200 separately materialized planner-bootstrap objects. It does not
describe provider calls, expert projections, or training rows.

The serial order is fixed:

```text
source registration
  -> planner train
  -> planner freeze (Q-only primary arm)
  -> frozen planner route/question commit
  -> expert dataset projection
  -> expert leaf release
```

The old 19,008 SWE-bench public export is only a candidate source. A future
source-selection manifest must explicitly select 19,000 IDs, record the eight
excluded IDs, carry its independent source-terms acceptance, and remain
body-free. Existing 320- and 1,000-record fixtures are not substitutes for the
required 3,200 planner-bootstrap objects.

The future source gate also requires translation evidence and real tool
trajectories. Neither a candidate count nor an old launch-ready marker can
replace those provenance requirements.

Final release identifiers are deliberately one-way:

```text
Teacher FINAL -> Producer attestation -> Consumer acceptance -> Training-input authority
```

Teacher FINAL may bind a normalized semantic root, but cannot contain an exact
physical SHA for a later attestation. This prevents a hash-DAG cycle.

Until both source manifests and the planner training/freeze evidence are
authenticated, the module remains non-live: no provider request, model load,
GPU run, or training authorization is produced.
