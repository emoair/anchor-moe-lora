# Formal authorization consumer v1

This additive overlay answers one narrow question: **may the current research
stack be treated as formally authorized training?** The answer in v1 is always
no. The overlay cannot launch training and contains no promotion path.

It authenticates these exact, version-isolated inputs before making that
decision:

- the Qwen prerequisite v2 consumer, its config, and its mandatory companion
  manifest sidecar;
- the multi-seed/independent-bundle blocked planner and config;
- the generic release v2 consumer and schema;
- the physical `formal_authorization_decision_v1.schema.json`, whose Draft
  2020-12 contract permits only the blocked v1 state.

Authenticated Python dependencies are compiled and executed from the exact
byte snapshots whose SHA-256 values are pinned in the config. Normal imported
module objects are not trusted. Every snapshot is checked again after the
decision, so replacement during evaluation fails closed.

## Machine conclusion

- formal-v3: `0/5`;
- protected source inventories: `2/6`;
- the controlled-factorial dataset remains a secondary proxy and cannot
  satisfy independent confirmation or bundle-generalization;
- `anchor.generic-train-release-lock.v2` is scoped to
  `research_proxy_only`, so an artifact satisfying that old schema is not a
  formal release;
- `training_authorized=false`, `formal_training_authorized=false`, and
  `formal=false`.

The decision reports zero provider, network, model, GPU, protected-body, and
training operations. Successful evaluation still exits with code 2 because
the authorization result is blocked.

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.formal_authorization_consumer `
  --config configs/research/formal_authorization_consumer_v1.yaml
```

This overlay does not modify the frozen Qwen prerequisite v1/v2 contracts and
does not consume Gold, held-out, scaffold, or record bodies.

## Execution enforcement

Formal execution is blocked at three independent entry points:

1. `run_formal_v3_lowmem.ps1` accepts only the repository-canonical
   `runs/formal-v3-training.lock`, acquires it before invoking this consumer,
   and rejects every v1 decision even if forged fields claim `ready=true`;
2. `anchor_mvp.training.cli` evaluates the gate before device probing,
   preflight/body reads, dataset validation, manifest writes, or runtime import;
3. `train_adapter()` repeats the gate before creating a progress/output path.

There is no ready path in v1. A future bare
`formal_training_authorized=true` decision is deliberately not enough. Any
future ready implementation must use a new, versioned v2-or-later decision and
an authenticated, launcher-held execution lease bound to the decision, run
config, adapter/rank/stage, and canonical lock identity. That decision/lease
contract is not available in v1, so launcher, direct CLI, and library execution
remain fail-closed.
