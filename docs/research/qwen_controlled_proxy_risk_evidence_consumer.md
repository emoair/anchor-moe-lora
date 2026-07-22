# Qwen controlled-proxy risk-evidence consumer v1

This consumer is an additive, fail-closed diagnostic overlay. It does not
modify the frozen Qwen prerequisite-v2 consumer and cannot authorize training
or a formal release.

## What it authenticates

The consumer executes three independent checks in order:

1. the existing Qwen prerequisite-v2 consumer, which must remain `blocked`;
2. the frozen Producer controlled-proxy follow-up auditor;
3. the frozen Producer Q+O risk-evidence companion auditor.

All local metadata are read through one immutable byte snapshot used for
hashing and parsing, then rechecked after evaluation. Mandatory SHA-256
sidecars use the exact `sha256sum` form with an LF terminator. Producer release
commit, parent, tree, mode, blob OID, blob bytes, and local bytes must agree.
Git replace refs, grafts, lazy fetch, external Git environment overrides,
symlinks, junctions, path traversal, duplicate YAML/JSON keys, and body-like
fields fail closed.

The overlay reads no Gold, heldout, scaffold, dataset, model, or adapter body.
It performs no provider, network, model, or GPU request.

## Interpretation boundary

`o_branch_retained` means the O branch retained after disabling the Q branch
inside a **jointly trained Q+O checkpoint**. It is not an independently trained
O-only arm. The retained fraction is a post-hoc branch ablation on the single
seed, short-context proxy at step 80. It does not establish additivity,
memorization, causality, statistical significance, broad bundle
generalization, or a formal winner.

An independently trained, equal-budget O-only control requires a separate
training receipt and a new versioned evidence contract.

## Run

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_controlled_proxy_risk_evidence_consumer `
  --config configs/research/qwen_controlled_proxy_risk_evidence_consumer_v1.yaml
```

A successful authentication prints a content-free decision with:

- `status=blocked`;
- `evidence_status=authenticated_non_authorizing_diagnostic`;
- `training_authorized=false`;
- `formal_training_authorized=false`;
- process exit code `2`.

Missing or drifting inputs fail closed. Passing this overlay never changes the
formal-v3 `0/5` or protected-inventory `2/6` gates.
