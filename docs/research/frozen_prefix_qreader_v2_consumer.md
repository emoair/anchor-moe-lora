# Frozen-prefix Q-reader V2 consumer

This is the model-free, additive consumer for Producer commit
`8c9fdfc71b94b5b41d6f3566e9f81baadcc0c267` on
`research/frozen-prefix-qreader-distillation-v2`.

It authenticates the Producer tracking ref, commit, tree, parent, exact
24-path change set, Git blob IDs, byte counts, and SHA-256 values. It then
validates the Producer profile, materializer, release-overlay contracts, and
the local mandatory `manifest.json.sha256` sidecar from single byte
snapshots. A second full Git/object and local-file check closes the TOCTOU
window.

It does **not** copy or read Gold, held-out, protected, or materialized
training records. It does not load a model, touch a GPU, call a provider, use
the network, train, release, tag, or authorize live execution.

## One-command preflight

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/research/preflight_frozen_prefix_qreader_v2_consumer.py
```

The command intentionally exits with code `2`. A successful contract
authentication prints one JSON object. The relevant fields look like:

```json
{
  "status": "producer_contract_ready_execution_blocked",
  "producer_contract_authenticated": true,
  "gates": {
    "training_authorized": false,
    "formal_training_authorized": false
  }
}
```

Exit code `2` means “the identity/contract is usable, but execution remains
blocked”; it is not a failed audit when the status above is present.

## What is machine-checked

- Exact five-role map:
  `planner`, `tool_policy`, `frontend_gen`, `frontend_review`,
  `security_gate`.
- Split grouping is `task_bundle_sha256`, before augmentation.
- One primary `concise_rationale_plus_json` view per role.
- `q_only` is primary; `o_only` and `q_plus_o` remain non-authorizing
  diagnostics.
- Current target, future, and forbidden bodies are excluded before
  serialization. Whole-board stringify and “serialize then mask” are
  forbidden.
- The route is an explicit two-request validate/commit boundary. Only
  committed text is re-encoded by the frozen base.
- Shared prefix is adapter-off. After routing, each expert has an append-only
  private tail KV. Cross-expert private-tail reuse is forbidden.
- No token index or physical KV tensor is emitted; committed scaffold
  re-encoding is adapter-off; wide LoRA is not inherited; source records are
  not rewritten.
- Exact reuse is limited to identical ordered prefix lineage. No physical
  Q-reader, full-generation shared-KV, naive in-stack Q-LoRA exact reuse, or
  token-level MoE claim is made.

## Why it remains blocked

This consumer authenticates a contract, not a live data/training release.
The V2 materialized training view, bundle profile, generic release lock,
execution decision/lease, byte-level TOCTOU lease, and live provider
distillation are absent. Formal-v3 is `0/5`; protected inventories are
`2/6`. Therefore training, formal training, release, and live authorization
are all false.

## Troubleshooting

- `producer_tracking_ref_unavailable`: the required tracking ref is not
  available locally. Update Git outside this preflight, then rerun. The
  preflight never fetches.
- `producer_provenance_mismatch`: the tracking ref no longer points to the
  frozen commit. Do not override it; obtain a new versioned contract.
- `producer_blob_identity_mismatch`: a pinned Producer blob does not match
  its frozen byte identity. Do not relax the hash.
- `manifest_sidecar_*`: restore the exact mandatory sidecar format
  `<64 lowercase hex><two spaces>manifest.json<LF>`.
- `*_contract_invalid`: the producer or consumer semantic boundary drifted.
  Version the contract rather than editing the frozen V2 meaning.

## Files

- Config:
  `configs/research/frozen_prefix_qreader_v2_consumer_v1.yaml`
- Manifest and mandatory sidecar:
  `fixtures/research/frozen_prefix_qreader_v2_consumer_v1/`
- Loader:
  `src/anchor_mvp/research/frozen_prefix_qreader_v2_consumer.py`
- CLI:
  `scripts/research/preflight_frozen_prefix_qreader_v2_consumer.py`
- Tests:
  `tests/test_frozen_prefix_qreader_v2_consumer.py`
