# Qwen training-prerequisite consumer gate

This command authenticates the producer's frozen explanation of why formal
training is currently blocked. It does not read Gold, heldout, or scaffold
JSONL bodies; load a tokenizer, model, or GPU; or promote a local mechanical
diagnostic into an A–F result.

```powershell
anchor-qwen-prerequisites `
  --config configs/research/qwen_train_prerequisite_consumer_v1.yaml
```

Successful authentication prints `status=blocked` and
`training_authorized=false`, then exits with status 2 to mean “the contract is
valid, but training remains blocked.” The config, four schemas, status manifest,
and exact `manifest.json.sha256` sidecar are bound by physical SHA-256. JSON
parsing and schema validation use the authenticated byte snapshot, followed by
a final identity/hash recheck. Paths must remain inside the repository and may
not traverse a symlink, junction, or other reparse point.

Frozen identities:

- producer commit: `a8efe5f55b72960b49bcb1ae3753b633afd14959`
- consumer config: `4fdc8173baaa9f14d93a288b18f38691be62bb1fb8e646c579a06d9c78bc1a8a`
- status schema: `e8d09abc26effcedc642125b4d84185f0e5072a23f5611f068274bd963c4f577`
- status manifest: `70c8f0a866c5fb41c4c3726638b55a66efab77f8b2ee31c27ad31ab55def67da`
- tokenizer-binding schema: `5b2e7c2e8e6efc1c9b7251fde853631e65806aca0364d9bb092ee9a07d135b25`
- toy-attestation schema: `7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea`
- formal release-lock schema: `119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa`

## Why it remains blocked

- The formal-v3 snapshot, final projector, generic execution contract,
  source-disjoint manifest, and release lock are unavailable.
- Only 85/256 strict five-stage chains exist; Review and Security remain below
  256 records.
- The tokenizer is an authenticated candidate source, not a bound artifact.
  Token coordinates may only be created after exact request-2 serialization.
- The six protected source classes have physical file hashes but no unified,
  frozen, recomputable source-ID/domain/namespace inventories. There is also no
  real toy generator, config, closed grammar, or attester. A `ready` toy
  attestation therefore cannot be minted honestly.

The producer must first freeze metadata-only inventories and real generation/
audit artifacts under a new compatible contract. The v1 contract must not be
mutated or relaxed, and zero hashes or test placeholders are not artifacts.
