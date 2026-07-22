# Qwen diagnostic-toy prerequisite consumer

This companion consumer validates the metadata-only toy prerequisite emitted
by Producer commit `744e23f975b13923903f5fabe04c32e74ea25dc4`.
It is version-isolated from the existing Qwen formal prerequisite gate, does
not modify v1, and never mints the legacy
`anchor.qwen-toy-source-disjoint-attestation.v1`.

## Usage

```powershell
anchor-qwen-toy-prerequisite `
  --config configs/research/qwen_toy_prerequisite_consumer_v1.yaml
```

This diagnostic is intentionally repository-local: run it from a source or
editable checkout whose copied contracts and metadata fixture can be
authenticated together. A standalone wheel is not a supported trust root.

The correct current result uses exit code `2` and returns `status=blocked`.
This is an authenticated research state, not a program failure:

- only the SWE-bench source and heldout ID inventories can be authenticated
  without reading protected bodies: `2/6 ready`;
- Gold partition, partial Gold export, legacy heldout, and synthetic scaffold
  remain `4/6 unavailable` and cannot be treated as empty sets;
- the request-local trigger receipt is still
  `pending_request_local_materialization`;
- `zero_intersection_claimed=false`, `v1_attestation_emitted=false`,
  `training_authorized=false`, and `formal_training_authorized=false`.

The consumer may read exactly 26 schemas, manifests, sidecars, and hashed-ID
files. `toy/diagnostic.jsonl` is neither copied into this repository nor read;
Gold, heldout, and scaffold bodies are also outside the whitelist. A single
byte snapshot drives parsing and hashing, followed by a final identity recheck;
file replacement or a reparse path fails closed.

Progress beyond this state requires body-free per-ID inventories for the four
missing classes and a request-local token/span/overhang receipt produced from
the complete request 2 by a bound tokenizer. Even then, this path remains
diagnostic-only and does not automatically authorize formal training.
