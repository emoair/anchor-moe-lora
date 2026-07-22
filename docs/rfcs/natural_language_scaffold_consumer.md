# Natural-language scaffold consumer

Status: contract-only MVP. It performs no model load, provider request, GPU
work, gradient update, or physical KV reuse.

## Data flow

1. Authenticate the frozen producer config, three schemas, smoke contract,
   manifest sidecar, manifest, and four JSONL partitions from one byte snapshot
   per file.
2. Validate every record against the closed producer schema. Unknown fields,
   unsafe paths, symlinks, SHA drift, count drift, and post-read replacement
   fail closed. The consumer uses the `jsonschema` Draft 2020-12 validator on
   all 20 published rows; its local structural checks are defense in depth, not
   a replacement for the published schema.
3. Cross-bind each scaffold record to the authenticated TaskBoard row using
   projector manifest SHA, source partition SHA, canonical source-line SHA,
   task bundle, task ID, source Gold, stage, role, language, segment plan, and
   target hash.
4. Build the user input only through `build_training_view()`. This hard filter
   removes forbidden/current/future bodies. `scaffold_text` is the assistant
   target; the original stage answer is never copied into the prompt.
5. Keep `json_only` and `concise_rationale_plus_json` as two physically and
   logically distinct ablation arms. A pair must have identical input and
   variant-specific targets. A future launcher must select exactly one arm.

## Two-request state machine

Request 1 uses the frozen base with the adapter off. It may produce a concise,
auditable rationale summary, strict routing JSON, tool trace, and expert trigger
candidate. Validation and explicit commit promote text only. Planner-private KV
is never transferred.

The committed scaffold must be re-encoded by the frozen base with the adapter
off, producing a new immutable lineage. Only Request 2 may activate the selected
expert, and only when tokenizer identity and a concrete adapter attestation are
bound. Same-request and generated mid-request switching remain prohibited.

The current synthetic fixture has unbound tokenizer/cache identity, no adapter
artifact, no re-encode attestation, and `execution_authorized=false`. Therefore
it can materialize Request-1 contract views but Request 2 and gradient training
remain fail-closed.

## Reproduce the low-memory preflight

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\research\preflight_natural_language_scaffold_consumer.py `
  --expected-consumer-config-sha256 `
  79cf993e4f4496b57786602bcbec3ac9048d4ad2a9fd6d5033bff64ab65c0640
```

The command prints content-free counts and hashes only. It does not print sample
bodies or heldout data. The expected config hash is mandatory: changing the
consumer config without deliberately updating the launch lock fails closed.

## Remaining gates before a training smoke

- frozen formal-v3 release lock;
- exact tokenizer, chat-template, trigger-text, and ordered-token identity;
- real adapter file and tensor-inventory attestation;
- committed-scaffold re-encode receipt and immutable lineage;
- explicit single-arm launch configuration;
- correctness, memory, throughput, and quality evaluation.

`q_only`, `q_plus_o`, and `wide_lora` are experiment labels, not proof of the
actual trained tensors and not execution authorization.
