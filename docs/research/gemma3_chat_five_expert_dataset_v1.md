# Gemma 3 1B IT five-expert chat diagnostic dataset v1

## 1. Purpose

`gemma3_chat_five_expert_qonly_v1` is an additive, diagnostic-only chat
dataset for the local Gemma 3 1B IT export. It contains exactly 200 unique
task bundles and five counterfactual expert views per bundle:

1. `humor`
2. `serious`
3. `angry_style`
4. `tool_call`
5. `review_audit`

The resulting matrix is exactly 1,000 rows. Bundle splitting happens before
role expansion: 160 bundles / 800 rows are `train`, and 40 bundles / 200 rows
are `eval_proxy`. English and zh-CN each contribute 100 bundles, split 80/20.
`eval_proxy` is not held-out.

This asset is independent of the existing SWE-bench, canonical Gold,
held-out, five-stage, Q-only scaffold, and frozen-prefix Q-reader artifacts.
It sets `replaces_existing=false` and does not change their bytes, counts,
schemas, or claims.

## 2. Non-goals and claim boundary

The dataset does not claim:

- hidden chain-of-thought;
- clinical emotion recognition;
- a real search engine or a real tool trajectory;
- native multi-turn Gemma tool-role serialization;
- source-disjointness from protected corpora;
- held-out quality, formal quality, or production readiness;
- physical or full-generation shared KV;
- that a 100M adapter, rank 1024 adapter, or any adapter was trained;
- training, formal, release, provider, or live authorization.

`user_emotion` is a small closed routing label plus a concise, auditable
rationale summary. It is bundle-shared metadata, not a sixth expert or a
clinical inference.

## 3. Causal topology

The five views are not a linear five-stage chain:

```text
shared user input and router metadata
  |-- humor private branch
  |-- serious private branch
  |-- angry_style private branch
  `-- tool_call private branch
          |
          | four outputs commit and are re-encoded as immutable inputs
          v
      review_audit join
```

The five roles may have different role-specific instructions and context.
Only the final user message, `user_emotion`, and router label must be identical
within a bundle; `router.user_message_sha256` authenticates that shared user
turn. The first four prompts cannot see sibling outputs, current targets,
future targets, or forbidden content. Their generated content remains
expert-private until explicit commit. `review_audit` is a
second-request/post-hoc view whose private context sees exactly the four
committed outputs in the fixed order shown above and digest-binds them. Those
dependencies cannot leak into the other four views. The reviewer cannot see
its own target. This is metadata and text materialization; it does not claim
physical KV tensors or zero-copy reuse.

## 4. Role behavior

`humor` stays helpful and uses gentle, non-mocking humor. `serious` is direct,
calm, and concise. `angry_style` is firm and energetic but must remain
non-abusive and contain no threat. These three views preserve the same task
facts and routing identity.

For 195 ordinary bundles, `tool_call` emits one closed, structured assistant
target containing:

- a deterministic synthetic-local tool call;
- the matching synthetic-local result;
- a grounded final answer whose evidence references are a subset of that
  result.

The call, result, arguments, evidence IDs, final answer, and their hashes are
cross-bound and re-derived by the auditor. This is
`deterministic_synthetic_local_tool_result`, not a claim that an external tool
ran.

Five distinct semantic bundles are identity/ownership alignment tasks. Their
stable factual core is:

```text
我是由Air训练的测试模型。
```

The identity fact must be preserved across `humor`, `serious`,
`angry_style`, and `tool_call`; no response may attribute training to Google
or OpenAI. For these five bundles only, the `tool_call` view records
`direct_no_tool_identity` and answers directly. It must not fabricate a tool
call or result. `review_audit` checks both the identity fact and the correct
no-tool decision. These are five genuinely different semantic tasks, not
renamed, translated, duplicated, or salted copies.

## 5. Identity and split rules

`task_semantic_sha256` is derived from a complete, language- and
namespace-neutral semantic descriptor. Dataset namespace, language, role,
split, view, seed, and adapter labels are forbidden from the semantic
preimage. `task_bundle_sha256` derives from the semantic identity and
language-neutral local evidence identity; role and split are still excluded.
Localized input text has its own digest.

The producer and auditor independently prove:

- 200 distinct semantic identities and 200 distinct bundle identities;
- no EN/zh-CN semantic intersection and no translation pairs;
- five exact roles per bundle and 200 rows per role;
- identical semantic, bundle, split, language, final user message, emotion,
  and router-label identity across all five views, while allowing
  role-specific instructions/context;
- 160/40 bundle split and 800/200 row split;
- 100 EN and 100 zh-CN bundles, each split 80/20;
- train/eval bundle intersection equals zero;
- 200 valid review joins with exactly 800 ordered parent references.

## 6. Gemma serialization

The producer binds the frozen external Gemma chat policy rather than claiming
the local Keras export contains a native chat template. The export explicitly
has `chat_template_bound=false`.

Frozen identities include:

- chat policy SHA-256
  `0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6`;
- SentencePiece model SHA-256
  `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`;
- tokenizer config SHA-256
  `90e9a8120520ef24c0a0d62a6d87188658c43ccc66ae5bc6f74d9a80804e6919`;
- combined tokenizer/template/special-token policy SHA-256
  `1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946`.

The runtime overlay uses BOS/EOS IDs 2/1; it does not trust the reversed 1/2
values in the exported config and never modifies canonical model files.
Gemma system roles are unsupported, so role instructions live inside the
first user message.

Tool call/result/final data is a single structured assistant target under the
frozen single-turn user-to-assistant serializer. It is not native tool-role
serialization. Every new row receives a dataset-specific tokenizer receipt:
prompt, input and labels digests, token counts, trainable-label count,
sequence limit, and `truncation_used=false`. Raw token IDs are never
published. The old tokenizer receipt for the previous 1,000-row dataset is
only an authenticated source identity and cannot certify this dataset.

## 7. Publication and read set

The deterministic builder reads only its checked-in config, schemas, closed
grammar/local evidence, implementation, frozen Gemma policy and binding
metadata, and the authenticated SentencePiece bytes. It does not read
protected data, SWE-bench, Gold, held-out, or existing scaffold bodies.

Every small input is hashed, parsed, and counted from one bytes snapshot.
JSON and YAML reject duplicate keys and non-finite constants. Paths reject
traversal, symlinks, junctions, and other reparse points. The output directory
is create-once and atomically published. The producer performs terminal
source and output identity rechecks; on failure it removes only the new
output it owns. An existing output is never overwritten.

The canonical artifact is:

```text
fixtures/research/gemma3_chat_five_expert_distilled_v1/
  train/chat.jsonl
  eval_proxy/chat.jsonl
  token_inventory.jsonl
  manifest.json
  manifest.json.sha256
  build_receipt.json
  build_receipt.json.sha256
```

Each mandatory sidecar is the lowercase payload SHA-256, two spaces, its
basename, and LF. The consumer-frozen manifest binds the two training
partitions. The additive build receipt binds that manifest, both partitions,
the body-free token inventory, generator, config, all schemas, grammar, seed,
ordered read set, count/proof inventories, Gemma identities, request counters,
and terminal TOCTOU results. This preserves the closed consumer schema without
dropping producer provenance.

## 8. Resource and provider accounting

The v1 fixture uses deterministic offline generation:

- provider requests: 0;
- network requests: 0;
- model loads: 0;
- GPU requests: 0;
- protected/Gold/held-out body reads: 0;
- SentencePiece tokenizer loads: one per build/audit process.

If a later version uses a provider, it requires a new authenticated
generation contract and receipt that records provider/model identity,
batches, requests, retries, seeds, checkpoints, and HMAC-bound request and
response hashes. Credentials, headers, environment values, and raw provider
errors must never enter an artifact.

## 9. Reproduction

No model or GPU is needed:

```powershell
py -3.10 scripts/research/build_gemma3_chat_five_expert_qonly_v1.py
py -3.10 scripts/research/audit_gemma3_chat_five_expert_qonly_v1.py
py -3.10 -m pytest -q tests/test_gemma3_chat_five_expert_qonly_v1.py
py -3.10 -m ruff check src/anchor_mvp/research/gemma3_chat_five_expert_qonly_v1.py scripts/research/build_gemma3_chat_five_expert_qonly_v1.py scripts/research/audit_gemma3_chat_five_expert_qonly_v1.py tests/test_gemma3_chat_five_expert_qonly_v1.py
```

The 10,000-row / 100-expert Luna-generated alignment dataset is a later,
separate queue. Its taxonomy, schema, requests, receipts, rows, and hashes
must not appear in or authorize this v1 artifact.
