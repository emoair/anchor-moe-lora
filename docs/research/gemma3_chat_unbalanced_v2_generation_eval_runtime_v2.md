# Gemma 3 unbalanced-v2 physical generation evaluation runtime v2

This runtime is the physical, diagnostic-only successor to the model-free
generation-evaluation contract v1. It does not authorize training, claim
held-out evaluation, select a release winner, or establish generalization.

## Safety boundary

The default command is validation. It reads the versioned config and schemas
only; it does not open evaluation bodies, import Torch/Transformers/PEFT/
bitsandbytes, request a GPU, or use a network.

Historical producer authentication is delegated to
`anchor_mvp.training.gemma3_chat_unbalanced_v2_runtime_source_v2`. The
evaluator never treats the current branch, worktree, remote head, or upstream
as release authority. The shared source gate authenticates the checked-in
sharded-v1 consumer receipt and sidecar, the sharded-v3 producer contract, the
P candidate commit/tree (`c5080249aa103d6cac09f0a55e51982b68524480` /
`f750c0020e7acacdfe04aee968f43e07689dc25e`), the R independent-attestation
commit/tree (`304be2b86f82aad7ddade80fae04528bf2801977` /
`9bf154d481623b696dbc8a4bdf276967710b7214`), read-set 88, the exact 23
payload + 23 sidecar Git blobs, logical identities, and the specific asset
blob OIDs. Evaluation JSONL is opened only through that authenticated
Git-object interface.

This separation intentionally allows the live producer branch to advance
without rewriting historical evidence. A missing historical commit, changed
tree, missing blob, byte/SHA drift, receipt drift, or attestation drift fails
before model load.

The tokenizer path is equally explicit and has a single authority: the shared
runtime-source module returns an authenticated in-memory SentencePiece
capability and its public identity. The evaluator never imports the legacy
tokenizer binding, never calls `AutoTokenizer`, and never reopens a tokenizer
model path. It applies the authenticated in-memory Gemma turn-token overlay,
including token IDs 105 and 106. The adapter-run receipt must bind the same
chat-template policy and consumer `serialization_inventory_sha256`. Every
final receipt proves `processor_source=authenticated_model_proto_bytes`,
`hf_auto_tokenizer_used=false`, and
`second_model_file_path_read=false`.

The closed teacher-v1 handoff is never consumable. The evaluator validates the
complete training receipt against the reviewed v2 training-receipt schema and
rejects any nested teacher identity containing a v1 version. Its config remains
`pending` until the unique reviewed v2 run receipt exists; this evaluator does
not guess or pre-freeze a candidate teacher-binding hash.

## Matrix and fairness

The fixed matrix is:

- tool comparison: base, Q-only, Q+micro-O over 400 records;
- planner comparison: base, Q-only, O-only, Q+micro-O over 240 records;
- router evaluation: planner Q-only over the 20 eval-proxy rows only;
- identity probe: base, correct route, wrong route over 50 rows;
- role/wrong-route proxy: correct route plus two deterministic wrong routes
  over 860 rows, covering humor, serious, angry style, review/audit, and tool
  routing.

For every asset, all arms share an exact record-order digest, serialization
digest, sampling digest, seed 1337, deterministic decoding, and
`max_new_tokens=256`. Any cross-arm lock drift fails closed.

## Physical execution

`--execute` is the only GPU path. Before loading the model it authenticates the
adapter run receipt, all nine adapter config/model files, adapter profiles,
base-model inventory, Q8 SCB inventory, and serialization identity. The
backend loads a local frozen 8-bit base with network access disabled, loads
the nine PEFT adapters as non-trainable, and checks the exact active-adapter
state on every switch. It rechecks frozen parameters, Q8 SCB identity,
base-file identity, termination, and Torch peak allocation/reservation.

All local authentication reads bind the path `lstat`, opened-handle `fstat`
before and after streaming, and final path `lstat` to one object, size,
timestamps, and non-reparse type. Config hashing reuses those authenticated
bytes and never reopens the path. Physical execution is Windows-only and fails
closed if handle leases are unavailable. Four tokenizer files are held under
read-only, deny-write/delete/rename leases while the shared in-memory
tokenizer capability consumes them. Before any ML import or path-based load,
the backend acquires another exact 23 leases: five base files plus config and
model files for each of nine adapters. Those leases remain open across
`safe_open`, base `from_pretrained`, adapter `from_pretrained`/`load_adapter`,
all generations, and the final identity/hash recheck.

Only aggregate counters, rates, latency/token totals, stratified summaries,
fairness locks, wrong-route deltas, and hash-bound source/runtime identities
are persisted. Prompts, targets, completions, record IDs, raw token IDs, and
per-record metrics are forbidden recursively from receipts and progress.
Progress files are immutable scalar arm summaries. Success and failure
receipts use create-once publication and SHA-256 sidecars under a single-GPU
exclusive lock.

The lock and publication paths are identity-bound. The GPU lock is acquired
with exclusive creation, pinned to its open-file object identity, and removed
only if the parent and current path still identify that same object. A
same-content replacement is preserved and the run fails closed. Each run
directory is claimed with an atomic no-replace directory creation. Progress
and sidecars are published with create-once hard links; `receipt.json` is the
last publication marker. Existing or concurrently won paths are never
overwritten, and failed/interrupted run directories and progress evidence are
never recursively deleted.

Owned temporary files and locks are removed on Windows only through
`FileDispositionInfo` on an authenticated delete-capable handle. There is no
path-name `unlink` fallback. If the handle identity cannot be proved, the
platform cannot provide the operation, or cleanup races with replacement, the
replacement and failure evidence are preserved and execution fails closed.

Metrics cover JSON/format structure, exact tool name and arguments, argument
grounding, planner label and topology, Air identity attribution, provider
misattribution absence, route accuracy, wrong-route deltas, refusal, output
bounds, and EOS termination. Aggregates are stratified by dataset stratum,
language, role, identity probe, router split, and wrong-route status.

## Commands

Validation (the default, zero body/model/GPU/network use):

```powershell
python scripts/research/run_gemma3_chat_unbalanced_v2_generation_eval_runtime_v2.py
```

Authenticated metadata dry run (requires a bound adapter receipt):

```powershell
python scripts/research/run_gemma3_chat_unbalanced_v2_generation_eval_runtime_v2.py `
  --dry-run `
  --source-repo <producer-repository> `
  --adapter-receipt <adapter-run-receipt>
```

Physical generation is the same command with `--execute`. It must not be run
concurrently with training or another evaluator.
