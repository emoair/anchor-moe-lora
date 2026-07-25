# Gemma 3 Chat FINAL Producer Git Provenance Overlay v1

## Outcome

This overlay attaches immutable Producer Git provenance to the already
completed Gemma 3 1B IT five-expert chat diagnostic. It does not modify the
training config, training receipt, adapters, evaluation receipt, or any
dataset file. It does not rerun training or evaluation.

The frozen Producer identity is:

- repository: `anchor-moe-lora-gemma3-chat-v1`;
- commit: `5a8343baa117f8ec2340debaa3fbc727ee993da2`;
- tree: `a9b87ad36773c31506118c13250af0a83fd5752d`;
- parent: `8ecef30d29d4c4a2e990ebc543dab21a6e3c33b1`;
- parent tree: `c19e2b485c6d7b062ca4fa00f3eaaa3bb8d146f2`;
- exact diff: 24 paths, comprising 23 additions and one modification.

The audit authenticates the commit object, lineage, exact diff, tree entries,
Git blob object IDs, blob sizes, and blob SHA-256 values. It then proves that
the 16 raw files actually consumed by the Consumer are byte-identical to the
corresponding Producer blobs.

## Scope and non-authority

The claim scope is
`additive_posthoc_git_provenance_non_authorizing`.

The attestation is diagnostic metadata only. In particular, it:

- does not change the legacy training identity;
- does not retroactively authorize training;
- does not authorize formal training or release;
- does not claim quality, source disjointness, or physical KV reuse;
- does not rerun training or evaluation;
- does not start the queued 10k/Luna job.

All `training_authorized`, `formal_training_authorized`, and `formal` fields
remain `false`.

## What is authenticated

### Producer Git

All 24 changed paths are enumerated in
`configs/research/gemma3_chat_final_producer_git_overlay_v1.json`. For every
path, the audit verifies:

1. the exact `A` or `M` diff status;
2. the `100644` tree mode and blob SHA-1;
3. the Git blob object hash preimage;
4. exact byte count and SHA-256;
5. parent absence for additions, or the exact parent blob for the modified
   `.gitattributes`.

The local branch and remote-tracking refs are observations only. Their values
are recorded at attestation time, but they are deliberately not immutable
gates. A later branch advance cannot invalidate the already-authenticated
commit/tree/blob identity. The Producer current checkout and worktree
cleanliness are irrelevant.

### Sixteen runtime copies

The sixteen Consumer files cover:

- Producer config and schemas;
- closed grammar;
- Producer implementation;
- manifest and mandatory sidecar;
- build receipt and mandatory sidecar;
- token inventory;
- train and eval-proxy partitions.

The JSONL and token-inventory files are handled only as opaque raw bytes. The
overlay does not parse records, inspect sample text, or emit token IDs.

### Existing diagnostic results

The legacy identities remain unchanged:

- training config SHA-256:
  `35dc9effc713730274937352c0f828c881939e503b6725225da7cb54604d9159`;
- Consumer training implementation SHA-256:
  `55a7c45eb8f4b6c590938a486b65f675920d39d46f94686685a04e08f3bde802`;
- training receipt SHA-256:
  `5988d67dc092a003a550dab42751791c8578dcca56bfcf49d7474a8eccb89a3a`;
- training receipt sidecar physical SHA-256:
  `cb1b5fe5675bb36511c5940ff7c5675b71710fef8d66ccb2719e685746a8c600`;
- evaluation receipt SHA-256:
  `ea3ca5e12d7af9bd3e40a0a81bd32dc74060cb051e319377f9e55b3b8c86956b`;
- evaluation sidecar physical SHA-256:
  `75a9d663aba21a9f77760d2937db3d854d2316ba508360b3eb8f7d9398879c0c`.

The audit also verifies ten phase receipts, ten mandatory phase sidecars, five
adapter configs, and five complete adapter safetensor files. Safetensor files
are streamed only for hashing; they are never loaded as models.

## TOCTOU and Git hardening

The audit:

- removes every inherited `GIT_*` environment variable;
- forces `GIT_NO_REPLACE_OBJECTS=1` and `GIT_NO_LAZY_FETCH=1`;
- rejects non-empty grafts and all replace refs;
- rejects symlinks and Windows reparse points;
- binds bytes, file stats, and file identity from one open handle;
- repeats the complete immutable Git authentication at the end;
- re-reads and re-hashes every Consumer input and adapter before publication.

No network command is used. A local remote-tracking ref is not described as a
live network check.

## Run

From the consumer repository root:

```powershell
$python = "C:\path\to\python.exe"
& $python scripts/research/audit_gemma3_chat_final_producer_git_overlay_v1.py `
  --producer-repo C:\path\to\anchor-moe-lora-gemma3-chat-v1 `
  --output-dir artifacts/diagnostics/gemma3_chat_final_producer_git_overlay_v1/producer-final-5a8343b
```

The explicit output directory must not already exist. The auditor writes an
atomic directory containing:

- `attestation.json`;
- `attestation.json.sha256`.

The sidecar format is exactly:

```text
<lowercase SHA-256><two spaces>attestation.json<LF>
```

The command returns exit code `0` only after the immutable Git identity,
runtime copies, legacy receipts, and adapters pass. Any mismatch returns exit
code `2`, a stable metadata-only error code, and all authorization fields
remain false.

## Tests

```powershell
$python = "C:\path\to\python.exe"
& $python -m pytest -q tests/test_gemma3_chat_final_producer_git_overlay_v1.py
```

The focused suite includes the real end-to-end audit plus negative tests for
config/claim drift, Producer/runtime mismatch, strict sidecars, duplicate or
non-finite metadata JSON, external schema references, path traversal, Git
replace refs, grafts, small-file and streamed-adapter TOCTOU, and output
overwrite attempts.

## Remaining work outside this overlay

This overlay intentionally does not improve the weak tool-call/review scores,
materialize the requested rebalanced dataset, train the future emotion-router
LoRA, implement physical Q-independent KV sharing, or start the 10k/Luna
queue. Those are separate additive experiments and remain pending.
