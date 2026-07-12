# Distillation-to-training handoff

This entry point makes the quota-exhaustion transition fail closed. Its default is an
offline dry-run: it does not prompt for a key, start OpenCode, call a teacher API, or
start a GPU job.

```powershell
.\scripts\run_distill_train_handoff.ps1
```

The PowerShell entry uses the repository source tree directly, so an editable package
install is not required.

The status is written atomically to
`runs/distill-train-handoff/status.json`; the append-only event stream is beside it.
Repeating the same config resumes the same state. Changing the config hash requires a
new `state_dir`, so a resume cannot silently change gates or datasets.

## Live sequence

```powershell
.\scripts\run_distill_train_handoff.ps1 -ConfirmLive -ConfirmTraining
```

`-ConfirmLive` is required before any quota-consuming child can run. The parent prompts
with masked input for each configured credential environment name. A key is never read
from a CLI option or config file and is passed only in the child environment. OpenCode
and teacher output is reduced to an exit code and hashes; it is not copied into the
handoff log. The training child receives neither teacher credential.

The order is fail-closed:

1. Run the patched OpenCode concurrency-1 execution gate.
2. Convert only the configured controlled raw session export. The accepted gold record
   and session candidate must share a sample ID, and tool calls must have complete,
   matching tool results.
3. Open only the remaining operator-configured positive-integer stages. A failed stage
   stops the requested sequence.
4. Run the existing general teacher automation for its configured quota epoch.
5. Only `provider_quota_exhausted` can trigger the default handoff. Temporary 429,
   cooldown, network/client deadline, 400/499, local request budget, and arbitrary
   failures do not trigger training. `automation_complete` is an explicit alternative
   config trigger, not an inferred HTTP outcome.
6. Validate all five curated expert JSONLs: canonical schema, minimum count per expert,
   global ID uniqueness, secret scan, held-out leakage, accepted execution gold, session
   conversion, and completed concurrency ramp. Quota exhaustion alone never means the
   data is sufficient.
7. Freeze the dataset SHA-256 bindings into `anchor.training-handoff.v1` plus a SHA
   sidecar. Every training job re-verifies this freeze.
8. Run the formal-v2 preflight and then one LoRA at a time. The profile must use
   `manual_active_labels_v2`, batch size 1, gradient checkpointing, and a configured
   maximum steady training peak of at most 9 GiB. A lock forbids parallel handoff-owned
   GPU jobs. Per-adapter logs and the trainer's normal progress/checkpoints remain under
   ignored runtime directories.

## Controlled session prerequisite

The stage-1 OpenCode harness must populate the `raw_export`, `capture`, and `workspace`
paths in the handoff config. If the patched build cannot yet produce that controlled
export, the coordinator stops at `execution_conversion_blocked`; it does not fall back
to the global OpenCode database or arbitrary historical chats.

The sample config currently points at the previously frozen formal-v1 dataset only so
the offline gate can be exercised. Before a new real run, curate the stopped automation
outputs into immutable per-expert JSONLs, update both the snapshot paths and formal-v2
training config to the same files, and use a new `state_dir`.
