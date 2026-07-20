# Formal-v3 full-scale A-F training

Formal-v3 is the training contract for the full public SWE-bench train bank.
It does not authorize a run by itself. It accepts only an immutable execution-
Gold snapshot and never trains directly from a growing distillation directory.

## Scale and split contract

- Source population: 19,008 candidate tasks.
- Five stage work orders per task: 95,040 total candidates.
- A task enters the snapshot only after its five stage records pass the real
  execution Gold gate. Rejected tasks are not backfilled with synthetic rows.
- `minimum_records_per_expert=256` is a quality floor, not a scale ceiling.
  The frozen accepted population may contain up to 19,008 Gold rows per stage.
- The source-bank split is applied before Gold selection:
  - `train`: at most 17,105 tasks; training only;
  - `validation-from-train`: at most 1,903 tasks; E/F rank calibration only;
  - external heldout: evaluation only, represented in the training snapshot by
    hashes and leak-audit metadata. Its body is neither copied nor read.
- The immutable manifest binds the actual accepted counts, balanced five-stage
  record counts, train/calibration ID-set hashes, pairwise-disjoint proof,
  calibration files, external-heldout manifest hash, and leakage-audit hash.

`prepare_full_v3_snapshot.py` creates this contract directly from the pinned
source-bank manifest and allowlists configured in `full_v3_snapshot.yaml`; it
is not a hand-written post-processing manifest.

The training preflight fails closed if train and calibration overlap, if a
heldout body is present/read/emitted, if accepted Gold exceeds 19,008, or if the
five train stages are not aligned to the same complete-chain count.

## Snapshot-sized exposure control

Inherited checked-in `max_steps` values are compatibility placeholders for the
older partial-data profiles. They are not the formal schedule, and a direct
B-F adapter run without materialization is rejected. Before B-F can run,
`scripts/train/materialize_formal_v3_schedule.py` verifies the frozen split and
writes a snapshot-bound config under:

```text
artifacts/formal_v3/schedules/<snapshot_sha256>/<arm>.json
```

The formal comparison uses one train epoch:

- B: one rank-16 mixed adapter sees all five train-Gold stage files.
- C/D/E/F: five independent adapters; each sees only its own stage file.

The shared low-memory control uses a conservative learning rate of `5e-5`,
`constant_with_warmup`, a `0.03` warmup ratio, and exactly this one
snapshot-derived epoch. The same optimization hyperparameters apply to B-F;
there is no arm-specific tuning. This replaces the earlier `2e-4` exploratory
setting to reduce small-snapshot overfitting risk, but it is still a controlled
baseline rather than a claim of optimal convergence.
- B-F have exactly the same total and per-stage sample exposure.
- With accumulation `g=4` and `N` accepted train chains, each stage uses
  `ceil(N/g)` optimizer steps. If `N` is not divisible by four, the runtime
  independently shuffles and pads each stage, then deterministically
  interleaves the five strata. Padding is at most three samples per stage.
  Planned and observed per-file exposures are both recorded; aggregate mixed
  shuffle is not accepted as proof of per-stage equality.

For the maximum canonical source split (`N=17,105`), a specialist job has
4,277 optimizer steps and 17,108 exposures; B has 21,385 optimizer steps. Every
B-F arm therefore has 85,540 group-level exposures. The historic 640-exposure
experiment is not a formal-v3 limit.

Every run manifest records requested/padded exposures, padding per stage,
derived optimizer steps, target epochs, snapshot hashes, and the B-F equality
invariant. Four approximately even safety checkpoints are derived from the
resolved step count.

## Low-memory sequence boundary

`formal_v3_lowmem_*` is a controlled 3080 Ti / 9 GiB profile, not a
full-trajectory training claim. It uses a 64-token window and the explicit
`formal_v3_lowmem_truncated_v1` contract: recent prompt context plus the start
of the assistant completion are retained. `full_trajectory_training=false` is
written into the config and manifest.

After an executed run, the manifest and checkpoint metadata are updated with
runtime-observed rendered-token maximum/mean, selected-token maximum/mean, and
the exact truncated-exposure count/fraction. Those statistics describe sample
exposures (including deterministic padding), not just unique rows. A future
cloud/full-context profile must use a separately audited no-truncation contract;
these low-memory results must never be relabelled as full-context training.

## A-F definitions

| Arm | Structure | Rank / budget | Training exposure |
| --- | --- | --- | --- |
| A | Frozen native Q4 base | no LoRA | none; preflight/evaluation baseline only |
| B | One mixed LoRA | rank 16; 10,387,456 parameters | all five train stages |
| C | Five routed experts | rank 16 each; rank sum 80 | one stage per expert |
| D | Five fixed small experts | `3/3/4/3/3`; rank sum 16, budget matched to B | one stage per expert |
| E | Calibration-adaptive experts | each expert <=16; no total-rank cap | one stage per expert |
| F | Same adaptive mechanism as E | rank sum 16 and parameters exactly matched to B | one stage per expert |

E and F require an immutable `anchor.lora-allocation.v1` plus SHA-256 sidecar.
It must be produced only from the calibration split and frozen before heldout
access. E must be non-uniform. F must have rank sum 16 and exactly 10,387,456
materialized trainable parameters. Both use the same attempted-allocation and
selection mechanism. Formal-v3 explicitly rejects the historical
`heuristic_preregistered_calibration_pending` manifests: `selection_status`
must be `calibration_selected_frozen`, calibration metrics must be present for
every attempted allocation, and the selected rank signature must be one of
those measured attempts.

## Live Gold bridge

The snapshot source is the authenticated full-bank coordinator export, not
`data/automated_v3` and not any older synthetic partition. After a terminal
live run, publish the strict Gold projection with:

```powershell
py -3.10 scripts/data/export_swebench_formal_gold.py
py -3.10 scripts/data/prepare_full_v3_snapshot.py --config configs/orchestration/full_v3_snapshot.yaml
```

The exporter reads the protocol-separated root-owned WSL train-receipt key only
in-process. Every accepted task must have five hash-bound stage artifacts, one
final review PASS, one security PASS, a matching final patch, at least one
successful nontrivial `anchor-validate` command with a model-visible real tool
result, successful sandbox cleanup, and an authenticated
`real_sandbox_self_verified` receipt. This train evidence explicitly sets
`not_official_swebench_pass=true`; official heldout evaluation remains separate.
The consumer independently re-hashes the publication manifest and every matched
candidate task/work-order shard, requires the same manifest hash in the run
manifest and status, reparses the exact terminal validator JSON from the
model-visible OpenCode export, and binds that result to the immutable train
sandbox image digest/ID, validator source hash, final patch, tool transcript,
five-stage lineage, and post-cleanup supervisor HMAC. A receipt signed for a
different shard, image, validator, patch, or validation state is not Gold.
A capped `stopped_checkpoint_resumable` checkpoint may be exported so completed
tasks remain usable, while incomplete or unverified tasks stay excluded and
retryable. The training projection retains real builder tool calls/results,
the sanitized OpenCode session export, and the exact workspace diff, while
removing explicit hidden-reasoning fields. No model-supplied verdict can replace
the supervisor receipt.

## Formal-v3 evaluation binding

Every materialized B–F training manifest carries
`anchor.formal-v3-af-evaluation.v1`: the same hash-only heldout binding,
normalization to `A=100`, A as the frozen Q4 baseline, B as one mixed adapter,
and C/D/E/F as five-stage serial runtime-LoRA hot swaps. Evaluation artifacts
must be isolated by formal-v3 arm and version; formal-v2 adapters, registries,
configs, and reports are forbidden inputs.

The checked-in formal-v3 evaluation control is
`configs/benchmark/formal_v3_af_control.json`. Its independent finalizer binds
the snapshot manifest **and its sidecar**, the common NF4/Q4 base inventory,
all B--F materialized schedules, execute manifests, checkpoint metadata,
progress state, adapter weights, frozen E/F calibration allocations, and only
the external-heldout metadata hashes. It rejects every `formal-v2` source.

The finalizer creates a new immutable version bundle under
`artifacts/formal_v3/evaluation/registries/<version>/`; it never overwrites an
existing bundle. The offline preflight does not open heldout case or fixture
bodies. Runtime outputs are restricted to
`runs/formal-v3/evaluation/<version>/...`; exact resume is limited to that same
version, registry, heldout hash, backend identity, sampling contract, and
checkpoint.

```powershell
# Read-only readiness inspection. Current checkout: BLOCKED until formal-v3
# snapshot, calibration allocations and completed B--F artifacts exist.
python scripts/benchmark/materialize_formal_v3_af.py `
  --version-id formal-v3-001

# Exclusively create the immutable registry/benchmark after training completes.
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 -Finalize

# Offline preflight only; no heldout body, API or GPU evaluation.
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001

# The only live form. It uses the same Q4 base, A=100, B's single mixed
# adapter and serial runtime-LoRA swaps for C/D/E/F.
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 -Execute -AuthorizeHeldoutAccess
```

No formal-v3 heldout evaluation has been executed or claimed. With the current
missing training artifacts the readiness command returns `BLOCKED` before any
heldout body is opened and before any API/GPU action.

## Read-only preflight and launch

```powershell
# Validates the snapshot and materializes B-F schedules. No API or GPU training.
.\scripts\train\formal_v3_preflight.ps1

# A is explicit and never creates a LoRA job, even with -Execute.
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm A

# Resource gates on the same frozen snapshot.
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm smoke -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm probe -Execute

# Formal matrix. Only the explicit -Execute form starts a GPU job.
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm B -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm C -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm D -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm E -AllocationManifest <E.json> -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm F -AllocationManifest <F.json> -Execute
```

Without `-Execute`, the launcher performs the same data/base checks and adapter
dry-run only. A process lock preserves single-GPU ownership. Completed experts
are skipped only when config, snapshot, final adapter, and checkpoint metadata
all match. Intra-expert safety checkpoints remain adapter-weight warm starts,
not exact optimizer/scheduler/RNG resume.
