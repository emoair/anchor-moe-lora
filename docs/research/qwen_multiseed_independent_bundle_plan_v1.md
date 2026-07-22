# Qwen multi-seed controlled-factorial dry-run plan

This additive planner prepares the next low-memory diagnostic without loading a
model, reading dataset bodies, using a GPU, or authorizing training. It first
re-runs the frozen prerequisite/risk consumer and accepts only its `blocked`
decision.

## Replication design

- Five master seeds: `1337`, `7331`, `104729`, `130363`, `20260723`.
- Independent `adapter_init`, `record_order`, and `cuda` seeds use the frozen
  Producer derivation. Deterministic Torch execution is mandatory.
- Checkpoints are `5/10/20/40/80`; step 80 is the only primary endpoint.
- The discovery track repeats the original Q-only, Q+O, and wide arms at
  1,376,256 trainable parameters. Its six throughput-only orders are all
  permutations.
- The mechanism track uses a separate common budget of 1,204,224 parameters:
  Q14, Q7+O7, wide Q4+O3+K6+V6, independently trained O14, and K12+V12.
  It has six pre-registered orders per seed and exact positional balance across
  all five seeds.
- Results cannot be compared across tracks. A retained O branch from a jointly
  trained Q+O checkpoint is never relabelled as independently trained O-only.

Fairness is explicit: 80 steps, LR `5e-5`, fully specified AdamW settings
with `foreach=false` and `fused=false`,
batch/accumulation `1/1`, sequence length 512, BF16+TF32/high, non-reentrant
gradient checkpointing, alpha/rank 2, and fresh base/adapter/optimizer state
with no resume. Deterministic Torch algorithms, deterministic cuDNN,
`cudnn.benchmark=false`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are required.
Each arm also freezes its expected trainable parameter and tensor counts. Run
and artifact keys include `track_id`. Arm orders are only
for separate throughput timing: warmup 1, repetitions 6, CUDA synchronization,
runtime/thermal receipts, one serial GPU job, and a 5 GiB peak-VRAM cap.

## Controlled-factorial confirmation

The future confirmation set remains absent. Its blueprint fixes 60 source
bundles, split by `task_bundle_sha256` into 40 train and 20 eval-proxy bundles
before five roles and two variants are expanded. EN/ZH and five information-flow
strata form ten cells, each with six bundles (`4 train + 2 eval_proxy`).

The three evaluation factors are:

1. old task / new template;
2. new task / old template;
3. new task / new template.

Task, template, and task-template-pair identities use separate common-domain
inventories. Old dimensions require membership; new dimensions require
non-overlap; every pair requires pair-level non-overlap. Global task or template
non-overlap is intentionally **not** claimed. A frozen rotation produces factor
quotas `train=13/14/13`, `eval=7/6/7`; real bundle assignment remains pending.

This secondary controlled-factorial probe cannot satisfy the Producer's
independent-confirmation gate and cannot establish bundle generalization. Both
blocked states are reported separately. These five plumbing files cannot
authorize materialization or training.

## Run

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_multiseed_independent_bundle_plan `
  --config configs/research/qwen_multiseed_independent_bundle_plan_v1.yaml
```

Exit code `2` is expected: it means the authenticated plan is still blocked.
`execution_ready`, `materialization_ready`, `training_authorized`, and
`formal_training_authorized` remain false until the dataset, three inventories,
factor membership/non-overlap proofs, and existing formal gates are present.
The plan binds physical SHA-256 snapshots of its own config and implementation
and rechecks both before returning. The prerequisite/risk consumer is not
imported from Python's module cache: its authenticated source-byte snapshot is
loaded in isolation and the evaluator from that exact snapshot is called.
