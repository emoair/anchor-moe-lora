# Formal-v2 low-memory training

`formal_v2_lowmem` keeps the frozen formal-v1 dataset snapshot, seed, NF4 base,
BF16 compute, TF32, sequence length 64, and sample exposure budget while replacing
the Trainer/TRL path with the smaller `manual_active_labels_v2` loop.

The completed 2026-07-11 two-step CUDA evidence and loader paging caveat are recorded
in [`formal_v2_probe_20260711.md`](formal_v2_probe_20260711.md).

## Promotion sequence

1. Reuse the executed formal-v1 one-step `smoke-gate` only when the gate verifies the
   identical base revision and frozen dataset snapshot; otherwise rerun it.
2. Run `run_formal_v2_lowmem.ps1 -Arm probe`; this uses `stage=train` for two optimizer
   steps and therefore exercises the actual multi-step formal-v2 branch.
3. Only promote B/C/D/E/F after the probe reports peak allocated and reserved VRAM at or
   below the hard 9 GiB limit.

Dry-run does not measure GPU memory. A passing dry-run must never be reported as a
passing GPU probe.

The matrix is A=native Q4, B=one mixed rank-16 LoRA, C=five full rank-16
specialists, D=manual `3/3/4/3/3`, E=calibration-adaptive Pareto ranks up to 16 per
stage without a B-sized total constraint, and F=the same adaptive mechanism under
exact B-sized rank/parameter constraints. E/F cannot enter held-out evaluation until
their calibration allocation manifests are frozen.

## 12 GB downgrade ladder

The preferred 2-step probe is rank 16, sequence 64, `q_proj/v_proj`, paged AdamW
8-bit, gradient checkpointing enabled, and strict single-GPU placement. If preflight
or the real peak guard fails, retry only as a clearly labeled feasibility probe:

The probe-only host-memory gate is 11GiB because the live machine had 11.46GiB free,
only 0.54GiB below the 12GiB formal gate. B/C/D/E/F retain the 12GiB requirement.

1. sequence 64 -> 48 -> 32;
2. rank 16 -> 8 -> 4;
3. keep `q_proj/v_proj`; only then try `q_proj` alone;
4. retain `paged_adamw_8bit`; use torch AdamW only if bitsandbytes itself fails;
5. retain gradient checkpointing throughout;
6. stop rather than enable CPU/disk offload or device-map paging.

Optimizer offload, base-model offload, and Windows shared-GPU-memory paging invalidate
latency/VRAM evidence and are forbidden, not downgrade options.

Before any formal long run, stop all other CUDA/compute workloads and avoid launching
GPU-heavy browser, broadcast, overlay, or inference tasks in parallel. The completed
probe proves only that the steady-state training window stays below 9GiB; model loading
still touched the physical VRAM ceiling and zero WDDM/OcuLink paging was not proven.
If loading pressure persists, first close optional GPU applications, then keep sequence
64 but lower rank 16 -> 8 -> 4, and only after that apply the sequence/target-module
feasibility ladder above. Do not hide loading pressure with CPU/disk offload.

## Safety checkpoints

The manual loop publishes a directory atomically at each `save_steps` boundary.
Each directory contains PEFT adapter weights plus `safety_checkpoint.json`.
These are crash-salvage checkpoints with `adapter_weights_warm_start_only` capability.

They do **not** contain optimizer, scheduler, scaler, or RNG state. Consequently they
must not be described as an exact continuation: recovery loads the adapter weights,
then explicitly restarts optimizer, scheduler, and sample scheduling. The long B run
saves at steps 25, 50, and 75 so an interrupted run retains useful adapter weights.
