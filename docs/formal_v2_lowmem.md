# Formal-v2 low-memory training

`formal_v2_lowmem` keeps the frozen formal-v1 dataset snapshot, seed, NF4 base,
BF16 compute, TF32, sequence length 64, and sample exposure budget while replacing
the Trainer/TRL path with the smaller `manual_active_labels_v2` loop.

## Promotion sequence

1. Run the legacy one-step `smoke-gate`; it proves load, backward, save, and reload.
2. Run `run_formal_v2_lowmem.ps1 -Arm probe`; this uses `stage=train` for two optimizer
   steps and therefore exercises the actual multi-step formal-v2 branch.
3. Only promote B/C/D after the probe reports peak allocated and reserved VRAM at or
   below the hard 9 GiB limit.

Dry-run does not measure GPU memory. A passing dry-run must never be reported as a
passing GPU probe.

## Safety checkpoints

The manual loop publishes a directory atomically at each `save_steps` boundary.
Each directory contains PEFT adapter weights plus `safety_checkpoint.json`.
These are crash-salvage checkpoints with `adapter_weights_warm_start_only` capability.

They do **not** contain optimizer, scheduler, scaler, or RNG state. Consequently they
must not be described as an exact continuation: recovery loads the adapter weights,
then explicitly restarts optimizer, scheduler, and sample scheduling. The long B run
saves at steps 25, 50, and 75 so an interrupted run retains useful adapter weights.
