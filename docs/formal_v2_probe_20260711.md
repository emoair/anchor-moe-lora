# Formal-v2 two-step GPU probe — 2026-07-11

## Verdict

The rank-16, sequence-64 `manual_active_labels_v2` profile completed two real CUDA
optimizer steps and produced two safety checkpoints plus the final PEFT adapter.
The training window stayed below the 9GiB gate. This establishes that the local
profile can train; it does **not** authorize a long B/C/D/E/F run.

Strict zero-paging promotion remains unproven. During model loading, PyTorch reported
14,844MiB reserved and zero free while an external `nvidia-smi` sample showed
12,002MiB of 12,288MiB physical VRAM in use. Post-load cleanup reduced the runtime
to 7,686MiB reserved with 3,342MiB free, and the measured training-window peak was
7.433GiB allocated / 7.945GiB reserved. The loader used strict `device_map={"": 0}`
and no explicit CPU/disk offload, but WDDM telemetry cannot prove that the transient
virtual reservation caused no system-level paging. Treat this as a loader warning.

## Environment and preflight

- GPU: NVIDIA GeForce RTX 3080 Ti, 12GiB, CUDA capability 8.6.
- Host: 22.79GiB physical RAM; the first preflight observed 11.46GiB available.
- Runtime: `<conda-env-python>`, Python 3.11.15,
  PyTorch 2.5.1+cu121, CUDA 12.1, BF16 supported.
- Core packages: Transformers 5.13.0, PEFT 0.19.1, TRL 1.8.0,
  bitsandbytes 0.48.2, Accelerate 1.14.0, Datasets 5.0.0.
- The Anaconda `base` environment was not used: it is Python 3.9 and lacks the
  full training dependency set.
- The probe-only host-memory threshold was reduced from 12GiB to 11GiB. Formal
  arms retain 12GiB.
- Training input was only
  `artifacts/formal_v1/dataset/data_frontend.jsonl`: 15 valid `frontend_gen`
  records, SHA-256
  `bfc7877df4f826ce09ed9f3e272f4366d24d696d168e502de5c82a2f3bb19ec3`.
  No session-export/failure records or held-out cases were training inputs.
- The reused formal-v1 smoke gate passed and matched both base revision and
  frozen dataset snapshot `a6cb9a4443c1f73775a1aa815e575a2ae1026b78cc6c4df9a3fb440670c19320`.

## Configuration

- Base: local Gemma 4 12B training-compatible bitsandbytes NF4, frozen.
- LoRA: rank 16, alpha 32, BF16, `q_proj/v_proj`, 10,387,456 trainable parameters.
- Sequence length 64, batch 1, accumulation 1, two optimizer steps.
- Paged AdamW 8-bit, gradient checkpointing, TF32 enabled.
- Single GPU only; no base/optimizer/disk offload.

## Results

| Evidence | Value |
| --- | ---: |
| Step 1 loss | 2.3589446545 |
| Step 2 loss | 2.8752079010 |
| Mean train loss | 2.6170762777 |
| Training-window peak allocated | 7.4328GiB |
| Training-window peak reserved | 7.9453GiB |
| Training window, including checkpoints | 16.449s |
| Total runtime, imports/load/train/save | 40.619s |
| Step-1 optimizer completion from trainer start | 15.204s |
| Step-2 optimizer completion delta | 1.128s |
| Safety checkpoints | Steps 1 and 2, 20,808,216 bytes each |
| Final adapter | Produced, unmerged PEFT adapter |

After exit, no Python training process remained. GPU usage returned to 886MiB with
11,201MiB free, confirming CUDA memory release.

## Evidence files

- Execute manifest:
  `artifacts/formal_v2/probe/manifests/frontend_gen-r16.execute.json`
- Progress ledger:
  `artifacts/formal_v2/probe/adapters/frontend_gen-r16.progress/events.jsonl`
- Final metadata:
  `artifacts/formal_v2/probe/adapters/frontend_gen-r16/checkpoint_metadata.json`
- Step checkpoints:
  `artifacts/formal_v2/probe/adapters/frontend_gen-r16/safety-checkpoints/`

The safety checkpoints contain adapter weights but not optimizer, scheduler, scaler,
or RNG state. They are warm-start salvage points, not exact-resume checkpoints.

## If a later probe fails

Apply the downgrade ladder in `formal_v2_lowmem.md`: sequence 64 -> 48 -> 32,
rank 16 -> 8 -> 4, then `q_proj`-only as a non-comparable last feasibility probe.
Keep gradient checkpointing and paged AdamW 8-bit. Stop instead of enabling CPU/disk
offload or accepting a loader/training window that depends on shared-memory paging.
