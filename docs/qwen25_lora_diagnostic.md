# Qwen2.5-1.5B one-step LoRA diagnostic

This is a mechanical plumbing test, not an A–F result and not a quality claim.
It uses one built-in toy exchange, one optimizer step, batch 1, sequence length
128–256, and a rank-4 or rank-8 LoRA attached only to `q_proj`. It never reads
Gold, heldout, formal-v3, or any project dataset.

Only the pinned, unpacked local Hugging Face checkpoint is accepted. GGUF is
rejected because llama.cpp inference serialization is not a PEFT training base.
The command forces Hugging Face and Transformers offline mode, and every model
load uses `local_files_only=True`.

```powershell
$env:ANCHOR_QWEN25_15B_HF_PATH = 'D:\LLM\models\qwen2.5-1.5b-instruct-hf'
anchor-qwen-diagnostic --config configs/training/qwen2_5_1_5b_lora_one_step_diagnostic.yaml --dry-run
anchor-qwen-diagnostic --config configs/training/qwen2_5_1_5b_lora_one_step_diagnostic.yaml --rank 4 --execute
```

Model ID, ModelScope source URL, Git revision, and the four required artifact
SHA-256 values are pinned in the config. The model worktree must be clean and
all tracked bytes must match the pinned revision. Identity values are not
accepted from environment variables.

Execution succeeds only if all of these gates pass:

- every one of the 28 transformer layers has exactly one trainable `q_proj`
  LoRA A/B pair with the expected shape;
- all trainable parameters, gradients, logits, and adapter deltas are finite,
  and at least one LoRA gradient is nonzero;
- the full frozen-base parameter hash remains unchanged after exactly one step;
- disabling the trained adapter changes the logits;
- a fresh base plus the saved adapter reproduces the trained next-token logits;
- the staged and final adapter files retain identical authenticated hashes.

Config, model, Git metadata, and output ancestors must be physical paths rather
than symlinks, junctions, or other reparse points. Outputs are confined to the
module-anchored `artifacts/diagnostics` directory and are never overwritten.

The loader authenticates a private hardlink snapshot and records the complete
source identity in the receipt. Because hardlinks share underlying inodes, this
is a `trusted_local_storage` mechanical diagnostic and is explicitly **not
release-grade** against a privileged writer mutating the checkpoint in place.
Before validation and reload, the saved adapter config is normalized to the
stable model ID `Qwen/Qwen2.5-1.5B-Instruct`; it never retains the temporary
`.model-snapshot` path or a personal absolute path.

## Local reference verification

A rank-4 run completed on an RTX 3080 Ti on 2026-07-22 in about 72 seconds.
The training loss was `1.47572124`; all 56 LoRA tensors and 344,064 trainable
parameters matched the contract. On the first step, all 28 B matrices had
nonzero gradients while the A matrices remained at zero gradient because B is
zero-initialized, as expected mathematically for the first LoRA step. Disabling
the adapter changed logits by a maximum of `0.603515625`; reloading the saved
adapter on a fresh base reproduced logits with maximum error `0.0`, and all 338
base-tensor hashes remained unchanged. `nvidia-smi` sampled 4,758 MiB of total
GPU memory; this is a mechanical observation, not a performance benchmark. Audit
counters were `provider_requests=0` and `external_dataset_inputs=0`.
