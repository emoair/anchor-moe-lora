# Qwen2.5-1.5B synthetic scaffold q_only diagnostic

This is a small local diagnostic, not formal training and not an A–F result. It
trains a rank-4 LoRA only on the 28 language-model `q_proj` modules, using the
100-record `synthetic_nl_scaffold_diagnostic_v1` fixture. It never reads Gold,
heldout, SWE-bench bodies, or provider output.

The runner is intentionally separate from the frozen one-step Qwen diagnostic.
It reuses that implementation's authenticated local Hugging Face identity,
private hardlink snapshot, frozen-base hash, strict q_proj scope, adapter
effect, and save/reload gates without changing the frozen contract.

## Fixed envelope

- Qwen2.5-1.5B-Instruct local Hugging Face checkpoint only; GGUF is rejected.
- `q_proj`, rank 4, alpha 8, batch 1, sequence length 512.
- BF16 compute, TF32 enabled, non-reentrant gradient checkpointing.
- Exactly 2 or 20 optimizer steps. The default is the 2-step smoke.
- 80 synthetic train records and 20 `eval_proxy` records.
- Pre- and post-training eval-proxy loss is reported. `eval_proxy` is not
  heldout and is not a quality result.
- Every complete chat-templated input/target must fit in 512 tokens. Any full
  view or target truncation fails before a model is loaded.
- Outputs are isolated by step count and are never overwritten.
- `training_authorized=false` and `formal=false` remain fixed. `--execute`
  records only `diagnostic_execution_user_requested=true`.

## Commands

Run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
$env:ANCHOR_QWEN25_15B_HF_PATH = 'D:\LLM\models\qwen2.5-1.5b-instruct-hf'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

conda run -n anchor-mvp python -m anchor_mvp.training.qwen_synthetic_scaffold_diagnostic `
  --config configs/training/qwen2_5_1_5b_synthetic_scaffold_qonly_v1.yaml `
  --max-steps 2 --dry-run `
  --preflight-output artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_qonly_preflight/step2-v2
```

The command prints the physical preflight receipt SHA-256. After reviewing the
receipt and local GPU headroom, pass that exact value to the explicitly
requested 2-step diagnostic:

```powershell
conda run -n anchor-mvp python -m anchor_mvp.training.qwen_synthetic_scaffold_diagnostic `
  --config configs/training/qwen2_5_1_5b_synthetic_scaffold_qonly_v1.yaml `
  --max-steps 2 --execute `
  --preflight-receipt artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_qonly_preflight/step2-v2/preflight.json `
  --preflight-receipt-sha256 <SHA256_PRINTED_BY_DRY_RUN>
```

Only after the 2-step artifact passes all gates should the separate 20-step
diagnostic be run by replacing `2` with `20`. It writes a different output
directory and does not resume from or overwrite the 2-step adapter.
Use a fresh `step20-v2` preflight directory for that run. Older receipt
identities are superseded and are never overwritten or silently reused.

## What dry-run does

Dry-run authenticates the fixture manifest, mandatory sidecar, schemas, and all
four partitions from single byte snapshots. It validates all 100 records,
confirms train/eval-proxy source-bundle disjointness, loads only the local
tokenizer to measure exact chat-templated lengths, and authenticates the local
HF checkpoint bytes/Git identity. It does not instantiate model weights, use a
GPU, contact a provider, or access the network.

Dry-run atomically publishes a versioned `preflight.json` and strict
`preflight.json.sha256` under the explicit output directory. The directory
must not already exist. Execute authenticates both files from byte snapshots,
recomputes the complete tokenizer-only preflight, and requires exact equality
before it can enter the model-loading path.
The receipt also binds per-record SHA-256 digests of the prompt IDs, full IDs,
and masked labels in sorted record-ID order, plus the exact Transformers and
Tokenizers runtime versions; raw token IDs are never emitted.

The final adapter directory similarly binds `adapter_config.json`,
`adapter_model.safetensors`, and `diagnostic_receipt.json`; the receipt has a
strict SHA-256 sidecar and records the authenticated preflight SHA, token-length
report, runner implementation SHA, and exact ordered train-record inventory.

## Claims boundary

Loss deltas from 20 synthetic eval-proxy records are proxy telemetry only. They
do not establish generalization, shared-KV correctness, physical KV reuse,
formal readiness, numerical equivalence, or a comparison against other LoRA
profiles.
