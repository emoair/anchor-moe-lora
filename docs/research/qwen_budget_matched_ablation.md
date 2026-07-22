# Qwen2.5-1.5B Budget-Matched LoRA Ablation (Diagnostic)

This experiment compares three LoRA projection scopes over the same frozen
base, 100-record synthetic fixture, fixed 80-record training order, and 80
optimizer steps. It is `controlled_proxy_only`, not formal-v3, held-out, or a
model-quality claim.

## Controlled budget

| Arm | Projection ranks | Trainable parameters | LoRA tensors |
|---|---|---:|---:|
| Q-only | `q=16` | 1,376,256 | 56 |
| Q+O | `q=8, o=8` | 1,376,256 | 112 |
| Wide | `q=5, o=4, k=6, v=6` | 1,376,256 | 224 |

All modules use `alpha/r=2`, dropout 0, BF16+TF32, sequence length 512,
batch size 1, gradient accumulation 1, learning rate 5e-5, and seed 1337.
Each of the 80 training records is consumed exactly once. All arms bind the
same ordered record-ID and tokenized-view digests.

## Results (2026-07-23)

The common eval-proxy macro-loss baseline is 3.102086. Eval-proxy contains
only two independent source bundles, so these are descriptive results without
significance claims.

| Arm | Final loss | Relative change | PPL | Train throughput | Train time |
|---|---:|---:|---:|---:|---:|
| Q-only | 2.089805 | -32.63% | 8.103 | 902.9 tok/s | 28.0 s |
| Q+O | **0.858610** | **-72.32%** | **2.386** | 814.5 tok/s | 31.0 s |
| Wide | 1.185087 | -61.80% | 3.288 | 621.9 tok/s | 40.6 s |

Both eval bundles improve in every arm. Base hashes remain identical before
training, after training, and after fresh reload. Adapter save/reload maximum
logit deltas are zero, and every LoRA tensor observes a finite non-zero
gradient. Peak allocated VRAM is 3,894,284,288 bytes (about 3.63 GiB) for all
three arms.

The offline aggregate audit is stored at
`artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_budget_matched_comparison_v1/comparison.json`
with SHA-256
`920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45`.
It reopens every final artifact directory, validates all saved safetensors
scopes/shapes/budgets, and binds an explicit eval token-view digest. CUDA
deterministic algorithms were not enabled and only one seed was run; both
limitations are machine-readable false claims in the comparison manifest.

The narrow conclusion is that Q+O is substantially more sample-efficient than
Q-only on this small synthetic short-horizon proxy. Wide remains better than
Q-only but trails Q+O at the same parameter budget. This supports continued
asymmetric-projection research; it neither proves generalization nor rejects
Q-Hijack for long-context role isolation or physical KV-reuse objectives.

## Execution

Create a versioned immutable preflight for each profile, then execute with its
explicit SHA:

```powershell
$env:PYTHONPATH = "src"
$env:ANCHOR_QWEN25_15B_HF_PATH = "D:\LLM\models\qwen2.5-1.5b-instruct-hf"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python -m anchor_mvp.training.qwen_budget_matched_ablation `
  --config configs/training/qwen2_5_1_5b_synthetic_scaffold_budget_matched_v1.yaml `
  --profile q_only --dry-run `
  --preflight-output artifacts/diagnostics/<new-preflight-dir>
```

Then use `--execute --preflight-receipt <preflight.json>
--preflight-receipt-sha256 <sha256>`. Outputs are non-overwriting; a rerun
requires a new versioned config/output namespace.

## Formal boundary

Formal-v3 still lacks its frozen snapshot, final projector, generic execution,
source-disjoint, and release lock (0/5). While any gate is missing,
`training_authorized=false`; these artifacts cannot be promoted to a formal
model or formal evaluation result.
