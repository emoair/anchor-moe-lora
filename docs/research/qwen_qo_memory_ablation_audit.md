# Q/O memory-contribution audit (diagnostic v1)

This read-only audit asks a narrow question about the trained equal-budget
`q_plus_o` adapter: how much of its teacher-forced fit is retained by the Q and
O LoRA branches separately?

The runner loads the authenticated Q+O step-80 adapter once and evaluates four
in-memory views:

1. `full`: Q and O deltas enabled;
2. `q_only_contribution`: O LoRA scaling is temporarily zero;
3. `o_only_contribution`: Q LoRA scaling is temporarily zero;
4. `adapter_off`: the PEFT adapter is disabled.

Scaling is restored after every view. Base and adapter tensor hashes are checked
before and after the audit. No adapter is merged or rewritten.

Each view covers all 80 synthetic training records, all 20 `eval_proxy`
records, and 20 deterministic OOD proxy records from four new source bundles.
The OOD generator is independent of the training targets; exact source-bundle
and body-hash overlap is rejected. Published receipts contain only hashes and
aggregate macro/micro teacher-forced losses. They never contain prompt, target,
held-out, or generated-answer bodies.

The generalization gap is descriptive only:

`eval_proxy_loss - train_loss`

No p-value, formal quality claim, held-out claim, or causal memory claim is
authorized by this diagnostic.

## Reproduced result

The read-only run completed in 70.2 seconds with 3.45 GiB peak allocated VRAM.
Base and adapter tensor hashes were identical before and after evaluation.

| View | train macro loss | same-template eval proxy | new-bundle OOD proxy |
|---|---:|---:|---:|
| adapter off | 3.0453 | 3.1021 | 3.0353 |
| Q only | 2.6004 | 2.6767 | 2.9832 |
| O only | 1.0231 | 1.0299 | 2.8666 |
| full Q+O | 0.8487 | 0.8586 | 2.8968 |

O-only accounts for 92.36% of the full same-template loss reduction, while
its OOD reduction is only 5.56%. Full Q+O reduces same-template loss by
72.32%, but OOD loss by only 4.56%. This is strong evidence of template/answer-
shape writeback in this diagnostic dataset, not evidence of broad task
generalization. The authenticated receipt is
[`results/qwen_qo_memory_ablation_audit_v1_receipt.json`](results/qwen_qo_memory_ablation_audit_v1_receipt.json).

Dry-run (no model/GPU):

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_qo_memory_ablation_audit `
  --config configs/research/qwen_qo_memory_ablation_audit_v1.yaml `
  --dry-run
```

GPU execution, only after review:

```powershell
$env:PYTHONPATH = "src"
$env:ANCHOR_QWEN25_15B_HF_PATH = "D:\LLM\models\qwen2.5-1.5b-instruct-hf"
python -m anchor_mvp.research.qwen_qo_memory_ablation_audit `
  --config configs/research/qwen_qo_memory_ablation_audit_v1.yaml `
  --execute
```

The output directory is atomically published without replacement at
`artifacts/diagnostics/qwen2_5_1_5b_qo_memory_ablation_audit_v1`.
