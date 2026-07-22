# Qwen2.5-1.5B 等参数 LoRA 对照（诊断）

本实验在同一冻结基座、同一 100 条纯合成数据、同一固定训练顺序和同一
80 optimizer-step 预算下，对比三种 LoRA 投影范围。它是
`controlled_proxy_only`，不是 formal-v3、held-out 或质量结论。

## 对照契约

| 组 | 投影与 rank | 可训练参数 | LoRA 张量 |
|---|---|---:|---:|
| Q-only | `q=16` | 1,376,256 | 56 |
| Q+O | `q=8, o=8` | 1,376,256 | 112 |
| Wide | `q=5, o=4, k=6, v=6` | 1,376,256 | 224 |

所有模块均使用 `alpha/r=2`、dropout=0、BF16+TF32、sequence length 512、
batch size 1、gradient accumulation 1、learning rate 5e-5 和 seed 1337。
80 条训练记录恰好各使用一次，无重复或遗漏；三组共享相同的有序 record-ID
和 tokenized-view 摘要。

## 2026-07-23 结果

共同基线 eval-proxy macro loss 为 3.102086。eval-proxy 只有两个独立 source
bundle，因此只报告描述性结果，不做显著性检验。

| 组 | 训练后 loss | 相对变化 | PPL | 训练吞吐 | 训练耗时 |
|---|---:|---:|---:|---:|---:|
| Q-only | 2.089805 | -32.63% | 8.103 | 902.9 tok/s | 28.0 s |
| Q+O | **0.858610** | **-72.32%** | **2.386** | 814.5 tok/s | 31.0 s |
| Wide | 1.185087 | -61.80% | 3.288 | 621.9 tok/s | 40.6 s |

三组的两个 eval bundle 均改善；基座训练前后与 fresh reload 哈希完全一致，
adapter 保存/重载最大 logit 差均为 0，所有 LoRA 张量都至少观察到一次非零且
有限的梯度。三组峰值 allocated VRAM 均为 3,894,284,288 bytes（约 3.63 GiB）。

离线聚合审计清单位于
`artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_budget_matched_comparison_v1/comparison.json`，
SHA-256 为 `920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45`。
它重新打开最终目录并逐张量验证 safetensors 的 scope、shape 和参数预算，同时
固定 eval token-view 摘要。当前未启用 CUDA deterministic algorithms，且只跑了
一个 seed；这两项在清单中明确为 false。

当前最小结论：在这个小型合成代理集和短程预算上，Q+O 明显优于 Q-only；
把相同预算进一步分散到 K/V 后，Wide 仍优于 Q-only，但不如 Q+O。它支持继续
研究非对称投影，但不能证明真实泛化，也不能否定 Q-Hijack 在更长上下文、
角色隔离或物理 KV 复用目标上的价值。

## 运行

先为每个 profile 生成不可变 preflight，再使用其 SHA 显式执行：

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

随后改用 `--execute --preflight-receipt <preflight.json>
--preflight-receipt-sha256 <sha256>`。输出目录禁止覆盖，因此复跑必须使用新的
版本化配置/输出命名空间。

## 正式训练边界

formal-v3 当前仍缺 frozen snapshot、final projector、generic execution、
source-disjoint 和 release lock（0/5）。任何一项缺失时，
`training_authorized=false`，本实验产物不得晋级为正式模型或正式评测结果。
