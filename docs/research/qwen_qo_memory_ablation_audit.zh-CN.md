# Q/O 记忆贡献消融审计（diagnostic v1）

这个只读审计只回答一个窄问题：在已经训练好的等参数量 `q_plus_o`
adapter 中，Q 与 O 两条 LoRA 分支各自保留了多少 teacher-forced 拟合效果？

脚本只加载一次经过认证的 Q+O step-80 adapter，并在内存中评估四种视图：

1. `full`：Q、O 增量都启用；
2. `q_only_contribution`：临时把 O LoRA scaling 置零；
3. `o_only_contribution`：临时把 Q LoRA scaling 置零；
4. `adapter_off`：通过 PEFT 上下文关闭整个 adapter。

每种视图结束后都会恢复 scaling；审计前后还会核验基座与 adapter
张量哈希。脚本不合并、不改写 adapter。

每种视图覆盖全部 80 条训练记录、20 条 `eval_proxy`，以及来自 4 个全新
source bundle 的 20 条确定性 OOD proxy。OOD 生成器独立于训练 target；若
source bundle 或正文精确哈希重叠，脚本会拒绝运行。发布 receipt 只包含哈希、
macro/micro teacher-forced loss 与描述性 gap，不包含 prompt、target、heldout
或生成答案正文。

generalization gap 仅按下式做描述：

`eval_proxy_loss - train_loss`

本审计不声明 p-value、正式质量、heldout 结果或因果“记忆”结论。

## 本地复现实测

只读实测耗时 70.2 秒，峰值 allocated 显存 3.45 GiB；评估前后基座与
adapter 张量哈希完全一致。

| 视图 | train macro loss | 同模板 eval proxy | 新 bundle OOD proxy |
|---|---:|---:|---:|
| adapter off | 3.0453 | 3.1021 | 3.0353 |
| Q only | 2.6004 | 2.6767 | 2.9832 |
| O only | 1.0231 | 1.0299 | 2.8666 |
| full Q+O | 0.8487 | 0.8586 | 2.8968 |

O-only 已解释完整同模板 loss 降幅的 92.36%，但在 OOD 上只下降 5.56%；
完整 Q+O 的同模板降幅是 72.32%，OOD 降幅却只有 4.56%。这对当前诊断集
而言是很强的“模板/答案形状写回”信号，不是广泛任务泛化的证据。认证后的
receipt 见
[`results/qwen_qo_memory_ablation_audit_v1_receipt.json`](results/qwen_qo_memory_ablation_audit_v1_receipt.json)。

无模型/GPU 预检：

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_qo_memory_ablation_audit `
  --config configs/research/qwen_qo_memory_ablation_audit_v1.yaml `
  --dry-run
```

审查通过后才执行 GPU 测试：

```powershell
$env:PYTHONPATH = "src"
$env:ANCHOR_QWEN25_15B_HF_PATH = "D:\LLM\models\qwen2.5-1.5b-instruct-hf"
python -m anchor_mvp.research.qwen_qo_memory_ablation_audit `
  --config configs/research/qwen_qo_memory_ablation_audit_v1.yaml `
  --execute
```

输出会以 atomic no-replace 方式发布到
`artifacts/diagnostics/qwen2_5_1_5b_qo_memory_ablation_audit_v1`。
