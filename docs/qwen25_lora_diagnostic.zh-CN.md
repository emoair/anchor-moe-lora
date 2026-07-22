# Qwen2.5-1.5B 单步 LoRA 诊断

这是机械管线诊断，不属于 A–F 正式实验，也不形成质量结论。入口只使用
一条内置玩具对话、一步优化、batch 1、128–256 序列长度，并且只给
`q_proj` 挂载 rank 4 或 rank 8 LoRA。它不会读取 Gold、heldout、formal-v3
或任何项目数据集。

入口只接受配置中锁定的、已解包的本地 Hugging Face 检查点。GGUF 是
llama.cpp 推理格式，不能作为 PEFT 训练底座，因此会被明确拒绝。执行时
强制 Hugging Face 与 Transformers 离线，所有模型加载均设置
`local_files_only=True`。

```powershell
$env:ANCHOR_QWEN25_15B_HF_PATH = 'D:\LLM\models\qwen2.5-1.5b-instruct-hf'
anchor-qwen-diagnostic --config configs/training/qwen2_5_1_5b_lora_one_step_diagnostic.yaml --dry-run
anchor-qwen-diagnostic --config configs/training/qwen2_5_1_5b_lora_one_step_diagnostic.yaml --rank 4 --execute
```

模型 ID、ModelScope 源地址、Git revision 和四个必需文件的 SHA-256 都固定
在配置中。模型工作树必须干净，所有受 Git 跟踪的字节必须与锁定 revision
一致；身份值不能由环境变量覆盖。

只有以下门槛全部通过才算诊断成功：

- 28 个 Transformer 层各自恰好包含一对形状正确且可训练的 `q_proj`
  LoRA A/B；
- 所有可训练参数、梯度、logits 和 adapter delta 都是有限数值，并至少
  观察到一个非零 LoRA 梯度；
- 恰好一步优化后，冻结底座的完整参数哈希不变；
- 关闭训练后的 adapter 会改变 logits；
- 全新加载底座与已保存 adapter 后，可复现训练后的 next-token logits；
- staging 与最终 adapter 文件的认证哈希完全一致。

配置、模型、Git 元数据以及输出路径的各级祖先都必须是物理路径，不能是
符号链接、junction 或其他 reparse point。输出只允许落在由模块位置锚定的
`artifacts/diagnostics` 中，而且不会覆盖已有目录。

加载器会认证一个私有硬链接快照，并在收据中记录完整来源身份。由于硬链接
仍共享底层 inode，这个入口属于要求 `trusted_local_storage` 的机械诊断；
面对可原地篡改检查点的特权写入者时，它明确**不具备发布级安全性**。
在校验和重新加载前，保存的 adapter 配置会被规范化为稳定模型 ID
`Qwen/Qwen2.5-1.5B-Instruct`，不会残留临时 `.model-snapshot` 路径或个人绝对路径。

## 本机参考验证

2026-07-22 在 RTX 3080 Ti 上完成一次 rank-4 实跑：总耗时约 72 秒，训练
loss 为 `1.47572124`；56 个 LoRA 张量、344,064 个可训练参数均符合契约。
首步共有 28 个 B 矩阵出现非零梯度（A 矩阵因零初始化的 B 而保持零梯度，
符合 LoRA 首步数学预期）。关闭 adapter 后的最大 logits 差为 `0.603515625`，
保存后在全新底座上重载的最大误差为 `0.0`，338 个底座张量哈希保持不变。
`nvidia-smi` 采样到的整卡显存为 4,758 MiB；这是机械验证观测值，不是性能基准。审计计数
为 `provider_requests=0`、`external_dataset_inputs=0`。
