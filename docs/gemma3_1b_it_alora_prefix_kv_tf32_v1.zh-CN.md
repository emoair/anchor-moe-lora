# Gemma 3 1B IT aLoRA Prefix-KV TF32 诊断

这是一个只用于机械可行性验证的诊断组，与 Qwen v1/v2 完全隔离。它用
float32 存储参数并启用 TF32 与 eager attention，只在 26 层 `q_proj` 上挂载
确定性的 rank-4 PEFT aLoRA。

诊断会执行 base 与 aLoRA 的完整/分段路径、adapter-off/active 前缀路径以及
缺失触发词对照组。请求采用版本化的 `anchor.gemma3-plaintext-r2.v1` 纯文本
序列化，由物理 SentencePiece 模型完整编码一次并显式添加 BOS。它不依赖
Windows 上缺失的 `tensorflow-text`，也不使用尚未绑定的 chat template。

HF 导出配置声明 BOS/EOS 为 `1/2`，物理 SentencePiece 模型则为 `2/1`。
诊断会同时记录两套身份，并强制以物理 SentencePiece 的 BOS `2` 为权威，
禁止静默混用。

```powershell
$env:PYTHONPATH = "src"
anchor-gemma3-alora-prefix-kv --config configs/research/gemma3_1b_it_alora_prefix_kv_tf32_v1.yaml --preflight
anchor-gemma3-alora-prefix-kv --config configs/research/gemma3_1b_it_alora_prefix_kv_tf32_v1.yaml --execute
```

仓库中的 RTX 3080 Ti 回执已通过全部诊断门：26 层前缀 KV 均位级一致，base
与 aLoRA 的完整/分段配对差均为零，确定性适配器效应非零，缺失触发词效应
为零；峰值已分配显存约 3.83 GiB。

这只代表本地 proxy signal 通过，不代表数值等价、质量提升、正式训练就绪、
多流执行、零拷贝 KV 或完整生成过程的 KV 共享。
