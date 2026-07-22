# aLoRA Prefix-KV TF32 v2 诊断档

FP32 v1 原样保留为高精度参考。TF32 v2 使用独立配置、Schema、输出目录和
receipt；它保留 float32 存储，但启用 Tensor Core TF32、
`matmul_precision=high`、确定性算法和
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

RTX 3080 Ti 实测 paired differential 为 `0.0301640`，adapter effect 为
`0.212040`，比值为 `0.142256`。v2 的非正式诊断预算同时要求绝对值不超过
`0.05`、相对值不超过 `0.20`，并继续要求 Prefix-KV bit-equal、argmax/贪心
序列一致等门。

该结果只能记为 `proxy_signal_passed=true`；它不证明数值等价、质量提升或正式
训练准备度，所以 receipt 明确写入 `numeric_equivalence=false` 和
`thresholds_formal=false`。

```powershell
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_tf32_v2.yaml --execute
```
