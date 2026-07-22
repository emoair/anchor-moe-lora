# aLoRA Prefix-KV TF32 v2 diagnostic profile

FP32 v1 remains the high-precision reference. TF32 v2 has an independent
config, schemas, output directory, and receipt. It retains float32 storage but
enables Tensor Core TF32, `matmul_precision=high`, deterministic algorithms,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

On the RTX 3080 Ti, the paired differential was `0.0301640`, the adapter
effect was `0.212040`, and their ratio was `0.142256`. The non-formal v2
diagnostic budget requires both an absolute value at most `0.05` and a relative
value at most `0.20`, while retaining the Prefix-KV bit-equality and
argmax/greedy gates.

This result is only `proxy_signal_passed=true`. It does not establish numeric
equivalence, quality improvement, or formal training readiness, so the receipt
also declares `numeric_equivalence=false` and `thresholds_formal=false`.

```powershell
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_tf32_v2.yaml --execute
```
