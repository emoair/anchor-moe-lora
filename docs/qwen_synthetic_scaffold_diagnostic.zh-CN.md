# Qwen2.5-1.5B 合成脚手架 q_only 诊断

这是小规模本地诊断，不是正式训练，也不是 A–F 评测结果。它只在
Qwen2.5-1.5B 的 28 个语言模型 `q_proj` 上训练 rank-4 LoRA，数据仅来自
100 条 `synthetic_nl_scaffold_diagnostic_v1` 合成夹具；绝不读取 Gold、
heldout、SWE-bench 正文或 provider 输出。

该 runner 与冻结的一步 Qwen 诊断完全隔离。它复用已经验证的本地 HF
身份认证、私有硬链接快照、冻结基座哈希、严格 q_proj 范围、adapter
效果以及保存/重载门控，但不修改原冻结契约。

## 固定资源边界

- 只接受本地 Qwen2.5-1.5B-Instruct Hugging Face checkpoint；拒绝 GGUF。
- `q_proj`、rank 4、alpha 8、batch 1、sequence length 512。
- BF16 计算、TF32 开启、非重入 gradient checkpointing。
- optimizer step 只能为 2 或 20；默认先做 2-step smoke。
- 80 条合成 train 与 20 条 `eval_proxy`；后者不是 heldout。
- 每条完整 chat-template 输入与 target 必须整体装入 512 tokens；任何截断都在模型加载前失败关闭。
- 2-step 与 20-step 输出隔离且禁止覆盖。
- `training_authorized=false`、`formal=false` 固定不变。

## 命令

在仓库根目录运行：

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

命令会打印物理 preflight receipt 的 SHA-256。人工检查 receipt 与显存余量后，
把该值原样传给显式授权的 2-step 诊断：

```powershell
conda run -n anchor-mvp python -m anchor_mvp.training.qwen_synthetic_scaffold_diagnostic `
  --config configs/training/qwen2_5_1_5b_synthetic_scaffold_qonly_v1.yaml `
  --max-steps 2 --execute `
  --preflight-receipt artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_qonly_preflight/step2-v2/preflight.json `
  --preflight-receipt-sha256 <DRY_RUN_打印的_SHA256>
```

只有 2-step 产物通过全部门控后，才可把 `2` 改为 `20`，使用独立的
preflight 与 adapter 输出目录运行 20-step；它不会续训或覆盖 2-step。
20-step 必须使用全新的 `step20-v2` preflight 目录；旧 receipt 身份只会标记为
superseded，绝不覆盖或静默复用。

## dry-run 与发布认证

dry-run 从单次字节快照认证 fixture manifest、强制 sidecar、schema 与四个
partition，验证 100 条记录及 train/eval-proxy source-bundle 隔离，并且只加载
本地 tokenizer 测量完整 chat-template token 长度。它不实例化模型权重、不用
GPU、不请求 provider，也不访问网络。

dry-run 会在显式指定、尚不存在的目录中，以原子 no-replace 方式发布版本化
`preflight.json` 与严格 `preflight.json.sha256`。execute 从同一字节快照认证
receipt 和 SHA，重新计算完整 tokenizer-only preflight，只有逐字段完全相等才会
进入模型加载路径。
receipt 还会按 record ID 排序，绑定每条 prompt IDs、完整 IDs 与 masked labels 的
SHA-256 摘要以及实际 Transformers/Tokenizers 运行时版本；绝不输出原始 token IDs。

最终 adapter 目录同样绑定 `adapter_config.json`、`adapter_model.safetensors` 与
`diagnostic_receipt.json`。receipt 有严格 SHA sidecar，并记录认证过的 preflight
SHA、token 长度报告、runner 实现 SHA 及实际有序训练记录清单。

## 结论边界

20 条合成 eval-proxy 的 loss 变化只属于 proxy telemetry，不能证明泛化、共享 KV
正确性、物理 KV 复用、正式就绪、数值等价或不同 LoRA profile 的优劣。
