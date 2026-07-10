# Anchor-MoE-LoRA 训练子系统

本子系统把 Gemma 4 12B Unified 当作共享冻结基座，分别训练三个任务 LoRA 和一个混合对照 LoRA。默认命令只做 dry-run，不下载模型，也不启动训练。

## 精度路线

训练默认使用非对齐的 `google/gemma-4-12B` 标准 BF16/FP16 Transformers checkpoint，revision 固定为 `56820d7d8cbe8e47975a53325439ed272e91cff2`：加载时由 bitsandbytes 在线量化为 4-bit NF4，启用 double quant；基座参数冻结，LoRA 参数使用 BF16。这个 base 当前没有官方训练版 Q4 checkpoint，所以默认路线是在线 NF4。也可以通过 `ANCHOR_BASE_MODEL_ID` 改成 Transformers/PEFT 能直接加载并训练的 4-bit checkpoint，并把 `ANCHOR_MODEL_LOAD_STRATEGY` 设为 `prequantized_peft_4bit`。代码使用 Gemma 4 官方 Transformers 类 `AutoModelForMultimodalLM`；`google/gemma-4-12B-it` 在这里只提供官方对话 processor/template，不提供训练基座权重，当前 MVP 只喂文本对话。

**QLoRA 本来就是在训练兼容的 4-bit 基座上挂接并训练 adapter；4-bit/Q4 这个位宽不是拒绝训练的理由。** 真正需要区分的是序列化格式和训练框架能否恢复计算图。这里有三类容易混淆的产物：

- 本项目的 **bitsandbytes NF4 QLoRA** 是训练路径：可从标准 BF16/FP16 checkpoint 在线量化，也可从框架兼容的 PEFT 4-bit checkpoint 开始；只有 LoRA 更新。
- Google 官方 **QAT W4A16 compressed-tensors** 面向 vLLM/SGLang 推理部署，当前 PEFT 训练入口不支持把该序列化直接当基座。
- **GGUF Q4_0** 面向 llama.cpp/LM Studio 推理，当前 Transformers + PEFT 入口不能直接加载它来训练 adapter。

因此配置校验拒绝的是带 `.gguf`、`-gguf` 或 `-w4a16-ct` 的推理专用 `model.id`，不是泛化地拒绝 4-bit 模型。

官方参考：[Gemma 4 模型与 QAT 格式说明](https://ai.google.dev/gemma/docs/core)、[Gemma QLoRA 指南](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)。

## 12 GB 显存边界

RTX 3080 Ti 12 GB 可以尝试 12B 的保守 QLoRA smoke，但余量很窄。官方估算 Gemma 4 12B 仅 4-bit 静态权重推理就约占 6.7 GB，训练还要承担激活、LoRA、梯度和优化器状态。因此默认值是：

- `max_seq_length=256`
- micro batch 1，梯度累积 4
- gradient checkpointing
- paged 8-bit AdamW
- rank 16、8 个 smoke steps
- 不训练 embedding、`lm_head`、视觉或音频投影

这是一张可执行性探针，不代表 rank 64 或长代码样本必然能放进 12 GB。先跑 rank 16；记录峰值显存和 step time 后，再依次放行 rank 32、64。若仍 OOM，优先缩短序列和样本，而不是关闭基座冻结或悄悄 CPU offload 后把延迟结果混进主实验。

## 数据契约

每个 JSONL 行必须是 UTF-8 JSON object：

```json
{
  "schema_version": "1.0",
  "id": "frontend-000001",
  "expert": "frontend_gen",
  "messages": [
    {"role": "system", "content": "Follow the injected frontend SOP."},
    {"role": "user", "content": "Build a responsive product landing page."},
    {"role": "assistant", "content": "clean target artifact"}
  ],
  "provenance": {
    "teacher": {"model": "k2.7"},
    "sop": {"sop_id": "frontend-sop-v1", "sha256": "..."}
  },
  "decision_trace": [
    {"check": "component boundary", "evidence": "two independent sections", "action": "split components"}
  ],
  "output": {"code": "clean target artifact"}
}
```

主实验的 `expert` 为 `planner`、`tool_policy`、`frontend_gen`、`frontend_review`、`security_gate`。旧的 `code_review` / `security_audit` 只为恢复早期成功样本而可读，不再是新训练配置的专家名。可审计方法记录在结构化 `decision_trace`，而不是不可验证的隐藏思维链字段。安全样本的最后一个 assistant target 必须且只能包含 `[BLOCK]` 或 `[PASS]` 之一，并与 `output.decision` 一致。验证器还检查消息角色、空内容、重复 ID、任务 output 和 provenance。`mixed_all` 不创造第六种 expert 标签，只把五份经过校验的数据合并加载。

默认路径：

| Adapter | Dataset |
| --- | --- |
| `planner` | `data/automated_v2/data_plan.jsonl` |
| `tool_policy` | `data/automated_v2/data_tool_policy.jsonl` |
| `frontend_gen` | `data/automated_v2/data_frontend.jsonl` |
| `frontend_review` | `data/automated_v2/data_review.jsonl` |
| `security_gate` | `data/automated_v2/data_security.jsonl` |
| `mixed_all` | 上述五份文件 |

## 环境与 dry-run

当前 Transformers 5.x 训练栈要求 Python 3.10 或更高。不要在现有 Python 3.9 `base` 里强行补包；创建或选择带 CUDA PyTorch 的 Python 3.10+ 环境，再在项目根目录安装依赖（脚本不会自动安装）：

```powershell
$env:ANCHOR_TRAIN_PYTHON = "C:\path\to\python310-env\python.exe"
& $env:ANCHOR_TRAIN_PYTHON -m pip install -r configs\training\requirements-qlora.txt
```

未设置 `ANCHOR_TRAIN_PYTHON` 时训练脚本使用系统 `py -3.10` 对应的解释器。dependency report 会把旧 Python、CPU-only Torch、缺包或不兼容版本标记为 `ready=false`。

Gemma 权重是 gated model，真实下载前需在 Hugging Face 接受许可证并设置有读取权限的 token。dry-run 不需要 token：

```powershell
.\scripts\train\run_adapter.ps1 -Adapter frontend_gen -Rank 16
.\scripts\train\validate_all.ps1
```

dry-run 会验证配置、探测依赖和 CUDA/BF16、验证已经存在的数据集，并写入 `artifacts/manifests/<adapter>-r<rank>.dry-run.json`。数据尚未生成时会在 manifest 记录 `exists=false`，但仍允许完成配置预检；加 `--require-data` 可把缺失数据变成失败。

one-step smoke-gate 使用标准库按需读取第一条 JSONL，并用轻量 PyTorch
优化循环执行一步；该路径不导入 TRL、HF Datasets 或 PyArrow。正式训练路径
暂时保留 TRL，并固定 `pyarrow>=21,<22`。本机 Python 3.11 上的 PyArrow 24.0.0
曾在导入 Datasets 时于 `arrow.dll` 原生崩溃，模型尚未加载；不要把这种无
Python traceback 的退出误判为显存不足。

Gemma 4 Unified 暂无 PEFT 自动 target mapping。RTX 3080 Ti 安全基线显式
使用语言塔的 `q_proj` 与 `v_proj`；本地权重清单确认这两个后缀未命中视觉或
音频塔。扩大到 `k_proj/o_proj` 或 MLP 投影必须作为单独显存消融实验。

长代码样本使用 assistant-preserving 截断。collator 通过 tokenizer offset
定位真实 assistant 内容（不使用 Gemma 4 generation prompt 的 token 数猜边界），
在 128-token one-step smoke 窗口中保留最近的 prompt 上下文与 assistant 开头监督 token。
正式训练如需覆盖完整代码，应另外做分块/长上下文消融，不能把首窗指标解释为完整代码学习。

如需覆盖模型或加载策略：

```powershell
$env:ANCHOR_BASE_MODEL_ID = "example/gemma-4-12B-bnb-nf4"
$env:ANCHOR_BASE_MODEL_REVISION = "<fixed-commit>"
$env:ANCHOR_MODEL_LOAD_STRATEGY = "prequantized_peft_4bit"
```

标准 BF16/FP16 checkpoint 保持默认的 `bnb_nf4_online`。正式实验必须固定 revision；不要让 `main` 漂移。

## 显式训练

本地缓存已有模型时：

```powershell
.\scripts\train\run_adapter.ps1 -Adapter frontend_gen -Rank 16 -Execute
```

只有明确允许联网拉权重时才加：

```powershell
.\scripts\train\run_adapter.ps1 -Adapter frontend_gen -Rank 16 -Execute -AllowModelDownload
```

训练会先强制检查 CUDA、BF16、显存、依赖和完整数据；检查 LoRA 挂载后所有非 LoRA 参数仍冻结；然后才进入 TRL。输出位于 `artifacts/adapters/<adapter>-r<rank>/`，其中包括 PEFT adapter、processor 和 `checkpoint_metadata.json`。manifest 记录配置哈希、数据 SHA-256、模型 revision、环境版本与精度路线。

## 放大前闭环门禁

正式训练不能直接从“配置看起来正确”跳到三专家长跑。先执行只读 preflight：

```powershell
.\scripts\train\preflight.ps1
```

它不会加载 23 GB 模型。门禁逐项核对：

- `data/live_smoke/data_frontend.jsonl`、`data_review.jsonl`、`data_security.jsonl` 三份 canonical JSONL 都存在；
- schema、跨文件 ID 唯一性、非空 assistant target、真人 teacher provenance；
- 非对齐 base 的固定 repo/revision、23,919,549,408-byte 文件、SHA-256 和 Hugging Face LFS OID 验证；
- Python/训练依赖、CUDA、BF16、至少 10.5 GiB 当前空闲显存，以及至少 12.0 GiB 当前空闲主机物理内存；
- 三专家 held-out case 文件完整。

默认 checksum 复用下载阶段已经计算并与 LFS OID 对上的 `anchor_download_manifest.json`，同时重新检查权重文件尺寸。若需要重新顺序读取整份 23 GB 文件：

```powershell
.\scripts\train\preflight.ps1 -DeepBaseChecksum
```

报告写入 `artifacts/manifests/preflight.dry-run.json`。任何一项失败都会返回非零状态；尤其三份真人样本未齐时，`--execute` 训练入口会硬拒绝启动。

三份数据和环境全部通过后，运行 one-sample/one-step QLoRA gate 的 dry-run：

```powershell
.\scripts\train\run_smoke_gate.ps1 -Adapter frontend_gen
```

只有 dry-run 显示 ready 后，才显式执行：

```powershell
.\scripts\train\run_smoke_gate.ps1 -Adapter frontend_gen -Execute
```

本地 23 GB base 会直接从 `models/google-gemma-4-12B-base` 加载，不会重复下载。若 `google/gemma-4-12B-it` 的 processor/template 尚未缓存，首次执行可加 `-AllowModelDownload`；这只允许补齐 processor 资产，训练基座仍由 checksum 门禁锁定为本地非对齐 base。

该命令使用 `gemma4_12b_qlora_one_step.yaml`：rank 16、seq 128、batch 1、gradient accumulation 1、单条真人样本、单步更新。它必须留下以下闭环证据：

- `global_step == 1` 且 loss 为有限数；
- 包含 base 加载与训练的 CUDA peak allocated/reserved VRAM；
- adapter config/weights 确实保存；
- 从磁盘卸载并重载 adapter 后，held-out logits 与保存前一致；
- held-out 的 pre/post next-token distribution 确实变化。

smoke 证据只做一次无 KV cache 的前向并读取最后一个位置的 next-token logits，
不会调用 `generate`。因此“输出差异”的硬门槛使用 logits 最大绝对差；reload
要求 logits 差异不超过 `1e-4`。三个最小 held-out case 位于
`configs/training/heldout_cases.jsonl`，从未进入训练数据。

成功证据写入 `artifacts/manifests/smoke-gate-frontend_gen-r16.execute.json` 和 adapter 的 `checkpoint_metadata.json`。之后普通 `run_adapter.ps1 -Execute` 还会重新做 preflight，并确认 smoke manifest 的 base revision、数据快照哈希和 passed 状态都未漂移；任一不一致都会阻止放大训练。

## Rank 消融和四组产物

```powershell
.\scripts\train\run_rank_ablation.ps1 -Adapter frontend_gen
```

默认仍只是三档 dry-run。真实消融必须显式加 `-Execute`。比较 16/32/64 时，模型 revision、数据哈希、seed、step 数、序列长度、batch 和评测集必须保持不变。五个专属 adapter 与 `mixed_all` 必须使用相同 rank 才能做 B/C 组公平对比。

## 已知风险

- 默认 base revision 已固定；覆盖 `ANCHOR_BASE_MODEL_ID` 时必须同时通过 `ANCHOR_BASE_MODEL_REVISION` 固定对应 commit hash。
- Gemma 4 与 Transformers/TRL 接口仍可能快速演进；当前版本下限来自官方 2026-07 QLoRA 指南。首次真实运行应保持 8 steps，不要直接长训。
- BF16 LoRA 满足本实验显存路线，但与常见的 FP32 adapter 优化数值行为不同；实验报告必须保留该信息。
- 当前 collator 只对最后一个 assistant turn 计算 loss，避免把 system/user prompt 训练成目标。
- Gemma 4 tokenizer 默认 left padding；collator 会按每行 `attention_mask` 将 prompt mask 平移到有效序列起点。若 `max_seq_length` 截断后没有任何 assistant target token，batch 会明确失败，不会用全 `-100` labels 静默训练。
- “4-bit checkpoint”必须具体到格式：只有 Transformers/PEFT 训练栈兼容的量化 checkpoint 才能作为输入；同为 4-bit 的 GGUF 或 vLLM compressed-tensors 不能据此自动视为可训练。
