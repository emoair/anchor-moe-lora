# 五角色 Q-only 消费端 v2

这是一个追加式、严格失败关闭的数据消费端。它会先认证完整的 1000 条纯合成诊断数据，再选择单一角色。数据由 200 个任务包乘以五个角色组成：`planner`、`tool_policy`、`frontend_gen`、`frontend_review` 和 `security_gate`。每个角色都有 160 条 `train` 与 40 条 `eval_proxy`。五个计划中的适配器彼此独立，均为 rank-4 `q_proj` LoRA。`O-only` 与 `Q+O` 只保留为诊断覆盖层标签，不复制数据，也不能进入主训练入口。

## 快速开始

在 `D:\LLM\anchor-moe-lora-neural-swarm` 下运行。默认的数据集模式不会加载 tokenizer、模型、CUDA、外部服务或训练器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1
```

Python 会按以下顺序探测：显式 `-PythonExecutable`、`ANCHOR_PYTHON`、当前
`CONDA_PREFIX`、仓库 `.venv`、`$HOME\.conda\envs\anchor-mvp`，最后才是
真实的 `python` 命令。无法正确响应 `--version` 的 WindowsApps 占位符会被拒绝。只检查 Python：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1 `
  -CheckPythonOnly
```

全部通过时，最后一行应为：

```text
[five-role preflight] PASS: all five roles authenticated serially; no model, CUDA, provider, or training request was made.
```

只检查单一角色：

```powershell
python scripts/research/prepare_synthetic_five_role_qonly_v2.py --role planner
```

预期 JSON 状态为 `passed_dataset_only_dry_run_training_blocked`。不使用辅助脚本时，可手动循环五个角色：

```powershell
$roles = "planner","tool_policy","frontend_gen","frontend_review","security_gate"
foreach ($role in $roles) {
  python scripts/research/prepare_synthetic_five_role_qonly_v2.py --role $role
  if ($LASTEXITCODE -ne 0) { throw "preflight failed: $role" }
}
```

可选的 tokenizer-only 长度预检完全离线，仍不加载模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1 `
  -TokenizerOnly
```

只有使用 `-TokenizerOnly` 时才能追加 `-PublishPreflight`。回执会写入角色隔离且禁止覆盖的目录：
`artifacts/diagnostics/qwen2_5_1_5b_synthetic_five_role_qonly_v2/<role>/preflight`。

## 预检实际证明什么

在应用 `--role` 过滤前，消费端会认证强制 manifest sidecar、记录与 manifest schema、两个分区的单次字节快照、全部 1000 条记录、200 个五角色任务包、固定阶段映射、bundle split、单元格配额，以及 200 个命名空间中性的语义身份。每个 bundle 的五份任务板 inventory 会逐项交叉绑定；当前目标也会反向绑定到自己的 segment hash。当前与未来阶段的 segment ID、摘要、目标 JSON 和序列化答案都不得进入 prompt。过滤发生在 tokenization 之前。

声明的训练数值配置为 BF16 计算、TF32 矩阵乘、micro-batch 1、序列长度 512、`use_cache=false`，并按串行方式规划五个 Q-only rank-4 任务。本模块只生成计划和预检结果，故意不提供训练执行入口。

## 专家私有尾部 KV 边界

相同的冻结前缀必须在 adapter-off 状态下只读共享。某个 Q-only 专家激活后，激活后的 prompt 与新生成 token 只能追加到该专家自己的私有尾部 KV。不同专家之间绝不能复用私有尾部。专家提交文本后，该文本必须重新编码进下一阶段的共享上下文。这样每个专家才能像独立 agent 一样工作，同时不会错误宣称“整段生成 KV 共享”。普通的 in-stack Q-LoRA 也不能宣称精确 KV 共享。因此 `runtime_private_tail_materialized=false` 与 `execution_authorized=false` 仍是硬门。

## 常见阻塞状态

- `five_role_fixture_identity_pending`：最终数据集哈希尚未锁入配置。
- `identity_mismatch` 或 SHA 错误：fixture 或配置已经漂移；应重新构建并审计，再一次性更新完整哈希集合。
- tokenizer 错误：检查配置中的本地 Qwen tokenizer 目录，以及锁定的 `tokenizer.json`、`tokenizer_config.json` 哈希。
- 缺少模型权重不会影响 dataset-only 或 tokenizer-only 检查。本消费端故意不提供训练命令，不能把预检计划误当成模型执行。
- 找不到可用 Python：执行 `conda activate anchor-mvp`、设置 `ANCHOR_PYTHON`，或传入
  `-PythonExecutable C:\path\to\python.exe`。
- `five_role_preflight_output_exists`：已发布回执禁止覆盖；请使用新命名空间，或明确归档旧回执。
- `forbidden_*`：因果过滤在 tokenization 前拒绝了数据，应重新构建。CLI 不会输出任何记录正文。

这批数据不是 held-out，不满足“两条独立的 600-record 确认轨”，不授权训练，也不能支持正式、质量、物理 KV 或多流结论。
