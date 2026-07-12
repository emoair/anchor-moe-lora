# Anchor-MoE-LoRA 简中端到端快速上手

> 当前项目绝对路径：`D:\LLM\anchor-moe-lora`
>
> **重要门禁：live gold 必须使用仓库构建并校验的 patched OpenCode，不得使用 PATH 中的官方裸版。**
> Windows 与 WSL/Linux 产物、一次性 Podman 沙箱、受控 raw export 和清理链已经接入；默认仍只做
> 离线预检。只有进程级 API key、held-out 分离和 patched capability probe 全部通过后，才允许
> 显式添加 `--confirm-live`。不要读取历史真实 session。

本文所有不带 `--confirm-live` 的示例均可离线执行。任何 live 命令都必须在完成文中门禁后手动运行。

## 1. 环境与路径

推荐使用已经准备好的独立 Conda 环境，不修改 `base`：

```powershell
conda activate anchor-mvp
Set-Location D:\LLM\anchor-moe-lora
python -m pip install -e ".[teacher,dev]"
python -m pytest -q
```

本机可直接定位解释器为：

```text
C:\Users\Air\.conda\envs\anchor-mvp\python.exe
```

若尚未安装环境，执行：

```powershell
.\scripts\environment\bootstrap_windows.ps1
```

确认当前运行的不是错误 Python：

```powershell
python -c "import sys, anchor_mvp; print(sys.executable); print(anchor_mvp.__file__)"
```

## 2. 第一次运行必须离线

先用 mock teacher 跑完五类数据，不访问网络、不消耗额度：

```powershell
$dryRun = Join-Path $env:TEMP "anchor-moe-lora-dry-run"
python -m anchor_mvp data `
  --config configs/data/smoke.yaml `
  --dry-run `
  --output-dir $dryRun `
  --seed-count 1 `
  --concurrency 1 `
  run
```

预期：`plan/tool_policy/frontend/review/security` 各写入 1 条，`errors` 为空。

检查 OpenCode batch、Skill 哈希和 held-out 分离，但不启动 OpenCode：

```powershell
python scripts/tooling/run_live.py `
  --batch-config configs/tooling/opencode_distillation_ramp.yaml
```

预期包含：`DRY RUN`、`candidate_count=1`、`requested_stages=1`。即使显示
`opencode_available=True`，也不代表已经满足 patched-binary live 门禁。

## 3. Provider preset、模型发现与手动选择

当前 preset：

| preset | 协议 | 默认 Base URL | 默认模型/选择方式 | key 环境变量 |
| --- | --- | --- | --- | --- |
| `kimi-code-openai` | OpenAI | `https://api.kimi.com/coding/v1` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-code-anthropic` | Anthropic | `https://api.kimi.com/coding/` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-platform-openai` | OpenAI | `https://api.moonshot.cn/v1` | 手动/发现 | `MOONSHOT_API_KEY` |
| `openai` | OpenAI | `https://api.openai.com/v1` | 手动/发现 | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `https://api.anthropic.com` | 手动/发现 | `ANTHROPIC_API_KEY` |
| `custom-openai` | OpenAI | 必填 | 手动/发现 | 默认 `TEACHER_API_KEY` |
| `custom-anthropic` | Anthropic | 必填 | 手动/发现 | 默认 `TEACHER_API_KEY` |

### 不带 key 的安全检查

以下命令不会发请求；结果应为 `missing_credential`：

```powershell
python -m anchor_mvp data --provider kimi-code-openai models
```

Kimi Code 没有在本项目中配置稳定的公开 quota API，因此以下命令应返回 `unsupported`：

```powershell
python -m anchor_mvp data --provider kimi-code-openai quota
```

`kimi-platform-openai` 才支持官方 balance capability；它需要 `MOONSHOT_API_KEY`，并且会发出
真实查询。Quota 查询只是信息，不会自动放宽 automation 预算。

### 发现模型

模型发现会调用标准 `/models` 接口，必须有 key：

```powershell
python -m anchor_mvp data `
  --provider custom-openai `
  --base-url https://gateway.example.com/v1 `
  --api-key-env TEACHER_API_KEY `
  models
```

不要把完整 endpoint 填入 `--base-url`。`.../models`、`.../messages`、
`.../chat/completions`、自然语言描述、缺少协议头或含空格的值都会在请求前被拒绝。

### 手动模型与 `--force-model`

对已知模型 ID，使用：

```powershell
python -m anchor_mvp data `
  --provider custom-openai `
  --base-url https://gateway.example.com/v1 `
  --api-key-env TEACHER_API_KEY `
  --model provider-model-id `
  --force-model `
  probe
```

`--force-model` 只表示跳过 discovery，不表示跳过生成请求；`probe` 仍会调用 provider。
`--model-index N` 只能用于本次 discovery 成功后返回的零基索引。Provider 返回顺序可能变化，
长期自动化优先固定 `--model`。

## 4. key 只进入当前进程环境

不要把 key 写入 YAML、JSON、`.env`、命令行参数、日志或 Git。PowerShell 7：

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
try {
  python -m anchor_mvp data `
    --config configs/data/default.yaml `
    --model kimi-for-coding `
    --force-model `
    probe
} finally {
  Remove-Item Env:KIMI_API_KEY -ErrorAction SilentlyContinue
}
```

只有完成离线 dry-run 后，才从 1 seed、并发 1 开始：

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
try {
  python -m anchor_mvp data `
    --config configs/data/default.yaml `
    --model kimi-for-coding `
    --force-model `
    --seed-count 1 `
    --concurrency 1 `
    run
} finally {
  Remove-Item Env:KIMI_API_KEY -ErrorAction SilentlyContinue
}
```

## 5. Automation、quota epoch 与失败账本

Automation 默认并发为 `1`。`concurrency_stages` 可由操作者填写任意非空正整数序列，代码中
没有固定 `1 -> 2 -> 4 -> 8` 或最大 8 的上限；每一档仍必须通过 schema、重复率、安全和
frozen held-out 门禁。查看状态：

```powershell
.\scripts\data\show_automation_status.ps1 -Config configs/data/automation.yaml
```

真正启动前检查 `configs/data/automation.yaml`：

- `quota_epoch_id`：仅在 provider 额度窗口确实重置后改成新 ID；
- `max_requests`、`max_output_tokens_total`：只约束当前 quota epoch；
- `audit_ledger`：跨窗口累计，不因 epoch 重置而删除；
- `max_failure_retries`：同一 `(seed, task, error-class)` 的有界重试；
- 超限失败进入 quarantine，不再重复扣费或调用下游。

Live 启动会消耗额度：

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
try {
  .\scripts\data\start_automation.ps1 -Config configs/data/automation.yaml
} finally {
  Remove-Item Env:KIMI_API_KEY -ErrorAction SilentlyContinue
}
```

不要通过删除 `status.json` 来“重置额度”；这会破坏审计链。修改新的 `quota_epoch_id`，旧窗口
会进入 `quota_history`，累计失败账本仍保留。

## 6. 受控 OpenCode execution gold

### 当前状态：patched 沙箱管线已构建，默认仍只预检

当前官方 OpenCode 版本：`1.17.18`。以下命令只预检：

```powershell
python scripts/tooling/run_live.py `
  --batch-config configs/tooling/opencode_distillation_ramp.yaml
```

只有在进程级 key 和 dry-run 均通过后才添加 `--confirm-live`。patched binary/runner 必须同时满足：

1. session 只能写入该样本的隔离 XDG data 目录；
2. 在 runtime 清理前得到受控 raw export；
3. sidecar 记录匹配的 session ID、validator 完整 stdout/stderr 和 public outcome；
4. raw export 与 sidecar 进入 fail-closed converter；
5. converter 完成后才删除隔离 session；
6. 不读取 `%USERPROFILE%\.local\share\opencode\opencode.db` 中的历史 session。

当前默认只配置一个候选、一个并发 stage。首次 live 放行命令：

```powershell
python scripts/tooling/run_live.py `
  --batch-config configs/tooling/opencode_distillation_ramp.yaml `
  --max-stages 1 `
  --confirm-live
```

当前只有 `sidex-p0-001-stable-status-sort` 具备一致的任务、fixture 和冻结验收合同；其余题目仍是
deferred。要增加并发，必须同时在 batch 配置中增加正整数 stage、样本数和已经审计的候选；单独把
`--max-stages` 调大仍会被拒绝。

### attempts 与 accepted gold 不是一回事

默认输出：

```text
artifacts/tooling/live_attempts.jsonl       所有尝试，含失败原因
artifacts/tooling/live_gold.accepted.jsonl  仅通过全部门禁的 accepted gold
```

失败、timeout、无 tool、无改动、修改冻结测试、缺少 public outcome 或 validator 失败的 attempt
不能进入 accepted gold。不要把 attempts 文件当训练集。

## 7. Raw export + sidecar 转候选 JSONL

官方 `opencode export --sanitize` 会删除任务正文、assistant 正文、tool input/result 和 diff，
所以只能分享/取证，不能做完整训练轨迹。Converter 会拒绝 `[redacted:...]` 占位符。

patched runner 应在隔离 session 尚存在时执行等价于：

```powershell
opencode export <controlled-session-id> > runs/capture/session.raw.json
```

不要加 `--sanitize`，也不要对历史真实 session 执行。Raw export 只能来自 disposable fixture，且必须
配套 `anchor.controlled-session-capture.v1` sidecar。转换：

```powershell
python scripts/tooling/convert_session_export.py `
  --export runs/capture/session.raw.json `
  --capture runs/capture/sidecar.json `
  --workspace runs/tooling-live/sample-workspace `
  --heldout-cases configs/benchmark/heldout_cases_v1.jsonl `
  --heldout-fixtures-root examples/benchmark/fixtures `
  --heldout-manifest artifacts/benchmark/heldout_v1/manifest.json `
  --output artifacts/tooling/session_candidates.jsonl `
  --quarantine artifacts/tooling/session_quarantine.jsonl
```

候选会保留：完整任务输入、公开 assistant 输出、按顺序且共享 `call_id` 的
`tool_call/tool_result`、受控 read/edit/apply_patch 结果、build/test/lint 完整 stdout/stderr、
最终 diff 和 public outcome。

以下任一命中会整条 quarantine，不会 REDACT 后混入训练：secret/credential、环境字段或环境读取、
工作区外路径、二进制/控制字符、大小超限、held-out 相似内容、损失性 sanitized export、未知 tool、
非 allowlist bash 命令、失败 validator、缺少 final diff/public outcome。

## 8. 如何确认样本真的可入训

只有同时满足以下条件才可从候选晋级训练快照：

- 来源是受控 fixture，而非历史 session；
- attempt `success=true`，且存在 accepted gold；
- task、初始源码、package、测试和脚本 SHA 与候选合同一致；
- `requires_changes=true` 时确实产生源码 diff；
- protected tests/package/TASK 没有被代理修改；
- build、test、lint 全部 `PASS` 且 exit code 为 0；
- public outcome 为 `completed`；
- trajectory 中至少有真实 `tool_call -> tool_result`；
- converter 输出候选而非 quarantine；
- secret、绝对路径和 frozen held-out 门禁通过；
- reasoning/thinking/system/provider metadata 不在候选 JSONL；
- 训练前再次冻结数据 SHA，并与 benchmark held-out 分离。

## 9. 常见故障

### HTTP 400 / `invalid_url`

通常是把自然语言或完整 endpoint 当作 Base URL。正确示例：

```text
https://api.kimi.com/coding/v1
https://api.kimi.com/coding/
```

错误示例包括缺少 `https://`、包含空格、`https://the repo for...`，或直接填
`.../chat/completions`。先用 preset，避免手拼 URL。

### HTTP 499 / `context canceled`

表示客户端在服务端返回前取消：用户中断、网络断开、wrapper timeout 都可能导致。它不等于服务端
5xx。检查 `wall_clock_deadline_seconds`、`timeout_seconds` 和外层终止信号；不要无限重试，因为请求
可能已经消耗额度。Automation 会持久化 cooldown/失败分类。

### 零 tool call

只输出建议、没有实际编辑/验证，不是 execution gold。检查：任务是否明确要求执行、Skill prompt 是否
加载、OpenCode JSONL 是否包含 tool event、权限是否允许 read/edit 和三条 npm 命令。项目通过
`requires_changes`、真实 diff、validator 与 public outcome 联合门禁阻止“嘴上完成”。

### `--sanitize` 后没有正文/tool result

这是官方预期行为，不是解析 bug。Sanitize 会主动抹掉这些字段。不要尝试从占位符恢复；改用受控 raw
export + sidecar converter。任何 sanitized export 都不能入训。

### 模型发现失败但已知模型可用

使用 `--model <真实模型ID> --force-model` 跳过 discovery。注意这不会跳过后续 probe/run 的真实请求。

### quota 已重置但 automation 仍显示旧失败

旧失败属于累计审计历史。更新 `quota_epoch_id` 开新窗口，不删除 ledger；已 quarantine 的重复失败仍
保持隔离，避免再次浪费额度。

## 10. A/B/C/D/E/F 对照不要混淆

所有组都使用同一冻结 Q4 基座、五阶段 DAG、数据切分、token cap 和评测题：

主路由的“五阶段”指五种专家，不代表固定五次调用：

`planner -> tool_policy -> (frontend <-> review，最多 2 轮) -> security`

审查 LoRA 只允许输出版本化公共 JSON：`PASS` 且 `issues=[]`，或
`REVISE` 且带精简问题列表。`REVISE` 会把当前代码和问题重新交给同一个
frontend LoRA；歧义、超时、字段不合规或轮次耗尽均 fail closed，security
不会运行。通过后，security 接收需求、最终代码和公共工具轨迹摘要。

注意兼容边界：旧 `data_review.jsonl` 的目标是“完整修复代码”，仅供旧
`PipelineRouter.run()`/旧 benchmark 使用；新主路由需要单独的
`review_verdict` adapter，以及 `data_review_verdict_v2.jsonl` 和
`data_frontend_revision_v2.jsonl`（`anchor.review-loop-data.v2`）。未配置
`review_verdict` 时主路由会直接阻断，绝不会把旧审查 LoRA 悄悄当成新契约。

| 组 | Adapter 设计 | 主要用途 |
| --- | --- | --- |
| A | 五阶段都不用 LoRA | 原生 Q4，指数基准 100 |
| B | 同一个 mixed rank-16 LoRA 复用五阶段 | 单 LoRA 混合基线 |
| C | 五个独立 rank-16 专家 | 最大容量 routed 对照，不与 B 做等参数结论 |
| D | 手工固定 `3/3/4/3/3`，总 rank/参数严格等于 B | B/D 等预算主对照 |
| E | 校准集上自适应非均匀分配；每阶段 rank `<=16`，总预算可变 | 容量—性能 Pareto 搜索 |
| F | 与 E 使用同一自适应算法，但总 rank/物化参数硬匹配 B | B/D/F 等预算主对照 |

因此公平预算的主结论来自 **B vs D vs F**；C 展示满容量 routed 上界，E 展示总预算可变的
Pareto 前沿。E/F 的 rank 只能在 calibration split 上冻结，不能窥视 held-out。

## 11. 架构与评测入口

- 五阶段 DAG 与 A/B/C/D/E/F 定义：`docs/architecture_five_stage_mvp.md`
- 评测控制变量：`docs/serving_benchmark.md`
- Provider 细节：`docs/teacher_providers.md`
- OpenCode session converter 的源码证据：`docs/opencode_session_distillation.md`
- 当前完成度和阻塞项：`docs/PROJECT_STATUS.md`
