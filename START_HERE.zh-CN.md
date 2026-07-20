# 从这里开始

[English](START_HERE.md) | [简体中文](START_HERE.zh-CN.md)

这是当前仓库最短、最不容易走错的操作入口。文档严格区分四种状态：代码存在、离线
检查通过、真实 LIVE 证据已生成、训练/评测已完成。

## 只需要记住三条命令

在 PowerShell 中运行。下面三条命令都不会读取 API key、发送 Provider 请求、启动 GPU
训练，也不会打开 heldout 题目正文。

```powershell
Set-Location D:\LLM\anchor-moe-lora

# 1. 紧凑只读状态。
.\anchor.ps1 -Action status

# 2. 启动或恢复本地控制面板。
.\anchor.ps1 -Action ui

# 3. 只运行正式全题库离线预检。
.\anchor.ps1 -Action distill-swebench
```

执行第 2 条后打开 <http://127.0.0.1:8765/>。页面默认选择 **Formal SWE-bench +
CC Switch + OpenCode**，可显示/控制 Start、安全 Stop、checkpoint Resume、并发、速度、
token/成本、重试、连接状态和精确的 fail-closed 原因。只有正式 LIVE 门禁通过后才需要
在密码框输入进程内 key；仅打开页面不会消耗额度。

## 正式数据主线

```text
公开 SWE-bench train 题库：19,008 张题卡 / 95,040 个工单
                               |
                               v
        planner -> tool_policy -> domain_builder -> domain_review -> security
                               |
                               v
     魔改 CC Switch 路由 + 魔改 OpenCode + WSL/Podman 通用 train 沙箱
                               |
                               v
       认证五阶段 Gold 导出 -> 不可变 formal-v3 训练快照
                               |
                               v
                  A / B / C / D / E / F 训练与评测
```

正式源固定为 `SWE-bench/SWE-bench` train revision
`7074ef12ea2a6f70a228943c1336553333c22786`。公开派生题库有 19,008 张题卡，
每题严格对应五个依赖绑定工单。9,504 个 `en-US` 与 9,504 个 `zh-CN` 目前是确定性的
语言路由分配，不能冒充“9,504 条中文正文已经生成”。

旧 c10、384+128 和 direct-API collector 仍可用于 synthetic 实验，但它们不是正式全题库
主线，`distill-swebench` 也绝不会失败后回退到 synthetic。

## 当前真实边界

| 部分 | 当前已经存在 | **尚未发生**的事情 |
| --- | --- | --- |
| 公开题库 | 19,008 张 train-only 题卡、95,040 个工单、确定性的 train/calibration allowlist；公开分片均小于 50 MiB | 官方 dev/test/heldout 题目正文不属于该题库 |
| Web 控制面 | 正式模式默认选中；生命周期、Provider profile、路由、门禁、速度、token/成本、重试、断连和原因码均已接入 | 页面不能绕过失败的正式门禁 |
| 魔改 CC Switch | 固定路由 profile、价格元数据、组件 manifest 和离线证明已存在 | 组件证明不等于真实 Provider pilot 已运行 |
| 魔改 OpenCode | 固定 source patch 和 v3 工具契约记录在 `patches/opencode/patch-manifest.json`；当前 patch `b61617124977d156f5702be23b46e7564325a4e796037e6faaa89ed42543106b` 已通过 clean apply 与离线契约检查 | 旧 binary bundle 不能因为“文件存在”就冒充当前 v3 patch |
| 通用 train 沙箱 | 题卡绑定的公开仓库与 `base_commit` 会在隔离工作区中物化；最后一次工具调用必须验证最终 diff/state，且训练 supervisor 会签发 HMAC 回执；该 repo+commit 自验证执行契约已 **READY** | 尚未运行真实 Provider 单题 pilot；`real_sandbox_self_verified` 明确不是官方 SWE-bench PASS |
| 官方 heldout 评测 | 与 train 蒸馏分离的正式评测契约仍保留 | 尚未完成；它不阻塞 train pilot、续跑或 Gold 导出，也绝不能进入训练数据 |
| 正式 Gold 桥接 | `scripts/data/export_swebench_formal_gold.py` 已实现从认证完整链 fail-closed 导出 | `artifacts/swebench/full-bank-live-v1/training-export/` 不存在，formal-v3 Gold 快照尚未发布 |
| A–F 训练 | 已实现按快照规模推导的 schedule 与 9 GiB/64-token 低显存 profile | 低显存 profile 不是 full-context；formal-v3 A–F 尚未训练 |
| Formal-v3 评测 | 已规定 A=100 与 A–F 绑定；可以做离线集成检查 | 当前不宣称存在 formal-v3 A–F heldout/GPU 评测结果 |

当前**通用 train 的 repo+commit 自验证执行契约已 READY**，但这不等于真实 LIVE 已发生。
本次更新尚未发出真实 Provider pilot 请求，也没有生成正式 Gold。操作者仍须先确认组件、
题库、执行、路由和 live-start 门禁全部通过，再在 WebUI 中为本次子进程输入 key；官方
heldout 评测状态不参与 train 启动门禁。

下一个真实动作是[快速上手第 4 节](QUICKSTART.zh-CN.md#4-先跑单题-pilot再逐档-resume)中的
`c1/cap1` 单题 pilot，而不是直接启动全题库。pilot 成功后，只续跑同一个认证 checkpoint：
`c8/cap16 -> c16/cap48 -> c24/cap96 -> c30/cap156`。任何已认证成功前缀都保持可用并在
Resume 时跳过；失败或未完成任务继续可重试，且绝不进入 Gold。

## A–F 就是六组都保留

| 组 | 控制变量结构 |
| --- | --- |
| A | 相同冻结原生 Q4 基座，无 LoRA、不训练；评测指数基准 100 |
| B | 一个 mixed rank-16 LoRA，在五阶段复用 |
| C | 五个独立满规格 rank-16 专家 |
| D | 五个固定小专家，rank 为 `3/3/4/3/3`，总预算严格对标 B |
| E | 五个 calibration 自适应专家，每个 rank `<=16`，总预算可变 |
| F | 与 E 使用相同自适应机制，但物化总参数量严格对标 B |

B–F 必须使用同一不可变快照，并保持总 sample exposure 和逐阶段 sample exposure 相等。
E/F 只能用 calibration-from-train 切分确定 rank。formal-v2 的 adapter、报告和 registry
不能冒充 formal-v3 证据。

## 不要从旧报告抄 OpenCode SHA

manifest 才是事实来源，当前 patch SHA 应动态读取：

```powershell
$OpenCodePatch = Get-Content patches\opencode\patch-manifest.json -Raw |
  ConvertFrom-Json
$OpenCodePatch.baseline_commit
$OpenCodePatch.patch_sha256
$OpenCodePatch.tool_contract_version
```

只有 bundle 的 source binding 与该 manifest 完全匹配，并通过验证/行为检查时，才能把它
认作当前构建。

## 接下来读哪一份

| 需要 | English | 简体中文 |
| --- | --- | --- |
| 完整操作手册 | [Quickstart](QUICKSTART.md) | [快速上手](QUICKSTART.zh-CN.md) |
| 正式总契约 | [Canonical contract](docs/CANONICAL_DISTILLATION_CONTRACT.md) | [正式契约](docs/CANONICAL_DISTILLATION_CONTRACT.zh-CN.md) |
| 全题库协调器 | [Coordinator guide](docs/swebench_ccswitch_live.md) | [协调器说明](docs/swebench_ccswitch_live.zh-CN.md) |
| A–F 训练 | [Formal-v3 training](docs/formal_v3_training.md) | [Formal-v3 训练](docs/formal_v3_training.zh-CN.md) |
| 控制面板 | [Dashboard guide](docs/distillation_dashboard.md) | [仪表盘指南](docs/distillation_dashboard.zh-CN.md) |
| 当前精确状态 | [Project status](docs/PROJECT_STATUS.md) | [项目状态（英文事实表）](docs/PROJECT_STATUS.md) |
| 全部公开文档 | [Documentation index](docs/README.md) | [文档索引](docs/README.zh-CN.md) |

凭据只进入当前进程。不得写入 YAML、`.env`、命令参数、日志、Git 或示例。未经新的明确
指令，不创建 Git tag，也不发布 GitHub Release。
