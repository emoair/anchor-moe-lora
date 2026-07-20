# Anchor-MoE-LoRA 简体中文快速上手

[English](QUICKSTART.md) | [简体中文](QUICKSTART.zh-CN.md)

本文只讲正式全题库主线，不把历史 synthetic collector 冒充正式证据。

## 0. 先认清当前边界

正式语料是固定版本的 `SWE-bench/SWE-bench` **train** 投影：

- 19,008 张公开题卡；
- 95,040 个依赖绑定工单，每题严格五个；
- `planner -> tool_policy -> domain_builder -> domain_review -> security`；
- 9,504 个 `en-US` 和 9,504 个 `zh-CN` 路由分配。

语言路由不等于中文翻译已完成，题卡也不等于 Gold。只有真实魔改 OpenCode 工具轨迹、
review/security PASS、至少一条非平凡公开验证命令成功、清理成功，以及 HMAC 认证的训练
回执都绑定同一个最终 patch，该题才能进入正式 Gold。其证据等级是
`real_sandbox_self_verified`，明确不声称官方 SWE-bench PASS。

截至 2026-07-18 的证据刷新：

- 公开题库、正式 WebUI、协调器、Gold 导出器、不可变快照桥接、A-F schedule 代码和
  formal-v3 评测集成均已实现并通过离线测试；
- 当前 OpenCode v3 patch 能对固定上游 clean apply，并通过离线契约测试；它的唯一
  正式身份来自 `patches/opencode/patch-manifest.json`；
- 通用 train 沙箱会物化题卡绑定的公开仓库与 `base_commit`，在最终工具调用后独立复核
  最终 diff/state 并签发 HMAC 训练回执；该 repo+commit 自验证执行契约已 **READY**；
- 该训练回执只表示 `real_sandbox_self_verified`，不表示官方 SWE-bench PASS；官方
  heldout/TestSpec 评测保持独立、尚未完成，并且不阻塞 train 蒸馏；
- 尚未运行任何真实 Provider 单题 pilot；
- 尚未完成 19,008 题正式 LIVE；
- `artifacts/swebench/full-bank-live-v1/training-export/` 不存在；
- formal-v3 A-F 训练和 heldout/GPU 评测均未执行。

因此当前可以确认的是“train 执行契约 READY”，不是“真实蒸馏已经完成”。第一次真实
动作仍必须是单题 pilot；在 pilot 产出认证回执前，不得声称已有 LIVE Gold。输入 key
不能绕过任何仍失败的组件、题库、路由或 checkpoint 门禁。

## 1. 在仓库目录打开 PowerShell

```powershell
Set-Location D:\LLM\anchor-moe-lora
```

`anchor.ps1` 按以下顺序寻找 Python 3.10+：`-PythonExe`、
`ANCHOR_MVP_PYTHON`、`.venv`、`anchor-mvp` Conda 环境、`py -3.11`。在本机当前
PowerShell 窗口固定已知环境：

```powershell
$env:ANCHOR_MVP_PYTHON = "$HOME\.conda\envs\anchor-mvp\python.exe"
& $env:ANCHOR_MVP_PYTHON -c "import sys; print(sys.executable); print(sys.version)"
```

如果该文件不存在，先建立/激活等价的 Python 3.10+ 环境并安装本仓库。不要把 Provider
key 写进 YAML、`.env`、命令行参数或 Git 文件：

```powershell
conda create -n anchor-mvp python=3.11 -y
conda activate anchor-mvp
python -m pip install -e ".[teacher,dev]"
$env:ANCHOR_MVP_PYTHON = (Get-Command python).Source
```

只有真正承担后续训练/推理的机器才需要额外安装 `training` 和 `serving` extras。

## 2. 第一次只运行三条零额度命令

下面三条命令不会读取 Provider key，不会请求模型，不会启动 GPU 训练，也不会打开
heldout 题目正文。

```powershell
# 紧凑只读状态。
.\anchor.ps1 -Action status

# 启动或重新连接本地控制面板。
.\anchor.ps1 -Action ui

# 只运行正式协调器离线门禁。
.\anchor.ps1 -Action distill-swebench
```

执行 `-Action ui` 后打开 <http://127.0.0.1:8765/>。页面默认选中 Formal
SWE-bench，显示组件、题库、执行、镜像/路由、本地化和 live-start 门禁及精确原因码；
同时提供并发、重试/重连、安全 Stop、checkpoint Resume、吞吐、ETA、请求/token/成本和
后端断连状态。

当前预期应看到通用 train 的 `execution_contract_ready` 为 **READY**，同时明确看到
“尚无真实 Provider pilot”的事实。其他组件、题库或路由门禁若失败，Start 仍保持禁用；
WebUI 不能把失败门禁改成成功。官方 heldout 评测是独立的非阻塞状态，不参与 train
Start 判定。

## 3. 核验魔改 OpenCode 与 CC Switch 身份

不要从旧报告复制 OpenCode 哈希，始终读取当前 manifest：

```powershell
$OpenCodePatch = Get-Content patches\opencode\patch-manifest.json -Raw |
  ConvertFrom-Json
$OpenCodePatch.baseline_commit
$OpenCodePatch.patch_sha256
$OpenCodePatch.tool_contract_version
```

当前 manifest 固定 OpenCode 1.17.18 上游和
`anchor.execution-tool-contract.v3`。只有 binary bundle 绑定同一份 source manifest，且
验证/行为测试通过时，才能视为当前构建。

魔改 CC Switch 负责 Provider/模型切换、精确 MAX 档位、计价和请求/token/成本统计。
正式 profile 是 GLM-5.2 MAX 与 Kimi-K3 MAX。组件 manifest 显示 ready，不等于
WSL/Podman 沙箱已在真实任务中成功连到该路由。

更长的离线组件诊断命令：

```powershell
.\anchor.ps1 -Action preflight -AllowIncomplete
```

`-AllowIncomplete` 只用于诊断，绝不授予 LIVE 权限。

## 4. 先跑单题 pilot，再逐档 Resume

通用 train 自验证链不依赖官方 heldout/TestSpec。它在隔离沙箱中检出题卡绑定的公开
仓库与 `base_commit`，只接受最后一次工具调用对最终 diff/state 的有效验证，再由训练
supervisor 复核并签发 HMAC 回执。仓库/提交、容器镜像、工具轨迹、最终 patch、验证结果
与清理状态任一不匹配都会 fail-closed。

本次仓库更新**尚未运行真实 Provider pilot**。推荐使用 WebUI：确认 train 的组件、题库、
执行、路由和 live-start 门禁全部 READY 后，才在密码框输入本次操作的 key。key 只传给
所选子进程环境，浏览器密码框随后清空；它不会写入配置、状态、日志或 Git。官方 heldout
评测即使仍显示“未完成”，也不阻塞 train pilot。

如果使用 CLI，key 只放在当前 PowerShell 进程。任务上限是同一个 checkpoint 的**累计值**：

```powershell
$env:ARK_CODING_API_KEY = '<只为当前 PowerShell 粘贴>'

# 1. 单题、并发 1 的真实 pilot。
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Concurrency 1 -MaxTasks 1

# 2. pilot 认证通过后，只 Resume 同一个 checkpoint，逐档扩大累计上限。
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 8  -MaxTasks 16
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 16 -MaxTasks 48
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 24 -MaxTasks 96
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 30 -MaxTasks 156

# 子进程退出后清除当前进程变量。
Remove-Item Env:ARK_CODING_API_KEY -ErrorAction SilentlyContinue
```

每一档结束后先检查认证成功数、失败原因、请求/token 增量、速度和 checkpoint 身份，再继续
下一档。已经通过完整五阶段、最终状态复核、HMAC 和清理检查的成功前缀会在 Resume 时按
哈希复核后跳过，始终可供后续 partial Gold 导出；失败或未完成任务保持可重试，绝不进入
Gold，也不会污染成功前缀。出现稳定的系统性并发错误时回到上一档；单个难题的验证失败只
重试该题，不应伪装成成功。

## 5. 先导出认证 Gold，再冻结快照

只有协调器进入终态后才执行。下面命令不会请求 Provider，也不会启动 GPU 训练。

```powershell
$Python = $env:ANCHOR_MVP_PYTHON

# 只导出完整且自验证训练回执通过 HMAC 认证的五段链。
& $Python scripts\data\export_swebench_formal_gold.py

# 发布不可变 formal-v3 数据快照。
& $Python scripts\data\prepare_full_v3_snapshot.py `
  --config configs\orchestration\full_v3_snapshot.yaml

Test-Path artifacts\swebench\full-bank-live-v1\training-export\partitions\manifest.json
Test-Path artifacts\formal_v3\dataset\manifest.json
```

五阶段链、真实工具轨迹、有效验证结果、diff、清理、review/security 决策、通用 train
沙箱镜像与 repo+commit 绑定或私有训练回执缺失时，该题会被剔除；认证产物一旦损坏或
被篡改则整次导出 fail-closed。
已封顶的 `stopped_checkpoint_resumable` checkpoint 可以导出，已认证成功样本保持可用，未完成
样本继续可重试。导出不声称官方 SWE-bench PASS，也不保留隐藏思维字段。绝不能直接从仍在
增长的 LIVE 输出目录训练。

中间封顶 checkpoint 必须用全新的版本化 `--output-dir`；导出目录不可变。默认规范
`training-export` 路径应保留给真正准备冻结的快照。封顶 manifest 可作为显式 partial
数据集使用，但不能声称全题库已完成。

任一 `Test-Path` 为 `False` 就停止。不得拿历史 `data/automated_v3`、c10、384+128、
mock 或 partial-Gold 目录替代。

## 6. 物化并训练完整 A-F 对照

正式矩阵如下：

| 组 | 控制结构 |
| --- | --- |
| A | 冻结原生 Q4 基座，无 LoRA、不训练；评测指数基准 100 |
| B | 一个 mixed rank-16 LoRA，在五阶段复用 |
| C | 五个独立满规格 rank-16 专家 |
| D | 五个固定小专家，rank 为 `3/3/4/3/3`，总预算严格对齐 B |
| E | 五个 calibration 自适应专家，每个 rank `<=16`，总预算可变 |
| F | 与 E 相同的自适应机制，但物化总参数量严格对齐 B |

B-F 必须使用同一个不可变快照，并保持总 sample exposure 和逐阶段 sample exposure 相等。
E/F 只能根据 calibration-from-train 确定并冻结 rank，不能先看 heldout。

先运行只生成 schedule 的预检：

```powershell
.\scripts\train\formal_v3_preflight.ps1
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm A
```

长训练前必须先做同一快照上的资源 smoke/probe：

```powershell
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm smoke -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm probe -Execute
```

只有 `-Execute` 会启动 GPU job。单 GPU 一次只跑一组：

```powershell
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm B -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm C -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm D -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm E -AllocationManifest <E.json> -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm F -AllocationManifest <F.json> -Execute
```

本地低显存 profile 目标约 9 GiB，但序列截断为 64 token；它明确不是 full-context 训练。
B–F 统一使用保守的一轮 `5e-5` 学习率，不允许某一组私下使用不同超参。后续更高精度
云端 profile 必须作为独立版本实验。

## 7. A-F 评测不得复用 formal-v2 产物

六组 formal-v3 registry 完整后，使用全新的 VersionId。第一组命令仅离线预检，不读取
heldout 正文：

```powershell
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001 -Finalize
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001
```

只有显式授权的命令才允许打开 heldout 并启动 GPU 评测：

```powershell
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 `
  -Execute `
  -AuthorizeHeldoutAccess
```

`-Resume` 只能续跑完全相同版本/输出 checkpoint。A 归一化为 100；B 是单一 mixed
adapter；C-F 是五阶段串行 runtime-LoRA 热切换。formal-v1/v2 的 adapter、报告和 registry
都不是 formal-v3 证据。

## 8. 快速排障

- **Start 禁用/BLOCKED：**看正式原因码，不要粘贴 key，也不要切换到 synthetic。
- **网页显示与后端断开：**运行 `.\anchor.ps1 -Action ui`，再刷新
  `http://127.0.0.1:8765/`。
- **HTTP 400 `invalid_url`：**工具收到自然语言或缺少 `http://`/`https://` 的地址；修复
  真正 URL 字段，不要重试错误文本。
- **HTTP 499/context canceled：**客户端、网络或超时在完成前取消请求；核对 checkpoint、
  路由和 timeout，再精确 Resume，不能另起重复任务。
- **token/成本未知：**面板只报告 Provider 返回的精确 usage，不会按文本长度偷偷估算。
- **中文数量是 9,504 但本地化缺失：**这是路由分配，不是中文正文已生成的证明。
- **训练预检 BLOCKED：**核对认证 training-export 和不可变快照；不能为了让命令跑起来
  而降低门禁。

## 9. 历史 collector 与发布边界

历史 synthetic 实验仍有显式入口：

```powershell
.\anchor.ps1 -Action distill-synthetic -ConfirmLegacySynthetic
```

它会直接调用 CompatibleTeacher，不会产生 formal-v3 所需的真实魔改 OpenCode 工具/
评测证据。旧 c10 与 384+128 配置只是历史对照，不是全题库失败后的回退方案。

公开 Git 只能包含源码、配置、文档、审计过且单文件小于 50 MiB 的公开 train 投影，以及
不含正文的 manifest/audit。必须排除 Provider key、私有 HMAC key/回执、未获准公开的
教师/session 正文、heldout 正文、权重、adapter、checkpoint、runs、日志和私有评测记录。

没有新的明确指令，不得创建 Git tag、GitHub Release 或版本包。
