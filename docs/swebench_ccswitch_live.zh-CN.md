# SWE-bench 全题库正式协调器

[English](swebench_ccswitch_live.md)

## WSL 路由监听地址探测

`sandbox_route_visibility` 固定为 `wsl-probed-host`。LIVE 启动不会再信任 WSL
默认路由的第一跳，因为 TUN 软件可能把 Windows 无法绑定的合成地址放在那里。协调器
优先探测 `127.0.0.1`。仅当某地址同时属于 WSL 默认网关和 Windows Preferred IPv4
时，才允许作为 NAT fallback；任意 WLAN/private 地址不会被暴露。每个候选都必须让
临时 Windows TCP socket 成功 bind/listen、让指定 WSL 发行版建立 TCP 连接，并由
Windows 监听器确实 accept。固定路由端口随后还要再次通过 Windows health 与 WSL TCP
检查。任一条件无法证明都会在 provider 请求之前 fail-closed。

这是面向 19,008 道纯 train 候选题和 95,040 份五阶段依赖工单的正式协调器，组合使用魔改 CC Switch 与魔改 OpenCode。它不使用 SWE-bench dev/test/Lite/Verified、gold patch、test patch、hint、heldout 正文或 oracle label。

## 零请求离线预检

```powershell
Set-Location D:\LLM\anchor-moe-lora
$Python = "$HOME\.conda\envs\anchor-mvp\python.exe"
& $Python scripts\tooling\run_swebench_ccswitch.py
# 等价统一入口
.\anchor.ps1 -Action distill-swebench
```

默认命令只读：发送 0 次模型请求，不读取凭据环境变量，不启动 CC Switch、OpenCode、沙箱、镜像获取或 GPU 任务。它分别报告：

- `component_ready`：固定版本的魔改二进制、清单和路由配置是否有效；
- `bank_ready`：题库数量、依赖、中英分配和五阶段工单是否完整；
- `execution_contract_ready`：持久化执行证明是否与当前本地 v3 重算结果完全一致；
- `live_start_allowed`：上述门禁是否全部通过。

`launch_ready` 仅为兼容字段，只表示组件和题库就绪，不代表 LIVE 获准。仅手写一个自称 `ready=true` 的 JSON 无法放行。

## 通用 train 仓库/提交自验证契约

公开 `train` 投影没有可用于训练时判分的官方 `test_patch`/TestSpec，因此正式蒸馏不伪造
“官方 PASS”。每道题都绑定公开仓库与准确的 `base_commit`，并按以下顺序执行：

1. 在隔离工作区物化准确仓库与提交；
2. 用不可变 digest 固定的通用 Python + Node train 沙箱执行魔改 OpenCode 工具轨迹；
3. 要求最后一次工具调用验证**最终**工作树，不能用“先验证、后修改”冒充成功；
4. 独立重算 changed paths、最终二进制 diff、未跟踪文件和 final-state 哈希；
5. 清理沙箱后，由受信 supervisor 签发内容哈希绑定的 HMAC 训练回执。

仓库/提交、题卡清单、容器 digest、OpenCode/CC Switch 身份、工具结果、最终 patch、验证状态、
清理结果或 HMAC 任一缺失/漂移都会 fail-closed。认证等级固定写为
`real_sandbox_self_verified` 和 `not_official_swebench_pass=true`。该 repo+commit 自验证执行
契约当前已 **READY**；READY 表示代码、镜像与离线证据通过，不表示已经调用过教师模型。

仓库执行沙箱使用 `--network=none --pull=never`；Provider 请求由沙箱外的固定协调器经魔改
CC Switch 发出。魔改 OpenCode 只接触当前任务的规范工作树。训练 HMAC key、supervisor
私有状态和验证回执密钥绝不进入 prompt、review、session export、日志或训练正文。

官方 heldout/TestSpec 评测是另一条协议：它尚未完成，不进入训练数据，也不阻塞 train
pilot、Resume 或 Gold 导出。只有后续显式评测才可以生成官方 heldout 结论。

## 检查固定版本依赖

检查或显式写入普通执行证明：

```powershell
& $Python scripts\tooling\build_swebench_execution_attestation.py
& $Python scripts\tooling\build_swebench_execution_attestation.py `
  --output artifacts/tooling/opencode-patched/multilang-execution-attestation.json
```

## 显式单题 Provider pilot

通用 train 执行契约 READY 后，真实 Provider 路径仍必须先用单题验证。pilot 只有在以下
内容全部绑定时才算成功：

- 最终 execution lock；
- OpenCode patch manifest、源 commit、bundle manifest 与 Linux 二进制；
- CC Switch route manifest；
- 代表任务的题卡、公开仓库、`base_commit` 与不可变 train 沙箱 digest；
- 最后一次工具调用结果、准确 final patch/state、成功清理和 supervisor HMAC 训练回执。

静态验证器会重新计算公开产物哈希，并要求 supervisor 在不导出密钥字节的前提下复核
最终状态和回执 HMAC。缺失或篡改仍然保持 BLOCKED。

重启电脑后，只有在你确定要消耗 provider/镜像流量时才执行：

```powershell
& $Python scripts\tooling\run_swebench_v3_representative_probe.py `
  --control-run-id operator-probe-0001 `
  --confirm-representative-live `
  --confirm-supervisor-network
```

这个窄入口要求静态代码、题库、路由、进程内 key 和 supervisor 前置条件全部通过；强制
并发 1、只跑 1 道 train 题，并使用独立运行目录。它会发出真实 Provider 请求，但绝不会
启动 19,008 题全量任务。只有完整五阶段、最终状态自验证、review/security、HMAC 和清理
均通过的 pilot 才进入认证成功前缀；这仍然不是官方 SWE-bench PASS。

本次仓库更新尚未运行真实 Provider pilot。执行 pilot 前，先确认零请求预检的 train 门禁
全部 READY；若仍有失败门禁，正式协调器会在 Provider 请求前拒绝：

```powershell
& $Python scripts\tooling\run_swebench_ccswitch.py --confirm-live `
  --control-run-id full-bank-0001
```

pilot 成功后只 Resume 同一个认证 checkpoint。下列任务上限都是累计值：

```powershell
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Concurrency 1  -MaxTasks 1
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 8  -MaxTasks 16
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 16 -MaxTasks 48
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 24 -MaxTasks 96
.\anchor.ps1 -Action distill-swebench -ConfirmLive -Resume -Concurrency 30 -MaxTasks 156
```

已经认证的成功前缀会在 Resume 时按哈希复核并跳过，可用于显式 partial Gold 导出；失败
或未完成任务保持可重试，绝不进入 Gold。稳定的系统性并发错误应触发停止并回退到上一档；
单个任务的验证失败只重试该任务。

## 五阶段记录契约

每道题固定顺序为：

`planner -> tool_policy -> domain_builder -> domain_review -> security`

9,504 个 `zh-CN` 是确定性语言分配，不代表翻译正文已经生成。实时本地化必须逐字保护代码块、行内代码、URL、路径、命令、JSON key 和工具名。

包含正文的 prompt、回复、真实工具调用/结果、OpenCode export 和 diff 只能写入 `artifacts/swebench/full-bank-live-v1/content-records/`。公开 status、checkpoint 元数据和代表性证明必须保持无正文并与哈希绑定。单独一个 security PASS 不能算任务完成；训练 Gold 必须具备严格完整五阶段轨迹、真实非平凡验证结果、成功清理，以及认证过的 `real_sandbox_self_verified` 训练回执。该回执不代表官方 SWE-bench PASS；heldout 评测保持独立。

## 证据边界

已经实现并通过离线测试：公开 repo+commit 物化、不可变通用 train 沙箱、最终状态独立复核、
HMAC 回执、并发/篡改 fail-closed、Resume 复核、成功前缀保留和 Gold 导出门禁。因此 train
自验证执行契约当前为 **READY**。

本次尚未产生的证据是：魔改 OpenCode + CC Switch 的真实 Provider 单题 pilot、19,008 题
全量 LIVE、正式 Gold 快照、formal-v3 A-F 训练，以及独立的官方 heldout/TestSpec 评测。
不得把“train READY”升级成这些结论；也不得反过来用 heldout 未完成阻塞 train pilot。
