# OpenCode 工具验证黄金层

## 结论

真实工具调用可做，而且适合成为蒸馏数据的“黄金验证层”，但不应把整段 OpenCode 会话或隐藏思维链直接并入训练集。本实现让每个样本进入独立目录，只允许读写当前项目及三个确定的 npm 验证命令，并把最终结果压成不含提示词、模型正文、隐藏思维链、命令输出和密钥的 canonical JSONL。

### 2026-07-17 当前实机状态（以此为准）

下面的结论来自本机二进制、配置、WSL 服务和无额度 dry preflight 的重新核验；不要再沿用
2026-07-10 的“OpenCode 尚未安装”旧结论。

- 项目只使用固定产物，不依赖全局 `opencode`：Windows 入口是
  `artifacts/tooling/opencode-patched/opencode-anchor.exe`，WSL/Linux 入口是
  `artifacts/tooling/opencode-patched/linux-x64/opencode-anchor`。两者都能启动并暴露
  `anchor run`、`anchor export`、`anchor cleanup`。
- 来源与字节身份由 `patches/opencode/patch-manifest.json`、两个平台 manifest 和
  `artifacts/tooling/opencode-patched/bundle-manifest.json` 共同固定。显示出来的开发版
  `--version` 字符串不是来源证明，提交、补丁和 SHA-256 才是。
- WSL 发行版为 `Ubuntu-22.04`；WSL 内 Podman `3.4.4` 可用，`podman.socket` 为 active。
  当前配置使用 `wsl-root-systemd` 监督器，不是隔离更强的专用 WSL 发行版。
- 唯一批量配置入口为 `configs/tooling/opencode_distillation_ramp.yaml`。安全默认值是单沙箱、
  `4G` 内存、`2` CPU、`256` PID、`900` 秒。代码允许更高并发，但本机尚未验收 8 个 live
  沙箱；不要把“可配置 8”写成“已经跑通 8”。
- 在 `$HOME\.conda\envs\anchor-mvp\python.exe`（Python 3.11）下，下面的 dry preflight
  已通过，且 `opencode_available=True`。它不读取 key、不发教师请求、不启动 live 蒸馏：

  ```powershell
  Set-Location D:\LLM\anchor-moe-lora
  $Python = "$HOME\.conda\envs\anchor-mvp\python.exe"
  & $Python scripts\tooling\run_live.py `
    --batch-config configs\tooling\opencode_distillation_ramp.yaml
  ```

- **最重要的边界**：`anchor_mvp.data.automation`（包括当前
  `automation.full_v3.ark_kimi_k3.delta128.c8.yaml`）仍是 direct-API synthetic
  collector，不会调用 OpenCode，也不会产生真实 `tool_call -> tool_result` 或沙箱侧车。
  OpenCode/沙箱“已配置”不等于已经接入 Kimi-K3 新增 128 题，更不等于 512 题工具闭环完成。
- 当前 strict execution Gold 仍为 0。正式纳入训练前，必须把 SWE-bench/执行题卡接到
  `run_live.py`（或等价受控入口），取得 controlled export、workspace diff 和沙箱审计，再走
  去重、heldout、分区与冻结。

完整、持续更新的证据边界见 [PROJECT_STATUS.md](PROJECT_STATUS.md)；日常启动顺序见
[中文入口](../START_HERE.zh-CN.md) 和 [English entry](../START_HERE.md)。

## 官方能力核验

- OpenCode 官方 CLI 支持非交互 `opencode run ... --format json`；JSON 模式输出 raw JSON events，也支持 `opencode export [sessionID] --sanitize` 导出脱敏会话。黄金层使用前者，只保留安全元数据；不会保存会话正文。[OpenCode CLI](https://dev.opencode.ai/docs/cli/)
- OpenCode 自定义 provider 对 `/v1/chat/completions` 应使用 `@ai-sdk/openai-compatible`，并通过 `options.baseURL`、环境变量形式的 `apiKey` 配置。[OpenCode Providers](https://opencode.ai/docs/providers)
- OpenCode 的 agent `steps` 是最大 agentic 迭代数；旧的 `maxSteps` 已弃用。权限支持默认拒绝、工具级规则和 bash 命令精确白名单。[OpenCode Agents](https://opencode.ai/docs/agents/)
- OpenCode 可将 `share` 设为 `disabled`，并用环境变量禁用默认插件、Claude Code 配置继承和 LSP 自动下载。[OpenCode Config](https://dev.opencode.ai/docs/config), [OpenCode CLI environment](https://dev.opencode.ai/docs/cli/)
- Kimi Code 的 OpenAI-compatible Base URL 是 `https://api.kimi.com/coding/v1`，模型 ID 是 `kimi-for-coding`。[Kimi Code Overview](https://www.kimi.com/code/docs/en/)
- Kimi 官方明确要求第三方工具保持真实身份标识，篡改 User-Agent 可能导致权益暂停。因此 OpenCode 路线必须使用真实 OpenCode/CLI 身份，不能伪装成 Claude Code。只有实际使用 Claude Code 客户端时，Claude Code 请求头才是真实身份。[Kimi Code Overview](https://www.kimi.com/code/docs/en/)

## 安全边界

配置文件：`configs/tooling/opencode_kimi.example.json`；生成代码：`src/anchor_mvp/tooling/`。

默认策略如下：

- 每个样本复制到 `sample-id--随机后缀` 独立目录；输入中的符号链接直接拒绝，避免路径逃逸。
- 允许 `read/edit/glob/grep/list/bash`，拒绝 `external_directory/task/skill/webfetch/websearch/lsp`。
- bash 默认拒绝，只允许：`npm run build --if-present`、`npm run test --if-present`、`npm run lint --if-present`。
- 当前正式配置不写死 agent 任务步数上限；终止边界是外层进程和沙箱默认 `900` 秒超时，
  单个验证命令默认 `300` 秒超时。Windows 超时会终止 OpenCode 进程树。只有操作员在诊断时
  显式传入 `--max-iterations`，才额外限制迭代数。
- 会话分享关闭；默认插件、Claude Code 配置继承、模型目录拉取和 LSP 下载关闭。OpenCode 的 XDG config/data/cache 全部重定向到样本内临时目录，事件归约完成后删除，避免持久化原始会话和隐藏思维链。
- key 只允许从 Provider 配置声明的宿主进程环境变量读取（默认 Kimi 示例为
  `KIMI_CODE_API_KEY`，Ark/GLM 示例为 `ARK_CODING_API_KEY`）；进入沙箱时统一改名为一次性子进程
  环境变量。配置及 gold 文件中不出现 key，也不设置伪造客户端身份的 User-Agent/header。
- `package.json` 中没有对应脚本时结果记为 `SKIP`，不会把 npm 的 `--if-present` 零退出码误报为真实通过。默认要求 `build` 必须存在且通过；可按样本要求 `test`、`lint` 也必须通过。

注意：当前已经叠加 WSL/Podman、无宿主凭据、资源限制和受控挂载，但仍是开发隔离层，
不是可承载恶意多租户代码的强安全边界。尚缺每 workspace 磁盘配额、强制 Provider-only
egress、专用禁用互操作的 WSL 发行版，以及真实崩溃/reaper 覆盖。

## 审计与 canonical gold JSONL

执行记录分为两层：`live_attempts.jsonl` 全量保留成功、失败和部分执行，
`live_gold.accepted.jsonl` 只保留 `success=true` 且公开结果状态为 `completed` 的可接受记录。
两者每行都按 key 排序并紧凑 JSON 编码，写入采用原子替换。记录：

- workspace ID、backend、成功状态、timeout、最大迭代、agent 退出码；
- build/test/lint 是否存在、退出码、耗时、输出 SHA-256；
- tool trace 的工具名、白名单命令、退出码、耗时；
- 非白名单命令只保留 SHA-256，不保存原命令；
- agent 修改文件的相对路径及修改前后 SHA-256；
- 结构化错误码，如 `invalid_url`、`client_cancelled`、`rate_limited`。

明确不保存：API key、完整环境变量、prompt、模型回复、thinking block、任意工具输出、源文件内容。OpenCode stdout/stderr 只在进程内完成归约，gold 中仅保存摘要哈希。

修改型任务必须在候选清单中显式设置 `requires_changes: true`；若 agent 没有产生
文件差异，attempt 会记录 `no_changes`，且不会进入 accepted gold。失败 attempt 不会
占用 accepted gold 的 `sample_id`，因此修复后的成功重试仍可入库。

旧版 `live_gold.jsonl` 可能混有失败 attempt。运行
`scripts/tooling/migrate_legacy_tool_gold.py` 默认只预览分类；只有显式增加 `--confirm`
才会创建新的 attempts/accepted 迁移文件。脚本不会删除或覆盖旧文件，迁移后的路径
切换必须由操作者另行确认。

## Kimi 路由预检（Windows / WSL）

`run_live.py` 的 live 模式会先解析 `api.kimi.com` 的全部 IPv4 地址，并在配置的
WSL distro 内逐个执行 `ip -j -4 route get <ip>`，只输出 `gateway/dev/src`、是否命中
`198.18.0.0/15` 或明显 TUN/虚拟网卡，以及可用的非 TUN default route。这个阶段不调用
Kimi API，不读取或打印 key；传给 `wsl.exe` 的环境也会删除常见 API key 变量。

新增参数 `--route-mode prompt|current|direct|abort`，默认 `prompt`：

- `prompt`：仅交互终端可用，展示审计结果后选择保持当前路由、仅 Kimi `/32` 临时直连或终止。
- `current`：显式接受当前路由，只读审计，绝不执行 route replace/delete。
- `direct`：为每个真实 Kimi IPv4 临时执行
  `ip -4 route replace <ip>/32 via <gateway> dev <dev>`；不会修改 default route 或 ip rule。
  进入 live run 前会再次 `route get` 验证，退出时恢复原有精确 `/32` route；原先没有 host
  route 时只删除本次新增的 `/32`。
- `abort`：在 DNS、WSL、OpenCode 和 API 请求前直接终止。

非 TTY 环境使用默认 `prompt` 会 fail closed，自动化必须显式传 `--route-mode current`、
`--route-mode direct` 或 `--route-mode abort`。batch 只建立一个 route guard，覆盖整个并发
stage，不会让 worker 反复修改 WSL 全局路由。若 DNS 本身返回 `198.18.0.0/15` fake-IP、
找不到唯一的非 TUN default route、替换后仍被 policy routing 导回 TUN，live run 都会拒绝。

示例：

```powershell
py -3.10 scripts/tooling/run_live.py --batch-config configs/tooling/opencode_distillation_ramp.yaml `
  --max-stages 1 --route-mode direct --confirm-live
```

临时恢复依赖 Python `try/finally`；进程被强制终止、WSL 或系统崩溃时无法保证执行恢复。
direct guard 使用主机临时目录中的独占锁；第二个 direct 实例会以 `route_lock_busy` 拒绝，
而不是覆盖第一个实例的快照。异常硬退出后仍应先用
`ip -4 route show exact <kimi-ip>/32` 人工核对，再开始下一次 live run。

## 400 / 499 处理

- 400 `invalid_url`：Kimi 工具错误通常是模型把描述性文字当 URL，或缺少协议。本层默认禁用 `webfetch/websearch`；Kimi Base URL 还会在启动前经过严格 HTTPS URL 校验，含空格、缺 scheme、query/fragment 的地址直接拒绝，不发送请求。
- 499 `context canceled`：这是客户端取消、网络中断或 wrapper timeout，不应当算服务端模型失败。runner 将进程树终止并记录 `timed_out=true`、`wrapper_timeout`；若事件文本中出现 499/context canceled，另记 `client_cancelled`。
- 不要对 499 无限重试；应先区分人工取消、外层超时和 WSL 网络异常。对 429 才按额度窗口退避。

## 离线验证

```powershell
py -3.10 scripts/tooling/run_mock.py
py -3.10 -m pytest tests -k tooling -q
```

mock 会创建隔离样本、修改一个文件、真实执行本机 npm build/test/lint 空载脚本，并生成 `artifacts/tooling/mock_gold.jsonl`。它不会启动 OpenCode 或调用 Kimi。

## 安装前检查与 dry-run

OpenCode 官方 Windows 文档支持 npm 安装并更推荐 WSL。当前未安装，先运行：

```powershell
scripts/tooling/check_opencode.ps1
```

该脚本只检查路径、版本、npm registry/prefix，并打印官方安装命令，不安装。若要让 npm 仅解析包元数据且不执行 package scripts：

```powershell
scripts/tooling/check_opencode.ps1 -RunNpmDryRun
```

它等价于 `npm install -g opencode-ai --dry-run --ignore-scripts`；仍可能访问 npm registry，但不会全局安装。只有用户明确授权后，才执行官方安装命令 `npm install -g opencode-ai`。[OpenCode Install](https://opencode.ai/en/docs)

## 单样本 live 入口

先审查 source、prompt 和 policy。未带确认参数时只做 dry-run 检查：

```powershell
py -3.10 scripts/tooling/run_live.py `
  --sample-id frontend-0001 `
  --source C:\path\to\audited-sample `
  --prompt-file C:\path\to\prompt.txt
```

真跑前，用进程级临时环境变量提供 key；不要写入 `.env`、JSON、命令行参数或 shell 历史。确认后增加 `--confirm-live`。入口一次只允许一个样本，避免误把额度瞬间打满；并行度由更外层调度器控制，每个并发任务仍必须使用独立目录。建议黄金工具验证先保持最多 8 并发，只有纯 API 蒸馏才考虑更高并发，并对 429/499 分别处理。

若确实要保留 OpenCode 原始会话用于人工取证，可单独运行官方 `opencode export <sessionID> --sanitize`，但导出文件不是训练集 gold，也不得进入自动蒸馏管线。
