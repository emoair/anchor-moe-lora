[English](teacher_providers.md) | [简体中文](teacher_providers.zh-CN.md)

# 教师模型服务商与模型选择

数据子系统支持 OpenAI 兼容的 Chat Completions 与 Anthropic 兼容的 Messages
协议。服务商配置只保存“装有凭据的环境变量名”，会拒绝 `api_key`、`token`、
`secret` 和 `authorization` 字段。项目不读取 `.env`，也不会把凭据值写入配置、
provenance、状态或日志。

本地控制面不绑定某一家服务商。目录预设只是快捷填写：基础 URL、协议、精确模型
ID、推理开关/强度、并发数、请求重试、进程重连和预算都可由操作者修改。模型发现
不是必需步骤；当服务商不支持 `GET .../models` 或发现失败时，勾选“强制使用手动
模型”并填写精确模型 ID。发现失败绝不允许控制面擅自替换模型。

当前正式教师配置中，GLM 5.2 与 Kimi-K3 的每个阶段都明确使用
`reasoning_effort: max`。生成的运行配置会记录这个精确值。若正式 MAX 配置选择更低
强度或关闭推理，启动会默认阻断，而不是静默降级。自定义/非正式配置可以选择
`low`、`medium`、`high` 或 `max`。

## 预设

| 预设 | 协议 | 官方/默认基础 URL | 默认模型 | Key 环境变量 |
| --- | --- | --- | --- | --- |
| `kimi-code-openai` | OpenAI | `https://api.kimi.com/coding/v1` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-code-anthropic` | Anthropic | `https://api.kimi.com/coding/` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-platform-openai` | OpenAI | `https://api.moonshot.cn/v1` | 手动/发现 | `MOONSHOT_API_KEY` |
| `openai` | OpenAI | `https://api.openai.com/v1` | 手动/发现 | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `https://api.anthropic.com` | 手动/发现 | `ANTHROPIC_API_KEY` |
| `custom-openai` | OpenAI | 配置中必填 | 手动/发现 | `TEACHER_API_KEY` |
| `custom-anthropic` | Anthropic | 配置中必填 | 手动/发现 | `TEACHER_API_KEY` |

Kimi Code 官方文档规定了两套基础 URL，以及稳定模型 ID `kimi-for-coding` 和
`kimi-for-coding-highspeed`。普通预设故意默认使用 `kimi-for-coding`；只有订阅允许时
才手动选择 HighSpeed。

## 发现、选择或强制指定模型

先在当前 PowerShell 进程设置 key，再列出模型；该命令不会发起生成请求：

```powershell
$env:TEACHER_API_KEY = "your-key"
py -m anchor_mvp.data models --provider custom-openai `
  --base-url https://gateway.example.com/v1 `
  --api-key-env TEACHER_API_KEY
```

发现过程调用协议标准端点：OpenAI 兼容协议使用携带 Bearer 认证的
`GET <base>/models`；Anthropic 兼容协议使用携带 `x-api-key` 与
`anthropic-version` 的 `GET <base>/v1/models`。输出会排序并从零编号。指定模型运行：

```powershell
py -m anchor_mvp.data run --config configs/data/provider.custom.example.yaml `
  --model provider-model-id --force-model
```

`--force-model`（或 `force_model: true`）会跳过发现。不强制时，
`discover_models: true` 只记录以下公开类别之一：`success`、
`missing_credential`、`auth_error`、`rate_limited`、`unsupported`、
`server_error`、`network_error` 或 `invalid_response`。发现失败不会阻止手动模型。
非交互选择只有在发现成功后才能使用 `model_index: N`；服务商排序可能变化时，显式
`model` 更可靠。

基础 URL 必须是带主机名的绝对 `http://` 或 `https://` URL。请求前会拒绝空白、
自然语言描述、内嵌凭据、查询字符串、fragment，以及完整的 `/models`、`/messages`
或 `/chat/completions` 端点。

每条蒸馏记录都会保存不含秘密的服务商 provenance：预设、选中模型与选择来源、
已验证基础 URL、实际协议、发现类别和模型数量。

## 可选额度查询

额度查询只提供信息，绝不会阻塞蒸馏：

```powershell
py -m anchor_mvp.data quota --provider kimi-platform-openai
```

Kimi Open Platform 公开了 `GET /v1/users/me/balance`，因此只有
`kimi-platform-openai` 实现 `moonshot_balance`。Kimi Code 会员文档要求在 Console
查看剩余额度和限流状态，没有稳定的公开额度端点，所以 Kimi Code 预设返回
`unsupported`。OpenAI、Anthropic 和自定义预设同样返回 `unsupported`，除非将来
明确实现稳定的官方端点。查询错误返回 `error`，不会改变生成行为。

## 旧配置迁移

原有的 `protocol`、`base_url`、`model`、`api_key_env` 扁平配置仍可用；缺少
`provider` 时会解释成匹配的 Kimi Code 预设。可以显式加入
`provider: kimi-code-anthropic` 或 `provider: kimi-code-openai`，再加入
`force_model: true` 保持原先固定模型/不发现行为。除非同步更新启动脚本，否则不要
重命名进程环境变量。

## 本地面板与网络路由语义

运行 `./anchor.ps1 ui`，再打开 `http://127.0.0.1:8765/`。凭据框只存在于内存/
进程中；开始、继续或模型发现后浏览器会清空它。子进程只通过
`ANCHOR_CONTROL_API_KEY` 收到所选凭据；YAML、JSON、argv 和日志都不会写入它。

面板默认的“直连”只表示“不继承代理 URL 环境变量，并设置 `NO_PROXY=*`”。它不会
把 socket 或进程绑定到物理网卡，也不能覆盖操作系统的 TUN/默认路由。面板会显示
代理/TUN 默认路由检测结果，并且绝不把此模式标成“已锁定物理网卡”。国内服务商和
大体积（尤其 10 GiB 以上）下载必须使用仓库专用的路由/下载预检，并在传输前确认
目标地址绑定的是启用中的物理适配器；无法证明物理直连时应停止，而不是宣称没有走
代理。

Windows 上的 CC Switch 路由组件与 WSL/Podman 可达性是两道独立门禁。哈希证明通过的
路由可执行文件只能标为 `component_ready`；正式协调器还必须从沙箱一侧证明路由可达，
之后才能称为端到端就绪。绝不能根据 Windows 上的 `http://127.0.0.1:...` 健康检查
推断容器可达。

完整题库清单中的英文/中文 9,504/9,504 仅描述语言路由。zh-CN localization manifest
生成并校验通过之前，中文正文尚未完成；缺少该运行输出会让 `training_ready=false`，但
不会阻止初始的 `launch_ready=true` 组件门禁。

## 主要文档

- [Kimi Code 服务端点、模型 ID 与 Console 额度说明](https://www.kimi.com/code/docs/en/)
- [Kimi Open Platform 余额端点](https://platform.kimi.com/docs/api/balance)
- [OpenAI 模型列表端点](https://developers.openai.com/api/reference/resources/models/methods/list)
- [Anthropic 模型列表端点](https://platform.claude.com/docs/en/api/models/list)
