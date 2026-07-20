# 双语发布规范与控制面板 i18n 检查表

[English](documentation_i18n.md) | [简体中文](documentation_i18n.zh-CN.md)

## 发布合同

- 英文使用 `name.md`；简体中文使用 `name.zh-CN.md`。
- 每一对文档开头都放置互相指向的 `English | 简体中文` 链接。
- 新增或实质修改的公开文档，只有在同一次变更中同时更新两个文件后才达到发布条件。
- 两个版本保持相同的章节顺序、安全边界、命令、代码标识、schema 名称、model ID、URL、
  hash、数字与许可证事实。只翻译说明，不翻译可执行值或机器可读值。
- 使用 UTF-8、有效 Unicode 和兼容 LF 的 Markdown。不得提交乱码或依赖本机 locale 的命令输出。
- 相对链接必须从各自文件正确解析。被链接目标有双语版本时，应尽量链接到同语言版本。
- 必须选择精确措辞时，英文是 schema reference language；中文必须表达相同行为，不能弱化门禁。
- 历史内部笔记可在成为 release-facing 内容前暂时保持单语；新的公开入口不适用该例外。

## 控制面板检查表

此检查表在蒸馏 dashboard 变为可交互控制面板时适用。它不授权存储或暴露 credential。

- 提供 `English` / `简体中文` 选择器；缺少翻译时确定性回退到英文。
- 只持久化 locale preference。API key、authorization header 或未脱敏 Provider error 绝不进入
  翻译状态、browser storage、URL，也不能在提交后留在 DOM 中。
- 本地化可见 label、help、validation summary、empty state、button、status name 和
  accessibility text。API field name、event type、model ID、path、CLI flag 与原始 Provider
  code 保持不变。
- 覆盖 `Start`、`Stop`、`Continue`、外部管理/只读任务、concurrency、requests per minute、
  wire attempts per minute、tokens per second、rows per minute、ETA、retry、cooldown 与
  unknown pricing。
- 两种语言都要明确区分 graceful stop 与 failure、resume 与新建 shard，以及“只监控外部进程”
  与 panel-owned worker。
- 存储中的原始结构化日志绝不翻译。界面在稳定 code 旁显示本地化解释，并保留原 code 供诊断。
- 按 locale 格式化展示用 timestamp、decimal separator 和 unit；API 仍输出 ISO 8601、整数
  token count，以及显式 currency/unit field。
- 两种语言都测试窄屏和宽屏。中文不得被裁切；英文控件必须容纳文本扩展。不能只用颜色表达含义。
- Host/Origin、CSRF/session、localhost binding、path validation 与 secret redaction 行为在
  所有 locale 下必须完全相同。
- 切换 locale 不得启动、停止、继续 worker，也不得产生其他状态变更。

## 术语表

| 稳定英文术语 | 简体中文显示 | 规则 |
| --- | --- | --- |
| control panel | 控制面板 | UI 名词；`control_plane` 标识保持不变。 |
| distillation | 蒸馏 | 不翻译 dataset/schema 标识。 |
| teacher model | 教师模型 | 显示名旁保留精确 model ID。 |
| planner | 规划专家 | 代码、配置和 trace 中保留 `planner`。 |
| tool policy | 工具策略 | 代码、配置和 trace 中保留 `tool_policy`。 |
| domain builder | 领域构建专家 | schema 中保留 `domain_builder`。 |
| domain review | 领域审查 | schema 中保留 `domain_review`。 |
| security gate | 安全门禁 | 按定义保留 `security`/`security_gate` 标识。 |
| task card | 题卡 | 不翻译 `card_id`。 |
| work order | 工作单 | 不翻译 `record_id` 或 `alignment_id`。 |
| shard | 分片 | 新的 effective configuration 必须使用新 shard。 |
| resume | 继续/断点恢复 | 不得暗示改变配置后可继续同一 shard。 |
| graceful stop | 平滑停止 | 与 cancel、crash 和 forced termination 区分。 |
| held-out | 留出集（held-out） | 首次使用时保留 `held-out`，避免改变 benchmark 含义。 |
| fail closed | 默认阻断（fail closed） | 不能弱化为仅警告。 |
| request rate | 请求速率 | 显式显示单位，例如 requests/min。 |
| wire attempt | 传输尝试 | retry 按独立 network attempt 计数。 |
| unknown price | 价格未知 | 不能显示为零成本。 |
| external/read-only | 外部进程/只读 | Panel 不能控制 attach 的外部 worker。 |
