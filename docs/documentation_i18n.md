# Bilingual publication policy and control-panel i18n checklist

[English](documentation_i18n.md) | [简体中文](documentation_i18n.zh-CN.md)

## Publication contract

- English uses `name.md`; Simplified Chinese uses `name.zh-CN.md`.
- Each pair starts with reciprocal `English | 简体中文` links.
- A new or materially changed public document is release-ready only when both
  files are updated in the same change.
- Both versions preserve the same section order, safety boundaries, commands,
  code identifiers, schema names, model IDs, URLs, hashes, numbers, and license
  facts. Translate explanations, not executable or machine-readable values.
- Use UTF-8, valid Unicode, and LF-compatible Markdown. Do not check in mojibake or
  locale-dependent command output.
- Relative links must resolve from each file. If a linked target has a bilingual
  pair, link to the matching language where practical.
- English is the schema-reference language when an exact wording must be chosen;
  the Chinese version must state the same behavior and may not weaken a gate.
- Historical internal notes may remain single-language until they become
  release-facing. New public entry points do not receive that exception.

## Control-panel checklist

This checklist applies when the distillation dashboard becomes interactive. It
does not authorize storing or exposing credentials.

- Provide an `English` / `简体中文` selector and use English as the deterministic
  fallback for missing strings.
- Persist only the locale preference. Never place API keys, authorization headers,
  or unredacted provider errors in translation state, browser storage, URLs, or
  the DOM after submission.
- Localize visible labels, help, validation summaries, empty states, buttons,
  status names, and accessibility text. Keep API field names, event types, model
  IDs, paths, CLI flags, and raw provider codes unchanged.
- Cover `Start`, `Stop`, `Continue`, externally managed/read-only jobs,
  concurrency, requests per minute, wire attempts per minute, tokens per second,
  rows per minute, ETA, retries, cooldown, and unknown pricing.
- Distinguish graceful stop from failure, resume from a fresh shard, and
  “monitor-only external process” from a panel-owned worker in both languages.
- Never translate raw structured logs in storage. Render a localized explanation
  beside the stable code and keep the original code available for diagnosis.
- Format display timestamps, decimal separators, and units by locale, while APIs
  continue to emit ISO 8601, integer token counts, and explicit currency/unit
  fields.
- Test narrow and wide layouts in both languages. Chinese text must not be clipped;
  English controls must tolerate expansion. Do not encode meaning by color alone.
- Host/Origin, CSRF/session, localhost binding, path validation, and secret
  redaction behavior must be identical across locales.
- A locale change must not restart, stop, resume, or otherwise mutate a worker.

## Terminology map

| Stable English term | Simplified Chinese display | Rule |
| --- | --- | --- |
| control panel | 控制面板 | UI noun; keep `control_plane` identifiers unchanged. |
| distillation | 蒸馏 | Do not translate dataset/schema identifiers. |
| teacher model | 教师模型 | Keep exact model ID beside the display name. |
| planner | 规划专家 | Keep `planner` in code, config, and traces. |
| tool policy | 工具策略 | Keep `tool_policy` in code, config, and traces. |
| domain builder | 领域构建专家 | Keep `domain_builder` in schemas. |
| domain review | 领域审查 | Keep `domain_review` in schemas. |
| security gate | 安全门禁 | Keep `security`/`security_gate` identifiers as defined. |
| task card | 题卡 | Do not translate `card_id`. |
| work order | 工作单 | Do not translate `record_id` or `alignment_id`. |
| shard | 分片 | A new effective configuration requires a new shard. |
| resume | 继续/断点恢复 | Never imply that changed config resumes the same shard. |
| graceful stop | 平滑停止 | Distinguish from cancel, crash, and forced termination. |
| held-out | 留出集（held-out） | Keep `held-out` on first use to preserve benchmark meaning. |
| fail closed | 默认阻断（fail closed） | Never soften to a warning-only meaning. |
| request rate | 请求速率 | Display unit explicitly, such as requests/min. |
| wire attempt | 传输尝试 | Counts retries as separate network attempts. |
| unknown price | 价格未知 | Must not render as zero cost. |
| external/read-only | 外部进程/只读 | The panel must not control an attached external worker. |
