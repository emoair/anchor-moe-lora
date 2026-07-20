# Anchor-MoE-LoRA 五阶段路由 adapter MVP

[English](architecture_five_stage_mvp.md) | [简体中文](architecture_five_stage_mvp.zh-CN.md)

每个 seed 的 MVP 路由严格按以下顺序执行：

`planner -> tool_policy -> (frontend_gen <-> frontend_review, bounded) -> security_gate`

所有 adapter 使用同一个冻结且序列化方式完全一致的 Q4/NF4 基座。模型 revision、量化设置、
tokenizer、本地产物 digest、阶段顺序和每阶段 token cap 是 A/B/C/D/E/F 全部实验组的
不变量。第一次实验的满容量 LoRA rank 是 16。只有在同一次变更中同时注册和评测配对的领域
reviewer，新的领域 coder 才有效。

| 阶段 | 输入 | 输出 | 失败行为 |
| --- | --- | --- | --- |
| `planner` | requirement | 摘要、有序步骤、约束 | 停止；不调用 policy/coder |
| `tool_policy` | plan + 惰性抽象提案 | 严格输出 `APPROVE`、`BLOCK` 或 `ESCALATE`，并给出理由和公开 trace | 停止/fail closed；标签只提供建议 |
| `frontend_gen` | requirement + plan + policy 建议；返修时还包括当前代码与公开 issues | 完整代码 | 停止；不调用 reviewer |
| `frontend_review` | requirement + 当前 candidate | 严格的公开 `anchor.domain-review-verdict.v2` JSON：`PASS` 且无 issues，或 `REVISE` 且带精简 issues | 歧义/错误/超时或轮次耗尽时 fail closed |
| `security_gate` | requirement + 最终 review 通过的代码 + 公开 tool trace 摘要 | 严格输出 `[PASS]` 或 `[BLOCK]` | 歧义/错误/超时均变为 BLOCK |

模型不能授权自身。运行时执行权限由非模型 allowlist、workspace 边界、副作用与显式审批规则决定。
模型给出的 `APPROVE` 永远不能覆盖确定性的 `BLOCK` 或 `ESCALATE`。

主运行时默认最多允许两轮 review。`REVISE` 会复用同一个领域 builder LoRA；reviewer 从不写
修复代码，也不输出私有推理。trace 可以包含重复的 `frontend` 和 `review` 尝试，但仍只使用
五种专家。Security 只在严格 `PASS` 后运行，并接收最终 candidate 与公开的
proposal/policy/cycle 摘要。

旧 v1 数据文件是 `data_plan.jsonl`、`data_tool_policy.jsonl`、`data_frontend.jsonl`、
`data_review.jsonl` 和 `data_security.jsonl`。v1 `data_review.jsonl` 的目标是完整修复代码，
只在兼容的 `PipelineRouter.run` 路径和旧 benchmark 记录中有效。它不会被静默当作 v2 verdict
adapter。主 v2 训练会在 schema `anchor.review-loop-data.v2` 下分别写入
`data_review_verdict_v2.jsonl` 和 `data_frontend_revision_v2.jsonl`；如果缺少
`review_verdict` adapter，`run_five_stage` 会 fail closed。每一条下游记录都会记录同 seed
来源记录的 ID。工具提案由 `anchor-inert-tool-proposals-v1` 在本地生成，不包含可执行参数或
URL，并持久化 `executed: false`。已有成功的三阶段 live 行按 seed provenance 识别，五阶段
resume 时绝不重写。

Benchmark 使用五个匹配阶段进行公平比较：

- A 在每个阶段都使用 Q4 基座，指数为 100。
- B 在所有阶段复用一个 mixed-data rank-16 LoRA（10,387,456 个可训练参数）。
- C 路由五个满容量 rank-16 专家（共存储 51,937,280 个可训练参数）。
- D 路由五个较小专家，rank 为 `3/3/4/3/3`；rank 总和与物化可训练参数量严格等于 B。
- E 是按复杂度自适应的非均匀路由组。每阶段最多 rank 16，但允许总 rank/参数在只使用
  calibration 的预算阶梯上变化；E 搜索容量/性能 Pareto 前沿。
- F 使用与 E 相同的复杂度自适应分配算法和 calibration split，但把总 rank 和物化 adapter
  参数硬性限制为与 B 完全相同。

C 衡量最大容量的路由架构。**B 对 D 对 F** 是主要等预算比较：B 是 mixed，D 是手工分配，
F 是同一硬预算下的算法分配。C 和 E 是容量/Pareto 比较，不能单独用于宣称等预算路由胜出。
D 固定；E 和 F 经 calibration 选择，并在 held-out 评测前冻结。单次调用 A/B 结果只作辅助；
改变序列化 Q4 产物会使比较失效。

E 和 F 属于后续分配实验，不属于 `formal-v1`。它们共享的初始复杂度先验是
`frontend_gen >= frontend_review >= planner >= tool_policy/security_gate`。候选分配与
选择规则位于 `configs/training/complexity_adaptive_lora.yaml`。Rank 只能使用单独的
calibration split 选择；冻结 held-out benchmark 不得影响 rank。每阶段 rank 最大为 16。
E 报告质量、物化参数、路由延迟和峰值 VRAM 的 calibration Pareto 前沿。F 使用相同复杂度
评估器和候选机制，但限制为与 B 完全相同的预算。

对 E 而言，总 rank 是搜索变量，因此输出是容量/性能 Pareto 点。对 F 而言，同一分配器被限制
为总 rank 16 和恰好 10,387,456 个物化可训练参数，与 B 匹配。这将等预算自适应分配的收益
（B/D/F）与因容量支出不同而产生的收益（C/E）区分开。

仓库内已检查的自适应 benchmark 条目仍为 `calibration_pending`。在每个选中分配具备
calibration snapshot hash、attempt ledger、冻结 rank、物化参数量和不可变 manifest hash
之前，held-out 门禁会拒绝它们。

## MVP 之后的路线图：Phase 2 上下文驱动专家路由

Phase 1 仍是当前 MVP，也是规范的 A/B/C/D/E/F 控制实验。其路径有意固定为：

`planner -> tool_policy -> (frontend_gen <-> frontend_review, bounded) -> security_gate`

任何 Phase 2 实现、样本、路由决策或指标都不得修改、重标、追加或追溯性重解释
Phase 1 的数据集、registry、held-out、benchmark 记录或 A--F 结论。必须先完成并冻结
Phase 1 评测，产出不可变 manifest，之后才能开始生成 Phase 2 数据。这样固定流水线才是
真实基线，而不是持续移动的目标。

MVP 评测冻结后立即启动独立的 Phase 2 实验：planner/router 观察任务当前的公开上下文，
自主选择要激活、调用和卸载的专家 LoRA。它可以跳过不必要的专家、回到先前专家，或创建
并汇合逻辑分支。因此，Phase 2 是有界状态机或任务图，而不是换了名称的
`review -> execute` 固定串行链。在 12 GB 单卡配置中，分支按逻辑交错执行，显存中仍最多
只有一个 active adapter；Phase 2 不代表同时激活多个 LoRA。

建议把 router 动作定义为有类型、可审计的协议，例如 `ACTIVATE`、`CALL`、`UNLOAD`、
`SKIP`、`BRANCH`、`JOIN`、`RETRY` 和 `STOP`。每个动作记录公开状态 digest、所选专家、
公开理由、已消耗预算、adapter 生命周期、工具结果摘要和下一状态 digest。最终权限仍由
确定性运行时策略掌握：学习型 router 不能绕过 workspace 边界、工具 allowlist、显式审批、
循环/调用预算或 fail-closed 安全规则。

Phase 2 必须重新蒸馏，不能把五个固定阶段的 target 换壳复用。新数据版本要包含全新的任务
和经校验的 router trajectory：教师模型接收累积公开上下文，选择下一动作/专家，看到公开的
专家或工具结果，再继续到终止状态。题型应包含合理跳过、重试、回环、分支和提前停止的正例，
也应包含 adapter 抖动、无进展重复调用、不安全迁移和预算耗尽等 negative/reject trajectory。
只保存公开决策和可观测结果，不要求隐藏思维链。数据集必须使用新的 schema/version、
snapshot hash、provenance graph、train/calibration split 和不可变 run ID；不得复制 Phase 1
held-out 的 prompt、oracle label、solution、样本文本或派生 task ID。

Phase 2 还必须使用全新的 held-out 和全新的 benchmark contract，并且只能在 Phase 1 评测
冻结后创建。新 held-out inventory 必须同时排除在 Phase 1 与 Phase 2 的训练/蒸馏输入之外，
并通过新的基于 hash 的泄漏审计。Router 调参与停止规则只能使用 train/calibration 数据。
为了公平比较架构，需要在这份新 held-out 上重新运行一个已冻结的固定路由参考组；绝不能把
Phase 2 在新题上的分数与 Phase 1 在旧题上的分数直接相减并宣称提升。

除任务质量与安全性外，Phase 2 benchmark 还必须报告：路由/动作有效率、有可审计标签时的
专家选择准确率、无效调用率与合理跳过率、循环终止、分支完成、adapter load/unload 次数、
adapter 抖动、每任务调用数与 token、端到端延迟、峰值 VRAM、fail-closed 率，以及匹配
调用/token/参数预算后的表现。结果必须冻结全部 route trace 和 evaluator 版本。Phase 2 使用
独立 registry namespace 和 benchmark 名称；历史 A--F 表继续只代表 Phase 1 固定路由结果。

## 两次调用的 live smoke

当 `data/live_smoke/seeds.jsonl` 已有一个 seed 时，以下命令最多调用教师两次，且不会触碰已有
成功的 frontend/review/security 行：

```powershell
python -m anchor_mvp data --config configs/data/default.yaml `
  --output-dir data/live_smoke --seed-count 1 --concurrency 1 `
  --tasks plan tool_policy --protocol openai --no-fallback `
  --thinking-enabled --thinking-effort low --stream-openai `
  --max-tokens 32768 --max-requests 2 --max-output-tokens-total 65536 `
  --max-retries 0 --wall-clock-deadline-seconds 900
```

预期请求：一次 `plan`，随后一次 `tool_policy`。如果任一响应未通过校验，第二份/下游数据集
不会晋级，也不会启用批量并发。
