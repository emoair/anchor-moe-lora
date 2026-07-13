# SWE 五阶段 trajectory adapter

`anchor_mvp.swebench.trajectory` 把一张已经通过分区/许可门禁的 SWE 题卡，和
受控 OpenCode session candidate，投影成一条完整的五阶段蒸馏记录。它是纯本地
校验与转换层：不调用教师 API、不启动 Docker/Podman、不执行命令、不读取 live
data，也不把结果自动追加到训练集。

## 阶段可见性

| 阶段 | 允许输入 | 必须输出/证明 |
| --- | --- | --- |
| Planner | 公开 `problem_statement`、base workspace inventory | `domain_id`、builder/reviewer expert、work items、tool proposals |
| Tool Policy | 公开需求、planner 的 tool proposals | 每个 proposal 恰好一次 `APPROVE`/`DENY` |
| Domain Builder | 仅公开需求、base inventory、已批准工具结果 | 生成的 diff、去 oracle 化执行摘要、受控 tool trace |
| Domain Review | 仅公开需求、生成 diff、结构化执行摘要 | `PASS` 或带公开 feedback 的 `REVISE` |
| Security | 仅公开需求、最终生成 diff、结构化执行摘要 | `PASS` 或带公开 finding 的 `BLOCK` |

返修可以产生多个 builder/review revision，但它们始终保留同一个
`alignment_id`。除最后一次 review 必须 `PASS` 外，前面的 review 必须是
`REVISE`。planner 选择的 domain、builder、reviewer 必须等于题卡路由；trusted
sandbox audit 记录的实际 builder 以及 review 输出记录的实际 reviewer 也必须
完全相同。

## OpenCode 输入边界

adapter 接受现有受控转换器生成的
`anchor.session-training-candidate.v1`，而不是未经处理的 OpenCode 原始 session。
二次校验包括：

- `source.kind` 必须是 `controlled-opencode-export`，workspace 必须严格为
  `<workspace>`；
- tool call/result 的 `call_id` 各自唯一、一对一、同工具、相邻，且 result 的
  sequence 必须等于 call sequence 加一；
- tool call 的工具和输入必须精确匹配 planner proposal，并且 proposal 已由
  tool-policy 明确批准；
- 所有结构化 path-bearing 参数与生成 diff 文件必须位于 `<workspace>` 下；
- 外部绝对路径、`..` escape、容器路径未规范化、环境/凭据/私有推理字段直接
  拒绝。

现有受控 candidate 的 `final_diff[].patch` 表示 agent **刚刚生成的 workspace
diff**，不是上游 benchmark 的 gold patch。adapter 只在这个唯一、严格限定的
schema 位置接收该字段，并立即改名为 `diff`；其他位置出现 `patch`，以及任何
`test_patch`、hints、`FAIL_TO_PASS`、`PASS_TO_PASS`、test-name、gold 或 oracle
字段，仍会整条 fail closed。validator stdout/stderr 不会进入 review/security
输入，执行信息只投影为状态和计数。

## Trusted sandbox audit bundle

一个看起来像 SHA-256 的字符串不是执行证明。adapter 同时需要 audit bundle
对象和由调用方从可信、model 不可写位置取得的预期 SHA-256，并执行以下重算：

- base workspace inventory binding；
- 每个 revision 的完整 controlled candidate hash；
- 规范化 tool trace hash；
- 生成 diff hash；
- 去 oracle 化 execution summary hash；
- 实际执行 builder expert；
- protected state 前后 hash 相等；
- `.anchor` 与 VCS snapshot exclusions 被显式声明；
- cleanup 状态为 `cleaned`。

bundle 自身的 canonical JSON digest 还必须等于调用方传入的 trusted digest。
adapter 只验证绑定关系，不自行宣称 bundle 的来源可信；正式 runner 必须在 agent
不可写的控制面生成并保存它。

## 输出不变量

成功记录固定包含五个阶段，且计数为一张卡、一个 alignment、一条完整 chain。
builder revision 的 `input` 恰好只有：

- `problem_statement`
- `base_workspace_inventory`
- `approved_tool_results`

review/security 的 `input` 恰好只有：

- `problem_statement`
- `diff`
- `execution_summary`

adapter 不提供 writer；调用方应先对输出做独立抽样/质量筛选，再决定是否写入某个
训练 shard。这样可以保持“先收集、后剔除”的数据策略，同时不绕过 held-out 和
许可门禁。
