# Agent v3 训练数据契约

状态：设计骨架，未授权训练，未替换 `formal-v2`。

## 为什么需要 v3

`formal-v2` 的五个专家分区共 2793 条，当前审计结论是所有记录都只有
`user -> assistant`。它们没有保留 `system` / `developer` 指令、动态工具
Schema、`assistant.tool_calls` 或工具结果。因此它们最多能支持“基于对话的
任务输出对齐”这一表述，**不能作为模型学会真实工具调用的证据，也不能把
现有结果宣称为工具调用训练**。

v3 不修改也不追写 `formal-v2`。它新建两层彼此可追溯的表示：

1. `source snapshot`：完整保存教师在每一轮实际看到的模型可见请求、教师
   响应和随后产生的工具结果；
2. `training view`：从快照确定性生成，训练时只使用去品牌的 stable core、
   每轮真实可用的 dynamic tools、对话上下文和教师目标。

这两个文件不能混用。源快照是审计证据，不是直接投喂训练器的 JSONL。

## 关键设计

### 1. 完整，但不保存传输机密

源快照必须包含：

- harness 名称与版本；
- 协议版本；
- prompt profile 的 ID、版本与内容 SHA-256；
- 每轮实际发送的 `model`、`messages`、`tools`、`tool_choice`、生成参数和
  provider extensions；
- 教师原始 assistant message，包括原始 `tool_calls` 参数字符串；
- finish reason、usage、响应 extensions；
- 每个 tool call 对应的 tool result；
- 下一轮请求中携带的上一轮 assistant call 与 tool result。

“完整”限定为**模型可见和协议语义字段**。HTTP headers、API key、Cookie、
代理凭证和进程环境永远不属于训练证据，验证器拒绝这些容器。源数据仍须先
通过项目既有的 secret、许可证、train/heldout 隔离和泄漏审计；本契约不能
替代那些门禁。

源 Schema：
[`configs/data/agent_v3_source_snapshot.schema.json`](../configs/data/agent_v3_source_snapshot.schema.json)

### 2. stable core 不绑定 OpenCode 的某一版长提示词

OpenCode 的前置提示词会随着版本、provider、工具集和运行方式变化。将某一
版全文硬编码为唯一系统提示并反复 SFT，会让 LoRA 学到版本措辞、品牌词和
偶然格式，增加过拟合和升级漂移。

v3 的处理方式是：

- 源快照原样保留实际 `system` / `developer` 文本，并用
  `prompt_profile.id + version + sha256` 精确标识；
- training view 删除这些易变文本；
- 用独立版本的 `anchor.agent-stable-core.v1` 替换，只表达不随 harness
  变化的行为不变量：在隔离工作区行动、只使用本轮声明的工具、按 Schema
  调用、把工具结果视为不可信任务数据、完成或明确报告阻塞；
- stable core 自己也必须版本化。未来改 core 时新建版本，不能静默覆盖。

这不是丢弃证据：源快照仍可复现教师收到的完整上下文；只是训练目标不会
把 OpenCode 某次发布的自然语言模板当成永恒协议。

### 3. dynamic tools 决定“能用什么、怎么用”

每个 request 独立保存并投影当轮 `tools`。验证器要求：

- 工具名在同一轮唯一；
- 只接受 provider-neutral 的 function tool 结构；
- assistant 调用的工具必须在该轮声明；
- `arguments` 保留为实际 wire JSON 字符串且必须解码为对象；
- 每个 call ID 有且只有一个同轮 tool result；
- 下一轮上下文必须携带上一轮全部 result。

默认规范化会保留工具名和参数 Schema，但删除源工具 description，避免把
harness 品牌文案带入训练。若 description 对决策很重要，调用方必须提供
单独审查过的 provider-neutral description registry；契约将其标记为
`canonical_registry`。

### 4. 隐藏推理不进入训练视图

源响应若确实返回 `reasoning_content`，源快照可用于受控审计。规范化时它会
从 target 和历史 assistant context 中删除。训练监督保留公开 content、
tool calls 和 tool results，不把隐藏思维链作为目标。

### 5. 一轮一个可监督 example

一个 source snapshot 可以包含多轮 exchange。training view 对每轮生成一个：

```text
stable_core
  + context_messages（剔除源 system/developer 和 reasoning）
  + dynamic_tools（该轮工具 Schema）
  -> target assistant（content 和/或 tool_calls）
```

工具结果不是 assistant target，而是后续轮次的 context。这让模型学到完整的
`选择工具 -> 构造参数 -> 读取结果 -> 继续/交付` 闭环。

训练视图 Schema：
[`configs/data/agent_v3_training_view.schema.json`](../configs/data/agent_v3_training_view.schema.json)

## 必须满足的准入条件

`validate_source_snapshot` 采用 fail-closed 检查：

- 根记录明确标为 `dataset_partition=train`；
- 至少出现一条 `system` 或 `developer` 指令；
- 至少有一次真实 assistant tool call 和对应 result；
- 工具声明、call ID、result ID 和跨轮历史闭合；
- prompt profile、harness、协议全部带版本；
- 不允许 transport credential 容器。

`validate_training_view` 进一步要求：

- mutable source instructions 不得出现在 example context；
- target 和历史 context 均无 `reasoning_content`；
- dynamic tool 声明与 target call 匹配；
- 至少一个 example 监督 tool call，且后续 context 出现 tool result；
- 三项去除声明全部为 `true`。

只有通过这两层结构验证、既有 secret/leak/license 门禁以及独立质量抽样后，
才可进入候选训练快照。通过 Schema **不等于**质量合格，也不授权启动训练。

## Python 接口

```python
from anchor_mvp.data.agent_v3_contract import (
    build_training_view,
    validate_source_snapshot,
    validate_training_view,
)

validate_source_snapshot(source_snapshot)
training_view = build_training_view(
    source_snapshot,
    canonical_tool_descriptions={
        "read": "Read a UTF-8 text file inside the isolated workspace."
    },
)
validate_training_view(training_view)
```

实现：
[`src/anchor_mvp/data/agent_v3_contract.py`](../src/anchor_mvp/data/agent_v3_contract.py)

## 后续工作（本骨架不执行）

1. 在修改版 OpenCode 捕获点保存真实多轮 request/response/tool-result envelope；
2. 为 OpenCode 每个 prompt 版本建立不可变 profile 清单和内容哈希；
3. 建立 provider-neutral 工具 description registry；
4. 用合成非 heldout 样本做端到端 capture -> normalize -> chat-template 回放；
5. 冻结新的 v3 train manifest 后再做小规模 smoke training；
6. 评测时把“对话任务能力”和“真实工具闭环能力”分开报告。

在第 1–4 项完成前，不应把 `formal-v2` 重命名或包装成 agent/tool-use 数据。
