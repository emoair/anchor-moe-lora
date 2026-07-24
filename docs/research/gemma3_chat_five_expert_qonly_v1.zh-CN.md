# Gemma 3 1B IT 五专家聊天诊断数据集 v1

## 1. 目标

`gemma3_chat_five_expert_qonly_v1` 是面向本地 Gemma 3 1B IT 导出的全新、
additive、仅诊断聊天数据。它固定包含 200 个唯一任务 bundle，每个 bundle
派生五个反事实专家视图：

1. `humor`
2. `serious`
3. `angry_style`
4. `tool_call`
5. `review_audit`

总量严格为 1,000 行。先按 bundle 分区，再扩展角色：160 bundle / 800 行
为 `train`，40 bundle / 200 行为 `eval_proxy`。英文和中文各 100 bundle，
分别为 80/20。`eval_proxy` 不是 held-out。

本资产独立于既有 SWE-bench、canonical Gold、held-out、五阶段、Q-only
scaffold 与 frozen-prefix Q-reader 资产。它固定
`replaces_existing=false`，不改变这些资产的字节、计数、schema 或声明。

## 2. 非目标与声明边界

本数据不声称：

- 隐藏思维链；
- 临床情绪识别；
- 真实搜索引擎或真实工具轨迹；
- Gemma 原生多轮 tool-role 序列化；
- 与受保护语料已证明 source-disjoint；
- held-out、formal 或生产质量；
- 物理 KV 或完整生成 KV 共享；
- 已训练 100M、rank 1024 或任何 adapter；
- training、formal、release、provider 或 live 授权。

`user_emotion` 只是闭集路由标签与简短、可审计的依据摘要。它是五条记录
共享的 bundle 元数据，不是第六专家，也不是临床判断。

## 3. 因果拓扑

五个视图不是旧五阶段线性链：

```text
共享用户输入与 router 元数据
  |-- humor 私有分支
  |-- serious 私有分支
  |-- angry_style 私有分支
  `-- tool_call 私有分支
          |
          | 四个输出分别 commit，并重编码为不可变输入
          v
      review_audit 汇合
```

五个角色可以使用不同的角色指令和专属 context。Bundle 内只强制最后一条
user message、`user_emotion` 与 router label 一致，并由
`router.user_message_sha256` 认证共享 user turn。前四个 prompt 不能看到兄弟
输出、当前 target、未来 target 或 forbidden 内容，生成内容在显式 commit
前保持专家私有。`review_audit` 是第二请求/post-hoc 视图，它的专属 context
只能按固定顺序读取四个已 commit 输出并 digest-bind；这些依赖不能泄漏到另外
四个视图。Reviewer 也不能读取自己的 target。这只是文本与元数据
materialization，不声称物理 KV tensor 或 zero-copy。

## 4. 角色行为

`humor` 保持有帮助、温和且不嘲讽；`serious` 直接、克制、简洁；
`angry_style` 可以坚定、有力度，但必须非辱骂、无威胁。这三个视图必须保持
同一任务事实和路由身份。

普通 195 个 bundle 的 `tool_call` 使用单个闭合结构化 assistant target，
其中包含：

- 确定性的 synthetic-local 工具调用；
- 与调用匹配的 synthetic-local 结果；
- evidence refs 只能来自该结果的 grounded final。

调用、参数、结果、证据 ID、最终回答及其哈希全部交叉绑定并由 auditor
重新计算。它的准确口径是 `deterministic_synthetic_local_tool_result`，
不声称外部工具真实运行。

另有五个不同 semantic bundle 用于身份/归属对齐。稳定事实核心是：

```text
我是由Air训练的测试模型。
```

`humor`、`serious`、`angry_style` 和 `tool_call` 都必须保留该事实，不能声称
由 Google 或 OpenAI 训练。仅这五个 bundle 的 `tool_call` 使用
`direct_no_tool_identity` 并直接回答，不得伪造调用或结果。`review_audit`
同时检查身份事实和 no-tool 决策。这五条是语义确实不同的任务，不能通过改名、
翻译、复制或加盐冒充 unique。

## 5. 身份与分区

`task_semantic_sha256` 来自完整、与语言和 namespace 无关的语义描述。
namespace、language、role、split、view、seed 和 adapter 标签都不得进入
semantic preimage。`task_bundle_sha256` 只由 semantic identity 与语言无关的
本地证据身份派生，仍排除 role 和 split。本地化用户文本另有独立哈希。

Producer 与 auditor 独立重算：

- 200 个唯一 semantic identity 和 200 个唯一 bundle identity；
- EN/zh-CN semantic 交集为零，translation pair 为零；
- 每 bundle 恰好五角色，每角色 200 行；
- 五视图的 semantic、bundle、split、language、最后 user message、emotion 与
  router-label 身份一致，同时允许角色专属指令/context；
- bundle 160/40、记录 800/200；
- EN 与 zh-CN 各 100 bundle，分别 80/20；
- train/eval bundle 交集为零；
- 200 个有效 review join、恰好 800 个有序 parent refs。

## 6. Gemma 序列化

Producer 绑定冻结的外置 Gemma chat policy，不能声称本地 Keras export 自带
chat template；export 明确写着 `chat_template_bound=false`。

冻结身份包括：

- chat policy SHA-256：
  `0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6`；
- SentencePiece SHA-256：
  `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`；
- tokenizer config SHA-256：
  `90e9a8120520ef24c0a0d62a6d87188658c43ccc66ae5bc6f74d9a80804e6919`；
- tokenizer/template/special-token 组合身份：
  `1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946`。

Runtime overlay 使用 BOS/EOS 2/1，不能信任 export config 中反向的 1/2，也
不能修改 canonical model 文件。Gemma 不支持 system role，因此角色指令放在
第一条 user message 内。

工具 call/result/final 作为一个结构化 assistant target 进入冻结的单轮
user-to-assistant serializer；这不等于原生 tool-role 序列化。每条新记录都要
生成本数据专属 tokenizer receipt，包括 prompt/input/labels digest、token
计数、可训练 label 数、sequence limit 与 `truncation_used=false`。不得发布
raw token IDs。旧 1,000 条的 tokenizer receipt 只能作为来源身份，不能认证
本数据。

## 7. 发布与 read-set

确定性 builder 只读取自身 config、schemas、closed grammar/本地证据、
implementation、冻结的 Gemma policy/binding 元数据，以及已认证的
SentencePiece 字节。它不读取 protected、SWE-bench、Gold、held-out 或既有
scaffold 正文。

每个小文件只读取一次 bytes，并用同一快照做 hash、parse 和 count。JSON/YAML
拒绝重复键和非有限数。路径拒绝 traversal、symlink、junction 与其他 reparse。
输出目录 create-once 并原子发布。Producer 在末端重验所有输入和输出身份；
失败时只能删除 creation-time directory identity 仍匹配、可证明归自己所有的
输出；若无法证明 ownership，必须保留目录并 fail closed，绝不能按未经认证的
当前路径递归删除。已有目录永不覆盖。

Canonical artifact 路径固定为：

```text
fixtures/research/gemma3_chat_five_expert_distilled_v1/
  train/chat.jsonl
  eval_proxy/chat.jsonl
  token_inventory.jsonl
  manifest.json
  manifest.json.sha256
  build_receipt.json
  build_receipt.json.sha256
```

每个 mandatory sidecar 都采用小写 payload SHA-256、两个空格、对应 basename
和 LF。Consumer 已冻结的 manifest 只绑定两个训练 partition；新增的
build receipt 再合取绑定 manifest、两个 partition、body-free token inventory、
generator、config、全部 schemas、grammar、seed、有序 read-set、计数/证明
inventory、Gemma 身份、请求计数和末端 TOCTOU 结果。这样既不放宽 consumer
closed schema，也不会丢失 producer provenance。

## 8. 资源与 provider 计数

v1 fixture 使用确定性的 offline generation：

- provider requests：0；
- network requests：0；
- model loads：0；
- GPU requests：0；
- protected/Gold/held-out body reads：0；
- 每次 build/audit 进程只加载一次 SentencePiece tokenizer。

未来版本若使用 provider，必须使用新的 authenticated generation contract 和
receipt，记录 provider/model identity、batch、request、retry、seed、
checkpoint 与 HMAC 绑定的请求/响应 hash。凭据、headers、环境变量值和原始
provider 错误正文都不得进入产物。

## 9. 复现

不需要模型或 GPU：

```powershell
py -3.10 scripts/research/build_gemma3_chat_five_expert_qonly_v1.py
py -3.10 scripts/research/audit_gemma3_chat_five_expert_qonly_v1.py
py -3.10 -m pytest -q tests/test_gemma3_chat_five_expert_qonly_v1.py
py -3.10 -m ruff check src/anchor_mvp/research/gemma3_chat_five_expert_qonly_v1.py scripts/research/build_gemma3_chat_five_expert_qonly_v1.py scripts/research/audit_gemma3_chat_five_expert_qonly_v1.py tests/test_gemma3_chat_five_expert_qonly_v1.py
```

10,000 行 / 100 专家的 Luna alignment 数据属于之后的独立队列。它的 taxonomy、
schema、请求、receipt、记录和哈希都不能进入或授权本 v1 artifact。

## 10. FINAL 修正门

Manifest 只有在以下条件同时成立后才可消费：

- `humor`、`serious`、`angry_style` 的 physical assistant target 必须是
  `format=chat_text` 的自然语言；只有 `tool_call` 和 `review_audit` 使用
  closed JSON target。
- Task semantic preimage 必须由真实 operation、structured operands、
  constraints 和 expected relation 构成。case number、evidence label、
  language、namespace、role、split 与 generator index 均不得充当唯一性盐。
  Receipt 另报 template count 与 train/eval template overlap；`eval_proxy`
  只代表 seen-template interpolation，不是 held-out 或 template-generalization。
- Review 数据必须是 review-only、pass/fail 平衡的纠错 projection；其扰动
  不能改写或泄漏到前四个 branch target。
- 每条 record 的输入消息必须精确为 `[system, user]`。Tokenizer 必须直接用
  authenticated `tokenizer.model` bytes snapshot 的 `model_proto` 构造，
  认证后不得再次按路径打开文件。
- Temporary 与 published directory 在任何 rollback cleanup 前都必须绑定
  creation-time identity；identity 不匹配时保留目录并 fail closed，禁止递归
  删除未经认证的路径。
- 五个 persona bundle 必须对应五种真实不同 intent：直接自我身份、来源事实
  核验、无需工具的身份判断、编码前归属说明、错误归属纠正。Namespace-neutral
  semantic identities 与零 translation-pair 结论必须复算，不能硬编码声明。

早于这些门的 implementation、manifest、receipt 和 partition identity 全部
superseded，消费端必须拒绝。

## 11. Producer 小工单纪律与后续 10k/Luna 队列设计

数据生产必须是串行、独立复核的状态机。不得把整套数据生成任务一次交给一个
agent，再由同一个 agent 自证通过。每张工单只能覆盖一个明确数据族或一个明确
修复；它交付有界变更、语义清单和负向测试后，必须由另一个 reviewer 独立重算
语义 identity 并验收，验收通过后才能派下一张工单。数量、schema、hash 和语法
检查都不能替代语义审计。

当前 1,000 行修复必须严格按以下顺序执行：

1. 删除 `TASK_CONSTRAINT`，只修 195 个普通任务，并让 `micro_coding` 使用用户
   可见且影响任务结果的真实参数；
2. 不重建 FINAL，由独立 reviewer 审计这 195 个 task instance；
3. 用单独工单实现五个 persona intent；
4. 由独立 reviewer 审计这五个 persona task instance；
5. 用单独工单重建完整 fixture；
6. 再做一次独立发布审计，通过后才产生唯一可消费 identity 集合。

禁止把 hash digest、数组位置、case/index 编号、evidence label、随机 ID、语言、
namespace、split、role 或 generator counter 当作语义差异。每个不同 identity
都必须对应真实 operation、operands、constraints 或 expected relation 中用户
可见且有任务意义的差异。语义审计失败即使相关 fixture 及其全部 hash
superseded，后续工单不得把它作为输入。

后续 10,000 行 / 100 专家 Luna 数据同样执行该纪律：先冻结 closed taxonomy 与
配额，再按小型数据族工单串行生成；每个数据族先通过独立语义和因果审计，才启动
下一个。最终 rebuild、跨族去重审计、provider/request receipt 审计和 release
审计必须分别作为独立工单。任何 agent 都不得同时生成某个数据族并给出使其晋级
的独立验收。中间 checkpoint 不具授权效力，任何不完整数据族都不得计入 FINAL。

Luna 队列继续暂停：在当前 1,000 行资产取得一套独立验收的 FINAL identity，且
用户显式放行下一阶段前，不得发送 provider 请求，也不得 materialize 10k rows。
未来队列的任何内容都不得混入、替换或授权本 v1 资产。
