# RFC：Producer 原生层级 Task-KV 元数据

状态：producer-v2 控制面 MVP
分支：`research/neural-swarm-kv`
外层 sidecar：`anchor.swebench-taskboard-sidecar.v2`
Segment plan：`anchor.hierarchical-task-kv-segment-plan.v1`

## 范围与唯一真相源

SWE-bench TaskBoard projector v2 会在每条外层 sidecar 的
`outer_sidecar.segment_plan` 中直接生成一个经过认证的 `segment_plan`。这个
producer 原生 plan 是当前唯一的 Task-KV plan。消费端不得再次派生、合并或把它
静默转换成已退役的 `anchor.taskboard-kv-segment-plan.v1` 结构。

规范文件如下：

- [`swebench_taskboard_projector_v2.yaml`](../../configs/research/swebench_taskboard_projector_v2.yaml)：
  当前 producer 策略；
- [`taskboard_projector_sidecar.schema.json`](../../configs/research/taskboard_projector_sidecar.schema.json)：
  封闭的外层 sidecar envelope；
- [`hierarchical_task_kv_segment_plan.schema.json`](../../configs/research/hierarchical_task_kv_segment_plan.schema.json)：
  需要独立认证的内层 plan schema；
- [`taskboard_projector_manifest.schema.json`](../../configs/research/taskboard_projector_manifest.schema.json)：
  producer manifest 契约；
- [`swebench_source_disjoint_manifest.schema.json`](../../configs/research/swebench_source_disjoint_manifest.schema.json)：
  source-disjoint 发布证据；
- [`generic_train_execution_contract.schema.json`](../../configs/research/generic_train_execution_contract.schema.json)：
  通用执行 envelope 契约；
- [`generic_train_release_lock.schema.json`](../../configs/research/generic_train_release_lock.schema.json)：
  下游训练 release lock。

旧的 `swebench_taskboard_projector_v1.yaml` 只保留为历史文件。v2 fixture 或
release 不得引用它。

冻结的 producer-v2 哈希如下：

| Artifact | SHA-256 |
| --- | --- |
| Projector config | `b36945a2693183f0b213da403afcf8bb5611f46298bb849434e7b7d5854ba943` |
| 外层 sidecar schema | `c1863bfab69ce2f2388ee37fadae951b14f3d5120706bab032cab3f9aab6bdc5` |
| 原生 segment-plan schema | `80f760497e0d21f7d4d532db758362a800e845e6919b18b23958caabc7f155bf` |
| Projector manifest schema | `2cd9dc98d2b2865ed0586abfe291e3f6d161686597fcd2a7884c5762d2195347` |
| Source-disjoint schema | `2a2aae532c25b324a96b929a6a396d55d051c765258a5da0ebb7547724c68f6b` |
| 通用 execution schema | `63c699fdb7932b9fe1593b044d6a588bb4234b349816ce70f65568ab7b0f0b3a` |
| 训练 release-lock schema | `119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa` |
| Fixture manifest | `595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac` |

消费端 CLI 将
[`hierarchical_task_kv_mvp.yaml`](../../configs/research/hierarchical_task_kv_mvp.yaml)
固定为 SHA-256 `f695e02cd2da8ca9c8d40fc99a0c33a4803b23dbc5e7f4cf296d40156315252d`。

partition 数量与哈希由该 fixture manifest 认证，此处不重复复制。

## Plan 是什么，不是什么

Plan 只包含有序 segment ID、正文哈希、所有权、可见性、commit state、依赖和
lineage 哈希。它不包含 prompt 正文、答案正文、token ID、tensor、设备指针、KV
page 或模型权重。

当前每个 plan 都固定为：

- `execution_mode: decoupled_frozen_prefix_producer_required`；
- `materialization: metadata_only_no_tensor_or_kv`；
- `full_generation_kv_shared_claimed: false`；
- `token_level_moe_claimed: false`。

它的 cache compatibility 被刻意设为 `identity_unbound`，并且
`cache_reuse_allowed: false`。因此，这份逻辑 plan 不声明运行时可以精确复用，
也不声明 CUDA overlap、显存降低、延迟降低、吞吐提升、质量保持或生产可用。

## CPU Cache 只是一种生命周期类比

任务共享前缀、下游不可变快照和专家私有 delta，可以类比为共享与私有缓存的
生命周期。这个类比只适用于所有权、可见性、fork、发布和 copy-on-write，不表示
Transformer attention stack 等价于 CPU cache hierarchy。

cache address 或查找 key 不是 Transformer 的 K tensor。代码、schema、指标与报告
不得混用这两个概念。

## 为什么普通 Q-LoRA 不能让全 stack KV 共享变得精确

在普通 decoder 中，层内 Q-LoRA 会改变 attention 输出，继而改变 residual hidden
state 和后续层计算 K/V 的输入。K-LoRA 或 V-LoRA 则直接改变被缓存的投影。因此：

- 仅做 Q specialization 不足以保证全 stack 精确共享；
- 禁止把朴素层内 Q-LoRA 的复用称为精确；
- 修改 K/V 的路径与未改变的共享 K/V page 不兼容。

未来若要实现精确路径，需要冻结且与专家无关的 prefix producer，以及解耦的专家
query/cross-attention/readout 路径。专家开始生成后再 fork 私有 delta state。
producer plan 只记录这个架构要求，并未实现或跑分验证它。

## 因果 Segment 构建

Producer 先按源任务切分，再创建 clean/noisy view；noisy view 绝不参与决定 split。
同一个源任务会生成 `planner`、`tool_policy`、`frontend_gen`、
`frontend_review` 和 `security_gate` 五个角色视图。

共享前缀遵循 `membership_rule: strict_all_five_role_visibility_intersection`。
block 按 TaskBoard 因果顺序序列化，并连接成唯一 prefix chain。禁止先独立编码
block 再任意拼接 KV。

三种 cache scope 为：

- `task_shared_prefix`：五角色全部可见、已 committed 的任务源 block；
- `downstream_task_shared_immutable`：已显式 committed 且对下游角色因果可见的
  上游专家输出；
- `expert_private_delta`：仅拥有者可见的 candidate 或 verified 私有上下文。

当前 target answer 绝不会作为输入 segment 输出。当前与未来 forbidden block 可以
物理存在于 TaskBoard 源对象中，但不得先插入再依靠 mask 隐藏；
`shared_then_mask_allowed` 与 `forbidden_current_future_preinsert_allowed` 均为
false。

新的专家输出从私有 scope 开始。只有显式 commit 且对下游因果可见时才能晋升；
模型输出不能自行授予共享权限。

## 内容寻址与 Lineage

单个 segment 的 ID 计算方式为：

```text
segment_id = "task-kv-segment-v1:" + sha256(canonical_json({
  task_bundle_sha256,
  source_block_id,
  content_sha256,
  producer_role,
  cache_scope
}))
```

根 lineage 绑定 task bundle、execution mode 与 `ordered_prefix_genesis`。每个后续
prefix lineage 绑定前一个 lineage、segment ID、serialization order 和原始
TaskBoard causal order。segment 还会列出此前所有 segment ID 作为 dependency。
因此，相同但无序的 block 集合不等于相同 prefix。

每条 plan 还绑定 task/bundle/base-board identity、producer/schema identity、源
Gold/file/snapshot identity、split、stage、expert 和 variant。消费端必须把这些
bindings 与外层 sidecar 及已认证 producer manifest 逐项交叉校验。

## 运行时兼容性继续 Fail Closed

在把逻辑前缀当成物理 cache hit 之前，未来 runtime 必须证明以下七项完全一致：

1. `model_architecture_sha256`；
2. `tokenizer_sha256`；
3. `token_order_sha256`；
4. `position_ids_sha256`；
5. `rope_config_sha256`；
6. `kv_producing_weights_sha256`；
7. `prefix_lineage_sha256`。

任何 identity 缺失或不匹配都必须得到 `cache_incompatible`。当前 producer 刻意没有
bound 变体，也不允许物理复用。

无模型 runtime 原型还要求调用方显式配置 `TrustedExactKVBinding`，绑定任务 bundle、
完整有序 page 列表、终端 lineage、compatibility identity 和 producer 执行模式。
Provider 不能只靠在 runtime context 里返回这些字符串就自行获得 `exact` 声明。
这仍是控制面信任证据，并非未来 CUDA 后端确实正确生成 tensor 的密码学证明。

所有元数据注册表都设有明确的 page、prefix、branch、inline byte 和 identity 上限。
Capability 碰撞会在发布前失败；旧版本但真实的 branch 句柄仍可释放引用；嵌套异步
迭代器在正常结束、取消或消费者提前关闭时都会执行有界的协作式清理。

## 消费端与发布要求

消费端必须：

1. 从单个不可变 bytes snapshot 读取每个 manifest/schema/partition；
2. 认证 producer 指定的本地 schema 精确字节；
3. 分别验证外层 sidecar 与内层 plan，禁止远程 `$ref` resolution；
4. 将外层 provenance 与内层每个 `bindings` 字段交叉绑定；
5. 对未知字段、未知 schema version、旧 v1 配置和任何 hash 漂移 fail closed；
6. 始终用 `task_bundle_sha256` 做 split group key，禁止五角色跨越
   train/calibration；
7. 真正排除 forbidden/current/future 正文，禁止 stringify 整个 TaskBoard；
8. 真实训练前要求 source-disjoint 证据和完全匹配的 release lock。

dry-run 必须保持 CPU-only、资源有界且不加载模型。它不得发送 provider request、
加载模型权重、分配 GPU tensor、读取 heldout 正文或改写 canonical Gold。

## 相关工作与致谢

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  说明 cache identity 需要绑定 parent lineage、有序 block token 和 LoRA ID 等额外
  identity；它的 cache key 不是 attention K tensor。
- [Punica](https://arxiv.org/abs/2310.18547) 提供多 LoRA batching 与共享 base weight
  serving kernel；这不等于跨 adapter KV 共享。
- [LRAgent](https://arxiv.org/abs/2602.01053) 研究共享 base component、adapter 相关
  low-rank cache component 与专用 attention kernel。本元数据契约不声称已实现其
  kernel 或复现其结果。
- [ForkKV](https://arxiv.org/abs/2604.06370) 提出 DualRadixTree、CoW cache management
  与 ResidualAttention，同时把 adapter hidden-state 分叉后的跨层 base-cache 复用
  描述为有损。本 RFC 不会把这种近似重新命名为精确。

本双语控制面 RFC 由 OpenAI GPT-5.6-sol 协助完成代码语境审计与架构梳理。相关
工作只按其真实贡献致谢；其测量或质量结果不会被写成 Anchor-MoE-LoRA 的结果。

## 明确不声明的成果

本 MVP 不声明 decoder KV block 可拼接、跨 adapter KV 可精确复用、完整生成 cache
可共享、已实现 token-level MoE、存在 CUDA overlap、获得显存/延迟/吞吐收益、
保持质量、完成 foundation-model training 或达到生产可用。在 runtime 实现与受控
评测方案另行批准前，不预设评测组。
