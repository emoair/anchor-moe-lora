# Gemma 3 unbalanced-v2 共享前缀 KV runtime v2

## 范围

这是 unbalanced-v2 规划师/专家架构的增量 diagnostic 边界切换探针，不授权训练、
formal 评测、发布或 live 服务。

运行时只通过 `gemma3_chat_unbalanced_v2_runtime_source_v2` 消费冻结的
sharded-v3 Producer lineage。该 source 模块独占不可变 Git/blob、checked-in
consumer preflight 和单份认证 SentencePiece 快照的认证责任。本运行时不调用 HF
tokenizer，也不会第二次按路径读取 `tokenizer.model`。

唯一接受的训练 receipt 为：

- schema：
  `anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-run-receipt.v2`
- status：`passed`
- mode：`full`
- phase/adapters：冻结顺序下精确九个训练臂

smoke receipt 和已废弃的 multiarm-v1 receipt 均拒绝。v2 receipt 必须绑定单一
冻结 base-file inventory、九阶段完全一致的 Q8 SCB identity，以及九个经过物理
文件认证的 adapter tree。

同一训练 receipt 还必须绑定唯一且经独立发布的 Teacher v2 identity：binding、
manifest、record schema、release receipt、外部 release attestation、shard
inventory、public identity，以及冻结的无正文计数。KV runtime 不会再次读取
teacher rows。Teacher v1 与当前互相冲突的临时 v2 candidate 名称均不可消费。
在 Producer 一次性返回唯一且签字的 FINAL v2 handoff 前，checked-in Teacher
字段有意保持为 `pending`，物理执行严格 fail-closed。

## 边界语义

handoff 严格按三阶段执行：

1. 对认证后的 ordered prefix 做一次 adapter-off frozen-base 真实 prefill；
2. 只启用一个被选中的 expert-planner，并追加私有 tail；
3. 只启用一个被选中的 output expert，继续私有 append-only tail。

交接的是 adapter-off prefill 实际产生的 cache。它被复制到私有分支后跨显式
adapter switch 传递，新 adapter 不重算此前前缀。source cache 与 sibling cache
必须保持逐层值不变。

boundary commit 绑定 Producer P/R/tree、source blob/record、训练 serialization、
单样本认证 serialization、tokenizer/model config、token 顺序、position、mask、
RoPE、model/base/Q8-SCB/adapter lineage、派生层拓扑以及 route/plan commit。

执行时从认证的 Gemma config 派生真实层拓扑。PASS 必须是 26 层，其中 22 层
sliding attention、4 层 full attention，并为每层记录前缀值摘要。

exact handoff 不是由两次 prefill 相等推断出来的，而是在两个 adapter 边界之后
分别验证。每一层都在 sequence axis `-2` 上切片：planner cache 已追加一个
tail token，output cache 累计追加两个 tail token；两个分支切片都必须与原始
adapter-off source cache 中对应的后缀逐值相等。full-attention 层保留全部 source
prefix；sliding-attention 层严格遵循认证 Transformers cache 语义，只保留最多
`sliding_window - 1` 的后缀（冻结窗口 512 时为 511 tokens）。receipt 会分别记录
source、branch、保留前缀计数，以及两个边界上的 K/V 摘要。

## 口径红线

物理 PASS 只能声明：

- planner 与 output 两个边界上的 retained-prefix K/V 逐层精确相等；
- adapter-switch 边界避免了前缀 prefill 重算。

永远不声明 shared storage、zero-copy、RDMA 或 full-generation KV sharing。
`DynamicCache` 的拼接可能重新分配 tensor，因此 pointer 相等只作为观察值，绝不
作为证明。

latent packet 当前禁用。未来若启用 soft prefix，必须冻结维度、serialization、
position 和 commit digest，且不得把它描述成隐藏 CoT。

## Fail-closed 执行

`--status`、`--validate`、`--dry-run` 全部 model-free。只有配置绑定 full-run
receipt/schema/sidecar/model 的物理 identity，并显式设置
`execute_authorized=true` 后，`--execute` 才可能运行。

execute 还必须满足：

- 仅本地加载模型；
- 独占 GPU 锁；
- 执行前后认证 base files 与九个 adapter tree；
- 推理前后 Q8 SCB identity 一致；
- cold recompute 与 handoff 的 logits/argmax 对比；
- source/sibling cache 隔离与 private-tail 校验；
- adapter、position、mask lineage 错误均被拒绝；
- receipt 原子 create-once 发布。

已有 receipt 永不覆盖。lock 与 staging 只在物理 file identity 仍属于本进程时
清理。prompt 正文、raw token IDs、raw logits、cache values 均不落盘；receipt
只允许 hash、布尔值、计数、延迟和峰值显存标量。

checked-in config 有意保持 model-free 且 execution-blocked。其验证测试不会触发
GPU、模型、provider 或网络。
