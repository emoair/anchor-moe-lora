# RFC：层级规划、激活对齐与只读 RDMA 交接

状态：冻结的 contract-only 研究红区
版本：`anchor.neural-swarm-hierarchical-planner-rdma.v1`

## 目标

这是一条 additive 架构线：先把请求拆成细粒度因果 DAG，再让稀疏规划链处理各子任务，
最终只由一个专家解码，即 **single output expert**。它不改当前冻结的 4,300 条数据，
也不授予训练、质量、formal、部署、zero-copy 或 RDMA 权限。

执行顺序不可变：

1. **P0**：adapter-off 的冻结底模公共前缀。
2. **P1**：一个总规划师提交，判断聊天还是干活，并识别工具、情绪或审查需求。
3. **P2**：按任务 DAG 拓扑顺序提交一至三个被选中的专家规划师；固定为模式、工具、
   情绪规划师。
4. **P3**：为被选链生成 content-addressed 的激活组 commit。
5. **P4**：仅一个输出专家接收获准的已提交 lineage，并追加自己的私有 tail。

禁止回到总规划师、投票、加权输出聚合或多专家共同解码。固定输出专家为幽默、严肃、
生气风格、工具调用、审查；最终选择数永远为一。审查保持离线旁路，不进入生成回路。

## 语义拆分与激活契约

每个任务 DAG 节点必须来自真实的 `operation`、`operands`、`constraints`、
`dependencies`、`expected_relation`。hash、行号、case、evidence ID 或语言标签只能
绑定身份，不能制造语义唯一性。DAG 必须无环且有界，每个子任务至多选择一个规划师。

规划交接使用固定 256 维 float32 packet，序列化为 `canonical_f32_le_v1`，绑定明确的
绝对位置、SHA-256 commit 和可复算边界。它是接口表示，不是隐藏思维链。对齐目标包括
被选链的对比相似性、错误路由 margin 分离和依赖一致性。数值阈值仍为
`pending_calibration`，禁止猜测或静默放宽。

## KV / cache lineage

显式 adapter switch boundary 后，可以交接上一阶段实际产生的 ordered prefix value；
这叫 boundary-switched execution，不等于用新 adapter 重算旧前缀。

当前 Hugging Face DynamicCache 只能声明 exact ordered prefix-value 与
prefill-compute handoff。尾部追加常会 cat/reallocate，因此相同 cache 对象不能证明相同
storage/data_ptr。当前口径固定为 `shared_storage=false`、`zero_copy=false`、
`full_generation_kv_shared=false`。

规划师私有 tail 不得输出。最终专家只接收已经提交的 lineage，随后维护 append-only
私有 tail。compatibility identity 必须绑定 token 顺序、位置、mask、RoPE、模型、
tokenizer、template、adapter lineage、attention implementation，以及 Gemma 3 的混合
注意力：22 个 window=512 的 sliding-window 层和 4 个 full-attention 层。

## 被选专家组与 RDMA 红线

被选组可以包含总规划师、DAG 选中的专家规划师和唯一输出专家；只可读取已经 commit、
不可变、content-addressed 的 activation/prefix pages。未被选专家零可见性。

RDMA 只优化运输，不提供路由正确性。正确性必须在 local-copy fallback 下保持一致，
否则 fail-closed。禁止 RDMA 写、future/uncommitted read、live private-KV 传输。本地
alias、CUDA IPC/peer access、GPUDirect RDMA 是三种不同 transport class，不能混用
attestation。

当前没有任何物理 RDMA 证明。未来必须验证注册内存区、owner/reader identity、只读
key scope、epoch/lease/revocation/ABA、防篡改 digest、owner 侧 pointer/offset/stride/
layout、token/position/mask/RoPE/model/tokenizer/template/adapter lineage、完成 fence 与
lifetime、stale read 拒绝，以及 22+4 混合注意力。只有 segmented/paged backend 且逐层
证据完成后，才允许把 `shared_storage=false` 或 `zero_copy=false` 改成 true。

## 串行对齐与评测

训练顺序固定为：总规划师训练并冻结、各专家规划师训练并冻结、被选链激活对齐、最后
在冻结上游 commit 上训练被管辖的输出专家。该架构对当前 4,300 条 FINAL 增加 0 行。

对照必须包含 oracle route、仅总规划师、未对齐的总+分链、已对齐链、错误路由、随机
路由和 local-copy transport。报告 DAG exactness、路由准确率、激活相似度、错误路由
margin/delta、唯一输出专家不变量、prefix-value equality、节省的 prefill compute、
物理可用时的传输 bytes/latency、stale/revoked 拒绝、fallback rate、tool grounding、
情绪风格准确率和身份一致性。

机器真相源是
`configs/research/neural_swarm_hierarchical_planner_rdma_v1.json`，由
`src/anchor_mvp/research/neural_swarm_hierarchical_planner_rdma_v1.py` 校验。
