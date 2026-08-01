# Planner-fused QKVO 22,200 串行适配器 v1

这是一个新增的、仅元数据的门控层。它不修改现有 3,520 条 multi-arm
训练运行时，也不能启动 provider、模型、GPU 或训练。

唯一可授权的顺序固定为：

`source admission → rank64 Planner Q+K+V+O 训练 → 独立 Planner gate/eval → merge 到原始 HF base → token-equivalence 证明 → fused Q8 → 不可变 question/route commit → 六专家投影 → 六个串行 Q+O 专家叶子 → 各自独立 Q8 package → fused-base route head/runtime`。

只有 `planner_qkvo_rank64` 可以授权 route commit。门控会显式拒绝
`planner_q_only`、`planner_q_plus_o`、`planner_q_plus_micro_o`、
`original_base`、`v14` 和 `quarantine`。Planner 的学习率比例为
`Q:K:V:O = 1:0.2:0.1:0.01`。每个专家是 Q+O 叶子，
`lr_O/lr_Q = 0.1`；六个专家必须通过前驱 package 严格串行，
并发固定为 1。

该链路认证的是物理元数据，而不是自报布尔值。它要求原始 HF base、
QKVO adapter、训练 receipt、带独立身份的 gate/eval、merged base、
输入/输出 tokenization 等价证明和 fused-Q8 artifact 都具有 sidecar
绑定的物理 identity。route commit 还必须有不可变 inventory；每个专家
投影也必须有独立的物理 inventory leaf。所有声明的资产必须位于同一个
受信任词法根目录下；链接/reparse 路径、越根路径、字节漂移和 sidecar
漂移均会 fail-closed。

22,200 仍由 3,200 个 planner-bootstrap 对象与 19,000 个 legacy coding
对象组成。当前 19,008 条公开 metadata 只是候选池；在可重算的 19,000
确定性 selection 与独立授权的 3,200 source-admission 资产到位之前，
该门控保持 blocked。它不声称数据、Planner 训练、专家训练或 fused runtime
已经存在。

物理 identity 只允许单向绑定：后续 Teacher FINAL、Producer attestation、
Consumer acceptance 或 training-input authority 可以绑定前序 identity；
前序文件不能回写后续文件的精确 SHA-256。
