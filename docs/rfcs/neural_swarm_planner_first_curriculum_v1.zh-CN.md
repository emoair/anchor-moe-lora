# Planner-first 专家课程 v1

状态：additive、model-free 预检契约  
Schema：`anchor.neural-swarm-planner-first-curriculum.v1`

## 为什么需要这一层

冻结的 multi-arm runtime 虽然是严格串行（`concurrency=1`），但物理臂顺序并不是
Planner-first：五个输出专家在前，三个 Planner 臂在后。专家数据也来自预先认证的静态
资产，并非由已经训练、冻结的 Planner 提交问题后再投影出来。

本层诚实记录这个事实，不把旧顺序换个名字冒充 Planner-first。它不修改现有 runtime、
配置、schema、launcher、数据或 GPU lock；旧 runtime 继续作为不可变的叶子执行器。

两个旧层级契约依赖早于 LF 属性。它们绑定 canonical LF Git-blob 字节；验证器只允许
这些完全相同字节在旧 Windows 工作树中的可逆 CRLF 展开。三个冻结训练 runtime pin
仍要求逐字节一致。

## 强制顺序

1. 先用认证的 Planner 资产训练主臂 `planner_q_only`。
2. 冻结该适配器，并绑定训练收据与参数 inventory。
3. 只允许这个冻结 Planner 产生不可变问题 payload；body-free manifest 记录 payload
   哈希、语义哈希、route commit，并为每道问题只选择一个专家。另有独立生成收据把
   manifest 绑定到冻结 adapter、生成实现和真实 model-request 计数；只在 manifest
   里抄一遍 adapter SHA 不算证明。
4. 把已提交的问题分别投影成五个专家 inventory。
5. 五个 inventory 全部闭合后，才给每个专家签发 leaf-preflight ticket。票据本身不授予
   GPU 权限，专家在现有单 GPU 约束下仍逐个串行执行。

诊断臂 `planner_o_only` 与 `planner_q_plus_micro_o` 不能解锁下游。每个问题 ID 都绑定
冻结 Planner 收据、Planner adapter、源记录、真实语义身份、不可变 payload、语言、
split 与唯一被选专家；其中任一字段变化，问题 ID 和 route commit 都必须变化。

每张专家票据还必须绑定问题生成收据。因此静态问题清单、未冻结 Planner 或诊断
Planner 臂都不能改名冒充“Planner 产生的专家数据”。

专家叶子顺序固定为 `humor`、`serious`、`angry_style`、`tool_call`、
`review_audit`。这是新增控制层的发放计划，不修改旧 `TRAINED_ARMS`。

## 严格单向，不回总控

拓扑只有一条方向：

`Planner 训练 -> Planner 冻结 -> 问题提交 -> 专家数据投影 -> 专家开工`

专家开始后，其输出和私有 KV tail 不得回 Planner 做投票、改写或第二次总汇。review
仍是明确的专家/离线旁路，不是必经的二次总控节点。

## 当前权限边界

当前 validator 完全 model-free。`expert_leaf_preflight_ready` 只表示上游身份完整，
可以进入新增叶子 launcher 的预检。冻结的旧 runtime 本身并不会校验这些票据，仍可按
旧静态顺序直接调用，所以 decision 同时明确
`planner_first_leaf_launcher_integrated=false` 与
`expert_training_execution_blocked=true`。它不表示本提交已经生成问题、执行训练，也
不授予 GPU、formal 或 release 权限。

机器真相源：

- `configs/research/neural_swarm_planner_first_curriculum_v1.json`
- `configs/research/neural_swarm_planner_first_curriculum_v1.schema.json`
- `configs/research/neural_swarm_planner_first_runtime_bundle_v1.schema.json`
- `configs/research/neural_swarm_planner_first_decision_v1.schema.json`
- `src/anchor_mvp/research/neural_swarm_planner_first_curriculum_v1.py`
