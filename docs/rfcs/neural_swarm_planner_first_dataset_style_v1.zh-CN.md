# Planner-first 数据样式 companion v1

这是一个 additive、metadata-only 的数据样式契约。它只约束 Planner
训练并冻结之后如何出题、如何把题目投影给专家；不会修改现有训练
runner，不会物化正文，不会启动模型，也不会授权 GPU 或训练。

## 只读训练证据

本设计只使用训练任务里的 body-free 事实：

- 最新 16,632 条数据的隔离门通过；
- 最新 500-step Planner/Router 数值训练阶段通过；
- 该尝试因执行 provenance 失败而保持 `consumed=false`；
- 历史 invalidated inventory 为 12,586 个 record ID、11,660 个规范化
  prompt；以及
- 本 companion 没有读取 holdout 正文。

因此不会把 provenance 失败误写成数据质量失败，也没有改训练工作树或
执行管线。

## 语义路由与物理投影分离

五个 expert asset 是“可用候选”，不是“本题五个都要激活”：

1. `humor / serious / angry_style` 中恰好选择一个主风格专家；
2. 只有任务确实需要工具时才加入 `tool_call`；
3. 只有任务确实需要旁路审查时才加入 `review_audit`；
4. identity 是所有已选投影共享的事实约束，不是第六输出专家，也不增加
   一行数据；
5. Planner 写入不可变 route commit 后退出，专家输出不回到 Planner
   做二次改写、投票或聚合。

每条物理投影仍只属于一个专家，所以兼容冻结父契约中的
`one_expert_per_question=true`：

```text
一个语义问题
  -> 一条主风格投影
  -> 0/1 条 tool sidecar
  -> 0/1 条 review sidecar
```

bundle validator 要求物理投影集合与真实 selected set 完全相等；禁止把
candidate inventory 直接复制成 activation set。

## 记录样式

每条 body-free projection row 绑定：

- semantic question、source record、source group、leakage family、prompt
  和 payload digest；
- frozen Planner receipt、adapter、question-generation receipt；
- 唯一 training expert 与 projection kind；
- `base / with_review / with_tool / with_tool_review` variant；
- identity 必需时的 provenance class 与 invariant digest；
- shared-prefix、salient、distractor、route evidence 的 segment identity；
- `target/future segment count=0`、不嵌正文、不存 raw token IDs。

真实问题 payload 必须放在独立不可变对象中，只通过 SHA-256 引用。

## 切分与防泄漏

本 metadata manifest 只允许 `train / calibration / eval_proxy`。holdout
不得作为行进入该 manifest，只能绑定 body-free inventory SHA；不得用来选
样式、挖 hard negative 或调参。

必须先切分，再扩语言视图、variant、hard negative 和专家投影。以下身份
禁止跨 split：

- `task_bundle_sha256`
- `source_group_sha256`
- `leakage_family_sha256`
- `prompt_digest_sha256`

历史 invalidated inventory 为强制输入，近重复口径固定为
NFKC/casefold/word/q-gram-5 Dice `< 0.90`。

## 路由 hard negative

Hard negative 是路由反事实，不是专家正样本。每条必须绑定正/负 route
commit，并且只允许改变一个语义变量：

- 主风格选错；
- 工具缺失或过路由；
- 审查缺失或过路由；
- identity bit 丢失；
- identity 错误归属。

不得从 holdout 派生，负 route commit 也不得进入正投影 inventory。

根据只读训练误差，本 companion 还强制覆盖 `serious + identity` 的
`base / with_review / with_tool_review`，并保留 `with_tool` 正向控制。
这里只冻结 metadata quota，不复用任何旧 holdout 正文。

## 当前边界

现有 GPU runner 尚未消费 v2 manifest。后续 additive launcher 必须先认证
并冻结 Planner，再生成外置问题 payload；question-generation receipt 还要
绑定本 style-contract SHA，随后才可验证 v2、导出父 v1 的 train-only
projection view，并严格串行调用 leaf runner。

在该集成完成独立审计前，所有授权仍为 false：

- `planner_first_leaf_launcher_integrated=false`
- `training_authorized=false`
- `gpu_authorized=false`
- `formal=false`
- `release_authorized=false`
