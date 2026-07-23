# 五角色 Q-only 合成诊断集 v1

## 边界

这是一份面向五个专家的确定性、仅诊断数据集：`planner`、
`tool_policy`、`frontend_gen`、`frontend_review` 与 `security_gate`。
数据目录由 OpenAI GPT-5.6-sol 辅助编写。本地构建器不会调用供应商、
网络、模型或 GPU；但“构建时请求为 0”不能被误报成“内容由 0 模型编写”。

它不是正式训练数据，不替代已冻结的 100 条夹具，也不满足未来独立轨和
factorial 轨各 600 条的物化要求。

## 精确规模

目录包含 200 个互不相同的 task bundle；每个 bundle 为五个角色各生成一条
主视图：

```text
200 个独立任务 × 5 个角色 × 1 个主视图 = 1000 条
```

每个角色恰好 200 条，其中 train 160 条、eval_proxy 40 条。英文和简体中文
各有 100 个独立任务语义；五个 stratum 在每种语言中各 20 个 bundle，并在
每个“语言 × stratum”单元内确定性切成 16 个 train 与 4 个 eval。一个任务
的五角色视图绝不跨 split，且共享 inner task ID 与 chain-root 哈希。

这 200 个任务不是中英翻译对：`translation_pair_count=0`，英文与中文的
语言无关语义身份集合交集为 0；五个角色均覆盖同一组 200 个
`task_semantic_sha256`。

## 输出与因果边界

唯一物化视图是 `concise_rationale_plus_json`。其中 rationale 是可审计的
简短决策摘要，不是隐藏思维链。每条 prompt 只能看见任务、约束、角色指令和
此前已提交阶段的短摘要。当前目标与未来目标只保留哈希绑定的库存项，并被列入
`forbidden_segment_ids`；它们的 ID、引用、正文、摘要和目标哈希均不得进入
物化 prompt 或序列化训练输入。此前阶段只提供短提交摘要以及非敏感的角色/阶段
标签，不提供目标哈希或私有 KV。

外部角色名保持兼容，`canonical_stage` 映射如下：

| role | canonical stage |
| --- | --- |
| `planner` | `planner` |
| `tool_policy` | `tool_policy` |
| `frontend_gen` | `domain_builder` |
| `frontend_review` | `domain_review` |
| `security_gate` | `security` |

## Q-only 主线

`q_only` 是唯一主训练臂；`o_only` 与 `q_plus_o` 仅是执行阶段的诊断对照。
它们不会生成重复数据行，也绝不会被序列化进 prompt 或 target。旧的
`wide_lora` 对照不被继承。数据集本身不选择胜者，不授权正式训练，也不声称
质量结论。

## 能力元数据

`simple_tool_search` 与 `micro_coding` 是为后续显式目录标注预留的能力名。
当前冻结目录没有权威的逐 bundle 能力标签，因此 manifest 将两者都记录为
`unavailable_no_explicit_catalog_label`，计数为 null，且不声称任何配额。
这让缺口可被机器审计，同时避免根据题目正文猜测分类，也不会增加行、视图或
训练臂。

## 构建与审计

在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"

python scripts/research/build_synthetic_five_role_qonly_diagnostic_v1.py build `
  --repo-root . `
  --config configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_five_role_qonly_diagnostic_v1

python scripts/research/build_synthetic_five_role_qonly_diagnostic_v1.py audit `
  --repo-root . `
  --config configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_five_role_qonly_diagnostic_v1

python -m pytest -q tests/test_synthetic_five_role_qonly_diagnostic_v1.py
```

构建器拒绝覆盖已有目录；它先在同父目录临时路径完成写入与全量审计，再重验
所有输入快照并原子发布。审计对每个文件只取一次 bytes 快照完成哈希、解析与
计数，最后再次做 TOCTOU 检查。所有文件必须小于 50 MB，并强制要求
`manifest.json.sha256`。

本任务不会创建 tag 或 release。
