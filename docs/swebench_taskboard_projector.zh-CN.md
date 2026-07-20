# SWE-bench TaskBoard 投影器契约

[English](swebench_taskboard_projector.md)

## 状态与范围

TaskBoard 投影器是一个确定性的 Gold 后置研究转换。它只能在
`anchor.training-snapshot.v2` 快照连同 manifest 和 SHA-256 sidecar
一起冻结后运行。其输出只是 query-specialization 研究线的 proxy
数据集，不是 canonical Gold，不是蒸馏阶段，也不能单独证明任何模型或训练
已经晋级。

投影器只读取正式的 `train` 和 `calibration` 分区。它不得读取或输出 heldout
记录正文，也不得修改源快照、manifest、SHA sidecar 或任何 canonical Gold
文件。

## 版本化契约

固定策略位于
[`swebench_taskboard_projector_v1.yaml`](../configs/research/swebench_taskboard_projector_v1.yaml)。
每条输出 JSONL 必须通过
[`taskboard_projector_sidecar.schema.json`](../configs/research/taskboard_projector_sidecar.schema.json)
验证，输出清单必须通过
[`taskboard_projector_manifest.schema.json`](../configs/research/taskboard_projector_manifest.schema.json)
验证。

每条 `anchor.swebench-taskboard-sidecar.v1` 记录都是一个 provenance wrapper，
内部包含一条严格验证的 `anchor.query-specialization.v1` `training_record`。
inner record 继续使用封闭字段集合：`schema_version`、`id`、`pair_id`、
`variant`、`language`、`split`、`role`、`task_board`、
`attention_targets` 和 `target`。不能向 inner object 添加 provenance；来源
Gold 记录及文件、冻结快照及 manifest、源任务包、基础 TaskBoard、投影器、
配置和 sidecar schema 的标识与小写 SHA-256 全部由 wrapper 绑定。

wrapper 的 `id`、`pair_id`、`variant` 和 `split` 必须分别等于
`training_record` 内的同名字段；`stage` 与 `expert` 必须遵守下表：

| Canonical stage | Research expert |
| --- | --- |
| `planner` | `planner` |
| `tool_policy` | `tool_policy` |
| `domain_builder` | `frontend_gen` |
| `domain_review` | `frontend_review` |
| `security` | `security_gate` |

这些跨字段相等关系及 stage/expert 映射由投影器在 JSON Schema 验证之外执行
语义校验。

## 分区、可见性与增强规则

投影器必须先按源任务划分 `train` 或 `calibration`，然后才能生成变体。同一
任务的五个 stage view 必须保留在源分区中。`train` 生成配对的 `clean` 与
`noisy` 记录；`calibration` 只生成 `clean` 记录。

签入策略和每个输出 manifest 都必须显式固定
`split_group_key=task_bundle_sha256`、
`task_id_cross_binding_key=training_record.task_board.task_id` 和
`all_five_role_views_same_split=true`。task bundle 摘要是源 task ID 与按顺序
排列的五阶段源记录绑定的 canonical SHA-256。发布前，投影器必须从 clean
角色视图重建该摘要，并校验 bundle/task ID 的正反向唯一映射、唯一 split、
完整五角色集合及 train clean/noisy 配对。

唯一允许的噪声策略是 `stale_duplicate_overlay`。noisy 记录只能把同一源任务
的一至多个 block 复制成 stale overlay block，不能引入其他任务的文字、标识
或事实。wrapper 的 `augmentation` 必须记录原 block ID 与 overlay block ID；
clean 记录的两个数组必须为空。禁止在划分 split 之前执行增强。

因果可见性采用 fail-closed。每个 stage 只能消费其映射 expert 可见且在该
stage 已可用的 block。未来 stage 或其他 forbidden block 只能作为结构化负
监督保留，不能渲染到模型可见 prompt 中。投影过程不得静默提升 block 的
`commit_state`。

## 实现与规模边界

输入和输出文件都从单次内存字节快照完成认证：SHA-256、字节数、记录数和
JSON 解析使用完全相同的字节，并在原子发布前再次复核完整 inventory。当前
研究实现仍会把已认证 Gold 和投影记录放入内存；目前只对 15 条集成夹具完成
验证，尚未证明全量题库的峰值内存和输出体积。对大型冻结快照运行前，必须先
做可测量的分片或流式 pilot，不能把本 MVP 当成全量吞吐承诺。

## 正式命令

正式命令必须显式传入冻结 manifest 的 SHA-256。snapshot 目录必须包含与其
匹配的 `manifest.json` 和 `manifest.json.sha256`，且不能是 heldout 来源目录
或输出目录。

```powershell
python scripts/data/project_swebench_taskboard.py `
  --config configs/research/swebench_taskboard_projector_v1.yaml `
  --snapshot-dir <FROZEN_TRAINING_SNAPSHOT_V2_DIRECTORY> `
  --snapshot-manifest-sha256 <FROZEN_MANIFEST_SHA256> `
  --output-dir <NEW_RESEARCH_OUTPUT_DIRECTORY>
```

若任一冻结输入绑定缺失或不匹配、源 split 不是 `train` 或 `calibration`、
运行将读取 heldout 正文，或者任一输出记录/manifest 验证失败，运行必须
fail closed。

## Manifest 必需证据

输出固定为 `train/clean.jsonl`、`train/noisy.jsonl` 和
`calibration/clean.jsonl` 三个文件。manifest 必须绑定精确的输入快照、
producer 身份与 schema/config 哈希（包括 manifest schema 自身的字节哈希）、
全部三个输出文件、唯一 task bundle 数、task ID 摘要，以及按 split、
variant、canonical stage、expert 和 language 汇总的记录数。有效 manifest
还必须证明以下固定不变量：

- `canonical_gold_written=false`
- `provider_requests=0`
- `heldout_content_read=false`
- `heldout_content_emitted=false`
- `split_preserved=true`
- `augmentation_applied_after_split=true`
- `claim_scope=research_proxy_only`

这些事实只能证明确定性投影遵守了研究边界。训练晋级仍需另行冻结训练
manifest，并通过未见任务评测、因果证据删除、干扰项不变性、严格 JSON
指标、跨角色分离度以及项目的正常审查门禁。

后续 hash-only generic execution、source-disjoint 与训练 release 冻结阶段
见 [`swebench_training_release.md`](swebench_training_release.md)。在真实冻结的
projector manifest 与已绑定 consumer contract 可用前，它们不得发布 ready
release lock。
