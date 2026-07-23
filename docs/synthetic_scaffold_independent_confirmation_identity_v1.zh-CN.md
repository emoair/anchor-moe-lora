# 合成脚手架独立确认身份 Producer v1

## 状态与范围

本文档规定一个纯元数据 Producer 产物，包含两个逻辑上严格分离的 60-bundle
track。产物不包含训练样本、prompt、答案、rationale、工具轨迹、受保护样本正文或
token ID；也不修改冻结的自然语言脚手架 fixture、canonical Gold、heldout 或任何
formal-v3 产物。

最终物理 SHA-256、字节数、分区身份、证明身份和有序 read-set 均由发布后的
manifest 绑定。在最终冻结前，本文档不复制这些身份。

两个 track 分别为：

1. `independent_confirmation`：60 个 bundle，其 namespace-neutral 任务语义相对
   discovery 全新，并且彼此不同；
2. `secondary_controlled_factorial`：60 个 bundle，按三因子设计有意复用 task 或
   template 身份。

两者不得合并。Secondary track 不能满足 Producer 所需的
independent-confirmation gate。

## 认证的 discovery bridge

Discovery 来源是冻结的 `synthetic_nl_scaffold_diagnostic_v1` Producer 元数据。
Bridge 仅消费新 config 绑定的精确 config、schema、closed grammar、implementation
identity、manifest 和 mandatory manifest sidecar，绝不打开四个 discovery JSONL
分区。

历史 fixture 有十个本地化 source-bundle view：五个英文、五个简体中文。旧
`source_bundle_id` preimage 包含语言、来源局部 key 和本地化元数据，因此它属于
grouping/provenance 身份，不是 namespace-neutral 语义身份。

Bridge 发布五个经 Producer 整理的、语言中立的完整 task semantic descriptor。
每个 descriptor 精确交叉绑定一个英文 view 和一个简体中文 view。两个本地化 view
共享同一个 `task_semantic_sha256`，同时保留不同的历史 source-bundle identity 和
localized-view digest。

这项双语映射是显式的 Producer-curated metadata assertion，不是自然语言字符串
语义等价的自动证明；单独一个粗粒度 archetype 也不能充当完整任务语义描述。

在共同域中重算历史 split 会暴露一个必须保留、不得隐藏或重写的事实：

```text
| discovery_train_task_semantic_ids
  intersection discovery_eval_proxy_task_semantic_ids | = 2
```

因此历史语言相关 split 并非 semantic-disjoint。现有 fixture 仍可作为 diagnostic
proxy，但其 eval-proxy 分区不能晋级为独立或 heldout 语义评测。

## 共同域身份

身份 preimage 使用严格 UTF-8 canonical JSON：递归 key 排序、紧凑分隔符、不做
Unicode normalization、禁止非有限数，末尾不加 LF。发布的 JSON 文档使用同一编码，
并只追加一个最终 LF。

### Task semantic identity

```text
task_semantic_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-task-identity.v1",
  "descriptor_schema_sha256": descriptor_schema_sha256,
  "ontology_sha256": ontology_sha256,
  "descriptor_atom_catalog_sha256": descriptor_atom_catalog_sha256,
  "descriptor": <完整语言中立符号化任务描述>
}))
```

`source_task_blueprint_sha256` 是 `task_semantic_sha256` 的 alias，不是第二个可独立
自报的 hash。

Descriptor 必须覆盖任务的 information-flow 行为、状态/证据拓扑、操作和状态转换
要求、约束类别及验收不变量。以下内容严禁进入 semantic preimage：

- 语言和本地化正文；
- source namespace、source-bundle ID 和 bundle key；
- split、factor、role、scaffold variant、noise 和 length augmentation；
- ordinal、UUID、时钟、随机盐及其他仅用于制造唯一性的字段。

这些排除项防止利用 namespace 或序号伪造 zero-overlap。每个 descriptor leaf 还必须
属于冻结的 `anchor.synthetic-scaffold-common-domain-atom-catalog.v1` 目录（241 个
atom；目录 SHA-256 为
`517f6b829bb78700b171a349d14541f75a9b76aa2a9267acb92a0e1a646d9545`）。未知的
language、locale、source、namespace、salt、nonce、seed、ordinal 或数字化 atom
一律 fail closed。只有 descriptor schema、ontology 和 atom catalog 三项上下文
SHA-256 完全相同时，task/template identity 才属于可比较的共同域。

### Template family identity

```text
template_family_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-template-identity.v1",
  "descriptor_schema_sha256": descriptor_schema_sha256,
  "ontology_sha256": ontology_sha256,
  "descriptor_atom_catalog_sha256": descriptor_atom_catalog_sha256,
  "descriptor": <完整语言中立结构化模板描述>
}))
```

Descriptor 表示脚手架接口和结构化渲染语义，而不是本地化措辞。本地化 template
view 可以有独立 view digest，但语言不能作为共同 family identity 的盐。

### Task-template pair identity

```text
task_template_pair_sha256 = SHA256(canonical_json({
  "domain": "anchor.controlled-factorial-task-template-pair.v1",
  "task_semantic_sha256": task_semantic_sha256,
  "template_family_sha256": template_family_sha256
}))
```

Pair identity 必须始终由认证的 task/template leaves 重算；record 不得自报或覆盖。

每个逻辑 inventory 都包含排序、去重后的共同域 leaves。逻辑 root 与物理文件
SHA-256 分开，并绑定 domain、count 和排序后的值。重复、缺失、乱序、格式错误或
未绑定的 leaf 一律 fail closed。

## Source bundle 与 split identity

`task_bundle_sha256` 继续作为来源/grouping identity。其 canonical preimage 绑定冻结的
task-bundle domain、track source namespace、source-bundle identity、language 和
`source_task_blueprint_sha256`，排除 role、variant、noise、length 及后续 causal
augmentation。

必须先按 `task_bundle_sha256` split，再扩增五角色和两种 scaffold variant。未来
materializer 必须将五角色、两变体全部绑定到同一个 bundle，并交叉绑定 inner
`task_board.task_id`。本 v1 只发布身份元数据，不生成每个 track 预期的 600 条训练
record。

`source_bundle_id` 和 `task_bundle_sha256` 都不能证明 namespace-neutral 语义独立性。

## Independent-confirmation track

Independent track 冻结以下结构：

- 60 个 source bundle 和 60 个唯一 `task_semantic_sha256`；
- 30 个英文和 30 个简体中文 bundle；
- track 内跨语言 translation pair 数为 0；
- 五个 information-flow strata；
- 十个 language/stratum cell，每 cell 六个 bundle；
- 每 cell 为 4 train + 2 eval_proxy；
- 总计 40 train + 20 eval_proxy bundle。

每个 task semantic leaf 都在与 discovery 完全相同的共同域中重算。Producer proof
对实际 discovery 与 independent leaves 逐 ID 做集合交集；namespaced ID、aggregate
count 或 source 名称都不能替代该证明。

该 track 的唯一正面结果是身份构造通过认证，以及 zero-overlap proof 可机械重算。
它不证明质量、泛化、统计显著性或训练就绪；`eval_proxy` 不是 heldout。

## Secondary controlled-factorial track

Secondary track 同样有 60 个 bundle，并保持相同 language/stratum 与每 cell
`4 train + 2 eval_proxy` 总数。它包含三个 factor，每个 20 bundle：

| Factor | Task membership | Template membership | Pair 要求 |
| --- | --- | --- | --- |
| `old_task_new_template` | 属于 discovery task inventory | 不属于 discovery template inventory | 不属于 discovery pair inventory |
| `new_task_old_template` | 不属于 discovery task inventory | 属于 discovery template inventory | 不属于 discovery pair inventory |
| `new_task_new_template` | 不属于 discovery task inventory | 不属于 discovery template inventory | 不属于 discovery pair inventory |

该 track 有意不声明 global task 或 template zero-overlap；global
task-template-pair zero-overlap 则是强制要求，并由 leaves 重算。

每个 language/stratum cell 中，每个 factor 有两个 bundle。冻结的 rotation 给一个
factor 分配 `2 train + 0 eval_proxy`，另外两个 factor 各分配
`1 train + 1 eval_proxy`；factor 内两个 bundle 按 `task_bundle_sha256` 排序决定
split。十个 cell 汇总后的 factor 配额为：

```text
factor 顺序： old_task_new_template,
              new_task_old_template,
              new_task_new_template
train:        13 / 14 / 13
eval_proxy:    7 /  6 /  7
```

Proof 发布逐 bundle membership truth value，并根据 discovery inventories 重算每个
布尔值。标签不能代替 membership。该 proof 通过也只代表 secondary factorial
plumbing，不能满足 independent confirmation 或 bundle generalization。

## 纯元数据产物与读取边界

产物只包含 closed-schema metadata record、IDs-only inventory、proof metadata、
manifest 和 `manifest.json.sha256`。其中不得包含 prompt、answer、rationale、task
text、constraint prose、tool trace、token ID、Gold row、heldout row 或 scaffold
record 正文。

Manifest 绑定显式、有序 read-set。Producer 侧物理读取 canonical config、四个
schema 和 implementation；从 sibling source repository 物理读取精确三个已认证来源
元数据文件（diagnostic config、manifest、mandatory sidecar），再加一个单独分类的
已认证 consumer-plan contract。Generator 与 closed-grammar identity 是从来源
manifest 读取的传递绑定，本 Producer 不打开它们的物理文件。Producer 不得递归发现
输入，也不得打开 discovery partition、Gold、heldout、protected scaffold JSONL、
provider、model、tokenizer 或 GPU 资源。

该 exact semantic-file read-set 与本地 Git provenance 读取分栏。Manifest 还逐项列出
并哈希固定的 11 项 Git metadata/object read-set：worktree/Git-dir 解析、graft/replace
检查、两个 commit object、四个精确 blob，以及 source-to-plan ancestry 检查。这些仅是
本地 object database 认证，不声称 live remote 或签名 provenance。

对每个结构化 JSON/YAML 输入和输出（implementation 与 sidecar 使用 raw-byte identity
检查），Producer 强制执行：

1. 只允许 repo-relative lexical path，拒绝 absolute path、`..` 和反斜杠；
2. 对每级路径组件拒绝 symlink、junction、reparse point 和非普通文件；
3. 只取一次 immutable bytes snapshot，并用同一 bytes 做 hash、parse、schema
   validation 和 count；
4. 严格拒绝 duplicate key 和非有限数；
5. 对同一 bytes 做 reparse；
6. 发布前重新检查 hash、size 和文件 identity，检测 TOCTOU；
7. 强制 exact artifact layout 和输出大小上限；
8. 写入同级临时目录，再做 atomic no-replace publish；
9. sidecar 必须是精确 lowercase 行：

```text
<manifest sha256><两个空格>manifest.json<LF>
```

目标已存在、发布竞争、source drift、同步篡改 manifest 或 sidecar drift 均 fail
closed。

## Claims 与授权边界

顶层 `claims` 对象全部保持 false。本产物不授权 training 或 formal release，也不
声称 heldout evaluation、quality、statistical significance、bundle generalization、
physical KV reuse、Q-reader implementation、zero-copy reuse 或 numeric equivalence。

Schema validation 或认证的 zero-overlap proof 等机械审计证据可以通过，但不能因此
晋级任何 claim。Provider、network、model、tokenizer、GPU、training、Gold-body、
heldout-body、scaffold-body 和 dataset-body 的运行计数全部保持 0。

## 复现

在 Producer 仓库根目录运行。`PYTHONPATH=src` 必须解析到本地 implementation；
`source-root` 只用于读取 manifest read-set 中精确认证的元数据文件。

受支持的信任路径是 fresh CLI process：`--config` 必须解析到 canonical config；
schema 和 implementation 使用实现内冻结的 canonical path，config 与四个 schema 的
SHA-256 也编译进 implementation；已加载 implementation 的路径与 bytes 在模块 import
时冻结，并持续重验到发布完成。替代 config、同步替换的平行 schema 信任根或 stale
imported-module 执行一律 fail closed。

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.swebench.synthetic_scaffold_independent_confirmation_identity build `
  --repo-root . `
  --source-root ../anchor-moe-lora-neural-swarm `
  --artifact fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1

python -m anchor_mvp.swebench.synthetic_scaffold_independent_confirmation_identity audit `
  --repo-root . `
  --source-root ../anchor-moe-lora-neural-swarm `
  --artifact fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1
```

`build` 使用 no-replace 语义，要求目标目录事先不存在；`audit` 用于认证冻结
artifact。需要做 byte-identical deterministic rebuild 时，必须使用另一个空
artifact 路径并比较精确 bytes，不能覆盖冻结 fixture。

## 已知限制与未完成项

- 历史 discovery semantic split intersection 仍为 2；本 Producer 记录事实，不重写
  历史。
- 五组双语 discovery 映射属于 curated metadata assertion，不是通用语义等价定理。
- Secondary track 的两个 factor cell 有意复用 discovery identity，因此不能满足
  independent confirmation。
- 两个 track 都尚未生成未来 600 条 role/variant record。
- Provider 请求、model load、tokenizer binding、GPU 执行、training、physical KV、
  multistream runtime、quality evaluation 和 performance evaluation 均不属于本产物。
- Protected-source inventories 与 formal-v3 snapshot/projector/generic execution/
  source-disjoint/release-lock gate 仍是独立前置条件。
- Consumer 在绑定最终 manifest、schema、config、implementation、partition、proof 和
  sidecar identity 前继续 fail closed。
- 所有最终 SHA-256 和精确 count 只以最终 manifest 为准；临时 hash 不得作为 frozen
  input 往返传播。
