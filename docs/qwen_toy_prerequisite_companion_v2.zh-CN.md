# Qwen toy 前置条件 companion v2

## 目标

本 companion 将 Consumer 的 request-local Qwen trigger receipt 认证为 frozen
Producer v1 前置包之上的非变异叠加层。它不改写 v1，只把 trigger 这一项从
`pending_request_local_materialization` 独立认证为 `ready_diagnostic_only`。

Consumer 的有效条件是合取关系：

```text
frozen v1 prerequisite AND companion v2
```

它不授权训练。六类受保护 source-ID inventory 仍只有 2/6 ready，四类
unavailable；因此零交集证明仍阻塞，v1 toy attestation 仍未生成。

## 非目标

本包不会：

- 修改 canonical Gold、held-out、scaffold 正文或任何 frozen v1 字节；
- 输出 prompt、answer、preview、content、原始 token IDs 或全局 token index；
- 执行 Consumer 实现，或启动 tokenizer、模型、provider、网络、GPU；
- 重新执行 request-local materialization；
- 宣称实时远端、签名 commit、formal threshold、数值等价、质量、物理 KV
  共享、多流、zero-copy 或完整生成 KV 共享；
- 绑定独立的 Gemma proxy 实验。

`proxy_signal_passed=true` 仅继承为 diagnostic provenance，不是 formal 或
numeric-equivalence 结论。

## 认证数据流

```text
Producer v1 metadata（744e23f 固定字节）
        +
Consumer Git blobs（固定 commit 7cb1f745 / tree 67ca22bd）
        |
        v
严格 bytes/hash/size 检查
        |
        +-- Consumer config 与 receipt 的 Draft 2020-12 真实校验
        +-- sidecar 精确格式与 canonical JSON 检查
        +-- 7cb1f745 原始 tree/parent 与 b0441e6 祖先检查
        +-- 7cb1f745 -> 本地 remote-tracking ref 祖先检查
        +-- 清除继承的 GIT_* override，拒绝 replace/grafts
        +-- request-2 投影和安全声明检查
        |
        v
临时四文件 fixture
        |
        +-- 同字节重解析
        +-- 发布前重验本地快照和 8 个 Git blobs
        +-- 再验 ancestry/tree/local-ref
        |
        v
原子、禁止覆盖式发布
```

源文件通过 binary `git cat-file blob <commit>:<path>` 读取，不读取 sibling
Consumer worktree 文件。`refs/remotes/origin/research/neural-swarm-kv` 只证明
固定 release commit 仍在本机观察到的 lineage 内；它不是网络 fetch，也不是
实时远端证明。`7cb1f745...` 仅提供内容寻址 provenance，不宣称签名认证。

## 为什么 receipt 必须是 request-local

权威边界来自 request 2 的完整序列化和一次完整 tokenization。trigger 覆盖
44 tokens 中的 `[25,33)`，坐标为 zero-based/end-exclusive；覆盖跨度的 leading
overhang 为 0，trailing overhang 为 1 UTF-8 byte / 1 code point。孤立 trigger
编码明确不是权威来源。

这保持“两请求”规则：Planner 输出先校验并 commit，再作为 Expert 请求输入。
Planner request-1 私有 KV 不得表示为可供 Expert 复用的 KV。companion 只复制
认证后的 metadata receipt，不复制原始 IDs 或请求正文。

## Manifest 契约

封闭的 Draft 2020-12 schema 绑定：

- `producer`：companion config/schema/implementation、基线 commit/tree、
  canonical JSON、单快照、重解析、末端重验和原子发布；
- `v1_dependency`：v1 manifest schema、pending-trigger schema、manifest、
  sidecar、inventory root、2/6 coverage 和四类缺口的精确身份；
- `consumer_dependency`：release commit/tree、baseline 语义、本地 tracking ref、
  有序八文件 path/SHA/bytes inventory 及 provenance 限制；
- `trigger_materialization`：receipt 副本、tokenizer/chat/R2/ordered-ID digest、
  digest 算法、span、overhang、出现次数及禁止复用/输出标志；
- `inventory_status`、`proof`：保持 v1 inventory 状态，零交集证明仍阻塞；
- `verification`：真实 schema、sidecar、投影、span、重解析、祖先及末端重验；
- `execution`：provider/network/model/GPU/protected-body/worktree 读取为 0，
  初始和末端 Git blob 读取各 8 次；
- `claims`：只允许 diagnostic trigger ready，其余 formal/training 声明均 false。

有序 Consumer artifact digest 为：

```text
SHA256(canonical_json({
  "domain": "anchor.qwen-toy-prerequisite-companion.consumer-artifacts.v2",
  "artifacts": [八个有序 path/SHA/bytes 条目]
}))
```

Canonical JSON 使用 UTF-8、key 排序、紧凑分隔符、不做 Unicode normalization、
禁止非有限数，文档末尾恰好一个 LF。Sidecar 为 lowercase SHA-256、两个 ASCII
空格、basename、一个 LF。

## 冻结身份

| Artifact | SHA-256 |
|---|---|
| Companion config | `21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f` |
| Companion manifest schema | `596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119` |
| Companion implementation | `dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e` |
| Fixture manifest | `7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115` |
| Manifest sidecar 物理文件 | `f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095` |
| Consumer artifact inventory | `bf38c88fd993d4804f9a624bbf98330a9d4572a6803a80c2f57a6beb05b5f567` |
| 复制的 source receipt | `ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e` |
| 复制的 source sidecar 物理文件 | `ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a` |

Producer 基线 commit 为
`744e23f975b13923903f5fabe04c32e74ea25dc4`，tree 为
`90cb962f5341717501fcb16caef13db8922f1cb4`。Consumer materialization commit
为 `7cb1f7454a76fa3c8c9f46d64da9f11244b51c54`，tree 为
`67ca22bd2f9d50642bf88e484408082abebe2126`，required ancestor 为
`b0441e6beaa07b180d7fc69e462b4d2babf21792`。

最终 Producer release commit 位于 artifact hash DAG 之外，并在 commit/push 后
回报；commit 无法安全地内嵌自身身份。Consumer 应同时绑定最终回报的 commit
和以上物理 SHA。

## Fail-closed 条件

出现下列任一情况，构建或审计必须失败：

- config/schema/implementation/v1 字节、Git blob path/SHA/size、原始 commit
  parent/tree、ancestry 或本地 ref lineage 漂移；
- 出现 Git replacement ref、legacy graft 文件或继承的 `GIT_*` 仓库 override；
  Git object 读取始终设置 `GIT_NO_REPLACE_OBJECTS=1`；
- sidecar 缺失、格式不精确、CRLF/BOM、文件名错误，或围绕篡改内容同步重算；
- JSON 有重复 key、非有限数、额外 schema 字段，或应 canonical 的字节不规范；
- request-2 digest、span、坐标语义、宽度、overhang、出现次数或 tokenization
  次数漂移；
- 任一 formal/training/KV/quality 禁止声明被晋级；
- v1 coverage 不是 2/6，或六类 inventory 齐备前出现零交集结果；
- 输出已存在、路径逃逸、出现 symlink/reparse point、首快照后输入变化，或原子
  发布失败。

聚焦负向测试覆盖：manifest+sidecar 同步篡改、receipt+sidecar 同步篡改、异常
sidecar、身份漂移、真实 schema 校验、逐字节确定性重建和禁止覆盖。

## 复现

使用已安装 dev dependencies 的 Python 3.10+ 环境。以下命令不会启动 provider、
网络、模型或 GPU：

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --artifact fixtures/research/qwen_toy_prerequisite_companion_v2
python -m pytest tests/test_qwen_toy_prerequisite_companion_v2.py -q
python -m ruff check `
  src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py `
  scripts/data/build_qwen_toy_prerequisite_companion_v2.py `
  scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  tests/test_qwen_toy_prerequisite_companion_v2.py
```

确定性重建必须使用不存在的输出目录：

```powershell
$env:PYTHONPATH = "src"
python scripts/data/build_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --output tmp/qwen-toy-prerequisite-companion-v2-rebuild
```

重建出的四个文件必须与发布 fixture 逐字节一致。

## 迁移与真实未完成项

Consumer 不得修改或放宽 v1，只能把 companion v2 加为第二个 mandatory
authenticated input，并仅投影 request-local trigger 字段。只认识 v1 的旧
Consumer 会继续安全阻塞。

真实未完成项：

- `gold_partition`、`partial_gold_export`、`legacy_heldout_cases`、
  `synthetic_scaffold` 的 body-free per-ID inventory；
- 六源 namespaced zero-intersection proof 和 v1 toy attestation；
- frozen formal-v3 snapshot/projector/generic/source-disjoint/release lock；
- formal tokenizer/adapter/tensor/release 身份、真实训练、物理 KV backend/CUDA、
  正式质量与性能评测。

所有独立 gate 未冻结并被消费前，`training_authorized` 与
`formal_training_authorized` 始终为 false。
