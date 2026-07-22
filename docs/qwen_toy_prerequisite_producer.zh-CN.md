# Qwen 诊断 Toy 前置 Producer

## 目标与非目标

这是一套 companion producer：在不修改 frozen
`anchor.qwen-toy-source-disjoint-attestation.v1` 的前提下，把 Qwen
diagnostic-only toy 验证所需的前置状态做成可机器认证的 artifact。它负责冻结
metadata-only source-ID inventory、从本地封闭语法确定性生成 toy 分区，并精确声明
哪些前置仍不可用。

它不会读取或输出 Gold、heldout、scaffold 样本正文；不会调用 provider、加载模型、
使用 GPU、授权正式训练、证明语义/内容唯一性或给出质量结论。不可用的 inventory
绝不能用 `count=0` 代替。

## 数据流

1. 对 producer config、schema、实现、closed grammar 和允许读取的源元数据分别做
   单次 bytes snapshot 认证。
2. 仅对同一份不可变 bytes 重解析；结果不一致即失败。
3. 为六类 protected source 分别生成 inventory manifest 和强制
   `manifest.json.sha256` sidecar。
4. 只有存在完整、body-free、逐 ID 集合时，才输出经过 domain separation 的 opaque
   ID leaves。
5. Toy 只从 closed grammar 与公开确定性 seed 生成，共 8 条 diagnostic 记录。
6. 独立 auditor 不 import generator，而是自行重建期望分区。
7. 发布前末端重验全部输入，写 audit/main manifest 与 sidecar，再原子发布目录。

Generator 的语义 read-set 固定为 generator implementation、pinned config、closed
grammar 三项；不会把 Python/操作系统的全部 I/O 冒充应用语义读追踪。

## 六类 protected source

固定顺序为 `swebench_source`、`gold_partition`、`partial_gold_export`、
`heldout`、`legacy_heldout_cases`、`synthetic_scaffold`。

- `swebench_source` 为 ready：从 manifest 认证的两份纯 ID allowlist 重算 19,008
  个 ID，不打开 candidate task 正文。
- `heldout` 为 ready：只消费 heldout manifest 已发布的 6 个 normalized case-ID
  digest，不打开 case body 文件。
- `gold_partition` 为 unavailable：完整 ID 只存在于 protected Gold JSONL 正文中。
- `partial_gold_export` 为 unavailable：没有独立 body-free 逐 ID inventory，不能拿
  aggregate record count 顶替。
- `legacy_heldout_cases` 为 unavailable：当前 protected JSONL 缺少能认证其精确 bytes
  的 body-free witness。
- `synthetic_scaffold` 为 unavailable：aggregate task digest 无法反推出可重算的逐 ID
  leaf 集合，除非打开 scaffold records。

因此 coverage 是 2/6。Artifact 刻意不输出 intersection count、intersection digest、
proof-input digest，固定 `zero_intersection_claimed=false`，也不会生成 frozen v1
attestation。

## ID 与哈希规则

物理文件身份是精确 bytes 的 SHA-256。JSON 使用 UTF-8、key 排序、compact separators、
不做 Unicode 规范化；发布文档末尾恰有一个 LF。Sidecar 格式是小写 digest、两个空格、
basename、一个 LF。

可用 source ID leaf 定义为：

```text
SHA256(UTF8(namespace) || NUL || UTF8(native_identifier))
```

Heldout 的 `native_identifier` 明确是已经发布的 normalized case-ID digest（小写
hex），不冒充 raw case ID。Inventory digest 是排序去重后的 leaf hex 以 LF 连接、
末尾无 LF 后计算 SHA-256。每份 source manifest 还包含 canonical JSON 哈希绑定的
domain policy 与 namespace inventory。

## Request-local trigger materialization

Companion contract 为已验证的 two-request aLoRA 边界预留 request-local receipt。未来
ready receipt 必须绑定完整 chat-templated request-2 的精确 serialization SHA、
tokenizer/chat-template 身份、ordered input token IDs 的摘要、零基且右开区间的 trigger
span，以及 boundary overhang。完整 R2 必须只序列化并 tokenize 一次。

孤立 trigger 编码明确为非权威；禁止输出 raw token IDs 和 global token index。激活语义
只能是 `next_request_input_activation_only`：request 1 生成的 trigger 不能在同一次请求
中热切换，Planner request-1 的私有 KV 也绝不能宣称可被 Expert 复用。当前 fixture
保持 pending，不包含任何 tokenizer 派生值。

## Fail-closed 条件

以下任一情况都会拒绝：路径穿越、symlink/reparse point、bytes/identity 漂移、
schema/config/implementation hash 漂移、非 canonical JSONL、重复或非法 identifier leaf、
六源顺序变化、sidecar 缺失/不匹配、toy 记录篡改、开放或带训练授权的 grammar、
unavailable inventory 携带 count/digest、六类未齐却声称 coverage 完整，或 coverage
不完整时声称零交集。

测试使用正式 Draft 2020-12 schema 验证每个 JSON 实例，并模拟同时篡改 toy 与主
manifest，确认独立 grammar rebuild 仍会拒绝。

## 复现

在仓库根目录执行（项目放入 `PYTHONPATH`）：

```powershell
$env:PYTHONPATH='src'
python scripts/data/build_qwen_toy_prerequisite.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_v1.json `
  --output <新的空输出目录>
python scripts/data/audit_qwen_toy_prerequisite.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_v1.json `
  --artifact <新的空输出目录>
python -m pytest tests/test_qwen_toy_prerequisite.py -q
```

构建固定为 8 条 toy 记录，provider/model/GPU/network 全为 0，protected sample-body
reads 为 0；不会覆盖已有输出目录。

## 版本与迁移纪律

Frozen `qwen_toy_source_disjoint_attestation.schema.json` 保持字节级 SHA-256
`7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea` 不变。
本 producer 是 companion prerequisite version，不是对 v1 的迁移。Consumer 可以认证
并展示具体缺口，但在六类逐 ID inventory 全部独立认证且完整 namespaced intersection
重算前，不得 mint frozen v1 attestation。

真实 Gold/partial/legacy/scaffold ID inventory producer、完成的 request-local tokenizer
receipt、formal-v3 release lock、真实训练以及质量/性能评测仍未完成。任何 diagnostic
信号都不能由此晋级为正式训练授权。
