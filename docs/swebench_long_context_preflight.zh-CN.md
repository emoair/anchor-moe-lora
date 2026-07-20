# 绑定 tokenizer 的长上下文 token inventory

[English](swebench_long_context_preflight.md)

## 状态与范围

这个 producer 在已认证 `anchor.swebench-taskboard-projector-manifest.v2`
下游生成 tokenizer-bound token inventory，绝不输出 estimated-token 近似值。
local 模式发布对其绑定本地 tokenizer 的精确计数；synthetic 模式只发布确定性
契约 fixture，其计数仅对声明的 synthetic tokenizer 精确，绝不是正式 Gemma 或
target-model exact inventory。若所选 tokenizer 身份不完整，运行必须在发布前
失败，且不得输出 null inventory。

签入契约位于
[`swebench_long_context_preflight_v1.yaml`](../configs/research/swebench_long_context_preflight_v1.yaml)。
虽然文件名保留了历史 preflight 名称，其 schema identity 是
`anchor.long-context-token-inventory-config.v1`。输出记录按
[`swebench_long_context_preflight_sidecar.schema.json`](../configs/research/swebench_long_context_preflight_sidecar.schema.json)
验证为 `anchor.long-context-token-inventory.v1`；inventory manifest 按
[`swebench_long_context_preflight_manifest.schema.json`](../configs/research/swebench_long_context_preflight_manifest.schema.json)
验证为 `anchor.long-context-token-inventory-manifest.v1`。两个 schema 都是封闭
schema，且不包含远程引用。

## 已认证 projector 输入

producer 只接受完整 projector v2 目录，其中必须存在精确 `manifest.json`、
强制 `manifest.json.sha256`、已认证 projector manifest/config/sidecar/
segment-plan schema 哈希，以及按顺序排列的三个固定分区：

- `train/clean.jsonl`
- `train/noisy.jsonl`
- `calibration/clean.jsonl`

每条源 JSONL 都按原始行字节认证。记录保存 `source_line_sha256`；禁止解析后按
canonical JSON 重序列化并伪造另一种行身份。manifest 绑定三个源分区和三个
输出分区的 SHA-256、字节数与记录数。发布还要求新的强制
`manifest.json.sha256`，并在原子发布前再次检查完整输入 inventory。

split 继续按 `task_bundle_sha256` 分组；`task_id_sha256` 是
`training_record.task_board.task_id` 精确 UTF-8 字节的 SHA-256。五个角色必须
保持在同一 split，且源 split 必须早于 augmentation。

## 强制 tokenizer 绑定

必须显式选择且只能选择一种 backend：

- `local_offline_tokenizer`：使用本地已认证 tokenizer assets；或
- `explicit_synthetic_tokenizer`：明确标注为 synthetic，只用于确定性 fixture。

manifest 必须绑定以下全部非 null 值：

- `tokenizer_id` 与 `tokenizer_revision`；
- `tokenizer_assets_sha256` 与 `tokenizer_runtime_sha256`；
- `chat_template_sha256`；
- `serialization_policy_sha256`；
- `special_token_policy_sha256`。

`tokenizer_label_source=caller_supplied_and_hash_bound` 明确可读 tokenizer 标签的
来源：标签由 caller 提供，manifest 再将其与已认证 tokenizer assets/runtime 一并
哈希绑定。标签本身不构成模型身份断言。

禁止网络访问。只有 tokenizer 名称、没有 revision/assets/runtime 哈希并不充分。
synthetic backend 绝不能冒充 Gemma tokenizer，也不能作为正式模型证据。

backend 必须确定完整发布模式：

| Backend | `inventory_mode` | `status` | `claim_scope` | Target-model match |
| --- | --- | --- | --- | --- |
| `explicit_synthetic_tokenizer` | `synthetic_fixture` | `synthetic_fixture_inventory_ready` | `synthetic_fixture_contract_only` | `not_applicable` |
| `local_offline_tokenizer` | `local_exact_tokenizer` | `exact_token_inventory_ready` | `exact_bound_tokenizer_inventory_only` | `consumer_verification_required` |

manifest 对 synthetic backend 固定 `synthetic_fixture_only=true`，对 local backend
固定为 `false`。即使 local exact 模式也只证明绑定 tokenizer 的计数；consumer
必须将其身份与实际 target model 交叉验证后，才能把这些 bucket 当作目标模型
bucket。

## 精确记录契约

每条 inventory 只包含标识、哈希、整数计数、bucket/gate 状态与 false 授权
声明。它绑定 task/role、源原始行、精确 segment plan、ordered segment-ID
摘要、terminal prefix lineage，以及全局 tokenizer binding 的 canonical hash。
记录不输出 segment 数组。

producer 只序列化因果上已选的 segment plan。current、future 与 forbidden
block 必须在序列化前排除。inventory 记录不得包含携带源文本、预览、渲染输入、
target、token ID 数组或 held-out 材料的字段或值。

语义不变量为：

```text
shared_prefix_input_tokens + private_delta_input_tokens = input_tokens
reserved_output_tokens = 4096
total_tokens = input_tokens + reserved_output_tokens
```

保留字段名 `shared_prefix_input_tokens` 是 consumer 契约标签；其值覆盖可见有序链中
全部 non-private segment，包括严格 task-shared prefix 与已因果提交的 downstream
task-shared immutable segment。不得把它解读为全部计数 token 都属于五角色严格共享
前缀交集。

已认证 serialization 与 special-token policy 负责确定 wrapper/special token 在两种
scope 计数之间的归属。token 数必须来自声明的精确 backend，不得使用 UTF-8 字节
启发式估计。

tokenizer 身份不能证明可复用的模型 KV 身份。因此每条记录固定
`cache_identity_status=identity_unbound` 与 `reuse_savings_tokens=0`。

## Bucket 与 gate

bucket 作用于绑定 tokenizer 的 `total_tokens`，使用闭区间上界；manifest 将依据
固定为 `bucket_basis=bound_tokenizer_total_tokens`：

| Bucket | Total tokens | Gate |
| --- | ---: | --- |
| `le_8k` | 1–8,192 | `measurement_candidate` |
| `le_16k` | 8,193–16,384 | `measurement_candidate` |
| `le_32k` | 16,385–32,768 | `measurement_candidate` |
| `le_64k` | 32,769–65,536 | `measurement_candidate` |
| `le_128k` | 65,537–131,072 | `measurement_candidate` |
| `le_256k` | 131,073–262,144 | `capability_only` |
| `le_1m` | 262,145–1,048,576 | `research_only_blocked` |
| `gt_1m` | 大于 1,048,576 | `reject` |

`measurement_candidate` 不是 evaluation pass。`capability_only` 仍需单独验证
model/runtime 能力。`research_only_blocked` 不得进入 allocation 或 execution，
`reject` 是终止状态。

synthetic bucket 只是在声明的 synthetic tokenizer 下成立，不是 Gemma 或
target-model bucket。local tokenizer bucket 在 consumer 证明绑定 tokenizer 就是
目标模型 tokenizer 之前，同样保持 consumer-unverified。

## Caller-supplied 模型元数据

manifest 只复述、不验证 caller-supplied Gemma4 12B 描述：48 个 text layer、
40 个 sliding-attention layer、8 个 global-attention layer、sliding window 1,024，
以及 `max_position_embeddings=262144`。它必须声明
`architecture_verified_by_preflight=false`。精确 tokenization 不能证明模型架构、
内存容量、RoPE 行为、KV 兼容性或长上下文质量。

## Manifest 证据与非授权边界

manifest 绑定 projector 身份、三个源分区、本 producer/config/record/manifest
schema、完整 tokenizer 身份、模型元数据声明、三个 inventory 文件，以及按
split、variant、stage、expert、bucket 与 gate 的计数。它还记录总行数、唯一
task bundle、task-ID 摘要、segment reference 与 unique segment ID 数。

每条记录与 manifest 都固定 `provider_requests=0`、
`evaluation_status=not_evaluated`、`quality_validated=false`、
`allocation_validated=false` 与 `execution_authorized=false`。manifest 还固定
`capability_validated=false`、`canonical_gold_written=false`、
`approximate_inventory_emitted=false` 和 `null_inventory_emitted=false`。由 backend
绑定的 mode、status、claim scope、target-model match 状态与
`synthetic_fixture_only` 标志都是强制机器可判定证据。两种模式都不授权
evaluation、allocation、execution 或 target-model capability claim。
