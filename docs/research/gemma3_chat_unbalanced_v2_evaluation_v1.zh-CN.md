# Gemma 3 chat unbalanced-v2 评测 v1

状态：model-free 契约已完成；本次变更没有执行物理 KV 实验、模型生成、GPU
运行或真实评测。

## 依赖边界

两个契约都只接受
`anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1`
且要求 `status=passed`。旧的未分片 `.v1` receipt 会被拒绝。检入配置不包含
Producer candidate 哈希、旧 44 文件身份、单个 `train/chat.jsonl` 假设或未发布
的分片身份。

评测器还要求：

- body-free 的多臂训练 run receipt 与 adapter inventory 身份；
- 已通过的物理 shared-prefix KV receipt；
- 所有 receipt 的 artifact-tree 与逻辑 shard-inventory 身份一致。

每项依赖 receipt 及其 Draft 2020-12 schema 都必须作为分别绑定配置 SHA-256
的物理普通文件快照进入门禁；已解析的 mapping 不能充当认证输入。加载器在读取
前核对路径与文件描述符身份，通过同一认证句柄只读一次，在解析 JSON 前核对
字节数、UTF-8/LF 与 SHA-256，并在最后再次核对路径身份，从而拒绝替换竞态。

Consumer、training、trusted-lineage-context、KV dependency projection
同时采用 closed exact-key 契约；顶层或嵌套字段新增、缺失、类型变化、
schema-version 漂移、非有限常量、重复键与 schema 弱化都会被拒绝。

检入配置的依赖身份均保持 `pending`，因此在经过 release review 的 consumer
handoff 绑定前，status/preflight 始终 fail-closed。

## Shared-prefix KV 口径

首个物理 probe 对照：

1. 每个分支分别重新 prefill；
2. adapter-off frozen base 只做一次 prefill，再在 `route_plan_commit` 边界
   显式本地复制。

v1 唯一允许的正向结论是：有序 retained prefix 的值完全一致，并完成
prefill-compute handoff。以下声明始终为 false：

- `shared_storage`；
- `zero_copy`；
- `full_generation_kv_shared`；
- `rdma`。

对象或指针相等不能升级上述声明；专家 tail 必须私有且 append-only。

Gemma 3 布局固定为 26 个 attention layer：22 个窗口为 512 的 sliding layer，
以及零基索引 5、11、17、23 的 4 个 full layer。边界长度为 `n` 时，full layer
保留 `[0,n)`，sliding layer 保留 `[max(0,n-512),n)`。合成用例覆盖 127、511、
512、513、1025。

物理 receipt 必须证明：

- isolated re-prefill 与 one-prefill-plus-copy 两个实验都真实执行；
- 26 层全部 retained span 的精确值摘要；
- 两个 arm 各自提供 first-token 分布与 argmax 摘要 operands；等价诊断必须
  由这些 operands 复算，不能信任 receipt 自报布尔值；
- source cache 未变、cache-miss 被明确拒绝、24 项注册 cross-mutation 全部
  被拒绝、private tail 不别名；
- 延迟、复制成本、device peak bytes 与重复次数。

完整 lineage 绑定模型、tokenizer、serialization、有序 prefix、token order、
position、attention mask、RoPE、adapter lineage、adapter-off 状态、route commit、
plan commit、boundary commit、训练 run、adapter inventory 与 cache presence。
两个实验必须绑定同一认证 source prefix 和完整 lineage。receipt 不持久化 raw
token IDs。该 lineage 必须来自独立物理 trusted-context 快照并绑定其 SHA-256，
不能由两个待验实验或 receipt 自行生成。它还冻结 template、attention
implementation、dtype、quantization、adapter-off policy 与 layer-layout 身份。
即使同时篡改两臂并重算 boundary commit，也必须因为不匹配外部 context 而拒绝。

## 评测矩阵

聚合评测器固定 4,990 个槽位：

| 分组 | 记录数 | 臂 / 路由数 | 槽位 |
| --- | ---: | ---: | ---: |
| 工具调用对照 | 400 | 3 | 1,200 |
| 规划师对照 | 240 | 4 | 960 |
| Router | 80 train proxy + 20 eval proxy | 1 | 100 |
| 身份认知 probe | 50 | 3 | 150 |
| 错误路由 proxy | 860 | 3 个 derangement | 2,580 |

工具调用臂为 base、Q-only、Q+micro-O；规划师臂为 base、Q-only、O-only、
Q+micro-O；身份认知比较 base、正确路由、错误路由；错误路由使用基于已认证
role label 的三个确定性 derangement。

工具分层为 seen-template interpolation（160）、argument-composition proxy
（120）、cross-language-transfer proxy（120）。规划师分层为 task
classification、expert selection、commit boundary，各 80 条。

## 仅聚合 receipt

receipt 只能包含计数、比例、差值、confusion 聚合与性能聚合；样本正文、prompt、
target、reference、raw token IDs、record IDs 与 per-record 数组全部拒绝。每个
比例与 router macro-F1 都从整数聚合复算。随后，八个 comparison 字段都使用
`decimal-string-quantized-1e-12-v1` 从这些计数重新计算：精确十进制除法、
`ROUND_HALF_EVEN`、固定 12 位字符串；外部提供的差值不受信任。

`eval_proxy` 不冒充 heldout，wrong-route 差值不冒充因果证明，也不宣称质量或
泛化已经验证。

## Model-free 命令

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1 --status
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1 --preflight `
  --consumer-receipt <consumer-receipt.json> `
  --consumer-receipt-schema <consumer-receipt.schema.json> `
  --training-receipt <training-run-receipt.json> `
  --training-receipt-schema <training-run-receipt.schema.json> `
  --trusted-lineage-context <trusted-lineage-context.json> `
  --trusted-lineage-context-schema <trusted-lineage-context.schema.json>
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_generation_eval_v1 --status
python -m anchor_mvp.research.gemma3_chat_unbalanced_v2_generation_eval_v1 --preflight `
  --consumer-receipt <consumer-receipt.json> `
  --consumer-receipt-schema <consumer-receipt.schema.json> `
  --training-receipt <training-run-receipt.json> `
  --training-receipt-schema <training-run-receipt.schema.json> `
  --kv-receipt <kv-probe-receipt.json> `
  --kv-receipt-schema <kv-probe-receipt.schema.json>
```

`--execute` 只是预留的 lazy interface；缺少已认证物理 backend 时会
fail-closed，并且不会导入 ML runtime。

## 非声明

本契约不证明训练质量、泛化能力、路由因果性、zero-copy cache sharing、RDMA
或 formal/release readiness。当前 model load、GPU request、provider request、
network request、protected read、Gold read、heldout body read、dataset body
read 与 raw-token-ID read 全部为 0。
