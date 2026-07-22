# 合成脚手架受控代理后续设计 v1

## 状态与非目标

本文冻结训练/Consumer 提交
`6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465` 的 Producer 侧纯元数据后续设计。
它是不授权、只追加的 overlay：不修改 canonical Gold、held-out、已冻结的自然语言
脚手架 v1/v2 或任何 formal-v3 artifact。

唯一正向结论是：在这次精确的单种子、seq512、80 step、等参数预算合同中，
`q_plus_o` 的 eval-proxy 终点 loss 最低。这不是 formal 胜者选择，也不证明统计显著性、
样本效率、Q/O 因果机制、长上下文泛化、物理 KV 复用、zero-copy、正式质量或训练授权。
20 条 eval 记录只来自两个独立 source bundle；CUDA deterministic algorithms 未启用。
等参数量不等于等计算量，三个臂分别有 56、112、224 个可训练 tensor。

## 认证的报告证据

上游 `comparison.json` 及 mandatory sidecar 是 Consumer 本地 ignored artifact，不在
该提交的 Git tree 中。因此 Producer 在
`fixtures/research/synthetic_scaffold_controlled_proxy_followup_v1/source/`
保存其纯元数据逐字节副本；副本不含 prompt、answer、token IDs、Gold/held-out 或
受保护 scaffold 正文。

Producer 认证的是 out-of-tree 报告的精确 bytes 与 closed semantic projection；审计
运行时不会重新打开报告引用的 preflight、receipt、adapter、model 或 Consumer Git
blob。冻结时只读交叉核验了 tracked schema/config/runner/auditor 哈希与 Consumer
commit，但这不等于重新执行训练或认证传递依赖。仅通过 JSON Schema 不够，必须执行
已哈希绑定的 Producer auditor。

| 身份 | SHA-256 |
|---|---|
| Consumer commit | `6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465` |
| Consumer tree | `2e40aaa1685b3da914275f4f61f8161138ffe6b5` |
| 源 comparison | `920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45` |
| 源 sidecar 物理文件 | `bbdeb5b3ac8f7890a93c736b19bcd920875f642174f99914b199d1d62ce06830` |
| 源 comparison schema | `3b1c81cc888f0b56e013d3afc1317ed78ccd62f7c53606aa2257ed6d389161e2` |
| 源训练 config | `490b0e18fce004a44b97b4c4ab2a3d0f9d0809e1d1f20a2fdc8f93938490e9c2` |
| 源 runner | `ea2b822f0aefc3f7deea9654849a2a5ea87b2e1c35356ad7615941bcb1cefd9b` |
| 源 auditor | `c7328cdc3083d52ee8d40de94ee0f18169d920cc448b75f2860868911c522b71` |
| Producer schema | `fe3878cac9d3be773a676c23025e79cc7f64063da03a53c54eb7f4b59594e0b6` |
| Producer contract | `06ed121b23570546eb00088c891b273806dab1a1f764c9e40fba527cbf6447df` |
| Producer auditor implementation | `588c396febab5de75b772a4f46d58f21c8456247097fccc345cb1c738b0093a3` |

上游 schema 只封闭顶层，内部 object 仍开放；其 JSON 认证也分别读取文件进行哈希和
解析。Producer 使用 closed semantic projection、同一个 immutable bytes snapshot 的
hash/parse/reparse、精确 mandatory sidecar、路径/reparse-point 拒绝和末端 fresh-byte
身份复验。

## 观测合同与解释

三个臂共同绑定同一个 Qwen2.5-1.5B-Instruct base fingerprint、80 条训练记录恰好各
一次、20 条 eval-proxy、seq512、80 step、学习率 `5e-5`、seed1337、BF16+TF32，
以及每臂 1,376,256 个可训练参数。历史 split key 是 `source_bundle_id`，不能改写为
`task_bundle_sha256`。

`2e05af50...e15` 不是 `model.safetensors` 文件 SHA，而是按模型
`named_parameters()` 迭代顺序生成的 SHA-256 fingerprint：每个 leaf 依次绑定 UTF-8
参数名、NUL、CPU-contiguous tensor 原始 bytes 的 ASCII SHA-256 和 LF。train-order 与
eval-view digest 也显式绑定 seeded-record/token-view 算法；LoRA rank 与 alpha/scaling
同时投影。

| Producer 标签 | Consumer profile | rank / alpha | 训练后 eval loss | 吞吐（tok/s） |
|---|---|---|---:|---:|
| `q_only` | `q_only` | Q16 / 32 | 2.0898047030 | 902.9 |
| `q_plus_o` | `q_plus_o` | Q8,O8 / 16,16 | 0.8586097196 | 814.5 |
| `wide_lora` | `wide_budget_matched` | Q5,O4,K6,V6 / 10,8,12,12 | 1.1850866765 | 621.9 |

Q+O 的中性规划标签只是“attention-query intervention +
attention-output-to-residual intervention”。证据只说明终点 proxy loss 更低，不说明
原因。吞吐只是单次描述信号：Q+O 比 Q-only 慢约 9.8%，Wide 慢约 31.1%，尚未做
随机化重复计时。

## 预注册复验阶段

合同冻结五个 master seed，并分别派生及哈希绑定 adapter-init、record-order、CUDA
三个 seed domain。各臂在同一 seed 下使用完全相同的样本和参数预算。性能测量与训练
分离：每臂一次 warm-up，并在每个 seed 内覆盖全部六种臂顺序排列；计时前后执行 CUDA
synchronize，并绑定 runtime、温度与时钟状态 receipt。

主指标预注册为相对共同 base 的 bundle-macro eval-loss delta：先在每个 seed 内对
bundle 等权聚合，再对 seed 等权聚合。区间采用 paired two-level seed/bundle
bootstrap，10,000 次、seed 20260723、95% interval。必须报告全部 seed、source bundle、
语言、角色、scaffold variant 和 5/10/20/40/80 step checkpoint。机制解释前必须增加
等预算 O-only 与 K+V 控制。

readiness 只看完整性与预注册覆盖，不依赖排名。合法结果可以复现 Q+O、打平/不确定，
也可以推翻排名；Producer 仍不选择胜者，通过后仍只是 controlled proxy。

## 计划中的 confirmation fixture 与 split

计划 fixture 有 60 个 source bundle，40 train、20 eval-proxy；英文/简中平衡；每
bundle 五角色、两个 paired variant，split 后共 600 条记录。五个预注册
information-flow stratum 各分配 12 个 bundle：8 train、4 eval-proxy；每个 stratum
内英文/简中分别为 4/4 train、2/2 eval-proxy，因此派生计数为 400 train、200
eval-proxy。生成前冻结 domain-separated blueprint-hash 排序与 balanced round-robin
分配算法。它必须先按
`task_bundle_sha256` split，再生成角色、variant、noise、length 或 causal augmentation。
task-bundle preimage 绑定固定 domain、confirmation source namespace、source-bundle
身份、语言和 `source_task_blueprint_sha256`，排除 role/variant/noise/length；五角色和
两 variant 必须共享同一 bundle，并与内部 `task_board.task_id` 交叉绑定。
`eval_proxy` 不是 held-out。

本合同只描述 strata；未来 fixture 会包含合成训练记录。它尚未生成，也尚未证明与
10-bundle discovery fixture 独立。合同绑定了 discovery source-bundle inventory hash，
但明确声明 namespaced bundle-ID 零交集不足以证明语义独立。必须为 discovery 与
confirmation 生成同一 common-domain、namespace-neutral 的
`source_task_blueprint_sha256` inventory。当前 discovery blueprint inventory 未发布，
confirmation inventory 待生成，zero-overlap unavailable，且
`independent_confirmation_claimed=false`。

## 长度与 KV gate

真实 tokenizer 绑定的长度计划从 8K/16K/32K 开始；每个 bucket 都表示
`total_tokens=input_tokens+reserved_output_tokens`，且不得超过 32,768。当前记录的本地
Qwen candidate revision/config/tokenizer/chat-template/serialization/special-token 身份
只属于 diagnostic candidate，不是 formal binding。此 preflight 禁止 provider 请求、
模型加载和 GPU 请求。

candidate config 报告 `max_position_embeddings=32768`，所以 64K、128K、256K 对当前
身份都 blocked，必须绑定新的模型或 RoPE；即使换身份，256K 也只做 capability gate。
1 Mi 保持 research-only/blocked。已有 Gemma 或 synthetic byte-tokenizer inventory
不能替代 Qwen 验证。

runtime gate 要求共享 frozen prefix 上所有 adapter 关闭，只允许 route boundary 后
激活 expert adapter。精确复用仅限 token order、position、RoPE、model 和 KV-producing
weights 完全相同的 ordered prefix lineage；expert tail KV 必须私有。任何普通
in-stack LoRA 臂都不能仅凭名称证明 exact full-stack reuse 或物理 Q-reader。

## Fail-closed 与真实未完成项

审计拒绝 schema/bytes 漂移、重复 JSON key、非有限数、sidecar 错误、绝对/穿越路径、
symlink/junction/reparse component、首末身份漂移、臂/alias/rank/alpha 漂移、参数预算
不等、共同 base/order/token-view 漂移、排名漂移、历史 split 改写、post-hoc 指标、
seed/permutation 漂移、未授权长度晋级、任何受保护正文读取及 blocked claim 晋级。

合同中的 provider/network/model/GPU 与 protected-body 零计数仅描述本次 Producer
元数据 overlay 的构建和审计；上游 controlled proxy 确实在本地 GPU 上加载并训练了
模型。

formal-v3 snapshot、projector、generic execution、source-disjoint、release lock 仍为
0/5；body-free protected inventory 仍为 2/6，六源 zero-intersection 未建立。多种子
复验、confirmation blueprint disjointness、长上下文证据、物理 KV 正确性/性能、真实
teacher 蒸馏数据与 formal 质量评测均未完成。合同固定
`training_authorized=false`、`formal_training_authorized=false`。

## 纯元数据复现

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_synthetic_scaffold_controlled_proxy_followup_v1.py `
  --repo-root .
python -m pytest -q tests/test_synthetic_scaffold_controlled_proxy_followup_v1.py
python -m ruff check `
  src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_followup.py `
  scripts/data/audit_synthetic_scaffold_controlled_proxy_followup_v1.py `
  tests/test_synthetic_scaffold_controlled_proxy_followup_v1.py
```

以上命令不发 provider 请求，也不加载模型/GPU。本轮不创建 tag/release；该 artifact
不能自行授权训练。
