# Qwen aLoRA 两请求 Prefix-KV 小规模探针

日期：2026-07-22。这个探针是本地机械可行性证据，不是正式训练、质量跑分、
tokenizer binding 或物理多流实现。它使用本地
`Qwen/Qwen2.5-1.5B-Instruct`、PEFT 0.19.1 和一个临时的 `q_proj` rank-4
aLoRA。为制造可观测的机械 effect，本地临时 B 矩阵使用确定性的非零随机值；
本轮没有训练、没有读取项目数据集、没有 provider 请求，也没有保存 adapter。

## 要验证的因果边界

Planner 的 request 1 私有 KV **不跨请求复用**。Planner 提交的短自然语言
scaffold 在 request 2 中重新序列化、重新编码。request 2 被分成：

1. 冻结基座编码、与 base 兼容的 invocation 前 prefix；
2. 唯一 invocation token 序列；
3. 从 invocation 起由专家 aLoRA 生效的 continuation。

探针比较两条路径：

- `full`：一次性对完整 request 2 做 aLoRA 前向；
- `reuse`：先关闭 adapter，用冻结基座计算 prefix KV；再把同一 KV 交给
  aLoRA，从 invocation 开始计算 suffix。

同时运行纯基座 `full/reuse` 对照，用来估计并校准 cache 分段造成的数值底噪；
这个差分不是非线性计算中的严格因果分离。

## 输入身份

- 完整 request 2：44 tokens；
- invocation 前 base-compatible prefix：25 tokens；
- invocation 覆盖 span：8 tokens；对应文本在完整序列化中恰好出现 1 次；
- active continuation：19 tokens，其中 invocation 8 tokens、其后 tail
  11 tokens；因此 `25 + 8 + 11 = 44`；
- invocation token span 只在完整 chat-template 序列化后、对整条 request 2
  tokenize 一次，再由 offset mapping 定位并核验字符 span 得到；token IDs
  由完整 tokenizer 编码生成。不能先孤立编码
  invocation 再搜索。此例孤立编码是 7 tokens，在完整输入中没有精确子序列；
  覆盖 `>\n` 边界的正式 span 是 8 tokens。这个 boundary overhang 必须写入
  request-local materialization receipt；
- invocation span 摘要为
  `fca583f23dcd62a9d0531744a7c70366558768a6483398150d1d5e94c13fbf8d`，
  完整 request-2 ordered token 摘要为
  `c78e409a9ec5b772c0b56f9efbd0f179bfbf35a21c6e8cbb3f6c998d9d1f57f2`。
  两者都只是本地 probe identity，不是全局 tokenizer binding。

## 结果

| 模式 | aLoRA full/reuse 最大差 | 纯基座 full/reuse 最大差 | argmax 一致率 | 观察 |
|---|---:|---:|---:|---|
| BF16 eager | 0.890625 | 0.90625 | 100% / 100% | full 与 cached attention 数值路径差异很大，不能用严格 max-logit 等价作门 |
| FP32 eager | 0.000282675 | 0.000255227 | 100% / 100% | aLoRA 差值与纯基座数值底噪同阶 |

FP32 下，`(aLoRA full-reuse) - (base full-reuse)` 的最大绝对差为
`0.000139236`，adapter 的最大可观察效果为 `0.0649891`，峰值 allocated
显存为 `6054.7 MiB`。另一轮 BF16 负对照中，缺少 invocation 时 adapter
效果严格为 `0.0`，验证了本次 PEFT 0.19.1 masking 对照；它不证明正式
tokenizer binding、服务路由或热插拔已完成。表中两个 argmax 值依次是
aLoRA 与 base，只来自这一条 44-token synthetic case，不能外推为质量结论。

## 当前结论

这组结果支持“request 2 的冻结基座 prefix KV 可以供 aLoRA suffix 使用”的
机械可行性；它不支持以下更强结论：

- Planner 私有 KV 可跨 request 复用；
- BF16/Q4 下 logits 逐元素完全相同；
- 普通 in-stack LoRA、任意 Q-LoRA 或多个专家生成过程可以共享完整 KV；
- 当前 single-slot runtime 已支持并发、多流或真实热插拔；
- tokenizer binding、toy source-disjoint 或 formal-v3 已完成。

正式 diagnostic gate 应使用 FP32 eager 作为高精度 reference path，并同时要求：

- 固定 eager、FP32、关闭 TF32；invocation 唯一且只从精确 request-2
  序列化结果生成；
- 缺 invocation 时 active 与 adapter-off 等价；有 invocation 时 adapter
  effect 至少为 `1e-3`；
- 定义 `E_base=base_split-base_full`、
  `E_alora=alora_split-alora_full`，要求
  `max(abs(E_alora-E_base)) <= 1e-3`；
- argmax/有限值/shape/cache lineage 均一致；
- invocation 之前 active 与 adapter-off 的逐层 prefix KV 必须 bit-equal；
- BF16 只作为性能与语义一致性观测：要求有限值、full/split argmax 与
  4--8 token greedy 序列一致，不使用 raw max-logit 严格等价门；
- 续写必须由 runtime 持续携带 aLoRA activation offset。PEFT 的
  `generate` 会维持该状态；手写逐 token decode 若丢失 offset 会静默退回
  base，且不能接受用户直接传入 offset。

## 运行时版本隔离

当前 `NaturalLanguageScaffoldRuntime v1` 是 single-slot 串行状态机，不能在
同一实例承载多个 R1/R2 pair。后续 diagnostic mux 不修改 v1：每个
`(run_id, stream_id)` 独占一个 v1 实例，采用 per-stream 原子状态迁移和
tombstone；只有相互独立的 pair 才能并发。真实 adapter registry、tokenizer
binding、activation receipt、physical KV lease 或 formal lock 任一缺失时，
仍必须保持 `diagnostic_only=true`、`formal=false`、
`training_authorized=false`。

上面的 44-token 数值来自最初手工 probe；它没有生成 receipt。随后新增的
独立 diagnostic 工具已绑定模型快照、tokenizer/chat-template、
Torch/PEFT/Transformers/CUDA、attention implementation、seed 与 adapter
recipe。它仍未接入多流 runtime，也不具备 formal 权限。所谓 base-compatible
prefix 只在有序 token、position/RoPE、mask、cache dtype/layout/device 与这些
身份全部一致时成立。

## 可重复执行结果

```powershell
# 仅验证契约，不导入 ML runtime
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml

# 验证本地四个模型资产，不加载权重
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml --preflight

# 唯一加载模型/GPU 的路径
anchor-qwen-alora-prefix-kv --config configs/research/qwen_alora_prefix_kv_diagnostic_v1.yaml --execute
```

2026-07-22 的正式 diagnostic-only 执行通过：request 2 为 48 tokens，
invocation 前 prefix 为 30 tokens，invocation 为 7 tokens，continuation 为
18 tokens。`paired differential=4.86374e-5`、
`adapter effect=0.205176`、missing-trigger effect=`0`；28 层 prefix KV
逐层 bit-equal，4-token greedy continuation 一致，峰值 allocated 显存
`5954.57 MiB`。结果写入
`artifacts/diagnostics/qwen_alora_prefix_kv_v1/diagnostic_receipt.json`。

这个新请求恰好没有早期 44-token case 的 `>\n` overhang，因此 invocation 是
7 tokens；两次结果不同正好证明 token span 必须是 request-local receipt，不能
全局写死。工具没有训练、没有保存 adapter、没有 provider/network/dataset
输入，并继续声明 `formal=false`、`training_authorized=false`。
