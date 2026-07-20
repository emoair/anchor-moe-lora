# Gemma 4 12B 静态长上下文预检

这是面向本地 Gemma 4 12B 配置的小规模、免模型容量检查。它只读取一个 YAML
并做整数 KV cache 计算；不会加载权重、调用 GPU/网络、读取 JSONL/数据正文或
held-out 内容。

固定的本地事实如下：

- 原生上下文 metadata 为 262,144 tokens；
- 模型身份为 `google/gemma-4-12B`，revision 固定为
  `56820d7d8cbe8e47975a53325439ed272e91cff2`，源 config SHA-256 为
  `14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098`；
- 共 48 个文本层，其中 40 个滑窗注意力层、8 个全局注意力层；
- 滑窗为 1,024；llama.cpp 分配为 1,024 + ubatch 256 = 1,280 cells；
- 全局层配置为 `rope_type=proportional`、theta 1,000,000，并单独固定
  `partial_rotary_factor=0.25`；
- 滑窗层默认 RoPE theta 10,000；
- 保守地分开存储 K/V：K 为 `q8_0`（每 32 元素 34 bytes），V 为
  `q4_0`（每 32 元素 18 bytes），并强制启用 Flash Attention；
- 1,280-cell 规则绑定 llama.cpp commit
  `33c718db1fbfe834f30eef28cf206f98736fe612`。

运行方式：

```powershell
python scripts/research/preflight_gemma4_12b_long_context.py `
  --config configs/research/gemma4_12b_long_context_preflight.yaml `
  --pretty
```

报告覆盖 8K、16K、32K、64K、128K、256K、512K 和 1Mi。256K 以内仅
表示 `native_metadata_only`，不代表运行时、质量、检索能力或训练已经验证。
512K 与 1Mi 超过原生 metadata，且尚未绑定外推缩放方案和实测，因此固定为
`research_only_blocked`。KV 数值只是量化张量 payload 的规划值，不是完整的
运行时实测；它不含 allocator 对齐、图/工作区、权重、输出和并发开销。

报告同时向蒸馏侧请求 `anchor.long-context-token-inventory.v1` 的纯标量交接
字段；该接口明确处于“等待 producer 冻结”状态，不冒充已冻结 schema。先按源
任务切分，再提供 token 数、lineage hash、预留输出和 bucket；禁止携带 prompt、
completion、TaskBoard、block 或 message 正文。本预检不读取该清单，只公布字段
与不变量。计数必须覆盖已绑定的 chat template、framing 和特殊 token；
`total_tokens` 表示真实 prompt+target 序列，`required_context_tokens` 则按 target
与预留输出中的较大者计算所需容量。

分阶段且不启动模型的路线是：先验证不缩放的原生 64K、128K、256K；512K 仅
研究 2x YaRN/linear；1Mi 则继续被门禁阻止，后续评估 4x YaRN 可行性或
LongRoPE 式训练路线。方法出处见 [YaRN](https://arxiv.org/abs/2309.00071) 与
[LongRoPE](https://arxiv.org/abs/2402.13753) 原论文。

在宣称 1Mi 之前，后续门禁必须绑定明确的 RoPE/外推方法、实测运行内存、
位置与检索质量、任务质量以及长上下文训练证据。仅有旋转位置编码机制并不能
证明百万上下文行为稳定。

一手依据包括 [Gemma 4 官方模型卡](https://ai.google.dev/gemma/docs/core/model_card_4)、
固定 revision 的 [12B 配置](https://huggingface.co/google/gemma-4-12B/blob/56820d7d8cbe8e47975a53325439ed272e91cff2/config.json)、
[Gemma 4 技术报告](https://arxiv.org/html/2607.02770) 与
[llama.cpp 服务参数](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。
