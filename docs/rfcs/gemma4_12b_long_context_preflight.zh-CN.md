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

上述默认命令只读取预检 config，不认证或解析 inventory JSONL。若要显式认证三份
已冻结 producer contract 文件与无正文的纯标量 fixture，请增加：

```powershell
python scripts/research/preflight_gemma4_12b_long_context_preflight.py `
  --config configs/research/gemma4_12b_long_context_preflight.yaml `
  --authenticate-producer-fixture `
  --pretty
```

报告覆盖 8K、16K、32K、64K、128K、256K、512K 和 1Mi。256K 以内仅
表示 `native_metadata_only`，不代表运行时、质量、检索能力或训练已经验证。
512K 与 1Mi 超过原生 metadata，且尚未绑定外推缩放方案和实测，因此固定为
`research_only_blocked`。KV 数值只是量化张量 payload 的规划值，不是完整的
运行时实测；它不含 allocator 对齐、图/工作区、权重、输出和并发开销。

## 已冻结的 synthetic token inventory 交接

蒸馏侧现已冻结纯标量 fixture 契约：

- producer：`anchor.long-context-token-inventory-producer.v1`；
- record schema：`anchor.long-context-token-inventory.v1`，SHA-256 为
  `aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f`；
- manifest schema：`anchor.long-context-token-inventory-manifest.v1`，SHA-256 为
  `8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0`；
- producer config SHA-256 为
  `79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe`；
- 源 projector manifest SHA-256 为
  `595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac`；
- fixture manifest SHA-256 为
  `73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e`。

fixture 包含 3 个认证分区、15 条纯标量记录、2 个 task bundle、3 组完整五角色
视图、89 个 segment 引用和 25 个唯一 segment；provider request 为 0。版本、声明
hash、物理 `manifest.json.sha256`、分区 hash 或 manifest count 任一漂移时，消费端
都会 fail closed。仍然先按 `task_bundle_sha256` 切分源任务，再做 augmentation；
prompt、completion、TaskBoard、block、message 以及 held-out 正文均不进入该交接。
显式认证只解析闭合的标量/hash 记录，用于重算 token 公式、clean/noisy private
delta 规则、role-stage 绑定、bundle split 隔离、完整五角色组、bucket/gate 以及
tokenizer binding 的 canonical hash。报告会如实标记已解析标量 JSONL，同时保持
`content_bodies_materialized=false`。

该 fixture 明确使用 synthetic tokenizer
`anchor.synthetic-fixture-utf8-byte`。它只证明 producer 与 consumer 可以认证并交换
纯标量 inventory 契约；**不能**证明其与本地 Gemma tokenizer 一致，也不能证明
256K 能力、1Mi 支持或质量、长上下文检索、运行时内存、KV 正确性或训练质量。
在绑定目标模型 tokenizer 并完成真实测量前，这些声明仍全部为 false。

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
