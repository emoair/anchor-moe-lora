# 致谢 / Acknowledgements

Anchor-MoE-LoRA 不是凭空长出来的。感谢开放 Skill、研究论文、模型、训练框架和
工程工具的作者与维护者，使这个受限硬件上的研究原型成为可能。

为避免把“使用了资产”“受到启发”和“依赖基础设施”混为一谈，本文将致谢分为
三类。列名不表示作者或项目方认可、参与或背书 Anchor-MoE-LoRA；本项目也不把
他人的实验结果当作自己的结果。

## 1. 直接随仓库提供并注入的 Skill 资产

下列文件作为 SOP 输入被直接复制到 `third_party/skills/`，并由
`configs/data/skill_sources.yaml` 固定来源 commit、文件 SHA-256、许可证标识和
许可证文件 SHA-256。它们保留各自的原始许可证；本仓库的 AGPL-3.0-or-later
不会替代这些第三方条款。

| 来源与作者/维护者 | 直接使用的资产 | 固定版本 | SPDX 与许可证文件 |
| --- | --- | --- | --- |
| [GitHub awesome-copilot](https://github.com/github/awesome-copilot)，GitHub 及项目贡献者；其中 `premium-frontend-ui` 元数据署名 [Utkarsh Patrikar](https://github.com/utkarsh232005) | `premium-frontend-ui`、`review-and-refactor`、`security-review` 三个 `SKILL.md` | [`30472ecf0fe34cc561df958c08501ecc5ca80ea4`](https://github.com/github/awesome-copilot/tree/30472ecf0fe34cc561df958c08501ecc5ca80ea4) | `MIT`；[本地许可证全文](third_party/skills/github-awesome-copilot/LICENSE) |
| [Anthropic Skills](https://github.com/anthropics/skills)，Anthropic | `frontend-design/SKILL.md` | [`9d2f1ae187231d8199c64b5b762e1bdf2244733d`](https://github.com/anthropics/skills/tree/9d2f1ae187231d8199c64b5b762e1bdf2244733d) | `Apache-2.0`；[该 Skill 的本地许可证全文](third_party/skills/anthropic-skills/frontend-design/LICENSE.txt) |

这些 Skill 只在本项目的确定性策略和运行时约束内使用。特别是：

- 星标只用于发现候选来源，不代表信任或质量证明；
- `security-review` 仅批准用于无凭据、惰性数据的合成 fixture；
- Skill 是过程约束和提示模具，不是未经验证的正确答案；
- 训练候选仍必须经过来源审计、执行验证、失败隔离和 held-out 泄漏门。

更细的逐文件说明见 [THIRD_PARTY_SKILLS.md](THIRD_PARTY_SKILLS.md)。

### 已审查但未纳入

[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) 曾作为候选来源
接受审查，但当前版本没有复制或注入其中任何文件。原因记录在
`configs/data/skill_sources.yaml`：当时审查到的 checkout 没有提供足以完成归属
审计的完整根许可证文本。因此，它不是本版本的直接依赖或训练资产。

## 2. 研究与架构启发

下表只表示相关工作帮助我们界定问题、设计控制组或选择测量方法。Anchor-MoE-LoRA
不声称复现这些工作，不声称与其训练数据或实现等价，也不借用其论文结果作为本项目
的性能证据。

| 工作 | 对本项目的启发与边界 |
| --- | --- |
| [LoRA](https://arxiv.org/abs/2106.09685) / [Microsoft 官方实现](https://github.com/microsoft/LoRA) | 低秩 adapter 的基础方法。Anchor 的贡献主张不包括发明 LoRA。 |
| [QLoRA](https://arxiv.org/abs/2305.14314) / [官方实现](https://github.com/artidoro/qlora) | 冻结量化基座并训练 adapter 的方法论先例。Anchor 只报告自身硬件、配置和产物。 |
| [LoRAMoE（ACL 2024）](https://aclanthology.org/2024.acl-long.106/) / [官方实现](https://github.com/Ablustrund/LoRAMoE) | 说明“冻结基座 + 多 LoRA + 路由”已有广义先例。其模型内 dense router 与 Anchor 的应用层固定 DAG 不同。 |
| [Mixture-of-LoRAs](https://arxiv.org/abs/2403.03432) | 分别训练领域 LoRA 并显式路由的直接相关工作；因此“task-routed LoRA”本身不作为 Anchor 的新颖性主张。 |
| [MoRAgent](https://openreview.net/pdf?id=rdeDanrYEj) | Agent 角色分解、角色 LoRA 和角色切换的近邻先例。Anchor 重点检验确定性授权、真实代码验证、失败闭合与预算匹配。 |
| [X-LoRA](https://arxiv.org/abs/2402.07148)、[MeteoRA](https://arxiv.org/abs/2405.13053)、[MoLE](https://arxiv.org/abs/2404.13628) | 动态、token/层级或模型内多 LoRA 组合的参照。Anchor 当前没有 learned gate，也不在一次 forward 中混合专家。 |
| [MoLA](https://arxiv.org/abs/2402.08562) | 非均匀专家/rank 预算的相关先例。Anchor 的 E 组仅是 calibration-only 的离线阶段预算搜索。 |
| [S-LoRA](https://arxiv.org/abs/2311.03285) 与 [Punica](https://arxiv.org/abs/2310.18547) | 多 adapter serving、缓存、批处理与切换开销的测量参照；本项目未声称实现新的 adapter kernel 或调度算法。 |

这些引用用于约束研究主张。当前核心问题仍是：在相同 Q4 基座、调用、token、数据
暴露和参数预算下，应用层阶段专用 adapter 是否优于一个 mixed adapter。只有真实
held-out、build/test、安全和资源测量完成后，才能回答这个问题。

## 3. 模型、教师与基础设施依赖

这些项目没有作为 Skill 资产复制进 `third_party/skills/`，但构成了本仓库实际或
可选的模型、训练、验证与 serving 路径。各自的许可证、模型条款和服务条款仍独立
适用。

### 模型与教师

- [Google Gemma 4](https://ai.google.dev/gemma/docs/core) 与
  [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B)：冻结基座和
  processor/模型接口来源。模型权重不因本项目代码采用 AGPL 而改变其原始条款。
- [Kimi Code](https://www.kimi.com/code/docs/en/)：合成教师调用所使用的官方 API
  服务。教师输出仍需经过本项目的数据 schema、公开决策轨迹、清洗和验证门；列入
  致谢不表示 Kimi/Moonshot AI 为本项目结果背书。

### 训练与数据栈

- [PyTorch](https://github.com/pytorch/pytorch) 与
  [torchao](https://github.com/pytorch/ao)；
- Hugging Face 的 [Transformers](https://github.com/huggingface/transformers)、
  [PEFT](https://github.com/huggingface/peft)、
  [TRL](https://github.com/huggingface/trl)、
  [Accelerate](https://github.com/huggingface/accelerate)、
  [Datasets](https://github.com/huggingface/datasets)、
  [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) 与
  [Safetensors](https://github.com/huggingface/safetensors)；
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)，用于本项目
  的 NF4/8-bit 优化器训练路径；
- [Apache Arrow / PyArrow](https://github.com/apache/arrow)、
  [SentencePiece](https://github.com/google/sentencepiece) 与
  [Protocol Buffers](https://github.com/protocolbuffers/protobuf)；
- [PyYAML](https://github.com/yaml/pyyaml) 和
  [HTTPX](https://github.com/encode/httpx)。

准确的直接 Python 依赖与版本范围以 [pyproject.toml](pyproject.toml) 和
`configs/training/requirements-qlora.txt` 为准，而不是以上概览列表。

### Agent 工具、推理与验证

- [OpenCode](https://github.com/anomalyco/opencode)：受限执行型蒸馏的外部 coding
  agent/CLI。Anchor 在独立 fixture 中施加自己的确定性工具策略、验证和 attempt
  ledger；没有把完整会话或隐藏推理保存为训练数据。
- [Podman](https://github.com/containers/podman)：一次性 WSL/Linux 沙箱的外部容器运行时。
  本项目只调用其公开 CLI，不随仓库再分发 Podman 二进制；感谢 Podman 社区维护
  rootless 容器、用户命名空间和资源限制等基础设施。
- [vLLM](https://github.com/vllm-project/vllm)：Linux/WSL2 路径上的主要多 LoRA
  推理与 OpenAI-compatible serving 候选。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：可选的本地低开销推理对照方向。
  当前仓库只记录转换和公平对比待办，不声称已经完成 Gemma 4 Unified + LoRA 的
  等价 serving，也没有把 llama.cpp 源码作为本仓库的一部分重新发布。
- [pytest](https://github.com/pytest-dev/pytest) 与
  [Ruff](https://github.com/astral-sh/ruff)：离线回归测试和静态检查。

## 4. 归属与复核入口

- 直接 Skill 的精确来源、commit、SHA-256 和审计状态：
  [`configs/data/skill_sources.yaml`](configs/data/skill_sources.yaml)
- 随仓库保留的第三方许可证：[`third_party/skills/`](third_party/skills/)
- 简明的第三方 Skill 清单：[THIRD_PARTY_SKILLS.md](THIRD_PARTY_SKILLS.md)
- 本项目自身许可证：[AGPL-3.0-or-later](LICENSE)

如果发现遗漏或归属错误，欢迎提交 issue 或修正。对上游作者和维护者最实际的感谢，
是保留准确归属、遵守许可证、不给他们强加未经证实的结论，并把可复现的改进继续
回馈给社区。

---

**English scope note.** Vendored and injected assets are listed only in Section 1
with pinned commits and local license copies. Section 2 records research inspiration,
not reproduction or result equivalence. Section 3 lists model, service, training, and
serving infrastructure; inclusion does not imply endorsement by any upstream project.
