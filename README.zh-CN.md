# Anchor-MoE-LoRA

[English](README.md) | [简体中文](README.zh-CN.md)

Anchor-MoE-LoRA 是一个可运行的研究脚手架：在同一个冻结基座模型上使用
**按任务路由的 LoRA 专家**。它通过可观测的应用层 DAG，依次处理规划、工具策略建议、
前端生成、配对前端审查，以及最终的防御性安全门禁。

仓库与发行名称是 `anchor-moe-lora`。Python 导入包仍为 `anchor_mvp`，预备的 Conda
环境仍为 `anchor-mvp`。下文命令均假设工作目录是仓库根目录。

端到端使用、密钥处理、Provider、自动蒸馏、OpenCode execution gold 与常见故障处理，
请参阅 [English quickstart](QUICKSTART.md) 或 [简中快速上手](QUICKSTART.zh-CN.md)。
当前 OpenCode live 门禁也以这两份文档为准。

这里的 MoE 不是神经网络内部的 Mixture-of-Experts 层。待验证的命题是：在调用次数、
token 与墙钟时间预算匹配的条件下，专用 adapter 加显式路由是否优于混合 adapter。

## 已实现内容

- 注入 SOP、异步且可断点恢复的教师数据生成。
- 严格的 `plan -> tool_policy -> frontend -> local benign mutation -> review -> security`
  数据依赖；下游教师不会自行编造或回显候选代码。
- 工具策略提案只是本地惰性数据；模型标签不会授予运行时权限。
- 规范 JSONL、公开 `decision_trace`、来源哈希、清洗与去重。
- Kimi Code Anthropic 优先、OpenAI 回退客户端；真实客户端身份、密钥脱敏、请求/token
  预算、probe 模式和确定性离线 mock。
- 面向 Gemma 4 12B 基座的五个专家 QLoRA 配置，以及一个 mixed 基线。
- 支持 NF4 在线量化与兼容的预量化 PEFT 加载；训练器会明确拒绝仅供推理的 GGUF 和
  W4A16 产物。
- vLLM/OpenAI-compatible 客户端、`frontend -> review -> security` DAG、fail-closed
  处理和结构化分阶段 trace。
- A/B/C/D 对照，以及计划中的 E/F 自适应 rank 组；延迟/token/VRAM/错误统计和
  Pass@1/TPR/FPR 指标挂钩。

## 本地环境

预备环境与 Anaconda `base` 隔离：

```powershell
conda activate anchor-mvp
cd <repo>
python -m pip install -e .
python -m pytest
```

若要在不修改 `base` 的前提下复现环境，请在 PowerShell 中运行
`scripts/environment/bootstrap_windows.ps1`。

训练路径已在一张 RTX 3080 Ti 12 GiB 显卡上实测，显卡通过 OCuLink PCIe 4.0 x4
连接。基于持久化 NF4 checkpoint，真实 rank-16 更新已完成 forward、backward、
paged 8-bit optimizer step、adapter 保存与重新加载。这只是资源受限条件下的可行性结果，
并不表示该硬件或最终模型质量最优。

已验证的低显存配置使用序列长度 64、batch size 1、冻结 NF4 基座权重、BF16 主计算，
并让符合条件的 FP32 矩阵乘法使用 TF32。归一化参数仍保持 FP32。rank 32/64、更长上下文
和更大 batch 必须通过实测峰值 VRAM 门禁；不预设它们一定能装入显存。

## 安全的首次运行

先在不访问网络、不消耗额度的情况下跑完整条数据路径：

```powershell
python -m anchor_mvp data --config configs/data/smoke.yaml --dry-run run
```

使用 Kimi Code 时，只把 key 写入当前 shell。不要把 key 写入 YAML、`.env`、命令参数、
日志或源码仓库：

```powershell
$env:KIMI_API_KEY = Read-Host -MaskInput "Kimi Code key"
python -m anchor_mvp data --config configs/data/default.yaml probe
python -m anchor_mvp data --config configs/data/default.yaml --seed-count 3 --concurrency 1 run
Remove-Item Env:KIMI_API_KEY
```

稳定模型 ID 是 `kimi-for-coding`。Kimi 文档要求调用方保留真实 User-Agent；本项目标识为
`anchor-moe-lora/0.1`，不会冒充 Claude Code。Thinking 会被显式启用和配置，使 probe
可以验证是否选择了 coding model 路径。隐藏推理会被丢弃；只有简短、可审计的决策产物进入数据集。

Anthropic live 生成使用非流式传输；所有 live transport 默认 HTTP read timeout 为 600 秒，
重试一次。probe 使用相同策略。请通过 `--timeout-seconds` 和 `--max-retries` 显式调节。
即使超时响应没有生成通过校验的 JSONL 行，Provider 也可能已经扣除额度，因此不要激进重试。

OpenAI-compatible Kimi 调用默认使用 SSE（`stream_openai: true`），与文档规定的第三方设置
一致。客户端只缓存最终 content delta 并丢弃 reasoning delta；完整响应结束后才原子写入 JSONL。
除非 probe 确认兼容，否则可选的 `stream_options.include_usage` 保持关闭。

mock 和单 seed 门禁通过后，可通过 `scripts/data/start_automation.ps1` 进行无人值守蒸馏。
并发默认为一；操作者可以配置任意非空正整数序列，代码不设静态上限。Automation 会持久化
Provider cooldown，执行请求/token/失败预算，并提供原子状态与 append-only 事件。每个并发档位
还会对五份当前训练 JSONL、五份 SOP 和冻结 held-out 语料重新检查；任何碰撞都会阻止升级并发。
Tool policy 与 security 使用独立的 low-effort worker，其他数据 worker 使用 medium effort。
脚本不会自动启动。冻结 manifest 与审计合同见
[held-out benchmark](docs/heldout_benchmark.md)。

## 模型与训练

非 instruction-tuned 基座固定为：

```text
google/gemma-4-12B
revision 56820d7d8cbe8e47975a53325439ed272e91cff2
```

固定的基座 snapshot 只导出一次，成为可重新加载的 Transformers/bitsandbytes NF4 训练
checkpoint。后续任务直接加载这个持久化 Q4 产物，避免反复在线量化及其主机内存与 PCIe 开销。
W4A16 部署副本仍是独立的推理产物。

无需下载或训练即可验证一个 adapter：

```powershell
python -m anchor_mvp train `
  --config configs/training/gemma4_12b_qlora_smoke.yaml `
  --adapter frontend_gen `
  --dry-run
```

只有在数据集存在、dry-run manifest 报告环境 ready 后，才使用
`--execute --allow-model-download`。参阅[训练细节](docs/training.md)。当前证据、被中断的实验
与剩余工作记录在[项目状态](docs/PROJECT_STATUS.md)中。

## Serving 与 benchmark

主要 A/B/C/D/E/F 对比有一个不可变的控制变量合同：所有组加载完全相同的本地序列化 Q4/NF4
基座产物（相同模型 revision、量化设置、tokenizer 与产物 digest），执行相同的五个有序阶段和
相同的每阶段 token cap，仅 adapter 分配/rank 分配不同。A 不使用 adapter；B 在全部阶段复用
一个 mixed adapter；C 路由五个完整 rank-16 专家；D 手工匹配 B 的总参数预算；E 搜索可变预算
的自适应 Pareto 分配；F 使用同一自适应分配器，但硬性限制为 B 的预算。报告展示 A 的绝对指标，
并把 A 归一化为指数 `100`。公平的等预算比较是 B/D/F；C/E 是容量/Pareto 对照。更换 Q4
产物会使实验失效。

官方 vLLM 只支持 Linux；真实 server 入口使用已安装的 WSL2 Ubuntu。先使用 3080 Ti 安全
配置：1–2K context、一个 sequence、一个 active adapter、CPU 缓存不活跃 adapter，以及
eager execution。只有 VRAM probe 通过后，才启用 CUDA graph 和吞吐增强功能。

应用可以连接任何 OpenAI-compatible 后端，因此可以把 vLLM 与更低开销的引擎比较，而不把
vLLM 视为强制依赖。参阅 [serving 与 benchmark 细节](docs/serving_benchmark.md)。

## 项目地图

```text
skills/                 版本化专家 SOP
src/anchor_mvp/data/    teacher、cleaning、schema、storage、pipeline
src/anchor_mvp/training QLoRA 配置、校验、runtime、manifest
src/anchor_mvp/serving/ backend client 与 routed adapter DAG
src/anchor_mvp/benchmark/ 公平基线、记录与指标
configs/                smoke 与实验配置
scripts/                模型、训练和 WSL serving 入口
tests/                  离线单元与集成测试
```

[文档索引](docs/README.zh-CN.md)列出公开的 English/简体中文文档对。新增或实质修改的
公开文档遵循[双语发布规范](docs/documentation_i18n.zh-CN.md)。

## 许可证

Copyright (C) 2026 emoair.

Anchor-MoE-LoRA 使用
[GNU Affero General Public License v3.0 or later](LICENSE) 许可。通过网络提供的修改版本
必须按许可证要求提供对应源码。捆绑的第三方 Skill 保留各自原始许可证和署名；参阅
[THIRD_PARTY_SKILLS.md](THIRD_PARTY_SKILLS.md)。

## 致谢

本项目由 OpenAI 的 GPT-5.6 SOL 辅助构建。

本项目建立在公开分享的 Skill、adapter 研究、模型/训练技术栈与 serving 工具之上。
直接引入的资产、研究启发和基础设施依赖分别记录在
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) 中；适用时会固定 commit 并标明原许可证位置。
隔离的 SWE task-card importer 会在
[docs/swebench_metadata_import.md](docs/swebench_metadata_import.md) 中记录 SWE-bench/SWE-smith
来源策略和官方上游致谢。
