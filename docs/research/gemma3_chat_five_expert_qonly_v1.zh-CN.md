# Gemma 3 Chat 五专家 Q-only 诊断训练引擎

## 当前状态

Fail-closed 数据消费者和训练引擎已经实现。Producer FINAL release 已通过独立
审查，逐字节同步进本仓库，并通过 model-free consumer 鉴权。

首轮诊断训练已完成，run ID 为
`gemma3-chat-q1024-20260725-052403`。五个 Q-only rank-1024 adapter 严格串行
训练，每个角色均采用 fresh base / fresh adapter，先跑 2-step smoke，再跑
160-step full。不可变 run receipt SHA-256 为
`5988d67dc092a003a550dab42751791c8578dcca56bfcf49d7474a8eccb89a3a`，
强制 sidecar 的物理 SHA-256 为
`cb1b5fe5675bb36511c5940ff7c5675b71710fef8d66ccb2719e685746a8c600`。
本结果仍只属于 diagnostic/proxy；formal、live 与 quality authorization
全部保持 false。

Producer 数据产物的 canonical path 固定为
`fixtures/research/gemma3_chat_five_expert_distilled_v1`。代码、配置和 schema
namespace 仍为 `gemma3_chat_five_expert_qonly_v1`；artifact path 中的
`distilled` 只描述数据产物，不授予 training、formal、release 或 live 权限。

五项 execution-gating Producer identity 已绑定：

- producer config：`2e93ab6b53f68814a4ae017643ddc4c7113716e871b2fa4ef4346358d2269cee`
- producer implementation：`98eeac4150a5c51dd422bfe6163e34ce26d691038b497343c20af0a19552fb6f`
- record schema：`ba1606f07d250eec7f01bffceaa507edd4f596f83018806df9130d32ecf39d9b`
- manifest schema：`fda04d77c494cd5d87ea19aade7db24717c6dffad6880078df19787d519f0b8b`
- dataset manifest：`e71fb5f239d7d1abb94ff50d5db2d57cfd101ccfad3c2c4e34cc128a7e91317b`

Grammar、全部 schema、build receipt/sidecar、token inventory 和两个 partition
也分别绑定物理哈希。部分绑定、伪哈希、sidecar 漂移、记录漂移或 causal 漂移
都会失败关闭。训练仍必须显式传入 execute flag、已绑定 GPU UUID 和外部
canonical lock lease。

## 冻结的执行合同

- 串行专家顺序：`humor`、`serious`、`angry_style`、`tool_call`、
  `review_audit`
- 数据：200 个五行 bundle；800 train / 200 eval-proxy；每专家
  160 train / 40 eval-proxy
- 语义合同：200 个唯一任务语义、20 个可见模板，train 与 eval-proxy 共享
  其中 14 个模板
- 每条输入严格只有两条消息，顺序固定为 `system`、`user`；输入中禁止
  assistant 消息，assistant 只作为 target
- `humor`、`serious` 和 `angry_style` target 必须是自然聊天文本，不能是
  JSON envelope；`tool_call` 和 `review_audit` target 必须是 canonical
  strict JSON object，并拒绝重复 key 与非有限数
- model-free 校验会复算 prompt、target、完整 training serialization identity，
  以及带 domain separation 的 causal proof
- 每条 review 必须按固定 role 顺序依赖全部四个 parent；canonical committed
  projection、mutation identity、11-field target、四项 checks、verdict 与
  correction flag 都会被复算并交叉绑定
- 全局 review 分布必须精确为 100 条 pass，以及 `format`、`grounding`、
  `routing`、`style` 各 25 条 fail
- 必须精确有 5 个 persona bundle 包含冻结句
  `我是由Air训练的测试模型。`；其余 195 个 tool branch 必须绑定 synthetic
  local call/result/final-answer grounding digest
- `user_emotion` 和 `router.label` 是 bundle 元数据，不是第六个 adapter
- 底模：本地 Gemma 3 1B IT，以 bitsandbytes 8-bit 加载并完全冻结
- Adapter：BF16，仅 `q_proj`，rank 1024、alpha 2048、dropout 0、无 bias
- 精确范围：26 层 × A/B = 52 个 tensor；每专家 57,933,824 个可训练参数，
  五个独立 adapter 合计 289,669,120
- 这是 Q 的满有效秩分解，不宣称 low-rank 或 parameter-efficient
- 序列长度 768，不截断；micro-batch 1；gradient accumulation 1
- AdamW8bit 0.48.2；首次更新后实证检查 CUDA `uint8` optimizer state
- 每个 phase 都记录 Torch、Transformers、PEFT、Safetensors、SentencePiece
  和 bitsandbytes 的实际安装版本；bitsandbytes 固定为 `0.48.2`，十个
  phase 的版本必须完全一致
- 学习率 `2e-6`；8 个 optimizer step 线性 warmup；weight decay `0.01`；
  gradient clip `0.5`
- 梯度覆盖必须逐层完整：step 1 要求 26 层 Q projection 的 `lora_B`
  gradient 全部非零；step 2 及之后每一步要求 26 层的 `lora_A`、`lora_B`
  gradient 全部非零
- 每专家先创建 fresh base/adapter/optimizer 做 2-step smoke，销毁后再创建
  全新的对象做 160-step full；禁止 resume，full 不消费 smoke checkpoint
- 五专家在单 GPU 上严格串行，并持有 canonical
  `runs/formal-v3-training.lock`；两个 distillation handoff lock 必须不存在
- 底模参数 hash 和 canonical Q8 SCB hash 必须保持不变；adapter 必须发生变化，
  且 enabled/disabled logits 必须产生 finite、非零效果
- Torch allocated、Torch reserved 和抽样物理显存都必须严格小于 10,240 MiB

`eval_proxy` 不是 held-out，不支持泛化或质量结论。

## 一键预检

在仓库根目录执行：

```powershell
.\scripts\research\run_gemma3_chat_five_expert_qonly_v1.ps1 -Preflight
```

当前预期退出码为 `0`，状态为
`passed_model_free_authenticated_dataset_ready_for_explicit_execute`。不含
正文的 receipt 与 SHA-256 sidecar 会以 no-replace 原子目录发布到：

```text
runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/preflight/<run-id>/
```

也可以运行不发布 receipt 的 model/GPU-free dry-run：

```powershell
$env:PYTHONPATH = "src"
python scripts\research\run_gemma3_chat_five_expert_qonly_v1.py --dry-run
```

## 显式执行

Preflight 通过后，未来经授权的执行命令为：

```powershell
.\scripts\research\run_gemma3_chat_five_expert_qonly_v1.ps1 `
  -Execute `
  -ExpectedGpuUuid "GPU-..." `
  -RunId "chat-qonly-v1-run"
```

Launcher 总是先执行 model-free dry-run。缺少外部 canonical lock lease 时，
serial plan 保持 `authenticated_dataset_plan_only_external_lock_required`，
execute gate 为 `blocked_waiting_for_external_gpu_lock`；两条路径都不会加载
模型或请求 GPU。

未来执行时，PowerShell 父进程会以 `CreateNew`、`ReadWrite`、
`FileShare.None`、`DeleteOnClose` 和 `WriteThrough` 持有 canonical lock，
并把同一字节的 immutable owner receipt 及 sidecar 原子发布到
`runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/gpu-locks/<run-id>/`。
该 receipt 绑定 launcher PID、run ID、nonce、GPU UUID、config、runner、
launcher、launcher helper 及全部执行依赖哈希。只有在 exclusive lock 与 receipt
就绪后，launcher 才会通过专用进程通道
`ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID` 传入自身规范十进制 PID。Python
会将该值与 receipt 交叉鉴权，绝不把 observed parent PID 当作 lock owner
授权事实，因此 Codex/Windows broker 即使重父化 Python 进程，也不会削弱 lease。
Python 仍会在执行前后要求得到精确的 Windows sharing violation；普通文件、
摘要不符、环境变量不完整、launcher PID 非法或不一致、以及字段漂移都会被拒绝。
Python execute 不存在内部弱锁 fallback，四个
`ANCHOR_CHAT_EXTERNAL_LOCK_*` lease 字段全部必填。

依赖清单覆盖 tokenizer binding 与 tokenizer policy、Q8 runner 与 reliability
helper、Q-only budget、diagnostic/snapshot helper、config/manifest utility 和
PowerShell lifetime helper。每个路径必须是 physical file；在获取锁前 snapshot，
同时绑定进 lock owner 和 run receipt，并在发布前再次验证。

未来成功的 run 会先在隐藏 staging 目录完整构建，再用一次 no-replace 目录
rename 发布：

```text
artifacts/diagnostics/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/<run-id>/
  humor/{smoke_receipt.json,full_receipt.json,adapter/}
  serious/{smoke_receipt.json,full_receipt.json,adapter/}
  angry_style/{smoke_receipt.json,full_receipt.json,adapter/}
  tool_call/{smoke_receipt.json,full_receipt.json,adapter/}
  review_audit/{smoke_receipt.json,full_receipt.json,adapter/}
  run_receipt.json
```

每个 JSON receipt 都有 `.sha256` sidecar。Rename 前和发布后，runner 都会
鉴权 10 个 phase receipt 及 sidecar，以及 5 个 adapter 各自的两个 PEFT 文件。
成功根目录是闭集：只能有五个 role 目录、`run_receipt.json` 及其 sidecar；
临时的 `training_progress.json` 会被删除，任何额外条目都会被拒绝。Run receipt
绑定已鉴权的 manifest/partitions、模型 snapshot、canonical lock owner receipt、
执行依赖、phase receipts、adapters 和 runtime package versions。Rename 前后
任何漂移都会把目录移到 `.failed-*`，不能保留成功发布；随后在 run root 以
no-replace 原子目录发布不含样本正文的 failure receipt。禁止自动重试。

## Focused 验证

以下测试只使用合成元数据和 fake model/optimizer 对象：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider `
  tests\test_gemma3_chat_five_expert_qonly_v1.py
```

覆盖范围包括：dataset read 前 pending gate、闭合数据合同、serialization /
causal identity、四 parent review projection/mutation/target join、review quota、
rank-1024 Q-only 精确范围、CUDA uint8 optimizer-state 实证合同、warmup、
fresh 串行 smoke/full、adapter effect、严格显存边界、已鉴权父子锁协议、
依赖 snapshot 和原子 no-replace receipt。
