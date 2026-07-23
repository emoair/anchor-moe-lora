# Gemma 3 1B IT 五角色 Q-only 诊断训练器

这是面向 1000 条中英双语五角色合成数据的 controlled-proxy 训练器，和
formal-v3 严格隔离。即使诊断训练成功，
`training_authorized=false`、`formal_training_authorized=false` 和
`eval_proxy_is_heldout=false` 也不会被提升。

## 实际运行内容

角色顺序固定：

1. `planner`
2. `tool_policy`
3. `frontend_gen`
4. `frontend_review`
5. `security_gate`

显卡里同时只驻留一个角色。每个角色都严格执行：

1. 全新 Gemma base 对象 + 全新 rank-4 `q_proj` LoRA，跑 2 个 smoke
   optimizer step；
2. 销毁模型、optimizer，并清理 CUDA cache；
3. 再创建一个全新 Gemma base 对象 + 全新 adapter，从 step 0 开始跑
   160 个 optimizer step。

full 阶段绝不读取 smoke 权重、optimizer 状态或 checkpoint。两阶段故意使用
相同 seed，因此全新 adapter 的初始摘要必须一致；对不上就立刻失败。

固定数值配置为 BF16 计算、TF32 矩阵乘、SDPA、microbatch 1、
gradient accumulation 1、bitsandbytes 0.48.2 `AdamW8bit` `2e-5`，以及严格
768 token、不允许截断。每个 Q-LoRA 张量的两个优化器动量都必须实际生成为
CUDA `uint8` 状态。bitsandbytes 0.48.2 要求兼容用的 `optim_bits` 构造参数
保持为 `32`，因此运行器以真实 state dtype 为准，不把该兼容参数误当成存储位宽；
Torch allocated 和 reserved 峰值门均为 23.4 GiB（23962 MiB）。该用户授权的
诊断上限有意覆盖专用显存与 WDDM 共享显存总预算；启动前物理 GPU 身份仍锁定
为 12 GiB。
1000 条样本实测长度为 449–665，其中 514 条超过 512，因此 512 不属于可接受
配置。

## Adapter 输出效果门

每个 smoke 和 full 阶段都必须证明的不只是 adapter 文件发生了变化。训练结束后，
运行器固定选择第一条训练记录，定位第一个受监督的 next-token 位置，并把前向输入
物理截断到该目标 token 之前；随后对完全相同的前缀执行两次前向：一次启用
Q-LoRA，一次进入 PEFT 的 `disable_adapter()` 上下文。这样未来 target 后缀不会
进入前向，也不会为其生成 logits。只有两组末位置 logits 及其绝对差都为有限值，
并且 `max_abs > 0`，该阶段才允许通过。

收据只记录 `max_abs`、`mean_abs`、词表宽度、长度和带命名空间的记录/视图哈希，
绝不包含样本正文或 token ID。这只是输出效果诊断，不代表质量、泛化或正式训练
结论。

## 独立 Agent 的 KV 边界

配置已经把目标运行边界写成不可漂移约束：

- 相同且有序的共享前缀必须在 adapter-off 状态下计算，并保持只读；
- Q-only 专家激活后，其激活后的 prompt token 与新生成 token 只能追加到该
  专家自己的私有 tail KV；
- 不同专家绝不能复用彼此的私有 tail；
- 跨角色只传递已经提交的文本，下一阶段必须重新编码这些文本。

这个“追加到专家私有 tail”的约束是让专家像独立 Agent 工作的必要条件。它不
等于整段生成 KV 共享，也不等于普通 in-stack Q-LoRA 的精确 KV 复用，更不是
token-level MoE。当前训练器已经绑定该契约，但还没有物理实现推理缓存，所以
仍诚实标记 `runtime_private_tail_materialized=false`。

## 纯 CPU/tokenizer 预检

默认启动不会查询 GPU：

```powershell
Set-Location D:\LLM\anchor-moe-lora-neural-swarm
.\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.ps1
```

也可以直接使用 Python：

```powershell
python `
  .\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.py `
  --dry-run
```

预检会认证：

- 本地 Gemma 导出的全部 5 个文件；
- tokenizer/template binding 与强制 SHA sidecar；
- 在筛选角色前完整认证 1000 条正式 consumer 记录；
- 精确 labels 与 768 token 零截断门；
- rank-4 五专家参数预算。

它不会请求 provider 或网络，也不会把模型张量载入 GPU。

## 显式启动诊断训练

执行前必须绑定显卡的完整物理 UUID；空值或 `UNBOUND` 都会被拒绝：

```powershell
$env:ANCHOR_GEMMA_GPU_UUID = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
.\scripts\research\run_gemma3_1b_it_five_role_qonly_v1.ps1 -Execute
```

也可以显式传 `-ExpectedGpuUuid`。启动器会：

- 加锁前连续采样 3 次，持锁后再采样 3 次；
- 核对固定 12 GiB 显卡身份、空闲显存、利用率和温度；
- 拒绝任何外来 compute PID；
- 用 `CreateNew + FileShare.None` 持有
  `runs/formal-v3-training.lock`；
- 拒绝旧版和 v3 handoff GPU lock；
- 整个五角色流程始终持锁；
- 发布带 SHA sidecar 的 lease 与 GPU attestation；
- 并发恒为 1。

Python 训练器只复制一次 5 个已认证模型文件，形成一次运行专属私有快照；10 个
全新模型对象都从同一只读快照加载。最后会重新哈希快照并删除。完整 adapter
和回执使用原子方式发布到：

`artifacts/diagnostics/gemma3_1b_it_five_role_qonly_v1/<run-id>`

失败时只发布不含样本正文的 failure receipt；不自动重试、不 resume、不偷偷
降配，也不发布半成品 adapter。

## 关键文件

- 配置：
  `configs/training/gemma3_1b_it_five_role_qonly_v1.yaml`
- 核心实现：
  `src/anchor_mvp/training/gemma3_five_role_qonly_v1.py`
- Python 入口：
  `scripts/research/run_gemma3_1b_it_five_role_qonly_v1.py`
- PowerShell 启动器：
  `scripts/research/run_gemma3_1b_it_five_role_qonly_v1.ps1`
- tokenizer 绑定：
  `fixtures/research/gemma3_1b_it_tokenizer_binding_v1/manifest.json`

本诊断不代表 tag 或 release。
