# Formal-v3：full_v3 冻结数据的低显存训练

Formal-v3 复用已经跑通过的 `manual_active_labels_v2` 低显存引擎，但不再从
formal-v1 或 `data/automated_v2` 继承训练数据。所有训练配置只接受未来冻结到
`artifacts/formal_v3/dataset/` 的 full_v3 快照；当前仍在增长或尚未通过质量门的
`data/automated_v3` 不能直接训练。

## 启动前硬门

预检必须同时满足：

1. 五个专家各有至少 128 条有效、真实教师样本；跨文件 ID 唯一且 target 非空。
2. `manifest.json` 使用 `anchor.training-snapshot.v2`，其
   `manifest.json.sha256` sidecar、五个文件的 basename/记录数/bytes/SHA-256、
   `source_partition_manifest_sha256` 和顶层 `snapshot_sha256` 全部一致。
3. 实际训练目录必须是 Transformers 可重载的 bitsandbytes NF4 目录，而不是
   GGUF、W4A16 推理包或任意“看起来像 Q4”的目录。预检会核对量化导出 manifest、
   来源权重 SHA、NF4/double-quant/BF16 compute-storage 配置、四个 safetensors
   分片的存在性与 bytes、index 的 shard/总大小/NF4 quant-state 绑定。
4. 冻结基座、`q_proj/v_proj`、BF16 LoRA、TF32、`paged_adamw_8bit`、batch 1、
   gradient checkpointing、序列长度 64 和训练峰值不超过 9 GiB 的契约不得漂移。
5. 正式训练前先完成同一快照上的 one-step smoke，再完成 two-step GPU probe。

默认的 NF4 校验使用 manifest + 文件大小，不会重新读取约 7.7 GB 分片。需要发布级
深校验时加 `-DeepBaseChecksum`，此时 BF16 来源权重和 NF4 分片都会重新计算 SHA-256。

```powershell
.\scripts\train\formal_v3_preflight.ps1
.\scripts\train\formal_v3_preflight.ps1 -DeepBaseChecksum
```

在冻结快照尚未生成时，预检失败是预期行为；这道失败用于阻止旧数据或增长中的数据
误入 GPU。

## 固定低显存参数

- Gemma 4 12B、训练兼容的预量化 bitsandbytes NF4、基座冻结
- NF4 double quant；BF16 compute/storage；LoRA BF16；Ampere TF32
- `max_seq_length=64`，micro-batch 1，梯度累积 4
- `paged_adamw_8bit`，gradient checkpointing，active-label-only loss
- 单专家 32 optimizer steps × 4 = 128 次样本曝光
- B 组 160 optimizer steps × 4 = 640 次样本曝光，与五个专家合计完全一致
- 专家每 8 steps、B 每 32 steps 原子保存一次安全 checkpoint

## A–F 控制组

| 组 | 结构 | Rank/预算 | 训练配置 |
| --- | --- | --- | --- |
| A | 原生 Q4 基座 | 无 LoRA | 不训练 |
| B | 一个 mixed LoRA | rank 16，10,387,456 参数 | `formal_v3_lowmem_mixed.yaml` |
| C | 五个独立专家 | 每个 rank 16，总 rank 80 | `formal_v3_lowmem_common.yaml` |
| D | 五个固定小专家 | `3/3/4/3/3`，总 rank 16，严格对标 B | `formal_v3_lowmem_budget.yaml` |
| E | 复杂度自适应专家 | 每专家最大 rank 16，不限制总预算 | `formal_v3_lowmem_adaptive.yaml` |
| F | 与 E 同一选择机制 | 总 rank 16、参数量严格对标 B | `formal_v3_lowmem_adaptive_budget.yaml` |

E/F 必须提供在 calibration split 上生成、且在打开 held-out 前冻结的
`anchor.lora-allocation.v1` JSON。它必须准确包含五个专家，绑定本次
`dataset_snapshot_sha256`，并提供同名 `.sha256` sidecar。launcher 还会强制检查：

- `mechanism_id=stage_complexity_calibration_pareto_v1`；
- calibration snapshot hash、完整 `attempted_allocations`、最终 `selected_ranks`、
  selection objectives、创建与冻结时间；
- held-out 尚未打开，且访问策略仍是 `forbidden_until_allocation_frozen`；
- base contract、`q_proj/v_proj` 与每 rank 649,216 参数保持一致；
- selected allocation 必须真实出现在 attempted allocations 中；
- E 必须是非均匀 rank，F 必须满足 rank 总和 16 与 10,387,456 个物化参数。

物化参数由 launcher 根据 ranks 重新计算，不能靠 manifest 自报数字绕过。

## 安全启动顺序

入口默认只跑 preflight；只有显式 `-Execute` 才会启动 GPU。一次只能选择一个组，
脚本使用原子 lock 保证同一时间只有一个 formal-v3 GPU launcher。

```powershell
# 数据冻结后先探针
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm smoke -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm probe -Execute

# MVP 主线：五个 rank-16 LoRA，严格逐个训练
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm C -Execute

# 其他对照组
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm B -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm D -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm E -AllocationManifest <E.json> -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm F -AllocationManifest <F.json> -Execute
```

不带 `-Execute` 时会执行相同的数据/基座预检和 adapter dry-run，不下载模型、不启动训练。

## 中断与恢复边界

五个专家是五个独立作业。重复运行同一组时，launcher 只有在 execute manifest、当前
配置指纹、冻结快照 hash、最终 adapter 文件与 `checkpoint_metadata.json` 全部匹配时，
才会跳过已完成专家，因此可在**专家作业边界**继续训练剩余专家。

单个专家内部的 safety checkpoint 目前只包含 LoRA 权重，不含 optimizer、scheduler
或 RNG 状态，能力被明确标记为 `adapter_weights_warm_start_only`。因此 launcher 遇到
partial/stale 输出会 fail closed，不会伪装成 exact resume，也不会静默从第 0 步覆盖。
如需利用该 checkpoint，必须先做显式 warm-start 审计并记录 optimizer/sample schedule
重新开始；在真正接入严格绑定的 warm-start CLI 前，不应把它称为断点续训。
