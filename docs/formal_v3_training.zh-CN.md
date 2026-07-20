# Formal-v3 全量 A–F 训练

Formal-v3 是面向完整公开 SWE-bench train 题库的训练契约，本身不代表已经授权启动。
它只接受通过真实执行 Gold gate 后冻结的不可变快照，绝不直接训练仍在增长的蒸馏目录。

## 规模与切分契约

- 源题库共 19,008 道候选题。
- 每题拆成五个阶段工单，总候选量 95,040。
- 只有五阶段记录都通过真实执行 Gold gate 的题目才能进入冻结快照；失败题不得用合成样本补齐。
- `minimum_records_per_expert=256` 只是质量下限，不是规模上限。每阶段实际冻结的
  Gold 最多可以达到 19,008 条。
- 先按源题库切分，再做 Gold 选择：
  - `train`：最多 17,105 题，只用于训练；
  - `validation-from-train`：最多 1,903 题，只用于 E/F 的 rank 校准；
  - 外部 heldout：只用于评测，在训练快照中只保留哈希和泄漏审计元数据，不复制、
    不读取正文。
- 不可变 manifest 绑定实际 Gold 数、五阶段等量记录数、train/calibration ID 集合哈希、
  两者互斥证明、calibration 文件、外部 heldout manifest 哈希和泄漏审计哈希。

`prepare_full_v3_snapshot.py` 会从 `full_v3_snapshot.yaml` 固定的源题库 manifest 和
allowlist 确定性生成上述契约，而不是在快照之后手写补一个 manifest。

只要 train 与 calibration 重叠、heldout 正文存在/被读取/被输出、Gold 数超过 19,008，
或五阶段 train 记录不能组成相同数量的完整链，训练预检就会 fail closed。

## 按快照实际规模推导 exposure

仓库里继承的 `max_steps` 只是兼容旧 partial-data 配置的占位值，不是正式训练步数；
未物化 schedule 的 B–F adapter 会被直接拒绝。B–F 运行前，
`scripts/train/materialize_formal_v3_schedule.py` 会先核验冻结切分，然后把绑定快照的
正式配置写到：

```text
artifacts/formal_v3/schedules/<snapshot_sha256>/<arm>.json
```

正式控制变量是一轮 train epoch：

- B：一个 rank-16 mixed LoRA，看五个阶段的全部 train Gold。
- C/D/E/F：五个独立 LoRA，每个专家只看自己的阶段。

共享低显存控制组使用保守学习率 `5e-5`、`constant_with_warmup`、`0.03`
warmup ratio，并且只训练这一轮由快照实际规模推导的 epoch。B–F 使用完全相同的优化
超参，不允许按组单独调参。该设置替代早期探索用的 `2e-4`，目的是降低小快照过拟合
风险；它仍只是受控基线，不代表已证明最优收敛。
- B–F 的总 sample exposure 和逐阶段 exposure 都完全相同。
- 当梯度累积 `g=4`、train 完整链数为 `N` 时，每阶段使用 `ceil(N/g)` 个 optimizer
  step。若 `N` 不能被 4 整除，runtime 会分别 shuffle/pad 每个阶段，再确定性交错五个
  stratum；每阶段最多 padding 3 条。manifest 同时记录计划值和运行时逐文件实际 exposure，
  不能再拿 mixed 数据整体 shuffle 的总数冒充逐阶段相等证明。

当源切分全部通过 Gold（`N=17,105`）时，每个专家 job 为 4,277 optimizer steps、
17,108 次 exposure；B 为 21,385 optimizer steps。B–F 每组总 exposure 都是 85,540。
过去的 640-exposure 实验不再是 formal-v3 上限。

每份运行 manifest 都会记录请求/补齐后的 exposure、每阶段 padding、推导出的 steps、
目标 epochs、快照哈希和 B–F exposure 相等约束。安全 checkpoint 频率按正式步数推导，
约均匀保存四次。

## 低显存序列边界

`formal_v3_lowmem_*` 是面向 3080 Ti / 9 GiB 的控制实验配置，不得宣称为完整轨迹训练。
它使用 64-token 窗口和显式 `formal_v3_lowmem_truncated_v1` 合同：保留最近的 prompt
上下文及 assistant completion 的开头。配置和 manifest 都写死
`full_trajectory_training=false`。

真正执行后，manifest 与 checkpoint metadata 会补写运行时观测到的 rendered-token
最大值/均值、selected-token 最大值/均值，以及精确的截断 exposure 数量/比例；统计单位是
sample exposure（包含确定性 padding），不是去重后的样本。后续云端/full-context 配置必须
另行使用经审计的 no-truncation 合同，不能把本低显存结果改名成 full-context 结果。

## A–F 定义

| 组 | 结构 | Rank / 预算 | 训练 exposure |
| --- | --- | --- | --- |
| A | 冻结原生 Q4 基座 | 无 LoRA | 不训练，只做预检和评测基线 |
| B | 一个 mixed LoRA | rank 16，10,387,456 参数 | 五阶段全部 train 数据 |
| C | 五个路由专家 | 每个 rank 16，总 rank 80 | 每专家一个阶段 |
| D | 五个固定小专家 | `3/3/4/3/3`，总 rank 16，严格对标 B | 每专家一个阶段 |
| E | calibration 自适应专家 | 每专家 <=16，不限制总 rank | 每专家一个阶段 |
| F | 与 E 相同自适应机制 | 总 rank 16，参数量严格对标 B | 每专家一个阶段 |

E/F 必须提供不可变 `anchor.lora-allocation.v1` 和同名 SHA-256 sidecar。分配只能使用
calibration split，并必须在打开 heldout 前冻结。E 必须是非均匀 rank；F 必须满足总
rank 16 和 10,387,456 个物化训练参数；两者必须使用同一套尝试/选择机制。
Formal-v3 会明确拒绝历史
`heuristic_preregistered_calibration_pending` manifest：`selection_status`
必须为 `calibration_selected_frozen`，每个尝试过的分配都必须带真实 calibration
指标，最终选中的 rank 组合也必须存在于这些已测尝试中。

## Live Gold 桥接

快照源必须是已认证的 full-bank coordinator 导出，不能再指向
`data/automated_v3`，也不能使用旧 synthetic partition。live 运行进入终态后执行：

```powershell
py -3.10 scripts/data/export_swebench_formal_gold.py
py -3.10 scripts/data/prepare_full_v3_snapshot.py --config configs/orchestration/full_v3_snapshot.yaml
```

导出器只在当前进程读取 WSL root 持有、协议隔离的训练回执 key。每道入选题必须同时具备
五阶段哈希绑定 artifact、唯一最终 review PASS、security PASS、完全匹配的 final patch、
至少一条成功且非平凡的 `anchor-validate` 命令及其模型可见真实工具结果、成功的沙箱清理，
以及 HMAC 认证的 `real_sandbox_self_verified` 回执。该训练证据明确写入
`not_official_swebench_pass=true`；官方 heldout 评测保持独立。
消费端会独立重算 publication manifest 以及所有匹配的 candidate task/work-order
分片 SHA/大小/记录数，要求 run manifest 与 status 绑定同一个题库 manifest 哈希，
再从模型可见的 OpenCode export 中重新解析终态 validator JSON，并将它与不可变训练
沙箱镜像 digest/ID、validator 源码哈希、final patch、工具轨迹、五阶段谱系和 cleanup 后
supervisor HMAC 对齐。任何签给其它分片、镜像、validator、patch 或验证终态的回执都不是 Gold。
已封顶的 `stopped_checkpoint_resumable` checkpoint 可以导出，已完成题保持可用，未完成或未验证题
继续排除并可重试。训练投影保留真实 builder 工具调用/返回、清理过显式隐藏推理字段的
OpenCode session export 和精确 workspace diff；模型自报的 verdict 不能替代 supervisor
回执。

## Formal-v3 评测绑定

每个物化后的 B–F 训练 manifest 都携带
`anchor.formal-v3-af-evaluation.v1`：绑定同一份只含哈希的 heldout，以 `A=100`
归一化，A 为冻结 Q4 基线，B 为单 mixed adapter，C/D/E/F 为五阶段串行 runtime-LoRA
热插拔。评测产物必须按 formal-v3 arm 与版本隔离；严禁混入 formal-v2 adapter、
registry、config 或 report。

正式评测控制文件已经落在
`configs/benchmark/formal_v3_af_control.json`。独立 finalizer 会同时绑定快照
manifest **及其 sidecar**、相同 NF4/Q4 基座清单、B--F 的物化 schedule、execute
manifest、checkpoint metadata、progress、adapter 权重、E/F 冻结 calibration 分配，
以及外部 heldout 的纯哈希元数据；任何 `formal-v2` 输入都会被拒绝。

finalizer 只会在
`artifacts/formal_v3/evaluation/registries/<version>/` 新建不可覆盖的版本 bundle。
离线 preflight 不打开 heldout 题目或 fixture 正文。运行结果只能写入
`runs/formal-v3/evaluation/<version>/...`；Resume 必须保持同一个 version、registry、
heldout 哈希、backend 身份、采样契约和 checkpoint。

```powershell
# 只读检查。当前仓库会 BLOCKED，直到 formal-v3 快照、calibration 分配和
# B--F 完整训练产物都存在。
python scripts/benchmark/materialize_formal_v3_af.py `
  --version-id formal-v3-001

# 训练完成后，排他式生成不可变 registry/benchmark bundle。
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 -Finalize

# 只做离线 preflight；不读 heldout 正文，不调用 API/GPU 评测。
.\scripts\benchmark\run_formal_v3_af.ps1 -VersionId formal-v3-001

# 唯一 live 形式：同一 Q4 基座，A=100，B 单 mixed adapter，
# C/D/E/F 五阶段串行 runtime-LoRA 热插拔。
.\scripts\benchmark\run_formal_v3_af.ps1 `
  -VersionId formal-v3-001 -Execute -AuthorizeHeldoutAccess
```

当前没有执行、也没有宣称完成 formal-v3 heldout 评测。由于训练产物缺失，只读检查会在
打开任何 heldout 正文、调用 API 或启动 GPU 前返回 `BLOCKED`。

## 只读预检与启动

```powershell
# 核验快照并生成 B–F 正式 schedule；不调用 API，不启动 GPU 训练。
.\scripts\train\formal_v3_preflight.ps1

# A 是显式 Q4 基线，不会创建 LoRA 训练任务。
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm A

# 同一冻结快照上的资源探针。
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm smoke -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm probe -Execute

# 正式矩阵；只有显式 -Execute 才会启动 GPU job。
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm B -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm C -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm D -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm E -AllocationManifest <E.json> -Execute
.\scripts\train\run_formal_v3_lowmem.ps1 -Arm F -AllocationManifest <F.json> -Execute
```

不带 `-Execute` 时，只做相同的数据/基座门禁和 adapter dry-run。进程锁保证单 GPU
所有权。只有配置、快照、最终 adapter 和 checkpoint metadata 全部匹配时才跳过已完成
专家。单专家内部 safety checkpoint 仍只是 adapter 权重 warm start，不是假装成包含
optimizer、scheduler、RNG 的精确断点续训。
