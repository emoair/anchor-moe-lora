# Qwen 多种子受控因子 dry-run 计划

这是一个附加式、低内存计划器。它不会加载模型、读取数据正文、调用 GPU 或
网络，也不会授权训练。运行时会复跑已冻结的 prerequisite/risk consumer，并且
只接受其 `blocked` 结果。

## 复现实验设计

- 五个 master seed：`1337`、`7331`、`104729`、`130363`、`20260723`。
- `adapter_init`、`record_order`、`cuda` 分别使用 Producer 已冻结的派生算法。
- checkpoint 固定为 `5/10/20/40/80`，只有 step 80 是主终点。
- discovery track 用旧预算 1,376,256 复现 Q-only、Q+O、wide 三臂。
- mechanism track 用共同预算 1,204,224 比较 Q14、Q7+O7、wide
  Q4+O3+K6+V6、独立 O14、K12+V12。
- 两条 track 禁止横向比较；retained-O 绝不能改名为独立 O-only。

公平性固定为：80 steps、LR `5e-5`、完整 AdamW 参数（`foreach=false`、
`fused=false`）、batch/累积 `1/1`、seq 512、BF16+TF32/high、非重入
gradient checkpoint、alpha/rank=2，每个 arm 重新加载 base/adapter/optimizer
且禁止 resume。还强制确定性 Torch 算法、确定性 cuDNN、
`cudnn.benchmark=false` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并冻结各 arm
预期的可训练参数量与张量数；run/artifact key 必须带 `track_id`。六种 arm
顺序只用于独立吞吐测量：warmup 1、计时 6 次、CUDA
同步、runtime/温度时钟 receipt、单 GPU 串行、并发上限 1、显存峰值 5 GiB。

## 受控因子确认集

真实确认集尚不存在。蓝图固定 60 个 source bundle，先按
`task_bundle_sha256` 切成 40 train / 20 eval-proxy，再扩增五角色和两个
variant。中英双语 × 五个固定 strata 组成十个 cell；每个 cell 六个 bundle，
严格为 `4 train + 2 eval_proxy`。

三种因子为旧任务/新模板、新任务/旧模板、新任务/新模板。任务、模板、任务-模板
pair 分别使用同域 inventory：旧维度验证 membership，新维度验证 non-overlap，
所有 pair 验证 pair-level non-overlap。配额固定为 `train=13/14/13`、
`eval=7/6/7`；真实 bundle identity 缺失时只输出配额表。

这个次级 controlled-factorial probe 不能满足 Producer 的独立确认门，也不能证明
bundle 泛化；输出会分别报告两者仍为 blocked。这五个 plumbing 文件不能晋级
materialization 或训练。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_multiseed_independent_bundle_plan `
  --config configs/research/qwen_multiseed_independent_bundle_plan_v1.yaml
```

退出码 `2` 是预期结果。在确认集、三套 inventory、membership/non-overlap 证明及
既有 formal 门控齐备前，所有 readiness/authorization/claims 都保持 false。计划
还会记录自身 config/implementation 的物理 SHA-256，并在返回前再次检查两者。
prerequisite/risk consumer 不会从 Python 模块缓存直接导入；计划器会先认证其源码
字节快照，再隔离加载并调用这份完全相同的快照。
