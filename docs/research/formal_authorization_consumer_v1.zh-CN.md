# 正式授权消费端 v1

这个增量覆盖层只回答一个问题：**当前研究管线是否已经获得正式训练授权？**
v1 的答案永远是否。它不能启动训练，也不存在自动晋级路径。

作出结论前，它会认证以下版本隔离的精确输入：

- Qwen prerequisite v2 消费端、配置和带强制 sidecar 的 companion
  manifest；
- multi-seed / independent-bundle 阻塞计划及其配置；
- generic release v2 消费端与 schema；
- 物理文件 `formal_authorization_decision_v1.schema.json`，其 Draft 2020-12
  契约只允许 v1 的阻塞状态。

Python 依赖不是从普通 import 缓存中直接调用，而是从已认证 SHA-256
的单次字节快照编译执行。决策结束后会再次核对所有快照，因此并发替换会
fail-closed。

## 机器结论

- formal-v3：`0/5`；
- 受保护来源 inventory：`2/6`；
- controlled-factorial 数据集仍是 secondary proxy，不能满足 independent
  confirmation 或 bundle-generalization；
- `anchor.generic-train-release-lock.v2` 的范围仅为
  `research_proxy_only`，满足旧 schema 不等于正式发布；
- `training_authorized=false`、`formal_training_authorized=false`、
  `formal=false`。

决策明确记录 provider、网络、模型、GPU、受保护正文和训练操作全部为零。
即使认证成功，CLI 仍返回退出码 2，因为授权结论保持阻塞。

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.formal_authorization_consumer `
  --config configs/research/formal_authorization_consumer_v1.yaml
```

该覆盖层不修改冻结的 Qwen prerequisite v1/v2 契约，也不会读取 Gold、
held-out、scaffold 或训练记录正文。

## 执行入口封锁

正式执行在三个相互独立的入口 fail-closed：

1. `run_formal_v3_lowmem.ps1` 只接受仓库唯一的
   `runs/formal-v3-training.lock`，先取得该单 GPU 锁，再调用本消费端；任何
   v1 决策都会被拒绝，即使伪造字段声称 `ready=true`；
2. `anchor_mvp.training.cli` 会在设备探测、preflight/正文读取、数据校验、
   manifest 写入和 runtime import 之前检查授权；
3. `train_adapter()` 会在创建进度或输出目录之前再次检查。

v1 不存在 ready 路径。未来即便出现裸的
`formal_training_authorized=true`，也不足以执行。任何 future-ready 实现必须使用
新的、版本化的 v2 或更高版本决策，并提供由持锁 launcher 生成、绑定决策、
运行配置、adapter/rank/stage 和 canonical 锁身份的认证 execution lease。v1
尚无该决策/lease 契约，所以 launcher、直接 CLI 和库调用仍会拒绝。
