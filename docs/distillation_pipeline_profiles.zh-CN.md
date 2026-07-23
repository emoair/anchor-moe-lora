# 版本化蒸馏管线 Profile

## 状态与范围

本文定义首个受版本控制的蒸馏管线 profile：
`task-level-moe-lora-v1`。

该 profile 是经过认证的 canonical 五阶段 Gold **之后**的纯元数据分流边界。
它不改变 SWE-bench source bank、teacher prompt、provider 路由、CC Switch、
patched OpenCode、沙箱、验证器、HMAC receipt、checkpoint/resume 逻辑或
canonical Gold 字节。它也不授权 live 执行、训练、release、provider 请求、
模型加载或 GPU 工作。

共同执行底座保持为
`anchor.swebench-five-stage-execution-core.v1`。未来的
`frozen-prefix-qreader-v2` 可以通过独立 TaskBoard/scaffold 投影消费同一份
不可变 canonical Gold，但不得修改本 v1 profile，也不得暗中重新解释其五个
直接分区。

## 为什么边界位于 canonical Gold 之后

现有执行协调器是刻意严格且面向 SWE-bench 的：五个阶段、task/work-order
schema、真实仓库工作区、验证命令和认证执行 receipt 都已固定。若把任意
prompt 模板当作 profile 字段，会削弱此契约，也会让两个消费版本无法复用
昂贵的 teacher 与沙箱执行。

因此，v1 profile 认证共同底座后只选择以下 post-Gold 视图：

| Canonical 阶段 | Task-level 专家 | 分区文件 |
| --- | --- | --- |
| `planner` | `planner` | `data_plan.jsonl` |
| `tool_policy` | `tool_policy` | `data_tool_policy.jsonl` |
| `domain_builder` | `frontend_gen` | `data_frontend.jsonl` |
| `domain_review` | `frontend_review` | `data_review.jsonl` |
| `security` | `security_gate` | `data_security.jsonl` |

TaskBoard 投影、自然语言脚手架、frozen-prefix Q-reader 声明和源记录重写在
本 profile 中全部为 false。

## 文件与身份

严格的 Draft 2020-12 schema：

`configs/orchestration/distillation_pipeline_profile.schema.json`

冻结后的纯元数据 manifest 还有独立的严格 Draft 2020-12 schema：

`configs/orchestration/distillation_profile_freeze_manifest.schema.json`

唯一允许的 v1 profile：

`configs/orchestration/profiles/task_level_moe_lora_v1.json`

profile 以项目相对路径、精确字节数和 SHA-256 绑定两份发布 schema 以及：

- `anchor.ps1`；
- full-bank builder、实现与配置；
- coordinator 实现与配置；
- HMAC execution-contract 与 receipt-runtime 实现；
- formal Gold 导出脚本与实现；
- profile 实现；
- profile CLI runner。

依赖角色与顺序均为闭集。未知字段、缺失绑定、重复 JSON key、非有限数、
路径逃逸、路径复用、symlink/reparse point、字节数漂移、hash 漂移或末端
TOCTOU 漂移一律拒绝。

## 离线预检

独立执行 profile 认证：

```powershell
py -3.10 scripts/data/run_distillation_profile.py preflight `
  --profile configs/orchestration/profiles/task_level_moe_lora_v1.json
```

输出仅包含 hash、路径、计数和 false 授权状态。它不读取 canonical Gold 或
heldout 正文，provider/network/model/GPU 请求均为 0。

未传 profile 时，原入口行为保持不变：

```powershell
.\anchor.ps1 -Action distill-swebench
```

显式选择已认证 v1：

```powershell
.\anchor.ps1 -Action distill-swebench `
  -DistillationProfile task-level-moe-lora-v1
```

入口先认证 profile，再运行现有 full-bank 和 coordinator 离线门禁。为避免
部分认证或歧义配置，`-DistillationProfile` 不得与 `-SWEConfig` 或
`-SWECoordinatorConfig` 同时使用。

profile 不会替代 `-ConfirmLive`。profile 中的 `live_authorized=false` 是
有意设置：只有现有四门协调器与操作员的显式动作才可能尝试 live 工作。
同理，profile 认证绝不能授权训练或 formal release。

## 冻结 profile 身份

在本地 `artifacts` 下冻结纯元数据 manifest：

```powershell
py -3.10 scripts/data/run_distillation_profile.py freeze `
  --profile configs/orchestration/profiles/task_level_moe_lora_v1.json `
  --output-dir artifacts/distillation-profiles/task-level-moe-lora-v1
```

freeze 为 create-once 且原子发布。manifest 在序列化前后都会通过已发布
freeze schema 的真实校验，只生成：

- `manifest.json`；
- mandatory `manifest.json.sha256`，格式严格为
  `<sha256>  manifest.json\n`。

输入按单次 bytes snapshot 读取，并在末端重新认证。现有输出绝不覆盖。
manifest 重申全部非授权状态，不记录任何样本正文。

## Fail-closed 矩阵

| 条件 | 结果 |
| --- | --- |
| Profile/schema/依赖路径、字节数或 SHA 漂移 | 拒绝 |
| 重复 key、未知字段、非有限值 | 拒绝 |
| Symlink、junction/reparse point 或项目根逃逸 | 拒绝 |
| 初始 snapshot 到末端重验之间依赖变化 | 拒绝 |
| Profile 与直接 SWE config override 同时使用 | 拒绝 |
| 任一 live/training/formal/release 授权为 true | Schema 拒绝 |
| v1 启用 TaskBoard、scaffold 或 frozen-prefix 声明 | Schema 拒绝 |
| Freeze 目标已存在或不在 `artifacts` 下 | 拒绝 |

## 迁移纪律

一旦被消费，本 v1 profile 即视为不可变。依赖身份变化必须发布新的 profile
版本和新的 frozen manifest；consumer 不得本地放宽或修补 hash。未来
frozen-prefix profile 必须使用独立 profile ID、schema、materializer、
manifest 与输出目录，同时以 SHA 绑定同一认证 canonical execution。现有
20-record scaffold fixture 不是大规模 producer，禁止仅靠 profile 开关将其
晋级。

## 验证

```powershell
py -3.10 -m pytest tests/test_distillation_profiles.py -q
py -3.10 -m ruff check `
  src/anchor_mvp/swebench/distillation_profile.py `
  scripts/data/run_distillation_profile.py `
  tests/test_distillation_profiles.py
py -3.10 -m py_compile `
  src/anchor_mvp/swebench/distillation_profile.py `
  scripts/data/run_distillation_profile.py
```

测试覆盖真实发布的 Draft 2020-12 schema、物理依赖认证、mandatory sidecar、
原子 create-once freeze、末端漂移、重复 key、symlink、授权晋级、CLI
metadata-only 输出、launcher override 冲突，以及未传 profile 时不变的 docs
入口。
