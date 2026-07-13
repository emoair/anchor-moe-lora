# SWE-bench / SWE-smith 元数据题卡接入

这是一条独立、默认拒绝的训练候选题卡入口。它只读取已经放在本地的小型
JSON/JSONL 元数据，不下载 Hugging Face 数据集、不下载或启动 Docker 镜像，
也不改写现有蒸馏数据、训练数据或冻结的 held-out benchmark schema。

当前 MVP 的职责只有三项：

1. 固定上游数据集的不可变 revision，并用精确题目 allowlist 选择候选题；
2. 在生成题卡前执行永久 held-out 隔离和逐仓库许可台账检查；
3. 生成不含标准补丁、测试和提示的任务卡，并验证后续五阶段链的完整性。

本版本没有容器 runner。题卡里的 `execution_sandbox_required=true` 是后续执行器
必须满足的契约，不代表 importer 已经执行或验证了仓库代码。

## 来源与分区政策

优先训练来源是官方 SWE-smith 的 `train` split，包括适用的语言专用
`SWE-smith-*` 数据集。普通 `SWE-bench/SWE-bench` 的 `train` split 也可以作为
可选训练来源，但同样必须通过完整的 allowlist、仓库隔离和许可台账。

以下数据永久只作 held-out：

- 普通 SWE-bench 的 `dev` 和 `test`，其中 `test` 对应 Full benchmark；
- SWE-bench Lite 的全部 benchmark 实例；
- SWE-bench Verified 的全部 benchmark 实例。

本地 held-out registry 必须同时登记 Full、Lite、Verified，缺少任何一个都拒绝
导入。门禁依次检查精确 `instance_id`、内容/工作区 `source_fingerprint`，并默认
执行更严格的仓库级隔离。一个训练仓库只要也出现在 held-out registry 中，即使
题目 ID 和描述不同，也会被拒绝。

数据集 revision 必须是完整的 40 位十六进制 commit，不能使用 `main`、`latest`
或版本标签。allowlist 和许可台账必须声明完全相同的 dataset ID 和 revision。

## 输入契约

本地训练元数据 JSONL 每行至少包含：

```json
{
  "instance_id": "owner__repo-123",
  "repo": "owner/repo",
  "problem_statement": "Issue text supplied to the solving agent.",
  "base_commit": "0123456789abcdef0123456789abcdef01234567"
}
```

SWE-smith 风格记录可以用 `image_name` 代替 `base_commit`。两者都存在也允许。
记录可以附带 `dataset_id`、`dataset_revision`、`split`，但一旦附带就必须与命令
行固定值完全一致。

输入中即使存在以下字段，importer 也绝不会把它们复制进题卡：

`patch`、`test_patch`、`hints_text`、`FAIL_TO_PASS`、`PASS_TO_PASS`、
`tests`、`test_cases`、gold solution 和 oracle 字段。

因此，题卡只保留求解所需的问题、仓库定位信息、来源/许可哈希以及路由标签；
它不是标准答案载体。训练语料仍需由受控 agent 在独立沙箱中实际解决、构建、
测试和审核后产生。

### 精确训练 allowlist

```json
{
  "schema_version": "anchor.swebench-train-allowlist.v1",
  "dataset_id": "SWE-bench/SWE-smith",
  "dataset_revision": "<40-hex-commit>",
  "split": "train",
  "instance_ids": ["owner__repo-123"]
}
```

allowlist 里的每个 ID 必须在本地来源 JSONL 中恰好出现一次。来源文件中的其他
记录会在许可检查之前被跳过。

### 永久 held-out registry

```json
{
  "schema_version": "anchor.swebench-heldout-registry.v1",
  "sources": [
    {
      "dataset_id": "SWE-bench/SWE-bench",
      "dataset_revision": "<40-hex-commit>",
      "split": "test",
      "metadata_jsonl": "heldout-full.jsonl"
    },
    {
      "dataset_id": "SWE-bench/SWE-bench_Lite",
      "dataset_revision": "<40-hex-commit>",
      "split": "test",
      "metadata_jsonl": "heldout-lite.jsonl"
    },
    {
      "dataset_id": "SWE-bench/SWE-bench_Verified",
      "dataset_revision": "<40-hex-commit>",
      "split": "test",
      "metadata_jsonl": "heldout-verified.jsonl"
    }
  ]
}
```

相对路径以 registry 文件所在目录为基准。每个 held-out JSONL 只需提供计算
ID、仓库和 fingerprint 的元数据，不需要补丁、测试或镜像。

### 仓库许可台账

```json
{
  "schema_version": "anchor.swebench-license-ledger.v1",
  "dataset_id": "SWE-bench/SWE-smith",
  "dataset_revision": "<40-hex-commit>",
  "repositories": {
    "owner/repo": {
      "spdx_id": "MIT",
      "license_file_sha256": "<64-hex-sha256>",
      "reviewed": true,
      "training_allowed": true,
      "metadata_redistribution_allowed": true,
      "attribution": "Project and copyright attribution"
    }
  }
}
```

上游数据集自己的许可证不自动替代仓库级审查。仓库缺项、布尔批准项不是严格的
`true`、归属为空、SPDX 或许可证哈希无效时，一律拒绝。题卡引用台账哈希和
许可证文件哈希，不复制 attribution 文本；发布数据时仍应从台账生成完整归属。

## 路由和完整链契约

每张卡都含有 `domain_id`、`language` 和 `task_kind`，供 planner 路由到领域
builder/reviewer。SWE 任务不会伪装成 frontend 任务。MVP 默认使用共享专家：

- `swe-shared-builder`
- `swe-shared-reviewer`

以后可以按语言或仓库类型换成领域专家，而不用改变题卡身份。`alignment_id` 只由
规范化的 `instance_id + repo` 生成，不随 split/revision 漂移；
`source_fingerprint` 则绑定规范化问题文本和 `base_commit`/`image_name` 工作区。

若传入 `--chain-index`，每张卡必须恰好有一行，且阶段必须按以下顺序完整出现：

1. `planner`
2. `tool_policy`
3. `domain_builder`
4. `domain_review`
5. `security`

chain index 同时记录 `planner_route` 和 `executed_route`，两者的 domain、builder
和 reviewer 必须相等，并与题卡路由契约相等。每行还必须用 SHA-256 绑定执行
沙箱审计记录。少一题、重复一题、少一个阶段或实际换了专家都会失败。

## 使用方法

先 dry-run。它完成全部读取和门禁，只把 content-free manifest 打到 stdout，
即使命令里传了输出路径也不会写文件：

```powershell
$env:PYTHONPATH = "src"
py -m anchor_mvp.swebench import `
  --source-jsonl fixtures\swe-smith-train.jsonl `
  --dataset-id SWE-bench/SWE-smith `
  --dataset-revision <40-hex-commit> `
  --train-allowlist configs\swe\train-allowlist.json `
  --heldout-registry configs\swe\heldout-registry.json `
  --license-ledger configs\swe\license-ledger.json `
  --dry-run
```

确认 manifest 后，非 dry-run 必须显式给出两个不同的输出：

```powershell
py -m anchor_mvp.swebench import `
  --source-jsonl fixtures\swe-smith-train.jsonl `
  --dataset-id SWE-bench/SWE-smith `
  --dataset-revision <40-hex-commit> `
  --train-allowlist configs\swe\train-allowlist.json `
  --heldout-registry configs\swe\heldout-registry.json `
  --license-ledger configs\swe\license-ledger.json `
  --cards-output artifacts\swe\cards.jsonl `
  --manifest-output artifacts\swe\manifest.json
```

manifest 只含策略、计数和集合/文件哈希，不含 instance ID、仓库名、问题文本、
commit、镜像名或单题 alignment。`content_emitted=false` 指的是 manifest 本身不
泄露这些内容；题卡 JSONL 会包含求解必需的问题和仓库元数据。

## 官方项目与致谢

本接入层借鉴并兼容以下官方开放项目的公开数据格式和研究定义，但没有复制其
benchmark 答案、容器镜像或 runner，也不暗示上游作者为本项目背书：

- [SWE-bench 官方仓库](https://github.com/SWE-bench/SWE-bench)、
  [官方数据卡](https://huggingface.co/datasets/SWE-bench/SWE-bench) 与
  [论文](https://arxiv.org/abs/2310.06770)。SWE-bench 代码仓库采用 MIT
  License；各实例涉及的第三方仓库许可证仍需独立审查。
- [SWE-smith 官方仓库](https://github.com/SWE-bench/SWE-smith)、
  [官方数据卡](https://huggingface.co/datasets/SWE-bench/SWE-smith) 与
  [论文](https://arxiv.org/abs/2504.21798)。感谢维护者公开大规模软件工程
  任务生成方法、元数据格式和研究结果。

任何后续实际引入的数据 snapshot 都必须另外记录精确 commit、文件 SHA-256、
当时适用的上游许可证和逐仓库归属；本页的链接不能代替具体 snapshot 审计。

题卡通过本页门禁后，如何接入受控 OpenCode 工具轨迹并生成五阶段训练合同，见
[SWE 五阶段 trajectory adapter](swebench_trajectory_adapter.md)。
