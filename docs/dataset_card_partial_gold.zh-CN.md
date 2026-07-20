# Partial Gold 数据集卡

[English](dataset_card_partial_gold.md) | [简体中文](dataset_card_partial_gold.zh-CN.md)

## 范围与用途

本次发布包含五个 Anchor 专家各自使用的阶段级合成 SFT 候选数据，分为两个相关包：

- 完整的已接收专家 Gold 导出，共 1,465 条；
- 确定性、等量的正式训练快照，每个专家 128 条，即 `128 x 5 = 640` 条。

两个包均明确标记为 `not_for_end_to_end_claim=true`。严格完整的五阶段链只有
85 条，未达到原门槛 256 条；源自动化最终状态为 `gate_blocked`，分区 manifest
也不是 `training_ready`。等量快照可用于文档规定的单专家 Partial Gold 探针，
不能据此声称完整路由系统已端到端跑通。

## 数据内容

已接收导出遵循 `anchor.per-expert-partial-gold-export.v1`：

| 阶段 | 条数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| `plan` | 384 | 6,502,043 | `c82d7806c161ee96942054c98e45c7068af2ebfa5437492c073c3f6526015932` |
| `tool_policy` | 384 | 6,557,791 | `599ebbd76b9937391bf17ddc2fe4f8086a79f9b832dc296a74605819528c54f3` |
| `frontend` | 346 | 12,602,666 | `c6f7c79756c064b7256ecb2e38bed495e6cd68f863e7073cf66afe8679fbeed2` |
| `review` | 203 | 9,509,163 | `d2c381957bad2661efd8965fa60d5b49164cb00228721cfcec178a1216292c87` |
| `security` | 148 | 4,328,526 | `dbef35c02e16cc4d1506653ec7b4c926c4161a4d8e269b06155196b0768a1cac` |

冻结快照遵循 `anchor.per-expert-partial-training-snapshot.v1`：

| 专家 | 条数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| `planner` | 128 | 2,134,537 | `d3c4245e900a6c6736c5cce3aa73c5c32052d86af475d0984e68ad3bec376673` |
| `tool_policy` | 128 | 2,173,743 | `b9682c416a68386a8e7dde138680c9b68d14c0e31b59e2cdb1e2881f679e2b2a` |
| `frontend_gen` | 128 | 4,634,188 | `77dda1691bdf10f3220222d6c59f72d0bdc6009096c32fede99318de96a8e45a` |
| `frontend_review` | 128 | 6,090,499 | `be1c66864801e96a0577397b85f70695115a984c26c4971a21bb12a98bb7ef11` |
| `security_gate` | 128 | 3,788,169 | `20eb4c7ef6d4fc72723401cccd39e0e503fb433323904a55c670c03b64d8928d` |

所有拟发布文件都小于 50 MiB。完整导出中最大的文件为 12.02 MiB，快照中最大的
文件为 5.81 MiB。运行日志、重试归档、隔离分区和 held-out 产物均不属于这两个包。

## 确定性选择与完整性

快照选择完全确定：对每个专家计算 `SHA256("20260711:" + record_id)`，升序排列后
取前 128 条；结果不依赖输入顺序或进程随机数状态。

- 源分区 manifest SHA-256：
  `4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7`
- 已接收导出 manifest SHA-256：
  `1b8e5b87957d7ec1e867813c95b8f7ab3bef55861e778b6ba9f197e6edf3f2ec`
- 快照 manifest SHA-256：
  `a0866e6afd7861d9ae827625db5f8a7b3273d4e629b79b09e56c5d0ce7599e28`
- 快照内容摘要：
  `2fe95635cfa441b7d5bed1262c307d37cef4f5592d56071e49150d7aa094acc7`

以下命令只读取 manifest 和文件哈希，不打印样本正文，同时检查 50 MiB 发布上限：

```powershell
$export = "data\automated_v3_shards\ark_max_retry2_offset300000_c10\training_exports\per_expert_partial_gold\4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7"
$snapshot = "artifacts\formal_partial_v1\dataset"

function Test-DatasetFiles([string]$dir, [string]$mapName) {
    $manifest = Get-Content -Raw (Join-Path $dir "manifest.json") | ConvertFrom-Json
    foreach ($entry in $manifest.$mapName.PSObject.Properties) {
        $binding = $entry.Value
        $path = Join-Path $dir $binding.path
        if ((Get-Item $path).Length -ne $binding.bytes) { throw "byte mismatch: $path" }
        if ((Get-Item $path).Length -ge 50MB) { throw "file is not below 50 MiB: $path" }
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $binding.sha256) { throw "SHA-256 mismatch: $path" }
    }
}

Test-DatasetFiles $export "gold_files"
Test-DatasetFiles $snapshot "files"
```

## 剔除项与局限

两个 manifest 均证明 `negative`、`reject`、`oracle_label_only` 和 `heldout` 已被
剔除。导出器还执行密钥模式和 held-out 来源检查，但自动检查不能保证已发现所有
隐私、版权或质量风险。

本次 Partial Gold 实验明确豁免了原覆盖率门槛：`review` 距离每阶段 256 条的目标
少 53 条，`security` 少 108 条。更重要的是，安全 Gold 中只有 3 条 `BLOCK`、
145 条 `PASS`，而 `BLOCK` 目标是 60 条。类别严重失衡，因此不得把本数据包装成
充分的安全训练集、已验证的安全分类器，也不得据此作自主安全决策。要声称安全
能力，必须另建平衡数据集并完成 held-out 评测。

## 复现快照并运行探针

在仓库根目录执行。冻结操作具备幂等性：若已存在完全匹配的快照，程序会校验它，
不会静默覆盖。

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python -m anchor_mvp.data.partial_snapshot `
  --export-dir "data\automated_v3_shards\ark_max_retry2_offset300000_c10\training_exports\per_expert_partial_gold\4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7" `
  --output-dir "artifacts\formal_partial_v1\dataset" `
  --per-expert 128 `
  --seed 20260711
```

使用带硬门禁的低内存启动器。`preflight` 只读；先执行单步 smoke，再执行两步
probe。以下命令不代表完成了正式训练或端到端验证。

```powershell
$python = (Get-Command python).Source
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm preflight -Python $python
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm smoke -Execute -Python $python
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm probe -Execute -Python $python
```

该配置保留 9 GiB CUDA 峰值硬限制；文档准入门槛为通用配置至少 12 GiB 可用主机
内存、仅 probe 路径至少 11 GiB。不要为了让高负载机器通过而降低这些门槛。

## 来源、致谢与条款边界

本仓库的题库设计、防泄漏边界或工具实现借鉴或集成了以下项目。列名不表示上游
背书；仅凭本次发布的两个 manifest，也不能断言某一条样本来自某个具名上游数据集。

- [SWE-bench](https://github.com/SWE-bench/SWE-bench) 及其
  [数据卡](https://huggingface.co/datasets/SWE-bench/SWE-bench)；
- [SWE-smith](https://github.com/SWE-bench/SWE-smith) 及其
  [数据卡](https://huggingface.co/datasets/SWE-bench/SWE-smith)；
- [OpenCode](https://github.com/anomalyco/opencode)，用于受控工具执行方案；
- [CC Switch](https://github.com/farion1231/cc-switch)，作为固定元数据和控制面设计
  参考，而非嵌入式路由器；
- 来自 [GitHub awesome-copilot](https://github.com/github/awesome-copilot) 与
  [Anthropic Skills](https://github.com/anthropics/skills) 的固定 SOP/Skill 输入；
  精确 commit、文件哈希和保留声明见
  [`configs/data/skill_sources.yaml`](../configs/data/skill_sources.yaml) 与
  [`THIRD_PARTY_SKILLS.md`](../THIRD_PARTY_SKILLS.md)。

必须区分各层条款。本项目代码和数据生产工具采用仓库的
[`AGPL-3.0-or-later`](../LICENSE)，但代码许可证不会自动重新许可教师输出、上游
数据集、benchmark 实例、第三方仓库、模型/API 输出或随附 Skill。OpenCode、
CC Switch 和各 Skill 均保留各自的上游声明。SWE 类实例还必须具备精确 snapshot
revision、逐仓库许可台账和归属记录，详见
[`swebench_metadata_import.md`](swebench_metadata_import.md)。本卡不会臆造或授予
一份独立的数据集许可证；发布者必须在再分发前于 release 元数据中写明已审核的
数据条款。
