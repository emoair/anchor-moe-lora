# SWE-bench 完整 train 题库

本文说明可复现、仅使用 train 的完整题库。配对英文版见
[swebench_full_bank.md](swebench_full_bank.md)。

## 固定范围

- 数据集：`SWE-bench/SWE-bench`，不可变 revision
  `7074ef12ea2a6f70a228943c1336553333c22786`，只使用 `train`。
- 源 parquet：106,492,326 bytes；SHA-256 为
  `0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69`。
- 源规模：35 个仓库、19,008 道题。
- 从 train 内确定性划分：17,105 条训练、1,903 条验证。验证行仍来自官方
  train split，不是公开 benchmark 的非训练分区。
- 确定性语言分配：9,504 条 `en-US`、9,504 条 `zh-CN`。分配不代表已经翻译。
- 每题五个有序工单，共 95,040 个：
  `planner -> tool_policy -> domain_builder -> domain_review -> security`。

原始 parquet 只保存在 `artifacts/swebench/source/`，不会进入公开导出。

2026-07-18 的本地离线核验结果为 `launch_ready=true`、
`publication_ready=true`。`training_ready=false` 是如实的运行前状态：只缺
`gold_manifest`、`zh_cn_localization_manifest` 和
`real_tool_results_manifest`，构建器不会伪造它们。

## 1. 下载固定源文件

除非显式加入 `-ConfirmDownload`，脚本只做 dry run。它只请求固定的 train
parquet。

```powershell
cd D:\LLM\anchor-moe-lora
.\scripts\data\download_swebench_train.ps1
```

物理网卡直连不是仅凭一个 IP 字符串就成立。脚本会用 interface index 把本地 IPv4
映射到 `Get-NetAdapter -Physical` 返回且状态为 `Up` 的网卡。虚拟、TUN、TAP、
VPN、已断开或未知网卡都会在执行 `curl` 前被拒绝。

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  ForEach-Object {
    $ip = $_
    Get-NetAdapter -Physical |
      Where-Object { $_.ifIndex -eq $ip.InterfaceIndex -and $_.Status -eq 'Up' } |
      Select-Object @{n='IPAddress';e={$ip.IPAddress}},Name,ifIndex,Status
  }

.\scripts\data\download_swebench_train.ps1 `
  -DirectPhysicalRoute `
  -SourceAddress 192.168.1.23 `
  -ConfirmDownload
```

示例地址必须替换成第一条命令实际打印的地址。

## 2. 重启后的离线预检

使用项目 Python 环境。预检只读：它只哈希本地文件并校验 JSON 元数据，不发送
Provider 请求，不启动沙箱，不读取密钥，也不需要 GPU。

```powershell
$py = 'C:\Users\Air\.conda\envs\anchor-mvp\python.exe'
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml `
  --preflight
```

需要给自动化返回非零失败码时，加入 `--require-launch-ready`：

```powershell
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml `
  --preflight --require-launch-ready
```

就绪状态拆成三组，彼此独立：

1. `launch_ready` 校验固定源文件的哈希、行数、train-only 范围，所有正式路由均为
   精确的 `reasoning_effort: max`，双目标魔改 OpenCode bundle 证明，以及 ready 的
   CC Switch route 证明。它**不要求**即将启动的运行自己才会产生的输出。
2. `training_ready` 校验运行完成后的 Gold、`zh-CN` 本地化和真实工具结果 manifest。
   它们是运行后产物，绝不是启动输入。
3. `publication_ready` 校验专用公开导出、MIT 来源归因、精确文件清单、结构字段与
   密钥扫描、train-only 绑定，以及严格的单文件大小上限。

这样就消除了“沙箱尚未启动，却要求沙箱结果已经存在”的循环门禁。三种状态可以独立
为真或为假。

## 3. 构建本地 staging 与公开投影

```powershell
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml
```

公开导出完成审计后，可执行
`& $py .\scripts\data\build_swebench_full_bank.py --config
.\configs\data\swebench_full_bank.formal.yaml
--refresh-hash-only-from-public`，在不解析 payload JSONL 或源 parquet 的前提下刷新
无正文 inventory 快照。

本地 staging 写入 `artifacts/swebench/full-bank-v1/`；审计后的公开投影写入
`datasets/public/swebench-full-bank-v1/`。命令只打印计数、路径和门禁状态，不打印
题目正文。

公开目录只包含：

- `source-metadata.train*.jsonl`，每行字段严格为 `repo`、`instance_id`、
  `base_commit`、`problem_statement`；
- `candidate-tasks/tasks*.jsonl` 与
  `candidate-work-orders/work-orders*.jsonl`；
- `allowlists/train.json` 与 `allowlists/validation-from-train.json`；
- `ATTRIBUTION.md` 与 `manifest.json`。

导出器在数据进入 Python 对象之前就做 parquet 列投影。答案改动、评测改动、提示、
评测标签、非训练 benchmark 记录和密钥都不是公开字段。每个公开文件必须严格小于
52,428,800 bytes；原始 parquet 明确排除。
公开 issue 正文中高置信度、形似密钥的字符串会被替换为 `[REDACTED_CREDENTIAL]`，
manifest 会记录替换次数。目录中出现未知文件会让
`publication_ready=false`；`.gitignore` 的允许列表与上述目录和文件族完全一致。

## 4. 训练产物 manifest 合同

每个运行后 manifest 都必须绑定数据集 ID、不可变 revision、`split: train` 和源
parquet SHA-256；必须声明 `complete: true`、`train_only: true`、
`contains_heldout: false`、正整数 `record_count`，并提供非空文件清单。清单中的哈希
和记录数都必须通过校验。三种 schema 为：

- `anchor.swebench-gold-manifest.v1`，并带 `gold_records: true`；
- `anchor.swebench-zh-cn-localization-manifest.v1`，并带 `locale: zh-CN`；
- `anchor.swebench-real-tool-results-manifest.v1`，并带
  `real_tool_results: true`。

候选题卡与工单是启动输入，不是训练 Gold。系统既不请求也不伪造隐藏思维链。

## 5. 归因与公开边界

`ATTRIBUTION.md` 在固定 revision 上标明 SWE-bench 上游仓库及其 MIT 软件许可证
（`SPDX-License-Identifier: MIT`）。它不会把 issue 正文或 35 个源仓库重新许可为
MIT；各自适用条款仍然有效。运行 Gold、本地化正文、工具结果、模型权重、日志、原始
parquet 和密钥继续位于被忽略的本地路径，不会复制到公开数据目录。

## 常见故障

- `launch_ready=false`：只检查 `gates.launch.missing` 和
  `gates.launch.invalid`；训练产物不能阻断启动。
- `route_not_ready`：重新构建并校验 CC Switch 组件证明；manifest 仅仅存在不代表
  ready。
- `training_ready=false`：检查三个运行后 manifest 及其绑定的 payload 哈希。
- `publication_ready=false`：查看不含正文的发布审计代码，移除未知文件或重建确定性
  导出。
- `parquet_metadata_unreadable`：在所选 Python 环境安装项目 training extra 和
  `pyarrow`。
- `file_size_limit`：调低 `publication.shard_rows`；禁止把上限提高到
  52,428,800 bytes 以上。
