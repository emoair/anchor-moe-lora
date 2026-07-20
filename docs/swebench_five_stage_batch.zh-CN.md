# SWE-bench / SWE-smith 五阶段题库与批量蒸馏

[English](swebench_five_stage_batch.md) | [简体中文](swebench_five_stage_batch.zh-CN.md)

这一入口把每一张已通过许可和分区门禁的 train 题卡拆成同一条依赖链：

1. `planner`：根据公开 issue 与基础工作区清单，决定交给哪个 builder/reviewer、如何拆任务、拟使用哪些工具；
2. `tool_policy`：逐项审查 planner 的工具提案，给出 `APPROVE` 或 `DENY`，但模型判定本身不替代运行时策略；
3. `domain_builder`：在隔离沙箱中运行受控 OpenCode，保留真实 tool call/result、生成的 workspace diff 与去 oracle 化执行摘要；
4. `domain_review`：领域 reviewer 对生成 diff 做 `PASS` 或提出 `REVISE`，返修时继续产生同一题的下一 revision；
5. `security`：只在领域审查通过后，对最终生成 diff 做 `PASS`/`BLOCK` 最终门禁。

每层记录均显式带有同一组 `card_id`、`instance_id`、`alignment_id`、
`source_fingerprint`，以及直接上游的 `record_id`。因此一题的五份训练目标不会在并发和断点恢复时串题。

## 数据边界

批量入口只接受 `anchor-swebench import` 写出的规范题卡和与该文件 SHA-256 绑定的 import
manifest。manifest 必须证明：

- 数据来自 `train`；
- Full dev/test、Lite、Verified 已登记为永久 held-out；
- 仓库许可逐项审核，未知仓库 fail closed；
- 题卡文件内容和数量未在导入后改变。

上游 `patch`、`test_patch`、`hints_text`、`FAIL_TO_PASS`、`PASS_TO_PASS`、隐藏测试名、
gold solution 和 oracle label 不得进入教师 prompt、replay response 或执行 bundle。唯一允许
出现 `patch` 这个键的位置，是受控 OpenCode candidate 的 `final_diff[].patch`；它表示 agent
本次刚生成的工作区 diff，并会在训练记录中立即投影成 `diff`，不是 benchmark 标准答案。

## 先只编译题库

以下命令会验证题卡，并在 stdout 输出 content-free 计数；默认不调用 API、不读取环境变量、不写文件：

```powershell
$env:PYTHONPATH = "src"
py -m anchor_mvp.swebench compile `
  --cards-jsonl artifacts\swebench\cards.train.jsonl `
  --import-manifest artifacts\swebench\import-manifest.train.json
```

确认后显式加 `--write` 才会写出每题五个 dependency-bound work order：

```powershell
py -m anchor_mvp.swebench compile `
  --cards-jsonl artifacts\swebench\cards.train.jsonl `
  --import-manifest artifacts\swebench\import-manifest.train.json `
  --work-orders-output artifacts\swebench\work-orders.jsonl `
  --write
```

也可以使用[示例配置](../configs/data/swebench_five_stage.example.yaml)：

```powershell
py -m anchor_mvp.swebench batch `
  --config configs\data\swebench_five_stage.example.yaml
```

示例默认 `mode: dry-run`，仍然不会写文件或访问网络。

## Replay 与 live

`mode: replay` 用离线 `record_id -> response` JSONL 跑完整闭环，适合 CI、schema 演进和无额度
调试。每行格式是：

```json
{"schema_version":"anchor.swebench-replay-response.v1","record_id":"swe-stage-v1:<sha256>","response":{}}
```

`mode: live` 复用项目现有 `provider_spec`、模型选择和 `CompatibleTeacher`。它必须同时满足配置
中的 `mode: live` 和命令行 `--allow-live`；配置只允许写 `api_key_env` 的变量名，禁止写 key
值。默认模式绝不会读取该变量。

两种运行模式都按完成顺序 `fsync` 追加：

- `requests.jsonl`：实际送给教师的公开请求；
- `stage_records.jsonl`：五阶段输入、输出、身份和依赖；
- `chains.jsonl`：经严格 trajectory adapter 二次校验的完整链；
- `manifest.json`：计数与文件哈希。

## 真实工具执行边界

当前 runner 不伪造 builder，也不自行把 benchmark 标准答案变成“工具轨迹”。它消费
`anchor.swebench-execution-bundle.v1`：其中必须是隔离沙箱内真实执行后，经现有
`convert_controlled_session` 导出的 `anchor.session-training-candidate.v1`、基础 workspace
inventory，以及保存在模型不可写位置的 trusted sandbox audit bundle。

因此，题库拆解、教师分层、断点持久化和最终验收已经闭环；仓库 checkout、沙箱生命周期和
OpenCode 进程调度仍由现有 tooling runner/control plane 负责。只有拿到真实 candidate 与审计
侧车后，SWE 批量 runner 才会写 builder/review/security 完整链。
