# Qwen controlled-proxy 风险证据消费器 v1

该消费器是一个只增不改、fail-closed 的诊断 overlay。它不修改冻结的
Qwen prerequisite-v2 消费器，也不能授权训练或正式发布。

## 它认证什么

消费器依次实际执行三项独立检查：

1. 现有 Qwen prerequisite-v2 消费器，其结果必须继续为 `blocked`；
2. 冻结的 Producer controlled-proxy follow-up auditor；
3. 冻结的 Producer Q+O risk-evidence companion auditor。

每个本地 metadata 文件只打开一次；同一份 bytes 同时用于解析与哈希，
结束前再做 identity、hash 和 bytes 重验。强制 sidecar 必须是带 LF 结尾的
标准 `sha256sum` 格式。Producer release 的 commit、parent、tree、文件模式、
blob OID、Git blob bytes 和本地 bytes 必须完全一致。

Git replace refs、grafts、lazy fetch、外部 `GIT_*` 覆盖、符号链接、junction、
路径越界、重复 YAML/JSON key 以及疑似正文键都会 fail-closed。

该 overlay 不读取 Gold、heldout、scaffold、dataset、模型或 adapter 正文，
不发起 provider、网络、模型或 GPU 请求。

## 解释边界

`o_branch_retained` 只表示：在一个**联合训练的 Q+O checkpoint** 中关闭 Q
分支后保留下来的 O 分支贡献。它不是独立训练的 O-only 对照组。

当前 retained fraction 是 step 80、单 seed、短上下文 proxy 上的事后分支
消融，不能证明分支可加、记忆、因果、统计显著性、广泛 bundle 泛化，
也不能宣布正式赢家。

真正的等预算、独立训练 O-only 对照必须拥有单独的训练 receipt 和新的
版本化证据契约。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.qwen_controlled_proxy_risk_evidence_consumer `
  --config configs/research/qwen_controlled_proxy_risk_evidence_consumer_v1.yaml
```

认证成功时会打印不含正文的 decision：

- `status=blocked`；
- `evidence_status=authenticated_non_authorizing_diagnostic`；
- `training_authorized=false`；
- `formal_training_authorized=false`；
- 进程退出码为 `2`。

任何缺失或漂移都会 fail-closed。即使本 overlay 通过，formal-v3 仍为
`0/5`，protected inventory 仍为 `2/6`。
