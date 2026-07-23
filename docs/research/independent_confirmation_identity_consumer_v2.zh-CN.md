# 独立确认身份 consumer v2

这个 additive consumer 会认证并独立复核 Producer commit
`09a6829084f76790e3488cb999a6755cd4d5f95e` 冻结的元数据产物。它不会替换或
修改任何 Producer v1 契约，也不具备训练授权能力。

## 哪一层变成 ready

只有元数据身份层变成 ready：

- 从 consumer 外部按物理 SHA-256 固定 Producer 的全部 19 个文件，并与
  Producer commit `09a6829...` 中的本地 Git blob 逐字节比对；
- 强制要求 `manifest.json.sha256` 存在，并且字节必须精确等于
  `<manifest-sha256>  manifest.json\n`；
- 用 Draft 2020-12 校验五份 JSON Schema，只允许 schema 内部 fragment ref；
- 哈希、解析、计数和 schema 校验共用同一次 bytes snapshot，覆盖 321 条 JSONL
  记录与两份 proof；
- 重新构造 241 项无 namespace 盐值的 descriptor atom 闭集；
- 独立重算 task、template、pair、task-bundle、inventory、membership、交集、
  配额和 matched-factor group，不复制 proof 的自报结果；
- 结束前再次按 bytes、SHA-256 和文件身份检查所有 snapshot，关闭 parse/hash
  之间的 TOCTOU 窗口。

真正的 independent 轨共有 60 个元数据 bundle（40 train、20 eval-proxy），
其中 task/template/pair inventory 是 `60/20/60`，相对 discovery 的交集是
`0/0/0`。

独立保留的 controlled-factorial 轨也有 60 个元数据 bundle 和 20 个完整三因子
分组；它相对 discovery 的交集是 `5/1/0`，train 配额是 `13/14/13`，
eval-proxy 配额是 `7/6/7`。这条轨道明确不能冒充 independent confirmation。

## 哪些仍然 blocked

元数据 ready 不代表训练记录已经生成，更不代表实验已经执行。因此 decision
把以下字段固定为 `false`：

- `records_materialized`
- `protected_source_disjoint`
- `independent_confirmation_executed`
- `controlled_factorial_executed`
- `quality_validated` 与 `generalization_validated`
- `training_authorized`、`formal_training_authorized` 与 `formal`

既有门禁也完全不变：protected inventory 仍为 `2/6`，formal-v3 仍为 `0/5`，
execution lease 不存在。即使元数据审计成功，CLI 仍返回退出码 2。

## 运行方式

在仓库根目录、项目环境中运行：

```powershell
$env:PYTHONPATH = "src"
python `
  -m anchor_mvp.research.independent_confirmation_identity_consumer `
  --config configs/research/independent_confirmation_identity_consumer_v2.yaml
```

预期语义结果：

```text
status=metadata_identity_ready_execution_blocked
metadata_identity_ready=true
process exit code=2
```

该过程不会加载模型、GPU，不会发 provider/network 请求，不会读取受保护数据集
正文，也不会启动训练。

## consumer 固定入口

- config：`configs/research/independent_confirmation_identity_consumer_v2.yaml`
  （`e69fa162dfc03e5c92168e1c529630a42b929027400870305829cad6877ea723`）
- blocked-only decision schema：
  `configs/research/independent_confirmation_identity_decision_v2.schema.json`
  （`84c9ebcbdf80b9ccb9e1a2f8c875642777bcef32b5ff6ead99a3059401ec609e`）
- implementation：
  `src/anchor_mvp/research/independent_confirmation_identity_consumer.py`
- tests：`tests/test_independent_confirmation_identity_consumer_v2.py`
