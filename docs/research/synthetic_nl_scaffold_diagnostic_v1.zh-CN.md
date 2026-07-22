# 合成自然语言脚手架诊断集 v1

## 用途

该产物是用于五角色脚手架接口的小型、确定性训练代理。它与 SWE-bench
Gold、partial Gold、held-out 题目以及旧脚手架 inventory 相互独立；生成器
不会读取这些来源的正文或路径。

它被严格限定为 **diagnostic only**：

- `formal=false`；
- `training_authorized=false`；
- `eval_proxy` 不是 held-out；
- 不声称已经得到 source-disjoint 或零交集证明；
- 不声称物理 KV 复用、数值等价或质量提升。

## 数据规模

闭合语法定义 10 个纯合成 source bundle，其中英文 5 个、简体中文 5 个。
每个 bundle 展开为五个角色和两种配对输出：

```text
10 bundles × 5 roles × 2 variants = 100 records
```

五角色为 `planner`、`tool_policy`、`frontend_gen`、`frontend_review` 和
`security_gate`。两种输出为 `json_only` 与
`concise_rationale_plus_json`。简短 rationale 是可审计的决策摘要，不是
隐藏思维链；配对的两条记录拥有完全相同的 canonical routing JSON。

数据先按 source bundle 切分，再展开角色或做任何增强。8 个 bundle 生成
80 条 train，2 个 bundle 生成 20 条 `eval_proxy`。角色、语言与 variant
严格均衡，同一 bundle 绝不会跨 split。

## 紧凑因果视图

真实训练 materializer 不会把整块 task board stringify。每个角色只能看到：

- 合成任务与三条约束；
- 当前角色指令；
- 已提交前序阶段的短摘要和本地 `S0`–`S4` 引用。

前序阶段的完整 target 不进入 prompt；当前和未来阶段的 segment ID、引用与
target 正文全部禁止。记录中的 hash 会把短摘要绑定回对应的完整合成 target。

数据集本身不绑定 tokenizer，也不发布 token 长度。后续 run preflight 必须
绑定 tokenizer，对完整 chat-template prompt + target 做一次无截断计数。本地
曾用 Qwen 2.5 1.5B tokenizer 做非持久化检查，当前 100 条均小于 1024 token；
这不是正式数据集声明。

## 对照标签

每条记录都同时允许以下三个实验标签：

- `q_only`；
- `q_plus_o`；
- `wide_lora`。

数据 producer 不指定 target modules，也不选择赢家。后续 diagnostic run
manifest 必须绑定真实 module map，并保证三组使用完全相同的 record inventory、
split 与 seed。

## 构建与审计

在仓库根目录和项目 Python 环境中执行：

```powershell
$env:PYTHONPATH = "src"

python scripts/research/build_synthetic_nl_scaffold_diagnostic_v1.py build `
  --repo-root . `
  --config configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_nl_scaffold_diagnostic_v1

python scripts/research/build_synthetic_nl_scaffold_diagnostic_v1.py audit `
  --repo-root . `
  --config configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml `
  --artifact fixtures/research/synthetic_nl_scaffold_diagnostic_v1

python -m pytest -q tests/test_synthetic_nl_scaffold_diagnostic_v1.py
```

构建器拒绝覆盖已有目录。它先在同一父目录构建临时树，完成全量审计、输入
snapshot 复验后再重命名发布。审计时每个文件只取一次 bytes snapshot，并用
同一份 bytes 完成 hash、解析和计数，最后再做 TOCTOU 复验。
`manifest.json.sha256` 必须存在，格式严格为
`<64hex>  manifest.json\n`。

## Read set 与发布纪律

声明的语义 read set 只包含本命名空间下的版本化 config、闭合语法、schema 与
生成器实现。provider、network、model、GPU、真实工具和 protected body 读取
计数全部为 0。

本工作不创建 tag 或 release，也不授权创建。没有后续明确指令和独立 formal
release lock 前，不得发布 tag 或 release。
