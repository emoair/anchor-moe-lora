# Gemma 3 聊天五专家生成评测 v1

这是 `gemma3_chat_five_expert_qonly_v1` 的训练后评测器。它在经过认证的
`eval_proxy` 上比较三组：

1. 关闭全部 adapter 的冻结 Q8 底模；
2. 与记录角色一致的正确 adapter；
3. 预先固定的错误路由 adapter。

五个角色固定为 `humor`、`serious`、`angry_style`、`tool_call`、
`review_audit`。错误路由映射是无不动点排列，不会把任何角色映射回自己。

仓库内 evaluator 配置已绑定 FINAL consumer implementation、consumer config、
manifest 和两个 partition。它有意让 training run ID 与 run-receipt SHA-256
保持 `pending`，因此干净检出默认 fail-closed；只有执行 `-Bind` 后生成的
run-local 不可变配置才能进入评测。

## 已完成的诊断结果

首轮绑定评测已经完成：

- 训练 run：`gemma3-chat-q1024-20260725-052403`
- 评测 run：`eval-chat-q1024-20260725-063309`
- 评测 receipt SHA-256：
  `ea3ca5e12d7af9bd3e40a0a81bd32dc74060cb051e319377f9e55b3b8c86956b`
- receipt sidecar 物理 SHA-256：
  `75a9d663aba21a9f77760d2937db3d854d2316ba508360b3eb8f7d9398879c0c`

本次共完成 75 个仅聚合观测：25 条不同的 eval-proxy 记录、5 个角色，
每条记录比较 3 个 arm。正确 adapter 的格式/代理通过率分别为：
`humor=0.8`、`serious=1.0`、`angry_style=1.0`；`tool_call` 与
`review_audit` 在三个 arm 中均为 `0.0`。宏平均 correct-minus-base 为
`+0.44`，correct-minus-wrong-route 为 `+0.08`。

这些结果提供了受控的风格路由信号，同时明确说明 1000 条诊断数据尚未教会
可靠的工具调用或审查对象输出。它们只是 proxy，不是 held-out 质量结论或
正式验证。

## 一键使用

若 Python 尚未加入 `PATH`，只需设置一次：

```powershell
$env:ANCHOR_TRAINING_PYTHON = "C:\path\to\python.exe"
```

查看门禁状态；此命令不会读取数据正文：

```powershell
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Status
```

训练完成后先一键生成不可变的完整绑定配置。此步骤会认证数据集、receipt 和
adapter，但不加载模型、不申请 GPU：

```powershell
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 `
  -Bind -TrainingRunId <training-run-id>
```

命令会输出新的 `bound_config.json` 路径。随后用它做 model-free preflight
并执行固定的小成本三组对照：

```powershell
$Bound = "runs\gemma3_chat_five_expert_qonly_generation_eval_v1\bindings\<training-run-id>\bound_config.json"
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Preflight -Config $Bound
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Execute -Config $Bound
```

仓库模板与每份生成的绑定配置都带强制 `sha256sum` 格式 sidecar。

结果原子发布到
`artifacts/diagnostics/gemma3_chat_five_expert_qonly_generation_eval_v1/<run-id>/`，
已有目录绝不覆盖。

## 会认证什么

以下身份必须同时成立：

- 最终 consumer 配置与实现；
- producer manifest 和两个物理分区；
- 完整训练 run receipt 及强制 sidecar；
- 十份 smoke/full 阶段 receipt 及 sidecar；
- 五个 PEFT adapter 各自的两个文件；
- 冻结本地 Gemma 3 模型文件契约。

只要有 pending、缺失、漂移或交叉绑定错误，就会在加载模型前关闭。生成前后
还会比较 Q8 SCB inventory 与本地模型文件。

## 仅聚合指标

receipt 不保存 prompt、target、生成回答、原始 token ID 或逐样本指标，只保存：

- humor、serious、angry_style 的自然语言格式有效率与参考三元字符相似度代理；
- 工具调用 JSON/闭合结构有效率与 evidence 字段一致率；
- 审查 JSON/闭合结构有效率与 dependency 字段一致率；
- 身份题中“由 Air 训练”的一致率及错误归属缺失率；
- EOS、重复 4-gram、生成速度和 Torch 峰值；
- 正确 adapter 相对错误路由、相对底模的差值。

风格指标明确只是 reference proxy，不冒充独立语义裁判。错误路由差值即使为正，
也只能作为受控诊断信号，不能直接证明路由因果、held-out 泛化或正式质量。

## 运行边界

底模按训练契约用 bitsandbytes INT8 加载，adapter 仅推理，贪心解码，生成使用
模型默认动态 KV。本评测不宣称 Q8 KV、完整生成 KV 精确共享、正式训练授权或
held-out 评测。
