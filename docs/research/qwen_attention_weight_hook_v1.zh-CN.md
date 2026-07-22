# Qwen Q+O 注意力权重 Hook v1

## 目的

这个独立诊断只回答一个窄问题：对于当前受控的 Q+O 代理适配器，分别观察
Q 与 O LoRA 分量时，选定层的注意力矩阵如何变化。它不会训练、修改、合并
或重新发布任何适配器。

工具在 eager attention 的第 0、13、27 个解码层安装 forward hook，并让一条
内联合成探针依次经过四种可逆 PEFT 视图：

1. `adapter_off`
2. `q_only_component`（临时把 `o_proj` LoRA scaling 置零）
3. `o_only_component`（临时把 `q_proj` LoRA scaling 置零）
4. `full`

每层输出一张五联图：四种绝对注意力矩阵，以及
`full - q_only_component` 差分。坐标轴只显示 query/key token 位置与
prompt/target 边界；不会输出 prompt 正文、target 正文或原始 token ID。

## 输出契约

输出目录严格只有以下文件：

- `summary.json`
- 强制校验旁车 `summary.json.sha256`
- `layer_00_attention.png`
- `layer_13_attention.png`
- `layer_27_attention.png`

摘要会认证每个按 head 取均值后的 float32 注意力矩阵，记录 target query
对 prompt 的注意力质量、熵，以及各模式差分范数。发布采用原子目录替换，
若目标已存在则拒绝覆盖。

## 解释边界

注意力权重只是诊断代理，不是解释，更不是因果归因。图上出现差异不能证明
路由、泛化、记忆或任务质量更好。这也只是一条探针的受控诊断，不是正式评测；
相关声明在机器可读摘要中全部保持为 false。

分量视图通过临时改变 PEFT `scaling` 映射实现，并在 `finally` 中恢复每一个
原始值。加载模型前会认证基座与适配器文件，磁盘权重不会被改动。

## 本地复现实测

已在 RTX 3080 Ti 上使用冻结的 Qwen2.5-1.5B 快照和 step-80 Q+O adapter
完成实测。单条 probe 共 79 tokens（prompt 44、target 35）；四个模式连同
模型加载与 PNG 渲染约耗时 39 秒。

第 0 层严格符合注意力方程的因果边界：`o_only_component == adapter_off`，
且 `full == q_only_component`。原因是 `O_proj` 位于本层 softmax 之后，无法
反过来改变本层已经算出的注意力权重。O 的影响会经残差流传给后层：
`full - q_only_component` 在第 13 层的平均/最大绝对差为
`0.000826 / 0.0615`，第 27 层为 `0.001908 / 0.1310`。第 27 层中，target
query 对 prompt 区域的注意力质量由 Q-only 的 `0.7467` 变为 full 的
`0.7253`。

这些结果只证明 O 会间接改变后层路由，不单独证明记忆。认证后的机器可读摘要见
[`results/qwen_attention_weight_hook_qpluso_v1_summary.json`](results/qwen_attention_weight_hook_qpluso_v1_summary.json)。

![第 0 层注意力五联图](figures/qwen_attention_weight_hook_qpluso_v1/layer_00_attention.png)

![第 13 层注意力五联图](figures/qwen_attention_weight_hook_qpluso_v1/layer_13_attention.png)

![第 27 层注意力五联图](figures/qwen_attention_weight_hook_qpluso_v1/layer_27_attention.png)

使用冻结本地模型复现：

```powershell
$env:PYTHONPATH = "src"
python scripts/research/run_qwen_attention_weight_hook.py --execute
```

运行器强制本地文件、eager attention、BF16 权重、TF32 矩阵乘法、batch=1、
禁用 KV cache、单条合成探针、0 网络请求、0 held-out 读取、0 受保护正文读取。

配置文件：`configs/research/qwen_attention_weight_hook_v1.yaml`。
