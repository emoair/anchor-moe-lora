# Q/O LoRA 记忆风险谱审计

这个诊断只回答一个窄问题：在总参数量相同的 Q+O 适配器里，是否出现了少量、
很强的残差流写回方向，并且这些方向与训练目标中的高频 token embedding 对齐。
它**不能**证明适配器里存着可直接读出的答案、源码、漏洞利用代码或任意字节串。

## 范围与隔离

- 仅 CPU；不做模型前向，不请求 GPU、provider 或网络。
- 只读取 80 条纯合成 **train** 分区；不打开 eval-proxy、held-out、Gold 或任何
  protected body。
- 先认证 Qwen 基模、tokenizer、比较清单、Q+O 与 wide adapter 的物理身份。
- 只从 `model.embed_tokens.weight` 切片所需的少量行，不实例化 1.5B 模型。
- receipt 只写聚合指标与 SHA-256 inventory，不写样本文本或 token ID。

## 数学方法

LoRA 更新为 `ΔW = (α/r) B A`。runner 不展开 1536×1536 稠密矩阵，而是做
thin QR：

`B = Q_B R_B`，`Aᵀ = Q_A R_A`

然后计算 `torch.linalg.svdvals((α/r) R_B R_Aᵀ)`。它与 `ΔW` 的非零奇异值
完全相同。谱范数严格取 `svdvals.max()`，绝不走本机已知有问题的
`torch.linalg.matrix_norm(..., ord=2)`。

每层会报告 Frobenius/谱范数、stable rank、基于能量熵的 effective rank、
top-1/2/4 奇异值能量占比及跨层能量集中度。O 投影的 B-column 子空间还会与
四组归一化 token embedding 比较：

1. 128 个目标高频 token；
2. 128 个仅出现在 prompt 的控制 token；
3. 128 个目标低频 token；
4. 128 个确定性、未在训练文本出现的随机词表控制 token。

选择结果只保留哈希，不序列化 token ID。投影能量定义为
`||Q_Bᵀ e_token||²`。

## 运行

```powershell
conda run -n anchor-mvp python scripts/research/audit_qwen_qo_spectral_memory.py `
  --config configs/research/qwen_qo_spectral_memory_audit_v1.yaml
```

命令会原子创建：

`artifacts/diagnostics/qwen_qo_spectral_memory_audit_v1/receipt.json`

以及强制的 `receipt.json.sha256`；目标目录已存在时会 fail-closed。

## 解释边界

较高的 top-1 能量占比加上高频目标对齐，只能说明存在低维模板/写回捷径风险，
不是逐字记忆的因果证明。本合成数据只有路由 JSON，没有真实漏洞利用正文，因此
“记住 exploit 代码”明确未测试。后续因果审计应加入无害唯一 canary、改写后的
抽取提示、模板族隔离评测和多随机种子。

## 本地复现实测

Q+O adapter 的 O-proj top-1 奇异能量为 `82.3186%`，能量有效秩为
`2.1161 / 8`，O 的总 delta 能量是 Q 的 `1.76715x`。O-column 子空间对
训练目标高频 token 的投影是确定性随机词表控制组的 `1.95911x`，而
prompt-only 控制只有 `1.00688x`。等参数 Wide 组也出现相同方向：top-1
能量 `86.6574%`，目标/随机对齐比 `2.04342x`。

O 能量并非在少数层爆炸：能量最高的四层只占 `24.34%`，跨层有效数量为
`25.71 / 28`。这支持“跨层广泛、层内低维的模板写回捷径”，但仍不证明逐字
答案或 exploit 代码被存入权重。认证结果见
[`results/qwen_qo_spectral_memory_audit_v1_receipt.json`](results/qwen_qo_spectral_memory_audit_v1_receipt.json)。
