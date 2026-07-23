# Gemma 3 1B IT 五专家 Q-only 参数预算

这是一个仅使用元数据、仅用于诊断的可行性契约。它不会加载模型、读取权重
张量、使用 GPU、请求模型供应商，也不授权训练。

## 一键复现审计

在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"
python -m anchor_mvp.research.gemma3_qonly_parameter_budget --repo-root .
python -m pytest tests/test_gemma3_qonly_parameter_budget.py -q
```

第一条命令会校验 Draft 2020-12 Schema、强制
`sha256sum` 格式 sidecar、契约与 Schema 的物理哈希，并从单次字节快照重算
表中所有整数。审计通过后仍会明确输出 `training_authorized: false`。

## 到底在计算什么

本地导出元数据显示：26 个 Transformer 层、隐藏维 1152、4 个宽度为 256
的 Query 头、基座参数 999,885,952。因此每层 `q_proj` 是
`1152→1024`。秩为 \(r\) 的 Q-only LoRA 参数量为：

```text
单专家 = 26 × r × (1152 + 1024) = 56,576r
五专家总存储 = 282,880r
单次正确路由的活跃参数 = 56,576r
```

默认每个请求只激活一个私有专家。把五份 checkpoint 相加不代表每个 token
同时跑五个 LoRA，更不代表系统自动“变成等效 2B 模型”。

| rank | 单专家参数 | 五专家总存储参数 | 定位 |
|---:|---:|---:|---|
| 4 | 226,304 | 1,131,520 | 小型 MVP |
| 8 | 452,608 | 2,263,040 | 小型 MVP |
| 16 | 905,216 | 4,526,080 | 小型 MVP |
| 32 | 1,810,432 | 9,052,160 | 中等 |
| 64 | 3,620,864 | 18,104,320 | 中等 |
| 256 | 14,483,456 | 72,417,280 | 总存储压力组 |
| 512 | 28,966,912 | 144,834,560 | 总存储压力组 |
| 542 | 30,664,192 | 153,320,960 | 接近 dense-Q 存储分界 |
| 1024 | 57,933,824 | 289,669,120 | 达有效满秩，但因子存储冗余 |
| 3535 | 199,996,160 | 999,980,800 | 不适合做基座参数量对齐 |

26 层完整 dense-Q 增量每个专家只有 30,670,848 个参数。LoRA 因子在
rank 542.1176 以上就比直接存一份 dense-Q 更大，而有效秩最多只有 1024。
因此 rank 3535 并不是“很大但仍有意义的 LoRA”，只是用大量冗余因子让五份
原始参数计数恰好比基座多 94,848。契约明确把它标成
`infeasible_for_base_parameter_parity`。

## 内存估算口径

JSON 契约使用可复算的明确口径：

- BF16 adapter checkpoint：2 字节/参数；
- BF16 梯度：2 字节/参数；
- FP32 master copy：4 字节/参数；
- 两份 FP32 Adam moment：8 字节/参数；
- 合计 adapter 训练状态：16 字节/可训练参数。

“单路由串行”估算是：BF16 基座 + 一个活跃 adapter 的 16 字节训练状态 +
另外四个非活跃 BF16 checkpoint。“五专家 optimizer 常驻”则为五个专家都
保留训练状态。两者都不包含 activation、KV Cache、CUDA context、kernel、
分配器碎片、dataloader 和框架 workspace，所以它们是会计基线，不是显存
峰值承诺。TF32 只影响符合条件的 FP32 矩阵乘路径，不改变这些存储数字。

## 公平对照

预留的主对照组为：

1. 冻结基座；
2. 单体 monolithic LoRA；
3. 五专家正确路由；
4. 五专家错误路由；
5. 五专家随机路由。

O-only 和 Q+O 只保留为诊断 overlay。活跃预算对齐时，单体 LoRA 与一次
路由到的单专家参数相同；总存储预算对齐时，单体 LoRA 与五专家总存储相同，
但单体的全部参数都会活跃。这是两个不同实验，不能拿五倍总存储去对比单份
LoRA 后宣称“同预算胜出”。

建议 proxy 指标为：工具调用 Schema 合法率、搜索结果有据回答正确率、微型
代码测试通过率、路由准确率、错误路由性能差值。本契约不作质量结论。

## 为什么目前不能真实开训

本地导出足够做预算，但尚不是完整训练身份：

- 没有绑定准确的官方模型 ID 和源 revision；
- `EXPORT_MANIFEST.json` 明确写着 `chat_template_bound=false`；
- 没有官方 chat template 的 SHA-256；
- `config.json` 声明 BOS=1、EOS=2，但 `tokenizer_config.json` 却把
  ID 1 映射为 `<eos>`、ID 2 映射为 `<bos>`；
- tokenizer/基座兼容性和 runner preflight 都没有通过。

真实训练前必须冻结并交叉校验 config、tokenizer 资产、权重、模型 ID、
revision 和官方 chat template。
