# Qwen 训练前置 consumer v2

## 目标

Consumer v2 是一个不读取正文、低内存的合取门。它在不改写任一来源的前提下，
同时认证 frozen v1 前置 consumer 与独立 companion v2 叠加层：

```text
已认证的 frozen v1 AND 已认证的 companion v2
```

它只补齐 request-local trigger 的**诊断级**缺口，不代表数据集、模型或正式训练
发布已经就绪。

## 当前结论

| 输入 | 已认证状态 | 含义 |
|---|---|---|
| Frozen 训练前置 v1 | blocked | 正式产物仍缺失。 |
| Frozen toy 前置 v1 | `pending_request_local_materialization` | frozen 字节按设计保持不变。 |
| Companion v2 | `ready_diagnostic_only` | request-2 trigger receipt 已获独立认证。 |
| 受保护 source-ID inventory | 2/6 ready | 只有 `swebench_source` 与 `heldout` ready。 |
| Consumer v2 结果 | blocked | `training_authorized=false`，`formal_training_authorized=false`。 |

尚未可用的四类 inventory 为 `gold_partition`、`partial_gold_export`、
`legacy_heldout_cases` 和 `synthetic_scaffold`。Calibration 不得冒充 held-out。

有效逻辑刻意分为两层：

```text
metadata_conjunction_verified = verify(frozen_v1) AND verify(companion_v2)

training_authorized =
    metadata_conjunction_verified
    AND 六类 source inventory 全部 ready
    AND zero-intersection proof ready
    AND formal-v3 snapshot ready
    AND final projector ready
    AND generic execution contract ready
    AND source-disjoint manifest ready
    AND formal release lock ready
```

目前第一行可以通过，但第二个表达式仍为 false。绝不能把
`ready_diagnostic_only` 晋级解释为“可以训练”。

## v2 实际认证什么

Consumer v2 会绑定并复验：

- frozen 训练前置 v1 的 config 与 implementation；
- frozen toy 前置 v1 的 config 与 implementation；
- companion v2 的 config、schema、implementation、canonical manifest 与
  强制 SHA-256 sidecar；
- Producer companion release commit 与精确 Git blobs；
- 复制的 request-local trigger receipt 与其强制 sidecar；
- companion 的 2/6 inventory 状态与所有禁止晋级的 claim；
- Planner request-1 私有 KV 不得作为 Expert request-2 KV 复用。

未知、缺失、只覆盖一半的 CLI 参数或任何 hash 漂移都会 fail closed。Frozen v1
仍以 frozen 代码执行；companion v2 是第二个强制输入，不是对 v1 的修改或替代。

## 声明边界

认证后的 trigger receipt 只证明：trigger span 来自一次完整 chat-template 的
request-2 精确序列化，以及一次完整 tokenization。它维持两请求协议：先校验并
commit Planner scaffold，再把已提交 scaffold 重新编码为 Expert 输入。

本 gate **不**声明：

- 已授权训练或正式训练；
- 六源 inventory 已完整或已经证明零交集；
- 数值等价、质量达标或正式 threshold；
- 物理 KV 共享、zero-copy、多流执行或完整生成阶段 KV 复用；
- 发生过 provider、模型、GPU 或网络执行。

## Frozen companion 身份

| Artifact | SHA-256 |
|---|---|
| Companion config | `21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f` |
| Companion schema | `596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119` |
| Companion implementation | `dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e` |
| Companion manifest | `7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115` |
| Manifest sidecar 物理文件 | `f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095` |
| Request-local receipt | `ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e` |
| Receipt sidecar 物理文件 | `ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a` |

Producer companion release commit：
`2648129d599a5041100278cb04b12291ffd8a482`。

## 低内存验证

在仓库根目录与项目 Python 环境中执行。下列命令只读取 metadata 与已认证的
hash-ID inventory；provider request、network request、model load、GPU request
和受保护正文读取均为 0。

```powershell
$env:PYTHONPATH = "src"

python scripts/data/audit_qwen_toy_prerequisite_companion_v2.py `
  --repo-root . `
  --config configs/research/qwen_toy_prerequisite_companion_v2.json `
  --artifact fixtures/research/qwen_toy_prerequisite_companion_v2

python -m anchor_mvp.research.qwen_train_prerequisite_consumer_v2 `
  --config configs/research/qwen_train_prerequisite_consumer_v2.yaml

python -m pytest -q `
  tests/test_qwen_train_prerequisite_consumer_v2.py `
  tests/test_qwen_toy_prerequisite_companion_v2.py
```

Consumer 命令当前会打印机器可读的 blocked decision，并按设计返回退出码 `2`。
在 inventory 只有 2/6 的现状下，返回 `0` 反而意味着发生了错误解锁。

## 尚缺的 formal-v3 gate

正式训练必须继续阻塞，直至以下产物均被独立冻结和认证：

- 四类缺失的 body-free per-ID inventory，以及六源 namespaced
  zero-intersection proof；
- formal-v3 training snapshot；
- final TaskBoard projector manifest；
- generic execution contract；
- source-disjoint manifest；
- formal release lock；
- trainable-base snapshot 与 tokenizer/base compatibility attestation；
- 正式 train/calibration/held-out partition 身份及其余 release bindings。

本 gate 不启动真实训练、provider 蒸馏、大模型加载、物理 KV/CUDA 测试，也不做
正式质量或性能评测。

## 发布纪律

本 consumer 只用于研究预检，不创建 tag 或 release，也不授权创建。没有后续明确
指令且 formal release lock 未完全满足前，不得发布 tag 或 release。
