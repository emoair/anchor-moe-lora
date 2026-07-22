# Qwen 请求内触发回执 v2

这份回执补齐原始 Qwen 44-token request-2 探针缺失的元数据。它认证一次
完整 chat-template 序列化、一次完整 tokenizer 调用，以及触发文本对应的
覆盖 token 区间；产物不会公开原始 token ID，也不会建立全局 token 索引。

## 范围边界

- 仅 tokenizer、全程离线：不读取模型权重、GGUF、Gold、heldout 或 scaffold
  JSONL 正文，不请求 GPU、provider 或网络。
- consumer 依赖基线绑定：
  `b0441e6beaa07b180d7fc69e462b4d2babf21792`。它必须是当前 checkout 的祖先
  （或与之相等），而不是永久要求 `HEAD == baseline`；实际执行的物化器另由
  config/schema/implementation 的物理 SHA-256 身份认证。
- producer 基线绑定：
  `744e23f975b13923903f5fabe04c32e74ea25dc4`。
- Qwen tokenizer revision：
  `3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34`。
- tokenizer binding SHA-256：
  `a76b0f60e5c1e2d92b8a8d9131f9afe9edfda3fcbf0221c4234359f70e806425`。

它仍然只是诊断型 companion artifact，**不能**授权训练、正式评测、数值
等价、KV 共享或正式阈值声明；引用的 TF32 结果仍仅是
`proxy_signal_passed`。

## 已冻结探针事实

| 字段 | 值 |
| --- | --- |
| 完整 request-2 token 数 | 44 |
| 触发覆盖区间 | `[25, 33)` |
| 下标语义 | 从零开始，右端不含 |
| 覆盖宽度 | 8 tokens |
| 前置 overhang | 0 UTF-8 字节 / 0 codepoints |
| 后置 overhang | 1 UTF-8 字节 / 1 codepoint |
| 完整 request-2 UTF-8 SHA-256 | `ed6adfcbd0052fdda52a5ab8c52ed04d6e55c7f62493f0d326d4e1b29d55c9f3` |
| 有序 token-ID 摘要 | `d989d46116cd50f30d5bba1be48a366e2a04efb8c156550d0f11a532f19121e6` |
| 触发 token-ID 摘要 | `1d6889128be1b4b84ae22999ffe267a1cc862209b7c38ef3f932a5e69851a412` |

有序 ID 摘要算法为
`sha256_concat_signed_int64_big_endian_v1`：按原顺序把每个 token ID 编码成
8 字节有符号大端整数，直接拼接后计算 SHA-256。规范 preimage 不包含 JSON
语法或分隔符。

## 复现

使用已锁定的 `anchor-mvp` 环境和本地已认证 tokenizer。默认目录为
`D:\LLM\models\qwen2.5-1.5b-instruct-hf`。

```powershell
$env:PYTHONPATH = "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
conda run -n anchor-mvp python `
  scripts\research\materialize_qwen_request_local_trigger_receipt_v2.py `
  --output runs\qwen_request_local_trigger_receipt_v2_rebuild\receipt.json
```

物化器拒绝覆盖既有输出目录。要验证仓库内现有产物而不替换它，请运行：

```powershell
$env:PYTHONPATH = "src"
conda run -n anchor-mvp python -m pytest -q `
  tests\test_qwen_request_local_trigger_receipt_v2.py
```

聚焦测试会在临时目录完成一次离线、逐字节一致的重建，并核验 mandatory
sidecar 格式：`<64 位小写十六进制>  receipt.json\n`。

## 产物位置

- 配置：`configs/research/qwen_request_local_trigger_receipt_v2.yaml`
- 配置 schema：
  `configs/research/qwen_request_local_trigger_receipt_v2_config.schema.json`
- 回执 schema：
  `configs/research/qwen_request_local_trigger_receipt_v2.schema.json`
- 物化器：
  `src/anchor_mvp/research/qwen_request_local_trigger_receipt_v2.py`
- 回执：`fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json`
- 强制 sidecar：
  `fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json.sha256`

下一步仍需 producer 发布并绑定 v2 companion schema/manifest，消费端前置状态
才能晋级。Formal-v3 release lock 与剩余 protected source inventory 是独立门槛。
