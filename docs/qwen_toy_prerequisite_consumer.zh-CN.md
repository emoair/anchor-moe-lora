# Qwen diagnostic toy 前置消费门

这个 companion consumer 验证 Producer commit
`744e23f975b13923903f5fabe04c32e74ea25dc4` 的 metadata-only toy
前置产物。它与既有 Qwen formal 前置门版本隔离，不修改旧 v1，也不会生成旧
`anchor.qwen-toy-source-disjoint-attestation.v1`。

## 使用

```powershell
anchor-qwen-toy-prerequisite `
  --config configs/research/qwen_toy_prerequisite_consumer_v1.yaml
```

该诊断有意限定为仓库内工具：请从源码或 editable checkout 运行，让复制的契约与
元数据 fixture 能在同一个信任根下共同认证。独立 wheel 不作为受支持的信任根。

正常的当前结果仍以退出码 `2` 返回 `status=blocked`。这不是程序错误，而是
已认证的研究状态：

- 6 类保护源中只有 SWE-bench source 与 heldout 的 ID inventory 可在不读
  正文的前提下认证，即 `2/6 ready`；
- Gold partition、partial Gold export、legacy heldout 与 synthetic scaffold
  为 `4/6 unavailable`，不能被当作空集；
- request-local trigger receipt 仍为
  `pending_request_local_materialization`；
- `zero_intersection_claimed=false`、`v1_attestation_emitted=false`、
  `training_authorized=false`、`formal_training_authorized=false`。

Consumer 只允许读取固定的 26 个 Schema、Manifest、sidecar 和哈希 ID 文件。
`toy/diagnostic.jsonl` 没有复制进本仓库，也不会被读取；Gold、heldout、
scaffold 正文同样不在读取白名单中。每个输入由单次 bytes snapshot 同时用于
解析与哈希，并在结束前重新验证文件身份，任何换档或 reparse path 都会
fail-closed。

下一步只有在其余四类获得 body-free、逐 ID 的正式 inventory，且完整
request 2 经绑定 tokenizer 生成 request-local token/span/overhang receipt 后，
才能继续生成 diagnostic-only 的零交集证明；它仍不会自动授权 formal 训练。
