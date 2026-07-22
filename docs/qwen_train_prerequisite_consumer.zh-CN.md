# Qwen 训练前置消费门

这个入口只认证 Producer 冻结的“为什么当前不能正式训练”状态，不读取
Gold、heldout、scaffold JSONL 正文，不加载 tokenizer、模型或 GPU，也不会
把本地机械诊断升级成正式 A–F 结论。

```powershell
anchor-qwen-prerequisites `
  --config configs/research/qwen_train_prerequisite_consumer_v1.yaml
```

认证成功时命令输出 `status=blocked`、`training_authorized=false`，并以退出码
2 表示“契约有效，但训练仍被阻止”。配置、四个 Schema、状态 manifest 和
严格 `manifest.json.sha256` sidecar 均绑定物理 SHA-256；JSON 解析和 Schema
验证使用同一份已认证字节，结束前再检查文件身份与哈希未漂移。路径必须位于
仓库内，且祖先不能是符号链接、junction 或其他 reparse point。

当前冻结身份：

- Producer commit：`a8efe5f55b72960b49bcb1ae3753b633afd14959`
- Consumer config：`4fdc8173baaa9f14d93a288b18f38691be62bb1fb8e646c579a06d9c78bc1a8a`
- Status Schema：`e8d09abc26effcedc642125b4d84185f0e5072a23f5611f068274bd963c4f577`
- Status manifest：`70c8f0a866c5fb41c4c3726638b55a66efab77f8b2ee31c27ad31ab55def67da`
- Tokenizer binding Schema：`5b2e7c2e8e6efc1c9b7251fde853631e65806aca0364d9bb092ee9a07d135b25`
- Toy attestation Schema：`7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea`
- Formal release-lock Schema：`119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa`

## 为什么仍然阻塞

- formal-v3 snapshot、final projector、generic execution、source-disjoint 和
  release lock 均不存在；
- 当前严格完整五段链为 85/256，Review 和 Security 仍不足 256；
- tokenizer 只是已认证候选来源，尚无绑定 manifest；token 位置只能在
  request 2 精确序列化后生成；
- 六类受保护数据只有物理文件哈希，没有统一、冻结且可重算的 source-ID、
  domain 与 namespace inventory；同时没有真实 toy generator、配置、closed
  grammar 或 attester。因此不能诚实生成 `ready` toy attestation。

后续必须由 Producer 先冻结 metadata-only inventory 与真实生成/审计工件，
再新增兼容契约；不能修改或放宽当前 v1，也不能用零哈希或测试占位符代替。
