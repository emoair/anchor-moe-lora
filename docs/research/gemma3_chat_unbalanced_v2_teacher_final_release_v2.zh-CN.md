# Teacher FINAL 集成控制器与外置发布 v2

## 目标

这套 additive v2 用来闭合 Gemma 3 Chat unbalanced-v2 蒸馏管线中的一个
明确生命周期缺口：

1. 单一进程持有唯一 `RuntimeSecretSlots`；
2. 串行执行 `exact1 -> bounded15 -> bulk_c30`，只有经过签名认证的
   provider rate-limit 才允许回退 `bulk_c16`；
3. 在运行时 HMAC 槽关闭前，认证 WAL、三个阶段 receipt、3,520 个终态
   job receipt 以及 Teacher 投影记录；
4. 原子创建不可变的 Teacher FINAL *candidate*；
5. 随后由不同 reviewer 对 candidate 做只读复算，并在独立目录中原子发布
   attestation、final manifest 和 v2 consumer binding。

冻结的 v1 控制器、配置和 schema 保持逐字节不变。v2 仅做加法，不修改
Producer FINAL 源数据字节。

## 为什么必须在同一 live 进程内完成 candidate

receipt HMAC 密钥只存在于控制器内存槽。新进程若要恢复它就必须落盘或重新
生成，这两种做法都会破坏认证语义，因此被明确禁止。candidate materialize
必须发生在 `RuntimeSecretSlots.close()` 之前；跨进程 HMAC resume 继续标记
为 unsupported。外部 reviewer 只能绑定已经由同进程认证过的 receipt
物理字节，不能宣称恢复或独立重算了 secret HMAC。

## Candidate 契约

finalizer 消费认证过的主训练投影，以及一份认证过的 router 混合文件：

- 五专家记录 3,440 条；
- router 混合文件实际解析 100 行，其中只输出 80 条 train；
- 另外 20 条 router `eval_proxy` 仅用于 split 过滤，进入 Teacher train
  shard 的数量为 0；
- 合计 accepted 3,520 条；
- identity 20 bundles / 100 records。

Producer 的 body-free identity-probe inventory 只绑定认证 digest，不读取其中
50 条 eval probe 正文（`identity_eval_probe_body_reads=0`）。这是一条严格限域
声明，不再把它扩张成“router 混合文件的 20 条 eval 行也未读取”。

每条输出只有十个 closed fields，交叉绑定口径为：

- `record_id_sha256 = sha256(UTF-8 source record ID)`；
- `source_content_sha256 = canonical_sha256(source messages)`；
- 认证过的 source serialization identity；
- `task_bundle_sha256`；
- 原始 Producer asset；
- 已验证 Teacher target 及其 UTF-8 SHA-256。

candidate 包含有序且不跨 bundle 切分的 shards、mandatory SHA-256
sidecars、closed manifest、output inventory、同进程 build receipt 和
release request。所有 payload 都小于 50 MiB。其状态始终是：

- `candidate=true`；
- `final=false`；
- `training_authorized=false`；
- `formal_training_authorized=false`；
- `live_authorized=false`。

consumer 不得直接消费 candidate。

## 外置发布契约

外部 v2 reviewer 用单次 bytes snapshot 与终端 TOCTOU 重验，逐项复算：

- candidate 全部物理文件、sidecar 和 physical-tree digest；
- 全部 record 对 frozen Draft 2020-12 record schema 的实例验证；
- record order、target、source join、shard 和 logical dataset digest；
- role counts 与 bundle boundary；
- 按真实 projection domain 从 Teacher 物理 rows 重算 identity：先从 humor、
  serious、angry-style、tool-call 四个角色中找出恰好 80 条包含 Air 句的
  anchors，再组成 20 bundles，且每 bundle 四类 anchor 各一条；
- 对这 20 个 bundle ID 做全体 rows 闭包：必须恰为 100 rows，每 bundle
  除四个 anchor 外再恰有一条 structured `review_audit`。review target
  不要求重复 Air 句；
- Producer P/R 父子关系、release tree 与 metadata-only physical-tree digest；
- integrated controller、base controller、batch、finalizer config、
  finalizer implementation 和各 schema 身份。

随后在自有 staging 目录写入六个文件：

- `independent_release_attestation.json` 及 sidecar；
- `final_manifest.json` 及 sidecar；
- `teacher_alignment_binding.v2.json` 及 sidecar。

六个文件通过一次 OS 级 no-replace 目录 rename 同时发布。目标目录已存在就
fail closed。失败的 staging 保留用于取证，禁止递归清理。

身份图有意设计为无环：attestation 先生成；final manifest 绑定 attestation，
但只声明 binding schema/path，不写 binding hash；最后由 terminal binding
绑定前两个 payload。没有任何文档声称自己的 hash。

外部 release 只表示“数据 artifact 已完成独立发布审查”。consumer
acceptance 仍为 pending，因此 training、formal training、live、quality 和
generalization 全部继续为 false。

外部 reviewer 不会声称仅从 structured review target 再次证明身份事实。
review 的身份正确性绑定同进程 finalizer 的 closed `identity_audit`、candidate
manifest、build-receipt digest，以及认证过的 body-free Producer
identity-probe inventory。

## 文件与版本

- 集成控制器：
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_integrated_controller_v2.py`
- 同进程 finalizer：
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_finalizer_v1.py`
- 外部 reviewer：
  `src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_v2.py`
- CLI wrapper：
  `scripts/data/review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py`
- candidate record/manifest/receipt schemas：
  `configs/data/gemma3_chat_unbalanced_v2_teacher_final_*_v2.schema.json`
- 外部 release schemas：
  `configs/data/gemma3_chat_unbalanced_v2_teacher_final_release_*_v2.schema.json`
  与
  `configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json`

本契约涉及的 JSON、Python、测试与文档都必须由 `.gitattributes` 强制 LF。
最终交接只能报告格式化和测试之后的物理 SHA-256；中间 hash 不得作为 release
identity。

## 复现命令

控制器 model-free validation：

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  -m anchor_mvp.data.gemma3_chat_unbalanced_v2_integrated_controller_v2 `
  --validate-only
```

独立只读 candidate validation：

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  scripts\data\review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py `
  --candidate data\gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1\candidate-<run-id> `
  --implementer-id producer.implementer `
  --reviewer-id independent.reviewer `
  --validate-only
```

独立原子发布：

```powershell
$env:PYTHONPATH = "src"
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe `
  scripts\data\review_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py `
  --candidate data\gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1\candidate-<run-id> `
  --implementer-id producer.implementer `
  --reviewer-id independent.reviewer `
  --publish
```

聚焦验证：

```powershell
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe -m pytest -q `
  tests\test_gemma3_chat_unbalanced_v2_integrated_controller_v2.py `
  tests\test_gemma3_chat_unbalanced_v2_teacher_finalizer_v1.py `
  tests\test_gemma3_chat_unbalanced_v2_teacher_final_release_v2.py
```

这些命令不加载模型/GPU，不发 provider 或网络请求。真实 live run 必须继续等待
独立 consumer gates 与内存 credential/HMAC slots 全部变绿。

## 兼容性与非目标

finalizer 使用 `datetime`、`timezone` 和 `timezone.utc`，不导入
`datetime.UTC`，因此可在 Python 3.10 解析与导入。integrated loader 用
digest-qualified 临时 module name 执行认证后的 finalizer bytes，只在 exec
期间注册 `sys.modules`；执行后验证 `__file__` 与 raw digest，并在成功或失败
时清除 cache entry。

本轮不做以下事情：

- 落盘或输出 credential/HMAC key；
- 支持 cross-process HMAC resume；
- 读取 identity eval-probe、heldout、Gold 或 protected sample 正文；认证过的
  router 混合文件会明确解析为 80 条 train 加 20 条 `eval_proxy`，后者不会
  输出到训练 shards；
- 授权训练，或启动 model/GPU/provider 请求；
- 声称质量、泛化、物理 KV 复用或 formal release。
