# Gemma 3 Chat FINAL Producer Git 溯源覆盖层 v1

## 结果

这个覆盖层给已经完成的 Gemma 3 1B IT 五专家聊天诊断补上不可变的
Producer Git 溯源。它不修改训练配置、训练 receipt、适配器、评测 receipt
或任何数据文件，也不会重跑训练或评测。

冻结的 Producer 身份为：

- 仓库：`anchor-moe-lora-gemma3-chat-v1`；
- commit：`5a8343baa117f8ec2340debaa3fbc727ee993da2`；
- tree：`a9b87ad36773c31506118c13250af0a83fd5752d`；
- parent：`8ecef30d29d4c4a2e990ebc543dab21a6e3c33b1`；
- parent tree：`c19e2b485c6d7b062ca4fa00f3eaaa3bb8d146f2`；
- 精确 diff：24 个路径，其中 23 个新增、1 个修改。

审计会认证 commit 对象、父子谱系、精确 diff、tree entry、Git blob
对象 ID、blob 字节数和 SHA-256，随后证明 Consumer 实际使用的 16 份
原始文件与相应 Producer blob 逐字节相等。

## 范围与非授权边界

claim scope 固定为
`additive_posthoc_git_provenance_non_authorizing`。

该 attestation 只是诊断元数据：

- 不改变既有训练身份；
- 不追溯授予训练权限；
- 不授予正式训练或发布权限；
- 不声称质量、来源不相交或物理 KV 复用；
- 不重跑训练或评测；
- 不启动排队中的 10k/Luna 任务。

`training_authorized`、`formal_training_authorized` 和 `formal` 始终为
`false`。

## 认证内容

### Producer Git

`configs/research/gemma3_chat_final_producer_git_overlay_v1.json` 精确枚举
24 个变更路径。每个路径都会验证：

1. 精确的 `A` 或 `M` diff 状态；
2. `100644` tree mode 与 blob SHA-1；
3. Git blob 对象哈希 preimage；
4. 精确字节数与 SHA-256；
5. 新增路径在 parent 中不存在；修改的 `.gitattributes` 则绑定准确的
   parent blob。

本地 branch ref 与 remote-tracking ref 只是在生成 attestation 时的观察值，
不是不可变门禁。未来分支前移不会反向破坏已经认证的 commit/tree/blob
身份。审计不要求 Producer 当前 checkout 指向该分支，也不要求工作区干净。

### 16 份运行时副本

Consumer 侧 16 份文件包括：

- Producer 配置和 schemas；
- closed grammar；
- Producer 实现；
- manifest 与 mandatory sidecar；
- build receipt 与 mandatory sidecar；
- token inventory；
- train 和 eval-proxy partitions。

JSONL 与 token inventory 只作为不透明原始字节参与哈希与等同性比较。覆盖层
不会解析记录、查看样本文本或输出 token IDs。

### 已有诊断结果

既有身份保持不变：

- 训练配置 SHA-256：
  `35dc9effc713730274937352c0f828c881939e503b6725225da7cb54604d9159`；
- Consumer 训练实现 SHA-256：
  `55a7c45eb8f4b6c590938a486b65f675920d39d46f94686685a04e08f3bde802`；
- 训练 receipt SHA-256：
  `5988d67dc092a003a550dab42751791c8578dcca56bfcf49d7474a8eccb89a3a`；
- 训练 receipt sidecar 物理 SHA-256：
  `cb1b5fe5675bb36511c5940ff7c5675b71710fef8d66ccb2719e685746a8c600`；
- 评测 receipt SHA-256：
  `ea3ca5e12d7af9bd3e40a0a81bd32dc74060cb051e319377f9e55b3b8c86956b`；
- 评测 sidecar 物理 SHA-256：
  `75a9d663aba21a9f77760d2937db3d854d2316ba508360b3eb8f7d9398879c0c`。

此外还会验证 10 份 phase receipt、10 份 mandatory phase sidecar、5 份
adapter config 以及 5 个完整 adapter safetensor 文件。Safetensor 仅流式读取
做哈希，不会作为模型加载。

## TOCTOU 与 Git 加固

审计会：

- 清除所有继承的 `GIT_*` 环境变量；
- 强制 `GIT_NO_REPLACE_OBJECTS=1` 和 `GIT_NO_LAZY_FETCH=1`；
- 拒绝非空 grafts 和任意 replace refs；
- 拒绝符号链接与 Windows reparse point；
- 用同一个打开句柄绑定字节、stat 与文件身份；
- 在结束前重新执行完整的不可变 Git 认证；
- 发布前重新读取并重哈希所有 Consumer 输入和适配器。

整个过程没有网络命令。本地 remote-tracking ref 不会冒充 live network
检查。

## 使用

在 consumer 仓库根目录下运行：

```powershell
$python = "C:\path\to\python.exe"
& $python scripts/research/audit_gemma3_chat_final_producer_git_overlay_v1.py `
  --producer-repo C:\path\to\anchor-moe-lora-gemma3-chat-v1 `
  --output-dir artifacts/diagnostics/gemma3_chat_final_producer_git_overlay_v1/producer-final-5a8343b
```

显式输出目录必须尚不存在。审计器会原子发布：

- `attestation.json`；
- `attestation.json.sha256`。

sidecar 格式严格为：

```text
<小写 SHA-256><两个空格>attestation.json<LF>
```

只有不可变 Git 身份、运行时副本、既有 receipts 与 adapters 全部通过时，
命令才返回 `0`。任何漂移都返回 `2` 和稳定的纯元数据错误码，所有授权字段
继续为 false。

## 测试

```powershell
$python = "C:\path\to\python.exe"
& $python -m pytest -q tests/test_gemma3_chat_final_producer_git_overlay_v1.py
```

聚焦套件包含一次真实端到端审计，以及 config/claim 漂移、Producer/runtime
不一致、sidecar 格式、重复键/非有限 JSON、外部 schema ref、路径穿越、Git
replace refs、grafts、小文件与流式 adapter TOCTOU、覆盖已有输出目录等负测。

## 本覆盖层之外的未完成项

该覆盖层不会改善当前较弱的工具调用/审查得分，不会生成重新配比的数据，
不会训练未来的情绪路由 LoRA，不会实现物理 Q 独立 KV 共享，也不会启动
10k/Luna 队列。这些都属于后续独立增量实验。
