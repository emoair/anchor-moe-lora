# Frozen-prefix Q-reader V2 消费端

这是 Producer 分支 `research/frozen-prefix-qreader-distillation-v2`、提交
`8c9fdfc71b94b5b41d6f3566e9f81baadcc0c267` 的模型无关、增量式消费端。

它会认证 Producer 的远端跟踪引用、commit、tree、parent、精确 24 路径
变更集、Git blob ID、字节数和 SHA-256；随后从同一份字节快照验证
Producer 的 profile、materializer、release-overlay 契约，以及本地强制
`manifest.json.sha256` sidecar。结束前会再次完整核验 Git 对象和本地文件，
封住 TOCTOU 换档窗口。

它**不会**复制或读取 Gold、heldout、protected 或已物化训练记录正文；
不会加载模型、触碰 GPU、请求 provider、访问网络、训练、发布、打 tag，
也不会授权线上执行。

## 一条命令完成预检

在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"
python scripts/research/preflight_frozen_prefix_qreader_v2_consumer.py
```

命令会**有意返回退出码 2**。契约认证成功时会输出一行 JSON；关键字段如下：

```json
{
  "status": "producer_contract_ready_execution_blocked",
  "producer_contract_authenticated": true,
  "gates": {
    "training_authorized": false,
    "formal_training_authorized": false
  }
}
```

看到上述状态时，退出码 2 表示“身份与契约可消费，但执行仍被门禁阻止”，
不表示审计失败。

## 机器实际核验的内容

- 五角色固定为 `planner`、`tool_policy`、`frontend_gen`、
  `frontend_review`、`security_gate`。
- 以 `task_bundle_sha256` 分组，先 split 后 augmentation。
- 每个角色只有一份 `concise_rationale_plus_json` primary view。
- `q_only` 是主实验；`o_only` 和 `q_plus_o` 仅为不授权的诊断控制。
- current target、future、forbidden 正文必须在序列化前排除；禁止整块
  TaskBoard stringify，也禁止“先序列化、后 mask”。
- 路由边界是显式两请求 validate/commit；只有已提交文本才由冻结底模
  重新编码。
- 共享前缀必须关闭 adapter；路由后每个专家维护 append-only 私有尾部
  KV，禁止跨专家复用私有尾部。
- 不发出 token index 或物理 KV tensor；已提交 scaffold 必须在 adapter-off
  状态重编码；不继承 wide LoRA，也不重写源记录。
- 精确复用范围仅限 identical ordered prefix lineage。不会宣称物理
  Q-reader 已实现、整段生成 KV 共享、普通 in-stack Q-LoRA 可精确复用，
  也不会宣称 token-level MoE。

## 为什么仍然阻塞

本消费端只认证契约，不等于真实数据或训练 release。当前缺少 V2 物化
训练视图、bundle profile、generic release lock、execution
decision/lease、数据字节 TOCTOU lease 和 live provider 蒸馏。
Formal-v3 仍为 `0/5`，protected inventory 仍为 `2/6`，所以 training、
formal training、release、live authorization 全部保持 false。

## 常见错误

- `producer_tracking_ref_unavailable`：本地没有所需 tracking ref。请在
  本预检之外更新 Git 后重跑；预检本身绝不 fetch。
- `producer_provenance_mismatch`：tracking ref 已不再指向冻结 commit。
  不要绕过，应建立新版本契约。
- `producer_blob_identity_mismatch`：Producer blob 与冻结字节身份不一致。
  不要放宽 hash。
- `manifest_sidecar_*`：恢复严格 sidecar 格式：
  `<64位小写十六进制><两个空格>manifest.json<LF>`。
- `*_contract_invalid`：Producer 或 consumer 语义发生漂移。应升级版本，
  不要修改冻结 V2 的含义。

## 文件位置

- 配置：
  `configs/research/frozen_prefix_qreader_v2_consumer_v1.yaml`
- Manifest 与强制 sidecar：
  `fixtures/research/frozen_prefix_qreader_v2_consumer_v1/`
- Loader：
  `src/anchor_mvp/research/frozen_prefix_qreader_v2_consumer.py`
- CLI：
  `scripts/research/preflight_frozen_prefix_qreader_v2_consumer.py`
- 测试：
  `tests/test_frozen_prefix_qreader_v2_consumer.py`
