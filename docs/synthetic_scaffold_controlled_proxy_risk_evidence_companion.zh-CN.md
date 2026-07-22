# 受控 proxy Q+O 风险证据 companion

## 范围与不可变依赖

`anchor.synthetic-scaffold-controlled-proxy-risk-evidence-companion.v1` 是
纯元数据、增量 companion，不替换也不修改 Producer commit
`23194f7b3c707e3531ac92a64863c2b2f523f81d` 的 frozen follow-up。接受前会
重新认证 frozen schema、contract、sidecar、implementation、comparison 和
comparison sidecar。该依赖同时校验本地 bytes 与 raw Producer
commit/tree/artifact blobs；仅复制同字节工作树不能冒充 Git provenance。

源身份是 Consumer commit
`58e9cd0c021ac0f01250746d44f199c1f616261d`，其直接 parent 为
`6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465`。审计器禁用 Git replacement
objects，直接读取 raw commit/tree 与三对 receipt/sidecar blob；绝不 fetch、
不读 Consumer 工作树，也不打开模型、adapter、PNG、数据分区、Gold、heldout、
scaffold 正文、prompt、answer 或 raw token IDs。

三份 receipt/sidecar SHA 分别为：

- Q/O 分支消融 `59750842e7bbad7fb06fcc64a1b9956dbd449e5591ba85f4abca021616da8ca3`
  / `ad18c248e9bb091797f229927febddd11c0520d1497943119e0da2401b657a31`；
- 静态谱风险 `c2fddd98ece4127ad3f17a19ffbd5bfa6e8d7f95588964b389e7e2970cfc8dd3`
  / `61b829f963b48dfe1b3f98337b6992dce0e5944595ec436e296b37b2986ac74a`；
- attention summary `fc1ce0168cacfd1ed46a7ffcc1b482e7593253e224e780ab7dc6f7b701bb58a4`
  / `1238605cf7abff130096e78071dd1f6ab2f916af93d85b29fdf3e054e1a13120`。

receipt 中的 config、implementation、model、adapter hashes 只是 receipt 声明的
传递身份；Producer 不声称已重新打开这些 artifact。

Producer 最终身份：schema
`c04ba5072c2892f111a913808559f1c3eca9864977159c387df09fa6b7081068`、
contract `352870bbea976c0b97df722fd3b188d731a8d463ec33f27bcd15bdb2e292ac28`、
mandatory sidecar 物理 SHA
`c75a91a78123fc1f583e9d053525c290fa39c3e6a811b22ecb48b11581c5503b`、
implementation
`2f707507014dc9e70546a024b8bf109f779cf0c33019439f8396563e795e5d3a`。

## 闭合观测与解释

联合训练 Q+O checkpoint 的同模板 teacher-forced macro loss，在 off、保留 Q、
保留 O、full 四模式下为 `3.1020863533020018`、`2.6766975283622743`、
`1.0298870503902435`、`0.8586097195744514`。保留 O 分支时保留了本次
off→full 降幅的 `0.9236553979475981`。这是联合训练 adapter 的事后分支消融，
不是独立训练的 O-only arm；分支效应不保证可加，也不证明 O-only 机制。

4 bundles/20 records synthetic OOD proxy 的对应 loss 为
`3.035316228866577`、`2.983178400993347`、`2.8666090965270996`、
`2.896842730045319`，full 相对 off 只改善 `0.04562078161884503`。这只是模板或
答案形状写回风险信号；它既非 heldout，也非预注册 confirmation fixture，不能
建立广泛泛化结论。

静态谱观测为 O top-1 能量 `0.82318570872`、有效秩 `2.11605570639/8`、O/Q
总 delta 能量 `1.767153939794`、target-frequent/random 投影
`1.95910776849`、prompt-control/random 投影 `1.006882507832`。token groups
未匹配频率或词性，所以只允许称相关性风险信号，不能证明逐字、答案、exploit
或因果记忆。

Attention Hook 只有一个 79-token 合成 probe、BF16+TF32、head-mean 聚合及
0/13/27 三层。full-minus-Q 分支 mean/max difference 为 `0/0`、
`0.0008257970912382007/0.061482757329940796`、
`0.0019083430524915457/0.1309814453125`。第 0 层为零不代表全栈不变；后层变化
仅与 O 更新沿 residual stream 传播相符，attention 不是解释或一般因果证明。

## 后续预注册边界

原非授权计划保持不变。唯一 primary endpoint 是 step-80 bundle-macro loss
delta；5/10/20/40 仅作次要学习曲线。Replication 仍至少需要 5 个 master
seeds、全部注册 arm order、相同 base/sample/token/order/budget/optimizer，以及
独立训练的等预算 O-only 与 K+V controls。5 seeds 最多支持 controlled-proxy
replication 信号，不能授权 formal significance。

Confirmation 前必须冻结 generator、namespace-neutral blueprint inventory、
`task_bundle_sha256` split、seeds、arm orders、endpoint 和统计方法；Discovery 与
confirmation 同时需要 ID 和 body-free blueprint disjointness。建议预注册
旧任务/新模板、新任务/旧模板、新任务/新模板三因子矩阵，并改变字段顺序、词法、
答案形状和安全 nonce。当前不存在这些 fixture 或 disjointness proof。

长上下文/cache 口径不变：当前只允许 8K/16K/32K diagnostic preflight；exact
reuse 仅限 identical ordered frozen-prefix lineage。本 companion 未实现 Q-reader、
物理 KV、CUDA zero-copy、多流共享或质量评测。

## Fail-closed 与计数

contract 是 closed Draft 2020-12 instance。本地输入执行 single-byte snapshot、
精确 sidecar、重复键/非有限数拒绝、同 bytes reparse 及末端重验；raw Git
commit/tree/blob 也在末端重读。同步替换 receipt+sidecar、指标/计数漂移、源 GPU
计数洗零、正文键、路径逃逸、reparse/TOCTOU 漂移或 promotion boolean 都会
fail closed。

源计数与 Producer 计数分开：消融与 Attention 各用了 1 次 model/GPU，谱审计为
CPU-only。Attention receipt 没有机器 `provider_requests` scalar，因此 companion
记录 `provider_requests_reported=false`，不伪造为 0。Producer 自身 provider、
network、model、GPU、protected-body 计数均为可认证的 0。

固定门禁仍是 formal-v3 0/5、protected inventories 2/6、multi-seed replication
未完成、独立 confirmation 未完成、`training_authorized=false`、
`formal_training_authorized=false`。

## 纯元数据复现

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py `
  --repo-root .
python -m pytest -q `
  tests/test_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py
python -m ruff check `
  src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_risk_evidence.py `
  scripts/data/audit_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py `
  tests/test_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py
```

运行前必须本地已有精确 Consumer commit object；审计器绝不 fetch。命令不会发起
provider/network 请求，不加载模型/GPU，也不创建 tag/release。
