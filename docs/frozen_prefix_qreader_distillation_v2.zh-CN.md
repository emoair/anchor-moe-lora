# Frozen-prefix Q-reader 蒸馏管线 v2

状态：这是增量式研究 profile。元数据可以达到 materialized，但执行、训练、
formal training 和 release authorization 均继续阻塞。

## 1. 目标与非目标

本版本不另造一套编排栈，而是复用已经认证的 SWE-bench 执行底座：五阶段调度、
CC Switch 路由、OpenCode 沙箱、真实工具调用/结果轨迹、验证器、HMAC receipt、
断点续跑、canonical Gold 导出、TaskBoard projector 和通用 release 合约。
V2 只在认证 Gold 之后增加版本化的 **frozen-prefix Q-reader /
prefix-branch producer-consumer** 研究视图。

Producer 边界严格限定为：

- canonical 五阶段 Gold 保持逐字节不变；
- 只有在 Gold 认证之后才生成按角色、按因果可见性过滤的视图；
- 固定五角色：`planner`、`tool_policy`、`frontend_gen`、
  `frontend_review`、`security_gate`；
- 共享前缀与专家私有尾部只生成认证元数据，不生成真实 KV tensor；
- `q_only` 是唯一 primary 标签，`o_only` 与 `q_plus_o` 只是
  non-authorizing diagnostic overlay。

本项目不宣称隐藏 CoT、无损继承推理、零 KV、O(1) attention、底模计算消失、
模型在同一请求中生成 sentinel 后 aLoRA 热切换、普通 in-stack Q-LoRA 的全栈
KV 精确共享、已经实现物理 cross-attention Q-reader、token-level MoE、
formal 质量或 formal 训练就绪。

## 2. 两版本边界与复用关系

`task-level-moe-lora-v1` 服务于之前的任务级 MoE-LoRA 分支；
`frozen-prefix-qreader-v2` 是独立增量版本。V2 不修改 v1 profile、v1 generic
release schema、v1 scaffold、canonical Gold 或 held-out。

两版共享：

1. source bank 与五阶段 work order 调度；
2. CC Switch provider/model 路由；
3. 受控 OpenCode 沙箱及真实 tool call/tool result；
4. 分阶段语义验证；
5. HMAC execution receipt 与 checkpoint/resume；
6. canonical Gold exporter；
7. deterministic TaskBoard projector 和 release gates。

差异仅在认证后的 post-Gold profile、bundle capability 元数据、training view
形状以及 downstream release overlay。这样既复用昂贵的 provider 执行逻辑，
又不允许一个版本的输出格式暗改另一个版本。

## 3. 数据流

```text
canonical Gold / TaskBoard
  -> serializer 之前完成 visibility filter
  -> route directive
  -> 单一 concise_rationale_plus_json primary scaffold
  -> 校验并显式 commit 文本
  -> frozen base 对短 committed scaffold 重编码
  -> 下一次请求触发 expert
  -> append-only expert-private KV tail
```

V2 大批量 materializer 严格地为每个 bundle/role 只输出一个 primary
`concise_rationale_plus_json` 视图，且 `pair_count=0`。旧 frozen scaffold v1
继续保留并且不修改其 `json_only` / `concise_rationale_plus_json` 配对消融资产；
V2 不继承该配对。V2 adapter diagnostic 是同一 record inventory 上的 execution
overlay，不是第二种 data view，也不能按重复行扩张计数。

必须先按 `task_bundle_sha256` split，再扩增 role/view/noise/causal 视图。
同一 bundle 五角色始终同 split；`eval_proxy` 绝不是 held-out。

current、future、forbidden block 正文必须在 prompt serialization、tokenizer、
shared prefix、日志和 receipt 之前排除。禁止先编码再用 mask 隐藏，也禁止
stringify 整块 TaskBoard。

## 4. 为什么采用“两请求 + 显式 commit”

aLoRA runtime 只扫描**下一请求输入**中的 invocation token。同一次生成过程中
模型吐出 sentinel 不会安全地中途切 adapter。因此：

1. Planner/base 的 request 1 生成 concise rationale、route JSON 和可选 sentinel；
2. verifier 校验 schema 并 commit 可见文本；
3. frozen-base producer 把这段短文本重新编码为 immutable downstream segment；
4. request 2 把 scaffold 作为输入，只在 invocation boundary 后启用专家 adapter。

Planner 私有 KV 由不同 adapter/hidden-state lineage 产生，不能直接交给 Expert。
跨边界的只有已 commit 文本，且必须由 frozen base 重编码。专家激活以后，
prompt 与新生成 token 只追加到该专家私有 tail；私有 KV 不跨专家迁移。

即使只训练 Q，当前 attention 层的输出也会改变后续 hidden state 和后续层 K/V。
所以 Q-only 是必要的控制项，不是全层 KV 可精确复用的证明。精确复用仅限于
token order、position、RoPE、tokenizer、model architecture、KV-producing
weights 与 ordered prefix lineage 全部一致的前缀。

## 5. Schema 与字段

V2 profile freeze 认证共享执行底座与 V2 专用边界。training-view producer 输出：

- record：`anchor.frozen-prefix-qreader-training-view.v2`；
- manifest：`anchor.frozen-prefix-qreader-training-view-manifest.v2`；
- `train.jsonl`、`eval_proxy.jsonl` 与 mandatory sidecar。

Body-free bundle producer 输出：

- record：`anchor.frozen-prefix-qreader-bundle-profile.v2`；
- manifest：`anchor.frozen-prefix-qreader-bundle-profile-manifest.v2`；
- `bundle_profiles.jsonl` 元数据 inventory。

每条记录绑定 bundle/source/split/language/information-flow stratum/role/
capability labels，以及 ordered segment/prefix lineage 和 route-boundary
architecture contract。Route JSON 使用确定性 key order 的 canonical UTF-8 JSON。
每条记录绑定
`training_view.routing_json_sha256=SHA256(canonical_route_json_bytes)`，同一
digest 还参与确定性 record identity preimage；materializer 在发布前重算并核对
这两层关系。不得用语言或 source namespace 作为逃避语义重合检测的盐。

`anchor.frozen-prefix-qreader-release-overlay.v1` 必须同时认证并合取：

1. 旧 `anchor.generic-train-release-lock.v2`；
2. V2 profile freeze manifest；
3. V2 training-view manifest + mandatory sidecar；
4. bundle-profile manifest + mandatory sidecar；
5. consumer diagnostic reference + mandatory sidecar。

Runtime path 与 expected manifest SHA 由 CLI 显式输入；checked-in config 只锁
schema version 与 producer implementation identity，不锁临时 runtime hash。
依赖 DAG 无环：V2 profile 可以绑定 overlay schema/module/CLI，但不能绑定
overlay config；config 依赖这些 code identity，五份 runtime manifest 最终只流向
overlay。实现会重新计算 DAG 并拒绝环。

Freeze CLI 的信任边界是专用新进程和 stdlib-only bootstrap；它不普通 import
任何 `anchor_mvp` package。Bootstrap 先把 executing CLI 的 canonical path/bytes、
config snapshot、profile manifest + strict sidecar，以及 implementation
path/bytes 同时与 config/profile binding 认证；随后只把这一次认证过的
implementation snapshot 编译进 digest-qualified private module。Package
`__init__`、缓存 bytecode、第二次 implementation 文件读取、修改后的 sibling CLI
和预加载 package state 都不能选择实际执行代码。Overlay 仍会在终点重查物理身份。
已有 Python 进程内嵌时，必须显式调用已经认证的 library API，不能冒充 CLI 边界。

Overlay 不接受“各 manifest 自报 hash 能彼此自洽”作为 producer 身份证明。它会从
profile freeze dependencies 重建 role-indexed map，逐字节认证 projector 的
config/implementation/CLI/schema，以及 materializer 的
config/implementation/schema/builder/auditor，并在终点再次检查这些 snapshot。
Training-view 与 bundle-profile 的 producer 字段必须同时匹配该 map 和实际加载的
manifest schema；projector manifest、sidecar/record、segment-plan schema 身份还
必须在 generic lock、profile、training view 与 hierarchical-KV contract 间一致。

Generic lock、profile freeze、training view 与 bundle profile 的 CLI path 必须
等于 checked-in project-relative canonical runtime directory；即使 bytes 相同，
临时目录副本也会被拒绝。输出绑定使用 `logical_manifest_path` /
`logical_sidecar_path`，并标记
`source_location_kind=project_canonical_runtime_dir`。Consumer reference 是唯一
external input；overlay 不发布其机器绝对路径，只记录 repository-logical path 与
`source_location_kind=external_consumer_reference`。

Consumer reference 使用 overlay schema 内嵌的严格**最小兼容 subschema**，并叠加
唯一 frozen manifest/sidecar hash 认证。该 subschema 只验证 producer 合取所需的
count、semantic identity、Q-only 标签、false gates 与 0-request audit；它不冒充、
也不替代 consumer 仓库的完整 publication schema。

Overlay 固定输出 `status=profile_materialized_execution_blocked`。即使 base lock
自报 `ready`，也不能抬高 `training_authorized`、
`formal_training_authorized` 或 `release_authorized`；它们固定为 `false`。

## 6. Producer、consumer、runtime 职责

Producer 负责认证元数据、因果可见性过滤、split-before-expansion、确定性视图、
create-once manifest/sidecar。Producer sidecar 不直接耦合 llama.cpp runtime。

Consumer 必须先认证 exact bytes、schema、hash、bundle/split/role cross-binding
和 forbidden-content 排除，再把记录交给训练侧；认证前记录保持 opaque。
Diagnostic metadata 不能授权 formal training。

Runtime 负责 request-local serialization、tokenizer/chat-template identity、
完整 request 2 单次 tokenize、trigger covering span/overhang、adapter 激活、
private tail 分配、物理 KV 行为、GPU lock 与 execution receipt。可选 aLoRA
capability 只能表示 `next_request_input_activation_only`，不是 cross-attention
Q-reader，也不证明物理共享 KV。

孤立 trigger token IDs 与全局 token index 均不权威。若需要 boundary，应对完整
request 2 只 tokenize 一次，记录 zero-based/end-exclusive covering span，并用
exact serialization SHA 绑定 request-local receipt。

## 7. 1000 条 reference 与 Gemma 边界

外部 diagnostic reference 固定为：

- manifest SHA-256：
  `a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed`；
- sidecar physical SHA-256：
  `7f238be47cc60af808421bbbdaefb6bbc5d5c0f617d976b66d5b2a87d767b0a0`；
- 200 unique task semantics × 5 roles × 1 primary view；
- train/eval_proxy = 800/200；
- EN/zh-CN = 100/100 semantics，translation pair = 0；
- 唯一 primary adapter label = `q_only`。

它只是 reference，不会转成 canonical Gold，也不继承 formal authority。

Gemma model-free tokenizer 观测固定 `seq_len=768` 和 strict no-truncation。
五角色最大完整 token 长度依次为 504/523/546/605/665；1000 条中有 514 条
超过 512，全部低于 768。旧的 Qwen declarative 512 不适用于本次 Gemma run。
在 Gemma tokenizer + official chat template combined identity 与 runner binding
最终冻结以前，执行继续 blocked。

## 8. Fail-closed 矩阵

| 条件 | 结果 |
| --- | --- |
| manifest 或 mandatory sidecar 缺失 | 拒绝 |
| manifest SHA 与 CLI 期望不一致 | 拒绝 |
| sidecar 不是严格的 `<sha>  manifest.json\n` | 拒绝 |
| schema path/hash/version 漂移 | 拒绝 |
| duplicate JSON key、非 UTF-8、非有限数值 | 拒绝 |
| 任一 metadata/config/schema/implementation 输入达到或超过 50 MB | 拒绝 |
| 认证路径包含 symlink/reparse point | 拒绝 |
| Producer artifact 来自非 canonical runtime directory | 拒绝 |
| snapshot 后到终点 recheck 之间输入变化 | 拒绝 |
| 输出已存在、与输入重叠、非 create-once 发布 | 拒绝 |
| generic v2 lock 不是 self-ready research proxy | 拒绝 |
| projector/bundle/view producer、strict projector sidecar 或传递 schema cross-binding 不一致 | 拒绝 |
| profile 绑定的 projector/materializer/builder/auditor 物理字节漂移 | 拒绝 |
| consumer reference count/split/language/Q-only identity 漂移 | 拒绝 |
| 任一输入宣称读取 Gold/heldout 正文或具有训练权限 | 拒绝 |
| tokenizer/template/runner identity 未知 | 继续 blocked |
| execution decision/lease/data-byte TOCTOU lease 缺失 | 继续 blocked |

失败只返回 content-free error code，不回显解析失败行。Overlay read-set 包含五份
manifest、五份 sidecar、checked-in schema/config/overlay code，以及用于物理身份
交叉检查的 17 份 profile-bound projector/materializer dependency 文件；绝不打开
JSONL partition、Gold 或 held-out 正文。

## 9. 从 0-request 到 live 的复现

静态与低内存验证：

```powershell
python -m pytest -q tests/test_frozen_prefix_qreader_release.py
python -m ruff check src/anchor_mvp/swebench/frozen_prefix_qreader_release.py scripts/data/freeze_frozen_prefix_qreader_release.py tests/test_frozen_prefix_qreader_release.py
python -m py_compile src/anchor_mvp/swebench/frozen_prefix_qreader_release.py scripts/data/freeze_frozen_prefix_qreader_release.py
```

先用各自 CLI 冻结 V2 profile 和小 fixture，再生成 blocked overlay：

```powershell
python scripts/data/freeze_frozen_prefix_qreader_release.py `
  --config-sha256 <checked-in-config-sha256> `
  --generic-release-dir artifacts/formal_v3/training_release/release_lock `
  --generic-release-manifest-sha256 <sha256> `
  --profile-freeze-dir artifacts/distillation-profiles/frozen-prefix-qreader-v2 `
  --profile-freeze-manifest-sha256 <sha256> `
  --training-view-dir artifacts/swebench/frozen-prefix-qreader-view-v2 `
  --training-view-manifest-sha256 <sha256> `
  --bundle-profile-dir artifacts/swebench/frozen-prefix-qreader-bundle-profile-v2 `
  --bundle-profile-manifest-sha256 <sha256> `
  --consumer-reference-dir <consumer-repo>/fixtures/research/synthetic_five_role_qonly_diagnostic_v1 `
  --consumer-reference-manifest-sha256 a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed `
  --output-dir artifacts/swebench/frozen-prefix-qreader-release-overlay-v1
```

该阶段 provider/network/model/GPU 请求均为 0。后续最小 live pilot 还必须显式
live flag、credential、cost cap、单 work order、checkpoint/resume 与 receipt
验证。只有 pilot、全量 projector streaming/peak-memory 测量、formal-v3 五件套、
protected inventory、tokenizer/runner binding、独立 execution decision 与 lease
全部完成后才能讨论扩量；overlay 自身永远没有授权能力。

## 10. 版本与 Git 戒律

Schema、config、implementation、fixture manifest、sidecar 与 runtime input 均按
physical-byte SHA-256 绑定。JSON 输出为 canonical UTF-8/sorted-key/compact/LF。
输入只读一次 bytes snapshot，从该 bytes 解析和验证，并在发布前与发布后做终点
identity recheck。

V1 保持兼容分支；V2 使用独立 branch 与 schema namespace。提交前只 stage 白名单，
执行 staged diff、credential、个人路径、单文件大小、JSON/YAML/UTF-8/LF、Ruff、
py_compile 和 focused test 检查；禁止 tag/release。只有 producer final freeze 后才
向 consumer 发送 runtime hash，绝不传播临时 hash。

迁移必须 opt-in。Consumer 必须显式支持 V2 profile 与 overlay；缺失、旧版本或未知
字段全部 fail closed，不能通过放宽 V1 loader 兼容。

V2 分支基线为
`524ca359eff128221ef4fa9f5a9e665abf64c7c3`（`task-level-moe-lora-v1`）。
本轮 scoped commit 白名单严格限定为：

```text
configs/orchestration/frozen_prefix_qreader_profile.schema.json
configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json
configs/orchestration/profiles/frozen_prefix_qreader_v2.json
configs/research/frozen_prefix_qreader_release_overlay_v1.json
configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json
configs/research/swebench_natural_language_scaffold_v2.yaml
configs/research/swebench_natural_language_scaffold_v2_bundle_profile.schema.json
configs/research/swebench_natural_language_scaffold_v2_bundle_profile_descriptor.schema.json
configs/research/swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json
configs/research/swebench_natural_language_scaffold_v2_config.schema.json
configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json
configs/research/swebench_natural_language_scaffold_v2_record.schema.json
docs/frozen_prefix_qreader_distillation_v2.md
docs/frozen_prefix_qreader_distillation_v2.zh-CN.md
scripts/data/audit_swebench_natural_language_scaffold_v2.py
scripts/data/build_swebench_natural_language_scaffold_v2.py
scripts/data/freeze_frozen_prefix_qreader_release.py
scripts/data/run_frozen_prefix_qreader_profile.py
src/anchor_mvp/swebench/frozen_prefix_qreader_profile.py
src/anchor_mvp/swebench/frozen_prefix_qreader_release.py
src/anchor_mvp/swebench/natural_language_scaffold_v2.py
tests/test_frozen_prefix_qreader_profile.py
tests/test_frozen_prefix_qreader_release.py
tests/test_swebench_natural_language_scaffold_v2.py
```

白名单之外的路径不得 stage。全部 schema/config/implementation 和生成的
manifest/sidecar 只按最终 commit bytes 与最终 runtime freeze 回报物理 SHA-256；
只有 commit、local HEAD、upstream HEAD 与 live remote HEAD 全部一致时才能称为
完成。

最终提交前的机器身份快照如下：

| 层 | 产物 | SHA-256 |
| --- | --- | --- |
| profile | profile schema | `5900f144c5aa25d359400b727fd5c8c31281b3a99792b3c27d783b357a2eb85a` |
| profile | freeze-manifest schema | `1310dd1c74f2f2f7c86bcfa3628925102a9c0b6f398df306dec99b446c60cfc5` |
| profile | checked-in profile | `f39ebde344d41ac29cf50d224795450d5d5da10534e382591d088e0b97224994` |
| profile | implementation | `973eea883e1e412083e3be8bc538428630d286fb2c09758493b3e2acaf18a944` |
| profile | runner | `1b1e88447ecfb53d6e91155ebbc0820d3d9d4df32ff441d8b31e455d01db3ccc` |
| materializer | config | `6fda2ff6bb6a92f8764daa8f68dc8226ae0839d2d9613dc240d9f0b75c9baee5` |
| materializer | config schema | `11f6a8555178f851a75341057750fadb415be7357e070275e4603453e03ec9e1` |
| materializer | descriptor / bundle-record / bundle-manifest schemas | `4614b0924ced82c483f6ead94e754e70426f7b12d45936eef40b43f18b21265a` / `42af26d742db8c06104f3955dfbd19c552c705c9237bd69455b42d132cb1ac5a` / `fbe7a543e0e8f19b27436ca882af629e44252ab5e7a5d246443709cadc39609f` |
| materializer | training-record / training-manifest schemas | `ac5176b53072a75439fd4f29f9e96e16416b6f690bed776df9e5509ae88b98c3` / `dae37cd63462ac6945547c32ca617d7a51d41700432292f989fc901491a2eb2c` |
| materializer | implementation | `3a3ff1eceed67f489e2edfb1df46f18615adf94128fe1c3ae499e1917e6228a3` |
| materializer | builder / auditor | `99a88e90d2419fca1cc6c35445058105fee7357962985063aefa01be7462b4d7` / `bf3f48476e0b174353c1363fcf70ab6aa518a504c9454ec634e02601d6fda08a` |
| release overlay | config / schema | `a6367f25654e7a5d2ea1d27cb56c50d19350a2af6cf8431664515983302dc611` / `03fd592a8fa98aee08a7193089dab8934663474b37de6ecc7b9cd5cf11eb5b91` |
| release overlay | implementation / 认证 bootstrap CLI | `35af412ad992ab6e19e53267d98a2188ee20bfe931d4884ce674bfb3764315c1` / `1c700cd6863bfb1862a1b60363dc5226914b9cdacf5e2be14f4bb61c3a980bdc` |
| runtime freeze | profile manifest / sidecar 物理字节 | `97f3fe1e8aa89bac107844413cd8a5da41ea6df474cf431687096a4e7972e255` / `72470ae4716733147ed654a27bc214151fe65816a54f1a1a93bdabbb7fd9c2eb` |

runtime freeze 是本地 ignored artifact，不是 committed fixture。这些身份不会授予
live、training、formal 或 release 权限。

## 11. 已验证、proxy 与未完成

已验证的契约事实包括：可复用的五阶段执行底座、metadata-only V2 schema、严格
causal filter、create-once 认证、1000 条 reference 的 count/hash，以及 model-free
Gemma token length 观测。此前 Qwen/Gemma aLoRA、Q-only/Q+O 结果均只是 proxy/smoke。

真实未完成项：

- current-identity live teacher pilot 与大批量 provider 蒸馏；
- 全 bank streaming projector 的 peak-memory 实测；
- formal-v3 frozen snapshot、final projector、source-disjoint、generic execution
  与 final release lock；
- 四类 protected source-ID inventory 与 zero-intersection proof；
- Gemma tokenizer/chat-template/runner 最终 binding；
- execution decision、single-GPU lease、data-byte TOCTOU lease；
- 物理 KV backend/CUDA、multistream correctness、Q-reader zero-copy；
- multi-seed independent-bundle 训练和质量/安全/性能评测。

在这些缺口闭合前，metadata ready 不得晋级 training/formal ready。
