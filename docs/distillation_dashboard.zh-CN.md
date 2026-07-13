[English](distillation_dashboard.md) | [简体中文](distillation_dashboard.zh-CN.md)

# 蒸馏仪表盘与本地控制面

这个独立页面用于观察蒸馏 JSONL 元数据，也可以选择性地启动一个经过严格配置的自动化子进程。它不会公开提示词、消息、模型输出、生成代码、分片绝对路径或凭据。

## 启动页面

在仓库根目录运行。可以在启动时以只读方式挂载当前外部 c10 采集任务：

```powershell
python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10
```

打开 `http://127.0.0.1:8765/`。服务只接受精确的 IPv4 回环地址；`localhost`、`::1`、`0.0.0.0` 和远程地址都会被拒绝。

页面支持 English 与简体中文。首次访问默认跟随浏览器语言；语言按钮只会在本地存储中保存不敏感的语言偏好。

不需要进程控制时使用只读监视模式：

```powershell
python scripts/observability/distillation_dashboard.py `
  --monitor-only `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10
```

若只需要一次终端摘要或不含内容的 JSON 快照：

```powershell
python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10 `
  --once

python scripts/observability/distillation_dashboard.py `
  --shard c10=data/automated_v3_shards/ark_max_retry2_offset300000_c10 `
  --once --json
```

`--shard` 可以重复指定，推荐使用 `标签=目录`。HTTP API 只返回操作员设置的标签。通过 `--shard` 传入或使用“只读挂载”添加的分片永远不会获得进程所有权，因此页面不能停止它。

## 实时遥测

页面每两秒轮询 `/api/snapshot`，显示：

- 种子行数，以及 `plan`、`tool_policy`、`frontend`、`review`、`security` 各阶段行数；
- 根据五个阶段 seed ID 交集计算的完整链数量；
- 来自状态审计账本的累计请求数和累计输出 token；
- 明确标为保留阶段小计的 input/output/total/cache token；若请求或使用量维度不完整，则标为已知下界；
- 审计账本输出 token 和请求的滚动速率，以及单独命名的保留行 token 与线路尝试速率；
- 每阶段和全部持久化行的滚动速率；
- 已接受种子、隔离种子拒绝数/速率以及近期不含内容的拒绝原因码；
- 重试、不含内容的错误类别计数、额度、ETA 和生命周期事件；
- 分片属于受管子进程还是外部只读挂载。

全部滚动速率使用最长 60 秒窗口内观测到的计数器差值。在获得两个可用观测值前显示 `unknown`；计数器重置时也显示 `unknown`，不会产生负速率。

Token 数量绝不会根据文本长度估算。累计请求数和累计输出 token 只来自 `status.json` 的 `audit_ledger`；当前 `quota_epoch` 只用于额度进度条，绝不会冒充全程累计值。保留阶段使用量只来自 `provenance.teacher.provider.completion.usage`。当审计账本没有 input/total 维度时，页面不会把它们声明成全局精确值。每个指标都有 `exact` 标志，并在适用时携带 `unknown_rows`。非精确值表示“已知小计 + 未知余量”。

状态检查点只有在相对最新保留数据没有过期时才有效。宽限期至少为 30 秒；若检查点策略更大，则扩展为 `3 * usage_checkpoint_policy.maximum_seconds + 5`。这既避免正常的检查点间隔被误判为停滞，也防止旧状态计数器与更新的 JSONL 数据混用。

`seed_rejections.jsonl` 使用独立的四字段白名单扫描器。仪表盘只读取 `error_class`、`reason`、`content_retained` 和 `observed_at`，不会实体化种子索引、响应哈希或其他字段。自由文本验证信息会缩减为固定原因码，例如 `active_payload_material`、`credential_like_material` 或 `invalid_json_object`；未知文本只会变成 `unclassified_validation`，不会原样返回。未明确声明 `content_retained: false` 的行只显示为 `metadata_policy_violation`。

## 启动新的受管分片

表单刻意不提供自由命令字段。基础配置下拉框只列出具备有效 SOP/输出目录、阶段、并发、种子数量和预算结构的严格自动化配置；任务卡和 SWE-bench 配置不会混入其中。需要填写：

- `configs/data` 中的严格基础配置和任务卡配置；
- `data/` 下一个新的相对输出目录；
- 与已登记配置或既有控制面清单不重叠的种子偏移区间；
- 并发数、供应商 URL/协议/模型和 API key；
- 传输超时/重试、自动化预算、冷却行为和监督器重连设置。

可选的 CC Switch 目录是只读元数据，来自 v3.16.5 固定版本的内置快照或经验证的活动快照。供应商/模型预设只填写 URL、协议和精确请求模型 ID，所有字段仍可手动修改。“检查固定版本差异”只执行离线只读刷新，绝不会下载或应用元数据。仪表盘不会读取 CC Switch 数据库、OpenCode 配置或供应商密钥。

只有在供应商绑定、精确别名、受支持协议、四个使用量维度（`input`、`output`、`cache_read`、`cache_write`）和经过审查的价格全部已知时，才显示固定价格成本；否则明确显示 `UNKNOWN`，不会把缺失维度当作零。

网络路由默认为“直连”。子进程会收到 `NO_PROXY=*`，并且不会继承代理 URL。只有显式选择“继承检测到的代理”才会把当前进程的代理环境复制给子进程。API 只公开 `proxy_detected` 布尔值，不会公开代理 URL 或凭据。

Key 只用于当前操作，复制到尽力可清零的内存槽，并通过 `ANCHOR_CONTROL_API_KEY` 传给子进程。它不会写入 YAML/JSON、API 响应、命令行参数或仪表盘日志。Start、Continue、模型发现之后，浏览器密码字段都会清空；子进程退出、停止、发现结束、显式清除 key 或服务关闭时，内存槽也会清空。仪表盘重启后必须重新输入。

Start 会生成不可变且不含秘密的文件：

```text
runs/control-plane/<run-id>/effective-config.yaml
runs/control-plane/<run-id>/control-manifest.json
```

其中记录基础配置、任务卡文件和 SOP 树的哈希，以及实际生效的供应商、传输、预算、并发、输出和调用参数。唯一的生产子进程命令是：

```text
<current-python> -m anchor_mvp.data.automation --config <effective-config>
```

只有选中时才附加 `--wait-cooldown`。进程使用 `shell=False`、固定仓库工作目录、净化后的环境、空标准输入输出以及新的进程组/会话。生成前会为输出目录取得排他所有权锁。

## 停止与精确续跑

“停止”首先向进程组发送协作式中断。超过有限宽限期后会终止进程树，必要时再强制杀死。采集器以 flush/fsync 写入已接受的 JSONL 行，但其核心当前没有专用的协作式信号检查点，因此停止后 `status.json` 可能仍看起来位于 worker 中间状态；已 fsync 的行仍然持久。

“精确续跑”会重新载入原始有效配置，并拒绝任何覆盖。它会检查配置 SHA-256、严格的清单/有效配置字段一致性、运行 ID 与目录身份、自动化语料绑定、当前基础配置和任务卡哈希、当前 SOP 树哈希、完成状态及所有权 token 锁。只有同一个内存控制器亲自观察到自己的子进程退出时，才可续跑看起来仍活跃的状态。仪表盘重启后，任何看起来活跃或由外部拥有的分片都只能只读挂载，以避免误启第二个写入者。

表单中的重连设置用于监督子进程意外退出，与自动化进程内部供应商请求重试所用的 `max_retries` 相互独立。监督器指数退避有上限，并且可被停止操作取消。

## 模型发现

“探测/载入模型”使用仅存在于内存中的 key 请求供应商模型列表。发现并非强制步骤；选择“强制模型”后，即使供应商不支持列表接口，也可以填写精确模型 ID。探测响应只保留语法安全的模型 ID 和一个小型状态枚举，供应商响应正文和凭据不会暴露。每个并发探测使用自己的请求局部 key，探测期间禁止 Start，并拒绝 HTTP 重定向，因此授权头不会跨来源传递。

## 浏览器与 HTTP 边界

页面没有外部脚本、样式表、字体或遥测。会改变状态的 POST 必须同时满足：

- 精确 `Host: 127.0.0.1:<绑定端口>`；
- 精确同源 `Origin`；
- HttpOnly、SameSite=Strict 的内存会话 cookie；
- `X-Anchor-CSRF: 1`；
- 一个严格 UTF-8、最大 16 KiB 的 `application/json` 请求体。

重复 JSON key、非有限数值、BOM、查询字符串、绝对请求目标、分块请求体、意外字段和 CORS 预检都会被拒绝。控制输入受工作区约束，并进行符号链接和范围检查。

JSONL 读取器会保留字节偏移和半行缓冲区，因此每次轮询不会重扫未改变的文件，并且只解码白名单元数据路径。畸形 JSONL 只以来源、行号和 SHA-256 表示，不会返回行片段。HTTP 响应使用 `no-store`、严格的内容安全策略、`nosniff` 和同源资源策略，并关闭默认请求日志。
