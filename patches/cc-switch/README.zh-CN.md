# 可审计的 CC Switch 路由补丁

本目录保存针对 CC Switch `v3.16.5`、提交
`8d1b3306d09a27b9d8fc29694791d8421aba5f93` 的 Anchor 补丁。它不是整仓复制，
也不冒充上游发布的二进制。

补丁新增无界面的 `anchor-opencode-route` 和独立逻辑应用类型
`anchor-opencode`。启动时必须指定隔离的 `CC_SWITCH_TEST_HOME`，因此不会复用
用户日常 CC Switch 数据库，也不会占用个人代理的 15721 端口。默认本地入口为：

- API：`http://127.0.0.1:15731/anchor/v1`
- Responses：`/anchor/v1/responses`
- Chat Completions：`/anchor/v1/chat/completions`
- 不含样本正文的存活/状态接口：`/anchor/health`、`/anchor/status`

协议、Base URL、模型 ID、模型发现策略、reasoning 字段与档位、价格、重试次数、
端口和 User-Agent 都由 profile 或启动参数决定，没有写死 GLM/Kimi。密钥值只存在于
启动进程的环境变量；profile、SQLite、manifest、patch 和日志只保存环境变量名称。
`/models` 发现是可选能力，`manual` 与 `force_manual_model` 是模型列表缺失或不可信时的
显式兜底。

网络策略也是显式配置：

- `direct`：默认值。隔离进程会清除继承的应用层代理变量并设置 `NO_PROXY=*`；这一步
  本身不能绕过 TUN。正式国内供应商 profile 还会设置 `require_physical_route=true`，
  启动器解析供应商实际路由，若命中非物理网卡便失败闭锁；启动器不会修改系统路由。
- `proxy`：只在 profile 中保存代理 URL 的环境变量名；代理 URL 本身不落盘。
- `inherit`：保留启动环境，沿用当前进程的常规代理行为。

国内供应商，以及模型、数据集等大文件传输，默认应选 `direct` 并通过物理路由预检，
只有操作者明确要求时才走 `proxy`。构建和验证路由不会下载模型或数据集。

`reasoning.effort=max` 在协议转换完成后注入，属于硬约束：CC Switch 不得把 `max`
降成 `high`/`xhigh`，也不得吞掉。仓内 GLM-5.2 和 Kimi-K3 正式 profile 都锁定这一点。
只有供应商明确要求顶层 `reasoning_effort` 时，才修改字段形式。

本轮正式 `kimi-k3-max` 教师配置使用用户指定的 Ark Coding Responses 地址和
`ARK_CODING_API_KEY`。通用 schema 与启动器仍可通过自定义 profile 接入 Kimi Code
或任意兼容 URL；Rust 路由层没有写死供应商身份。

构建与验收：

```powershell
pwsh -File scripts/tooling/build_patched_ccswitch.ps1
py scripts/tooling/validate_ccswitch_route.py --require-ready
```

先把密钥放入当前进程环境，再启动：

```powershell
pwsh -File scripts/tooling/start_patched_ccswitch_route.ps1 `
  -ProfilePath patches/cc-switch/profiles/glm-5.2-max.json
```

只有固定补丁成功应用、Rust 测试与真实二进制行为测试通过、二进制哈希写入 manifest
之后，`ready` 才能为 `true`；否则正式蒸馏必须失败闭锁。

上游采用 MIT 许可。致谢与来源见 `THIRD_PARTY_NOTICES.md`：
https://github.com/farion1231/cc-switch
