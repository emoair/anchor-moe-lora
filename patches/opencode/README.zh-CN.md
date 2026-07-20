# OpenCode v1.17.18 Anchor 补丁

[English](README.md)

本目录是魔改 OpenCode 二进制的 canonical 源码契约。`patch-manifest.json`
固定上游提交 `b1fc8113948b518835c2a39ece49553cffe9b30c`、Bun 1.3.14、补丁
SHA-256、必跑测试以及 `anchor.execution-tool-contract.v3`。

正式 v3 的隔离边界如下：

- 模型容器始终使用 `network=none`，唯一可写工作树固定挂载为 `/testbed`；
- CC Switch 位于模型容器之外。supervisor 持有 Unix socket，并只连接一个固定的
  私有/回环 CC Switch 目标；容器内桥只监听 `127.0.0.1:18080`；
- relay 拒绝 CONNECT、绝对 URL、重复或不精确的 Host，并只允许固定的
  health/models/responses/chat 路径；
- 模型容器及 route socket 销毁后，才允许启动 fresh 的离线导出或系统私有
  validator 容器；后者既没有 route socket，也没有本地 route token。

上游教师密钥只属于 CC Switch，绝不会挂载进模型容器。
`ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN=anchor-local-route` 只是固定、非机密的本地客户端
token；supervisor socket 被销毁后，它本身不能建立任何网络能力。

`anchor export`（包括 `exportSandboxed`）只导出原始会话/转录，不是正式评测
receipt，也不会生成 `validator_version_sha256`、`validation_state_sha256` 或最终
变更路径的精确哈希集合。只有可信 coordinator/supervisor 在终态 validator 覆盖
重新计算的最终 diff 且清理成功后，才能生成这些证明。

修改源码补丁不会自动升级仓库里的旧二进制和 bundle manifest。Windows 与 Linux
产物必须从此精确补丁重新构建，并把 v3 契约和新补丁哈希写入各自 manifest；旧 v2
二进制或 bundle 在此之前必须保持 `not-ready`。

## 可复现核验

在固定上游提交的干净 checkout 中执行：

```text
git apply --check v1.17.18-anchor-distillation.patch
git apply v1.17.18-anchor-distillation.patch
bun test test/anchor/sandbox.test.ts test/util/process.test.ts test/session/initial-tool-call.test.ts
bun run --cwd packages/opencode typecheck
```

需要生成可审计的 Windows/Linux 二进制时，请使用仓库内的双平台构建脚本。不要从本
目录发布 package、tag 或 release。

## 许可证

补丁基于 `patch-manifest.json` 固定的 OpenCode 上游版本。OpenCode 上游源码及补丁
所包含的上游片段继续遵循 `LICENSE.upstream` 中的 MIT 许可证和版权声明。

Anchor-MoE-LoRA 自行编写的外围代码和文档遵循仓库根目录的
AGPL-3.0-or-later；这不会替换或移除 OpenCode 上游 MIT 声明。
