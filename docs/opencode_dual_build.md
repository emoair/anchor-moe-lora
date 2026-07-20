# OpenCode 双平台构建

本说明只覆盖仓库内 OpenCode 补丁产物的构建与校验；不涉及蒸馏执行器、训练或任何 API 调用。

## 当前 readiness 边界

canonical source patch 与 `patch-manifest.json` 已升级为
`anchor.execution-tool-contract.v3`。正式 v3 构建要求模型容器固定
`network=none`、工作目录固定 `/testbed`，并且只通过 supervisor 持有的
Unix socket 和容器内固定 loopback bridge 访问 CC Switch。构建脚本和 bundle
校验器会拒绝 v2 契约。

修改补丁不会自动升级仓库中已经存在的二进制。只要 Windows 与 Linux 两个平台
尚未从当前补丁重新构建并写入匹配的 v3 manifest，旧 v2 二进制与旧 bundle 就必须
保持 `not-ready`，不得用于正式蒸馏或 A–F 评测。

在 canonical OpenCode patch 更新并完成审计前，不要执行下面的构建命令。两个构建器会从同一 `patches/opencode/patch-manifest.json` 读取官方仓库、baseline commit、canonical patch、Bun 版本和必跑测试清单。

## 产物布局与来源约束

两个 target 必须使用独立、干净的 checkout 和独立的 Bun cache。构建器不会 `git reset`、`git clean` 或复用已应用补丁的脏目录；中断后请指定新的 checkout 路径。

bundle 成员使用对称、相对 artifact root 的路径：

```text
artifacts/tooling/opencode-patched/
  windows-x64/opencode-anchor.exe
  linux-x64/opencode-anchor
  windows-x64.manifest.json
  linux-x64.manifest.json
  bundle-manifest.json
```

根目录的 `opencode-anchor.exe` 与旧 `manifest.json` 是 Windows 运行器的兼容副本；它们不是 bundle 成员路径。`assemble_opencode_bundle.py` 会拒绝绝对路径、`..` 路径、二进制哈希不匹配，或两个 target 的 baseline/patch/Bun version/tool contract/lockfile SHA 不一致。

## Windows x64

Windows 构建器默认创建 `runs/opencode-build/worktrees/windows-x64`，并要求 Bun 1.3.14、node-gyp 13.0.1 和显式 Bun 路径。可选的 `-BunSha256` 会把本地 Bun 可执行文件固定到已审计字节。

```powershell
$repo = (Resolve-Path .).Path
$bun = Join-Path $repo 'runs\opencode-build\tools\bun-v1.3.14\bun-windows-x64\bun.exe'
$bunSha = (Get-FileHash -LiteralPath $bun -Algorithm SHA256).Hash.ToLowerInvariant()
$nodeGyp = Join-Path $repo 'runs\opencode-build\tools\node-gyp-13\node_modules\.bin\node-gyp.cmd'

.\scripts\tooling\build_patched_opencode.ps1 `
  -BunPath $bun `
  -BunSha256 $bunSha `
  -NodeGypPath $nodeGyp
```

## WSL Ubuntu Linux x64

Linux checkout 和 `node_modules` 必须在 WSL ext4（默认 `$HOME/.cache/...`），不要在 Windows DrvFS/9p 挂载目录上构建。固定 Bun 路径和可执行文件 SHA-256 示例为：

```bash
BUN_PATH="$HOME/.cache/anchor-moe-lora/toolchains/bun-1.3.14/bun"
BUN_SHA256=9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74

test "$(sha256sum "$BUN_PATH" | awk '{print $1}')" = "$BUN_SHA256"
"$BUN_PATH" --version  # 必须输出 1.3.14
```

确认 canonical patch 更新后，使用下列命令构建 Linux x64：

```bash
export REPO_ROOT=/path/to/anchor-moe-lora
export BUN_PATH="$HOME/.cache/anchor-moe-lora/toolchains/bun-1.3.14/bun"
export BUN_SHA256=9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74

bash "$REPO_ROOT/scripts/tooling/build_patched_opencode_wsl.sh" \
  --bun-path "$BUN_PATH" \
  --bun-sha256 "$BUN_SHA256"
```

该脚本还会预检 `git`、`python3`、`make`、`gcc`、`g++`，运行 manifest 指定的 focused tests、typecheck 和当前 Linux target 的离线 `--version` smoke test。它不会使用 Windows 的 `bun.exe`、node-gyp 或 `node_modules`。

已核验的官方 Linux x64 Bun ZIP archive 长度为 `35,969,274` bytes，SHA-256 为：

```bash
BUN_ZIP_SHA256=951ee2aee855f08595aeec6225226a298d3fea83a3dcd6465c09cbccdf7e848f
BUN_ZIP=/path/to/bun-linux-x64.zip

test "$(wc -c < "$BUN_ZIP" | tr -d '[:space:]')" = 35969274
test "$(sha256sum "$BUN_ZIP" | awk '{print $1}')" = "$BUN_ZIP_SHA256"
```

archive 校验与上面的 `BUN_SHA256` 是两个不同层级：前者固定下载 ZIP，后者固定已安装的 Linux Bun 可执行文件；重新下载、解压或替换任一文件时都应分别复核。

## 统一 bundle manifest

两个 platform manifest 和二进制都已生成后，再运行：

```bash
python3 "$REPO_ROOT/scripts/tooling/assemble_opencode_bundle.py" \
  --artifact-root "$REPO_ROOT/artifacts/tooling/opencode-patched"
```

只检查、不写入 `bundle-manifest.json`：

```bash
python3 "$REPO_ROOT/scripts/tooling/assemble_opencode_bundle.py" \
  --artifact-root "$REPO_ROOT/artifacts/tooling/opencode-patched" \
  --check
```
