# OpenCode v1.17.18 Anchor patch

[简体中文](README.zh-CN.md)

This directory is the canonical source contract for the modified OpenCode
binary. `patch-manifest.json` pins upstream commit
`b1fc8113948b518835c2a39ece49553cffe9b30c`, Bun 1.3.14, the patch digest,
focused tests, and `anchor.execution-tool-contract.v3`.

Formal v3 changes the sandbox boundary in four material ways:

- the model container is always `network=none` and its only writable worktree is
  the canonical `/testbed` bind mount;
- CC Switch stays outside the model container. A supervisor-owned Unix socket
  reaches one fixed private/loopback CC Switch target, while an internal bridge
  exposes only `127.0.0.1:18080` inside the container;
- the relay rejects CONNECT, absolute-form URLs, duplicate or non-exact Host
  headers, and every path except the fixed health/models/responses/chat paths;
- model container and route socket are destroyed before a fresh offline export
  or system-private validator container. The latter receives neither the route
  socket nor the local route token.

The upstream provider credential belongs to CC Switch and is never mounted into
the model container. `ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN=anchor-local-route` is a
non-secret, fixed local client token and is useful only while the supervisor
socket exists.

`anchor export` (including `exportSandboxed`) is only a raw session/transcript
export. It is not an official-evaluation receipt and does not emit
`validator_version_sha256`, `validation_state_sha256`, or an exact hash set for
the final changed paths. Those attestations are produced only by the trusted
coordinator/supervisor after a terminal validator covers the recomputed final
diff and cleanup succeeds.

The checked-in binaries and bundle manifests are not automatically upgraded by
changing this patch. A v2 binary or bundle must remain not-ready until both
platforms are rebuilt from this exact patch and their manifests bind the v3
contract and new patch SHA-256.

## Reproducible verification

From a clean checkout of the pinned upstream commit:

```text
git apply --check v1.17.18-anchor-distillation.patch
git apply v1.17.18-anchor-distillation.patch
bun test test/anchor/sandbox.test.ts test/util/process.test.ts test/session/initial-tool-call.test.ts
bun run --cwd packages/opencode typecheck
```

Use the repository build launchers for attested Windows/Linux artifacts; do not
publish a package, tag, or release from this directory.

## License notices

The patch files in this directory describe Anchor-MoE-LoRA changes against the
pinned OpenCode source revision recorded in `patch-manifest.json`. OpenCode's
upstream source and the portions represented by these patches remain covered by
the upstream MIT license and copyright notice in `LICENSE.upstream`.

Anchor-MoE-LoRA-authored surrounding code and documentation are distributed
under this repository's AGPL-3.0-or-later license. Nothing in the repository's
root license removes or replaces the upstream MIT notice for OpenCode-derived
material.
