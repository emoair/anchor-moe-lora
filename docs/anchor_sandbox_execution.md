# Anchor sandbox execution

The live execution-distillation harness invokes the patched OpenCode binary's
`anchor` subcommands. The binary remains the trusted host-side orchestrator: on this WSL2
host it starts rootful Podman under a resource-limited systemd unit; native Linux defaults
to direct rootless Podman. It streams the inner OpenCode
JSONL back to Python. Python never builds a shell command from a prompt, tool argument,
path, or resource setting.

## Prerequisites

- a WSL2 Linux distribution with Podman and systemd, or a native Linux host with rootless
  Podman configured for the intended non-root user;
- the repository-local, attested patched OpenCode binary;
- a pinned sandbox image and, when configured, an attested Linux OpenCode artifact;
- cgroup support sufficient for the selected systemd/Podman memory, CPU, and PID limits.

A dedicated sandbox distribution with automount and Windows-process interop disabled is
the stronger deployment. The current compatibility path still uses WSL path conversion,
but it mounts only the copied task workspace, per-run state, read-only config, and read-only
Linux OpenCode binary into the container; it never mounts `/mnt/c`, `/mnt/d`, a home
directory, SSH/Git credentials, or a Docker/Podman socket wholesale. WSL documents the
per-distribution controls in its [advanced configuration guide](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

## Operator configuration

`configs/tooling/opencode_distillation_ramp.yaml` defaults to one serialized stage:

```yaml
concurrency_stages: [1]
samples_per_stage: [1]
anchor_sandbox:
  linux_executable: artifacts/tooling/opencode-patched/linux-x64/opencode-anchor
  wsl_distro: Ubuntu-22.04
  supervisor: wsl-root-systemd
  memory: 4G
  cpus: 2
  pids: 256
  timeout_seconds: 900
retain_workspace: false
```

Stages are optional. When supplied, `concurrency_stages` and `samples_per_stage` must
be equally long non-empty lists of positive integers; no static `1,2,4,8` or `max=8`
limit exists in the Python layer. The safe default remains one job. Resource settings are
operator-owned config only and are passed as argv; model prompts, tool calls, and session
exports cannot select them. `linux_executable` is deliberately set to the bundled Linux
x64 artifact: Debian/WSL jobs must not attempt to run the Windows executable. The harness
removes each copied workspace only after validation and controlled-session capture finish;
set `retain_workspace: true` (or single-mode `--retain-workspace`) only for short-lived
local debugging. On this Windows host, `wsl_distro: Ubuntu-22.04` and
`supervisor: wsl-root-systemd` are required. The patched CLI accepts `supervisor: direct`
for a native Linux host; it derives that default only when the option is omitted.

The executor issues the following argument-vector shape, never `sh -c` or `bash -lc`:

```text
opencode anchor run --run-id <sample-id> --workspace <absolute-host-workspace>
  --config <absolute-host-config> [--linux-executable <absolute-host-artifact>]
  [--wsl-distro <name>] [--supervisor direct|wsl-root-systemd]
  [--memory 4G] [--cpus 2] [--pids 256] [--timeout 900]
  --model anchor-kimi/kimi-for-coding --agent anchor-distiller
  --variant medium --title <title> <prompt>
```

The matching controlled export is `opencode anchor export ... --session <ses_...>` and
receives the same `--linux-executable`, `--wsl-distro`, `--supervisor`, and resource
arguments as the corresponding run. On Windows, omitting `--wsl-distro` is invalid;
Linux may omit it. `--supervisor` is an explicit value, not a boolean switch.
Every finalizer, including conversion failures, calls `opencode anchor cleanup --run-id
<sample-id> --workspace <absolute-host-workspace>` and treats a non-zero cleanup exit as
a quarantined execution attempt. The Podman command itself must also be one-shot and
reap its container; a launcher crash still requires the anchor-side stale-job sweeper.

Container paths rooted at `/workspace` are normalized to `<workspace>` in retained public
records. Traversal such as `/workspace/../outside` remains a hard quarantine condition;
real host paths outside the copied fixture are never accepted as training data.

## Security boundary

This is a local, least-privilege development isolation layer, **not** a guarantee for
malicious or multi-tenant code. It does not defend against a compromised Windows host,
WSL kernel, Podman/image vulnerability, hardware side channel, or data deliberately made
available inside the job. A read-only mounted executable only prevents that specific file
from being written back; it does not protect credentials inherited through the environment
or files mounted beside it.

The patched launcher injects the provider token into the inner, single-use container so
OpenCode can make the model request. It must never appear in argv, generated config, the
workspace, session exports, Gold records, or logs. This weakens the isolation boundary:
untrusted code that can read its process environment may exfiltrate that token through
allowed network egress. Use a quota-limited, revocable distillation-only credential and
restrict egress to the approved provider endpoint or an authenticated egress proxy; do not
reuse a production key. The sandbox must receive no other production credential, host
socket, or real worktree. `--privileged`, host PID/IPC/network namespaces, device mappings,
unconfined seccomp, user-selected images, and arbitrary bind mounts are unsupported.
On native Linux, Podman's rootless mode maps container root through a user namespace. The
current Windows compatibility path instead uses rootful Podman because Ubuntu 22.04's
Podman 3.4 cgroup integration fails under this WSL kernel; its outer systemd unit applies
the limits while the container drops all capabilities, enables `no-new-privileges`, uses
a read-only root filesystem, and receives only disposable bind mounts. This remains a
weaker boundary than a dedicated VM or gVisor-style sandbox.
See [Podman rootless](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
and [`podman run`](https://docs.podman.io/en/latest/markdown/podman-run.1.html).

## Attribution and distribution

- [OpenCode](https://github.com/anomalyco/opencode) is upstream MIT-licensed. This fork
  retains its upstream MIT license and copyright notices; a distributed modified binary
  must ship the corresponding source/patch provenance and third-party notices.
- [Podman](https://github.com/containers/podman) is Apache-2.0. It is an external runtime
  prerequisite here, not a bundled component. If a release starts bundling its binary or
  code, include its license and any required notices in that release artifact.
- A pinned sandbox image and any Linux OpenCode artifact are supply-chain inputs. Record
  their digest/hash and SBOM or package-license inventory before publishing a runnable
  bundle.
