[English](teacher_providers.md) | [简体中文](teacher_providers.zh-CN.md)

# Teacher providers and model selection

The data subsystem supports OpenAI-compatible Chat Completions and Anthropic-compatible
Messages providers. A provider config stores only the environment-variable **name** that
contains a credential. It rejects `api_key`, `token`, `secret`, and `authorization` fields.
The project does not load `.env` files and never writes credential values to config,
provenance, status, or logs.

The local control panel is provider-neutral. A catalog preset is only a convenience:
the base URL, protocol, exact model ID, reasoning switch/effort, concurrency, retry,
reconnect, and budgets remain operator-controlled. Model discovery is optional; when
`GET .../models` is unsupported or fails, select **Force model** and enter the exact
model ID manually. A failed discovery never authorizes the panel to substitute a model.

For the current formal teacher profiles, both GLM 5.2 and Kimi-K3 use explicit
`reasoning_effort: max` for every stage. The generated run records that exact setting.
Selecting a lower effort or disabling reasoning for a formal MAX profile fails closed;
it does not silently downgrade the run. Custom/non-formal profiles may choose any
supported effort (`low`, `medium`, `high`, or `max`).

## Presets

| Preset | Protocol | Official/default base URL | Default model | Key environment variable |
| --- | --- | --- | --- | --- |
| `kimi-code-openai` | OpenAI | `https://api.kimi.com/coding/v1` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-code-anthropic` | Anthropic | `https://api.kimi.com/coding/` | `kimi-for-coding` | `KIMI_API_KEY` |
| `kimi-platform-openai` | OpenAI | `https://api.moonshot.cn/v1` | manual/discovered | `MOONSHOT_API_KEY` |
| `openai` | OpenAI | `https://api.openai.com/v1` | manual/discovered | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `https://api.anthropic.com` | manual/discovered | `ANTHROPIC_API_KEY` |
| `custom-openai` | OpenAI | required in config | manual/discovered | `TEACHER_API_KEY` |
| `custom-anthropic` | Anthropic | required in config | manual/discovered | `TEACHER_API_KEY` |

Kimi Code's official documentation specifies both base URLs and stable model IDs
`kimi-for-coding` and `kimi-for-coding-highspeed`. The ordinary preset intentionally
defaults to `kimi-for-coding`; select HighSpeed manually only when the subscription permits it.

## Discover, choose, or force a model

Set the key in the current shell, then list models without making a generation request:

```powershell
$env:TEACHER_API_KEY = "your-key"
py -m anchor_mvp.data models --provider custom-openai `
  --base-url https://gateway.example.com/v1 `
  --api-key-env TEACHER_API_KEY
```

Discovery calls the protocol-standard endpoint: OpenAI-compatible `GET <base>/models`
with Bearer authentication, or Anthropic-compatible `GET <base>/v1/models` with
`x-api-key` and `anthropic-version`. Output is sorted and indexed from zero. Run with a
specific model:

```powershell
py -m anchor_mvp.data run --config configs/data/provider.custom.example.yaml `
  --model provider-model-id --force-model
```

`--force-model` (or `force_model: true`) skips discovery. Without it,
`discover_models: true` records one of these public categories: `success`,
`missing_credential`, `auth_error`, `rate_limited`, `unsupported`, `server_error`,
`network_error`, or `invalid_response`. A failed discovery never prevents a manually
specified model. For non-interactive selection, use `model_index: N` only after successful
discovery; explicit `model` is safer when provider ordering may change.

Base URLs must be absolute `http://` or `https://` URLs with a hostname. Whitespace,
natural-language descriptions, embedded credentials, query strings, fragments, and full
`/models`, `/messages`, or `/chat/completions` endpoints are rejected before any request.

Every distilled record stores non-secret provider provenance: preset, selected model and
selection source, validated base URL, active protocol, and discovery category/model count.

## Optional quota capability

Quota lookup is informational and never blocks distillation:

```powershell
py -m anchor_mvp.data quota --provider kimi-platform-openai
```

The Kimi Open Platform documents `GET /v1/users/me/balance`, so only
`kimi-platform-openai` implements the `moonshot_balance` capability. Kimi Code membership
documentation directs users to its Console for remaining quota and rate-limit status and
does not document a stable public quota endpoint; Kimi Code presets therefore return
`unsupported`. OpenAI, Anthropic, and custom presets also return `unsupported` unless a
future stable official endpoint is explicitly implemented. Lookup errors return `error`
and do not alter generation behavior.

## Migration from existing configs

Existing flat configs using `protocol`, `base_url`, `model`, and `api_key_env` continue to
work and are interpreted as the matching Kimi Code preset when `provider` is absent.
Add `provider: kimi-code-anthropic` or `provider: kimi-code-openai` explicitly, then add
`force_model: true` to preserve the prior fixed-model/no-discovery behavior. Do not rename
the process environment variable unless the launching scripts are updated at the same time.

## Local panel and network-route semantics

Open the local control panel with `./anchor.ps1 ui`, then use
`http://127.0.0.1:8765/`. The credential field is RAM/process-only and is cleared from
the browser after Start, Continue, or model discovery. The child receives the selected
credential only through `ANCHOR_CONTROL_API_KEY`; it is not written to YAML, JSON,
argv, or logs.

The panel's default **direct** mode means “do not inherit proxy URL variables and set
`NO_PROXY=*`.” It does **not** bind a socket or process to a physical NIC and cannot
override an operating-system TUN/default route. The panel reports proxy/TUN default-route
detection and never labels this mode as physical-NIC-pinned. For domestic providers and
large (especially 10 GiB+) downloads, use the repository's dedicated route/download
preflight and verify an up physical adapter before transferring data. If the physical
route cannot be proven, stop rather than claiming proxy-free transfer.

The Windows CC Switch route component and WSL/Podman reachability are separate
gates. A hash-attested route executable is `component_ready`; it is not E2E-ready
until the formal coordinator proves, from the sandbox side, that the route is
reachable. Never infer container reachability from a Windows
`http://127.0.0.1:...` health check.

The 9,504/9,504 English/Chinese counts in the full-bank manifest describe only
locale routing. Chinese text is not complete until the zh-CN localization
manifest is produced and validated; that missing runtime output keeps
`training_ready=false` without blocking the initial `launch_ready=true` gate.

## Primary documentation

- [Kimi Code service endpoints, model IDs, and console quota guidance](https://www.kimi.com/code/docs/en/)
- [Kimi Open Platform balance endpoint](https://platform.kimi.com/docs/api/balance)
- [OpenAI model list endpoint](https://developers.openai.com/api/reference/resources/models/methods/list)
- [Anthropic model list endpoint](https://platform.claude.com/docs/en/api/models/list)
