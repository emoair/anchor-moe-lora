# Auditable CC Switch route patch

This directory contains an Anchor-specific patch against CC Switch `v3.16.5`
at commit `8d1b3306d09a27b9d8fc29694791d8421aba5f93`. It is not a fork snapshot and
does not vendor upstream binaries.

The patch adds a headless `anchor-opencode-route` binary and the independent
logical application type `anchor-opencode`. Its SQLite state is isolated by a
mandatory `CC_SWITCH_TEST_HOME`, so it never reuses the user's normal CC Switch
database or the personal proxy on port 15721. The default local route is:

- API base: `http://127.0.0.1:15731/anchor/v1`
- Responses: `/anchor/v1/responses`
- Chat: `/anchor/v1/chat/completions`
- content-free liveness/status: `/anchor/health`, `/anchor/status`

All upstream parameters are runtime data. A profile can select protocol, base
URL, model ID, model discovery policy, reasoning field/effort, price data,
retry count, port, and User-Agent. The secret itself is never written to a
profile, database, manifest, patch, or log: only its environment-variable name
is persisted. Model discovery is optional; `manual` and `force_manual_model`
remain the fail-safe path when `/models` is absent or misleading.

Network policy is explicit and auditable:

- `direct` (default) clears inherited application-proxy variables and sets
  `NO_PROXY=*` in the isolated route process. This alone does not bypass a TUN.
  Checked-in domestic-provider profiles therefore set
  `require_physical_route=true`; the launcher resolves the provider route and
  fails closed when it selects a non-physical adapter. It never edits routes.
- `proxy` reads the proxy URL from the environment-variable name in
  `network.proxy_url_env`; the URL itself is not stored in the profile.
- `inherit` preserves the launcher's environment and follows its normal proxy
  behavior.

Use `direct` plus the physical-route preflight for mainland-China providers and
for model/dataset or other large resource transfers unless an operator
explicitly chooses a proxy. The route does not download any model or dataset as
part of build or validation.

`reasoning.effort=max` is applied as a post-transform body override. This is a
hard contract: CC Switch must forward the literal `max`; it must not silently
map it to `high`/`xhigh` or remove it. The checked-in GLM-5.2 and Kimi-K3
profiles both lock this behavior. Change `reasoning.field` only when a provider
requires the top-level `reasoning_effort` form.

The formal `kimi-k3-max` teacher profile intentionally uses the user-selected
Ark Coding Responses endpoint and `ARK_CODING_API_KEY`. The generic schema and
launcher still support Kimi Code or any other compatible URL through a custom
profile; provider identities are not hard-coded into the Rust route.

Build and verify:

```powershell
pwsh -File scripts/tooling/build_patched_ccswitch.ps1
py scripts/tooling/validate_ccswitch_route.py --require-ready
```

Start with a process-only credential already set:

```powershell
pwsh -File scripts/tooling/start_patched_ccswitch_route.ps1 `
  -ProfilePath patches/cc-switch/profiles/glm-5.2-max.json
```

The build script never installs Rust automatically. Until the pinned patch is
applied, tests pass, and the binary hash is recorded, the route manifest stays
`ready=false`; distillation must fail closed.

Upstream is MIT licensed. See `THIRD_PARTY_NOTICES.md` and the upstream project:
https://github.com/farion1231/cc-switch
