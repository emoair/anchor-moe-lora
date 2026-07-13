# CC Switch metadata adapter

This package exposes a small, secret-free metadata input for the distillation
control plane. It does not install, launch, import, or parse the CC Switch
application. The only network operation is hashing a fixed allowlist of official
GitHub raw files at commit `8d1b3306d09a27b9d8fc29694791d8421aba5f93`.

CC Switch v3.16.5 does not publish a stable provider/pricing JSON feed. Therefore
the adapter ships an explicitly reviewed fixture. A network check proves that the
pinned source anchors still have their expected byte lengths and SHA-256 values;
it does not turn TypeScript or Rust into live configuration.

## Commands

```powershell
$state = "$HOME/.anchor-moe-lora/ccswitch-metadata"
python -m anchor_mvp.integrations.ccswitch_metadata check --state-dir $state
python -m anchor_mvp.integrations.ccswitch_metadata diff --state-dir $state
python -m anchor_mvp.integrations.ccswitch_metadata apply --state-dir $state
python -m anchor_mvp.integrations.ccswitch_metadata rollback --state-dir $state
```

Add `--offline` to `check`, `diff`, or `apply` to use the last verified snapshot,
or the bundled reviewed snapshot when no network-verified copy exists. A network
timeout also falls back this way. A size or SHA-256 mismatch is an integrity error
and never falls back silently.

The default active output is `STATE_DIR/active.json`; pass `--target PATH` when a
different consumer path is required. Every command emits one content-safe JSON
object. `apply` writes a rollback journal and replaces the target atomically.

## Consumer contract

- Parse the JSON; do not inject it as HTML.
- Treat provider `base_url` values and model IDs as selectable defaults, not proof
  that an account can access them.
- Store only an environment-variable or OS-vault reference beside a selected
  provider. The snapshot intentionally has no key/header/token fields.
- Resolve only `model_aliases` whose `match` is `exact`.
- Display currency and `per_1m_tokens` beside every estimate.
- If any billable dimension is `unknown`, show cost as unavailable. Never render
  unknown as zero.
- Provider/model discovery is a separate runtime operation. Keep it same-origin
  with the selected provider and never send a bearer credential to a user-supplied
  discovery URL.

The helper `estimate_cost()` uses `Decimal`, preserves the OpenAI-compatible vs.
Anthropic cache-input distinction recorded in the snapshot, and returns an
explicit unavailable result for unknown models or prices.

## Provider/channel pricing overlay

Current provider prices that are not present in CC Switch v3.16.5 live in a
separate, project-owned audited overlay. In particular, GLM-5.2 pay-as-you-go
pricing is scoped to the exact Zhipu endpoint, while Volcengine Ark Coding Plan is
marked `subscription_quota` with unavailable marginal token cost. An Ark model ID
can never select Zhipu rates by model name alone. Cache storage prices with
per-hour units are also kept separate from the token estimator's `cache_write`
dimension. See `docs/glm52_provider_pricing.md`.

## State and trust

- `verification_cache.json` stores ETags and verified hashes, not source bodies.
- `last_verified_snapshot.json` is the last snapshot accepted after all source
  anchors passed verification.
- `backup_index.json` and `backups/` implement target-specific rollback.
- `.sync.lock` prevents concurrent apply/rollback. After an abnormal process exit,
  inspect that no sync process is alive before removing a stale lock.
- Directories are set to mode `0700` and files to `0600` where the OS supports
  POSIX permissions. No secret is expected in any of them.

Upstream attribution and the MIT text are in `NOTICE.txt`. The broader integration
decision record is in `docs/cc_switch_v3_16_5_integration.md`.
