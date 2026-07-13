"""Secret-safe teacher provider presets, discovery, selection, and capabilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


ProviderProtocol = Literal["openai", "openai_responses", "anthropic"]
DiscoveryStatus = Literal[
    "success",
    "skipped",
    "skipped_force_model",
    "missing_credential",
    "auth_error",
    "rate_limited",
    "unsupported",
    "server_error",
    "network_error",
    "invalid_response",
]


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    protocol: ProviderProtocol
    base_url: str | None
    default_model: str | None
    api_key_env: str
    quota_capability: str | None = None


@dataclass(frozen=True)
class ProviderSpec:
    preset: str
    protocol: ProviderProtocol
    base_url: str
    api_key_env: str
    default_model: str | None
    quota_capability: str | None


@dataclass(frozen=True)
class ModelDiscovery:
    status: DiscoveryStatus
    endpoint: str | None
    models: tuple[str, ...] = ()
    http_status: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "endpoint": self.endpoint,
            "model_count": len(self.models),
            "models": list(self.models),
            "choices": [
                {"index": index, "id": model} for index, model in enumerate(self.models)
            ],
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class ProviderSelection:
    spec: ProviderSpec
    model: str
    model_source: str
    discovery: ModelDiscovery

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "preset": self.spec.preset,
            "protocol": self.spec.protocol,
            "base_url": self.spec.base_url,
            "api_key_env": self.spec.api_key_env,
            "model": self.model,
            "model_source": self.model_source,
            "discovery": self.discovery.to_public_dict(),
        }


PRESETS: dict[str, ProviderPreset] = {
    "kimi-code-openai": ProviderPreset(
        name="kimi-code-openai",
        protocol="openai",
        base_url="https://api.kimi.com/coding/v1",
        default_model="kimi-for-coding",
        api_key_env="KIMI_API_KEY",
    ),
    "kimi-code-anthropic": ProviderPreset(
        name="kimi-code-anthropic",
        protocol="anthropic",
        base_url="https://api.kimi.com/coding/",
        default_model="kimi-for-coding",
        api_key_env="KIMI_API_KEY",
    ),
    "kimi-platform-openai": ProviderPreset(
        name="kimi-platform-openai",
        protocol="openai",
        base_url="https://api.moonshot.cn/v1",
        default_model=None,
        api_key_env="MOONSHOT_API_KEY",
        quota_capability="moonshot_balance",
    ),
    "openai": ProviderPreset(
        name="openai",
        protocol="openai",
        base_url="https://api.openai.com/v1",
        default_model=None,
        api_key_env="OPENAI_API_KEY",
    ),
    "anthropic": ProviderPreset(
        name="anthropic",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        default_model=None,
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "custom-openai": ProviderPreset(
        name="custom-openai",
        protocol="openai",
        base_url=None,
        default_model=None,
        api_key_env="TEACHER_API_KEY",
    ),
    "custom-openai-responses": ProviderPreset(
        name="custom-openai-responses",
        protocol="openai_responses",
        base_url=None,
        default_model=None,
        api_key_env="TEACHER_API_KEY",
    ),
    "custom-anthropic": ProviderPreset(
        name="custom-anthropic",
        protocol="anthropic",
        base_url=None,
        default_model=None,
        api_key_env="TEACHER_API_KEY",
    ),
}

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_CONFIG_KEYS = frozenset(
    {"api_key", "apikey", "secret", "token", "authorization"}
)


def reject_inline_secrets(config: Mapping[str, Any]) -> None:
    """Reject credential values in config; environment-variable names remain allowed."""

    offending = sorted(
        str(key) for key in config if str(key).casefold() in _SECRET_CONFIG_KEYS
    )
    if offending:
        raise ValueError(
            "credentials must not be stored in config; use api_key_env to name a process "
            f"environment variable (found: {', '.join(offending)})"
        )


def validate_base_url(value: str, *, name: str = "base_url") -> str:
    """Return a canonical HTTP(S) base URL or reject ambiguous/natural-language input."""

    candidate = value.strip()
    if (
        candidate != value
        or not candidate
        or any(character.isspace() for character in candidate)
    ):
        raise ValueError(f"{name} must be one absolute HTTP(S) URL without whitespace")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{name} must start with http:// or https://")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"{name} must contain a hostname and must not contain credentials"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain a query string or fragment")
    if parsed.path.casefold().endswith(
        ("/chat/completions", "/messages", "/models", "/responses")
    ):
        raise ValueError(f"{name} must be a base URL, not a full API endpoint")
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc, parsed.path or "", "", "")
    )


def provider_spec(
    config: Mapping[str, Any],
    *,
    preset_name: str | None = None,
    protocol: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> ProviderSpec:
    """Resolve a new preset or a backward-compatible flat teacher config."""

    reject_inline_secrets(config)
    requested_preset = preset_name or _optional_text(config.get("provider"))
    if requested_preset is None:
        # Migration path: historical flat configs were Kimi Code protocol configs.
        legacy_protocol = str(protocol or config.get("protocol", "anthropic"))
        if legacy_protocol == "anthropic":
            requested_preset = "kimi-code-anthropic"
        elif legacy_protocol == "openai":
            requested_preset = "kimi-code-openai"
        elif legacy_protocol == "openai_responses":
            requested_preset = "custom-openai-responses"
        else:
            raise ValueError("protocol must be anthropic, openai, or openai_responses")
    if requested_preset not in PRESETS:
        raise ValueError(
            f"unknown provider preset {requested_preset!r}; choose one of {sorted(PRESETS)}"
        )
    preset = PRESETS[requested_preset]
    resolved_protocol = str(protocol or config.get("protocol", preset.protocol))
    if resolved_protocol not in {"openai", "openai_responses", "anthropic"}:
        raise ValueError("protocol must be anthropic, openai, or openai_responses")
    if resolved_protocol != preset.protocol:
        raise ValueError(
            f"provider preset {requested_preset} requires protocol {preset.protocol}; "
            f"choose a {resolved_protocol} preset instead"
        )
    resolved_base = (
        base_url or _optional_text(config.get("base_url")) or preset.base_url
    )
    if resolved_base is None:
        raise ValueError(f"base_url is required for provider preset {requested_preset}")
    resolved_env = (
        api_key_env or _optional_text(config.get("api_key_env")) or preset.api_key_env
    )
    if not _ENV_NAME.fullmatch(resolved_env):
        raise ValueError(
            "api_key_env must be a valid process environment-variable name"
        )
    validated_base = validate_base_url(resolved_base)
    quota_capability = (
        preset.quota_capability
        if preset.base_url is not None
        and validated_base == validate_base_url(preset.base_url)
        else None
    )
    return ProviderSpec(
        preset=requested_preset,
        protocol=resolved_protocol,  # type: ignore[arg-type]
        base_url=validated_base,
        api_key_env=resolved_env,
        default_model=preset.default_model,
        quota_capability=quota_capability,
    )


def discover_models(
    spec: ProviderSpec, *, timeout_seconds: float = 20.0
) -> ModelDiscovery:
    """Fetch the provider's official protocol model-list endpoint without leaking its key."""

    endpoint = model_list_endpoint(spec.base_url, spec.protocol)
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return ModelDiscovery("missing_credential", endpoint)
    headers = {"Accept": "application/json", "User-Agent": "anchor-moe-lora/0.1"}
    if spec.protocol in {"openai", "openai_responses"}:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    request = Request(endpoint, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - validated URL
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        status = error.code
        if status in {401, 403}:
            category: DiscoveryStatus = "auth_error"
        elif status == 429:
            category = "rate_limited"
        elif status in {404, 405, 501}:
            category = "unsupported"
        elif status >= 500:
            category = "server_error"
        else:
            category = "invalid_response"
        return ModelDiscovery(category, endpoint, http_status=status)
    except (OSError, TimeoutError, URLError):
        return ModelDiscovery("network_error", endpoint)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ModelDiscovery("invalid_response", endpoint)
    models = _model_ids(payload)
    if not models:
        return ModelDiscovery("invalid_response", endpoint)
    return ModelDiscovery("success", endpoint, tuple(models))


def select_provider_model(
    spec: ProviderSpec,
    *,
    requested_model: str | None,
    discover: bool,
    force_model: bool,
    model_index: int | None = None,
    timeout_seconds: float = 20.0,
) -> ProviderSelection:
    """Select an explicit/default/discovered model with deterministic fallback behavior."""

    explicit = _optional_text(requested_model)
    if force_model:
        model = explicit or spec.default_model
        if model is None:
            raise ValueError(
                "--force-model requires --model for a provider without a default"
            )
        return ProviderSelection(
            spec,
            model,
            "manual" if explicit else "preset_default",
            ModelDiscovery("skipped_force_model", None),
        )
    discovery = (
        discover_models(spec, timeout_seconds=timeout_seconds)
        if discover
        else ModelDiscovery("skipped", None)
    )
    if model_index is not None:
        if discovery.status != "success":
            raise ValueError("model_index requires successful model discovery")
        if model_index < 0 or model_index >= len(discovery.models):
            raise ValueError(
                f"model_index must be between 0 and {len(discovery.models) - 1}"
            )
        return ProviderSelection(
            spec, discovery.models[model_index], "discovered_index", discovery
        )
    if explicit:
        return ProviderSelection(spec, explicit, "manual", discovery)
    if spec.default_model:
        return ProviderSelection(spec, spec.default_model, "preset_default", discovery)
    if discovery.status == "success":
        raise ValueError(
            "models were discovered; choose one with --model or --model-index"
        )
    raise ValueError(
        "no model selected; specify --model (discovery failure never blocks manual selection)"
    )


def query_quota(spec: ProviderSpec, *, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Query an optional documented provider capability; never gate generation on it."""

    if spec.quota_capability != "moonshot_balance":
        return {
            "status": "unsupported",
            "capability": None,
            "provider": spec.preset,
            "reason": "no stable official quota API is configured for this preset",
        }
    endpoint = _append_path(spec.base_url, "users/me/balance")
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return {
            "status": "missing_credential",
            "capability": spec.quota_capability,
            "provider": spec.preset,
        }
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - validated URL
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return {
            "status": "error",
            "capability": spec.quota_capability,
            "provider": spec.preset,
            "http_status": error.code,
        }
    except (OSError, TimeoutError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status": "error",
            "capability": spec.quota_capability,
            "provider": spec.preset,
        }
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping):
        return {
            "status": "error",
            "capability": spec.quota_capability,
            "provider": spec.preset,
        }
    allowed = ("available_balance", "voucher_balance", "cash_balance")
    return {
        "status": "success",
        "capability": spec.quota_capability,
        "provider": spec.preset,
        "balance": {name: data[name] for name in allowed if name in data},
    }


def model_list_endpoint(base_url: str, protocol: ProviderProtocol) -> str:
    base = validate_base_url(base_url)
    if protocol == "anthropic" and not urlsplit(base).path.rstrip("/").endswith("/v1"):
        return _append_path(base, "v1/models")
    return _append_path(base, "models")


def _append_path(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        return []
    identifiers = {
        str(item["id"]).strip()
        for item in payload["data"]
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    }
    return sorted(identifiers)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
