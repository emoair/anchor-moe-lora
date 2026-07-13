from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlparse

from .policy import ToolPolicy


DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "kimi-for-coding"
DEFAULT_VARIANT = "medium"
DEFAULT_OUTPUT_TOKENS = 32768
DEFAULT_HEADER_TIMEOUT_MS = 30_000
DEFAULT_CHUNK_TIMEOUT_MS = 60_000
PROVIDER_ID = "anchor-kimi"
AGENT_ID = "anchor-distiller"

# The patched sandbox currently materializes its one-use Podman secret under
# this child-only name.  ``key_env`` below is the configurable host-side source;
# OpenCode config files intentionally reference only this value-less alias.
SANDBOX_API_KEY_ENV = "KIMI_CODE_API_KEY"
SUPPORTED_PROVIDER_NPM = frozenset({"@ai-sdk/openai-compatible", "@ai-sdk/openai"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ROUTE_HOST = re.compile(
    r"^(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


def validate_route_host(value: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _ROUTE_HOST.fullmatch(value)
    ):
        raise ValueError(
            "route_host must be a literal DNS host without a scheme or path"
        )
    return value.casefold()


def validate_base_url(value: str, *, route_host: str = "api.kimi.com") -> str:
    expected_host = validate_route_host(route_host)
    parsed = urlparse(value)
    if (
        value != value.strip()
        or any(char.isspace() for char in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.hostname.casefold() != expected_host
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base URL must be a literal HTTPS URL for route_host without credentials, "
            "whitespace, query, or fragment"
        )
    return value.rstrip("/")


@dataclass(frozen=True)
class OpenCodeProvider:
    """Audited provider coordinates; credentials remain process-local.

    The seven fields deliberately match the live-batch YAML contract.  Provider
    package names are allowlisted so a configuration file cannot make OpenCode
    install an arbitrary npm package in a live sandbox.
    """

    provider_id: str = PROVIDER_ID
    npm: str = "@ai-sdk/openai-compatible"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    variant: str = DEFAULT_VARIANT
    key_env: str = SANDBOX_API_KEY_ENV
    route_host: str = "api.kimi.com"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.provider_id):
            raise ValueError("provider_id must be a non-empty OpenCode identifier")
        if self.npm not in SUPPORTED_PROVIDER_NPM:
            raise ValueError(
                "provider npm must be an audited OpenAI-compatible or OpenAI Responses package"
            )
        if (
            not self.model
            or self.model != self.model.strip()
            or any(char.isspace() or ord(char) < 32 for char in self.model)
        ):
            raise ValueError("model must be a non-empty identifier")
        if not _IDENTIFIER.fullmatch(self.variant):
            raise ValueError("variant must be a non-empty OpenCode identifier")
        if not _ENVIRONMENT_NAME.fullmatch(self.key_env):
            raise ValueError("key_env must be a valid environment variable name")
        route_host = validate_route_host(self.route_host)
        base_url = validate_base_url(self.base_url, route_host=route_host)
        object.__setattr__(self, "route_host", route_host)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_mapping(cls, value: object) -> "OpenCodeProvider":
        if not isinstance(value, Mapping):
            raise ValueError("provider must be an object")
        names = {
            "provider_id",
            "npm",
            "base_url",
            "model",
            "variant",
            "key_env",
            "route_host",
        }
        unknown = sorted(set(value) - names)
        missing = sorted(names - set(value))
        if unknown:
            raise ValueError(
                "provider contains unverified fields: " + ", ".join(map(str, unknown))
            )
        if missing:
            raise ValueError("provider is missing fields: " + ", ".join(missing))
        return cls(**{name: str(value[name]) for name in names})

    @property
    def is_responses(self) -> bool:
        return self.npm == "@ai-sdk/openai"

    @property
    def context_tokens(self) -> int:
        return 128_000 if self.is_responses else 262_144

    @property
    def model_name(self) -> str:
        return "Kimi for Coding" if self == DEFAULT_PROVIDER else self.model

    @property
    def provider_name(self) -> str:
        if self == DEFAULT_PROVIDER:
            return "Kimi Code (official OpenAI-compatible endpoint)"
        return f"Anchor audited provider ({self.provider_id})"


DEFAULT_PROVIDER = OpenCodeProvider()


def _provider_options(provider: OpenCodeProvider) -> dict[str, object]:
    result: dict[str, object] = {
        "baseURL": provider.base_url,
        "apiKey": f"{{env:{SANDBOX_API_KEY_ENV}}}",
    }
    if provider.is_responses:
        # This is consumed by OpenCode's transform and must not reach the wire.
        # It suppresses the otherwise automatic prompt_cache_key.  Any other
        # Responses fields are checked by the offline wire probe before live use.
        result["setCacheKey"] = False
    else:
        result.update(
            {
                "includeUsage": False,
                "headerTimeout": DEFAULT_HEADER_TIMEOUT_MS,
                "chunkTimeout": DEFAULT_CHUNK_TIMEOUT_MS,
            }
        )
    return result


def build_opencode_config(
    policy: ToolPolicy,
    *,
    provider: OpenCodeProvider = DEFAULT_PROVIDER,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    """Build a keyless, fail-closed OpenCode configuration.

    ``base_url`` and ``model`` remain as compatibility shims for older callers;
    new code should supply one immutable :class:`OpenCodeProvider`.
    """

    if base_url is not None or model is not None:
        provider = replace(
            provider,
            base_url=base_url if base_url is not None else provider.base_url,
            model=model if model is not None else provider.model,
        )
    permission = policy.opencode_permissions()
    agent: dict[str, object] = {
        "description": "Isolated coding task with a fail-closed tool policy",
        "mode": "primary",
        "permission": permission,
    }
    if policy.max_iterations is not None:
        agent["steps"] = policy.max_iterations
    model_config: dict[str, object] = {
        "name": provider.model_name,
        "reasoning": True,
        "limit": {
            "context": provider.context_tokens,
            "output": DEFAULT_OUTPUT_TOKENS,
        },
        "variants": {
            provider.variant: {"reasoningEffort": provider.variant},
        },
    }
    if not provider.is_responses:
        model_config["interleaved"] = {"field": "reasoning_content"}
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider.provider_id}/{provider.model}",
        "default_agent": AGENT_ID,
        "share": "disabled",
        "lsp": False,
        "provider": {
            provider.provider_id: {
                "npm": provider.npm,
                "name": provider.provider_name,
                "options": _provider_options(provider),
                "models": {provider.model: model_config},
            }
        },
        "permission": permission,
        "agent": {
            AGENT_ID: agent,
        },
    }


def write_opencode_config(
    path: str | Path,
    policy: ToolPolicy,
    *,
    provider: OpenCodeProvider = DEFAULT_PROVIDER,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_opencode_config(policy, provider=provider),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
