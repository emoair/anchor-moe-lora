from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from .policy import ToolPolicy


DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "kimi-for-coding"
DEFAULT_VARIANT = "thinking"
PROVIDER_ID = "kimi-code"
AGENT_ID = "anchor-gold"


def validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        value != value.strip()
        or any(char.isspace() for char in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname != "api.kimi.com"
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base URL must be a literal official Kimi HTTPS URL without credentials, "
            "whitespace, query, or fragment"
        )
    return value.rstrip("/")


def build_opencode_config(
    policy: ToolPolicy,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, object]:
    """Build a keyless, fail-closed OpenCode configuration.

    The API key is intentionally referenced only through an environment variable.
    No User-Agent/header override is supplied: Kimi requires the real client identity.
    """

    base_url = validate_base_url(base_url)
    if not model or any(char.isspace() for char in model):
        raise ValueError("model must be a non-empty identifier")
    permission = policy.opencode_permissions()
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{PROVIDER_ID}/{model}",
        "default_agent": AGENT_ID,
        "share": "disabled",
        "lsp": False,
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Kimi Code (official OpenAI-compatible endpoint)",
                "options": {
                    "baseURL": base_url,
                    "apiKey": "{env:KIMI_CODE_API_KEY}",
                },
                "models": {
                    model: {
                        "name": "Kimi for Coding",
                        "limit": {"context": 262144, "output": 32768},
                        "variants": {
                            DEFAULT_VARIANT: {"reasoningEffort": "high"},
                        },
                    }
                },
            }
        },
        "permission": permission,
        "agent": {
            AGENT_ID: {
                "description": "Isolated coding task with a fail-closed tool policy",
                "mode": "primary",
                "steps": policy.max_iterations,
                "requireInitialToolCall": True,
                "permission": permission,
            }
        },
    }


def write_opencode_config(path: str | Path, policy: ToolPolicy) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_opencode_config(policy), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
