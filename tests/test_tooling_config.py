import json
from pathlib import Path

import pytest

from anchor_mvp.tooling import OpenCodeProvider, ToolPolicy, build_opencode_config
from anchor_mvp.tooling.config import SANDBOX_API_KEY_ENV, validate_base_url


ROOT = Path(__file__).resolve().parents[1]


def test_keyless_kimi_config_matches_checked_in_example():
    generated = build_opencode_config(ToolPolicy())
    example = json.loads(
        (ROOT / "configs" / "tooling" / "opencode_kimi.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert generated == example
    serialized = json.dumps(generated)
    assert "sk-" not in serialized
    assert generated["provider"]["anchor-kimi"]["options"] == {
        "baseURL": "https://api.kimi.com/coding/v1",
        "apiKey": "{env:KIMI_CODE_API_KEY}",
        "includeUsage": False,
        "headerTimeout": 30000,
        "chunkTimeout": 60000,
    }
    assert "headers" not in generated["provider"]["anchor-kimi"]["options"]
    assert "steps" not in generated["agent"]["anchor-distiller"]
    assert "requireInitialToolCall" not in generated["agent"]["anchor-distiller"]
    assert generated["share"] == "disabled"
    model = generated["provider"]["anchor-kimi"]["models"]["kimi-for-coding"]
    assert model["reasoning"] is True
    assert model["interleaved"] == {"field": "reasoning_content"}
    assert model["limit"] == {"context": 262144, "output": 32768}
    assert model["variants"] == {
        "medium": {"reasoningEffort": "medium"},
    }


def test_explicit_agent_step_limit_is_forwarded_without_an_unbounded_sentinel():
    generated = build_opencode_config(ToolPolicy(max_iterations=7))

    assert generated["agent"]["anchor-distiller"]["steps"] == 7


def test_invalid_or_descriptive_urls_are_rejected_before_requests():
    for value in (
        "the repo for the contents of the path",
        "api.kimi.com/coding/v1",
        "https://the repo for the contents of the path",
        "http://api.kimi.com/coding/v1",
    ):
        try:
            validate_base_url(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"URL should have been rejected: {value}")


def _ark_provider() -> OpenCodeProvider:
    return OpenCodeProvider(
        provider_id="anchor-ark-glm52",
        npm="@ai-sdk/openai",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="glm-5-2-260617",
        variant="max",
        key_env="ARK_CODING_API_KEY",
        route_host="ark.cn-beijing.volces.com",
    )


def test_ark_glm52_uses_responses_with_max_and_a_value_less_sandbox_key_alias():
    provider = _ark_provider()
    generated = build_opencode_config(ToolPolicy(), provider=provider)
    configured = generated["provider"][provider.provider_id]

    assert generated["model"] == "anchor-ark-glm52/glm-5-2-260617"
    assert configured["npm"] == "@ai-sdk/openai"
    assert configured["options"] == {
        "baseURL": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "apiKey": f"{{env:{SANDBOX_API_KEY_ENV}}}",
        "setCacheKey": False,
    }
    model = configured["models"][provider.model]
    assert model["variants"] == {"max": {"reasoningEffort": "max"}}
    assert model["limit"] == {"context": 128000, "output": 32768}
    assert "interleaved" not in model
    serialized = json.dumps(generated)
    assert "ARK_CODING_API_KEY" not in serialized
    assert "reasoningSummary" not in serialized
    assert '"store"' not in serialized
    assert '"include"' not in serialized


def test_provider_contract_rejects_unverified_fields_and_route_mismatch():
    value = {
        "provider_id": "anchor-ark-glm52",
        "npm": "@ai-sdk/openai",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "glm-5-2-260617",
        "variant": "max",
        "key_env": "ARK_CODING_API_KEY",
        "route_host": "ark.cn-beijing.volces.com",
    }
    with pytest.raises(ValueError, match="unverified fields"):
        OpenCodeProvider.from_mapping({**value, "headers": {"x": "y"}})
    with pytest.raises(ValueError, match="route_host"):
        OpenCodeProvider.from_mapping({**value, "route_host": "example.com"})
