from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from anchor_mvp.data import provider as module
from anchor_mvp.data.provider import (
    PRESETS,
    discover_models,
    model_list_endpoint,
    provider_spec,
    query_quota,
    select_provider_model,
    validate_base_url,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_kimi_code_presets_use_official_protocol_urls_and_model() -> None:
    openai = PRESETS["kimi-code-openai"]
    anthropic = PRESETS["kimi-code-anthropic"]
    assert (openai.base_url, openai.default_model) == (
        "https://api.kimi.com/coding/v1",
        "kimi-for-coding",
    )
    assert (anthropic.base_url, anthropic.default_model) == (
        "https://api.kimi.com/coding/",
        "kimi-for-coding",
    )


@pytest.mark.parametrize(
    "value",
    [
        "api.example.com/v1",
        "use the kimi URL",
        "ftp://api.example.com/v1",
        "https://key@example.com/v1",
        "https://api.example.com/v1?key=value",
        "https://api.example.com/v1/chat/completions",
    ],
)
def test_base_url_rejects_ambiguous_or_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_base_url(value)


def test_config_rejects_inline_credentials() -> None:
    with pytest.raises(ValueError, match="must not be stored in config"):
        provider_spec(
            {
                "provider": "custom-openai",
                "base_url": "https://example.com/v1",
                "api_key": "secret",
            }
        )


def test_custom_provider_requires_url_and_valid_env_name() -> None:
    with pytest.raises(ValueError, match="base_url is required"):
        provider_spec({"provider": "custom-openai"})
    with pytest.raises(ValueError, match="environment-variable name"):
        provider_spec(
            {
                "provider": "custom-openai",
                "base_url": "https://example.com/v1",
                "api_key_env": "not valid",
            }
        )


def test_named_preset_rejects_protocol_mismatch() -> None:
    with pytest.raises(ValueError, match="requires protocol openai"):
        provider_spec({"provider": "kimi-code-openai", "protocol": "anthropic"})


def test_protocol_model_list_endpoints() -> None:
    assert model_list_endpoint("https://api.openai.com/v1", "openai") == (
        "https://api.openai.com/v1/models"
    )
    assert model_list_endpoint("https://api.anthropic.com", "anthropic") == (
        "https://api.anthropic.com/v1/models"
    )
    assert model_list_endpoint("https://api.kimi.com/coding/", "anthropic") == (
        "https://api.kimi.com/coding/v1/models"
    )
    assert (
        model_list_endpoint(
            "https://ark.cn-beijing.volces.com/api/coding/v3", "openai_responses"
        )
        == "https://ark.cn-beijing.volces.com/api/coding/v3/models"
    )


def test_custom_openai_responses_provider_preserves_ark_versioned_base() -> None:
    spec = provider_spec(
        {
            "provider": "custom-openai-responses",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "api_key_env": "ARK_TEST_KEY",
        }
    )
    assert spec.protocol == "openai_responses"
    assert spec.base_url == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert spec.api_key_env == "ARK_TEST_KEY"


def test_responses_full_endpoint_is_rejected_as_base_url() -> None:
    with pytest.raises(ValueError, match="base URL"):
        validate_base_url("https://ark.cn-beijing.volces.com/api/coding/v3/responses")


def test_openai_discovery_reads_key_only_from_environment(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        return _Response(
            {"object": "list", "data": [{"id": "z"}, {"id": "a"}, {"id": "a"}]}
        )

    monkeypatch.setenv("TEST_TEACHER_KEY", "secret-for-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    spec = provider_spec(
        {
            "provider": "custom-openai",
            "base_url": "https://example.com/v1",
            "api_key_env": "TEST_TEACHER_KEY",
        }
    )
    result = discover_models(spec)
    public = result.to_public_dict()
    assert result.status == "success"
    assert result.models == ("a", "z")
    assert captured == {
        "url": "https://example.com/v1/models",
        "authorization": "Bearer secret-for-test",
    }
    assert "secret-for-test" not in json.dumps(public)


def test_anthropic_discovery_uses_required_headers(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(request.headers)
        return _Response({"data": [{"id": "claude-test"}], "has_more": False})

    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    spec = provider_spec({"provider": "anthropic", "api_key_env": "ANTHROPIC_TEST_KEY"})
    assert discover_models(spec).models == ("claude-test",)
    assert captured["X-api-key"] == "secret-for-test"
    assert captured["Anthropic-version"] == "2023-06-01"


def test_discovery_failure_allows_manual_selection(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "not found", {}, io.BytesIO())

    monkeypatch.setenv("TEST_TEACHER_KEY", "secret-for-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    spec = provider_spec(
        {
            "provider": "custom-openai",
            "base_url": "https://example.com/v1",
            "api_key_env": "TEST_TEACHER_KEY",
        }
    )
    selected = select_provider_model(
        spec,
        requested_model="manual-model",
        discover=True,
        force_model=False,
    )
    assert selected.model == "manual-model"
    assert selected.model_source == "manual"
    assert selected.discovery.status == "unsupported"


def test_force_model_skips_discovery(monkeypatch) -> None:
    monkeypatch.setattr(
        module, "discover_models", lambda *args, **kwargs: pytest.fail("called")
    )
    spec = provider_spec({"provider": "kimi-code-openai"})
    selected = select_provider_model(
        spec,
        requested_model="manual-model",
        discover=True,
        force_model=True,
    )
    assert selected.model == "manual-model"
    assert selected.discovery.status == "skipped_force_model"


def test_model_index_selects_discovered_model(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TEACHER_KEY", "secret-for-test")
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda request, timeout: _Response({"data": [{"id": "b"}, {"id": "a"}]}),
    )
    spec = provider_spec(
        {
            "provider": "custom-openai",
            "base_url": "https://example.com/v1",
            "api_key_env": "TEST_TEACHER_KEY",
        }
    )
    selected = select_provider_model(
        spec,
        requested_model=None,
        discover=True,
        force_model=False,
        model_index=1,
    )
    assert selected.model == "b"
    assert selected.model_source == "discovered_index"


def test_kimi_code_quota_is_explicitly_unsupported() -> None:
    result = query_quota(provider_spec({"provider": "kimi-code-openai"}))
    assert result["status"] == "unsupported"
    assert result["capability"] is None


def test_moonshot_capability_is_disabled_when_official_base_is_overridden() -> None:
    spec = provider_spec(
        {
            "provider": "kimi-platform-openai",
            "base_url": "https://gateway.example.com/v1",
        }
    )
    assert query_quota(spec)["status"] == "unsupported"


def test_moonshot_official_balance_capability_is_non_secret(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        assert request.full_url == "https://api.moonshot.cn/v1/users/me/balance"
        assert request.headers["Authorization"] == "Bearer secret-for-test"
        return _Response(
            {
                "status": True,
                "data": {
                    "available_balance": 4.5,
                    "voucher_balance": 3.0,
                    "cash_balance": 1.5,
                },
            }
        )

    monkeypatch.setenv("MOONSHOT_API_KEY", "secret-for-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    result = query_quota(provider_spec({"provider": "kimi-platform-openai"}))
    assert result["status"] == "success"
    assert result["balance"]["available_balance"] == 4.5
    assert "secret-for-test" not in json.dumps(result)
