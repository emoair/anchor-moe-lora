from __future__ import annotations

from copy import deepcopy

import pytest

from anchor_mvp.integrations.ccswitch_metadata.provider_pricing import (
    estimate_channel_cost,
    load_provider_pricing,
    match_channel,
    overlay_sha256,
    public_channel_price,
    resolve_channel_model,
    validate_provider_pricing,
)
from anchor_mvp.integrations.ccswitch_metadata.schema import SchemaError


ARK_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
ZHIPU_PAYG_URL = "https://open.bigmodel.cn/api/paas/v4"


def _channel(overlay: dict[str, object], channel_id: str) -> dict[str, object]:
    return next(  # type: ignore[return-value]
        item
        for item in overlay["channels"]
        if item["id"] == channel_id  # type: ignore[index]
    )


def test_provider_pricing_is_audited_provider_scoped_metadata() -> None:
    overlay = load_provider_pricing()

    validate_provider_pricing(overlay)
    assert len(overlay_sha256(overlay)) == 64
    zhipu = _channel(overlay, "zhipu-glm-payg")
    ark = _channel(overlay, "volcengine-ark-coding-plan")
    zhipu_price = public_channel_price(zhipu)
    ark_price = public_channel_price(ark)

    assert zhipu_price == {
        "known": True,
        "billing_mode": "per_token",
        "price_status": "known",
        "currency": "CNY",
        "basis": "per_1m_tokens",
        "input": "8",
        "output": "28",
        "cache_read": "2",
        "cache_write": "unknown",
        "cache_storage_per_1m_token_hour": "0",
        "cache_storage_price_status": "promotional_free_valid_as_of",
        "valid_as_of": "2026-07-14",
        "source_scope": "official_provider_channel",
        "source_url": "https://bigmodel.cn/pricing",
        "retrieved_date": "2026-07-14",
    }
    assert ark_price["known"] is False
    assert ark_price["billing_mode"] == "subscription_quota"
    assert ark_price["basis"] == "subscription_quota"
    assert ark_price["source_url"] == ("https://www.volcengine.com/activity/codingplan")


def test_zhipu_payg_cost_uses_cny_and_preserves_cache_storage_units() -> None:
    overlay = load_provider_pricing()
    channel = match_channel(
        overlay,
        base_url=ZHIPU_PAYG_URL,
        protocol="openai",
    )
    assert channel is not None

    estimate = estimate_channel_cost(
        channel,
        request_model_id="glm-5.2",
        protocol="openai_compatible",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=200_000,
    )
    with_cache_creation = estimate_channel_cost(
        channel,
        request_model_id="glm-5.2",
        protocol="openai_compatible",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=200_000,
        cache_write_tokens=1,
    )

    assert estimate["known"] is True
    assert estimate["currency"] == "CNY"
    assert estimate["components"] == {
        "input": "6.4",
        "output": "2.8",
        "cache_read": "0.4",
        "cache_write": "0",
    }
    assert estimate["total"] == "9.6"
    assert with_cache_creation["known"] is False
    assert with_cache_creation["reason"] == "unknown_price"
    assert with_cache_creation["unknown_dimensions"] == ["cache_write"]


def test_ark_alias_and_subscription_price_do_not_leak_across_channels() -> None:
    overlay = load_provider_pricing()
    ark = match_channel(
        overlay,
        base_url=ARK_URL,
        protocol="openai_responses",
    )
    zhipu = match_channel(
        overlay,
        base_url=ZHIPU_PAYG_URL,
        protocol="openai",
    )

    assert ark is not None
    assert zhipu is not None
    assert resolve_channel_model(ark, "glm-5-2-260617") == "glm-5.2"
    assert resolve_channel_model(ark, "glm-5.2") is None
    assert resolve_channel_model(zhipu, "glm-5-2-260617") is None
    assert resolve_channel_model(zhipu, "glm-5.2") == "glm-5.2"
    assert (
        match_channel(
            overlay,
            base_url="https://example.invalid/v1",
            protocol="openai_responses",
        )
        is None
    )

    estimate = estimate_channel_cost(
        ark,
        request_model_id="glm-5-2-260617",
        protocol="openai_responses",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )
    assert estimate["known"] is False
    assert estimate["billing_mode"] == "subscription_quota"
    assert estimate["reason"] == "subscription_quota_no_marginal_token_price"
    assert estimate["total"] is None


def test_schema_rejects_cross_unit_cache_price_and_unapproved_ark_alias() -> None:
    overlay = load_provider_pricing()
    invalid_cache = deepcopy(overlay)
    _channel(invalid_cache, "zhipu-glm-payg")["billing"]["cache_write"] = "0"  # type: ignore[index]
    with pytest.raises(SchemaError, match="cache_write"):
        validate_provider_pricing(invalid_cache)

    invalid_alias = deepcopy(overlay)
    _channel(invalid_alias, "volcengine-ark-coding-plan")["model"][  # type: ignore[index]
        "request_model_ids"
    ].append("glm-5.2")
    with pytest.raises(SchemaError, match="request_model_ids"):
        validate_provider_pricing(invalid_alias)
