from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "observability" / "distillation_catalog.py"
SPEC = importlib.util.spec_from_file_location("distillation_catalog_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
catalog_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog_module
SPEC.loader.exec_module(catalog_module)


def _tokens(*, cache_exact: bool = True) -> dict[str, dict[str, object]]:
    return {
        "input": {"value": 1_000_000, "exact": True},
        "output": {"value": 100_000, "exact": True},
        "cache_read": {"value": 200_000, "exact": cache_exact},
        "cache_write": {"value": 0, "exact": True},
    }


def test_catalog_is_content_safe_complete_and_pinned(tmp_path: Path) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)

    public = catalog.public()

    assert public["content_safe"] is True
    assert public["secrets_read"] is False
    assert public["provenance"]["source_tag"] == "v3.16.5"
    assert len(public["provenance"]["snapshot_sha256"]) == 64
    assert public["update_status"]["automatic_apply"] is False
    assert public["update_status"]["network_checked"] is False
    provider = next(item for item in public["providers"] if item["id"] == "kimi")
    assert provider["display_name"] == "Kimi"
    assert provider["base_url"] == "https://api.moonshot.cn/v1"
    assert provider["control_protocol"] == "openai"
    assert provider["model_presets"][0]["request_model_id"] == "kimi-k2.7-code"
    model = next(item for item in public["models"] if item["id"] == "gpt-5.5")
    assert model["pricing"] == {
        "known": True,
        "currency": "USD",
        "basis": "per_1m_tokens",
        "input": "5",
        "output": "30",
        "cache_read": "0.50",
        "cache_write": "0",
        "source_scope": "cc_switch_global_estimate",
    }
    serialized = json.dumps(public, ensure_ascii=False).casefold()
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "opencode.json" not in serialized
    assert "cc-switch.db" not in serialized
    assert str(tmp_path).casefold() not in serialized


def test_valid_active_snapshot_is_read_only_and_diffed(tmp_path: Path) -> None:
    bundled = catalog_module.load_bundled_snapshot()
    active = deepcopy(bundled)
    active["providers"][0]["display_name"] = "DeepSeek Locally Reviewed"
    catalog_module.validate_snapshot(active)
    (tmp_path / "active.json").write_text(
        json.dumps(active, ensure_ascii=False), encoding="utf-8"
    )

    catalog = catalog_module.CatalogService(state_dir=tmp_path)
    public = catalog.public()

    assert public["provenance"]["origin"] == "active_validated_snapshot"
    assert public["update_status"]["state"] == "active_differs_from_pinned"
    assert public["update_status"]["difference"]["sections"]["providers"][
        "changed"
    ] == ["deepseek"]
    assert public["update_status"]["automatic_apply"] is False


def test_pinned_cost_requires_exact_alias_price_and_all_token_dimensions(
    tmp_path: Path,
) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)

    exact = catalog.pinned_cost(
        request_model_id="gpt-5.5-low",
        runtime_protocol="openai",
        base_url="https://custom.example/v1",
        token_metrics=_tokens(),
        binding_exact=True,
    )
    missing_cache = catalog.pinned_cost(
        request_model_id="gpt-5.5-low",
        runtime_protocol="openai",
        base_url="https://custom.example/v1",
        token_metrics=_tokens(cache_exact=False),
        binding_exact=True,
    )
    unknown_price = catalog.pinned_cost(
        request_model_id="glm-5.1",
        runtime_protocol="openai",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        token_metrics=_tokens(),
        binding_exact=True,
    )

    assert exact["known"] is True
    assert exact["exact"] is True
    assert exact["canonical_model_id"] == "gpt-5.5"
    assert exact["total"] == "7.1"
    assert missing_cache["known"] is False
    assert missing_cache["reason"] == "cache_read_usage_unknown"
    assert missing_cache["total"] is None
    assert unknown_price["known"] is False
    assert unknown_price["reason"] == "unknown_price"
    assert unknown_price["total"] is None


def test_glm52_cost_is_scoped_to_provider_endpoint_and_billing_channel(
    tmp_path: Path,
) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)

    zhipu = catalog.pinned_cost(
        request_model_id="glm-5.2",
        runtime_protocol="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        token_metrics=_tokens(),
        binding_exact=True,
    )
    ark = catalog.pinned_cost(
        request_model_id="glm-5-2-260617",
        runtime_protocol="openai_responses",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        token_metrics=_tokens(),
        binding_exact=True,
    )
    ark_unverified_alias = catalog.pinned_cost(
        request_model_id="glm-5.2",
        runtime_protocol="openai_responses",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        token_metrics=_tokens(),
        binding_exact=True,
    )
    custom = catalog.pinned_cost(
        request_model_id="glm-5.2",
        runtime_protocol="openai",
        base_url="https://custom.example/v1",
        token_metrics=_tokens(),
        binding_exact=True,
    )

    assert zhipu["known"] is True
    assert zhipu["provider_id"] == "zhipu-glm-payg"
    assert zhipu["currency"] == "CNY"
    assert zhipu["total"] == "9.6"
    assert zhipu["source_url"] == "https://bigmodel.cn/pricing"
    assert zhipu["retrieved_date"] == "2026-07-14"
    assert ark["known"] is False
    assert ark["provider_id"] == "volcengine-ark-coding-plan"
    assert ark["canonical_model_id"] == "glm-5.2"
    assert ark["billing_mode"] == "subscription_quota"
    assert ark["reason"] == "subscription_quota_no_marginal_token_price"
    assert ark["total"] is None
    assert ark_unverified_alias["reason"] == "exact_alias_unknown"
    assert custom["reason"] == "exact_alias_unknown"


def test_catalog_exposes_glm52_channel_price_provenance(tmp_path: Path) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)

    public = catalog.public()
    providers = {provider["id"]: provider for provider in public["providers"]}
    glm52 = next(model for model in public["models"] if model["id"] == "glm-5.2")

    zhipu_price = providers["zhipu-glm-payg"]["model_presets"][0]["pricing"]
    ark_price = providers["volcengine-ark-coding-plan"]["model_presets"][0]["pricing"]
    assert zhipu_price["input"] == "8"
    assert zhipu_price["output"] == "28"
    assert zhipu_price["cache_read"] == "2"
    assert zhipu_price["cache_write"] == "unknown"
    assert zhipu_price["currency"] == "CNY"
    assert ark_price["known"] is False
    assert ark_price["billing_mode"] == "subscription_quota"
    assert glm52["pricing"]["billing_mode"] == "provider_scoped"
    assert len(glm52["pricing_channels"]) == 2
    assert public["provider_pricing"]["retrieved_date"] == "2026-07-14"


def test_combined_cost_preserves_common_pricing_provenance_and_rejects_mixed(
    tmp_path: Path,
) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)
    overlay_cost = catalog.pinned_cost(
        request_model_id="glm-5.2",
        runtime_protocol="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        token_metrics=_tokens(),
        binding_exact=True,
    )
    cc_cost = catalog.pinned_cost(
        request_model_id="gpt-5.5-low",
        runtime_protocol="openai",
        base_url="https://custom.example/v1",
        token_metrics=_tokens(),
        binding_exact=True,
    )

    overlay_total = catalog.combined_cost([overlay_cost, overlay_cost])
    cc_total = catalog.combined_cost([cc_cost, cc_cost])
    mixed_total = catalog.combined_cost([overlay_cost, cc_cost])

    assert overlay_total["known"] is True
    assert overlay_total["total"] == "19.2"
    assert overlay_total["source_tag"] == "provider-pricing-2026-07-14"
    assert overlay_total["snapshot_sha256"] == overlay_cost["snapshot_sha256"]
    assert cc_total["known"] is True
    assert cc_total["total"] == "14.2"
    assert cc_total["source_tag"] == "v3.16.5"
    assert cc_total["snapshot_sha256"] == cc_cost["snapshot_sha256"]
    assert mixed_total["known"] is False
    assert mixed_total["reason"] == "mixed_pricing_provenance"
    assert mixed_total["source_tag"] is None
    assert mixed_total["snapshot_sha256"] is None


def test_combined_unknown_ark_cost_keeps_overlay_provenance(tmp_path: Path) -> None:
    catalog = catalog_module.CatalogService(state_dir=tmp_path)
    ark_cost = catalog.pinned_cost(
        request_model_id="glm-5-2-260617",
        runtime_protocol="openai_responses",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        token_metrics=_tokens(),
        binding_exact=True,
    )

    total = catalog.combined_cost([ark_cost])
    missing = dict(ark_cost)
    missing.pop("snapshot_sha256")
    missing_total = catalog.combined_cost([missing])

    assert total["known"] is False
    assert total["reason"] == "one_or_more_shard_costs_unknown"
    assert total["source_tag"] == "provider-pricing-2026-07-14"
    assert total["snapshot_sha256"] == ark_cost["snapshot_sha256"]
    assert missing_total["reason"] == "pricing_provenance_unknown"
    assert missing_total["source_tag"] is None
    assert missing_total["snapshot_sha256"] is None


def test_active_snapshot_cannot_shadow_audited_pricing_channel_id(
    tmp_path: Path,
) -> None:
    active = deepcopy(catalog_module.load_bundled_snapshot())
    cloned_provider = deepcopy(
        next(
            provider
            for provider in active["providers"]
            if provider["id"] == "zhipu-glm"
        )
    )
    cloned_provider["id"] = "zhipu-glm-payg"
    active["providers"].append(cloned_provider)
    glm51 = next(model for model in active["models"] if model["id"] == "glm-5.1")
    glm51["provider_ids"].append("zhipu-glm-payg")
    catalog_module.validate_snapshot(active)
    (tmp_path / "active.json").write_text(
        json.dumps(active, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(catalog_module.SchemaError, match="provider IDs collide"):
        catalog_module.CatalogService(state_dir=tmp_path)
