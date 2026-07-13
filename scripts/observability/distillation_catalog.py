"""Read-only, secret-free CC Switch metadata catalog for the local dashboard."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = WORKSPACE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from anchor_mvp.integrations.ccswitch_metadata.pricing import (  # noqa: E402
    estimate_cost,
    resolve_model_id,
)
from anchor_mvp.integrations.ccswitch_metadata.provider_pricing import (  # noqa: E402
    OVERLAY_SOURCE_TAG,
    estimate_channel_cost,
    load_provider_pricing,
    match_channel,
    overlay_sha256,
    public_channel_price,
    resolve_channel_model,
)
from anchor_mvp.integrations.ccswitch_metadata.schema import (  # noqa: E402
    SchemaError,
    validate_snapshot,
)
from anchor_mvp.integrations.ccswitch_metadata.sync import (  # noqa: E402
    IntegrityError,
    MetadataStore,
    default_state_dir,
    load_bundled_snapshot,
    semantic_diff,
    snapshot_sha256,
)


CATALOG_SCHEMA = "anchor.distillation-catalog.v1"
PRICE_DIMENSIONS = ("input", "output", "cache_read", "cache_write")
CONTROL_PROTOCOL = {
    "openai_compatible": "openai",
    "openai_responses": "openai_responses",
    "anthropic": "anthropic",
}
BILLING_PROTOCOL = {
    "openai": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "openai_responses": "openai_responses",
    "anthropic": "anthropic",
}


class CatalogService:
    """Load only validated bundled/active snapshots; never touch provider secrets."""

    def __init__(self, *, state_dir: Path | None = None) -> None:
        self.state_dir = (state_dir or default_state_dir()).expanduser().resolve()
        self.lock = threading.RLock()
        self.snapshot: dict[str, Any] = {}
        self.provider_pricing = load_provider_pricing()
        self.origin = "unavailable"
        self.active_status = "unchecked"
        self.difference: dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        bundled = load_bundled_snapshot()
        validate_snapshot(bundled)
        active: dict[str, Any] | None = None
        active_status = "absent"
        try:
            active = MetadataStore(self.state_dir).current()
        except (IntegrityError, SchemaError, OSError):
            active_status = "invalid_ignored"
        else:
            if active is not None:
                validate_snapshot(active)
                active_status = "validated"
        selected = active if active is not None else bundled
        self._reject_provider_id_collisions(selected, self.provider_pricing)
        difference = (
            semantic_diff(active, bundled)
            if active is not None
            else {
                "changed": False,
                "current_sha256": snapshot_sha256(bundled),
                "candidate_sha256": snapshot_sha256(bundled),
                "sections": {},
            }
        )
        with self.lock:
            self.snapshot = selected
            self.origin = (
                "active_validated_snapshot"
                if active is not None
                else "bundled_reviewed_snapshot"
            )
            self.active_status = active_status
            self.difference = difference

    def public(self, *, refresh: bool = False) -> dict[str, object]:
        if refresh:
            self.refresh()
        with self.lock:
            snapshot = self.snapshot
            origin = self.origin
            active_status = self.active_status
            difference = self.difference
            selected_sha = snapshot_sha256(snapshot)
            providers = self._providers(snapshot, self.provider_pricing)
            models = self._models(snapshot, self.provider_pricing)
            source = snapshot["source"]
            pinned_sha = difference.get("candidate_sha256") or selected_sha
            return {
                "schema_version": CATALOG_SCHEMA,
                "content_safe": True,
                "secrets_read": False,
                "providers": providers,
                "models": models,
                "provider_pricing": {
                    "source_tag": OVERLAY_SOURCE_TAG,
                    "snapshot_sha256": overlay_sha256(self.provider_pricing),
                    "retrieved_date": self.provider_pricing["retrieved_date"],
                    "scope": "official_provider_channel",
                    "channels": [
                        {
                            "provider_id": channel["id"],
                            "billing_mode": channel["billing"]["billing_mode"],
                            "source_url": channel["source"]["url"],
                            "retrieved_date": channel["source"]["retrieved_date"],
                        }
                        for channel in self.provider_pricing["channels"]
                    ],
                },
                "provenance": {
                    "origin": origin,
                    "snapshot_sha256": selected_sha,
                    "source_tag": source["source_tag"],
                    "source_commit": source["source_commit"],
                    "repository": source["repository"],
                    "license": source["license"],
                    "adapter_mode": source["adapter"]["mode"],
                    "source_files": [
                        {
                            "path": item["path"],
                            "role": item["role"],
                            "size": item["size"],
                            "sha256": item["sha256"],
                        }
                        for item in source["files"]
                    ],
                },
                "update_status": {
                    "mode": "offline_pinned_diff_only",
                    "pinned_tag": source["source_tag"],
                    "pinned_sha256": pinned_sha,
                    "selected_sha256": selected_sha,
                    "active_status": active_status,
                    "state": (
                        "active_differs_from_pinned"
                        if difference.get("changed")
                        else "pinned_snapshot_selected"
                    ),
                    "difference": difference,
                    "automatic_apply": False,
                    "network_checked": False,
                },
            }

    def status(self, *, refresh: bool = False) -> dict[str, object]:
        public = self.public(refresh=refresh)
        return {
            "schema_version": CATALOG_SCHEMA,
            "content_safe": True,
            "provenance": public["provenance"],
            "update_status": public["update_status"],
        }

    def pinned_cost(
        self,
        *,
        request_model_id: str | None,
        runtime_protocol: str | None,
        base_url: str | None,
        token_metrics: Mapping[str, Mapping[str, object]],
        binding_exact: bool,
    ) -> dict[str, object]:
        with self.lock:
            snapshot = self.snapshot
            selected_sha = snapshot_sha256(snapshot)
            source_tag = str(snapshot["source"]["source_tag"])
            pricing_channel = match_channel(
                self.provider_pricing,
                base_url=base_url,
                protocol=runtime_protocol,
            )
            provider_id = (
                str(pricing_channel["id"])
                if pricing_channel is not None
                else self._provider_id(snapshot, base_url, runtime_protocol)
            )
            result_base: dict[str, object] = {
                "known": False,
                "exact": False,
                "currency": None,
                "total": None,
                "request_model_id": request_model_id,
                "canonical_model_id": None,
                "provider_id": provider_id,
                "source_tag": source_tag,
                "snapshot_sha256": selected_sha,
                "scope": "persisted_stage_provider_usage",
            }
            if not binding_exact or request_model_id is None:
                return {**result_base, "reason": "provider_binding_unknown"}
            protocol = BILLING_PROTOCOL.get(runtime_protocol or "")
            if protocol is None:
                return {**result_base, "reason": "billing_protocol_unknown"}
            if pricing_channel is not None:
                canonical = resolve_channel_model(pricing_channel, request_model_id)
                result_base.update(
                    {
                        "source_tag": OVERLAY_SOURCE_TAG,
                        "snapshot_sha256": overlay_sha256(self.provider_pricing),
                        "billing_mode": pricing_channel["billing"]["billing_mode"],
                        "price_status": pricing_channel["billing"]["price_status"],
                        "source_url": pricing_channel["source"]["url"],
                        "retrieved_date": pricing_channel["source"]["retrieved_date"],
                    }
                )
            else:
                canonical = resolve_model_id(
                    snapshot,
                    request_model_id,
                    provider_id=provider_id or "*",
                )
            if canonical is None:
                return {**result_base, "reason": "exact_alias_unknown"}
            result_base["canonical_model_id"] = canonical
            if (
                pricing_channel is not None
                and pricing_channel["billing"]["billing_mode"] == "subscription_quota"
            ):
                return {
                    **result_base,
                    "currency": pricing_channel["billing"]["currency"],
                    "basis": pricing_channel["billing"]["basis"],
                    "reason": "subscription_quota_no_marginal_token_price",
                }
            counts: dict[str, int] = {}
            for dimension in PRICE_DIMENSIONS:
                metric = token_metrics.get(dimension)
                if not isinstance(metric, Mapping) or metric.get("exact") is not True:
                    return {
                        **result_base,
                        "reason": f"{dimension}_usage_unknown",
                    }
                value = metric.get("value")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    return {
                        **result_base,
                        "reason": f"{dimension}_usage_unknown",
                    }
                counts[dimension] = value
            if pricing_channel is not None:
                estimate = estimate_channel_cost(
                    pricing_channel,
                    request_model_id=request_model_id,
                    protocol=protocol,
                    input_tokens=counts["input"],
                    output_tokens=counts["output"],
                    cache_read_tokens=counts["cache_read"],
                    cache_write_tokens=counts["cache_write"],
                )
            else:
                estimate = estimate_cost(
                    snapshot,
                    request_model_id=request_model_id,
                    provider_id=provider_id or "*",
                    protocol=protocol,
                    input_tokens=counts["input"],
                    output_tokens=counts["output"],
                    cache_read_tokens=counts["cache_read"],
                    cache_write_tokens=counts["cache_write"],
                )
            if estimate.get("known") is not True:
                return {
                    **result_base,
                    "canonical_model_id": estimate.get("canonical_model_id"),
                    "currency": estimate.get("currency"),
                    "reason": str(estimate.get("reason") or "unknown_price"),
                }
            result = {
                **result_base,
                "known": True,
                "exact": True,
                "reason": None,
                "currency": estimate["currency"],
                "total": estimate["total"],
                "canonical_model_id": estimate["canonical_model_id"],
                "basis": estimate["basis"],
                "components": estimate["components"],
                "billable_tokens": estimate["billable_tokens"],
            }
            for field in (
                "billing_mode",
                "price_status",
                "source_url",
                "retrieved_date",
            ):
                if field in estimate:
                    result[field] = estimate[field]
            return result

    def combined_cost(self, costs: Sequence[object]) -> dict[str, object]:
        public_costs = [item for item in costs if isinstance(item, Mapping)]
        if not public_costs:
            return self._unknown_total("no_shards", source_tag=None, source_sha=None)
        provenance: set[tuple[str, str]] = set()
        expected_provenance = {
            (
                str(self.snapshot["source"]["source_tag"]),
                snapshot_sha256(self.snapshot),
            ),
            (OVERLAY_SOURCE_TAG, overlay_sha256(self.provider_pricing)),
        }
        for item in public_costs:
            source_tag = item.get("source_tag")
            source_sha = item.get("snapshot_sha256")
            if (
                not isinstance(source_tag, str)
                or not source_tag
                or not isinstance(source_sha, str)
                or len(source_sha) != 64
                or any(character not in "0123456789abcdef" for character in source_sha)
            ):
                return self._unknown_total(
                    "pricing_provenance_unknown", source_tag=None, source_sha=None
                )
            provenance.add((source_tag, source_sha))
        if len(provenance) != 1:
            return self._unknown_total(
                "mixed_pricing_provenance", source_tag=None, source_sha=None
            )
        common_source_tag, common_source_sha = next(iter(provenance))
        if (common_source_tag, common_source_sha) not in expected_provenance:
            return self._unknown_total(
                "pricing_provenance_unknown", source_tag=None, source_sha=None
            )
        if any(item.get("known") is not True for item in public_costs):
            return self._unknown_total(
                "one_or_more_shard_costs_unknown",
                source_tag=common_source_tag,
                source_sha=common_source_sha,
            )
        currencies = {str(item.get("currency")) for item in public_costs}
        if len(currencies) != 1:
            return self._unknown_total(
                "mixed_currency",
                source_tag=common_source_tag,
                source_sha=common_source_sha,
            )
        try:
            total = sum(Decimal(str(item["total"])) for item in public_costs)
        except (InvalidOperation, KeyError, TypeError):
            return self._unknown_total(
                "invalid_cost_component",
                source_tag=common_source_tag,
                source_sha=common_source_sha,
            )
        with self.lock:
            return {
                "known": True,
                "exact": True,
                "reason": None,
                "currency": next(iter(currencies)),
                "total": _decimal_text(total),
                "source_tag": common_source_tag,
                "snapshot_sha256": common_source_sha,
                "scope": "sum_persisted_stage_provider_usage",
            }

    @staticmethod
    def _unknown_total(
        reason: str, *, source_tag: str | None, source_sha: str | None
    ) -> dict[str, object]:
        return {
            "known": False,
            "exact": False,
            "reason": reason,
            "currency": None,
            "total": None,
            "source_tag": source_tag,
            "snapshot_sha256": source_sha,
            "scope": "sum_persisted_stage_provider_usage",
        }

    @staticmethod
    def _reject_provider_id_collisions(
        snapshot: Mapping[str, Any], provider_pricing: Mapping[str, Any]
    ) -> None:
        snapshot_ids = {str(provider["id"]) for provider in snapshot["providers"]}
        overlay_ids = {str(channel["id"]) for channel in provider_pricing["channels"]}
        collisions = sorted(snapshot_ids & overlay_ids)
        if collisions:
            raise SchemaError(
                "CC Switch snapshot provider IDs collide with audited pricing "
                f"channels: {collisions}"
            )

    @staticmethod
    def _provider_id(
        snapshot: Mapping[str, Any], base_url: str | None, runtime_protocol: str | None
    ) -> str | None:
        catalog_protocol = BILLING_PROTOCOL.get(runtime_protocol or "")
        if base_url is None or catalog_protocol is None:
            return None
        matches = [
            str(provider["id"])
            for provider in snapshot["providers"]
            if provider["base_url"] == base_url
            and provider["protocol"] == catalog_protocol
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _providers(
        snapshot: Mapping[str, Any], provider_pricing: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        models = {item["id"]: item for item in snapshot["models"]}
        pricing = {item["model_id"]: item for item in snapshot["pricing"]}
        result: list[dict[str, object]] = []
        for provider in snapshot["providers"]:
            provider_id = str(provider["id"])
            presets: list[dict[str, object]] = []
            for request_id in provider["request_model_ids"]:
                canonical = resolve_model_id(
                    snapshot, request_id, provider_id=provider_id
                )
                model = models.get(canonical)
                price = pricing.get(canonical)
                presets.append(
                    {
                        "request_model_id": request_id,
                        "canonical_model_id": canonical,
                        "display_name": (
                            model["display_name"] if model is not None else request_id
                        ),
                        "alias_match": "exact" if canonical is not None else "unknown",
                        "pricing": _public_price(price),
                    }
                )
            result.append(
                {
                    "id": provider_id,
                    "display_name": provider["display_name"],
                    "category": provider["category"],
                    "base_url": provider["base_url"],
                    "protocol": provider["protocol"],
                    "control_protocol": CONTROL_PROTOCOL[provider["protocol"]],
                    "model_discovery": provider["model_discovery"],
                    "model_presets": presets,
                }
            )
        known_ids = {str(item["id"]) for item in result}
        overlay_ids = {str(channel["id"]) for channel in provider_pricing["channels"]}
        collisions = sorted(known_ids & overlay_ids)
        if collisions:
            raise SchemaError(
                "CC Switch snapshot provider IDs collide with audited pricing "
                f"channels: {collisions}"
            )
        for channel in provider_pricing["channels"]:
            provider_id = str(channel["id"])
            protocol = str(channel["protocols"][0])
            model = channel["model"]
            price = public_channel_price(channel)
            result.append(
                {
                    "id": provider_id,
                    "display_name": channel["display_name"],
                    "category": channel["category"],
                    "base_url": channel["base_urls"][0],
                    "protocol": protocol,
                    "control_protocol": CONTROL_PROTOCOL[protocol],
                    "supported_protocols": [
                        CONTROL_PROTOCOL[item] for item in channel["protocols"]
                    ],
                    "model_discovery": "unsupported",
                    "model_presets": [
                        {
                            "request_model_id": request_id,
                            "canonical_model_id": model["canonical_model_id"],
                            "display_name": model["display_name"],
                            "alias_match": "exact",
                            "pricing": price,
                        }
                        for request_id in model["request_model_ids"]
                    ],
                }
            )
        return result

    @staticmethod
    def _models(
        snapshot: Mapping[str, Any], provider_pricing: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        prices = {item["model_id"]: item for item in snapshot["pricing"]}
        aliases: dict[str, list[dict[str, str]]] = {}
        for alias in snapshot["model_aliases"]:
            aliases.setdefault(alias["canonical_model_id"], []).append(
                {
                    "provider_id": alias["provider_id"],
                    "alias": alias["alias"],
                    "match": alias["match"],
                }
            )
        result = [
            {
                "id": model["id"],
                "display_name": model["display_name"],
                "provider_ids": list(model["provider_ids"]),
                "context_tokens": model["context_tokens"],
                "max_output_tokens": model["max_output_tokens"],
                "aliases": aliases.get(model["id"], []),
                "pricing": _public_price(prices.get(model["id"])),
            }
            for model in snapshot["models"]
        ]
        channels_by_model: dict[str, list[Mapping[str, Any]]] = {}
        for channel in provider_pricing["channels"]:
            canonical = str(channel["model"]["canonical_model_id"])
            channels_by_model.setdefault(canonical, []).append(channel)
        existing = {str(model["id"]): model for model in result}
        for canonical, channels in channels_by_model.items():
            pricing_channels = [
                {
                    "provider_id": channel["id"],
                    "pricing": public_channel_price(channel),
                }
                for channel in channels
            ]
            aliases_for_channels = [
                {
                    "provider_id": channel["id"],
                    "alias": alias,
                    "match": "exact",
                }
                for channel in channels
                for alias in channel["model"]["request_model_ids"]
            ]
            if canonical in existing:
                existing[canonical]["pricing_channels"] = pricing_channels
                existing[canonical]["aliases"].extend(aliases_for_channels)
                continue
            representative = channels[0]["model"]
            model_result: dict[str, object] = {
                "id": canonical,
                "display_name": representative["display_name"],
                "provider_ids": [channel["id"] for channel in channels],
                "context_tokens": representative["context_tokens"],
                "max_output_tokens": representative["max_output_tokens"],
                "aliases": aliases_for_channels,
                "pricing": {
                    "known": False,
                    "billing_mode": "provider_scoped",
                    "price_status": "select_provider_channel",
                    "currency": None,
                    "basis": None,
                    "source_scope": "official_provider_channel",
                },
                "pricing_channels": pricing_channels,
            }
            result.append(model_result)
            existing[canonical] = model_result
        return result


def _public_price(value: Mapping[str, Any] | None) -> dict[str, object] | None:
    if value is None:
        return None
    known = all(value[name] != "unknown" for name in PRICE_DIMENSIONS)
    return {
        "known": known,
        "currency": value["currency"],
        "basis": value["basis"],
        "input": value["input"],
        "output": value["output"],
        "cache_read": value["cache_read"],
        "cache_write": value["cache_write"],
        "source_scope": value["source_scope"],
    }


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
