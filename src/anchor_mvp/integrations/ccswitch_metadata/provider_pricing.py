"""Audited provider/channel pricing that must not be treated as global model data."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .schema import SchemaError


OVERLAY_PATH = Path(__file__).with_name("fixtures") / "provider_pricing_2026_07_14.json"
OVERLAY_SOURCE_TAG = "provider-pricing-2026-07-14"
PRICE_DIMENSIONS = ("input", "output", "cache_read", "cache_write")
_PROTOCOLS = frozenset({"openai_compatible", "openai_responses"})
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?$")
_EXPECTED_CHANNELS: dict[str, dict[str, object]] = {
    "zhipu-glm-payg": {
        "protocols": ["openai_compatible"],
        "base_urls": ["https://open.bigmodel.cn/api/paas/v4"],
        "request_model_ids": ["glm-5.2"],
        "billing": {
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
        },
        "source_url": "https://bigmodel.cn/pricing",
    },
    "volcengine-ark-coding-plan": {
        "protocols": ["openai_responses", "openai_compatible"],
        "base_urls": ["https://ark.cn-beijing.volces.com/api/coding/v3"],
        "request_model_ids": ["glm-5-2-260617"],
        "billing": {
            "billing_mode": "subscription_quota",
            "price_status": "marginal_token_cost_unknown",
            "currency": "CNY",
            "basis": "subscription_quota",
            "input": "unknown",
            "output": "unknown",
            "cache_read": "unknown",
            "cache_write": "unknown",
            "cache_storage_per_1m_token_hour": "unknown",
            "cache_storage_price_status": "unknown",
            "valid_as_of": "2026-07-14",
        },
        "source_url": "https://www.volcengine.com/activity/codingplan",
    },
}


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate provider-pricing JSON key: {key}")
        result[key] = value
    return result


def load_provider_pricing() -> dict[str, Any]:
    """Load the bundled, project-owned pricing overlay and validate it strictly."""

    try:
        raw = OVERLAY_PATH.read_bytes()
    except OSError as exc:
        raise SchemaError(f"cannot read provider pricing overlay: {exc}") from exc
    if len(raw) > 64 * 1024:
        raise SchemaError("provider pricing overlay exceeds 65536 bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
        raise SchemaError(f"invalid provider pricing overlay: {exc}") from exc
    validate_provider_pricing(value)
    return value


def validate_provider_pricing(value: Any) -> None:
    root = _object(
        value,
        "$",
        {"schema_version", "content_kind", "retrieved_date", "channels"},
    )
    if root["schema_version"] != 1:
        raise SchemaError("provider pricing schema_version must be 1")
    if root["content_kind"] != "anchor_provider_channel_pricing":
        raise SchemaError("provider pricing content_kind is unsupported")
    if root["retrieved_date"] != "2026-07-14":
        raise SchemaError("provider pricing retrieved_date is not the audited date")
    channels = _list(root["channels"], "$.channels", maximum=16)
    seen: set[str] = set()
    for index, channel_value in enumerate(channels):
        path = f"$.channels[{index}]"
        channel = _object(
            channel_value,
            path,
            {
                "id",
                "display_name",
                "category",
                "protocols",
                "base_urls",
                "model",
                "billing",
                "source",
            },
        )
        channel_id = _text(channel["id"], f"{path}.id", maximum=80)
        if channel_id in seen:
            raise SchemaError(f"duplicate provider pricing channel: {channel_id}")
        seen.add(channel_id)
        expected = _EXPECTED_CHANNELS.get(channel_id)
        if expected is None:
            raise SchemaError(f"unapproved provider pricing channel: {channel_id}")
        _text(channel["display_name"], f"{path}.display_name", maximum=120)
        if channel["category"] != "first_party":
            raise SchemaError(f"{path}.category must be first_party")
        protocols = _list(channel["protocols"], f"{path}.protocols", maximum=4)
        if protocols != expected["protocols"] or any(
            protocol not in _PROTOCOLS for protocol in protocols
        ):
            raise SchemaError(f"{path}.protocols differ from the audited channel")
        base_urls = _list(channel["base_urls"], f"{path}.base_urls", maximum=4)
        for url_index, url in enumerate(base_urls):
            _https_url(url, f"{path}.base_urls[{url_index}]")
        if base_urls != expected["base_urls"]:
            raise SchemaError(f"{path}.base_urls differ from the audited channel")
        model = _validate_model(channel["model"], f"{path}.model")
        if model["request_model_ids"] != expected["request_model_ids"]:
            raise SchemaError(
                f"{path}.model.request_model_ids differ from the audited channel"
            )
        billing = _validate_billing(channel["billing"], f"{path}.billing")
        if dict(billing) != expected["billing"]:
            raise SchemaError(f"{path}.billing differs from the audited price")
        source = _object(
            channel["source"],
            f"{path}.source",
            {"url", "publisher", "retrieved_date", "note"},
        )
        _https_url(source["url"], f"{path}.source.url")
        if source["url"] != expected["source_url"]:
            raise SchemaError(f"{path}.source.url differs from the audited source")
        _text(source["publisher"], f"{path}.source.publisher", maximum=80)
        if source["retrieved_date"] != root["retrieved_date"]:
            raise SchemaError(f"{path}.source.retrieved_date is inconsistent")
        _text(source["note"], f"{path}.source.note", maximum=500)
    if seen != set(_EXPECTED_CHANNELS):
        raise SchemaError("provider pricing overlay does not contain every channel")


def overlay_sha256(overlay: Mapping[str, Any]) -> str:
    validate_provider_pricing(overlay)
    canonical = json.dumps(
        overlay,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def match_channel(
    overlay: Mapping[str, Any],
    *,
    base_url: str | None,
    protocol: str | None,
) -> Mapping[str, Any] | None:
    """Resolve a billing channel by exact endpoint and protocol, never by model."""

    validate_provider_pricing(overlay)
    normalized_protocol = _normalize_protocol(protocol)
    normalized_url = _normalize_base_url(base_url)
    if normalized_protocol is None or normalized_url is None:
        return None
    matches = [
        channel
        for channel in overlay["channels"]
        if normalized_protocol in channel["protocols"]
        and normalized_url
        in {_normalize_base_url(item) for item in channel["base_urls"]}
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_channel_model(
    channel: Mapping[str, Any], request_model_id: str
) -> str | None:
    """Resolve only an alias explicitly approved for the already-matched channel."""

    folded = request_model_id.casefold()
    if any(
        folded == str(alias).casefold()
        for alias in channel["model"]["request_model_ids"]
    ):
        return str(channel["model"]["canonical_model_id"])
    return None


def estimate_channel_cost(
    channel: Mapping[str, Any],
    *,
    request_model_id: str,
    protocol: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> dict[str, Any]:
    """Estimate one already-matched provider channel or return explicit unknown."""

    canonical = resolve_channel_model(channel, request_model_id)
    source = channel["source"]
    billing = channel["billing"]
    result_base = {
        "request_model_id": request_model_id,
        "canonical_model_id": canonical,
        "provider_id": channel["id"],
        "billing_mode": billing["billing_mode"],
        "price_status": billing["price_status"],
        "currency": billing["currency"],
        "basis": billing["basis"],
        "source_tag": OVERLAY_SOURCE_TAG,
        "source_url": source["url"],
        "retrieved_date": source["retrieved_date"],
    }
    if canonical is None:
        return {
            **result_base,
            "known": False,
            "reason": "exact_alias_unknown",
            "total": None,
            "components": None,
        }
    if billing["billing_mode"] == "subscription_quota":
        return {
            **result_base,
            "known": False,
            "reason": "subscription_quota_no_marginal_token_price",
            "total": None,
            "components": None,
        }
    normalized_protocol = _normalize_protocol(protocol)
    if normalized_protocol not in channel["protocols"]:
        raise SchemaError("protocol does not belong to the matched pricing channel")
    counts = {
        "input": _token_count(input_tokens, "input_tokens"),
        "output": _token_count(output_tokens, "output_tokens"),
        "cache_read": _token_count(cache_read_tokens, "cache_read_tokens"),
        "cache_write": _token_count(cache_write_tokens, "cache_write_tokens"),
    }
    billable = {
        "input": max(0, counts["input"] - counts["cache_read"]),
        "output": counts["output"],
        "cache_read": counts["cache_read"],
        "cache_write": counts["cache_write"],
    }
    unknown_dimensions = sorted(
        name
        for name, count in billable.items()
        if count > 0 and billing[name] == "unknown"
    )
    if unknown_dimensions:
        return {
            **result_base,
            "known": False,
            "reason": "unknown_price",
            "unknown_dimensions": unknown_dimensions,
            "total": None,
            "components": None,
        }
    divisor = Decimal(1_000_000)
    components: dict[str, str] = {}
    total = Decimal(0)
    for name, count in billable.items():
        if count == 0:
            component = Decimal(0)
        else:
            component = Decimal(str(billing[name])) * Decimal(count) / divisor
        components[name] = _decimal_text(component)
        total += component
    return {
        **result_base,
        "known": True,
        "reason": None,
        "billable_tokens": billable,
        "components": components,
        "total": _decimal_text(total),
    }


def public_channel_price(channel: Mapping[str, Any]) -> dict[str, object]:
    billing = channel["billing"]
    source = channel["source"]
    return {
        "known": billing["price_status"] == "known",
        "billing_mode": billing["billing_mode"],
        "price_status": billing["price_status"],
        "currency": billing["currency"],
        "basis": billing["basis"],
        "input": billing["input"],
        "output": billing["output"],
        "cache_read": billing["cache_read"],
        "cache_write": billing["cache_write"],
        "cache_storage_per_1m_token_hour": billing["cache_storage_per_1m_token_hour"],
        "cache_storage_price_status": billing["cache_storage_price_status"],
        "valid_as_of": billing["valid_as_of"],
        "source_scope": "official_provider_channel",
        "source_url": source["url"],
        "retrieved_date": source["retrieved_date"],
    }


def _validate_model(value: Any, path: str) -> Mapping[str, Any]:
    model = _object(
        value,
        path,
        {
            "canonical_model_id",
            "display_name",
            "request_model_ids",
            "context_tokens",
            "max_output_tokens",
        },
    )
    if model["canonical_model_id"] != "glm-5.2":
        raise SchemaError(f"{path}.canonical_model_id must be glm-5.2")
    _text(model["display_name"], f"{path}.display_name", maximum=120)
    aliases = _list(model["request_model_ids"], f"{path}.request_model_ids", maximum=8)
    if not aliases:
        raise SchemaError(f"{path}.request_model_ids cannot be empty")
    folded: set[str] = set()
    for index, alias in enumerate(aliases):
        alias_text = _text(alias, f"{path}.request_model_ids[{index}]", maximum=120)
        if any(character.isspace() for character in alias_text):
            raise SchemaError(f"{path}.request_model_ids contains whitespace")
        if alias_text.casefold() in folded:
            raise SchemaError(f"{path}.request_model_ids contains a duplicate")
        folded.add(alias_text.casefold())
    for field in ("context_tokens", "max_output_tokens"):
        count = model[field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SchemaError(f"{path}.{field} must be a positive integer")
    return model


def _validate_billing(value: Any, path: str) -> Mapping[str, Any]:
    billing = _object(
        value,
        path,
        {
            "billing_mode",
            "price_status",
            "currency",
            "basis",
            "input",
            "output",
            "cache_read",
            "cache_write",
            "cache_storage_per_1m_token_hour",
            "cache_storage_price_status",
            "valid_as_of",
        },
    )
    if billing["currency"] != "CNY":
        raise SchemaError(f"{path}.currency must be CNY")
    if billing["billing_mode"] == "per_token":
        if billing["price_status"] != "known":
            raise SchemaError(f"{path}.price_status must be known")
        if billing["basis"] != "per_1m_tokens":
            raise SchemaError(f"{path}.basis must be per_1m_tokens")
        for field in ("input", "output", "cache_read"):
            _price(billing[field], f"{path}.{field}", allow_unknown=False)
        if billing["cache_write"] != "unknown":
            raise SchemaError(
                f"{path}.cache_write must remain unknown because cache storage "
                "has different units"
            )
        _price(
            billing["cache_storage_per_1m_token_hour"],
            f"{path}.cache_storage_per_1m_token_hour",
            allow_unknown=False,
        )
        if billing["cache_storage_price_status"] != "promotional_free_valid_as_of":
            raise SchemaError(
                f"{path}.cache_storage_price_status must mark promotional pricing"
            )
    elif billing["billing_mode"] == "subscription_quota":
        if billing["price_status"] != "marginal_token_cost_unknown":
            raise SchemaError(f"{path}.price_status must keep marginal cost unknown")
        if billing["basis"] != "subscription_quota":
            raise SchemaError(f"{path}.basis must be subscription_quota")
        for field in PRICE_DIMENSIONS:
            if billing[field] != "unknown":
                raise SchemaError(f"{path}.{field} must remain unknown")
        if billing["cache_storage_per_1m_token_hour"] != "unknown":
            raise SchemaError(
                f"{path}.cache_storage_per_1m_token_hour must remain unknown"
            )
        if billing["cache_storage_price_status"] != "unknown":
            raise SchemaError(f"{path}.cache_storage_price_status must remain unknown")
    else:
        raise SchemaError(f"{path}.billing_mode is unsupported")
    if billing["valid_as_of"] != "2026-07-14":
        raise SchemaError(f"{path}.valid_as_of is not the audited date")
    return billing


def _object(value: Any, path: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise SchemaError(f"{path} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{path} has unknown fields: {sorted(unknown)}")
    return value


def _list(value: Any, path: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise SchemaError(f"{path} must be a non-empty array up to {maximum} items")
    return value


def _text(value: Any, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SchemaError(f"{path} must be a non-empty string up to {maximum} chars")
    if any(ord(character) < 32 for character in value) or "<" in value or ">" in value:
        raise SchemaError(f"{path} contains unsafe text")
    return value


def _https_url(value: Any, path: str) -> str:
    text = _text(value, path, maximum=500)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SchemaError(f"{path} must be a credential-free HTTPS URL")
    return text


def _price(value: Any, path: str, *, allow_unknown: bool) -> str:
    if allow_unknown and value == "unknown":
        return value
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise SchemaError(f"{path} must be a non-negative decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise SchemaError(f"{path} is not decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise SchemaError(f"{path} must be finite and non-negative")
    return value


def _token_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer")
    return value


def _normalize_protocol(value: str | None) -> str | None:
    if value == "openai":
        return "openai_compatible"
    return value if value in _PROTOCOLS else None


def _normalize_base_url(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.rstrip("/")


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
