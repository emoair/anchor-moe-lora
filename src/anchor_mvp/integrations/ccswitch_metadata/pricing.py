"""Decimal-only token cost estimates over a validated metadata snapshot."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .schema import SchemaError, validate_snapshot


def resolve_model_id(
    snapshot: Mapping[str, Any], request_model_id: str, *, provider_id: str = "*"
) -> str | None:
    """Resolve only explicit exact aliases; never guess by prefix or fuzzy matching."""

    validate_snapshot(snapshot)
    model_ids = {model["id"] for model in snapshot["models"]}
    if request_model_id in model_ids:
        return request_model_id
    folded = request_model_id.casefold()
    for wanted_provider in (provider_id, "*"):
        for alias in snapshot["model_aliases"]:
            if (
                alias["provider_id"] == wanted_provider
                and alias["alias"].casefold() == folded
            ):
                return str(alias["canonical_model_id"])
    return None


def estimate_cost(
    snapshot: Mapping[str, Any],
    *,
    request_model_id: str,
    protocol: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    provider_id: str = "*",
    multiplier: str = "1",
) -> dict[str, Any]:
    """Return a transparent estimate or an explicit unavailable result.

    CC Switch treats OpenAI-compatible input counts as including cache-read tokens,
    while Anthropic input counts are already fresh-input counts. This function keeps
    that distinction and never turns an unknown price into a zero-dollar estimate.
    """

    validate_snapshot(snapshot)
    if protocol not in {"openai_compatible", "openai_responses", "anthropic"}:
        raise SchemaError(f"unsupported billing protocol: {protocol}")
    counts = {
        "input": _token_count(input_tokens, "input_tokens"),
        "output": _token_count(output_tokens, "output_tokens"),
        "cache_read": _token_count(cache_read_tokens, "cache_read_tokens"),
        "cache_write": _token_count(cache_write_tokens, "cache_write_tokens"),
    }
    try:
        multiplier_decimal = Decimal(multiplier)
    except (InvalidOperation, TypeError) as exc:
        raise SchemaError("multiplier must be a non-negative decimal string") from exc
    if not multiplier_decimal.is_finite() or multiplier_decimal < 0:
        raise SchemaError("multiplier must be finite and non-negative")

    canonical = resolve_model_id(snapshot, request_model_id, provider_id=provider_id)
    if canonical is None:
        return {
            "known": False,
            "reason": "unknown_model",
            "request_model_id": request_model_id,
            "canonical_model_id": None,
            "currency": None,
            "total": None,
            "components": None,
        }
    pricing = next(
        item for item in snapshot["pricing"] if item["model_id"] == canonical
    )

    billable_input = counts["input"]
    if protocol in {"openai_compatible", "openai_responses"}:
        billable_input = max(0, billable_input - counts["cache_read"])
    billable = {
        "input": billable_input,
        "output": counts["output"],
        "cache_read": counts["cache_read"],
        "cache_write": counts["cache_write"],
    }
    unknown_dimensions = sorted(
        name
        for name, count in billable.items()
        if count > 0 and pricing[name] == "unknown"
    )
    if unknown_dimensions:
        return {
            "known": False,
            "reason": "unknown_price",
            "unknown_dimensions": unknown_dimensions,
            "request_model_id": request_model_id,
            "canonical_model_id": canonical,
            "currency": pricing["currency"],
            "total": None,
            "components": None,
        }

    divisor = Decimal(snapshot["token_billing"]["basis_divisor"])
    components: dict[str, str] = {}
    total = Decimal(0)
    for name, count in billable.items():
        if count == 0:
            component = Decimal(0)
        else:
            component = Decimal(pricing[name]) * Decimal(count) / divisor
        components[name] = _decimal_text(component)
        total += component
    total *= multiplier_decimal
    return {
        "known": True,
        "reason": None,
        "request_model_id": request_model_id,
        "canonical_model_id": canonical,
        "currency": pricing["currency"],
        "basis": pricing["basis"],
        "billable_tokens": billable,
        "multiplier": _decimal_text(multiplier_decimal),
        "components": components,
        "total": _decimal_text(total),
    }


def _token_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer")
    return value


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
