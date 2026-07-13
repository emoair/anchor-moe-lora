"""Strict, fail-closed schema for dashboard-safe metadata snapshots."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .constants import (
    ALLOWED_SOURCE_URLS,
    EXPECTED_SOURCE_FILES,
    MAX_SNAPSHOT_BYTES,
    RAW_BASE,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TAG,
)


class SchemaError(ValueError):
    """A snapshot is unsafe, ambiguous, or incompatible with this adapter."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ALIAS_RE = re.compile(r"^[^\s<>]{1,200}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?$")
_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "apiKey",
        "authorization",
        "headers",
        "password",
        "refresh_token",
        "secret",
        "secret_access_key",
        "token",
    }
)


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_snapshot_bytes(data: bytes, *, origin: str = "snapshot") -> dict[str, Any]:
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise SchemaError(f"{origin} exceeds {MAX_SNAPSHOT_BYTES} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{origin} must be UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, SchemaError) as exc:
        raise SchemaError(f"invalid {origin}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{origin} root must be an object")
    validate_snapshot(value)
    return value


def load_snapshot(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise SchemaError(f"cannot read snapshot {source}: {exc}") from exc
    return parse_snapshot_bytes(data, origin=str(source))


def safe_json_bytes(snapshot: Mapping[str, Any]) -> bytes:
    """Serialize validated metadata without HTML-significant literal characters."""

    validate_snapshot(snapshot)
    text = json.dumps(
        snapshot,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    text = text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (text + "\n").encode("utf-8")


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    root = _object(
        snapshot,
        "$",
        {
            "schema_version",
            "content_kind",
            "source",
            "providers",
            "models",
            "model_aliases",
            "pricing",
            "token_billing",
        },
    )
    if root["schema_version"] != 1:
        raise SchemaError("$.schema_version must be 1")
    if root["content_kind"] != "cc_switch_metadata_snapshot":
        raise SchemaError("$.content_kind is unsupported")
    _reject_unsafe_content(root, "$")
    _validate_source(root["source"])

    providers = _validate_providers(root["providers"])
    models = _validate_models(root["models"], providers)
    aliases = _validate_aliases(root["model_aliases"], providers, models)
    _validate_provider_model_references(root["providers"], models, aliases)
    _validate_pricing(root["pricing"], models)
    _validate_billing(root["token_billing"])


def _validate_source(value: Any) -> None:
    source = _object(
        value,
        "$.source",
        {
            "repository",
            "source_tag",
            "source_commit",
            "fetched_at",
            "license",
            "notice",
            "adapter",
            "files",
        },
    )
    expected = {
        "repository": SOURCE_REPOSITORY,
        "source_tag": SOURCE_TAG,
        "source_commit": SOURCE_COMMIT,
        "license": "MIT",
    }
    for key, wanted in expected.items():
        if source[key] != wanted:
            raise SchemaError(f"$.source.{key} must equal the pinned value {wanted!r}")
    _timestamp(source["fetched_at"], "$.source.fetched_at")

    notice = _object(
        source["notice"],
        "$.source.notice",
        {"path", "copyright", "license_file", "upstream_license_url"},
    )
    if notice["path"] != "NOTICE.txt" or notice["license_file"] != "NOTICE.txt":
        raise SchemaError("$.source.notice must point to the bundled NOTICE.txt")
    if notice["copyright"] != "Copyright (c) 2025 Jason Young":
        raise SchemaError("$.source.notice.copyright does not match v3.16.5")
    if notice["upstream_license_url"] != RAW_BASE + "LICENSE":
        raise SchemaError("$.source.notice.upstream_license_url is not commit-pinned")

    adapter = _object(
        source["adapter"],
        "$.source.adapter",
        {"mode", "dynamic_upstream", "note"},
    )
    if adapter["mode"] != "audited_fixture":
        raise SchemaError("$.source.adapter.mode must be audited_fixture")
    if adapter["dynamic_upstream"] != "unsupported_unstructured_source":
        raise SchemaError("dynamic upstream extraction must remain unsupported")
    _text(adapter["note"], "$.source.adapter.note", maximum=500)

    files = _list(source["files"], "$.source.files", maximum=32)
    if len(files) != len(EXPECTED_SOURCE_FILES):
        raise SchemaError(
            "$.source.files must contain every pinned verification anchor"
        )
    seen: set[str] = set()
    for index, value in enumerate(files):
        path = f"$.source.files[{index}]"
        item = _object(value, path, {"path", "url", "size", "sha256", "role"})
        source_path = _text(item["path"], f"{path}.path", maximum=300)
        if source_path in seen:
            raise SchemaError(f"duplicate source path: {source_path}")
        seen.add(source_path)
        expected_file = EXPECTED_SOURCE_FILES.get(source_path)
        if expected_file is None:
            raise SchemaError(f"unapproved source path: {source_path}")
        if (
            item["url"] not in ALLOWED_SOURCE_URLS
            or item["url"] != RAW_BASE + source_path
        ):
            raise SchemaError(f"{path}.url is not on the commit-pinned allowlist")
        if item["size"] != expected_file["size"]:
            raise SchemaError(f"{path}.size differs from the pinned source")
        if item["sha256"] != expected_file["sha256"] or not _SHA_RE.fullmatch(
            str(item["sha256"])
        ):
            raise SchemaError(f"{path}.sha256 differs from the pinned source")
        if item["role"] != expected_file["role"]:
            raise SchemaError(f"{path}.role differs from the pinned source")
    if seen != set(EXPECTED_SOURCE_FILES):
        raise SchemaError("$.source.files is incomplete")


def _validate_providers(value: Any) -> dict[str, Mapping[str, Any]]:
    items = _list(value, "$.providers", maximum=64)
    providers: dict[str, Mapping[str, Any]] = {}
    for index, value_item in enumerate(items):
        path = f"$.providers[{index}]"
        item = _object(
            value_item,
            path,
            {
                "id",
                "display_name",
                "category",
                "protocol",
                "base_url",
                "request_model_ids",
                "model_discovery",
            },
        )
        provider_id = _identifier(item["id"], f"{path}.id")
        if provider_id in providers:
            raise SchemaError(f"duplicate provider id: {provider_id}")
        _text(item["display_name"], f"{path}.display_name", maximum=100)
        if item["category"] not in {"first_party", "aggregator", "custom"}:
            raise SchemaError(f"{path}.category is unsupported")
        if item["protocol"] not in {
            "openai_compatible",
            "openai_responses",
            "anthropic",
            "custom",
        }:
            raise SchemaError(f"{path}.protocol is unsupported")
        if item["base_url"] is None:
            if item["category"] != "custom":
                raise SchemaError(
                    f"{path}.base_url may be null only for custom providers"
                )
        else:
            _https_url(item["base_url"], f"{path}.base_url")
        request_models = _list(
            item["request_model_ids"], f"{path}.request_model_ids", maximum=64
        )
        if item["category"] != "custom" and not request_models:
            raise SchemaError(f"{path}.request_model_ids cannot be empty")
        seen_models: set[str] = set()
        for model_index, request_id in enumerate(request_models):
            request_id = _alias(request_id, f"{path}.request_model_ids[{model_index}]")
            if request_id.casefold() in seen_models:
                raise SchemaError(
                    f"duplicate request model id in provider {provider_id}"
                )
            seen_models.add(request_id.casefold())
        if item["model_discovery"] not in {
            "same_origin_models_endpoint",
            "unsupported",
        }:
            raise SchemaError(f"{path}.model_discovery is unsupported")
        providers[provider_id] = item
    if not providers:
        raise SchemaError("$.providers cannot be empty")
    return providers


def _validate_models(
    value: Any, providers: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    items = _list(value, "$.models", maximum=256)
    models: dict[str, Mapping[str, Any]] = {}
    for index, value_item in enumerate(items):
        path = f"$.models[{index}]"
        item = _object(
            value_item,
            path,
            {
                "id",
                "display_name",
                "provider_ids",
                "context_tokens",
                "max_output_tokens",
            },
        )
        model_id = _identifier(item["id"], f"{path}.id")
        if model_id in models:
            raise SchemaError(f"duplicate model id: {model_id}")
        _text(item["display_name"], f"{path}.display_name", maximum=120)
        provider_ids = _list(item["provider_ids"], f"{path}.provider_ids", maximum=32)
        if not provider_ids:
            raise SchemaError(f"{path}.provider_ids cannot be empty")
        seen: set[str] = set()
        for provider_index, provider_id_value in enumerate(provider_ids):
            provider_id = _identifier(
                provider_id_value, f"{path}.provider_ids[{provider_index}]"
            )
            if provider_id not in providers:
                raise SchemaError(f"{path} references unknown provider {provider_id}")
            if provider_id in seen:
                raise SchemaError(f"{path} repeats provider {provider_id}")
            seen.add(provider_id)
        _token_limit(item["context_tokens"], f"{path}.context_tokens")
        _token_limit(item["max_output_tokens"], f"{path}.max_output_tokens")
        models[model_id] = item
    if not models:
        raise SchemaError("$.models cannot be empty")
    return models


def _validate_aliases(
    value: Any,
    providers: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    items = _list(value, "$.model_aliases", maximum=1024)
    aliases: dict[tuple[str, str], str] = {}
    for index, value_item in enumerate(items):
        path = f"$.model_aliases[{index}]"
        item = _object(
            value_item,
            path,
            {"provider_id", "alias", "canonical_model_id", "match"},
        )
        provider_id = item["provider_id"]
        if provider_id != "*":
            provider_id = _identifier(provider_id, f"{path}.provider_id")
            if provider_id not in providers:
                raise SchemaError(f"{path} references unknown provider {provider_id}")
        alias = _alias(item["alias"], f"{path}.alias")
        canonical = _identifier(
            item["canonical_model_id"], f"{path}.canonical_model_id"
        )
        if canonical not in models:
            raise SchemaError(f"{path} references unknown model {canonical}")
        if item["match"] != "exact":
            raise SchemaError(f"{path}.match must be exact")
        key = (provider_id, alias.casefold())
        if key in aliases:
            raise SchemaError(f"duplicate model alias for {provider_id}: {alias}")
        aliases[key] = canonical
    return aliases


def _validate_provider_model_references(
    provider_values: Any,
    models: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[tuple[str, str], str],
) -> None:
    for provider in provider_values:
        provider_id = provider["id"]
        for request_id in provider["request_model_ids"]:
            canonical = (
                request_id
                if request_id in models
                else aliases.get(
                    (provider_id, request_id.casefold()),
                    aliases.get(("*", request_id.casefold())),
                )
            )
            if canonical is None:
                raise SchemaError(
                    f"provider {provider_id} request model {request_id!r} has no exact alias"
                )
            if provider_id not in models[canonical]["provider_ids"]:
                raise SchemaError(
                    f"model {canonical} does not declare provider {provider_id}"
                )


def _validate_pricing(value: Any, models: Mapping[str, Mapping[str, Any]]) -> None:
    items = _list(value, "$.pricing", maximum=256)
    pricing_models: set[str] = set()
    for index, value_item in enumerate(items):
        path = f"$.pricing[{index}]"
        item = _object(
            value_item,
            path,
            {
                "model_id",
                "currency",
                "basis",
                "input",
                "output",
                "cache_read",
                "cache_write",
                "source_scope",
                "note",
            },
        )
        model_id = _identifier(item["model_id"], f"{path}.model_id")
        if model_id not in models:
            raise SchemaError(f"{path} references unknown model {model_id}")
        if model_id in pricing_models:
            raise SchemaError(f"duplicate pricing model: {model_id}")
        pricing_models.add(model_id)
        if not isinstance(item["currency"], str) or not _CURRENCY_RE.fullmatch(
            item["currency"]
        ):
            raise SchemaError(f"{path}.currency must be an ISO-style 3-letter code")
        if item["basis"] != "per_1m_tokens":
            raise SchemaError(f"{path}.basis must be per_1m_tokens")
        prices = [
            _price(item[name], f"{path}.{name}")
            for name in ("input", "output", "cache_read", "cache_write")
        ]
        if item["source_scope"] == "unsupported_by_pinned_cc_switch":
            if prices != ["unknown"] * 4:
                raise SchemaError(f"{path} unsupported prices must all be unknown")
        elif item["source_scope"] == "cc_switch_global_estimate":
            if prices[0] == "unknown" or prices[1] == "unknown":
                raise SchemaError(
                    f"{path} known estimates require input and output prices"
                )
        else:
            raise SchemaError(f"{path}.source_scope is unsupported")
        _text(item["note"], f"{path}.note", maximum=500)
    if pricing_models != set(models):
        missing = sorted(set(models) - pricing_models)
        extra = sorted(pricing_models - set(models))
        raise SchemaError(
            f"pricing must explicitly cover every model; missing={missing}, extra={extra}"
        )


def _validate_billing(value: Any) -> None:
    item = _object(
        value,
        "$.token_billing",
        {
            "basis_divisor",
            "dimensions",
            "input_token_semantics",
            "unknown_price_policy",
            "multiplier_policy",
            "rounding",
        },
    )
    if item["basis_divisor"] != 1_000_000:
        raise SchemaError("$.token_billing.basis_divisor must be 1000000")
    if item["dimensions"] != ["input", "output", "cache_read", "cache_write"]:
        raise SchemaError("$.token_billing.dimensions is unsupported")
    semantics = _object(
        item["input_token_semantics"],
        "$.token_billing.input_token_semantics",
        {"openai_compatible", "openai_responses", "anthropic"},
    )
    if semantics != {
        "openai_compatible": "input_includes_cache_read",
        "openai_responses": "input_includes_cache_read",
        "anthropic": "input_excludes_cache_read",
    }:
        raise SchemaError("$.token_billing.input_token_semantics is unsupported")
    if item["unknown_price_policy"] != "unavailable_not_zero":
        raise SchemaError("unknown prices must never be coerced to zero")
    if item["multiplier_policy"] != "after_component_sum":
        raise SchemaError("$.token_billing.multiplier_policy is unsupported")
    if item["rounding"] != "decimal_no_binary_float":
        raise SchemaError("$.token_billing.rounding is unsupported")


def _object(value: Any, path: str, required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - required
    if missing:
        raise SchemaError(f"{path} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{path} has unknown fields: {sorted(unknown)}")
    return value


def _list(value: Any, path: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    if len(value) > maximum:
        raise SchemaError(f"{path} exceeds {maximum} items")
    return value


def _text(value: Any, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SchemaError(f"{path} must be a non-empty string up to {maximum} chars")
    if any(ord(character) < 32 for character in value) or "<" in value or ">" in value:
        raise SchemaError(f"{path} contains unsafe characters")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise SchemaError(f"{path} is not a safe identifier")
    return value


def _alias(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ALIAS_RE.fullmatch(value):
        raise SchemaError(f"{path} is not a safe exact model alias")
    return value


def _https_url(value: Any, path: str) -> str:
    value = _text(value, path, maximum=500)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SchemaError(f"{path} must be an HTTPS base URL without credentials/query")
    return value


def _timestamp(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SchemaError(f"{path} must be an RFC3339 UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SchemaError(f"{path} must be an RFC3339 UTC timestamp") from exc


def _token_limit(value: Any, path: str) -> None:
    if value == "unknown":
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 10_000_000
    ):
        raise SchemaError(f"{path} must be a positive integer or 'unknown'")


def _price(value: Any, path: str) -> str:
    if value == "unknown":
        return value
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise SchemaError(f"{path} must be a non-negative decimal string or 'unknown'")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise SchemaError(f"{path} is not a decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise SchemaError(f"{path} must be finite and non-negative")
    return value


def _reject_unsafe_content(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_KEYS:
                raise SchemaError(
                    f"{path} contains forbidden secret-bearing field {key!r}"
                )
            _reject_unsafe_content(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unsafe_content(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if (
            any(ord(character) < 32 for character in value)
            or "<" in value
            or ">" in value
        ):
            raise SchemaError(f"{path} contains unsafe text")
