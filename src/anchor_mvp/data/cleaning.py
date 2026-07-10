"""Teacher response extraction, cleaning, safety checks, and deduplication."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .schema import DataValidationError, SeedDemand, TaskType


_PREFACE = re.compile(
    r"^\s*(?:sure[,!.]?|okay[,!.]?|certainly[,!.]?|here(?:'s| is)[^:\n]*:?)\s*",
    flags=re.IGNORECASE,
)

# Actual exploit strings do not belong in this dataset. Labels and inert tokens do.
_ACTIVE_PAYLOADS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"\b(?:union\s+select|drop\s+table)\b", re.IGNORECASE),
    re.compile(r"\b(?:eval|exec)\s*\(", re.IGNORECASE),
    re.compile(r"\b(?:stratum\+tcp|coinhive)\b", re.IGNORECASE),
)

_SECURITY_REVIEWED_CODE_FORMS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\binnerHTML\b", re.IGNORECASE),
    # Reject raw HTML-style inline handlers such as ``onerror=...`` while
    # allowing ordinary React/JSX event props whose value is a code expression,
    # for example ``onClick={handleClick}``.
    re.compile(r"<[^>]*\bon[A-Za-z]+\s*=\s*(?!\{)", re.IGNORECASE),
)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the first complete JSON object without trusting prose or fences."""

    candidate = _PREFACE.sub("", raw.strip())
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```\s*$", "", candidate)
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(candidate) if char == "{"]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise DataValidationError("teacher response does not contain a valid JSON object")


def extract_frontend_payload(raw: str) -> tuple[dict[str, Any], str]:
    """Accept canonical JSON or one fenced TS/JS code artifact."""

    try:
        return extract_json_object(raw), "json"
    except DataValidationError:
        blocks = re.findall(
            r"```(?:tsx|typescript|jsx|javascript|ts|js)\s*\n([\s\S]*?)```",
            raw,
            flags=re.IGNORECASE,
        )
        if len(blocks) != 1 or not blocks[0].strip():
            raise
        return (
            {
                "decision_trace": [],
                "output": {"language": "tsx", "code": blocks[0].strip()},
            },
            "fenced_code",
        )


def contains_active_payload(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(contains_active_payload(key) or contains_active_payload(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_active_payload(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _ACTIVE_PAYLOADS)
    return False


def sanitize_security_seed(seed: SeedDemand) -> SeedDemand:
    request = seed.request
    for pattern in _ACTIVE_PAYLOADS:
        request = pattern.sub("[DEFENSIVE_ACTIVE_CONTENT_PLACEHOLDER]", request)
    return SeedDemand(
        seed_id=seed.seed_id,
        title=seed.title,
        request=request,
        category=seed.category,
        tags=seed.tags,
    )


def validate_safe_payload(task_type: TaskType, payload: Mapping[str, Any]) -> None:
    if task_type in ("frontend", "review"):
        output = payload.get("output", {})
        code = output.get("code", "") if isinstance(output, Mapping) else ""
        if isinstance(code, str) and len(code) > 12_000:
            raise DataValidationError(
                f"{task_type} output.code exceeds the 12000-character hard limit"
            )
    if task_type in ("plan", "tool_policy") and contains_active_payload(payload):
        raise DataValidationError(
            f"{task_type} records may contain defensive labels, not active payloads"
        )
    if task_type == "security" and contains_active_payload(payload):
        raise DataValidationError("security records may contain labels/placeholders, not active payloads")
    if task_type == "security":
        raw_input = payload.get("input", {})
        reviewed_code = raw_input.get("reviewed_code", "") if isinstance(raw_input, Mapping) else ""
        if isinstance(reviewed_code, str) and any(
            pattern.search(reviewed_code) for pattern in _SECURITY_REVIEWED_CODE_FORMS
        ):
            raise DataValidationError(
                "security reviewed_code must use safe code or inert placeholders"
            )
    if task_type == "review" and contains_active_payload(payload):
        raise DataValidationError("review records may contain bugs, not active payloads")
