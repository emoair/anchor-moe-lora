"""Teacher response extraction, cleaning, safety checks, and deduplication."""

from __future__ import annotations

from hashlib import sha256
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

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?<![A-Za-z0-9_-])ark-[A-Za-z0-9][A-Za-z0-9_-]{20,127}"
        r"(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)"
        r"\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        re.IGNORECASE,
    ),
)

_SECURITY_REVIEWED_CODE_FORMS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\binnerHTML\b", re.IGNORECASE),
    # Reject raw HTML-style inline handlers such as ``onerror=...`` while
    # allowing ordinary React/JSX event props whose value is a code expression,
    # for example ``onClick={handleClick}``.
    re.compile(r"<[^>]*\bon[A-Za-z]+\s*=\s*(?!\{)", re.IGNORECASE),
)

SECURITY_FIXTURE_GENERATOR_VERSION = "anchor-inert-security-fixtures-v1"
SECURITY_FIXTURE_ORACLE_VERSION = "anchor-security-fixture-gold-v1"

_SECURITY_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "fixture_id": "boundary-react-escaped-text",
        "kind": "benign_boundary",
        "marker": "[INERT_SECURITY_BOUNDARY:REACT_ESCAPED_TEXT]",
        "decision": "PASS",
        "rationale": "The inert boundary fixture describes framework-escaped text rendering.",
        "findings": [],
    },
    {
        "fixture_id": "negative-untrusted-html-sink",
        "kind": "safe_negative",
        "marker": "[INERT_SECURITY_NEGATIVE:UNTRUSTED_HTML_SINK]",
        "decision": "BLOCK",
        "rationale": "The inert negative fixture labels an untrusted HTML sink requiring remediation.",
        "findings": ["INERT_UNTRUSTED_HTML_SINK"],
    },
)


def build_inert_security_fixture(
    reviewed_code: str, index: int
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Attach one non-executable security fixture and its deterministic gold label."""

    fixture = _SECURITY_FIXTURES[index % len(_SECURITY_FIXTURES)]
    marker = str(fixture["marker"])
    candidate = (
        f"{reviewed_code.rstrip()}\n\n/* {marker} descriptor-only; never execute */"
    )
    if contains_active_payload(
        candidate
    ):  # defensive invariant; fixture text must remain inert
        raise DataValidationError(
            "security fixture generator produced active payload material"
        )
    output = {
        "decision": fixture["decision"],
        "rationale": fixture["rationale"],
        "findings": list(fixture["findings"]),
    }
    canonical = json.dumps(
        output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    manifest = {
        "generator": SECURITY_FIXTURE_GENERATOR_VERSION,
        "fixture_id": fixture["fixture_id"],
        "kind": fixture["kind"],
        "expected_decision": fixture["decision"],
        "active_payload_present": False,
        "gold_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return candidate, output, manifest


def deterministic_security_fixture_oracle(
    reviewed_code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the trusted label from a stored inert fixture, never a model claim.

    The fixture marker is a local, descriptor-only suffix added by
    :func:`build_inert_security_fixture`.  Rebuilding the same suffix lets the
    bulk gate reject stale teacher-decided rows that merely resemble this
    schema, without executing or interpreting the submitted code.
    """

    for index, fixture in enumerate(_SECURITY_FIXTURES):
        suffix = f"\n\n/* {fixture['marker']} descriptor-only; never execute */"
        if not reviewed_code.endswith(suffix):
            continue
        source = reviewed_code[: -len(suffix)]
        candidate, output, manifest = build_inert_security_fixture(source, index)
        if candidate != reviewed_code:
            raise DataValidationError(
                "security fixture is not a canonical deterministic suffix"
            )
        return output, manifest
    raise DataValidationError(
        "security reviewed_code lacks a recognized inert fixture suffix"
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
        return any(
            contains_active_payload(key) or contains_active_payload(item)
            for key, item in value.items()
        )
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
        card_id=seed.card_id,
        seed_index=seed.seed_index,
        template_id=seed.template_id,
        source_kind=seed.source_kind,
        source_digest=seed.source_digest,
    )


def redact_active_payload_material(value: Any) -> tuple[Any, int]:
    """Replace executable payload markers in descriptive teacher structures.

    This is intentionally used only for non-code planning/policy records.  It
    preserves the teacher's surrounding explanation while ensuring that an
    otherwise defensive mention such as ``avoid eval(...)`` cannot place a
    live payload marker in the training corpus.  Code-producing stages remain
    fail-closed and are never passed through this redactor.
    """

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        total = 0
        for key, item in value.items():
            clean_item, count = redact_active_payload_material(item)
            redacted[key] = clean_item
            total += count
        return redacted, total
    if isinstance(value, list):
        redacted_items: list[Any] = []
        total = 0
        for item in value:
            clean_item, count = redact_active_payload_material(item)
            redacted_items.append(clean_item)
            total += count
        return redacted_items, total
    if isinstance(value, tuple):
        clean_items, total = redact_active_payload_material(list(value))
        return tuple(clean_items), total
    if not isinstance(value, str):
        return value, 0
    text = value
    total = 0
    for pattern in _ACTIVE_PAYLOADS:
        text, count = pattern.subn("[DEFENSIVE_ACTIVE_CONTENT_PLACEHOLDER]", text)
        total += count
    return text, total


def validate_safe_payload(task_type: TaskType, payload: Mapping[str, Any]) -> None:
    if contains_secret_material(payload):
        raise DataValidationError(
            f"{task_type} record contains credential-like material"
        )
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
        raise DataValidationError(
            "security records may contain labels/placeholders, not active payloads"
        )
    if task_type == "security":
        raw_input = payload.get("input", {})
        reviewed_code = (
            raw_input.get("reviewed_code", "") if isinstance(raw_input, Mapping) else ""
        )
        if isinstance(reviewed_code, str) and any(
            pattern.search(reviewed_code) for pattern in _SECURITY_REVIEWED_CODE_FORMS
        ):
            raise DataValidationError(
                "security reviewed_code must use safe code or inert placeholders"
            )
    if task_type == "review" and contains_active_payload(payload):
        raise DataValidationError(
            "review records may contain bugs, not active payloads"
        )


def contains_secret_material(value: Any) -> bool:
    """Detect credential-like material without retaining or reporting its value."""

    if isinstance(value, Mapping):
        return any(
            contains_secret_material(key) or contains_secret_material(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_secret_material(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _SECRET_PATTERNS)
    return False
