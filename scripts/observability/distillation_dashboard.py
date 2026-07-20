#!/usr/bin/env python3
"""Content-free distillation telemetry and strict local subprocess control.

The dashboard intentionally has no import dependency on ``anchor_mvp`` so it can
observe a running collection without changing its process or package state.
Only whitelisted metadata is materialized from JSONL rows. Prompt, message,
output, code, environment, and credential fields are skipped by the selective
JSON scanner and are never placed in a snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from http.cookies import SimpleCookie
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

try:
    from distillation_control import (
        ControlError,
        ControlPlane,
        csrf_cookie_value,
        csrf_token,
    )
    from distillation_catalog import CatalogService
except ModuleNotFoundError:  # Imported as a repository namespace module in tests.
    from scripts.observability.distillation_control import (
        ControlError,
        ControlPlane,
        csrf_cookie_value,
        csrf_token,
    )
    from scripts.observability.distillation_catalog import CatalogService


SCHEMA_VERSION = "anchor.distillation-dashboard.v1"
DIAGNOSTIC_SCHEMA_VERSION = "anchor.distillation-diagnostics.v1"
DIAGNOSTIC_REASON_CODES = frozenset(
    {
        "telemetry_cold_start",
        "telemetry_warming",
        "status_stale",
        "file_parse_error",
        "provider_cooldown",
        "rate_limit",
        "quota",
        "client_deadline",
        "process_exit",
        "unknown_counter",
    }
)
STAGE_FILES = {
    "plan": "data_plan.jsonl",
    "tool_policy": "data_tool_policy.jsonl",
    "frontend": "data_frontend.jsonl",
    "review": "data_review.jsonl",
    "security": "data_security.jsonl",
}
STAGE_ORDER = tuple(STAGE_FILES)
SEED_FILE = "seeds.jsonl"
SEED_REJECTIONS_FILE = "seed_rejections.jsonl"
ATTEMPTS_FILE = Path("automation") / "attempts.jsonl"
STATUS_FILE = Path("automation") / "status.json"
MAX_PARSE_ERRORS = 50
MAX_LOG_ENTRIES = 160
MAX_POST_BODY_BYTES = 16_384
STATUS_FRESHNESS_GRACE_SECONDS = 30
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SAFE_ENUM_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
NUMBER_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

PathKey = tuple[str, ...]

RECORD_PATHS: frozenset[PathKey] = frozenset(
    {
        ("seed_id",),
        ("provenance", "seed_id"),
        (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
            "input_tokens",
        ),
        (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
            "output_tokens",
        ),
        (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
            "total_tokens",
        ),
        (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
            "cache_read_tokens",
        ),
        (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
            "cache_write_tokens",
        ),
        ("provenance", "teacher", "model"),
        ("provenance", "teacher", "base_url"),
        ("provenance", "teacher", "protocol"),
        ("provenance", "teacher", "provider", "model"),
        ("provenance", "teacher", "provider", "base_url"),
        ("provenance", "teacher", "provider", "protocol"),
        (
            "provenance",
            "teacher",
            "provider",
            "attempts",
            "wire_attempts",
        ),
        (
            "provenance",
            "teacher",
            "provider",
            "attempts",
            "retry_count",
        ),
        (
            "provenance",
            "teacher",
            "generation_params",
            "max_output_tokens_total",
        ),
        (
            "provenance",
            "teacher",
            "generation_params",
            "max_requests",
        ),
    }
)
ATTEMPT_PATHS: frozenset[PathKey] = frozenset(
    {
        ("error_class",),
        ("task_type",),
        ("attempt_number",),
        ("observed_at",),
        ("outcome",),
    }
)
SEED_REJECTION_PATHS: frozenset[PathKey] = frozenset(
    {
        ("error_class",),
        ("reason",),
        ("content_retained",),
        ("observed_at",),
    }
)
STATUS_PATHS: frozenset[PathKey] = frozenset(
    {
        ("state",),
        ("updated_at",),
        ("cooldown_until",),
        ("quota_epoch", "requests_used"),
        ("quota_epoch", "output_tokens_used"),
        ("quota_epoch", "max_requests"),
        ("quota_epoch", "max_output_tokens_total"),
        ("audit_ledger", "requests_total"),
        ("audit_ledger", "output_tokens_total"),
        ("usage_checkpoint_policy", "maximum_seconds"),
    }
)


class MetadataJsonError(ValueError):
    """A content-free parse error for one metadata source."""


class SelectiveJsonScanner:
    """Validate JSON while materializing only explicitly selected scalar paths.

    Values outside selected path prefixes are syntax-scanned in their byte form.
    In particular, message and code strings are not decoded into Python strings.
    """

    def __init__(self, raw: bytes, selected: Iterable[PathKey]) -> None:
        self.raw = raw
        self.length = len(raw)
        self.position = 0
        self.selected = frozenset(selected)
        self.prefixes = {
            path[:index] for path in self.selected for index in range(len(path) + 1)
        }
        self.values: dict[PathKey, object] = {}

    def scan(self) -> dict[PathKey, object]:
        self._space()
        self._parse_value(())
        self._space()
        if self.position != self.length:
            raise MetadataJsonError("trailing data")
        return self.values

    def _space(self) -> None:
        while self.position < self.length and self.raw[self.position] in b" \t\r\n":
            self.position += 1

    def _parse_value(self, path: PathKey) -> None:
        self._space()
        if self.position >= self.length:
            raise MetadataJsonError("unexpected end")
        if path not in self.prefixes:
            self._skip_value()
            return
        marker = self.raw[self.position]
        if marker == ord("{"):
            self._parse_object(path)
            return
        if marker == ord("["):
            self._skip_value()
            return
        value = self._parse_scalar(decode=path in self.selected)
        if path in self.selected:
            self.values[path] = value

    def _parse_object(self, path: PathKey) -> None:
        self.position += 1
        self._space()
        if self._take(ord("}")):
            return
        while True:
            if self.position >= self.length or self.raw[self.position] != ord('"'):
                raise MetadataJsonError("object key expected")
            key = self._string(decode=True)
            if not isinstance(key, str):
                raise MetadataJsonError("invalid object key")
            self._space()
            if not self._take(ord(":")):
                raise MetadataJsonError("colon expected")
            self._parse_value((*path, key))
            self._space()
            if self._take(ord("}")):
                return
            if not self._take(ord(",")):
                raise MetadataJsonError("object separator expected")
            self._space()

    def _skip_value(self) -> None:
        self._space()
        if self.position >= self.length:
            raise MetadataJsonError("unexpected end")
        marker = self.raw[self.position]
        if marker == ord('"'):
            self._string(decode=False)
            return
        if marker == ord("{"):
            self.position += 1
            self._space()
            if self._take(ord("}")):
                return
            while True:
                if self.position >= self.length or self.raw[self.position] != ord('"'):
                    raise MetadataJsonError("object key expected")
                self._string(decode=False)
                self._space()
                if not self._take(ord(":")):
                    raise MetadataJsonError("colon expected")
                self._skip_value()
                self._space()
                if self._take(ord("}")):
                    return
                if not self._take(ord(",")):
                    raise MetadataJsonError("object separator expected")
                self._space()
        elif marker == ord("["):
            self.position += 1
            self._space()
            if self._take(ord("]")):
                return
            while True:
                self._skip_value()
                self._space()
                if self._take(ord("]")):
                    return
                if not self._take(ord(",")):
                    raise MetadataJsonError("array separator expected")
                self._space()
        else:
            self._parse_scalar(decode=False)

    def _parse_scalar(self, *, decode: bool) -> object:
        if self.position >= self.length:
            raise MetadataJsonError("unexpected end")
        if self.raw[self.position] == ord('"'):
            return self._string(decode=decode)
        for literal, value in ((b"true", True), (b"false", False), (b"null", None)):
            if self.raw.startswith(literal, self.position):
                self.position += len(literal)
                return value if decode else None
        match = NUMBER_RE.match(self.raw, self.position)
        if match is None:
            raise MetadataJsonError("scalar expected")
        token = match.group(0)
        self.position = match.end()
        if not decode:
            return None
        try:
            return json.loads(token.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MetadataJsonError("invalid number") from error

    def _string(self, *, decode: bool) -> object:
        start = self.position
        self.position += 1
        while self.position < self.length:
            byte = self.raw[self.position]
            if byte == ord('"'):
                self.position += 1
                if not decode:
                    return None
                token = self.raw[start : self.position]
                try:
                    value = json.loads(token.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise MetadataJsonError("invalid string") from error
                if not isinstance(value, str):
                    raise MetadataJsonError("invalid string")
                return value
            if byte < 0x20:
                raise MetadataJsonError("control byte in string")
            if byte == ord("\\"):
                self.position += 1
                if self.position >= self.length:
                    raise MetadataJsonError("incomplete escape")
                escape = self.raw[self.position]
                if escape == ord("u"):
                    end = self.position + 5
                    if end > self.length or any(
                        char not in b"0123456789abcdefABCDEF"
                        for char in self.raw[self.position + 1 : end]
                    ):
                        raise MetadataJsonError("invalid unicode escape")
                    self.position = end
                    continue
                if escape not in b'"\\/bfnrt':
                    raise MetadataJsonError("invalid escape")
            self.position += 1
        raise MetadataJsonError("unterminated string")

    def _take(self, marker: int) -> bool:
        if self.position < self.length and self.raw[self.position] == marker:
            self.position += 1
            return True
        return False


def scan_metadata(raw: bytes, selected: Iterable[PathKey]) -> dict[PathKey, object]:
    """Return selected scalar metadata without materializing body fields."""

    return SelectiveJsonScanner(raw, selected).scan()


def _safe_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_enum(value: object, *, fallback_prefix: str) -> str:
    if isinstance(value, str) and SAFE_ENUM_RE.fullmatch(value):
        return value
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:10]
    return f"{fallback_prefix}-{digest}"


def _seed_rejection_reason(value: object) -> str:
    """Map persisted validation text to a finite content-free reason code."""

    if not isinstance(value, str):
        return "unclassified_validation"
    normalized = " ".join(value.strip().casefold().split())
    if normalized == "seed contains active payload material":
        return "active_payload_material"
    if normalized == "seed contains credential-like material":
        return "credential_like_material"
    if "valid json object" in normalized:
        return "invalid_json_object"
    if normalized == "seed request is empty":
        return "empty_request"
    if "task-card" in normalized or "task card" in normalized:
        return "task_card_binding"
    if normalized.startswith("seed "):
        return "seed_schema"
    return "unclassified_validation"


def _safe_observed_at(value: object) -> str | None:
    if not isinstance(value, str) or not 10 <= len(value) <= 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _safe_model_binding(value: object) -> str | None:
    if isinstance(value, str) and SAFE_MODEL_ID_RE.fullmatch(value):
        return value
    return None


def _safe_protocol_binding(value: object) -> str | None:
    if value in {"openai", "openai_compatible", "openai_responses", "anthropic"}:
        return str(value)
    return None


def _safe_base_url_binding(value: object) -> str | None:
    if not isinstance(value, str) or value != value.strip() or len(value) > 2048:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    return value


def _iso_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _metric(
    value: int | float | None,
    *,
    exact: bool,
    unknown_rows: int = 0,
    source: str,
) -> dict[str, object]:
    return {
        "value": value,
        "exact": exact,
        "unknown_rows": unknown_rows,
        "source": source,
    }


def _sum_public_metrics(metrics: Sequence[object], *, source: str) -> dict[str, object]:
    values: list[float] = []
    exact = bool(metrics)
    unknown = 0
    for item in metrics:
        if not isinstance(item, Mapping) or item.get("value") is None:
            exact = False
            unknown += 1
            continue
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            exact = False
            unknown += 1
            continue
        values.append(float(value))
        exact = exact and item.get("exact") is True
    if not values:
        return _metric(None, exact=False, unknown_rows=unknown, source=source)
    return _metric(
        round(sum(values), 6),
        exact=exact,
        unknown_rows=unknown,
        source=source,
    )


def _public_connection_diagnostics(
    shards: Sequence[Mapping[str, object]],
    control: Mapping[str, object] | None,
    *,
    observed_at: str,
) -> dict[str, object]:
    """Compose one finite diagnostic view without process discovery or secrets."""

    reasons: list[str] = []
    shard_observed_at: list[str] = []
    summaries: list[str] = []
    for shard in shards:
        diagnostics = shard.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        summaries.append(str(diagnostics.get("summary", "attention")))
        candidate = _safe_observed_at(diagnostics.get("observed_at"))
        if candidate is not None:
            shard_observed_at.append(candidate)
        raw_reasons = diagnostics.get("reason_codes")
        if isinstance(raw_reasons, list):
            reasons.extend(
                reason
                for reason in raw_reasons
                if isinstance(reason, str) and reason in DIAGNOSTIC_REASON_CODES
            )

    output_label = control.get("output_label") if control is not None else None
    managed = isinstance(output_label, str) and any(
        shard.get("label") == output_label for shard in shards
    )
    process_state = control.get("process_state") if control is not None else None
    if managed:
        ownership = "managed"
        process_alive: bool | None = process_state in {
            "starting",
            "running",
            "stopping",
            "terminating",
        }
        if process_state in {"exited", "failed", "reconnect_wait"}:
            process_alive = False
            reasons.append("process_exit")
    else:
        ownership = "external_read_only"
        process_alive = None

    reconnect = control.get("reconnect") if control is not None else None
    reconnect_public: dict[str, object] = {
        "applicable": managed,
        "used": None,
        "maximum": None,
        "next_at": None,
    }
    if managed and isinstance(reconnect, Mapping):
        reconnect_public["used"] = _safe_nonnegative_int(reconnect.get("used"))
        reconnect_public["maximum"] = _safe_nonnegative_int(reconnect.get("maximum"))
        reconnect_public["next_at"] = _safe_observed_at(reconnect.get("next_at"))

    exit_code = control.get("exit_code") if managed and control is not None else None
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None
    exit_signal = None
    if exit_code is not None and -64 <= exit_code < 0:
        exit_signal = -exit_code

    reasons = list(dict.fromkeys(reasons))
    if not set(reasons) <= DIAGNOSTIC_REASON_CODES:
        raise RuntimeError("diagnostic reason schema drift")
    hard_reasons = set(reasons) - {
        "telemetry_cold_start",
        "telemetry_warming",
    }
    if hard_reasons:
        summary = "attention"
    elif any(item == "normal_warming" for item in summaries):
        summary = "normal_warming"
    elif summaries and all(item == "complete" for item in summaries):
        summary = "complete"
    else:
        summary = "running"
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "collector_alive": True,
        "process_alive": process_alive,
        "ownership": ownership,
        "summary": summary,
        "reason_codes": reasons,
        "observed_at": max(shard_observed_at, default=observed_at),
        "reconnect": reconnect_public,
        "last_exit": {"code": exit_code, "signal": exit_signal},
    }


@dataclass
class StageAggregate:
    rows: int = 0
    seed_ids: set[str] = field(default_factory=set)
    unknown_seed_rows: int = 0
    token_values: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "total": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    token_unknown_rows: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "total": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    wire_attempts: int = 0
    wire_unknown_rows: int = 0
    retries: int = 0
    retry_unknown_rows: int = 0
    output_token_budgets: set[int] = field(default_factory=set)
    request_budgets: set[int] = field(default_factory=set)
    provider_models: set[str] = field(default_factory=set)
    provider_protocols: set[str] = field(default_factory=set)
    provider_base_urls: set[str] = field(default_factory=set)
    provider_model_unknown_rows: int = 0
    provider_protocol_unknown_rows: int = 0
    provider_base_url_unknown_rows: int = 0

    def add(self, metadata: Mapping[PathKey, object]) -> None:
        self.rows += 1
        seed = metadata.get(("provenance", "seed_id"))
        if isinstance(seed, str) and seed:
            self.seed_ids.add(seed)
        else:
            self.unknown_seed_rows += 1
        usage_root = (
            "provenance",
            "teacher",
            "provider",
            "completion",
            "usage",
        )
        for public_name, provider_name in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("total", "total_tokens"),
            ("cache_read", "cache_read_tokens"),
            ("cache_write", "cache_write_tokens"),
        ):
            value = _safe_nonnegative_int(metadata.get((*usage_root, provider_name)))
            if value is None:
                self.token_unknown_rows[public_name] += 1
            else:
                self.token_values[public_name] += value
        attempts_root = (
            "provenance",
            "teacher",
            "provider",
            "attempts",
        )
        wire = _safe_nonnegative_int(metadata.get((*attempts_root, "wire_attempts")))
        if wire is None or wire < 1:
            self.wire_unknown_rows += 1
        else:
            self.wire_attempts += wire
        retry = _safe_nonnegative_int(metadata.get((*attempts_root, "retry_count")))
        if retry is None:
            self.retry_unknown_rows += 1
        else:
            self.retries += retry
        generation_root = ("provenance", "teacher", "generation_params")
        token_budget = _safe_nonnegative_int(
            metadata.get((*generation_root, "max_output_tokens_total"))
        )
        request_budget = _safe_nonnegative_int(
            metadata.get((*generation_root, "max_requests"))
        )
        if token_budget:
            self.output_token_budgets.add(token_budget)
        if request_budget:
            self.request_budgets.add(request_budget)
        provider_root = ("provenance", "teacher", "provider")
        teacher_root = ("provenance", "teacher")
        model = _safe_model_binding(metadata.get((*provider_root, "model")))
        if model is None:
            model = _safe_model_binding(metadata.get((*teacher_root, "model")))
        if model is None:
            self.provider_model_unknown_rows += 1
        else:
            self.provider_models.add(model)
        protocol = _safe_protocol_binding(metadata.get((*provider_root, "protocol")))
        if protocol is None:
            protocol = _safe_protocol_binding(metadata.get((*teacher_root, "protocol")))
        if protocol is None:
            self.provider_protocol_unknown_rows += 1
        else:
            self.provider_protocols.add(protocol)
        base_url = _safe_base_url_binding(metadata.get((*provider_root, "base_url")))
        if base_url is None:
            base_url = _safe_base_url_binding(metadata.get((*teacher_root, "base_url")))
        if base_url is None:
            self.provider_base_url_unknown_rows += 1
        else:
            self.provider_base_urls.add(base_url)


@dataclass
class SeedAggregate:
    rows: int = 0
    seed_ids: set[str] = field(default_factory=set)
    unknown_seed_rows: int = 0

    def add(self, metadata: Mapping[PathKey, object]) -> None:
        self.rows += 1
        seed = metadata.get(("seed_id",))
        if isinstance(seed, str) and seed:
            self.seed_ids.add(seed)
        else:
            self.unknown_seed_rows += 1


@dataclass
class AttemptAggregate:
    rows: int = 0
    by_type: Counter[str] = field(default_factory=Counter)

    def add(self, metadata: Mapping[PathKey, object]) -> None:
        self.rows += 1
        error_class = _safe_enum(
            metadata.get(("error_class",)), fallback_prefix="unknown-error"
        )
        self.by_type[error_class] += 1


@dataclass
class SeedRejectionAggregate:
    rows: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)
    recent: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=8))

    def add(self, metadata: Mapping[PathKey, object]) -> None:
        self.rows += 1
        if metadata.get(("content_retained",)) is not False:
            reason = "metadata_policy_violation"
        else:
            reason = _seed_rejection_reason(metadata.get(("reason",)))
        error_class = _safe_enum(
            metadata.get(("error_class",)), fallback_prefix="unknown-error"
        )
        self.by_reason[reason] += 1
        self.recent.append(
            {
                "reason": reason,
                "error_class": error_class,
                "observed_at": _safe_observed_at(metadata.get(("observed_at",))),
            }
        )


@dataclass(frozen=True)
class ParseError:
    source: str
    line: int
    sha256: str

    def public(self) -> dict[str, object]:
        return {"source": self.source, "line": self.line, "sha256": self.sha256}


class IncrementalJsonl:
    """Append-aware JSONL reader with per-file aggregate state."""

    def __init__(self, path: Path, source: str, kind: str) -> None:
        self.path = path
        self.source = source
        self.kind = kind
        self.offset = 0
        self.partial = b""
        self.line_number = 0
        self.identity: tuple[int, int] | None = None
        self.last_mtime: float | None = None
        self.parse_errors: deque[ParseError] = deque(maxlen=MAX_PARSE_ERRORS)
        self.bytes_read_total = 0
        self.aggregate: (
            StageAggregate | SeedAggregate | AttemptAggregate | SeedRejectionAggregate
        )
        self._reset_aggregate()

    def _reset_aggregate(self) -> None:
        if self.kind == "stage":
            self.aggregate = StageAggregate()
        elif self.kind == "seed":
            self.aggregate = SeedAggregate()
        elif self.kind == "attempt":
            self.aggregate = AttemptAggregate()
        elif self.kind == "seed_rejection":
            self.aggregate = SeedRejectionAggregate()
        else:
            raise ValueError(f"unknown JSONL kind: {self.kind}")
        self.parse_errors.clear()

    def _reset_file(self) -> None:
        self.offset = 0
        self.partial = b""
        self.line_number = 0
        self._reset_aggregate()

    def refresh(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self.offset or self.line_number:
                self.identity = None
                self.last_mtime = None
                self._reset_file()
                return True
            return False
        identity = (int(stat.st_dev), int(stat.st_ino))
        if self.identity is not None and (
            identity != self.identity or stat.st_size < self.offset
        ):
            self._reset_file()
        self.identity = identity
        self.last_mtime = stat.st_mtime
        if stat.st_size == self.offset:
            return False
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        self.offset += len(chunk)
        self.bytes_read_total += len(chunk)
        if not chunk:
            return False
        blocks = (self.partial + chunk).split(b"\n")
        self.partial = blocks.pop()
        for block in blocks:
            self.line_number += 1
            raw = block[:-1] if block.endswith(b"\r") else block
            if not raw.strip():
                continue
            try:
                selected = RECORD_PATHS
                if self.kind == "seed":
                    selected = frozenset({("seed_id",)})
                elif self.kind == "attempt":
                    selected = ATTEMPT_PATHS
                elif self.kind == "seed_rejection":
                    selected = SEED_REJECTION_PATHS
                metadata = scan_metadata(raw, selected)
                self.aggregate.add(metadata)
            except (MetadataJsonError, ValueError, TypeError):
                self.parse_errors.append(
                    ParseError(
                        source=self.source,
                        line=self.line_number,
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
        return True


class StatusReader:
    """Cached selective reader for the non-append automation status file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.signature: tuple[int, int] | None = None
        self.metadata: dict[PathKey, object] = {}
        self.last_mtime: float | None = None
        self.invalid_sha256: str | None = None

    def refresh(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            changed = self.signature is not None
            self.signature = None
            self.metadata = {}
            self.last_mtime = None
            self.invalid_sha256 = None
            return changed
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return False
        self.signature = signature
        self.last_mtime = stat.st_mtime
        raw = self.path.read_bytes()
        try:
            self.metadata = scan_metadata(raw, STATUS_PATHS)
            self.invalid_sha256 = None
        except (MetadataJsonError, ValueError, TypeError):
            self.metadata = {}
            self.invalid_sha256 = hashlib.sha256(raw).hexdigest()
        return True


def _sum_token_metric(stages: Sequence[StageAggregate], name: str) -> dict[str, object]:
    value = sum(stage.token_values[name] for stage in stages)
    unknown = sum(stage.token_unknown_rows[name] for stage in stages)
    return _metric(
        value,
        exact=unknown == 0,
        unknown_rows=unknown,
        source="retained_stage_provider_usage_subtotal",
    )


def _known_percent(value: int, budgets: set[int]) -> dict[str, object]:
    if not budgets:
        return _metric(None, exact=False, source="unknown")
    budget = max(budgets)
    percent = min(100.0, value * 100.0 / budget) if budget else None
    return _metric(
        round(percent, 4) if percent is not None else None,
        exact=len(budgets) == 1,
        source="provider_usage_lower_bound",
    )


class ShardMonitor:
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        self.seed_reader = IncrementalJsonl(path / SEED_FILE, "seeds", "seed")
        self.rejection_reader = IncrementalJsonl(
            path / SEED_REJECTIONS_FILE, "seed_rejections", "seed_rejection"
        )
        self.stage_readers = {
            stage: IncrementalJsonl(path / filename, stage, "stage")
            for stage, filename in STAGE_FILES.items()
        }
        self.attempt_reader = IncrementalJsonl(
            path / ATTEMPTS_FILE, "automation_attempts", "attempt"
        )
        self.status_reader = StatusReader(path / STATUS_FILE)
        self.history: deque[tuple[float, int]] = deque(maxlen=360)
        self.rate_history: deque[tuple[float, dict[str, tuple[int | None, bool]]]] = (
            deque(maxlen=360)
        )

    def refresh(self, now_monotonic: float) -> bool:
        changed = self.seed_reader.refresh()
        changed = self.rejection_reader.refresh() or changed
        for reader in self.stage_readers.values():
            changed = reader.refresh() or changed
        changed = self.attempt_reader.refresh() or changed
        changed = self.status_reader.refresh() or changed
        rows = sum(self._stage(stage).rows for stage in STAGE_ORDER)
        if not self.history or changed or now_monotonic - self.history[-1][0] >= 10:
            self.history.append((now_monotonic, rows))
        counters = self._rate_counters()
        if (
            not self.rate_history
            or counters != self.rate_history[-1][1]
            or now_monotonic - self.rate_history[-1][0] >= 2
        ):
            self.rate_history.append((now_monotonic, counters))
        cutoff = now_monotonic - 600
        while len(self.history) > 2 and self.history[0][0] < cutoff:
            self.history.popleft()
        while len(self.rate_history) > 2 and self.rate_history[0][0] < cutoff:
            self.rate_history.popleft()
        return changed

    def _rate_counters(self) -> dict[str, tuple[int | None, bool]]:
        stages = [self._stage(name) for name in STAGE_ORDER]
        counters: dict[str, tuple[int | None, bool]] = {}
        for name in STAGE_ORDER:
            stage = self._stage(name)
            counters[f"stage.{name}"] = (
                stage.rows,
                not self.stage_readers[name].parse_errors,
            )
        counters["stage.total"] = (
            sum(stage.rows for stage in stages),
            all(not reader.parse_errors for reader in self.stage_readers.values()),
        )
        rejections = self._rejections()
        counters["seed_rejections"] = (
            rejections.rows,
            not self.rejection_reader.parse_errors,
        )
        for token_name in ("input", "output", "total"):
            counters[f"retained_token.{token_name}"] = (
                sum(stage.token_values[token_name] for stage in stages),
                sum(stage.token_unknown_rows[token_name] for stage in stages) == 0,
            )
        wire_unknown = sum(stage.wire_unknown_rows for stage in stages)
        counters["wire"] = (
            sum(stage.wire_attempts for stage in stages),
            wire_unknown == 0,
        )
        status = self.status_reader.metadata
        audit_requests = _safe_nonnegative_int(
            status.get(("audit_ledger", "requests_total"))
        )
        audit_output = _safe_nonnegative_int(
            status.get(("audit_ledger", "output_tokens_total"))
        )
        fresh = self._status_is_fresh()
        counters["requests"] = (
            audit_requests,
            fresh and audit_requests is not None,
        )
        counters["provider_output"] = (
            audit_output,
            fresh and audit_output is not None,
        )
        return counters

    def _rolling_rate(
        self, key: str, multiplier: float, *, source: str = "rolling_60s"
    ) -> dict[str, object]:
        if len(self.rate_history) < 2:
            return _metric(None, exact=False, source=source)
        latest_time, latest = self.rate_history[-1]
        latest_value, latest_exact = latest.get(key, (None, False))
        if latest_value is None:
            return _metric(None, exact=False, source=source)
        cutoff = latest_time - 60
        candidates = [
            point
            for point in self.rate_history
            if point[0] >= cutoff and point[0] < latest_time
        ]
        if not candidates:
            return _metric(None, exact=False, source=source)
        earlier_time, earlier = candidates[0]
        earlier_value, earlier_exact = earlier.get(key, (None, False))
        if earlier_value is None or latest_value < earlier_value:
            return _metric(None, exact=False, source=source)
        elapsed = latest_time - earlier_time
        if elapsed <= 0:
            return _metric(None, exact=False, source=source)
        value = (latest_value - earlier_value) * multiplier / elapsed
        return _metric(
            round(value, 6),
            exact=latest_exact and earlier_exact,
            source=source,
        )

    def _public_rates(self) -> dict[str, object]:
        stage_rates = {
            name: self._rolling_rate(f"stage.{name}", 60.0)
            for name in (*STAGE_ORDER, "total")
        }
        return {
            "requests_per_minute": self._rolling_rate(
                "requests", 60.0, source="rolling_audit_ledger_60s"
            ),
            "provider_output_tokens_per_second": self._rolling_rate(
                "provider_output", 1.0, source="rolling_audit_ledger_60s"
            ),
            "wire_attempts_per_minute": self._rolling_rate(
                "wire", 60.0, source="rolling_retained_rows_60s"
            ),
            "seed_rejections_per_minute": self._rolling_rate("seed_rejections", 60.0),
            "retained_tokens_per_second": {
                name: self._rolling_rate(
                    f"retained_token.{name}",
                    1.0,
                    source="rolling_retained_rows_60s",
                )
                for name in ("input", "output", "total")
            },
            "stage_rows_per_minute": stage_rates,
        }

    def _public_diagnostics(
        self,
        *,
        state: str,
        status_fresh: bool,
        parse_error_count: int,
        audit_requests: int | None,
        audit_output_tokens: int | None,
        latest_data_mtime: float | None,
    ) -> dict[str, object]:
        """Return finite, content-free reasons for unavailable telemetry.

        The reason list is deliberately derived only from local counters and a
        small set of persisted workload states. Provider messages and arbitrary
        status strings never enter this public structure.
        """

        reasons: list[str] = []
        if len(self.rate_history) < 2:
            reasons.append("telemetry_cold_start")
        elif self.rate_history[-1][0] - self.rate_history[0][0] < 60:
            reasons.append("telemetry_warming")
        if not status_fresh:
            reasons.append("status_stale")
        if parse_error_count or self.status_reader.invalid_sha256 is not None:
            reasons.append("file_parse_error")
        if state == "cooldown":
            reasons.extend(("provider_cooldown", "rate_limit"))
        elif state == "provider_quota_exhausted":
            reasons.append("quota")
        elif state == "client_deadline":
            reasons.append("client_deadline")
        if status_fresh and (audit_requests is None or audit_output_tokens is None):
            reasons.append("unknown_counter")
        reasons = list(dict.fromkeys(reasons))
        if not set(reasons) <= DIAGNOSTIC_REASON_CODES:
            raise RuntimeError("diagnostic reason schema drift")

        hard_reasons = set(reasons) - {
            "telemetry_cold_start",
            "telemetry_warming",
        }
        if state == "complete" and not hard_reasons:
            summary = "complete"
        elif state == "running" and not hard_reasons:
            summary = "normal_warming" if reasons else "running"
        elif state == "cooldown":
            summary = "waiting"
        else:
            summary = "attention"
        observed_at = _safe_observed_at(
            self.status_reader.metadata.get(("updated_at",))
        ) or _iso_from_timestamp(latest_data_mtime)
        cooldown_until = None
        if state == "cooldown":
            cooldown_until = _safe_observed_at(
                self.status_reader.metadata.get(("cooldown_until",))
            )
        return {
            "summary": summary,
            "reason_codes": reasons,
            "observed_at": observed_at,
            "cooldown_until": cooldown_until,
        }

    def _stage(self, name: str) -> StageAggregate:
        aggregate = self.stage_readers[name].aggregate
        if not isinstance(aggregate, StageAggregate):
            raise TypeError("stage reader aggregate mismatch")
        return aggregate

    def _seeds(self) -> SeedAggregate:
        aggregate = self.seed_reader.aggregate
        if not isinstance(aggregate, SeedAggregate):
            raise TypeError("seed reader aggregate mismatch")
        return aggregate

    def _attempts(self) -> AttemptAggregate:
        aggregate = self.attempt_reader.aggregate
        if not isinstance(aggregate, AttemptAggregate):
            raise TypeError("attempt reader aggregate mismatch")
        return aggregate

    def _rejections(self) -> SeedRejectionAggregate:
        aggregate = self.rejection_reader.aggregate
        if not isinstance(aggregate, SeedRejectionAggregate):
            raise TypeError("seed rejection reader aggregate mismatch")
        return aggregate

    def _provider_binding(self) -> dict[str, object]:
        stages = [self._stage(name) for name in STAGE_ORDER]
        row_count = sum(stage.rows for stage in stages)

        def exact_value(values: set[str], unknown_rows: int) -> tuple[str | None, bool]:
            exact = row_count > 0 and unknown_rows == 0 and len(values) == 1
            return (next(iter(values)) if exact else None, exact)

        model, model_exact = exact_value(
            set().union(*(stage.provider_models for stage in stages)),
            sum(stage.provider_model_unknown_rows for stage in stages),
        )
        protocol, protocol_exact = exact_value(
            set().union(*(stage.provider_protocols for stage in stages)),
            sum(stage.provider_protocol_unknown_rows for stage in stages),
        )
        base_url, base_url_exact = exact_value(
            set().union(*(stage.provider_base_urls for stage in stages)),
            sum(stage.provider_base_url_unknown_rows for stage in stages),
        )
        return {
            "request_model_id": model,
            "runtime_protocol": protocol,
            "base_url": base_url,
            "exact": model_exact and protocol_exact and base_url_exact,
        }

    def _throughput(self) -> float | None:
        if len(self.history) < 2:
            return None
        latest_time, latest_rows = self.history[-1]
        earlier = next(
            (
                point
                for point in self.history
                if point[1] != latest_rows and latest_time > point[0]
            ),
            None,
        )
        if earlier is None:
            return None
        seconds = latest_time - earlier[0]
        rows = latest_rows - earlier[1]
        if seconds <= 0 or rows <= 0:
            return None
        return rows / seconds

    def _latest_data_mtime(self) -> float | None:
        mtimes = [
            self.seed_reader.last_mtime,
            self.rejection_reader.last_mtime,
            self.attempt_reader.last_mtime,
        ]
        mtimes.extend(reader.last_mtime for reader in self.stage_readers.values())
        present = [value for value in mtimes if value is not None]
        return max(present) if present else None

    def _status_is_fresh(self) -> bool:
        status_time = self.status_reader.last_mtime
        data_time = self._latest_data_mtime()
        if status_time is None:
            return False
        checkpoint_seconds = self.status_reader.metadata.get(
            ("usage_checkpoint_policy", "maximum_seconds")
        )
        if isinstance(checkpoint_seconds, bool) or not isinstance(
            checkpoint_seconds, (int, float)
        ):
            checkpoint_seconds = 0
        grace = max(
            STATUS_FRESHNESS_GRACE_SECONDS,
            max(0.0, float(checkpoint_seconds)) * 3 + 5,
        )
        return data_time is None or status_time + grace >= data_time

    def public(self, catalog: CatalogService | None = None) -> dict[str, object]:
        seeds = self._seeds()
        stages = [self._stage(name) for name in STAGE_ORDER]
        stage_seed_sets = [stage.seed_ids for stage in stages]
        complete = len(set.intersection(*stage_seed_sets)) if stage_seed_sets else 0
        parse_errors = [
            error.public()
            for reader in [
                self.seed_reader,
                self.rejection_reader,
                *self.stage_readers.values(),
            ]
            for error in reader.parse_errors
        ]
        parse_errors.extend(
            error.public() for error in self.attempt_reader.parse_errors
        )
        complete_exact = not parse_errors and all(
            stage.unknown_seed_rows == 0 for stage in stages
        )
        retained_token_metrics = {
            name: _sum_token_metric(stages, name)
            for name in ("input", "output", "total", "cache_read", "cache_write")
        }
        wire_value = sum(stage.wire_attempts for stage in stages)
        wire_unknown = sum(stage.wire_unknown_rows for stage in stages)
        retry_value = sum(stage.retries for stage in stages)
        retry_unknown = sum(stage.retry_unknown_rows for stage in stages)
        latest_data_mtime = self._latest_data_mtime()
        status_fresh = self._status_is_fresh()
        status = self.status_reader.metadata
        current_epoch_requests = _safe_nonnegative_int(
            status.get(("quota_epoch", "requests_used"))
        )
        audit_requests = _safe_nonnegative_int(
            status.get(("audit_ledger", "requests_total"))
        )
        audit_output_tokens = _safe_nonnegative_int(
            status.get(("audit_ledger", "output_tokens_total"))
        )
        current_epoch_output_tokens = _safe_nonnegative_int(
            status.get(("quota_epoch", "output_tokens_used"))
        )
        request_budget = _safe_nonnegative_int(
            status.get(("quota_epoch", "max_requests"))
        )
        state_value = status.get(("state",)) if status_fresh else None
        if state_value is not None:
            state = _safe_enum(state_value, fallback_prefix="unknown-state")
        elif self.status_reader.last_mtime is not None:
            state = "status-stale"
        else:
            state = "untracked"
        row_count = sum(stage.rows for stage in stages)
        throughput = self._throughput()
        remaining = sum(max(seeds.rows - stage.rows, 0) for stage in stages)
        eta = remaining / throughput if throughput and remaining else None
        output_budgets = set().union(*(stage.output_token_budgets for stage in stages))
        if status_fresh:
            status_token_budget = _safe_nonnegative_int(
                status.get(("quota_epoch", "max_output_tokens_total"))
            )
            if status_token_budget:
                output_budgets = {status_token_budget}
        output_budget = _known_percent(
            int(current_epoch_output_tokens or 0), output_budgets
        )
        output_budget["exact"] = bool(
            output_budget["exact"]
            and status_fresh
            and current_epoch_output_tokens is not None
        )
        output_budget["source"] = "current_quota_epoch"
        if status_fresh and current_epoch_requests is not None and request_budget:
            request_percent = _metric(
                round(min(100.0, current_epoch_requests * 100.0 / request_budget), 4),
                exact=True,
                source="current_quota_epoch",
            )
        else:
            request_percent = _metric(None, exact=False, source="unknown")
        attempt_errors = dict(sorted(self._attempts().by_type.items()))
        rejections = self._rejections()
        status_error = None
        if self.status_reader.invalid_sha256:
            status_error = {
                "source": "automation_status",
                "sha256": self.status_reader.invalid_sha256,
            }
        binding = self._provider_binding()
        if catalog is None:
            pinned_cost: dict[str, object] = {
                "known": False,
                "exact": False,
                "reason": "catalog_unavailable",
                "currency": None,
                "total": None,
                "scope": "persisted_stage_provider_usage",
            }
        else:
            pinned_cost = catalog.pinned_cost(
                request_model_id=binding["request_model_id"],  # type: ignore[arg-type]
                runtime_protocol=binding["runtime_protocol"],  # type: ignore[arg-type]
                base_url=binding["base_url"],  # type: ignore[arg-type]
                token_metrics=retained_token_metrics,  # type: ignore[arg-type]
                binding_exact=bool(binding["exact"]),
            )
        missing_request_usage = max((audit_requests or 0) - row_count, 0)
        provider_tokens = {
            "input": _metric(
                int(retained_token_metrics["input"]["value"]),
                exact=False,
                unknown_rows=(
                    missing_request_usage
                    + int(retained_token_metrics["input"]["unknown_rows"])
                ),
                source="retained_stage_provider_usage_lower_bound",
            ),
            "output": _metric(
                audit_output_tokens,
                exact=status_fresh and audit_output_tokens is not None,
                source="audit_ledger_checkpoint",
            ),
            "total": _metric(
                int(retained_token_metrics["total"]["value"]),
                exact=False,
                unknown_rows=(
                    missing_request_usage
                    + int(retained_token_metrics["total"]["unknown_rows"])
                ),
                source="retained_stage_provider_usage_lower_bound",
            ),
        }
        diagnostics = self._public_diagnostics(
            state=state,
            status_fresh=status_fresh,
            parse_error_count=len(parse_errors),
            audit_requests=audit_requests,
            audit_output_tokens=audit_output_tokens,
            latest_data_mtime=latest_data_mtime,
        )
        return {
            "label": self.label,
            "state": state,
            "seeds": _metric(
                seeds.rows,
                exact=not self.seed_reader.parse_errors,
                unknown_rows=seeds.unknown_seed_rows,
                source="seeds_jsonl",
            ),
            "seed_rejections": {
                **_metric(
                    rejections.rows,
                    exact=not self.rejection_reader.parse_errors,
                    source="seed_rejections_jsonl",
                ),
                "by_reason": dict(sorted(rejections.by_reason.items())),
                "recent": list(reversed(rejections.recent)),
                "content_retained": False,
            },
            "complete_chains": _metric(
                complete,
                exact=complete_exact,
                unknown_rows=sum(stage.unknown_seed_rows for stage in stages),
                source="seed_id_intersection",
            ),
            "stages": {
                name: {
                    "rows": self._stage(name).rows,
                    "exact": not self.stage_readers[name].parse_errors,
                }
                for name in STAGE_ORDER
            },
            "completed_stage_records": row_count,
            "tokens": provider_tokens,
            "retained_stage_tokens": retained_token_metrics,
            "pinned_cost": pinned_cost,
            "requests": _metric(
                audit_requests,
                exact=status_fresh and audit_requests is not None,
                source="audit_ledger_checkpoint",
            ),
            "wire_attempts": _metric(
                wire_value,
                exact=wire_unknown == 0,
                unknown_rows=wire_unknown,
                source="provider_attempts",
            ),
            "retries": _metric(
                retry_value,
                exact=retry_unknown == 0,
                unknown_rows=retry_unknown,
                source="provider_attempts",
            ),
            "budget": {
                "request_percent": request_percent,
                "output_token_percent": output_budget,
            },
            "errors": {
                "by_type": attempt_errors,
                "invalid_json_lines": parse_errors,
                "status_error": status_error,
            },
            "updated_at": _iso_from_timestamp(latest_data_mtime),
            "throughput_rows_per_second": (
                round(throughput, 6) if throughput is not None else None
            ),
            "eta_seconds": round(eta, 2) if eta is not None else None,
            "rates": self._public_rates(),
            "diagnostics": diagnostics,
        }


class DashboardEngine:
    def __init__(
        self,
        shards: Sequence[tuple[str, Path]],
        *,
        catalog: CatalogService | None = None,
    ) -> None:
        self.monitors = [ShardMonitor(label, path) for label, path in shards]
        self.catalog = catalog
        self.lock = threading.Lock()
        self.logs: deque[dict[str, object]] = deque(maxlen=MAX_LOG_ENTRIES)
        self.previous: dict[str, dict[str, object]] = {}

    def attach_shard(self, label: str, path: Path) -> None:
        """Attach an existing/new shard as a read-only monitor.

        Attachment never grants process ownership. The control plane separately
        tracks only children it started itself.
        """

        with self.lock:
            for monitor in self.monitors:
                if monitor.label == label:
                    if monitor.path == path:
                        return
                    raise ControlError(
                        409, "duplicate_label", "Monitor label is already used"
                    )
                if monitor.path == path:
                    return
            self.monitors.append(ShardMonitor(label, path))

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            now_monotonic = time.monotonic()
            for monitor in self.monitors:
                monitor.refresh(now_monotonic)
            shards = [monitor.public(self.catalog) for monitor in self.monitors]
            self._record_changes(shards)
            return self._compose(shards)

    def _record_changes(self, shards: Sequence[dict[str, object]]) -> None:
        at = datetime.now(timezone.utc).isoformat()
        for shard in shards:
            label = str(shard["label"])
            stages = shard["stages"]
            if not isinstance(stages, Mapping):
                continue
            current_rows = {
                name: int(stages[name]["rows"])
                for name in STAGE_ORDER
                if isinstance(stages.get(name), Mapping)
            }
            state = str(shard["state"])
            rejection_metric = shard.get("seed_rejections", {})
            rejection_count = (
                int(rejection_metric.get("value", 0))
                if isinstance(rejection_metric, Mapping)
                else 0
            )
            invalid = shard.get("errors", {}).get("invalid_json_lines", [])
            invalid_count = len(invalid) if isinstance(invalid, list) else 0
            current: dict[str, object] = {
                "rows": current_rows,
                "state": state,
                "rejections": rejection_count,
                "invalid": invalid_count,
            }
            previous = self.previous.get(label)
            if previous is None:
                self._log(at, "info", "shard_observed", label, "metadata scan ready")
            else:
                old_rows = previous.get("rows", {})
                if isinstance(old_rows, Mapping):
                    for stage in STAGE_ORDER:
                        before = int(old_rows.get(stage, 0))
                        after = current_rows.get(stage, 0)
                        if after != before:
                            self._log(
                                at,
                                "info",
                                "stage_rows",
                                label,
                                f"{stage}: {before} -> {after}",
                            )
                if previous.get("state") != state:
                    self._log(
                        at,
                        "info",
                        "state",
                        label,
                        f"{previous.get('state')} -> {state}",
                    )
                if int(previous.get("rejections", 0)) != rejection_count:
                    self._log(
                        at,
                        "warning",
                        "seed_rejections",
                        label,
                        f"rejections: {rejection_count}",
                    )
                if int(previous.get("invalid", 0)) != invalid_count:
                    self._log(
                        at,
                        "warning",
                        "invalid_json",
                        label,
                        f"invalid lines: {invalid_count}",
                    )
            self.previous[label] = current

    def _log(
        self,
        at: str,
        level: str,
        event: str,
        shard: str,
        detail: str,
    ) -> None:
        self.logs.append(
            {
                "at": at,
                "level": level,
                "event": event,
                "shard": shard,
                "detail": detail,
            }
        )

    def _compose(self, shards: Sequence[dict[str, object]]) -> dict[str, object]:
        monitors = self.monitors
        unique_seeds = set().union(*(monitor._seeds().seed_ids for monitor in monitors))
        global_stage_sets = [
            set().union(*(monitor._stage(stage).seed_ids for monitor in monitors))
            for stage in STAGE_ORDER
        ]
        complete = len(set.intersection(*global_stage_sets)) if global_stage_sets else 0
        all_stages = [
            monitor._stage(stage) for monitor in monitors for stage in STAGE_ORDER
        ]
        retained_tokens = {
            name: _sum_token_metric(all_stages, name)
            for name in ("input", "output", "total", "cache_read", "cache_write")
        }
        complete_unknown = sum(stage.unknown_seed_rows for stage in all_stages)
        invalid_count = sum(
            len(reader.parse_errors)
            for monitor in monitors
            for reader in [
                monitor.seed_reader,
                monitor.rejection_reader,
                *monitor.stage_readers.values(),
            ]
        )
        rejection_count = sum(monitor._rejections().rows for monitor in monitors)
        rejection_exact = all(
            not monitor.rejection_reader.parse_errors for monitor in monitors
        )
        rejection_reasons: Counter[str] = Counter()
        for monitor in monitors:
            rejection_reasons.update(monitor._rejections().by_reason)
        request_metrics = [shard["requests"] for shard in shards]
        request_values = [
            metric.get("value")
            for metric in request_metrics
            if isinstance(metric, Mapping) and metric.get("value") is not None
        ]
        request_exact = bool(request_metrics) and all(
            isinstance(metric, Mapping) and metric.get("exact") is True
            for metric in request_metrics
        )
        provider_tokens = {
            name: _sum_public_metrics(
                [
                    shard.get("tokens", {}).get(name)
                    if isinstance(shard.get("tokens"), Mapping)
                    else None
                    for shard in shards
                ],
                source=(
                    "sum_audit_ledger_checkpoints"
                    if name == "output"
                    else "sum_retained_stage_provider_usage_lower_bounds"
                ),
            )
            for name in ("input", "output", "total")
        }
        wire_value = sum(stage.wire_attempts for stage in all_stages)
        wire_unknown = sum(stage.wire_unknown_rows for stage in all_stages)
        retry_value = sum(stage.retries for stage in all_stages)
        retry_unknown = sum(stage.retry_unknown_rows for stage in all_stages)
        rates = [
            float(value)
            for shard in shards
            if (value := shard.get("throughput_rows_per_second")) is not None
        ]
        throughput = sum(rates) if rates else None
        remaining = sum(
            max(monitor._seeds().rows - monitor._stage(stage).rows, 0)
            for monitor in monitors
            for stage in STAGE_ORDER
        )
        eta = remaining / throughput if throughput and remaining else None
        stage_rows = {
            stage: sum(monitor._stage(stage).rows for monitor in monitors)
            for stage in STAGE_ORDER
        }
        request_rate_metrics: list[object] = []
        wire_rate_metrics: list[object] = []
        rejection_rate_metrics: list[object] = []
        provider_output_rate_metrics: list[object] = []
        token_rate_metrics: dict[str, list[object]] = {
            "input": [],
            "output": [],
            "total": [],
        }
        stage_rate_metrics: dict[str, list[object]] = {
            name: [] for name in (*STAGE_ORDER, "total")
        }
        for shard in shards:
            rates = shard.get("rates")
            if not isinstance(rates, Mapping):
                continue
            request_rate_metrics.append(rates.get("requests_per_minute"))
            wire_rate_metrics.append(rates.get("wire_attempts_per_minute"))
            rejection_rate_metrics.append(rates.get("seed_rejections_per_minute"))
            provider_output_rate_metrics.append(
                rates.get("provider_output_tokens_per_second")
            )
            token_rates = rates.get("retained_tokens_per_second")
            if isinstance(token_rates, Mapping):
                for name in token_rate_metrics:
                    token_rate_metrics[name].append(token_rates.get(name))
            row_rates = rates.get("stage_rows_per_minute")
            if isinstance(row_rates, Mapping):
                for name in stage_rate_metrics:
                    stage_rate_metrics[name].append(row_rates.get(name))
        public_rates = {
            "requests_per_minute": _sum_public_metrics(
                request_rate_metrics, source="sum_shard_rolling_audit_counters_60s"
            ),
            "provider_output_tokens_per_second": _sum_public_metrics(
                provider_output_rate_metrics,
                source="sum_shard_rolling_audit_counters_60s",
            ),
            "wire_attempts_per_minute": _sum_public_metrics(
                wire_rate_metrics, source="sum_shard_rolling_retained_rows_60s"
            ),
            "seed_rejections_per_minute": _sum_public_metrics(
                rejection_rate_metrics, source="sum_shard_rolling_60s"
            ),
            "retained_tokens_per_second": {
                name: _sum_public_metrics(
                    values, source="sum_shard_rolling_retained_rows_60s"
                )
                for name, values in token_rate_metrics.items()
            },
            "stage_rows_per_minute": {
                name: _sum_public_metrics(values, source="sum_shard_rolling_60s")
                for name, values in stage_rate_metrics.items()
            },
        }
        if self.catalog is None:
            pinned_cost: dict[str, object] = {
                "known": False,
                "exact": False,
                "reason": "catalog_unavailable",
                "currency": None,
                "total": None,
                "scope": "sum_persisted_stage_provider_usage",
            }
        else:
            pinned_cost = self.catalog.combined_cost(
                [shard.get("pinned_cost") for shard in shards]
            )
        generated_at = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "privacy": {
                "content_free": True,
                "absolute_paths_returned": False,
                "provider_usage_only": True,
            },
            "totals": {
                "seeds": _metric(
                    len(unique_seeds),
                    exact=all(
                        not monitor.seed_reader.parse_errors
                        and monitor._seeds().unknown_seed_rows == 0
                        for monitor in monitors
                    ),
                    source="unique_seed_ids",
                ),
                "seed_rejections": {
                    **_metric(
                        rejection_count,
                        exact=rejection_exact,
                        source="seed_rejections_jsonl",
                    ),
                    "by_reason": dict(sorted(rejection_reasons.items())),
                    "content_retained": False,
                },
                "stage_rows": stage_rows,
                "complete_chains": _metric(
                    complete,
                    exact=complete_unknown == 0 and invalid_count == 0,
                    unknown_rows=complete_unknown,
                    source="global_seed_id_intersection",
                ),
                "tokens": provider_tokens,
                "retained_stage_tokens": retained_tokens,
                "pinned_cost": pinned_cost,
                "requests": _metric(
                    (
                        sum(int(value) for value in request_values)
                        if request_values
                        else None
                    ),
                    exact=request_exact,
                    unknown_rows=sum(
                        1
                        for metric in request_metrics
                        if not isinstance(metric, Mapping)
                        or metric.get("value") is None
                    ),
                    source="sum_audit_ledger_checkpoints",
                ),
                "wire_attempts": _metric(
                    wire_value,
                    exact=wire_unknown == 0,
                    unknown_rows=wire_unknown,
                    source="retained_stage_provider_attempts",
                ),
                "retries": _metric(
                    retry_value,
                    exact=retry_unknown == 0,
                    unknown_rows=retry_unknown,
                    source="retained_stage_provider_attempts",
                ),
                "throughput_rows_per_second": (
                    round(throughput, 6) if throughput is not None else None
                ),
                "eta_seconds": round(eta, 2) if eta is not None else None,
                "rates": public_rates,
            },
            "shards": list(shards),
            "logs": list(self.logs),
            "diagnostics": _public_connection_diagnostics(
                shards, None, observed_at=generated_at
            ),
        }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        engine: DashboardEngine,
        index_html: bytes,
        controller: ControlPlane | None = None,
    ) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("control dashboard binds only to 127.0.0.1")
        self.engine = engine
        self.catalog = engine.catalog
        self.index_html = index_html
        self.controller = controller
        self.session_token = csrf_token()
        self.session_cookie = csrf_cookie_value(self.session_token)
        self._closed_once = False
        super().__init__(address, DashboardHandler)
        port = int(self.server_address[1])
        self.expected_host = f"127.0.0.1:{port}"
        self.expected_origin = f"http://127.0.0.1:{port}"

    def server_close(self) -> None:
        if not self._closed_once:
            self._closed_once = True
            if self.controller is not None:
                self.controller.close()
            for index in range(len(self.session_token)):
                self.session_token[index] = 0
            self.session_cookie = ""
        super().server_close()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._host_allowed() or self._unsafe_request_target():
            return
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            self._error(HTTPStatus.BAD_REQUEST, "query_not_allowed")
            return
        path = parsed.path
        if path == "/api/snapshot":
            snapshot = self.server.engine.snapshot()
            control = (
                self.server.controller.public()
                if self.server.controller is not None
                else {"enabled": False, "process_state": "monitor_only"}
            )
            snapshot["control"] = control
            snapshot["diagnostics"] = _public_connection_diagnostics(
                snapshot["shards"],  # type: ignore[arg-type]
                control,
                observed_at=str(snapshot["generated_at"]),
            )
            payload = json.dumps(
                snapshot,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", payload)
            return
        if path == "/api/control/options":
            if self.server.controller is None:
                self._error(HTTPStatus.NOT_FOUND, "control_disabled")
                return
            self._json(HTTPStatus.OK, self.server.controller.options())
            return
        if path == "/api/control/formal-status":
            if self.server.controller is None:
                self._error(HTTPStatus.NOT_FOUND, "control_disabled")
                return
            self._json(HTTPStatus.OK, self.server.controller.formal_status())
            return
        if path == "/api/catalog":
            if self.server.catalog is None:
                self._error(HTTPStatus.NOT_FOUND, "catalog_disabled")
                return
            self._json(HTTPStatus.OK, self.server.catalog.public())
            return
        if path == "/api/catalog/status":
            if self.server.catalog is None:
                self._error(HTTPStatus.NOT_FOUND, "catalog_disabled")
                return
            self._json(HTTPStatus.OK, self.server.catalog.status(refresh=True))
            return
        if path in {"/", "/index.html"}:
            self._send(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                self.server.index_html,
                extra_headers={
                    "Set-Cookie": (
                        f"AnchorSession={self.server.session_cookie}; Path=/; "
                        "HttpOnly; SameSite=Strict"
                    )
                },
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.close_connection = True
        if not self._host_allowed() or self._unsafe_request_target():
            return
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            self._error(HTTPStatus.BAD_REQUEST, "query_not_allowed")
            return
        if self.server.controller is None:
            self._error(HTTPStatus.NOT_FOUND, "control_disabled")
            return
        if not self._mutation_authorized():
            return
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/control/formal-start":
                result = self.server.controller.start_formal(payload, resume=False)
            elif parsed.path == "/api/control/formal-continue":
                result = self.server.controller.start_formal(payload, resume=True)
            elif parsed.path == "/api/control/formal-stop":
                if set(payload) != {"run_id"}:
                    raise ControlError(
                        400,
                        "invalid_formal_stop",
                        "Formal stop requires only the active run ID",
                    )
                result = self.server.controller.stop_formal(payload.get("run_id"))
            elif parsed.path == "/api/control/start":
                result = self.server.controller.start_new(payload)
            elif parsed.path == "/api/control/continue":
                result = self.server.controller.continue_run(payload)
            elif parsed.path == "/api/control/stop":
                if set(payload) != {"run_id"}:
                    raise ControlError(
                        400, "invalid_stop", "Stop requires only the active run ID"
                    )
                result = self.server.controller.stop(payload.get("run_id"))
            elif parsed.path == "/api/control/clear-key":
                if payload:
                    raise ControlError(
                        400, "invalid_clear", "Clear credential accepts no fields"
                    )
                result = self.server.controller.clear_credential()
            elif parsed.path == "/api/control/models":
                result = self.server.controller.probe_models(payload)
            elif parsed.path == "/api/control/attach":
                result = self.server.controller.attach_monitor(payload)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found")
                return
        except ControlError as error:
            self._json(error.status, {"error": error.code, "message": error.message})
            return
        except (OSError, ValueError, RuntimeError):
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")
            return
        self._json(HTTPStatus.OK, result)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    def _host_allowed(self) -> bool:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or not hmac.compare_digest(
            hosts[0], self.server.expected_host
        ):
            self._error(HTTPStatus.FORBIDDEN, "host_rejected")
            return False
        return True

    def _unsafe_request_target(self) -> bool:
        if self.path.startswith(("http://", "https://")):
            self._error(HTTPStatus.BAD_REQUEST, "absolute_target_rejected")
            return True
        return False

    def _mutation_authorized(self) -> bool:
        origins = self.headers.get_all("Origin") or []
        if len(origins) != 1 or not hmac.compare_digest(
            origins[0], self.server.expected_origin
        ):
            self._error(HTTPStatus.FORBIDDEN, "origin_rejected")
            return False
        csrf_headers = self.headers.get_all("X-Anchor-CSRF") or []
        if len(csrf_headers) != 1 or csrf_headers[0] != "1":
            self._error(HTTPStatus.FORBIDDEN, "csrf_rejected")
            return False
        cookie_headers = self.headers.get_all("Cookie") or []
        if len(cookie_headers) != 1:
            self._error(HTTPStatus.FORBIDDEN, "csrf_rejected")
            return False
        session_parts = [
            part
            for part in cookie_headers[0].split(";")
            if part.partition("=")[0].strip() == "AnchorSession"
        ]
        if len(session_parts) != 1:
            self._error(HTTPStatus.FORBIDDEN, "csrf_rejected")
            return False
        cookie = SimpleCookie()
        try:
            cookie.load(cookie_headers[0])
        except Exception:
            self._error(HTTPStatus.FORBIDDEN, "csrf_rejected")
            return False
        morsel = cookie.get("AnchorSession")
        if morsel is None or not hmac.compare_digest(
            morsel.value, self.server.session_cookie
        ):
            self._error(HTTPStatus.FORBIDDEN, "csrf_rejected")
            return False
        return True

    def _read_json_body(self) -> dict[str, object]:
        if self.headers.get_all("Transfer-Encoding"):
            raise ControlError(
                400, "transfer_encoding_rejected", "Chunked requests are not accepted"
            )
        if self.headers.get_all("Expect"):
            raise ControlError(
                400, "expect_rejected", "Expect headers are not accepted"
            )
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise ControlError(
                411, "content_length_required", "One Content-Length is required"
            )
        length = int(lengths[0])
        if length < 2:
            raise ControlError(400, "invalid_json", "JSON object is required")
        if length > MAX_POST_BODY_BYTES:
            raise ControlError(413, "request_too_large", "Control request is too large")
        content_types = self.headers.get_all("Content-Type") or []
        if len(content_types) != 1:
            raise ControlError(
                415, "json_required", "Content-Type must be application/json"
            )
        normalized = ";".join(
            part.strip().casefold() for part in content_types[0].split(";")
        )
        if normalized not in {"application/json", "application/json;charset=utf-8"}:
            raise ControlError(
                415, "json_required", "Content-Type must be application/json"
            )
        self.connection.settimeout(5)
        body = self.rfile.read(length)
        if len(body) != length:
            raise ControlError(400, "invalid_json", "JSON object is required")
        try:
            text_body = body.decode("utf-8")
            if text_body.startswith("\ufeff"):
                raise ValueError("BOM rejected")
            value = json.loads(
                text_body,
                object_pairs_hook=_strict_json_pairs,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ControlError(
                400, "invalid_json", "JSON object is required"
            ) from error
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise ControlError(400, "invalid_json", "JSON object is required")
        return value

    def _json(self, status: int | HTTPStatus, value: Mapping[str, object]) -> None:
        body = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _error(self, status: int | HTTPStatus, code: str) -> None:
        self._json(status, {"error": code})

    def _send(
        self,
        status: int | HTTPStatus,
        content_type: str,
        body: bytes,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        if extra_headers is not None:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        # Request targets can contain operator-provided query text. Keep the
        # observability process content-free by suppressing the default log.
        return


def _strict_json_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> object:
    raise ValueError("non-finite value")


def parse_shards(values: Sequence[str]) -> list[tuple[str, Path]]:
    shards: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for index, raw in enumerate(values, start=1):
        if "=" in raw:
            label, path_text = raw.split("=", 1)
        else:
            path_text = raw
            candidate = Path(path_text).name
            label = candidate if LABEL_RE.fullmatch(candidate) else f"shard-{index:02d}"
        if not LABEL_RE.fullmatch(label):
            raise ValueError(f"shard {index} has an invalid label")
        if label in labels:
            raise ValueError(f"duplicate shard label: {label}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"shard {index} is not a directory")
        labels.add(label)
        shards.append((label, path))
    return shards


def _terminal_summary(snapshot: Mapping[str, object]) -> str:
    totals = snapshot.get("totals", {})
    if not isinstance(totals, Mapping):
        return "snapshot unavailable"
    complete = totals.get("complete_chains", {})
    complete_value = complete.get("value") if isinstance(complete, Mapping) else None
    tokens = totals.get("tokens", {})
    output = tokens.get("output", {}) if isinstance(tokens, Mapping) else {}
    output_value = output.get("value") if isinstance(output, Mapping) else None
    output_exact = output.get("exact") if isinstance(output, Mapping) else False
    suffix = "" if output_exact else "+unknown"
    return (
        f"complete_chains={complete_value} "
        f"provider_output_tokens={output_value}{suffix} "
        f"shards={len(snapshot.get('shards', []))}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local distillation telemetry and strict subprocess control"
    )
    parser.add_argument(
        "--shard",
        action="append",
        metavar="[LABEL=]DIRECTORY",
        help="repeat for each shard; labels are the only names returned by the API",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--ccswitch-state-dir",
        type=Path,
        default=None,
        help="optional validated CC Switch metadata adapter state directory",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="disable process-control endpoints and serve telemetry only",
    )
    parser.add_argument("--once", action="store_true", help="print one snapshot")
    parser.add_argument(
        "--json",
        action="store_true",
        help="with --once, print the complete content-free JSON snapshot",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json and not args.once:
        parser.error("--json requires --once")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.host != "127.0.0.1":
        parser.error("dashboard control plane binds only to 127.0.0.1")
    try:
        shards = parse_shards(args.shard or [])
    except ValueError as error:
        parser.error(str(error))
    catalog = CatalogService(state_dir=args.ccswitch_state_dir)
    engine = DashboardEngine(shards, catalog=catalog)
    if args.once:
        snapshot = engine.snapshot()
        if args.json:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        else:
            print(_terminal_summary(snapshot))
        return 0
    asset = Path(__file__).with_name("dashboard_assets") / "index.html"
    if not asset.is_file():
        parser.error("dashboard HTML asset is missing")
    controller = None
    if not args.monitor_only:
        workspace_root = Path(__file__).resolve().parents[2]
        try:
            controller = ControlPlane(
                workspace_root,
                attach_callback=engine.attach_shard,
            )
        except ValueError as error:
            parser.error(str(error))
    server = DashboardServer(
        (args.host, args.port), engine, asset.read_bytes(), controller
    )
    host, port = server.server_address[:2]
    print(f"distillation dashboard: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
