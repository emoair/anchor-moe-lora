"""Fail-closed local control plane for distillation subprocesses.

This module is intentionally separate from the training package. It generates
immutable, secret-free automation configs and starts only one fixed Python
module with ``shell=False``. The HTTP layer lives in
``distillation_dashboard.py``; this file owns validation, process lifecycle,
cross-process shard locks, and RAM-only credentials.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, getproxies
from uuid import uuid4

import yaml


CONTROL_SCHEMA = "anchor.distillation-control.v1"
MANIFEST_SCHEMA = "anchor.distillation-control-manifest.v1"
CONTROL_KEY_ENV = "ANCHOR_CONTROL_API_KEY"
LOCK_NAME = ".anchor-control.lock"
MAX_MODEL_RESPONSE_BYTES = 2 * 1024 * 1024
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SECRET_ENV_RE = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE
)
SECRET_CONFIG_KEYS = frozenset(
    {"api_key", "apikey", "secret", "token", "authorization", "password"}
)
ALLOWED_PROTOCOLS = frozenset({"openai", "openai_responses", "anthropic"})
NETWORK_ROUTE_MODES = frozenset({"direct", "inherit"})
PROVIDER_BY_PROTOCOL = {
    "openai": "custom-openai",
    "openai_responses": "custom-openai-responses",
    "anthropic": "custom-anthropic",
}
START_FIELDS = frozenset(
    {
        "base_config",
        "output_dir",
        "seed_index_offset",
        "concurrency",
        "base_url",
        "protocol",
        "api_key",
        "model",
        "force_model",
        "task_card_config",
        "timeout_seconds",
        "max_retries",
        "reconnect_attempts",
        "reconnect_backoff_seconds",
        "cooldown_seconds",
        "cooldown_poll_seconds",
        "wall_clock_deadline_seconds",
        "max_requests",
        "max_output_tokens_total",
        "discovery_timeout_seconds",
        "wait_cooldown",
        "network_route",
    }
)
CONTINUE_FIELDS = frozenset({"run_id", "api_key"})
PROBE_FIELDS = frozenset(
    {"base_url", "protocol", "api_key", "model", "force_model", "timeout_seconds"}
)
ATTACH_FIELDS = frozenset({"output_dir", "label"})
CONTROL_PUBLIC_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "process_state",
        "workload_state",
        "run_id",
        "generation",
        "exit_code",
        "output_label",
        "base_config",
        "started_at",
        "finished_at",
        "launch_config_sha256",
        "corpus_binding_sha256",
        "credential_loaded",
        "can_start",
        "can_stop",
        "can_continue",
        "last_error_code",
        "reconnect",
        "events",
    }
)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControlError(RuntimeError):
    """A public, content-free control-plane failure."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class StrictLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _strict_mapping(
    loader: StrictLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError("duplicate YAML mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping
)


def load_strict_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.casefold() == ".json":
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        else:
            value = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError("base config is not valid strict YAML/JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("base config root must be a string-keyed mapping")
    if _contains_inline_secret(value):
        raise ValueError("base config contains an inline credential field")
    return dict(value)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON mapping key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON numbers are not allowed")


def _contains_inline_secret(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in SECRET_CONFIG_KEYS:
                return True
            if _contains_inline_secret(child):
                return True
    elif isinstance(value, list):
        return any(_contains_inline_secret(child) for child in value)
    return False


def validate_base_url(value: object) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ControlError(400, "invalid_base_url", "Base URL must be one HTTP(S) URL")
    if any(character.isspace() or ord(character) < 0x20 for character in value):
        raise ControlError(400, "invalid_base_url", "Base URL must be one HTTP(S) URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ControlError(400, "invalid_base_url", "Base URL must use http or https")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ControlError(400, "invalid_base_url", "Base URL host is invalid")
    if parsed.query or parsed.fragment:
        raise ControlError(
            400, "invalid_base_url", "Base URL cannot contain query or fragment"
        )
    if parsed.path.casefold().endswith(
        ("/chat/completions", "/messages", "/models", "/responses")
    ):
        raise ControlError(
            400, "invalid_base_url", "Use a base URL, not an API endpoint"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise ControlError(
            400, "invalid_base_url", "Base URL port is invalid"
        ) from error
    if port is not None and not 1 <= port <= 65535:
        raise ControlError(400, "invalid_base_url", "Base URL port is invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "", "", ""))


def validate_api_key(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 8192:
        raise ControlError(400, "invalid_credential", "Credential is required")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ControlError(400, "invalid_credential", "Credential format is invalid")
    return value


def _text(payload: Mapping[str, object], name: str, pattern: re.Pattern[str]) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ControlError(400, f"invalid_{name}", f"{name} is invalid")
    return value


def _integer(
    payload: Mapping[str, object], name: str, minimum: int, maximum: int
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlError(400, f"invalid_{name}", f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ControlError(
            400, f"invalid_{name}", f"{name} is outside the allowed range"
        )
    return value


def _number(
    payload: Mapping[str, object], name: str, minimum: float, maximum: float
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError(400, f"invalid_{name}", f"{name} must be numeric")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ControlError(
            400, f"invalid_{name}", f"{name} is outside the allowed range"
        )
    return numeric


def _boolean(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ControlError(400, f"invalid_{name}", f"{name} must be boolean")
    return value


def _network_route(value: object) -> str:
    if value not in NETWORK_ROUTE_MODES:
        raise ControlError(
            400, "invalid_network_route", "Network route must be direct or inherit"
        )
    return str(value)


def _proxy_detected() -> bool:
    try:
        return bool(getproxies())
    except (OSError, ValueError):
        return False


def _require_fields(payload: Mapping[str, object], allowed: frozenset[str]) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ControlError(400, "unknown_fields", "Request contains unsupported fields")


@dataclass(frozen=True)
class StartSpec:
    base_config: str
    output_dir: str
    seed_index_offset: int
    concurrency: int
    base_url: str
    protocol: str
    model: str
    force_model: bool
    task_card_config: str
    timeout_seconds: float
    max_retries: int
    reconnect_attempts: int
    reconnect_backoff_seconds: float
    cooldown_seconds: int
    cooldown_poll_seconds: int
    wall_clock_deadline_seconds: float
    max_requests: int
    max_output_tokens_total: int
    discovery_timeout_seconds: float
    wait_cooldown: bool
    network_route: str

    @property
    def provider(self) -> str:
        return PROVIDER_BY_PROTOCOL[self.protocol]


@dataclass(frozen=True)
class GeneratedRun:
    run_id: str
    run_dir: Path
    effective_config: Path
    manifest_path: Path
    output_dir: Path
    output_relative: str
    output_label: str
    base_config: str
    launch_config_sha256: str
    reconnect_attempts: int
    reconnect_backoff_seconds: float
    wait_cooldown: bool
    network_route: str


class SecretSlot:
    """Best-effort zeroizable RAM credential slot."""

    def __init__(self) -> None:
        self._value: bytearray | None = None

    @property
    def configured(self) -> bool:
        return self._value is not None

    def set(self, value: str) -> None:
        self.clear()
        self._value = bytearray(value.encode("ascii"))

    def reveal(self) -> str:
        if self._value is None:
            raise ControlError(
                409, "credential_missing", "Credential must be supplied again"
            )
        return self._value.decode("ascii")

    def clear(self) -> None:
        if self._value is not None:
            for index in range(len(self._value)):
                self._value[index] = 0
        self._value = None


class WorkspacePolicy:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root.expanduser().resolve()
        self.config_root = (self.root / "configs" / "data").resolve()
        self.data_root = (self.root / "data").resolve()
        self.runs_root = (self.root / "runs" / "control-plane").resolve()
        if not self.config_root.is_dir() or not self.data_root.is_dir():
            raise ValueError("workspace is missing configs/data or data")

    def options(self) -> dict[str, object]:
        configs: list[dict[str, object]] = []
        for path in sorted(self.config_root.glob("*")):
            if path.suffix.casefold() not in {".yaml", ".yml", ".json"}:
                continue
            relative = path.relative_to(self.root).as_posix()
            try:
                mapping = load_strict_mapping(path)
            except ValueError:
                continue
            if _is_automation_base_mapping(mapping):
                configs.append({"id": relative, "valid": True})
        task_cards = [
            path.relative_to(self.root).as_posix()
            for path in sorted(self.config_root.glob("task_cards*.yaml"))
            if path.is_file() and not path.is_symlink()
        ]
        runs: list[dict[str, object]] = []
        if self.runs_root.is_dir():
            for manifest in sorted(self.runs_root.glob("*/control-manifest.json")):
                try:
                    value = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if (
                    not isinstance(value, dict)
                    or value.get("schema_version") != MANIFEST_SCHEMA
                ):
                    continue
                run_id = manifest.parent.name
                output_label = value.get("output_label")
                if RUN_ID_RE.fullmatch(run_id) and isinstance(output_label, str):
                    runs.append({"run_id": run_id, "output_label": output_label})
        return {
            "base_configs": configs,
            "task_card_configs": task_cards,
            "protocols": sorted(ALLOWED_PROTOCOLS),
            "network_route_modes": ["direct", "inherit"],
            "proxy_detected": _proxy_detected(),
            "runs": runs[-50:],
            "limits": {
                "concurrency_max": 64,
                "request_body_bytes": 16_384,
            },
        }

    def config_path(self, relative: object) -> tuple[str, Path]:
        if not isinstance(relative, str) or not relative:
            raise ControlError(
                400, "invalid_base_config", "Choose a registered base config"
            )
        candidate = self._relative_file(relative, self.config_root)
        if candidate.suffix.casefold() not in {".yaml", ".yml", ".json"}:
            raise ControlError(
                400, "invalid_base_config", "Choose a registered base config"
            )
        return candidate.relative_to(self.root).as_posix(), candidate

    def task_card_path(self, relative: object) -> tuple[str, Path]:
        if not isinstance(relative, str) or not relative:
            raise ControlError(
                400, "invalid_task_card_config", "Choose a task-card config"
            )
        candidate = self._relative_file(relative, self.config_root)
        if candidate.suffix.casefold() not in {".yaml", ".yml", ".json"}:
            raise ControlError(
                400, "invalid_task_card_config", "Choose a task-card config"
            )
        return candidate.relative_to(self.root).as_posix(), candidate

    def _relative_file(self, relative: str, allowed_root: Path) -> Path:
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts or not raw.parts:
            raise ControlError(
                400, "invalid_path", "Path must use a registered workspace file"
            )
        lexical = self.root / raw
        if self._has_linklike_component(lexical, allowed_root):
            raise ControlError(
                400, "invalid_path", "Symlinked control inputs are not allowed"
            )
        candidate = lexical.resolve()
        if not candidate.is_relative_to(allowed_root) or not candidate.is_file():
            raise ControlError(
                400, "invalid_path", "Path must use a registered workspace file"
            )
        return candidate

    def output_path(self, relative: object, *, must_exist: bool) -> tuple[str, Path]:
        if not isinstance(relative, str) or not relative:
            raise ControlError(400, "invalid_output_dir", "Output directory is invalid")
        raw = Path(relative)
        if raw.is_absolute() or len(raw.parts) < 2 or raw.parts[0] != "data":
            raise ControlError(
                400, "invalid_output_dir", "Output must be a data subdirectory"
            )
        if any(not PATH_SEGMENT_RE.fullmatch(part) for part in raw.parts[1:]):
            raise ControlError(400, "invalid_output_dir", "Output directory is invalid")
        if ".." in raw.parts:
            raise ControlError(400, "invalid_output_dir", "Output directory is invalid")
        lexical = self.root / raw
        if self._has_linklike_component(lexical, self.data_root):
            raise ControlError(
                400, "invalid_output_dir", "Symlinked output paths are not allowed"
            )
        candidate = lexical.resolve(strict=False)
        if not candidate.is_relative_to(self.data_root) or candidate == self.data_root:
            raise ControlError(
                400, "invalid_output_dir", "Output must be a data subdirectory"
            )
        if must_exist and not candidate.is_dir():
            raise ControlError(
                404, "output_not_found", "Monitored output directory does not exist"
            )
        return raw.as_posix(), candidate

    @staticmethod
    def _has_linklike_component(candidate: Path, allowed_root: Path) -> bool:
        try:
            relative = candidate.relative_to(allowed_root)
        except ValueError:
            return True
        current = allowed_root
        for part in relative.parts:
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                return True
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            if current.is_symlink() or (
                attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
            ):
                return True
        return False

    def run_path(self, run_id: object) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise ControlError(400, "invalid_run_id", "Run ID is invalid")
        lexical = self.runs_root / run_id
        if self._has_linklike_component(lexical, self.runs_root):
            raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
        candidate = lexical.resolve()
        if not candidate.is_relative_to(self.runs_root) or not candidate.is_dir():
            raise ControlError(404, "run_not_found", "Control run was not found")
        return candidate

    def reserved_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for path in sorted(self.config_root.glob("*")):
            if path.suffix.casefold() not in {".yaml", ".yml", ".json"}:
                continue
            try:
                mapping = load_strict_mapping(path)
                ranges.append(_range_from_mapping(mapping))
            except (ValueError, TypeError):
                continue
        if self.runs_root.is_dir():
            for manifest in self.runs_root.glob("*/control-manifest.json"):
                try:
                    value = json.loads(manifest.read_text(encoding="utf-8"))
                    start = int(value["seed_index_offset"])
                    count = int(value["raw_collection_target"])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                ranges.append((start, start + max(1, count)))
        return ranges


def _range_from_mapping(mapping: Mapping[str, object]) -> tuple[int, int]:
    start = int(mapping.get("seed_index_offset", 0))
    raw_target = mapping.get("raw_collection_target")
    if raw_target is None:
        counts = mapping.get("stage_seed_counts", [1])
        if not isinstance(counts, list) or not counts:
            count = 1
        else:
            count = max(int(item) for item in counts)
    else:
        count = int(raw_target)
    if start < 0 or count < 1:
        raise ValueError("invalid seed range")
    return start, start + count


def _is_automation_base_mapping(mapping: Mapping[str, object]) -> bool:
    required = {
        "sop_dir",
        "output_dir",
        "concurrency_stages",
        "stage_seed_counts",
        "max_requests",
        "max_output_tokens_total",
    }
    if not required.issubset(mapping):
        return False
    if not all(
        isinstance(mapping.get(name), str) and bool(str(mapping[name]).strip())
        for name in ("sop_dir", "output_dir")
    ):
        return False
    concurrency = mapping.get("concurrency_stages")
    targets = mapping.get("stage_seed_counts")
    if not isinstance(concurrency, list) or not isinstance(targets, list):
        return False
    if not concurrency or len(concurrency) != len(targets):
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in [*concurrency, *targets]
    ):
        return False
    if list(sorted(set(targets))) != targets:
        return False
    for name in ("max_requests", "max_output_tokens_total"):
        value = mapping.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return False
    raw_target = mapping.get("raw_collection_target")
    if raw_target is not None and (
        isinstance(raw_target, bool)
        or not isinstance(raw_target, int)
        or raw_target < targets[-1]
    ):
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        if item.is_symlink():
            raise ValueError("symlinked SOP input is not allowed")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def parse_start_spec(
    payload: Mapping[str, object], policy: WorkspacePolicy
) -> tuple[StartSpec, str, dict[str, Any]]:
    _require_fields(payload, START_FIELDS)
    base_relative, base_path = policy.config_path(payload.get("base_config"))
    base = load_strict_mapping(base_path)
    if not _is_automation_base_mapping(base):
        raise ControlError(
            400,
            "invalid_base_config",
            "Base config is not an automation configuration",
        )
    output_relative, _ = policy.output_path(payload.get("output_dir"), must_exist=False)
    task_relative, _ = policy.task_card_path(payload.get("task_card_config"))
    protocol = payload.get("protocol")
    if protocol not in ALLOWED_PROTOCOLS:
        raise ControlError(400, "invalid_protocol", "Protocol is invalid")
    model = _text(payload, "model", MODEL_RE)
    spec = StartSpec(
        base_config=base_relative,
        output_dir=output_relative,
        seed_index_offset=_integer(payload, "seed_index_offset", 0, 2_000_000_000),
        concurrency=_integer(payload, "concurrency", 1, 64),
        base_url=validate_base_url(payload.get("base_url")),
        protocol=str(protocol),
        model=model,
        force_model=_boolean(payload, "force_model"),
        task_card_config=task_relative,
        timeout_seconds=_number(payload, "timeout_seconds", 1, 3600),
        max_retries=_integer(payload, "max_retries", 0, 10),
        reconnect_attempts=_integer(payload, "reconnect_attempts", 0, 20),
        reconnect_backoff_seconds=_number(
            payload, "reconnect_backoff_seconds", 0.1, 3600
        ),
        cooldown_seconds=_integer(payload, "cooldown_seconds", 1, 172_800),
        cooldown_poll_seconds=_integer(payload, "cooldown_poll_seconds", 1, 3600),
        wall_clock_deadline_seconds=_number(
            payload, "wall_clock_deadline_seconds", 1, 86_400
        ),
        max_requests=_integer(payload, "max_requests", 1, 10_000_000),
        max_output_tokens_total=_integer(
            payload, "max_output_tokens_total", 1, 10_000_000_000
        ),
        discovery_timeout_seconds=_number(payload, "discovery_timeout_seconds", 1, 120),
        wait_cooldown=_boolean(payload, "wait_cooldown"),
        network_route=_network_route(payload.get("network_route")),
    )
    credential = validate_api_key(payload.get("api_key"))
    return spec, credential, base


def generate_run(
    policy: WorkspacePolicy, spec: StartSpec, base: Mapping[str, Any]
) -> GeneratedRun:
    output_relative, output_dir = policy.output_path(spec.output_dir, must_exist=False)
    if output_dir.exists():
        raise ControlError(
            409, "new_shard_required", "New runs require a unique empty shard"
        )
    raw_target = _range_from_mapping(base)[1] - _range_from_mapping(base)[0]
    requested = (spec.seed_index_offset, spec.seed_index_offset + raw_target)
    for reserved in policy.reserved_ranges():
        if requested[0] < reserved[1] and reserved[0] < requested[1]:
            raise ControlError(
                409, "offset_conflict", "Seed offset range is already reserved"
            )
    run_id = uuid4().hex
    run_dir = policy.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    base_path = policy.root / spec.base_config
    task_path = policy.root / spec.task_card_config
    effective = dict(base)
    stage_counts = effective.get("stage_seed_counts")
    stage_count = (
        len(stage_counts) if isinstance(stage_counts, list) and stage_counts else 1
    )
    effective.update(
        {
            "provider": spec.provider,
            "protocol": spec.protocol,
            "base_url": spec.base_url,
            "model": spec.model,
            "force_model": spec.force_model,
            "discover_models": not spec.force_model,
            "api_key_env": CONTROL_KEY_ENV,
            "output_dir": output_relative,
            "task_card_config": spec.task_card_config,
            "seed_index_offset": spec.seed_index_offset,
            "concurrency_stages": [spec.concurrency] * stage_count,
            "timeout_seconds": spec.timeout_seconds,
            "max_retries": spec.max_retries,
            "wall_clock_deadline_seconds": spec.wall_clock_deadline_seconds,
            "max_requests": spec.max_requests,
            "max_output_tokens_total": spec.max_output_tokens_total,
            "cooldown_seconds": spec.cooldown_seconds,
            "cooldown_poll_seconds": spec.cooldown_poll_seconds,
            "discovery_timeout_seconds": spec.discovery_timeout_seconds,
            "quota_epoch_id": f"control-{run_id}",
        }
    )
    effective.pop("monotonic_expansion_from", None)
    for key in tuple(effective):
        if key.startswith("fallback_"):
            effective.pop(key, None)
    sop_raw = effective.get("sop_dir", "skills")
    sop_path = (policy.root / str(sop_raw)).resolve()
    try:
        sop_digest = _tree_sha256(sop_path)
    except ValueError as error:
        raise ControlError(
            409, "untrusted_sop", "SOP tree contains an unsafe entry"
        ) from error
    control_provenance = {
        "schema_version": CONTROL_SCHEMA,
        "run_id": run_id,
        "created_at": _iso(),
        "base_config": spec.base_config,
        "base_config_sha256": _sha256_file(base_path),
        "task_card_config_sha256": _sha256_file(task_path),
        "sop_tree_sha256": sop_digest,
        "output_dir": output_relative,
        "seed_index_offset": spec.seed_index_offset,
        "raw_collection_target": raw_target,
        "concurrency": spec.concurrency,
        "provider_profile": {
            "provider": spec.provider,
            "protocol": spec.protocol,
            "base_url": spec.base_url,
            "model": spec.model,
            "force_model": spec.force_model,
        },
        "transport": {
            "timeout_seconds": spec.timeout_seconds,
            "max_retries": spec.max_retries,
            "wall_clock_deadline_seconds": spec.wall_clock_deadline_seconds,
        },
        "supervisor": {
            "reconnect_attempts": spec.reconnect_attempts,
            "reconnect_backoff_seconds": spec.reconnect_backoff_seconds,
            "wait_cooldown": spec.wait_cooldown,
        },
        "network_route": spec.network_route,
        "credential_env": CONTROL_KEY_ENV,
        "credential_persisted": False,
        "invocation": {
            "module": "anchor_mvp.data.automation",
            "shell": False,
            "wait_cooldown": spec.wait_cooldown,
        },
    }
    effective["control_plane"] = control_provenance
    if _contains_inline_secret(effective):
        raise ControlError(
            500, "config_generation_failed", "Effective config was rejected"
        )
    config_bytes = yaml.safe_dump(
        effective, allow_unicode=True, sort_keys=False
    ).encode("utf-8")
    launch_sha = hashlib.sha256(config_bytes).hexdigest()
    effective_path = run_dir / "effective-config.yaml"
    _exclusive_write(effective_path, config_bytes)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_at": control_provenance["created_at"],
        "base_config": spec.base_config,
        "output_dir": output_relative,
        "output_label": output_dir.name,
        "seed_index_offset": spec.seed_index_offset,
        "raw_collection_target": raw_target,
        "launch_config_sha256": launch_sha,
        "corpus_binding_sha256": None,
        "effective_config": "effective-config.yaml",
        "reconnect_attempts": spec.reconnect_attempts,
        "reconnect_backoff_seconds": spec.reconnect_backoff_seconds,
        "wait_cooldown": spec.wait_cooldown,
        "network_route": spec.network_route,
        "credential_persisted": False,
    }
    manifest_path = run_dir / "control-manifest.json"
    _exclusive_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        ),
    )
    return GeneratedRun(
        run_id=run_id,
        run_dir=run_dir,
        effective_config=effective_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        output_relative=output_relative,
        output_label=output_dir.name,
        base_config=spec.base_config,
        launch_config_sha256=launch_sha,
        reconnect_attempts=spec.reconnect_attempts,
        reconnect_backoff_seconds=spec.reconnect_backoff_seconds,
        wait_cooldown=spec.wait_cooldown,
        network_route=spec.network_route,
    )


def _exclusive_write(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_generated_run(policy: WorkspacePolicy, run_id: object) -> GeneratedRun:
    run_dir = policy.run_path(run_id)
    manifest_path = run_dir / "control-manifest.json"
    effective_path = run_dir / "effective-config.yaml"
    if (
        manifest_path.is_symlink()
        or effective_path.is_symlink()
        or not manifest_path.is_file()
        or not effective_path.is_file()
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlError(
            409, "run_not_trusted", "Control run cannot be resumed"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    try:
        effective = load_strict_mapping(effective_path)
        effective_sha = _sha256_file(effective_path)
    except (OSError, ValueError) as error:
        raise ControlError(
            409, "run_not_trusted", "Control run cannot be resumed"
        ) from error
    expected_sha = manifest.get("launch_config_sha256")
    if (
        not isinstance(expected_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or not hmac.compare_digest(expected_sha, effective_sha)
    ):
        raise ControlError(
            409, "launch_config_changed", "Effective config no longer matches"
        )
    control = effective.get("control_plane")
    if (
        not isinstance(control, Mapping)
        or control.get("schema_version") != CONTROL_SCHEMA
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    directory_run_id = run_dir.name
    if not (
        hmac.compare_digest(str(manifest.get("run_id", "")), directory_run_id)
        and hmac.compare_digest(str(control.get("run_id", "")), directory_run_id)
        and hmac.compare_digest(str(run_id), directory_run_id)
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")

    manifest_output = manifest.get("output_dir")
    effective_output = effective.get("output_dir")
    control_output = control.get("output_dir")
    if not (
        isinstance(manifest_output, str)
        and isinstance(effective_output, str)
        and isinstance(control_output, str)
        and manifest_output == effective_output == control_output
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    output_relative, output_dir = policy.output_path(
        effective.get("output_dir"), must_exist=True
    )
    output_label = output_dir.name
    if manifest.get("output_label") != output_label:
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")

    manifest_base = manifest.get("base_config")
    control_base = control.get("base_config")
    if not (
        isinstance(manifest_base, str)
        and isinstance(control_base, str)
        and manifest_base == control_base
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    base_config, base_path = policy.config_path(control.get("base_config"))
    base_sha = control.get("base_config_sha256")
    if not isinstance(base_sha, str) or not hmac.compare_digest(
        base_sha, _sha256_file(base_path)
    ):
        raise ControlError(409, "source_drift", "Base config no longer matches")

    task_relative, task_path = policy.task_card_path(effective.get("task_card_config"))
    task_sha = control.get("task_card_config_sha256")
    if (
        effective.get("task_card_config") != task_relative
        or not isinstance(task_sha, str)
        or not hmac.compare_digest(task_sha, _sha256_file(task_path))
    ):
        raise ControlError(409, "source_drift", "Task-card config no longer matches")

    sop_raw = effective.get("sop_dir", "skills")
    if not isinstance(sop_raw, str) or not sop_raw:
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    sop_relative = Path(sop_raw)
    sop_lexical = policy.root / sop_relative
    if (
        sop_relative.is_absolute()
        or ".." in sop_relative.parts
        or policy._has_linklike_component(sop_lexical, policy.root)
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    sop_path = sop_lexical.resolve()
    if (
        not sop_path.is_relative_to(policy.root)
        or sop_path.is_symlink()
        or not sop_path.is_dir()
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    try:
        sop_sha = _tree_sha256(sop_path)
    except ValueError as error:
        raise ControlError(409, "source_drift", "SOP tree no longer matches") from error
    recorded_sop_sha = control.get("sop_tree_sha256")
    if not isinstance(recorded_sop_sha, str) or not hmac.compare_digest(
        recorded_sop_sha, str(sop_sha)
    ):
        raise ControlError(409, "source_drift", "SOP tree no longer matches")

    supervisor = control.get("supervisor")
    invocation = control.get("invocation")
    if not isinstance(supervisor, Mapping) or not isinstance(invocation, Mapping):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    try:
        reconnect_attempts = int(supervisor["reconnect_attempts"])
        reconnect_backoff = float(supervisor["reconnect_backoff_seconds"])
        wait_cooldown = supervisor["wait_cooldown"]
        network_route = str(control["network_route"])
    except (KeyError, TypeError, ValueError) as error:
        raise ControlError(
            409, "run_not_trusted", "Control run cannot be resumed"
        ) from error
    if (
        isinstance(wait_cooldown, bool) is False
        or manifest.get("reconnect_attempts") != reconnect_attempts
        or manifest.get("reconnect_backoff_seconds") != reconnect_backoff
        or manifest.get("wait_cooldown") is not wait_cooldown
        or network_route not in NETWORK_ROUTE_MODES
        or manifest.get("network_route") != network_route
        or effective.get("api_key_env") != CONTROL_KEY_ENV
        or control.get("credential_env") != CONTROL_KEY_ENV
        or control.get("credential_persisted") is not False
        or invocation.get("module") != "anchor_mvp.data.automation"
        or invocation.get("shell") is not False
        or invocation.get("wait_cooldown") is not wait_cooldown
    ):
        raise ControlError(409, "run_not_trusted", "Control run cannot be resumed")
    return GeneratedRun(
        run_id=directory_run_id,
        run_dir=run_dir,
        effective_config=effective_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        output_relative=output_relative,
        output_label=output_label,
        base_config=base_config,
        launch_config_sha256=effective_sha,
        reconnect_attempts=reconnect_attempts,
        reconnect_backoff_seconds=reconnect_backoff,
        wait_cooldown=wait_cooldown,
        network_route=network_route,
    )


class ProcessLike(Protocol):
    pid: int

    def wait(self, timeout: float | None = None) -> int: ...

    def poll(self) -> int | None: ...

    def send_signal(self, signal_value: int) -> None: ...


PopenFactory = Callable[..., ProcessLike]
AttachCallback = Callable[[str, Path], None]
ProbeBackend = Callable[[str, str, str, float], dict[str, object]]


class SystemSignaler:
    def __init__(
        self,
        platform_name: str | None = None,
        *,
        killpg: Callable[[int, int], None] | None = None,
        getpgid: Callable[[int], int] | None = None,
        run: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    ) -> None:
        self.platform_name = platform_name or os.name
        self.killpg = killpg or getattr(os, "killpg", None)
        self.getpgid = getpgid or getattr(os, "getpgid", None)
        self.run = run or subprocess.run
        if self.platform_name != "nt" and (self.killpg is None or self.getpgid is None):
            raise ValueError("POSIX process-group signaling is unavailable")

    def popen_group_kwargs(self) -> dict[str, object]:
        if self.platform_name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return {"start_new_session": True}

    def graceful(self, process: ProcessLike) -> None:
        try:
            if self.platform_name == "nt":
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", 1))
            else:
                assert self.killpg is not None and self.getpgid is not None
                self.killpg(self.getpgid(process.pid), signal.SIGINT)
        except (OSError, ProcessLookupError):
            return

    def terminate(self, process: ProcessLike) -> None:
        try:
            if self.platform_name == "nt":
                self.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                assert self.killpg is not None and self.getpgid is not None
                self.killpg(self.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return

    def kill(self, process: ProcessLike) -> None:
        try:
            if self.platform_name == "nt":
                self.terminate(process)
            else:
                assert self.killpg is not None and self.getpgid is not None
                self.killpg(self.getpgid(process.pid), getattr(signal, "SIGKILL", 9))
        except (OSError, ProcessLookupError):
            return


@dataclass
class ManagedJob:
    generated: GeneratedRun
    generation: str
    process_state: str = "starting"
    workload_state: str = "unknown"
    process: ProcessLike | None = None
    exit_code: int | None = None
    started_at: str = field(default_factory=_iso)
    finished_at: str | None = None
    last_error_code: str | None = None
    stop_requested: bool = False
    reconnect_used: int = 0
    next_reconnect_at: str | None = None
    reconnect_cancel: threading.Event = field(default_factory=threading.Event)
    lock_path: Path | None = None
    lock_owner_token: str | None = None


class ControlPlane:
    def __init__(
        self,
        workspace_root: Path,
        *,
        attach_callback: AttachCallback | None = None,
        popen_factory: PopenFactory | None = None,
        command_builder: Callable[[GeneratedRun], list[str]] | None = None,
        signaler: SystemSignaler | None = None,
        probe_backend: ProbeBackend | None = None,
        graceful_timeout_seconds: float = 8.0,
        terminate_timeout_seconds: float = 3.0,
    ) -> None:
        self.policy = WorkspacePolicy(workspace_root)
        self.attach_callback = attach_callback
        self.popen_factory = popen_factory or subprocess.Popen
        self._custom_command_builder = command_builder is not None
        self.command_builder = command_builder or self._default_command
        self.signaler = signaler or SystemSignaler()
        self.probe_backend = probe_backend or discover_models
        self.graceful_timeout_seconds = graceful_timeout_seconds
        self.terminate_timeout_seconds = terminate_timeout_seconds
        self.secret = SecretSlot()
        self.lock = threading.RLock()
        self.probe_lock = threading.Lock()
        self.probe_active = False
        self.job: ManagedJob | None = None
        self.events: deque[dict[str, object]] = deque(maxlen=100)
        self.closed = False

    def options(self) -> dict[str, object]:
        return self.policy.options()

    def start_new(self, payload: Mapping[str, object]) -> dict[str, object]:
        spec, credential, base = parse_start_spec(payload, self.policy)
        with self.lock:
            self._require_inactive()
            generated = generate_run(self.policy, spec, base)
            lock_owner_token: str | None = None
            try:
                try:
                    generated.output_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError as error:
                    raise ControlError(
                        409,
                        "new_shard_required",
                        "New runs require a unique empty shard",
                    ) from error
                lock_path, lock_owner_token = self._acquire_shard_lock(generated)
                self.secret.set(credential)
                job = ManagedJob(
                    generated=generated,
                    generation=uuid4().hex,
                    lock_path=lock_path,
                    lock_owner_token=lock_owner_token,
                )
                self.job = job
                self._event("starting", generated.output_label)
                self._spawn(job)
                if self.attach_callback is not None:
                    self.attach_callback(generated.output_label, generated.output_dir)
            except Exception:
                self.secret.clear()
                self._cleanup_failed_start(generated, lock_owner_token)
                self.job = None
                raise
            return self.public()

    def continue_run(self, payload: Mapping[str, object]) -> dict[str, object]:
        _require_fields(payload, CONTINUE_FIELDS)
        credential = validate_api_key(payload.get("api_key"))
        generated = _load_generated_run(self.policy, payload.get("run_id"))
        with self.lock:
            previous_job = self.job
            known_terminal_owner = bool(
                previous_job is not None
                and previous_job.generated.run_id == generated.run_id
                and previous_job.process_state
                not in {
                    "starting",
                    "running",
                    "stopping",
                    "terminating",
                    "reconnect_wait",
                }
            )
            self._require_inactive()
            status = self._load_workload_status(generated)
            if status is None:
                raise ControlError(
                    409, "resume_status_missing", "Resume requires automation status"
                )
            state = status.get("state")
            if state == "complete":
                raise ControlError(
                    409, "already_complete", "Completed runs are not resumed"
                )
            # A freshly-started control process cannot prove that a status left
            # in the middle of a worker is stale. Treat it as externally owned
            # even when the advisory lock is absent. Only the same in-memory
            # controller that observed its child exit may resume that state.
            if (
                state == "running" or status.get("current_worker") is not None
            ) and not known_terminal_owner:
                raise ControlError(
                    409,
                    "unmanaged_owner",
                    "Active-looking shard state is attach-only",
                )
            if generated.output_dir.joinpath(LOCK_NAME).exists():
                raise ControlError(
                    409, "unmanaged_owner", "Shard has an unmanaged owner; attach only"
                )
            binding = status.get("config_binding_sha256")
            if not isinstance(binding, str) or not re.fullmatch(
                r"[0-9a-f]{64}", binding
            ):
                raise ControlError(
                    409, "binding_missing", "Corpus binding is unavailable"
                )
            self._record_corpus_binding(generated, binding)
            lock_path, lock_owner_token = self._acquire_shard_lock(generated)
            self.secret.set(credential)
            job = ManagedJob(
                generated=generated,
                generation=uuid4().hex,
                lock_path=lock_path,
                lock_owner_token=lock_owner_token,
                workload_state=str(state or "unknown"),
            )
            self.job = job
            self._event("continuing", generated.output_label)
            try:
                self._spawn(job)
                if self.attach_callback is not None:
                    self.attach_callback(generated.output_label, generated.output_dir)
            except Exception:
                self._release_shard_lock(job)
                self.secret.clear()
                self.job = None
                raise
            return self.public()

    def attach_monitor(self, payload: Mapping[str, object]) -> dict[str, object]:
        _require_fields(payload, ATTACH_FIELDS)
        output_relative, output_dir = self.policy.output_path(
            payload.get("output_dir"), must_exist=True
        )
        del output_relative
        label = _text(payload, "label", LABEL_RE)
        if self.attach_callback is None:
            raise ControlError(
                409, "monitor_unavailable", "Monitor attachment is unavailable"
            )
        self.attach_callback(label, output_dir)
        self._event("attached_read_only", label)
        return {"attached": True, "label": label, "managed": False}

    def stop(self, job_id: object) -> dict[str, object]:
        if not isinstance(job_id, str) or not RUN_ID_RE.fullmatch(job_id):
            raise ControlError(400, "invalid_run_id", "Run ID is invalid")
        with self.lock:
            job = self.job
            if job is None or job.generated.run_id != job_id:
                raise ControlError(
                    409, "stale_job", "Requested job is not the active job"
                )
            if job.process_state in {"stopping", "terminating"}:
                return self.public()
            if job.process_state == "reconnect_wait":
                job.stop_requested = True
                job.reconnect_cancel.set()
                self._finalize(job, exit_code=job.exit_code, failed=False)
                return self.public()
            if job.process is None or job.process.poll() is not None:
                raise ControlError(409, "not_running", "Managed process is not running")
            job.stop_requested = True
            job.process_state = "stopping"
            job.reconnect_cancel.set()
            process = job.process
            generation = job.generation
            self._event("graceful_stop_requested", job.generated.output_label)
            threading.Thread(
                target=self._stop_worker,
                args=(generation, process),
                name=f"anchor-stop-{generation[:8]}",
                daemon=True,
            ).start()
            return self.public()

    def clear_credential(self) -> dict[str, object]:
        with self.lock:
            self.secret.clear()
            self._event("credential_cleared", "control-plane")
            return {"credential_loaded": False}

    def probe_models(self, payload: Mapping[str, object]) -> dict[str, object]:
        _require_fields(payload, PROBE_FIELDS)
        base_url = validate_base_url(payload.get("base_url"))
        protocol = payload.get("protocol")
        if protocol not in ALLOWED_PROTOCOLS:
            raise ControlError(400, "invalid_protocol", "Protocol is invalid")
        credential = validate_api_key(payload.get("api_key"))
        model = payload.get("model")
        if model not in {None, ""} and (
            not isinstance(model, str) or not MODEL_RE.fullmatch(model)
        ):
            raise ControlError(400, "invalid_model", "Model ID is invalid")
        _boolean(payload, "force_model")
        timeout = _number(payload, "timeout_seconds", 1, 120)
        with self.probe_lock:
            with self.lock:
                if self.job is not None and self.job.process_state in {
                    "starting",
                    "running",
                    "stopping",
                    "terminating",
                    "reconnect_wait",
                }:
                    raise ControlError(
                        409, "process_active", "Stop the active run before probing"
                    )
                self.probe_active = True
            try:
                # The probe credential is request-local. It must never enter the
                # child-process SecretSlot shared with Start/Continue.
                result = self.probe_backend(
                    base_url, str(protocol), credential, timeout
                )
            finally:
                with self.lock:
                    self.probe_active = False
        models = result.get("models", [])
        if not isinstance(models, list):
            models = []
        safe_models = [
            value
            for value in models
            if isinstance(value, str)
            and MODEL_RE.fullmatch(value)
            and not hmac.compare_digest(value, credential)
        ][:500]
        status = result.get("status")
        safe_status = (
            status
            if status
            in {
                "success",
                "auth_error",
                "rate_limited",
                "unsupported",
                "server_error",
                "network_error",
                "invalid_response",
            }
            else "invalid_response"
        )
        return {
            "status": safe_status,
            "models": safe_models,
            "model_count": len(safe_models),
        }

    def public(self) -> dict[str, object]:
        with self.lock:
            job = self.job
            if job is None:
                value: dict[str, object] = {
                    "schema_version": CONTROL_SCHEMA,
                    "enabled": True,
                    "process_state": "idle",
                    "workload_state": "unknown",
                    "run_id": None,
                    "generation": None,
                    "exit_code": None,
                    "output_label": None,
                    "base_config": None,
                    "started_at": None,
                    "finished_at": None,
                    "launch_config_sha256": None,
                    "corpus_binding_sha256": None,
                    "credential_loaded": self.secret.configured,
                    "can_start": True,
                    "can_stop": False,
                    "can_continue": False,
                    "last_error_code": None,
                    "reconnect": {"used": 0, "maximum": 0, "next_at": None},
                    "events": list(self.events),
                }
                return value
            status = self._load_workload_status(job.generated)
            if status is not None:
                state = status.get("state")
                if isinstance(state, str) and re.fullmatch(
                    r"[A-Za-z0-9_.:-]{1,80}", state
                ):
                    job.workload_state = state
                binding = status.get("config_binding_sha256")
                if isinstance(binding, str) and re.fullmatch(r"[0-9a-f]{64}", binding):
                    try:
                        self._record_corpus_binding(job.generated, binding)
                    except ControlError:
                        job.last_error_code = "corpus_binding_changed"
            manifest = self._manifest(job.generated)
            corpus_binding = manifest.get("corpus_binding_sha256")
            active = job.process_state in {
                "starting",
                "running",
                "stopping",
                "terminating",
                "reconnect_wait",
            }
            value = {
                "schema_version": CONTROL_SCHEMA,
                "enabled": True,
                "process_state": job.process_state,
                "workload_state": job.workload_state,
                "run_id": job.generated.run_id,
                "generation": job.generation,
                "exit_code": job.exit_code,
                "output_label": job.generated.output_label,
                "base_config": job.generated.base_config,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "launch_config_sha256": job.generated.launch_config_sha256,
                "corpus_binding_sha256": corpus_binding,
                "credential_loaded": self.secret.configured,
                "can_start": not active,
                "can_stop": active,
                "can_continue": not active and job.workload_state != "complete",
                "last_error_code": job.last_error_code,
                "reconnect": {
                    "used": job.reconnect_used,
                    "maximum": job.generated.reconnect_attempts,
                    "next_at": job.next_reconnect_at,
                },
                "events": list(self.events),
            }
            if set(value) != CONTROL_PUBLIC_FIELDS:
                raise RuntimeError("control public schema drift")
            return value

    def close(self) -> None:
        with self.lock:
            self.closed = True
            job = self.job
            if (
                job is not None
                and job.process is not None
                and job.process.poll() is None
            ):
                job.stop_requested = True
                job.reconnect_cancel.set()
                self.signaler.graceful(job.process)
            self.secret.clear()

    def _spawn(self, job: ManagedJob) -> None:
        credential = self.secret.reveal()
        argv = self.command_builder(job.generated)
        self._validate_argv(argv, job.generated)
        environment = self._child_environment(credential, job.generated.network_route)
        kwargs: dict[str, object] = {
            "cwd": str(self.policy.root),
            "env": environment,
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        kwargs.update(self.signaler.popen_group_kwargs())
        try:
            process = self.popen_factory(argv, **kwargs)
        except Exception as error:
            job.process_state = "failed"
            job.last_error_code = "spawn_failed"
            raise ControlError(
                500, "spawn_failed", "Distillation process could not start"
            ) from error
        finally:
            environment.pop(CONTROL_KEY_ENV, None)
            credential = ""
        job.process = process
        job.process_state = "running"
        self._event("process_started", job.generated.output_label)
        threading.Thread(
            target=self._supervise,
            args=(job.generation,),
            name=f"anchor-reaper-{job.generation[:8]}",
            daemon=True,
        ).start()

    def _supervise(self, generation: str) -> None:
        while True:
            with self.lock:
                job = self.job
                if job is None or job.generation != generation or job.process is None:
                    return
                process = job.process
            exit_code = process.wait()
            with self.lock:
                job = self.job
                if (
                    job is None
                    or job.generation != generation
                    or job.process is not process
                ):
                    return
                job.exit_code = exit_code
                if job.stop_requested or self.closed:
                    self._finalize(job, exit_code=exit_code, failed=False)
                    return
                if (
                    exit_code != 0
                    and job.reconnect_used < job.generated.reconnect_attempts
                    and self.secret.configured
                ):
                    job.reconnect_used += 1
                    job.process_state = "reconnect_wait"
                    delay = min(
                        3600.0,
                        job.generated.reconnect_backoff_seconds
                        * (2 ** (job.reconnect_used - 1)),
                    )
                    job.next_reconnect_at = (
                        datetime.now(timezone.utc) + timedelta(seconds=delay)
                    ).isoformat()
                    self._event("reconnect_wait", job.generated.output_label)
                    cancel = job.reconnect_cancel
                else:
                    self._finalize(job, exit_code=exit_code, failed=exit_code != 0)
                    return
            if cancel.wait(delay):
                with self.lock:
                    current = self.job
                    if current is not None and current.generation == generation:
                        self._finalize(current, exit_code=exit_code, failed=False)
                return
            with self.lock:
                current = self.job
                if (
                    current is None
                    or current.generation != generation
                    or current.stop_requested
                ):
                    return
                try:
                    self._spawn_without_reaper(current)
                except ControlError:
                    self._finalize(current, exit_code=exit_code, failed=True)
                    return

    def _spawn_without_reaper(self, job: ManagedJob) -> None:
        credential = self.secret.reveal()
        argv = self.command_builder(job.generated)
        self._validate_argv(argv, job.generated)
        environment = self._child_environment(credential, job.generated.network_route)
        kwargs: dict[str, object] = {
            "cwd": str(self.policy.root),
            "env": environment,
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        kwargs.update(self.signaler.popen_group_kwargs())
        try:
            job.process = self.popen_factory(argv, **kwargs)
        except Exception as error:
            job.last_error_code = "reconnect_spawn_failed"
            raise ControlError(
                500, "spawn_failed", "Distillation process could not restart"
            ) from error
        finally:
            environment.pop(CONTROL_KEY_ENV, None)
            credential = ""
        job.process_state = "running"
        job.next_reconnect_at = None
        self._event("process_reconnected", job.generated.output_label)

    def _stop_worker(self, generation: str, process: ProcessLike) -> None:
        self.signaler.graceful(process)
        try:
            process.wait(timeout=self.graceful_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        with self.lock:
            job = self.job
            if (
                job is None
                or job.generation != generation
                or job.process is not process
            ):
                return
            job.process_state = "terminating"
            self._event("graceful_stop_timeout", job.generated.output_label)
        self.signaler.terminate(process)
        try:
            process.wait(timeout=self.terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            self.signaler.kill(process)

    def _finalize(
        self, job: ManagedJob, *, exit_code: int | None, failed: bool
    ) -> None:
        if self.job is not job:
            return
        job.exit_code = exit_code
        job.finished_at = _iso()
        job.next_reconnect_at = None
        job.process_state = "failed" if failed else "exited"
        if failed and job.last_error_code is None:
            job.last_error_code = "child_exit_nonzero"
        self._release_shard_lock(job)
        self.secret.clear()
        self._event(job.process_state, job.generated.output_label)

    def _require_inactive(self) -> None:
        if self.closed:
            raise ControlError(409, "control_closed", "Control plane is closing")
        if self.probe_active:
            raise ControlError(
                409, "probe_active", "Wait for model discovery to finish"
            )
        if self.job is not None and self.job.process_state in {
            "starting",
            "running",
            "stopping",
            "terminating",
            "reconnect_wait",
        }:
            raise ControlError(
                409, "process_active", "A managed process is already active"
            )

    def _default_command(self, generated: GeneratedRun) -> list[str]:
        argv = [
            sys.executable,
            "-m",
            "anchor_mvp.data.automation",
            "--config",
            str(generated.effective_config),
        ]
        if generated.wait_cooldown:
            argv.append("--wait-cooldown")
        return argv

    def _validate_argv(self, argv: Sequence[str], generated: GeneratedRun) -> None:
        expected = self._default_command(generated)
        if not self._custom_command_builder and list(argv) != expected:
            raise ControlError(
                500, "argv_rejected", "Internal process command was rejected"
            )
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ControlError(
                500, "argv_rejected", "Internal process command was rejected"
            )

    def _child_environment(self, credential: str, network_route: str) -> dict[str, str]:
        allowed_names = {
            "PATH",
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "PROGRAMDATA",
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONUTF8",
            "PYTHONIOENCODING",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "NO_PROXY",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
        }
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in allowed_names and not SECRET_ENV_RE.search(name)
        }
        source_root = str(self.policy.root / "src")
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root + os.pathsep + old_pythonpath if old_pythonpath else source_root
        )
        environment["PYTHONUTF8"] = "1"
        if network_route == "direct":
            environment["NO_PROXY"] = "*"
            environment["no_proxy"] = "*"
        elif network_route == "inherit":
            for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
            ):
                value = os.environ.get(name)
                if value is not None:
                    environment[name] = value
        else:
            raise ControlError(
                500, "invalid_network_route", "Stored network route is invalid"
            )
        environment[CONTROL_KEY_ENV] = credential
        return environment

    def _acquire_shard_lock(self, generated: GeneratedRun) -> tuple[Path, str]:
        lock_path = generated.output_dir / LOCK_NAME
        owner_token = secrets.token_hex(32)
        content = json.dumps(
            {
                "schema_version": CONTROL_SCHEMA,
                "run_id": generated.run_id,
                "launch_config_sha256": generated.launch_config_sha256,
                "owner_token": owner_token,
                "created_at": _iso(),
            },
            sort_keys=True,
        ).encode("utf-8")
        try:
            _exclusive_write(lock_path, content)
        except FileExistsError as error:
            raise ControlError(
                409, "unmanaged_owner", "Shard is already owned"
            ) from error
        return lock_path, owner_token

    def _release_shard_lock(self, job: ManagedJob) -> None:
        path = job.lock_path
        token = job.lock_owner_token
        if path is not None and token is not None:
            self._unlink_owned_lock(path, job.generated, token)
        job.lock_path = None
        job.lock_owner_token = None

    def _cleanup_failed_start(
        self, generated: GeneratedRun, owner_token: str | None
    ) -> None:
        lock_path = generated.output_dir / LOCK_NAME
        if owner_token is not None:
            self._unlink_owned_lock(lock_path, generated, owner_token)
        try:
            if generated.output_dir.is_dir() and not any(
                generated.output_dir.iterdir()
            ):
                generated.output_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def _unlink_owned_lock(
        path: Path, generated: GeneratedRun, owner_token: str
    ) -> bool:
        try:
            stat_before = path.stat()
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
            return False
        if not isinstance(value, Mapping):
            return False
        recorded_run = value.get("run_id")
        recorded_sha = value.get("launch_config_sha256")
        recorded_owner = value.get("owner_token")
        if not (
            isinstance(recorded_run, str)
            and isinstance(recorded_sha, str)
            and isinstance(recorded_owner, str)
            and hmac.compare_digest(recorded_run, generated.run_id)
            and hmac.compare_digest(recorded_sha, generated.launch_config_sha256)
            and hmac.compare_digest(recorded_owner, owner_token)
        ):
            return False
        try:
            stat_after = path.stat()
            if (stat_before.st_dev, stat_before.st_ino, stat_before.st_mtime_ns) != (
                stat_after.st_dev,
                stat_after.st_ino,
                stat_after.st_mtime_ns,
            ):
                return False
            path.unlink()
        except (FileNotFoundError, OSError):
            return False
        return True

    def _load_workload_status(
        self, generated: GeneratedRun
    ) -> dict[str, object] | None:
        path = generated.output_dir / "automation" / "status.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(value, dict):
            return None
        return {
            "state": value.get("state"),
            "current_worker": value.get("current_worker"),
            "config_binding_sha256": value.get("config_binding_sha256"),
        }

    def _manifest(self, generated: GeneratedRun) -> dict[str, object]:
        try:
            value = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _record_corpus_binding(self, generated: GeneratedRun, binding: str) -> None:
        manifest = self._manifest(generated)
        observed = manifest.get("corpus_binding_sha256")
        if observed is None:
            manifest["corpus_binding_sha256"] = binding
            _atomic_json(generated.manifest_path, manifest)
        elif not isinstance(observed, str) or not hmac.compare_digest(
            observed, binding
        ):
            raise ControlError(
                409, "corpus_binding_changed", "Corpus binding no longer matches"
            )

    def _event(self, event: str, target: str) -> None:
        self.events.append(
            {"at": _iso(), "event": event, "target": target, "content_free": True}
        )


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        _request: Request,
        _fp: object,
        _code: int,
        _msg: str,
        _headers: object,
        _new_url: str,
    ) -> None:
        return None


def discover_models(
    base_url: str, protocol: str, api_key: str, timeout_seconds: float
) -> dict[str, object]:
    suffix = (
        "v1/models"
        if protocol == "anthropic"
        and not urlsplit(base_url).path.rstrip("/").endswith("/v1")
        else "models"
    )
    endpoint = f"{base_url.rstrip('/')}/{suffix}"
    headers = {"Accept": "application/json", "User-Agent": "anchor-control/0.1"}
    if protocol == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, headers=headers, method="GET")
    try:
        opener = build_opener(_RejectRedirects())
        with opener.open(  # noqa: S310 - validated URL, redirects disabled
            request, timeout=timeout_seconds
        ) as response:
            raw = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
    except HTTPError as error:
        if error.code in {401, 403}:
            status = "auth_error"
        elif error.code == 429:
            status = "rate_limited"
        elif error.code in {404, 405, 501}:
            status = "unsupported"
        elif 300 <= error.code < 400:
            status = "invalid_response"
        elif error.code >= 500:
            status = "server_error"
        else:
            status = "invalid_response"
        return {"status": status, "models": []}
    except (OSError, TimeoutError, URLError):
        return {"status": "network_error", "models": []}
    if len(raw) > MAX_MODEL_RESPONSE_BYTES:
        return {"status": "invalid_response", "models": []}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "invalid_response", "models": []}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        return {"status": "invalid_response", "models": []}
    models = sorted(
        {
            str(item["id"])
            for item in payload["data"]
            if isinstance(item, Mapping)
            and isinstance(item.get("id"), str)
            and MODEL_RE.fullmatch(str(item["id"]))
        }
    )
    return {"status": "success" if models else "invalid_response", "models": models}


def csrf_token() -> bytearray:
    return bytearray(secrets.token_bytes(32))


def csrf_cookie_value(token: bytearray) -> str:
    return bytes(token).hex()
