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
import math
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
REASONING_EFFORTS = frozenset({"low", "medium", "high", "max"})
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
        "reasoning_enabled",
        "reasoning_effort",
        "pricing_route",
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
        "provider_profile",
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
    payload: Mapping[str, object],
    name: str,
    minimum: int,
    maximum: int | None,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlError(400, f"invalid_{name}", f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
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


def _reasoning_effort(value: object) -> str:
    if value not in REASONING_EFFORTS:
        raise ControlError(
            400,
            "invalid_reasoning_effort",
            "Reasoning effort must be low, medium, high, or max",
        )
    return str(value)


def _pricing_route(value: object) -> str:
    if not isinstance(value, str) or LABEL_RE.fullmatch(value) is None:
        raise ControlError(
            400,
            "invalid_pricing_route",
            "Pricing route must be manual or one catalog provider ID",
        )
    return value


_STAGE_REASONING_KEYS = (
    "thinking_effort_seed",
    "thinking_effort_plan",
    "thinking_effort_tool_policy",
    "thinking_effort_frontend",
    "thinking_effort_review",
    "thinking_effort_security",
)


def _required_reasoning_effort(base: Mapping[str, Any]) -> str | None:
    """Return a provider-neutral formal-profile floor when one is declared.

    New profiles may declare ``reasoning_policy.required`` explicitly. Existing
    formal GLM/Kimi profiles are recognized by their complete all-stage MAX
    contract, without hard-coding provider or model IDs.
    """

    policy = base.get("reasoning_policy")
    if policy is not None:
        if not isinstance(policy, Mapping) or set(policy) != {"required", "effort"}:
            raise ControlError(
                400,
                "invalid_reasoning_policy",
                "reasoning_policy must contain only required and effort",
            )
        required = policy.get("required")
        effort = policy.get("effort")
        if not isinstance(required, bool) or effort not in REASONING_EFFORTS:
            raise ControlError(
                400,
                "invalid_reasoning_policy",
                "reasoning_policy required/effort is invalid",
            )
        return str(effort) if required else None
    if (
        base.get("thinking_enabled") is True
        and base.get("thinking_effort") == "max"
        and all(base.get(key) == "max" for key in _STAGE_REASONING_KEYS)
    ):
        return "max"
    return None


def _proxy_detected() -> bool:
    try:
        return bool(getproxies())
    except (OSError, ValueError):
        return False


def _default_route_audit() -> dict[str, object]:
    """Return a content-free, read-only default-route warning.

    ``NO_PROXY`` can bypass proxy environment variables, but it cannot override
    a TUN adapter selected by the operating-system routing table.  This probe is
    deliberately observational: it never adds, replaces, or deletes a route.
    """

    result: dict[str, object] = {
        "status": "unsupported" if os.name != "nt" else "unknown",
        "virtual_default_route_detected": None,
        "physical_default_route_detected": None,
        "physical_route_pinned": False,
        "direct_semantics": "proxy_env_bypass_only",
    }
    if os.name != "nt":
        return result
    script = (
        "$ErrorActionPreference='Stop';"
        "@(Get-NetRoute -AddressFamily IPv4 -DestinationPrefix '0.0.0.0/0' "
        "| Select-Object ifIndex,InterfaceAlias,NextHop,RouteMetric "
        "| ConvertTo-Json -Compress)"
    )
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return result
        decoded = json.loads(completed.stdout)
        routes = decoded if isinstance(decoded, list) else [decoded]
        virtual = False
        physical = False
        for raw in routes:
            if not isinstance(raw, Mapping):
                continue
            alias = str(raw.get("InterfaceAlias", "")).casefold()
            next_hop = str(raw.get("NextHop", ""))
            looks_virtual = any(
                marker in alias
                for marker in ("tun", "tap", "clash", "flclash", "wintun", "vpn")
            ) or next_hop.startswith("198.18.") or next_hop.startswith("198.19.")
            if looks_virtual:
                virtual = True
            elif next_hop and next_hop != "0.0.0.0":
                physical = True
        result.update(
            {
                "status": "observed",
                "virtual_default_route_detected": virtual,
                "physical_default_route_detected": physical,
            }
        )
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return result


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
    reasoning_enabled: bool
    reasoning_effort: str
    pricing_route: str
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
    provider_profile: dict[str, object]
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
        formal_preflight = self.formal_preflight()
        formal_route = dict(formal_preflight["formal_route"])
        formal_dataset = dict(formal_preflight["formal_dataset"])
        formal_execution = dict(formal_preflight["execution_contract"])
        formal_gates = {
            name: formal_preflight[name]
            for name in (
                "component_ready",
                "bank_ready",
                "execution_contract_ready",
                "live_start_allowed",
                "reason_code",
            )
        }
        return {
            "base_configs": configs,
            "task_card_configs": task_cards,
            "protocols": sorted(ALLOWED_PROTOCOLS),
            "network_route_modes": ["direct", "inherit"],
            "proxy_detected": _proxy_detected(),
            "default_route_audit": _default_route_audit(),
            "formal_route": formal_route,
            "formal_dataset": formal_dataset,
            "formal_execution": formal_execution,
            "formal_gates": formal_gates,
            "formal_preflight": formal_preflight,
            "runs": runs[-50:],
            "limits": {
                "concurrency_min": 1,
                "concurrency_max": None,
                "concurrency_default": 1,
                "request_body_bytes": 16_384,
            },
        }

    def formal_preflight(self) -> dict[str, object]:
        """Return coordinator-compatible gates without reading task bodies.

        This payload is deliberately narrower than the full formal coordinator
        preflight.  It reads only public manifests, component hashes, and the
        locked v3 execution attestation.  Candidate JSONL, heldout files,
        credentials, provider routes, and sandboxes are outside this code path.
        Missing or malformed fields therefore fail closed instead of being
        inferred as ready.
        """

        formal_route = self._formal_route_status()
        formal_dataset = self._formal_dataset_status()
        formal_execution = self._formal_execution_status()
        component_ready = bool(
            formal_route.get("component_ready") is True
            and formal_execution.get("bundle_present") is True
        )
        bank_ready = formal_dataset.get("bank_ready") is True
        execution_contract_ready = formal_execution.get("ready") is True
        live_start_allowed = bool(
            component_ready and bank_ready and execution_contract_ready
        )
        if not component_ready:
            gate_reason = "formal_component_not_ready"
        elif not bank_ready:
            gate_reason = "formal_bank_not_ready"
        elif not execution_contract_ready:
            raw_reason = formal_execution.get("reason_code")
            gate_reason = (
                str(raw_reason)
                if isinstance(raw_reason, str) and raw_reason
                else "formal_execution_contract_not_ready"
            )
        else:
            raw_reason = formal_execution.get("reason_code")
            gate_reason = (
                str(raw_reason)
                if isinstance(raw_reason, str) and raw_reason
                else "generic_train_execution_contract_ready"
            )
        return {
            "schema_version": "anchor.swebench-ccswitch-preflight.v1",
            "content_free": True,
            "offline": True,
            "provider_requests": 0,
            "credentials_read": False,
            "sample_bodies_read": False,
            "sample_bodies_printed": False,
            "heldout_files_read": False,
            "component_ready": component_ready,
            "bank_ready": bank_ready,
            "execution_contract_ready": execution_contract_ready,
            "live_start_allowed": live_start_allowed,
            "reason_code": gate_reason,
            "formal_route": formal_route,
            "formal_dataset": formal_dataset,
            "execution_contract": formal_execution,
            "live_started": False,
        }

    def _formal_route_status(self) -> dict[str, object]:
        """Separate component evidence from WSL/container reachability.

        The dashboard does not probe or mutate WSL/Podman networking, so even a
        validated Windows route binary is never presented as end-to-end ready.
        """

        manifest_path = (
            self.root
            / "artifacts"
            / "tooling"
            / "ccswitch-patched"
            / "route-manifest.json"
        )
        component_ready = False
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                binary = manifest.get("binary") if isinstance(manifest, Mapping) else None
                patch = manifest.get("patch") if isinstance(manifest, Mapping) else None
                if isinstance(binary, Mapping) and isinstance(patch, Mapping):
                    binary_path = (self.root / str(binary.get("path", ""))).resolve()
                    patch_path = (self.root / str(patch.get("path", ""))).resolve()
                    component_ready = bool(
                        manifest.get("schema_version")
                        == "anchor.ccswitch-route-manifest.v1"
                        and manifest.get("ready") is True
                        and binary_path.is_relative_to(self.root)
                        and patch_path.is_relative_to(self.root)
                        and binary_path.is_file()
                        and patch_path.is_file()
                        and _sha256_file(binary_path) == binary.get("sha256")
                        and _sha256_file(patch_path) == patch.get("sha256")
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                component_ready = False
        return {
            "component_ready": component_ready,
            "live_route_container_reachable": None,
            "reachability_state": "not_probed_by_dashboard",
            "e2e_ready": False,
        }

    def _formal_dataset_status(self) -> dict[str, object]:
        manifest_path = (
            self.root / "artifacts" / "swebench" / "full-bank-v1" / "manifest.json"
        )
        counts: dict[str, int] = {}
        localization_present = False
        bank_ready = False
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                bilingual = (
                    manifest.get("bilingual")
                    if isinstance(manifest, Mapping)
                    else None
                )
                raw_counts = (
                    bilingual.get("counts") if isinstance(bilingual, Mapping) else None
                )
                if isinstance(raw_counts, Mapping):
                    for locale in ("en-US", "zh-CN"):
                        value = raw_counts.get(locale)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            counts[locale] = value
                localization_present = bool(
                    isinstance(bilingual, Mapping)
                    and bilingual.get("translation_manifest_present") is True
                )
                bank_ready = bool(
                    manifest.get("schema_version")
                    == "anchor.swebench-full-bank-manifest.v2"
                    and manifest.get("launch_ready") is True
                    and counts == {"en-US": 9504, "zh-CN": 9504}
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                counts = {}
                localization_present = False
                bank_ready = False
        return {
            "bank_ready": bank_ready,
            "locale_assignment_counts": counts,
            "language_routing_only": True,
            "zh_cn_localization_manifest_present": localization_present,
        }

    def _formal_execution_status(self) -> dict[str, object]:
        """Use the same locked v3 verifier as the formal coordinator."""

        bundle_path = (
            self.root
            / "artifacts"
            / "tooling"
            / "opencode-patched"
            / "bundle-manifest.json"
        )
        coordinator_config = (
            self.root / "configs" / "data" / "swebench_five_stage.ccswitch.yaml"
        )
        observed_version: str | None = None
        bundle_present = False
        if bundle_path.is_file() and not bundle_path.is_symlink():
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                source = bundle.get("source") if isinstance(bundle, Mapping) else None
                contract = (
                    source.get("tool_contract")
                    if isinstance(source, Mapping)
                    else None
                )
                observed = (
                    source.get("tool_contract_version")
                    if isinstance(source, Mapping)
                    else None
                )
                if isinstance(observed, str):
                    observed_version = observed
                bundle_present = bool(
                    bundle.get("schema_version") == "anchor.patched-opencode.bundle.v1"
                    and isinstance(contract, Mapping)
                    and contract.get("version") == observed_version
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                bundle_present = False
                observed_version = None
        generic_verification: Mapping[str, object] = {
            "mode": "generic_train_repo_base_commit",
            "not_official_swebench_pass": True,
            "ready": False,
            "reason_code": "generic_train_execution_contract_not_ready",
            "remaining_gates": ["execution_lock_invalid"],
            "official_evaluation_contract_ready": False,
            "official_evaluation_remaining_gates": [],
        }
        try:
            config = load_strict_mapping(coordinator_config)
            spec = config.get("execution_contract")
            if not isinstance(spec, Mapping):
                raise ValueError("execution contract spec missing")
            attestation_path = (self.root / str(spec.get("attestation", ""))).resolve()
            lock_path = (self.root / str(spec.get("lock", ""))).resolve()
            lock_sha256 = spec.get("lock_sha256")
            if (
                not attestation_path.is_relative_to(self.root)
                or not lock_path.is_relative_to(self.root)
                or not isinstance(lock_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", lock_sha256)
            ):
                raise ValueError("execution contract path/hash invalid")
            source_root = self.root / "src"
            tooling_root = self.root / "scripts" / "tooling"
            for import_root in (source_root, tooling_root):
                if str(import_root) not in sys.path:
                    sys.path.insert(0, str(import_root))
            from run_swebench_ccswitch import (  # noqa: PLC0415
                CoordinatorConfig,
                _distillation_execution_contract_gate,
            )

            coordinator = CoordinatorConfig.load(coordinator_config)
            result = _distillation_execution_contract_gate(coordinator)
            if isinstance(result, Mapping):
                generic_verification = result
        except (ImportError, OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            pass
        return {
            "bundle_present": bundle_present,
            "observed_tool_contract_version": observed_version,
            "required_tool_contract_version": "anchor.execution-tool-contract.v3",
            "mode": generic_verification.get("mode"),
            "not_official_swebench_pass": True,
            "ready": generic_verification.get("ready") is True,
            "reason_code": str(
                generic_verification.get(
                    "reason_code", "generic_train_execution_contract_not_ready"
                )
            ),
            "remaining_gates": list(
                generic_verification.get("remaining_gates", ())
            ),
            "official_evaluation_contract_ready": (
                generic_verification.get("official_evaluation_contract_ready") is True
            ),
            "official_evaluation_remaining_gates": list(
                generic_verification.get("official_evaluation_remaining_gates", ())
            ),
            "capability_gap": (
                None
                if generic_verification.get("ready") is True
                else "python_repository_validation_not_attested"
            ),
        }

    def formal_runtime_status(self) -> dict[str, object]:
        """Read only the coordinator's explicitly content-free status file."""

        path = (
            self.root
            / "artifacts"
            / "swebench"
            / "full-bank-live-v1"
            / "status.json"
        )
        empty: dict[str, object] = {
            "available": False,
            "state": "not_started",
            "submitted_tasks": 0,
            "completed_tasks": 0,
            "expected_tasks": 19008,
            "active_tasks": 0,
            "counts": {"completed": 0, "blocked": 0, "failed": 0},
            "stage_counts": {
                "planner": 0,
                "tool_policy": 0,
                "domain_builder": 0,
                "domain_review": 0,
                "security": 0,
            },
            "failure_counts": {},
            "requests": {
                "provider_requests": None,
                "provider_successes": None,
                "provider_failures": None,
                "retry_attempts": None,
            },
            "tokens": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_input_tokens": None,
            },
            "tasks_per_minute": None,
            "provider_output_tokens_per_second": None,
            "eta_seconds": None,
            "stage_progress_available": False,
            "token_metrics_available": False,
            "identity_verified": False,
            "fresh": False,
            "formal_lifecycle_control_available": False,
            "content_free": True,
            "control_run_id": None,
            "checkpoint_id": None,
            "config_sha256": None,
            "execution_lock_sha256": None,
            "resume_mode": None,
            "request_failure_counts": {},
            "last_error_code": None,
        }
        if not path.is_file() or path.is_symlink():
            return empty
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {**empty, "state": "invalid_status"}
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version")
            not in {
                "anchor.swebench-ccswitch-status.v1",
                "anchor.swebench-ccswitch-status.v2",
            }
            or value.get("content_free") is not True
        ):
            return {**empty, "state": "invalid_status"}
        schema = str(value.get("schema_version"))
        state = value.get("state")
        submitted = value.get("submitted_tasks")
        completed = value.get("completed_tasks")
        elapsed = value.get("elapsed_seconds")
        rate = value.get("tasks_per_minute")
        counts = value.get("counts")
        if (
            state
            not in {
                "running",
                "starting",
                "completed",
                "completed_with_failures",
                "failed",
                "stopped",
                "stopped_checkpoint_resumable",
            }
            or not isinstance(submitted, int)
            or isinstance(submitted, bool)
            or submitted < 0
            or not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed < 0
            or not isinstance(counts, Mapping)
        ):
            return {**empty, "state": "invalid_status"}
        safe_counts: dict[str, int] = {}
        for key in ("completed", "blocked", "failed"):
            item = counts.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                return {**empty, "state": "invalid_status"}
            safe_counts[key] = item
        expected = value.get("expected_tasks", 19008)
        active = value.get("active_tasks", 0)
        if (
            not isinstance(expected, int)
            or isinstance(expected, bool)
            or expected != 19008
            or not isinstance(active, int)
            or isinstance(active, bool)
            or active < 0
            or completed > submitted
            or submitted > expected
            or sum(safe_counts.values()) != completed
            or active > submitted - completed
            or (state not in {"running", "starting"} and active != 0)
        ):
            return {**empty, "state": "invalid_status"}
        stage_counts = dict(empty["stage_counts"])
        failure_counts: dict[str, int] = {}
        requests = dict(empty["requests"])
        tokens = dict(empty["tokens"])
        token_rate: float | None = None
        control_run_id: str | None = None
        checkpoint_id: str | None = None
        config_sha256: str | None = None
        execution_lock_sha256: str | None = None
        resume_mode: bool | None = None
        request_failure_counts: dict[str, int] = {}
        last_error_code: str | None = None
        if schema.endswith(".v2"):
            raw_stages = value.get("stage_counts")
            raw_failures = value.get("failure_counts")
            raw_requests = value.get("requests")
            raw_tokens = value.get("tokens")
            raw_request_failures = value.get("request_failure_counts")
            if not all(
                isinstance(item, Mapping)
                for item in (
                    raw_stages,
                    raw_failures,
                    raw_requests,
                    raw_tokens,
                    raw_request_failures,
                )
            ):
                return {**empty, "state": "invalid_status"}
            assert isinstance(raw_stages, Mapping)
            assert isinstance(raw_failures, Mapping)
            assert isinstance(raw_requests, Mapping)
            assert isinstance(raw_tokens, Mapping)
            assert isinstance(raw_request_failures, Mapping)
            control_run_id = value.get("control_run_id")  # type: ignore[assignment]
            checkpoint_id = value.get("checkpoint_id")  # type: ignore[assignment]
            config_sha256 = value.get("config_sha256")  # type: ignore[assignment]
            execution_lock_sha256 = value.get("execution_lock_sha256")  # type: ignore[assignment]
            resume_mode = value.get("resume_mode")  # type: ignore[assignment]
            if (
                not isinstance(control_run_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", control_run_id)
                or not all(
                    isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
                    for item in (
                        checkpoint_id,
                        config_sha256,
                        execution_lock_sha256,
                    )
                )
                or not isinstance(resume_mode, bool)
            ):
                return {**empty, "state": "invalid_status"}
            for key in stage_counts:
                item = raw_stages.get(key)
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    return {**empty, "state": "invalid_status"}
                stage_counts[key] = item
            for key, item in raw_failures.items():
                if (
                    not isinstance(key, str)
                    or not re.fullmatch(r"[a-z0-9_]{1,80}", key)
                    or not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < 0
                ):
                    return {**empty, "state": "invalid_status"}
                failure_counts[key] = item
            for key, item in raw_request_failures.items():
                if (
                    not isinstance(key, str)
                    or not re.fullmatch(r"[a-z0-9_]{1,80}", key)
                    or not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < 0
                ):
                    return {**empty, "state": "invalid_status"}
                request_failure_counts[key] = item
            raw_last_error = value.get("last_error_code")
            if raw_last_error is not None and (
                not isinstance(raw_last_error, str)
                or not re.fullmatch(r"[a-z0-9_]{1,80}", raw_last_error)
            ):
                return {**empty, "state": "invalid_status"}
            last_error_code = raw_last_error
            for key in requests:
                item = raw_requests.get(key)
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    return {**empty, "state": "invalid_status"}
                requests[key] = item
            if requests["provider_successes"] + requests["provider_failures"] > requests["provider_requests"]:
                return {**empty, "state": "invalid_status"}
            for key in tokens:
                item = raw_tokens.get(key)
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    return {**empty, "state": "invalid_status"}
                tokens[key] = item
            raw_token_rate = value.get("provider_output_tokens_per_second")
            if (
                isinstance(raw_token_rate, (int, float))
                and not isinstance(raw_token_rate, bool)
                and math.isfinite(float(raw_token_rate))
                and float(raw_token_rate) >= 0
            ):
                token_rate = float(raw_token_rate)
            else:
                return {**empty, "state": "invalid_status"}
        numeric_rate: float | None
        eta_seconds: float | None = None
        if schema.endswith(".v2"):
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
                or not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(float(rate))
                or float(rate) < 0
            ):
                return {**empty, "state": "invalid_status"}
            numeric_rate = float(rate)
            raw_eta = value.get("eta_seconds")
            if raw_eta is not None:
                if (
                    not isinstance(raw_eta, (int, float))
                    or isinstance(raw_eta, bool)
                    or not math.isfinite(float(raw_eta))
                    or float(raw_eta) < 0
                ):
                    return {**empty, "state": "invalid_status"}
                eta_seconds = float(raw_eta)
        else:
            numeric_rate = (
                float(rate)
                if isinstance(rate, (int, float))
                and not isinstance(rate, bool)
                and math.isfinite(float(rate))
                and float(rate) > 0
                else None
            )
            if (
                numeric_rate is None
                and isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and elapsed > 0
            ):
                numeric_rate = completed * 60.0 / float(elapsed)
            if state == "running" and numeric_rate:
                eta_seconds = max(
                    0.0, (expected - completed) * 60.0 / numeric_rate
                )
        raw_updated = value.get("updated_at")
        try:
            if isinstance(raw_updated, str):
                updated = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    raise ValueError("timezone required")
            else:
                updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            updated = updated.astimezone(timezone.utc)
            updated_at = updated.isoformat()
            age_seconds = (datetime.now(timezone.utc) - updated).total_seconds()
            if age_seconds < 0:
                raise ValueError("future status timestamp")
        except (OSError, ValueError):
            return {**empty, "state": "invalid_status"}
        fresh = state not in {"running", "starting"} or age_seconds <= 10.0
        return {
            **empty,
            "available": True,
            "state": state,
            "status_schema": schema,
            "submitted_tasks": submitted,
            "completed_tasks": completed,
            "expected_tasks": expected,
            "active_tasks": active,
            "counts": safe_counts,
            "stage_counts": stage_counts,
            "failure_counts": failure_counts,
            "requests": requests,
            "tokens": tokens,
            "tasks_per_minute": numeric_rate,
            "provider_output_tokens_per_second": token_rate,
            "eta_seconds": eta_seconds,
            "stage_progress_available": schema.endswith(".v2"),
            "token_metrics_available": schema.endswith(".v2"),
            "fresh": fresh,
            "status_age_seconds": age_seconds,
            "updated_at": updated_at,
            "control_run_id": control_run_id,
            "checkpoint_id": checkpoint_id,
            "config_sha256": config_sha256,
            "execution_lock_sha256": execution_lock_sha256,
            "resume_mode": resume_mode,
            "request_failure_counts": request_failure_counts,
            "last_error_code": last_error_code,
        }

    def formal_local_binding(self) -> dict[str, object]:
        config_path = (
            self.root / "configs" / "data" / "swebench_five_stage.ccswitch.yaml"
        )
        output_root = (
            self.root / "artifacts" / "swebench" / "full-bank-live-v1"
        )
        try:
            config = load_strict_mapping(config_path)
            execution = config.get("execution_contract")
            if not isinstance(execution, Mapping):
                raise ValueError("execution contract missing")
            lock_sha256 = execution.get("lock_sha256")
            if not isinstance(lock_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", lock_sha256
            ):
                raise ValueError("execution lock hash invalid")
            failed_startup_rearmable = False
            try:
                source_root = self.root / "src"
                tooling_root = self.root / "scripts" / "tooling"
                for import_root in (source_root, tooling_root):
                    if str(import_root) not in sys.path:
                        sys.path.insert(0, str(import_root))
                from run_swebench_ccswitch import (  # noqa: PLC0415
                    can_rearm_failed_start,
                )

                failed_startup_rearmable = can_rearm_failed_start(output_root)
            except (ImportError, OSError, ValueError):
                failed_startup_rearmable = False
            return {
                "ready": True,
                "config_sha256": _sha256_file(config_path),
                "execution_lock_sha256": lock_sha256,
                "status_exists": (output_root / "status.json").is_file(),
                "checkpoint_exists": (
                    output_root / "checkpoint.events.jsonl"
                ).is_file(),
                "failed_startup_rearmable": failed_startup_rearmable,
            }
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            return {
                "ready": False,
                "config_sha256": None,
                "execution_lock_sha256": None,
                "status_exists": False,
                "checkpoint_exists": False,
                "failed_startup_rearmable": False,
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
    reasoning_enabled_value = payload.get(
        "reasoning_enabled", base.get("thinking_enabled", True)
    )
    if not isinstance(reasoning_enabled_value, bool):
        raise ControlError(
            400,
            "invalid_reasoning_enabled",
            "reasoning_enabled must be boolean",
        )
    reasoning_enabled = reasoning_enabled_value
    reasoning_effort = _reasoning_effort(
        payload.get("reasoning_effort", base.get("thinking_effort", "medium"))
    )
    required_reasoning = _required_reasoning_effort(base)
    if required_reasoning is not None and (
        not reasoning_enabled or reasoning_effort != required_reasoning
    ):
        raise ControlError(
            409,
            "formal_reasoning_required",
            f"Selected formal profile requires reasoning effort {required_reasoning}",
        )
    spec = StartSpec(
        base_config=base_relative,
        output_dir=output_relative,
        seed_index_offset=_integer(payload, "seed_index_offset", 0, 2_000_000_000),
        concurrency=_integer(payload, "concurrency", 1, None),
        base_url=validate_base_url(payload.get("base_url")),
        protocol=str(protocol),
        model=model,
        force_model=_boolean(payload, "force_model"),
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
        pricing_route=_pricing_route(payload.get("pricing_route", "manual")),
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
    provider_profile: dict[str, object] = {
        "provider": spec.provider,
        "protocol": spec.protocol,
        "base_url": spec.base_url,
        "model": spec.model,
        "force_model": spec.force_model,
        "reasoning_enabled": spec.reasoning_enabled,
        "reasoning_effort": spec.reasoning_effort,
        "pricing_route": spec.pricing_route,
    }
    effective.update(
        {
            "provider": spec.provider,
            "protocol": spec.protocol,
            "base_url": spec.base_url,
            "model": spec.model,
            "force_model": spec.force_model,
            "discover_models": not spec.force_model,
            "thinking_enabled": spec.reasoning_enabled,
            "thinking_effort": spec.reasoning_effort,
            **{
                key: spec.reasoning_effort
                for key in _STAGE_REASONING_KEYS
            },
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
        "provider_profile": provider_profile,
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
        "provider_profile": provider_profile,
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
        provider_profile=provider_profile,
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

    profile = control.get("provider_profile")
    if not isinstance(profile, Mapping):
        raise ControlError(409, "run_not_trusted", "Provider profile is missing")
    protocol = profile.get("protocol")
    model = profile.get("model")
    force_model = profile.get("force_model")
    reasoning_enabled = profile.get(
        "reasoning_enabled", effective.get("thinking_enabled", False)
    )
    reasoning_effort = profile.get(
        "reasoning_effort", effective.get("thinking_effort", "medium")
    )
    pricing_route = profile.get("pricing_route", "manual")
    try:
        base_url = validate_base_url(profile.get("base_url"))
        safe_pricing_route = _pricing_route(pricing_route)
    except ControlError as error:
        raise ControlError(
            409, "run_not_trusted", "Provider profile is invalid"
        ) from error
    if (
        protocol not in ALLOWED_PROTOCOLS
        or not isinstance(model, str)
        or MODEL_RE.fullmatch(model) is None
        or not isinstance(force_model, bool)
        or not isinstance(reasoning_enabled, bool)
        or reasoning_effort not in REASONING_EFFORTS
        or profile.get("provider") != PROVIDER_BY_PROTOCOL[protocol]
        or effective.get("protocol") != protocol
        or effective.get("base_url") != base_url
        or effective.get("model") != model
        or effective.get("force_model") is not force_model
        or effective.get("thinking_enabled") is not reasoning_enabled
        or effective.get("thinking_effort") != reasoning_effort
    ):
        raise ControlError(409, "run_not_trusted", "Provider profile is invalid")
    provider_profile: dict[str, object] = {
        "provider": PROVIDER_BY_PROTOCOL[protocol],
        "protocol": protocol,
        "base_url": base_url,
        "model": model,
        "force_model": force_model,
        "reasoning_enabled": reasoning_enabled,
        "reasoning_effort": reasoning_effort,
        "pricing_route": safe_pricing_route,
    }
    manifest_profile = manifest.get("provider_profile")
    if manifest_profile is not None and manifest_profile != provider_profile:
        raise ControlError(409, "run_not_trusted", "Provider profile changed")

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
        provider_profile=provider_profile,
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


@dataclass
class FormalJob:
    run_id: str
    concurrency: int
    resume_mode: bool
    config_sha256: str
    execution_lock_sha256: str
    expected_checkpoint_id: str | None
    max_tasks: int | None = None
    process_state: str = "starting"
    process: ProcessLike | None = None
    exit_code: int | None = None
    started_at: str = field(default_factory=_iso)
    finished_at: str | None = None
    stop_requested: bool = False
    last_error_code: str | None = None


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
        self.formal_secret = SecretSlot()
        self.lock = threading.RLock()
        self.probe_lock = threading.Lock()
        self.probe_active = False
        self.job: ManagedJob | None = None
        self.formal_job: FormalJob | None = None
        self.formal_gates: dict[str, object] = {}
        self.formal_execution: dict[str, object] = {}
        self.events: deque[dict[str, object]] = deque(maxlen=100)
        self.closed = False

    def options(self) -> dict[str, object]:
        value = self.policy.options()
        gates = value.get("formal_gates")
        execution = value.get("formal_execution")
        if isinstance(gates, Mapping):
            self.formal_gates = dict(gates)
        if isinstance(execution, Mapping):
            self.formal_execution = dict(execution)
        return value

    def formal_status(self) -> dict[str, object]:
        runtime = self.policy.formal_runtime_status()
        binding = self.policy.formal_local_binding()
        with self.lock:
            if not self.formal_gates:
                self.options()
            gates = dict(self.formal_gates)
            execution = dict(self.formal_execution)
            job = self.formal_job
            active = bool(
                job is not None
                and job.process is not None
                and job.process.poll() is None
            )
            local_binding_matches = bool(
                runtime.get("available") is True
                and binding.get("ready") is True
                and runtime.get("config_sha256") == binding.get("config_sha256")
                and runtime.get("execution_lock_sha256")
                == binding.get("execution_lock_sha256")
            )
            identity_verified = False
            status_reason: str | None = None
            if job is not None and runtime.get("available") is True:
                identity_verified = bool(
                    local_binding_matches
                    and runtime.get("control_run_id") == job.run_id
                    and runtime.get("resume_mode") is job.resume_mode
                    and (
                        job.expected_checkpoint_id is None
                        or runtime.get("checkpoint_id")
                        == job.expected_checkpoint_id
                    )
                    and runtime.get("fresh") is True
                )
                if runtime.get("fresh") is not True:
                    status_reason = "formal_status_stale"
                elif not identity_verified:
                    status_reason = "formal_status_identity_mismatch"
            elif runtime.get("state") in {"running", "starting"}:
                status_reason = "formal_status_historical_unbound"
            process_consistent = bool(
                not active
                or (
                    identity_verified
                    and runtime.get("state")
                    in {"starting", "running", "failed"}
                )
            )
            if active and not process_consistent and status_reason is None:
                status_reason = "formal_process_status_mismatch"
            checkpoint_bound = bool(
                local_binding_matches
                and binding.get("checkpoint_exists") is True
                and isinstance(runtime.get("checkpoint_id"), str)
            )
            checkpoint_terminal = runtime.get("state") in {
                "completed",
                "completed_with_failures",
            }
            gate_reason = gates.get("reason_code") or execution.get("reason_code")
            public_runtime_state = runtime.get("state")
            if job is None and runtime.get("state") in {"running", "starting"}:
                public_runtime_state = "historical_unbound"
            elif active and status_reason == "formal_status_stale":
                public_runtime_state = "stale_status"
            elif active and status_reason is not None:
                public_runtime_state = "untrusted_status"
            status = dict(runtime)
            status.update(
                {
                    "state": public_runtime_state,
                    "schema_version": "anchor.formal-control-public.v1",
                    "target": "formal_swebench_ccswitch",
                    "process_state": (
                        job.process_state if job is not None else "not_started"
                    ),
                    "run_id": job.run_id if job is not None else None,
                    "exit_code": job.exit_code if job is not None else None,
                    "started_at": job.started_at if job is not None else None,
                    "finished_at": job.finished_at if job is not None else None,
                    "credential_loaded": self.formal_secret.configured,
                    "concurrency": job.concurrency if job is not None else 1,
                    "max_tasks": job.max_tasks if job is not None else None,
                    "gates": gates,
                    "reason_code": status_reason or gate_reason,
                    "identity_verified": identity_verified,
                    "local_binding_matches": local_binding_matches,
                    "process_consistent": process_consistent,
                    "telemetry_trusted": bool(
                        identity_verified and process_consistent
                    ),
                    "can_start": bool(
                        gates.get("live_start_allowed") is True
                        and not active
                        and (
                            binding.get("status_exists") is not True
                            or binding.get("failed_startup_rearmable") is True
                        )
                        and binding.get("checkpoint_exists") is not True
                    ),
                    "can_stop": active,
                    "can_continue": bool(
                        gates.get("live_start_allowed") is True
                        and not active
                        and checkpoint_bound
                        and not checkpoint_terminal
                    ),
                    "pause_semantics": "graceful_stop_then_checkpoint_resume",
                    "formal_lifecycle_control_available": True,
                }
            )
            return status

    def start_formal(self, payload: Mapping[str, object], *, resume: bool) -> dict[str, object]:
        required = {"api_key", "concurrency"}
        allowed = required | {"max_tasks"}
        if not required.issubset(payload) or not set(payload).issubset(allowed):
            raise ControlError(
                400,
                "invalid_formal_start",
                "Formal start accepts api_key, concurrency, and optional max_tasks",
            )
        concurrency = _integer(payload, "concurrency", 1, None)
        max_tasks = (
            _integer(payload, "max_tasks", 1, 19008)
            if "max_tasks" in payload
            else None
        )
        credential = validate_api_key(payload.get("api_key"))
        options = self.options()
        gates = options.get("formal_gates")
        reason = (
            gates.get("reason_code")
            if isinstance(gates, Mapping)
            else "formal_gate_unavailable"
        )
        if not isinstance(gates, Mapping) or gates.get("live_start_allowed") is not True:
            credential = ""
            raise ControlError(
                409,
                str(reason or "formal_live_blocked"),
                "Formal LIVE is blocked by the execution-contract gate",
            )
        with self.lock:
            if self.closed:
                raise ControlError(409, "control_closed", "Control plane is closed")
            if self.job is not None and self.job.process is not None and self.job.process.poll() is None:
                raise ControlError(409, "legacy_active", "Legacy shard process is active")
            current = self.formal_job
            if current is not None and current.process is not None and current.process.poll() is None:
                raise ControlError(409, "formal_active", "Formal coordinator is active")
            runtime = self.policy.formal_runtime_status()
            binding = self.policy.formal_local_binding()
            if binding.get("ready") is not True:
                raise ControlError(409, "formal_binding_unavailable", "Formal binding is unavailable")
            local_match = bool(
                runtime.get("available") is True
                and runtime.get("config_sha256") == binding.get("config_sha256")
                and runtime.get("execution_lock_sha256")
                == binding.get("execution_lock_sha256")
            )
            if resume:
                if (
                    binding.get("status_exists") is not True
                    or binding.get("checkpoint_exists") is not True
                    or not local_match
                    or not isinstance(runtime.get("checkpoint_id"), str)
                    or runtime.get("state")
                    in {"completed", "completed_with_failures"}
                ):
                    raise ControlError(
                        409,
                        "formal_resume_binding_invalid",
                        "No matching resumable formal checkpoint is available",
                    )
                expected_checkpoint_id = str(runtime["checkpoint_id"])
            else:
                if (
                    (
                        binding.get("status_exists") is True
                        and binding.get("failed_startup_rearmable") is not True
                    )
                    or binding.get("checkpoint_exists") is True
                ):
                    raise ControlError(
                        409,
                        "formal_checkpoint_exists_use_resume",
                        "A formal checkpoint exists; use Continue",
                    )
                expected_checkpoint_id = None
            self.formal_secret.set(credential)
            credential = ""
            job = FormalJob(
                run_id=f"formal-{uuid4().hex[:16]}",
                concurrency=concurrency,
                resume_mode=resume,
                config_sha256=str(binding["config_sha256"]),
                execution_lock_sha256=str(binding["execution_lock_sha256"]),
                expected_checkpoint_id=expected_checkpoint_id,
                max_tasks=max_tasks,
            )
            self.formal_job = job
            try:
                self._spawn_formal(job)
            except Exception:
                self.formal_secret.clear()
                self.formal_job = None
                raise
            return self.formal_status()

    def stop_formal(self, run_id: object) -> dict[str, object]:
        if not isinstance(run_id, str):
            raise ControlError(400, "invalid_formal_stop", "Formal run ID is required")
        with self.lock:
            job = self.formal_job
            if job is None or not hmac.compare_digest(job.run_id, run_id):
                raise ControlError(409, "formal_run_mismatch", "Formal run ID does not match")
            if job.process is None or job.process.poll() is not None:
                raise ControlError(409, "formal_not_active", "Formal coordinator is not active")
            job.stop_requested = True
            job.process_state = "stopping"
            self.signaler.graceful(job.process)
            return self.formal_status()

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
            legacy_active = bool(
                self.job is not None
                and self.job.process is not None
                and self.job.process.poll() is None
            )
            formal_active = bool(
                self.formal_job is not None
                and self.formal_job.process is not None
                and self.formal_job.process.poll() is None
            )
            if legacy_active or formal_active:
                raise ControlError(
                    409,
                    "active_credential_resident",
                    "Safe-pause the active run before clearing its credential",
                )
            self.secret.clear()
            self.formal_secret.clear()
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
                    "provider_profile": None,
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
                "provider_profile": dict(job.generated.provider_profile),
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
            formal = self.formal_job
            if (
                formal is not None
                and formal.process is not None
                and formal.process.poll() is None
            ):
                formal.stop_requested = True
                formal.process_state = "stopping"
                self.signaler.graceful(formal.process)
            self.secret.clear()
            self.formal_secret.clear()

    def _spawn_formal(self, job: FormalJob) -> None:
        credential = self.formal_secret.reveal()
        script = (
            self.policy.root / "scripts" / "tooling" / "run_swebench_ccswitch.py"
        ).resolve()
        config = (
            self.policy.root
            / "configs"
            / "data"
            / "swebench_five_stage.ccswitch.yaml"
        ).resolve()
        if (
            not script.is_relative_to(self.policy.root)
            or not config.is_relative_to(self.policy.root)
            or not script.is_file()
            or not config.is_file()
        ):
            raise ControlError(
                500,
                "formal_entrypoint_missing",
                "Fixed formal coordinator entrypoint is missing",
            )
        argv = [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--confirm-live",
            "--control-run-id",
            job.run_id,
            "--concurrency",
            str(job.concurrency),
        ]
        if job.resume_mode:
            argv.append("--resume")
        if job.max_tasks is not None:
            argv.extend(("--max-tasks", str(job.max_tasks)))
        environment = os.environ.copy()
        environment["ARK_CODING_API_KEY"] = credential
        try:
            process = self.popen_factory(
                argv,
                cwd=str(self.policy.root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                **self.signaler.popen_group_kwargs(),
            )
        except OSError as error:
            raise ControlError(
                500,
                "formal_spawn_failed",
                "Formal coordinator could not start",
            ) from error
        finally:
            environment.pop("ARK_CODING_API_KEY", None)
            credential = ""
        job.process = process
        job.process_state = "running"
        threading.Thread(
            target=self._watch_formal,
            args=(job.run_id, process),
            daemon=True,
            name=f"anchor-formal-watch-{job.run_id[-8:]}",
        ).start()

    def _watch_formal(self, run_id: str, process: ProcessLike) -> None:
        try:
            return_code = process.wait()
        except (OSError, subprocess.SubprocessError):
            return_code = 1
        with self.lock:
            job = self.formal_job
            if job is None or not hmac.compare_digest(job.run_id, run_id):
                return
            job.exit_code = return_code
            job.finished_at = _iso()
            if job.stop_requested:
                job.process_state = "stopped_checkpoint_resumable"
            elif return_code == 0:
                runtime = self.policy.formal_runtime_status()
                job.process_state = (
                    "stopped_checkpoint_resumable"
                    if runtime.get("state") == "stopped_checkpoint_resumable"
                    else "completed"
                )
            else:
                job.process_state = "exited"
                job.last_error_code = "formal_process_exit"
            self.formal_secret.clear()

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
