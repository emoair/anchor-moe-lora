"""Checkpointed five-stage SWE-bench full-bank coordinator.

The default invocation is a read-only offline preflight.  Network access,
credential lookup, provider processes, git checkouts, and OpenCode sandboxes are
reachable only after the operator passes ``--confirm-live``.  Runtime records
are content-bearing artifacts; stdout/stderr and the checkpoint ledger remain
content-free by design.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.full_bank import (  # noqa: E402
    FullBankConfig,
    preflight_full_bank,
)
from anchor_mvp.swebench.schema import CHAIN_STAGES, canonical_json  # noqa: E402
from anchor_mvp.tooling.config import (  # noqa: E402
    OpenCodeProvider,
    build_opencode_config,
)
from anchor_mvp.tooling.models import AgentExecution  # noqa: E402
from anchor_mvp.tooling.policy import ToolPolicy  # noqa: E402
from anchor_mvp.tooling.route_diagnostics import (  # noqa: E402
    ROUTE_FAILURE_DIAGNOSTIC_NAME,
    ROUTE_STARTUP_ERROR_CODES,
    RouteDiagnosticSource,
    validate_route_failure_diagnostic,
    write_route_failure_diagnostic,
)
from anchor_mvp.tooling.runner import (  # noqa: E402
    AnchorSandboxOptions,
    OpenCodeExecutor,
)
from anchor_mvp.tooling.swebench_execution_v3 import (  # noqa: E402
    DISTILLATION_VALIDATED_CODE_SUFFIXES,
    DISTILLATION_VALIDATED_NON_CODE_SUFFIXES,
    DISTILLATION_VALIDATION_STATE_SCHEMA,
    ExecutionContractError,
    approved_builder_policy,
    candidate_artifact_set_sha256,
    distillation_lineage_sha256,
    distillation_tool_evidence,
    distillation_validation_state_sha256,
    verify_execution_attestation,
    verify_distillation_execution_receipt,
)
from anchor_mvp.tooling.swebench_runtime_v3 import (  # noqa: E402
    issue_distillation_execution_receipt_after_cleanup,
    load_distillation_supervisor_receipt_key,
)

CONFIG_SCHEMA = "anchor.swebench-ccswitch-coordinator.v1"
TASK_SCHEMA = "anchor.swebench-candidate-task.v1"
ORDER_SCHEMA = "anchor.swebench-candidate-work-order.v1"
EVENT_SCHEMA = "anchor.swebench-ccswitch-event.v1"
RUN_MANIFEST_SCHEMA = "anchor.swebench-ccswitch-run-manifest.v1"
STATUS_SCHEMA = "anchor.swebench-ccswitch-status.v2"
TRANSPORT_EVENT_SCHEMA = "anchor.swebench-ccswitch-transport-event.v2"
RESPONSE_ENVELOPE_EVENT_SCHEMA = "anchor.swebench-response-envelope-event.v1"
PROFILE_SCHEMA = "anchor.ccswitch-route-profile.v1"
ROUTE_MANIFEST_SCHEMA = "anchor.ccswitch-route-manifest.v1"
PUBLIC_MANIFEST_SCHEMA = "anchor.swebench-publication-manifest.v1"
MULTILANG_EXECUTION_ATTESTATION_SCHEMA = "anchor.multilang-execution-attestation.v1"
MULTILANG_EXECUTION_CONTRACT_VERSION = "anchor.execution-tool-contract.v3"
EXECUTION_ATTESTATION_MISSING = "multilang_execution_attestation_missing"
EXECUTION_ATTESTATION_INVALID = "multilang_execution_attestation_invalid"
EXECUTION_ATTESTATION_UNVERIFIED = "multilang_execution_attestation_unverified"
EXPECTED_STAGES = tuple(CHAIN_STAGES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,80}$")
_CONTRACT_RETRY_SCHEMA = "anchor.swebench-contract-retry.v1"
_BUILDER_POLICY_PUBLIC_ERROR_CODES = {
    "builder_policy_input_invalid": "v3_builder_policy_input_invalid",
    "builder_policy_decision_invalid": "v3_builder_policy_decision_invalid",
    "builder_policy_proposal_invalid": "v3_builder_policy_proposal_invalid",
    "builder_policy_tool_outside_global_allowlist": "v3_builder_policy_tool_invalid",
    "builder_policy_bash_input_invalid": "v3_builder_policy_bash_input_invalid",
    "builder_policy_bash_command_invalid": "v3_builder_policy_bash_command_invalid",
    "builder_policy_proposal_binding_ambiguous": (
        "v3_builder_policy_proposal_binding_ambiguous"
    ),
    "builder_policy_coverage_invalid": "v3_builder_policy_coverage_invalid",
}
_BUILDER_POLICY_FALLBACK_ERROR_CODE = "v3_builder_policy_contract_invalid"
_CONTRACT_RETRY_GUIDANCE = {
    "planner_structure_invalid": (
        "Return the exact planner JSON contract with non-empty work_items and "
        "tool_proposals arrays."
    ),
    "planner_proposal_invalid": (
        "Replace malformed proposals with unique proposal_id, allowed tool, and "
        "object input fields."
    ),
    "planner_tool_invalid": (
        "Use only the explicitly allowed planner tools and preserve unique proposal IDs."
    ),
    "planner_bash_binding_invalid": (
        "For bash, use exactly one proven anchor-validate command in the input object."
    ),
    "planner_duplicate_family": (
        "Emit exactly one proposal total from the shared edit/write/apply_patch "
        "permission family. The planner proposes and never approves tools."
    ),
    "planner_write_required": (
        "Include exactly one proposal from the shared edit/write/apply_patch permission "
        "family."
    ),
    "tool_policy_structure_invalid": (
        "Return the exact tool-policy JSON contract with a non-empty decisions array."
    ),
    "tool_policy_decision_invalid": (
        "Emit each planner proposal_id exactly once with only APPROVE or DENY."
    ),
    "tool_policy_missing_decision": (
        "Copy every current planner proposal_id exactly once without additions or omissions."
    ),
    "tool_policy_write_required": (
        "Approve exactly one available proposal in the shared edit/write/apply_patch "
        "permission family."
    ),
    "tool_policy_rebind_planner": (
        "Re-evaluate the corrected planner and copy every current proposal_id exactly once."
    ),
}
_RESPONSE_STATUSES = frozenset(
    {"completed", "incomplete", "failed", "cancelled", "canceled", "error"}
)
_MAX_OUTPUT_BUDGET_REASONS = frozenset(
    {
        "length",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "output_token_limit",
        "token_limit",
    }
)
_RESPONSE_INCOMPLETE_REASONS = frozenset(
    {"content_filter", "max_output_tokens", "none", "unknown"}
)
_RESPONSE_OUTPUT_ITEM_TYPES = frozenset(
    {
        "code_interpreter_call",
        "computer_call",
        "custom_tool_call",
        "file_search_call",
        "function_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_approval_request",
        "mcp_call",
        "message",
        "reasoning",
        "web_search_call",
    }
)
_RESPONSE_CONTENT_PART_TYPES = frozenset(
    {
        "input_file",
        "input_image",
        "input_text",
        "output_text",
        "refusal",
        "summary_text",
    }
)
_RESPONSE_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
)
_RESPONSE_USAGE_LIMIT = 1_000_000_000_000
_RESPONSE_TYPE_COUNT_LIMIT = 1_000_000
_SEMANTIC_RETRY_MAX_OUTPUT_TOKENS = 65_536
PROVIDER_NETWORK_MODES = frozenset({"direct", "proxy", "inherit"})
PRE_PROVIDER_STARTUP_ERROR_CODES = ROUTE_STARTUP_ERROR_CODES | frozenset(
    {
        "backend_startup_failed",
        "live_credential_missing",
        "opencode_behavioral_attestation_failed",
        "sandbox_validator_artifact_mismatch",
        "sandbox_validator_image_digest_invalid",
        "sandbox_validator_version_mismatch",
        "v3_runtime_adapter_startup_failed",
        "wsl_probed_host_discovery_failed",
    }
)
_REPOSITORY = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})/[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})$"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIRECT_OPENER = build_opener(ProxyHandler({}))
TRAIN_SANDBOX_NODE_IMAGE = (
    "docker.io/library/node@sha256:"
    "a149cd71dccd68704a07d4e4ca3e610c27301852b0f556865cfdb6e2856f8bed"
)
TRAIN_SANDBOX_PYTHON_IMAGE = (
    "docker.io/library/python@sha256:"
    "5c5e0496473632460861e691a03cce82205c38556d9c0be4e6cb5915380f1e50"
)
TRAIN_SANDBOX_CONTAINERFILE = (
    PROJECT_ROOT / "configs" / "tooling" / "train-sandbox.Containerfile"
)
TRAIN_SANDBOX_VALIDATOR = (
    PROJECT_ROOT / "scripts" / "tooling" / "train_sandbox_validate.py"
)
TRAIN_SANDBOX_VALIDATOR_SCHEMA = "anchor.train-sandbox-validation.v1"
TRAIN_SANDBOX_VALIDATOR_VERSION = "1.0.1"
TRAIN_SANDBOX_IMAGE_FAMILY = "python-bookworm+node22-bookworm+anchor-validator"
TRAIN_SANDBOX_IMAGE_REFERENCE = (
    "localhost/anchor-train-sandbox@sha256:"
    "a8a183c8a59d4c6a376ea6551ef14dabe73573bb739a7f045fe6180f30bd9671"
)
TRAIN_SANDBOX_IMAGE_ID = (
    "a7411e7ae2cfabf3f87e24279138a1a158f068016bd952fe3661634d7ea02924"
)


class CoordinatorError(RuntimeError):
    """Expected fail-closed coordinator error with a content-free code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoordinatorError(f"invalid_{label}")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CoordinatorError(f"invalid_{label}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CoordinatorError(f"invalid_{label}")
    return value


def _project_path(root: Path, value: object, label: str) -> Path:
    candidate = (root / _text(value, label)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CoordinatorError(f"path_escape_{label}") from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _progress_rates(
    *,
    completed_tasks: int,
    completed_task_baseline: int,
    expected_tasks: int,
    elapsed_seconds: float,
    state: str,
) -> tuple[int, float, float | None]:
    """Return attempt-local completions/rate and total-work ETA.

    ``completed_tasks`` is cumulative during a resumed scan because checkpointed
    terminal tasks are replayed through ``run_chain``.  A rate based on that
    cumulative value would report the old run's work as if this process had just
    completed it.  Keep the public total for remaining-work accounting, but
    subtract the checkpoint baseline for this attempt's throughput.
    """

    current_run_completed = max(0, completed_tasks - completed_task_baseline)
    elapsed = max(elapsed_seconds, 0.001)
    tasks_per_minute = current_run_completed * 60.0 / elapsed
    eta_seconds = (
        max(0.0, (expected_tasks - completed_tasks) * 60.0 / tasks_per_minute)
        if state == "running" and tasks_per_minute > 0
        else None
    )
    return current_run_completed, tasks_per_minute, eta_seconds


def _terminal_state_for_run(
    *,
    max_tasks: int | None,
    submitted_tasks: int,
    expected_tasks: int,
    failed_tasks: int,
) -> str:
    """Keep a successful capped pilot resumable instead of sealing the bank.

    ``--max-tasks`` is an operator-controlled checkpoint boundary, not proof that
    the full bank completed.  Its prefix therefore remains explicitly resumable
    until the expected task count has actually been submitted; failed or partial
    tasks can be retried while hash-bound completed stages stay immutable.
    """

    if max_tasks is not None and submitted_tasks < expected_tasks:
        return "stopped_checkpoint_resumable"
    return "completed" if failed_tasks == 0 else "completed_with_failures"


def _checkpoint_identity(config: "CoordinatorConfig") -> dict[str, str]:
    config_sha256 = _sha256_file(config.config_path)
    source_bank_manifest_sha256 = _sha256_file(config.bank_manifest)
    try:
        output_dir = config.runtime.output_dir.relative_to(PROJECT_ROOT).as_posix()
        bank_manifest = config.bank_manifest.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise CoordinatorError("checkpoint_identity_path_escape") from exc
    binding = {
        "config_sha256": config_sha256,
        "execution_lock_sha256": config.execution_contract.lock_sha256,
        "source_bank_manifest_sha256": source_bank_manifest_sha256,
        "output_dir": output_dir,
        "bank_manifest": bank_manifest,
        "expected_tasks": config.expected_tasks,
        "stages": list(EXPECTED_STAGES),
    }
    return {
        "checkpoint_id": sha256(canonical_json(binding).encode("utf-8")).hexdigest(),
        "config_sha256": config_sha256,
        "execution_lock_sha256": config.execution_contract.lock_sha256,
        "source_bank_manifest_sha256": source_bank_manifest_sha256,
    }


def _zero_int_mapping(value: object, keys: Iterable[str]) -> bool:
    expected = set(keys)
    return (
        isinstance(value, Mapping)
        and set(value) == expected
        and all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and value.get(key) == 0
            for key in expected
        )
    )


def _rearmable_failed_start(
    output: Path,
) -> tuple[Mapping[str, Any], bytes] | None:
    """Return one exact zero-work startup failure, otherwise fail closed.

    The status may bind an older coordinator config or execution lock. A stale
    binding is safe to archive only because every work, request, token, ledger,
    and receipt signal is independently required to be absent or zero.
    """

    status_path = output / "status.json"
    if (
        not output.is_dir()
        or output.is_symlink()
        or not status_path.is_file()
        or status_path.is_symlink()
    ):
        return None
    try:
        raw = status_path.read_bytes()
        if not raw or len(raw) > 64 * 1024:
            return None
        status = _mapping(json.loads(raw.decode("utf-8")), "failed_start_status")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, CoordinatorError):
        return None

    expected_status_keys = {
        "schema_version",
        "control_run_id",
        "checkpoint_id",
        "config_sha256",
        "execution_lock_sha256",
        "source_bank_manifest_sha256",
        "resume_mode",
        "state",
        "submitted_tasks",
        "active_tasks",
        "completed_tasks",
        "completed_task_baseline",
        "current_run_completed_tasks",
        "expected_tasks",
        "counts",
        "stage_counts",
        "failure_counts",
        "requests",
        "request_failure_counts",
        "tokens",
        "elapsed_seconds",
        "tasks_per_minute",
        "provider_output_tokens_per_second",
        "eta_seconds",
        "updated_at",
        "last_error_code",
        "content_free",
    }
    code = status.get("last_error_code")
    if (
        set(status) != expected_status_keys
        or status.get("schema_version") != STATUS_SCHEMA
        or status.get("content_free") is not True
        or status.get("state") != "failed"
        or status.get("resume_mode") is not False
        or not isinstance(code, str)
        or not _ERROR_CODE.fullmatch(code)
        or not _CONTROL_RUN_ID.fullmatch(str(status.get("control_run_id", "")))
        or status.get("expected_tasks") != 19008
    ):
        return None
    updated_at = status.get("updated_at")
    if not isinstance(updated_at, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", updated_at
    ):
        return None
    if not all(
        isinstance(status.get(name), str) and _SHA256.fullmatch(str(status.get(name)))
        for name in (
            "checkpoint_id",
            "config_sha256",
            "execution_lock_sha256",
            "source_bank_manifest_sha256",
        )
    ):
        return None
    if any(
        not isinstance(status.get(name), int)
        or isinstance(status.get(name), bool)
        or status.get(name) != 0
        for name in (
            "submitted_tasks",
            "active_tasks",
            "completed_tasks",
            "completed_task_baseline",
            "current_run_completed_tasks",
        )
    ):
        return None
    if not _zero_int_mapping(status.get("counts"), ("completed", "blocked", "failed")):
        return None
    if not _zero_int_mapping(status.get("stage_counts"), EXPECTED_STAGES):
        return None
    if not _zero_int_mapping(
        status.get("requests"),
        (
            "provider_requests",
            "provider_successes",
            "provider_failures",
            "retry_attempts",
        ),
    ):
        return None
    if not _zero_int_mapping(
        status.get("tokens"),
        ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"),
    ):
        return None
    failure_counts = status.get("failure_counts")
    rates = (
        status.get("elapsed_seconds"),
        status.get("tasks_per_minute"),
        status.get("provider_output_tokens_per_second"),
    )
    if (
        status.get("request_failure_counts") != {}
        or not isinstance(failure_counts, Mapping)
        or set(failure_counts) != {code}
        or not isinstance(failure_counts.get(code), int)
        or isinstance(failure_counts.get(code), bool)
        or failure_counts.get(code) != 1
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0
            for value in rates
        )
        or status.get("eta_seconds") is not None
    ):
        return None

    forbidden = (
        "checkpoint.events.jsonl",
        "manifest.json",
        "content-records",
        "usage.events.jsonl",
        "transport.events.jsonl",
        "system-private",
    )
    if any((output / name).exists() for name in forbidden):
        return None
    diagnostic_path = output / ROUTE_FAILURE_DIAGNOSTIC_NAME
    try:
        for path in output.rglob("*"):
            if path.is_symlink():
                return None
            if path.is_file() and path not in {status_path, diagnostic_path}:
                return None
        if diagnostic_path.exists():
            if not diagnostic_path.is_file() or diagnostic_path.is_symlink():
                return None
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if code in ROUTE_STARTUP_ERROR_CODES:
                validate_route_failure_diagnostic(
                    diagnostic,
                    expected_startup_error_code=code,
                )
            else:
                validate_route_failure_diagnostic(
                    diagnostic,
                    expected_classified_error_code=code,
                )
        elif code not in PRE_PROVIDER_STARTUP_ERROR_CODES:
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return status, raw


def can_rearm_failed_start(output: Path) -> bool:
    """Read-only control-plane predicate for a fresh zero-work pilot retry."""

    return _rearmable_failed_start(output) is not None


def _archive_rearmable_failed_start(output: Path) -> Path:
    inspected = _rearmable_failed_start(output)
    if inspected is None:
        raise CoordinatorError("checkpoint_exists_use_resume")
    status, raw = inspected
    archive_root = output.with_name(output.name + ".failed-startups")
    if archive_root.exists() and (
        not archive_root.is_dir() or archive_root.is_symlink()
    ):
        raise CoordinatorError("failed_startup_archive_invalid")
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
        prefix = f"{str(status['checkpoint_id'])[:12]}-{sha256(raw).hexdigest()[:12]}-"
        reservation = Path(tempfile.mkdtemp(prefix=prefix, dir=archive_root))
        target = reservation / "attempt"
        output.rename(target)
    except OSError as exc:
        if "reservation" in locals():
            try:
                reservation.rmdir()
            except OSError:
                pass
        raise CoordinatorError("failed_startup_archive_failed") from exc
    return target


def _validate_checkpoint_mode(
    config: "CoordinatorConfig",
    *,
    resume: bool,
    identity: Mapping[str, str],
) -> None:
    output = config.runtime.output_dir
    status_path = output / "status.json"
    events_path = output / "checkpoint.events.jsonl"
    manifest_path = output / "manifest.json"
    if not resume:
        if status_path.exists() and can_rearm_failed_start(output):
            _archive_rearmable_failed_start(output)
        if (
            status_path.exists()
            or events_path.exists()
            or manifest_path.exists()
            or (output.is_dir() and any(output.iterdir()))
        ):
            raise CoordinatorError("checkpoint_exists_use_resume")
        return
    if not status_path.is_file() or not events_path.is_file():
        raise CoordinatorError("resume_checkpoint_missing")
    try:
        status = _mapping(
            json.loads(status_path.read_text(encoding="utf-8")),
            "resume_status",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinatorError("resume_status_invalid") from exc
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("content_free") is not True
    ):
        raise CoordinatorError("resume_status_invalid")
    for name in (
        "checkpoint_id",
        "config_sha256",
        "execution_lock_sha256",
        "source_bank_manifest_sha256",
    ):
        if status.get(name) != identity[name]:
            raise CoordinatorError("resume_checkpoint_binding_mismatch")
    if status.get("state") in {"completed", "completed_with_failures"}:
        raise CoordinatorError("resume_checkpoint_already_terminal")


@dataclass(frozen=True)
class RouteSpec:
    alias: str
    profile: Path
    port: int


@dataclass(frozen=True)
class RuntimeSpec:
    output_dir: Path
    repository_cache: Path
    workspace_root: Path
    concurrency: int
    max_revisions: int
    request_timeout_seconds: int
    sandbox_timeout_seconds: int
    max_output_tokens: int
    max_retries: int
    provider_network_mode: str
    wsl_distro: str
    sandbox_supervisor: str
    sandbox_memory: str
    sandbox_cpus: str
    sandbox_pids: int
    sandbox_route_visibility: str
    retain_router_state: bool


@dataclass(frozen=True)
class ExecutionContractSpec:
    attestation: Path
    lock: Path
    lock_sha256: str
    required_schema: str
    required_tool_contract_version: str


@dataclass(frozen=True)
class CoordinatorConfig:
    config_path: Path
    bank_root: Path
    bank_manifest: Path
    tasks_glob: str
    work_orders_glob: str
    expected_tasks: int
    expected_work_orders: int
    required_split: str
    full_bank_config: Path
    opencode_bundle_manifest: Path
    opencode_patch_manifest: Path
    opencode_windows_binary: Path
    opencode_linux_binary: Path
    ccswitch_manifest: Path
    ccswitch_launcher: Path
    execution_contract: ExecutionContractSpec
    routes: Mapping[str, RouteSpec]
    runtime: RuntimeSpec

    @classmethod
    def load(cls, path: Path) -> "CoordinatorConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = PROJECT_ROOT.resolve()
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "bank",
            "components",
            "execution_contract",
            "routes",
            "runtime",
        }:
            raise CoordinatorError("invalid_config_root")
        if raw.get("schema_version") != CONFIG_SCHEMA:
            raise CoordinatorError("unsupported_config_schema")
        bank = _mapping(raw.get("bank"), "bank")
        if set(bank) != {
            "root",
            "manifest",
            "tasks_glob",
            "work_orders_glob",
            "expected_tasks",
            "expected_work_orders",
            "required_split",
        }:
            raise CoordinatorError("invalid_bank_config")
        components = _mapping(raw.get("components"), "components")
        if set(components) != {
            "full_bank_config",
            "opencode_bundle_manifest",
            "opencode_patch_manifest",
            "opencode_windows_binary",
            "opencode_linux_binary",
            "ccswitch_manifest",
            "ccswitch_launcher",
        }:
            raise CoordinatorError("invalid_component_config")
        execution_raw = _mapping(raw.get("execution_contract"), "execution_contract")
        if set(execution_raw) != {
            "attestation",
            "lock",
            "lock_sha256",
            "required_schema",
            "required_tool_contract_version",
        }:
            raise CoordinatorError("invalid_execution_contract_config")
        required_schema = _text(
            execution_raw.get("required_schema"),
            "execution_contract_required_schema",
        )
        required_tool_contract_version = _text(
            execution_raw.get("required_tool_contract_version"),
            "execution_contract_required_tool_contract_version",
        )
        lock_sha256 = _text(
            execution_raw.get("lock_sha256"), "execution_contract_lock_sha256"
        )
        if not _SHA256.fullmatch(lock_sha256):
            raise CoordinatorError("invalid_execution_contract_lock_sha256")
        if (
            required_schema != MULTILANG_EXECUTION_ATTESTATION_SCHEMA
            or required_tool_contract_version != MULTILANG_EXECUTION_CONTRACT_VERSION
        ):
            raise CoordinatorError("unsupported_execution_contract_config")
        routes_raw = _mapping(raw.get("routes"), "routes")
        if set(routes_raw) != {"glm52_max", "kimi_k3_max"}:
            raise CoordinatorError("invalid_route_aliases")
        routes: dict[str, RouteSpec] = {}
        ports: set[int] = set()
        for alias, value in routes_raw.items():
            route = _mapping(value, "route")
            if set(route) != {"profile", "port"}:
                raise CoordinatorError("invalid_route_config")
            port = _positive_int(route.get("port"), "route_port")
            if port > 65535 or port in ports:
                raise CoordinatorError("invalid_route_port")
            ports.add(port)
            routes[str(alias)] = RouteSpec(
                alias=str(alias),
                profile=_project_path(root, route.get("profile"), "route_profile"),
                port=port,
            )
        runtime = _mapping(raw.get("runtime"), "runtime")
        expected_runtime = {
            "output_dir",
            "repository_cache",
            "workspace_root",
            "concurrency",
            "max_revisions",
            "request_timeout_seconds",
            "sandbox_timeout_seconds",
            "max_output_tokens",
            "max_retries",
            "provider_network_mode",
            "wsl_distro",
            "sandbox_supervisor",
            "sandbox_memory",
            "sandbox_cpus",
            "sandbox_pids",
            "sandbox_route_visibility",
            "retain_router_state",
        }
        if set(runtime) != expected_runtime:
            raise CoordinatorError("invalid_runtime_config")
        retain = runtime.get("retain_router_state")
        if not isinstance(retain, bool):
            raise CoordinatorError("invalid_retain_router_state")
        visibility = _text(
            runtime.get("sandbox_route_visibility"), "sandbox_route_visibility"
        )
        if visibility != "wsl-probed-host":
            raise CoordinatorError("unsupported_sandbox_route_visibility")
        provider_network_mode = _text(
            runtime.get("provider_network_mode"), "provider_network_mode"
        )
        if provider_network_mode not in PROVIDER_NETWORK_MODES:
            raise CoordinatorError("invalid_provider_network_mode")
        runtime_spec = RuntimeSpec(
            output_dir=_project_path(root, runtime.get("output_dir"), "output_dir"),
            repository_cache=_project_path(
                root, runtime.get("repository_cache"), "repository_cache"
            ),
            workspace_root=_project_path(
                root, runtime.get("workspace_root"), "workspace_root"
            ),
            concurrency=_positive_int(runtime.get("concurrency"), "concurrency"),
            max_revisions=_positive_int(runtime.get("max_revisions"), "max_revisions"),
            request_timeout_seconds=_positive_int(
                runtime.get("request_timeout_seconds"), "request_timeout_seconds"
            ),
            sandbox_timeout_seconds=_positive_int(
                runtime.get("sandbox_timeout_seconds"), "sandbox_timeout_seconds"
            ),
            max_output_tokens=_positive_int(
                runtime.get("max_output_tokens"), "max_output_tokens"
            ),
            max_retries=_positive_int(runtime.get("max_retries"), "max_retries"),
            provider_network_mode=provider_network_mode,
            wsl_distro=_text(runtime.get("wsl_distro"), "wsl_distro"),
            sandbox_supervisor=_text(
                runtime.get("sandbox_supervisor"), "sandbox_supervisor"
            ),
            sandbox_memory=_text(runtime.get("sandbox_memory"), "sandbox_memory"),
            sandbox_cpus=_text(runtime.get("sandbox_cpus"), "sandbox_cpus"),
            sandbox_pids=_positive_int(runtime.get("sandbox_pids"), "sandbox_pids"),
            sandbox_route_visibility=visibility,
            retain_router_state=retain,
        )
        return cls(
            config_path=path.resolve(),
            bank_root=_project_path(root, bank.get("root"), "bank_root"),
            bank_manifest=_project_path(root, bank.get("manifest"), "bank_manifest"),
            tasks_glob=_text(bank.get("tasks_glob"), "tasks_glob"),
            work_orders_glob=_text(bank.get("work_orders_glob"), "work_orders_glob"),
            expected_tasks=_positive_int(bank.get("expected_tasks"), "expected_tasks"),
            expected_work_orders=_positive_int(
                bank.get("expected_work_orders"), "expected_work_orders"
            ),
            required_split=_text(bank.get("required_split"), "required_split"),
            full_bank_config=_project_path(
                root, components.get("full_bank_config"), "full_bank_config"
            ),
            opencode_bundle_manifest=_project_path(
                root,
                components.get("opencode_bundle_manifest"),
                "opencode_bundle_manifest",
            ),
            opencode_patch_manifest=_project_path(
                root,
                components.get("opencode_patch_manifest"),
                "opencode_patch_manifest",
            ),
            opencode_windows_binary=_project_path(
                root,
                components.get("opencode_windows_binary"),
                "opencode_windows_binary",
            ),
            opencode_linux_binary=_project_path(
                root,
                components.get("opencode_linux_binary"),
                "opencode_linux_binary",
            ),
            ccswitch_manifest=_project_path(
                root, components.get("ccswitch_manifest"), "ccswitch_manifest"
            ),
            ccswitch_launcher=_project_path(
                root, components.get("ccswitch_launcher"), "ccswitch_launcher"
            ),
            execution_contract=ExecutionContractSpec(
                attestation=_project_path(
                    root,
                    execution_raw.get("attestation"),
                    "execution_contract_attestation",
                ),
                lock=_project_path(
                    root,
                    execution_raw.get("lock"),
                    "execution_contract_lock",
                ),
                lock_sha256=lock_sha256,
                required_schema=required_schema,
                required_tool_contract_version=required_tool_contract_version,
            ),
            routes=routes,
            runtime=runtime_spec,
        )


@dataclass(frozen=True)
class TaskChain:
    task: Mapping[str, Any]
    orders: tuple[Mapping[str, Any], ...]
    source_bank_manifest_sha256: str = ""
    candidate_task_artifact_path: str = ""
    candidate_task_artifact_sha256: str = ""
    candidate_work_order_artifacts: tuple[tuple[str, str], ...] = ()

    @property
    def task_id(self) -> str:
        return str(self.task["task_id"])

    @property
    def candidate_work_order_artifacts_sha256(self) -> str:
        try:
            return candidate_artifact_set_sha256(
                [
                    {"path": path, "sha256": digest}
                    for path, digest in self.candidate_work_order_artifacts
                ]
            )
        except ExecutionContractError as exc:
            raise CoordinatorError("candidate_artifact_binding_invalid") from exc


@dataclass
class GenericWorkspaceHandle:
    task_id: str
    instance_id: str
    repo: str
    base_commit: str
    workspace: Path
    image_digest: str
    image_id_sha256: str
    validation_capabilities: tuple[str, ...]
    last_sample_id: str | None = None
    last_executor: OpenCodeExecutor | None = None


@dataclass(frozen=True)
class PendingDistillationReceipt:
    bindings: Mapping[str, Any]
    final_patch: bytes
    builder_output: Mapping[str, Any]


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CoordinatorError("invalid_bank_jsonl") from exc
            if not isinstance(value, Mapping):
                raise CoordinatorError("invalid_bank_row")
            yield value


def _validate_task(task: Mapping[str, Any], config: CoordinatorConfig) -> None:
    if task.get("schema_version") != TASK_SCHEMA:
        raise CoordinatorError("task_schema_mismatch")
    task_id = str(task.get("task_id", ""))
    if not task_id.startswith("swe-full-v1:") or not _SHA256.fullmatch(
        task_id.removeprefix("swe-full-v1:")
    ):
        raise CoordinatorError("task_id_invalid")
    source = _mapping(task.get("source"), "task_source")
    if source.get("split") != config.required_split:
        raise CoordinatorError("non_train_task_rejected")
    public_input = _mapping(task.get("public_input"), "public_input")
    statement = public_input.get("problem_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise CoordinatorError("problem_statement_missing")
    chain = _mapping(task.get("chain_contract"), "chain_contract")
    if (
        tuple(chain.get("stages", ())) != EXPECTED_STAGES
        or chain.get("dependency_order") != "strict"
        or chain.get("real_sandbox_required_for_builder") is not True
    ):
        raise CoordinatorError("task_chain_contract_mismatch")
    route = _mapping(task.get("routing_contract"), "routing_contract")
    if route.get("reasoning_effort") != "max":
        raise CoordinatorError("task_reasoning_not_max")
    providers = _mapping(route.get("providers_by_stage"), "providers_by_stage")
    if set(providers) != set(EXPECTED_STAGES):
        raise CoordinatorError("task_stage_routes_mismatch")
    if any(str(value) not in config.routes for value in providers.values()):
        raise CoordinatorError("task_unknown_route")


def _validate_orders(
    task: Mapping[str, Any], orders: Sequence[Mapping[str, Any]]
) -> None:
    if len(orders) != len(EXPECTED_STAGES):
        raise CoordinatorError("work_order_cardinality_mismatch")
    previous = str(task["task_id"])
    providers = _mapping(
        _mapping(task.get("routing_contract"), "routing_contract").get(
            "providers_by_stage"
        ),
        "providers_by_stage",
    )
    for stage, order in zip(EXPECTED_STAGES, orders):
        if (
            order.get("schema_version") != ORDER_SCHEMA
            or order.get("task_id") != task["task_id"]
            or order.get("stage") != stage
            or order.get("upstream_record_ids") != [previous]
            or order.get("reasoning_effort") != "max"
            or order.get("provider_alias") != providers[stage]
        ):
            raise CoordinatorError("work_order_dependency_mismatch")
        record_id = str(order.get("record_id", ""))
        if not record_id.startswith("swe-full-stage-v1:"):
            raise CoordinatorError("work_order_id_invalid")
        previous = record_id


def _publication_inventory(
    config: CoordinatorConfig,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    """Load the exact audited public-bank inventory without trusting flags alone."""

    try:
        raw = config.bank_manifest.read_bytes()
        manifest_sha = sha256(raw).hexdigest()
        manifest = _mapping(json.loads(raw.decode("utf-8")), "public_manifest")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinatorError("public_bank_manifest_invalid") from exc
    if (
        (
            expected_manifest_sha256 is not None
            and manifest_sha != expected_manifest_sha256
        )
        or manifest.get("schema_version") != PUBLIC_MANIFEST_SCHEMA
        or manifest.get("train_only") is not True
        or manifest.get("source_split") != "train"
        or manifest.get("publication_ready") is not True
    ):
        raise CoordinatorError("public_bank_manifest_invalid")
    values = manifest.get("files")
    if not isinstance(values, list) or not values:
        raise CoordinatorError("public_bank_inventory_invalid")
    inventory: dict[str, Mapping[str, Any]] = {}
    for raw_item in values:
        item = _mapping(raw_item, "public_bank_inventory_item")
        relative = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        records = item.get("records")
        if (
            set(item) != {"path", "sha256", "bytes", "records"}
            or not isinstance(relative, str)
            or relative in inventory
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or isinstance(records, bool)
            or not isinstance(records, int)
            or records < 0
        ):
            raise CoordinatorError("public_bank_inventory_invalid")
        inventory[relative] = item
    return manifest_sha, inventory


def _read_bound_jsonl(
    path: Path,
    *,
    bank_root: Path,
    inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    try:
        relative = path.resolve().relative_to(bank_root.resolve()).as_posix()
    except ValueError as exc:
        raise CoordinatorError("public_bank_artifact_path_escape") from exc
    binding = inventory.get(relative)
    if binding is None:
        raise CoordinatorError("public_bank_artifact_unbound")
    try:
        raw = path.read_bytes()
        text_value = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CoordinatorError("public_bank_artifact_invalid") from exc
    if sha256(raw).hexdigest() != binding.get("sha256") or len(raw) != binding.get(
        "bytes"
    ):
        raise CoordinatorError("public_bank_artifact_hash_mismatch")
    rows: list[Mapping[str, Any]] = []
    try:
        for line in text_value.split("\n"):
            if not line:
                continue
            rows.append(_mapping(json.loads(line), "public_bank_row"))
    except (json.JSONDecodeError, CoordinatorError) as exc:
        raise CoordinatorError("public_bank_artifact_invalid") from exc
    if len(rows) != binding.get("records"):
        raise CoordinatorError("public_bank_artifact_record_count_mismatch")
    return tuple(rows)


def iter_task_chains(
    config: CoordinatorConfig,
    *,
    expected_manifest_sha256: str | None = None,
) -> Iterable[TaskChain]:
    manifest_sha, inventory = _publication_inventory(
        config,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    task_paths = sorted(config.bank_root.glob(config.tasks_glob))
    order_paths = sorted(config.bank_root.glob(config.work_orders_glob))
    if not task_paths or len(task_paths) != len(order_paths):
        raise CoordinatorError("bank_shard_pairing_failed")
    inventory_task_paths = {
        relative for relative in inventory if Path(relative).match(config.tasks_glob)
    }
    inventory_order_paths = {
        relative
        for relative in inventory
        if Path(relative).match(config.work_orders_glob)
    }
    actual_task_paths = {
        path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        for path in task_paths
    }
    actual_order_paths = {
        path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        for path in order_paths
    }
    if (
        inventory_task_paths != actual_task_paths
        or inventory_order_paths != actual_order_paths
    ):
        raise CoordinatorError("public_bank_inventory_path_mismatch")
    for task_path, order_path in zip(task_paths, order_paths):
        task_relative = (
            task_path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        )
        order_relative = (
            order_path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        )
        task_binding = inventory[task_relative]
        order_binding = inventory[order_relative]
        order_groups: dict[str, list[Mapping[str, Any]]] = {}
        for order in _read_bound_jsonl(
            order_path,
            bank_root=config.bank_root,
            inventory=inventory,
        ):
            order_groups.setdefault(str(order.get("task_id", "")), []).append(order)
        for task in _read_bound_jsonl(
            task_path,
            bank_root=config.bank_root,
            inventory=inventory,
        ):
            _validate_task(task, config)
            task_id = str(task["task_id"])
            orders = tuple(order_groups.pop(task_id, ()))
            _validate_orders(task, orders)
            yield TaskChain(
                task=task,
                orders=orders,
                source_bank_manifest_sha256=manifest_sha,
                candidate_task_artifact_path=task_relative,
                candidate_task_artifact_sha256=str(task_binding["sha256"]),
                candidate_work_order_artifacts=(
                    (order_relative, str(order_binding["sha256"])),
                ),
            )
        if order_groups:
            raise CoordinatorError("orphan_work_orders")


def _validate_profile(path: Path, alias: str) -> Mapping[str, Any]:
    profile = _mapping(json.loads(path.read_text(encoding="utf-8")), "profile")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise CoordinatorError("route_profile_schema_mismatch")
    if _mapping(profile.get("reasoning"), "reasoning").get("effort") != "max":
        raise CoordinatorError("route_profile_reasoning_not_max")
    key_env = str(profile.get("key_env", ""))
    if not _ENV_NAME.fullmatch(key_env):
        raise CoordinatorError("route_profile_key_env_invalid")
    selection = _mapping(profile.get("model_selection"), "model_selection")
    if selection.get("force_manual_model") is not True:
        raise CoordinatorError("formal_route_model_not_forced")
    expected_profile = {
        "glm52_max": "glm-5.2-max",
        "kimi_k3_max": "kimi-k3-max",
    }.get(alias)
    if profile.get("profile_id") != expected_profile:
        raise CoordinatorError("route_profile_alias_mismatch")
    return profile


def _execution_contract_gate(config: CoordinatorConfig) -> dict[str, Any]:
    """Re-probe and verify the locked v3 execution environment fail-closed."""

    path = config.execution_contract.attestation
    try:
        display_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # CoordinatorConfig.load never permits this in production.  Keeping the
        # gate false still makes direct unit construction safe and observable.
        display_path = str(path)
    metadata: dict[str, Any] = {
        "attestation_path": display_path,
        "lock_path": config.execution_contract.lock.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "lock_sha256": config.execution_contract.lock_sha256,
        "required_schema": config.execution_contract.required_schema,
        "required_tool_contract_version": (
            config.execution_contract.required_tool_contract_version
        ),
        "observed_schema": None,
    }
    verification = verify_execution_attestation(
        PROJECT_ROOT,
        path,
        config.execution_contract.lock,
        expected_lock_sha256=config.execution_contract.lock_sha256,
    )
    probe = verification.get("current_probe")
    if isinstance(probe, Mapping):
        observed_schema = probe.get("schema_version")
        metadata["observed_schema"] = (
            observed_schema if isinstance(observed_schema, str) else None
        )
    return {
        **metadata,
        "ready": verification.get("ready") is True,
        "reason_code": str(
            verification.get("reason_code", EXECUTION_ATTESTATION_INVALID)
        ),
        "remaining_gates": list(verification.get("remaining_gates", ())),
        "current_probe": probe if isinstance(probe, Mapping) else None,
    }


def _distillation_execution_contract_gate(
    config: CoordinatorConfig,
) -> dict[str, Any]:
    """Authorize train distillation without claiming an official SWE-bench PASS.

    The 19,008 public train rows are generic ``repo + base_commit`` work items;
    they do not carry a usable private TestSpec/test_patch.  Official TestSpec
    and heldout proof therefore remain an independent evaluation contract and
    must never be a prerequisite for producing train trajectories.
    """

    inspected = _execution_contract_gate(config)
    probe = inspected.get("current_probe")
    bindings = probe.get("bindings") if isinstance(probe, Mapping) else None
    dataset = bindings.get("dataset") if isinstance(bindings, Mapping) else None
    validator = bindings.get("validator") if isinstance(bindings, Mapping) else None
    opencode = bindings.get("opencode") if isinstance(bindings, Mapping) else None
    supervisor = (
        bindings.get("supervisor_private_state")
        if isinstance(bindings, Mapping)
        else None
    )
    remaining: list[str] = []
    if not isinstance(dataset, Mapping) or dataset.get("present_and_bound") is not True:
        remaining.append("source_train_parquet_binding_failed")
    validator_ready = (
        isinstance(validator, Mapping)
        and validator.get("code_bound") is True
        and validator.get("self_test") is True
        and validator.get("rejects_arbitrary_commands") is True
        and validator.get("distillation_validator_version")
        == TRAIN_SANDBOX_VALIDATOR_VERSION
        and validator.get("distillation_validator_family") == TRAIN_SANDBOX_IMAGE_FAMILY
        and validator.get("distillation_image_reference")
        == TRAIN_SANDBOX_IMAGE_REFERENCE
        and validator.get("distillation_image_id_sha256") == TRAIN_SANDBOX_IMAGE_ID
        and validator.get("distillation_result_schema")
        == TRAIN_SANDBOX_VALIDATOR_SCHEMA
        and validator.get("distillation_allowed_actions") == ["compile", "test"]
    )
    if not validator_ready:
        remaining.append("generic_train_validator_binding_invalid")
    if (
        not isinstance(opencode, Mapping)
        or opencode.get("tool_contract_version")
        != config.execution_contract.required_tool_contract_version
    ):
        remaining.append("opencode_tool_contract_v3_missing")
    if (
        not isinstance(opencode, Mapping)
        or opencode.get("model_isolation_contract") is not True
    ):
        remaining.append("post_agent_validator_isolation_contract_missing")
    if (
        not isinstance(supervisor, Mapping)
        or supervisor.get("distillation_receipt_key_metadata_valid") is not True
    ):
        remaining.append("distillation_receipt_key_missing_or_invalid")
    ready = not remaining
    return {
        "attestation_path": inspected["attestation_path"],
        "lock_path": inspected["lock_path"],
        "lock_sha256": inspected["lock_sha256"],
        "required_schema": inspected["required_schema"],
        "required_tool_contract_version": inspected["required_tool_contract_version"],
        "observed_schema": inspected["observed_schema"],
        "mode": "generic_train_repo_base_commit",
        "not_official_swebench_pass": True,
        "ready": ready,
        "reason_code": (
            "generic_train_execution_contract_ready"
            if ready
            else "generic_train_execution_contract_not_ready"
        ),
        "remaining_gates": remaining,
        "current_probe": probe if isinstance(probe, Mapping) else None,
        "official_evaluation_contract_ready": inspected["ready"],
        "official_evaluation_remaining_gates": inspected["remaining_gates"],
    }


def offline_preflight(config: CoordinatorConfig) -> dict[str, Any]:
    """Read only local metadata; never inspect an environment credential."""

    required_files = (
        config.bank_manifest,
        config.full_bank_config,
        config.opencode_bundle_manifest,
        config.opencode_patch_manifest,
        config.opencode_windows_binary,
        config.opencode_linux_binary,
        config.ccswitch_manifest,
        config.ccswitch_launcher,
        *(route.profile for route in config.routes.values()),
    )
    if any(not path.is_file() for path in required_files):
        raise CoordinatorError("required_component_missing")
    public_manifest = _mapping(
        json.loads(config.bank_manifest.read_text(encoding="utf-8")),
        "public_manifest",
    )
    if (
        public_manifest.get("schema_version") != PUBLIC_MANIFEST_SCHEMA
        or public_manifest.get("train_only") is not True
        or public_manifest.get("source_split") != "train"
        or public_manifest.get("publication_ready") is not True
    ):
        raise CoordinatorError("public_bank_manifest_invalid")
    route_manifest = _mapping(
        json.loads(config.ccswitch_manifest.read_text(encoding="utf-8")),
        "route_manifest",
    )
    if (
        route_manifest.get("schema_version") != ROUTE_MANIFEST_SCHEMA
        or route_manifest.get("ready") is not True
        or route_manifest.get("secret_persisted") is not False
    ):
        raise CoordinatorError("ccswitch_route_not_ready")
    binary = _mapping(route_manifest.get("binary"), "route_binary")
    route_binary = _project_path(PROJECT_ROOT, binary.get("path"), "route_binary")
    if not route_binary.is_file() or _sha256_file(route_binary) != binary.get("sha256"):
        raise CoordinatorError("ccswitch_binary_attestation_failed")
    profiles = {
        alias: _validate_profile(route.profile, alias)
        for alias, route in config.routes.items()
    }
    full_config = FullBankConfig.load(PROJECT_ROOT, config.full_bank_config)
    full_report = preflight_full_bank(full_config)
    publication = _mapping(
        _mapping(full_report.get("gates"), "full_bank_gates").get("publication"),
        "full_bank_publication_gate",
    )
    audited_export = _mapping(
        _mapping(publication.get("checks"), "publication_checks").get("audited_export"),
        "audited_export",
    )
    source_bank_manifest_sha256 = _sha256_file(config.bank_manifest)
    expected_manifest_path = config.bank_manifest.relative_to(PROJECT_ROOT).as_posix()
    if (
        full_report.get("publication_ready") is not True
        or publication.get("ready") is not True
        or audited_export.get("validated") is not True
        or audited_export.get("manifest_path") != expected_manifest_path
        or audited_export.get("sha256") != source_bank_manifest_sha256
        or audited_export.get("errors") != []
    ):
        raise CoordinatorError("full_bank_publication_gate_failed")
    if not full_report.get("launch_ready"):
        raise CoordinatorError("full_bank_launch_gate_failed")
    task_count = 0
    order_count = 0
    locale_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    for chain in iter_task_chains(
        config,
        expected_manifest_sha256=source_bank_manifest_sha256,
    ):
        task_count += 1
        order_count += len(chain.orders)
        locale = str(
            _mapping(chain.task.get("bilingual"), "bilingual").get(
                "requested_locale", ""
            )
        )
        locale_counts[locale] = locale_counts.get(locale, 0) + 1
        for order in chain.orders:
            alias = str(order["provider_alias"])
            provider_counts[alias] = provider_counts.get(alias, 0) + 1
    if (
        task_count != config.expected_tasks
        or order_count != config.expected_work_orders
        or order_count != task_count * len(EXPECTED_STAGES)
    ):
        raise CoordinatorError("bank_cardinality_mismatch")
    if locale_counts != {"en-US": task_count // 2, "zh-CN": task_count // 2}:
        raise CoordinatorError("bilingual_assignment_mismatch")
    execution_contract = _distillation_execution_contract_gate(config)
    component_ready = True
    bank_ready = True
    execution_contract_ready = bool(execution_contract["ready"])
    live_start_allowed = component_ready and bank_ready and execution_contract_ready
    return {
        "schema_version": "anchor.swebench-ccswitch-preflight.v1",
        "offline": True,
        "provider_requests": 0,
        "credentials_read": False,
        "sample_bodies_printed": False,
        "heldout_files_read": False,
        "task_count": task_count,
        "work_order_count": order_count,
        "stages": list(EXPECTED_STAGES),
        "locale_counts": dict(sorted(locale_counts.items())),
        "provider_work_order_counts": dict(sorted(provider_counts.items())),
        "profiles": {
            alias: {
                "profile_id": profile["profile_id"],
                "model_id": profile["model_selection"]["manual_model_id"],
                "reasoning_effort": profile["reasoning"]["effort"],
            }
            for alias, profile in sorted(profiles.items())
        },
        "component_ready": component_ready,
        "bank_ready": bank_ready,
        "source_bank_manifest_sha256": source_bank_manifest_sha256,
        "execution_contract_ready": execution_contract_ready,
        "live_start_allowed": live_start_allowed,
        "reason_code": execution_contract["reason_code"],
        "execution_contract": execution_contract,
        # Backward-compatible component+bank gate.  This is intentionally not
        # a live-launch authorization; callers must use live_start_allowed.
        "launch_ready": True,
        "live_started": False,
    }


@dataclass(frozen=True)
class StageArtifact:
    stage: str
    revision: int
    output: Mapping[str, Any]
    provider_alias: str


class Backend(Protocol):
    def prepare(self, chain: TaskChain) -> Mapping[str, Any]: ...

    def teacher(
        self,
        *,
        chain: TaskChain,
        order: Mapping[str, Any],
        stage: str,
        revision: int,
        context: Mapping[str, Any],
        prepared: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def builder(
        self,
        *,
        chain: TaskChain,
        order: Mapping[str, Any],
        revision: int,
        context: Mapping[str, Any],
        prepared: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def restore_builder(
        self, output: Mapping[str, Any], prepared: Mapping[str, Any]
    ) -> None: ...

    def finalization_outcome(
        self, chain: TaskChain, *, revision: int
    ) -> str | None: ...

    def finalize(
        self,
        *,
        chain: TaskChain,
        revision: int,
        builder: Mapping[str, Any],
        security: Mapping[str, Any],
        prepared: Mapping[str, Any],
        stage_records: Mapping[str, Mapping[str, Any]],
    ) -> None: ...

    def cleanup(self, chain: TaskChain, prepared: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class RecordStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.content = self.root / "content-records"
        self.events = self.root / "checkpoint.events.jsonl"
        self._lock = threading.Lock()
        self._completed: dict[tuple[str, str, int], tuple[Path, str]] = {}
        self._stage_counts: dict[str, int] = {stage: 0 for stage in EXPECTED_STAGES}
        self._failure_counts: dict[str, int] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.events.is_file():
            return
        for value in _read_jsonl(self.events):
            if value.get("schema_version") != EVENT_SCHEMA:
                raise CoordinatorError("checkpoint_schema_mismatch")
            if value.get("status") != "completed":
                if value.get("status") == "failed":
                    code = str(value.get("error_code", "unknown_failure"))
                    self._failure_counts[code] = self._failure_counts.get(code, 0) + 1
                continue
            relative = Path(str(value.get("artifact", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise CoordinatorError("checkpoint_artifact_path_invalid")
            artifact = self.root / relative
            digest = str(value.get("artifact_sha256", ""))
            if not artifact.is_file() or _sha256_file(artifact) != digest:
                raise CoordinatorError("checkpoint_artifact_hash_mismatch")
            key = (
                str(value.get("task_id", "")),
                str(value.get("stage", "")),
                int(value.get("revision", 1)),
            )
            first_seen = key not in self._completed
            self._completed[key] = (artifact, digest)
            stage = key[1]
            if first_seen and stage in self._stage_counts:
                self._stage_counts[stage] += 1

    def completed(self, task_id: str, stage: str, revision: int) -> bool:
        return (task_id, stage, revision) in self._completed

    def finished_outcome(self, task_id: str) -> str | None:
        if not self.completed(task_id, "security", 1):
            return None
        output = self.load_output(task_id, "security", 1)
        _validate_stage_output("security", output)
        return "blocked" if output.get("decision") == "BLOCK" else "completed"

    def final_revision(self, task_id: str, max_revisions: int) -> int:
        revisions: list[int] = []
        for revision in range(1, max_revisions + 1):
            if not self.completed(task_id, "domain_review", revision):
                continue
            output = self.load_output(task_id, "domain_review", revision)
            _validate_stage_output("domain_review", output)
            if output.get("decision") == "PASS":
                revisions.append(revision)
        if len(revisions) != 1:
            raise CoordinatorError("resume_final_revision_ambiguous")
        return revisions[0]

    def load_output(self, task_id: str, stage: str, revision: int) -> Mapping[str, Any]:
        artifact, _ = self._completed[(task_id, stage, revision)]
        value = _mapping(json.loads(artifact.read_text(encoding="utf-8")), "artifact")
        return _mapping(value.get("output"), "artifact_output")

    def lineage_records(
        self, task_id: str, *, final_revision: int
    ) -> dict[str, dict[str, Any]]:
        revisions = {
            "planner": 1,
            "tool_policy": 1,
            "domain_builder": final_revision,
            "domain_review": final_revision,
            "security": 1,
        }
        result: dict[str, dict[str, Any]] = {}
        for stage in EXPECTED_STAGES:
            revision = revisions[stage]
            record = self._completed.get((task_id, stage, revision))
            if record is None or not _SHA256.fullmatch(record[1]):
                raise CoordinatorError("distillation_lineage_checkpoint_missing")
            result[stage] = {
                "revision": revision,
                "artifact_sha256": record[1],
            }
        return result

    def write(
        self,
        *,
        chain: TaskChain,
        order: Mapping[str, Any],
        stage: str,
        revision: int,
        context: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> None:
        task_digest = chain.task_id.rsplit(":", 1)[-1]
        suffix = f".r{revision}" if revision > 1 else ""
        key = (chain.task_id, stage, revision)
        replacement_suffix = ".contract-retry" if key in self._completed else ""
        artifact = (
            self.content
            / task_digest[:2]
            / task_digest
            / f"{stage}{suffix}{replacement_suffix}.json"
        )
        value = {
            "schema_version": "anchor.swebench-ccswitch-stage-artifact.v1",
            "task_id": chain.task_id,
            "record_id": order["record_id"],
            "stage": stage,
            "revision": revision,
            "provider_alias": order["provider_alias"],
            "reasoning_effort": "max",
            "input": dict(context),
            "output": dict(output),
        }
        _atomic_json(artifact, value)
        digest = _sha256_file(artifact)
        relative = artifact.relative_to(self.root).as_posix()
        event = {
            "schema_version": EVENT_SCHEMA,
            "task_id": chain.task_id,
            "stage": stage,
            "revision": revision,
            "status": "completed",
            "provider_alias": order["provider_alias"],
            "artifact": relative,
            "artifact_sha256": digest,
        }
        encoded = canonical_json(event) + "\n"
        with self._lock:
            first_seen = key not in self._completed
            with self.events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._completed[key] = (artifact, digest)
            if first_seen and stage in self._stage_counts:
                self._stage_counts[stage] += 1

    def failure(self, task_id: str, stage: str, revision: int, code: str) -> None:
        event = {
            "schema_version": EVENT_SCHEMA,
            "task_id": task_id,
            "stage": stage,
            "revision": revision,
            "status": "failed",
            "error_code": code,
        }
        with self._lock:
            with self.events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._failure_counts[code] = self._failure_counts.get(code, 0) + 1

    def public_metrics(self) -> dict[str, dict[str, int]]:
        """Return aggregate, content-free checkpoint counters."""

        with self._lock:
            return {
                "stage_counts": dict(self._stage_counts),
                "failure_counts": dict(sorted(self._failure_counts.items())),
            }


class LocalizationStore:
    """Source-bound zh-CN overlays; the index contains no sample text."""

    def __init__(self, root: Path, *, source_binding: Mapping[str, str]) -> None:
        self.root = root.resolve()
        self.content = self.root / "content"
        self.index = self.root / "checkpoint.events.jsonl"
        self.source_binding = dict(source_binding)
        self._lock = threading.Lock()
        self._records: dict[str, tuple[Path, str, str]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        if self.index.is_file():
            for value in _read_jsonl(self.index):
                if (
                    value.get("schema_version")
                    != "anchor.swebench-localization-event.v1"
                ):
                    raise CoordinatorError("localization_checkpoint_schema_mismatch")
                relative = Path(str(value.get("artifact", "")))
                artifact = self.root / relative
                artifact_sha = str(value.get("artifact_sha256", ""))
                source_sha = str(value.get("source_sha256", ""))
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not artifact.is_file()
                    or _sha256_file(artifact) != artifact_sha
                    or not _SHA256.fullmatch(source_sha)
                ):
                    raise CoordinatorError("localization_checkpoint_invalid")
                self._records[str(value.get("task_id", ""))] = (
                    artifact,
                    artifact_sha,
                    source_sha,
                )

    def get(self, task_id: str, source_sha: str) -> str | None:
        record = self._records.get(task_id)
        if record is None or record[2] != source_sha:
            return None
        value = _mapping(
            json.loads(record[0].read_text(encoding="utf-8")), "localization"
        )
        translated = value.get("translated_problem_statement")
        return (
            translated if isinstance(translated, str) and translated.strip() else None
        )

    def put(self, task_id: str, source_sha: str, translated: str) -> None:
        digest = task_id.rsplit(":", 1)[-1]
        artifact = self.content / digest[:2] / f"{digest}.json"
        _atomic_json(
            artifact,
            {
                "schema_version": "anchor.swebench-zh-cn-localization.v1",
                "task_id": task_id,
                "source_sha256": source_sha,
                "locale": "zh-CN",
                "translated_problem_statement": translated,
            },
        )
        artifact_sha = _sha256_file(artifact)
        event = {
            "schema_version": "anchor.swebench-localization-event.v1",
            "task_id": task_id,
            "source_sha256": source_sha,
            "artifact": artifact.relative_to(self.root).as_posix(),
            "artifact_sha256": artifact_sha,
        }
        with self._lock:
            with self.index.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._records[task_id] = (artifact, artifact_sha, source_sha)

    @property
    def count(self) -> int:
        return len(self._records)

    def write_gate(self, path: Path, *, expected_count: int) -> None:
        if self.count != expected_count:
            raise CoordinatorError("localization_gate_incomplete")
        files = [
            {
                "path": artifact.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": artifact_sha,
                "records": 1,
            }
            for artifact, artifact_sha, _ in sorted(
                self._records.values(), key=lambda item: item[0].as_posix()
            )
        ]
        _atomic_json(
            path,
            {
                "schema_version": "anchor.swebench-zh-cn-localization-manifest.v1",
                "source": dict(self.source_binding),
                "complete": True,
                "train_only": True,
                "contains_heldout": False,
                "locale": "zh-CN",
                "record_count": self.count,
                "files": files,
            },
        )


def _stage_context(
    chain: TaskChain,
    stage: str,
    outputs: Mapping[str, Mapping[str, Any]],
    prepared: Mapping[str, Any],
    revision: int,
) -> dict[str, Any]:
    task = chain.task
    source = _mapping(task.get("source"), "task_source")
    context: dict[str, Any] = {
        "task_id": chain.task_id,
        "stage": stage,
        "revision": revision,
        "requested_locale": _mapping(task.get("bilingual"), "bilingual").get(
            "requested_locale"
        ),
        "domain_label": _mapping(task.get("routing_contract"), "routing_contract").get(
            "domain_label", "general"
        ),
        "source": {
            "dataset_id": source.get("dataset_id"),
            "dataset_revision": source.get("dataset_revision"),
            "split": "train",
            "instance_id": source.get("instance_id"),
            "repo": source.get("repo"),
            "base_commit": source.get("base_commit"),
        },
        "problem_statement": prepared.get(
            "localized_problem_statement",
            _mapping(task.get("public_input"), "public_input").get("problem_statement"),
        ),
        "localization": prepared.get(
            "localization",
            {"locale": "en-US", "source_bound": True},
        ),
    }
    if stage == "planner":
        context["controlled_workspace_inventory"] = prepared.get(
            "workspace_inventory", {}
        )
    elif stage == "tool_policy":
        context["planner"] = outputs["planner"]
    elif stage == "domain_builder":
        context["planner"] = outputs["planner"]
        context["tool_policy"] = outputs["tool_policy"]
        if "domain_review" in outputs:
            context["revision_feedback"] = outputs["domain_review"]
    elif stage == "domain_review":
        context["builder"] = outputs["domain_builder"]
    elif stage == "security":
        context["builder"] = outputs["domain_builder"]
        context["domain_review"] = outputs["domain_review"]
    return context


def _validate_stage_output(stage: str, value: Mapping[str, Any]) -> None:
    if not value:
        raise CoordinatorError(f"empty_{stage}_output")
    if stage == "planner":
        if value.get("schema_version") != "anchor.swebench-planner-output.v1":
            raise CoordinatorError("planner_schema_mismatch")
        if not isinstance(value.get("work_items"), list) or not isinstance(
            value.get("tool_proposals"), list
        ):
            raise CoordinatorError("planner_shape_invalid")
    elif stage == "tool_policy":
        if value.get("schema_version") != "anchor.swebench-tool-policy-output.v1":
            raise CoordinatorError("tool_policy_schema_mismatch")
        decisions = value.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise CoordinatorError("tool_policy_shape_invalid")
        if not any(
            isinstance(item, Mapping) and item.get("decision") == "APPROVE"
            for item in decisions
        ):
            raise CoordinatorError("tool_policy_approved_nothing")
    elif stage == "domain_builder":
        if (
            value.get("schema_version")
            != "controlled-opencode-export+real-tool-results"
        ):
            raise CoordinatorError("builder_schema_mismatch")
        calls = value.get("tool_calls")
        results = value.get("tool_results")
        if (
            not isinstance(calls, list)
            or not calls
            or not isinstance(results, list)
            or not results
        ):
            raise CoordinatorError("builder_missing_real_tool_trace")
    elif stage == "domain_review":
        if value.get("schema_version") != "anchor.swebench-domain-review-output.v1":
            raise CoordinatorError("review_schema_mismatch")
        if value.get("decision") not in {"PASS", "REVISE"}:
            raise CoordinatorError("review_verdict_invalid")
    elif stage == "security":
        if value.get("schema_version") != "anchor.swebench-security-output.v1":
            raise CoordinatorError("security_schema_mismatch")
        if value.get("decision") not in {"PASS", "BLOCK"}:
            raise CoordinatorError("security_verdict_invalid")


@dataclass(frozen=True)
class _BuilderPolicyFailure:
    public_code: str
    retry_stage: str | None
    retry_reason: str | None


def _execution_contract_error_key(error: ExecutionContractError) -> str | None:
    """Return only an exact allowlisted internal code, never exception text."""

    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    key = error.args[0]
    return key if key in _BUILDER_POLICY_PUBLIC_ERROR_CODES else None


def _public_builder_policy_error(error: ExecutionContractError) -> str:
    key = _execution_contract_error_key(error)
    if key is None:
        return _BUILDER_POLICY_FALLBACK_ERROR_CODE
    return _BUILDER_POLICY_PUBLIC_ERROR_CODES[key]


def _builder_policy_failure(
    planner: Mapping[str, Any],
    tool_policy: Mapping[str, Any],
) -> _BuilderPolicyFailure | None:
    """Deep-check the pair and classify one fixed, content-free correction."""

    try:
        approved_builder_policy(planner, tool_policy)
    except ExecutionContractError as error:
        key = _execution_contract_error_key(error)
        public_code = _public_builder_policy_error(error)
        retry_stage: str | None = None
        retry_reason: str | None = None
        if key == "builder_policy_input_invalid":
            proposals = planner.get("tool_proposals")
            if not isinstance(proposals, list) or not proposals:
                retry_stage = "planner"
                retry_reason = "planner_structure_invalid"
            else:
                retry_stage = "tool_policy"
                retry_reason = "tool_policy_structure_invalid"
        elif key == "builder_policy_decision_invalid":
            retry_stage = "tool_policy"
            retry_reason = "tool_policy_decision_invalid"
        elif key == "builder_policy_proposal_invalid":
            retry_stage = "planner"
            retry_reason = "planner_proposal_invalid"
        elif key == "builder_policy_tool_outside_global_allowlist":
            retry_stage = "planner"
            retry_reason = "planner_tool_invalid"
        elif key in {
            "builder_policy_bash_input_invalid",
            "builder_policy_bash_command_invalid",
        }:
            retry_stage = "planner"
            retry_reason = "planner_bash_binding_invalid"
        elif key == "builder_policy_proposal_binding_ambiguous":
            retry_stage = "planner"
            retry_reason = "planner_duplicate_family"
        elif key == "builder_policy_coverage_invalid":
            proposals = planner.get("tool_proposals")
            decisions = tool_policy.get("decisions")
            proposal_ids = {
                item.get("proposal_id")
                for item in proposals
                if isinstance(proposals, list)
                and isinstance(item, Mapping)
                and isinstance(item.get("proposal_id"), str)
            }
            decision_ids = {
                item.get("proposal_id")
                for item in decisions
                if isinstance(decisions, list)
                and isinstance(item, Mapping)
                and isinstance(item.get("proposal_id"), str)
            }
            if proposal_ids != decision_ids:
                retry_stage = "tool_policy"
                retry_reason = "tool_policy_missing_decision"
            else:
                write_proposal_ids = {
                    item.get("proposal_id")
                    for item in proposals
                    if isinstance(proposals, list)
                    and isinstance(item, Mapping)
                    and isinstance(item.get("proposal_id"), str)
                    and isinstance(item.get("tool"), str)
                    and ToolPolicy.normalize_tool(str(item["tool"])) == "edit"
                }
                if write_proposal_ids:
                    retry_stage = "tool_policy"
                    retry_reason = "tool_policy_write_required"
                else:
                    retry_stage = "planner"
                    retry_reason = "planner_write_required"
        return _BuilderPolicyFailure(
            public_code=public_code,
            retry_stage=retry_stage,
            retry_reason=retry_reason,
        )
    return None


def _structure_retry_reason(stage: str, code: str) -> str | None:
    if stage == "planner" and code == "planner_duplicate_family":
        return "planner_duplicate_family"
    retryable = {
        "planner": {
            "empty_planner_output",
            "planner_schema_mismatch",
            "planner_shape_invalid",
        },
        "tool_policy": {
            "empty_tool_policy_output",
            "tool_policy_schema_mismatch",
            "tool_policy_shape_invalid",
            "tool_policy_approved_nothing",
        },
    }
    if code not in retryable.get(stage, set()):
        return None
    return f"{stage}_structure_invalid"


def _planner_has_duplicate_non_bash_family(value: Mapping[str, Any]) -> bool:
    """Detect redundant permission proposals before spending a policy request."""

    proposals = value.get("tool_proposals")
    if not isinstance(proposals, list):
        return False
    seen: set[str] = set()
    for item in proposals:
        if not isinstance(item, Mapping):
            continue
        tool = item.get("tool")
        if not isinstance(tool, str) or tool == "bash":
            continue
        family = ToolPolicy.normalize_tool(tool)
        if family in seen:
            return True
        seen.add(family)
    return False


def _contract_retry_context(
    context: Mapping[str, Any], stage: str, reason: str | None
) -> Mapping[str, Any]:
    if reason is None:
        return context
    if reason not in _CONTRACT_RETRY_GUIDANCE:
        raise CoordinatorError("contract_retry_reason_invalid")
    return {
        **context,
        "contract_retry": {
            "schema_version": _CONTRACT_RETRY_SCHEMA,
            "stage": stage,
            "reason_code": reason,
        },
    }


def _resolve_planner_and_tool_policy(
    chain: TaskChain,
    backend: Backend,
    store: RecordStore,
    orders: Mapping[str, Mapping[str, Any]],
    prepared: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Resolve, deep-validate, then checkpoint the planner/policy pair."""

    outputs: dict[str, Mapping[str, Any]] = {}
    contexts: dict[str, Mapping[str, Any]] = {}
    attempts = {"planner": 0, "tool_policy": 0}
    generated = {"planner": False, "tool_policy": False}

    def obtain(stage: str, retry_reason: str | None = None) -> Mapping[str, Any]:
        context = _stage_context(chain, stage, outputs, prepared, 1)
        if attempts[stage] == 0 and store.completed(chain.task_id, stage, 1):
            attempts[stage] = 1
            output = store.load_output(chain.task_id, stage, 1)
        else:
            if attempts[stage] >= 2:
                raise CoordinatorError(f"{stage}_semantic_retry_exhausted")
            attempts[stage] += 1
            context = _contract_retry_context(context, stage, retry_reason)
            output = backend.teacher(
                chain=chain,
                order=orders[stage],
                stage=stage,
                revision=1,
                context=context,
                prepared=prepared,
            )
            generated[stage] = True
        contexts[stage] = context
        return output

    def obtain_valid(stage: str, retry_reason: str | None = None) -> Mapping[str, Any]:
        reason = retry_reason
        while True:
            output = obtain(stage, reason)
            try:
                _validate_stage_output(stage, output)
                if stage == "planner" and _planner_has_duplicate_non_bash_family(
                    output
                ):
                    raise CoordinatorError("planner_duplicate_family")
            except CoordinatorError as error:
                reason = _structure_retry_reason(stage, error.code)
                if reason is None or attempts[stage] >= 2:
                    raise
                continue
            return output

    outputs["planner"] = obtain_valid("planner")
    outputs["tool_policy"] = obtain_valid("tool_policy")
    while True:
        failure = _builder_policy_failure(outputs["planner"], outputs["tool_policy"])
        if failure is None:
            break
        if (
            failure.retry_stage is None
            or failure.retry_reason is None
            or attempts[failure.retry_stage] >= 2
        ):
            raise CoordinatorError(failure.public_code)
        if failure.retry_stage == "planner":
            outputs["planner"] = obtain_valid("planner", failure.retry_reason)
            if attempts["tool_policy"] >= 2:
                raise CoordinatorError(failure.public_code)
            outputs["tool_policy"] = obtain_valid(
                "tool_policy", "tool_policy_rebind_planner"
            )
        else:
            outputs["tool_policy"] = obtain_valid("tool_policy", failure.retry_reason)

    # Neither stage is persisted until the pair passes the deep intersection.
    for stage in ("planner", "tool_policy"):
        if generated[stage]:
            store.write(
                chain=chain,
                order=orders[stage],
                stage=stage,
                revision=1,
                context=contexts[stage],
                output=outputs[stage],
            )
    return outputs


def _project_agent_tool_trace(
    execution: AgentExecution,
    planner: Mapping[str, Any],
    tool_policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one real trace into correlated call and result projections."""

    # Revalidate the immutable authorization pair at the trace boundary.  This
    # prevents duplicate decision IDs or malformed proposals from being folded
    # by the lookup maps below if this helper is ever called outside builder().
    try:
        approved_builder_policy(planner, tool_policy)
    except ExecutionContractError as error:
        raise CoordinatorError(
            "builder_tool_trace_proposal_binding_invalid"
        ) from error
    decisions = {
        str(item.get("proposal_id")): str(item.get("decision"))
        for item in tool_policy.get("decisions", [])
        if isinstance(item, Mapping)
    }
    proposals: list[Mapping[str, Any]] = [
        item for item in planner.get("tool_proposals", []) if isinstance(item, Mapping)
    ]
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in execution.trace:
        if item.source != "agent":
            continue
        if item.sequence < 1 or item.sequence in seen:
            raise CoordinatorError("builder_tool_trace_sequence_invalid")
        seen.add(item.sequence)
        matches = []
        for proposal in proposals:
            proposal_tool = proposal.get("tool")
            if not isinstance(proposal_tool, str) or (
                ToolPolicy.normalize_tool(proposal_tool)
                != ToolPolicy.normalize_tool(item.tool)
            ):
                continue
            proposal_input = proposal.get("input")
            if item.tool == "bash" and (
                not isinstance(proposal_input, Mapping)
                or proposal_input.get("command") != item.command
            ):
                continue
            if decisions.get(str(proposal.get("proposal_id"))) == "APPROVE":
                matches.append(proposal)
        if len(matches) != 1:
            raise CoordinatorError("builder_tool_trace_proposal_binding_invalid")
        if not isinstance(item.input_sha256, str) or not _SHA256.fullmatch(
            item.input_sha256
        ):
            raise CoordinatorError("builder_tool_trace_input_binding_missing")
        proposal = matches[0]
        proposal_id = str(proposal["proposal_id"])
        proposal_input = _mapping(proposal.get("input"), "planner_tool_input")
        tool_input: Mapping[str, Any] = (
            {"command": item.command}
            if item.tool == "bash" and isinstance(item.command, str)
            else dict(proposal_input)
        )
        command_sha256 = (
            sha256(item.command.encode("utf-8")).hexdigest()
            if isinstance(item.command, str)
            else None
        )
        invocation_sha256 = sha256(
            canonical_json(
                {
                    "tool": item.tool,
                    "input": dict(tool_input),
                    "actual_input_sha256": item.input_sha256,
                    "planner_proposal_id": proposal_id,
                }
            ).encode("utf-8")
        ).hexdigest()
        calls.append(
            {
                "sequence": item.sequence,
                "tool": item.tool,
                "input": dict(tool_input),
                "input_provenance": "planner-approved-authorization-scope",
                "actual_input_sha256": item.input_sha256,
                "command": item.command,
                "command_sha256": command_sha256,
                "invocation_sha256": invocation_sha256,
                "planner_proposal_id": proposal_id,
                "tool_policy_decision": (decisions.get(proposal_id)),
                "execution_scope": "isolated-instance-container",
            }
        )
        results.append(
            {
                "sequence": item.sequence,
                "tool": item.tool,
                "status": item.status,
                "exit_code": item.exit_code,
                "duration_ms": item.duration_ms,
                "output_sha256": item.output_sha256,
                "actual_input_sha256": item.input_sha256,
                "command_sha256": command_sha256,
                "invocation_sha256": invocation_sha256,
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            }
        )
    return calls, results


def run_chain(
    chain: TaskChain, backend: Backend, store: RecordStore, max_revisions: int
) -> str:
    prepared: Mapping[str, Any] = {}
    outputs: dict[str, Mapping[str, Any]] = {}
    orders = {str(order["stage"]): order for order in chain.orders}
    stage = "prepare"
    revision = 1
    try:
        if store.completed(chain.task_id, "security", 1):
            security = store.load_output(chain.task_id, "security", 1)
            _validate_stage_output("security", security)
            if security.get("decision") == "BLOCK":
                return "blocked"
            stored_revisions = [
                item
                for item in range(1, max_revisions + 1)
                if store.completed(chain.task_id, "domain_review", item)
                and store.load_output(chain.task_id, "domain_review", item).get(
                    "decision"
                )
                == "PASS"
            ]
            if len(stored_revisions) != 1:
                raise CoordinatorError("resume_final_revision_ambiguous")
            finalization_outcome = backend.finalization_outcome(
                chain,
                revision=stored_revisions[0],
            )
            if finalization_outcome is not None:
                return finalization_outcome
        prepared = backend.prepare(chain)
        stage = "tool_policy"
        outputs.update(
            _resolve_planner_and_tool_policy(
                chain,
                backend,
                store,
                orders,
                prepared,
            )
        )
        for revision in range(1, max_revisions + 1):
            stage = "domain_builder"
            order = orders[stage]
            context = _stage_context(chain, stage, outputs, prepared, revision)
            if store.completed(chain.task_id, stage, revision):
                builder = store.load_output(chain.task_id, stage, revision)
                _validate_stage_output(stage, builder)
                builder_loaded = True
            else:
                builder = backend.builder(
                    chain=chain,
                    order=order,
                    revision=revision,
                    context=context,
                    prepared=prepared,
                )
                _validate_stage_output(stage, builder)
                store.write(
                    chain=chain,
                    order=order,
                    stage=stage,
                    revision=revision,
                    context=context,
                    output=builder,
                )
                builder_loaded = False
            outputs[stage] = builder
            stage = "domain_review"
            order = orders[stage]
            context = _stage_context(chain, stage, outputs, prepared, revision)
            review_loaded = store.completed(chain.task_id, stage, revision)
            builder_restored = False
            if builder_loaded and not review_loaded:
                # The process crashed after persisting the cumulative builder
                # diff but before review.  Materialise that exact checkpoint in
                # the fresh workspace before the reviewer inspects it.
                backend.restore_builder(builder, prepared)
                builder_restored = True
            if review_loaded:
                review = store.load_output(chain.task_id, stage, revision)
                _validate_stage_output(stage, review)
            else:
                review = backend.teacher(
                    chain=chain,
                    order=order,
                    stage=stage,
                    revision=revision,
                    context=context,
                    prepared=prepared,
                )
                _validate_stage_output(stage, review)
                store.write(
                    chain=chain,
                    order=order,
                    stage=stage,
                    revision=revision,
                    context=context,
                    output=review,
                )
            outputs[stage] = review
            if review.get("decision") == "PASS":
                if builder_loaded and not builder_restored:
                    # A checkpointed PASS followed by a crash still needs its
                    # cumulative diff restored before the security stage.
                    backend.restore_builder(builder, prepared)
                break
            next_builder_loaded = revision < max_revisions and store.completed(
                chain.task_id, "domain_builder", revision + 1
            )
            if builder_loaded and not builder_restored and not next_builder_loaded:
                # If the next cumulative builder checkpoint already exists it
                # supersedes this diff and will be restored from the clean base
                # on its own iteration.  Otherwise the next live builder must
                # continue from the latest materialised revision.
                backend.restore_builder(builder, prepared)
        else:
            raise CoordinatorError("review_revision_budget_exhausted")
        stage = "security"
        order = orders[stage]
        context = _stage_context(chain, stage, outputs, prepared, revision)
        if store.completed(chain.task_id, stage, 1):
            security = store.load_output(chain.task_id, stage, 1)
            _validate_stage_output(stage, security)
        else:
            security = backend.teacher(
                chain=chain,
                order=order,
                stage=stage,
                revision=1,
                context=context,
                prepared=prepared,
            )
            _validate_stage_output(stage, security)
            store.write(
                chain=chain,
                order=order,
                stage=stage,
                revision=1,
                context=context,
                output=security,
            )
        if security.get("decision") == "BLOCK":
            return "blocked"
        backend.finalize(
            chain=chain,
            revision=revision,
            builder=outputs["domain_builder"],
            security=security,
            prepared=prepared,
            stage_records=store.lineage_records(
                chain.task_id,
                final_revision=revision,
            ),
        )
        return "completed"
    except CoordinatorError as error:
        store.failure(chain.task_id, stage, revision, error.code)
        return "failed"
    except Exception:  # noqa: BLE001 - never put exception/body text in the ledger
        store.failure(chain.task_id, stage, revision, "unexpected_backend_failure")
        return "failed"
    finally:
        try:
            backend.cleanup(chain, prepared)
        except Exception:  # noqa: BLE001 - cleanup is separately content-free
            store.failure(chain.task_id, "cleanup", 1, "sandbox_cleanup_failed")
            # A leaked container/worktree/private mount is never a successful
            # task, even if a model/security verdict was already persisted.
            return "failed"


class ReplayBackend:
    """Deterministic offline backend used by schema and resume tests."""

    def __init__(
        self,
        responses: Mapping[str, Mapping[str, Any]],
        *,
        finalization_already_completed: bool = True,
        finalization_failed: bool = False,
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []
        self.events: list[tuple[str, str, int]] = []
        self.cleanup_count = 0
        self.restored_revisions: list[int] = []
        self.finalization_already_completed = finalization_already_completed
        self.finalization_failed = finalization_failed
        self.finalized_revisions: list[int] = []

    def prepare(self, chain: TaskChain) -> Mapping[str, Any]:
        del chain
        return {"workspace_inventory": {"files": []}}

    def _response(self, stage: str, revision: int) -> Mapping[str, Any]:
        key = f"{stage}:r{revision}"
        value = self.responses.get(key, self.responses.get(stage))
        if value is None:
            raise CoordinatorError("replay_response_missing")
        self.calls.append((stage, revision))
        self.events.append(("call", stage, revision))
        return dict(value)

    def teacher(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._response(str(kwargs["stage"]), int(kwargs["revision"]))

    def builder(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._response("domain_builder", int(kwargs["revision"]))

    def restore_builder(
        self, output: Mapping[str, Any], prepared: Mapping[str, Any]
    ) -> None:
        del prepared
        revision = int(output.get("revision", 1))
        self.restored_revisions.append(revision)
        self.events.append(("restore", "domain_builder", revision))

    def finalization_outcome(self, chain: TaskChain, *, revision: int) -> str | None:
        del chain, revision
        if self.finalization_failed:
            return "failed"
        return "completed" if self.finalization_already_completed else None

    def finalize(self, **kwargs: Any) -> None:
        revision = int(kwargs["revision"])
        self.finalized_revisions.append(revision)
        self.events.append(("finalize", "official_eval", revision))

    def cleanup(self, chain: TaskChain, prepared: Mapping[str, Any]) -> None:
        del chain, prepared
        self.cleanup_count += 1

    def close(self) -> None:
        return


@dataclass(frozen=True)
class LocalRouteOpenCodeProvider(OpenCodeProvider):
    """OpenCode provider restricted to one RFC1918/loopback HTTP route.

    The shared provider class intentionally requires public HTTPS.  This
    subclass is narrower, not broader: it accepts only the ephemeral CC Switch
    route proven reachable from the sandbox, and its credential name must be
    the non-provider placeholder used for local client syntax.
    """

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError as exc:
            raise ValueError("local CC Switch route requires a literal IP") from exc
        if (
            parsed.scheme != "http"
            or not (address.is_private or address.is_loopback)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/anchor/v1")
            or self.route_host != str(address)
            or self.key_env != "ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN"
            or self.npm != "@ai-sdk/openai"
            or self.variant != "max"
        ):
            raise ValueError("invalid audited local CC Switch provider")
        if not self.provider_id.startswith("anchor-") or not self.model.strip():
            raise ValueError("invalid audited local CC Switch identity")


class LiveBackend:
    """Real CC Switch + patched OpenCode backend, created only after confirmation."""

    def __init__(self, config: CoordinatorConfig) -> None:
        self.config = config
        self._full_config = FullBankConfig.load(PROJECT_ROOT, config.full_bank_config)
        self._localizations = LocalizationStore(
            config.runtime.output_dir / "localization",
            source_binding={
                "dataset_id": self._full_config.dataset_id,
                "dataset_revision": self._full_config.dataset_revision,
                "split": "train",
                "parquet_sha256": self._full_config.source_parquet_sha256,
            },
        )
        self._profiles = {
            alias: _validate_profile(route.profile, alias)
            for alias, route in config.routes.items()
        }
        # This is the first credential read in the program.  It is unreachable
        # from offline_preflight and the values are never stored or printed.
        for profile in self._profiles.values():
            key_env = str(profile["key_env"])
            if not os.environ.get(key_env, "").strip():
                raise CoordinatorError("live_credential_missing")
        self._processes: list[subprocess.Popen[bytes]] = []
        self._route_log_handles: list[Any] = []
        self._route_urls: dict[str, str] = {}
        self._repo_locks: dict[str, threading.Lock] = {}
        self._repo_locks_guard = threading.Lock()
        self._providers: dict[str, LocalRouteOpenCodeProvider] = {}
        self._executors: dict[str, OpenCodeExecutor] = {}
        self._runtime_handles: dict[str, GenericWorkspaceHandle] = {}
        self._runtime_handles_lock = threading.Lock()
        self._pending_receipts: dict[str, PendingDistillationReceipt] = {}
        self._pending_receipts_lock = threading.Lock()
        checkpoint_identity = _checkpoint_identity(config)
        self._checkpoint_id = checkpoint_identity["checkpoint_id"]
        self._source_bank_manifest_sha256 = checkpoint_identity[
            "source_bank_manifest_sha256"
        ]
        self._config_sha256 = _sha256_file(config.config_path)
        self._private_root = config.runtime.output_dir / "system-private"
        self._usage_lock = threading.Lock()
        self._usage_path = config.runtime.output_dir / "usage.events.jsonl"
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
        self._transport_lock = threading.Lock()
        self._transport_path = config.runtime.output_dir / "transport.events.jsonl"
        self._transport_totals = {
            "provider_requests": 0,
            "provider_successes": 0,
            "provider_failures": 0,
            "retry_attempts": 0,
        }
        self._transport_failure_counts: dict[str, int] = {}
        self._response_envelope_lock = threading.Lock()
        self._response_envelope_path = (
            config.runtime.output_dir / "response-envelope.events.jsonl"
        )
        self._route_startup_error_code: str | None = None
        self._route_failure_public_code: str | None = None
        self._route_diagnostic_path = (
            config.runtime.output_dir / ROUTE_FAILURE_DIAGNOSTIC_NAME
        )
        self._load_public_telemetry()
        self._output_token_baseline = self._usage_totals["output_tokens"]
        try:
            self._start_routes()
            try:
                self._route_diagnostic_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise CoordinatorError("route_diagnostic_cleanup_failed") from exc
            self._prepare_v3_runtime()
        except Exception as error:
            if (
                isinstance(error, CoordinatorError)
                and error.code in ROUTE_STARTUP_ERROR_CODES
            ):
                self._route_startup_error_code = error.code
            self.close()
            if self._route_failure_public_code is not None:
                raise CoordinatorError(self._route_failure_public_code) from error
            raise

    @staticmethod
    def _normalize_route_address_candidates(
        values: Iterable[str],
        *,
        include_loopback: bool = True,
    ) -> tuple[str, ...]:
        """Return unique local IPv4 candidates with loopback first.

        A TUN adapter can install a synthetic WSL default route whose next hop
        is not an address Windows is allowed to bind.  Candidate ownership is
        therefore derived from Windows addresses, not the WSL route table.
        The benchmark block is intentionally excluded because transparent
        proxy/TUN software commonly uses it for synthetic endpoints.
        """

        candidates = ["127.0.0.1"] if include_loopback else []
        seen = set(candidates)
        benchmark_network = ipaddress.ip_network("198.18.0.0/15")
        for value in values:
            try:
                address = ipaddress.ip_address(str(value).strip())
            except ValueError:
                continue
            if not isinstance(address, ipaddress.IPv4Address):
                continue
            if (
                address.is_unspecified
                or address.is_multicast
                or address.is_link_local
                or (address.is_loopback and not include_loopback)
                or address in benchmark_network
                or not (address.is_private or address.is_loopback)
            ):
                continue
            normalized = str(address)
            if normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
        return tuple(candidates)

    def _windows_preferred_ipv4_addresses(self) -> tuple[str, ...]:
        raw: list[str] = []
        try:
            raw.extend(
                str(sockaddr[0])
                for _family, _kind, _proto, _canonical, sockaddr in socket.getaddrinfo(
                    socket.gethostname(),
                    0,
                    family=socket.AF_INET,
                    type=socket.SOCK_STREAM,
                )
            )
        except OSError:
            pass

        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-Command",
                    (
                        "$ErrorActionPreference='SilentlyContinue'; "
                        "Get-NetIPAddress -AddressFamily IPv4 | "
                        "Where-Object { $_.AddressState -eq 'Preferred' } | "
                        "Sort-Object InterfaceIndex,PrefixLength,IPAddress | "
                        "ForEach-Object { $_.IPAddress }"
                    ),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            raw.extend(completed.stdout.splitlines())
        return self._normalize_route_address_candidates(
            raw,
            include_loopback=False,
        )

    def _wsl_default_gateway_addresses(self) -> tuple[str, ...]:
        try:
            completed = subprocess.run(
                [
                    "wsl.exe",
                    "--distribution",
                    self.config.runtime.wsl_distro,
                    "--user",
                    "root",
                    "--exec",
                    "bash",
                    "-lc",
                    (
                        "ip -4 route show default | "
                        'awk \'{for (i=1; i<=NF; i++) if ($i == "via") '
                        "print $(i+1)}'"
                    ),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()
        if completed.returncode != 0:
            return ()
        return self._normalize_route_address_candidates(
            completed.stdout.splitlines(),
            include_loopback=False,
        )

    def _route_address_candidates(self) -> tuple[str, ...]:
        """Prefer mirrored localhost; restrict NAT fallback to host gateways.

        Exposing an unauthenticated router on an arbitrary WLAN/private address
        would widen the trust boundary.  A non-loopback fallback is eligible
        only when it is both a WSL default gateway and a Windows Preferred IPv4
        address; the subsequent socket probe still has to prove bind ownership
        and WSL reachability.
        """

        windows_addresses = set(self._windows_preferred_ipv4_addresses())
        gateways = self._wsl_default_gateway_addresses()
        return (
            "127.0.0.1",
            *(gateway for gateway in gateways if gateway in windows_addresses),
        )

    def _wsl_tcp_probe(self, host: str, port: int) -> bool:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        if not isinstance(address, ipaddress.IPv4Address) or not 1 <= port <= 65535:
            return False
        command = "timeout 8 bash -c '</dev/tcp/" + str(address) + "/" + str(port) + "'"
        try:
            completed = subprocess.run(
                [
                    "wsl.exe",
                    "--distribution",
                    self.config.runtime.wsl_distro,
                    "--user",
                    "root",
                    "--exec",
                    "bash",
                    "-lc",
                    command,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def _probe_route_address(self, host: str) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_EXCLUSIVEADDRUSE,
                        1,
                    )
                listener.bind((host, 0))
                listener.listen(1)
                listener.settimeout(2.0)
                port = int(listener.getsockname()[1])
                if not self._wsl_tcp_probe(host, port):
                    return False
                connection, _peer = listener.accept()
                connection.close()
                return True
        except OSError:
            return False

    def _discover_route_listen_address(self) -> str:
        if self._probe_route_address("127.0.0.1"):
            return "127.0.0.1"
        for candidate in self._route_address_candidates():
            if candidate == "127.0.0.1":
                continue
            if self._probe_route_address(candidate):
                return candidate
        raise CoordinatorError("wsl_probed_host_discovery_failed")

    def _start_routes(self) -> None:
        route_address = self._discover_route_listen_address()
        self._route_listen_address = route_address
        route_root = self.config.runtime.output_dir / "route-runtime"
        route_root.mkdir(parents=True, exist_ok=True)
        for alias, spec in self.config.routes.items():
            profile = json.loads(spec.profile.read_text(encoding="utf-8"))
            profile["route"]["listen_address"] = route_address
            profile["route"]["port"] = spec.port
            runtime_profile = route_root / f"{alias}.profile.json"
            _atomic_json(runtime_profile, profile)
            state = route_root / f"{alias}.state"
            command = [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.config.ccswitch_launcher),
                "-ProfilePath",
                str(runtime_profile),
                "-ManifestPath",
                str(self.config.ccswitch_manifest),
                "-StateHome",
                str(state),
                "-Port",
                str(spec.port),
                "-NetworkMode",
                self.config.runtime.provider_network_mode,
            ]
            stdout_handle = (route_root / f"{alias}.stdout.log").open("ab")
            stderr_handle = (route_root / f"{alias}.stderr.log").open("ab")
            self._route_log_handles.extend((stdout_handle, stderr_handle))
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            self._processes.append(process)
            base = f"http://{route_address}:{spec.port}/anchor/v1"
            self._wait_route(
                process,
                f"http://{route_address}:{spec.port}/anchor/health",
            )
            self._verify_wsl_tcp(route_address, spec.port)
            self._route_urls[alias] = base

    def _wait_route(self, process: subprocess.Popen[bytes], health: str) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise CoordinatorError("ccswitch_route_exited")
            try:
                with _DIRECT_OPENER.open(health, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, HTTPError, URLError):
                time.sleep(0.25)
        raise CoordinatorError("ccswitch_route_health_timeout")

    def _verify_wsl_tcp(self, host: str, port: int) -> None:
        if not self._wsl_tcp_probe(host, port):
            raise CoordinatorError("ccswitch_route_not_visible_from_wsl")

    def _prepare_v3_runtime(self) -> None:
        image_reference = self._ensure_train_sandbox_image()
        for alias, profile in self._profiles.items():
            selection = _mapping(profile["model_selection"], "model_selection")
            provider = LocalRouteOpenCodeProvider(
                provider_id=f"anchor-{alias.replace('_', '-')}",
                npm="@ai-sdk/openai",
                base_url="http://127.0.0.1:18080/anchor/v1",
                model=str(selection["manual_model_id"]),
                variant="max",
                key_env="ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN",
                route_host="127.0.0.1",
            )
            self._providers[alias] = provider
        try:
            self._receipt_key = load_distillation_supervisor_receipt_key(
                self.config.runtime.wsl_distro
            )
            for alias, provider in self._providers.items():
                route = self.config.routes[alias]
                executor = OpenCodeExecutor(
                    executable=str(self.config.opencode_windows_binary),
                    extra_environment={"ANCHOR_PODMAN_IMAGE": image_reference},
                    patch_manifest=self.config.opencode_patch_manifest,
                    sandbox_options=AnchorSandboxOptions(
                        linux_executable=self.config.opencode_linux_binary,
                        wsl_distro=self.config.runtime.wsl_distro,
                        supervisor=self.config.runtime.sandbox_supervisor,
                        memory=self.config.runtime.sandbox_memory,
                        cpus=self.config.runtime.sandbox_cpus,
                        pids=self.config.runtime.sandbox_pids,
                        timeout_seconds=self.config.runtime.sandbox_timeout_seconds,
                        route_host=self._route_listen_address,
                        route_port=route.port,
                    ),
                    provider=provider,
                )
                probe_config = (
                    self.config.runtime.output_dir
                    / "model-config"
                    / f"probe-{alias}.opencode.json"
                )
                self._write_local_route_config(
                    probe_config,
                    ToolPolicy(allowed_tools=("read",), allowed_commands=()),
                    provider,
                )
                passed, _reason = executor.probe_patched(probe_config)
                # This generated probe config contains no provider credential or
                # sample content.  A transient Windows scanner/file-share lock
                # must not invalidate an otherwise completed behavioral proof;
                # the next run overwrites it atomically before reuse.
                try:
                    probe_config.unlink(missing_ok=True)
                except OSError:
                    pass
                if not passed:
                    raise CoordinatorError("opencode_behavioral_attestation_failed")
                self._executors[alias] = executor
        except (OSError, ExecutionContractError) as exc:
            raise CoordinatorError("v3_runtime_adapter_startup_failed") from exc

    @staticmethod
    def _write_local_route_config(
        path: Path,
        policy: ToolPolicy,
        provider: LocalRouteOpenCodeProvider,
    ) -> None:
        payload = build_opencode_config(policy, provider=provider)
        provider_config = _mapping(
            _mapping(payload["provider"], "opencode_provider")[provider.provider_id],
            "opencode_provider_config",
        )
        options = _mapping(provider_config["options"], "opencode_provider_options")
        provider_config = dict(provider_config)
        provider_config["options"] = {
            **dict(options),
            "apiKey": "{env:ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN}",
        }
        payload = dict(payload)
        payload["provider"] = {
            **dict(_mapping(payload["provider"], "opencode_provider")),
            provider.provider_id: provider_config,
        }
        _atomic_json(path, payload)

    def _wsl_root_run(
        self,
        args: Sequence[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        safe_environment = {
            key: value
            for key in ("SystemRoot", "SYSTEMROOT", "WINDIR", "PATH")
            if (value := os.environ.get(key))
        }
        return subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                self.config.runtime.wsl_distro,
                "--user",
                "root",
                "--exec",
                *args,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=safe_environment,
        )

    def _ensure_train_sandbox_image(self) -> str:
        """Verify the prebuilt, digest-pinned Python+Node train image exactly."""

        if (
            not TRAIN_SANDBOX_VALIDATOR.is_file()
            or not TRAIN_SANDBOX_CONTAINERFILE.is_file()
            or _sha256_file(TRAIN_SANDBOX_VALIDATOR)
            != "f42de489ef86a213b76904d83b856b604cc957506909a1b783a8e369dfd8dd56"
            or _sha256_file(TRAIN_SANDBOX_CONTAINERFILE)
            != "6df6f9eecf1547a9daf756bd117f4528483d7bfea58fccb3cf8035ecb81c8075"
        ):
            raise CoordinatorError("sandbox_validator_artifact_mismatch")
        self._validator_version_sha256 = _sha256_file(TRAIN_SANDBOX_VALIDATOR)
        inspected = self._wsl_root_run(
            [
                "podman",
                "image",
                "inspect",
                "--format",
                "{{.Id}}|{{.Digest}}|{{json .RepoDigests}}",
                TRAIN_SANDBOX_IMAGE_REFERENCE,
            ],
            timeout=60,
        )
        rendered = inspected.stdout.decode("utf-8", errors="replace").strip()
        image_id, separator, remainder = rendered.partition("|")
        image_digest, separator_two, repo_digests = remainder.partition("|")
        if (
            inspected.returncode != 0
            or not separator
            or not separator_two
            or image_id != TRAIN_SANDBOX_IMAGE_ID
            or image_digest != TRAIN_SANDBOX_IMAGE_REFERENCE.rsplit("@", 1)[1]
            or TRAIN_SANDBOX_IMAGE_REFERENCE not in repo_digests
        ):
            raise CoordinatorError("sandbox_validator_image_digest_invalid")
        version = self._wsl_root_run(
            [
                "podman",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=all",
                "--security-opt=no-new-privileges",
                "--cgroups=disabled",
                "--entrypoint=/usr/local/bin/anchor-validate",
                TRAIN_SANDBOX_IMAGE_REFERENCE,
                "--version",
            ],
            timeout=60,
        )
        if (
            version.returncode != 0
            or version.stdout.decode("utf-8", errors="strict").strip()
            != TRAIN_SANDBOX_VALIDATOR_VERSION
        ):
            raise CoordinatorError("sandbox_validator_version_mismatch")
        self._generic_image_digest = image_digest
        self._generic_image_id_sha256 = image_id
        self._generic_image_reference = TRAIN_SANDBOX_IMAGE_REFERENCE
        return TRAIN_SANDBOX_IMAGE_REFERENCE

    def _repo_lock(self, repo: str) -> threading.Lock:
        with self._repo_locks_guard:
            return self._repo_locks.setdefault(repo, threading.Lock())

    @staticmethod
    def _git(args: Sequence[str], cwd: Path | None = None) -> None:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise CoordinatorError("repository_materialization_failed")

    @staticmethod
    def _git_capture(args: Sequence[str], cwd: Path | None = None) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise CoordinatorError("repository_materialization_failed")
        return completed.stdout

    @staticmethod
    def _git_commit_has_missing_objects(cache: Path, commit: str) -> bool:
        """Return whether a promised commit still lacks reachable objects.

        ``git clone --filter=blob:none`` deliberately leaves worktree blobs as
        promised objects.  ``rev-list`` reports those objects with ``?`` while
        its subprocess timeout keeps a damaged repository from stalling a
        formal launch indefinitely.
        """

        completed = subprocess.run(
            [
                "git",
                "rev-list",
                "--objects",
                "--missing=print",
                commit,
            ],
            cwd=cache,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise CoordinatorError("repository_materialization_failed")
        return any(line.startswith(b"?") for line in completed.stdout.splitlines())

    @staticmethod
    def _disable_partial_clone_filter(cache: Path) -> None:
        completed = subprocess.run(
            [
                "git",
                "config",
                "--unset-all",
                "remote.origin.partialclonefilter",
            ],
            cwd=cache,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        # Git returns 5 when the requested key does not exist.  That already
        # satisfies the desired state.
        if completed.returncode not in {0, 5}:
            raise CoordinatorError("repository_materialization_failed")

    @staticmethod
    def _repository_url(repo: str) -> str:
        if not _REPOSITORY.fullmatch(repo):
            raise CoordinatorError("repository_identifier_invalid")
        return f"https://github.com/{repo}.git"

    def _safe_remove_workspace(self, workspace: Path) -> None:
        root = self.config.runtime.workspace_root.resolve()
        target = workspace.resolve()
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise CoordinatorError("workspace_cleanup_path_escape") from exc
        if not relative.parts or len(relative.parts) != 1:
            raise CoordinatorError("workspace_cleanup_path_invalid")
        if target.exists():

            def retry_readonly(
                function: Any,
                path: str,
                exc_info: tuple[type[BaseException], BaseException, Any],
            ) -> None:
                error = exc_info[1]
                if not isinstance(error, PermissionError):
                    raise error
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                function(path)

            # ``git clone --no-local`` marks copied pack files read-only on
            # Windows.  Clearing only that attribute inside the already
            # boundary-checked one-task workspace makes cleanup idempotent
            # without broadening its path authority.
            shutil.rmtree(target, onerror=retry_readonly)

    def _materialize_repository(
        self,
        *,
        task_id: str,
        repo: str,
        base_commit: str,
    ) -> Path:
        if (
            not _SHA256.fullmatch(task_id.rsplit(":", 1)[-1])
            or not _REPOSITORY.fullmatch(repo)
            or not _COMMIT.fullmatch(base_commit)
        ):
            raise CoordinatorError("repository_materialization_input_invalid")
        cache_root = self.config.runtime.repository_cache.resolve()
        workspace_root = self.config.runtime.workspace_root.resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        workspace_root.mkdir(parents=True, exist_ok=True)
        repo_digest = sha256(repo.encode("utf-8")).hexdigest()
        cache = cache_root / f"{repo_digest}.git"
        workspace = workspace_root / task_id.rsplit(":", 1)[-1]
        remote = self._repository_url(repo)
        with self._repo_lock(repo):
            if not cache.exists():
                self._git(
                    [
                        "clone",
                        "--bare",
                        "--filter=blob:none",
                        "--no-tags",
                        remote,
                        str(cache),
                    ]
                )
            if (
                self._git_capture(
                    ["rev-parse", "--is-bare-repository"], cwd=cache
                ).strip()
                != b"true"
            ):
                raise CoordinatorError("repository_cache_invalid")
            self._git(["remote", "set-url", "origin", remote], cwd=cache)
            fetched = subprocess.run(
                [
                    "git",
                    "fetch",
                    "--force",
                    "--filter=blob:none",
                    "--no-tags",
                    "origin",
                    base_commit,
                ],
                cwd=cache,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=900,
                check=False,
            )
            if fetched.returncode != 0:
                self._git(
                    [
                        "fetch",
                        "--force",
                        "--filter=blob:none",
                        "--no-tags",
                        "origin",
                        "+refs/heads/*:refs/remotes/origin/*",
                    ],
                    cwd=cache,
                )
            self._git(["cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=cache)
            if self._git_commit_has_missing_objects(cache, base_commit):
                # A configured partial-clone filter is implicitly reused by
                # later fetches.  Remove it before --refetch so the exact
                # target commit is hydrated with every worktree blob.
                self._disable_partial_clone_filter(cache)
                self._git(
                    [
                        "fetch",
                        "--force",
                        "--no-tags",
                        "--refetch",
                        "origin",
                        base_commit,
                    ],
                    cwd=cache,
                )
                if self._git_commit_has_missing_objects(cache, base_commit):
                    raise CoordinatorError("repository_materialization_objects_missing")
            self._safe_remove_workspace(workspace)
            materialize_branch = f"anchor-materialize/{base_commit}"
            materialize_ref = f"refs/heads/{materialize_branch}"
            self._git(["update-ref", materialize_ref, base_commit], cwd=cache)
            try:
                # A shared local clone writes a Windows cache path into
                # .git/objects/info/alternates.  That path is intentionally
                # outside the Linux validator's sole /testbed mount, so Git
                # inspection fails inside the sandbox.  A depth-one,
                # --no-local clone through a temporary cache ref copies the
                # exact snapshot into a self-contained worktree without
                # downloading it from GitHub again.
                self._git(
                    [
                        "clone",
                        "--no-checkout",
                        "--single-branch",
                        "--branch",
                        materialize_branch,
                        "--depth",
                        "1",
                        "--no-local",
                        cache.as_uri(),
                        str(workspace),
                    ]
                )
            finally:
                deleted = subprocess.run(
                    ["git", "update-ref", "-d", materialize_ref],
                    cwd=cache,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
                if deleted.returncode != 0:
                    self._safe_remove_workspace(workspace)
                    raise CoordinatorError("repository_materialization_failed")
            self._git(["remote", "set-url", "origin", remote], cwd=workspace)
            # The validator runs inside Linux.  Pin checkout normalization in
            # the repository itself so Windows global autocrlf settings cannot
            # make a clean host checkout appear modified in the container.
            self._git(["config", "core.autocrlf", "false"], cwd=workspace)
            self._git(["config", "core.filemode", "false"], cwd=workspace)
            self._git(["config", "core.safecrlf", "false"], cwd=workspace)
            # A --no-checkout clone can already have HEAD and the index at the
            # requested commit while its worktree is empty.  --force is
            # required to materialize the files instead of accepting that
            # misleading no-op state.
            self._git(["checkout", "--detach", "--force", base_commit], cwd=workspace)
        observed = self._git_capture(["rev-parse", "HEAD"], cwd=workspace)
        status = self._git_capture(
            ["status", "--porcelain=v1", "--untracked-files=no"], cwd=workspace
        )
        if observed.decode("ascii", errors="strict").strip() != base_commit or status:
            self._safe_remove_workspace(workspace)
            raise CoordinatorError("repository_materialization_binding_failed")
        return workspace

    def _run_train_validator(
        self,
        workspace: Path,
        *,
        mode: str,
    ) -> subprocess.CompletedProcess[bytes]:
        if mode not in {"compile", "test"}:
            raise CoordinatorError("sandbox_validation_mode_invalid")
        converted = self._wsl_root_run(
            ["wslpath", "-a", "-u", str(workspace)],
            timeout=30,
        )
        native = converted.stdout.decode("utf-8", errors="replace").strip()
        if converted.returncode != 0 or not native.startswith("/"):
            raise CoordinatorError("sandbox_validation_capability_missing")
        return self._wsl_root_run(
            [
                "podman",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=all",
                "--security-opt=no-new-privileges",
                "--cgroups=disabled",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
                "--workdir=/testbed",
                "--mount",
                f"type=bind,src={native},dst=/testbed,rw",
                "--entrypoint=/usr/local/bin/anchor-validate",
                self._generic_image_reference,
                mode,
            ],
            timeout=min(self.config.runtime.sandbox_timeout_seconds, 360),
        )

    @staticmethod
    def _parse_train_validator_result(
        stdout: bytes | str,
        *,
        expected_mode: str,
    ) -> dict[str, Any]:
        try:
            text_value = (
                stdout.decode("utf-8", errors="strict")
                if isinstance(stdout, bytes)
                else stdout
            )
            lines = [line for line in text_value.splitlines() if line.strip()]
            value = _mapping(json.loads(lines[-1]), "train_validator_result")
        except (
            IndexError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            CoordinatorError,
        ) as exc:
            raise CoordinatorError("sandbox_validation_result_invalid") from exc
        expected_fields = {
            "schema_version",
            "validator_version",
            "mode",
            "success",
            "not_official_swebench_pass",
            "validation_level",
            "changed_paths",
            "changed_paths_sha256",
            "final_state_sha256",
            "validators",
        }
        changed = value.get("changed_paths")
        validators = value.get("validators")
        if (
            set(value) != expected_fields
            or value.get("schema_version") != TRAIN_SANDBOX_VALIDATOR_SCHEMA
            or value.get("validator_version") != TRAIN_SANDBOX_VALIDATOR_VERSION
            or value.get("mode") != expected_mode
            or value.get("success") is not True
            or value.get("not_official_swebench_pass") is not True
            or value.get("validation_level")
            != ("native_test" if expected_mode == "test" else "syntax")
            or not isinstance(changed, list)
            or not changed
            or not all(isinstance(item, str) and item for item in changed)
            or changed != sorted(set(changed))
            or not isinstance(validators, list)
            or not validators
            or not all(isinstance(item, str) and item for item in validators)
            or validators != sorted(set(validators))
            or not _SHA256.fullmatch(str(value.get("changed_paths_sha256", "")))
            or not _SHA256.fullmatch(str(value.get("final_state_sha256", "")))
            or value.get("changed_paths_sha256")
            != sha256(canonical_json(changed).encode("utf-8")).hexdigest()
        ):
            raise CoordinatorError("sandbox_validation_result_invalid")
        allowed_suffixes = (
            DISTILLATION_VALIDATED_CODE_SUFFIXES
            | DISTILLATION_VALIDATED_NON_CODE_SUFFIXES
        )
        if any(
            item.startswith("/")
            or "\\" in item
            or any(part in {"", ".", ".."} for part in Path(item).parts)
            or Path(item).suffix.casefold() not in allowed_suffixes
            for item in changed
        ) or not any(
            Path(item).suffix.casefold() in DISTILLATION_VALIDATED_CODE_SUFFIXES
            for item in changed
        ):
            raise CoordinatorError("sandbox_validation_result_invalid")
        return dict(value)

    def _validation_capabilities(self, workspace: Path) -> tuple[str, ...]:
        probe_root = workspace / ".anchor-capability-probe"
        if probe_root.exists():
            raise CoordinatorError("sandbox_validation_capability_missing")
        try:
            probe_root.mkdir()
            (probe_root / "probe.py").write_text(
                "value: int = 1\n", encoding="utf-8", newline="\n"
            )
            (probe_root / "probe.js").write_text(
                "export const value = 1;\n", encoding="utf-8", newline="\n"
            )
            checked = self._run_train_validator(workspace, mode="compile")
            result = self._parse_train_validator_result(
                checked.stdout,
                expected_mode="compile",
            )
            if (
                checked.returncode != 0
                or result["changed_paths"]
                != [
                    ".anchor-capability-probe/probe.js",
                    ".anchor-capability-probe/probe.py",
                ]
                or result["validators"] != ["node-check", "python-compile"]
            ):
                raise CoordinatorError("sandbox_validation_capability_missing")
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)
        status = self._git_capture(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            cwd=workspace,
        )
        if status:
            raise CoordinatorError("sandbox_validation_capability_missing")
        return ("anchor-validate compile",)

    def _workspace_inventory(
        self,
        workspace: Path,
        capabilities: tuple[str, ...],
    ) -> dict[str, Any]:
        files = 0
        javascript = 0
        python = 0
        for directory, names, filenames in os.walk(workspace):
            names[:] = [name for name in names if name not in {".git", ".anchor"}]
            files += len(filenames)
            javascript += sum(
                name.casefold().endswith((".js", ".mjs", ".cjs")) for name in filenames
            )
            python += sum(name.casefold().endswith(".py") for name in filenames)
        return {
            "file_count": files,
            "javascript_file_count": javascript,
            "python_file_count": python,
            "package_json_present": (workspace / "package.json").is_file(),
            "validation_capabilities": list(capabilities),
            "image_family": TRAIN_SANDBOX_IMAGE_FAMILY,
            "validator_version": TRAIN_SANDBOX_VALIDATOR_VERSION,
            "validator_sha256": self._validator_version_sha256,
        }

    def prepare(self, chain: TaskChain) -> Mapping[str, Any]:
        if (
            chain.source_bank_manifest_sha256 != self._source_bank_manifest_sha256
            or not _SHA256.fullmatch(chain.candidate_task_artifact_sha256)
            or not _SHA256.fullmatch(chain.candidate_work_order_artifacts_sha256)
        ):
            raise CoordinatorError("candidate_artifact_binding_invalid")
        source = _mapping(chain.task.get("source"), "task_source")
        instance_id = _text(source.get("instance_id"), "source_instance_id")
        repo = _text(source.get("repo"), "source_repo")
        base_commit = _text(source.get("base_commit"), "source_base_commit")
        workspace = self._materialize_repository(
            task_id=chain.task_id,
            repo=repo,
            base_commit=base_commit,
        )
        handle = GenericWorkspaceHandle(
            task_id=chain.task_id,
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            workspace=workspace,
            image_digest=self._generic_image_digest,
            image_id_sha256=self._generic_image_id_sha256,
            validation_capabilities=(),
        )
        with self._runtime_handles_lock:
            if chain.task_id in self._runtime_handles:
                raise CoordinatorError("v3_runtime_duplicate_workspace")
            self._runtime_handles[chain.task_id] = handle
        capabilities = self._validation_capabilities(workspace)
        handle.validation_capabilities = capabilities
        inventory = self._workspace_inventory(workspace, capabilities)
        prepared: dict[str, Any] = {
            "runtime_task_id": chain.task_id,
            "workspace": str(workspace),
            "canonical_workspace": "/testbed",
            "workspace_source": "exact-public-repo-base-commit",
            "image_digest": handle.image_digest,
            "base_commit": handle.base_commit,
            "workspace_inventory": inventory,
            "not_official_swebench_pass": True,
            "source_bank_manifest_sha256": chain.source_bank_manifest_sha256,
            "candidate_task_artifact_sha256": (chain.candidate_task_artifact_sha256),
            "candidate_work_order_artifacts_sha256": (
                chain.candidate_work_order_artifacts_sha256
            ),
        }
        bilingual = _mapping(chain.task.get("bilingual"), "bilingual")
        if bilingual.get("requested_locale") == "zh-CN":
            statement = str(
                _mapping(chain.task.get("public_input"), "public_input")[
                    "problem_statement"
                ]
            )
            source_sha = sha256(statement.encode("utf-8")).hexdigest()
            translated = self._localizations.get(chain.task_id, source_sha)
            if translated is None:
                translated = self._translate_zh(chain.task_id, statement, source_sha)
                self._localizations.put(chain.task_id, source_sha, translated)
            prepared["localized_problem_statement"] = translated
            prepared["localization"] = {
                "locale": "zh-CN",
                "source_sha256": source_sha,
                "source_bound": True,
                "protected_fragments_preserved": True,
            }
        return prepared

    @staticmethod
    def _protected_fragments(statement: str) -> tuple[str, ...]:
        fragments: set[str] = set()
        fragments.update(re.findall(r"```[\s\S]*?```", statement))
        fragments.update(re.findall(r"`[^`\r\n]+`", statement))
        fragments.update(re.findall(r"https?://[^\s)\]}]+", statement))
        fragments.update(
            re.findall(
                r"(?<![\w.-])(?:[\w.-]+/)+[\w.@+-]+(?:\.[A-Za-z0-9]+)?",
                statement,
            )
        )
        return tuple(
            sorted((item for item in fragments if item), key=len, reverse=True)
        )

    def _translate_zh(self, task_id: str, statement: str, source_sha: str) -> str:
        del task_id
        protected = self._protected_fragments(statement)
        response = self._request_json(
            "glm52_max",
            {
                "model": self._profiles["glm52_max"]["model_selection"][
                    "manual_model_id"
                ],
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the public SWE-bench train issue into Simplified Chinese. "
                            "Preserve code fences, inline code, identifiers, paths, URLs, commands, "
                            "JSON keys, and tool names byte-for-byte. Do not solve the issue and do "
                            "not add hidden reasoning. Return strict JSON with schema_version, "
                            "source_sha256, and translated_problem_statement."
                        ),
                    },
                    {
                        "role": "user",
                        "content": canonical_json(
                            {
                                "source_sha256": source_sha,
                                "problem_statement": statement,
                            }
                        ),
                    },
                ],
                "reasoning": {"effort": "max"},
                "max_output_tokens": self.config.runtime.max_output_tokens,
            },
        )
        value = self._strict_json(self._response_text(response))
        translated = value.get("translated_problem_statement")
        if (
            value.get("schema_version") != "anchor.swebench-zh-cn-localization.v1"
            or value.get("source_sha256") != source_sha
            or not isinstance(translated, str)
            or not translated.strip()
            or not re.search(r"[\u4e00-\u9fff]", translated)
            or any(fragment not in translated for fragment in protected)
        ):
            raise CoordinatorError("zh_localization_fidelity_failed")
        return translated

    def _request_json(
        self, alias: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        url = self._route_urls[alias].rstrip("/") + "/responses"
        request = Request(
            url,
            data=canonical_json(dict(payload)).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "claude-code",
                "Authorization": "Bearer anchor-local-route",
            },
            method="POST",
        )
        last_error = "teacher_request_failed"
        for attempt in range(self.config.runtime.max_retries + 1):
            try:
                with _DIRECT_OPENER.open(
                    request, timeout=self.config.runtime.request_timeout_seconds
                ) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
            except (OSError, HTTPError, URLError, json.JSONDecodeError) as error:
                retry = attempt < self.config.runtime.max_retries
                self._record_transport_event(
                    alias,
                    status="failure",
                    retry=retry,
                    error_code=self._transport_error_code(error),
                )
                if not retry:
                    break
                time.sleep(min(2**attempt, 8))
                continue
            value = _mapping(decoded, "teacher_response")
            self._record_transport_event(alias, status="success", retry=False)
            self._record_usage(alias, value)
            self._record_response_envelope(alias, value)
            return value
        raise CoordinatorError(last_error)

    def _load_public_telemetry(self) -> None:
        if self._usage_path.is_file():
            for event in _read_jsonl(self._usage_path):
                if event.get("schema_version") != "anchor.swebench-ccswitch-usage.v1":
                    raise CoordinatorError("usage_checkpoint_schema_mismatch")
                usage = _mapping(event.get("usage"), "usage_checkpoint")
                for name in self._usage_totals:
                    value = usage.get(name)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                    ):
                        self._usage_totals[name] += value
        if self._transport_path.is_file():
            for event in _read_jsonl(self._transport_path):
                if event.get("schema_version") != TRANSPORT_EVENT_SCHEMA:
                    raise CoordinatorError("transport_checkpoint_schema_mismatch")
                status = event.get("status")
                retry = event.get("retry_scheduled")
                error_code = event.get("error_code")
                if (
                    status not in {"success", "failure"}
                    or not isinstance(retry, bool)
                    or (
                        status == "failure"
                        and (
                            not isinstance(error_code, str)
                            or not _ERROR_CODE.fullmatch(error_code)
                        )
                    )
                    or (status == "success" and error_code is not None)
                ):
                    raise CoordinatorError("transport_checkpoint_invalid")
                self._transport_totals["provider_requests"] += 1
                self._transport_totals[
                    "provider_successes" if status == "success" else "provider_failures"
                ] += 1
                if retry:
                    self._transport_totals["retry_attempts"] += 1
                if status == "failure":
                    self._transport_failure_counts[error_code] = (
                        self._transport_failure_counts.get(error_code, 0) + 1
                    )

    @staticmethod
    def _transport_error_code(error: BaseException) -> str:
        if isinstance(error, HTTPError):
            return f"http_{int(error.code)}"
        if isinstance(error, json.JSONDecodeError):
            return "response_json_invalid"
        if isinstance(error, TimeoutError):
            return "transport_timeout"
        if isinstance(error, URLError):
            return "url_error"
        return "transport_os_error"

    def _record_transport_event(
        self,
        alias: str,
        *,
        status: str,
        retry: bool,
        error_code: str | None = None,
    ) -> None:
        if alias not in self.config.routes or status not in {"success", "failure"}:
            raise CoordinatorError("transport_event_invalid")
        if (status == "success" and error_code is not None) or (
            status == "failure"
            and (
                not isinstance(error_code, str) or not _ERROR_CODE.fullmatch(error_code)
            )
        ):
            raise CoordinatorError("transport_event_invalid")
        event = {
            "schema_version": TRANSPORT_EVENT_SCHEMA,
            "provider_alias": alias,
            "status": status,
            "retry_scheduled": retry,
            "error_code": error_code,
        }
        self._transport_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transport_lock:
            with self._transport_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._transport_totals["provider_requests"] += 1
            self._transport_totals[
                "provider_successes" if status == "success" else "provider_failures"
            ] += 1
            if retry:
                self._transport_totals["retry_attempts"] += 1
            if status == "failure":
                assert error_code is not None
                self._transport_failure_counts[error_code] = (
                    self._transport_failure_counts.get(error_code, 0) + 1
                )

    def _record_usage(self, alias: str, response: Mapping[str, Any]) -> None:
        raw = response.get("usage")
        if not isinstance(raw, Mapping):
            return
        allowed = (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        )
        usage = {
            name: int(raw[name])
            for name in allowed
            if isinstance(raw.get(name), int)
            and not isinstance(raw.get(name), bool)
            and int(raw[name]) >= 0
        }
        if not usage:
            return
        event = {
            "schema_version": "anchor.swebench-ccswitch-usage.v1",
            "provider_alias": alias,
            "usage": usage,
        }
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._usage_lock:
            with self._usage_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            for name, value in usage.items():
                self._usage_totals[name] += value

    @staticmethod
    def _response_status(response: Mapping[str, Any]) -> str:
        raw = response.get("status")
        if not isinstance(raw, str):
            return "unknown"
        normalized = raw.strip().casefold()
        return normalized if normalized in _RESPONSE_STATUSES else "unknown"

    @staticmethod
    def _response_incomplete_reason(response: Mapping[str, Any]) -> str:
        details = response.get("incomplete_details")
        if not isinstance(details, Mapping):
            return "none"
        raw = details.get("reason")
        if not isinstance(raw, str):
            return "unknown"
        normalized = re.sub(r"[^a-z0-9]+", "_", raw.strip().casefold()).strip("_")
        if normalized in _MAX_OUTPUT_BUDGET_REASONS:
            return "max_output_tokens"
        if normalized == "content_filter":
            return normalized
        return "unknown"

    @staticmethod
    def _public_response_text(response: Mapping[str, Any]) -> str | None:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        chunks: list[str] = []
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping) or item.get("type") != "message":
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if (
                        isinstance(part, Mapping)
                        and part.get("type") == "output_text"
                        and isinstance(part.get("text"), str)
                        and str(part["text"]).strip()
                    ):
                        chunks.append(str(part["text"]))
        text = "\n".join(chunks).strip()
        return text or None

    @staticmethod
    def _bounded_response_type(value: object, allowed: frozenset[str]) -> str:
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in allowed:
                return normalized
        return "other"

    @staticmethod
    def _increment_response_type_count(counts: dict[str, int], name: str) -> None:
        counts[name] = min(counts.get(name, 0) + 1, _RESPONSE_TYPE_COUNT_LIMIT)

    @classmethod
    def _response_envelope(cls, response: Mapping[str, Any]) -> dict[str, Any]:
        item_counts: dict[str, int] = {}
        content_counts: dict[str, int] = {}
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    cls._increment_response_type_count(item_counts, "other")
                    continue
                item_type = cls._bounded_response_type(
                    item.get("type"), _RESPONSE_OUTPUT_ITEM_TYPES
                )
                cls._increment_response_type_count(item_counts, item_type)
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    part_type = (
                        cls._bounded_response_type(
                            part.get("type"), _RESPONSE_CONTENT_PART_TYPES
                        )
                        if isinstance(part, Mapping)
                        else "other"
                    )
                    cls._increment_response_type_count(content_counts, part_type)
        raw_usage = response.get("usage")
        usage = (
            {
                name: int(raw_usage[name])
                for name in _RESPONSE_USAGE_FIELDS
                if isinstance(raw_usage.get(name), int)
                and not isinstance(raw_usage.get(name), bool)
                and 0 <= int(raw_usage[name]) <= _RESPONSE_USAGE_LIMIT
            }
            if isinstance(raw_usage, Mapping)
            else {}
        )
        reason = cls._response_incomplete_reason(response)
        assert reason in _RESPONSE_INCOMPLETE_REASONS
        return {
            "status": cls._response_status(response),
            "incomplete_reason": reason,
            "output_item_type_counts": dict(sorted(item_counts.items())),
            "content_part_type_counts": dict(sorted(content_counts.items())),
            "usage": usage,
            "has_final_text": cls._public_response_text(response) is not None,
        }

    def _record_response_envelope(
        self, alias: str, response: Mapping[str, Any]
    ) -> None:
        if alias not in self.config.routes:
            raise CoordinatorError("response_envelope_event_invalid")
        event = {
            "schema_version": RESPONSE_ENVELOPE_EVENT_SCHEMA,
            "provider_alias": alias,
            **self._response_envelope(response),
        }
        self._response_envelope_path.parent.mkdir(parents=True, exist_ok=True)
        with self._response_envelope_lock:
            with self._response_envelope_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def public_telemetry(self, *, elapsed_seconds: float) -> dict[str, Any]:
        """Return exact aggregate counters without provider content or identifiers."""

        with self._transport_lock:
            transport = dict(self._transport_totals)
            transport_failures = dict(sorted(self._transport_failure_counts.items()))
        with self._usage_lock:
            usage = dict(self._usage_totals)
        current_output = max(
            0,
            usage["output_tokens"] - self._output_token_baseline,
        )
        output_rate = current_output / elapsed_seconds if elapsed_seconds > 0 else None
        return {
            "requests": transport,
            "request_failure_counts": transport_failures,
            "tokens": usage,
            "provider_output_tokens_per_second": (
                round(output_rate, 4) if output_rate is not None else None
            ),
        }

    @staticmethod
    def _response_text(response: Mapping[str, Any]) -> str:
        status = LiveBackend._response_status(response)
        if status != "completed":
            if status == "incomplete":
                if (
                    LiveBackend._response_incomplete_reason(response)
                    == "max_output_tokens"
                ):
                    raise CoordinatorError("teacher_output_budget_exhausted")
                raise CoordinatorError("teacher_response_incomplete")
            if status in {"cancelled", "canceled"}:
                raise CoordinatorError("teacher_response_cancelled")
            if status == "failed":
                raise CoordinatorError("teacher_response_failed")
            if status == "error":
                raise CoordinatorError("teacher_response_error")
            raise CoordinatorError("teacher_response_status_invalid")
        text = LiveBackend._public_response_text(response)
        if text is None:
            raise CoordinatorError("teacher_output_text_missing")
        return text

    @staticmethod
    def _strict_json(text: str) -> Mapping[str, Any]:
        candidate = text.strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as direct_error:
            fenced = re.fullmatch(
                r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
                candidate,
                flags=re.IGNORECASE,
            )
            if fenced is None:
                raise CoordinatorError(
                    "teacher_output_not_strict_json"
                ) from direct_error
            try:
                value = json.loads(fenced.group("body").strip())
            except json.JSONDecodeError as fenced_error:
                raise CoordinatorError(
                    "teacher_output_not_strict_json"
                ) from fenced_error
        return _mapping(value, "teacher_output")

    def teacher(self, **kwargs: Any) -> Mapping[str, Any]:
        order = _mapping(kwargs["order"], "order")
        alias = str(order["provider_alias"])
        stage = str(kwargs["stage"])
        context = _mapping(kwargs["context"], "context")
        locale = str(context.get("requested_locale", "en-US"))
        schema = str(order["required_output_schema"])
        contract = {
            "planner": {
                "schema_version": "anchor.swebench-planner-output.v1",
                "alignment_id": context["task_id"],
                "domain_id": context.get("domain_label", "general"),
                "builder_expert_id": "domain-builder",
                "reviewer_expert_id": "domain-review",
                "work_items": ["ordered public implementation step"],
                "tool_proposals": [
                    {
                        "proposal_id": "read-source",
                        "tool": "read",
                        "purpose": "inspect one exact workspace-relative path",
                        "input": {"path": "workspace-relative path"},
                    },
                    {
                        "proposal_id": "edit-source",
                        "tool": "edit",
                        "purpose": "make the planned implementation change",
                        "input": {"path": "workspace-relative path"},
                    },
                    {
                        "proposal_id": "validate-public",
                        "tool": "bash",
                        "purpose": "run a public model-visible validation",
                        "input": {"command": "anchor-validate compile"},
                    },
                ],
            },
            "tool_policy": {
                "schema_version": "anchor.swebench-tool-policy-output.v1",
                "alignment_id": context["task_id"],
                "executed_expert_id": "tool-policy",
                "decisions": [
                    {
                        "proposal_id": "copy every planner proposal_id",
                        "decision": "APPROVE or DENY",
                        "reason": "public reason",
                    }
                ],
            },
            "domain_review": {
                "schema_version": "anchor.swebench-domain-review-output.v1",
                "alignment_id": context["task_id"],
                "revision": int(kwargs["revision"]),
                "executed_expert_id": "domain-review",
                "decision": "PASS or REVISE",
                "feedback": ["public actionable feedback when REVISE"],
            },
            "security": {
                "schema_version": "anchor.swebench-security-output.v1",
                "alignment_id": context["task_id"],
                "executed_expert_id": "security",
                "decision": "PASS or BLOCK",
                "findings": ["public finding when BLOCK"],
            },
        }[stage]
        instruction = (
            f"Return exactly one JSON object using schema_version {schema}. "
            "Do not reveal hidden chain-of-thought. Reasons, work items, feedback, and findings "
            "are concise public decision evidence. Do not add fields beyond the contract. "
            "Never use benchmark gold patches, hidden tests, hints, or oracle labels. "
            f"Write human-readable fields in {locale}. Stage={stage}. "
            + (
                "For planner, emit a task-specific minimal proposal for every tool the builder "
                "will need. Allowed tools are read, grep, glob, list, edit, write, apply_patch, "
                "and bash. At least one of edit/write/apply_patch is required. Do not grant a "
                "generic tool bundle. Emit exactly one proposal total from the shared "
                "edit/write/apply_patch permission family; the planner proposes tools and never "
                "approves them. Emit at most one proposal for every other canonical non-bash "
                "OpenCode permission family. One proposal may authorize repeated calls in that "
                "family. Each bash proposal must be one unique exact command selected only from "
                "controlled_workspace_inventory.validation_capabilities; never propose a "
                "validator command that prepare did not prove available. "
                if stage == "planner"
                else ""
            )
            + (
                "For tool_policy, copy every planner proposal_id exactly once and independently "
                "APPROVE or DENY it; do not add or omit proposal IDs. If an old or abnormal "
                "planner contains multiple proposals in one canonical non-bash permission "
                "family, APPROVE at most one of them. "
                if stage == "tool_policy"
                else ""
            )
            + f"Exact JSON shape: {canonical_json(contract)}"
        )
        retry = context.get("contract_retry")
        if (
            isinstance(retry, Mapping)
            and retry.get("schema_version") == _CONTRACT_RETRY_SCHEMA
            and retry.get("stage") == stage
            and isinstance(retry.get("reason_code"), str)
        ):
            guidance = _CONTRACT_RETRY_GUIDANCE.get(str(retry["reason_code"]))
            if guidance is not None:
                instruction += (
                    " This is the single directed contract-correction retry. "
                    + guidance
                )
        payload = {
            "model": self._profiles[alias]["model_selection"]["manual_model_id"],
            "input": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": canonical_json(dict(context))},
            ],
            "reasoning": {"effort": "max"},
            "max_output_tokens": self.config.runtime.max_output_tokens,
        }
        response = self._request_json(alias, payload)
        try:
            text = self._response_text(response)
        except CoordinatorError as error:
            if error.code != "teacher_output_budget_exhausted":
                raise
            retry_payload = dict(payload)
            retry_payload["input"] = [
                {
                    "role": "system",
                    "content": (
                        instruction
                        + " The previous response exhausted its output budget before the final "
                        "answer. On this single retry, emit only the complete required JSON "
                        "object as concisely as possible."
                    ),
                },
                {"role": "user", "content": canonical_json(dict(context))},
            ]
            retry_payload["reasoning"] = {"effort": "max"}
            retry_payload["max_output_tokens"] = max(
                self.config.runtime.max_output_tokens,
                _SEMANTIC_RETRY_MAX_OUTPUT_TOKENS,
            )
            response = self._request_json(alias, retry_payload)
            text = self._response_text(response)
        return self._strict_json(text)

    def _terminal_validator_result_from_export(
        self,
        export: Mapping[str, Any],
        *,
        command: str,
        output_sha256: str,
    ) -> dict[str, Any]:
        """Recover only the final validator JSON whose raw output hash was traced."""

        if command not in {"anchor-validate compile", "anchor-validate test"}:
            raise CoordinatorError("sandbox_validation_result_invalid")
        expected_mode = command.rsplit(" ", 1)[1]
        matches: list[dict[str, Any]] = []
        messages = export.get("messages")
        if not isinstance(messages, list):
            raise CoordinatorError("sandbox_validation_result_invalid")
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if (
                    not isinstance(part, Mapping)
                    or part.get("type") != "tool"
                    or part.get("tool") != "bash"
                ):
                    continue
                state = part.get("state")
                if not isinstance(state, Mapping) or state.get("status") != "completed":
                    continue
                tool_input = state.get("input")
                raw_output = state.get("output")
                if (
                    not isinstance(tool_input, Mapping)
                    or tool_input.get("command") != command
                    or not isinstance(raw_output, str)
                    or sha256(raw_output.encode("utf-8", errors="replace")).hexdigest()
                    != output_sha256
                ):
                    continue
                matches.append(
                    self._parse_train_validator_result(
                        raw_output,
                        expected_mode=expected_mode,
                    )
                )
        if len(matches) != 1:
            raise CoordinatorError("sandbox_validation_result_invalid")
        return matches[0]

    def builder(self, **kwargs: Any) -> Mapping[str, Any]:
        order = _mapping(kwargs["order"], "order")
        context = _mapping(kwargs["context"], "context")
        _mapping(kwargs["prepared"], "prepared")
        chain = kwargs["chain"]
        revision = int(kwargs["revision"])
        alias = str(order["provider_alias"])
        provider = self._providers[alias]
        try:
            policy = approved_builder_policy(
                _mapping(context["planner"], "planner"),
                _mapping(context["tool_policy"], "tool_policy"),
                timeout_seconds=self.config.runtime.sandbox_timeout_seconds,
            )
        except ExecutionContractError as exc:
            raise CoordinatorError(_public_builder_policy_error(exc)) from exc
        config_path = (
            self.config.runtime.output_dir
            / "model-config"
            / chain.task_id.rsplit(":", 1)[-1]
            / f"revision-{revision}.opencode.json"
        )
        self._write_local_route_config(config_path, policy, provider)
        prompt = (
            "Execute the approved SWE-bench train task in this disposable workspace. "
            "Use real local tools and finish with a concise public outcome. Never inspect or request "
            "benchmark gold patches, test_patch, hints, oracle labels, dev, test, Lite, or Verified data.\n"
            + canonical_json(dict(context))
        )
        sample_id = f"{chain.task_id.rsplit(':', 1)[-1][:48]}-r{revision}"
        with self._runtime_handles_lock:
            handle = self._runtime_handles.get(chain.task_id)
        if handle is None:
            raise CoordinatorError("v3_runtime_workspace_missing")
        executor = self._executors[alias]
        handle.last_sample_id = sample_id
        handle.last_executor = executor
        execution: AgentExecution
        export: Mapping[str, Any] | None = None
        cleanup_failed = False
        try:
            execution = executor.run(
                sample_id=sample_id,
                prompt=prompt,
                workspace=handle.workspace,
                config_path=config_path,
                policy=policy,
            )
            export_path = (
                Path(execution.controlled_export_path)
                if execution.controlled_export_path
                else None
            )
            if export_path is not None and export_path.is_file():
                loaded = json.loads(export_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    export = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CoordinatorError("v3_opencode_builder_failed") from exc
        finally:
            try:
                executor.cleanup_sandbox(
                    sample_id=sample_id,
                    workspace=handle.workspace,
                )
            except (OSError, RuntimeError, ValueError):
                cleanup_failed = True
            config_path.unlink(missing_ok=True)
        if cleanup_failed:
            raise CoordinatorError("sandbox_cleanup_failed")
        tool_calls, tool_results = _project_agent_tool_trace(
            execution,
            _mapping(context["planner"], "planner"),
            _mapping(context["tool_policy"], "tool_policy"),
        )
        binary_diff = self._capture_workspace_diff(handle.workspace)
        if (
            execution.exit_code != 0
            or export is None
            or not binary_diff
            or execution.rejected_events != 0
        ):
            raise CoordinatorError("v3_opencode_builder_failed")
        output = {
            "schema_version": "controlled-opencode-export+real-tool-results",
            "revision": revision,
            "opencode_session_export": dict(export),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "workspace_diff": binary_diff.decode("utf-8", errors="strict"),
            "public_outcome": (
                asdict(execution.public_outcome) if execution.public_outcome else None
            ),
            "agent_stdout_sha256": execution.stdout_sha256,
            "agent_stderr_sha256": execution.stderr_sha256,
            "not_official_swebench_pass": True,
        }
        try:
            distillation_tool_evidence(output)
            terminal_call = max(tool_calls, key=lambda item: item["sequence"])
            terminal_result = max(tool_results, key=lambda item: item["sequence"])
            terminal_command = terminal_call.get("command")
            terminal_output_sha256 = terminal_result.get("output_sha256")
            if not isinstance(terminal_command, str) or not isinstance(
                terminal_output_sha256, str
            ):
                raise CoordinatorError("sandbox_validation_result_invalid")
            visible_validation = self._terminal_validator_result_from_export(
                export,
                command=terminal_command,
                output_sha256=terminal_output_sha256,
            )
            independent = self._run_train_validator(
                handle.workspace,
                mode=terminal_command.rsplit(" ", 1)[1],
            )
            independent_validation = self._parse_train_validator_result(
                independent.stdout,
                expected_mode=terminal_command.rsplit(" ", 1)[1],
            )
            if (
                independent.returncode != 0
                or independent_validation != visible_validation
            ):
                raise CoordinatorError("sandbox_validation_state_recompute_mismatch")
            changed_files = self._changed_files(handle.workspace)
            if [item["path"] for item in changed_files] != visible_validation[
                "changed_paths"
            ]:
                raise CoordinatorError("sandbox_validation_state_recompute_mismatch")
            changed_files_sha = sha256(
                canonical_json(changed_files).encode("utf-8")
            ).hexdigest()
            final_patch_sha = sha256(binary_diff).hexdigest()
            output["validation_state"] = {
                "schema_version": DISTILLATION_VALIDATION_STATE_SCHEMA,
                "final_patch_sha256": final_patch_sha,
                "changed_files": changed_files,
                "changed_files_sha256": changed_files_sha,
                "terminal_validation_output_sha256": terminal_result["output_sha256"],
                "terminal_command_sha256": terminal_call["command_sha256"],
                "validator_version_sha256": self._validator_version_sha256,
                "validator_result": visible_validation,
            }
            output["validation_state_sha256"] = distillation_validation_state_sha256(
                output,
                final_patch_sha256=final_patch_sha,
                validator_version_sha256=self._validator_version_sha256,
            )
            output["validator_version_sha256"] = self._validator_version_sha256
        except ExecutionContractError as exc:
            raise CoordinatorError("v3_public_validation_evidence_missing") from exc
        return output

    def _capture_workspace_diff(self, workspace: Path) -> bytes:
        self._git(
            [
                "-c",
                "core.safecrlf=false",
                "add",
                "--intent-to-add",
                "--all",
                "--",
                ".",
            ],
            cwd=workspace,
        )
        try:
            return self._git_capture(
                [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    "HEAD",
                    "--",
                    ".",
                    ":(exclude).anchor",
                ],
                cwd=workspace,
            )
        finally:
            self._git(["reset", "--mixed", "HEAD"], cwd=workspace)

    def _changed_files(self, workspace: Path) -> list[dict[str, str]]:
        self._git(
            ["-c", "core.safecrlf=false", "add", "--intent-to-add", "--all"],
            cwd=workspace,
        )
        try:
            deleted = self._git_capture(
                [
                    "-c",
                    "core.quotepath=false",
                    "diff",
                    "--name-only",
                    "-z",
                    "--diff-filter=D",
                    "HEAD",
                    "--",
                    ".",
                    ":(exclude).anchor",
                ],
                cwd=workspace,
            )
            changed = self._git_capture(
                [
                    "-c",
                    "core.quotepath=false",
                    "diff",
                    "--name-only",
                    "-z",
                    "--diff-filter=ACMR",
                    "HEAD",
                    "--",
                    ".",
                    ":(exclude).anchor",
                ],
                cwd=workspace,
            )
        finally:
            self._git(["reset", "--mixed", "HEAD"], cwd=workspace)
        if deleted.strip(b"\x00"):
            raise CoordinatorError("sandbox_validation_changed_path_unsupported")
        try:
            names = sorted(
                {
                    item.decode("utf-8", errors="strict")
                    for item in changed.split(b"\x00")
                    if item
                }
            )
        except UnicodeDecodeError as exc:
            raise CoordinatorError("sandbox_validation_changed_path_invalid") from exc
        if not names:
            raise CoordinatorError("sandbox_validation_changed_path_missing")
        root = workspace.resolve()
        result: list[dict[str, str]] = []
        code_file_count = 0
        allowed_suffixes = (
            DISTILLATION_VALIDATED_CODE_SUFFIXES
            | DISTILLATION_VALIDATED_NON_CODE_SUFFIXES
        )
        for relative in names:
            if (
                relative.startswith("/")
                or "\\" in relative
                or Path(relative).suffix.casefold() not in allowed_suffixes
            ):
                raise CoordinatorError("sandbox_validation_changed_path_unsupported")
            candidate = root / relative
            if candidate.is_symlink():
                raise CoordinatorError("sandbox_validation_changed_path_unsupported")
            path = candidate.resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise CoordinatorError(
                    "sandbox_validation_changed_path_escape"
                ) from exc
            if not path.is_file():
                raise CoordinatorError("sandbox_validation_changed_path_unsupported")
            result.append(
                {
                    "path": relative.replace("\\", "/"),
                    "sha256": _sha256_file(path),
                }
            )
            if Path(relative).suffix.casefold() in DISTILLATION_VALIDATED_CODE_SUFFIXES:
                code_file_count += 1
        if code_file_count < 1:
            raise CoordinatorError("sandbox_validation_changed_path_unsupported")
        return result

    def restore_builder(
        self, output: Mapping[str, Any], prepared: Mapping[str, Any]
    ) -> None:
        diff = output.get("workspace_diff")
        if not isinstance(diff, str) or not diff.strip() or "\x00" in diff:
            raise CoordinatorError("builder_resume_diff_invalid")
        task_id = prepared.get("runtime_task_id")
        if not isinstance(task_id, str):
            raise CoordinatorError("v3_runtime_resume_handle_missing")
        with self._runtime_handles_lock:
            handle = self._runtime_handles.get(task_id)
        if handle is None:
            raise CoordinatorError("v3_runtime_resume_handle_missing")
        try:
            observed = self._capture_workspace_diff(handle.workspace)
            if observed:
                raise CoordinatorError("builder_resume_workspace_not_clean")
            completed = subprocess.run(
                ["git", "apply", "--binary", "--whitespace=nowarn"],
                cwd=handle.workspace,
                input=diff.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
            if completed.returncode != 0 or self._capture_workspace_diff(
                handle.workspace
            ) != diff.encode("utf-8"):
                raise CoordinatorError("builder_resume_diff_rejected")
        except (OSError, UnicodeEncodeError, subprocess.TimeoutExpired) as exc:
            raise CoordinatorError("builder_resume_diff_rejected") from exc

    def finalization_outcome(self, chain: TaskChain, *, revision: int) -> str | None:
        del revision
        task_digest = sha256(chain.task_id.encode("utf-8")).hexdigest()
        directory = self._private_root / task_digest
        try:
            receipt = _mapping(
                json.loads(
                    (directory / "distillation-execution-receipt.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "distillation_receipt",
            )
            patch = (directory / "final.patch").read_bytes()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, CoordinatorError):
            return None
        source = _mapping(chain.task.get("source"), "task_source")
        expected = {
            "checkpoint_id": self._checkpoint_id,
            "config_sha256": self._config_sha256,
            "execution_lock_sha256": self.config.execution_contract.lock_sha256,
            "source_bank_manifest_sha256": (chain.source_bank_manifest_sha256),
            "candidate_task_artifact_sha256": (chain.candidate_task_artifact_sha256),
            "candidate_work_order_artifacts_sha256": (
                chain.candidate_work_order_artifacts_sha256
            ),
            "task_id_sha256": task_digest,
            "instance_id_sha256": sha256(
                str(source.get("instance_id", "")).encode("utf-8")
            ).hexdigest(),
            "repo_sha256": sha256(
                str(source.get("repo", "")).encode("utf-8")
            ).hexdigest(),
            "base_commit": str(source.get("base_commit", "")),
            "image_digest": self._generic_image_digest,
            "image_id_sha256": self._generic_image_id_sha256,
            "final_patch_sha256": sha256(patch).hexdigest(),
            "validator_version_sha256": self._validator_version_sha256,
        }
        if not verify_distillation_execution_receipt(
            receipt,
            trusted_receipt_key=self._receipt_key,
        ):
            return None
        if any(receipt.get(name) != value for name, value in expected.items()):
            return None
        return "completed"

    def finalize(self, **kwargs: Any) -> None:
        chain = kwargs["chain"]
        if int(kwargs["revision"]) < 1:
            raise CoordinatorError("v3_runtime_revision_invalid")
        builder = _mapping(kwargs["builder"], "builder")
        security = _mapping(kwargs["security"], "security")
        stage_records = _mapping(kwargs["stage_records"], "stage_records")
        if security.get("decision") != "PASS":
            raise CoordinatorError("v3_runtime_finalize_without_security_pass")
        diff = builder.get("workspace_diff")
        if not isinstance(diff, str) or not diff.strip():
            raise CoordinatorError("v3_runtime_final_diff_missing")
        with self._runtime_handles_lock:
            handle = self._runtime_handles.get(chain.task_id)
        if handle is None:
            raise CoordinatorError("v3_runtime_workspace_missing")
        try:
            final_patch = self._capture_workspace_diff(handle.workspace)
            if not final_patch or final_patch != diff.encode("utf-8"):
                raise CoordinatorError("v3_runtime_final_diff_binding_failed")
            transcript_sha, validation_sha = distillation_tool_evidence(builder)
            validation_state_sha = distillation_validation_state_sha256(
                builder,
                final_patch_sha256=sha256(final_patch).hexdigest(),
                validator_version_sha256=self._validator_version_sha256,
            )
            if (
                builder.get("validation_state_sha256") != validation_state_sha
                or builder.get("validator_version_sha256")
                != self._validator_version_sha256
            ):
                raise CoordinatorError("v3_runtime_validation_state_binding_failed")
            task_id_sha = sha256(chain.task_id.encode("utf-8")).hexdigest()
            lineage_sha = distillation_lineage_sha256(
                checkpoint_id=self._checkpoint_id,
                config_sha256=self._config_sha256,
                execution_lock_sha256=self.config.execution_contract.lock_sha256,
                task_id_sha256=task_id_sha,
                stage_records=stage_records,
            )
            bindings = {
                "checkpoint_id": self._checkpoint_id,
                "config_sha256": self._config_sha256,
                "execution_lock_sha256": self.config.execution_contract.lock_sha256,
                "source_bank_manifest_sha256": (chain.source_bank_manifest_sha256),
                "candidate_task_artifact_sha256": (
                    chain.candidate_task_artifact_sha256
                ),
                "candidate_work_order_artifacts_sha256": (
                    chain.candidate_work_order_artifacts_sha256
                ),
                "task_id_sha256": task_id_sha,
                "instance_id_sha256": sha256(
                    handle.instance_id.encode("utf-8")
                ).hexdigest(),
                "repo_sha256": sha256(handle.repo.encode("utf-8")).hexdigest(),
                "base_commit": handle.base_commit,
                "image_digest": handle.image_digest,
                "image_id_sha256": handle.image_id_sha256,
                "final_patch_sha256": sha256(final_patch).hexdigest(),
                "tool_transcript_sha256": transcript_sha,
                "validation_evidence_sha256": validation_sha,
                "validation_state_sha256": validation_state_sha,
                "validator_version_sha256": self._validator_version_sha256,
                "lineage_sha256": lineage_sha,
            }
        except ExecutionContractError as exc:
            raise CoordinatorError("v3_runtime_self_verification_failed") from exc
        with self._pending_receipts_lock:
            self._pending_receipts[chain.task_id] = PendingDistillationReceipt(
                bindings=bindings,
                final_patch=final_patch,
                builder_output=dict(builder),
            )

    def cleanup(self, chain: TaskChain, prepared: Mapping[str, Any]) -> None:
        del prepared
        with self._runtime_handles_lock:
            handle = self._runtime_handles.pop(chain.task_id, None)
        if handle is None:
            return
        try:
            if handle.last_executor is not None and handle.last_sample_id is not None:
                handle.last_executor.cleanup_sandbox(
                    sample_id=handle.last_sample_id,
                    workspace=handle.workspace,
                )
            self._safe_remove_workspace(handle.workspace)
        except (OSError, RuntimeError, ValueError, CoordinatorError) as exc:
            raise CoordinatorError("sandbox_cleanup_failed") from exc
        with self._pending_receipts_lock:
            pending = self._pending_receipts.pop(chain.task_id, None)
        if pending is None:
            return
        try:
            issue_distillation_execution_receipt_after_cleanup(
                private_root=self._private_root,
                bindings=pending.bindings,
                final_patch=pending.final_patch,
                builder_output=pending.builder_output,
                trusted_receipt_key=self._receipt_key,
            )
        except (OSError, ExecutionContractError) as exc:
            raise CoordinatorError("distillation_receipt_issue_failed") from exc

    def close(self) -> None:
        route_sources = [
            RouteDiagnosticSource(
                route_alias=alias,
                exit_code=process.poll(),
                stderr_path=(
                    self.config.runtime.output_dir
                    / "route-runtime"
                    / f"{alias}.stderr.log"
                ),
            )
            for alias, process in zip(self.config.routes, self._processes)
        ]
        for process in reversed(self._processes):
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                        check=False,
                    )
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
        for handle in self._route_log_handles:
            try:
                handle.close()
            except OSError:
                pass
        if self._route_startup_error_code is not None:
            try:
                diagnostic = write_route_failure_diagnostic(
                    self._route_diagnostic_path,
                    startup_error_code=self._route_startup_error_code,
                    sources=route_sources,
                )
            except (OSError, ValueError):
                self._route_failure_public_code = None
            else:
                self._route_failure_public_code = str(
                    diagnostic["classified_error_code"]
                )
        if not self.config.runtime.retain_router_state:
            shutil.rmtree(
                self.config.runtime.output_dir / "route-runtime", ignore_errors=True
            )

    @property
    def localization_count(self) -> int:
        return self._localizations.count

    def write_localization_gate(self) -> None:
        path = self._full_config.gate_paths["training"]["zh_cn_localization_manifest"]
        self._localizations.write_gate(path, expected_count=9504)


def run_live(
    config: CoordinatorConfig,
    *,
    control_run_id: str,
    resume: bool,
    concurrency: int | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    if not _CONTROL_RUN_ID.fullmatch(control_run_id):
        raise CoordinatorError("invalid_control_run_id")
    identity = _checkpoint_identity(config)
    _validate_checkpoint_mode(config, resume=resume, identity=identity)
    store = RecordStore(config.runtime.output_dir)
    status_path = config.runtime.output_dir / "status.json"
    started_at = time.time()
    checkpoint_metrics = store.public_metrics()
    # A persisted security PASS is not a terminal result.  The resume baseline
    # grows only when the task is a persisted BLOCK or its system-private
    # official-eval receipt authenticates as PASS/FAIL against this checkpoint.
    completed_task_baseline = 0
    startup_status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "control_run_id": control_run_id,
        "checkpoint_id": identity["checkpoint_id"],
        "config_sha256": identity["config_sha256"],
        "execution_lock_sha256": identity["execution_lock_sha256"],
        "source_bank_manifest_sha256": identity["source_bank_manifest_sha256"],
        "resume_mode": resume,
        "state": "starting",
        "submitted_tasks": 0,
        "active_tasks": 0,
        "completed_tasks": 0,
        "completed_task_baseline": completed_task_baseline,
        "current_run_completed_tasks": 0,
        "expected_tasks": config.expected_tasks,
        "counts": {"completed": 0, "blocked": 0, "failed": 0},
        "stage_counts": checkpoint_metrics["stage_counts"],
        "failure_counts": checkpoint_metrics["failure_counts"],
        "requests": {
            "provider_requests": 0,
            "provider_successes": 0,
            "provider_failures": 0,
            "retry_attempts": 0,
        },
        "request_failure_counts": {},
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        },
        "elapsed_seconds": 0.0,
        "tasks_per_minute": 0.0,
        "provider_output_tokens_per_second": 0.0,
        "eta_seconds": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_error_code": None,
        "content_free": True,
    }
    _atomic_json(status_path, startup_status)
    try:
        backend = LiveBackend(config)
    except Exception as error:  # noqa: BLE001 - emit only a stable content-free code
        code = (
            error.code
            if isinstance(error, CoordinatorError)
            else "backend_startup_failed"
        )
        _atomic_json(
            status_path,
            {
                **startup_status,
                "state": "failed",
                "failure_counts": {code: 1},
                "last_error_code": code,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        if isinstance(error, CoordinatorError):
            raise
        raise CoordinatorError(code) from error
    counts = {"completed": 0, "blocked": 0, "failed": 0}
    submitted = 0
    active_tasks = 0
    progress_lock = threading.Lock()
    terminal_state = "failed"
    monitor_stop = threading.Event()

    def write_public_status(state: str) -> None:
        elapsed = max(time.time() - started_at, 0.001)
        with progress_lock:
            safe_counts = dict(counts)
            safe_submitted = submitted
            safe_active = active_tasks
            safe_baseline = completed_task_baseline
        completed_tasks = sum(safe_counts.values())
        current_run_completed, tasks_per_minute, eta_seconds = _progress_rates(
            completed_tasks=completed_tasks,
            completed_task_baseline=safe_baseline,
            expected_tasks=config.expected_tasks,
            elapsed_seconds=elapsed,
            state=state,
        )
        checkpoint_metrics = store.public_metrics()
        telemetry = backend.public_telemetry(elapsed_seconds=elapsed)
        _atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA,
                "control_run_id": control_run_id,
                "checkpoint_id": identity["checkpoint_id"],
                "config_sha256": identity["config_sha256"],
                "execution_lock_sha256": identity["execution_lock_sha256"],
                "source_bank_manifest_sha256": identity["source_bank_manifest_sha256"],
                "resume_mode": resume,
                "state": state,
                "submitted_tasks": safe_submitted,
                "active_tasks": safe_active,
                "completed_tasks": completed_tasks,
                "completed_task_baseline": safe_baseline,
                "current_run_completed_tasks": current_run_completed,
                "expected_tasks": config.expected_tasks,
                "counts": safe_counts,
                "stage_counts": checkpoint_metrics["stage_counts"],
                "failure_counts": checkpoint_metrics["failure_counts"],
                "requests": telemetry["requests"],
                "request_failure_counts": telemetry["request_failure_counts"],
                "tokens": telemetry["tokens"],
                "elapsed_seconds": round(elapsed, 3),
                "tasks_per_minute": round(tasks_per_minute, 4),
                "provider_output_tokens_per_second": telemetry[
                    "provider_output_tokens_per_second"
                ],
                "eta_seconds": round(eta_seconds, 3)
                if eta_seconds is not None
                else None,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_error_code": None,
                "content_free": True,
            },
        )

    def monitor_status() -> None:
        while not monitor_stop.wait(1.0):
            write_public_status("running")

    write_public_status("running")
    monitor = threading.Thread(
        target=monitor_status,
        daemon=True,
        name="anchor-formal-content-free-status",
    )
    monitor.start()
    try:
        workers = concurrency or config.runtime.concurrency
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future[str], str] = {}
            chains = iter(
                iter_task_chains(
                    config,
                    expected_manifest_sha256=identity["source_bank_manifest_sha256"],
                )
            )

            def submit_one() -> bool:
                nonlocal submitted, active_tasks, completed_task_baseline
                while True:
                    if max_tasks is not None and submitted >= max_tasks:
                        return False
                    try:
                        chain = next(chains)
                    except StopIteration:
                        return False
                    prior = store.finished_outcome(chain.task_id) if resume else None
                    if prior == "completed":
                        revision = store.final_revision(
                            chain.task_id,
                            config.runtime.max_revisions,
                        )
                        # Missing, tampered, stale, or key-rotated receipts are
                        # deliberately not terminal and are replayed through
                        # the fresh hidden official evaluator below.
                        prior = backend.finalization_outcome(
                            chain,
                            revision=revision,
                        )
                    with progress_lock:
                        submitted += 1
                        if prior is not None:
                            counts[prior] += 1
                            completed_task_baseline += 1
                    if prior is not None:
                        continue
                    future = pool.submit(
                        run_chain,
                        chain,
                        backend,
                        store,
                        config.runtime.max_revisions,
                    )
                    futures[future] = chain.task_id
                    with progress_lock:
                        active_tasks += 1
                    return True

            for _ in range(workers * 2):
                if not submit_one():
                    break
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future, None)
                    try:
                        status = future.result()
                    except Exception:  # noqa: BLE001
                        status = "failed"
                    with progress_lock:
                        active_tasks -= 1
                        counts[status] += 1
                    print(
                        canonical_json(
                            {
                                "completed_tasks": sum(counts.values()),
                                "status": status,
                            }
                        )
                    )
                    submit_one()
        if max_tasks is None and submitted == config.expected_tasks:
            backend.write_localization_gate()
        terminal_state = _terminal_state_for_run(
            max_tasks=max_tasks,
            submitted_tasks=submitted,
            expected_tasks=config.expected_tasks,
            failed_tasks=counts["failed"],
        )
    except KeyboardInterrupt:
        terminal_state = "stopped_checkpoint_resumable"
        raise
    finally:
        monitor_stop.set()
        monitor.join(timeout=3.0)
        write_public_status(terminal_state)
        backend.close()
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "control_run_id": control_run_id,
        "checkpoint_id": identity["checkpoint_id"],
        "config_sha256": identity["config_sha256"],
        "execution_lock_sha256": identity["execution_lock_sha256"],
        "source_bank_manifest_sha256": identity["source_bank_manifest_sha256"],
        "resume_mode": resume,
        "task_count_requested": submitted,
        "counts": counts,
        "checkpoint": store.events.relative_to(PROJECT_ROOT).as_posix(),
        "reasoning_effort": "max",
        "real_sandbox_required": True,
        "zh_cn_localization_records": backend.localization_count,
        "content_bearing_records_logged_to_console": False,
    }
    _atomic_json(config.runtime.output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preflight or explicitly confirmed SWE-bench CC Switch run"
    )
    parser.add_argument(
        "--config",
        default="configs/data/swebench_five_stage.ccswitch.yaml",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Start provider routes, repository materialization, and real OpenCode sandboxes",
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--control-run-id",
        help="Content-free process-attempt identity supplied by the local control plane",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the checkpoint bound to the same config and execution lock",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = CoordinatorConfig.load(
            _project_path(PROJECT_ROOT, args.config, "config")
        )
        report = offline_preflight(config)
        if not args.confirm_live:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if report.get("live_start_allowed") is not True:
            reason = report.get("reason_code")
            if not isinstance(reason, str) or not reason:
                reason = "execution_contract_not_ready"
            raise CoordinatorError(reason)
        if args.concurrency is not None and args.concurrency < 1:
            raise CoordinatorError("invalid_cli_concurrency")
        if args.max_tasks is not None and args.max_tasks < 1:
            raise CoordinatorError("invalid_cli_max_tasks")
        if not args.resume and (args.max_tasks != 1 or args.concurrency != 1):
            raise CoordinatorError("generic_train_representative_probe_required")
        if not isinstance(args.control_run_id, str) or not _CONTROL_RUN_ID.fullmatch(
            args.control_run_id
        ):
            raise CoordinatorError("invalid_control_run_id")
        result = run_live(
            config,
            control_run_id=args.control_run_id,
            resume=bool(args.resume),
            concurrency=args.concurrency,
            max_tasks=args.max_tasks,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["counts"]["failed"] == 0 else 3
    except CoordinatorError as error:
        print(f"SWE-bench coordinator refused: {error.code}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        print("SWE-bench coordinator refused: invalid_local_artifact", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
