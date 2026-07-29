"""Additive vNext teacher-alignment acceleration primitives.

The historical v1/v2 controllers remain closed.  This module keeps their WAL
entry format readable by the independent full-replay auditor while replacing
online full-history authentication with a held sequence/chain-tip cursor.
Only body-free identities, counts, usage, and authenticated strata enter the
vNext checkpoints and freeze inventories.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch
from . import gemma3_chat_unbalanced_v2_live_controller as closed
from . import gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2 as selector
from . import teacher as teacher_transport


CONFIG_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.config.v1"
)
CONTRACT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.contract.v1"
)
RECEIPT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.job-receipt.v1"
)
CHECKPOINT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.delta-checkpoint.v1"
)
THROUGHPUT_TELEMETRY_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "provider-throughput-telemetry.v1"
)
THROUGHPUT_MONITOR_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "provider-throughput-monitor-snapshot.v1"
)
_THROUGHPUT_SEMANTICS = (
    "provider_terminal_usage_rolling_window; provider_reported_usage_only; "
    "not_committed_or_accepted_counts; "
    "idle_reads_do_not_advance_observed_at_or_synthesize_usage; "
    "any_unknown_row_or_forbidden_restart_makes_rates_UNKNOWN"
)
_THROUGHPUT_DISK_KEYS = frozenset(
    {
        "schema_version",
        "provider_input_tokens_per_second",
        "provider_output_tokens_per_second",
        "window_seconds",
        "window_limit_seconds",
        "terminal_jobs",
        "observed_at",
        "exact",
        "exact_rows",
        "unknown_rows",
        "error",
        "semantics",
        "controller_run_id",
        "content_retained",
        "raw_token_ids_retained",
        "credential_retained",
    }
)
ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "isolated-transport-frame.v1"
)
_ISOLATED_TRANSPORT_MAX_BYTES = 16 * 1024 * 1024
_ISOLATED_TRANSPORT_MIN_DEADLINE_SECONDS = 0.3
_ISOLATED_TRANSPORT_CLEANUP_RESERVE_SECONDS = 0.25
FREEZE_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.freeze-attestation.v1"
)
INVENTORY_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "body-free-inventory-row.v1"
)
TRAINING_INVENTORY_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "training-record-inventory-row.v1"
)
COMMITTED_SHARDS_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "committed-shards-manifest.v2"
)
TEACHER_RECORD_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
)
TEACHER_RECORD_NAMESPACE = "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
TEACHER_TRAINING_SHARD_NAME = "teacher_training_shard-00000.jsonl"
TRAINING_RECORD_INVENTORY_NAME = "training_record_inventory.jsonl"
TEACHER_TRAINING_SHARD_MAX_BYTES_EXCLUSIVE = 50 * 1024 * 1024
IMPLEMENTATION_PATH = Path(__file__).absolute()
DEFAULT_CONFIG_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json"
)
DEFAULT_TRANSITION_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_transition_v1.schema.json"
)
DEFAULT_EXTERNAL_ATTESTATION_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_"
    "external_attestation_v1.schema.json"
)
DEFAULT_BULK_PROFILE_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "teacher_alignment.bulk_c30.yaml"
)
LAUNCH_STAGING_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "external-attestation-staging.v1"
)
LAUNCH_AUTHORIZATION_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.launch-authorization.v1"
)
_SAFE_STOP_REASON = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_STYLE_ROLES = frozenset({"humor", "serious", "angry"})
_JSON_KEYS = {
    "humor": frozenset({"final_answer"}),
    "serious": frozenset({"final_answer"}),
    "angry": frozenset({"final_answer"}),
    "tool": frozenset({"final_answer", "tool_call", "evidence_ids"}),
    "review": frozenset({"verdict", "faults", "correction"}),
    "router": frozenset({"route", "plan", "stop"}),
}
_COVERAGE_SELECTION_KINDS = frozenset(
    {"full_core", "tool_review_depth", "identity", "router"}
)
CoverageKey = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]
_ALLOWED_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"reserved"}),
    "reserved": frozenset({"dispatching"}),
    "dispatching": frozenset({"retryable", "uncertain", "committed", "rejected"}),
    "retryable": frozenset({"reserved"}),
    "uncertain": frozenset(),
    "committed": frozenset(),
    "rejected": frozenset(),
}
_LEGACY_CONTROLLER_MARKERS = (
    "gemma3_chat_unbalanced_v2_live_controller",
    "gemma3_chat_unbalanced_v2_integrated_controller_v2",
)
_VNEXT_CONTROLLER_MARKERS = ("gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1",)
_CONTROLLER_INVENTORY_SCOPE = (
    "matching_controller_entrypoints_excluding_authenticated_candidate_process_tree"
)
_CONTROLLER_INVENTORY_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "trusted-controller-inventory.v1"
)
JOB_PROVENANCE_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext.job-provenance.v1"
)
PRODUCER_LAUNCH_REVIEWED_PATHS = (
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_contract_v2.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_metadata_overlay_v2.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_preimages_v2.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_selector_v2.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "vnext_checkpoint_v1.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_contract_v1.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "vnext_external_attestation_v1.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "vnext_producer_release_v1.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "vnext_receipt_v1.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
    "vnext_transition_v1.schema.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json",
    "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.schema.json",
    "scripts/data/run_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
    "scripts/observability/distillation_dashboard.py",
    "scripts/research/benchmark_gemma3_teacher_alignment_vnext_wal_v1.py",
    "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
    "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
    "tests/test_distillation_dashboard.py",
    "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
    "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
)
RELEASE_KIND_STAGE_MAP = {
    "producer_launch_L": [
        "code_components",
        "schemas",
        "contracts",
    ],
    "producer_payload_P": [
        "selection_overlay",
        "selector_preimages",
        "candidate_inventory",
        "selected_inventory",
        "accepted_inventory",
        "rejected_inventory",
        "quarantine_inventory",
        "checkpoints",
        "committed_shards",
        "teacher_training_shard",
        "training_record_inventory",
    ],
    "producer_final_R": [
        "freeze_attestation",
        "external_attestation",
    ],
    "consumer_code_P": [
        "consumer_code",
        "consumer_schemas",
        "consumer_contract",
    ],
    "consumer_pin_C": ["consumer_pin_only"],
}
PRODUCER_RELEASE_ROOT = (
    "data/gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "teacher_alignment_vnext_v1/release_v1"
)
_PRODUCER_PAYLOAD_P_FILES = (
    "accepted_inventory.jsonl",
    "accepted_inventory.jsonl.sha256",
    "authenticated_metadata_overlay.jsonl",
    "authenticated_metadata_overlay.jsonl.sha256",
    "authenticated_metadata_overlay_manifest.json",
    "authenticated_metadata_overlay_manifest.json.sha256",
    "candidate_inventory.jsonl",
    "candidate_inventory.jsonl.sha256",
    "checkpoint_manifest.json",
    "checkpoint_manifest.json.sha256",
    "checkpoints_inventory.json",
    "checkpoints_inventory.json.sha256",
    "committed_shards_manifest.json",
    "committed_shards_manifest.json.sha256",
    "producer_launch_inventory.json",
    "producer_launch_inventory.json.sha256",
    "quarantine_inventory.jsonl",
    "quarantine_inventory.jsonl.sha256",
    "rejected_inventory.jsonl",
    "rejected_inventory.jsonl.sha256",
    "selected_inventory.jsonl",
    "selected_inventory.jsonl.sha256",
    "selection_manifest.json",
    "selection_manifest.json.sha256",
    "selection_preimage.json",
    "selection_preimage.json.sha256",
    "selector_preimage_manifest.json",
    "selector_preimage_manifest.json.sha256",
    "teacher_training_shard-00000.jsonl",
    "teacher_training_shard-00000.jsonl.sha256",
    "training_record_inventory.jsonl",
    "training_record_inventory.jsonl.sha256",
)
_PRODUCER_R_RELEASE_FILES = (
    "external_attestation.json",
    "external_attestation.json.sha256",
    "freeze_attestation.json",
    "freeze_attestation.json.sha256",
)
PRODUCER_PAYLOAD_P_PATHS = tuple(
    f"{PRODUCER_RELEASE_ROOT}/{name}" for name in _PRODUCER_PAYLOAD_P_FILES
)
PRODUCER_R_RELEASE_PATHS = tuple(
    f"{PRODUCER_RELEASE_ROOT}/{name}" for name in _PRODUCER_R_RELEASE_FILES
)
PRODUCER_L_BASE_DIFF_STATUS = tuple(
    sorted(
        [
            (".gitattributes", "M"),
            *(
                (
                    path,
                    (
                        "M"
                        if path
                        in {
                            "scripts/observability/distillation_dashboard.py",
                            "tests/test_distillation_dashboard.py",
                        }
                        else "A"
                    ),
                )
                for path in PRODUCER_LAUNCH_REVIEWED_PATHS
            ),
        ]
    )
)
PRODUCER_PAYLOAD_P_DIFF_STATUS = tuple((path, "A") for path in PRODUCER_PAYLOAD_P_PATHS)
_EXPECTED_ROUTER_RECORD_CELLS = {
    "humor": {"en": 14, "zh": 13},
    "serious": {"en": 13, "zh": 13},
    "angry": {"en": 13, "zh": 14},
}


@dataclass(frozen=True)
class VNextPolicy:
    """Closed online policy loaded from the versioned config."""

    path: Path
    physical_sha256: str
    schema_path: Path
    contract_sha256: str
    schema_sha256: str
    implementation_path: Path
    implementation_sha256: str
    checkpoint_schema_path: Path
    checkpoint_schema_sha256: str
    receipt_schema_path: Path
    receipt_schema_sha256: str
    selector_contract_sha256: str
    selector_schema_path: Path
    selector_schema_sha256: str
    selector_implementation_sha256: str
    teacher_record_schema_path: Path
    teacher_record_schema_sha256: str
    transition_schema_path: Path
    transition_schema_sha256: str
    base_profile_path: Path
    base_profile_sha256: str
    output_root_parent: Path | None
    old_canonical_root: Path
    old_archive_parent: Path
    controller_lease_name: str
    rejection_window: int
    rejection_min_samples: int
    rejection_max_rate: float
    claims: dict[str, Any]
    consumer_code_P: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalCursor:
    sequence: int
    chain_tip_sha256: str
    genesis_sha256: str
    job_heads: dict[str, str | None] = field(default_factory=dict)


@dataclass
class StopAfterGroupLatch:
    requested: bool = False
    effective: bool = False
    reason_code: str | None = None
    requested_at_sequence: int | None = None
    effective_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class FullReplayAudit:
    stage: str
    wal_entries: int
    terminal_events: int
    phase_receipts: int
    chain_tip_sha256: str


class StopAfterGroup(batch.AdapterError):
    """A safe stop that became effective only after a durable group checkpoint."""

    def __init__(self, reason_code: str = "vnext_stop_after_group") -> None:
        super().__init__(reason_code)


class RejectionRateFuse:
    """Latch when a bounded terminal window indicates systematic rejection."""

    def __init__(
        self,
        *,
        window: int,
        min_samples: int,
        max_rate: float,
    ) -> None:
        if (
            window < 1
            or min_samples < 1
            or min_samples > window
            or not math.isfinite(max_rate)
            or not 0.0 <= max_rate <= 1.0
        ):
            raise batch.AdapterError("vnext_rejection_fuse_config_invalid")
        self.window = window
        self.min_samples = min_samples
        self.max_rate = max_rate
        self._terminal: deque[bool] = deque(maxlen=window)
        self.latched = False

    def observe(self, rejected: Sequence[bool]) -> dict[str, Any]:
        self._terminal.extend(bool(item) for item in rejected)
        samples = len(self._terminal)
        rejected_count = sum(self._terminal)
        rate = rejected_count / samples if samples else 0.0
        if samples >= self.min_samples and rate > self.max_rate:
            self.latched = True
        return {
            "window": self.window,
            "samples": samples,
            "rejected": rejected_count,
            "rate": rate,
            "max_rate": self.max_rate,
            "latched": self.latched,
        }


@dataclass(frozen=True)
class _ThroughputSample:
    observed_monotonic: float
    input_tokens: int | None
    output_tokens: int | None


class ProviderThroughputAccumulator:
    """O(1)-amortized provider-usage telemetry at terminal boundaries."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        observed_at_clock: Callable[[], str] = batch._iso,
        window_limit_seconds: float = 60.0,
    ) -> None:
        if not math.isfinite(window_limit_seconds) or window_limit_seconds <= 0:
            raise batch.AdapterError("vnext_throughput_window_invalid")
        self._monotonic_clock = monotonic_clock
        self._observed_at_clock = observed_at_clock
        self._window_limit_seconds = float(window_limit_seconds)
        self._samples: deque[_ThroughputSample] = deque()
        self._input_tokens = 0
        self._output_tokens = 0
        self._exact_rows = 0
        self._unknown_rows = 0
        self._started_monotonic = self._read_monotonic()
        self._last_monotonic: float | None = None
        self._observed_at: str | None = None
        self._sticky_error: str | None = None

    def _read_monotonic(self) -> float:
        value = self._monotonic_clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise batch.AdapterError("vnext_throughput_clock_invalid")
        return float(value)

    @staticmethod
    def _usage_tokens(
        usage: Mapping[str, Any] | None,
    ) -> tuple[int, int] | None:
        if not isinstance(usage, Mapping):
            return None
        values = tuple(
            usage.get(name)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            return None
        input_tokens, output_tokens, total_tokens = values
        if total_tokens != input_tokens + output_tokens:
            return None
        return input_tokens, output_tokens

    def mark_restart_forbidden(self) -> dict[str, Any]:
        """Keep telemetry UNKNOWN when the closed policy forbids resume."""

        self._sticky_error = "cross_process_resume_forbidden"
        self._observed_at = self._observed_at_clock()
        return self.snapshot()

    def observe_terminal(
        self,
        usage: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        now = self._read_monotonic()
        if self._last_monotonic is not None and now < self._last_monotonic:
            self._sticky_error = "monotonic_clock_regressed"
        self._last_monotonic = now
        self._observed_at = self._observed_at_clock()
        tokens = self._usage_tokens(usage)
        if tokens is None:
            sample = _ThroughputSample(now, None, None)
            self._unknown_rows += 1
        else:
            input_tokens, output_tokens = tokens
            sample = _ThroughputSample(
                now,
                input_tokens,
                output_tokens,
            )
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._exact_rows += 1
        self._samples.append(sample)
        cutoff = now - self._window_limit_seconds
        while self._samples and self._samples[0].observed_monotonic < cutoff:
            expired = self._samples.popleft()
            if expired.input_tokens is None:
                self._unknown_rows -= 1
            else:
                self._input_tokens -= expired.input_tokens
                self._output_tokens -= expired.output_tokens or 0
                self._exact_rows -= 1
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        if self._last_monotonic is None:
            window_seconds = 0.0
        else:
            window_seconds = max(
                0.0,
                self._last_monotonic
                - max(
                    self._started_monotonic,
                    self._last_monotonic - self._window_limit_seconds,
                ),
            )
        error = self._sticky_error
        if error is None and self._unknown_rows:
            error = "provider_usage_missing_or_invalid"
        if error is None and not self._samples:
            error = "no_terminal_usage_observed"
        if error is None and window_seconds <= 0:
            error = "throughput_window_nonpositive"
        exact = error is None
        input_rate: float | str = "UNKNOWN"
        output_rate: float | str = "UNKNOWN"
        if exact:
            input_rate = round(self._input_tokens / window_seconds, 6)
            output_rate = round(self._output_tokens / window_seconds, 6)
        return {
            "schema_version": THROUGHPUT_TELEMETRY_SCHEMA_VERSION,
            "provider_input_tokens_per_second": input_rate,
            "provider_output_tokens_per_second": output_rate,
            "window_seconds": round(window_seconds, 6),
            "window_limit_seconds": self._window_limit_seconds,
            "terminal_jobs": len(self._samples),
            "observed_at": self._observed_at,
            "exact": exact,
            "exact_rows": self._exact_rows,
            "unknown_rows": self._unknown_rows,
            "error": error,
            "semantics": _THROUGHPUT_SEMANTICS,
            "content_retained": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        }


@dataclass
class AbsoluteDeadlineCompatibleTeacher(batch.CompatibleTeacher):
    """One-attempt-per-process transport with a hard parent deadline.

    An async context manager can ignore cancellation while unwinding.  Each
    paid wire attempt therefore runs inside a child process.  The credential
    and request travel only over an anonymous stdin pipe; the fixed command,
    sanitized environment, and filesystem never contain them.  On timeout the
    parent kills and joins the child and drains/closes its pipes before return.
    """

    isolated_worker_command_factory: Callable[[], Sequence[str]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _active_isolated_pids: set[int] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _last_isolated_pid: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        idempotency_key: str | None = None,
    ) -> str:
        if (
            self.protocol != "openai"
            or self.fallback_protocol is not None
            or self.stream_openai
        ):
            raise batch.TeacherError(
                "vnext transport requires non-streaming OpenAI Chat without fallback"
            )
        self._completion_context.set(None)
        validated_key = teacher_transport._validate_idempotency_key(idempotency_key)
        if self.wall_clock_deadline_seconds < _ISOLATED_TRANSPORT_MIN_DEADLINE_SECONDS:
            self._set_attempt_context(wire_attempts=0, retry_reasons=())
            raise batch.ClientDeadlineExceeded(self.wall_clock_deadline_seconds)
        deadline_at = time.monotonic() + self.wall_clock_deadline_seconds
        try:
            return await self._complete_before_deadline(
                system=system,
                user=user,
                idempotency_key=validated_key,
                deadline_at=deadline_at,
            )
        except (TimeoutError, asyncio.TimeoutError):
            raise batch.ClientDeadlineExceeded(
                self.wall_clock_deadline_seconds
            ) from None

    @property
    def active_isolated_process_count(self) -> int:
        return len(self._active_isolated_pids)

    def _worker_command(self) -> tuple[str, ...]:
        command = (
            tuple(self.isolated_worker_command_factory())
            if self.isolated_worker_command_factory is not None
            else (
                sys.executable,
                str(
                    batch.REPO_ROOT / "scripts/data/"
                    "run_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py"
                ),
                "--isolated-transport-worker",
            )
        )
        if not command or any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or "\r" in item
            or "\n" in item
            for item in command
        ):
            raise batch.TeacherError("vnext isolated transport command is invalid")
        return command

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed = (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
        )
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment["PYTHONPATH"] = str(batch.REPO_ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    async def _isolated_wire_attempt(
        self,
        frame: Mapping[str, Any],
        *,
        deadline_at: float,
    ) -> dict[str, Any]:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        raw = _canonical_bytes(frame, newline=True)
        if len(raw) > _ISOLATED_TRANSPORT_MAX_BYTES:
            raise batch.TeacherError(
                "vnext isolated transport request frame is too large"
            )
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = await asyncio.create_subprocess_exec(
            *self._worker_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(batch.REPO_ROOT),
            env=self._worker_environment(),
            creationflags=creation_flags,
        )
        self._last_isolated_pid = process.pid
        self._active_isolated_pids.add(process.pid)
        communication = asyncio.create_task(process.communicate(raw))
        try:
            done, _ = await asyncio.wait(
                {communication},
                timeout=max(0.0, deadline_at - time.monotonic()),
            )
            if communication not in done:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                # OS termination is the explicit hard boundary.  Await the
                # same communicate task to join and close every pipe/handle;
                # it is never detached or left pending.
                await communication
                raise asyncio.TimeoutError
            stdout, _ = communication.result()
        except BaseException:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            if not communication.done():
                await communication
            raise
        finally:
            if process.returncode is None:
                await process.wait()
            self._active_isolated_pids.discard(process.pid)
        if process.returncode != 0 or len(stdout) > _ISOLATED_TRANSPORT_MAX_BYTES:
            raise batch.TeacherError("vnext isolated teacher transport failed")
        value = _strict_json_mapping(
            stdout,
            reason="vnext_isolated_transport_response_invalid",
        )
        if value.get("schema_version") != ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION:
            raise batch.TeacherError(
                "vnext isolated teacher transport response was invalid"
            )
        return value

    def _credential(self) -> str:
        if self.credential_provider is None:
            raise batch.TeacherError("vnext credential memory slot is unavailable")
        try:
            credential = self.credential_provider()
        except Exception:
            raise batch.TeacherError(
                "vnext credential memory slot is unavailable"
            ) from None
        if (
            not isinstance(credential, str)
            or not credential
            or credential != credential.strip()
            or any(marker in credential for marker in ("\x00", "\r", "\n"))
        ):
            raise batch.TeacherError("vnext credential memory slot is unavailable")
        return credential

    def _request_parts(
        self,
        *,
        system: str,
        user: str,
        idempotency_key: str | None,
        credential: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = teacher_transport._openai_endpoint(self.base_url)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_enabled:
            payload["reasoning_effort"] = self.thinking_effort
        else:
            payload["temperature"] = self.temperature
            if self.responses_thinking_policy == "explicit_disabled":
                payload["thinking"] = {"type": "disabled"}
        if self._chat_response_format_json is not None:
            payload["response_format"] = json.loads(self._chat_response_format_json)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credential}",
            "User-Agent": self.user_agent,
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        return endpoint, headers, payload

    @staticmethod
    def _retryable_async_transport(error: Exception) -> bool:
        return (
            isinstance(error, (ConnectionError, OSError))
            or type(error).__module__.split(".", 1)[0] == "httpx"
        )

    def _set_attempt_context(
        self,
        *,
        wire_attempts: int,
        retry_reasons: Sequence[str],
        completion: Mapping[str, Any] | None = None,
    ) -> None:
        self._completion_context.set(
            {
                "protocol": "openai",
                "base_url": self.base_url,
                "completion": dict(completion) if completion else None,
                "attempts": {
                    "wire_attempts": wire_attempts,
                    "retry_count": max(0, wire_attempts - 1),
                    "max_retries": self.max_retries,
                    "retry_reasons": list(retry_reasons),
                },
            }
        )

    async def _complete_before_deadline(
        self,
        *,
        system: str,
        user: str,
        idempotency_key: str | None,
        deadline_at: float,
    ) -> str:
        credential = self._credential()
        endpoint, headers, payload = self._request_parts(
            system=system,
            user=user,
            idempotency_key=idempotency_key,
            credential=credential,
        )
        retry_reasons: list[str] = []
        last_reason = "transport_failure"
        self._set_attempt_context(wire_attempts=0, retry_reasons=())
        wire_deadline = deadline_at - _ISOLATED_TRANSPORT_CLEANUP_RESERVE_SECONDS
        for attempt in range(self.max_retries + 1):
            if wire_deadline - time.monotonic() <= 0:
                raise asyncio.TimeoutError
            self._budget.reserve_request()
            self._set_attempt_context(
                wire_attempts=attempt + 1,
                retry_reasons=retry_reasons,
            )
            response = await self._isolated_wire_attempt(
                {
                    "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
                    "kind": "request",
                    "endpoint": endpoint,
                    "headers": headers,
                    "payload": payload,
                    "timeout_seconds": min(
                        self.timeout_seconds,
                        max(0.001, wire_deadline - time.monotonic()),
                    ),
                },
                deadline_at=wire_deadline,
            )
            kind = response.get("kind")
            if kind == "timeout":
                raise asyncio.TimeoutError
            if kind == "transport_error":
                if response.get("retryable") is not True:
                    raise batch.TeacherError("vnext isolated teacher transport failed")
                last_reason = "transport_failure"
            elif kind == "response":
                status = response.get("status_code")
                response_headers = response.get("headers")
                if not isinstance(status, int) or not isinstance(
                    response_headers,
                    Mapping,
                ):
                    raise batch.TeacherError(
                        "vnext isolated teacher response was invalid"
                    )
                retry_after = teacher_transport._parse_retry_after(
                    response_headers.get("Retry-After")
                )
                if status == 429:
                    raise batch.RateLimitError(retry_after)
                if not 200 <= status < 300:
                    if not teacher_transport._is_retryable_http_status(status):
                        raise batch.TeacherError(
                            "vnext teacher endpoint returned a non-retryable status"
                        )
                    last_reason = f"http_{status}"
                else:
                    body = response.get("body")
                    if not isinstance(body, Mapping):
                        raise batch.TeacherError(
                            "vnext teacher response was not valid JSON"
                        )
                    content, _ = teacher_transport._response_content(
                        "openai",
                        body,
                    )
                    output_tokens, completion = (
                        teacher_transport._openai_chat_completion(
                            body,
                            current_credential=credential,
                        )
                    )
                    if credential in content:
                        raise batch.TeacherError(
                            "vnext teacher response contained current credential"
                        )
                    self._budget.add_output(output_tokens)
                    self._set_attempt_context(
                        wire_attempts=attempt + 1,
                        retry_reasons=retry_reasons,
                        completion=completion,
                    )
                    return content
            else:
                raise batch.TeacherError(
                    "vnext isolated teacher transport response was invalid"
                )
            if attempt >= self.max_retries:
                break
            retry_reasons.append(last_reason)
            self._set_attempt_context(
                wire_attempts=attempt + 1,
                retry_reasons=retry_reasons,
            )
            remaining = wire_deadline - time.monotonic()
            delay = teacher_transport._retry_delay_seconds(attempt, 0.0)
            if remaining <= delay:
                raise asyncio.TimeoutError
            await asyncio.sleep(delay)
        raise batch.TeacherError("vnext teacher request failed after bounded retries")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _physical_read(
    path: Path,
    *,
    reason: str,
    stop: Path,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    """Read one regular file through a no-follow handle and recheck its name."""

    candidate = path.absolute()
    boundary = stop.absolute()
    try:
        candidate.relative_to(boundary)
    except ValueError:
        raise batch.AdapterError(reason) from None
    closed._reject_reparse_chain(candidate, stop=boundary)
    try:
        before = candidate.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if (
        candidate.is_symlink()
        or bool(int(getattr(before, "st_file_attributes", 0)) & 0x400)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > maximum_bytes
    ):
        raise batch.AdapterError(reason)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise batch.AdapterError(reason) from error
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise batch.AdapterError(reason)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise batch.AdapterError(reason)
        after_handle = os.fstat(descriptor)
        if identity != (
            after_handle.st_dev,
            after_handle.st_ino,
            after_handle.st_size,
            after_handle.st_mtime_ns,
        ):
            raise batch.AdapterError(reason)
    finally:
        os.close(descriptor)
    try:
        after_name = candidate.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if identity != (
        after_name.st_dev,
        after_name.st_ino,
        after_name.st_size,
        after_name.st_mtime_ns,
    ):
        raise batch.AdapterError(reason)
    closed._reject_reparse_chain(candidate, stop=boundary)
    return b"".join(chunks)


def _named_entry_absent(path: Path, *, stop: Path) -> bool:
    candidate = path.absolute()
    boundary = stop.absolute()
    try:
        candidate.relative_to(boundary)
    except ValueError:
        raise batch.AdapterError("vnext_named_entry_scope_invalid") from None
    closed._reject_reparse_chain(candidate.parent, stop=boundary)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return True
    except OSError as error:
        raise batch.AdapterError("vnext_named_entry_identity_unknown") from error
    return False


def _named_regular_directory(path: Path, *, stop: Path) -> bool:
    candidate = path.absolute()
    if _named_entry_absent(candidate, stop=stop):
        return False
    try:
        identity = candidate.lstat()
    except OSError as error:
        raise batch.AdapterError("vnext_named_entry_identity_unknown") from error
    return (
        stat.S_ISDIR(identity.st_mode)
        and not candidate.is_symlink()
        and not bool(int(getattr(identity, "st_file_attributes", 0)) & 0x400)
    )


def _canonical_bytes(value: Mapping[str, Any], *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_BINARY", 0)),
            0o600,
        )
    except FileExistsError as error:
        raise batch.AdapterError("vnext_append_identity_collision") from error
    try:
        batch._write_all(descriptor, value)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    batch._fsync_parent(path.parent)


def _strict_json_mapping(raw: bytes, *, reason: str) -> dict[str, Any]:
    value = closed._strict_json(raw, reason=reason)
    if not isinstance(value, dict):
        raise batch.AdapterError(reason)
    return value


def _physical_jsonl(
    path: Path,
    *,
    id_field: str,
    stop: Path,
    reason: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _physical_read(path, reason=reason, stop=stop)
    if raw and not raw.endswith(b"\n"):
        raise batch.AdapterError(reason)
    rows = [_strict_json_mapping(line, reason=reason) for line in raw.splitlines()]
    seen: set[str] = set()
    for row in rows:
        identity = row.get(id_field)
        if not isinstance(identity, str) or identity in seen:
            raise batch.AdapterError(reason)
        seen.add(identity)
    return rows


def _repo_file(raw: Any, *, reason: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise batch.AdapterError(reason)
    candidate = Path(raw)
    path = (
        candidate.absolute()
        if candidate.is_absolute()
        else (batch.REPO_ROOT / candidate).absolute()
    )
    try:
        path.relative_to(batch.REPO_ROOT.absolute())
    except ValueError:
        raise batch.AdapterError(reason) from None
    closed._reject_reparse_chain(path, stop=batch.REPO_ROOT.absolute())
    if not path.is_file():
        raise batch.AdapterError(reason)
    return path


def _repo_path(raw: Any, *, reason: str) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise batch.AdapterError(reason)
    candidate = Path(raw)
    path = (
        candidate.absolute()
        if candidate.is_absolute()
        else (batch.REPO_ROOT / candidate).absolute()
    )
    try:
        path.relative_to(batch.REPO_ROOT.absolute())
    except ValueError:
        raise batch.AdapterError(reason) from None
    return path


def _verified_file(
    path_value: Any,
    digest_value: Any,
    *,
    reason: str,
) -> tuple[Path, bytes]:
    path = _repo_file(path_value, reason=reason)
    if (
        not isinstance(digest_value, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
    ):
        raise batch.AdapterError(reason)
    raw = _physical_read(
        path,
        reason=reason,
        stop=batch.REPO_ROOT.absolute(),
    )
    if not hmac.compare_digest(_sha256_bytes(raw), digest_value):
        raise batch.AdapterError(reason)
    return path, raw


def load_vnext_policy(path: Path = DEFAULT_CONFIG_PATH) -> VNextPolicy:
    """Load and hash-bind only the additive vNext contract and closed ancestors."""

    config_path = _repo_file(path.as_posix(), reason="vnext_config_path_invalid")
    raw = _physical_read(
        config_path,
        reason="vnext_config_snapshot_drift",
        stop=batch.REPO_ROOT.absolute(),
    )
    value = _strict_json_mapping(raw, reason="vnext_config_invalid")
    schema_path, schema_raw = _verified_file(
        value.get("schema_path"),
        value.get("schema_sha256"),
        reason="vnext_config_schema_identity_drift",
    )
    schema = _strict_json_mapping(schema_raw, reason="vnext_config_schema_invalid")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("vnext_config_schema_mismatch") from error
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise batch.AdapterError("vnext_config_version_drift")
    _verified_file(
        value.get("contract_path"),
        value.get("contract_sha256"),
        reason="vnext_contract_identity_drift",
    )
    implementation_path, implementation_raw = _verified_file(
        value.get("implementation_path"),
        value.get("implementation_sha256"),
        reason="vnext_implementation_identity_drift",
    )
    if implementation_path != IMPLEMENTATION_PATH:
        raise batch.AdapterError("vnext_implementation_path_drift")
    checkpoint_schema_path, _ = _verified_file(
        value.get("checkpoint_schema_path"),
        value.get("checkpoint_schema_sha256"),
        reason="vnext_checkpoint_schema_identity_drift",
    )
    receipt_schema_path, _ = _verified_file(
        value.get("receipt_schema_path"),
        value.get("receipt_schema_sha256"),
        reason="vnext_receipt_schema_identity_drift",
    )
    selector_schema_path: Path | None = None
    teacher_record_schema_path: Path | None = None
    transition_schema_path: Path | None = None
    for path_key, sha_key, reason in (
        (
            "selector_contract_path",
            "selector_contract_sha256",
            "vnext_selector_contract_identity_drift",
        ),
        (
            "selector_schema_path",
            "selector_schema_sha256",
            "vnext_selector_schema_identity_drift",
        ),
        (
            "selector_implementation_path",
            "selector_implementation_sha256",
            "vnext_selector_implementation_identity_drift",
        ),
        (
            "teacher_record_schema_path",
            "teacher_record_schema_sha256",
            "vnext_teacher_record_schema_identity_drift",
        ),
        (
            "transition_schema_path",
            "transition_schema_sha256",
            "vnext_transition_schema_identity_drift",
        ),
    ):
        verified_path, _ = _verified_file(
            value.get(path_key),
            value.get(sha_key),
            reason=reason,
        )
        if path_key == "selector_schema_path":
            selector_schema_path = verified_path
        elif path_key == "teacher_record_schema_path":
            teacher_record_schema_path = verified_path
        elif path_key == "transition_schema_path":
            transition_schema_path = verified_path
    for anchor in value["closed_ancestors"]:
        _verified_file(
            anchor.get("path"),
            anchor.get("sha256"),
            reason="vnext_closed_ancestor_identity_drift",
        )
    launch = value["launch"]
    base_profile_path, _ = _verified_file(
        launch.get("base_profile_path"),
        launch.get("base_profile_sha256"),
        reason="vnext_base_profile_identity_drift",
    )
    if (
        launch.get("output_root_parent") is not None
        or launch.get("runtime_root_scope") != "external_absolute_parent_from_cli"
    ):
        raise batch.AdapterError("vnext_output_root_parent_invalid")
    output_root_parent = None
    old_canonical_root = _repo_path(
        launch.get("old_canonical_root"),
        reason="vnext_old_canonical_root_invalid",
    )
    old_archive_parent = _repo_path(
        launch.get("old_archive_parent"),
        reason="vnext_old_archive_parent_invalid",
    )
    fuse = value["rejection_fuse"]
    if (
        selector_schema_path is None
        or teacher_record_schema_path is None
        or transition_schema_path is None
    ):
        raise batch.AdapterError("vnext_schema_identity_drift")
    return VNextPolicy(
        path=config_path,
        physical_sha256=_sha256_bytes(raw),
        schema_path=schema_path,
        contract_sha256=str(value["contract_sha256"]),
        schema_sha256=str(value["schema_sha256"]),
        implementation_path=implementation_path,
        implementation_sha256=_sha256_bytes(implementation_raw),
        checkpoint_schema_path=checkpoint_schema_path,
        checkpoint_schema_sha256=str(value["checkpoint_schema_sha256"]),
        receipt_schema_path=receipt_schema_path,
        receipt_schema_sha256=str(value["receipt_schema_sha256"]),
        selector_contract_sha256=str(value["selector_contract_sha256"]),
        selector_schema_path=selector_schema_path,
        selector_schema_sha256=str(value["selector_schema_sha256"]),
        selector_implementation_sha256=str(value["selector_implementation_sha256"]),
        teacher_record_schema_path=teacher_record_schema_path,
        teacher_record_schema_sha256=str(value["teacher_record_schema_sha256"]),
        transition_schema_path=transition_schema_path,
        transition_schema_sha256=str(value["transition_schema_sha256"]),
        base_profile_path=base_profile_path,
        base_profile_sha256=str(launch["base_profile_sha256"]),
        output_root_parent=output_root_parent,
        old_canonical_root=old_canonical_root,
        old_archive_parent=old_archive_parent,
        controller_lease_name=str(launch["controller_lease_name"]),
        rejection_window=int(fuse["window"]),
        rejection_min_samples=int(fuse["min_samples"]),
        rejection_max_rate=float(fuse["max_rate"]),
        claims=dict(value["claims"]),
        consumer_code_P=dict(value["consumer_code_P"]),
    )


def _tool_family(source: batch.SourceRecord) -> str | None:
    tools = source.teacher_contract["allowed_tools"]
    if source.role == "tool":
        if not isinstance(tools, list) or len(tools) != 1:
            raise batch.AdapterError("vnext_tool_family_cardinality_invalid")
        tool = tools[0]
        if not isinstance(tool, str) or batch.SAFE_ID_RE.fullmatch(tool) is None:
            raise batch.AdapterError("vnext_tool_family_invalid")
        return tool
    if tools:
        raise batch.AdapterError("vnext_non_tool_family_present")
    return None


def _normalized_language(value: str) -> str:
    if value == "en":
        return "en"
    if value in {"zh", "zh-CN"}:
        return "zh"
    raise batch.AdapterError("vnext_language_invalid")


def body_free_stratum(
    source: batch.SourceRecord,
    selection_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the authenticated metadata needed for coverage and targeted gaps."""

    identity = source.identity_class
    if identity is not None and identity not in batch.IDENTITY_CLASSES:
        raise batch.AdapterError("vnext_identity_class_invalid")
    validation_role = source.role
    if selection_view is None:
        if validation_role == "identity":
            raise batch.AdapterError("vnext_identity_output_expert_mapping_missing")
        output_role = validation_role
        tool_family = _tool_family(source)
        selection_kind = "source_native"
        identity_parent_class = None
        review_verdict = None
        review_fault_type = None
        router_label = None
    else:
        expected_source = _sha256_text(source.record_id)
        if (
            selection_view.get("source_record_id_sha256") != expected_source
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(selection_view.get("source_content_sha256")),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(selection_view.get("source_serialization_identity_sha256")),
            )
            is None
            or selection_view.get("task_bundle_sha256")
            != _sha256_text(source.task_bundle)
            or selection_view.get("validation_contract_role") != validation_role
            or selection_view.get("language") != _normalized_language(source.language)
            or selection_view.get("identity_class") != identity
        ):
            raise batch.AdapterError("vnext_selection_view_source_drift")
        output_role = selection_view.get("output_expert_role")
        if output_role not in {
            "humor",
            "serious",
            "angry",
            "tool",
            "review",
            "router",
        }:
            raise batch.AdapterError("vnext_output_expert_role_invalid")
        if (
            selection_view.get("training_asset")
            != (selector.TRAINING_ASSET_BY_OUTPUT_ROLE[output_role])
        ):
            raise batch.AdapterError("vnext_training_asset_drift")
        tool_family = selection_view.get("tool_family")
        selection_kind = selection_view.get("selection_kind")
        identity_parent_class = selection_view.get("identity_parent_class")
        review_verdict = selection_view.get("review_verdict")
        review_fault_type = selection_view.get("review_fault_type")
        router_label = selection_view.get("router_label")
        if selection_kind not in _COVERAGE_SELECTION_KINDS:
            raise batch.AdapterError("vnext_selection_kind_invalid")
        if validation_role == "identity":
            if tool_family is not None:
                raise batch.AdapterError("vnext_identity_tool_direct_contract_drift")
        elif validation_role == "tool" and tool_family != _tool_family(source):
            raise batch.AdapterError("vnext_selection_tool_family_drift")
        elif validation_role != "tool":
            kind = selection_view.get("selection_kind")
            if kind in {"full_core", "tool_review_depth"}:
                if (
                    not isinstance(tool_family, str)
                    or batch.SAFE_ID_RE.fullmatch(tool_family) is None
                ):
                    raise batch.AdapterError("vnext_selection_tool_family_drift")
            elif tool_family is not None:
                raise batch.AdapterError("vnext_selection_tool_family_drift")
        if (
            selection_kind == "identity"
            and identity_parent_class not in batch.IDENTITY_CLASSES
        ) or (selection_kind != "identity" and identity_parent_class is not None):
            raise batch.AdapterError("vnext_selection_identity_parent_class_drift")
        if selection_kind == "tool_review_depth":
            if (
                review_verdict not in {"pass", "fail"}
                or (review_verdict == "pass" and review_fault_type is not None)
                or (
                    review_verdict == "fail"
                    and review_fault_type not in selector.FAULT_TYPES
                )
            ):
                raise batch.AdapterError("vnext_selection_review_stratum_drift")
        elif review_verdict is not None or review_fault_type is not None:
            raise batch.AdapterError("vnext_selection_review_stratum_drift")
        if (
            selection_kind == "router"
            and router_label not in {"humor", "serious", "angry"}
        ) or (selection_kind != "router" and router_label is not None):
            raise batch.AdapterError("vnext_selection_router_stratum_drift")
    return {
        "role": output_role,
        "validation_contract_role": validation_role,
        "language": _normalized_language(source.language),
        "tool_family": tool_family,
        "identity_class": identity,
        "identity_parent_class": identity_parent_class,
        "selection_kind": selection_kind,
        "review_verdict": review_verdict,
        "review_fault_type": review_fault_type,
        "router_label": router_label,
        "source_record_id_sha256": _sha256_text(source.record_id),
    }


def _coverage_key(
    stratum: Mapping[str, Any],
) -> CoverageKey:
    return (
        str(stratum["role"]),
        str(stratum["validation_contract_role"]),
        str(stratum["language"]),
        str(stratum.get("tool_family") or ""),
        str(stratum.get("identity_class") or ""),
        str(stratum.get("identity_parent_class") or ""),
        str(stratum.get("selection_kind") or ""),
        str(stratum.get("review_verdict") or ""),
        str(stratum.get("review_fault_type") or ""),
        str(stratum.get("router_label") or ""),
    )


def _coverage_rows(
    coverage: Mapping[CoverageKey, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (
        role,
        validation_contract_role,
        language,
        tool_family,
        identity_class,
        identity_parent_class,
        selection_kind,
        review_verdict,
        review_fault_type,
        router_label,
    ), counts in sorted(coverage.items()):
        rows.append(
            {
                "role": role,
                "validation_contract_role": validation_contract_role,
                "language": language,
                "tool_family": tool_family or None,
                "identity_class": identity_class or None,
                "identity_parent_class": (identity_parent_class or None),
                "selection_kind": selection_kind,
                "review_verdict": review_verdict or None,
                "review_fault_type": review_fault_type or None,
                "router_label": router_label or None,
                "accepted": int(counts.get("accepted", 0)),
                "rejected": int(counts.get("rejected", 0)),
                "quarantine": int(counts.get("quarantine", 0)),
            }
        )
    return rows


def coverage_gaps(
    expected: Mapping[CoverageKey, int],
    coverage: Mapping[
        CoverageKey,
        Mapping[str, int],
    ],
) -> list[dict[str, Any]]:
    """Return only body-free, count-level deficits for targeted supplementation."""

    gaps: list[dict[str, Any]] = []
    for key, requested in sorted(expected.items()):
        accepted = int(coverage.get(key, {}).get("accepted", 0))
        if accepted >= requested:
            continue
        (
            role,
            validation_contract_role,
            language,
            tool_family,
            identity_class,
            identity_parent_class,
            selection_kind,
            review_verdict,
            review_fault_type,
            router_label,
        ) = key
        gaps.append(
            {
                "role": role,
                "validation_contract_role": validation_contract_role,
                "language": language,
                "tool_family": tool_family or None,
                "identity_class": identity_class or None,
                "identity_parent_class": (identity_parent_class or None),
                "selection_kind": selection_kind,
                "review_verdict": review_verdict or None,
                "review_fault_type": review_fault_type or None,
                "router_label": router_label or None,
                "requested": requested,
                "accepted": accepted,
                "missing": requested - accepted,
            }
        )
    return gaps


def fast_validate_closed_grammar(
    source: batch.SourceRecord,
    text: str,
) -> None:
    """Run a single-record, body-free-error, closed-grammar precheck."""

    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > 256 * 1024
        or "\x00" in text
    ):
        raise batch.AdapterError("vnext_closed_grammar_envelope_invalid")
    clean = text.strip()
    if source.role == "identity":
        if clean not in set(batch.IDENTITY_OUTPUT_TEMPLATES.values()):
            raise batch.AdapterError("vnext_closed_grammar_identity_invalid")
        return
    required = _JSON_KEYS.get(source.role)
    if required is None or not clean.startswith("{") or not clean.endswith("}"):
        raise batch.AdapterError("vnext_closed_grammar_envelope_invalid")
    value = closed._strict_json(
        clean.encode("utf-8"),
        reason="vnext_closed_grammar_json_invalid",
    )
    if not isinstance(value, dict) or frozenset(value) != required:
        raise batch.AdapterError("vnext_closed_grammar_keys_invalid")
    if source.role in _STYLE_ROLES:
        answer = value.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise batch.AdapterError("vnext_closed_grammar_value_invalid")
    elif source.role == "tool":
        call = value.get("tool_call")
        if (
            not isinstance(value.get("final_answer"), str)
            or not isinstance(call, dict)
            or frozenset(call) != {"name", "arguments"}
            or not isinstance(call.get("arguments"), dict)
            or not isinstance(value.get("evidence_ids"), list)
        ):
            raise batch.AdapterError("vnext_closed_grammar_value_invalid")
    elif source.role == "review":
        if value.get("verdict") not in {"pass", "fail"} or not isinstance(
            value.get("faults"), list
        ):
            raise batch.AdapterError("vnext_closed_grammar_value_invalid")
    elif source.role == "router":
        if (
            not isinstance(value.get("route"), str)
            or not isinstance(value.get("plan"), list)
            or value.get("stop") is not True
        ):
            raise batch.AdapterError("vnext_closed_grammar_value_invalid")


def fast_parse_candidate_output(
    source: batch.SourceRecord,
    text: str,
) -> dict[str, Any]:
    """Apply only structural and irreversible safety gates online."""

    fast_validate_closed_grammar(source, text)
    clean = text.strip()
    if source.role in _STYLE_ROLES:
        parsed = closed._strict_json(
            clean.encode("utf-8"),
            reason="vnext_closed_grammar_json_invalid",
        )
        assert isinstance(parsed, dict)
        value: Any = str(parsed["final_answer"]).strip()
        collision_text = value
        result = {"kind": "natural_text", "value": value}
    elif source.role == "identity":
        collision_text = clean
        result = {"kind": "natural_text", "value": clean}
    else:
        parsed = closed._strict_json(
            clean.encode("utf-8"),
            reason="vnext_closed_grammar_json_invalid",
        )
        batch._reject_reasoning_keys(parsed)
        collision_text = clean
        result = {"kind": "structured_json", "value": parsed}
    normalized = collision_text.casefold()
    if any(marker in normalized for marker in batch.REASONING_MARKERS):
        raise batch.AdapterError("teacher_reasoning_marker_rejected")
    if batch.SECRET_LIKE_RE.search(collision_text):
        raise batch.AdapterError("teacher_secret_like_output")
    output_sha256 = _sha256_text(collision_text)
    if output_sha256 in set(source.guards["target_leakage_sha256"]):
        if not (
            source.role == "identity"
            and batch._identity_target_collision_allowed(
                source,
                output_sha256,
            )
        ):
            raise batch.AdapterError("teacher_target_leakage_rejected")
    return result


def _merkle_root(rows: Sequence[Mapping[str, Any]]) -> str:
    leaves = [
        hashlib.sha256(b"leaf\x00" + _canonical_bytes(row)).digest() for row in rows
    ]
    if not leaves:
        return hashlib.sha256(b"empty\x00").hexdigest()
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(b"node\x00" + leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _git_bytes(
    *args: str,
    reason: str,
    repo_root: Path | None = None,
) -> bytes:
    root = (repo_root or batch.REPO_ROOT).absolute()
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if result.returncode != 0 or result.stderr:
        raise batch.AdapterError(reason)
    return bytes(result.stdout)


def authenticate_producer_launch_L(
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Bind a globally clean reviewed producer commit/tree before any write."""

    root = (repo_root or batch.REPO_ROOT).absolute()
    status = _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        reason="vnext_producer_launch_L_dirty_tree",
        repo_root=root,
    )
    if status:
        raise batch.AdapterError("vnext_producer_launch_L_dirty_tree")
    commit_sha1 = (
        _git_bytes(
            "rev-parse",
            "HEAD^{commit}",
            reason="vnext_producer_launch_L_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    tree_sha1 = (
        _git_bytes(
            "rev-parse",
            "HEAD^{tree}",
            reason="vnext_producer_launch_L_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit_sha1) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree_sha1) is None
        or tuple(sorted(PRODUCER_LAUNCH_REVIEWED_PATHS))
        != PRODUCER_LAUNCH_REVIEWED_PATHS
        or len(set(PRODUCER_LAUNCH_REVIEWED_PATHS))
        != len(PRODUCER_LAUNCH_REVIEWED_PATHS)
    ):
        raise batch.AdapterError("vnext_producer_launch_L_git_identity_invalid")
    parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            commit_sha1,
            reason="vnext_producer_launch_L_parent_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    if (
        len(parent_line) != 2
        or parent_line[0] != commit_sha1
        or re.fullmatch(r"[0-9a-f]{40}", parent_line[1]) is None
    ):
        raise batch.AdapterError("vnext_producer_launch_L_parent_invalid")
    audited_base = _verify_audited_base_to_L(
        repo_root=root,
        audited_base=parent_line[1],
        producer_L=commit_sha1,
        expected_path_status=PRODUCER_L_BASE_DIFF_STATUS,
    )
    inventory: list[dict[str, Any]] = []
    for relative_path in PRODUCER_LAUNCH_REVIEWED_PATHS:
        physical_path = root / relative_path
        physical = _physical_read(
            physical_path,
            reason="vnext_producer_launch_L_file_snapshot_drift",
            stop=root,
        )
        blob_sha1 = (
            _git_bytes(
                "rev-parse",
                f"{commit_sha1}:{relative_path}",
                reason="vnext_producer_launch_L_blob_missing",
                repo_root=root,
            )
            .decode("ascii")
            .strip()
        )
        if re.fullmatch(r"[0-9a-f]{40}", blob_sha1) is None:
            raise batch.AdapterError("vnext_producer_launch_L_blob_invalid")
        committed = _git_bytes(
            "cat-file",
            "blob",
            blob_sha1,
            reason="vnext_producer_launch_L_blob_invalid",
            repo_root=root,
        )
        if not hmac.compare_digest(committed, physical):
            raise batch.AdapterError("vnext_producer_launch_L_blob_physical_drift")
        inventory.append(
            {
                "path": relative_path,
                "blob_sha1": blob_sha1,
                "sha256": _sha256_bytes(physical),
                "bytes": len(physical),
                "release_stage": "producer_launch_L",
            }
        )
    return {
        "stage": "producer_launch_L",
        "status": "authenticated_clean_reviewed_tree",
        "audited_base": audited_base,
        "commit_sha1": commit_sha1,
        "tree_sha1": tree_sha1,
        "reviewed_file_inventory": inventory,
        "reviewed_file_inventory_root_sha256": _sha256_bytes(
            _canonical_bytes(inventory)
        ),
        "reviewed_file_count": len(inventory),
        "global_clean_tree": True,
        "procedurally_independent_review_present": False,
        "content_retained": False,
    }


def _authenticate_consumer_code_P(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "stage",
        "status",
        "blocker_reason_code",
        "commit_sha1",
        "tree_sha1",
        "file_inventory_root_sha256",
        "implementation_sha256",
        "schema_set_root_sha256",
        "contract_sha256",
        "router_record_cells",
        "independently_audited",
        "content_retained",
    }
    if set(value) != required:
        raise batch.AdapterError("consumer_code_P_binding_invalid")
    if value.get("router_record_cells") != _EXPECTED_ROUTER_RECORD_CELLS:
        raise batch.AdapterError("consumer_code_P_router_source_truth_drift")
    if value.get("status") != "audited_compatible":
        reason = value.get("blocker_reason_code")
        if not isinstance(reason, str) or _SAFE_STOP_REASON.fullmatch(reason) is None:
            reason = "consumer_code_P_uncommitted_unreviewed"
        raise batch.AdapterError(reason)
    if (
        value.get("stage") != "consumer_code_P"
        or value.get("blocker_reason_code") is not None
        or value.get("independently_audited") is not True
        or value.get("content_retained") is not False
        or any(
            re.fullmatch(r"[0-9a-f]{40}", str(value.get(field))) is None
            for field in ("commit_sha1", "tree_sha1")
        )
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value.get(field))) is None
            for field in (
                "file_inventory_root_sha256",
                "implementation_sha256",
                "schema_set_root_sha256",
                "contract_sha256",
            )
        )
    ):
        raise batch.AdapterError("consumer_code_P_binding_invalid")
    return dict(value)


def _git_commit_inventory(
    *,
    repo_root: Path,
    commit_sha1: str,
    paths: Sequence[str],
    release_stage: str,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for relative_path in paths:
        blob_sha1 = (
            _git_bytes(
                "rev-parse",
                f"{commit_sha1}:{relative_path}",
                reason="vnext_release_blob_missing",
                repo_root=repo_root,
            )
            .decode("ascii")
            .strip()
        )
        if re.fullmatch(r"[0-9a-f]{40}", blob_sha1) is None:
            raise batch.AdapterError("vnext_release_blob_invalid")
        committed = _git_bytes(
            "cat-file",
            "blob",
            blob_sha1,
            reason="vnext_release_blob_invalid",
            repo_root=repo_root,
        )
        inventory.append(
            {
                "path": relative_path,
                "blob_sha1": blob_sha1,
                "sha256": _sha256_bytes(committed),
                "bytes": len(committed),
                "release_stage": release_stage,
            }
        )
    return inventory


def _git_name_status_diff(
    *,
    repo_root: Path,
    parent_commit: str,
    child_commit: str,
    reason: str,
) -> tuple[bytes, tuple[tuple[str, str], ...]]:
    raw = _git_bytes(
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        parent_commit,
        child_commit,
        reason=reason,
        repo_root=repo_root,
    )
    parts = raw.decode("utf-8").split("\x00")
    if parts and parts[-1] == "":
        parts.pop()
    if len(parts) % 2:
        raise batch.AdapterError(reason)
    rows = tuple((parts[index + 1], parts[index]) for index in range(0, len(parts), 2))
    if tuple(sorted(rows)) != rows or len({path for path, _ in rows}) != len(rows):
        raise batch.AdapterError(reason)
    return raw, rows


def _verify_audited_base_to_L(
    *,
    repo_root: Path,
    audited_base: str,
    producer_L: str,
    expected_path_status: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    expected = tuple(expected_path_status)
    if (
        tuple(sorted(expected)) != expected
        or len({path for path, _status in expected}) != len(expected)
        or any(status not in {"A", "M"} for _path, status in expected)
    ):
        raise batch.AdapterError("vnext_producer_L_allowlist_invalid")
    base_commit = (
        _git_bytes(
            "rev-parse",
            f"{audited_base}^{{commit}}",
            reason="vnext_producer_audited_base_invalid",
            repo_root=repo_root,
        )
        .decode("ascii")
        .strip()
    )
    if (
        base_commit != audited_base
        or re.fullmatch(r"[0-9a-f]{40}", base_commit) is None
    ):
        raise batch.AdapterError("vnext_producer_audited_base_invalid")
    parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            producer_L,
            reason="vnext_producer_L_parent_invalid",
            repo_root=repo_root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    if parent_line != [producer_L, base_commit]:
        raise batch.AdapterError("vnext_producer_L_not_unique_direct_child")
    raw_diff, observed = _git_name_status_diff(
        repo_root=repo_root,
        parent_commit=base_commit,
        child_commit=producer_L,
        reason="vnext_producer_L_base_diff_invalid",
    )
    if observed != expected:
        raise batch.AdapterError("vnext_producer_L_base_diff_allowlist_drift")
    base_tree = (
        _git_bytes(
            "rev-parse",
            f"{base_commit}^{{tree}}",
            reason="vnext_producer_audited_base_invalid",
            repo_root=repo_root,
        )
        .decode("ascii")
        .strip()
    )
    return {
        "commit_sha1": base_commit,
        "tree_sha1": base_tree,
        "producer_L_direct_child_sha1": producer_L,
        "exact_global_diff_allowlist": True,
        "raw_diff_sha256": _sha256_bytes(raw_diff),
        "path_status_root_sha256": _sha256_bytes(
            _canonical_bytes(
                [{"path": path, "status": status} for path, status in expected]
            )
        ),
        "path_count": len(expected),
        "content_retained": False,
    }


def _verify_producer_L_to_P_to_R(
    *,
    repo_root: Path,
    audited_base: str,
    producer_L: str,
    producer_P: str,
    producer_R: str,
    reviewed_paths: Sequence[str],
    producer_payload_P_paths: Sequence[str],
    producer_R_release_paths: Sequence[str],
    release_root: str,
    producer_L_base_diff_status: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Verify exact B -> L -> P -> R commits without self-reported stages."""

    root = repo_root.absolute()
    reviewed = tuple(reviewed_paths)
    p_paths = tuple(producer_payload_P_paths)
    r_paths = tuple(producer_R_release_paths)
    l_status = tuple(producer_L_base_diff_status)
    l_paths = tuple(path for path, _status in l_status)
    if (
        tuple(sorted(reviewed)) != reviewed
        or tuple(sorted(p_paths)) != p_paths
        or tuple(sorted(r_paths)) != r_paths
        or tuple(sorted(l_status)) != l_status
        or tuple(sorted(l_paths)) != l_paths
        or set(l_paths) != {".gitattributes", *reviewed}
        or len(set((*reviewed, *p_paths, *r_paths)))
        != len(reviewed) + len(p_paths) + len(r_paths)
        or not release_root
        or release_root.startswith("/")
        or ".." in Path(release_root).parts
    ):
        raise batch.AdapterError("vnext_release_exact_path_set_invalid")
    if _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        reason="vnext_release_dirty_tree",
        repo_root=root,
    ):
        raise batch.AdapterError("vnext_release_dirty_tree")
    l_commit = (
        _git_bytes(
            "rev-parse",
            f"{producer_L}^{{commit}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    p_commit = (
        _git_bytes(
            "rev-parse",
            f"{producer_P}^{{commit}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    r_commit = (
        _git_bytes(
            "rev-parse",
            f"{producer_R}^{{commit}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    if (
        l_commit != producer_L
        or p_commit != producer_P
        or r_commit != producer_R
        or re.fullmatch(r"[0-9a-f]{40}", l_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", p_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", r_commit) is None
    ):
        raise batch.AdapterError("vnext_release_git_identity_invalid")
    audited_base_proof = _verify_audited_base_to_L(
        repo_root=root,
        audited_base=audited_base,
        producer_L=l_commit,
        expected_path_status=l_status,
    )
    p_parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            p_commit,
            reason="vnext_release_parent_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    r_parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            r_commit,
            reason="vnext_release_parent_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    if p_parent_line != [p_commit, l_commit] or r_parent_line != [
        r_commit,
        p_commit,
    ]:
        raise batch.AdapterError("vnext_release_not_unique_direct_child")
    l_tree = (
        _git_bytes(
            "rev-parse",
            f"{l_commit}^{{tree}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    p_tree = (
        _git_bytes(
            "rev-parse",
            f"{p_commit}^{{tree}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    r_tree = (
        _git_bytes(
            "rev-parse",
            f"{r_commit}^{{tree}}",
            reason="vnext_release_git_identity_invalid",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    raw_p_diff, l_to_p_rows = _git_name_status_diff(
        repo_root=root,
        parent_commit=l_commit,
        child_commit=p_commit,
        reason="vnext_release_diff_invalid",
    )
    if l_to_p_rows != tuple((path, "A") for path in p_paths):
        raise batch.AdapterError("vnext_release_diff_allowlist_drift")
    raw_r_diff, p_to_r_rows = _git_name_status_diff(
        repo_root=root,
        parent_commit=p_commit,
        child_commit=r_commit,
        reason="vnext_release_diff_invalid",
    )
    if p_to_r_rows != tuple((path, "A") for path in r_paths):
        raise batch.AdapterError("vnext_release_diff_allowlist_drift")
    p_namespace = tuple(
        sorted(
            item
            for item in _git_bytes(
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                p_commit,
                "--",
                release_root,
                reason="vnext_release_namespace_invalid",
                repo_root=root,
            )
            .decode("utf-8")
            .split("\x00")
            if item
        )
    )
    r_namespace = tuple(
        sorted(
            item
            for item in _git_bytes(
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                r_commit,
                "--",
                release_root,
                reason="vnext_release_namespace_invalid",
                repo_root=root,
            )
            .decode("utf-8")
            .split("\x00")
            if item
        )
    )
    if p_namespace != p_paths or r_namespace != tuple(sorted((*p_paths, *r_paths))):
        raise batch.AdapterError("vnext_release_namespace_exact_set_drift")
    l_inventory = _git_commit_inventory(
        repo_root=root,
        commit_sha1=l_commit,
        paths=l_paths,
        release_stage="producer_launch_L",
    )
    p_inventory = _git_commit_inventory(
        repo_root=root,
        commit_sha1=p_commit,
        paths=p_paths,
        release_stage="producer_payload_P",
    )
    r_inventory = _git_commit_inventory(
        repo_root=root,
        commit_sha1=r_commit,
        paths=r_paths,
        release_stage="producer_final_R",
    )
    for row in l_inventory:
        for descendant in (p_commit, r_commit):
            descendant_blob = (
                _git_bytes(
                    "rev-parse",
                    f"{descendant}:{row['path']}",
                    reason="vnext_release_L_blob_not_preserved",
                    repo_root=root,
                )
                .decode("ascii")
                .strip()
            )
            if descendant_blob != row["blob_sha1"]:
                raise batch.AdapterError("vnext_release_L_blob_not_preserved")
    for row in p_inventory:
        r_blob = (
            _git_bytes(
                "rev-parse",
                f"{r_commit}:{row['path']}",
                reason="vnext_release_P_blob_not_preserved",
                repo_root=root,
            )
            .decode("ascii")
            .strip()
        )
        if r_blob != row["blob_sha1"]:
            raise batch.AdapterError("vnext_release_P_blob_not_preserved")
    path_stage_map = (
        [
            {
                "path": path,
                "release_stage": "producer_launch_L",
            }
            for path in l_paths
        ]
        + [
            {
                "path": path,
                "release_stage": "producer_payload_P",
            }
            for path in p_paths
        ]
        + [
            {
                "path": path,
                "release_stage": "producer_final_R",
            }
            for path in r_paths
        ]
    )
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "producer-L-to-P-to-R-proof.v1"
        ),
        "audited_base": audited_base_proof,
        "producer_launch_L": {
            "commit_sha1": l_commit,
            "tree_sha1": l_tree,
            "direct_parent_sha1": audited_base_proof["commit_sha1"],
            "file_inventory": l_inventory,
            "file_inventory_root_sha256": _sha256_bytes(_canonical_bytes(l_inventory)),
        },
        "producer_payload_P": {
            "commit_sha1": p_commit,
            "tree_sha1": p_tree,
            "direct_parent_sha1": l_commit,
            "raw_diff_sha256": _sha256_bytes(raw_p_diff),
            "file_inventory": p_inventory,
            "file_inventory_root_sha256": _sha256_bytes(_canonical_bytes(p_inventory)),
        },
        "producer_final_R": {
            "commit_sha1": r_commit,
            "tree_sha1": r_tree,
            "direct_parent_sha1": p_commit,
            "raw_diff_sha256": _sha256_bytes(raw_r_diff),
            "file_inventory": r_inventory,
            "file_inventory_root_sha256": _sha256_bytes(_canonical_bytes(r_inventory)),
        },
        "release_kind_stage_map": RELEASE_KIND_STAGE_MAP,
        "producer_L_exact_path_set_root_sha256": _sha256_bytes(
            _canonical_bytes(list(l_paths))
        ),
        "producer_P_exact_path_set_root_sha256": _sha256_bytes(
            _canonical_bytes(list(p_paths))
        ),
        "producer_R_exact_path_set_root_sha256": _sha256_bytes(
            _canonical_bytes(list(r_paths))
        ),
        "path_stage_map": path_stage_map,
        "path_stage_map_root_sha256": _sha256_bytes(_canonical_bytes(path_stage_map)),
        "global_clean_tree": True,
        "unique_direct_children": True,
        "exact_raw_diff_allowlist": True,
        "release_self_pin_present": False,
        "procedurally_independent_review_present": False,
        "content_retained": False,
    }


def verify_producer_L_to_P_to_R(
    *,
    audited_base: str,
    producer_L: str,
    producer_P: str,
    producer_R: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Public verifier with immutable production L/P/R path sets."""

    return _verify_producer_L_to_P_to_R(
        repo_root=(repo_root or batch.REPO_ROOT),
        audited_base=audited_base,
        producer_L=producer_L,
        producer_P=producer_P,
        producer_R=producer_R,
        reviewed_paths=PRODUCER_LAUNCH_REVIEWED_PATHS,
        producer_payload_P_paths=PRODUCER_PAYLOAD_P_PATHS,
        producer_R_release_paths=PRODUCER_R_RELEASE_PATHS,
        release_root=PRODUCER_RELEASE_ROOT,
        producer_L_base_diff_status=PRODUCER_L_BASE_DIFF_STATUS,
    )


def _materialize_producer_R(
    *,
    repo_root: Path,
    release_root: str,
    producer_P_release_paths: Sequence[str],
    producer_R_release_paths: Sequence[str],
    freeze_preimage_path: Path,
    freeze_preimage_sha256: str,
    external_review_receipt_path: Path,
    external_review_receipt_sha256: str,
    external_schema_path: Path = DEFAULT_EXTERNAL_ATTESTATION_SCHEMA_PATH,
) -> dict[str, Any]:
    """Create R's exact four files once, or roll back the whole operation."""

    root = repo_root.absolute()
    p_paths = tuple(producer_P_release_paths)
    r_paths = tuple(producer_R_release_paths)
    if (
        tuple(sorted(p_paths)) != p_paths
        or tuple(sorted(r_paths)) != r_paths
        or len(r_paths) != 4
        or {Path(path).name for path in r_paths}
        != {
            "freeze_attestation.json",
            "freeze_attestation.json.sha256",
            "external_attestation.json",
            "external_attestation.json.sha256",
        }
        or any(Path(path).parent.as_posix() != release_root for path in r_paths)
    ):
        raise batch.AdapterError("vnext_R_materializer_path_set_invalid")
    if _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        reason="vnext_R_materializer_dirty_tree",
        repo_root=root,
    ):
        raise batch.AdapterError("vnext_R_materializer_dirty_tree")
    destination = root / release_root
    closed._reject_reparse_chain(destination, stop=root)
    if not destination.is_dir():
        raise batch.AdapterError("vnext_R_materializer_release_root_missing")
    observed_p = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in destination.iterdir()
            if path.is_file()
        )
    )
    if observed_p != p_paths:
        raise batch.AdapterError("vnext_R_materializer_P_file_set_drift")
    targets = tuple(root / path for path in r_paths)
    if any(path.exists() or path.is_symlink() for path in targets):
        raise batch.AdapterError("vnext_R_materializer_no_replace")
    freeze_raw = _physical_read(
        freeze_preimage_path,
        reason="vnext_R_freeze_preimage_snapshot_drift",
        stop=root,
    )
    freeze_digest = _sha256_bytes(freeze_raw)
    if re.fullmatch(
        r"[0-9a-f]{64}", freeze_preimage_sha256
    ) is None or not hmac.compare_digest(freeze_digest, freeze_preimage_sha256):
        raise batch.AdapterError("vnext_R_freeze_preimage_identity_drift")
    freeze_sidecar = _physical_read(
        freeze_preimage_path.with_name(f"{freeze_preimage_path.name}.sha256"),
        reason="vnext_R_freeze_preimage_sidecar_drift",
        stop=root,
    )
    if freeze_sidecar != (f"{freeze_digest}  {freeze_preimage_path.name}\n").encode(
        "ascii"
    ):
        raise batch.AdapterError("vnext_R_freeze_preimage_sidecar_drift")
    external_raw = _physical_read(
        external_review_receipt_path,
        reason="vnext_R_external_review_snapshot_drift",
        stop=root,
    )
    external_digest = _sha256_bytes(external_raw)
    if re.fullmatch(
        r"[0-9a-f]{64}", external_review_receipt_sha256
    ) is None or not hmac.compare_digest(
        external_digest,
        external_review_receipt_sha256,
    ):
        raise batch.AdapterError("vnext_R_external_review_identity_drift")
    freeze = _strict_json_mapping(
        freeze_raw,
        reason="vnext_R_freeze_preimage_invalid",
    )
    external = _strict_json_mapping(
        external_raw,
        reason="vnext_R_external_review_invalid",
    )
    schema_raw = _physical_read(
        external_schema_path,
        reason="vnext_R_external_schema_snapshot_drift",
        stop=batch.REPO_ROOT.absolute(),
    )
    schema = _strict_json_mapping(
        schema_raw,
        reason="vnext_R_external_schema_invalid",
    )
    try:
        Draft202012Validator(schema).validate(external)
    except ValidationError as error:
        raise batch.AdapterError("vnext_R_external_review_schema_mismatch") from error
    reviewer = external["reviewer_receipt"]
    if not isinstance(reviewer, Mapping) or hmac.compare_digest(
        str(reviewer["reviewer_identity_sha256"]),
        str(reviewer["implementer_identity_sha256"]),
    ):
        raise batch.AdapterError("vnext_R_reviewer_not_independent")
    audited_base = external["audited_base"]
    producer_L = external["producer_launch_L"]
    producer_P = external["producer_payload_P"]
    freeze_producer_L = freeze.get("producer_launch_L")
    freeze_binding = external["freeze_preimage"]
    if (
        not isinstance(audited_base, Mapping)
        or not isinstance(producer_L, Mapping)
        or not isinstance(producer_P, Mapping)
        or not isinstance(freeze_producer_L, Mapping)
        or not isinstance(freeze_binding, Mapping)
        or any(
            producer_L.get(key) != freeze_producer_L.get(key)
            for key in (
                "commit_sha1",
                "tree_sha1",
                "reviewed_file_inventory_root_sha256",
                "reviewed_file_count",
            )
        )
        or producer_L.get("direct_parent_sha1") != audited_base.get("commit_sha1")
        or freeze_producer_L.get("audited_base_commit_sha1")
        != audited_base.get("commit_sha1")
        or producer_P.get("direct_parent_sha1") != producer_L.get("commit_sha1")
        or freeze_binding.get("freeze_attestation_sha256") != freeze_digest
        or freeze_binding.get("checkpoint_manifest_root_sha256")
        != freeze.get("checkpoint_manifest_root_sha256")
        or freeze_binding.get("selector_identity_sha256")
        != freeze.get("selector_identity_sha256")
        or freeze_binding.get("selected_set_root_sha256")
        != freeze.get("selected_set_root_sha256")
        or freeze_binding.get("accepted_strata_root_sha256")
        != freeze.get("accepted_strata_root_sha256")
        or freeze_binding.get("counts") != freeze.get("counts")
        or freeze_binding.get("coverage_gaps") != freeze.get("coverage_gaps")
        or freeze_binding.get("coverage_complete") is not True
        or freeze.get("coverage_gaps") != []
    ):
        raise batch.AdapterError("vnext_R_freeze_preimage_binding_drift")
    head = (
        _git_bytes(
            "rev-parse",
            "HEAD^{commit}",
            reason="vnext_R_producer_P_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    tree = (
        _git_bytes(
            "rev-parse",
            "HEAD^{tree}",
            reason="vnext_R_producer_P_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    if head != producer_P.get("commit_sha1") or tree != producer_P.get("tree_sha1"):
        raise batch.AdapterError("vnext_R_producer_P_identity_drift")
    p_parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            head,
            reason="vnext_R_producer_P_parent_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    l_commit = str(producer_L["commit_sha1"])
    b_commit = str(audited_base["commit_sha1"])
    l_parent_line = (
        _git_bytes(
            "rev-list",
            "--parents",
            "-n",
            "1",
            l_commit,
            reason="vnext_R_producer_L_parent_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    if p_parent_line != [head, l_commit] or l_parent_line != [l_commit, b_commit]:
        raise batch.AdapterError("vnext_R_producer_stage_parent_drift")
    raw_l_diff, l_rows = _git_name_status_diff(
        repo_root=root,
        parent_commit=b_commit,
        child_commit=l_commit,
        reason="vnext_R_producer_L_diff_drift",
    )
    raw_p_diff, p_rows = _git_name_status_diff(
        repo_root=root,
        parent_commit=l_commit,
        child_commit=head,
        reason="vnext_R_producer_P_diff_drift",
    )
    p_inventory = _git_commit_inventory(
        repo_root=root,
        commit_sha1=head,
        paths=p_paths,
        release_stage="producer_payload_P",
    )
    if (
        producer_L.get("tree_sha1")
        != _git_bytes(
            "rev-parse",
            f"{l_commit}^{{tree}}",
            reason="vnext_R_producer_L_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        or audited_base.get("tree_sha1")
        != _git_bytes(
            "rev-parse",
            f"{b_commit}^{{tree}}",
            reason="vnext_R_audited_base_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
        or audited_base.get("producer_L_raw_diff_sha256") != _sha256_bytes(raw_l_diff)
        or audited_base.get("producer_L_path_status_root_sha256")
        != _sha256_bytes(
            _canonical_bytes(
                [{"path": path, "status": status} for path, status in l_rows]
            )
        )
        or p_rows != tuple((path, "A") for path in p_paths)
        or producer_P.get("raw_diff_sha256") != _sha256_bytes(raw_p_diff)
        or producer_P.get("file_inventory_root_sha256")
        != _sha256_bytes(_canonical_bytes(p_inventory))
    ):
        raise batch.AdapterError("vnext_R_producer_provenance_drift")
    output_by_name = {
        "freeze_attestation.json": freeze_raw,
        "freeze_attestation.json.sha256": (
            f"{freeze_digest}  freeze_attestation.json\n"
        ).encode("ascii"),
        "external_attestation.json": external_raw,
        "external_attestation.json.sha256": (
            f"{external_digest}  external_attestation.json\n"
        ).encode("ascii"),
    }
    created: list[Path] = []
    try:
        for target in targets:
            _exclusive_bytes(target, output_by_name[target.name])
            created.append(target)
    except BaseException:
        for target in reversed(created):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise
    inventory = [
        {
            "path": target.relative_to(root).as_posix(),
            "sha256": _sha256_bytes(output_by_name[target.name]),
            "bytes": len(output_by_name[target.name]),
        }
        for target in targets
    ]
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "producer-R-materialization-receipt.v1"
        ),
        "producer_P_commit_sha1": head,
        "files": inventory,
        "file_inventory_root_sha256": _merkle_root(inventory),
        "created_once": True,
        "no_replace": True,
        "release_self_pin_present": False,
        "R_identity_proof_written_into_R": False,
        "external_R_proof_required": True,
        "consumer_pin_C_required": True,
        "content_retained": False,
    }


def materialize_producer_R(
    *,
    freeze_preimage_path: Path,
    freeze_preimage_sha256: str,
    external_review_receipt_path: Path,
    external_review_receipt_sha256: str,
) -> dict[str, Any]:
    """Production R materializer with immutable exact paths."""

    return _materialize_producer_R(
        repo_root=batch.REPO_ROOT,
        release_root=PRODUCER_RELEASE_ROOT,
        producer_P_release_paths=PRODUCER_PAYLOAD_P_PATHS,
        producer_R_release_paths=PRODUCER_R_RELEASE_PATHS,
        freeze_preimage_path=freeze_preimage_path,
        freeze_preimage_sha256=freeze_preimage_sha256,
        external_review_receipt_path=external_review_receipt_path,
        external_review_receipt_sha256=external_review_receipt_sha256,
    )


def derive_selected_vnext_jobs(
    base_jobs: Sequence[batch.AlignmentJob],
    selection_views: Mapping[str, Mapping[str, Any]],
    *,
    identity_derivative_specs: Sequence[Mapping[str, Any]],
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
) -> tuple[batch.AlignmentJob, ...]:
    """Bind exact selector rows to physical or persisted derivative jobs."""

    by_source: dict[str, batch.AlignmentJob] = {}
    for job in base_jobs:
        source_hash = _sha256_text(job.source.record_id)
        if source_hash in by_source:
            raise batch.AdapterError("vnext_source_record_identity_duplicate")
        by_source[source_hash] = job
    specs_by_key = {
        str(spec["derived_teacher_job_idempotency_key"]): dict(spec)
        for spec in identity_derivative_specs
    }
    if len(identity_derivative_specs) != 100 or len(specs_by_key) != 100:
        raise batch.AdapterError("vnext_identity_derivative_spec_inventory_drift")
    consumed_specs: set[str] = set()
    selected: list[batch.AlignmentJob] = []
    # ``selection_views`` is constructed directly from the authenticated
    # selected-candidate JSONL. Preserve that physical candidate order so the
    # later training inventory can be zipped to it without inventing a second
    # ordering domain.
    for key, raw_view in selection_views.items():
        view = dict(raw_view)
        if view.get("idempotency_key", key) != key:
            raise batch.AdapterError("vnext_selector_idempotency_key_drift")
        parent_source_sha256 = str(view["parent_source_record_id_sha256"])
        base_job = by_source.get(parent_source_sha256)
        if base_job is None:
            raise batch.AdapterError("vnext_selector_source_record_missing")
        output_role = str(view.get("output_expert_role"))
        if view.get("selection_kind") == "identity":
            spec = specs_by_key.get(key)
            if spec is None:
                raise batch.AdapterError("vnext_identity_derivative_spec_missing")
            spec_sha256 = str(spec["identity_derivative_spec_sha256"])
            consumed_specs.add(spec_sha256)
            expected_view = {
                "identity_derivative_spec_sha256": spec_sha256,
                "idempotency_key": spec["derived_teacher_job_idempotency_key"],
                "teacher_request_idempotency_key": spec[
                    "derived_teacher_job_idempotency_key"
                ],
                "source_record_id_sha256": spec["derived_record_id_sha256"],
                "source_content_sha256": spec["derived_source_content_sha256"],
                "source_serialization_identity_sha256": spec[
                    "derived_source_serialization_identity_sha256"
                ],
                "teacher_request_record_id_sha256": spec["derived_record_id_sha256"],
                "parent_source_record_id_sha256": spec[
                    "parent_source_record_id_sha256"
                ],
                "task_bundle_sha256": spec["derived_task_bundle_sha256"],
                "parent_task_bundle_sha256": spec["parent_task_bundle_sha256"],
                "source_semantic_sha256": spec["derived_semantic_sha256"],
                "output_expert_role": spec["output_expert_role"],
                "training_asset": selector.TRAINING_ASSET_BY_OUTPUT_ROLE[
                    str(spec["output_expert_role"])
                ],
                "validation_contract_role": spec["validation_contract_role"],
                "language": spec["language"],
                "identity_class": spec["target_identity_class"],
                "identity_parent_class": spec["source_identity_class"],
            }
            if any(
                view.get(field) != expected for field, expected in expected_view.items()
            ):
                raise batch.AdapterError("vnext_identity_derivative_view_drift")
            parent_safe = {
                "parent_source_record_id_sha256": _sha256_text(
                    base_job.source.record_id
                ),
                "parent_task_bundle_sha256": _sha256_text(base_job.source.task_bundle),
                "parent_source_semantic_sha256": _sha256_text(base_job.source.semantic),
                "parent_source_line_sha256": (base_job.source.source_line_sha256),
                "parent_content_identity_sha256": (
                    base_job.source.content_identity_sha256
                ),
                "parent_base_prompt_template_sha256": (base_job.prompt_template_sha256),
                "parent_base_idempotency_key": (base_job.idempotency_key),
                "parent_source_role": base_job.source.role,
            }
            if any(
                spec.get(field) != expected for field, expected in parent_safe.items()
            ):
                raise batch.AdapterError(
                    "vnext_identity_derivative_parent_lineage_drift"
                )
            selected_job = selector.identity_derivative_job_from_spec(
                spec,
                inventory=inventory,
                config=config,
            )
            if selected_job.idempotency_key != key:
                raise batch.AdapterError("vnext_identity_derivative_job_reuse")
        else:
            if key in specs_by_key:
                raise batch.AdapterError("vnext_identity_derivative_spec_kind_drift")
            if (
                view.get("validation_contract_role") != base_job.source.role
                or output_role != base_job.source.role
                or view.get("source_record_id_sha256") != parent_source_sha256
                or view.get("teacher_request_record_id_sha256") != parent_source_sha256
                or view.get("teacher_request_idempotency_key")
                != base_job.idempotency_key
            ):
                raise batch.AdapterError("vnext_selector_validation_role_drift")
            selected_job = base_job
        body_free_stratum(selected_job.source, view)
        if not hmac.compare_digest(key, selected_job.idempotency_key):
            raise batch.AdapterError("vnext_selector_idempotency_identity_drift")
        selected.append(selected_job)
    if (
        len(selected) != 320
        or len(consumed_specs) != 100
        or consumed_specs
        != {
            str(spec["identity_derivative_spec_sha256"])
            for spec in identity_derivative_specs
        }
    ):
        raise batch.AdapterError("vnext_minimal320_job_set_drift")
    return tuple(selected)


def authenticated_job_prompts(
    job: batch.AlignmentJob,
    selection_view: Mapping[str, Any],
    *,
    identity_derivative_spec: Mapping[str, Any] | None,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
) -> tuple[str, str]:
    """Reconstruct the exact wire prompts and authenticate their bound hashes."""

    if selection_view.get("selection_kind") == "identity":
        if identity_derivative_spec is None:
            raise batch.AdapterError("vnext_identity_derivative_spec_missing")
        system, user, teacher_input = selector.identity_derivative_prompts_from_spec(
            identity_derivative_spec,
            inventory=inventory,
            config=config,
        )
        if (
            identity_derivative_spec["identity_derivative_spec_sha256"]
            != selection_view.get("identity_derivative_spec_sha256")
            or job.source.teacher_input != teacher_input
            or batch._user_prompt(job.source) != user
            or job.prompt_template_sha256
            != identity_derivative_spec["derived_prompt_contract_sha256"]
            or job.output_schema_sha256
            != identity_derivative_spec["derived_output_contract_sha256"]
            or _sha256_text(user) != identity_derivative_spec["user_prompt_sha256"]
            or _sha256_text(teacher_input)
            != identity_derivative_spec["teacher_input_sha256"]
        ):
            raise batch.AdapterError("vnext_identity_derivative_wire_prompt_drift")
        return system, user
    if identity_derivative_spec is not None:
        raise batch.AdapterError("vnext_identity_derivative_spec_kind_drift")
    system = batch._system_prompt(job.source.role)
    user = batch._user_prompt(job.source)
    if _sha256_text(system) != job.prompt_template_sha256:
        raise batch.AdapterError("vnext_wire_prompt_identity_drift")
    return system, user


def prompt_binding(
    job: batch.AlignmentJob,
    selection_view: Mapping[str, Any],
    *,
    identity_derivative_spec: Mapping[str, Any] | None,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
) -> dict[str, Any]:
    system, user = authenticated_job_prompts(
        job,
        selection_view,
        identity_derivative_spec=identity_derivative_spec,
        inventory=inventory,
        config=config,
    )
    return {
        "system_prompt_sha256": _sha256_text(system),
        "user_prompt_sha256": _sha256_text(user),
        "teacher_input_sha256": _sha256_text(job.source.teacher_input),
        "prompt_version": (
            str(identity_derivative_spec["v5_prompt_version"])
            if identity_derivative_spec is not None
            else batch.PROMPT_VERSION
        ),
        "prompt_contract_sha256": job.prompt_template_sha256,
        "physical_output_schema_sha256": (
            str(identity_derivative_spec["derived_output_schema_sha256"])
            if identity_derivative_spec is not None
            else job.output_schema_sha256
        ),
        "output_contract_sha256": job.output_schema_sha256,
        "identity_derivative_spec_sha256": (
            str(identity_derivative_spec["identity_derivative_spec_sha256"])
            if identity_derivative_spec is not None
            else None
        ),
        "response_contract": (
            str(identity_derivative_spec["response_contract"])
            if identity_derivative_spec is not None
            else "source_native"
        ),
        "tool_invocation_allowed": (
            bool(identity_derivative_spec["tool_invocation_allowed"])
            if identity_derivative_spec is not None
            else job.source.role == "tool"
        ),
        "content_retained": False,
    }


def _authenticated_file_digest(
    path: Path,
    *,
    stop: Path,
    reason: str,
) -> str:
    raw = _physical_read(path, reason=reason, stop=stop)
    digest = _sha256_bytes(raw)
    sidecar = _physical_read(
        path.with_name(f"{path.name}.sha256"),
        reason=reason,
        stop=stop,
    )
    if not hmac.compare_digest(
        sidecar,
        f"{digest}  {path.name}\n".encode("ascii"),
    ):
        raise batch.AdapterError(reason)
    return digest


def build_job_provenance(
    *,
    policy: VNextPolicy,
    config: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    overlay: selector.AuthenticatedMetadataOverlay,
    preimages: selector.SelectorPreimages,
    selection_manifest: Mapping[str, Any],
    producer_launch_L: Mapping[str, Any],
    consumer_code_P: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the closed body-free provenance embedded in every job receipt."""

    consumer = _authenticate_consumer_code_P(consumer_code_P)
    producer_required = {
        "stage",
        "status",
        "audited_base",
        "commit_sha1",
        "tree_sha1",
        "reviewed_file_inventory",
        "reviewed_file_inventory_root_sha256",
        "reviewed_file_count",
        "global_clean_tree",
        "procedurally_independent_review_present",
        "content_retained",
    }
    if (
        set(producer_launch_L) != producer_required
        or producer_launch_L.get("stage") != "producer_launch_L"
        or producer_launch_L.get("status") != "authenticated_clean_reviewed_tree"
        or producer_launch_L.get("global_clean_tree") is not True
        or producer_launch_L.get("procedurally_independent_review_present") is not False
        or producer_launch_L.get("content_retained") is not False
        or not isinstance(producer_launch_L.get("audited_base"), Mapping)
        or producer_launch_L["audited_base"].get("producer_L_direct_child_sha1")
        != producer_launch_L.get("commit_sha1")
        or producer_launch_L.get("reviewed_file_count")
        != len(PRODUCER_LAUNCH_REVIEWED_PATHS)
        or producer_launch_L.get("reviewed_file_inventory_root_sha256")
        != _sha256_bytes(
            _canonical_bytes(producer_launch_L.get("reviewed_file_inventory"))
        )
    ):
        raise batch.AdapterError("vnext_producer_launch_L_binding_invalid")
    overlay_manifest_sha256 = _authenticated_file_digest(
        overlay.manifest_path,
        stop=overlay.directory,
        reason="vnext_overlay_manifest_provenance_drift",
    )
    preimage_manifest_sha256 = _authenticated_file_digest(
        preimages.preimage_manifest_path,
        stop=preimages.directory,
        reason="vnext_preimage_manifest_provenance_drift",
    )
    if (
        preimages.manifest.get("selected_set_root_sha256")
        != selection_manifest.get("selected_set_root_sha256")
        or preimages.manifest.get("selector_identity_sha256")
        != selection_manifest.get("selector_identity_sha256")
        or preimages.manifest.get("selected_count") != 320
        or preimages.manifest.get("candidate_count") != 3960
    ):
        raise batch.AdapterError("vnext_selector_provenance_binding_drift")
    payload = {
        "schema_version": JOB_PROVENANCE_SCHEMA_VERSION,
        "runtime": {
            "vnext_config_sha256": policy.physical_sha256,
            "vnext_config_schema_sha256": policy.schema_sha256,
            "vnext_contract_sha256": policy.contract_sha256,
            "vnext_implementation_sha256": policy.implementation_sha256,
            "receipt_schema_sha256": policy.receipt_schema_sha256,
            "checkpoint_schema_sha256": policy.checkpoint_schema_sha256,
            "base_profile_sha256": config.physical_sha256,
            "campaign_sha256": config.campaign_sha256,
            "model_binding_sha256": config.model_binding_sha256,
            "teacher_implementation_sha256": config.contract_hashes[
                "teacher_implementation_path"
            ],
        },
        "source": {
            "source_identity_sha256": inventory.source_identity_sha256,
            "manifest_sha256": inventory.manifest_sha256,
            "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        },
        "selector": {
            "selector_contract_sha256": policy.selector_contract_sha256,
            "selector_schema_sha256": policy.selector_schema_sha256,
            "selector_implementation_sha256": (policy.selector_implementation_sha256),
            "overlay_manifest_sha256": overlay_manifest_sha256,
            "overlay_rows_root_sha256": overlay.manifest["overlay_rows_root_sha256"],
            "overlay_row_count": overlay.manifest["row_count"],
            "preimage_manifest_sha256": preimage_manifest_sha256,
            "preimage_schema_sha256": preimages.manifest["preimage_schema_sha256"],
            "candidate_schema_sha256": preimages.manifest["candidate_schema_sha256"],
            "candidate_set_root_sha256": preimages.manifest[
                "candidate_set_root_sha256"
            ],
            "candidate_count": preimages.manifest["candidate_count"],
            "selected_set_root_sha256": preimages.manifest["selected_set_root_sha256"],
            "selected_count": preimages.manifest["selected_count"],
            "selector_identity_sha256": preimages.manifest["selector_identity_sha256"],
            "identity_derivative_plan_sha256": preimages.manifest[
                "identity_derivative_plan_sha256"
            ],
        },
        "producer_launch_L": {
            "stage": "producer_launch_L",
            "audited_base_commit_sha1": producer_launch_L["audited_base"][
                "commit_sha1"
            ],
            "audited_base_tree_sha1": producer_launch_L["audited_base"]["tree_sha1"],
            "audited_base_path_status_root_sha256": producer_launch_L["audited_base"][
                "path_status_root_sha256"
            ],
            "commit_sha1": producer_launch_L["commit_sha1"],
            "tree_sha1": producer_launch_L["tree_sha1"],
            "reviewed_file_inventory_root_sha256": producer_launch_L[
                "reviewed_file_inventory_root_sha256"
            ],
            "reviewed_file_count": producer_launch_L["reviewed_file_count"],
            "global_clean_tree": True,
            "procedurally_independent_review_present": False,
        },
        "consumer_code_P": consumer,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    return {
        **payload,
        "provenance_identity_sha256": _sha256_bytes(_canonical_bytes(payload)),
    }


def _validate_job_provenance(
    value: Mapping[str, Any],
    *,
    policy: VNextPolicy,
    config: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    selection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    identity = payload.pop("provenance_identity_sha256", None)
    if (
        set(payload)
        != {
            "schema_version",
            "runtime",
            "source",
            "selector",
            "producer_launch_L",
            "consumer_code_P",
            "content_retained",
            "raw_token_ids_retained",
            "credential_retained",
        }
        or identity != _sha256_bytes(_canonical_bytes(payload))
        or payload.get("schema_version") != JOB_PROVENANCE_SCHEMA_VERSION
        or payload.get("content_retained") is not False
        or payload.get("raw_token_ids_retained") is not False
        or payload.get("credential_retained") is not False
    ):
        raise batch.AdapterError("vnext_job_provenance_invalid")
    runtime = payload.get("runtime")
    source = payload.get("source")
    selector_binding = payload.get("selector")
    producer = payload.get("producer_launch_L")
    if (
        runtime
        != {
            "vnext_config_sha256": policy.physical_sha256,
            "vnext_config_schema_sha256": policy.schema_sha256,
            "vnext_contract_sha256": policy.contract_sha256,
            "vnext_implementation_sha256": policy.implementation_sha256,
            "receipt_schema_sha256": policy.receipt_schema_sha256,
            "checkpoint_schema_sha256": policy.checkpoint_schema_sha256,
            "base_profile_sha256": config.physical_sha256,
            "campaign_sha256": config.campaign_sha256,
            "model_binding_sha256": config.model_binding_sha256,
            "teacher_implementation_sha256": config.contract_hashes[
                "teacher_implementation_path"
            ],
        }
        or source
        != {
            "source_identity_sha256": inventory.source_identity_sha256,
            "manifest_sha256": inventory.manifest_sha256,
            "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        }
        or not isinstance(selector_binding, Mapping)
        or selector_binding.get("selector_contract_sha256")
        != policy.selector_contract_sha256
        or selector_binding.get("selector_schema_sha256")
        != policy.selector_schema_sha256
        or selector_binding.get("selector_implementation_sha256")
        != policy.selector_implementation_sha256
        or selector_binding.get("selected_set_root_sha256")
        != selection_manifest.get("selected_set_root_sha256")
        or selector_binding.get("selector_identity_sha256")
        != selection_manifest.get("selector_identity_sha256")
        or selector_binding.get("candidate_count") != 3960
        or selector_binding.get("selected_count") != 320
        or not isinstance(producer, Mapping)
        or producer.get("stage") != "producer_launch_L"
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(producer.get("audited_base_commit_sha1")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(producer.get("audited_base_tree_sha1")),
        )
        is None
        or producer.get("global_clean_tree") is not True
        or producer.get("procedurally_independent_review_present") is not False
    ):
        raise batch.AdapterError("vnext_job_provenance_binding_drift")
    _authenticate_consumer_code_P(
        payload.get("consumer_code_P")
        if isinstance(payload.get("consumer_code_P"), Mapping)
        else {}
    )
    return dict(value)


def _transport_from_persisted_output(
    source: batch.SourceRecord,
    output: Mapping[str, Any],
) -> str:
    if set(output) != {"kind", "value"}:
        raise batch.AdapterError("vnext_final_output_shape_invalid")
    value = output["value"]
    if source.role in _STYLE_ROLES:
        return json.dumps(
            {"final_answer": value},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if source.role == "identity":
        if not isinstance(value, str):
            raise batch.AdapterError("vnext_final_identity_value_invalid")
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _assert_identity_invariant(
    text: Any,
    identity_class: str,
    *,
    exact: bool,
) -> None:
    if not isinstance(text, str):
        raise batch.AdapterError("vnext_final_identity_invariant_missing")
    template = selector.V5_IDENTITY_TEMPLATES[identity_class]
    if text != template if exact else template not in text:
        raise batch.AdapterError("vnext_final_identity_template_mismatch")
    if exact:
        return
    # The selected immutable template is the only attribution-bearing span.
    # Removing it must leave no second Air/provider claim—affirmative,
    # contradictory, or another negation.
    remainder = text.replace(template, "", 1).casefold()
    if any(provider in remainder for provider in ("air", "google", "openai")):
        raise batch.AdapterError("vnext_final_identity_extra_attribution")


def final_quality_validate(
    job: batch.AlignmentJob,
    record: Mapping[str, Any],
    selection_view: Mapping[str, Any],
    *,
    identity_derivative_spec: Mapping[str, Any] | None = None,
    inventory: batch.SourceInventory | None = None,
    config: batch.AdapterConfig | None = None,
) -> None:
    """Revalidate persisted candidate bytes and selector-specific semantics."""

    expected_record_role = selection_view.get("output_expert_role")
    if (
        record.get("idempotency_key") != job.idempotency_key
        or record.get("record_id") != job.source.record_id
        or record.get("task_bundle") != job.source.task_bundle
        or record.get("semantic") != job.source.semantic
        or record.get("role") != expected_record_role
        or record.get("language") != job.source.language
    ):
        raise batch.AdapterError("vnext_final_record_source_binding_drift")
    source_binding = record.get("source_binding")
    if (
        not isinstance(source_binding, Mapping)
        or any(
            source_binding.get(field) != selection_view.get(field)
            for field in (
                "source_record_id_sha256",
                "source_content_sha256",
                "source_serialization_identity_sha256",
                "task_bundle_sha256",
                "training_asset",
            )
        )
        or source_binding.get("selected_candidate_sha256")
        != selection_view.get("candidate_id_sha256")
    ):
        raise batch.AdapterError("vnext_final_training_projection_binding_drift")
    if inventory is None or config is None:
        if selection_view.get("selection_kind") == "identity":
            raise batch.AdapterError("vnext_identity_derivative_audit_context_missing")
    else:
        prompt_binding(
            job,
            selection_view,
            identity_derivative_spec=identity_derivative_spec,
            inventory=inventory,
            config=config,
        )
    teacher_binding = record.get("teacher_binding")
    if not isinstance(teacher_binding, Mapping):
        raise batch.AdapterError("vnext_final_teacher_binding_missing")
    expected_prompt_version = (
        identity_derivative_spec["v5_prompt_version"]
        if identity_derivative_spec is not None
        else batch.PROMPT_VERSION
    )
    if (
        teacher_binding.get("prompt_version") != expected_prompt_version
        or teacher_binding.get("prompt_template_sha256") != job.prompt_template_sha256
        or teacher_binding.get("output_schema_sha256") != job.output_schema_sha256
        or teacher_binding.get("prompt_template_id")
        != job.source.teacher_contract["prompt_template_id"]
        or teacher_binding.get("output_schema_id")
        != job.source.teacher_contract["output_schema_id"]
    ):
        raise batch.AdapterError("vnext_final_teacher_binding_drift")
    output = record.get("output")
    if not isinstance(output, Mapping):
        raise batch.AdapterError("vnext_final_output_missing")
    transport = _transport_from_persisted_output(job.source, output)
    validated = batch.validate_teacher_output(job.source, transport)
    if dict(output) != validated:
        raise batch.AdapterError("vnext_final_output_normalization_drift")
    structured = validated.get("value")
    if selection_view.get("selection_kind") == "identity":
        if identity_derivative_spec is None:
            raise batch.AdapterError("vnext_identity_derivative_spec_missing")
        identity_class = str(selection_view["identity_class"])
        output_role = str(selection_view["output_expert_role"])
        if (
            identity_derivative_spec["target_identity_class"] != identity_class
            or identity_derivative_spec["output_expert_role"] != output_role
            or identity_derivative_spec["tool_invocation_allowed"] is not False
        ):
            raise batch.AdapterError("vnext_final_identity_contract_drift")
        if output_role == "tool":
            if (
                job.source.role != "identity"
                or selection_view["validation_contract_role"] != "identity"
                or validated.get("kind") != "natural_text"
            ):
                raise batch.AdapterError("vnext_final_identity_tool_asset_drift")
            _assert_identity_invariant(
                structured,
                identity_class,
                exact=True,
            )
        elif output_role in _STYLE_ROLES:
            _assert_identity_invariant(
                structured,
                identity_class,
                exact=False,
            )
        elif output_role == "review":
            if (
                not isinstance(structured, Mapping)
                or structured.get("verdict") != "fail"
                or structured.get("faults") != ["missing_identity_answer"]
            ):
                raise batch.AdapterError(
                    "vnext_final_identity_review_contract_mismatch"
                )
            _assert_identity_invariant(
                structured.get("correction"),
                identity_class,
                exact=True,
            )
        else:
            raise batch.AdapterError("vnext_final_identity_output_role_invalid")
    if (
        selection_view.get("selection_kind") == "tool_review_depth"
        and selection_view.get("output_expert_role") == "review"
    ):
        expected_verdict = selection_view.get("review_verdict")
        expected_faults = (
            [selection_view["review_fault_type"]] if expected_verdict == "fail" else []
        )
        if (
            not isinstance(structured, Mapping)
            or structured.get("verdict") != expected_verdict
            or structured.get("faults") != expected_faults
        ):
            raise batch.AdapterError("vnext_final_review_contract_mismatch")
    if selection_view.get("selection_kind") == "router":
        if not isinstance(structured, Mapping) or structured.get(
            "route"
        ) != selection_view.get("router_label"):
            raise batch.AdapterError("vnext_final_router_contract_mismatch")


class VNextAuthenticatedBatchState(closed.AuthenticatedBatchState):
    """vNext state: one startup replay, O(1) online auth, one final replay."""

    def __init__(
        self,
        config: batch.AdapterConfig,
        *,
        hmac_key: bytes,
        controller_binding: Mapping[str, Any],
        policy: VNextPolicy,
        selection_manifest: Mapping[str, Any],
        selection_views: Mapping[str, Mapping[str, Any]],
        identity_derivative_specs: Mapping[str, Mapping[str, Any]] | None = None,
        job_provenance: Mapping[str, Any] | None = None,
        jobs: Sequence[batch.AlignmentJob] = (),
        stop_probe: Callable[[], bool] | None = None,
        final_quality_validator: Callable[..., None] = final_quality_validate,
        monotonic_clock: Callable[[], float] = time.monotonic,
        observed_at_clock: Callable[[], str] = batch._iso,
    ) -> None:
        super().__init__(
            config,
            hmac_key=hmac_key,
            controller_binding=controller_binding,
        )
        self.policy = policy
        self.selection_manifest = dict(selection_manifest)
        self._selection_views = {
            str(key): dict(value) for key, value in selection_views.items()
        }
        self._identity_derivative_specs = {
            str(key): dict(value)
            for key, value in (identity_derivative_specs or {}).items()
        }
        self._job_provenance = (
            dict(job_provenance) if job_provenance is not None else {}
        )
        self._configured_jobs = tuple(jobs)
        if (
            self.selection_manifest.get("counts", {}).get("selected") != 320
            or not isinstance(
                self.selection_manifest.get("selector_identity_sha256"),
                str,
            )
            or not isinstance(
                self.selection_manifest.get("selected_set_root_sha256"),
                str,
            )
        ):
            raise batch.AdapterError("vnext_minimal320_manifest_invalid")
        self.checkpoint_dir = self.automation_dir / "vnext_delta_checkpoints_v1"
        self.freeze_dir = self.root / "vnext_freeze_v1"
        self.throughput_path = self.automation_dir / "vnext_provider_throughput_v1.json"
        self._append_lock = threading.RLock()
        self._cursor: WalCursor | None = None
        self._replay_cache = batch.ReplayState()
        self._online_ready = False
        self._bootstrapping = False
        self._signed_phase_receipt_ids: set[str] = set()
        self._checkpoint_sequence = 0
        self._prior_checkpoint_sha256: str | None = None
        self._full_replay_counts = Counter()
        self._coverage: dict[CoverageKey, dict[str, int]] = {}
        self._accepted_role_counts: Counter[str] = Counter()
        self._accepted_language_counts: Counter[str] = Counter()
        self._expected_coverage: dict[CoverageKey, int] = {}
        self._jobs_by_key: dict[str, batch.AlignmentJob] = {}
        self._pending_group_id: str | None = None
        self._pending_group_strata: dict[str, dict[str, Any]] = {}
        self._last_checkpoint: dict[str, Any] | None = None
        self._stop_probe = stop_probe
        self._final_quality_validator = final_quality_validator
        self._stop_latch = StopAfterGroupLatch()
        self._throughput = ProviderThroughputAccumulator(
            monotonic_clock=monotonic_clock,
            observed_at_clock=observed_at_clock,
        )
        self._fuse = RejectionRateFuse(
            window=policy.rejection_window,
            min_samples=policy.rejection_min_samples,
            max_rate=policy.rejection_max_rate,
        )

    @property
    def cursor(self) -> WalCursor:
        if self._cursor is None:
            raise batch.AdapterError("vnext_wal_cursor_uninitialized")
        return self._cursor

    @property
    def full_replay_counts(self) -> dict[str, int]:
        return dict(self._full_replay_counts)

    @property
    def stop_latch(self) -> StopAfterGroupLatch:
        with self._append_lock:
            return StopAfterGroupLatch(**vars(self._stop_latch))

    def initialize(
        self,
        inventory: batch.SourceInventory,
        *,
        jobs: Sequence[batch.AlignmentJob] = (),
    ) -> None:
        if not jobs:
            jobs = self._configured_jobs
        self._ensure_genesis()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        closed._reject_reparse_chain(
            self.checkpoint_dir,
            stop=self.root,
        )
        if any(self.checkpoint_dir.glob("*.json")):
            # The policy binds an in-memory HMAC key and explicitly forbids
            # cross-process resume. Authenticate the checkpoint chain first,
            # publish UNKNOWN telemetry, then stop before any dispatch path.
            self._load_checkpoint_cursor()
            raise batch.AdapterError("vnext_cross_process_resume_forbidden")
        self._jobs_by_key = {job.idempotency_key: job for job in jobs}
        if len(self._jobs_by_key) != len(jobs):
            raise batch.AdapterError("vnext_job_identity_duplicate")
        if set(self._jobs_by_key) != set(self._selection_views):
            raise batch.AdapterError("vnext_minimal320_job_set_drift")
        self._job_provenance = _validate_job_provenance(
            self._job_provenance,
            policy=self.policy,
            config=self.config,
            inventory=inventory,
            selection_manifest=self.selection_manifest,
        )
        self._expected_coverage = dict(
            Counter(
                _coverage_key(
                    body_free_stratum(
                        job.source,
                        self._selection_views[job.idempotency_key],
                    )
                )
                for job in jobs
            )
        )
        self._bootstrapping = True
        try:
            batch.BatchState.initialize(self, inventory)
        finally:
            self._bootstrapping = False
        self._replay_cache = batch.BatchState.replay(self)
        self._rebuild_coverage_from_receipts()
        self._load_checkpoint_cursor()
        self.full_replay_audit(stage="startup")
        self._online_ready = True

    def _load_checkpoint_cursor(self) -> None:
        paths = sorted(self.checkpoint_dir.glob("*.json"))
        previous: str | None = None
        for sequence, path in enumerate(paths):
            raw = _physical_read(
                path,
                reason="vnext_checkpoint_snapshot_drift",
                stop=self.root,
            )
            value = _strict_json_mapping(raw, reason="vnext_checkpoint_invalid")
            payload = self._verify_signed_value(
                value,
                reason="vnext_checkpoint_hmac_drift",
            )
            sidecar_path = path.with_name(f"{path.name}.sha256")
            sidecar = _physical_read(
                sidecar_path,
                reason="vnext_checkpoint_sidecar_missing",
                stop=self.root,
            )
            expected_sidecar = f"{_sha256_bytes(raw)}  {path.name}\n".encode("ascii")
            if (
                payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or payload.get("checkpoint_sequence") != sequence
                or payload.get("prior_checkpoint_sha256") != previous
                or not hmac.compare_digest(sidecar, expected_sidecar)
            ):
                raise batch.AdapterError("vnext_checkpoint_chain_drift")
            previous = _sha256_bytes(raw)
        self._checkpoint_sequence = len(paths)
        self._prior_checkpoint_sha256 = previous
        if paths:
            self._write_throughput_telemetry(self._throughput.mark_restart_forbidden())

    def _write_throughput_telemetry(
        self,
        telemetry: Mapping[str, Any],
    ) -> None:
        batch._atomic_write_json(
            self.throughput_path,
            {
                **dict(telemetry),
                "controller_run_id": self.controller_binding["controller_run_id"],
            },
        )

    def _observe_terminal_throughput(
        self,
        usage: Mapping[str, Any] | None,
    ) -> None:
        with self._append_lock:
            self._write_throughput_telemetry(self._throughput.observe_terminal(usage))

    def observe_provider_terminal_usage(
        self,
        usage: Mapping[str, Any] | None,
    ) -> None:
        """Persist one provider terminal before any group can commit it."""

        self._observe_terminal_throughput(usage)

    def _rebuild_coverage_from_receipts(self) -> None:
        self._coverage.clear()
        self._accepted_role_counts.clear()
        self._accepted_language_counts.clear()
        for path, outcome_name in (
            (self.receipts_path, "accepted"),
            (self.rejections_path, "rejected"),
        ):
            rows = closed._stable_jsonl(path, id_field="id")
            for receipt in rows:
                batch._verify_receipt(receipt, self.hmac_key or b"")
                stratum = receipt.get("stratum")
                if not isinstance(stratum, Mapping):
                    raise batch.AdapterError("vnext_receipt_stratum_missing")
                key = _coverage_key(stratum)
                counts = self._coverage.setdefault(
                    key,
                    {"accepted": 0, "rejected": 0, "quarantine": 0},
                )
                counts[outcome_name] += 1
                if outcome_name == "accepted":
                    self._accepted_role_counts[str(stratum["role"])] += 1
                    self._accepted_language_counts[str(stratum["language"])] += 1

    def full_replay_audit(self, *, stage: str) -> FullReplayAudit:
        if stage not in {"startup", "final_freeze"}:
            raise batch.AdapterError("vnext_full_replay_stage_invalid")
        with self._append_lock:
            if self._full_replay_counts[stage] != 0:
                raise batch.AdapterError("vnext_full_replay_duplicate")
            (
                entries,
                authoritative_events,
                phase_receipts,
                job_heads,
            ) = self._read_entries()
            aggregate_raw = (
                _physical_read(
                    self.events_path,
                    reason="vnext_events_snapshot_drift",
                    stop=self.root,
                )
                if self.events_path.exists()
                else b""
            )
            aggregate_events = closed._parse_jsonl_snapshot(
                aggregate_raw,
                reason="vnext_events_projection_invalid",
            )
            if aggregate_events != authoritative_events:
                raise batch.AdapterError("vnext_events_projection_drift")
            closed._validate_event_transitions(authoritative_events)
            physical_phase_receipts = closed._stable_jsonl(
                self.phase_receipts_path,
                id_field="id",
            )
            if physical_phase_receipts != phase_receipts:
                raise batch.AdapterError("vnext_phase_receipt_projection_drift")
            self._terminal_cross_bind(authoritative_events, entries)
            paths = self._entry_paths()
            genesis_sha256 = closed._sha256_file(self.genesis_path)
            tip = closed._sha256_file(paths[-1]) if paths else genesis_sha256
            observed_cursor = WalCursor(
                sequence=len(entries),
                chain_tip_sha256=tip,
                genesis_sha256=genesis_sha256,
                job_heads=dict(job_heads),
            )
            if self._cursor is not None and (
                self._cursor.sequence != observed_cursor.sequence
                or not hmac.compare_digest(
                    self._cursor.chain_tip_sha256,
                    observed_cursor.chain_tip_sha256,
                )
                or self._cursor.job_heads != observed_cursor.job_heads
            ):
                raise batch.AdapterError("vnext_held_cursor_drift")
            self._cursor = observed_cursor
            self._signed_phase_receipt_ids = {
                str(receipt["id"]) for receipt in phase_receipts
            }
            self._full_replay_counts[stage] += 1
            return FullReplayAudit(
                stage=stage,
                wal_entries=len(entries),
                terminal_events=sum(
                    event.get("state") in {"committed", "rejected"}
                    for event in authoritative_events
                ),
                phase_receipts=len(phase_receipts),
                chain_tip_sha256=tip,
            )

    def _online_guard_locked(self) -> None:
        cursor = self.cursor
        if cursor.sequence == 0:
            observed = closed._sha256_file(self.genesis_path)
        else:
            tip_path = self.wal_dir / f"{cursor.sequence - 1:016d}.json"
            observed = closed._sha256_file(tip_path)
        next_path = self.wal_dir / f"{cursor.sequence:016d}.json"
        if (
            not hmac.compare_digest(observed, cursor.chain_tip_sha256)
            or next_path.exists()
        ):
            raise batch.AdapterError("vnext_online_chain_tip_drift")

    def authenticate_authority(
        self,
        *,
        allow_inflight_prepare: bool = False,
    ) -> None:
        del allow_inflight_prepare
        if not self._online_ready:
            return
        with self._append_lock:
            self._online_guard_locked()

    def _append_wal_entry(
        self,
        *,
        entry_kind: str,
        job_keys: Sequence[str],
        body: Mapping[str, Any],
        allow_inflight_prepare: bool = False,
    ) -> str:
        del allow_inflight_prepare
        with self._append_lock:
            cursor = self.cursor
            self._online_guard_locked()
            keys = list(job_keys)
            if len(keys) != len(set(keys)) or not all(
                isinstance(key, str) and key for key in keys
            ):
                raise batch.AdapterError("vnext_wal_job_key_invalid")
            payload = {
                "schema_version": closed.WAL_CHECKPOINT_SCHEMA_VERSION,
                "sequence": cursor.sequence,
                "entry_kind": entry_kind,
                "controller_run_id": self.controller_binding["controller_run_id"],
                "process_id": os.getpid(),
                "previous_entry_sha256": cursor.chain_tip_sha256,
                "job_keys": keys,
                "previous_job_entry_sha256": {
                    key: cursor.job_heads.get(key) for key in keys
                },
                "active_profile_id": self.config.profile_id,
                "active_profile_sha256": self.config.physical_sha256,
                "campaign_sha256": self.config.campaign_sha256,
                "source_identity_sha256": self.controller_binding[
                    "source_identity_sha256"
                ],
                "manifest_sha256": self.controller_binding["manifest_sha256"],
                "controller_config_sha256": self.controller_binding[
                    "controller_config_sha256"
                ],
                "teacher_implementation_sha256": self.controller_binding[
                    "teacher_implementation_sha256"
                ],
                "body": dict(body),
                "content_retained": False,
            }
            path = self.wal_dir / f"{cursor.sequence:016d}.json"
            batch._exclusive_write(path, self._signed_value(payload))
            digest = closed._sha256_file(path)
            cursor.sequence += 1
            cursor.chain_tip_sha256 = digest
            for key in keys:
                cursor.job_heads[key] = digest
            return digest

    def recover_group_commits(self) -> int:
        if self._bootstrapping:
            return batch.BatchState.recover_group_commits(self)
        if self._inflight_group_id is None:
            return 0
        return self._recover_one_group(self._inflight_group_id)

    def _recover_one_group(self, group_id: str) -> int:
        path = self.groups_dir / f"{group_id}.json"
        raw = _physical_read(
            path,
            reason="vnext_group_snapshot_drift",
            stop=self.root,
        )
        group = _strict_json_mapping(raw, reason="vnext_group_invalid")
        payload = dict(group)
        group_sha256 = payload.pop("group_sha256", None)
        if (
            group.get("schema_version") != batch.GROUP_SCHEMA_VERSION
            or group.get("group_id") != group_id
            or not isinstance(group_sha256, str)
            or not hmac.compare_digest(
                group_sha256,
                batch._hash_object(payload),
            )
        ):
            raise batch.AdapterError("vnext_group_identity_drift")
        record_store = self._record_store or batch.JsonlStore(
            self.records_path,
            id_field="idempotency_key",
        )
        rejection_store = self._rejection_store or batch.JsonlStore(
            self.rejections_path,
            id_field="idempotency_key",
        )
        receipt_store = self._receipt_store or batch.JsonlStore(
            self.receipts_path,
            id_field="id",
        )
        event_store = self._event_store or batch.JsonlStore(
            self.events_path,
            id_field="event_id",
        )
        recovered = 0
        for receipt in [*group["receipts"], *group["rejections"]]:
            if not isinstance(receipt, Mapping):
                raise batch.AdapterError("vnext_group_receipt_invalid")
            batch._verify_receipt(receipt, self.hmac_key or b"")
            key = str(receipt.get("idempotency_key"))
            expected = self._pending_group_strata.get(key)
            if expected is None or receipt.get("stratum") != expected:
                raise batch.AdapterError("vnext_group_stratum_binding_drift")
        for receipt in group["receipts"]:
            if receipt_store.append(receipt):
                recovered += 1
        for record in group["records"]:
            if not isinstance(record, Mapping):
                raise batch.AdapterError("vnext_group_record_invalid")
            if record_store.append(record):
                recovered += 1
        for rejection in group["rejections"]:
            if rejection_store.append(rejection):
                recovered += 1
        terminal_events: list[Mapping[str, Any]] = []
        for event in group["events"]:
            if not isinstance(event, Mapping):
                raise batch.AdapterError("vnext_group_event_invalid")
            if event_store.append(event):
                recovered += 1
            terminal_events.append(event)
        self._record_store = record_store
        self._rejection_store = rejection_store
        self._receipt_store = receipt_store
        self._event_store = event_store
        self._apply_terminal_delta(group, terminal_events)
        return recovered

    def _refresh_event_cache(self) -> None:
        if self._online_ready or (not self._bootstrapping and self._cursor is not None):
            return
        batch.BatchState._refresh_event_cache(self)

    def _apply_terminal_delta(
        self,
        group: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        replay = self._replay_cache
        rows_by_key: dict[str, Mapping[str, Any]] = {}
        for row in [*group["records"], *group["rejections"]]:
            key = str(row["idempotency_key"])
            if key in rows_by_key:
                raise batch.AdapterError("vnext_group_terminal_duplicate")
            rows_by_key[key] = row
            usage = row.get("usage")
            attempts = row.get("attempts")
            if (
                not isinstance(usage, Mapping)
                or not isinstance(attempts, Mapping)
                or ProviderThroughputAccumulator._usage_tokens(usage) is None
            ):
                raise batch.AdapterError("vnext_group_usage_invalid")
            replay.input_tokens += int(usage["input_tokens"])
            replay.output_tokens += int(usage["output_tokens"])
            replay.request_attempts += int(attempts["wire_attempts"])
            if int(attempts.get("retry_count", 0)) > 0:
                replay.retried.add(key)
        for event in events:
            key = str(event["idempotency_key"])
            state = str(event["state"])
            previous = replay.latest_state.get(key)
            if state not in _ALLOWED_TRANSITIONS.get(previous, frozenset()):
                raise batch.AdapterError("vnext_event_transition_drift")
            replay.latest_state[key] = state
            replay.events_by_job.setdefault(key, []).append(dict(event))
            replay.uncertain.discard(key)
            self._active_reservation_cache.pop(key, None)
            if state == "committed":
                if key in replay.completed | replay.rejected:
                    raise batch.AdapterError("vnext_terminal_identity_duplicate")
                replay.completed.add(key)
            elif state == "rejected":
                if key in replay.completed | replay.rejected:
                    raise batch.AdapterError("vnext_terminal_identity_duplicate")
                replay.rejected.add(key)
            else:
                raise batch.AdapterError("vnext_group_terminal_state_invalid")
        self._event_counter += len(events)
        replay.cost_units = replay.request_attempts

    def replay(self) -> batch.ReplayState:
        if not self._online_ready and not self._bootstrapping:
            return batch.BatchState.replay(self)
        self.authenticate_authority()
        # Runner consumers treat ReplayState as read-only.  Returning the held
        # aggregate avoids rebuilding O(history) sets and event lists per group.
        return self._replay_cache

    async def append_event(self, **kwargs: Any) -> dict[str, Any]:
        async with self._controller_wal_lock:
            self.authenticate_authority()
            job = kwargs.get("job")
            state_name = kwargs.get("state")
            if not isinstance(job, batch.AlignmentJob) or not isinstance(
                state_name,
                str,
            ):
                raise batch.AdapterError("vnext_event_input_invalid")
            previous = self._replay_cache.latest_state.get(job.idempotency_key)
            if state_name not in _ALLOWED_TRANSITIONS.get(previous, frozenset()):
                raise batch.AdapterError("vnext_event_transition_drift")
            event = await batch.BatchState.append_event(self, **kwargs)
            self._append_wal_entry(
                entry_kind="EVENT",
                job_keys=[job.idempotency_key],
                body={"event": event},
            )
            self._apply_online_event(event)
            self.authenticate_authority()
            return event

    def _apply_online_event(self, event: Mapping[str, Any]) -> None:
        replay = self._replay_cache
        key = str(event["idempotency_key"])
        state_name = str(event["state"])
        replay.latest_state[key] = state_name
        replay.events_by_job.setdefault(key, []).append(dict(event))
        if state_name in {"retryable", "uncertain"}:
            attempts = event.get("attempts")
            if not isinstance(attempts, Mapping):
                raise batch.AdapterError("vnext_failure_attempts_missing")
            replay.request_attempts += int(attempts["wire_attempts"])
            replay.cost_units = replay.request_attempts
            if int(attempts.get("retry_count", 0)) > 0:
                replay.retried.add(key)
        if state_name in {"dispatching", "uncertain"}:
            replay.uncertain.add(key)
        elif state_name != "reserved":
            replay.uncertain.discard(key)

    def request_stop_after_group(self, reason_code: str) -> None:
        if _SAFE_STOP_REASON.fullmatch(reason_code) is None:
            raise batch.AdapterError("vnext_stop_reason_invalid")
        with self._append_lock:
            if self._stop_latch.effective:
                return
            if not self._stop_latch.requested:
                self._stop_latch.requested = True
                self._stop_latch.reason_code = reason_code
                self._stop_latch.requested_at_sequence = self.cursor.sequence

    def _observe_stop_probe(self) -> None:
        if self._stop_probe is not None and self._stop_probe():
            self.request_stop_after_group("kill_switch_armed")

    def stop_at_durable_group_boundary(self, reason_code: str) -> None:
        """Make an already-durable checkpoint the safe-stop boundary."""

        with self._append_lock:
            if (
                self._pending_group_id is not None
                or self._checkpoint_sequence < 1
                or self._prior_checkpoint_sha256 is None
            ):
                raise batch.AdapterError("vnext_zero_group_kill_switch_hard_block")
            self.request_stop_after_group(reason_code)
            self._stop_latch.effective = True
            self._stop_latch.effective_checkpoint_sha256 = self._prior_checkpoint_sha256
            raise StopAfterGroup(reason_code)

    def commit_group(
        self,
        pending: Sequence[batch.PendingCommit],
        *,
        inventory: batch.SourceInventory,
        phase: str,
        expected_job_keys: Sequence[str] | None = None,
    ) -> None:
        if not pending:
            return
        with self._append_lock:
            self._observe_stop_probe()
            keys = [item.job.idempotency_key for item in pending]
            expected = (
                list(expected_job_keys) if expected_job_keys is not None else keys
            )
            if len(keys) != len(set(keys)):
                raise batch.AdapterError("vnext_group_job_duplicate")
            if (
                len(expected) != len(set(expected))
                or len(keys) != len(expected)
                or set(keys) != set(expected)
            ):
                raise batch.AdapterError("vnext_group_expected_inventory_incomplete")
            for item in pending:
                previous = self._replay_cache.latest_state.get(item.job.idempotency_key)
                if (
                    previous != "dispatching"
                    or item.terminal_state not in {"committed", "rejected"}
                    or (item.output_record is None)
                    != (item.terminal_state == "rejected")
                ):
                    raise batch.AdapterError("vnext_group_terminal_invalid")
            prior_tip = self.cursor.chain_tip_sha256
            prior_sequence = self.cursor.sequence
            group_id = batch._hash_object({"phase": phase, "keys": keys})
            self._pending_group_id = group_id
            self._pending_group_strata = {
                item.job.idempotency_key: body_free_stratum(
                    item.job.source,
                    self._selection_views.get(item.job.idempotency_key),
                )
                for item in pending
            }
            try:
                closed.AuthenticatedBatchState.commit_group(
                    self,
                    pending,
                    inventory=inventory,
                    phase=phase,
                )
                checkpoint = self._write_delta_checkpoint(
                    pending,
                    group_id=group_id,
                    phase=phase,
                    prior_chain_tip_sha256=prior_tip,
                    prior_sequence=prior_sequence,
                )
                replay = self._replay_cache
                if (
                    set(keys) & replay.uncertain
                    or not set(keys) <= replay.completed | replay.rejected
                ):
                    raise batch.AdapterError("vnext_group_terminal_replay_incomplete")
                fuse = self._fuse.observe(
                    [item.terminal_state == "rejected" for item in pending]
                )
                if self._stop_latch.requested:
                    self._stop_latch.effective = True
                    self._stop_latch.effective_checkpoint_sha256 = _sha256_bytes(
                        _canonical_bytes(checkpoint, newline=True)
                    )
                if fuse["latched"]:
                    raise batch.AdapterError("vnext_systematic_rejection_rate_fuse")
                if self._stop_latch.effective:
                    raise StopAfterGroup()
            finally:
                self._pending_group_id = None
                self._pending_group_strata = {}

    def _write_delta_checkpoint(
        self,
        pending: Sequence[batch.PendingCommit],
        *,
        group_id: str,
        phase: str,
        prior_chain_tip_sha256: str,
        prior_sequence: int,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        usage = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retried": 0,
        }
        accepted = 0
        rejected = 0
        delta_coverage: dict[CoverageKey, dict[str, int]] = {}
        for item in pending:
            receipt = item.receipt
            stratum = body_free_stratum(
                item.job.source,
                self._selection_views.get(item.job.idempotency_key),
            )
            if receipt.get("stratum") != stratum:
                raise batch.AdapterError("vnext_checkpoint_stratum_drift")
            outcome = str(receipt["outcome"])
            outcome_name = "accepted" if outcome == "succeeded" else "rejected"
            if outcome_name == "accepted":
                accepted += 1
            else:
                rejected += 1
            key = _coverage_key(stratum)
            counts = self._coverage.setdefault(
                key,
                {"accepted": 0, "rejected": 0, "quarantine": 0},
            )
            counts[outcome_name] += 1
            delta_counts = delta_coverage.setdefault(
                key,
                {"accepted": 0, "rejected": 0, "quarantine": 0},
            )
            delta_counts[outcome_name] += 1
            if outcome_name == "accepted":
                self._accepted_role_counts[stratum["role"]] += 1
                self._accepted_language_counts[stratum["language"]] += 1
            attempts = receipt["attempts"]
            receipt_usage = receipt["usage"]
            usage["requests"] += int(attempts["wire_attempts"])
            usage["input_tokens"] += int(receipt_usage["input_tokens"])
            usage["output_tokens"] += int(receipt_usage["output_tokens"])
            usage["retried"] += int(int(attempts.get("retry_count", 0)) > 0)
            rows.append(
                {
                    "idempotency_key": item.job.idempotency_key,
                    "outcome": outcome_name,
                    "reason_code": item.event_reason,
                    "receipt_sha256": batch._hash_object(receipt),
                    "output_record_sha256": receipt["output_record_sha256"],
                    "stratum": stratum,
                    "content_retained": False,
                }
            )
        rows.sort(key=lambda row: str(row["idempotency_key"]))
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_sequence": self._checkpoint_sequence,
            "controller_run_id": self.controller_binding["controller_run_id"],
            "group_id": group_id,
            "phase": phase,
            "profile_id": self.config.profile_id,
            "profile_sha256": self.config.physical_sha256,
            "selector_identity_sha256": self.selection_manifest[
                "selector_identity_sha256"
            ],
            "selected_set_root_sha256": self.selection_manifest[
                "selected_set_root_sha256"
            ],
            "selected_set_count": 320,
            "prior_checkpoint_sha256": self._prior_checkpoint_sha256,
            "prior_chain_tip_sha256": prior_chain_tip_sha256,
            "new_chain_tip_sha256": self.cursor.chain_tip_sha256,
            "wal_sequence_start": prior_sequence,
            "wal_sequence_end": self.cursor.sequence,
            "delta_inventory": rows,
            "delta_inventory_root_sha256": _merkle_root(rows),
            "counts": {
                "requested": len(pending),
                "completed": len(pending),
                "accepted": accepted,
                "rejected": rejected,
                "quarantine": 0,
            },
            "acceptance_scope": "online_structural_candidate_only",
            "usage": usage,
            "coverage": _coverage_rows(delta_coverage),
            "stop_after_group_requested": self._stop_latch.requested,
            "content_retained": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        }
        checkpoint = self._signed_value(payload)
        schema_raw = _physical_read(
            self.policy.checkpoint_schema_path,
            reason="vnext_checkpoint_schema_snapshot_drift",
            stop=batch.REPO_ROOT.absolute(),
        )
        schema = _strict_json_mapping(
            schema_raw,
            reason="vnext_checkpoint_schema_invalid",
        )
        try:
            Draft202012Validator(schema).validate(checkpoint)
        except ValidationError as error:
            raise batch.AdapterError("vnext_checkpoint_schema_mismatch") from error
        filename = f"{self._checkpoint_sequence:016d}-{group_id}.json"
        path = self.checkpoint_dir / filename
        raw = _canonical_bytes(checkpoint, newline=True)
        _exclusive_bytes(path, raw)
        digest = _sha256_bytes(raw)
        _exclusive_bytes(
            path.with_name(f"{path.name}.sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
        self._checkpoint_sequence += 1
        self._prior_checkpoint_sha256 = digest
        self._last_checkpoint = checkpoint
        return checkpoint

    def write_status(
        self,
        *,
        inventory: batch.SourceInventory,
        jobs: Sequence[batch.AlignmentJob],
        phase_jobs: Sequence[batch.AlignmentJob],
        phase: str,
        state_name: str,
        started_monotonic: float | None,
        cooldown_until: str | None = None,
        blockers: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Write status from held counters without scanning the selected set."""

        replay = self.replay()
        total = len(self._jobs_by_key)
        if len(jobs) != total or len(phase_jobs) != total:
            raise batch.AdapterError("vnext_status_job_set_drift")
        succeeded = len(replay.completed)
        rejected = len(replay.rejected)
        uncertain = len(replay.uncertain)
        queued = total - succeeded - rejected - uncertain
        if queued < 0:
            raise batch.AdapterError("vnext_status_counter_drift")
        elapsed = (
            max(0.000001, time.monotonic() - started_monotonic)
            if started_monotonic is not None
            else None
        )
        rate = succeeded / elapsed if elapsed is not None else None
        eta = queued / rate if rate and queued else None
        status = {
            "schema_version": batch.STATUS_SCHEMA_VERSION,
            "dataset_kind": batch.DATASET_KIND,
            "state": state_name,
            "profile": self.config.profile_id,
            "phase": phase,
            "updated_at": batch._iso(),
            "total": total,
            "queued": queued,
            "inflight": uncertain,
            "succeeded": succeeded,
            "rejected": rejected,
            "retried": len(replay.retried),
            "role_counts": dict(sorted(self._accepted_role_counts.items())),
            "language_counts": dict(sorted(self._accepted_language_counts.items())),
            "provider_usage": {
                "requests": replay.request_attempts,
                "input_tokens": replay.input_tokens,
                "output_tokens": replay.output_tokens,
                "usage_only": True,
            },
            "provider_throughput": self._throughput.snapshot(),
            "rate": {
                "jobs_per_second": (round(rate, 6) if rate is not None else None),
                "eta_seconds": round(eta, 2) if eta is not None else None,
            },
            "cooldown_until": cooldown_until,
            "hashes": {
                "source": inventory.source_identity_sha256,
                "config": self.config.physical_sha256,
                "campaign": self.config.campaign_sha256,
                "model": self.config.model_binding_sha256,
                "implementation": self.config.implementation_sha256,
                "contracts": batch._hash_object(self.config.contract_hashes),
                "vnext_config": self.policy.physical_sha256,
                "selector": self.selection_manifest["selector_identity_sha256"],
            },
            "resume": {
                "completed": succeeded,
                "uncertain": uncertain,
                "cross_process_resume": False,
                "duplicate_paid_call_prevention": (
                    "fresh_root_and_uncertain_never_redispatched"
                ),
            },
            "cost_guard": {
                "basis": self.config.limits.cost_basis,
                "used_units": replay.cost_units,
                "maximum_units": self.config.limits.max_cost_units,
                "marginal_currency_cost_known": False,
            },
            "kill_switch": {
                "armed": self._stop_latch.requested,
                "checked_before_each_group": True,
                "stop_after_group": True,
            },
            "blockers": list(dict.fromkeys(blockers)),
            "content_free": True,
        }
        batch._atomic_write_json(self.status_path, status)
        return status

    def append_phase_receipt(self, **kwargs: Any) -> dict[str, Any]:
        self.authenticate_authority()
        receipt = batch.BatchState.append_phase_receipt(self, **kwargs)
        receipt_id = str(receipt["id"])
        if receipt_id not in self._signed_phase_receipt_ids:
            self._append_wal_entry(
                entry_kind="PHASE_RECEIPT",
                job_keys=[],
                body={"receipt": receipt},
            )
            self._signed_phase_receipt_ids.add(receipt_id)
        self.authenticate_authority()
        return receipt

    def freeze_handoff(
        self,
        *,
        inventory: batch.SourceInventory,
        jobs: Sequence[batch.AlignmentJob],
        terminal_transition_recheck: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Full final replay and body-free accepted/rejected/quarantine handoff."""

        with self._append_lock:
            required_terminal_recheck = {
                "schema_version",
                "transition_evidence_sha256",
                "policy_revalidation_sha256",
                "old_controller_pid",
                "old_archive_identity_sha256",
                "old_archive_file_count",
                "controller_count",
                "candidate_controller_count",
                "candidate_launch_process_count",
                "candidate_controller_pid",
                "controller_inventory_sha256",
                "all_matching_inventory_sha256",
                "canonical_root_absent",
                "vnext_root_present",
                "content_retained",
                "recheck_sha256",
            }
            terminal_payload = dict(terminal_transition_recheck)
            terminal_digest = terminal_payload.pop("recheck_sha256", None)
            if (
                set(terminal_transition_recheck) != required_terminal_recheck
                or not isinstance(terminal_digest, str)
                or not hmac.compare_digest(
                    terminal_digest,
                    _sha256_bytes(_canonical_bytes(terminal_payload)),
                )
                or terminal_payload["controller_count"] != 0
                or terminal_payload["candidate_controller_count"] != 1
                or not isinstance(
                    terminal_payload["candidate_launch_process_count"],
                    int,
                )
                or terminal_payload["candidate_launch_process_count"] < 1
                or terminal_payload["candidate_controller_pid"] != os.getpid()
                or terminal_payload["canonical_root_absent"] is not True
                or terminal_payload["vnext_root_present"] is not True
                or terminal_payload["content_retained"] is not False
            ):
                raise batch.AdapterError("vnext_freeze_transition_recheck_invalid")
            self._observe_stop_probe()
            replay = self.replay()
            if replay.uncertain:
                raise batch.AdapterError("vnext_freeze_uncertain_forbidden")
            final_audit = self.full_replay_audit(stage="final_freeze")
            jobs_by_key = {job.idempotency_key: job for job in jobs}
            if len(jobs_by_key) != len(jobs):
                raise batch.AdapterError("vnext_freeze_job_identity_duplicate")
            accepted, rejected = self._audit_final_candidates(
                jobs_by_key=jobs_by_key,
                inventory=inventory,
            )
            terminal = {str(row["idempotency_key"]) for row in [*accepted, *rejected]}
            quarantine = [
                {
                    "schema_version": INVENTORY_SCHEMA_VERSION,
                    "idempotency_key": key,
                    "outcome": "quarantine",
                    "reason_code": "nonterminal_at_freeze",
                    "receipt_sha256": None,
                    "output_record_sha256": None,
                    "stratum": body_free_stratum(
                        jobs_by_key[key].source,
                        self._selection_views.get(key),
                    ),
                    "content_retained": False,
                }
                for key in sorted(set(jobs_by_key) - terminal)
            ]
            freeze_dir = self.freeze_dir
            freeze_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_manifest = self._write_checkpoint_manifest(freeze_dir)
            inventories = {
                "accepted": accepted,
                "rejected": rejected,
                "quarantine": quarantine,
            }
            authenticated_inventories: dict[str, list[dict[str, Any]]] = {}
            inventory_bindings: dict[str, dict[str, Any]] = {}
            for name, rows in inventories.items():
                path = freeze_dir / f"{name}_inventory.jsonl"
                authenticated_rows, binding = (
                    self._write_and_authenticate_freeze_inventory(
                        path,
                        rows=rows,
                        outcome=name,
                        jobs_by_key=jobs_by_key,
                    )
                )
                authenticated_inventories[name] = authenticated_rows
                inventory_bindings[name] = binding
            accepted = authenticated_inventories["accepted"]
            rejected = authenticated_inventories["rejected"]
            quarantine = authenticated_inventories["quarantine"]
            final_coverage: dict[CoverageKey, dict[str, int]] = {}
            accepted_role_counts: Counter[str] = Counter()
            accepted_language_counts: Counter[str] = Counter()
            for name, rows in authenticated_inventories.items():
                for row in rows:
                    stratum = row["stratum"]
                    key = _coverage_key(stratum)
                    counts = final_coverage.setdefault(
                        key,
                        {"accepted": 0, "rejected": 0, "quarantine": 0},
                    )
                    counts[name] += 1
                    if name == "accepted":
                        accepted_role_counts[str(stratum["role"])] += 1
                        accepted_language_counts[str(stratum["language"])] += 1
            requested = len(jobs)
            coverage_rows = _coverage_rows(final_coverage)
            payload = {
                "schema_version": FREEZE_SCHEMA_VERSION,
                "state": "candidate_pending_external_attestation",
                "controller_run_id": self.controller_binding["controller_run_id"],
                "source_identity_sha256": inventory.source_identity_sha256,
                "manifest_sha256": inventory.manifest_sha256,
                "selector_identity_sha256": self.selection_manifest[
                    "selector_identity_sha256"
                ],
                "selected_set_root_sha256": self.selection_manifest[
                    "selected_set_root_sha256"
                ],
                "selected_set_count": 320,
                "pre_dispatch_transition_recheck_sha256": (
                    self.controller_binding["terminal_transition_recheck_sha256"]
                ),
                "pre_freeze_transition_recheck_sha256": terminal_digest,
                "old_archive_identity_sha256": terminal_payload[
                    "old_archive_identity_sha256"
                ],
                "controller_inventory_sha256": terminal_payload[
                    "controller_inventory_sha256"
                ],
                "checkpoint_manifest_root_sha256": checkpoint_manifest[
                    "checkpoint_manifest_root_sha256"
                ],
                "checkpoint_manifest_sha256": checkpoint_manifest[
                    "checkpoint_manifest_sha256"
                ],
                "final_chain_tip_sha256": final_audit.chain_tip_sha256,
                "final_wal_entries": final_audit.wal_entries,
                "counts": {
                    "requested": requested,
                    "completed": len(accepted) + len(rejected),
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                    "quarantine": len(quarantine),
                },
                "inventory_sha256": {
                    name: binding["sha256"]
                    for name, binding in inventory_bindings.items()
                },
                "inventory_bindings": inventory_bindings,
                "coverage": coverage_rows,
                "accepted_strata_root_sha256": _merkle_root(
                    [row for row in coverage_rows if int(row["accepted"]) > 0]
                ),
                "output_expert_role_counts": dict(sorted(accepted_role_counts.items())),
                "language_counts": dict(sorted(accepted_language_counts.items())),
                "selection_counts": dict(self.selection_manifest["counts"]),
                "coverage_gaps": coverage_gaps(
                    self._expected_coverage,
                    final_coverage,
                ),
                "full_replay_counts": self.full_replay_counts,
                "job_provenance_identity_sha256": self._job_provenance[
                    "provenance_identity_sha256"
                ],
                "producer_launch_L": dict(self._job_provenance["producer_launch_L"]),
                "consumer_code_P": dict(self._job_provenance["consumer_code_P"]),
                "external_attestation_required": True,
                "training_authorized": False,
                "formal_training_authorized": False,
                "content_retained": False,
                "raw_token_ids_retained": False,
                "credential_retained": False,
            }
            attestation = self._signed_value(payload)
            path = freeze_dir / "freeze_attestation.json"
            raw = _canonical_bytes(attestation, newline=True)
            _exclusive_bytes(path, raw)
            digest = _sha256_bytes(raw)
            _exclusive_bytes(
                path.with_name(f"{path.name}.sha256"),
                f"{digest}  {path.name}\n".encode("ascii"),
            )
            if (
                _authenticated_file_digest(
                    path,
                    stop=freeze_dir,
                    reason="vnext_freeze_attestation_output_drift",
                )
                != digest
                or _strict_json_mapping(
                    _physical_read(
                        path,
                        reason="vnext_freeze_attestation_output_drift",
                        stop=freeze_dir,
                    ),
                    reason="vnext_freeze_attestation_output_drift",
                )
                != attestation
            ):
                raise batch.AdapterError("vnext_freeze_attestation_output_drift")
            return attestation

    def _audit_final_candidates(
        self,
        *,
        jobs_by_key: Mapping[str, batch.AlignmentJob],
        inventory: batch.SourceInventory,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records = _physical_jsonl(
            self.records_path,
            id_field="idempotency_key",
            stop=self.root,
            reason="vnext_final_records_snapshot_drift",
        )
        receipts = _physical_jsonl(
            self.receipts_path,
            id_field="id",
            stop=self.root,
            reason="vnext_final_receipts_snapshot_drift",
        )
        online_rejections = _physical_jsonl(
            self.rejections_path,
            id_field="id",
            stop=self.root,
            reason="vnext_final_rejections_snapshot_drift",
        )
        records_by_key = {str(record["idempotency_key"]): record for record in records}
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        candidate_keys: set[str] = set()
        for receipt in receipts:
            batch._verify_receipt(receipt, self.hmac_key or b"")
            key = str(receipt.get("idempotency_key"))
            job = jobs_by_key.get(key)
            record = records_by_key.get(key)
            if job is None or record is None or key in candidate_keys:
                raise batch.AdapterError("vnext_final_candidate_join_drift")
            candidate_keys.add(key)
            if receipt.get("outcome") != "succeeded" or not hmac.compare_digest(
                str(receipt.get("output_record_sha256")),
                batch._hash_object(record),
            ):
                raise batch.AdapterError("vnext_final_candidate_hash_drift")
            stratum = body_free_stratum(
                job.source,
                self._selection_views.get(key),
            )
            if receipt.get("stratum") != stratum:
                raise batch.AdapterError("vnext_final_stratum_drift")
            if receipt.get("provenance") != self._job_provenance:
                raise batch.AdapterError("vnext_final_job_provenance_drift")
            expected_prompt_binding = prompt_binding(
                job,
                self._selection_views[key],
                identity_derivative_spec=(self._identity_derivative_specs.get(key)),
                inventory=inventory,
                config=self.config,
            )
            if receipt.get("prompt_binding") != expected_prompt_binding:
                raise batch.AdapterError("vnext_final_prompt_binding_drift")
            outcome = "accepted"
            reason_code = "final_quality_validated"
            try:
                self._final_quality_validator(
                    job,
                    record,
                    self._selection_views[key],
                    identity_derivative_spec=(self._identity_derivative_specs.get(key)),
                    inventory=inventory,
                    config=self.config,
                )
            except batch.AdapterError as error:
                outcome = "rejected"
                reason_code = f"final_quality_{error.reason_code}"
            row = {
                "schema_version": INVENTORY_SCHEMA_VERSION,
                "idempotency_key": key,
                "outcome": outcome,
                "reason_code": reason_code,
                "receipt_sha256": batch._hash_object(receipt),
                "output_record_sha256": batch._hash_object(record),
                "stratum": stratum,
                "content_retained": False,
            }
            (accepted if outcome == "accepted" else rejected).append(row)
        if set(records_by_key) != candidate_keys:
            raise batch.AdapterError("vnext_final_orphan_record")
        for receipt in online_rejections:
            batch._verify_receipt(receipt, self.hmac_key or b"")
            key = str(receipt.get("idempotency_key"))
            job = jobs_by_key.get(key)
            if (
                job is None
                or key in candidate_keys
                or receipt.get("outcome") != "rejected"
                or receipt.get("output_record_sha256") is not None
            ):
                raise batch.AdapterError("vnext_final_online_rejection_drift")
            candidate_keys.add(key)
            stratum = body_free_stratum(
                job.source,
                self._selection_views.get(key),
            )
            if receipt.get("stratum") != stratum:
                raise batch.AdapterError("vnext_final_stratum_drift")
            if receipt.get("provenance") != self._job_provenance:
                raise batch.AdapterError("vnext_final_job_provenance_drift")
            expected_prompt_binding = prompt_binding(
                job,
                self._selection_views[key],
                identity_derivative_spec=(self._identity_derivative_specs.get(key)),
                inventory=inventory,
                config=self.config,
            )
            if receipt.get("prompt_binding") != expected_prompt_binding:
                raise batch.AdapterError("vnext_final_prompt_binding_drift")
            rejected.append(
                {
                    "schema_version": INVENTORY_SCHEMA_VERSION,
                    "idempotency_key": key,
                    "outcome": "rejected",
                    "reason_code": receipt["reason_code"],
                    "receipt_sha256": batch._hash_object(receipt),
                    "output_record_sha256": None,
                    "stratum": stratum,
                    "content_retained": False,
                }
            )
        return (
            sorted(accepted, key=lambda row: str(row["idempotency_key"])),
            sorted(rejected, key=lambda row: str(row["idempotency_key"])),
        )

    def _write_and_authenticate_freeze_inventory(
        self,
        path: Path,
        *,
        rows: Sequence[Mapping[str, Any]],
        outcome: str,
        jobs_by_key: Mapping[str, batch.AlignmentJob],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if outcome not in {"accepted", "rejected", "quarantine"}:
            raise batch.AdapterError("vnext_freeze_inventory_outcome_invalid")
        ordered = [dict(row) for row in rows]
        if ordered != sorted(
            ordered,
            key=lambda row: str(row["idempotency_key"]),
        ):
            raise batch.AdapterError("vnext_freeze_inventory_order_drift")
        raw = b"".join(_canonical_bytes(row, newline=True) for row in ordered)
        _exclusive_bytes(path, raw)
        digest = _sha256_bytes(raw)
        _exclusive_bytes(
            path.with_name(f"{path.name}.sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
        authenticated_digest = _authenticated_file_digest(
            path,
            stop=self.freeze_dir,
            reason="vnext_freeze_inventory_sidecar_drift",
        )
        reloaded = _physical_jsonl(
            path,
            id_field="idempotency_key",
            stop=self.freeze_dir,
            reason="vnext_freeze_inventory_snapshot_drift",
        )
        required = {
            "schema_version",
            "idempotency_key",
            "outcome",
            "reason_code",
            "receipt_sha256",
            "output_record_sha256",
            "stratum",
            "content_retained",
        }
        if (
            not hmac.compare_digest(digest, authenticated_digest)
            or reloaded != ordered
            or len({str(row.get("idempotency_key")) for row in reloaded})
            != len(reloaded)
        ):
            raise batch.AdapterError("vnext_freeze_inventory_identity_drift")
        for row in reloaded:
            key = str(row.get("idempotency_key"))
            job = jobs_by_key.get(key)
            receipt_sha256 = row.get("receipt_sha256")
            output_sha256 = row.get("output_record_sha256")
            if (
                set(row) != required
                or row.get("schema_version") != INVENTORY_SCHEMA_VERSION
                or row.get("outcome") != outcome
                or job is None
                or row.get("content_retained") is not False
                or _SAFE_STOP_REASON.fullmatch(str(row.get("reason_code"))) is None
                or row.get("stratum")
                != body_free_stratum(
                    job.source,
                    self._selection_views.get(key),
                )
                or (
                    receipt_sha256 is not None
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(receipt_sha256),
                    )
                    is None
                )
                or (
                    output_sha256 is not None
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(output_sha256),
                    )
                    is None
                )
            ):
                raise batch.AdapterError("vnext_freeze_inventory_row_invalid")
            if outcome == "accepted" and (
                receipt_sha256 is None or output_sha256 is None
            ):
                raise batch.AdapterError("vnext_freeze_inventory_row_invalid")
            if outcome == "quarantine" and (
                receipt_sha256 is not None or output_sha256 is not None
            ):
                raise batch.AdapterError("vnext_freeze_inventory_row_invalid")
        return reloaded, {
            "file": path.name,
            "sha256": authenticated_digest,
            "inventory_root_sha256": _merkle_root(reloaded),
            "count": len(reloaded),
            "outcome": outcome,
            "content_retained": False,
        }

    def _write_checkpoint_manifest(
        self,
        freeze_dir: Path,
    ) -> dict[str, Any]:
        checkpoint_paths = sorted(self.checkpoint_dir.glob("*.json"))
        expected_names = {
            name
            for path in checkpoint_paths
            for name in (path.name, f"{path.name}.sha256")
        }
        observed_names = {path.name for path in self.checkpoint_dir.iterdir()}
        if observed_names != expected_names:
            raise batch.AdapterError("vnext_checkpoint_manifest_file_set_drift")
        schema_raw = _physical_read(
            self.policy.checkpoint_schema_path,
            reason="vnext_checkpoint_schema_snapshot_drift",
            stop=batch.REPO_ROOT.absolute(),
        )
        if not hmac.compare_digest(
            _sha256_bytes(schema_raw),
            self.policy.checkpoint_schema_sha256,
        ):
            raise batch.AdapterError("vnext_checkpoint_schema_identity_drift")
        schema = _strict_json_mapping(
            schema_raw,
            reason="vnext_checkpoint_schema_invalid",
        )
        validator = Draft202012Validator(schema)
        entries: list[dict[str, Any]] = []
        for sequence, path in enumerate(checkpoint_paths):
            raw = _physical_read(
                path,
                reason="vnext_checkpoint_manifest_snapshot_drift",
                stop=self.root,
            )
            value = _strict_json_mapping(
                raw,
                reason="vnext_checkpoint_manifest_entry_invalid",
            )
            payload = self._verify_signed_value(
                value,
                reason="vnext_checkpoint_manifest_hmac_drift",
            )
            try:
                validator.validate(value)
            except ValidationError as error:
                raise batch.AdapterError(
                    "vnext_checkpoint_manifest_schema_mismatch"
                ) from error
            sidecar_digest = _authenticated_file_digest(
                path,
                stop=self.checkpoint_dir,
                reason="vnext_checkpoint_manifest_sidecar_drift",
            )
            if payload.get(
                "checkpoint_sequence"
            ) != sequence or sidecar_digest != _sha256_bytes(raw):
                raise batch.AdapterError("vnext_checkpoint_manifest_sequence_drift")
            entries.append(
                {
                    "checkpoint_sequence": sequence,
                    "checkpoint_sha256": _sha256_bytes(raw),
                    "group_id": payload["group_id"],
                    "delta_inventory_root_sha256": payload[
                        "delta_inventory_root_sha256"
                    ],
                    "new_chain_tip_sha256": payload["new_chain_tip_sha256"],
                }
            )
        payload = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
                "checkpoint-manifest.v1"
            ),
            "selector_identity_sha256": self.selection_manifest[
                "selector_identity_sha256"
            ],
            "selected_set_root_sha256": self.selection_manifest[
                "selected_set_root_sha256"
            ],
            "selected_set_count": 320,
            "checkpoint_count": len(entries),
            "entries": entries,
            "checkpoint_manifest_root_sha256": _merkle_root(entries),
            "content_retained": False,
            "raw_token_ids_retained": False,
        }
        manifest = self._signed_value(payload)
        path = freeze_dir / "checkpoint_manifest.json"
        raw = _canonical_bytes(manifest, newline=True)
        _exclusive_bytes(path, raw)
        digest = _sha256_bytes(raw)
        _exclusive_bytes(
            path.with_name(f"{path.name}.sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
        if (
            _authenticated_file_digest(
                path,
                stop=freeze_dir,
                reason="vnext_checkpoint_manifest_output_drift",
            )
            != digest
            or _strict_json_mapping(
                _physical_read(
                    path,
                    reason="vnext_checkpoint_manifest_output_drift",
                    stop=freeze_dir,
                ),
                reason="vnext_checkpoint_manifest_output_drift",
            )
            != manifest
        ):
            raise batch.AdapterError("vnext_checkpoint_manifest_output_drift")
        return {
            **manifest,
            "checkpoint_manifest_sha256": digest,
        }

    def stage_external_attestation(
        self,
        freeze_attestation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Stage, but never forge, the independent release decision."""

        self._verify_signed_value(
            freeze_attestation,
            reason="vnext_freeze_attestation_hmac_drift",
        )
        freeze_path = self.freeze_dir / "freeze_attestation.json"
        freeze_raw = _physical_read(
            freeze_path,
            reason="vnext_freeze_attestation_snapshot_drift",
            stop=self.root,
        )
        if _strict_json_mapping(
            freeze_raw,
            reason="vnext_freeze_attestation_invalid",
        ) != dict(freeze_attestation):
            raise batch.AdapterError("vnext_freeze_attestation_identity_drift")
        freeze_sha256 = _authenticated_file_digest(
            freeze_path,
            stop=self.freeze_dir,
            reason="vnext_freeze_attestation_sidecar_drift",
        )
        final_binding_payload = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
                "producer-final-binding-candidate.v1"
            ),
            "state": "awaiting_producer_final_R_and_independent_attestation",
            "controller_run_id": self.controller_binding["controller_run_id"],
            "producer_launch_L": dict(self._job_provenance["producer_launch_L"]),
            "consumer_code_P": dict(self._job_provenance["consumer_code_P"]),
            "job_provenance_identity_sha256": self._job_provenance[
                "provenance_identity_sha256"
            ],
            "selector_identity_sha256": self.selection_manifest[
                "selector_identity_sha256"
            ],
            "selected_set_root_sha256": self.selection_manifest[
                "selected_set_root_sha256"
            ],
            "selected_set_count": 320,
            "freeze_attestation_sha256": freeze_sha256,
            "checkpoint_manifest_root_sha256": freeze_attestation[
                "checkpoint_manifest_root_sha256"
            ],
            "inventory_bindings": dict(freeze_attestation["inventory_bindings"]),
            "counts": dict(freeze_attestation["counts"]),
            "coverage_gaps": list(freeze_attestation["coverage_gaps"]),
            "producer_final_R_present": False,
            "external_attestation_present": False,
            "consumer_pin_C_present": False,
            "procedurally_independent_review_present": False,
            "release_self_pin_present": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "content_retained": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        }
        final_binding = self._signed_value(final_binding_payload)
        final_binding_path = self.freeze_dir / "producer_final_binding_candidate.json"
        final_binding_raw = _canonical_bytes(
            final_binding,
            newline=True,
        )
        _exclusive_bytes(final_binding_path, final_binding_raw)
        final_binding_sha256 = _sha256_bytes(final_binding_raw)
        _exclusive_bytes(
            final_binding_path.with_name(f"{final_binding_path.name}.sha256"),
            (f"{final_binding_sha256}  {final_binding_path.name}\n").encode("ascii"),
        )
        if (
            _authenticated_file_digest(
                final_binding_path,
                stop=self.freeze_dir,
                reason="vnext_final_binding_candidate_output_drift",
            )
            != final_binding_sha256
        ):
            raise batch.AdapterError("vnext_final_binding_candidate_output_drift")
        payload = {
            "schema_version": LAUNCH_STAGING_SCHEMA_VERSION,
            "state": "awaiting_independent_external_attestation",
            "controller_run_id": self.controller_binding["controller_run_id"],
            "selector_identity_sha256": self.selection_manifest[
                "selector_identity_sha256"
            ],
            "selected_set_root_sha256": self.selection_manifest[
                "selected_set_root_sha256"
            ],
            "selected_set_count": 320,
            "freeze_attestation_sha256": freeze_sha256,
            "producer_final_binding_candidate_sha256": (final_binding_sha256),
            "checkpoint_manifest_root_sha256": freeze_attestation[
                "checkpoint_manifest_root_sha256"
            ],
            "counts": dict(freeze_attestation["counts"]),
            "consumer_code_P": dict(self._job_provenance["consumer_code_P"]),
            "external_attestation_present": False,
            "consumer_integration_ready": False,
            "release_blockers": [
                "independent_external_attestation_missing",
                "producer_final_R_missing",
                "consumer_pin_C_missing",
            ],
            "training_authorized": False,
            "formal_training_authorized": False,
            "content_retained": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        }
        staging = self._signed_value(payload)
        path = self.freeze_dir / "external_attestation_staging.json"
        raw = _canonical_bytes(staging, newline=True)
        _exclusive_bytes(path, raw)
        digest = _sha256_bytes(raw)
        _exclusive_bytes(
            path.with_name(f"{path.name}.sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
        return {**staging, "staging_sha256": digest}

    def _freeze_inventory_rows(
        self,
        path: Path,
        *,
        outcome: str,
        jobs_by_key: Mapping[str, batch.AlignmentJob],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for receipt in closed._stable_jsonl(path, id_field="id"):
            batch._verify_receipt(receipt, self.hmac_key or b"")
            key = str(receipt.get("idempotency_key"))
            job = jobs_by_key.get(key)
            if job is None:
                raise batch.AdapterError("vnext_freeze_receipt_source_drift")
            stratum = body_free_stratum(
                job.source,
                self._selection_views.get(key),
            )
            if receipt.get("stratum") != stratum:
                raise batch.AdapterError("vnext_freeze_stratum_drift")
            result.append(
                {
                    "schema_version": INVENTORY_SCHEMA_VERSION,
                    "idempotency_key": key,
                    "outcome": outcome,
                    "reason_code": receipt["reason_code"],
                    "receipt_sha256": batch._hash_object(receipt),
                    "output_record_sha256": receipt["output_record_sha256"],
                    "stratum": stratum,
                    "content_retained": False,
                }
            )
        return sorted(result, key=lambda row: str(row["idempotency_key"]))


class VNextBatchRunner(batch.BatchRunner):
    """Batch runner with fast grammar, vNext receipts, and safe stop latching."""

    def __init__(
        self,
        config: batch.AdapterConfig,
        inventory: batch.SourceInventory,
        *,
        teachers: Mapping[str, batch.BatchTeacher],
        runtime_slots: batch.RuntimeSecretSlots,
        controller_binding: Mapping[str, Any],
        policy: VNextPolicy,
        selection_manifest: Mapping[str, Any],
        selection_views: Mapping[str, Mapping[str, Any]],
        identity_derivative_specs: Sequence[Mapping[str, Any]],
        job_provenance: Mapping[str, Any],
        launch_authorization: Mapping[str, Any],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            config,
            inventory,
            teachers=teachers,
            runtime_slots=runtime_slots,
            environ=environ,
        )
        batch._verify_receipt(launch_authorization, self.hmac_key)
        launch_payload = dict(launch_authorization)
        launch_payload.pop("hmac_sha256", None)
        if (
            launch_payload.get("schema_version") != LAUNCH_AUTHORIZATION_SCHEMA_VERSION
            or launch_payload.get("live_authorized") is not True
            or launch_payload.get("vnext_config_sha256") != policy.physical_sha256
            or launch_payload.get("output_root") != str(config.output_root)
            or policy.claims.get("live_authorized") is not True
            or policy.claims.get("training_authorized") is not False
        ):
            raise batch.AdapterError("vnext_launch_authorization_invalid")
        self.launch_authorization = launch_payload
        self._selection_views = {
            str(key): dict(value) for key, value in selection_views.items()
        }
        self._identity_derivative_specs = {
            str(spec["derived_teacher_job_idempotency_key"]): dict(spec)
            for spec in identity_derivative_specs
        }
        if (
            len(identity_derivative_specs) != 100
            or len(self._identity_derivative_specs) != 100
        ):
            raise batch.AdapterError("vnext_identity_derivative_spec_inventory_drift")
        selected_keys = set(selection_views)
        selected_jobs = derive_selected_vnext_jobs(
            self.jobs,
            selection_views,
            identity_derivative_specs=identity_derivative_specs,
            inventory=inventory,
            config=config,
        )
        if (
            len(selected_jobs) != 320
            or {job.idempotency_key for job in selected_jobs} != selected_keys
        ):
            raise batch.AdapterError("vnext_minimal320_job_set_drift")
        self.jobs = selected_jobs
        self.state = VNextAuthenticatedBatchState(
            config,
            hmac_key=self.hmac_key,
            controller_binding=controller_binding,
            policy=policy,
            selection_manifest=selection_manifest,
            selection_views=selection_views,
            identity_derivative_specs=(self._identity_derivative_specs),
            job_provenance=job_provenance,
            jobs=selected_jobs,
            stop_probe=lambda: batch._kill_switch_armed(
                self.config,
                self.environ,
            ),
        )

    def _output_record(
        self,
        job: batch.AlignmentJob,
        output: Mapping[str, Any],
        usage: Mapping[str, int],
        attempts: Mapping[str, Any],
    ) -> dict[str, Any]:
        view = self._selection_views[job.idempotency_key]
        output_role = str(view["output_expert_role"])
        asset_job = (
            replace(
                job,
                source=replace(job.source, role=output_role),
            )
            if output_role != job.source.role
            else job
        )
        record = super()._output_record(
            asset_job,
            output,
            usage,
            attempts,
        )
        if view.get("candidate_id_sha256") != selector.candidate_identity_sha256(view):
            raise batch.AdapterError("vnext_selected_candidate_identity_drift")
        source_binding = dict(record["source_binding"])
        source_binding.update(
            {
                "selected_candidate_sha256": view["candidate_id_sha256"],
                "source_record_id_sha256": view["source_record_id_sha256"],
                "source_content_sha256": view["source_content_sha256"],
                "source_serialization_identity_sha256": view[
                    "source_serialization_identity_sha256"
                ],
                "task_bundle_sha256": view["task_bundle_sha256"],
                "training_asset": view["training_asset"],
            }
        )
        record["source_binding"] = source_binding
        spec = self._identity_derivative_specs.get(job.idempotency_key)
        if spec is not None:
            teacher_binding = dict(record["teacher_binding"])
            teacher_binding["prompt_version"] = spec["v5_prompt_version"]
            teacher_binding["prompt_template_id"] = job.source.teacher_contract[
                "prompt_template_id"
            ]
            teacher_binding["prompt_template_sha256"] = job.prompt_template_sha256
            teacher_binding["output_schema_id"] = job.source.teacher_contract[
                "output_schema_id"
            ]
            teacher_binding["output_schema_sha256"] = job.output_schema_sha256
            record["teacher_binding"] = teacher_binding
        return record

    async def run(self, phase: str = "bulk") -> batch.RunReport:
        if (
            self.launch_authorization.get("live_authorized") is not True
            or self.launch_authorization.get("output_root")
            != str(self.config.output_root)
            or not Path(
                str(self.launch_authorization["controller_lease_path"])
            ).is_file()
        ):
            raise batch.AdapterError("vnext_launch_authorization_not_live")
        if phase != "bulk" or phase not in self.config.limits.allowed_phases:
            raise batch.AdapterError("vnext_only_bulk_phase_allowed")
        if not self.runtime_slots.loaded:
            raise batch.AdapterError("controller_credential_slot_unloaded")
        batch.validate_heldout_receipt(self.config)
        if batch._kill_switch_armed(self.config, self.environ):
            raise batch.AdapterError("kill_switch_armed")
        self.state.initialize(self.inventory)
        phase_jobs = tuple(self.jobs)
        replay = self.state.replay()
        target_keys = {job.idempotency_key for job in phase_jobs}
        if replay.uncertain & target_keys:
            raise batch.AdapterError("uncertain_dispatch_not_redispatched")
        pending_jobs = [
            job
            for job in phase_jobs
            if job.idempotency_key not in replay.completed
            and job.idempotency_key not in replay.rejected
        ]
        started = time.monotonic()
        self.state.write_status(
            inventory=self.inventory,
            jobs=self.jobs,
            phase_jobs=phase_jobs,
            phase=phase,
            state_name="running",
            started_monotonic=started,
        )
        try:
            for offset in range(
                0,
                len(pending_jobs),
                self.config.group_commit_size,
            ):
                if batch._kill_switch_armed(self.config, self.environ):
                    self.state.stop_at_durable_group_boundary("kill_switch_armed")
                group_jobs = pending_jobs[
                    offset : offset + self.config.group_commit_size
                ]
                replay = self.state.replay()
                outcomes = await asyncio.gather(
                    *(
                        self._execute_job(
                            job,
                            phase=phase,
                            replay=replay,
                            retry_limit=self.config.limits.max_retries,
                        )
                        for job in group_jobs
                    ),
                    return_exceptions=True,
                )
                failures = [
                    item for item in outcomes if isinstance(item, BaseException)
                ]
                if failures:
                    for failure_type in (
                        batch.ProviderQuotaExhausted,
                        batch.RateLimitError,
                        batch.ClientDeadlineExceeded,
                        batch.BudgetExceeded,
                        batch.TeacherError,
                        batch.AdapterError,
                    ):
                        selected = next(
                            (
                                item
                                for item in failures
                                if isinstance(item, failure_type)
                            ),
                            None,
                        )
                        if selected is not None:
                            raise selected
                    raise batch.AdapterError("unexpected_worker_failure")
                terminal = [
                    item for item in outcomes if isinstance(item, batch.PendingCommit)
                ]
                if len(terminal) != len(group_jobs):
                    raise batch.AdapterError(
                        "vnext_group_expected_inventory_incomplete"
                    )
                self.state.commit_group(
                    terminal,
                    inventory=self.inventory,
                    phase=phase,
                    expected_job_keys=[job.idempotency_key for job in group_jobs],
                )
                self.state.write_status(
                    inventory=self.inventory,
                    jobs=self.jobs,
                    phase_jobs=phase_jobs,
                    phase=phase,
                    state_name="running",
                    started_monotonic=started,
                )
        except StopAfterGroup:
            self.state.write_status(
                inventory=self.inventory,
                jobs=self.jobs,
                phase_jobs=phase_jobs,
                phase=phase,
                state_name="stopped_after_group",
                started_monotonic=started,
                blockers=["vnext_stop_after_group"],
            )
            raise
        except (
            batch.AdapterError,
            batch.BudgetExceeded,
            batch.ClientDeadlineExceeded,
            batch.TeacherError,
        ):
            self.state.write_status(
                inventory=self.inventory,
                jobs=self.jobs,
                phase_jobs=phase_jobs,
                phase=phase,
                state_name="blocked",
                started_monotonic=started,
            )
            raise
        status = self.state.write_status(
            inventory=self.inventory,
            jobs=self.jobs,
            phase_jobs=phase_jobs,
            phase=phase,
            state_name="generation_complete_pending_freeze",
            started_monotonic=started,
        )
        return batch.RunReport(
            phase=phase,
            total=int(status["total"]),
            queued=int(status["queued"]),
            succeeded=int(status["succeeded"]),
            rejected=int(status["rejected"]),
            retried=int(status["retried"]),
            requests=int(status["provider_usage"]["requests"]),
            input_tokens=int(status["provider_usage"]["input_tokens"]),
            output_tokens=int(status["provider_usage"]["output_tokens"]),
            state=str(status["state"]),
        )

    async def _execute_job(
        self,
        job: batch.AlignmentJob,
        *,
        phase: str,
        replay: batch.ReplayState,
        retry_limit: int,
    ) -> batch.PendingCommit | None:
        await self.state.reserve_budget(job, replay, phase)
        await self.state.append_event(
            job=job,
            phase=phase,
            state="dispatching",
            reason_code="provider_dispatch",
        )
        teacher = self.teachers[job.source.role]
        batch._set_teacher_retry_limit(teacher, retry_limit)
        selection_view = self._selection_views[job.idempotency_key]
        derivative_spec = self._identity_derivative_specs.get(job.idempotency_key)
        system_prompt, user_prompt = authenticated_job_prompts(
            job,
            selection_view,
            identity_derivative_spec=derivative_spec,
            inventory=self.inventory,
            config=self.config,
        )
        try:
            text = await teacher.complete(
                system=system_prompt,
                user=user_prompt,
                idempotency_key=job.idempotency_key,
            )
        except batch.ProviderQuotaExhausted:
            self.state.observe_provider_terminal_usage(None)
            await self.state.append_event(
                job=job,
                phase=phase,
                state="retryable",
                reason_code="provider_quota_exhausted",
                attempts=batch._safe_attempts_after_failure(teacher),
            )
            raise
        except batch.RateLimitError:
            self.state.observe_provider_terminal_usage(None)
            await self.state.append_event(
                job=job,
                phase=phase,
                state="retryable",
                reason_code="provider_rate_limit",
                attempts=batch._safe_attempts_after_failure(teacher),
            )
            raise
        except (
            batch.ClientDeadlineExceeded,
            batch.BudgetExceeded,
            batch.TeacherError,
        ) as error:
            self.state.observe_provider_terminal_usage(None)
            await self.state.append_event(
                job=job,
                phase=phase,
                state="uncertain",
                reason_code=batch._body_free_provider_failure_reason(
                    error,
                    teacher,
                ),
                attempts=batch._safe_attempts_after_failure(teacher),
            )
            raise
        self.state._observe_stop_probe()
        provenance = teacher.provider_provenance
        try:
            usage, attempts = batch._usage_from_provenance(provenance)
        except batch.AdapterError as error:
            self.state.observe_provider_terminal_usage(None)
            await self.state.append_event(
                job=job,
                phase=phase,
                state="uncertain",
                reason_code=error.reason_code,
                attempts=batch._safe_attempts_after_failure(teacher),
            )
            raise
        self.state.observe_provider_terminal_usage(usage)
        try:
            output = fast_parse_candidate_output(job.source, text)
        except batch.AdapterError as error:
            receipt = self._job_receipt(
                job,
                usage=usage,
                attempts=attempts,
                output_record_sha256=None,
                overlay_identity_sha256=None,
                outcome="rejected",
                reason_code=error.reason_code,
            )
            return batch.PendingCommit(
                job=job,
                output_record=None,
                receipt=receipt,
                terminal_state="rejected",
                event_reason=error.reason_code,
            )
        record = self._output_record(job, output, usage, attempts)
        receipt = self._job_receipt(
            job,
            usage=usage,
            attempts=attempts,
            output_record_sha256=batch._hash_object(record),
            overlay_identity_sha256=record["derived_planner_overlay"][
                "overlay_identity_sha256"
            ],
            outcome="succeeded",
            reason_code="online_candidate_committed",
        )
        return batch.PendingCommit(
            job=job,
            output_record=record,
            receipt=receipt,
            terminal_state="committed",
            event_reason="online_candidate_committed",
        )

    def _job_receipt(
        self,
        job: batch.AlignmentJob,
        *,
        usage: Mapping[str, int],
        attempts: Mapping[str, Any],
        output_record_sha256: str | None,
        overlay_identity_sha256: str | None,
        outcome: str,
        reason_code: str,
    ) -> dict[str, Any]:
        stratum = body_free_stratum(
            job.source,
            self.state._selection_views.get(job.idempotency_key),
        )
        selection_view = self._selection_views[job.idempotency_key]
        derivative_spec = self._identity_derivative_specs.get(job.idempotency_key)
        payload = {
            "id": batch._hash_object(
                {
                    "idempotency_key": job.idempotency_key,
                    "outcome": outcome,
                    "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
                }
            ),
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "receipt_kind": "job",
            "idempotency_key": job.idempotency_key,
            "role": stratum["role"],
            "validation_contract_role": stratum["validation_contract_role"],
            "language": stratum["language"],
            "stratum": stratum,
            "outcome": outcome,
            "reason_code": reason_code,
            "output_record_sha256": output_record_sha256,
            "planner_lineage": batch._planner_lineage_receipt(
                job,
                overlay_identity_sha256=overlay_identity_sha256,
            ),
            "prompt_binding": prompt_binding(
                job,
                selection_view,
                identity_derivative_spec=derivative_spec,
                inventory=self.inventory,
                config=self.config,
            ),
            "usage": dict(usage),
            "attempts": dict(attempts),
            "provenance": dict(self.state._job_provenance),
            "committed_at": batch._iso(),
            "content_retained": False,
            "planner_body_retained": False,
            "raw_token_ids_retained": False,
        }
        receipt = {
            **payload,
            "hmac_sha256": batch._receipt_hmac(payload, self.hmac_key),
        }
        schema_raw = _physical_read(
            self.state.policy.receipt_schema_path,
            reason="vnext_receipt_schema_snapshot_drift",
            stop=batch.REPO_ROOT.absolute(),
        )
        schema = _strict_json_mapping(
            schema_raw,
            reason="vnext_receipt_schema_invalid",
        )
        try:
            Draft202012Validator(schema).validate(receipt)
        except ValidationError as error:
            raise batch.AdapterError("vnext_receipt_schema_mismatch") from error
        return receipt


def build_vnext_teachers(
    config: batch.AdapterConfig,
    runtime_slots: batch.RuntimeSecretSlots,
) -> dict[str, AbsoluteDeadlineCompatibleTeacher]:
    workers: dict[str, AbsoluteDeadlineCompatibleTeacher] = {}
    owner: AbsoluteDeadlineCompatibleTeacher | None = None
    for role in batch.ROLES:
        teacher = AbsoluteDeadlineCompatibleTeacher(
            base_url=batch.BASE_URL,
            model=batch.MODEL,
            protocol=batch.PROTOCOL,
            fallback_protocol=None,
            fallback_base_url=batch.BASE_URL,
            api_key_env="",
            credential_provider=runtime_slots.read_provider_credential,
            user_agent=str(config.provider["user_agent"]),
            timeout_seconds=min(
                config.limits.timeout_seconds,
                config.limits.wall_clock_deadline_seconds,
            ),
            wall_clock_deadline_seconds=(config.limits.wall_clock_deadline_seconds),
            max_retries=config.limits.max_retries,
            temperature=0.2,
            max_tokens=None,
            thinking_enabled=False,
            thinking_effort="low",
            thinking_budget_tokens=0,
            responses_thinking_policy="explicit_disabled",
            chat_response_format=(
                batch.CHAT_JSON_RESPONSE_FORMAT
                if role in batch.CHAT_JSON_MODE_ROLES
                else None
            ),
            stream_openai=False,
            stream_options_include_usage=False,
            max_requests=config.limits.max_requests,
            max_output_tokens_total=config.limits.max_output_tokens_total,
            provider_preset=batch.PROVIDER_PRESET,
            model_source="forced_config",
            discovery_status="skipped_force_model",
            discovery_model_count=0,
        )
        if owner is None:
            owner = teacher
        else:
            teacher.share_usage_budget(owner)
        workers[role] = teacher
    return workers


def _nonnegative_finite_number(value: Any, *, positive: bool = False) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return False
    return float(value) > 0 if positive else float(value) >= 0


def public_provider_throughput_snapshot(path: Path) -> dict[str, Any]:
    """Read one body-free vNext telemetry snapshot without polling or writes."""

    candidate = path.absolute()
    raw = _physical_read(
        candidate,
        reason="vnext_throughput_telemetry_unreadable",
        stop=candidate.parent.absolute(),
        maximum_bytes=64 * 1024,
    )
    value = _strict_json_mapping(
        raw,
        reason="vnext_throughput_telemetry_invalid",
    )
    input_rate = value.get("provider_input_tokens_per_second")
    output_rate = value.get("provider_output_tokens_per_second")
    exact = value.get("exact")
    terminal_jobs = value.get("terminal_jobs")
    exact_rows = value.get("exact_rows")
    unknown_rows = value.get("unknown_rows")
    error = value.get("error")
    numeric_rates = _nonnegative_finite_number(
        input_rate
    ) and _nonnegative_finite_number(output_rate)
    unknown_rates = input_rate == "UNKNOWN" and output_rate == "UNKNOWN"
    if (
        set(value) != _THROUGHPUT_DISK_KEYS
        or value.get("schema_version") != THROUGHPUT_TELEMETRY_SCHEMA_VERSION
        or value.get("semantics") != _THROUGHPUT_SEMANTICS
        or not _nonnegative_finite_number(value.get("window_seconds"))
        or not _nonnegative_finite_number(
            value.get("window_limit_seconds"),
            positive=True,
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (terminal_jobs, exact_rows, unknown_rows)
        )
        or exact_rows + unknown_rows != terminal_jobs
        or not isinstance(value.get("observed_at"), str)
        or not value["observed_at"]
        or not isinstance(exact, bool)
        or (
            error is not None
            and (
                not isinstance(error, str) or _SAFE_STOP_REASON.fullmatch(error) is None
            )
        )
        or re.fullmatch(r"[0-9a-f]{32}", str(value.get("controller_run_id"))) is None
        or value.get("content_retained") is not False
        or value.get("raw_token_ids_retained") is not False
        or value.get("credential_retained") is not False
        or (
            exact
            and (
                not numeric_rates
                or unknown_rates
                or error is not None
                or terminal_jobs < 1
                or exact_rows != terminal_jobs
                or unknown_rows != 0
                or not _nonnegative_finite_number(
                    value.get("window_seconds"),
                    positive=True,
                )
            )
        )
        or (
            not exact
            and (not unknown_rates or numeric_rates or not isinstance(error, str))
        )
    ):
        raise batch.AdapterError("vnext_throughput_telemetry_invalid")
    return {
        "schema_version": THROUGHPUT_MONITOR_SCHEMA_VERSION,
        "mode": "read_only_single_snapshot",
        "live_reload": False,
        "telemetry_sha256": _sha256_bytes(raw),
        "provider_input_tokens_per_second": input_rate,
        "provider_output_tokens_per_second": output_rate,
        "window_seconds": value["window_seconds"],
        "window_limit_seconds": value["window_limit_seconds"],
        "terminal_jobs": terminal_jobs,
        "observed_at": value["observed_at"],
        "exact": exact,
        "exact_rows": exact_rows,
        "unknown_rows": unknown_rows,
        "error": error,
        "semantics": _THROUGHPUT_SEMANTICS,
        "write_operations": 0,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }


def public_dry_run(policy: VNextPolicy) -> dict[str, Any]:
    consumer = policy.consumer_code_P
    blocker = consumer.get("blocker_reason_code")
    consumer_ready = (
        consumer.get("stage") == "consumer_code_P"
        and consumer.get("status") == "audited_compatible"
        and blocker is None
        and consumer.get("independently_audited") is True
    )
    if not consumer_ready and (
        not isinstance(blocker, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,95}", blocker) is None
    ):
        blocker = "consumer_code_P_binding_invalid"
    blockers = ([] if consumer_ready else [str(blocker)]) + [
        "vnext_transition_evidence_required"
    ]
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "state": "candidate",
        "mode": "dry_run",
        "config_sha256": policy.physical_sha256,
        "contract_sha256": policy.contract_sha256,
        "schema_sha256": policy.schema_sha256,
        "online_authentication": "o1_held_sequence_chain_tip",
        "startup_full_replay": True,
        "final_freeze_full_replay": True,
        "delta_checkpoint_hmac": True,
        "delta_checkpoint_sidecar": True,
        "absolute_nonstreaming_chat_deadline": True,
        "stop_after_group": True,
        "body_free_strata": True,
        "closed_grammar_fast_gate": True,
        "systematic_rejection_fuse": True,
        "provider_requests": 0,
        "gpu_requests": 0,
        "transition_gate_satisfied": False,
        "blockers": blockers,
        "cross_process_resume": False,
        "consumer_code_P_ready": consumer_ready,
        "consumer_integration_ready": False,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
        **policy.claims,
    }


def _pid_alive(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        synchronize = 0x00100000
        query_limited_information = 0x00001000
        handle = kernel32.OpenProcess(
            synchronize | query_limited_information,
            False,
            pid,
        )
        if not handle:
            # ERROR_INVALID_PARAMETER is the documented nonexistent-PID case.
            # Access denied and all other failures are fail-closed as alive.
            return ctypes.get_last_error() != 87
        try:
            wait_result = kernel32.WaitForSingleObject(handle, 0)
            if wait_result == 0x00000102:  # WAIT_TIMEOUT
                return True
            if wait_result == 0x00000000:  # WAIT_OBJECT_0
                return False
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _controller_kind(arguments: Sequence[str]) -> str | None:
    normalized = tuple(
        str(argument).strip().strip('"').replace("\\", "/").lower()
        for argument in arguments
    )
    if any(
        marker in argument
        for argument in normalized
        for marker in _LEGACY_CONTROLLER_MARKERS
    ):
        return "legacy"
    if any(
        marker in argument
        for argument in normalized
        for marker in _VNEXT_CONTROLLER_MARKERS
    ):
        return "vnext"
    return None


def _body_free_process_row(
    *,
    pid: int,
    parent_pid: int,
    creation_identity: int,
    executable: str,
    arguments: Sequence[str],
    kind: str,
) -> dict[str, Any]:
    launch_arguments = [str(argument).replace("\\", "/") for argument in arguments[1:]]
    while launch_arguments and re.fullmatch(
        r"-(?:[23](?:\.\d+)?|v:[^\s]+)",
        launch_arguments[0],
        flags=re.IGNORECASE,
    ):
        launch_arguments.pop(0)
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "creation_identity": creation_identity,
        "executable_path_sha256": _sha256_text(executable.replace("\\", "/").lower()),
        "argv_sha256": _sha256_bytes(
            b"\x00".join(str(argument).encode("utf-8") for argument in arguments)
        ),
        "argument_count": len(arguments),
        "controller_kind": kind,
        "launch_command_identity_sha256": _sha256_bytes(
            b"\x00".join(argument.encode("utf-8") for argument in launch_arguments)
        ),
        "content_retained": False,
    }


def _windows_controller_process_rows() -> list[dict[str, Any]]:
    """Use kernel process handles; no shell/CIM inventory is trusted."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x00001000
    synchronize = 0x00100000
    wait_timeout = 0x00000102
    process_command_line_information = 60

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    toolhelp_snapshot_process = 0x00000002

    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CreateToolhelp32Snapshot.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQueryInformationProcess.restype = wintypes.LONG
    shell32.CommandLineToArgvW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(
        toolhelp_snapshot_process,
        0,
    )
    if snapshot == wintypes.HANDLE(-1).value:
        raise batch.AdapterError("vnext_controller_inventory_enumeration_failed")
    process_names: dict[int, tuple[str, int]] = {}
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise batch.AdapterError("vnext_controller_inventory_enumeration_failed")
        while True:
            process_names[int(entry.th32ProcessID)] = (
                str(entry.szExeFile),
                int(entry.th32ParentProcessID),
            )
            ctypes.set_last_error(0)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                if ctypes.get_last_error() != 18:
                    raise batch.AdapterError(
                        "vnext_controller_inventory_enumeration_failed"
                    )
                break
    finally:
        kernel32.CloseHandle(snapshot)

    rows: list[dict[str, Any]] = []
    for pid, (enumerated_name, parent_pid) in sorted(process_names.items()):
        if pid < 1:
            continue
        enumerated_name = enumerated_name.lower()
        python_candidate = (
            enumerated_name.startswith("python")
            or enumerated_name.startswith("pypy")
            or enumerated_name == "py.exe"
        )
        if not python_candidate:
            continue
        handle = kernel32.OpenProcess(
            process_query_limited_information | synchronize,
            False,
            pid,
        )
        if not handle:
            raise batch.AdapterError(
                "vnext_controller_inventory_python_handle_unreadable"
            )
        try:
            initial_wait = kernel32.WaitForSingleObject(handle, 0)
            if initial_wait == 0x00000000:
                continue
            if initial_wait != wait_timeout:
                raise batch.AdapterError("vnext_controller_inventory_liveness_unknown")
            image_capacity = wintypes.DWORD(32768)
            image_buffer = ctypes.create_unicode_buffer(image_capacity.value)
            if not kernel32.QueryFullProcessImageNameW(
                handle,
                0,
                image_buffer,
                ctypes.byref(image_capacity),
            ):
                raise batch.AdapterError("vnext_controller_inventory_image_unreadable")
            executable = image_buffer.value
            executable_name = Path(executable).name.lower()
            if executable_name != enumerated_name:
                raise batch.AdapterError(
                    "vnext_controller_inventory_executable_name_drift"
                )

            command_capacity = 65536
            command_buffer = ctypes.create_string_buffer(command_capacity)
            needed = wintypes.ULONG()
            status = ntdll.NtQueryInformationProcess(
                handle,
                process_command_line_information,
                command_buffer,
                command_capacity,
                ctypes.byref(needed),
            )
            if status != 0 and needed.value > command_capacity:
                command_capacity = int(needed.value) + 2
                command_buffer = ctypes.create_string_buffer(command_capacity)
                status = ntdll.NtQueryInformationProcess(
                    handle,
                    process_command_line_information,
                    command_buffer,
                    command_capacity,
                    ctypes.byref(needed),
                )
            if status != 0:
                # An unreadable same-user Python command could be a controller.
                raise batch.AdapterError(
                    "vnext_controller_inventory_commandline_unreadable"
                )
            command = ctypes.cast(
                command_buffer,
                ctypes.POINTER(_UnicodeString),
            ).contents
            command_line = ctypes.wstring_at(
                command.Buffer,
                command.Length // ctypes.sizeof(ctypes.c_wchar),
            )
            argument_count = ctypes.c_int()
            argv_pointer = shell32.CommandLineToArgvW(
                command_line,
                ctypes.byref(argument_count),
            )
            if not argv_pointer:
                raise batch.AdapterError(
                    "vnext_controller_inventory_commandline_invalid"
                )
            try:
                arguments = tuple(
                    argv_pointer[index] for index in range(argument_count.value)
                )
            finally:
                kernel32.LocalFree(argv_pointer)
            kind = _controller_kind(arguments)
            if kind is None:
                continue

            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                raise batch.AdapterError(
                    "vnext_controller_inventory_handle_identity_failed"
                )
            creation_identity = (int(creation.dwHighDateTime) << 32) | int(
                creation.dwLowDateTime
            )
            final_wait = kernel32.WaitForSingleObject(handle, 0)
            if final_wait == 0x00000000:
                continue
            if final_wait != wait_timeout:
                raise batch.AdapterError("vnext_controller_inventory_liveness_unknown")
            rows.append(
                _body_free_process_row(
                    pid=pid,
                    parent_pid=parent_pid,
                    creation_identity=creation_identity,
                    executable=executable,
                    arguments=arguments,
                    kind=kind,
                )
            )
        finally:
            kernel32.CloseHandle(handle)
    return rows


def _posix_controller_process_rows() -> list[dict[str, Any]]:
    proc = Path("/proc")
    if not proc.is_dir():
        arguments = tuple([sys.executable, *sys.argv])
        kind = _controller_kind(arguments)
        if kind is None:
            return []
        return [
            _body_free_process_row(
                pid=os.getpid(),
                parent_pid=os.getppid(),
                creation_identity=0,
                executable=sys.executable,
                arguments=arguments,
                kind=kind,
            )
        ]
    rows: list[dict[str, Any]] = []
    for child in proc.iterdir():
        if not child.name.isdecimal():
            continue
        pid = int(child.name)
        try:
            command_stat_before = (child / "cmdline").lstat()
            descriptor = os.open(
                child / "cmdline",
                os.O_RDONLY | int(getattr(os, "O_NOFOLLOW", 0)),
            )
            try:
                opened = os.fstat(descriptor)
                raw = os.read(descriptor, 1024 * 1024)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (command_stat_before.st_dev, command_stat_before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ) or (opened.st_dev, opened.st_ino, opened.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
            ):
                raise batch.AdapterError("vnext_controller_inventory_snapshot_drift")
            arguments = tuple(
                part.decode("utf-8", errors="strict")
                for part in raw.rstrip(b"\x00").split(b"\x00")
                if part
            )
            kind = _controller_kind(arguments)
            if kind is None:
                continue
            executable = os.readlink(child / "exe")
            stat_fields = (child / "stat").read_text(encoding="ascii").split()
            creation_identity = int(stat_fields[21])
            parent_pid = int(stat_fields[3])
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            UnicodeDecodeError,
            ValueError,
            OSError,
        ):
            continue
        rows.append(
            _body_free_process_row(
                pid=pid,
                parent_pid=parent_pid,
                creation_identity=creation_identity,
                executable=executable,
                arguments=arguments,
                kind=kind,
            )
        )
    return rows


def _trusted_controller_inventory() -> dict[str, Any]:
    rows = (
        _windows_controller_process_rows()
        if os.name == "nt"
        else _posix_controller_process_rows()
    )
    rows = sorted(
        rows,
        key=lambda row: (
            int(row["pid"]),
            int(row["creation_identity"]),
            str(row["controller_kind"]),
        ),
    )
    current = [
        row
        for row in rows
        if row["pid"] == os.getpid() and row["controller_kind"] == "vnext"
    ]
    if len(current) != 1:
        raise batch.AdapterError("vnext_controller_inventory_candidate_count_drift")
    rows_by_pid = {int(row["pid"]): row for row in rows}
    candidate_tree_pids = {os.getpid()}
    parent_pid = int(current[0]["parent_pid"])
    while parent_pid in rows_by_pid:
        parent = rows_by_pid[parent_pid]
        if (
            parent["controller_kind"] != "vnext"
            or parent["launch_command_identity_sha256"]
            != current[0]["launch_command_identity_sha256"]
            or int(parent["creation_identity"]) > int(current[0]["creation_identity"])
        ):
            break
        candidate_tree_pids.add(parent_pid)
        parent_pid = int(parent["parent_pid"])
    conflicts = [row for row in rows if int(row["pid"]) not in candidate_tree_pids]
    conflict_payload = {
        "schema_version": _CONTROLLER_INVENTORY_SCHEMA_VERSION,
        "scope": _CONTROLLER_INVENTORY_SCOPE,
        "controllers": conflicts,
        "content_retained": False,
    }
    all_payload = {
        "schema_version": _CONTROLLER_INVENTORY_SCHEMA_VERSION,
        "scope": "matching_controller_entrypoints_including_candidate_pid",
        "controllers": rows,
        "content_retained": False,
    }
    return {
        "controller_count": len(conflicts),
        "candidate_controller_count": len(current),
        "candidate_launch_process_count": len(candidate_tree_pids),
        "candidate_controller_pid": os.getpid(),
        "controller_inventory_sha256": _sha256_bytes(
            _canonical_bytes(conflict_payload)
        ),
        "all_matching_inventory_sha256": _sha256_bytes(_canonical_bytes(all_payload)),
        "content_retained": False,
    }


class _VNextControllerLease:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._raw: bytes | None = None
        self._descriptor: int | None = None
        self._identity: tuple[int, int] | None = None

    @staticmethod
    def _unlock_close(descriptor: int, *, locked: bool) -> None:
        try:
            if locked:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def acquire(self) -> str:
        payload = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2."
                "teacher-alignment-vnext.controller-lease.v1"
            ),
            "controller_run_id": self.run_id,
            "pid": os.getpid(),
            "content_retained": False,
        }
        raw = _canonical_bytes(payload, newline=True)
        descriptor: int | None = None
        locked = False
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_RDWR | int(getattr(os, "O_BINARY", 0)),
                0o600,
            )
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size < 1:
                    os.write(descriptor, b"\x00")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            locked = True
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            batch._write_all(descriptor, raw)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            batch._fsync_parent(self.path.parent)
        except (OSError, ImportError, batch.AdapterError) as error:
            if descriptor is not None:
                self._unlock_close(descriptor, locked=locked)
            raise batch.AdapterError("vnext_controller_lease_already_held") from error
        assert descriptor is not None
        self._descriptor = descriptor
        self._identity = (opened.st_dev, opened.st_ino)
        self._raw = raw
        return _sha256_bytes(raw)

    def authenticate(self) -> None:
        if self._raw is None:
            raise batch.AdapterError("vnext_controller_lease_not_held")
        if self._descriptor is None or self._identity is None:
            raise batch.AdapterError("vnext_controller_lease_not_held")
        closed._reject_reparse_chain(self.path, stop=self.path.parent)
        held = os.fstat(self._descriptor)
        if self._identity != (held.st_dev, held.st_ino):
            raise batch.AdapterError("vnext_controller_lease_identity_drift")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = held.st_size
        while remaining:
            chunk = os.read(self._descriptor, min(remaining, 65536))
            if not chunk:
                raise batch.AdapterError("vnext_controller_lease_snapshot_drift")
            chunks.append(chunk)
            remaining -= len(chunk)
        observed = b"".join(chunks)
        held_after = os.fstat(self._descriptor)
        named = self.path.lstat()
        if not hmac.compare_digest(observed, self._raw) or (
            held.st_dev,
            held.st_ino,
            held.st_size,
            held.st_mtime_ns,
        ) != (
            held_after.st_dev,
            held_after.st_ino,
            held_after.st_size,
            held_after.st_mtime_ns,
        ):
            raise batch.AdapterError("vnext_controller_lease_identity_drift")
        if self._identity != (named.st_dev, named.st_ino):
            raise batch.AdapterError("vnext_controller_lease_identity_drift")
        closed._reject_reparse_chain(self.path, stop=self.path.parent)

    def release(self) -> None:
        if self._raw is None:
            return
        error: BaseException | None = None
        try:
            self.authenticate()
        except BaseException as caught:
            error = caught
        finally:
            if self._descriptor is not None:
                try:
                    self._unlock_close(self._descriptor, locked=True)
                except BaseException as caught:
                    if error is None:
                        error = caught
        self._descriptor = None
        self._identity = None
        self._raw = None
        if error is not None:
            raise batch.AdapterError("vnext_controller_lease_release_failed") from error


def _transition_value(
    path: Path,
    *,
    expected_sha256: str | None,
    schema_path: Path,
    schema_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    evidence_path = path.absolute()
    if not evidence_path.is_file():
        raise batch.AdapterError("vnext_transition_evidence_missing")
    raw = _physical_read(
        evidence_path,
        reason="vnext_transition_evidence_snapshot_drift",
        stop=evidence_path.parent,
    )
    if expected_sha256 is not None and not hmac.compare_digest(
        _sha256_bytes(raw), expected_sha256
    ):
        raise batch.AdapterError("vnext_transition_evidence_identity_drift")
    value = _strict_json_mapping(
        raw,
        reason="vnext_transition_evidence_invalid",
    )
    schema_raw = _physical_read(
        schema_path,
        reason="vnext_transition_schema_snapshot_drift",
        stop=batch.REPO_ROOT.absolute(),
    )
    if schema_sha256 is not None and not hmac.compare_digest(
        _sha256_bytes(schema_raw),
        schema_sha256,
    ):
        raise batch.AdapterError("vnext_transition_schema_identity_drift")
    schema = _strict_json_mapping(
        schema_raw,
        reason="vnext_transition_schema_invalid",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("vnext_transition_gate_blocked") from error
    return value, raw


def _revalidate_policy_physical_identity(
    policy: VNextPolicy,
) -> str:
    pins = (
        (
            policy.path,
            policy.physical_sha256,
            "vnext_config_snapshot_drift",
        ),
        (
            policy.schema_path,
            policy.schema_sha256,
            "vnext_config_schema_identity_drift",
        ),
        (
            policy.implementation_path,
            policy.implementation_sha256,
            "vnext_implementation_identity_drift",
        ),
        (
            policy.transition_schema_path,
            policy.transition_schema_sha256,
            "vnext_transition_schema_identity_drift",
        ),
    )
    observed: list[dict[str, Any]] = []
    raws: dict[Path, bytes] = {}
    for path, expected, reason in pins:
        raw = _physical_read(
            path,
            reason=reason,
            stop=batch.REPO_ROOT.absolute(),
        )
        digest = _sha256_bytes(raw)
        if not hmac.compare_digest(digest, expected):
            raise batch.AdapterError(reason)
        raws[path] = raw
        observed.append(
            {
                "path_sha256": _sha256_text(
                    path.relative_to(batch.REPO_ROOT.absolute()).as_posix()
                ),
                "sha256": digest,
            }
        )
    config = _strict_json_mapping(
        raws[policy.path],
        reason="vnext_config_invalid",
    )
    schema = _strict_json_mapping(
        raws[policy.schema_path],
        reason="vnext_config_schema_invalid",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(config)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("vnext_config_schema_mismatch") from error
    expected_paths = {
        "schema_path": policy.schema_path,
        "implementation_path": policy.implementation_path,
        "transition_schema_path": policy.transition_schema_path,
    }
    for key, expected_path in expected_paths.items():
        if (
            _repo_file(
                config.get(key),
                reason="vnext_config_pinned_path_drift",
            )
            != expected_path
        ):
            raise batch.AdapterError("vnext_config_pinned_path_drift")
    expected_hashes = {
        "schema_sha256": policy.schema_sha256,
        "implementation_sha256": policy.implementation_sha256,
        "transition_schema_sha256": policy.transition_schema_sha256,
    }
    if any(
        not hmac.compare_digest(str(config.get(key)), expected)
        for key, expected in expected_hashes.items()
    ):
        raise batch.AdapterError("vnext_config_pinned_identity_drift")
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": (
                    "anchor.gemma3-chat-unbalanced-v2."
                    "teacher-alignment-vnext.policy-revalidation.v1"
                ),
                "pins": observed,
                "content_retained": False,
            }
        )
    )


def _archived_root_identity(
    root: Path,
    *,
    expected_pid: int,
) -> tuple[str, int]:
    archive = root.absolute()
    if not archive.is_dir():
        raise batch.AdapterError("vnext_old_archive_root_missing")
    closed._reject_reparse_chain(
        archive,
        stop=batch.REPO_ROOT.absolute(),
    )
    rows: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(
        archive,
        followlinks=False,
    ):
        current_path = Path(current)
        closed._reject_reparse_chain(current_path, stop=archive)
        for name in directories:
            child = current_path / name
            child_stat = child.lstat()
            if child.is_symlink() or bool(
                int(getattr(child_stat, "st_file_attributes", 0)) & 0x400
            ):
                raise batch.AdapterError("vnext_old_archive_reparse_forbidden")
        for name in filenames:
            path = current_path / name
            raw = _physical_read(
                path,
                reason="vnext_old_archive_file_snapshot_drift",
                stop=archive,
            )
            rows.append(
                {
                    "relative_path": path.relative_to(archive).as_posix(),
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                }
            )
            if len(rows) > 100000:
                raise batch.AdapterError("vnext_old_archive_file_count_invalid")
    rows.sort(key=lambda item: str(item["relative_path"]))
    required = {
        "automation/controller_wal_v1/genesis.json",
        "automation/controller_wal_v1/process_lock.json",
        "automation/status.json",
    }
    if not required <= {str(item["relative_path"]) for item in rows}:
        raise batch.AdapterError("vnext_old_archive_terminal_metadata_missing")
    observed_pids: set[int] = set()
    for relative in (
        "automation/controller_wal_v1/genesis.json",
        "automation/controller_wal_v1/process_lock.json",
    ):
        raw = _physical_read(
            archive / relative,
            reason="vnext_old_archive_process_identity_drift",
            stop=archive,
        )
        value = _strict_json_mapping(
            raw,
            reason="vnext_old_archive_process_identity_invalid",
        )
        process_id = value.get("process_id")
        if not isinstance(process_id, int):
            raise batch.AdapterError("vnext_old_archive_process_identity_invalid")
        observed_pids.add(process_id)
    if observed_pids != {expected_pid}:
        raise batch.AdapterError("vnext_old_archive_process_pid_drift")
    identity = _sha256_bytes(
        b"anchor.vnext.old-archive-tree.v1\x00" + _canonical_bytes({"files": rows})
    )
    return identity, len(rows)


def _runtime_output_parent(
    policy: VNextPolicy,
    output_root: Path,
) -> Path:
    """Resolve the caller-supplied absolute runtime parent outside the repo."""

    if not output_root.is_absolute():
        raise batch.AdapterError("vnext_output_root_must_be_absolute")
    output = output_root.absolute()
    parent = output.parent
    if policy.output_root_parent is not None and parent != policy.output_root_parent:
        raise batch.AdapterError("vnext_output_root_parent_invalid")
    try:
        parent.relative_to(batch.REPO_ROOT.absolute())
    except ValueError:
        pass
    else:
        raise batch.AdapterError("vnext_output_root_inside_repository_forbidden")
    if (
        parent == output
        or output.name in {"", ".", ".."}
        or batch.SAFE_ID_RE.fullmatch(output.name) is None
    ):
        raise batch.AdapterError("vnext_unique_output_root_invalid")
    return parent


def validate_transition_evidence(
    path: Path,
    *,
    schema_path: Path = DEFAULT_TRANSITION_SCHEMA_PATH,
    expected_sha256: str | None = None,
    policy: VNextPolicy | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the attested transition and, for execute, physical facts."""

    if policy is not None:
        if schema_path.absolute() != policy.transition_schema_path:
            raise batch.AdapterError("vnext_transition_schema_path_drift")
        schema_sha256 = policy.transition_schema_sha256
    else:
        schema_sha256 = None
    value, raw = _transition_value(
        path,
        expected_sha256=expected_sha256,
        schema_path=schema_path,
        schema_sha256=schema_sha256,
    )
    if bool(value["old_controller_pid_alive"]) or _pid_alive(
        int(value["old_controller_pid"])
    ):
        raise batch.AdapterError("vnext_old_controller_still_alive")
    if policy is not None:
        if output_root is None:
            raise batch.AdapterError("vnext_transition_output_root_missing")
        old_canonical = Path(str(value["old_canonical_root"])).absolute()
        archive_root = Path(str(value["old_archive_root"])).absolute()
        attested_vnext = Path(str(value["vnext_root"])).absolute()
        runtime_parent = _runtime_output_parent(policy, output_root)
        if (
            old_canonical != policy.old_canonical_root
            or attested_vnext != output_root.absolute()
            or archive_root.parent != policy.old_archive_parent
            or not _named_regular_directory(
                archive_root,
                stop=batch.REPO_ROOT.absolute(),
            )
            or not _named_entry_absent(
                policy.old_canonical_root,
                stop=batch.REPO_ROOT.absolute(),
            )
            or not _named_entry_absent(
                output_root,
                stop=runtime_parent,
            )
        ):
            raise batch.AdapterError("vnext_transition_physical_fact_drift")
        closed._reject_reparse_chain(
            archive_root,
            stop=batch.REPO_ROOT.absolute(),
        )
        archive_identity, archive_files = _archived_root_identity(
            archive_root,
            expected_pid=int(value["old_controller_pid"]),
        )
        if not hmac.compare_digest(
            archive_identity,
            str(value["old_archive_identity_sha256"]),
        ):
            raise batch.AdapterError("vnext_old_archive_identity_drift")
    else:
        archive_files = None
    return {
        "transition_gate_satisfied": True,
        "legacy_controller_pid_verified_dead": int(value["old_controller_pid"]),
        "controller_scope": "exact_legacy_pid_plus_archive_and_vnext_os_lease",
        "old_canonical_archive_state": "archived_entire_root_no_replace",
        "canonical_root_absent": True,
        "vnext_root_absent": True,
        "legacy_disposition": "historical_quarantine_only",
        "legacy_partial_uncertain_consumable": False,
        "credential_channel": "anonymous_stdin_same_process",
        "runtime_hmac_scope": "same_process_memory_only",
        "evidence_sha256": _sha256_bytes(raw),
        "old_archive_identity_sha256": str(value["old_archive_identity_sha256"]),
        "controller_inventory_sha256": str(value["controller_inventory_sha256"]),
        "controller_count": int(value["controller_count"]),
        "old_archive_file_count": archive_files,
        "content_retained": False,
    }


def _terminal_transition_recheck(
    *,
    policy: VNextPolicy,
    transition_evidence: Path,
    transition_evidence_sha256: str,
    output_root: Path,
    lease: _VNextControllerLease,
    process_inventory_probe: Callable[
        [], Mapping[str, Any]
    ] = _trusted_controller_inventory,
) -> dict[str, Any]:
    """Repeat every launch fact while the fixed OS lease is still held."""

    lease.authenticate()
    policy_revalidation_sha256 = _revalidate_policy_physical_identity(policy)
    value, raw = _transition_value(
        transition_evidence,
        expected_sha256=transition_evidence_sha256,
        schema_path=policy.transition_schema_path,
        schema_sha256=policy.transition_schema_sha256,
    )
    old_pid = int(value["old_controller_pid"])
    old_canonical = Path(str(value["old_canonical_root"])).absolute()
    archive_root = Path(str(value["old_archive_root"])).absolute()
    attested_vnext = Path(str(value["vnext_root"])).absolute()
    output = output_root.absolute()
    runtime_parent = _runtime_output_parent(policy, output)
    if (
        bool(value["old_controller_pid_alive"])
        or _pid_alive(old_pid)
        or old_canonical != policy.old_canonical_root
        or not _named_entry_absent(
            old_canonical,
            stop=batch.REPO_ROOT.absolute(),
        )
        or archive_root.parent != policy.old_archive_parent
        or not _named_regular_directory(
            archive_root,
            stop=batch.REPO_ROOT.absolute(),
        )
        or attested_vnext != output
        or not _named_regular_directory(
            output,
            stop=runtime_parent,
        )
    ):
        raise batch.AdapterError("vnext_transition_terminal_fact_drift")
    closed._reject_reparse_chain(
        archive_root,
        stop=batch.REPO_ROOT.absolute(),
    )
    closed._reject_reparse_chain(
        output,
        stop=runtime_parent,
    )
    archive_identity, archive_files = _archived_root_identity(
        archive_root,
        expected_pid=old_pid,
    )
    if not hmac.compare_digest(
        archive_identity,
        str(value["old_archive_identity_sha256"]),
    ):
        raise batch.AdapterError("vnext_old_archive_identity_drift")
    process_inventory = dict(process_inventory_probe())
    required_inventory = {
        "controller_count",
        "candidate_controller_count",
        "candidate_launch_process_count",
        "candidate_controller_pid",
        "controller_inventory_sha256",
        "all_matching_inventory_sha256",
        "content_retained",
    }
    if (
        set(process_inventory) != required_inventory
        or process_inventory["controller_count"] != 0
        or process_inventory["candidate_controller_count"] != 1
        or not isinstance(
            process_inventory["candidate_launch_process_count"],
            int,
        )
        or process_inventory["candidate_launch_process_count"] < 1
        or process_inventory["candidate_controller_pid"] != os.getpid()
        or process_inventory["content_retained"] is not False
        or int(value["controller_count"]) != 0
        or value["controller_inventory_scope"] != _CONTROLLER_INVENTORY_SCOPE
        or not hmac.compare_digest(
            str(process_inventory["controller_inventory_sha256"]),
            str(value["controller_inventory_sha256"]),
        )
    ):
        raise batch.AdapterError("vnext_controller_inventory_drift")
    lease.authenticate()
    payload = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2."
            "teacher-alignment-vnext.terminal-transition-recheck.v1"
        ),
        "transition_evidence_sha256": _sha256_bytes(raw),
        "policy_revalidation_sha256": policy_revalidation_sha256,
        "old_controller_pid": old_pid,
        "old_archive_identity_sha256": archive_identity,
        "old_archive_file_count": archive_files,
        "controller_count": 0,
        "candidate_controller_count": 1,
        "candidate_launch_process_count": process_inventory[
            "candidate_launch_process_count"
        ],
        "candidate_controller_pid": os.getpid(),
        "controller_inventory_sha256": process_inventory["controller_inventory_sha256"],
        "all_matching_inventory_sha256": process_inventory[
            "all_matching_inventory_sha256"
        ],
        "canonical_root_absent": True,
        "vnext_root_present": True,
        "content_retained": False,
    }
    return {
        **payload,
        "recheck_sha256": _sha256_bytes(_canonical_bytes(payload)),
    }


def _exclusive_vnext_root(policy: VNextPolicy, output_root: Path) -> None:
    parent = _runtime_output_parent(policy, output_root)
    parent.mkdir(parents=True, exist_ok=True)
    closed._reject_reparse_chain(
        parent,
        stop=parent,
    )
    if output_root.parent != parent or not _named_entry_absent(
        output_root, stop=parent
    ):
        raise batch.AdapterError("vnext_unique_output_root_invalid")
    try:
        os.mkdir(output_root, 0o700)
    except FileExistsError as error:
        raise batch.AdapterError("vnext_output_root_no_replace") from error
    except OSError as error:
        raise batch.AdapterError("vnext_output_root_create_failed") from error
    batch._fsync_parent(parent)


def _read_anonymous_credential(stream: Any) -> bytes:
    try:
        raw = stream.read(batch.RuntimeSecretSlots._MAX_CREDENTIAL_BYTES + 1)
    except Exception as error:
        raise batch.AdapterError("credential_stdin_read_failed") from error
    if not isinstance(raw, bytes):
        raise batch.AdapterError("controller_credential_slot_invalid")
    if raw.endswith(b"\r\n"):
        return raw[:-2]
    if raw.endswith(b"\n"):
        return raw[:-1]
    return raw


def _signed_launch_authorization(
    *,
    policy: VNextPolicy,
    output_root: Path,
    transition_sha256: str,
    lease_path: Path,
    lease_sha256: str,
    controller_run_id: str,
    terminal_recheck: Mapping[str, Any],
    hmac_key: bytes,
) -> dict[str, Any]:
    payload = {
        "schema_version": LAUNCH_AUTHORIZATION_SCHEMA_VERSION,
        "controller_run_id": controller_run_id,
        "vnext_config_sha256": policy.physical_sha256,
        "transition_evidence_sha256": transition_sha256,
        "output_root": str(output_root),
        "controller_lease_path": str(lease_path),
        "controller_lease_sha256": lease_sha256,
        "terminal_transition_recheck_sha256": terminal_recheck["recheck_sha256"],
        "old_archive_identity_sha256": terminal_recheck["old_archive_identity_sha256"],
        "controller_inventory_sha256": terminal_recheck["controller_inventory_sha256"],
        "live_authorized": True,
        "cross_process_resume": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "content_retained": False,
        "credential_retained": False,
    }
    return {
        **payload,
        "hmac_sha256": batch._receipt_hmac(payload, hmac_key),
    }


def _persist_launch_authorization(
    output_root: Path,
    value: Mapping[str, Any],
) -> None:
    raw = _canonical_bytes(value, newline=True)
    path = output_root / "launch_authorization.json"
    _exclusive_bytes(path, raw)
    digest = _sha256_bytes(raw)
    _exclusive_bytes(
        path.with_name(f"{path.name}.sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )


def _rename_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    before_rename: Callable[[], None],
) -> None:
    """Atomically publish one authenticated directory without replacement."""

    before_rename()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination)):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError("vnext_P_materializer_no_replace")
            raise OSError(error, "vnext_P_materializer_atomic_publish_failed")
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise batch.AdapterError("vnext_P_materializer_atomic_publish_unsupported")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        ):
            error = ctypes.get_errno()
            if error == 17:
                raise FileExistsError("vnext_P_materializer_no_replace")
            raise OSError(error, "vnext_P_materializer_atomic_publish_failed")
        return
    raise batch.AdapterError("vnext_P_materializer_atomic_publish_unsupported")


def _canonical_jsonl_rows(
    raw: bytes,
    *,
    reason: str,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not raw:
        if allow_empty:
            return []
        raise batch.AdapterError(reason)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise batch.AdapterError(reason)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        value = closed._strict_json(line, reason=reason)
        if not isinstance(value, dict) or _canonical_bytes(value) != line:
            raise batch.AdapterError(reason)
        rows.append(value)
    return rows


def _teacher_record_validator(policy: VNextPolicy) -> Draft202012Validator:
    raw = _physical_read(
        policy.teacher_record_schema_path,
        reason="vnext_teacher_record_schema_snapshot_drift",
        stop=batch.REPO_ROOT.absolute(),
    )
    if not hmac.compare_digest(
        _sha256_bytes(raw),
        policy.teacher_record_schema_sha256,
    ):
        raise batch.AdapterError("vnext_teacher_record_schema_identity_drift")
    schema = _strict_json_mapping(
        raw,
        reason="vnext_teacher_record_schema_invalid",
    )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise batch.AdapterError("vnext_teacher_record_schema_invalid") from error
    return Draft202012Validator(schema)


def _canonical_teacher_target(
    job: batch.AlignmentJob,
    runtime_record: Mapping[str, Any],
) -> str:
    output = runtime_record.get("output")
    if not isinstance(output, Mapping):
        raise batch.AdapterError("vnext_teacher_target_output_missing")
    transport = _transport_from_persisted_output(job.source, output)
    validated = batch.validate_teacher_output(job.source, transport)
    if set(validated) != {"kind", "value"} or _canonical_bytes(
        validated
    ) != _canonical_bytes(dict(output)):
        raise batch.AdapterError("vnext_teacher_target_revalidation_drift")
    kind = validated["kind"]
    value = validated["value"]
    if kind == "natural_text":
        if not isinstance(value, str) or not value:
            raise batch.AdapterError("vnext_teacher_target_natural_invalid")
        return value
    if kind == "structured_json" and isinstance(value, Mapping):
        return _canonical_bytes(value).decode("utf-8")
    raise batch.AdapterError("vnext_teacher_target_transport_invalid")


def _project_teacher_record(
    *,
    selected: Mapping[str, Any],
    job: batch.AlignmentJob,
    runtime_record: Mapping[str, Any],
    validator: Draft202012Validator,
) -> tuple[dict[str, Any], str, str]:
    hash_fields = (
        "source_record_id_sha256",
        "source_content_sha256",
        "source_serialization_identity_sha256",
        "task_bundle_sha256",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(selected.get(field))) is None
        for field in hash_fields
    ) or selected.get("training_asset") not in set(
        selector.TRAINING_ASSET_BY_OUTPUT_ROLE.values()
    ):
        raise batch.AdapterError("vnext_teacher_selected_projection_invalid")
    source_binding = runtime_record.get("source_binding")
    projection_fields = (*hash_fields, "training_asset")
    if (
        runtime_record.get("idempotency_key") != job.idempotency_key
        or runtime_record.get("record_id") != job.source.record_id
        or runtime_record.get("task_bundle") != job.source.task_bundle
        or runtime_record.get("hidden_reasoning_retained") is not False
        or not isinstance(source_binding, Mapping)
        or source_binding.get("selected_candidate_sha256")
        != selected.get("candidate_id_sha256")
        or any(
            source_binding.get(field) != selected.get(field)
            for field in projection_fields
        )
        or _sha256_text(str(runtime_record["record_id"]))
        != selected["source_record_id_sha256"]
        or _sha256_text(str(runtime_record["task_bundle"]))
        != selected["task_bundle_sha256"]
    ):
        raise batch.AdapterError("vnext_teacher_runtime_projection_drift")
    target = _canonical_teacher_target(job, runtime_record)
    target_sha256 = _sha256_text(target)
    teacher_record = {
        "schema_version": TEACHER_RECORD_SCHEMA_VERSION,
        "namespace": TEACHER_RECORD_NAMESPACE,
        "record_id_sha256": selected["source_record_id_sha256"],
        "source_content_sha256": selected["source_content_sha256"],
        "source_serialization_identity_sha256": selected[
            "source_serialization_identity_sha256"
        ],
        "task_bundle_sha256": selected["task_bundle_sha256"],
        "source_asset": selected["training_asset"],
        "decision": "accepted",
        "teacher_target": target,
        "teacher_target_sha256": target_sha256,
    }
    try:
        validator.validate(teacher_record)
    except ValidationError as error:
        raise batch.AdapterError("vnext_teacher_record_schema_mismatch") from error
    teacher_record_sha256 = _sha256_bytes(_canonical_bytes(teacher_record))
    return teacher_record, target_sha256, teacher_record_sha256


def _build_teacher_training_payloads(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
    runtime_records: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    state: VNextAuthenticatedBatchState,
    inventory: batch.SourceInventory,
    validator: Draft202012Validator,
    hmac_key: bytes,
    max_shard_bytes_exclusive: int = TEACHER_TRAINING_SHARD_MAX_BYTES_EXCLUSIVE,
) -> tuple[bytes, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    if max_shard_bytes_exclusive < 1:
        raise batch.AdapterError("vnext_teacher_shard_limit_invalid")
    materialized_selected = [dict(row) for row in selected_rows]
    if (
        len(materialized_selected) != 320
        or materialized_selected
        != sorted(
            materialized_selected,
            key=lambda row: str(row["candidate_id_sha256"]),
        )
        or len({str(row["idempotency_key"]) for row in materialized_selected}) != 320
    ):
        raise batch.AdapterError("vnext_teacher_selected_inventory_drift")
    selected_by_key = {
        str(row["idempotency_key"]): (index, row)
        for index, row in enumerate(materialized_selected)
    }
    accepted_by_key = {
        str(row.get("idempotency_key")): dict(row) for row in accepted_rows
    }
    if (
        not accepted_by_key
        or len(accepted_by_key) != len(accepted_rows)
        or not set(accepted_by_key) <= set(selected_by_key)
        or len(accepted_by_key) > 320
    ):
        raise batch.AdapterError("vnext_teacher_accepted_inventory_drift")
    records_by_key = {
        str(row.get("idempotency_key")): dict(row) for row in runtime_records
    }
    if len(records_by_key) != len(runtime_records):
        raise batch.AdapterError("vnext_teacher_runtime_record_duplicate")
    receipts_by_key: dict[str, dict[str, Any]] = {}
    for raw_receipt in runtime_receipts:
        receipt = dict(raw_receipt)
        batch._verify_receipt(receipt, hmac_key)
        key = str(receipt.get("idempotency_key"))
        if key in receipts_by_key:
            raise batch.AdapterError("vnext_teacher_runtime_receipt_duplicate")
        receipts_by_key[key] = receipt

    teacher_records: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for selected_ordinal, selected in enumerate(materialized_selected):
        key = str(selected["idempotency_key"])
        accepted = accepted_by_key.get(key)
        if accepted is None:
            continue
        runtime_record = records_by_key.get(key)
        receipt = receipts_by_key.get(key)
        job = state._jobs_by_key.get(key)
        selection_view = state._selection_views.get(key)
        if (
            runtime_record is None
            or receipt is None
            or job is None
            or selection_view != selected
            or accepted.get("outcome") != "accepted"
            or accepted.get("receipt_sha256") != batch._hash_object(receipt)
            or accepted.get("output_record_sha256")
            != batch._hash_object(runtime_record)
            or receipt.get("outcome") != "succeeded"
            or receipt.get("output_record_sha256")
            != accepted.get("output_record_sha256")
        ):
            raise batch.AdapterError("vnext_teacher_accepted_runtime_join_drift")
        state._final_quality_validator(
            job,
            runtime_record,
            selection_view,
            identity_derivative_spec=state._identity_derivative_specs.get(key),
            inventory=inventory,
            config=state.config,
        )
        teacher_record, target_sha256, teacher_record_sha256 = _project_teacher_record(
            selected=selected,
            job=job,
            runtime_record=runtime_record,
            validator=validator,
        )
        training_row = {
            "schema_version": TRAINING_INVENTORY_SCHEMA_VERSION,
            "training_ordinal": len(training_rows),
            "selected_ordinal": selected_ordinal,
            "candidate_id_sha256": selected["candidate_id_sha256"],
            "selected_candidate_sha256": _sha256_bytes(_canonical_bytes(selected)),
            "idempotency_key": key,
            "source_record_id_sha256": selected["source_record_id_sha256"],
            "source_content_sha256": selected["source_content_sha256"],
            "source_serialization_identity_sha256": selected[
                "source_serialization_identity_sha256"
            ],
            "task_bundle_sha256": selected["task_bundle_sha256"],
            "source_asset": selected["training_asset"],
            "decision": "accepted",
            "receipt_sha256": accepted["receipt_sha256"],
            "output_record_sha256": accepted["output_record_sha256"],
            "teacher_target_sha256": target_sha256,
            "teacher_record_sha256": teacher_record_sha256,
            "content_retained": False,
            "raw_token_ids_retained": False,
        }
        required = {
            "schema_version",
            "training_ordinal",
            "selected_ordinal",
            "candidate_id_sha256",
            "selected_candidate_sha256",
            "idempotency_key",
            "source_record_id_sha256",
            "source_content_sha256",
            "source_serialization_identity_sha256",
            "task_bundle_sha256",
            "source_asset",
            "decision",
            "receipt_sha256",
            "output_record_sha256",
            "teacher_target_sha256",
            "teacher_record_sha256",
            "content_retained",
            "raw_token_ids_retained",
        }
        hash_fields = required - {
            "schema_version",
            "training_ordinal",
            "selected_ordinal",
            "source_asset",
            "decision",
            "content_retained",
            "raw_token_ids_retained",
        }
        if set(training_row) != required or any(
            re.fullmatch(r"[0-9a-f]{64}", str(training_row[field])) is None
            for field in hash_fields
        ):
            raise batch.AdapterError("vnext_teacher_training_inventory_row_invalid")
        teacher_records.append(teacher_record)
        training_rows.append(training_row)

    if len(training_rows) != len(accepted_rows):
        raise batch.AdapterError("vnext_teacher_training_inventory_omission")
    shard_raw = b"".join(
        _canonical_bytes(record, newline=True) for record in teacher_records
    )
    training_inventory_raw = b"".join(
        _canonical_bytes(row, newline=True) for row in training_rows
    )
    if (
        not shard_raw
        or len(shard_raw) >= max_shard_bytes_exclusive
        or len(
            _canonical_jsonl_rows(
                shard_raw,
                reason="vnext_teacher_shard_physical_drift",
            )
        )
        != len(training_rows)
        or len(
            _canonical_jsonl_rows(
                training_inventory_raw,
                reason="vnext_teacher_training_inventory_physical_drift",
            )
        )
        != len(training_rows)
    ):
        raise batch.AdapterError("vnext_teacher_shard_size_or_count_invalid")
    return shard_raw, training_inventory_raw, training_rows, teacher_records


def _materialize_producer_payload_P(
    *,
    repo_root: Path,
    release_root: str,
    producer_payload_P_paths: Sequence[str],
    runtime_slots: batch.RuntimeSecretSlots,
    lease_authenticate: Callable[[], None],
    producer_launch_L: Mapping[str, Any],
    overlay: selector.AuthenticatedMetadataOverlay,
    preimages: selector.SelectorPreimages,
    state: VNextAuthenticatedBatchState,
    inventory: batch.SourceInventory,
    freeze_attestation: Mapping[str, Any],
    staging: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize exact P32 while the credential/HMAC and lease remain live."""

    root = repo_root.absolute()
    paths = tuple(producer_payload_P_paths)
    expected_names = tuple(Path(path).name for path in paths)
    primary_names = tuple(
        name for name in expected_names if not name.endswith(".sha256")
    )
    if (
        tuple(sorted(paths)) != paths
        or len(paths) != 32
        or len(set(paths)) != 32
        or len(primary_names) != 16
        or any(Path(path).parent.as_posix() != release_root for path in paths)
        or {f"{name}.sha256" for name in primary_names}
        != {name for name in expected_names if name.endswith(".sha256")}
    ):
        raise batch.AdapterError("vnext_P_materializer_path_set_invalid")
    if not runtime_slots.loaded:
        raise batch.AdapterError("vnext_P_materializer_runtime_closed")
    hmac_key = runtime_slots.receipt_hmac_key()
    lease_authenticate()
    if _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        reason="vnext_P_materializer_dirty_tree",
        repo_root=root,
    ):
        raise batch.AdapterError("vnext_P_materializer_dirty_tree")
    head = (
        _git_bytes(
            "rev-parse",
            "HEAD^{commit}",
            reason="vnext_P_materializer_launch_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    tree = (
        _git_bytes(
            "rev-parse",
            "HEAD^{tree}",
            reason="vnext_P_materializer_launch_identity_drift",
            repo_root=root,
        )
        .decode("ascii")
        .strip()
    )
    if (
        producer_launch_L.get("stage") != "producer_launch_L"
        or head != producer_launch_L.get("commit_sha1")
        or tree != producer_launch_L.get("tree_sha1")
        or producer_launch_L.get("global_clean_tree") is not True
        or freeze_attestation.get("full_replay_counts")
        != {"startup": 1, "final_freeze": 1}
        or staging.get("state") != "awaiting_independent_external_attestation"
    ):
        raise batch.AdapterError("vnext_P_materializer_launch_identity_drift")

    def authenticated_source(path: Path, *, stop: Path) -> bytes:
        raw = _physical_read(
            path,
            reason="vnext_P_materializer_source_drift",
            stop=stop,
        )
        if not hmac.compare_digest(
            _authenticated_file_digest(
                path,
                stop=stop,
                reason="vnext_P_materializer_source_drift",
            ),
            _sha256_bytes(raw),
        ):
            raise batch.AdapterError("vnext_P_materializer_source_drift")
        return raw

    freeze_raw = authenticated_source(
        state.freeze_dir / "freeze_attestation.json",
        stop=state.freeze_dir,
    )
    authenticated_freeze = _strict_json_mapping(
        freeze_raw,
        reason="vnext_P_materializer_freeze_invalid",
    )
    freeze_payload = state._verify_signed_value(
        authenticated_freeze,
        reason="vnext_P_materializer_freeze_hmac_drift",
    )
    if (
        authenticated_freeze != dict(freeze_attestation)
        or freeze_payload.get("state") != "candidate_pending_external_attestation"
        or freeze_payload.get("selected_set_count") != 320
    ):
        raise batch.AdapterError("vnext_P_materializer_freeze_identity_drift")

    selector_schema_raw = _physical_read(
        state.policy.selector_schema_path,
        reason="vnext_P_materializer_selector_schema_drift",
        stop=root,
    )
    if not hmac.compare_digest(
        _sha256_bytes(selector_schema_raw),
        state.policy.selector_schema_sha256,
    ):
        raise batch.AdapterError("vnext_P_materializer_selector_schema_drift")
    _candidate_universe, authenticated_selection = (
        selector.authenticate_selector_preimages(
            preimages,
            overlay=overlay,
            inventory=inventory,
            config=state.config,
            contract_sha256=state.policy.selector_contract_sha256,
            candidate_schema_path=state.policy.selector_schema_path,
        )
    )
    selected_raw = authenticated_source(
        preimages.selected_rows_path,
        stop=preimages.directory,
    )
    selected_rows = _canonical_jsonl_rows(
        selected_raw,
        reason="vnext_P_materializer_selected_inventory_drift",
    )
    expected_selected_rows = [
        candidate.as_dict() for candidate in authenticated_selection.candidates
    ]
    selected_keys = tuple(str(row["idempotency_key"]) for row in selected_rows)
    if (
        selected_rows != expected_selected_rows
        or authenticated_selection.manifest != state.selection_manifest
        or freeze_payload.get("selected_set_root_sha256")
        != authenticated_selection.manifest.get("selected_set_root_sha256")
        or freeze_payload.get("selector_identity_sha256")
        != authenticated_selection.manifest.get("selector_identity_sha256")
        or preimages.manifest.get("selected_set_root_sha256")
        != freeze_payload.get("selected_set_root_sha256")
        or tuple(state._selection_views) != selected_keys
        or [dict(state._selection_views[key]) for key in selected_keys] != selected_rows
    ):
        raise batch.AdapterError("vnext_P_materializer_selected_inventory_drift")

    accepted_raw = authenticated_source(
        state.freeze_dir / "accepted_inventory.jsonl",
        stop=state.freeze_dir,
    )
    accepted_rows = _canonical_jsonl_rows(
        accepted_raw,
        reason="vnext_P_materializer_accepted_inventory_drift",
    )
    accepted_binding = freeze_payload.get("inventory_bindings", {}).get("accepted")
    accepted_count = len(accepted_rows)
    if (
        not isinstance(accepted_binding, Mapping)
        or freeze_payload.get("inventory_sha256", {}).get("accepted")
        != _sha256_bytes(accepted_raw)
        or accepted_binding.get("sha256") != _sha256_bytes(accepted_raw)
        or accepted_binding.get("inventory_root_sha256") != _merkle_root(accepted_rows)
        or accepted_binding.get("count") != accepted_count
        or accepted_binding.get("outcome") != "accepted"
        or freeze_payload.get("counts", {}).get("accepted") != accepted_count
        or any(row.get("outcome") != "accepted" for row in accepted_rows)
    ):
        raise batch.AdapterError("vnext_P_materializer_accepted_inventory_drift")

    runtime_records = _physical_jsonl(
        state.records_path,
        id_field="idempotency_key",
        stop=state.root,
        reason="vnext_P_materializer_runtime_records_drift",
    )
    runtime_receipts = _physical_jsonl(
        state.receipts_path,
        id_field="id",
        stop=state.root,
        reason="vnext_P_materializer_runtime_receipts_drift",
    )
    accepted_keys = {str(row["idempotency_key"]) for row in accepted_rows}
    if {str(row["idempotency_key"]) for row in runtime_records} != accepted_keys or {
        str(row["idempotency_key"]) for row in runtime_receipts
    } != accepted_keys:
        raise batch.AdapterError("vnext_P_materializer_runtime_join_drift")
    teacher_validator = _teacher_record_validator(state.policy)
    (
        teacher_shard_raw,
        training_inventory_raw,
        training_rows,
        teacher_records,
    ) = _build_teacher_training_payloads(
        selected_rows=selected_rows,
        accepted_rows=accepted_rows,
        runtime_records=runtime_records,
        runtime_receipts=runtime_receipts,
        state=state,
        inventory=inventory,
        validator=teacher_validator,
        hmac_key=hmac_key,
    )
    teacher_shard_sidecar_raw = (
        f"{_sha256_bytes(teacher_shard_raw)}  {TEACHER_TRAINING_SHARD_NAME}\n"
    ).encode("ascii")
    training_inventory_sidecar_raw = (
        f"{_sha256_bytes(training_inventory_raw)}  {TRAINING_RECORD_INVENTORY_NAME}\n"
    ).encode("ascii")

    checkpoint_entries: list[dict[str, Any]] = []
    for checkpoint_path in sorted(state.checkpoint_dir.glob("*.json")):
        raw = authenticated_source(checkpoint_path, stop=state.checkpoint_dir)
        checkpoint_entries.append(
            {
                "path": checkpoint_path.relative_to(state.root).as_posix(),
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
            }
        )
    checkpoints_inventory = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "checkpoints-inventory.v1"
        ),
        "checkpoint_count": len(checkpoint_entries),
        "files": checkpoint_entries,
        "file_inventory_root_sha256": _merkle_root(checkpoint_entries),
        "checkpoint_manifest_root_sha256": freeze_attestation[
            "checkpoint_manifest_root_sha256"
        ],
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    shard_release_path = f"{release_root}/{TEACHER_TRAINING_SHARD_NAME}"
    shard_sidecar_release_path = f"{shard_release_path}.sha256"
    training_inventory_release_path = f"{release_root}/{TRAINING_RECORD_INVENTORY_NAME}"
    training_inventory_sidecar_release_path = (
        f"{training_inventory_release_path}.sha256"
    )
    committed_file_entries = [
        {
            "path": shard_release_path,
            "sha256": _sha256_bytes(teacher_shard_raw),
            "bytes": len(teacher_shard_raw),
            "kind": "teacher_training_shard",
        },
        {
            "path": shard_sidecar_release_path,
            "sha256": _sha256_bytes(teacher_shard_sidecar_raw),
            "bytes": len(teacher_shard_sidecar_raw),
            "kind": "sha256_sidecar",
        },
        {
            "path": training_inventory_release_path,
            "sha256": _sha256_bytes(training_inventory_raw),
            "bytes": len(training_inventory_raw),
            "kind": "training_record_inventory",
        },
        {
            "path": training_inventory_sidecar_release_path,
            "sha256": _sha256_bytes(training_inventory_sidecar_raw),
            "bytes": len(training_inventory_sidecar_raw),
            "kind": "sha256_sidecar",
        },
    ]
    freeze_counts = dict(freeze_payload["counts"])
    exact_requested_set_complete = freeze_counts == {
        "requested": 320,
        "completed": 320,
        "accepted": 320,
        "rejected": 0,
        "quarantine": 0,
    }
    committed_shards = {
        "schema_version": COMMITTED_SHARDS_SCHEMA_VERSION,
        "release_root": release_root,
        "shard_count": 1,
        "files": committed_file_entries,
        "file_count": len(committed_file_entries),
        "file_inventory_root_sha256": _merkle_root(committed_file_entries),
        "teacher_training_shard": {
            "path": shard_release_path,
            "sidecar_path": shard_sidecar_release_path,
            "schema_version": TEACHER_RECORD_SCHEMA_VERSION,
            "schema_sha256": state.policy.teacher_record_schema_sha256,
            "row_count": accepted_count,
            "bytes": len(teacher_shard_raw),
            "max_bytes_exclusive": TEACHER_TRAINING_SHARD_MAX_BYTES_EXCLUSIVE,
            "sha256": _sha256_bytes(teacher_shard_raw),
            "record_inventory_root_sha256": _merkle_root(teacher_records),
        },
        "training_record_inventory": {
            "path": training_inventory_release_path,
            "sidecar_path": training_inventory_sidecar_release_path,
            "schema_version": TRAINING_INVENTORY_SCHEMA_VERSION,
            "row_count": accepted_count,
            "bytes": len(training_inventory_raw),
            "sha256": _sha256_bytes(training_inventory_raw),
            "inventory_root_sha256": _merkle_root(training_rows),
        },
        "counts": freeze_counts,
        "partial_accepted_subset": not exact_requested_set_complete,
        "exact_requested_set_complete": exact_requested_set_complete,
        "accepted_selected_order_root_sha256": _sha256_bytes(
            _canonical_bytes([row["candidate_id_sha256"] for row in training_rows])
        ),
        "teacher_target_order_root_sha256": _sha256_bytes(
            _canonical_bytes([row["teacher_target_sha256"] for row in training_rows])
        ),
        "zip_alignment": {
            "row_count": accepted_count,
            "training_ordinals_contiguous": True,
            "selected_order_filtered_accepted": True,
            "teacher_record_sha256_bound": True,
        },
        "accepted_inventory_sha256": _sha256_bytes(accepted_raw),
        "selected_set_root_sha256": freeze_payload["selected_set_root_sha256"],
        "accepted_strata_root_sha256": freeze_payload["accepted_strata_root_sha256"],
        "final_chain_tip_sha256": freeze_payload["final_chain_tip_sha256"],
        "final_wal_entries": freeze_payload["final_wal_entries"],
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    launch_inventory = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "producer-launch-L-inventory.v1"
        ),
        "producer_launch_L": dict(producer_launch_L),
        "content_retained": False,
    }
    primary_raw = {
        "accepted_inventory.jsonl": accepted_raw,
        "authenticated_metadata_overlay.jsonl": authenticated_source(
            overlay.rows_path,
            stop=overlay.directory,
        ),
        "authenticated_metadata_overlay_manifest.json": authenticated_source(
            overlay.manifest_path,
            stop=overlay.directory,
        ),
        "candidate_inventory.jsonl": authenticated_source(
            preimages.candidate_rows_path,
            stop=preimages.directory,
        ),
        "checkpoint_manifest.json": authenticated_source(
            state.freeze_dir / "checkpoint_manifest.json",
            stop=state.freeze_dir,
        ),
        "checkpoints_inventory.json": _canonical_bytes(
            checkpoints_inventory,
            newline=True,
        ),
        "committed_shards_manifest.json": _canonical_bytes(
            committed_shards,
            newline=True,
        ),
        "producer_launch_inventory.json": _canonical_bytes(
            launch_inventory,
            newline=True,
        ),
        "quarantine_inventory.jsonl": authenticated_source(
            state.freeze_dir / "quarantine_inventory.jsonl",
            stop=state.freeze_dir,
        ),
        "rejected_inventory.jsonl": authenticated_source(
            state.freeze_dir / "rejected_inventory.jsonl",
            stop=state.freeze_dir,
        ),
        "selected_inventory.jsonl": selected_raw,
        "selection_manifest.json": authenticated_source(
            preimages.selection_manifest_path,
            stop=preimages.directory,
        ),
        "selection_preimage.json": authenticated_source(
            preimages.selection_preimage_path,
            stop=preimages.directory,
        ),
        "selector_preimage_manifest.json": authenticated_source(
            preimages.preimage_manifest_path,
            stop=preimages.directory,
        ),
        TEACHER_TRAINING_SHARD_NAME: teacher_shard_raw,
        TRAINING_RECORD_INVENTORY_NAME: training_inventory_raw,
    }
    if tuple(sorted(primary_raw)) != primary_names:
        raise batch.AdapterError("vnext_P_materializer_path_set_invalid")

    destination = root / release_root
    targets = tuple(root / path for path in paths)
    if (
        destination.exists()
        or destination.is_symlink()
        or any(target.exists() or target.is_symlink() for target in targets)
    ):
        raise batch.AdapterError("vnext_P_materializer_no_replace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    closed._reject_reparse_chain(destination.parent, stop=root)
    staging_directory = destination.parent / (
        f".{destination.name}.staging-{secrets.token_hex(16)}"
    )
    try:
        staging_directory.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise batch.AdapterError("vnext_P_materializer_staging_collision") from error
    staging_stat = staging_directory.lstat()
    staging_identity = (
        staging_stat.st_dev,
        staging_stat.st_ino,
        stat.S_IFMT(staging_stat.st_mode),
        int(getattr(staging_stat, "st_file_attributes", 0)),
    )
    output_by_name: dict[str, bytes] = {}
    for name, raw in primary_raw.items():
        output_by_name[name] = raw
        output_by_name[f"{name}.sha256"] = f"{_sha256_bytes(raw)}  {name}\n".encode(
            "ascii"
        )
    for name in expected_names:
        _exclusive_bytes(staging_directory / name, output_by_name[name])
    if (
        lambda observed: (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
            int(getattr(observed, "st_file_attributes", 0)),
        )
    )(staging_directory.lstat()) != staging_identity or tuple(
        sorted(path.name for path in staging_directory.iterdir())
    ) != expected_names:
        raise batch.AdapterError("vnext_P_materializer_staging_drift")
    release_file_inventory: list[dict[str, Any]] = []
    for path, name in zip(paths, expected_names, strict=True):
        staged = staging_directory / name
        raw = _physical_read(
            staged,
            reason="vnext_P_materializer_staging_drift",
            stop=staging_directory,
        )
        if not hmac.compare_digest(raw, output_by_name[name]):
            raise batch.AdapterError("vnext_P_materializer_staging_drift")
        release_file_inventory.append(
            {
                "path": path,
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
                "release_stage": "producer_payload_P",
            }
        )

    def authenticate_before_publish() -> None:
        lease_authenticate()
        if not runtime_slots.loaded:
            raise batch.AdapterError("vnext_P_materializer_runtime_closed")
        observed = staging_directory.lstat()
        if (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
            int(getattr(observed, "st_file_attributes", 0)),
        ) != staging_identity:
            raise batch.AdapterError("vnext_P_materializer_staging_drift")
        observed_head = (
            _git_bytes(
                "rev-parse",
                "HEAD^{commit}",
                reason="vnext_P_materializer_launch_identity_drift",
                repo_root=root,
            )
            .decode("ascii")
            .strip()
        )
        observed_tree = (
            _git_bytes(
                "rev-parse",
                "HEAD^{tree}",
                reason="vnext_P_materializer_launch_identity_drift",
                repo_root=root,
            )
            .decode("ascii")
            .strip()
        )
        if observed_head != head or observed_tree != tree:
            raise batch.AdapterError("vnext_P_materializer_launch_identity_drift")

    try:
        _rename_directory_noreplace(
            staging_directory,
            destination,
            before_rename=authenticate_before_publish,
        )
    except FileExistsError as error:
        raise batch.AdapterError("vnext_P_materializer_no_replace") from error
    batch._fsync_parent(destination.parent)
    observed_status = _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        reason="vnext_P_materializer_output_drift",
        repo_root=root,
    )
    status_parts = tuple(
        entry for entry in observed_status.decode("utf-8").split("\x00") if entry
    )
    status_entries = tuple(
        sorted(entry[3:] for entry in status_parts if entry.startswith("?? "))
    )
    if status_entries != paths or any(
        not entry.startswith("?? ") for entry in status_parts
    ):
        raise batch.AdapterError("vnext_P_materializer_output_drift")
    for target in targets:
        raw = _physical_read(
            target,
            reason="vnext_P_materializer_output_drift",
            stop=root,
        )
        if not hmac.compare_digest(raw, output_by_name[target.name]):
            raise batch.AdapterError("vnext_P_materializer_output_drift")
    lease_authenticate()
    if not runtime_slots.loaded:
        raise batch.AdapterError("vnext_P_materializer_runtime_closed")
    binding_payload = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "producer-payload-P-binding.v1"
        ),
        "state": "producer_payload_P_materialized_awaiting_git_commit",
        "producer_launch_L_commit_sha1": head,
        "producer_launch_L_tree_sha1": tree,
        "producer_payload_P_file_count": len(release_file_inventory),
        "producer_payload_P_file_inventory_root_sha256": _merkle_root(
            release_file_inventory
        ),
        "freeze_attestation_sha256": _authenticated_file_digest(
            state.freeze_dir / "freeze_attestation.json",
            stop=state.freeze_dir,
            reason="vnext_P_materializer_freeze_binding_drift",
        ),
        "external_attestation_staging_sha256": staging["staging_sha256"],
        "training_authorized": False,
        "formal_training_authorized": False,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    binding = {
        **binding_payload,
        "hmac_sha256": batch._receipt_hmac(binding_payload, hmac_key),
    }
    binding_path = state.freeze_dir / "producer_payload_P_binding.json"
    binding_raw = _canonical_bytes(binding, newline=True)
    _exclusive_bytes(binding_path, binding_raw)
    binding_digest = _sha256_bytes(binding_raw)
    _exclusive_bytes(
        binding_path.with_name(f"{binding_path.name}.sha256"),
        f"{binding_digest}  {binding_path.name}\n".encode("ascii"),
    )
    reloaded = _strict_json_mapping(
        _physical_read(
            binding_path,
            reason="vnext_P_materializer_binding_drift",
            stop=state.freeze_dir,
        ),
        reason="vnext_P_materializer_binding_drift",
    )
    reloaded_payload = dict(reloaded)
    observed_hmac = reloaded_payload.pop("hmac_sha256", None)
    if (
        reloaded != binding
        or not isinstance(observed_hmac, str)
        or not hmac.compare_digest(
            observed_hmac,
            batch._receipt_hmac(reloaded_payload, runtime_slots.receipt_hmac_key()),
        )
        or _authenticated_file_digest(
            binding_path,
            stop=state.freeze_dir,
            reason="vnext_P_materializer_binding_drift",
        )
        != binding_digest
    ):
        raise batch.AdapterError("vnext_P_materializer_binding_drift")
    lease_authenticate()
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "producer-payload-P-materialization-receipt.v1"
        ),
        "producer_launch_L_commit_sha1": head,
        "files": release_file_inventory,
        "file_inventory_root_sha256": _merkle_root(release_file_inventory),
        "binding_sha256": binding_digest,
        "created_once": True,
        "atomic_directory_publish": True,
        "no_replace": True,
        "runtime_slots_open": True,
        "lease_authenticated": True,
        "content_retained": False,
    }


async def execute_vnext_from_process_channel(
    policy: VNextPolicy,
    transition_evidence: Path,
    transition_evidence_sha256: str,
    output_root: Path,
    receiver: Callable[[], bytes],
    *,
    profile_loader: Callable[[Path], batch.AdapterConfig] = batch.load_config,
    source_loader: Callable[
        [batch.AdapterConfig], batch.SourceInventory
    ] = batch.load_source_inventory,
    candidate_deriver: Callable[
        [
            batch.SourceInventory,
            batch.AdapterConfig,
            selector.AuthenticatedMetadataOverlay,
        ],
        Sequence[selector.SelectorCandidate],
    ] = selector.derive_candidates_from_source,
    teacher_builder: Callable[
        [batch.AdapterConfig, batch.RuntimeSecretSlots],
        Mapping[str, batch.BatchTeacher],
    ] = build_vnext_teachers,
    runner_factory: Callable[..., VNextBatchRunner] = VNextBatchRunner,
    environ: Mapping[str, str] | None = None,
    hmac_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    """Execute generation and freeze under one credential/HMAC lifetime."""

    active_environ = environ if environ is not None else os.environ
    closed._reject_environment_credentials(active_environ)
    consumer_code_P = _authenticate_consumer_code_P(policy.consumer_code_P)
    if policy.claims.get("live_authorized") is not True:
        raise batch.AdapterError("vnext_release_not_live_authorized")
    producer_launch_L = authenticate_producer_launch_L()
    output_root = output_root.absolute()
    transition = validate_transition_evidence(
        transition_evidence,
        schema_path=policy.transition_schema_path,
        expected_sha256=transition_evidence_sha256,
        policy=policy,
        output_root=output_root,
    )
    run_id = secrets.token_hex(16)
    runtime_parent = _runtime_output_parent(policy, output_root)
    runtime_parent.mkdir(parents=True, exist_ok=True)
    lease_path = runtime_parent / policy.controller_lease_name
    lease = _VNextControllerLease(lease_path, run_id)
    lease_sha256 = lease.acquire()
    try:
        lease.authenticate()
        _exclusive_vnext_root(policy, output_root)
        slots = batch.RuntimeSecretSlots.from_process_channel(
            receiver,
            hmac_factory=hmac_factory,
        )
        try:
            pre_dispatch_transition = _terminal_transition_recheck(
                policy=policy,
                transition_evidence=transition_evidence,
                transition_evidence_sha256=transition_evidence_sha256,
                output_root=output_root,
                lease=lease,
            )
            profile = profile_loader(policy.base_profile_path)
            if (
                profile.physical_sha256 != policy.base_profile_sha256
                or profile.profile_id != "bulk_c30"
                or profile.limits.allowed_phases != ("bulk",)
            ):
                raise batch.AdapterError("vnext_base_profile_binding_drift")
            output = dict(profile.output)
            output["root"] = str(output_root)
            profile = replace(profile, output=output)
            batch.validate_heldout_receipt(profile)
            inventory = source_loader(profile)
            overlay = selector.build_authenticated_metadata_overlay(
                inventory,
                profile,
                output_root=output_root,
            )
            selector.authenticate_metadata_overlay(
                overlay,
                inventory=inventory,
                config=profile,
            )
            candidates = tuple(candidate_deriver(inventory, profile, overlay))
            selection = selector.select_minimal_320(
                candidates,
                contract_sha256=policy.selector_contract_sha256,
                source_identity_sha256=inventory.source_identity_sha256,
            )
            preimages = selector.persist_selector_preimages(
                candidates,
                selection,
                overlay=overlay,
                inventory=inventory,
                config=profile,
                contract_sha256=policy.selector_contract_sha256,
            )
            _reloaded_candidates, selection = selector.authenticate_selector_preimages(
                selector.load_selector_preimages(preimages.directory),
                overlay=overlay,
                inventory=inventory,
                config=profile,
                contract_sha256=(policy.selector_contract_sha256),
            )
            identity_derivative_specs = tuple(
                dict(spec)
                for spec in overlay.manifest["identity_derivative_plan"][
                    "derivative_specs"
                ]
            )
            views = {
                item.idempotency_key: item.as_dict() for item in selection.candidates
            }
            job_provenance = build_job_provenance(
                policy=policy,
                config=profile,
                inventory=inventory,
                overlay=overlay,
                preimages=preimages,
                selection_manifest=selection.manifest,
                producer_launch_L=producer_launch_L,
                consumer_code_P=consumer_code_P,
            )
            launch = _signed_launch_authorization(
                policy=policy,
                output_root=output_root,
                transition_sha256=transition["evidence_sha256"],
                lease_path=lease_path,
                lease_sha256=lease_sha256,
                controller_run_id=run_id,
                terminal_recheck=pre_dispatch_transition,
                hmac_key=slots.receipt_hmac_key(),
            )
            _persist_launch_authorization(output_root, launch)
            binding = {
                "controller_config_sha256": policy.physical_sha256,
                "controller_implementation_sha256": (policy.implementation_sha256),
                "teacher_implementation_sha256": (
                    profile.contract_hashes["teacher_implementation_path"]
                ),
                "source_identity_sha256": inventory.source_identity_sha256,
                "manifest_sha256": inventory.manifest_sha256,
                "protected_test_identity_sha256": (
                    inventory.readonly_test_identity_sha256
                ),
                "controller_run_id": run_id,
                "terminal_transition_recheck_sha256": (
                    pre_dispatch_transition["recheck_sha256"]
                ),
                "accepted_profiles": {
                    profile.profile_id: profile.physical_sha256,
                },
            }
            runner = runner_factory(
                profile,
                inventory,
                teachers=teacher_builder(profile, slots),
                runtime_slots=slots,
                controller_binding=binding,
                policy=policy,
                selection_manifest=selection.manifest,
                selection_views=views,
                identity_derivative_specs=(identity_derivative_specs),
                job_provenance=job_provenance,
                launch_authorization=launch,
                environ=active_environ,
            )
            safely_stopped = False
            try:
                report = await runner.run("bulk")
            except StopAfterGroup:
                replay = runner.state.replay()
                checkpoint_paths = sorted(runner.state.checkpoint_dir.glob("*.json"))
                if (
                    replay.uncertain
                    or not checkpoint_paths
                    or not checkpoint_paths[-1]
                    .with_name(f"{checkpoint_paths[-1].name}.sha256")
                    .is_file()
                    or not runner.state.stop_latch.effective
                ):
                    raise batch.AdapterError(
                        "vnext_safe_stop_terminal_evidence_missing"
                    ) from None
                safely_stopped = True
                report = batch.RunReport(
                    phase="bulk",
                    total=len(runner.jobs),
                    queued=(
                        len(runner.jobs) - len(replay.completed) - len(replay.rejected)
                    ),
                    succeeded=len(replay.completed),
                    rejected=len(replay.rejected),
                    retried=len(replay.retried),
                    requests=replay.request_attempts,
                    input_tokens=replay.input_tokens,
                    output_tokens=replay.output_tokens,
                    state="safe_stopped_after_complete_group",
                )
            pre_freeze_transition = _terminal_transition_recheck(
                policy=policy,
                transition_evidence=transition_evidence,
                transition_evidence_sha256=transition_evidence_sha256,
                output_root=output_root,
                lease=lease,
            )
            freeze = runner.state.freeze_handoff(
                inventory=inventory,
                jobs=runner.jobs,
                terminal_transition_recheck=pre_freeze_transition,
            )
            staging = runner.state.stage_external_attestation(freeze)
            producer_payload_P = _materialize_producer_payload_P(
                repo_root=batch.REPO_ROOT,
                release_root=PRODUCER_RELEASE_ROOT,
                producer_payload_P_paths=PRODUCER_PAYLOAD_P_PATHS,
                runtime_slots=slots,
                lease_authenticate=lease.authenticate,
                producer_launch_L=producer_launch_L,
                overlay=overlay,
                preimages=preimages,
                state=runner.state,
                inventory=inventory,
                freeze_attestation=freeze,
                staging=staging,
            )
            lease.authenticate()
            if not (
                (runner.state.freeze_dir / "freeze_attestation.json").is_file()
                and (
                    runner.state.freeze_dir / "external_attestation_staging.json"
                ).is_file()
                and (
                    runner.state.freeze_dir / "producer_payload_P_binding.json"
                ).is_file()
            ):
                raise batch.AdapterError("vnext_terminal_artifact_recheck_failed")
            return {
                "schema_version": LAUNCH_STAGING_SCHEMA_VERSION,
                "state": "awaiting_independent_external_attestation",
                "mode": "execute",
                "controller_run_id": run_id,
                "selected": len(runner.jobs),
                "generation_state": report.state,
                "safe_stopped_after_complete_group": safely_stopped,
                "freeze_counts": dict(freeze["counts"]),
                "staging_sha256": staging["staging_sha256"],
                "producer_payload_P_binding_sha256": producer_payload_P[
                    "binding_sha256"
                ],
                "consumer_integration_ready": False,
                "release_blockers": [
                    "independent_external_attestation_missing",
                    "producer_final_R_missing",
                    "consumer_pin_C_missing",
                ],
                "training_authorized": False,
                "formal_training_authorized": False,
                "credential_retained": False,
                "content_retained": False,
            }
        finally:
            slots.close()
    finally:
        lease.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Additive teacher-alignment vNext controller."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--materialize-producer-r", action="store_true")
    modes.add_argument("--telemetry-snapshot", type=Path)
    parser.add_argument("--credential-stdin", action="store_true")
    parser.add_argument("--transition-evidence", type=Path)
    parser.add_argument("--transition-evidence-sha256")
    parser.add_argument("--vnext-root", type=Path)
    parser.add_argument("--freeze-preimage", type=Path)
    parser.add_argument("--freeze-preimage-sha256")
    parser.add_argument("--external-review-receipt", type=Path)
    parser.add_argument("--external-review-receipt-sha256")
    return parser


def _isolated_transport_contains_secret(
    value: Any,
    secret: str,
) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, Mapping):
        return any(
            _isolated_transport_contains_secret(key, secret)
            or _isolated_transport_contains_secret(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_isolated_transport_contains_secret(item, secret) for item in value)
    return False


async def _isolated_transport_request(
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    if set(frame) != {
        "schema_version",
        "kind",
        "endpoint",
        "headers",
        "payload",
        "timeout_seconds",
    } or (
        frame.get("schema_version") != ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION
        or frame.get("kind") != "request"
        or not isinstance(frame.get("endpoint"), str)
        or not isinstance(frame.get("headers"), Mapping)
        or not isinstance(frame.get("payload"), Mapping)
        or isinstance(frame.get("timeout_seconds"), bool)
        or not isinstance(frame.get("timeout_seconds"), (int, float))
        or not math.isfinite(float(frame["timeout_seconds"]))
        or float(frame["timeout_seconds"]) <= 0
    ):
        return {
            "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
            "kind": "invalid_request",
        }
    headers = dict(frame["headers"])
    authorization = headers.get("Authorization")
    idempotency_key = headers.get("Idempotency-Key")
    if (
        not isinstance(authorization, str)
        or not authorization.startswith("Bearer ")
        or len(authorization) <= len("Bearer ")
        or not isinstance(idempotency_key, str)
        or teacher_transport._validate_idempotency_key(idempotency_key)
        != idempotency_key
    ):
        return {
            "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
            "kind": "invalid_request",
        }
    credential = authorization[len("Bearer ") :]
    try:
        import httpx
    except ImportError:
        return {
            "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
            "kind": "transport_error",
            "retryable": False,
        }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(float(frame["timeout_seconds"])),
            follow_redirects=False,
        ) as client:
            response = await client.post(
                str(frame["endpoint"]),
                headers=headers,
                json=dict(frame["payload"]),
            )
        try:
            body = response.json()
        except Exception:
            body = None
        result: dict[str, Any] = {
            "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
            "kind": "response",
            "status_code": response.status_code,
            "headers": {
                "Retry-After": response.headers.get("Retry-After"),
            },
            "body": body,
        }
    except Exception as error:
        if any("Timeout" in item.__name__ for item in type(error).__mro__):
            result = {
                "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
                "kind": "timeout",
            }
        else:
            result = {
                "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
                "kind": "transport_error",
                "retryable": isinstance(error, httpx.RequestError),
            }
    if _isolated_transport_contains_secret(result, credential):
        return {
            "schema_version": ISOLATED_TRANSPORT_FRAME_SCHEMA_VERSION,
            "kind": "credential_echo",
        }
    return result


def _isolated_transport_worker_main() -> int:
    try:
        raw = sys.stdin.buffer.read(_ISOLATED_TRANSPORT_MAX_BYTES + 1)
        if len(raw) > _ISOLATED_TRANSPORT_MAX_BYTES:
            return 2
        frame = _strict_json_mapping(
            raw,
            reason="vnext_isolated_transport_request_invalid",
        )
        result = asyncio.run(_isolated_transport_request(frame))
        output = _canonical_bytes(result, newline=True)
        if len(output) > _ISOLATED_TRANSPORT_MAX_BYTES:
            return 2
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except BaseException:
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == ["--isolated-transport-worker"]:
        return _isolated_transport_worker_main()
    try:
        closed._reject_argv_credentials(raw_argv)
        args = _parser().parse_args(raw_argv)
        if args.telemetry_snapshot is not None:
            if args.credential_stdin or any(
                value is not None
                for value in (
                    args.transition_evidence,
                    args.transition_evidence_sha256,
                    args.vnext_root,
                    args.freeze_preimage,
                    args.freeze_preimage_sha256,
                    args.external_review_receipt,
                    args.external_review_receipt_sha256,
                )
            ):
                raise batch.AdapterError("vnext_telemetry_snapshot_argument_invalid")
            result = public_provider_throughput_snapshot(args.telemetry_snapshot)
        else:
            if args.execute != args.credential_stdin:
                raise batch.AdapterError(
                    "vnext_execute_requires_anonymous_credential_stdin"
                )
            policy = load_vnext_policy(args.config)
            if args.dry_run:
                if any(
                    value is not None
                    for value in (
                        args.transition_evidence,
                        args.transition_evidence_sha256,
                        args.vnext_root,
                    )
                ):
                    raise batch.AdapterError("vnext_dry_run_execute_argument_forbidden")
                result = public_dry_run(policy)
            elif args.materialize_producer_r:
                if (
                    args.credential_stdin
                    or args.transition_evidence is not None
                    or args.transition_evidence_sha256 is not None
                    or args.vnext_root is not None
                    or args.freeze_preimage is None
                    or args.external_review_receipt is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(args.freeze_preimage_sha256),
                    )
                    is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(args.external_review_receipt_sha256),
                    )
                    is None
                ):
                    raise batch.AdapterError("vnext_R_materializer_argument_invalid")
                result = materialize_producer_R(
                    freeze_preimage_path=args.freeze_preimage,
                    freeze_preimage_sha256=str(args.freeze_preimage_sha256),
                    external_review_receipt_path=(args.external_review_receipt),
                    external_review_receipt_sha256=str(
                        args.external_review_receipt_sha256
                    ),
                )
            else:
                if (
                    args.transition_evidence is None
                    or not isinstance(args.transition_evidence_sha256, str)
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        args.transition_evidence_sha256,
                    )
                    is None
                    or args.vnext_root is None
                ):
                    raise batch.AdapterError("vnext_execute_launch_argument_missing")
                closed._validate_anonymous_os_channel(sys.stdin.buffer)
                result = asyncio.run(
                    execute_vnext_from_process_channel(
                        policy,
                        args.transition_evidence,
                        args.transition_evidence_sha256,
                        args.vnext_root,
                        lambda: _read_anonymous_credential(sys.stdin.buffer),
                    )
                )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as error:
        print(
            json.dumps(
                closed._blocked(closed._body_free_reason(error)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
