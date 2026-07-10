from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


SCHEMA_VERSION = "anchor.tool-gold.v1"
ValidationStatus = Literal["PASS", "FAIL", "SKIP", "TIMEOUT"]


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    prompt: str
    source_dir: Path
    required_validations: tuple[str, ...] = ("build",)


@dataclass(frozen=True)
class ToolTraceEntry:
    sequence: int
    source: Literal["agent", "validator"]
    tool: str
    status: str
    command: str | None = None
    command_sha256: str | None = None
    exit_code: int | None = None
    duration_ms: float | None = None
    output_sha256: str | None = None


@dataclass(frozen=True)
class FileChange:
    path: str
    operation: Literal["added", "modified", "deleted"]
    before_sha256: str | None
    after_sha256: str | None


@dataclass(frozen=True)
class ValidationResult:
    name: str
    command: str
    script_present: bool
    status: ValidationStatus
    exit_code: int | None = None
    duration_ms: float = 0.0
    output_sha256: str | None = None


@dataclass(frozen=True)
class AgentExecution:
    exit_code: int
    timed_out: bool
    duration_ms: float
    trace: tuple[ToolTraceEntry, ...] = ()
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    rejected_events: int = 0
    error_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldRecord:
    sample_id: str
    backend: str
    success: bool
    workspace_id: str
    max_iterations: int
    timeout_seconds: float
    agent_exit_code: int
    timed_out: bool
    duration_ms: float
    validations: tuple[ValidationResult, ...]
    tool_trace: tuple[ToolTraceEntry, ...]
    changed_files: tuple[FileChange, ...]
    rejected_events: int = 0
    error_codes: tuple[str, ...] = ()
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
