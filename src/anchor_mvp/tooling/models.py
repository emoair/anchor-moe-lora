from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Literal


SCHEMA_VERSION = "anchor.tool-gold.v1"
ValidationStatus = Literal["PASS", "FAIL", "SKIP", "TIMEOUT"]
PublicOutcomeStatus = Literal["completed", "blocked", "partial"]


@dataclass(frozen=True)
class SkillProvenance:
    source_id: str
    repository: str
    commit: str
    license: str
    license_sha256: str
    bundle_sha256: str
    instruction_audit_sha256: str


@dataclass(frozen=True)
class PublicDecisionStep:
    check: str
    evidence: str
    action: str


@dataclass(frozen=True)
class PublicOutcome:
    status: PublicOutcomeStatus
    decision_trace: tuple[PublicDecisionStep, ...]
    repair_summaries: tuple[str, ...]
    final_summary: str
    schema_version: str = "anchor.public-outcome.v1"


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    prompt: str
    source_dir: Path
    required_validations: tuple[str, ...] = ("build",)
    skill_provenance: tuple[SkillProvenance, ...] = ()
    protected_files: tuple[tuple[str, str], ...] = ()
    input_files: tuple[tuple[str, str], ...] = ()
    requires_changes: bool = False


def sample_contract_sha256(sample: SampleSpec) -> str:
    """Bind the public task to its immutable local acceptance files."""

    if not sample.protected_files and not sample.input_files:
        return hashlib.sha256(sample.prompt.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "prompt": sample.prompt,
            "input_files": list(sample.input_files),
            "protected_files": list(sample.protected_files),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    public_outcome: PublicOutcome | None = None
    controlled_session_id: str | None = None
    controlled_export_path: str | None = None
    isolated_runtime_path: str | None = None
    opencode_version: str | None = None


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
    task_bundle_sha256: str
    agent_stdout_sha256: str | None
    agent_stderr_sha256: str | None
    skill_provenance: tuple[SkillProvenance, ...] = ()
    public_outcome: PublicOutcome | None = None
    rejected_events: int = 0
    error_codes: tuple[str, ...] = ()
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
