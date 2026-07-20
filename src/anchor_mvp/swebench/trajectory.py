"""Local-only adapter from SWE task cards to five-stage distillation records.

The adapter validates already-controlled OpenCode session candidates. It does not
run an agent, call a model API, start a container, or claim that an audit digest is
trusted merely because it is well formed. The caller must provide the digest of a
trusted sandbox-audit bundle produced outside the model-controlled workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .schema import (
    MAX_PROBLEM_STATEMENT_CHARS,
    SWEBenchValidationError,
    TaskCard,
    canonical_json,
    digest_value,
)


TRAJECTORY_SCHEMA_VERSION = "anchor.swebench-five-stage-trajectory.v1"
WORKSPACE_INVENTORY_SCHEMA_VERSION = "anchor.swebench-workspace-inventory.v1"
PLANNER_OUTPUT_SCHEMA_VERSION = "anchor.swebench-planner-output.v1"
TOOL_POLICY_OUTPUT_SCHEMA_VERSION = "anchor.swebench-tool-policy-output.v1"
REVIEW_OUTPUT_SCHEMA_VERSION = "anchor.swebench-domain-review-output.v1"
SECURITY_OUTPUT_SCHEMA_VERSION = "anchor.swebench-security-output.v1"
SANDBOX_AUDIT_SCHEMA_VERSION = "anchor.swebench-sandbox-audit-bundle.v1"
OPENCODE_CANDIDATE_SCHEMA_VERSION = "anchor.session-training-candidate.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")
_CALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OPENCODE_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$")
_WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
_POSIX_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|var|tmp|opt|etc)(?:/|$)"
)
_ORACLE_TEXT_MARKER = re.compile(
    r"\b(?:FAIL_TO_PASS|PASS_TO_PASS|test_patch|hints_text|gold_patch|"
    r"gold_solution|oracle_text)\b",
    re.IGNORECASE,
)

_FORBIDDEN_NORMALIZED_KEYS = frozenset(
    {
        "patch",
        "patches",
        "testpatch",
        "hint",
        "hints",
        "hintstext",
        "testname",
        "testnames",
        "tests",
        "testcases",
        "failtopass",
        "passtopass",
        "gold",
        "goldpatch",
        "goldsolution",
        "oracle",
        "oracletext",
        "oraclefields",
        "reasoning",
        "thinking",
        "chainofthought",
        "environment",
        "env",
        "apikey",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "proxyauthorization",
        "password",
        "secretkey",
        "privatekey",
        "cookie",
        "systemprompt",
        "system",
        "systemmessage",
        "provider",
        "model",
        "modelid",
        "session",
        "sessionid",
        "messageid",
        "rawcallid",
    }
)
_FORBIDDEN_KEY_PREFIXES = (
    "patch",
    "test",
    "hint",
    "oracle",
    "gold",
    "failtopass",
    "passtopass",
)
_PATH_KEYS = frozenset(
    {
        "path",
        "file",
        "filepath",
        "file_path",
        "cwd",
        "directory",
        "root",
        "workspace",
    }
)
_SNAPSHOT_EXCLUSIONS = (".anchor", ".git", ".hg", ".svn")


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SWEBenchValidationError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise SWEBenchValidationError(f"{label} has unexpected fields")


def _safe_text(value: object, label: str, *, max_length: int = 100_000) -> str:
    if not isinstance(value, str):
        raise SWEBenchValidationError(f"{label} must be text")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or len(text) > max_length or "\x00" in text:
        raise SWEBenchValidationError(f"{label} is empty or outside its size limit")
    if _ORACLE_TEXT_MARKER.search(text):
        raise SWEBenchValidationError("oracle metadata marker entered trajectory text")
    return text


def _retained_text(
    value: object,
    label: str,
    *,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    """Validate controlled output while retaining its exact public bytes as text."""

    if (
        not isinstance(value, str)
        or (not allow_empty and not value.strip())
        or len(value) > max_length
        or "\x00" in value
    ):
        raise SWEBenchValidationError(f"{label} is empty or outside its size limit")
    if _ORACLE_TEXT_MARKER.search(value):
        raise SWEBenchValidationError("oracle metadata marker entered trajectory text")
    return value


def _sha256(value: object, label: str) -> str:
    candidate = str(value)
    if not _SHA256.fullmatch(candidate):
        raise SWEBenchValidationError(f"{label} must be one SHA-256 digest")
    return candidate


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SWEBenchValidationError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise SWEBenchValidationError(f"{label} is below its minimum")
    return value


def _reject_forbidden_material(
    value: object,
    *,
    allow_generated_diff_patch: bool = False,
    path: tuple[str, ...] = (),
) -> None:
    """Reject oracle/sensitive fields before selecting any retained subset.

    The controlled OpenCode v1 candidate calls its *generated* final diff
    ``final_diff[].patch``. That one schema-bound key is accepted only at that
    exact location and is projected to ``diff`` in the resulting SWE record.
    Upstream benchmark ``patch`` fields are rejected everywhere else.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = _normalized_key(key)
            generated_patch = (
                allow_generated_diff_patch
                and normalized == "patch"
                and len(path) == 2
                and path[0] == "final_diff"
                and path[1].isdigit()
            )
            forbidden_key = (
                normalized in _FORBIDDEN_NORMALIZED_KEYS
                or normalized.startswith(_FORBIDDEN_KEY_PREFIXES)
            )
            if forbidden_key and not generated_patch:
                raise SWEBenchValidationError(
                    "forbidden oracle, credential, environment, or reasoning field"
                )
            _reject_forbidden_material(
                child,
                allow_generated_diff_patch=allow_generated_diff_patch,
                path=(*path, key),
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_material(
                child,
                allow_generated_diff_patch=allow_generated_diff_patch,
                path=(*path, str(index)),
            )
    elif isinstance(value, str) and _ORACLE_TEXT_MARKER.search(value):
        raise SWEBenchValidationError("oracle metadata marker entered trajectory text")


def _workspace_path(value: object, label: str, *, allow_root: bool = False) -> str:
    if not isinstance(value, str):
        raise SWEBenchValidationError(f"{label} must be a workspace path")
    if value == "<workspace>":
        if allow_root:
            return value
        raise SWEBenchValidationError(f"{label} must name a file below the workspace")
    if not value.startswith("<workspace>/") or "\\" in value:
        raise SWEBenchValidationError(f"{label} is not workspace-bound")
    suffix = value.removeprefix("<workspace>/")
    pure = PurePosixPath(suffix)
    if (
        not suffix
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != suffix
    ):
        raise SWEBenchValidationError(f"{label} escapes or is not canonical")
    return value


def _validate_tool_paths(value: object, *, key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            normalized = str(child_key).casefold()
            if normalized in _PATH_KEYS:
                _workspace_path(child, f"tool input {child_key}", allow_root=True)
            _validate_tool_paths(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_tool_paths(child, key=key)
    elif isinstance(value, str):
        if "<workspace>/../" in value or "<workspace>\\.." in value:
            raise SWEBenchValidationError("tool input contains a workspace escape")
        if _WINDOWS_ABSOLUTE.search(value) or _POSIX_ABSOLUTE.search(value):
            raise SWEBenchValidationError(
                "tool input contains an external absolute path"
            )
        if "/workspace/" in value:
            raise SWEBenchValidationError(
                "container workspace paths must use the canonical sentinel"
            )


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    sha256: str
    byte_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkspaceFile":
        _exact_fields(value, {"path", "sha256", "byte_count"}, "workspace file")
        return cls(
            path=_workspace_path(value.get("path"), "workspace inventory path"),
            sha256=_sha256(value.get("sha256"), "workspace file sha256"),
            byte_count=_positive_int(
                value.get("byte_count"),
                "workspace file byte_count",
                allow_zero=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class WorkspaceInventory:
    files: tuple[WorkspaceFile, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkspaceInventory":
        _reject_forbidden_material(value)
        _exact_fields(
            value,
            {"schema_version", "workspace", "files"},
            "workspace inventory",
        )
        if value.get("schema_version") != WORKSPACE_INVENTORY_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported workspace-inventory schema")
        _workspace_path(
            value.get("workspace"), "workspace inventory root", allow_root=True
        )
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise SWEBenchValidationError("workspace inventory files must be a list")
        files = tuple(
            sorted(
                (
                    WorkspaceFile.from_mapping(
                        _mapping(item, "workspace inventory file")
                    )
                    for item in raw_files
                ),
                key=lambda item: item.path,
            )
        )
        if len({item.path for item in files}) != len(files):
            raise SWEBenchValidationError("workspace inventory repeats one path")
        return cls(files=files)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKSPACE_INVENTORY_SCHEMA_VERSION,
            "workspace": "<workspace>",
            "files": [item.to_dict() for item in self.files],
        }

    @property
    def binding_sha256(self) -> str:
        return digest_value(self.to_dict())


@dataclass(frozen=True)
class ToolProposal:
    proposal_id: str
    tool: str
    purpose: str
    input: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolProposal":
        _exact_fields(
            value,
            {"proposal_id", "tool", "purpose", "input"},
            "tool proposal",
        )
        proposal_id = str(value.get("proposal_id", ""))
        tool = str(value.get("tool", ""))
        if not _SAFE_ID.fullmatch(proposal_id) or not _SAFE_ID.fullmatch(tool):
            raise SWEBenchValidationError("tool proposal uses an unsafe id")
        purpose = _safe_text(
            value.get("purpose"), "tool proposal purpose", max_length=4000
        )
        tool_input = _mapping(value.get("input"), "tool proposal input")
        _validate_tool_paths(tool_input)
        return cls(
            proposal_id=proposal_id,
            tool=tool,
            purpose=purpose,
            input=dict(tool_input),
        )

    @property
    def signature(self) -> str:
        return digest_value({"tool": self.tool, "input": self.input})

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "tool": self.tool,
            "purpose": self.purpose,
            "input": dict(self.input),
        }


@dataclass(frozen=True)
class PlannerOutput:
    alignment_id: str
    domain_id: str
    builder_expert_id: str
    reviewer_expert_id: str
    work_items: tuple[str, ...]
    tool_proposals: tuple[ToolProposal, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], card: TaskCard) -> "PlannerOutput":
        _reject_forbidden_material(value)
        _exact_fields(
            value,
            {
                "schema_version",
                "alignment_id",
                "domain_id",
                "builder_expert_id",
                "reviewer_expert_id",
                "work_items",
                "tool_proposals",
            },
            "planner output",
        )
        if value.get("schema_version") != PLANNER_OUTPUT_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported planner-output schema")
        raw_work_items = value.get("work_items")
        if not isinstance(raw_work_items, list) or not raw_work_items:
            raise SWEBenchValidationError("planner work_items must be non-empty")
        work_items = tuple(
            _safe_text(item, "planner work item", max_length=8000)
            for item in raw_work_items
        )
        raw_proposals = value.get("tool_proposals")
        if not isinstance(raw_proposals, list) or not raw_proposals:
            raise SWEBenchValidationError("planner tool_proposals must be non-empty")
        proposals = tuple(
            ToolProposal.from_mapping(_mapping(item, "tool proposal"))
            for item in raw_proposals
        )
        if len({item.proposal_id for item in proposals}) != len(proposals):
            raise SWEBenchValidationError("planner repeats one proposal_id")
        if len({item.signature for item in proposals}) != len(proposals):
            raise SWEBenchValidationError(
                "planner emits ambiguous duplicate tool proposals"
            )
        output = cls(
            alignment_id=str(value.get("alignment_id", "")),
            domain_id=str(value.get("domain_id", "")),
            builder_expert_id=str(value.get("builder_expert_id", "")),
            reviewer_expert_id=str(value.get("reviewer_expert_id", "")),
            work_items=work_items,
            tool_proposals=proposals,
        )
        if (
            output.alignment_id != card.alignment_id
            or output.domain_id != card.domain_id
            or output.builder_expert_id != card.builder_expert_id
            or output.reviewer_expert_id != card.reviewer_expert_id
        ):
            raise SWEBenchValidationError(
                "planner route differs from the task-card routing contract"
            )
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PLANNER_OUTPUT_SCHEMA_VERSION,
            "alignment_id": self.alignment_id,
            "domain_id": self.domain_id,
            "builder_expert_id": self.builder_expert_id,
            "reviewer_expert_id": self.reviewer_expert_id,
            "work_items": list(self.work_items),
            "tool_proposals": [item.to_dict() for item in self.tool_proposals],
        }


@dataclass(frozen=True)
class ToolDecision:
    proposal_id: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ToolPolicyOutput:
    alignment_id: str
    executed_expert_id: str
    decisions: tuple[ToolDecision, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        card: TaskCard,
        planner: PlannerOutput,
        expected_expert_id: str,
    ) -> "ToolPolicyOutput":
        _reject_forbidden_material(value)
        _exact_fields(
            value,
            {
                "schema_version",
                "alignment_id",
                "executed_expert_id",
                "decisions",
            },
            "tool-policy output",
        )
        if value.get("schema_version") != TOOL_POLICY_OUTPUT_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported tool-policy-output schema")
        raw_decisions = value.get("decisions")
        if not isinstance(raw_decisions, list) or not raw_decisions:
            raise SWEBenchValidationError("tool-policy decisions must be non-empty")
        decisions: list[ToolDecision] = []
        for raw in raw_decisions:
            item = _mapping(raw, "tool-policy decision")
            _exact_fields(
                item,
                {"proposal_id", "decision", "reason"},
                "tool-policy decision",
            )
            proposal_id = str(item.get("proposal_id", ""))
            decision = str(item.get("decision", ""))
            if not _SAFE_ID.fullmatch(proposal_id) or decision not in {
                "APPROVE",
                "DENY",
            }:
                raise SWEBenchValidationError("tool-policy decision is invalid")
            decisions.append(
                ToolDecision(
                    proposal_id=proposal_id,
                    decision=decision,
                    reason=_safe_text(
                        item.get("reason"),
                        "tool-policy decision reason",
                        max_length=4000,
                    ),
                )
            )
        expected_ids = {item.proposal_id for item in planner.tool_proposals}
        actual_ids = [item.proposal_id for item in decisions]
        if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
            raise SWEBenchValidationError(
                "tool-policy must decide every planner proposal exactly once"
            )
        output = cls(
            alignment_id=str(value.get("alignment_id", "")),
            executed_expert_id=str(value.get("executed_expert_id", "")),
            decisions=tuple(decisions),
        )
        if output.alignment_id != card.alignment_id:
            raise SWEBenchValidationError(
                "tool-policy alignment_id differs from the card"
            )
        if output.executed_expert_id != expected_expert_id:
            raise SWEBenchValidationError("unexpected tool-policy expert executed")
        return output

    @property
    def approved_ids(self) -> frozenset[str]:
        return frozenset(
            item.proposal_id for item in self.decisions if item.decision == "APPROVE"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
            "alignment_id": self.alignment_id,
            "executed_expert_id": self.executed_expert_id,
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class SessionProjection:
    revision: int
    candidate_sha256: str
    source_sha256: str
    tool_trace: tuple[Mapping[str, Any], ...]
    approved_results: tuple[Mapping[str, Any], ...]
    generated_diff: tuple[Mapping[str, Any], ...]
    execution_summary: Mapping[str, Any]


def _proposal_by_signature(planner: PlannerOutput) -> Mapping[str, ToolProposal]:
    return {item.signature: item for item in planner.tool_proposals}


def _project_session_candidate(
    value: Mapping[str, Any],
    *,
    revision: int,
    card: TaskCard,
    planner: PlannerOutput,
    policy: ToolPolicyOutput,
) -> SessionProjection:
    _reject_forbidden_material(value, allow_generated_diff_patch=True)
    expected_fields = {
        "schema_version",
        "sample_id",
        "source",
        "skill_provenance",
        "trajectory",
        "final_diff",
        "validators",
        "public_outcome",
    }
    _exact_fields(value, expected_fields, "OpenCode session candidate")
    if value.get("schema_version") != OPENCODE_CANDIDATE_SCHEMA_VERSION:
        raise SWEBenchValidationError("unsupported OpenCode session-candidate schema")
    if not _SAMPLE_ID.fullmatch(str(value.get("sample_id", ""))):
        raise SWEBenchValidationError("OpenCode sample_id is invalid")

    source = _mapping(value.get("source"), "OpenCode source")
    _exact_fields(
        source,
        {"kind", "opencode_version", "source_sha256", "workspace", "tool_contract"},
        "OpenCode source",
    )
    if source.get("kind") != "controlled-opencode-export":
        raise SWEBenchValidationError("OpenCode source is not a controlled export")
    if not _OPENCODE_VERSION.fullmatch(str(source.get("opencode_version", ""))):
        raise SWEBenchValidationError("OpenCode version is invalid")
    source_sha256 = _sha256(source.get("source_sha256"), "OpenCode source sha256")
    _workspace_path(
        source.get("workspace"), "OpenCode source workspace", allow_root=True
    )
    tool_contract = _mapping(source.get("tool_contract"), "OpenCode tool contract")
    version = tool_contract.get("version")
    if not isinstance(version, str) or not version.startswith(
        "anchor.execution-tool-contract."
    ):
        raise SWEBenchValidationError("OpenCode tool contract is unsupported")

    raw_trajectory = value.get("trajectory")
    if not isinstance(raw_trajectory, list):
        raise SWEBenchValidationError("OpenCode trajectory must be a list")
    proposals_by_signature = _proposal_by_signature(planner)
    calls: dict[str, tuple[str, str]] = {}
    results: set[str] = set()
    projected_trace: list[Mapping[str, Any]] = []
    approved_results: list[Mapping[str, Any]] = []
    previous_sequence = 0
    pending_call: tuple[str, str, str] | None = None
    for raw_event in raw_trajectory:
        event = _mapping(raw_event, "OpenCode trajectory event")
        sequence = _positive_int(event.get("sequence"), "trajectory sequence")
        if sequence <= previous_sequence:
            raise SWEBenchValidationError(
                "OpenCode trajectory sequence is not increasing"
            )
        previous_sequence = sequence
        event_type = event.get("type")
        if event_type in {"user_input", "assistant_output"}:
            _exact_fields(
                event,
                {"type", "sequence", "content"},
                "OpenCode text event",
            )
            _safe_text(event.get("content"), "OpenCode public text")
            if pending_call is not None:
                raise SWEBenchValidationError("tool result is not adjacent to its call")
            continue
        if event_type == "tool_call":
            _exact_fields(
                event,
                {"type", "sequence", "call_id", "tool", "input"},
                "OpenCode tool call",
            )
            if pending_call is not None:
                raise SWEBenchValidationError("tool calls and results are not adjacent")
            call_id = str(event.get("call_id", ""))
            tool = str(event.get("tool", ""))
            tool_input = _mapping(event.get("input"), "OpenCode tool input")
            if (
                not _CALL_ID.fullmatch(call_id)
                or call_id in calls
                or not _SAFE_ID.fullmatch(tool)
            ):
                raise SWEBenchValidationError(
                    "OpenCode tool call is invalid or duplicated"
                )
            _validate_tool_paths(tool_input)
            signature = digest_value({"tool": tool, "input": tool_input})
            proposal = proposals_by_signature.get(signature)
            if proposal is None or proposal.proposal_id not in policy.approved_ids:
                raise SWEBenchValidationError(
                    "OpenCode tool call was not explicitly approved"
                )
            calls[call_id] = (tool, proposal.proposal_id)
            pending_call = (call_id, tool, proposal.proposal_id)
            projected_trace.append(
                {
                    "type": "tool_call",
                    "sequence": sequence,
                    "call_id": call_id,
                    "proposal_id": proposal.proposal_id,
                    "tool": tool,
                    "input": dict(tool_input),
                }
            )
            continue
        if event_type == "tool_result":
            _exact_fields(
                event,
                {"type", "sequence", "call_id", "tool", "status", "content"},
                "OpenCode tool result",
            )
            call_id = str(event.get("call_id", ""))
            tool = str(event.get("tool", ""))
            status = str(event.get("status", ""))
            if (
                pending_call is None
                or pending_call[:2] != (call_id, tool)
                or call_id in results
                or status not in {"completed", "error", "rejected"}
                or sequence != projected_trace[-1]["sequence"] + 1
            ):
                raise SWEBenchValidationError("tool call/result pairing is invalid")
            content = _retained_text(
                event.get("content"),
                "OpenCode tool result content",
                max_length=200_000,
                allow_empty=True,
            )
            proposal_id = pending_call[2]
            result = {
                "type": "tool_result",
                "sequence": sequence,
                "call_id": call_id,
                "proposal_id": proposal_id,
                "tool": tool,
                "status": status,
                "content": content,
            }
            projected_trace.append(result)
            approved_results.append(
                {
                    "call_id": call_id,
                    "proposal_id": proposal_id,
                    "tool": tool,
                    "status": status,
                    "content": content,
                }
            )
            results.add(call_id)
            pending_call = None
            continue
        raise SWEBenchValidationError("unsupported OpenCode trajectory event type")
    if pending_call is not None or not calls or set(calls) != results:
        raise SWEBenchValidationError("tool call/result sets do not match")

    raw_diffs = value.get("final_diff")
    if not isinstance(raw_diffs, list) or not raw_diffs:
        raise SWEBenchValidationError("OpenCode final_diff must be non-empty")
    generated_diff: list[Mapping[str, Any]] = []
    for raw_diff in raw_diffs:
        diff = _mapping(raw_diff, "OpenCode generated diff")
        _exact_fields(
            diff,
            {"file", "patch", "additions", "deletions", "status"},
            "OpenCode generated diff",
        )
        additions = _positive_int(
            diff.get("additions"), "diff additions", allow_zero=True
        )
        deletions = _positive_int(
            diff.get("deletions"), "diff deletions", allow_zero=True
        )
        status = str(diff.get("status", ""))
        if status not in {"added", "deleted", "modified"}:
            raise SWEBenchValidationError("generated diff status is invalid")
        generated_diff.append(
            {
                "path": _workspace_path(diff.get("file"), "generated diff file"),
                "diff": _retained_text(
                    diff.get("patch"), "generated diff", max_length=300_000
                ),
                "additions": additions,
                "deletions": deletions,
                "status": status,
            }
        )
    if len({str(item["path"]) for item in generated_diff}) != len(generated_diff):
        raise SWEBenchValidationError("generated diff repeats one workspace path")

    raw_validators = value.get("validators")
    if not isinstance(raw_validators, list):
        raise SWEBenchValidationError("OpenCode validators must be a list")
    passed = 0
    failed = 0
    validator_names: set[str] = set()
    for raw_validator in raw_validators:
        validator = _mapping(raw_validator, "OpenCode validator")
        _exact_fields(
            validator,
            {"name", "status", "exit_code", "command", "stdout", "stderr"},
            "OpenCode validator",
        )
        validator_name = _safe_text(
            validator.get("name"), "validator label", max_length=80
        )
        if validator_name in validator_names:
            raise SWEBenchValidationError("OpenCode validators repeat one name")
        validator_names.add(validator_name)
        _safe_text(validator.get("command"), "validator command", max_length=4000)
        for output_key in ("stdout", "stderr"):
            output = validator.get(output_key)
            if not isinstance(output, str) or len(output) > 200_000:
                raise SWEBenchValidationError("validator output is invalid")
        exit_code = validator.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise SWEBenchValidationError("validator exit_code must be an integer")
        if validator.get("status") == "PASS" and exit_code == 0:
            passed += 1
        else:
            failed += 1

    provenance = value.get("skill_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise SWEBenchValidationError("OpenCode skill provenance must be non-empty")
    provenance_fields = {
        "source_id",
        "repository",
        "commit",
        "license",
        "license_sha256",
        "bundle_sha256",
        "instruction_audit_sha256",
    }
    for raw_item in provenance:
        item = _mapping(raw_item, "OpenCode skill provenance")
        _exact_fields(item, provenance_fields, "OpenCode skill provenance")
        _safe_text(item.get("source_id"), "skill source_id", max_length=200)
        repository = _safe_text(
            item.get("repository"), "skill repository", max_length=2000
        )
        if not repository.startswith("https://"):
            raise SWEBenchValidationError("skill repository must use https")
        if not re.fullmatch(r"[0-9a-f]{40}", str(item.get("commit", ""))):
            raise SWEBenchValidationError("skill commit must be immutable")
        _safe_text(item.get("license"), "skill license", max_length=200)
        for key in (
            "license_sha256",
            "bundle_sha256",
            "instruction_audit_sha256",
        ):
            _sha256(item.get(key), f"skill {key}")

    public_outcome = _mapping(value.get("public_outcome"), "OpenCode public outcome")
    _exact_fields(
        public_outcome,
        {
            "schema_version",
            "status",
            "decision_trace",
            "repair_summaries",
            "final_summary",
        },
        "OpenCode public outcome",
    )
    if public_outcome.get("schema_version") != "anchor.public-outcome.v1":
        raise SWEBenchValidationError("OpenCode public outcome schema is unsupported")
    if not isinstance(public_outcome.get("decision_trace"), list) or not isinstance(
        public_outcome.get("repair_summaries"), list
    ):
        raise SWEBenchValidationError("OpenCode public outcome lists are invalid")
    _safe_text(
        public_outcome.get("final_summary"),
        "OpenCode final summary",
        max_length=20_000,
    )
    outcome_status = str(public_outcome.get("status", ""))
    if outcome_status not in {"completed", "partial", "blocked"}:
        raise SWEBenchValidationError("OpenCode public outcome status is invalid")
    summary = {
        "status": outcome_status,
        "exit_code": 0 if failed == 0 and outcome_status == "completed" else 1,
        "tool_call_count": len(calls),
        "checks_run": passed + failed,
        "checks_passed": passed,
        "checks_failed": failed,
    }
    return SessionProjection(
        revision=revision,
        candidate_sha256=digest_value(value),
        source_sha256=source_sha256,
        tool_trace=tuple(projected_trace),
        approved_results=tuple(approved_results),
        generated_diff=tuple(generated_diff),
        execution_summary=summary,
    )


def _validate_review_output(
    value: Mapping[str, Any],
    *,
    card: TaskCard,
    revision: int,
) -> Mapping[str, Any]:
    _reject_forbidden_material(value)
    _exact_fields(
        value,
        {
            "schema_version",
            "alignment_id",
            "revision",
            "executed_expert_id",
            "decision",
            "feedback",
        },
        "domain-review output",
    )
    if value.get("schema_version") != REVIEW_OUTPUT_SCHEMA_VERSION:
        raise SWEBenchValidationError("unsupported domain-review-output schema")
    if (
        value.get("alignment_id") != card.alignment_id
        or value.get("revision") != revision
        or value.get("executed_expert_id") != card.reviewer_expert_id
    ):
        raise SWEBenchValidationError(
            "domain-review alignment, revision, or executed expert differs"
        )
    decision = str(value.get("decision", ""))
    if decision not in {"PASS", "REVISE"}:
        raise SWEBenchValidationError("domain-review decision is invalid")
    raw_feedback = value.get("feedback")
    if not isinstance(raw_feedback, list):
        raise SWEBenchValidationError("domain-review feedback must be a list")
    feedback = [
        _safe_text(item, "domain-review feedback", max_length=8000)
        for item in raw_feedback
    ]
    if decision == "REVISE" and not feedback:
        raise SWEBenchValidationError("REVISE requires public feedback")
    return {
        "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "revision": revision,
        "executed_expert_id": card.reviewer_expert_id,
        "decision": decision,
        "feedback": feedback,
    }


def _validate_security_output(
    value: Mapping[str, Any],
    *,
    card: TaskCard,
    expected_expert_id: str,
) -> Mapping[str, Any]:
    _reject_forbidden_material(value)
    _exact_fields(
        value,
        {
            "schema_version",
            "alignment_id",
            "executed_expert_id",
            "decision",
            "findings",
        },
        "security output",
    )
    if value.get("schema_version") != SECURITY_OUTPUT_SCHEMA_VERSION:
        raise SWEBenchValidationError("unsupported security-output schema")
    if (
        value.get("alignment_id") != card.alignment_id
        or value.get("executed_expert_id") != expected_expert_id
    ):
        raise SWEBenchValidationError(
            "security alignment or executed expert differs from the contract"
        )
    decision = str(value.get("decision", ""))
    if decision not in {"PASS", "BLOCK"}:
        raise SWEBenchValidationError("security decision is invalid")
    raw_findings = value.get("findings")
    if not isinstance(raw_findings, list):
        raise SWEBenchValidationError("security findings must be a list")
    findings = [
        _safe_text(item, "security finding", max_length=8000) for item in raw_findings
    ]
    if decision == "BLOCK" and not findings:
        raise SWEBenchValidationError("BLOCK requires at least one public finding")
    return {
        "schema_version": SECURITY_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "executed_expert_id": expected_expert_id,
        "decision": decision,
        "findings": findings,
    }


def _validate_sandbox_audit(
    value: Mapping[str, Any],
    *,
    expected_digest: str,
    card: TaskCard,
    inventory: WorkspaceInventory,
    sessions: Sequence[SessionProjection],
) -> str:
    _reject_forbidden_material(value)
    _exact_fields(
        value,
        {
            "schema_version",
            "alignment_id",
            "source_fingerprint",
            "workspace",
            "workspace_binding_sha256",
            "snapshot_exclusions",
            "protected_state_before_sha256",
            "protected_state_after_sha256",
            "cleanup_status",
            "sessions",
        },
        "sandbox audit bundle",
    )
    if value.get("schema_version") != SANDBOX_AUDIT_SCHEMA_VERSION:
        raise SWEBenchValidationError("unsupported sandbox-audit schema")
    if value.get("alignment_id") != card.alignment_id:
        raise SWEBenchValidationError(
            "sandbox audit alignment_id differs from the card"
        )
    if value.get("source_fingerprint") != card.source_fingerprint:
        raise SWEBenchValidationError(
            "sandbox audit source_fingerprint differs from the card"
        )
    _workspace_path(value.get("workspace"), "sandbox audit workspace", allow_root=True)
    if value.get("workspace_binding_sha256") != inventory.binding_sha256:
        raise SWEBenchValidationError("sandbox audit binds a different base workspace")
    exclusions = value.get("snapshot_exclusions")
    if exclusions != list(_SNAPSHOT_EXCLUSIONS):
        raise SWEBenchValidationError(
            "sandbox audit snapshot exclusions are incomplete"
        )
    before = _sha256(
        value.get("protected_state_before_sha256"),
        "protected state before sha256",
    )
    after = _sha256(
        value.get("protected_state_after_sha256"),
        "protected state after sha256",
    )
    if before != after:
        raise SWEBenchValidationError(
            "protected sandbox state changed during execution"
        )
    if value.get("cleanup_status") != "cleaned":
        raise SWEBenchValidationError("sandbox cleanup is not attested as complete")
    raw_sessions = value.get("sessions")
    if not isinstance(raw_sessions, list) or len(raw_sessions) != len(sessions):
        raise SWEBenchValidationError(
            "sandbox audit session count differs from revisions"
        )
    for projection, raw in zip(sessions, raw_sessions):
        item = _mapping(raw, "sandbox audit session")
        _exact_fields(
            item,
            {
                "revision",
                "executed_expert_id",
                "candidate_sha256",
                "tool_trace_sha256",
                "generated_diff_sha256",
                "execution_summary_sha256",
            },
            "sandbox audit session",
        )
        if (
            item.get("revision") != projection.revision
            or item.get("executed_expert_id") != card.builder_expert_id
            or item.get("candidate_sha256") != projection.candidate_sha256
            or item.get("tool_trace_sha256")
            != digest_value(list(projection.tool_trace))
            or item.get("generated_diff_sha256")
            != digest_value(list(projection.generated_diff))
            or item.get("execution_summary_sha256")
            != digest_value(projection.execution_summary)
        ):
            raise SWEBenchValidationError(
                "sandbox audit session route or hashes do not match the validated export"
            )
        for key in (
            "candidate_sha256",
            "tool_trace_sha256",
            "generated_diff_sha256",
            "execution_summary_sha256",
        ):
            _sha256(item.get(key), f"sandbox audit {key}")
    digest = digest_value(value)
    if _sha256(expected_digest, "trusted sandbox audit sha256") != digest:
        raise SWEBenchValidationError("trusted sandbox audit digest does not match")
    return digest


def adapt_task_card_trajectory(
    *,
    card: TaskCard,
    workspace_inventory: Mapping[str, Any],
    planner_output: Mapping[str, Any],
    tool_policy_output: Mapping[str, Any],
    opencode_session_exports: Sequence[Mapping[str, Any]],
    review_outputs: Sequence[Mapping[str, Any]],
    security_output: Mapping[str, Any],
    sandbox_audit_bundle: Mapping[str, Any],
    trusted_sandbox_audit_sha256: str,
    tool_policy_expert_id: str = "tool-policy",
    security_expert_id: str = "security-audit",
) -> dict[str, Any]:
    """Validate and project one task card into one complete five-stage chain."""

    if not _SAFE_ID.fullmatch(tool_policy_expert_id) or not _SAFE_ID.fullmatch(
        security_expert_id
    ):
        raise SWEBenchValidationError("stage expert id is unsafe")
    _safe_text(
        card.problem_statement,
        "task-card problem_statement",
        max_length=MAX_PROBLEM_STATEMENT_CHARS,
    )
    inventory = WorkspaceInventory.from_mapping(workspace_inventory)
    planner = PlannerOutput.from_mapping(planner_output, card)
    policy = ToolPolicyOutput.from_mapping(
        tool_policy_output,
        card=card,
        planner=planner,
        expected_expert_id=tool_policy_expert_id,
    )
    if not policy.approved_ids:
        raise SWEBenchValidationError("tool-policy approved no executable proposal")
    if not opencode_session_exports:
        raise SWEBenchValidationError("one or more OpenCode revisions are required")
    sessions = tuple(
        _project_session_candidate(
            _mapping(value, "OpenCode session candidate"),
            revision=index,
            card=card,
            planner=planner,
            policy=policy,
        )
        for index, value in enumerate(opencode_session_exports, 1)
    )
    if len(review_outputs) != len(sessions):
        raise SWEBenchValidationError("every builder revision needs one domain review")
    reviews = tuple(
        _validate_review_output(
            _mapping(value, "domain-review output"),
            card=card,
            revision=index,
        )
        for index, value in enumerate(review_outputs, 1)
    )
    if any(item["decision"] != "REVISE" for item in reviews[:-1]):
        raise SWEBenchValidationError("non-final revisions must request REVISE")
    if reviews[-1]["decision"] != "PASS":
        raise SWEBenchValidationError("final domain review must PASS before security")
    security = _validate_security_output(
        security_output,
        card=card,
        expected_expert_id=security_expert_id,
    )
    audit_sha256 = _validate_sandbox_audit(
        sandbox_audit_bundle,
        expected_digest=trusted_sandbox_audit_sha256,
        card=card,
        inventory=inventory,
        sessions=sessions,
    )

    builder_revisions: list[dict[str, Any]] = []
    review_revisions: list[dict[str, Any]] = []
    for session, review in zip(sessions, reviews):
        public_review_input = {
            "problem_statement": card.problem_statement,
            "diff": list(session.generated_diff),
            "execution_summary": dict(session.execution_summary),
        }
        builder_revisions.append(
            {
                "alignment_id": card.alignment_id,
                "revision": session.revision,
                "executed_expert_id": card.builder_expert_id,
                "input": {
                    "problem_statement": card.problem_statement,
                    "base_workspace_inventory": inventory.to_dict(),
                    "approved_tool_results": list(session.approved_results),
                },
                "execution_trace": list(session.tool_trace),
                "execution_evidence": {
                    "candidate_sha256": session.candidate_sha256,
                    "source_sha256": session.source_sha256,
                    "sandbox_audit_sha256": audit_sha256,
                },
                "output": {
                    "diff": list(session.generated_diff),
                    "execution_summary": dict(session.execution_summary),
                },
            }
        )
        review_revisions.append(
            {
                "alignment_id": card.alignment_id,
                "revision": session.revision,
                "executed_expert_id": card.reviewer_expert_id,
                "input": public_review_input,
                "output": dict(review),
            }
        )

    final_session = sessions[-1]
    public_final_input = {
        "problem_statement": card.problem_statement,
        "diff": list(final_session.generated_diff),
        "execution_summary": dict(final_session.execution_summary),
    }
    result: dict[str, Any] = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "card_id": card.card_id,
        "alignment_id": card.alignment_id,
        "source_fingerprint": card.source_fingerprint,
        "sandbox_audit_sha256": audit_sha256,
        "routing_contract": {
            "domain_id": card.domain_id,
            "builder_expert_id": card.builder_expert_id,
            "reviewer_expert_id": card.reviewer_expert_id,
            "planner_selection_matches_execution": True,
        },
        "stages": [
            {
                "stage": "planner",
                "input": {
                    "problem_statement": card.problem_statement,
                    "base_workspace_inventory": inventory.to_dict(),
                },
                "output": planner.to_dict(),
            },
            {
                "stage": "tool_policy",
                "executed_expert_id": tool_policy_expert_id,
                "input": {
                    "problem_statement": card.problem_statement,
                    "tool_proposals": [
                        item.to_dict() for item in planner.tool_proposals
                    ],
                },
                "output": policy.to_dict(),
            },
            {
                "stage": "domain_builder",
                "executed_expert_id": card.builder_expert_id,
                "revisions": builder_revisions,
            },
            {
                "stage": "domain_review",
                "executed_expert_id": card.reviewer_expert_id,
                "revisions": review_revisions,
            },
            {
                "stage": "security",
                "executed_expert_id": security_expert_id,
                "input": public_final_input,
                "output": dict(security),
            },
        ],
        "counts": {
            "task_card_count": 1,
            "alignment_count": 1,
            "complete_chain_count": 1,
            "revision_count": len(sessions),
        },
    }
    _reject_forbidden_material(result)
    # Canonical round-trip catches values that are not JSON serializable before
    # a training writer receives the record.
    json.loads(canonical_json(result))
    return result


def project_task_card_builder_sessions(
    *,
    card: TaskCard,
    workspace_inventory: Mapping[str, Any],
    planner_output: Mapping[str, Any],
    tool_policy_output: Mapping[str, Any],
    opencode_session_exports: Sequence[Mapping[str, Any]],
    tool_policy_expert_id: str = "tool-policy",
) -> tuple[dict[str, Any], ...]:
    """Validate controlled sessions and expose only review-safe builder results.

    This is the pre-review half of :func:`adapt_task_card_trajectory`.  It does
    not claim sandbox-audit trust or produce a complete chain; the final adapter
    still recomputes and verifies that proof after review and security outputs
    exist.
    """

    if not _SAFE_ID.fullmatch(tool_policy_expert_id):
        raise SWEBenchValidationError("stage expert id is unsafe")
    inventory = WorkspaceInventory.from_mapping(workspace_inventory)
    planner = PlannerOutput.from_mapping(planner_output, card)
    policy = ToolPolicyOutput.from_mapping(
        tool_policy_output,
        card=card,
        planner=planner,
        expected_expert_id=tool_policy_expert_id,
    )
    if not policy.approved_ids:
        raise SWEBenchValidationError("tool-policy approved no executable proposal")
    if not opencode_session_exports:
        raise SWEBenchValidationError("one or more OpenCode revisions are required")
    sessions = tuple(
        _project_session_candidate(
            _mapping(value, "OpenCode session candidate"),
            revision=index,
            card=card,
            planner=planner,
            policy=policy,
        )
        for index, value in enumerate(opencode_session_exports, 1)
    )
    result = tuple(
        {
            "alignment_id": card.alignment_id,
            "source_fingerprint": card.source_fingerprint,
            "revision": session.revision,
            "executed_expert_id": card.builder_expert_id,
            "input": {
                "problem_statement": card.problem_statement,
                "base_workspace_inventory": inventory.to_dict(),
                "approved_tool_results": list(session.approved_results),
            },
            "execution_trace": list(session.tool_trace),
            "execution_evidence": {
                "candidate_sha256": session.candidate_sha256,
                "source_sha256": session.source_sha256,
                "sandbox_audit_pending": True,
            },
            "output": {
                "diff": list(session.generated_diff),
                "execution_summary": dict(session.execution_summary),
            },
        }
        for session in sessions
    )
    _reject_forbidden_material(result)
    json.loads(canonical_json(result))
    return result


__all__ = [
    "OPENCODE_CANDIDATE_SCHEMA_VERSION",
    "PLANNER_OUTPUT_SCHEMA_VERSION",
    "REVIEW_OUTPUT_SCHEMA_VERSION",
    "SANDBOX_AUDIT_SCHEMA_VERSION",
    "SECURITY_OUTPUT_SCHEMA_VERSION",
    "TOOL_POLICY_OUTPUT_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "WORKSPACE_INVENTORY_SCHEMA_VERSION",
    "WorkspaceInventory",
    "adapt_task_card_trajectory",
    "project_task_card_builder_sessions",
]
