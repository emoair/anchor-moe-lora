"""Fail-closed conversion of controlled OpenCode exports into training candidates."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from ..benchmark.heldout import verify_heldout_manifest
from .tool_contract import (
    ALLOWED_NPM_COMMANDS,
    ALLOWED_VALIDATOR_COMMANDS,
    EXECUTION_TOOLS,
    PATH_REQUIRED_TOOLS,
    SEARCH_TOOLS,
    contract_descriptor,
    normalized_command,
    validate_search_input,
)


CANDIDATE_SCHEMA_VERSION = "anchor.session-training-candidate.v1"
STAGING_SCHEMA_VERSION = "anchor.session-candidate-staging.v1"
CAPTURE_SCHEMA_VERSION = "anchor.controlled-session-capture.v1"
QUARANTINE_SCHEMA_VERSION = "anchor.session-quarantine.v1"
ALLOWED_TOOLS = EXECUTION_TOOLS
ALLOWED_VALIDATORS = frozenset({"build", "test", "lint"})
ALLOWED_BASH_COMMANDS = ALLOWED_NPM_COMMANDS
PATH_KEYS = frozenset({"path", "file", "filepath", "file_path", "cwd", "directory"})
FORBIDDEN_KEYS = frozenset(
    {"env", "environment", "reasoning", "thinking", "chain_of_thought", "chain-of-thought"}
)
SECRET_FIELD_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "authorization",
        "password",
        "secret_key",
        "private_key",
    }
)
CAPTURE_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "sample_id",
        "session_id",
        "opencode_version",
        "validators",
        "public_outcome",
        "final_diff",
        "skill_provenance",
        "quality",
    }
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        re.IGNORECASE,
    ),
)
HIDDEN_REASONING_MARKER = re.compile(
    r"<\/?thinking>|<\/?reasoning>|chain[- ]of[- ]thought|[\"'](?:reasoning|thinking)[\"']\s*:",
    re.IGNORECASE,
)
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\r\n\t\"'<>|]*)")
POSIX_ABSOLUTE = re.compile(r"(?m)(?<![A-Za-z0-9_.-])(/(?:home|Users|var|tmp|opt|etc)/[^\r\n\t\"'<>|]*)")
CONTAINER_WORKSPACE = re.compile(r"(?<![A-Za-z0-9_.-])/workspace(?=$|[\\/])")
WORKSPACE_REFERENCE = re.compile(r"<workspace>(?:[\\/][^\r\n\t\"'<>|]*)?")


class QuarantineError(ValueError):
    """The entire capture is unsafe and must not be partially redacted into training."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SessionConversionPolicy:
    workspace_root: Path
    heldout_cases: Path
    heldout_fixtures_root: Path | None = None
    heldout_manifest: Path | None = None
    max_text_bytes: int = 131_072
    max_record_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not self.workspace_root.is_absolute():
            raise ValueError("workspace_root must be absolute")
        if self.max_text_bytes < 1 or self.max_record_bytes < self.max_text_bytes:
            raise ValueError("invalid session conversion size limits")
        if (self.heldout_fixtures_root is None) != (self.heldout_manifest is None):
            raise ValueError("held-out fixture root and manifest must be configured together")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _similarity(left: str, right: str) -> float:
    a = _normalized_text(left)
    b = _normalized_text(right)
    if not a or not b:
        return 0.0
    if a == b or (len(a) >= 24 and a in b) or (len(b) >= 24 and b in a):
        return 1.0
    ratio = len(a) / len(b)
    if ratio < 0.4 or ratio > 2.5:
        return 0.0
    a_tokens = a.split()
    b_tokens = b.split()
    width = 3 if min(len(a_tokens), len(b_tokens)) >= 5 else 1
    a_shingles = {tuple(a_tokens[index : index + width]) for index in range(len(a_tokens) - width + 1)}
    b_shingles = {tuple(b_tokens[index : index + width]) for index in range(len(b_tokens) - width + 1)}
    union = a_shingles | b_shingles
    jaccard = len(a_shingles & b_shingles) / len(union) if union else 0.0
    return max(jaccard, SequenceMatcher(None, a, b, autojunk=False).ratio())


def _heldout_needles(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    identifiers: set[str] = set()
    requirements: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("held-out case must be an object")
        for key in ("case_id", "seed_id", "case_family", "namespace", "seed_namespace"):
            if value.get(key):
                identifiers.add(str(value[key]).casefold())
        for text in (
            value.get("requirement"),
            value.get("security_intent_label"),
            value.get("review_mutation", {}).get("marker")
            if isinstance(value.get("review_mutation"), Mapping)
            else None,
        ):
            if text:
                requirements.add(_normalized_text(str(text)))
        for key in ("plan_required_concepts", "tool_proposal_labels"):
            items = value.get(key)
            if isinstance(items, list):
                requirements.update(_normalized_text(str(item)) for item in items if str(item).strip())
    return frozenset(identifiers), frozenset(requirements)


class _SafetyGate:
    def __init__(self, policy: SessionConversionPolicy) -> None:
        self.policy = policy
        self.workspace = policy.workspace_root.resolve()
        if policy.heldout_fixtures_root is not None and policy.heldout_manifest is not None:
            verify_heldout_manifest(
                policy.heldout_cases,
                policy.heldout_fixtures_root,
                policy.heldout_manifest,
            )
        self.workspace_windows = str(self.workspace).replace("/", "\\")
        self.workspace_posix = self.workspace.as_posix()
        self.identifiers, self.requirements = _heldout_needles(policy.heldout_cases)

    def text(
        self, value: object, *, label: str, check_hidden_reasoning: bool = True
    ) -> str:
        if not isinstance(value, str):
            raise QuarantineError(f"{label}_not_text")
        if "\x00" in value or any(ord(char) < 9 for char in value):
            raise QuarantineError("binary_or_control_text")
        if len(value.encode("utf-8")) > self.policy.max_text_bytes:
            raise QuarantineError("text_size_limit")
        if "[redacted:" in value:
            raise QuarantineError("official_sanitize_is_lossy")
        normalized = self._normalize_workspace_paths(value)
        if any(pattern.search(normalized) for pattern in SECRET_PATTERNS):
            raise QuarantineError("secret_detected")
        if check_hidden_reasoning and HIDDEN_REASONING_MARKER.search(normalized):
            raise QuarantineError("hidden_reasoning_in_public_text")
        folded = normalized.casefold()
        compact = _normalized_text(normalized)
        if any(identifier in folded for identifier in self.identifiers):
            raise QuarantineError("heldout_leakage")
        if any(_similarity(requirement, compact) >= 0.86 for requirement in self.requirements):
            raise QuarantineError("heldout_leakage")
        return normalized

    def raw_sensitive_scan(self, value: object) -> None:
        """Scan even dropped fields so a sensitive hit quarantines the whole capture."""

        if isinstance(value, str):
            self.text(value, label="raw_export", check_hidden_reasoning=False)
            return
        if value is None or isinstance(value, (bool, int, float)):
            return
        if isinstance(value, list):
            for item in value:
                self.raw_sensitive_scan(item)
            return
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).casefold()
                if key in {"env", "environment"}:
                    raise QuarantineError("forbidden_environment_field")
                if key in SECRET_FIELD_KEYS:
                    raise QuarantineError("secret_field_detected")
                self.raw_sensitive_scan(item)
            return
        raise QuarantineError("raw_export_unsupported_value")

    def value(
        self,
        value: object,
        *,
        label: str,
        check_hidden_reasoning: bool = True,
    ) -> object:
        if isinstance(value, str):
            return self.text(
                value,
                label=label,
                check_hidden_reasoning=check_hidden_reasoning,
            )
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, list):
            return [
                self.value(
                    item,
                    label=label,
                    check_hidden_reasoning=check_hidden_reasoning,
                )
                for item in value
            ]
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                if key.casefold() in FORBIDDEN_KEYS:
                    raise QuarantineError("forbidden_environment_or_reasoning_field")
                if key.casefold() in PATH_KEYS:
                    result[key] = self.path(item, label=label)
                else:
                    result[key] = self.value(
                        item,
                        label=label,
                        check_hidden_reasoning=check_hidden_reasoning,
                    )
            return result
        raise QuarantineError(f"{label}_unsupported_value")

    def path(self, value: object, *, label: str) -> str:
        text = self.text(value, label=label)
        if text.startswith("<workspace>"):
            if not WORKSPACE_REFERENCE.fullmatch(text):
                raise QuarantineError("workspace_reference_invalid")
            return self._normalize_workspace_reference(text)
        if text.startswith("/"):
            raise QuarantineError("absolute_path_outside_workspace")
        candidate = Path(text)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.workspace / candidate).resolve()
        try:
            relative = resolved.relative_to(self.workspace).as_posix()
        except ValueError as error:
            raise QuarantineError("workspace_escape") from error
        return "<workspace>" if relative == "." else f"<workspace>/{relative}"

    def _normalize_workspace_paths(self, value: str) -> str:
        normalized = re.sub(
            re.escape(self.workspace_windows),
            "<workspace>",
            value,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            re.escape(self.workspace_posix),
            "<workspace>",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = CONTAINER_WORKSPACE.sub("<workspace>", normalized)
        if WINDOWS_ABSOLUTE.search(normalized) or POSIX_ABSOLUTE.search(normalized):
            raise QuarantineError("absolute_path_outside_workspace")
        return WORKSPACE_REFERENCE.sub(
            lambda match: self._normalize_workspace_reference(match.group(0)), normalized
        )

    @staticmethod
    def _normalize_workspace_reference(value: str) -> str:
        """Canonicalize a local or container workspace reference without allowing escape."""

        if value == "<workspace>":
            return value
        suffix = value.removeprefix("<workspace>").replace("\\", "/")
        if not suffix.startswith("/"):
            raise QuarantineError("workspace_reference_invalid")
        parts = PurePosixPath(suffix.lstrip("/")).parts
        if any(part == ".." for part in parts):
            raise QuarantineError("workspace_escape")
        normalized = tuple(part for part in parts if part not in {"", "."})
        return "<workspace>" if not normalized else "<workspace>/" + "/".join(normalized)


def _as_mapping(value: object, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuarantineError(code)
    return value


def _message_role(message: Mapping[str, Any]) -> str:
    info = _as_mapping(message.get("info"), code="message_info_missing")
    role = str(info.get("role", ""))
    if role not in {"user", "assistant"}:
        raise QuarantineError("unsupported_message_role")
    return role


def _text_parts(
    message: Mapping[str, Any], gate: _SafetyGate, *, role: str
) -> list[str]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        raise QuarantineError("message_parts_missing")
    result: list[str] = []
    for part in parts:
        item = _as_mapping(part, code="part_not_object")
        if item.get("type") != "text" or item.get("ignored") is True:
            continue
        # The audited user prompt legitimately states that private
        # chain-of-thought must not be revealed.  That instruction is input,
        # not leaked model reasoning.  Assistant-visible text remains subject
        # to the hidden-reasoning marker gate, while official reasoning parts
        # are dropped separately below.
        result.append(
            gate.text(
                item.get("text"),
                label="message_text",
                check_hidden_reasoning=role == "assistant",
            )
        )
    return result


def _validate_bash_input(value: Mapping[str, Any], gate: _SafetyGate) -> dict[str, object]:
    normalized = gate.value(value, label="bash_input")
    assert isinstance(normalized, dict)
    command = normalized.get("command")
    if not isinstance(command, str) or normalized_command(command) not in ALLOWED_BASH_COMMANDS:
        raise QuarantineError("bash_command_not_allowed")
    return normalized


def _tool_interaction(
    part: Mapping[str, Any], gate: _SafetyGate, *, sequence: int, call_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    tool = str(part.get("tool", ""))
    if tool not in ALLOWED_TOOLS:
        raise QuarantineError("tool_not_allowed")
    state = _as_mapping(part.get("state"), code="tool_state_missing")
    status = str(state.get("status", ""))
    if status not in {"completed", "error"}:
        raise QuarantineError("tool_state_incomplete")
    raw_input = _as_mapping(state.get("input"), code="tool_input_missing")
    tool_input = (
        _validate_bash_input(raw_input, gate)
        if tool == "bash"
        else gate.value(raw_input, label=f"{tool}_input")
    )
    if tool in PATH_REQUIRED_TOOLS:
        if not any(key.casefold() in PATH_KEYS for key in raw_input):
            raise QuarantineError("tool_path_missing")
    if tool in SEARCH_TOOLS:
        try:
            validate_search_input(tool, raw_input)
        except ValueError as error:
            raise QuarantineError(str(error)) from error
    result_field = "output" if status == "completed" else "error"
    result = gate.text(state.get(result_field), label=f"{tool}_result")
    return (
        {
            "type": "tool_call",
            "sequence": sequence,
            "call_id": call_id,
            "tool": tool,
            "input": tool_input,
        },
        {
            "type": "tool_result",
            "sequence": sequence + 1,
            "call_id": call_id,
            "tool": tool,
            "status": status,
            "content": result,
        },
    )


def _trajectory(messages: Sequence[object], gate: _SafetyGate) -> list[dict[str, object]]:
    trajectory: list[dict[str, object]] = []
    call_index = 0
    sequence = 0
    for raw_message in messages:
        message = _as_mapping(raw_message, code="message_not_object")
        role = _message_role(message)
        for text in _text_parts(message, gate, role=role):
            sequence += 1
            trajectory.append(
                {
                    "type": "user_input" if role == "user" else "assistant_output",
                    "sequence": sequence,
                    "content": text,
                }
            )
        parts = message.get("parts")
        assert isinstance(parts, list)
        if role == "assistant":
            for raw_part in parts:
                part = _as_mapping(raw_part, code="part_not_object")
                if part.get("type") == "reasoning":
                    continue
                if part.get("type") != "tool":
                    continue
                call_index += 1
                sequence += 1
                tool_call, tool_result = _tool_interaction(
                    part,
                    gate,
                    sequence=sequence,
                    call_id=f"call_{call_index:04d}",
                )
                trajectory.extend((tool_call, tool_result))
                sequence += 1
    if not any(item["type"] == "user_input" for item in trajectory):
        raise QuarantineError("user_input_missing")
    if not any(item["type"] == "assistant_output" for item in trajectory):
        raise QuarantineError("assistant_output_missing")
    return trajectory


def _validators(value: object, gate: _SafetyGate) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise QuarantineError("validators_missing")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in value:
        item = _as_mapping(raw, code="validator_not_object")
        name = str(item.get("name", ""))
        if name not in ALLOWED_VALIDATORS or name in seen:
            raise QuarantineError("validator_invalid")
        seen.add(name)
        status = str(item.get("status", ""))
        exit_code = item.get("exit_code")
        if status != "PASS" or exit_code != 0:
            raise QuarantineError("validator_failed")
        command = gate.text(item.get("command"), label="validator_command")
        if normalized_command(command) not in ALLOWED_VALIDATOR_COMMANDS:
            raise QuarantineError("validator_command_not_allowed")
        result.append(
            {
                "name": name,
                "status": status,
                "exit_code": 0,
                "command": command,
                "stdout": gate.text(item.get("stdout", ""), label="validator_stdout"),
                "stderr": gate.text(item.get("stderr", ""), label="validator_stderr"),
            }
        )
    if seen != ALLOWED_VALIDATORS:
        raise QuarantineError("validators_incomplete")
    return result


def _staging_validators(
    value: object, gate: _SafetyGate, labels: set[str]
) -> list[dict[str, object]]:
    """Retain complete safe validator output; classify quality after collection."""

    if not isinstance(value, list):
        raise QuarantineError("validators_not_array")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in value:
        item = _as_mapping(raw, code="validator_not_object")
        name = str(item.get("name", ""))
        if name not in ALLOWED_VALIDATORS or name in seen:
            raise QuarantineError("validator_invalid")
        seen.add(name)
        status = str(item.get("status", ""))
        if status not in {"PASS", "FAIL", "SKIP", "TIMEOUT"}:
            raise QuarantineError("validator_status_invalid")
        exit_code = item.get("exit_code")
        if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
            raise QuarantineError("validator_exit_code_invalid")
        command = gate.text(item.get("command"), label="validator_command")
        if normalized_command(command) not in ALLOWED_VALIDATOR_COMMANDS:
            labels.add("validator_command_not_allowed")
        if status != "PASS" or exit_code != 0:
            labels.add("validator_failed")
        result.append(
            {
                "name": name,
                "status": status,
                "exit_code": exit_code,
                "command": command,
                "stdout": gate.text(item.get("stdout", ""), label="validator_stdout"),
                "stderr": gate.text(item.get("stderr", ""), label="validator_stderr"),
            }
        )
    if seen != ALLOWED_VALIDATORS:
        labels.add("validators_incomplete")
    return result


def _public_outcome(value: object, gate: _SafetyGate) -> dict[str, object]:
    item = _as_mapping(value, code="public_outcome_missing")
    if item.get("schema_version") != "anchor.public-outcome.v1" or item.get("status") != "completed":
        raise QuarantineError("public_outcome_invalid")
    allowed = {
        "schema_version": item["schema_version"],
        "status": item["status"],
        "decision_trace": item.get("decision_trace", []),
        "repair_summaries": item.get("repair_summaries", []),
        "final_summary": item.get("final_summary", ""),
    }
    normalized = gate.value(allowed, label="public_outcome")
    assert isinstance(normalized, dict)
    return normalized


def _staging_public_outcome(
    value: object, gate: _SafetyGate, labels: set[str]
) -> dict[str, object] | None:
    if value is None:
        labels.add("public_outcome_missing")
        return None
    item = _as_mapping(value, code="public_outcome_not_object")
    status = str(item.get("status", ""))
    if item.get("schema_version") != "anchor.public-outcome.v1" or status not in {
        "completed",
        "blocked",
        "partial",
    }:
        labels.add("public_outcome_missing")
        return None
    allowed = {
        "schema_version": item["schema_version"],
        "status": status,
        "decision_trace": item.get("decision_trace", []),
        "repair_summaries": item.get("repair_summaries", []),
        "final_summary": item.get("final_summary", ""),
    }
    normalized = gate.value(allowed, label="public_outcome")
    assert isinstance(normalized, dict)
    if status != "completed":
        labels.add(f"task_{status}")
    return normalized


def _final_diff(value: object, gate: _SafetyGate) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise QuarantineError("final_diff_missing")
    result: list[dict[str, object]] = []
    for raw in value:
        item = _as_mapping(raw, code="diff_not_object")
        file = gate.path(item.get("file"), label="diff_file")
        patch = gate.text(item.get("patch"), label="diff_patch")
        if not patch.strip():
            raise QuarantineError("diff_patch_missing")
        try:
            additions = int(item.get("additions", 0))
            deletions = int(item.get("deletions", 0))
        except (TypeError, ValueError) as error:
            raise QuarantineError("diff_count_invalid") from error
        status = str(item.get("status", "modified"))
        if additions < 0 or deletions < 0 or status not in {"added", "deleted", "modified"}:
            raise QuarantineError("diff_metadata_invalid")
        result.append(
            {
                "file": file,
                "patch": patch,
                "additions": additions,
                "deletions": deletions,
                "status": status,
            }
        )
    return result


def _staging_final_diff(
    value: object, gate: _SafetyGate, labels: set[str]
) -> list[dict[str, object]]:
    if value is None or value == []:
        labels.add("final_diff_missing")
        return []
    return _final_diff(value, gate)


def _staging_trajectory(
    messages: Sequence[object], gate: _SafetyGate, labels: set[str]
) -> list[dict[str, object]]:
    """Retain safe complete call/result pairs even when a tool reports an error."""

    trajectory: list[dict[str, object]] = []
    call_index = 0
    sequence = 0
    for raw_message in messages:
        message = _as_mapping(raw_message, code="message_not_object")
        role = _message_role(message)
        for text in _text_parts(message, gate, role=role):
            sequence += 1
            trajectory.append(
                {
                    "type": "user_input" if role == "user" else "assistant_output",
                    "sequence": sequence,
                    "content": text,
                }
            )
        parts = message.get("parts")
        assert isinstance(parts, list)
        if role != "assistant":
            continue
        for raw_part in parts:
            part = _as_mapping(raw_part, code="part_not_object")
            if part.get("type") == "reasoning":
                continue
            if part.get("type") != "tool":
                continue
            tool = gate.text(part.get("tool", ""), label="tool_name")
            state = _as_mapping(part.get("state"), code="tool_state_missing")
            status = str(state.get("status", ""))
            if status not in {"completed", "error", "rejected"}:
                raise QuarantineError("tool_state_incomplete")
            raw_input = _as_mapping(state.get("input"), code="tool_input_missing")
            tool_input = gate.value(raw_input, label=f"{tool or 'tool'}_input")
            assert isinstance(tool_input, dict)
            if tool not in ALLOWED_TOOLS:
                labels.add("tool_not_allowed")
            elif tool in PATH_REQUIRED_TOOLS and not any(
                key.casefold() in PATH_KEYS for key in raw_input
            ):
                labels.add("tool_path_missing")
            if tool == "bash":
                command = tool_input.get("command")
                if not isinstance(command, str) or normalized_command(command) not in ALLOWED_BASH_COMMANDS:
                    labels.add("bash_command_not_allowed")
            if tool in SEARCH_TOOLS:
                try:
                    validate_search_input(tool, raw_input)
                except ValueError:
                    labels.add("search_input_not_allowed")
            if status == "completed":
                raw_result = state.get("output")
            elif status == "error":
                raw_result = state.get("error")
            else:
                raw_result = state.get(
                    "error",
                    state.get("message", state.get("output", "tool call rejected by policy")),
                )
            result = gate.text(raw_result, label=f"{tool or 'tool'}_result")
            call_index += 1
            sequence += 1
            call_id = f"call_{call_index:04d}"
            trajectory.append(
                {
                    "type": "tool_call",
                    "sequence": sequence,
                    "call_id": call_id,
                    "tool": tool,
                    "input": tool_input,
                }
            )
            sequence += 1
            trajectory.append(
                {
                    "type": "tool_result",
                    "sequence": sequence,
                    "call_id": call_id,
                    "tool": tool,
                    "status": status,
                    "content": result,
                }
            )
            if status == "error":
                labels.add("tool_error")
            elif status == "rejected":
                labels.add("tool_rejected")
    if not any(item["type"] == "user_input" for item in trajectory):
        raise QuarantineError("user_input_missing")
    if not any(item["type"] == "assistant_output" for item in trajectory):
        raise QuarantineError("assistant_output_missing")
    if not any(item["type"] == "tool_call" for item in trajectory):
        labels.add("tool_calls_missing")
    return trajectory


def _skill_provenance(value: object, gate: _SafetyGate) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise QuarantineError("skill_provenance_missing")
    required = {
        "source_id",
        "repository",
        "commit",
        "license",
        "license_sha256",
        "bundle_sha256",
        "instruction_audit_sha256",
    }
    result: list[dict[str, object]] = []
    for raw in value:
        item = _as_mapping(raw, code="skill_provenance_not_object")
        if set(item) != required:
            raise QuarantineError("skill_provenance_fields_invalid")
        normalized = gate.value(item, label="skill_provenance")
        assert isinstance(normalized, dict)
        if not str(normalized["repository"]).startswith("https://"):
            raise QuarantineError("skill_repository_invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(normalized["commit"])):
            raise QuarantineError("skill_commit_invalid")
        for name in ("license_sha256", "bundle_sha256", "instruction_audit_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(normalized[name])):
                raise QuarantineError("skill_hash_invalid")
        result.append(normalized)
    return result


def _capture_quality(value: object, gate: _SafetyGate, labels: set[str]) -> dict[str, object]:
    item = _as_mapping(value, code="capture_quality_missing")
    allowed = {"agent_exit_code", "timed_out", "rejected_events", "error_codes"}
    if set(item) != allowed:
        raise QuarantineError("capture_quality_fields_invalid")
    exit_code = item.get("agent_exit_code")
    rejected = item.get("rejected_events")
    timed_out = item.get("timed_out")
    errors = item.get("error_codes")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not isinstance(timed_out, bool)
        or isinstance(rejected, bool)
        or not isinstance(rejected, int)
        or rejected < 0
        or not isinstance(errors, list)
        or any(not isinstance(code, str) or not code for code in errors)
    ):
        raise QuarantineError("capture_quality_invalid")
    normalized_errors = [gate.text(code, label="capture_error_code") for code in errors]
    if exit_code != 0:
        labels.add("agent_exit_nonzero")
    if timed_out:
        labels.add("task_timed_out")
    if rejected:
        labels.add("tool_rejected")
    labels.update(normalized_errors)
    return {
        "agent_exit_code": exit_code,
        "timed_out": timed_out,
        "rejected_events": rejected,
        "error_codes": normalized_errors,
    }


def convert_controlled_session(
    export_data: Mapping[str, Any],
    capture: Mapping[str, Any],
    policy: SessionConversionPolicy,
) -> dict[str, object]:
    """Convert a raw export plus trusted sidecar; quarantine on any unsafe retained field."""

    if capture.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise QuarantineError("capture_schema_invalid")
    if set(capture).difference(CAPTURE_KEYS):
        raise QuarantineError("capture_unknown_field")
    if capture.get("source") != "opencode-export-controlled-fixture":
        raise QuarantineError("capture_source_untrusted")
    info = _as_mapping(export_data.get("info"), code="session_info_missing")
    messages = export_data.get("messages")
    if not isinstance(messages, list):
        raise QuarantineError("session_messages_missing")
    if str(capture.get("session_id", "")) != str(info.get("id", "")):
        raise QuarantineError("capture_session_mismatch")
    gate = _SafetyGate(policy)
    gate.raw_sensitive_scan(export_data)
    gate.raw_sensitive_scan(capture)
    if gate.path(info.get("directory"), label="session_directory") != "<workspace>":
        raise QuarantineError("capture_workspace_mismatch")
    sample_id = gate.text(capture.get("sample_id"), label="sample_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample_id):
        raise QuarantineError("sample_id_invalid")
    trajectory = _trajectory(messages, gate)
    summary = info.get("summary")
    summary_diffs = summary.get("diffs") if isinstance(summary, Mapping) else None
    diff_source = capture.get("final_diff", summary_diffs)
    opencode_version = gate.text(capture.get("opencode_version"), label="opencode_version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", opencode_version):
        raise QuarantineError("opencode_version_invalid")
    candidate: dict[str, object] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "source": {
            "kind": "controlled-opencode-export",
            "opencode_version": opencode_version,
            "source_sha256": _sha256_bytes(_json_bytes(export_data)),
            "workspace": "<workspace>",
            "tool_contract": contract_descriptor(),
        },
        "trajectory": trajectory,
        "final_diff": _final_diff(diff_source, gate),
        "validators": _validators(capture.get("validators"), gate),
        "public_outcome": _public_outcome(capture.get("public_outcome"), gate),
    }
    if capture.get("skill_provenance") is not None:
        candidate["skill_provenance"] = _skill_provenance(
            capture.get("skill_provenance"), gate
        )
    if len(_json_bytes(candidate)) > policy.max_record_bytes:
        raise QuarantineError("record_size_limit")
    return candidate


def convert_controlled_session_staging(
    export_data: Mapping[str, Any],
    capture: Mapping[str, Any],
    policy: SessionConversionPolicy,
) -> dict[str, object]:
    """Collect a structurally complete, safe session before offline quality filtering.

    Tool errors, blocked/partial task outcomes, validation failures, policy rejections,
    and non-zero exits are retained as labels. Secrets, held-out leakage, unsafe paths,
    malformed structures, and hidden reasoning in retained public text still fail closed.
    """

    if capture.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise QuarantineError("capture_schema_invalid")
    if set(capture).difference(CAPTURE_KEYS):
        raise QuarantineError("capture_unknown_field")
    if capture.get("source") != "opencode-export-controlled-fixture":
        raise QuarantineError("capture_source_untrusted")
    info = _as_mapping(export_data.get("info"), code="session_info_missing")
    messages = export_data.get("messages")
    if not isinstance(messages, list):
        raise QuarantineError("session_messages_missing")
    if str(capture.get("session_id", "")) != str(info.get("id", "")):
        raise QuarantineError("capture_session_mismatch")
    gate = _SafetyGate(policy)
    gate.raw_sensitive_scan(export_data)
    gate.raw_sensitive_scan(capture)
    if gate.path(info.get("directory"), label="session_directory") != "<workspace>":
        raise QuarantineError("capture_workspace_mismatch")
    sample_id = gate.text(capture.get("sample_id"), label="sample_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample_id):
        raise QuarantineError("sample_id_invalid")
    opencode_version = gate.text(capture.get("opencode_version"), label="opencode_version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", opencode_version):
        raise QuarantineError("opencode_version_invalid")

    labels: set[str] = set()
    trajectory = _staging_trajectory(messages, gate, labels)
    summary = info.get("summary")
    summary_diffs = summary.get("diffs") if isinstance(summary, Mapping) else None
    diff_source = capture.get("final_diff", summary_diffs)
    validators = _staging_validators(capture.get("validators"), gate, labels)
    public_outcome = _staging_public_outcome(capture.get("public_outcome"), gate, labels)
    final_diff = _staging_final_diff(diff_source, gate, labels)
    provenance = _skill_provenance(capture.get("skill_provenance"), gate)
    execution = _capture_quality(capture.get("quality"), gate, labels)

    candidate: dict[str, object] = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "sample_id": sample_id,
        "source": {
            "kind": "controlled-opencode-export",
            "opencode_version": opencode_version,
            "source_sha256": _sha256_bytes(_json_bytes(export_data)),
            "workspace": "<workspace>",
            "tool_contract": contract_descriptor(),
        },
        "skill_provenance": provenance,
        "trajectory": trajectory,
        "final_diff": final_diff,
        "validators": validators,
        "public_outcome": public_outcome,
        "quality": {
            "labels": sorted(labels),
            "strict_gold_eligible": not labels,
            "execution": execution,
        },
    }
    if len(_json_bytes(candidate)) > policy.max_record_bytes:
        raise QuarantineError("record_size_limit")
    return candidate


def quarantine_record(
    *, sample_id: str | None, code: str, export_bytes: bytes
) -> dict[str, object]:
    safe_sample = sample_id if sample_id and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", sample_id) else None
    return {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "sample_id": safe_sample,
        "reason_code": code,
        "source_sha256": _sha256_bytes(export_bytes),
        "content_retained": False,
    }


def append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
