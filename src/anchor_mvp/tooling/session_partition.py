"""Offline collect-then-filter partitioning for controlled OpenCode sessions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .session_export import (
    ALLOWED_BASH_COMMANDS,
    ALLOWED_TOOLS,
    ALLOWED_VALIDATOR_COMMANDS,
    ALLOWED_VALIDATORS,
    CANDIDATE_SCHEMA_VERSION,
    SECRET_PATTERNS,
    STAGING_SCHEMA_VERSION,
    PATH_KEYS,
    QuarantineError,
    SessionConversionPolicy,
    _SafetyGate,
)
from .tool_contract import (
    PATH_REQUIRED_TOOLS,
    SEARCH_TOOLS,
    contract_descriptor,
    normalized_command,
    validate_search_input,
)


PARTITION_REJECT_SCHEMA_VERSION = "anchor.session-partition-reject.v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject(*, line_number: int, reason: str, source: bytes) -> dict[str, object]:
    return {
        "schema_version": PARTITION_REJECT_SCHEMA_VERSION,
        "line_number": line_number,
        "reason_code": reason,
        "source_sha256": _sha256(source),
        "content_retained": False,
    }


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuarantineError(code)
    return value


def derive_quality_labels(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Recompute quality labels instead of trusting staging metadata."""

    labels: set[str] = set()
    quality = _mapping(record.get("quality"), "quality_missing")
    if set(quality) != {"labels", "strict_gold_eligible", "execution"}:
        raise QuarantineError("quality_fields_invalid")
    execution = _mapping(quality.get("execution"), "quality_execution_missing")
    if set(execution) != {
        "agent_exit_code",
        "timed_out",
        "rejected_events",
        "error_codes",
    }:
        raise QuarantineError("quality_execution_fields_invalid")
    exit_code = execution.get("agent_exit_code")
    timed_out = execution.get("timed_out")
    rejected = execution.get("rejected_events")
    errors = execution.get("error_codes")
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
        raise QuarantineError("quality_execution_invalid")
    if exit_code != 0:
        labels.add("agent_exit_nonzero")
    if timed_out:
        labels.add("task_timed_out")
    if rejected:
        labels.add("tool_rejected")
    labels.update(errors)

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list):
        raise QuarantineError("trajectory_not_array")
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    has_user = False
    has_assistant = False
    for item_value in trajectory:
        item = _mapping(item_value, "trajectory_item_not_object")
        kind = item.get("type")
        if kind == "user_input":
            has_user = True
        elif kind == "assistant_output":
            has_assistant = True
        elif kind == "tool_call":
            call_id = str(item.get("call_id", ""))
            tool = str(item.get("tool", ""))
            if not call_id or call_id in calls or not isinstance(item.get("input"), Mapping):
                raise QuarantineError("tool_call_invalid")
            calls[call_id] = tool
            if tool not in ALLOWED_TOOLS:
                labels.add("tool_not_allowed")
            elif tool in PATH_REQUIRED_TOOLS and not any(
                str(key).casefold() in PATH_KEYS for key in item["input"]
            ):
                labels.add("tool_path_missing")
            if tool == "bash":
                command = item["input"].get("command")
                if not isinstance(command, str) or normalized_command(command) not in ALLOWED_BASH_COMMANDS:
                    labels.add("bash_command_not_allowed")
            if tool in SEARCH_TOOLS:
                try:
                    validate_search_input(tool, item["input"])
                except ValueError:
                    labels.add("search_input_not_allowed")
        elif kind == "tool_result":
            call_id = str(item.get("call_id", ""))
            tool = str(item.get("tool", ""))
            status = item.get("status")
            if not call_id or call_id in results or status not in {
                "completed",
                "error",
                "rejected",
            }:
                raise QuarantineError("tool_result_invalid")
            results[call_id] = tool
            if status == "error":
                labels.add("tool_error")
            elif status == "rejected":
                labels.add("tool_rejected")
        else:
            raise QuarantineError("trajectory_type_invalid")
    if not has_user:
        raise QuarantineError("user_input_missing")
    if not has_assistant:
        raise QuarantineError("assistant_output_missing")
    if not calls:
        labels.add("tool_calls_missing")
    if calls != results:
        raise QuarantineError("tool_call_result_mismatch")

    validators = record.get("validators")
    if not isinstance(validators, list):
        raise QuarantineError("validators_not_array")
    names: set[str] = set()
    for item_value in validators:
        item = _mapping(item_value, "validator_not_object")
        if set(item) != {"name", "status", "exit_code", "command", "stdout", "stderr"}:
            raise QuarantineError("validator_fields_invalid")
        name = str(item.get("name", ""))
        if name not in ALLOWED_VALIDATORS or name in names:
            raise QuarantineError("validator_invalid")
        names.add(name)
        command = item.get("command")
        if not isinstance(item.get("stdout"), str) or not isinstance(item.get("stderr"), str):
            raise QuarantineError("validator_output_invalid")
        if not isinstance(command, str) or normalized_command(command) not in (
            ALLOWED_VALIDATOR_COMMANDS
        ):
            labels.add("validator_command_not_allowed")
        if item.get("status") != "PASS" or item.get("exit_code") != 0:
            labels.add("validator_failed")
    if names != ALLOWED_VALIDATORS:
        labels.add("validators_incomplete")

    final_diff = record.get("final_diff")
    if not isinstance(final_diff, list):
        raise QuarantineError("final_diff_not_array")
    if not final_diff:
        labels.add("final_diff_missing")

    outcome = record.get("public_outcome")
    if outcome is None:
        labels.add("public_outcome_missing")
    elif isinstance(outcome, Mapping):
        if set(outcome) != {
            "schema_version",
            "status",
            "decision_trace",
            "repair_summaries",
            "final_summary",
        }:
            raise QuarantineError("public_outcome_fields_invalid")
        if (
            not isinstance(outcome.get("decision_trace"), list)
            or not isinstance(outcome.get("repair_summaries"), list)
            or not isinstance(outcome.get("final_summary"), str)
        ):
            raise QuarantineError("public_outcome_shape_invalid")
        status = outcome.get("status")
        if outcome.get("schema_version") != "anchor.public-outcome.v1" or status not in {
            "completed",
            "blocked",
            "partial",
        }:
            labels.add("public_outcome_invalid")
        elif status != "completed":
            labels.add(f"task_{status}")
    else:
        raise QuarantineError("public_outcome_not_object")
    return tuple(sorted(labels))


def validate_staging_record(
    record: Mapping[str, Any], policy: SessionConversionPolicy
) -> tuple[str, ...]:
    required_record_fields = {
        "schema_version",
        "sample_id",
        "source",
        "skill_provenance",
        "trajectory",
        "final_diff",
        "validators",
        "public_outcome",
        "quality",
    }
    if set(record) != required_record_fields:
        raise QuarantineError("staging_fields_invalid")
    if record.get("schema_version") != STAGING_SCHEMA_VERSION:
        raise QuarantineError("staging_schema_invalid")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample_id
    ):
        raise QuarantineError("sample_id_invalid")
    source = _mapping(record.get("source"), "source_missing")
    if set(source) != {
        "kind",
        "opencode_version",
        "source_sha256",
        "workspace",
        "tool_contract",
    }:
        raise QuarantineError("source_fields_invalid")
    if (
        source.get("kind") != "controlled-opencode-export"
        or source.get("workspace") != "<workspace>"
        or not re.fullmatch(r"\d+\.\d+\.\d+", str(source.get("opencode_version", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("source_sha256", "")))
        or source.get("tool_contract") != contract_descriptor()
    ):
        raise QuarantineError("source_invalid")
    provenance = record.get("skill_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise QuarantineError("skill_provenance_missing")
    provenance_fields = {
        "source_id",
        "repository",
        "commit",
        "license",
        "license_sha256",
        "bundle_sha256",
        "instruction_audit_sha256",
    }
    for value in provenance:
        item = _mapping(value, "skill_provenance_not_object")
        if set(item) != provenance_fields:
            raise QuarantineError("skill_provenance_fields_invalid")
        if (
            not str(item.get("repository", "")).startswith("https://")
            or not re.fullmatch(r"[0-9a-f]{40}", str(item.get("commit", "")))
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get(name, "")))
                for name in ("license_sha256", "bundle_sha256", "instruction_audit_sha256")
            )
        ):
            raise QuarantineError("skill_provenance_invalid")

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list):
        raise QuarantineError("trajectory_not_array")
    previous_sequence = 0
    for value in trajectory:
        item = _mapping(value, "trajectory_item_not_object")
        sequence = item.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= previous_sequence
        ):
            raise QuarantineError("trajectory_sequence_invalid")
        previous_sequence = sequence
        kind = item.get("type")
        if kind in {"user_input", "assistant_output"}:
            if set(item) != {"type", "sequence", "content"} or not isinstance(
                item.get("content"), str
            ):
                raise QuarantineError("trajectory_text_invalid")
        elif kind == "tool_call":
            if set(item) != {"type", "sequence", "call_id", "tool", "input"}:
                raise QuarantineError("tool_call_fields_invalid")
        elif kind == "tool_result":
            if set(item) != {
                "type",
                "sequence",
                "call_id",
                "tool",
                "status",
                "content",
            } or not isinstance(item.get("content"), str):
                raise QuarantineError("tool_result_fields_invalid")

    final_diff = record.get("final_diff")
    if not isinstance(final_diff, list):
        raise QuarantineError("final_diff_not_array")
    for value in final_diff:
        item = _mapping(value, "diff_not_object")
        if set(item) != {"file", "patch", "additions", "deletions", "status"}:
            raise QuarantineError("diff_fields_invalid")
        additions = item.get("additions")
        deletions = item.get("deletions")
        if (
            not isinstance(item.get("file"), str)
            or not str(item["file"]).startswith("<workspace>/")
            or not isinstance(item.get("patch"), str)
            or not str(item["patch"]).strip()
            or isinstance(additions, bool)
            or not isinstance(additions, int)
            or additions < 0
            or isinstance(deletions, bool)
            or not isinstance(deletions, int)
            or deletions < 0
            or item.get("status") not in {"added", "deleted", "modified"}
        ):
            raise QuarantineError("diff_invalid")
    gate = _SafetyGate(policy)
    gate.raw_sensitive_scan(record)
    # User input may legitimately contain the instruction "do not reveal
    # chain-of-thought". Re-scan the whole record for secrets, held-out text,
    # unsafe paths, and canonical form without treating that input as leaked
    # model reasoning. Assistant-visible output is checked explicitly below.
    normalized = gate.value(
        record,
        label="staging_record",
        check_hidden_reasoning=False,
    )
    if normalized != record:
        raise QuarantineError("staging_record_not_canonical")
    for item in record["trajectory"]:
        if item.get("type") == "assistant_output":
            gate.text(item.get("content"), label="assistant_output")
    if record.get("public_outcome") is not None:
        gate.value(record["public_outcome"], label="public_outcome")
    labels = derive_quality_labels(record)
    quality = _mapping(record.get("quality"), "quality_missing")
    stored = quality.get("labels")
    if not isinstance(stored, list) or any(not isinstance(item, str) for item in stored):
        raise QuarantineError("quality_labels_invalid")
    if tuple(sorted(set(stored))) != labels:
        raise QuarantineError("quality_labels_mismatch")
    if quality.get("strict_gold_eligible") is not (not labels):
        raise QuarantineError("quality_eligibility_mismatch")
    return labels


def strict_candidate(record: Mapping[str, Any]) -> dict[str, object]:
    """Project a validated, label-free staging record to the compatible v1 schema."""

    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "sample_id": record["sample_id"],
        "source": record["source"],
        "skill_provenance": record["skill_provenance"],
        "trajectory": record["trajectory"],
        "final_diff": record["final_diff"],
        "validators": record["validators"],
        "public_outcome": record["public_outcome"],
    }


def _write_atomic(path: Path, records: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")
    os.replace(temporary, path)


def partition_staging_jsonl(
    *,
    staging_path: Path,
    gold_path: Path,
    negative_path: Path,
    reject_path: Path,
    policy: SessionConversionPolicy,
) -> dict[str, int]:
    paths = {path.resolve() for path in (staging_path, gold_path, negative_path, reject_path)}
    if len(paths) != 4:
        raise ValueError("staging and partition output paths must be distinct")
    gold: list[Mapping[str, object]] = []
    negative: list[Mapping[str, object]] = []
    rejects: list[Mapping[str, object]] = []
    seen_gold: set[str] = set()
    for line_number, raw_line in enumerate(staging_path.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            continue
        if any(pattern.search(raw_line.decode("utf-8", errors="ignore")) for pattern in SECRET_PATTERNS):
            rejects.append(_reject(line_number=line_number, reason="secret_detected", source=raw_line))
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise QuarantineError("staging_not_object")
            labels = validate_staging_record(value, policy)
            sample_id = str(value["sample_id"])
            if not labels and sample_id in seen_gold:
                raise QuarantineError("duplicate_gold_sample_id")
        except (UnicodeDecodeError, json.JSONDecodeError):
            rejects.append(
                _reject(line_number=line_number, reason="invalid_json_or_encoding", source=raw_line)
            )
        except (OSError, ValueError) as error:
            reason = error.code if isinstance(error, QuarantineError) else "partition_validation_error"
            rejects.append(_reject(line_number=line_number, reason=reason, source=raw_line))
        else:
            if labels:
                negative.append(value)
            else:
                seen_gold.add(sample_id)
                gold.append(strict_candidate(value))
    _write_atomic(gold_path, gold)
    _write_atomic(negative_path, negative)
    _write_atomic(reject_path, rejects)
    return {"gold": len(gold), "negative": len(negative), "reject": len(rejects)}
