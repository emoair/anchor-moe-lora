"""Streaming validation for distilled expert JSONL datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERTS = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
    # Legacy names remain readable so existing successful live rows resume.
    "code_review",
    "security_audit",
)
ROLES = ("system", "user", "assistant")
SECURITY_LABELS = ("[BLOCK]", "[PASS]")
TOOL_POLICY_LABELS = ("APPROVE", "BLOCK", "ESCALATE")


class DatasetValidationError(ValueError):
    """Raised when a distilled record cannot be used safely for SFT."""


def validate_record(record: Any, *, source: str = "<record>") -> str:
    """Validate one canonical record and return its expert name."""

    if not isinstance(record, Mapping):
        raise DatasetValidationError(f"{source}: record must be a JSON object")
    if record.get("schema_version") != "1.0":
        raise DatasetValidationError(f"{source}: schema_version must be '1.0'")
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise DatasetValidationError(f"{source}: id must be a non-empty string")
    expert = record.get("expert")
    if expert not in EXPERTS:
        raise DatasetValidationError(f"{source}: expert must be one of {EXPERTS}")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DatasetValidationError(f"{source}: messages must contain at least user and assistant turns")
    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise DatasetValidationError(f"{source}: messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise DatasetValidationError(f"{source}: messages[{index}].role must be one of {ROLES}")
        if not isinstance(content, str) or not content.strip():
            raise DatasetValidationError(f"{source}: messages[{index}].content must be non-empty text")
        roles.append(role)
    if "user" not in roles:
        raise DatasetValidationError(f"{source}: at least one user turn is required")
    if roles[-1] != "assistant":
        raise DatasetValidationError(f"{source}: the final message must be the training target (assistant)")

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise DatasetValidationError(f"{source}: provenance must be an object")
    generator = provenance.get("generator")
    teacher = provenance.get("teacher")
    teacher_model = teacher.get("model") if isinstance(teacher, Mapping) else None
    if not (
        (isinstance(generator, str) and generator.strip())
        or (isinstance(teacher_model, str) and teacher_model.strip())
    ):
        raise DatasetValidationError(
            f"{source}: provenance needs generator or teacher.model"
        )

    trace = record.get("decision_trace")
    if not isinstance(trace, list) or not trace:
        raise DatasetValidationError(f"{source}: decision_trace must be a non-empty list")
    for index, step in enumerate(trace):
        if not isinstance(step, Mapping):
            raise DatasetValidationError(f"{source}: decision_trace[{index}] must be an object")
        for field in ("check", "evidence", "action"):
            value = step.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(
                    f"{source}: decision_trace[{index}].{field} must be non-empty text"
                )

    output = record.get("output")
    if not isinstance(output, Mapping):
        raise DatasetValidationError(f"{source}: output must be an object")
    if expert == "planner":
        summary = output.get("summary")
        steps = output.get("steps")
        if not isinstance(summary, str) or not summary.strip():
            raise DatasetValidationError(f"{source}: planner output.summary is required")
        if not isinstance(steps, list) or not steps:
            raise DatasetValidationError(f"{source}: planner output.steps is required")

    if expert == "tool_policy":
        assistant = str(messages[-1]["content"]).strip()
        decision = output.get("decision")
        rationale = output.get("rationale")
        if decision not in TOOL_POLICY_LABELS or assistant != decision:
            raise DatasetValidationError(
                f"{source}: tool_policy target must exactly match one of {TOOL_POLICY_LABELS}"
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise DatasetValidationError(f"{source}: tool_policy output.rationale is required")

    if expert in ("frontend_gen", "frontend_review", "code_review"):
        code = output.get("code")
        if not isinstance(code, str) or not code.strip():
            raise DatasetValidationError(f"{source}: {expert} output.code is required")

    if expert in ("security_gate", "security_audit"):
        assistant = str(messages[-1]["content"])
        present = [label for label in SECURITY_LABELS if label in assistant]
        if len(present) != 1:
            raise DatasetValidationError(
                f"{source}: security target must contain exactly one of {SECURITY_LABELS}"
            )
        decision = output.get("decision")
        rationale = output.get("rationale")
        if decision not in ("BLOCK", "PASS") or f"[{decision}]" != present[0]:
            raise DatasetValidationError(
                f"{source}: output.decision must match the assistant security label"
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise DatasetValidationError(f"{source}: security output.rationale is required")
    return str(expert)


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{dataset_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc


def validate_jsonl(
    path: str | Path,
    *,
    allowed_experts: Iterable[str] | None = None,
    max_errors: int = 20,
) -> dict[str, Any]:
    """Validate a file without loading it into memory and return a compact report."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise DatasetValidationError(f"dataset does not exist: {dataset_path}")
    allowed = set(allowed_experts) if allowed_experts is not None else set(EXPERTS)
    unknown_allowed = allowed - set(EXPERTS)
    if unknown_allowed:
        raise DatasetValidationError(f"unknown allowed experts: {sorted(unknown_allowed)}")

    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    errors: list[str] = []
    total = 0
    for line_number, record in iter_jsonl(dataset_path):
        total += 1
        source = f"{dataset_path}:{line_number}"
        try:
            expert = validate_record(record, source=source)
            if expert not in allowed:
                raise DatasetValidationError(
                    f"{source}: expert {expert!r} is not valid for this adapter dataset"
                )
            identifier = record["id"]
            if identifier in seen_ids:
                raise DatasetValidationError(f"{source}: duplicate id {identifier!r}")
            seen_ids.add(identifier)
            counts[expert] += 1
        except DatasetValidationError as exc:
            errors.append(str(exc))
            if len(errors) >= max_errors:
                break

    if not total:
        errors.append(f"{dataset_path}: file contains no records")
    report = {
        "path": str(dataset_path),
        "records_seen": total,
        "valid_records": sum(counts.values()),
        "experts": dict(sorted(counts.items())),
        "errors": errors,
        "ok": not errors,
    }
    if errors:
        raise DatasetValidationError("\n".join(errors))
    return report
