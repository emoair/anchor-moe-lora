from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.schema import (  # noqa: E402
    DatasetValidationError,
    validate_jsonl,
    validate_record,
)


def record(identifier: str = "security-1", label: str = "[BLOCK]") -> dict:
    return {
        "schema_version": "1.0",
        "id": identifier,
        "expert": "security_audit",
        "messages": [
            {"role": "user", "content": "Add unescaped HTML from the URL."},
            {"role": "assistant", "content": label},
        ],
        "provenance": {"generator": "fixture", "skill_id": "security-v1"},
        "decision_trace": [
            {
                "check": "DOM sink",
                "evidence": "innerHTML receives URL data",
                "action": "block",
            }
        ],
        "output": {
            "decision": label.strip("[]")
            if label in ("[BLOCK]", "[PASS]")
            else "BLOCK",
            "rationale": "Untrusted HTML reaches a DOM sink.",
        },
    }


def canonical_record(expert: str) -> dict:
    if expert == "planner":
        output = {
            "summary": "实现一个无障碍计数器。",
            "steps": [{"id": "P1", "goal": "Implement", "deliverable": "Counter"}],
        }
        assistant = json.dumps(output, ensure_ascii=False, sort_keys=True)
    elif expert == "tool_policy":
        output = {"decision": "APPROVE", "rationale": "The bounded edit is safe."}
        assistant = "APPROVE"
    elif expert in ("frontend_gen", "frontend_review"):
        output = {
            "language": "tsx",
            "code": "\nexport const Counter = () => <main>0</main>;\n",
        }
        assistant = "export const Counter = () => <main>0</main>;"
    elif expert == "security_gate":
        output = {"decision": "PASS", "rationale": "No unsafe data flow is present."}
        assistant = "[PASS]"
    else:  # pragma: no cover - the test parameters enumerate the public experts
        raise AssertionError(expert)
    return {
        "schema_version": "1.0",
        "id": f"{expert}-canonical",
        "expert": expert,
        "messages": [
            {"role": "user", "content": "Complete the bounded task."},
            {"role": "assistant", "content": assistant},
        ],
        "provenance": {"generator": "fixture"},
        "decision_trace": [
            {"check": "contract", "evidence": "fixture", "action": "accept"}
        ],
        "output": output,
    }


def test_valid_security_record() -> None:
    assert validate_record(record()) == "security_audit"


@pytest.mark.parametrize(
    "expert",
    ["planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate"],
)
def test_valid_canonical_target_for_each_public_expert(expert: str) -> None:
    assert validate_record(canonical_record(expert)) == expert


@pytest.mark.parametrize("label", ["No decision", "[BLOCK] and [PASS]"])
def test_security_record_requires_one_label(label: str) -> None:
    with pytest.raises(DatasetValidationError, match="exactly one"):
        validate_record(record(label=label))


def test_jsonl_stream_validation_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "security.jsonl"
    path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    report = validate_jsonl(path, allowed_experts=["security_audit"])
    assert report["ok"] is True
    assert report["valid_records"] == 1

    path.write_text(
        json.dumps(record()) + "\n" + json.dumps(record()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        validate_jsonl(path)


@pytest.mark.parametrize(
    ("expert", "tampered_assistant"),
    [
        ("frontend_gen", "export const Counter = () => <main>1</main>;"),
        (
            "planner",
            '{"steps": [{"deliverable": "Counter", "goal": "Skip", "id": "P1"}], '
            '"summary": "实现一个无障碍计数器。"}',
        ),
        ("security_gate", "[PASS]\nNo unsafe data flow is present."),
    ],
)
def test_record_and_jsonl_reject_tampered_assistant_output_target(
    tmp_path: Path, expert: str, tampered_assistant: str
) -> None:
    tampered = canonical_record(expert)
    tampered["messages"][-1]["content"] = tampered_assistant

    with pytest.raises(
        DatasetValidationError, match="canonical target derived from output"
    ):
        validate_record(tampered)

    path = tmp_path / f"{expert}.jsonl"
    path.write_text(json.dumps(tampered, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(
        DatasetValidationError, match="canonical target derived from output"
    ):
        validate_jsonl(path, allowed_experts=[expert])
