from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.schema import DatasetValidationError, validate_jsonl, validate_record  # noqa: E402


def record(identifier: str = "security-1", label: str = "[BLOCK]") -> dict:
    return {
        "schema_version": "1.0",
        "id": identifier,
        "expert": "security_audit",
        "messages": [
            {"role": "user", "content": "Add unescaped HTML from the URL."},
            {"role": "assistant", "content": f"{label}\nUntrusted HTML reaches a DOM sink."},
        ],
        "provenance": {"generator": "fixture", "skill_id": "security-v1"},
        "decision_trace": [
            {"check": "DOM sink", "evidence": "innerHTML receives URL data", "action": "block"}
        ],
        "output": {
            "decision": label.strip("[]") if label in ("[BLOCK]", "[PASS]") else "BLOCK",
            "rationale": "Untrusted HTML reaches a DOM sink.",
        },
    }


def test_valid_security_record() -> None:
    assert validate_record(record()) == "security_audit"


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
