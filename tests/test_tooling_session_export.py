from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor_mvp.tooling.session_export import (
    QuarantineError,
    SessionConversionPolicy,
    convert_controlled_session,
    quarantine_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _policy(tmp_path: Path, *, heldout: str = "Frozen cobalt acceptance phrase"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cases = tmp_path / "heldout.jsonl"
    cases.write_text(
        json.dumps(
            {
                "case_id": "synthetic-heldout-case",
                "seed_id": "synthetic-heldout-seed",
                "case_family": "synthetic-heldout-family",
                "requirement": heldout,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return SessionConversionPolicy(workspace.resolve(), cases), workspace


def _export(workspace: Path) -> dict[str, object]:
    source = workspace / "src/status-list.js"
    return {
        "info": {
            "id": "ses_controlled_mock",
            "directory": str(workspace),
            "summary": {
                "diffs": [
                    {
                        "file": str(source),
                        "patch": "@@ -1 +1 @@\n-export const value = 1;\n+export const value = 2;",
                        "additions": 1,
                        "deletions": 1,
                        "status": "modified",
                    }
                ]
            },
        },
        "messages": [
            {
                "info": {"id": "msg_user", "role": "user"},
                "parts": [
                    {
                        "id": "part_user",
                        "type": "text",
                        "text": "Update the controlled fixture and validate it.",
                    }
                ],
            },
            {
                "info": {"id": "msg_assistant", "role": "assistant"},
                "parts": [
                    {
                        "id": "part_text",
                        "type": "text",
                        "text": "I will inspect, edit, and validate the fixture.",
                    },
                    {
                        "id": "part_reasoning",
                        "type": "reasoning",
                        "text": "private model reasoning must never be retained",
                    },
                    {
                        "id": "part_read",
                        "type": "tool",
                        "callID": "raw-call-read",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": str(source)},
                            "output": f"<file>{source}\nexport const value = 1;</file>",
                        },
                    },
                    {
                        "id": "part_edit",
                        "type": "tool",
                        "callID": "raw-call-edit",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {
                                "filePath": str(source),
                                "oldString": "export const value = 1;",
                                "newString": "export const value = 2;",
                            },
                            "output": "Edit applied successfully.",
                        },
                    },
                    {
                        "id": "part_patch",
                        "type": "tool",
                        "callID": "raw-call-patch",
                        "tool": "apply_patch",
                        "state": {
                            "status": "completed",
                            "input": {"patch": "*** Begin Patch\n*** End Patch"},
                            "output": "Patch applied to one controlled file.",
                        },
                    },
                    {
                        "id": "part_bash",
                        "type": "tool",
                        "callID": "raw-call-bash",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "npm run test --if-present"},
                            "output": "TAP version 13\n# pass 3\n# fail 0",
                        },
                    },
                    {
                        "id": "part_final",
                        "type": "text",
                        "text": "The public task is complete and all validators passed.",
                    },
                ],
            },
        ],
    }


def _capture() -> dict[str, object]:
    return {
        "schema_version": "anchor.controlled-session-capture.v1",
        "source": "opencode-export-controlled-fixture",
        "sample_id": "controlled-session-001",
        "session_id": "ses_controlled_mock",
        "opencode_version": "1.17.18",
        "validators": [
            {
                "name": name,
                "status": "PASS",
                "exit_code": 0,
                "command": f"npm run {name}",
                "stdout": f"{name}: complete output retained",
                "stderr": "",
            }
            for name in ("build", "test", "lint")
        ],
        "public_outcome": {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [
                {
                    "check": "validators",
                    "evidence": "build, test, and lint passed",
                    "action": "keep the controlled fixture diff",
                }
            ],
            "repair_summaries": ["Changed one fixture value."],
            "final_summary": "Controlled fixture validation passed.",
        },
    }


def test_controlled_export_retains_safe_tool_results_and_drops_reasoning(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    candidate = convert_controlled_session(_export(workspace), _capture(), policy)

    assert candidate["schema_version"] == "anchor.session-training-candidate.v1"
    trajectory = candidate["trajectory"]
    assert isinstance(trajectory, list)
    calls = [item for item in trajectory if item["type"] == "tool_call"]
    results = [item for item in trajectory if item["type"] == "tool_result"]
    assert [item["call_id"] for item in calls] == [
        "call_0001",
        "call_0002",
        "call_0003",
        "call_0004",
    ]
    assert [item["call_id"] for item in results] == [
        "call_0001",
        "call_0002",
        "call_0003",
        "call_0004",
    ]
    assert [item["tool"] for item in calls] == ["read", "edit", "apply_patch", "bash"]
    assert results[0]["content"].endswith("export const value = 1;</file>")
    assert results[3]["content"] == "TAP version 13\n# pass 3\n# fail 0"
    assert calls[0]["input"]["filePath"] == "<workspace>/src/status-list.js"
    assert all(result["sequence"] == call["sequence"] + 1 for call, result in zip(calls, results))
    serialized = json.dumps(candidate, ensure_ascii=False)
    assert "private model reasoning" not in serialized
    assert str(workspace) not in serialized
    assert candidate["final_diff"][0]["file"] == "<workspace>/src/status-list.js"
    assert candidate["validators"][1]["stdout"] == "test: complete output retained"
    assert candidate["source"]["tool_contract"]["version"] == (
        "anchor.execution-tool-contract.v2"
    )


def test_secret_in_tool_result_quarantines_entire_capture(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    exported["messages"][1]["parts"][2]["state"]["output"] = (
        "credential sk-example-secret-value-123456"
    )

    with pytest.raises(QuarantineError, match="secret_detected"):
        convert_controlled_session(exported, _capture(), policy)

    record = quarantine_record(
        sample_id="controlled-session-001",
        code="secret_detected",
        export_bytes=json.dumps(exported).encode(),
    )
    assert record["content_retained"] is False
    assert "credential" not in json.dumps(record)


def test_secret_in_dropped_reasoning_still_quarantines_entire_capture(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    exported["messages"][1]["parts"][1]["text"] = "sk-private-reasoning-secret-123456"

    with pytest.raises(QuarantineError, match="secret_detected"):
        convert_controlled_session(exported, _capture(), policy)


def test_workspace_escape_in_tool_input_quarantines_capture(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    exported["messages"][1]["parts"][2]["state"]["input"]["filePath"] = (
        str(tmp_path / "outside.txt")
    )

    with pytest.raises(QuarantineError, match="absolute_path_outside_workspace|workspace_escape"):
        convert_controlled_session(exported, _capture(), policy)


def test_heldout_text_in_public_output_quarantines_capture(tmp_path: Path):
    heldout = "Frozen cobalt acceptance phrase"
    policy, workspace = _policy(tmp_path, heldout=heldout)
    exported = _export(workspace)
    exported["messages"][1]["parts"][-1]["text"] = heldout

    with pytest.raises(QuarantineError, match="heldout_leakage"):
        convert_controlled_session(exported, _capture(), policy)


def test_official_sanitized_export_is_rejected_as_lossy(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    exported["messages"][0]["parts"][0]["text"] = "[redacted:text:part_user]"

    with pytest.raises(QuarantineError, match="official_sanitize_is_lossy"):
        convert_controlled_session(exported, _capture(), policy)


def test_failed_or_environment_reading_tools_are_not_candidates(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    bash = exported["messages"][1]["parts"][5]
    bash["state"]["input"]["command"] = "printenv"

    with pytest.raises(QuarantineError, match="bash_command_not_allowed"):
        convert_controlled_session(exported, _capture(), policy)


def test_checked_in_heldout_manifest_is_verified_before_conversion(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = SessionConversionPolicy(
        workspace.resolve(),
        ROOT / "configs/benchmark/heldout_cases_v1.jsonl",
        ROOT / "examples/benchmark/fixtures",
        ROOT / "artifacts/benchmark/heldout_v1/manifest.json",
    )

    candidate = convert_controlled_session(_export(workspace), _capture(), policy)

    assert candidate["sample_id"] == "controlled-session-001"


def test_v2_contract_retains_write_and_workspace_search_results(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    source = workspace / "src" / "status-list.js"
    extra = [
        ("write", {"filePath": str(source), "content": "export const ok = true"}, "Wrote file"),
        ("grep", {"pattern": "export\\s+const", "path": str(workspace / "src")}, "src/status-list.js:1"),
        ("glob", {"pattern": "**/*.js", "path": str(workspace / "src")}, "src/status-list.js"),
        ("list", {"path": str(workspace / "src")}, "status-list.js"),
    ]
    parts = exported["messages"][1]["parts"]
    insert_at = len(parts) - 1
    for index, (tool, tool_input, output) in enumerate(extra, 1):
        parts.insert(
            insert_at + index - 1,
            {
                "id": f"part_v2_{index}",
                "type": "tool",
                "callID": f"raw-v2-{index}",
                "tool": tool,
                "state": {"status": "completed", "input": tool_input, "output": output},
            },
        )

    candidate = convert_controlled_session(exported, _capture(), policy)
    calls = [item for item in candidate["trajectory"] if item["type"] == "tool_call"]
    results = [item for item in candidate["trajectory"] if item["type"] == "tool_result"]
    assert [item["tool"] for item in calls][-4:] == ["write", "grep", "glob", "list"]
    assert calls[-4]["input"]["filePath"] == "<workspace>/src/status-list.js"
    assert calls[-3]["input"]["path"] == "<workspace>/src"
    assert [item["call_id"] for item in calls] == [item["call_id"] for item in results]


def test_v2_contract_rejects_glob_traversal_and_search_secret_result(tmp_path: Path):
    policy, workspace = _policy(tmp_path)
    exported = _export(workspace)
    parts = exported["messages"][1]["parts"]
    parts.insert(
        -1,
        {
            "id": "bad-glob",
            "type": "tool",
            "callID": "bad-glob",
            "tool": "glob",
            "state": {
                "status": "completed",
                "input": {"pattern": "../**/*", "path": str(workspace)},
                "output": "none",
            },
        },
    )
    with pytest.raises(QuarantineError, match="glob_pattern_escapes_workspace"):
        convert_controlled_session(exported, _capture(), policy)

    parts[-2]["state"]["input"]["pattern"] = "**/*.js"
    parts[-2]["state"]["output"] = "sk-search-result-secret-123456"
    with pytest.raises(QuarantineError, match="secret_detected"):
        convert_controlled_session(exported, _capture(), policy)
