from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.swebench.schema import (
    LicenseReference,
    SWEBenchValidationError,
    TaskCard,
    digest_value,
)
from anchor_mvp.swebench.trajectory import (
    OPENCODE_CANDIDATE_SCHEMA_VERSION,
    PLANNER_OUTPUT_SCHEMA_VERSION,
    REVIEW_OUTPUT_SCHEMA_VERSION,
    SANDBOX_AUDIT_SCHEMA_VERSION,
    SECURITY_OUTPUT_SCHEMA_VERSION,
    TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    WORKSPACE_INVENTORY_SCHEMA_VERSION,
    WorkspaceInventory,
    adapt_task_card_trajectory,
)


def _card() -> TaskCard:
    return TaskCard.from_metadata(
        dataset_id="SWE-bench/SWE-smith",
        dataset_revision="a" * 40,
        split="train",
        instance_id="smith-calculator-1",
        repo="fixture/calculator",
        problem_statement="Make addition preserve integer values and formatting.",
        base_commit="b" * 40,
        license_reference=LicenseReference(
            spdx_id="MIT",
            license_file_sha256="c" * 64,
            ledger_sha256="d" * 64,
        ),
        domain_id="python-repository",
        language="python",
        task_kind="issue-resolution",
        builder_expert_id="swe-shared-builder",
        reviewer_expert_id="swe-shared-reviewer",
    )


def _inventory() -> dict[str, Any]:
    return {
        "schema_version": WORKSPACE_INVENTORY_SCHEMA_VERSION,
        "workspace": "<workspace>",
        "files": [
            {
                "path": "<workspace>/src/calculator.py",
                "sha256": "e" * 64,
                "byte_count": 41,
            },
            {
                "path": "<workspace>/pyproject.toml",
                "sha256": "f" * 64,
                "byte_count": 92,
            },
        ],
    }


def _planner(card: TaskCard) -> dict[str, Any]:
    return {
        "schema_version": PLANNER_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "domain_id": card.domain_id,
        "builder_expert_id": card.builder_expert_id,
        "reviewer_expert_id": card.reviewer_expert_id,
        "work_items": [
            "Inspect the implementation.",
            "Apply the smallest source correction and validate it.",
        ],
        "tool_proposals": [
            {
                "proposal_id": "inspect-source",
                "tool": "read",
                "purpose": "Read the implementation file.",
                "input": {"filePath": "<workspace>/src/calculator.py"},
            },
            {
                "proposal_id": "edit-source",
                "tool": "edit",
                "purpose": "Apply the planned source correction.",
                "input": {
                    "filePath": "<workspace>/src/calculator.py",
                    "oldString": "return int(lhs) + int(rhs)",
                    "newString": "return lhs + rhs",
                },
            },
        ],
    }


def _policy(card: TaskCard) -> dict[str, Any]:
    return {
        "schema_version": TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "executed_expert_id": "tool-policy",
        "decisions": [
            {
                "proposal_id": "inspect-source",
                "decision": "APPROVE",
                "reason": "Workspace-local read is required.",
            },
            {
                "proposal_id": "edit-source",
                "decision": "APPROVE",
                "reason": "The edit is workspace-local and scoped.",
            },
        ],
    }


def _candidate(revision: int) -> dict[str, Any]:
    suffix = str(revision)
    generated_patch = (
        "@@ -1 +1 @@\n-return int(lhs) + int(rhs)\n+return lhs + rhs"
        if revision == 1
        else "@@ -1 +1 @@\n-return lhs+rhs\n+return lhs + rhs"
    )
    return {
        "schema_version": OPENCODE_CANDIDATE_SCHEMA_VERSION,
        "sample_id": f"smith-calculator-{revision}",
        "source": {
            "kind": "controlled-opencode-export",
            "opencode_version": "1.2.3",
            "source_sha256": suffix * 64,
            "workspace": "<workspace>",
            "tool_contract": {
                "version": "anchor.execution-tool-contract.v2",
                "tools": ["read", "edit"],
            },
        },
        "skill_provenance": [
            {
                "source_id": "fixture-skill",
                "repository": "https://example.invalid/fixture/skill",
                "commit": "8" * 40,
                "license": "MIT",
                "license_sha256": "7" * 64,
                "bundle_sha256": "6" * 64,
                "instruction_audit_sha256": "5" * 64,
            }
        ],
        "trajectory": [
            {
                "type": "user_input",
                "sequence": 1,
                "content": "Repair the public calculator issue.",
            },
            {
                "type": "assistant_output",
                "sequence": 2,
                "content": "I will inspect the workspace source.",
            },
            {
                "type": "tool_call",
                "sequence": 3,
                "call_id": "call_0001",
                "tool": "read",
                "input": {"filePath": "<workspace>/src/calculator.py"},
            },
            {
                "type": "tool_result",
                "sequence": 4,
                "call_id": "call_0001",
                "tool": "read",
                "status": "completed",
                "content": "def add(lhs, rhs):\n    return int(lhs) + int(rhs)",
            },
            {
                "type": "tool_call",
                "sequence": 5,
                "call_id": "call_0002",
                "tool": "edit",
                "input": {
                    "filePath": "<workspace>/src/calculator.py",
                    "oldString": "return int(lhs) + int(rhs)",
                    "newString": "return lhs + rhs",
                },
            },
            {
                "type": "tool_result",
                "sequence": 6,
                "call_id": "call_0002",
                "tool": "edit",
                "status": "completed",
                "content": "Workspace edit completed.",
            },
            {
                "type": "assistant_output",
                "sequence": 7,
                "content": "The scoped change is complete.",
            },
        ],
        # This is the controlled export's generated diff, not an upstream gold patch.
        "final_diff": [
            {
                "file": "<workspace>/src/calculator.py",
                "patch": generated_patch,
                "additions": 1,
                "deletions": 1,
                "status": "modified",
            }
        ],
        "validators": [
            {
                "name": name,
                "status": "PASS",
                "exit_code": 0,
                "command": command,
                "stdout": "check completed",
                "stderr": "",
            }
            for name, command in (
                ("build", "python -m compileall src"),
                ("test", "python -m pytest -q"),
                ("lint", "ruff check src"),
            )
        ],
        "public_outcome": {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [],
            "repair_summaries": [],
            "final_summary": "The public issue was repaired.",
        },
    }


def _projected_trace() -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_call",
            "sequence": 3,
            "call_id": "call_0001",
            "proposal_id": "inspect-source",
            "tool": "read",
            "input": {"filePath": "<workspace>/src/calculator.py"},
        },
        {
            "type": "tool_result",
            "sequence": 4,
            "call_id": "call_0001",
            "proposal_id": "inspect-source",
            "tool": "read",
            "status": "completed",
            "content": "def add(lhs, rhs):\n    return int(lhs) + int(rhs)",
        },
        {
            "type": "tool_call",
            "sequence": 5,
            "call_id": "call_0002",
            "proposal_id": "edit-source",
            "tool": "edit",
            "input": {
                "filePath": "<workspace>/src/calculator.py",
                "oldString": "return int(lhs) + int(rhs)",
                "newString": "return lhs + rhs",
            },
        },
        {
            "type": "tool_result",
            "sequence": 6,
            "call_id": "call_0002",
            "proposal_id": "edit-source",
            "tool": "edit",
            "status": "completed",
            "content": "Workspace edit completed.",
        },
    ]


def _projected_diff(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    source = candidate["final_diff"][0]
    return [
        {
            "path": source["file"],
            "diff": source["patch"],
            "additions": source["additions"],
            "deletions": source["deletions"],
            "status": source["status"],
        }
    ]


def _execution_summary() -> dict[str, Any]:
    return {
        "status": "completed",
        "exit_code": 0,
        "tool_call_count": 2,
        "checks_run": 3,
        "checks_passed": 3,
        "checks_failed": 0,
    }


def _audit(
    card: TaskCard,
    inventory: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SANDBOX_AUDIT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "source_fingerprint": card.source_fingerprint,
        "workspace": "<workspace>",
        "workspace_binding_sha256": WorkspaceInventory.from_mapping(
            inventory
        ).binding_sha256,
        "snapshot_exclusions": [".anchor", ".git", ".hg", ".svn"],
        "protected_state_before_sha256": "9" * 64,
        "protected_state_after_sha256": "9" * 64,
        "cleanup_status": "cleaned",
        "sessions": [
            {
                "revision": revision,
                "executed_expert_id": card.builder_expert_id,
                "candidate_sha256": digest_value(candidate),
                "tool_trace_sha256": digest_value(_projected_trace()),
                "generated_diff_sha256": digest_value(_projected_diff(candidate)),
                "execution_summary_sha256": digest_value(_execution_summary()),
            }
            for revision, candidate in enumerate(candidates, 1)
        ],
    }


def _review(card: TaskCard, revision: int, decision: str) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "revision": revision,
        "executed_expert_id": card.reviewer_expert_id,
        "decision": decision,
        "feedback": (
            ["Normalize spacing before final approval."] if decision == "REVISE" else []
        ),
    }


def _security(card: TaskCard) -> dict[str, Any]:
    return {
        "schema_version": SECURITY_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "executed_expert_id": "security-audit",
        "decision": "PASS",
        "findings": [],
    }


def _arguments(revision_count: int = 1) -> dict[str, Any]:
    card = _card()
    inventory = _inventory()
    candidates = [_candidate(revision) for revision in range(1, revision_count + 1)]
    audit = _audit(card, inventory, candidates)
    reviews = [
        _review(
            card,
            revision,
            "PASS" if revision == revision_count else "REVISE",
        )
        for revision in range(1, revision_count + 1)
    ]
    return {
        "card": card,
        "workspace_inventory": inventory,
        "planner_output": _planner(card),
        "tool_policy_output": _policy(card),
        "opencode_session_exports": candidates,
        "review_outputs": reviews,
        "security_output": _security(card),
        "sandbox_audit_bundle": audit,
        "trusted_sandbox_audit_sha256": digest_value(audit),
    }


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key.casefold())
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def test_one_card_produces_one_complete_five_stage_chain() -> None:
    args = _arguments()
    record = adapt_task_card_trajectory(**args)

    assert record["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert [stage["stage"] for stage in record["stages"]] == [
        "planner",
        "tool_policy",
        "domain_builder",
        "domain_review",
        "security",
    ]
    assert record["counts"] == {
        "task_card_count": 1,
        "alignment_count": 1,
        "complete_chain_count": 1,
        "revision_count": 1,
    }
    builder = record["stages"][2]["revisions"][0]
    assert set(builder["input"]) == {
        "problem_statement",
        "base_workspace_inventory",
        "approved_tool_results",
    }
    review_input = record["stages"][3]["revisions"][0]["input"]
    security_input = record["stages"][4]["input"]
    assert set(review_input) == {"problem_statement", "diff", "execution_summary"}
    assert set(security_input) == {"problem_statement", "diff", "execution_summary"}
    assert review_input == security_input
    assert "patch" not in _all_keys(record)
    assert "test_patch" not in _all_keys(record)
    assert record["routing_contract"]["planner_selection_matches_execution"] is True


def test_revision_cycles_keep_one_alignment_and_the_same_experts() -> None:
    args = _arguments(revision_count=2)
    record = adapt_task_card_trajectory(**args)
    card = args["card"]

    assert record["counts"]["revision_count"] == 2
    builders = record["stages"][2]["revisions"]
    reviews = record["stages"][3]["revisions"]
    assert {item["alignment_id"] for item in builders + reviews} == {card.alignment_id}
    assert {item["executed_expert_id"] for item in builders} == {card.builder_expert_id}
    assert {item["executed_expert_id"] for item in reviews} == {card.reviewer_expert_id}
    assert [item["output"]["decision"] for item in reviews] == ["REVISE", "PASS"]


def test_planner_route_and_trusted_actual_builder_must_match_card() -> None:
    args = _arguments()
    args["planner_output"]["builder_expert_id"] = "wrong-builder"
    with pytest.raises(SWEBenchValidationError, match="planner route"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["sandbox_audit_bundle"]["sessions"][0]["executed_expert_id"] = "wrong-builder"
    args["trusted_sandbox_audit_sha256"] = digest_value(args["sandbox_audit_bundle"])
    with pytest.raises(SWEBenchValidationError, match="route or hashes"):
        adapt_task_card_trajectory(**args)


def test_tool_calls_require_adjacent_matching_results_and_approval() -> None:
    args = _arguments()
    args["opencode_session_exports"][0]["trajectory"][3]["call_id"] = "orphan"
    with pytest.raises(SWEBenchValidationError, match="pairing"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["tool_policy_output"]["decisions"][1]["decision"] = "DENY"
    with pytest.raises(SWEBenchValidationError, match="not explicitly approved"):
        adapt_task_card_trajectory(**args)


def test_tool_inputs_and_generated_diffs_are_workspace_bound() -> None:
    args = _arguments()
    args["opencode_session_exports"][0]["trajectory"][2]["input"]["filePath"] = (
        "<workspace>/../secret.txt"
    )
    with pytest.raises(SWEBenchValidationError, match="workspace escape|canonical"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["opencode_session_exports"][0]["final_diff"][0]["file"] = (
        "C:\\outside\\calculator.py"
    )
    with pytest.raises(SWEBenchValidationError, match="workspace-bound"):
        adapt_task_card_trajectory(**args)


def test_oracle_fields_and_markers_fail_closed_before_projection() -> None:
    args = _arguments()
    args["planner_output"]["patch"] = "upstream answer"
    with pytest.raises(SWEBenchValidationError, match="forbidden oracle"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["opencode_session_exports"][0]["test_patch"] = "upstream tests"
    with pytest.raises(SWEBenchValidationError, match="forbidden oracle"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["opencode_session_exports"][0]["trajectory"][3]["content"] = (
        "FAIL_TO_PASS: hidden_case_name"
    )
    with pytest.raises(SWEBenchValidationError, match="oracle metadata marker"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["review_outputs"][0]["test_names"] = ["hidden_case_name"]
    with pytest.raises(SWEBenchValidationError, match="forbidden oracle"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["security_output"]["oracle"] = "known answer"
    with pytest.raises(SWEBenchValidationError, match="forbidden oracle"):
        adapt_task_card_trajectory(**args)


def test_audit_bundle_is_recomputed_and_must_match_trusted_digest() -> None:
    args = _arguments()
    args["trusted_sandbox_audit_sha256"] = "0" * 64
    with pytest.raises(SWEBenchValidationError, match="digest does not match"):
        adapt_task_card_trajectory(**args)

    args = _arguments()
    args["sandbox_audit_bundle"]["sessions"][0]["candidate_sha256"] = "0" * 64
    args["trusted_sandbox_audit_sha256"] = digest_value(args["sandbox_audit_bundle"])
    with pytest.raises(SWEBenchValidationError, match="route or hashes"):
        adapt_task_card_trajectory(**args)


def test_adapter_is_local_only_and_does_not_create_files(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    adapt_task_card_trajectory(**_arguments())
    assert list(tmp_path.iterdir()) == before
