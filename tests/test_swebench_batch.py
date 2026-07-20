from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.swebench.batch import (
    EXECUTION_BUNDLE_SCHEMA_VERSION,
    BatchConfig,
    ReplayTeacher,
    compile_manifest,
    compile_work_orders,
    load_validated_cards,
    run_batch,
    stage_record_id,
)
from anchor_mvp.swebench.importer import IMPORT_MANIFEST_SCHEMA_VERSION
from anchor_mvp.swebench.schema import LicenseReference, TaskCard, digest_value
from anchor_mvp.swebench.trajectory import (
    OPENCODE_CANDIDATE_SCHEMA_VERSION,
    PLANNER_OUTPUT_SCHEMA_VERSION,
    REVIEW_OUTPUT_SCHEMA_VERSION,
    SANDBOX_AUDIT_SCHEMA_VERSION,
    SECURITY_OUTPUT_SCHEMA_VERSION,
    TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
    WORKSPACE_INVENTORY_SCHEMA_VERSION,
    WorkspaceInventory,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _card(dataset_id: str = "SWE-bench/SWE-smith") -> TaskCard:
    return TaskCard.from_metadata(
        dataset_id=dataset_id,
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


def _card_artifacts(tmp_path: Path) -> tuple[TaskCard, Path, Path]:
    card = _card()
    cards = tmp_path / "cards.jsonl"
    _write_jsonl(cards, [card.to_dict()])
    manifest = {
        "schema_version": IMPORT_MANIFEST_SCHEMA_VERSION,
        "source": {
            "dataset_id": card.source.dataset_id,
            "dataset_revision": card.source.dataset_revision,
            "split": "train",
        },
        "partition": {
            "full_lite_verified_permanent_deny": True,
            "heldout_variant_row_counts": {"full": 1, "lite": 1, "verified": 1},
        },
        "license_gate": {
            "unknown_repository_policy": "fail_closed",
            "approved_repository_count": 1,
        },
        "cards": {
            "card_count": 1,
            "cards_file_sha256": sha256(cards.read_bytes()).hexdigest(),
        },
        "content_emitted": False,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return card, cards, manifest_path


def _inventory() -> dict[str, Any]:
    return {
        "schema_version": WORKSPACE_INVENTORY_SCHEMA_VERSION,
        "workspace": "<workspace>",
        "files": [
            {
                "path": "<workspace>/src/calculator.py",
                "sha256": "e" * 64,
                "byte_count": 41,
            }
        ],
    }


def _planner(card: TaskCard) -> dict[str, Any]:
    return {
        "schema_version": PLANNER_OUTPUT_SCHEMA_VERSION,
        "alignment_id": card.alignment_id,
        "domain_id": card.domain_id,
        "builder_expert_id": card.builder_expert_id,
        "reviewer_expert_id": card.reviewer_expert_id,
        "work_items": ["Inspect the implementation and apply the smallest correction."],
        "tool_proposals": [
            {
                "proposal_id": "inspect-source",
                "tool": "read",
                "purpose": "Read the workspace implementation.",
                "input": {"filePath": "<workspace>/src/calculator.py"},
            }
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
                "reason": "The read is confined to the controlled workspace.",
            }
        ],
    }


def _candidate() -> dict[str, Any]:
    return {
        "schema_version": OPENCODE_CANDIDATE_SCHEMA_VERSION,
        "sample_id": "smith-calculator-1",
        "source": {
            "kind": "controlled-opencode-export",
            "opencode_version": "1.2.3",
            "source_sha256": "1" * 64,
            "workspace": "<workspace>",
            "tool_contract": {
                "version": "anchor.execution-tool-contract.v2",
                "tools": ["read"],
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
            {"type": "user_input", "sequence": 1, "content": "Repair the calculator."},
            {
                "type": "tool_call",
                "sequence": 2,
                "call_id": "call-1",
                "tool": "read",
                "input": {"filePath": "<workspace>/src/calculator.py"},
            },
            {
                "type": "tool_result",
                "sequence": 3,
                "call_id": "call-1",
                "tool": "read",
                "status": "completed",
                "content": "def add(lhs, rhs): return int(lhs) + int(rhs)",
            },
            {"type": "assistant_output", "sequence": 4, "content": "Repair completed."},
        ],
        "final_diff": [
            {
                "file": "<workspace>/src/calculator.py",
                "patch": "@@ -1 +1 @@\n-return int(lhs) + int(rhs)\n+return lhs + rhs",
                "additions": 1,
                "deletions": 1,
                "status": "modified",
            }
        ],
        "validators": [],
        "public_outcome": {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [],
            "repair_summaries": [],
            "final_summary": "The public issue was repaired.",
        },
    }


def _projection_values() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    trace = [
        {
            "type": "tool_call",
            "sequence": 2,
            "call_id": "call-1",
            "proposal_id": "inspect-source",
            "tool": "read",
            "input": {"filePath": "<workspace>/src/calculator.py"},
        },
        {
            "type": "tool_result",
            "sequence": 3,
            "call_id": "call-1",
            "proposal_id": "inspect-source",
            "tool": "read",
            "status": "completed",
            "content": "def add(lhs, rhs): return int(lhs) + int(rhs)",
        },
    ]
    diff = [
        {
            "path": "<workspace>/src/calculator.py",
            "diff": "@@ -1 +1 @@\n-return int(lhs) + int(rhs)\n+return lhs + rhs",
            "additions": 1,
            "deletions": 1,
            "status": "modified",
        }
    ]
    summary = {
        "status": "completed",
        "exit_code": 0,
        "tool_call_count": 1,
        "checks_run": 0,
        "checks_passed": 0,
        "checks_failed": 0,
    }
    return trace, diff, summary


def _execution_bundle(card: TaskCard) -> dict[str, Any]:
    candidate = _candidate()
    inventory = _inventory()
    trace, diff, summary = _projection_values()
    audit = {
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
                "revision": 1,
                "executed_expert_id": card.builder_expert_id,
                "candidate_sha256": digest_value(candidate),
                "tool_trace_sha256": digest_value(trace),
                "generated_diff_sha256": digest_value(diff),
                "execution_summary_sha256": digest_value(summary),
            }
        ],
    }
    return {
        "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
        "card_id": card.card_id,
        "instance_id": card.source.instance_id,
        "source_fingerprint": card.source_fingerprint,
        "workspace_inventory": inventory,
        "opencode_session_exports": [candidate],
        "sandbox_audit_bundle": audit,
        "trusted_sandbox_audit_sha256": digest_value(audit),
    }


def _responses(card: TaskCard) -> dict[str, dict[str, Any]]:
    plan_id = stage_record_id(card, "planner", upstream_record_ids=(card.card_id,))
    policy_id = stage_record_id(card, "tool_policy", upstream_record_ids=(plan_id,))
    builder_id = stage_record_id(
        card, "domain_builder", revision=1, upstream_record_ids=(policy_id,)
    )
    review_id = stage_record_id(
        card, "domain_review", revision=1, upstream_record_ids=(builder_id,)
    )
    security_id = stage_record_id(card, "security", upstream_record_ids=(review_id,))
    return {
        plan_id: _planner(card),
        policy_id: _policy(card),
        review_id: {
            "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "revision": 1,
            "executed_expert_id": card.reviewer_expert_id,
            "decision": "PASS",
            "feedback": [],
        },
        security_id: {
            "schema_version": SECURITY_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "executed_expert_id": "security-audit",
            "decision": "PASS",
            "findings": [],
        },
    }


def test_compile_splits_every_card_into_five_linked_work_orders(tmp_path: Path) -> None:
    card, cards_path, manifest_path = _card_artifacts(tmp_path)
    cards = load_validated_cards(cards_path, manifest_path)
    orders = compile_work_orders(cards)

    assert len(orders) == 5
    assert [order["stage"] for order in orders] == [
        "planner",
        "tool_policy",
        "domain_builder",
        "domain_review",
        "security",
    ]
    assert {order["identity"]["source_fingerprint"] for order in orders} == {
        card.source_fingerprint
    }
    assert orders[0]["upstream_record_ids"] == [card.card_id]
    for previous, current in zip(orders, orders[1:]):
        assert current["upstream_record_ids"] == [previous["record_id"]]
    assert compile_manifest(cards)["teacher_requests_sent"] == 0


def test_card_manifest_binding_and_permanent_heldout_proof_fail_closed(
    tmp_path: Path,
) -> None:
    _, cards_path, manifest_path = _card_artifacts(tmp_path)
    cards_path.write_text(
        cards_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differs from its import manifest"):
        load_validated_cards(cards_path, manifest_path)

    _, cards_path, manifest_path = _card_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partition"]["full_lite_verified_permanent_deny"] = False
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="held-out proof"):
        load_validated_cards(cards_path, manifest_path)


def test_ordinary_swebench_train_cards_remain_supported(tmp_path: Path) -> None:
    card = _card("SWE-bench/SWE-bench")
    cards_path = tmp_path / "ordinary-cards.jsonl"
    _write_jsonl(cards_path, [card.to_dict()])
    manifest_path = tmp_path / "ordinary-manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": IMPORT_MANIFEST_SCHEMA_VERSION,
            "source": {
                "dataset_id": card.source.dataset_id,
                "dataset_revision": card.source.dataset_revision,
                "split": "train",
            },
            "partition": {
                "full_lite_verified_permanent_deny": True,
                "heldout_variant_row_counts": {
                    "full": 1,
                    "lite": 1,
                    "verified": 1,
                },
            },
            "license_gate": {
                "unknown_repository_policy": "fail_closed",
                "approved_repository_count": 1,
            },
            "cards": {
                "card_count": 1,
                "cards_file_sha256": sha256(cards_path.read_bytes()).hexdigest(),
            },
            "content_emitted": False,
        },
    )
    assert load_validated_cards(cards_path, manifest_path) == (card,)


def test_replay_batch_keeps_real_tool_trace_and_completes_all_five_stages(
    tmp_path: Path,
) -> None:
    card, cards_path, manifest_path = _card_artifacts(tmp_path)
    bundles = tmp_path / "bundles.jsonl"
    _write_jsonl(bundles, [_execution_bundle(card)])
    config = BatchConfig(
        cards_jsonl=cards_path,
        import_manifest=manifest_path,
        execution_bundles_jsonl=bundles,
        replay_responses_jsonl=tmp_path / "unused.jsonl",
        output_dir=tmp_path / "output",
        mode="replay",
        concurrency=1,
    )
    result = asyncio.run(run_batch(config, teacher=ReplayTeacher(_responses(card))))

    assert result.manifest["complete_chain_count"] == 1
    chain = result.chains[0]
    assert [stage["stage"] for stage in chain["stages"]] == [
        "planner",
        "tool_policy",
        "domain_builder",
        "domain_review",
        "security",
    ]
    builder = chain["stages"][2]["revisions"][0]
    assert [event["type"] for event in builder["execution_trace"]] == [
        "tool_call",
        "tool_result",
    ]
    stage_rows = [
        json.loads(line)
        for line in (config.output_dir / "stage_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(stage_rows) == 5
    assert {row["record_id"] for row in stage_rows} == {
        order["record_id"] for order in compile_work_orders((card,))
    }
    assert {row["identity"]["source_fingerprint"] for row in stage_rows} == {
        card.source_fingerprint
    }


def test_execution_bundle_rejects_upstream_answer_fields(tmp_path: Path) -> None:
    card, cards_path, manifest_path = _card_artifacts(tmp_path)
    bundle = _execution_bundle(card)
    bundle["gold_patch"] = "answer material"
    bundles = tmp_path / "bundles.jsonl"
    _write_jsonl(bundles, [bundle])
    config = BatchConfig(
        cards_jsonl=cards_path,
        import_manifest=manifest_path,
        execution_bundles_jsonl=bundles,
        output_dir=tmp_path / "output",
        mode="replay",
    )
    with pytest.raises(ValueError, match="unexpected fields|forbidden"):
        asyncio.run(run_batch(config, teacher=ReplayTeacher({})))


def test_teacher_oracle_shaped_output_is_rejected_before_stage_persistence(
    tmp_path: Path,
) -> None:
    card, cards_path, manifest_path = _card_artifacts(tmp_path)
    bundles = tmp_path / "bundles.jsonl"
    _write_jsonl(bundles, [_execution_bundle(card)])
    responses = _responses(card)
    plan_id = stage_record_id(card, "planner", upstream_record_ids=(card.card_id,))
    responses[plan_id]["expected_test_names"] = ["private_case"]
    config = BatchConfig(
        cards_jsonl=cards_path,
        import_manifest=manifest_path,
        execution_bundles_jsonl=bundles,
        output_dir=tmp_path / "output",
        mode="replay",
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        asyncio.run(run_batch(config, teacher=ReplayTeacher(responses)))
    assert not (config.output_dir / "stage_records.jsonl").exists()
