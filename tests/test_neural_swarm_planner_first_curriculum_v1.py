from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from anchor_mvp.research.neural_swarm_planner_first_curriculum_v1 import (
    DECISION_VERSION,
    EXPERT_ASSETS,
    EXPERT_LEAF_ARMS,
    OBSERVED_LEGACY_ORDER,
    PlannerFirstCurriculumError,
    build_question_row,
    build_runtime_decision,
    canonical_sha256,
    main,
    validate_contract,
)


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "configs/research/neural_swarm_planner_first_curriculum_v1.json"
CONTRACT_SCHEMA = (
    REPO / "configs/research/neural_swarm_planner_first_curriculum_v1.schema.json"
)
RUNTIME_SCHEMA = (
    REPO / "configs/research/neural_swarm_planner_first_runtime_bundle_v1.schema.json"
)
DECISION_SCHEMA = (
    REPO / "configs/research/neural_swarm_planner_first_decision_v1.schema.json"
)
LEGACY_CONFIG = (
    REPO
    / "configs/training/gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.yaml"
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _bundle() -> dict:
    training = {
        "schema_version": "anchor.neural-swarm-planner-training-receipt.v1",
        "stage": "planner_train",
        "status": "passed",
        "arm": "planner_q_only",
        "source_asset": "planner_router",
        "run_id_sha256": _sha("planner-run"),
        "base_model_sha256": _sha("base"),
        "tokenizer_sha256": _sha("tokenizer"),
        "chat_template_sha256": _sha("template"),
        "training_record_inventory_sha256": _sha("planner-records"),
        "adapter_sha256": _sha("planner-adapter"),
        "completed_steps": 80,
        "training_authorized": False,
        "formal": False,
    }
    training_sha = canonical_sha256(training)
    freeze = {
        "schema_version": "anchor.neural-swarm-planner-freeze-receipt.v1",
        "stage": "planner_freeze",
        "status": "passed",
        "planner_training_receipt_sha256": training_sha,
        "planner_adapter_sha256": training["adapter_sha256"],
        "frozen_parameter_inventory_sha256": _sha("frozen-parameters"),
        "adapter_frozen": True,
        "mutation_after_freeze_allowed": False,
    }
    freeze_sha = canonical_sha256(freeze)
    rows = [
        build_question_row(
            planner_freeze_receipt_sha256=freeze_sha,
            planner_adapter_sha256=training["adapter_sha256"],
            ordinal=ordinal,
            expert_asset=expert,
            language="en" if ordinal % 2 == 0 else "zh-CN",
            source_record_id_sha256=_sha(f"source-{expert}"),
            question_semantic_sha256=_sha(f"semantic-{expert}"),
            question_payload_sha256=_sha(f"payload-{expert}"),
        )
        for ordinal, expert in enumerate(EXPERT_ASSETS)
    ]
    manifest = {
        "schema_version": "anchor.neural-swarm-planner-question-manifest.v1",
        "stage": "planner_question_commit",
        "status": "passed",
        "planner_freeze_receipt_sha256": freeze_sha,
        "planner_adapter_sha256": training["adapter_sha256"],
        "question_count": len(rows),
        "ordered_question_ids_sha256": canonical_sha256(
            [row["question_id_sha256"] for row in rows]
        ),
        "rows": rows,
    }
    generation = {
        "schema_version": (
            "anchor.neural-swarm-planner-question-generation-receipt.v1"
        ),
        "stage": "planner_question_commit",
        "status": "passed",
        "generation_mode": "frozen_planner_inference",
        "generation_run_id_sha256": _sha("question-generation-run"),
        "generator_implementation_sha256": _sha("question-generator"),
        "planner_freeze_receipt_sha256": freeze_sha,
        "planner_adapter_sha256": training["adapter_sha256"],
        "question_manifest_sha256": canonical_sha256(manifest),
        "model_request_count": len(rows),
        "provider_request_count": 0,
        "raw_hidden_reasoning_persisted": False,
        "training_authorized": False,
        "formal": False,
    }
    return {
        "schema_version": "anchor.neural-swarm-planner-first-runtime-bundle.v1",
        "planner_training_receipt": training,
        "planner_freeze_receipt": freeze,
        "planner_question_manifest": manifest,
        "planner_question_generation_receipt": generation,
    }


def test_current_runtime_is_serial_but_not_planner_first() -> None:
    value = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    observed = tuple(value["arms"]["ordered"])
    assert observed == OBSERVED_LEGACY_ORDER
    assert observed[:5] == (
        "humor_q_only",
        "serious_q_only",
        "angry_style_q_only",
        "review_audit_q_only",
        "tool_q_only",
    )
    assert observed.index("planner_q_only") == 6
    assert value["training"]["strictly_serial"] is True
    assert value["training"]["concurrency"] == 1


def test_contract_and_runtime_schemas_are_closed_draft_202012() -> None:
    contract = _contract()
    bundle = _bundle()
    for path, value in (
        (CONTRACT_SCHEMA, contract),
        (RUNTIME_SCHEMA, bundle),
    ):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    result = validate_contract(contract, repo_root=REPO)
    assert result["status"] == "passed"
    assert result["legacy_runtime_planner_first"] is False
    assert result["planner_first_additive_gate_required"] is True
    assert result["existing_training_runtime_modified"] is False


def test_valid_chain_releases_five_serial_leaf_preflight_tickets() -> None:
    decision = build_runtime_decision(_contract(), _bundle(), repo_root=REPO)
    schema = json.loads(DECISION_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(decision)

    assert decision["schema_version"] == DECISION_VERSION
    assert decision["status"] == "expert_leaf_preflight_ready"
    assert decision["stage"] == "expert_leaf_release"
    assert decision["expert_execution_policy"] == {
        "strictly_serial": True,
        "max_concurrency": 1,
        "planner_reentry_after_expert_start": False,
    }
    tickets = decision["expert_tickets"]
    assert tuple(ticket["expert_asset"] for ticket in tickets) == EXPERT_ASSETS
    assert tuple(ticket["leaf_runner_arm"] for ticket in tickets) == EXPERT_LEAF_ARMS
    assert [ticket["execution_order"] for ticket in tickets] == list(range(5))
    assert all(ticket["question_count"] == 1 for ticket in tickets)
    assert all(ticket["leaf_preflight_eligible"] is True for ticket in tickets)
    assert all(ticket["gpu_authorized"] is False for ticket in tickets)
    assert decision["existing_training_runtime_modified"] is False
    assert decision["planner_first_leaf_launcher_integrated"] is False
    assert decision["expert_training_execution_blocked"] is True
    assert decision["gpu_authorized"] is False
    assert decision["training_authorized"] is False
    assert decision["formal"] is False


def test_questions_are_bound_to_the_frozen_planner_and_body_free() -> None:
    bundle = _bundle()
    freeze_sha = canonical_sha256(bundle["planner_freeze_receipt"])
    adapter_sha = bundle["planner_training_receipt"]["adapter_sha256"]
    manifest = bundle["planner_question_manifest"]
    assert manifest["planner_freeze_receipt_sha256"] == freeze_sha
    assert manifest["planner_adapter_sha256"] == adapter_sha
    for row in manifest["rows"]:
        assert "question_text" not in row
        assert "prompt" not in row
        assert "answer" not in row
        assert set(row) == {
            "question_id_sha256",
            "ordinal",
            "expert_asset",
            "language",
            "split",
            "source_record_id_sha256",
            "question_semantic_sha256",
            "question_payload_sha256",
            "route_commit_sha256",
        }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value["planner_training_receipt"].update(
                {"arm": "planner_o_only"}
            ),
            "planner_training_receipt_semantics_invalid",
        ),
        (
            lambda value: value["planner_freeze_receipt"].update(
                {"adapter_frozen": False}
            ),
            "planner_adapter_must_be_frozen",
        ),
        (
            lambda value: value["planner_freeze_receipt"].update(
                {"planner_training_receipt_sha256": _sha("forged")}
            ),
            "planner_freeze_training_receipt_binding_mismatch",
        ),
        (
            lambda value: value["planner_question_manifest"].update(
                {"planner_freeze_receipt_sha256": _sha("forged")}
            ),
            "question_manifest_freeze_binding_mismatch",
        ),
        (
            lambda value: value["planner_question_generation_receipt"].update(
                {"question_manifest_sha256": _sha("forged")}
            ),
            "question_generation_manifest_binding_mismatch",
        ),
        (
            lambda value: value["planner_question_generation_receipt"].update(
                {"generation_mode": "static_fixture"}
            ),
            "planner_question_generation_receipt_semantics_invalid",
        ),
        (
            lambda value: value["planner_question_manifest"]["rows"][0].update(
                {"expert_asset": "serious"}
            ),
            "question_row_identity_mismatch",
        ),
        (
            lambda value: value["planner_question_manifest"]["rows"].pop(),
            "question_manifest_count_mismatch",
        ),
    ],
)
def test_dependency_bypasses_fail_closed(mutation, error: str) -> None:
    bundle = _bundle()
    mutation(bundle)
    with pytest.raises(PlannerFirstCurriculumError, match=error):
        build_runtime_decision(_contract(), bundle, repo_root=REPO)


def test_missing_expert_coverage_fails_even_with_self_reported_count_repaired() -> None:
    bundle = _bundle()
    manifest = bundle["planner_question_manifest"]
    removed = manifest["rows"].pop()
    manifest["question_count"] = len(manifest["rows"])
    manifest["ordered_question_ids_sha256"] = canonical_sha256(
        [row["question_id_sha256"] for row in manifest["rows"]]
    )
    with pytest.raises(
        PlannerFirstCurriculumError,
        match=f"question_manifest_{removed['expert_asset']}_coverage_missing",
    ):
        build_runtime_decision(_contract(), bundle, repo_root=REPO)


def test_question_identity_cannot_be_reused_after_route_or_payload_change() -> None:
    bundle = _bundle()
    row = bundle["planner_question_manifest"]["rows"][0]
    row["question_payload_sha256"] = _sha("different-payload")
    with pytest.raises(
        PlannerFirstCurriculumError,
        match="question_row_identity_mismatch",
    ):
        build_runtime_decision(_contract(), bundle, repo_root=REPO)


def test_frozen_leaf_runtime_pin_drift_fails_closed() -> None:
    contract = copy.deepcopy(_contract())
    contract["frozen_leaf_runtime"]["runtime_implementation"]["sha256"] = _sha("forged")
    with pytest.raises(
        PlannerFirstCurriculumError,
        match="runtime_implementation_identity_drift",
    ):
        validate_contract(contract, repo_root=REPO)


def test_cli_without_planner_receipt_is_blocked_and_does_not_start_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "planner-first",
            str(CONTRACT),
            "--repo-root",
            str(REPO),
        ],
    )
    assert main() == 2
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "blocked_planner_training_receipt_required"
    assert value["blockers"] == ["planner_training_receipt_required"]
    assert value["gpu_authorized"] is False
    assert value["training_authorized"] is False


def test_existing_training_runtime_physical_bytes_remain_frozen() -> None:
    expected = {
        "configs/training/gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.yaml": (
            "7ab575e1b897fcf46812208730482a4733d4f86c2548afae3cc2f9a9a5417947"
        ),
        "src/anchor_mvp/training/gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.py": (
            "452798bf4f63f72f600093df1a9abe4c5a1bf5f24c430e23044546ccb3214154"
        ),
        "src/anchor_mvp/training/gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1.py": (
            "8cdf360a102eca9e8ff45780486337581263e2e24c88a6e29dc92f6ec2a049ae"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == digest


def test_hierarchical_dependencies_bind_canonical_lf_git_bytes() -> None:
    contract = _contract()
    dependency = contract["hierarchical_dependency"]
    for key in ("contract", "implementation"):
        relative = dependency[key]["path"]
        raw = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw).hexdigest() == dependency[key]["sha256"]
        attrs = subprocess.check_output(
            ["git", "check-attr", "text", "eol", "--", relative],
            cwd=REPO,
            text=True,
            encoding="utf-8",
        )
        assert "text: set" in attrs
        assert "eol: lf" in attrs
