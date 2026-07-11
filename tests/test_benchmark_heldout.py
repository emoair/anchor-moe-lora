import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from anchor_mvp.benchmark import (
    HeldoutBenchmarkRunner,
    HeldoutGateError,
    check_training_leakage,
    load_cases_jsonl,
    load_specs,
    verify_heldout_manifest,
)
from anchor_mvp.benchmark.heldout_mock import heldout_mock_handler
from anchor_mvp.benchmark.heldout import validate_primary_specs
from anchor_mvp.serving import MockBackend


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl"
FIXTURES = ROOT / "examples" / "benchmark" / "fixtures"
MANIFEST = ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json"


def test_frozen_manifest_and_sidecar_verify():
    digest = verify_heldout_manifest(CASES, FIXTURES, MANIFEST)

    assert digest == "1ac7240d700a67458dc713b66ff085f1e51795b26cdacff688063bc60af3194c"


def test_leak_checker_fails_closed_without_emitting_content(tmp_path):
    requirement = load_cases_jsonl(CASES)[0].requirement
    training = tmp_path / "training.jsonl"
    training.write_text(
        json.dumps({"id": "private", "input": {"requirement": requirement}}) + "\n",
        encoding="utf-8",
    )

    audit = check_training_leakage(CASES, FIXTURES, MANIFEST, [training])

    serialized = json.dumps(audit)
    assert audit["status"] == "FAIL"
    assert audit["collision_count"] >= 1
    assert audit["content_emitted"] is False
    assert requirement not in serialized
    assert "private" not in serialized


def test_manifest_detects_case_change(tmp_path):
    changed = tmp_path / "cases.jsonl"
    changed.write_text(CASES.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(HeldoutGateError, match="changed after freeze"):
        verify_heldout_manifest(changed, FIXTURES, MANIFEST)


def test_mock_primary_arms_use_five_expert_types_with_bounded_review_attempts():
    specs = load_specs(ROOT / "configs" / "benchmark" / "heldout_q4_v1.json")
    cases = load_cases_jsonl(CASES)[:1]
    models = {model for spec in specs for model in spec.stage_models.values()}
    backend = MockBackend(handlers={model: heldout_mock_handler for model in models})
    records = asyncio.run(
        HeldoutBenchmarkRunner(
            backend,
            sample_vram=False,
            backend_label="mock",
            manifest_sha256=verify_heldout_manifest(CASES, FIXTURES, MANIFEST),
        ).run_suite(specs, cases)
    )

    assert [record.baseline for record in records] == [
        "base_matched_calls",
        "mixed_matched_calls",
        "c_pipeline",
        "d_budget_matched_pipeline",
    ]
    assert all(record.call_count == 7 for record in records)
    assert all(
        [stage["stage"] for stage in record.stages]
        == ["planner", "tool_policy", "frontend", "review", "frontend", "review", "security"]
        for record in records
    )
    assert all(
        list(dict.fromkeys(stage["stage"] for stage in record.stages))
        == ["planner", "tool_policy", "frontend", "review", "security"]
        for record in records
    )
    assert all(record.review_repair_pass is True for record in records)
    assert all(record.tool_policy_decision == "APPROVE" for record in records)
    assert all(record.deterministic_tool_policy_decision == "APPROVE" for record in records)
    assert all(record.evaluation["model_tool_policy_was_executed"] is False for record in records)


def test_live_primary_gate_accepts_exported_q4_artifact_digest():
    specs = load_specs(ROOT / "configs" / "benchmark" / "heldout_q4_v1.json")

    validate_primary_specs(specs, require_verified_q4=True)
    assert {spec.q4_artifact_sha256 for spec in specs} == {
        "96ebac04f4d2c64d4b21142bb6e05d94656c3c7fb243fdbd43c4b4457eca0156"
    }


def test_budget_matched_routed_arm_exactly_matches_mixed_parameter_count():
    specs = load_specs(ROOT / "configs" / "benchmark" / "heldout_q4_budget_v1.json")

    by_name = {spec.name: spec for spec in specs}
    mixed = by_name["mixed_matched_calls"]
    budget = by_name["d_budget_matched_pipeline"]
    full = by_name["c_pipeline"]
    assert budget.adapter_trainable_parameters == mixed.adapter_trainable_parameters
    assert sum(budget.stage_adapter_ranks.values()) == 16
    assert tuple(budget.stage_adapter_ranks.values()) == (3, 3, 4, 3, 3)
    assert full.adapter_trainable_parameters == 5 * mixed.adapter_trainable_parameters
    assert set(by_name) == {
        "base_matched_calls",
        "mixed_matched_calls",
        "c_pipeline",
        "d_budget_matched_pipeline",
        "e_adaptive_pareto_pipeline",
        "f_adaptive_budget_matched_pipeline",
    }
    with pytest.raises(HeldoutGateError, match="calibration-selected and frozen"):
        validate_primary_specs(specs, require_verified_q4=True)


def test_frozen_adaptive_arms_enforce_shared_mechanism_and_exact_f_budget():
    specs = load_specs(ROOT / "configs" / "benchmark" / "heldout_q4_budget_v1.json")
    frozen = []
    for spec in specs:
        if spec.name == "e_adaptive_pareto_pipeline":
            spec = replace(
                spec,
                status="ready",
                allocation_frozen=True,
                allocation_manifest_sha256="a" * 64,
                adapter_trainable_parameters=24_670_208,
                stage_adapter_ranks={
                    "planner": 4,
                    "tool_policy": 2,
                    "frontend": 16,
                    "review": 12,
                    "security": 4,
                },
            )
        elif spec.name == "f_adaptive_budget_matched_pipeline":
            spec = replace(
                spec,
                status="ready",
                allocation_frozen=True,
                allocation_manifest_sha256="b" * 64,
                stage_adapter_ranks={
                    "planner": 2,
                    "tool_policy": 2,
                    "frontend": 6,
                    "review": 4,
                    "security": 2,
                },
            )
        frozen.append(spec)

    validate_primary_specs(frozen, require_verified_q4=True)

    invalid = [
        replace(spec, stage_adapter_ranks={**spec.stage_adapter_ranks, "security": 3})
        if spec.name == "f_adaptive_budget_matched_pipeline"
        else spec
        for spec in frozen
    ]
    with pytest.raises(HeldoutGateError, match="sum exactly"):
        validate_primary_specs(invalid, require_verified_q4=True)
