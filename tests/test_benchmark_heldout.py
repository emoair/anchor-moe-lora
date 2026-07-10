import asyncio
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


def test_mock_primary_arms_are_exactly_five_stage_matched():
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
    ]
    assert all(record.call_count == 5 for record in records)
    assert all(
        [stage["stage"] for stage in record.stages]
        == ["planner", "tool_policy", "frontend", "review", "security"]
        for record in records
    )
    assert all(record.tool_policy_decision == "APPROVE" for record in records)
    assert all(record.deterministic_tool_policy_decision == "APPROVE" for record in records)
    assert all(record.evaluation["model_tool_policy_was_executed"] is False for record in records)
