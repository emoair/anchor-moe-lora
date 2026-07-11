from __future__ import annotations

import copy
from pathlib import Path

import pytest

from anchor_mvp.curriculum import (
    CurriculumValidationError,
    expected_stage_flow,
    validate_curriculum,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/curriculum/collaboration_v2.yaml"


def test_curriculum_has_15_unique_continuous_candidates():
    manifest = validate_curriculum(MANIFEST, ROOT)

    assert len(manifest["tasks"]) == 15
    assert len({task["seed_id"] for task in manifest["tasks"]}) == 15
    assert {task["domain"] for task in manifest["tasks"]} >= {
        "frontend-web",
        "python-cli",
        "node-ts-utility",
        "code-repair",
        "accessibility-ui",
        "security-inert",
    }
    assert any(task["max_cycles"] == 2 for task in manifest["tasks"])


def test_stage_flow_carries_verified_builder_artifacts_to_review_and_safety():
    stages = expected_stage_flow(2)

    assert stages[4]["input_refs"][-1] == "domain_review_1.output"
    assert "builder_2.tool_trace" in stages[-1]["input_refs"]
    assert "builder_2.diff" in stages[-1]["input_refs"]
    assert "builder_2.validators" in stages[-1]["input_refs"]
    assert stages[-1]["input_refs"][-1] == "domain_review_2.output"


def test_flow_rejects_independent_stage_prompt_splicing():
    manifest = validate_curriculum(MANIFEST, ROOT)
    task = copy.deepcopy(manifest["tasks"][0])
    task["stage_flow"][2]["input_refs"] = ["context"]

    assert task["stage_flow"] != expected_stage_flow(
        task["max_cycles"], task["required_builder"], task["reviewer"]
    )


def test_flow_rejects_unsupported_unbounded_revision_loop():
    with pytest.raises(CurriculumValidationError, match="max_cycles"):
        expected_stage_flow(3)
