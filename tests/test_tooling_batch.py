import json
from pathlib import Path

import pytest
import yaml

from anchor_mvp.tooling import (
    LiveBatchConfig,
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SkillSourceRegistry,
    load_candidate_samples,
    run_live_batch,
    verify_execution_split,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_batch_preflight_has_strict_ramp_and_no_heldout_collision():
    config = LiveBatchConfig.load(
        ROOT, ROOT / "configs/tooling/opencode_distillation_ramp.yaml"
    )
    registry = SkillSourceRegistry(ROOT, config.skill_registry)
    identifiers, requirements = verify_execution_split(
        ROOT, config.split_policy, config.candidate_manifest
    )

    samples = load_candidate_samples(
        ROOT,
        config.candidate_manifest,
        registry,
        heldout_identifiers=identifiers,
        heldout_requirements=requirements,
    )

    assert config.concurrency_stages == (1, 2, 4, 8)
    assert len(samples) == 15
    assert len({sample.sample_id for sample in samples}) == 15


def test_split_policy_rejects_changed_heldout_input(tmp_path):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("tasks: []\n", encoding="utf-8")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"case_id": "heldout-1"}) + "\n", encoding="utf-8")
    policy = tmp_path / "split.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": "anchor.execution-split-policy.v1",
                "candidate_inputs": ["candidate.yaml"],
                "heldout_inputs": [
                    {"path": "heldout.jsonl", "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="held-out input hash mismatch"):
        verify_execution_split(tmp_path, policy, candidate)


def test_failure_in_one_sample_is_isolated_from_siblings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "package.json").write_text(
        json.dumps({"scripts": {"build": 'node -e "process.exit(0)"'}}),
        encoding="utf-8",
    )
    outcome = PublicOutcome(
        status="completed",
        decision_trace=(PublicDecisionStep("check", "evidence", "action"),),
        repair_summaries=(),
        final_summary="done",
    )
    config = LiveBatchConfig(
        candidate_manifest=tmp_path / "unused",
        split_policy=tmp_path / "unused",
        skill_registry=tmp_path / "unused",
        workspace_root=tmp_path / "runs",
        gold_output=tmp_path / "gold.jsonl",
        concurrency_stages=(1, 2, 4, 8),
        samples_per_stage=(1, 2, 1, 1),
        minimum_stage_success_rate=0.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"s-{index}", "task", source) for index in range(5))
    delegate = MockAgentExecutor(public_outcome=outcome)

    class OneFailureExecutor:
        backend_name = "one-failure-mock"

        def run(self, **kwargs):
            if kwargs["sample_id"] == "s-1":
                raise RuntimeError("synthetic isolated failure")
            return delegate.run(**kwargs)

    executor = OneFailureExecutor()
    stages = run_live_batch(samples=samples, config=config, executor=executor)

    assert len(stages) == 2
    assert sum(len(stage.records) for stage in stages) == 3
    failed = next(record for stage in stages for record in stage.records if record.sample_id == "s-1")
    assert failed.success is False
    assert failed.error_codes == ("isolated_sample_exception", "public_outcome_missing")
    assert sum(record.success for stage in stages for record in stage.records) == 2
