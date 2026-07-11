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
    BatchStageResult,
    batch_run_succeeded,
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
    stages = run_live_batch(
        samples=samples, config=config, executor=executor, max_stages=4
    )

    assert len(stages) == 2
    assert sum(len(stage.records) for stage in stages) == 3
    failed = next(record for stage in stages for record in stage.records if record.sample_id == "s-1")
    assert failed.success is False
    assert failed.error_codes == ("isolated_sample_exception", "public_outcome_missing")
    assert sum(record.success for stage in stages for record in stage.records) == 2


def test_live_batch_defaults_to_only_the_first_ramp_stage(tmp_path):
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
        samples_per_stage=(1, 2, 4, 8),
        minimum_stage_success_rate=1.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"safe-{index}", "task", source) for index in range(15))

    stages = run_live_batch(
        samples=samples,
        config=config,
        executor=MockAgentExecutor(public_outcome=outcome),
    )

    assert len(stages) == 1
    assert stages[0].concurrency == 1
    assert len(stages[0].records) == 1


def test_live_batch_runs_only_explicitly_requested_stage_slice(tmp_path):
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
        samples_per_stage=(1, 2, 4, 8),
        minimum_stage_success_rate=1.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"slice-{index}", "task", source) for index in range(15))

    stages = run_live_batch(
        samples=samples,
        config=config,
        executor=MockAgentExecutor(public_outcome=outcome),
        max_stages=2,
    )

    assert [stage.concurrency for stage in stages] == [1, 2]
    assert sum(len(stage.records) for stage in stages) == 3
    assert batch_run_succeeded(stages, requested_stages=2) is True
    assert batch_run_succeeded(stages, requested_stages=4) is False


def test_batch_success_semantics_require_every_requested_gate():
    passed = BatchStageResult(1, (), True)
    failed = BatchStageResult(2, (), False)

    assert batch_run_succeeded((passed,), requested_stages=1) is True
    assert batch_run_succeeded((passed,), requested_stages=2) is False
    assert batch_run_succeeded((passed, failed), requested_stages=2) is False


@pytest.mark.parametrize("value", [0, 5])
def test_live_batch_rejects_out_of_range_stage_limit(tmp_path, value):
    config = LiveBatchConfig(
        candidate_manifest=tmp_path / "unused",
        split_policy=tmp_path / "unused",
        skill_registry=tmp_path / "unused",
        workspace_root=tmp_path / "runs",
        gold_output=tmp_path / "gold.jsonl",
    )

    with pytest.raises(ValueError, match="max_stages"):
        run_live_batch(
            samples=(),
            config=config,
            executor=MockAgentExecutor(),
            max_stages=value,
        )
