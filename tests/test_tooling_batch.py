import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from anchor_mvp.tooling import (
    ControlledSessionCapture,
    LiveBatchConfig,
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SkillSourceRegistry,
    BatchStageResult,
    ToolPolicy,
    batch_run_succeeded,
    build_opencode_config,
    load_candidate_samples,
    run_live_batch,
    verify_execution_split,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_batch_preflight_defaults_to_one_sandboxed_stage_and_no_heldout_collision():
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

    assert config.concurrency_stages == (1,)
    assert config.samples_per_stage == (1,)
    assert config.anchor_sandbox_options().memory == "4G"
    assert config.anchor_sandbox_options().linux_executable == (
        ROOT / "artifacts/tooling/opencode-patched/linux-x64/opencode-anchor"
    )
    assert config.anchor_sandbox_options().wsl_distro == "Ubuntu-22.04"
    assert config.anchor_sandbox_options().supervisor == "wsl-root-systemd"
    assert config.retain_workspace is False
    assert config.max_iterations is None
    generated = build_opencode_config(
        ToolPolicy(
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
        )
    )
    assert "steps" not in generated["agent"]["anchor-distiller"]
    assert config.attempts_output == ROOT / "artifacts/tooling/live_attempts.jsonl"
    assert config.opencode_executable == (
        ROOT / "artifacts/tooling/opencode-patched/opencode-anchor.exe"
    )
    assert len(samples) == 1
    assert samples[0].sample_id == "sidex-p0-001-stable-status-sort"
    assert samples[0].requires_changes is True
    assert "Stable status-list sorting" in samples[0].prompt
    assert {path for path, _ in samples[0].protected_files} == {
        "TASK.md",
        "package.json",
        "scripts/build.mjs",
        "scripts/lint.mjs",
        "test/status-list.test.js",
    }
    assert {path for path, _ in samples[0].input_files} == {"src/status-list.js"}


def test_ark_batch_profile_has_independent_outputs_and_responses_provider():
    kimi = LiveBatchConfig.load(
        ROOT, ROOT / "configs/tooling/opencode_distillation_ramp.yaml"
    )
    ark = LiveBatchConfig.load(
        ROOT, ROOT / "configs/tooling/opencode_distillation_ramp.ark_glm52.yaml"
    )

    assert ark.provider.provider_id == "anchor-ark-glm52"
    assert ark.provider.npm == "@ai-sdk/openai"
    assert ark.provider.model == "glm-5-2-260617"
    assert ark.provider.variant == "max"
    assert ark.provider.key_env == "ARK_CODING_API_KEY"
    assert ark.provider.route_host == "ark.cn-beijing.volces.com"
    assert ark.attempts_output != kimi.attempts_output
    assert ark.session_staging != kimi.session_staging
    assert ark.session_candidates != kimi.session_candidates
    assert ark.session_quarantine != kimi.session_quarantine


def test_stage_one_fixture_rejects_bug_and_accepts_the_required_repair(tmp_path):
    source = ROOT / "fixtures/execution/sidex-fixture-v0-status-list"
    fixture = tmp_path / "fixture"
    shutil.copytree(source, fixture)
    npm = shutil.which("npm")
    assert npm is not None

    initial = subprocess.run(
        [npm, "run", "test"],
        cwd=fixture,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert initial.returncode != 0
    assert "sorts numeric priority ascending without mutation" in initial.stdout

    (fixture / "src/status-list.js").write_text(
        """export function sortStatusRows(rows, direction) {
  if (direction !== "asc" && direction !== "desc") {
    throw new TypeError("direction must be asc or desc");
  }
  const sign = direction === "asc" ? 1 : -1;
  return rows
    .map((row, index) => ({ row, index }))
    .sort((left, right) =>
      sign * (Number(left.row.priority) - Number(right.row.priority)) ||
      left.index - right.index
    )
    .map(({ row }) => row);
}
""",
        encoding="utf-8",
    )
    for script in ("build", "test", "lint"):
        result = subprocess.run(
            [npm, "run", script],
            cwd=fixture,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


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
                "heldout_inputs": [{"path": "heldout.jsonl", "sha256": "0" * 64}],
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
        concurrency_stages=(1, 2, 5),
        samples_per_stage=(1, 2, 1),
        minimum_stage_success_rate=0.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"s-{index}", "task", source) for index in range(4))
    delegate = MockAgentExecutor(public_outcome=outcome)

    class OneFailureExecutor:
        backend_name = "one-failure-mock"

        def run(self, **kwargs):
            if kwargs["sample_id"] == "s-1":
                raise RuntimeError("synthetic isolated failure")
            return delegate.run(**kwargs)

    executor = OneFailureExecutor()
    stages = run_live_batch(
        samples=samples, config=config, executor=executor, max_stages=3
    )

    assert len(stages) == 2
    assert sum(len(stage.records) for stage in stages) == 3
    failed = next(
        record
        for stage in stages
        for record in stage.records
        if record.sample_id == "s-1"
    )
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
        concurrency_stages=(1, 3, 9),
        samples_per_stage=(1, 2, 3),
        minimum_stage_success_rate=1.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"safe-{index}", "task", source) for index in range(6))

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
        concurrency_stages=(1, 3, 9),
        samples_per_stage=(1, 2, 3),
        minimum_stage_success_rate=1.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(SampleSpec(f"slice-{index}", "task", source) for index in range(6))

    stages = run_live_batch(
        samples=samples,
        config=config,
        executor=MockAgentExecutor(public_outcome=outcome),
        max_stages=2,
    )

    assert [stage.concurrency for stage in stages] == [1, 3]
    assert sum(len(stage.records) for stage in stages) == 3
    assert batch_run_succeeded(stages, requested_stages=2) is True
    assert batch_run_succeeded(stages, requested_stages=4) is False


def test_batch_success_semantics_require_every_requested_gate():
    passed = BatchStageResult(1, (), True)
    failed = BatchStageResult(2, (), False)

    assert batch_run_succeeded((passed,), requested_stages=1) is True
    assert batch_run_succeeded((passed,), requested_stages=2) is False
    assert batch_run_succeeded((passed, failed), requested_stages=2) is False


def test_collection_stage_does_not_apply_strict_quality_success_rate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "package.json").write_text(
        json.dumps({"scripts": {"build": 'node -e "process.exit(0)"'}}),
        encoding="utf-8",
    )
    config = LiveBatchConfig(
        candidate_manifest=tmp_path / "unused",
        split_policy=tmp_path / "unused",
        skill_registry=tmp_path / "unused",
        workspace_root=tmp_path / "runs",
        gold_output=tmp_path / "gold.jsonl",
        concurrency_stages=(2,),
        samples_per_stage=(2,),
        minimum_stage_success_rate=1.0,
    )
    from anchor_mvp.tooling import SampleSpec

    samples = tuple(
        SampleSpec(f"collect-{index}", "task", source) for index in range(2)
    )
    capture = ControlledSessionCapture(
        candidates_path=(tmp_path / "candidates.jsonl").resolve(),
        quarantine_path=(tmp_path / "quarantine.jsonl").resolve(),
        heldout_cases=(tmp_path / "heldout.jsonl").resolve(),
        heldout_fixtures_root=(tmp_path / "heldout-fixtures").resolve(),
        heldout_manifest=(tmp_path / "heldout-manifest.json").resolve(),
        staging_path=(tmp_path / "staging.jsonl").resolve(),
        mode="collect",
    )

    class CollectExecutor(MockAgentExecutor):
        session_capture = capture

        def finalize_capture(self, **kwargs):
            return True, None

    stages = run_live_batch(
        samples=samples,
        config=config,
        executor=CollectExecutor(exit_code=1),
        collection_mode=True,
    )

    assert all(record.success is False for record in stages[0].records)
    assert stages[0].passed_gate is True


@pytest.mark.parametrize("value", [0, 2])
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


def test_live_batch_accepts_operator_selected_positive_stage_values(tmp_path):
    config = LiveBatchConfig(
        candidate_manifest=tmp_path / "unused",
        split_policy=tmp_path / "unused",
        skill_registry=tmp_path / "unused",
        workspace_root=tmp_path / "runs",
        gold_output=tmp_path / "gold.jsonl",
        concurrency_stages=(1, 3, 17),
        samples_per_stage=(1, 1, 1),
    )

    assert config.concurrency_stages == (1, 3, 17)
    with pytest.raises(ValueError, match="positive integers"):
        LiveBatchConfig(
            candidate_manifest=tmp_path / "unused",
            split_policy=tmp_path / "unused",
            skill_registry=tmp_path / "unused",
            workspace_root=tmp_path / "runs",
            gold_output=tmp_path / "gold.jsonl",
            concurrency_stages=(1, 0),
            samples_per_stage=(1, 1),
        )
