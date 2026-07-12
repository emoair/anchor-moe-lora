from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.automation import (  # noqa: E402
    AutomationConfig,
    AutomationRunner,
    _build_teachers,
    chargeable_failure_count,
    evaluate_gate,
    evaluate_heldout_scale_gate,
    main,
    partition_collected_records,
)
from anchor_mvp.data.cli import _simple_config  # noqa: E402
from anchor_mvp.data.pipeline import DistillationPipeline  # noqa: E402
from anchor_mvp.data.teacher import (  # noqa: E402
    ClientDeadlineExceeded,
    MockTeacher,
    RateLimitError,
)
from anchor_mvp.training.schema import validate_jsonl  # noqa: E402


def config(tmp_path: Path, **overrides) -> AutomationConfig:
    values = {
        "sop_dir": ROOT / "skills",
        "output_dir": tmp_path,
        "concurrency_stages": (1,),
        "stage_seed_counts": (1,),
        "min_success_rate": 1.0,
        "max_duplicate_rate": 0.0,
        "max_safety_violations": 0,
        "max_failures": 0,
        "max_requests": 40,
        "max_output_tokens_total": 100_000,
        "cooldown_seconds": 300,
        "cooldown_poll_seconds": 1,
    }
    values.update(overrides)
    return AutomationConfig(**values)


def test_mock_automation_defaults_to_one_gated_stage_and_resumes(tmp_path: Path) -> None:
    settings = config(tmp_path)
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "complete"
    assert [stage["concurrency"] for stage in status["stages"]] == [1]
    assert all(stage["gate"]["passed"] for stage in status["stages"])
    assert status["metrics"]["records"] == 5
    assert status["metrics"]["throughput_records_per_second"] > 0
    assert status["metrics"]["eta_seconds"] == 0
    assert status["quota_epoch"]["requests_used"] <= 40
    assert status["audit_ledger"]["requests_total"] == status["quota_epoch"]["requests_used"]
    assert settings.status_path.is_file()

    experts = {
        "plan": "planner",
        "tool_policy": "tool_policy",
        "frontend": "frontend_gen",
        "review": "frontend_review",
        "security": "security_gate",
    }
    for task, expert in experts.items():
        report = validate_jsonl(tmp_path / f"data_{task}.jsonl", allowed_experts=[expert])
        assert report["valid_records"] == 1

    events_before = settings.events_path.read_text(encoding="utf-8").splitlines()
    event_types = [json.loads(line)["type"] for line in events_before]
    assert event_types.count("stage_started") == 1
    assert event_types.count("gate_passed") == 1
    assert event_types[-1] == "automation_completed"

    resumed = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())
    assert resumed["state"] == "complete"
    assert settings.events_path.read_text(encoding="utf-8").splitlines() == events_before
    assert all(len((tmp_path / f"data_{task}.jsonl").read_text(encoding="utf-8").splitlines()) == 1 for task in experts)


def test_gate_detects_duplicates_and_training_schema_failure(tmp_path: Path) -> None:
    settings = config(tmp_path)
    asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())
    frontend = tmp_path / "data_frontend.jsonl"
    first = frontend.read_text(encoding="utf-8").splitlines()[0]
    with frontend.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    gate = evaluate_gate(settings, 1)
    assert gate["passed"] is False
    assert gate["duplicate_count"] >= 1
    assert gate["duplicate_rate"] > 0
    assert gate["training_schema_ok"] is False


def test_collect_first_does_not_retry_soft_quality_gate(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        collection_policy="collect_then_partition",
        minimum_label_counts={"tool_policy": {"BLOCK": 2}},
        max_stagnant_gate_rounds=1,
    )
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "complete"
    assert status["stage_index"] == 1
    assert status["partition"]["training_ready"] is False
    assert status["partition"]["label_quota_errors"]
    events = [
        json.loads(line)["type"]
        for line in settings.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("stage_started") == 1
    assert "gate_retry_scheduled" not in events
    assert "collection_partitioned" in events


def test_collect_first_partitions_partial_output_when_budget_closes(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        collection_policy="collect_then_partition",
        max_requests=1,
        max_failures=20,
    )
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "budget_exhausted"
    assert status["partition"]["training_ready"] is False
    assert settings.quality_staging_path.is_file()
    assert (settings.partition_dir / "manifest.json").is_file()


def test_offline_partition_retains_quality_negative_and_content_free_reject(
    tmp_path: Path,
) -> None:
    settings = config(tmp_path, collection_policy="collect_then_partition")
    _distill_for_gate(settings, seed_count=1)
    frontend = settings.output_dir / "data_frontend.jsonl"
    source = json.loads(frontend.read_text(encoding="utf-8").splitlines()[0])
    duplicate = json.loads(json.dumps(source))
    secret = json.loads(json.dumps(source))
    secret["id"] = "secret-bearing-record"
    secret["output"]["code"] += "\n// api_key=sk-example-secret-value-123456"
    with frontend.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
        handle.write(json.dumps(secret, ensure_ascii=False) + "\n")
    tool_policy = settings.output_dir / "data_tool_policy.jsonl"
    tool_record = json.loads(tool_policy.read_text(encoding="utf-8").splitlines()[0])
    authoritative = tool_record["output"]["decision"]
    tool_record["provenance"]["teacher_observed_decision"] = (
        "BLOCK" if authoritative != "BLOCK" else "APPROVE"
    )
    _write_jsonl(tool_policy, [tool_record])

    manifest = partition_collected_records(settings, 1)

    assert manifest["negative_count"] >= 3
    assert manifest["reject_count"] == 1
    negatives = settings.partition_dir / "negative.jsonl"
    assert "duplicate_record_id" in negatives.read_text(encoding="utf-8")
    assert "teacher_label_disagreement" in negatives.read_text(encoding="utf-8")
    reject = (settings.partition_dir / "reject.jsonl").read_text(encoding="utf-8")
    assert "secret_detected" in reject
    assert "sk-example-secret-value-123456" not in reject
    staging = settings.quality_staging_path.read_text(encoding="utf-8")
    assert "sk-example-secret-value-123456" not in staging


def test_failure_attempt_ledger_is_content_free(tmp_path: Path) -> None:
    settings = config(tmp_path, max_failures=2)
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    error = "frontend:seed-1: DataValidationError: api_key=sk-do-not-store-123456"

    runner._record_report_failures([error])
    runner._append_failure_attempts([error])

    retained = settings.attempts_path.read_text(encoding="utf-8")
    assert "DataValidationError" in retained
    assert "sk-do-not-store-123456" not in retained
    assert "teacher_content_retained\": false" in retained


def test_partition_only_cli_does_not_require_provider_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anchor_mvp.data import automation as module

    config_path = tmp_path / "collect.yaml"
    config_path.write_text(
        f'sop_dir: "{(ROOT / "skills").as_posix()}"\n'
        f'output_dir: "{tmp_path.as_posix()}"\n'
        "concurrency_stages: [1]\n"
        "stage_seed_counts: [1]\n"
        "collection_policy: collect_then_partition\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setattr(
        module,
        "partition_collected_records",
        lambda unused: {"training_ready": True, "gold_count": 5},
    )

    assert main(["--config", str(config_path), "--partition-only"]) == 0


class _RateLimitedTeacher:
    model = "rate-limit-mock"
    base_url = "mock://rate-limit"
    protocol = "mock"
    generation_params = {"deterministic": True}

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise RateLimitError(120)


class _DeadlineTeacher(_RateLimitedTeacher):
    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise ClientDeadlineExceeded(30)


def test_rate_limit_persists_five_hour_floor_and_resume_does_not_call(tmp_path: Path) -> None:
    settings = config(tmp_path, cooldown_seconds=18_000, max_failures=2)
    teacher = _RateLimitedTeacher()
    status = asyncio.run(AutomationRunner(config=settings, teacher=teacher).run())
    assert status["state"] == "cooldown"
    assert status["cooldown_until"]
    assert teacher.calls == 1
    events = [json.loads(line) for line in settings.events_path.read_text(encoding="utf-8").splitlines()]
    cooldown = next(event for event in events if event["type"] == "rate_limit_cooldown")
    assert cooldown["data"]["retry_after_seconds"] == 120
    assert cooldown["data"]["cooldown_seconds"] == 18_000

    second_teacher = _RateLimitedTeacher()
    resumed = asyncio.run(AutomationRunner(config=settings, teacher=second_teacher).run())
    assert resumed["state"] == "cooldown"
    assert second_teacher.calls == 0


def test_request_budget_stops_before_gate_upgrade(tmp_path: Path) -> None:
    settings = config(tmp_path, max_requests=1, max_failures=10)
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())
    assert status["state"] == "budget_exhausted"
    assert status["stage_index"] == 0
    assert status["quota_epoch"]["requests_used"] == 1
    events = [json.loads(line)["type"] for line in settings.events_path.read_text(encoding="utf-8").splitlines()]
    assert "gate_passed" not in events
    assert events[-1] == "budget_exhausted"


def test_failure_budget_stops_at_configured_limit(tmp_path: Path) -> None:
    settings = config(tmp_path, max_failures=2)
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    runner.status["quota_epoch"]["failures_used"] = 2

    assert runner._budget_exhausted() == "failure_budget"


def test_dependency_cascades_are_not_charged_as_independent_failures() -> None:
    errors = [
        "frontend:seed-a: DataValidationError: invalid code",
        "review:seed-a: UpstreamDependencyError: frontend row required",
        "security:seed-a: UpstreamDependencyError: review row required",
    ]

    assert chargeable_failure_count(errors) == 1


def test_failure_identity_is_charged_once_and_quarantined_after_bounded_retries(
    tmp_path: Path,
) -> None:
    settings = config(tmp_path, max_failures=10, max_failure_retries=2)
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    error = "frontend:seed-a: ValueError: invalid code"
    cascade = "review:seed-a: UpstreamDependencyError: frontend row required"

    runner._record_report_failures([error, error, cascade])
    runner._record_report_failures([error, cascade])
    runner._record_report_failures([error, cascade])

    epoch = runner.status["quota_epoch"]
    entries = list(runner.status["audit_ledger"]["failure_entries"].values())
    assert epoch["failures_used"] == 1
    assert len(epoch["charged_failure_keys"]) == 1
    assert runner.status["audit_ledger"]["failure_observations_total"] == 3
    assert len(entries) == 1
    assert entries[0]["attempts_total"] == 3
    assert entries[0]["quarantined"] is True
    assert runner._quarantined_seed_ids_for_task("plan") == frozenset()
    assert runner._quarantined_seed_ids_for_task("tool_policy") == frozenset()
    assert runner._quarantined_seed_ids_for_task("frontend") == frozenset({"seed-a"})
    assert runner._quarantined_seed_ids_for_task("security") == frozenset({"seed-a"})


def test_schema_v1_budget_exhaustion_migrates_without_erasing_history(tmp_path: Path) -> None:
    settings = config(tmp_path, quota_epoch_id="reset-window-2")
    template = AutomationRunner(config=settings, teacher=MockTeacher()).status
    legacy = dict(template)
    legacy["schema_version"] = "1.0"
    legacy["state"] = "budget_exhausted"
    legacy["stage_index"] = 3
    legacy["budgets"] = {
        "requests_used": 1200,
        "output_tokens_used": 456_000,
        "failures_used": 276,
        "max_requests": 1200,
        "max_output_tokens_total": 20_000_000,
        "max_failures": 200,
    }
    legacy.pop("quota_epoch")
    legacy.pop("quota_history")
    legacy.pop("audit_ledger")
    settings.status_path.parent.mkdir(parents=True)
    settings.status_path.write_text(json.dumps(legacy), encoding="utf-8")

    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    migrated = runner.status

    assert migrated["schema_version"] == "2.0"
    assert migrated["state"] == "ready"
    assert migrated["stage_index"] == 0
    assert migrated["stages"] == []
    assert migrated["quota_epoch"]["epoch_id"] == "reset-window-2"
    assert migrated["quota_epoch"]["requests_used"] == 0
    assert migrated["quota_epoch"]["failures_used"] == 0
    assert migrated["quota_history"][0]["failures_used"] == 276
    assert migrated["audit_ledger"]["requests_total"] == 1200
    assert migrated["audit_ledger"]["legacy_unkeyed_failures"] == 276
    assert migrated["migration_history"][0]["legacy_status"]["budgets"]["failures_used"] == 276
    assert migrated["migration_history"][0]["resume_policy"] == "fresh_epoch_stage_zero"
    assert migrated["migration_history"][0]["previous_stage_index"] == 3

    # A one-stage v2 config must perform stage zero, not mistake the legacy
    # stage index for a completed v2 ramp.
    resumed = asyncio.run(runner.run())
    assert resumed["state"] == "complete"
    assert [stage["index"] for stage in resumed["stages"]] == [0]


def test_new_quota_epoch_resets_window_but_retains_durable_failure_ledger(
    tmp_path: Path,
) -> None:
    first_settings = config(tmp_path, quota_epoch_id="window-1", max_failures=10)
    first = AutomationRunner(config=first_settings, teacher=MockTeacher())
    first.status["quota_epoch"]["requests_used"] = 9
    first.status["audit_ledger"]["requests_total"] = 9
    first._record_report_failures(["plan:seed-a: ValueError: invalid plan"])
    first._save_status()

    second_settings = config(tmp_path, quota_epoch_id="window-2", max_failures=10)
    second = AutomationRunner(config=second_settings, teacher=MockTeacher())

    assert second.status["quota_epoch"]["epoch_id"] == "window-2"
    assert second.status["quota_epoch"]["requests_used"] == 0
    assert second.status["quota_epoch"]["failures_used"] == 0
    assert second.status["quota_history"][-1]["epoch_id"] == "window-1"
    assert second.status["quota_history"][-1]["requests_used"] == 9
    assert second.status["audit_ledger"]["requests_total"] == 9
    assert len(second.status["audit_ledger"]["failure_entries"]) == 1


def test_client_deadline_has_distinct_persisted_classification(tmp_path: Path) -> None:
    settings = config(tmp_path, max_failures=2)
    teacher = _DeadlineTeacher()
    status = asyncio.run(AutomationRunner(config=settings, teacher=teacher).run())
    assert status["state"] == "client_deadline"
    assert status["last_client_deadline"]["worker"] == "seed"
    events = [json.loads(line) for line in settings.events_path.read_text(encoding="utf-8").splitlines()]
    deadline = next(event for event in events if event["type"] == "client_deadline")
    assert deadline["data"]["classification"] == "client_deadline"
    assert not any(event["type"] == "stage_failed" for event in events)


def test_real_worker_factory_uses_low_security_effort_and_shared_budget() -> None:
    workers = _build_teachers(
        {
            "max_tokens": 8192,
            "thinking_budget_tokens": 1024,
            "max_requests": 10,
            "max_output_tokens_total": 10000,
        },
        dry_run=False,
    )
    assert workers["frontend"].generation_params["thinking_effort"] == "medium"
    assert workers["plan"].generation_params["thinking_effort"] == "medium"
    assert workers["tool_policy"].generation_params["thinking_effort"] == "low"
    assert workers["review"].generation_params["thinking_effort"] == "medium"
    assert workers["security"].generation_params["thinking_effort"] == "low"
    assert len({worker.usage_budget_id for worker in workers.values()}) == 1


def test_concurrency_stages_accept_any_positive_operator_values(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        concurrency_stages=(1, 3, 11),
        stage_seed_counts=(1, 2, 3),
    )
    assert settings.concurrency_stages == (1, 3, 11)

    with pytest.raises(ValueError, match="positive integers"):
        config(tmp_path, concurrency_stages=(1, 0), stage_seed_counts=(1, 2))


@pytest.mark.parametrize(
    ("config_name", "expected_stages", "expected_targets"),
    [
        ("automation.mock.yaml", (1,), (1,)),
        ("automation.yaml", (1,), (128,)),
        ("automation.full_v3.yaml", (1,), (128,)),
        ("automation.full_v3.fast.yaml", (10,), (128,)),
    ],
)
def test_canonical_stage_lists_load_and_legacy_scalar_remains_compatible(
    config_name: str,
    expected_stages: tuple[int, ...],
    expected_targets: tuple[int, ...],
) -> None:
    raw = _simple_config(ROOT / "configs" / "data" / config_name)
    assert raw["concurrency_stages"] == list(expected_stages)
    assert raw["stage_seed_counts"] == list(expected_targets)
    loaded = AutomationConfig.from_mapping(raw, repo_root=ROOT)
    assert loaded.concurrency_stages == expected_stages
    assert loaded.stage_seed_counts == expected_targets

    legacy_scalar = AutomationConfig.from_mapping(
        {
            "sop_dir": "skills",
            "output_dir": "data/legacy-scalar-test",
            "concurrency_stages": 1,
            "stage_seed_counts": 1,
        },
        repo_root=ROOT,
    )
    assert legacy_scalar.concurrency_stages == (1,)
    assert legacy_scalar.stage_seed_counts == (1,)


def test_full_v3_config_has_an_isolated_full_corpus_state_directory() -> None:
    raw = _simple_config(ROOT / "configs" / "data" / "automation.full_v3.yaml")
    loaded = AutomationConfig.from_mapping(raw, repo_root=ROOT)

    assert loaded.output_dir == (ROOT / "data" / "automated_v3").resolve()
    assert loaded.status_path == loaded.output_dir / "automation" / "status.json"
    assert loaded.status_path != (ROOT / "data" / "automated_v2" / "automation" / "status.json")
    assert loaded.concurrency_stages == (1,)
    assert loaded.stage_seed_counts == (128,)
    assert loaded.collection_policy == "collect_then_partition"
    assert loaded.max_requests == 1200
    assert loaded.minimum_label_counts == {
        "tool_policy": {"APPROVE": 40, "ESCALATE": 40, "BLOCK": 40},
        "security": {"PASS": 60, "BLOCK": 60},
    }
    assert loaded.artifact_validation_fixture == (
        ROOT / "examples" / "data" / "fixtures" / "tsx-fragment"
    ).resolve()


def test_fast_v3_profile_is_opt_in_and_cannot_mix_state_with_serial_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    serial_raw = _simple_config(ROOT / "configs" / "data" / "automation.full_v3.yaml")
    fast_raw = _simple_config(ROOT / "configs" / "data" / "automation.full_v3.fast.yaml")
    serial = AutomationConfig.from_mapping(serial_raw, repo_root=ROOT)
    fast = AutomationConfig.from_mapping(fast_raw, repo_root=ROOT)

    assert serial.output_dir == fast.output_dir
    assert serial.concurrency_stages == (1,)
    assert fast.concurrency_stages == (10,)
    assert serial.stage_seed_counts == fast.stage_seed_counts == (128,)
    assert serial.collection_policy == fast.collection_policy == "collect_then_partition"
    assert serial.minimum_label_counts == fast.minimum_label_counts
    assert serial.status_binding_sha256 != fast.status_binding_sha256
    assert serial.quota_epoch_id != fast.quota_epoch_id

    # The checked-in fast profile loads through the read-only operator status
    # command without constructing a teacher or mutating output/state.
    status_path = fast.status_path
    before = status_path.read_bytes() if status_path.exists() else None
    assert main(["--config", str(ROOT / "configs" / "data" / "automation.full_v3.fast.yaml"), "--status-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "state" in payload
    after = status_path.read_bytes() if status_path.exists() else None
    assert after == before

    # A state created by one ramp profile cannot be reopened with the other,
    # even when the operator chooses a different quota epoch.
    shared_output = tmp_path / "shared-output"
    serialized = config(
        shared_output,
        quota_epoch_id="serialized-window",
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
    )
    serialized_runner = AutomationRunner(config=serialized, teacher=MockTeacher())
    serialized_runner._save_status()
    fast_settings = config(
        shared_output,
        quota_epoch_id="fast-window",
        concurrency_stages=(10,),
        stage_seed_counts=(1,),
    )
    with pytest.raises(ValueError, match="config binding mismatch"):
        AutomationRunner(config=fast_settings, teacher=MockTeacher())


def _distill_for_gate(settings: AutomationConfig, *, seed_count: int) -> None:
    report = asyncio.run(
        DistillationPipeline(
            teacher=MockTeacher(),
            sop_dir=settings.sop_dir,
            output_dir=settings.output_dir,
        ).run(seed_count=seed_count)
    )
    assert report.errors == ()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_quality_gate_requires_current_oracles_and_isolated_artifact_builds(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        stage_seed_counts=(4,),
        minimum_label_counts={
            "tool_policy": {"APPROVE": 1, "ESCALATE": 1, "BLOCK": 1},
            "security": {"PASS": 1, "BLOCK": 1},
        },
        artifact_validation_fixture=ROOT / "examples" / "data" / "fixtures" / "tsx-fragment",
        artifact_validation_workspace_root=tmp_path / "artifact-workspaces",
        artifact_validation_timeout_seconds=15,
    )
    _distill_for_gate(settings, seed_count=4)

    gate = evaluate_gate(settings, 4)
    assert gate["passed"] is True
    assert gate["deterministic_oracle_ok"] is True
    assert gate["label_counts"] == {
        "tool_policy": {"APPROVE": 2, "ESCALATE": 1, "BLOCK": 1},
        "security": {"PASS": 2, "BLOCK": 2},
    }
    assert gate["artifact_validation"]["passed"] is True
    assert gate["artifact_validation"]["checked"] == {"frontend": 4, "review": 4}

    tool_path = tmp_path / "data_tool_policy.jsonl"
    tool_records = [json.loads(line) for line in tool_path.read_text(encoding="utf-8").splitlines()]
    tool_records[0]["provenance"].pop("label_oracle")
    _write_jsonl(tool_path, tool_records)
    security_path = tmp_path / "data_security.jsonl"
    security_records = [
        json.loads(line) for line in security_path.read_text(encoding="utf-8").splitlines()
    ]
    security_records[0]["provenance"].pop("security_fixture")
    _write_jsonl(security_path, security_records)

    rejected = evaluate_gate(settings, 4)
    assert rejected["passed"] is False
    assert rejected["deterministic_oracle_ok"] is False
    assert any(error.startswith("tool_policy:") for error in rejected["oracle_errors"])
    assert any(error.startswith("security:") for error in rejected["oracle_errors"])


def test_quality_gate_fails_closed_for_missing_label_quota_and_review_regression(
    tmp_path: Path,
) -> None:
    quota_settings = config(
        tmp_path / "quota",
        minimum_label_counts={
            "tool_policy": {"APPROVE": 1, "ESCALATE": 1, "BLOCK": 1},
            "security": {"PASS": 1, "BLOCK": 1},
        },
    )
    _distill_for_gate(quota_settings, seed_count=1)
    quota_gate = evaluate_gate(quota_settings, 1)
    assert quota_gate["passed"] is False
    assert quota_gate["deterministic_oracle_ok"] is True
    assert quota_gate["label_quota_ok"] is False
    assert {error.split(":", 2)[1] for error in quota_gate["label_quota_errors"]} == {
        "ESCALATE",
        "BLOCK",
    }

    settings = config(
        tmp_path / "execution",
        stage_seed_counts=(4,),
        artifact_validation_fixture=ROOT / "examples" / "data" / "fixtures" / "tsx-fragment",
        artifact_validation_workspace_root=tmp_path / "execution-workspaces",
        artifact_validation_timeout_seconds=15,
    )
    _distill_for_gate(settings, seed_count=4)
    review_path = settings.output_dir / "data_review.jsonl"
    review_records = [
        json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()
    ]
    review_records[0]["output"]["code"] = review_records[0]["input"]["candidate_code"]
    _write_jsonl(review_path, review_records)

    rejected = evaluate_gate(settings, 4)
    assert rejected["passed"] is False
    assert rejected["artifact_validation"]["passed"] is False
    assert any(
        "repair_does_not_restore_frontend_source" in error
        for error in rejected["artifact_validation"]["errors"]
    )


def _heldout_paths() -> dict[str, Path]:
    return {
        "heldout_cases": ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl",
        "heldout_fixtures_root": ROOT / "examples" / "benchmark" / "fixtures",
        "heldout_manifest": ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json",
        "heldout_leak_audit": ROOT
        / "artifacts"
        / "benchmark"
        / "heldout_v1"
        / "leak_audit.prebulk.json",
    }


def test_heldout_gate_rescans_five_outputs_and_records_manifest(tmp_path: Path) -> None:
    settings = config(tmp_path, **_heldout_paths())
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "complete"
    assert status["heldout_gate"]["status"] == "PASS"
    assert status["heldout_gate"]["collision_count"] == 0
    assert status["heldout_gate"]["training_source_count"] == 5
    assert status["heldout_gate"]["sop_source_count"] == 5
    assert status["heldout_gate"]["manifest_sha256"] == (
        "1ac7240d700a67458dc713b66ff085f1e51795b26cdacff688063bc60af3194c"
    )
    events = [
        json.loads(line)
        for line in settings.events_path.read_text(encoding="utf-8").splitlines()
    ]
    heldout_events = [event for event in events if event["type"] == "heldout_leakage_gate"]
    assert len(heldout_events) == 1
    assert all(event["data"]["passed"] for event in heldout_events)

    heldout_requirement = json.loads(
        (ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )["requirement"]
    with (tmp_path / "data_plan.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "synthetic-leak", "text": heldout_requirement}) + "\n")
    collision = evaluate_heldout_scale_gate(settings)
    assert collision["passed"] is False
    assert collision["status"] == "FAIL"
    assert collision["collision_count"] >= 1
    assert collision["content_emitted"] is False


def test_heldout_collision_blocks_concurrency_upgrade(tmp_path: Path, monkeypatch) -> None:
    from anchor_mvp.data import automation as module

    settings = config(tmp_path)
    monkeypatch.setattr(
        module,
        "evaluate_heldout_scale_gate",
        lambda unused: {
            "enabled": True,
            "passed": False,
            "status": "FAIL",
            "manifest_sha256": "frozen",
            "collision_count": 1,
            "content_emitted": False,
        },
    )
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "gate_blocked"
    assert status["stage_index"] == 0
    assert status["current_concurrency"] == 1
    assert status["last_gate"]["heldout_leakage"]["collision_count"] == 1
    events = [
        json.loads(line)["type"]
        for line in settings.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events.index("heldout_leakage_gate") < events.index("gate_blocked")
    assert "gate_passed" not in events
