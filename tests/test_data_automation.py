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
)
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
        "concurrency_stages": (1, 2, 4, 8),
        "stage_seed_counts": (1, 2, 3, 4),
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


def test_mock_automation_ramps_gates_and_resumes(tmp_path: Path) -> None:
    settings = config(tmp_path)
    status = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())

    assert status["state"] == "complete"
    assert [stage["concurrency"] for stage in status["stages"]] == [1, 2, 4, 8]
    assert all(stage["gate"]["passed"] for stage in status["stages"])
    assert status["metrics"]["records"] == 20
    assert status["metrics"]["throughput_records_per_second"] > 0
    assert status["metrics"]["eta_seconds"] == 0
    assert status["budgets"]["requests_used"] <= 40
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
        assert report["valid_records"] == 4

    events_before = settings.events_path.read_text(encoding="utf-8").splitlines()
    event_types = [json.loads(line)["type"] for line in events_before]
    assert event_types.count("stage_started") == 4
    assert event_types.count("gate_passed") == 4
    assert event_types[-1] == "automation_completed"

    resumed = asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())
    assert resumed["state"] == "complete"
    assert settings.events_path.read_text(encoding="utf-8").splitlines() == events_before
    assert all(len((tmp_path / f"data_{task}.jsonl").read_text(encoding="utf-8").splitlines()) == 4 for task in experts)


def test_gate_detects_duplicates_and_training_schema_failure(tmp_path: Path) -> None:
    settings = config(tmp_path)
    asyncio.run(AutomationRunner(config=settings, teacher=MockTeacher()).run())
    frontend = tmp_path / "data_frontend.jsonl"
    first = frontend.read_text(encoding="utf-8").splitlines()[0]
    with frontend.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    gate = evaluate_gate(settings, 4)
    assert gate["passed"] is False
    assert gate["duplicate_count"] >= 1
    assert gate["duplicate_rate"] > 0
    assert gate["training_schema_ok"] is False


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
    assert status["budgets"]["requests_used"] == 1
    events = [json.loads(line)["type"] for line in settings.events_path.read_text(encoding="utf-8").splitlines()]
    assert "gate_passed" not in events
    assert events[-1] == "budget_exhausted"


def test_failure_budget_stops_at_configured_limit(tmp_path: Path) -> None:
    settings = config(tmp_path, max_failures=2)
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    runner.status["budgets"]["failures_used"] = 2

    assert runner._budget_exhausted() == "failure_budget"


def test_dependency_cascades_are_not_charged_as_independent_failures() -> None:
    errors = [
        "frontend:seed-a: DataValidationError: invalid code",
        "review:seed-a: UpstreamDependencyError: frontend row required",
        "security:seed-a: UpstreamDependencyError: review row required",
    ]

    assert chargeable_failure_count(errors) == 1


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


def test_concurrency_ramp_is_not_configurable_past_gate_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 1,2,4,8"):
        config(tmp_path, concurrency_stages=(1, 2, 8))


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
    assert len(heldout_events) == 4
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
