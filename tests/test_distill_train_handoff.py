from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

from anchor_mvp.data import teacher as teacher_module
from anchor_mvp.data.automation import AutomationConfig, AutomationRunner
from anchor_mvp.data.teacher import CompatibleTeacher, ProviderQuotaExhausted
from anchor_mvp.handoff import (
    EXPERTS,
    HandoffConfig,
    _credential_free_parent_environment,
    _run_child,
    classify_distillation_terminal,
    evaluate_snapshot,
    inspect_execution_artifacts,
    inspect_lowmem_training,
    run_coordinator,
    verify_frozen_handoff,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _gold(sample_id: str = "sample-1") -> dict:
    return {
        "schema_version": "anchor.tool-gold.v1",
        "sample_id": sample_id,
        "success": True,
        "public_outcome": {"status": "completed"},
    }


def _session(sample_id: str = "sample-1", *, include_result: bool = True) -> dict:
    trajectory = [
        {"type": "user_input", "sequence": 1, "content": "Build it"},
        {
            "type": "tool_call",
            "sequence": 2,
            "call_id": "call_0001",
            "tool": "read",
            "input": {"path": "<workspace>/a.txt"},
        },
    ]
    if include_result:
        trajectory.append(
            {
                "type": "tool_result",
                "sequence": 3,
                "call_id": "call_0001",
                "tool": "read",
                "status": "completed",
                "content": "ok",
            }
        )
    trajectory.append({"type": "assistant_output", "sequence": 4, "content": "Done"})
    return {
        "schema_version": "anchor.session-training-candidate.v1",
        "sample_id": sample_id,
        "trajectory": trajectory,
        "validators": [{"name": "build", "status": "PASS", "exit_code": 0}],
        "public_outcome": {"status": "completed"},
    }


def test_execution_gate_requires_matching_tool_result_and_sample(tmp_path: Path) -> None:
    gold = _write_jsonl(tmp_path / "gold.jsonl", [_gold()])
    sessions = _write_jsonl(tmp_path / "sessions.jsonl", [_session(include_result=False)])
    blocked = inspect_execution_artifacts(gold, sessions, minimum_gold=1)
    assert blocked["passed"] is False
    assert "session_candidate_rejected:1" in blocked["errors"]

    _write_jsonl(sessions, [_session()])
    passed = inspect_execution_artifacts(gold, sessions, minimum_gold=1)
    assert passed["passed"] is True
    assert passed["matched_sample_count"] == 1


def test_handoff_config_rejects_inline_credentials_before_any_run(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "anchor.distill-train-handoff.config.v1",
                "project_root": str(ROOT),
                "api_key": "sk-this-must-never-be-in-config",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret-valued field is forbidden"):
        HandoffConfig(path)


@pytest.mark.parametrize(
    ("state", "classification"),
    [
        ("provider_quota_exhausted", "provider_quota_exhausted"),
        ("cooldown", "temporary_rate_limit"),
        ("client_deadline", "client_deadline"),
        ("budget_exhausted", "local_safety_budget"),
        ("failed", "failed"),
        ("complete", "automation_complete"),
    ],
)
def test_terminal_classification_is_state_based(state: str, classification: str) -> None:
    assert classify_distillation_terminal({"state": state}) == classification


def test_generic_429_is_not_provider_quota_but_explicit_quota_is(monkeypatch) -> None:
    teacher = CompatibleTeacher(max_retries=0, fallback_protocol=None)
    errors = iter(
        [
            teacher_module._ProtocolError("anthropic", 429, "slow down"),
            teacher_module._ProtocolError("anthropic", 429, "quota exhausted"),
        ]
    )

    def fake_request(*args, **kwargs):
        raise next(errors)

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    with pytest.raises(teacher_module.RateLimitError) as temporary:
        asyncio.run(teacher._with_retries("anthropic", "https://example.invalid", "s", "u", 1))
    assert type(temporary.value) is teacher_module.RateLimitError
    with pytest.raises(ProviderQuotaExhausted):
        asyncio.run(teacher._with_retries("anthropic", "https://example.invalid", "s", "u", 1))


class _QuotaTeacher:
    model = "quota-mock"
    base_url = "mock://quota"
    protocol = "mock"
    generation_params: dict = {}
    provider_provenance: dict = {}

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise ProviderQuotaExhausted()


def test_automation_persists_explicit_provider_quota_terminal(tmp_path: Path) -> None:
    config = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path / "output",
        concurrency_stages=(1, 3, 7, 11),
        stage_seed_counts=(1, 2, 3, 4),
        quota_epoch_id="test-epoch",
    )
    first_teacher = _QuotaTeacher()
    status = asyncio.run(AutomationRunner(config=config, teacher=first_teacher).run())
    assert status["state"] == "provider_quota_exhausted"
    assert status["quota_epoch"]["close_reason"] == "provider_quota_exhausted"
    assert status["cooldown_until"] is None
    assert first_teacher.calls == 1
    resumed_teacher = _QuotaTeacher()
    resumed = asyncio.run(AutomationRunner(config=config, teacher=resumed_teacher).run())
    assert resumed["state"] == "provider_quota_exhausted"
    assert resumed_teacher.calls == 0


def _formal_v1_datasets() -> dict[str, Path]:
    base = ROOT / "artifacts" / "formal_v1" / "dataset"
    return {
        "planner": base / "data_plan.jsonl",
        "tool_policy": base / "data_tool_policy.jsonl",
        "frontend_gen": base / "data_frontend.jsonl",
        "frontend_review": base / "data_review.jsonl",
        "security_gate": base / "data_security.jsonl",
    }


def test_snapshot_passes_all_gates_but_minimum_count_remains_independent() -> None:
    common = dict(
        datasets=_formal_v1_datasets(),
        heldout_cases=ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl",
        heldout_fixtures_root=ROOT / "examples" / "benchmark" / "fixtures",
        heldout_manifest=ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json",
        execution_gate={"passed": True},
        ramp_complete=True,
    )
    passed = evaluate_snapshot(
        **common, minimum_records={expert: 15 for expert in EXPERTS}
    )
    assert passed["passed"] is True
    assert passed["secret_findings"] == []

    insufficient = evaluate_snapshot(
        **common, minimum_records={expert: 16 for expert in EXPERTS}
    )
    assert insufficient["passed"] is False
    assert all(f"minimum_records:{expert}" in insufficient["errors"] for expert in EXPERTS)


def test_formal_v2_profile_is_sequential_low_memory_eligible() -> None:
    report = inspect_lowmem_training(
        ROOT / "configs" / "training" / "formal_v2_lowmem_common.yaml",
        EXPERTS,
        dataset_bindings=_formal_v1_datasets(),
    )
    assert report["passed"] is True
    assert report["maximum_training_peak_vram_gib"] <= 9.0


def test_offline_coordinator_freezes_handoff_without_api_or_training(tmp_path: Path) -> None:
    runtime_relative = Path("runs") / f"pytest-handoff-{uuid4().hex}"
    runtime = ROOT / runtime_relative
    try:
        gold = _write_jsonl(runtime / "gold.jsonl", [_gold()])
        sessions = _write_jsonl(runtime / "sessions.jsonl", [_session()])
        handoff = runtime / "training-handoff.json"
        config_value = {
            "schema_version": "anchor.distill-train-handoff.config.v1",
            "project_root": str(ROOT),
            "state_dir": str(runtime_relative / "state"),
            "execution": {
                "accepted_gold": str(gold.relative_to(ROOT)),
                "session_candidates": str(sessions.relative_to(ROOT)),
                "minimum_accepted_gold": 1,
                "concurrency_stages": [1],
            },
            "distillation": {
                "dry_run_terminal_state": "provider_quota_exhausted",
                "dry_run_quota_epoch": "mock-quota-epoch",
            },
            "snapshot": {
                "datasets": {
                    expert: str(path.relative_to(ROOT))
                    for expert, path in _formal_v1_datasets().items()
                },
                "minimum_records_per_expert": 15,
                "heldout_cases": "configs/benchmark/heldout_cases_v1.jsonl",
                "heldout_fixtures_root": "examples/benchmark/fixtures",
                "heldout_manifest": "artifacts/benchmark/heldout_v1/manifest.json",
            },
            "training": {
                "handoff_trigger": "provider_quota_exhausted",
                "handoff_manifest": str(handoff.relative_to(ROOT)),
                "config": "configs/training/formal_v2_lowmem_common.yaml",
                "adapters": list(EXPERTS),
                "auto_start": True,
                "max_parallel_gpu_jobs": 1,
                "logs_dir": str((runtime / "logs").relative_to(ROOT)),
            },
        }
        config_path = tmp_path / "handoff.json"
        config_path.write_text(json.dumps(config_value), encoding="utf-8")
        loaded_config = HandoffConfig(config_path)
        result = run_coordinator(
            loaded_config, confirm_live=False, confirm_training=False
        )
        assert result["phase"] == "handoff_ready"
        assert result["training"]["jobs"] == []
        frozen = verify_frozen_handoff(handoff)
        assert frozen["trigger"] == "provider_quota_exhausted"
        assert frozen["training"]["execution"] == "sequential_single_lora"
        first_sha = result["handoff"]["sha256"]
        resumed = run_coordinator(
            loaded_config, confirm_live=False, confirm_training=False
        )
        assert resumed["handoff"]["sha256"] == first_sha
        assert verify_frozen_handoff(handoff)["created_at"] == frozen["created_at"]
    finally:
        if runtime.is_dir() and runtime.resolve().parent == (ROOT / "runs").resolve():
            shutil.rmtree(runtime)


def test_child_secret_echo_is_discarded_and_never_logged(tmp_path: Path) -> None:
    secret = "sk-this-is-a-test-secret-not-a-real-key"
    log = tmp_path / "must-not-exist.log"
    with pytest.raises(RuntimeError, match="output contained a credential"):
        _run_child(
            [sys.executable, "-c", "import os; print(os.environ['TEST_ONLY_KEY'])"],
            environment={**os.environ, "TEST_ONLY_KEY": secret},
            secrets=(secret,),
            log_path=log,
        )
    assert not log.exists()


def test_inherited_credentials_are_hidden_from_opencode_environment(monkeypatch) -> None:
    monkeypatch.setenv("UNRELATED_TEST_API_KEY", "inherited-secret")
    with _credential_free_parent_environment():
        assert "UNRELATED_TEST_API_KEY" not in os.environ
    assert os.environ["UNRELATED_TEST_API_KEY"] == "inherited-secret"
