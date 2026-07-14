from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.automation import (  # noqa: E402
    AutomationConfig,
    AutomationRunner,
    _archive_quality_retry_namespace,
    main,
)
from anchor_mvp.data.pipeline import DistillationPipeline  # noqa: E402
from anchor_mvp.data.quality_retry import (  # noqa: E402
    QUALITY_FEEDBACK_CODES,
    build_quality_retry_plan,
    prepare_quality_retry_projection,
    restore_quality_retry_projection,
)
from anchor_mvp.data.schema import (  # noqa: E402
    DistilledRecord,
    ExpertSOP,
    SeedDemand,
)
from anchor_mvp.data.teacher import MockTeacher  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(task: str, seed_id: str) -> dict:
    return {
        "id": f"record_{task}_{seed_id}",
        "provenance": {"seed_id": seed_id},
    }


def _planner_config(tmp_path: Path, *, target: int = 1) -> SimpleNamespace:
    state_dir = tmp_path / "automation"
    return SimpleNamespace(
        output_dir=tmp_path,
        state_dir=state_dir,
        quality_staging_path=state_dir / "quality_staging.jsonl",
        raw_records_per_task=target,
    )


def test_quality_plan_uses_earliest_root_and_downstream_closure(tmp_path: Path) -> None:
    config = _planner_config(tmp_path)
    seed_id = "seed-a"
    _write_jsonl(tmp_path / "seeds.jsonl", [{"seed_id": seed_id, "seed_index": 0}])
    for task in ("plan", "tool_policy", "frontend", "review", "security"):
        _write_jsonl(tmp_path / f"data_{task}.jsonl", [_record(task, seed_id)])
    _write_jsonl(
        config.quality_staging_path,
        [
            {
                "id": "quality-review",
                "task_type": "review",
                "source_record_id": f"record_review_{seed_id}",
                "disposition": "negative",
                "quality": {
                    "labels": ["artifact_validation_failed"],
                    "audit_labels": [],
                },
            },
            {
                "id": "quality-security",
                "task_type": "security",
                "source_record_id": f"record_security_{seed_id}",
                "disposition": "negative",
                "quality": {
                    "labels": ["teacher_label_disagreement"],
                    "audit_labels": [],
                },
            },
        ],
    )
    plan = build_quality_retry_plan(
        config,
        {"raw_collection_target": 1},
        generation=1,
        artifact_gate={
            "record_failures": {
                f"review:record_review_{seed_id}": [
                    "mutation_recompute_failed:MutationUnavailable"
                ]
            }
        },
    )

    assert plan["seeds"][0]["root_task"] == "frontend"
    assert plan["seeds"][0]["tasks"] == ["frontend", "review", "security"]
    jobs = {job["task_type"]: job for job in plan["jobs"]}
    assert jobs["frontend"]["quality_feedback_codes"] == ["review_mutation_unavailable"]
    assert jobs["security"]["quality_feedback_codes"] == ["public_rubric_disagreement"]
    assert all(
        set(job["quality_feedback_codes"]).issubset(QUALITY_FEEDBACK_CODES)
        for job in plan["jobs"]
    )
    assert "MutationUnavailable" not in json.dumps(plan)


def test_quality_projection_archives_full_original_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config = _planner_config(tmp_path)
    before: dict[str, bytes] = {}
    for task in ("plan", "tool_policy", "frontend", "review", "security"):
        path = tmp_path / f"data_{task}.jsonl"
        _write_jsonl(path, [_record(task, "seed-a"), _record(task, "seed-b")])
        before[task] = path.read_bytes()
    plan = {
        "schema_version": "anchor.automation-quality-retry-plan.v1",
        "generation": 1,
        "seed_count": 1,
        "job_count": 3,
        "seeds": [],
        "jobs": [
            {
                "seed_id": "seed-a",
                "task_type": task,
                "generation": 1,
                "retry_of": [f"record_{task}_seed-a"],
                "quality_feedback_codes": ["upstream_quality_rebuild"],
            }
            for task in ("frontend", "review", "security")
        ],
    }

    first = prepare_quality_retry_projection(config, plan)
    second = prepare_quality_retry_projection(config, plan)
    assert first == second
    generation = config.state_dir / "quality_retry" / "generation-1"
    for task in ("plan", "tool_policy", "frontend", "review", "security"):
        assert (generation / f"active_data_{task}.jsonl").read_bytes() == before[task]
    assert len(_read(tmp_path / "data_plan.jsonl")) == 2
    assert len(_read(tmp_path / "data_frontend.jsonl")) == 1
    assert (
        _read(tmp_path / "data_frontend.jsonl")[0]["provenance"]["seed_id"] == "seed-b"
    )
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_history_recoverable"] is True

    corrupted = json.loads(json.dumps(manifest))
    corrupted["files"].pop("security")
    (generation / "manifest.json").write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="bind all task files"):
        prepare_quality_retry_projection(config, plan)
    (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (tmp_path / "data_frontend.jsonl").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="active collection changed"):
        prepare_quality_retry_projection(config, plan)
    restored = restore_quality_retry_projection(config, 1)
    assert restored["partial_retry_history_recoverable"] is True
    assert restore_quality_retry_projection(config, 1) == restored
    for task in ("plan", "tool_policy", "frontend", "review", "security"):
        assert (tmp_path / f"data_{task}.jsonl").read_bytes() == before[task]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_quality_feedback_changes_record_id_and_is_canonical() -> None:
    seed = SeedDemand(seed_id="seed-a", title="A", request="Build a status card")
    sop = ExpertSOP(
        sop_id="plan-v1",
        task_type="plan",
        source="test",
        content="Make a public plan.",
        sha256="a" * 64,
    )
    payload = {
        "decision_trace": [
            {"check": "scope", "evidence": "one card", "action": "plan"}
        ],
        "output": {
            "summary": "Build one status card.",
            "steps": [{"id": "P1", "goal": "Build", "deliverable": "Card"}],
            "constraints": ["local"],
        },
    }
    common = {
        "payload": payload,
        "task_type": "plan",
        "seed": seed,
        "sop": sop,
        "teacher_model": "mock",
        "teacher_base_url": "mock://teacher",
        "teacher_protocol": "mock",
        "generation_params": {},
        "template_sha256": "b" * 64,
    }
    original = DistilledRecord.from_teacher_payload(**common)
    retry = DistilledRecord.from_teacher_payload(
        **common,
        quality_retry={
            "generation": 1,
            "retry_of": [original.id],
            "quality_feedback_codes": ["raw_record_missing"],
        },
    )
    assert retry.id != original.id
    assert retry.input["quality_retry"] == retry.provenance["quality_retry"]
    assert "QUALITY RETRY METADATA" in retry.messages[0]["content"]


def test_planned_seed_overrides_quarantine_scope(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "seeds.jsonl",
        [
            {
                "seed_id": "seed-a",
                "title": "Status card",
                "request": "Build an accessible local status card.",
                "category": "standard",
                "tags": [],
            }
        ],
    )
    pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
        quality_feedback={
            "seed-a": {
                "generation": 1,
                "retry_of": [],
                "quality_feedback_codes": ["raw_record_missing"],
            }
        },
    )
    report = asyncio.run(
        pipeline.run(
            seed_count=1,
            tasks=["plan"],
            excluded_seed_ids=["seed-a"],
        )
    )
    assert report.written_by_task["plan"] == 1
    record = _read(tmp_path / "data_plan.jsonl")[0]
    assert record["provenance"]["quality_retry"]["generation"] == 1
    resumed = asyncio.run(
        pipeline.run(
            seed_count=1,
            tasks=["plan"],
            excluded_seed_ids=["seed-a"],
        )
    )
    assert resumed.written_by_task["plan"] == 0
    assert len(_read(tmp_path / "data_plan.jsonl")) == 1


class _PlanFormatTeacher(MockTeacher):
    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: plan" in user:
            return "{}"
        return await super().complete(system=system, user=user)


def test_offline_prepare_recovers_gate_blocked_without_teacher_or_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path / "collection",
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
        collection_policy="collect_then_partition",
        max_quality_retry_rounds=0,
        max_failures=20,
        max_requests=40,
        max_output_tokens_total=100_000,
    )
    blocked = asyncio.run(
        AutomationRunner(config=settings, teacher=_PlanFormatTeacher()).run()
    )
    assert blocked["state"] == "gate_blocked"
    # Simulate the immediately preceding status-binding schema, which had no
    # quality-retry policy field.  The offline command may migrate only the
    # equivalent default policy.
    blocked["config_binding_sha256"] = settings.pre_quality_retry_status_binding_sha256
    blocked.pop("quality_retry_policy", None)
    settings.status_path.write_text(json.dumps(blocked), encoding="utf-8")

    config_path = tmp_path / "automation.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"sop_dir: {settings.sop_dir.as_posix()}",
                f"output_dir: {settings.output_dir.as_posix()}",
                "concurrency_stages: [1]",
                "stage_seed_counts: [1]",
                "collection_policy: collect_then_partition",
                "max_quality_retry_rounds: 2",
                "max_failures: 20",
                "max_requests: 40",
                "max_output_tokens_total: 100000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    assert main(["--config", str(config_path), "--partition-only"]) == 3
    migrated = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert migrated["config_binding_sha256"] != (
        settings.pre_quality_retry_status_binding_sha256
    )
    assert migrated["state"] == "gate_blocked"
    assert main(["--config", str(config_path), "--prepare-quality-retry"]) == 0
    recovered = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "quality_retry_ready"
    assert recovered["quality_retry"]["prepared_offline"] is True
    assert recovered["migration_history"][-1]["migration_type"] == (
        "quality_retry_policy_v2_to_v3"
    )
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--restore-quality-generation",
                "1",
            ]
        )
        == 0
    )
    restored = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert restored["quality_retry"]["state"] == "restored"
    assert restored["active_projection_incomplete"] is False
    assert restored["partition_stale_reason"] == "quality_retry_generation_restored"


def test_quality_retry_budget_change_is_bound_and_rejected(tmp_path: Path) -> None:
    settings = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
        max_quality_retry_rounds=2,
    )
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    runner._save_status()
    changed = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
        max_quality_retry_rounds=3,
    )
    with pytest.raises(ValueError, match="config binding mismatch"):
        AutomationRunner(config=changed, teacher=MockTeacher())


def test_quality_retry_namespace_archive_hashes_untrusted_status_and_replays(
    tmp_path: Path,
) -> None:
    settings = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path / "collection",
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
    )
    retry_root = settings.state_dir / "quality_retry"
    retry_root.mkdir(parents=True)
    (retry_root / "marker.txt").write_text("retained", encoding="utf-8")
    status = {
        "run_id": r"..\..\escaped",
        "config_binding_sha256": "not-a-trusted-path-component",
        "active_projection_incomplete": False,
    }

    first = _archive_quality_retry_namespace(
        settings,
        status,
        reason="monotonic-expansion",
    )
    assert first is not None
    archived = (settings.output_dir / first).resolve()
    assert archived.parent == (settings.state_dir / "quality_retry_history").resolve()
    assert archived.name.startswith("archive-")
    assert "escaped" not in archived.name
    assert (archived / "marker.txt").read_text(encoding="utf-8") == "retained"
    assert not retry_root.exists()

    # Simulate a crash after os.replace and before the caller commits status.
    second = _archive_quality_retry_namespace(
        settings,
        status,
        reason="monotonic-expansion",
    )
    assert second == first


def test_stagnant_quality_runs_all_configured_rounds_before_blocking(
    tmp_path: Path,
) -> None:
    settings = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency_stages=(1,),
        stage_seed_counts=(1,),
        collection_policy="collect_then_partition",
        max_quality_retry_rounds=2,
        max_stagnant_gate_rounds=5,
        max_failure_retries=0,
        max_failures=20,
        max_requests=40,
        max_output_tokens_total=100_000,
    )
    teacher = _PlanFormatTeacher()
    status = asyncio.run(AutomationRunner(config=settings, teacher=teacher).run())

    assert status["state"] == "gate_blocked"
    assert status["quality_retry"]["generation"] == 2
    assert status["quality_retry"]["stagnant_quality_rounds"] == 2
    assert (
        settings.state_dir / "quality_retry" / "generation-1" / "manifest.json"
    ).is_file()
    assert (
        settings.state_dir / "quality_retry" / "generation-2" / "manifest.json"
    ).is_file()
    events = _read(settings.events_path)
    blocked = [event for event in events if event["type"] == "collection_gate_blocked"]
    assert blocked[-1]["data"]["reason"] == "quality_retry_rounds_exhausted"
