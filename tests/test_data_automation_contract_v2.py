from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.artifact_validation import validate_tsx_fragment  # noqa: E402
from anchor_mvp.data.automation import (  # noqa: E402
    AutomationConfig,
    AutomationRunner,
    partition_collected_records,
)
from anchor_mvp.data.pipeline import DistillationPipeline  # noqa: E402
from anchor_mvp.data.prompts import task_prompt  # noqa: E402
from anchor_mvp.data.schema import SeedDemand  # noqa: E402
from anchor_mvp.data.sops import load_sop  # noqa: E402
from anchor_mvp.data.teacher import MockTeacher  # noqa: E402


def _config(tmp_path: Path, **overrides: object) -> AutomationConfig:
    values: dict[str, object] = {
        "sop_dir": ROOT / "skills",
        "output_dir": tmp_path,
        "concurrency_stages": (1,),
        "stage_seed_counts": (3,),
        "min_success_rate": 1.0,
        "max_duplicate_rate": 0.0,
        "max_safety_violations": 0,
        "max_failures": 20,
        "max_requests": 100,
        "max_output_tokens_total": 1_000_000,
        "cooldown_seconds": 300,
        "cooldown_poll_seconds": 1,
        "collection_policy": "collect_then_partition",
    }
    values.update(overrides)
    return AutomationConfig(**values)


def _distill(settings: AutomationConfig, count: int) -> None:
    report = asyncio.run(
        DistillationPipeline(
            teacher=MockTeacher(),
            sop_dir=settings.sop_dir,
            output_dir=settings.output_dir,
        ).run(seed_count=count)
    )
    assert report.errors == ()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_raw_overcollection_makes_one_quality_negative_reachable(tmp_path: Path) -> None:
    settings = _config(
        tmp_path,
        raw_collection_target=3,
        minimum_gold_records_per_task={task: 2 for task in (
            "plan", "tool_policy", "frontend", "review", "security"
        )},
    )
    _distill(settings, 3)

    tool_path = settings.output_dir / "data_tool_policy.jsonl"
    tool_records = [
        json.loads(line) for line in tool_path.read_text(encoding="utf-8").splitlines()
    ]
    # Tampering only the observed label leaves provenance that says the trace
    # came from teacher agreement. The partitioner must not pretend this is an
    # oracle-normalized disagreement.
    tool_records[1]["provenance"]["teacher_observed_decision"] = "APPROVE"
    _write_jsonl(tool_path, tool_records)

    manifest = partition_collected_records(settings)

    assert manifest["raw_collection_target"] == 3
    assert manifest["minimum_gold_records_per_task"] == {
        task: 2 for task in ("plan", "tool_policy", "frontend", "review", "security")
    }
    assert manifest["gold_by_task"]["tool_policy"] == 2
    assert manifest["negative_count"] == 1
    assert manifest["coverage_complete"] is True
    assert manifest["training_ready"] is True
    assert manifest["unresolved_disagreements_by_task"] == {"tool_policy": 1}

    legacy_strict = _config(settings.output_dir)
    strict_manifest = partition_collected_records(legacy_strict)
    assert strict_manifest["minimum_gold_records_per_task"]["tool_policy"] == 3
    assert strict_manifest["coverage_complete"] is False
    assert strict_manifest["coverage_shortfalls"] == {"tool_policy": 1}
    assert strict_manifest["training_ready"] is False


def test_raw_gold_contract_resume_is_explicit_and_other_changes_fail_closed(
    tmp_path: Path,
) -> None:
    legacy = _config(
        tmp_path,
        stage_seed_counts=(2,),
        quota_epoch_id="legacy-window",
    )
    legacy_runner = AutomationRunner(config=legacy, teacher=MockTeacher())
    legacy_runner._save_status()

    overcollection = _config(
        tmp_path,
        stage_seed_counts=(2,),
        raw_collection_target=3,
        minimum_gold_records_per_task={task: 2 for task in (
            "plan", "tool_policy", "frontend", "review", "security"
        )},
        quota_epoch_id="overcollection-window",
    )
    assert overcollection.legacy_status_binding_sha256 == legacy.status_binding_sha256
    assert overcollection.status_binding_sha256 != legacy.status_binding_sha256

    resumed = AutomationRunner(config=overcollection, teacher=MockTeacher())
    migration = resumed.status["migration_history"][-1]
    assert migration["migration_type"] == "raw_gold_contract_v1_to_v2"
    assert migration["raw_collection_target"] == 3
    assert migration["minimum_gold_records_per_task"]["frontend"] == 2
    assert migration["resume_policy"] == (
        "preserve_append_only_rows_and_resume_missing_raw_rows"
    )
    assert resumed.status["config_binding_sha256"] == overcollection.status_binding_sha256

    incompatible = _config(
        tmp_path,
        stage_seed_counts=(3,),
        raw_collection_target=4,
        minimum_gold_records_per_task={task: 2 for task in (
            "plan", "tool_policy", "frontend", "review", "security"
        )},
        quota_epoch_id="incompatible-window",
    )
    with pytest.raises(ValueError, match="config binding mismatch"):
        AutomationRunner(config=incompatible, teacher=MockTeacher())


def test_isolated_secret_reject_is_quarantined_without_blocking_clean_gold(
    tmp_path: Path,
) -> None:
    settings = _config(
        tmp_path,
        stage_seed_counts=(2,),
        raw_collection_target=2,
        minimum_gold_records_per_task={task: 1 for task in (
            "plan", "tool_policy", "frontend", "review", "security"
        )},
    )
    _distill(settings, 2)
    frontend_path = settings.output_dir / "data_frontend.jsonl"
    records = [
        json.loads(line) for line in frontend_path.read_text(encoding="utf-8").splitlines()
    ]
    secret = json.loads(json.dumps(records[0]))
    secret["id"] = "isolated-secret-record"
    secret["messages"][0]["content"] += "\nISOLATED AUDIT COPY"
    secret["output"]["code"] += "\n// api_key=sk-example-secret-value-123456"
    records.append(secret)
    _write_jsonl(frontend_path, records)

    manifest = partition_collected_records(settings)

    assert manifest["reject_count"] == 1
    assert manifest["partition_complete"] is True
    assert manifest["rejects_quarantined"] is True
    assert manifest["reject_reason_counts"] == {
        "secret_detected": 1,
        "unsafe_payload": 1,
    }
    assert manifest["training_ready"] is True
    reject_text = (settings.partition_dir / "reject.jsonl").read_text(encoding="utf-8")
    assert "sk-example-secret-value-123456" not in reject_text
    assert "source_record_sha256" in reject_text


def test_malformed_collection_replaces_any_stale_ready_manifest(tmp_path: Path) -> None:
    settings = _config(
        tmp_path,
        stage_seed_counts=(1,),
        raw_collection_target=1,
        minimum_gold_records_per_task={task: 1 for task in (
            "plan", "tool_policy", "frontend", "review", "security"
        )},
    )
    _distill(settings, 1)
    assert partition_collected_records(settings)["training_ready"] is True
    (settings.output_dir / "data_plan.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed collection record"):
        partition_collected_records(settings)

    blocked = json.loads(
        (settings.partition_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert blocked["partition_complete"] is False
    assert blocked["training_ready"] is False
    assert blocked["corpus_blocker"] == "malformed_collection"
    assert blocked["content_emitted"] is False


def test_tsx_validator_accepts_types_and_arrow_attributes_but_rejects_bad_tags(
    tmp_path: Path,
) -> None:
    valid = """import { useState } from 'react';
const choices = new Set<string>();
const TypedCard = () => {
  const [open, setOpen] = useState<boolean>(false);
  return (<button type="button" onClick={() => setOpen((value) => !value)}>
    {open ? choices.size : 0}
  </button>);
};
export default TypedCard;
"""
    accepted = validate_tsx_fragment(
        valid,
        fixture_root=ROOT / "examples" / "data" / "fixtures" / "tsx-fragment",
        workspace_root=tmp_path / "accepted",
        timeout_seconds=15,
    )
    assert accepted["passed"] is True

    invalid = "export function Broken(){return (<div><span>bad</div>)}"
    rejected = validate_tsx_fragment(
        invalid,
        fixture_root=ROOT / "examples" / "data" / "fixtures" / "tsx-fragment",
        workspace_root=tmp_path / "rejected",
        timeout_seconds=15,
    )
    assert rejected["passed"] is False


def test_tool_policy_prompt_has_unambiguous_oracle_precedence() -> None:
    seed = SeedDemand(
        seed_id="seed-policy-prompt",
        title="Bounded local editor",
        request="Build a local editor.",
        category="developer-tool",
        tags=("local",),
    )
    sop = load_sop(ROOT / "skills" / "tool_policy.md")
    _, user = task_prompt(
        "tool_policy",
        seed,
        sop,
        0,
        task_input={
            "plan": {"summary": "local", "steps": [{"id": "P1"}]},
            "tool_proposals": [
                {
                    "capability": "workspace.write_derived_file",
                    "resource_scope": "workspace-generated-output",
                    "side_effect": "reversible",
                }
            ],
        },
    )
    assert "Otherwise ESCALATE if any proposal writes" in user
    assert "bounded reversible write always requires explicit human approval" in user
    assert "is never APPROVE" in user
