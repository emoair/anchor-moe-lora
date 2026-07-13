from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anchor_mvp.data import snapshot as snapshot_module
from anchor_mvp.data.snapshot import EXPERTS, SnapshotConfig, prepare_snapshot
from anchor_mvp.handoff import HandoffConfig
from anchor_mvp.training.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _record(expert: str, marker: str) -> dict:
    output: dict = {}
    assistant = marker
    if expert == "planner":
        output = {"summary": marker, "steps": ["implement", "verify"]}
    elif expert == "tool_policy":
        assistant = "APPROVE"
        output = {"decision": "APPROVE", "rationale": marker}
    elif expert in {"frontend_gen", "frontend_review"}:
        output = {"code": f"export const marker = {marker!r};"}
    elif expert == "security_gate":
        assistant = f"[PASS] {marker}"
        output = {"decision": "PASS", "rationale": marker}
    return {
        "schema_version": "1.0",
        "id": f"{expert}-id",
        "expert": expert,
        "messages": [
            {"role": "user", "content": f"request {marker}"},
            {"role": "assistant", "content": assistant},
        ],
        "provenance": {"generator": "unit-test"},
        "decision_trace": [
            {"check": "contract", "evidence": "fixture", "action": "accept"}
        ],
        "output": output,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path, *, ready: bool = True, marker: str = "PRIVATE_TRAINING_BODY"
) -> SnapshotConfig:
    collection = tmp_path / "collection"
    gold = collection / "partitions" / "gold"
    names = {
        "planner": ("plan", "data_plan.jsonl"),
        "tool_policy": ("tool_policy", "data_tool_policy.jsonl"),
        "frontend_gen": ("frontend", "data_frontend.jsonl"),
        "frontend_review": ("review", "data_review.jsonl"),
        "security_gate": ("security", "data_security.jsonl"),
    }
    counts: dict[str, int] = {}
    for expert, (task, filename) in names.items():
        records = [_record(expert, marker)]
        _write_jsonl(collection / filename, records)
        _write_jsonl(gold / filename, records)
        counts[task] = 1
    quality_staging = collection / "automation" / "quality_staging.jsonl"
    negative = collection / "partitions" / "negative.jsonl"
    reject = collection / "partitions" / "reject.jsonl"
    _write_jsonl(quality_staging, [{"partition_index": index} for index in range(5)])
    _write_jsonl(negative, [])
    _write_jsonl(reject, [])
    heldout_gate = {
        "status": "PASS",
        "passed": True,
        "collision_count": 0,
        "content_emitted": False,
        "manifest_sha256": "a" * 64,
        "prebulk_audit_sha256": "b" * 64,
        "heldout_text_for_leak_test": "PRIVATE_HELDOUT_BODY",
    }
    manifest = {
        "schema_version": "anchor.automation-partition-manifest.v2",
        "collection_policy": "collect_then_partition",
        "seed_target": 1,
        "raw_collection_target": 1,
        "minimum_gold_records_per_task": {task: 1 for task in counts},
        "staged_count": 5,
        "gold_count": 5,
        "negative_count": 0,
        "reject_count": 0,
        "partition_complete": True,
        "rejects_quarantined": True,
        "gold_integrity_ok": True,
        "reject_reason_counts": {},
        "reject_rate": 0.0,
        "gold_by_task": counts,
        "gold_label_counts": {},
        "label_quota_errors": [],
        "coverage_complete": ready,
        "coverage_shortfalls": {} if ready else {},
        "raw_by_task": {task: 1 for task in counts},
        "raw_collection_complete": True,
        "raw_collection_shortfalls": {},
        "training_ready": ready,
        "heldout_gate": heldout_gate,
        "quality_staging_sha256": sha256_file(quality_staging),
        "negative_sha256": sha256_file(negative),
        "reject_sha256": sha256_file(reject),
    }
    partition_manifest = collection / "partitions" / "manifest.json"
    partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    status = {
        "schema_version": "2.0",
        "state": "provider_quota_exhausted",
        "partition": manifest,
        "heldout_gate": heldout_gate,
    }
    automation_status = collection / "automation" / "status.json"
    automation_status.parent.mkdir(parents=True, exist_ok=True)
    automation_status.write_text(json.dumps(status), encoding="utf-8")
    config_path = tmp_path / "snapshot.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: anchor.training-snapshot-config.v1",
                f"project_root: {tmp_path.as_posix()}",
                f"partition_manifest: {partition_manifest.as_posix()}",
                f"automation_status: {automation_status.as_posix()}",
                f"collection_dir: {collection.as_posix()}",
                f"gold_dir: {gold.as_posix()}",
                f"snapshot_dir: {(tmp_path / 'snapshot').as_posix()}",
                f"readiness_report: {(tmp_path / 'readiness.json').as_posix()}",
                "expected_minimum_gold_records_per_expert: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return SnapshotConfig.load(config_path)


def test_not_ready_writes_metadata_report_only_and_never_freezes(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, ready=False)
    result = prepare_snapshot(config)
    persisted = config.readiness_report.read_text(encoding="utf-8")

    assert result["status"] == "blocked"
    assert result["training_ready"] is False
    assert result["freeze_performed"] is False
    assert result["execution_gate"]["evaluated"] is False
    assert not config.snapshot_dir.exists()
    assert "PRIVATE_TRAINING_BODY" not in persisted
    assert "PRIVATE_HELDOUT_BODY" not in persisted


def test_ready_snapshot_is_atomic_hashed_and_idempotent(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    result = prepare_snapshot(config)

    assert result["status"] == "frozen"
    assert result["training_ready"] is True
    assert result["freeze_performed"] is True
    manifest_path = config.snapshot_dir / "manifest.json"
    sidecar = config.snapshot_dir / "manifest.json.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "anchor.training-snapshot.v2"
    assert tuple(manifest["files"]) == EXPERTS
    assert manifest["source_partition_manifest_sha256"] == sha256_file(
        config.partition_manifest
    )
    expected_parts = []
    for expert in EXPERTS:
        item = manifest["files"][expert]
        assert Path(item["path"]).name == item["path"]
        assert item["source_sha256"] == item["sha256"]
        assert sha256_file(config.snapshot_dir / item["path"]) == item["sha256"]
        expected_parts.append(
            f"{expert}:{item['path']}:{item['sha256']}:{item['records']}"
        )
    assert (
        manifest["snapshot_sha256"]
        == hashlib.sha256("\n".join(expected_parts).encode()).hexdigest()
    )
    assert sidecar.read_text(encoding="ascii").split()[0] == sha256_file(manifest_path)

    resumed = prepare_snapshot(config)
    assert resumed["status"] == "already_frozen"
    assert resumed["freeze_performed"] is False
    assert (
        resumed["snapshot"]["manifest_sha256"] == result["snapshot"]["manifest_sha256"]
    )


def test_copy_failure_leaves_no_partial_snapshot(tmp_path: Path, monkeypatch) -> None:
    config = _fixture(tmp_path)
    original = snapshot_module.shutil.copyfile
    calls = 0

    def fail_second_copy(source: Path, destination: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated private failure details")
        return original(source, destination)

    monkeypatch.setattr(snapshot_module.shutil, "copyfile", fail_second_copy)
    result = prepare_snapshot(config)

    assert result["status"] == "freeze_failed"
    assert result["training_ready"] is False
    assert not config.snapshot_dir.exists()
    assert not list(config.snapshot_dir.parent.glob(".snapshot.tmp-*"))
    persisted = config.readiness_report.read_text(encoding="utf-8")
    assert "simulated private failure details" not in persisted


def test_quarantined_reject_does_not_block_clean_gold_snapshot(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    planner_raw = config.collection_dir / "data_plan.jsonl"
    extra = _record("planner", "REJECTED_PRIVATE_BODY")
    extra["id"] = "planner-rejected-id"
    with planner_raw.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")

    reject_path = config.partition_manifest.parent / "reject.jsonl"
    _write_jsonl(
        reject_path,
        [
            {
                "id": "reject-id",
                "schema_version": "anchor.automation-partition-reject.v1",
                "task_type": "plan",
                "source_record_sha256": "c" * 64,
                "reason_codes": ["unsafe_payload"],
                "content_retained": False,
            }
        ],
    )
    staging_path = config.collection_dir / "automation" / "quality_staging.jsonl"
    _write_jsonl(staging_path, [{"partition_index": index} for index in range(6)])

    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "seed_target": 2,
            "raw_collection_target": 2,
            "raw_by_task": {
                "plan": 2,
                "tool_policy": 1,
                "frontend": 1,
                "review": 1,
                "security": 1,
            },
            "raw_collection_complete": False,
            "raw_collection_shortfalls": {
                "tool_policy": 1,
                "frontend": 1,
                "review": 1,
                "security": 1,
            },
            "staged_count": 6,
            "reject_count": 1,
            "reject_reason_counts": {"unsafe_payload": 1},
            "reject_rate": 1 / 6,
            "quality_staging_sha256": sha256_file(staging_path),
            "reject_sha256": sha256_file(reject_path),
        }
    )
    config.partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    status = json.loads(config.automation_status.read_text(encoding="utf-8"))
    status["partition"] = manifest
    config.automation_status.write_text(json.dumps(status), encoding="utf-8")

    result = prepare_snapshot(config)
    report_text = config.readiness_report.read_text(encoding="utf-8")
    assert result["status"] == "frozen"
    assert result["source"]["reject_count"] == 1
    assert "REJECTED_PRIVATE_BODY" not in report_text


def test_checked_in_full_v3_configs_bind_new_state_and_immutable_paths() -> None:
    snapshot = SnapshotConfig.load(
        ROOT / "configs" / "orchestration" / "full_v3_snapshot.yaml"
    )
    assert snapshot.expected_minimum_gold_records_per_expert == 128
    assert snapshot.snapshot_dir == ROOT / "artifacts" / "formal_v3" / "dataset"

    handoff = HandoffConfig(
        ROOT / "configs" / "orchestration" / "distill_train_handoff_v3.yaml"
    )
    assert handoff.state_dir == ROOT / "runs" / "distill-train-handoff-v3"
    assert handoff.snapshot["minimum_records_per_expert"] == 128
    assert set(handoff.snapshot["datasets"].values()) == {
        "artifacts/formal_v3/dataset/data_plan.jsonl",
        "artifacts/formal_v3/dataset/data_tool_policy.jsonl",
        "artifacts/formal_v3/dataset/data_frontend.jsonl",
        "artifacts/formal_v3/dataset/data_review.jsonl",
        "artifacts/formal_v3/dataset/data_security.jsonl",
    }
