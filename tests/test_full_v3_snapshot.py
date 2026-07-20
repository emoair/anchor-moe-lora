from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from anchor_mvp.data import snapshot as snapshot_module
from anchor_mvp.data.cleaning import build_inert_security_fixture
from anchor_mvp.data.mutator import mutate_frontend_code
from anchor_mvp.data.snapshot import EXPERTS, SnapshotConfig, prepare_snapshot
from anchor_mvp.handoff import HandoffConfig
from anchor_mvp.training.manifest import sha256_file
from anchor_mvp.training.config import load_training_config, select_adapter
from anchor_mvp.training.preflight import (
    inspect_dataset_snapshot_manifest,
    inspect_gate_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "formal_v3_materializer_e2e",
    ROOT / "scripts/train/materialize_formal_v3_schedule.py",
)
assert MATERIALIZER_SPEC is not None and MATERIALIZER_SPEC.loader is not None
MATERIALIZER = importlib.util.module_from_spec(MATERIALIZER_SPEC)
sys.modules[MATERIALIZER_SPEC.name] = MATERIALIZER
MATERIALIZER_SPEC.loader.exec_module(MATERIALIZER)


def _record(
    expert: str,
    marker: str,
    *,
    suffix: str | None = None,
    instance_id: str | None = None,
) -> dict:
    record_id = f"{expert}-id" if suffix is None else f"{expert}-{suffix}-id"
    seed_id = "seed-fixture" if suffix is None else f"seed-{suffix}"
    planner_id = "planner-id" if suffix is None else f"planner-{suffix}-id"
    policy_id = (
        "tool_policy-id" if suffix is None else f"tool_policy-{suffix}-id"
    )
    frontend_id = (
        "frontend_gen-id" if suffix is None else f"frontend_gen-{suffix}-id"
    )
    review_id = (
        "frontend_review-id"
        if suffix is None
        else f"frontend_review-{suffix}-id"
    )
    requirement = f"request {marker}"
    plan_output = {"summary": marker, "steps": ["implement", "verify"]}
    tool_output = {"decision": "APPROVE", "rationale": marker}
    frontend_code = (
        f'export const Marker = () => <main aria-label="fixture">{marker}</main>;'
    )
    output: dict = {}
    record_input: dict = {"requirement": requirement}
    assistant = marker
    if expert == "planner":
        output = plan_output
        assistant = json.dumps(output, ensure_ascii=False, sort_keys=True)
    elif expert == "tool_policy":
        assistant = "APPROVE"
        output = tool_output
        record_input.update(
            {"plan": plan_output, "tool_proposals": [{"kind": "read_only"}]}
        )
    elif expert == "frontend_gen":
        output = {"code": frontend_code}
        record_input.update({"plan": plan_output, "tool_policy": tool_output})
        assistant = output["code"].strip()
    elif expert == "frontend_review":
        candidate, mutation = mutate_frontend_code(
            frontend_code, source_record_id=frontend_id
        )
        output = {"code": frontend_code}
        record_input.update(
            {
                "candidate_code": candidate.strip(),
                "known_benign_defect": mutation.known_benign_defect.strip(),
            }
        )
        assistant = output["code"].strip()
    elif expert == "security_gate":
        candidate, output, _fixture = build_inert_security_fixture(frontend_code, 0)
        record_input["reviewed_code"] = candidate.strip()
        assistant = f"[{output['decision']}]"
    provenance = {"generator": "unit-test", "seed_id": seed_id}
    if instance_id is not None:
        provenance["instance_id"] = instance_id
    if expert == "tool_policy":
        provenance["source_plan_record_id"] = planner_id
    elif expert == "frontend_gen":
        provenance.update(
            {
                "source_plan_record_id": planner_id,
                "source_tool_policy_record_id": policy_id,
            }
        )
    elif expert == "frontend_review":
        _candidate, mutation = mutate_frontend_code(
            frontend_code, source_record_id=frontend_id
        )
        provenance.update(
            {
                "source_frontend_record_id": frontend_id,
                "mutation": mutation.to_dict(),
            }
        )
    elif expert == "security_gate":
        _candidate, expected_output, fixture = build_inert_security_fixture(
            frontend_code, 0
        )
        provenance.update(
            {
                "source_review_record_id": review_id,
                "security_fixture": fixture,
                "label_oracle": {
                    "oracle": "anchor-security-fixture-gold-v1",
                    "decision": expected_output["decision"],
                    "sha256": fixture["gold_sha256"],
                },
            }
        )
    return {
        "schema_version": "1.0",
        "id": record_id,
        "expert": expert,
        "messages": [
            {"role": "user", "content": requirement},
            {"role": "assistant", "content": assistant},
        ],
        "input": record_input,
        "provenance": provenance,
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
    tmp_path: Path,
    *,
    ready: bool = True,
    marker: str = "PRIVATE_TRAINING_BODY",
    formal_split: bool = False,
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
    train_instances: list[str] = []
    calibration_instances: list[str] = []
    if formal_split:
        train_allowlist = json.loads(
            (
                ROOT
                / "datasets/public/swebench-full-bank-v1/allowlists/train.json"
            ).read_text(encoding="utf-8")
        )
        calibration_allowlist = json.loads(
            (
                ROOT
                / "datasets/public/swebench-full-bank-v1/allowlists/validation-from-train.json"
            ).read_text(encoding="utf-8")
        )
        train_instances = list(train_allowlist["instance_ids"][:256])
        calibration_instances = [calibration_allowlist["instance_ids"][0]]
    instances = train_instances + calibration_instances
    counts: dict[str, int] = {}
    for expert, (task, filename) in names.items():
        if formal_split:
            records = [
                _record(
                    expert,
                    f"{marker}-{index}",
                    suffix=f"{index:04d}",
                    instance_id=instance,
                )
                for index, instance in enumerate(instances)
            ]
        else:
            records = [_record(expert, marker)]
        _write_jsonl(collection / filename, records)
        _write_jsonl(gold / filename, records)
        counts[task] = len(records)
    gold_files = {
        task: {
            "path": filename,
            "records": counts[task],
            "bytes": (gold / filename).stat().st_size,
            "sha256": sha256_file(gold / filename),
        }
        for _expert, (task, filename) in names.items()
    }
    quality_staging = collection / "automation" / "quality_staging.jsonl"
    negative = collection / "partitions" / "negative.jsonl"
    reject = collection / "partitions" / "reject.jsonl"
    task_bank = collection / "partitions" / "task_bank.jsonl"
    chain_count = len(instances) if formal_split else 1
    _write_jsonl(
        quality_staging,
        [{"partition_index": index} for index in range(5 * chain_count)],
    )
    _write_jsonl(negative, [])
    _write_jsonl(reject, [])
    task_bank_rows = (
        [
            {
                "alignment_id": f"alignment-{index:04d}",
                "card_id": f"card-{index:04d}",
                "seed_id": f"seed-{index:04d}",
                # Mirror canonical TaskCard.to_dict().
                "source": {"instance_id": instance},
            }
            for index, instance in enumerate(instances)
        ]
        if formal_split
        else [
            {
                "alignment_id": "alignment-fixture",
                "card_id": "card-fixture",
                "seed_id": "seed-fixture",
            }
        ]
    )
    _write_jsonl(task_bank, task_bank_rows)
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
        "seed_target": chain_count,
        "raw_collection_target": chain_count,
        "minimum_gold_records_per_task": {
            task: (256 if formal_split else 1) for task in counts
        },
        "staged_count": 5 * chain_count,
        "gold_count": 5 * chain_count,
        "negative_count": 0,
        "reject_count": 0,
        "partition_complete": True,
        "rejects_quarantined": True,
        "gold_integrity_ok": True,
        "reject_reason_counts": {},
        "reject_rate": 0.0,
        "gold_by_task": counts,
        "gold_files": gold_files,
        "gold_label_counts": {},
        "label_quota_errors": [],
        "coverage_complete": ready,
        "coverage_shortfalls": {} if ready else {},
        "raw_by_task": {task: chain_count for task in counts},
        "raw_collection_complete": True,
        "raw_collection_shortfalls": {},
        "lineage_complete": True,
        "complete_chain_count": chain_count,
        "minimum_complete_chain_count": 256 if formal_split else 1,
        "complete_chain_count_sufficient": True,
        "lineage_edge_error_count": 0,
        "lineage_edge_errors_by_edge": {},
        "lineage_edge_errors": [],
        "lineage_chain_error_count": 0,
        "lineage_chain_errors_by_code": {},
        "lineage_chain_errors": [],
        "near_duplicate_gate": {"passed": True, "policy_id": "fixture-v1"},
        "task_card_coverage": {
            "passed": True,
            "cardinality_equal": True,
            "complete_chain_count": chain_count,
            "card_count": chain_count,
            "unique_alignment_id_count": chain_count,
        },
        "task_bank_file": {
            "path": "task_bank.jsonl",
            "records": chain_count,
            "bytes": task_bank.stat().st_size,
            "sha256": sha256_file(task_bank),
        },
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
    formal_lines: list[str] = []
    if formal_split:
        metadata_root = tmp_path / "formal-metadata"
        source_public = ROOT / "datasets/public/swebench-full-bank-v1"
        (metadata_root / "allowlists").mkdir(parents=True)
        shutil.copyfile(
            source_public / "manifest.json", metadata_root / "manifest.json"
        )
        shutil.copyfile(
            source_public / "allowlists/train.json",
            metadata_root / "allowlists/train.json",
        )
        shutil.copyfile(
            source_public / "allowlists/validation-from-train.json",
            metadata_root / "allowlists/validation-from-train.json",
        )
        shutil.copyfile(
            ROOT / "artifacts/benchmark/heldout_v1/manifest.json",
            metadata_root / "heldout-manifest.json",
        )
        shutil.copyfile(
            ROOT / "artifacts/benchmark/heldout_v1/leak_audit.prebulk.json",
            metadata_root / "heldout-leak-audit.json",
        )
        formal_lines = [
            "formal_v3_split:",
            f"  source_bank_manifest: {(metadata_root / 'manifest.json').as_posix()}",
            f"  train_allowlist: {(metadata_root / 'allowlists/train.json').as_posix()}",
            "  calibration_allowlist: "
            + (metadata_root / "allowlists/validation-from-train.json").as_posix(),
            f"  heldout_manifest: {(metadata_root / 'heldout-manifest.json').as_posix()}",
            "  heldout_leak_audit: "
            + (metadata_root / "heldout-leak-audit.json").as_posix(),
        ]
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
                "expected_minimum_gold_records_per_expert: "
                + ("256" if formal_split else "1"),
                *formal_lines,
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


def test_snapshot_fails_closed_when_v2_partition_lacks_lineage_proof(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    for field in (
        "lineage_complete",
        "complete_chain_count",
        "minimum_complete_chain_count",
        "complete_chain_count_sufficient",
        "lineage_edge_error_count",
        "lineage_edge_errors_by_edge",
        "lineage_edge_errors",
        "lineage_chain_error_count",
        "lineage_chain_errors_by_code",
        "lineage_chain_errors",
    ):
        manifest.pop(field)
    config.partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    status = json.loads(config.automation_status.read_text(encoding="utf-8"))
    status["partition"] = manifest
    config.automation_status.write_text(json.dumps(status), encoding="utf-8")

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert result["status"] == "blocked"
    assert "partition_complete_chain_count_invalid" in result["blockers"]
    assert "partition_lineage_edge_summary_invalid" in result["blockers"]
    assert "partition_lineage_chain_summary_invalid" in result["blockers"]
    assert "partition_lineage_incomplete" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_fails_closed_when_partition_lacks_gold_file_bindings(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    manifest.pop("gold_files")
    config.partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    status = json.loads(config.automation_status.read_text(encoding="utf-8"))
    status["partition"] = manifest
    config.automation_status.write_text(json.dumps(status), encoding="utf-8")

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "partition_gold_file_bindings_invalid" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_rejects_gold_file_drift_from_partition_binding(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    plan_path = config.gold_dir / "data_plan.jsonl"
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "gold_file_binding_mismatch:planner" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_requires_near_duplicate_and_task_card_cardinality_gates(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    manifest["near_duplicate_gate"]["passed"] = False
    manifest["task_card_coverage"]["unique_alignment_id_count"] = 0
    config.partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    status = json.loads(config.automation_status.read_text(encoding="utf-8"))
    status["partition"] = manifest
    config.automation_status.write_text(json.dumps(status), encoding="utf-8")

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "partition_near_duplicate_gate_not_passed" in result["blockers"]
    assert "partition_task_card_coverage_invalid" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_rejects_task_bank_drift_from_partition_binding(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    task_bank = config.partition_manifest.parent / "task_bank.jsonl"
    task_bank.write_text(task_bank.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "task_bank_file_binding_mismatch" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_detects_task_bank_change_during_validation(
    tmp_path: Path, monkeypatch
) -> None:
    config = _fixture(tmp_path)
    task_bank = config.partition_manifest.parent / "task_bank.jsonl"
    original = snapshot_module._validate_task_bank_jsonl
    changed = False

    def mutate_after_validation(path: Path) -> int:
        nonlocal changed
        result = original(path)
        if path == task_bank and not changed:
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(
        snapshot_module, "_validate_task_bank_jsonl", mutate_after_validation
    )

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "task_bank_file_changed_during_read" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_detects_gold_change_during_validation(
    tmp_path: Path, monkeypatch
) -> None:
    config = _fixture(tmp_path)
    original = snapshot_module.validate_jsonl
    changed = False

    def mutate_after_validation(path: Path, *args, **kwargs):
        nonlocal changed
        result = original(path, *args, **kwargs)
        if (
            path.name == "data_plan.jsonl"
            and path.parent == config.gold_dir
            and not changed
        ):
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(snapshot_module, "validate_jsonl", mutate_after_validation)

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "gold_file_changed_during_read:planner" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_recomputes_lineage_from_strict_gold_files(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    security_path = config.gold_dir / "data_security.jsonl"
    security = json.loads(security_path.read_text(encoding="utf-8"))
    security["provenance"]["source_review_record_id"] = "missing-review-id"
    _write_jsonl(security_path, [security])

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "partition_lineage_recompute_mismatch" in result["blockers"]
    assert not config.snapshot_dir.exists()


def test_snapshot_rejects_assistant_output_target_mismatch(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    frontend_path = config.gold_dir / "data_frontend.jsonl"
    frontend = json.loads(frontend_path.read_text(encoding="utf-8"))
    frontend["messages"][-1]["content"] = "export const Tampered = () => null;"
    _write_jsonl(frontend_path, [frontend])

    result = prepare_snapshot(config)

    assert result["training_ready"] is False
    assert "gold_schema_invalid:frontend_gen" in result["blockers"]
    assert not config.snapshot_dir.exists()


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
    assert manifest["source_gate"]["lineage_complete"] is True
    assert manifest["source_gate"]["complete_chain_count"] == 1
    assert manifest["source_gate"]["minimum_complete_chain_count"] == 1
    assert manifest["source_gate"]["lineage_edge_error_count"] == 0
    assert manifest["source_gate"]["lineage_chain_error_count"] == 0
    assert manifest["source_gate"]["near_duplicate_gate"]["passed"] is True
    assert manifest["source_gate"]["task_card_coverage"]["cardinality_equal"] is True
    assert manifest["source_gate"]["task_bank_file"]["records"] == 1
    assert set(manifest["source_gate"]["gold_files"]) == {
        "plan",
        "tool_policy",
        "frontend",
        "review",
        "security",
    }
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
    task_bank = manifest["task_bank_file"]
    assert task_bank["path"] == "task_bank.jsonl"
    assert task_bank["records"] == 1
    assert task_bank["source_sha256"] == task_bank["sha256"]
    assert sha256_file(config.snapshot_dir / task_bank["path"]) == task_bank["sha256"]
    expected_parts.append(
        f"task_bank:{task_bank['path']}:{task_bank['sha256']}:{task_bank['records']}"
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


def test_formal_split_prepare_materialize_and_preflight_close_end_to_end(
    tmp_path: Path,
) -> None:
    snapshot_config = _fixture(tmp_path, formal_split=True)
    prepared = prepare_snapshot(snapshot_config)

    assert prepared["status"] == "frozen"
    assert prepared["formal_v3_split"]["heldout_content_read"] is False
    manifest = json.loads(
        (snapshot_config.snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    split = manifest["split_contract"]
    assert manifest["population_contract"]["gold_accepted_tasks"] == 257
    assert split["partitions"]["train"]["gold_task_count"] == 256
    assert split["partitions"]["calibration"]["gold_task_count"] == 1
    assert split["partitions"]["heldout"]["content_read"] is False
    assert "files" not in split["partitions"]["heldout"]
    assert manifest["task_bank_file"]["records"] == 256
    assert split["partitions"]["calibration"]["task_bank_file"]["records"] == 1
    frozen_task_card = json.loads(
        (snapshot_config.snapshot_dir / "task_bank.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert "instance_id" not in frozen_task_card
    assert isinstance(frozen_task_card["source"]["instance_id"], str)

    profile = load_training_config(
        ROOT / "configs/training/formal_v3_lowmem_common.yaml"
    )
    profile = {key: value for key, value in profile.items() if not key.startswith("_")}
    profile["paths"]["project_root"] = "."
    for expert, (_task, filename) in {
        "planner": ("plan", "data_plan.jsonl"),
        "tool_policy": ("tool_policy", "data_tool_policy.jsonl"),
        "frontend_gen": ("frontend", "data_frontend.jsonl"),
        "frontend_review": ("review", "data_review.jsonl"),
        "security_gate": ("security", "data_security.jsonl"),
    }.items():
        relative = (Path("snapshot") / filename).as_posix()
        profile["scale_gate"]["required_datasets"][expert] = relative
        profile["adapters"][expert]["datasets"] = [relative]
    profile["adapters"]["mixed_all"]["datasets"] = [
        (Path("snapshot") / filename).as_posix()
        for filename in (
            "data_plan.jsonl",
            "data_tool_policy.jsonl",
            "data_frontend.jsonl",
            "data_review.jsonl",
            "data_security.jsonl",
        )
    ]
    profile["scale_gate"]["dataset_snapshot"].update(
        {
            "manifest": "snapshot/manifest.json",
            "sidecar": "snapshot/manifest.json.sha256",
        }
    )
    config_path = tmp_path / "formal-v3-C.json"
    config_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    resolved_path = tmp_path / "resolved-C.json"
    materialized = MATERIALIZER.materialize(config_path, "C", resolved_path)
    resolved = load_training_config(resolved_path)
    selected = select_adapter(resolved, "planner", 16)
    datasets = inspect_gate_datasets(selected, tmp_path)
    snapshot_report = inspect_dataset_snapshot_manifest(selected, tmp_path, datasets)

    assert materialized["training_started"] is False
    assert materialized["exposure_plan"]["records_per_stage"] == 256
    assert materialized["exposure_plan"]["max_steps_per_adapter_job"] == 64
    assert snapshot_report["passed"] is True
    assert snapshot_report["split_contract"]["train_records_per_expert"] == {
        expert: 256 for expert in EXPERTS
    }


def test_formal_identity_reader_accepts_real_task_card_and_stage_shapes() -> None:
    instance = "project__repo-123"

    assert snapshot_module._record_instance_id(
        {"source": {"instance_id": instance}},
        task_bank_by_seed={},
        task_bank_by_alignment={},
    ) == instance
    assert snapshot_module._record_instance_id(
        {"input": {"identity": {"instance_id": instance}}},
        task_bank_by_seed={},
        task_bank_by_alignment={},
    ) == instance


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
    assert snapshot.expected_minimum_gold_records_per_expert == 256
    assert snapshot.snapshot_dir == ROOT / "artifacts" / "formal_v3" / "dataset"
    assert snapshot.partition_manifest == (
        ROOT
        / "artifacts/swebench/full-bank-live-v1/training-export/partitions/manifest.json"
    )
    assert snapshot.gold_dir == (
        ROOT
        / "artifacts/swebench/full-bank-live-v1/training-export/partitions/gold"
    )
    assert snapshot.formal_v3_split is not None
    assert snapshot.formal_v3_split.train_allowlist == (
        ROOT / "datasets/public/swebench-full-bank-v1/allowlists/train.json"
    )
    assert snapshot.formal_v3_split.calibration_allowlist == (
        ROOT
        / "datasets/public/swebench-full-bank-v1/allowlists/validation-from-train.json"
    )

    handoff = HandoffConfig(
        ROOT / "configs" / "orchestration" / "distill_train_handoff_v3.yaml"
    )
    assert handoff.state_dir == ROOT / "runs" / "distill-train-handoff-v3"
    assert handoff.snapshot["minimum_records_per_expert"] == 256
    assert handoff.distillation["automation_config"] == (
        "configs/data/automation.full_v3.ark_glm52.max384.c8.yaml"
    )
    assert handoff.distillation["credential_env"] == "ARK_CODING_API_KEY"
    assert set(handoff.snapshot["datasets"].values()) == {
        "artifacts/formal_v3/dataset/data_plan.jsonl",
        "artifacts/formal_v3/dataset/data_tool_policy.jsonl",
        "artifacts/formal_v3/dataset/data_frontend.jsonl",
        "artifacts/formal_v3/dataset/data_review.jsonl",
        "artifacts/formal_v3/dataset/data_security.jsonl",
    }
