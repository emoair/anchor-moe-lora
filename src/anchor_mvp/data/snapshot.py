"""Fail-closed preparation of immutable full-v3 training snapshots.

The readiness report contains counts, hashes, and gate codes only.  It never
copies record bodies or held-out text.  A snapshot is published only after the
partition manifest, automation status, and all five expert files pass together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

import yaml

from .cleaning import contains_secret_material
from ..training.manifest import sha256_file
from ..training.schema import DatasetValidationError, iter_jsonl, validate_jsonl


CONFIG_SCHEMA = "anchor.training-snapshot-config.v1"
PARTITION_SCHEMA = "anchor.automation-partition-manifest.v2"
LEGACY_PARTITION_SCHEMA = "anchor.automation-partition-manifest.v1"
READINESS_SCHEMA = "anchor.training-snapshot-readiness.v1"
SNAPSHOT_SCHEMA = "anchor.training-snapshot.v2"

EXPERT_SOURCES = {
    "planner": ("plan", "data_plan.jsonl"),
    "tool_policy": ("tool_policy", "data_tool_policy.jsonl"),
    "frontend_gen": ("frontend", "data_frontend.jsonl"),
    "frontend_review": ("review", "data_review.jsonl"),
    "security_gate": ("security", "data_security.jsonl"),
}
EXPERTS = tuple(EXPERT_SOURCES)
SAFE_TERMINAL_STATES = frozenset({"provider_quota_exhausted", "complete"})
_SHA256_HEX = frozenset("0123456789abcdef")


class SnapshotPreparationError(RuntimeError):
    """A content-free failure that is safe to persist in the readiness report."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(root: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project_root") from exc
    return resolved


@dataclass(frozen=True)
class SnapshotConfig:
    root: Path
    partition_manifest: Path
    automation_status: Path
    collection_dir: Path
    gold_dir: Path
    snapshot_dir: Path
    readiness_report: Path
    expected_minimum_gold_records_per_expert: int

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotConfig":
        config_path = Path(path).resolve()
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != CONFIG_SCHEMA
        ):
            raise ValueError("unsupported training snapshot config schema")

        def required_path(name: str) -> object:
            raw = value.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{name} must be a non-empty path")
            return raw

        root = (config_path.parent / str(value.get("project_root", "../.."))).resolve()
        if not root.is_dir():
            raise ValueError("project_root is missing")
        expected = value.get("expected_minimum_gold_records_per_expert")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError(
                "expected_minimum_gold_records_per_expert must be a positive integer"
            )
        config = cls(
            root=root,
            partition_manifest=_inside(
                root, required_path("partition_manifest"), "partition_manifest"
            ),
            automation_status=_inside(
                root, required_path("automation_status"), "automation_status"
            ),
            collection_dir=_inside(
                root, required_path("collection_dir"), "collection_dir"
            ),
            gold_dir=_inside(root, required_path("gold_dir"), "gold_dir"),
            snapshot_dir=_inside(root, required_path("snapshot_dir"), "snapshot_dir"),
            readiness_report=_inside(
                root, required_path("readiness_report"), "readiness_report"
            ),
            expected_minimum_gold_records_per_expert=expected,
        )
        if (
            config.snapshot_dir == config.gold_dir
            or config.snapshot_dir == config.collection_dir
        ):
            raise ValueError("snapshot_dir must be separate from collection inputs")
        try:
            config.snapshot_dir.relative_to(config.collection_dir)
        except ValueError:
            pass
        else:
            raise ValueError("snapshot_dir must not be inside collection_dir")
        try:
            config.collection_dir.relative_to(config.snapshot_dir)
        except ValueError:
            pass
        else:
            raise ValueError("snapshot_dir must not contain collection_dir")
        try:
            config.readiness_report.relative_to(config.snapshot_dir)
        except ValueError:
            pass
        else:
            raise ValueError("readiness_report must be outside snapshot_dir")
        if config.readiness_report == config.partition_manifest:
            raise ValueError(
                "readiness_report must not overwrite the partition manifest"
            )
        return config


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _count_nonempty(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _safe_nonnegative(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.casefold()).issubset(_SHA256_HEX)
    )


def _snapshot_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    parts = [
        f"{expert}:{files[expert]['path']}:{files[expert]['sha256']}:{files[expert]['records']}"
        for expert in EXPERTS
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def evaluate_readiness(config: SnapshotConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a public metadata-only report plus private freeze inputs."""

    blockers: list[str] = []

    def block(code: str) -> None:
        if code not in blockers:
            blockers.append(code)

    partition = _read_mapping(config.partition_manifest)
    partition_sha = (
        sha256_file(config.partition_manifest)
        if config.partition_manifest.is_file()
        else None
    )
    if partition is None:
        block("partition_manifest_missing_or_invalid")
        partition = {}
    partition_schema = partition.get("schema_version")
    if partition_schema == LEGACY_PARTITION_SCHEMA:
        block("partition_schema_legacy_without_separate_gold_target")
    elif partition_schema != PARTITION_SCHEMA:
        block("partition_schema_invalid")
    if partition.get("collection_policy") != "collect_then_partition":
        block("partition_collection_policy_invalid")
    if partition.get("training_ready") is not True:
        block("partition_training_ready_false")
    if partition.get("coverage_complete") is not True:
        block("partition_coverage_incomplete")
    if partition.get("partition_complete") is not True:
        block("partition_incomplete")
    if partition.get("rejects_quarantined") is not True:
        block("partition_rejects_not_quarantined")
    if partition.get("gold_integrity_ok") is not True:
        block("partition_gold_integrity_not_passed")
    reject_count = _safe_nonnegative(partition.get("reject_count"))
    if reject_count is None:
        block("partition_reject_count_invalid")
    staged_count = _safe_nonnegative(partition.get("staged_count"))
    negative_count = _safe_nonnegative(partition.get("negative_count"))
    quota_errors = partition.get("label_quota_errors")
    if not isinstance(quota_errors, list) or quota_errors:
        block("partition_label_quotas_incomplete")
    raw_collection_target = _safe_nonnegative(partition.get("raw_collection_target"))
    if raw_collection_target is None and partition_schema == LEGACY_PARTITION_SCHEMA:
        # Legacy seed_target is useful for capacity audit only. It is never
        # accepted as proof of the separate strict-gold floor.
        raw_collection_target = _safe_nonnegative(partition.get("seed_target"))
    if raw_collection_target is None or raw_collection_target < 1:
        block("partition_raw_collection_target_invalid")

    raw_minimums = partition.get("minimum_gold_records_per_task")
    minimums: dict[str, int | None] = {}
    for _expert, (task, _filename) in EXPERT_SOURCES.items():
        value = (
            raw_minimums.get(task)
            if isinstance(raw_minimums, Mapping)
            else (
                config.expected_minimum_gold_records_per_expert
                if partition_schema == LEGACY_PARTITION_SCHEMA
                else None
            )
        )
        minimum = _safe_nonnegative(value)
        minimums[task] = minimum
        if (
            partition_schema == PARTITION_SCHEMA
            and minimum != config.expected_minimum_gold_records_per_expert
        ):
            block(f"partition_minimum_gold_mismatch:{task}")
    if raw_collection_target is not None and any(
        minimum is not None and minimum > raw_collection_target
        for minimum in minimums.values()
    ):
        block("partition_minimum_gold_exceeds_raw_target")

    gold_by_task_raw = partition.get("gold_by_task")
    gold_by_task = gold_by_task_raw if isinstance(gold_by_task_raw, Mapping) else {}
    normalized_gold: dict[str, int | None] = {}
    for _expert, (task, _filename) in EXPERT_SOURCES.items():
        count = _safe_nonnegative(gold_by_task.get(task))
        normalized_gold[task] = count
        if count is None:
            block(f"partition_gold_count_invalid:{task}")
        elif minimums[task] is None or count < minimums[task]:
            block(f"partition_gold_count_below_target:{task}")
    declared_gold = _safe_nonnegative(partition.get("gold_count"))
    known_gold = [item for item in normalized_gold.values() if item is not None]
    if len(known_gold) != len(EXPERTS) or declared_gold != sum(known_gold):
        block("partition_gold_total_mismatch")
    if (
        None in {staged_count, declared_gold, negative_count, reject_count}
        or staged_count != declared_gold + negative_count + reject_count
    ):
        block("partition_disposition_total_mismatch")
    if partition_schema == PARTITION_SCHEMA:
        staging_path = config.collection_dir / "automation" / "quality_staging.jsonl"
        negative_path = config.partition_manifest.parent / "negative.jsonl"
        reject_path = config.partition_manifest.parent / "reject.jsonl"
        sidecars = (
            (
                "quality_staging",
                staging_path,
                staged_count,
                partition.get("quality_staging_sha256"),
            ),
            (
                "negative",
                negative_path,
                negative_count,
                partition.get("negative_sha256"),
            ),
            ("reject", reject_path, reject_count, partition.get("reject_sha256")),
        )
        for name, path, expected_count, expected_sha in sidecars:
            if (
                not path.is_file()
                or expected_count is None
                or _count_nonempty(path) != expected_count
                or not _is_sha256(expected_sha)
                or sha256_file(path) != expected_sha
            ):
                block(f"partition_sidecar_integrity:{name}")
        if reject_path.is_file():
            allowed_reject_keys = {
                "id",
                "schema_version",
                "task_type",
                "source_record_sha256",
                "reason_codes",
                "content_retained",
            }
            try:
                for _line_number, reject in iter_jsonl(reject_path):
                    if (
                        not isinstance(reject, Mapping)
                        or set(reject) != allowed_reject_keys
                        or reject.get("content_retained") is not False
                        or not _is_sha256(reject.get("source_record_sha256"))
                        or not isinstance(reject.get("reason_codes"), list)
                        or not reject["reason_codes"]
                        or not all(
                            isinstance(reason, str) and reason
                            for reason in reject["reason_codes"]
                        )
                    ):
                        block("partition_reject_payload_not_quarantined")
                        break
            except (OSError, DatasetValidationError, ValueError):
                block("partition_reject_payload_not_quarantined")
    computed_shortfalls = {
        task: minimum - (normalized_gold.get(task) or 0)
        for task, minimum in minimums.items()
        if minimum is not None and (normalized_gold.get(task) or 0) < minimum
    }
    if (
        partition_schema == PARTITION_SCHEMA
        and partition.get("coverage_shortfalls") != computed_shortfalls
    ):
        block("partition_coverage_shortfalls_mismatch")

    partition_heldout = partition.get("heldout_gate")
    partition_heldout_passed = bool(
        isinstance(partition_heldout, Mapping)
        and partition_heldout.get("status") == "PASS"
        and partition_heldout.get("passed") is True
        and _safe_nonnegative(partition_heldout.get("collision_count")) == 0
        and partition_heldout.get("content_emitted") is False
    )
    if not partition_heldout_passed:
        block("partition_heldout_gate_not_passed")

    status = _read_mapping(config.automation_status)
    status_sha = (
        sha256_file(config.automation_status)
        if config.automation_status.is_file()
        else None
    )
    heldout_public: dict[str, Any] = {
        "status": None,
        "passed": False,
        "collision_count": None,
        "content_emitted": None,
        "manifest_sha256": None,
        "prebulk_audit_sha256": None,
    }
    if status is None:
        block("automation_status_missing_or_invalid")
        status = {}
    state = status.get("state")
    if state not in SAFE_TERMINAL_STATES:
        block("automation_state_not_safe_terminal")
    if status.get("partition") != partition:
        block("automation_partition_binding_mismatch")
    heldout = status.get("heldout_gate")
    if isinstance(heldout, Mapping):
        heldout_public = {
            "status": heldout.get("status")
            if heldout.get("status") in {"PASS", "FAIL"}
            else None,
            "passed": heldout.get("passed") is True,
            "collision_count": _safe_nonnegative(heldout.get("collision_count")),
            "content_emitted": heldout.get("content_emitted") is True,
            "manifest_sha256": (
                heldout.get("manifest_sha256")
                if isinstance(heldout.get("manifest_sha256"), str)
                else None
            ),
            "prebulk_audit_sha256": (
                heldout.get("prebulk_audit_sha256")
                if isinstance(heldout.get("prebulk_audit_sha256"), str)
                else None
            ),
        }
    if not (
        heldout_public["status"] == "PASS"
        and heldout_public["passed"] is True
        and heldout_public["collision_count"] == 0
        and heldout_public["content_emitted"] is False
    ):
        block("heldout_gate_not_passed")

    files: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    duplicate_ids = 0
    for expert, (task, filename) in EXPERT_SOURCES.items():
        source = config.gold_dir / filename
        records = _count_nonempty(source)
        metadata: dict[str, Any] = {
            "path": filename,
            "exists": source.is_file(),
            "records": records,
            "bytes": source.stat().st_size if source.is_file() else 0,
            "sha256": sha256_file(source) if source.is_file() else None,
            "schema_valid": False,
            "secret_scan_passed": False,
        }
        files[expert] = metadata
        if not source.is_file():
            block(f"gold_file_missing:{expert}")
            continue
        try:
            validation = validate_jsonl(source, allowed_experts=[expert])
            metadata["schema_valid"] = validation.get("ok") is True
            has_secret = False
            for _line_number, record in iter_jsonl(source):
                if contains_secret_material(record):
                    has_secret = True
                identifier = str(record.get("id", ""))
                if identifier in all_ids:
                    duplicate_ids += 1
                all_ids.add(identifier)
            metadata["secret_scan_passed"] = not has_secret
            if has_secret:
                block(f"gold_secret_detected:{expert}")
        except (OSError, DatasetValidationError, ValueError):
            block(f"gold_schema_invalid:{expert}")
        declared = normalized_gold.get(task)
        if declared is None or records != declared:
            block(f"gold_file_count_mismatch:{expert}")
    if duplicate_ids:
        block("gold_cross_expert_duplicate_ids")

    capacity: dict[str, dict[str, int | None]] = {}
    actual_raw_by_task: dict[str, int] = {}
    unreachable: list[str] = []
    maximum_total = 0
    for _expert, (task, filename) in EXPERT_SOURCES.items():
        collected = _count_nonempty(config.collection_dir / filename)
        actual_raw_by_task[task] = collected
        current_gold = normalized_gold.get(task) or 0
        remaining = max((raw_collection_target or 0) - collected, 0)
        maximum = current_gold + remaining
        maximum_total += maximum
        minimum = minimums.get(task)
        capacity[task] = {
            "collected_records": collected,
            "current_gold": current_gold,
            "raw_collection_target": raw_collection_target,
            "minimum_gold_required": minimum,
            "remaining_raw_collection_slots": remaining,
            "maximum_possible_gold_from_current_seed_target": maximum,
            "maximum_possible_gold_from_raw_collection_target": maximum,
        }
        if minimum is None or maximum < minimum:
            unreachable.append(task)
    if unreachable:
        block("coverage_unreachable_with_current_seed_target")
    if partition_schema == PARTITION_SCHEMA:
        if partition.get("raw_by_task") != actual_raw_by_task:
            block("partition_raw_counts_mismatch")
        computed_raw_shortfalls = {
            task: (raw_collection_target or 0) - count
            for task, count in actual_raw_by_task.items()
            if count < (raw_collection_target or 0)
        }
        if partition.get("raw_collection_shortfalls") != computed_raw_shortfalls:
            block("partition_raw_shortfalls_mismatch")
        if partition.get("raw_collection_complete") != (not computed_raw_shortfalls):
            block("partition_raw_completion_mismatch")

    report: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA,
        "generated_at": _utc_now(),
        "scope": "data_snapshot_only",
        "training_ready": not blockers,
        "freeze_performed": False,
        "status": "ready" if not blockers else "blocked",
        "execution_gate": {
            "evaluated": False,
            "required_separately": True,
            "note": "strict accepted gold and session candidates are not inferred from this report",
        },
        "source": {
            "partition_manifest_sha256": partition_sha,
            "automation_status_sha256": status_sha,
            "automation_state": state if isinstance(state, str) else None,
            "raw_collection_target": raw_collection_target,
            "minimum_gold_records_per_task": minimums,
            "expected_minimum_gold_records_per_expert": (
                config.expected_minimum_gold_records_per_expert
            ),
            "gold_count": declared_gold,
            "gold_by_task": normalized_gold,
            "reject_count": reject_count,
            "negative_count": negative_count,
            "staged_count": staged_count,
            "partition_complete": partition.get("partition_complete") is True,
            "rejects_quarantined": partition.get("rejects_quarantined") is True,
            "gold_integrity_ok": partition.get("gold_integrity_ok") is True,
            "label_quota_error_count": len(quota_errors)
            if isinstance(quota_errors, list)
            else None,
            "heldout_gate": heldout_public,
        },
        "capacity": {
            "by_task": capacity,
            "maximum_possible_gold_from_current_seed_target": maximum_total,
            "maximum_possible_gold_from_raw_collection_target": maximum_total,
            "coverage_unreachable_without_overcollection_or_lower_gold_target": unreachable,
        },
        "files": files,
        "cross_expert_duplicate_id_count": duplicate_ids,
        "blockers": blockers,
        "snapshot": None,
    }
    private = {
        "partition_manifest_sha256": partition_sha,
        "automation_status_sha256": status_sha,
        "partition": partition,
        "status": status,
        "files": {
            expert: {
                "source": config.gold_dir / filename,
                "path": filename,
                "records": files[expert]["records"],
                "bytes": files[expert]["bytes"],
                "sha256": files[expert]["sha256"],
            }
            for expert, (_task, filename) in EXPERT_SOURCES.items()
        },
        "heldout_gate": heldout_public,
    }
    return report, private


def _verify_frozen_snapshot(
    output_dir: Path, *, expected_partition_sha256: str | None
) -> tuple[dict[str, Any], str]:
    manifest_path = output_dir / "manifest.json"
    sidecar = output_dir / "manifest.json.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise SnapshotPreparationError("snapshot_existing_incomplete")
    sidecar_parts = sidecar.read_text(encoding="ascii").split()
    manifest_sha = sha256_file(manifest_path)
    if not sidecar_parts or sidecar_parts[0] != manifest_sha:
        raise SnapshotPreparationError("snapshot_manifest_sidecar_mismatch")
    manifest = _read_mapping(manifest_path)
    if manifest is None or manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise SnapshotPreparationError("snapshot_manifest_invalid")
    if manifest.get("source_partition_manifest_sha256") != expected_partition_sha256:
        raise SnapshotPreparationError("snapshot_source_partition_conflict")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping) or tuple(raw_files) != EXPERTS:
        raise SnapshotPreparationError("snapshot_file_binding_invalid")
    files: dict[str, Mapping[str, Any]] = {}
    for expert in EXPERTS:
        item = raw_files.get(expert)
        if not isinstance(item, Mapping):
            raise SnapshotPreparationError("snapshot_file_binding_invalid")
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise SnapshotPreparationError("snapshot_file_path_invalid")
        dataset = output_dir / relative
        if not dataset.is_file():
            raise SnapshotPreparationError("snapshot_dataset_missing")
        if sha256_file(dataset) != item.get("sha256"):
            raise SnapshotPreparationError("snapshot_dataset_hash_mismatch")
        if dataset.stat().st_size != item.get("bytes"):
            raise SnapshotPreparationError("snapshot_dataset_size_mismatch")
        if _count_nonempty(dataset) != item.get("records"):
            raise SnapshotPreparationError("snapshot_dataset_count_mismatch")
        validate_jsonl(dataset, allowed_experts=[expert])
        files[expert] = item
    if _snapshot_digest(files) != manifest.get("snapshot_sha256"):
        raise SnapshotPreparationError("snapshot_digest_mismatch")
    return dict(manifest), manifest_sha


def _freeze(
    config: SnapshotConfig, private: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    partition_sha = private.get("partition_manifest_sha256")
    if not isinstance(partition_sha, str):
        raise SnapshotPreparationError("partition_manifest_hash_missing")
    if config.snapshot_dir.exists():
        return _verify_frozen_snapshot(
            config.snapshot_dir, expected_partition_sha256=partition_sha
        )

    config.snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        config.snapshot_dir.parent / f".{config.snapshot_dir.name}.tmp-{uuid4().hex}"
    )
    temporary.mkdir()
    try:
        manifest_files: dict[str, dict[str, Any]] = {}
        private_files = private.get("files")
        if not isinstance(private_files, Mapping):
            raise SnapshotPreparationError("freeze_inputs_invalid")
        for expert in EXPERTS:
            item = private_files.get(expert)
            if not isinstance(item, Mapping):
                raise SnapshotPreparationError("freeze_inputs_invalid")
            source = item.get("source")
            filename = item.get("path")
            if not isinstance(source, Path) or not isinstance(filename, str):
                raise SnapshotPreparationError("freeze_inputs_invalid")
            destination = temporary / filename
            shutil.copyfile(source, destination)
            copied_sha = sha256_file(destination)
            if copied_sha != item.get("sha256"):
                raise SnapshotPreparationError("source_changed_during_copy")
            validation = validate_jsonl(destination, allowed_experts=[expert])
            records = int(validation["valid_records"])
            if records != item.get("records"):
                raise SnapshotPreparationError("source_changed_during_copy")
            manifest_files[expert] = {
                "path": filename,
                "records": records,
                "bytes": destination.stat().st_size,
                "sha256": copied_sha,
                "source_sha256": copied_sha,
            }

        if sha256_file(config.partition_manifest) != partition_sha:
            raise SnapshotPreparationError("partition_changed_during_copy")
        status_sha = private.get("automation_status_sha256")
        if (
            not isinstance(status_sha, str)
            or sha256_file(config.automation_status) != status_sha
        ):
            raise SnapshotPreparationError("automation_status_changed_during_copy")
        for item in private_files.values():
            if not isinstance(item, Mapping) or sha256_file(item["source"]) != item.get(
                "sha256"
            ):
                raise SnapshotPreparationError("source_changed_during_copy")

        manifest: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA,
            "created_at": _utc_now(),
            "source_partition_manifest_sha256": partition_sha,
            "source_automation_status_sha256": status_sha,
            "selection": "all strict-gold partition records; no resampling",
            "total_records": sum(item["records"] for item in manifest_files.values()),
            "snapshot_sha256": _snapshot_digest(manifest_files),
            "source_gate": {
                "raw_collection_target": private["partition"].get(
                    "raw_collection_target"
                ),
                "minimum_gold_records_per_task": private["partition"].get(
                    "minimum_gold_records_per_task"
                ),
                "collection_policy": private["partition"].get("collection_policy"),
                "gold_count": private["partition"].get("gold_count"),
                "partition_complete": private["partition"].get("partition_complete"),
                "rejects_quarantined": private["partition"].get("rejects_quarantined"),
                "reject_count": private["partition"].get("reject_count"),
                "gold_integrity_ok": private["partition"].get("gold_integrity_ok"),
                "heldout_gate": private["heldout_gate"],
            },
            "files": manifest_files,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest_sha = sha256_file(manifest_path)
        (temporary / "manifest.json.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
        )
        os.replace(temporary, config.snapshot_dir)
        return manifest, manifest_sha
    except FileExistsError:
        return _verify_frozen_snapshot(
            config.snapshot_dir, expected_partition_sha256=partition_sha
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_snapshot(config: SnapshotConfig) -> dict[str, Any]:
    """Write readiness metadata and atomically publish only a fully ready snapshot."""

    report, private = evaluate_readiness(config)
    if not report["training_ready"]:
        _atomic_json(config.readiness_report, report)
        return report
    try:
        existed = config.snapshot_dir.exists()
        manifest, manifest_sha = _freeze(config, private)
    except (OSError, ValueError, SnapshotPreparationError):
        report["training_ready"] = False
        report["status"] = "freeze_failed"
        report["blockers"] = [*report["blockers"], "snapshot_freeze_failed"]
        _atomic_json(config.readiness_report, report)
        return report
    report["freeze_performed"] = not existed
    report["status"] = "already_frozen" if existed else "frozen"
    report["snapshot"] = {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha,
        "snapshot_sha256": manifest["snapshot_sha256"],
        "total_records": manifest["total_records"],
    }
    _atomic_json(config.readiness_report, report)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fail-closed full-v3 snapshot readiness and atomic freeze"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = SnapshotConfig.load(args.config)
        report = prepare_snapshot(config)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"frozen", "already_frozen"} else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
