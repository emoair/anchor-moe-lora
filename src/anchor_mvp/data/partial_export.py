"""Explicit export of independently trainable per-expert strict gold.

This is intentionally separate from the full-v3 snapshot publisher.  It may
waive corpus-size and end-to-end DAG claims, but it never waives partition
integrity, secret scanning, reject quarantine, or held-out leakage checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

from ..benchmark.heldout import file_sha256
from ..training.schema import DatasetValidationError, iter_jsonl, validate_jsonl
from .cleaning import contains_secret_material
from .schema import TASK_TYPES


EXPORT_SCHEMA_VERSION = "anchor.per-expert-partial-gold-export.v1"
TRAINING_MODE = "per_expert_partial_gold"
_EXPERT_BY_TASK = {
    "plan": "planner",
    "tool_policy": "tool_policy",
    "frontend": "frontend_gen",
    "review": "frontend_review",
    "security": "security_gate",
}
_STOPPED_STATES = frozenset(
    {
        "complete",
        "provider_quota_exhausted",
        "budget_exhausted",
        "client_deadline",
        "failed",
        "gate_blocked",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_existing_export(path: Path, source_sha256: str) -> dict[str, Any]:
    manifest = _load_mapping(path / "manifest.json", label="partial export manifest")
    source = manifest.get("source")
    if (
        manifest.get("schema_version") != EXPORT_SCHEMA_VERSION
        or not isinstance(source, Mapping)
        or source.get("partition_manifest_sha256") != source_sha256
    ):
        raise ValueError("existing partial export does not match its source")
    files = manifest.get("gold_files")
    if not isinstance(files, Mapping):
        raise ValueError("existing partial export file manifest is invalid")
    for task in TASK_TYPES:
        metadata = files.get(task)
        output = path / f"data_{task}.jsonl"
        if (
            not isinstance(metadata, Mapping)
            or not output.is_file()
            or file_sha256(output) != metadata.get("sha256")
            or output.stat().st_size != metadata.get("bytes")
        ):
            raise ValueError("existing partial export file binding mismatch")
    return manifest


def export_partial_expert_gold(config: Any) -> dict[str, Any]:
    """Publish only the five strict-gold files under an explicit waiver mode."""

    # Import lazily so this module can remain a narrow export helper while the
    # automation CLI calls it after automation.py has finished initialization.
    from .automation import (  # noqa: PLC0415
        AUTOMATION_SCHEMA_VERSION,
        _verify_offline_partition_inputs,
        _verify_status_config_binding,
    )

    status = _load_mapping(config.status_path, label="automation status")
    if status.get("schema_version") != AUTOMATION_SCHEMA_VERSION:
        raise ValueError("partial export requires an automation-v2 status")
    _verify_status_config_binding(config, status)
    if (
        status.get("state") not in _STOPPED_STATES
        or status.get("current_worker") not in (None, "")
        or status.get("active_projection_incomplete") is True
        or status.get("partition_stale_reason") is not None
    ):
        raise ValueError("partial export requires a stopped, current partition")

    partition = status.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("automation status has no partition manifest")
    _verify_offline_partition_inputs(config, partition)
    if partition.get("partition_complete") is not True:
        raise ValueError("partial export requires complete partition accounting")
    if partition.get("rejects_quarantined") is not True:
        raise ValueError("partial export requires quarantined rejects")
    if partition.get("gold_integrity_ok") is not True:
        raise ValueError("partial export requires clean strict gold")
    heldout = partition.get("heldout_gate")
    if not isinstance(heldout, Mapping) or heldout.get("passed") is not True:
        raise ValueError("partial export requires a passing held-out leakage gate")

    partition_manifest_path = config.partition_dir / "manifest.json"
    partition_manifest_sha256 = file_sha256(partition_manifest_path)
    declared_files = partition.get("gold_files")
    if not isinstance(declared_files, Mapping) or set(declared_files) != set(
        TASK_TYPES
    ):
        raise ValueError("partition gold file manifest is incomplete")

    verified_files: dict[str, dict[str, Any]] = {}
    for task in TASK_TYPES:
        filename = f"data_{task}.jsonl"
        source = config.partition_dir / "gold" / filename
        metadata = declared_files.get(task)
        if not isinstance(metadata, Mapping) or metadata.get("path") != filename:
            raise ValueError("partition gold file path is invalid")
        if not source.is_file() or source.is_symlink():
            raise ValueError("partition gold file is missing or indirect")
        with source.open("r", encoding="utf-8") as handle:
            record_count = sum(1 for line in handle if line.strip())
        observed = {
            "path": filename,
            "records": record_count,
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        }
        if any(metadata.get(key) != value for key, value in observed.items()):
            raise ValueError("partition gold file binding mismatch")
        if observed["records"] < 1:
            raise ValueError("partial export requires at least one gold row per expert")
        try:
            validation = validate_jsonl(source, allowed_experts=[_EXPERT_BY_TASK[task]])
            if validation.get("ok") is not True:
                raise ValueError("partition gold schema validation failed")
            for _line_number, record in iter_jsonl(source):
                provenance = record.get("provenance")
                if contains_secret_material(record):
                    raise ValueError("strict gold safety scan failed")
                if (
                    isinstance(provenance, Mapping)
                    and provenance.get("source_kind") == "swebench_heldout"
                ):
                    raise ValueError("held-out source found in strict gold")
        except DatasetValidationError as exc:
            raise ValueError("partition gold schema validation failed") from exc
        verified_files[task] = observed

    export_root = config.output_dir / "training_exports" / TRAINING_MODE
    destination = export_root / partition_manifest_sha256
    if destination.exists():
        return _verify_existing_export(destination, partition_manifest_sha256)

    minimums = partition.get("minimum_gold_records_per_task")
    gold_by_task = partition.get("gold_by_task")
    if not isinstance(minimums, Mapping) or not isinstance(gold_by_task, Mapping):
        raise ValueError("partition coverage metadata is invalid")
    quota_errors = partition.get("label_quota_errors")
    if not isinstance(quota_errors, list):
        raise ValueError("partition label quota metadata is invalid")
    task_card_coverage = partition.get("task_card_coverage")
    task_card_passed = bool(
        isinstance(task_card_coverage, Mapping)
        and task_card_coverage.get("passed") is True
    )

    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "training_mode": TRAINING_MODE,
        "not_for_end_to_end_claim": True,
        "source": {
            "partition_manifest_sha256": partition_manifest_sha256,
            "automation_status_sha256": file_sha256(config.status_path),
            "automation_state": status.get("state"),
            "partition_training_ready": partition.get("training_ready") is True,
        },
        "strict_complete_chains": int(partition.get("complete_chain_count", 0)),
        "gold_files": verified_files,
        "gold_records_by_task": {
            task: int(gold_by_task.get(task, 0)) for task in TASK_TYPES
        },
        "waivers": {
            "complete_chain": {
                "applied": partition.get("complete_chain_count_sufficient") is not True,
                "observed": int(partition.get("complete_chain_count", 0)),
                "required": int(partition.get("minimum_complete_chain_count", 0)),
            },
            "coverage": {
                "applied": (
                    partition.get("coverage_complete") is not True
                    or not task_card_passed
                ),
                "minimum_gold_records_per_task": {
                    task: int(minimums.get(task, 0)) for task in TASK_TYPES
                },
                "coverage_shortfalls": dict(partition.get("coverage_shortfalls", {})),
                "task_card_coverage_passed": task_card_passed,
            },
            "label_quota": {
                "applied": bool(quota_errors),
                "error_count": len(quota_errors),
                "errors": [str(error) for error in quota_errors],
            },
        },
        "excluded": {
            "negative": True,
            "reject": True,
            "oracle_label_only": True,
            "heldout": True,
        },
    }

    export_root.mkdir(parents=True, exist_ok=True)
    temporary = export_root / f".tmp-{partition_manifest_sha256}-{uuid4().hex}"
    try:
        temporary.mkdir()
        for task in TASK_TYPES:
            filename = f"data_{task}.jsonl"
            shutil.copyfile(
                config.partition_dir / "gold" / filename, temporary / filename
            )
            if file_sha256(temporary / filename) != verified_files[task]["sha256"]:
                raise ValueError("partial export copy verification failed")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.replace(temporary, destination)
        except FileExistsError:
            return _verify_existing_export(destination, partition_manifest_sha256)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
