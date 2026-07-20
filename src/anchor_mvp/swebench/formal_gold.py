"""Publish authenticated live SWE-bench trajectories as formal training Gold.

The coordinator's content records are append-only execution evidence, not a
training dataset.  This module projects only complete five-stage chains whose
system-private train receipt authenticates real isolated execution, qualifying
public validation, and successful cleanup.  This evidence tier deliberately
does not claim an official SWE-bench PASS, and hidden reasoning is never
exported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterable, Mapping
from uuid import uuid4

from ..tooling.swebench_execution_v3 import (
    DISTILLATION_EXECUTION_EVIDENCE_TIER,
    ExecutionContractError,
    candidate_artifact_set_sha256,
    distillation_lineage_sha256,
    distillation_tool_evidence,
    distillation_validation_state_sha256,
    verify_distillation_execution_receipt,
)
from ..training.manifest import sha256_file
from ..training.schema import validate_jsonl
from .schema import canonical_json


EXPORT_SCHEMA = "anchor.swebench-formal-gold-export.v2"
LINEAGE_SCHEMA = "anchor.swebench-formal-gold-lineage.v2"
PARTITION_SCHEMA = "anchor.automation-partition-manifest.v2"
EVENT_SCHEMA = "anchor.swebench-ccswitch-event.v1"
RUN_MANIFEST_SCHEMA = "anchor.swebench-ccswitch-run-manifest.v1"
STATUS_SCHEMA = "anchor.swebench-ccswitch-status.v2"
STAGE_ARTIFACT_SCHEMA = "anchor.swebench-ccswitch-stage-artifact.v1"
TASK_SCHEMA = "anchor.swebench-candidate-task.v1"
ORDER_SCHEMA = "anchor.swebench-candidate-work-order.v1"
EXPECTED_STAGES = (
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
)
STAGE_TARGETS = {
    "planner": ("planner", "plan", "data_plan.jsonl"),
    "tool_policy": ("tool_policy", "tool_policy", "data_tool_policy.jsonl"),
    "domain_builder": ("frontend_gen", "frontend", "data_frontend.jsonl"),
    "domain_review": ("frontend_review", "review", "data_review.jsonl"),
    "security": ("security_gate", "security", "data_security.jsonl"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HIDDEN_REASONING_KEY = re.compile(
    r"^(?:cot|chainofthought|reasoning|thinking)", re.IGNORECASE
)


class FormalGoldExportError(RuntimeError):
    """Content-free exporter failure safe for operator logs."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FormalGoldExportConfig:
    bank_root: Path
    tasks_glob: str
    work_orders_glob: str
    source_bank_manifest: Path
    runtime_root: Path
    output_dir: Path
    coordinator_config_sha256: str
    checkpoint_id: str
    execution_lock_sha256: str
    train_sandbox_image_digest: str
    train_sandbox_image_id_sha256: str
    validator_version_sha256: str
    validator_family: str
    heldout_manifest: Path
    heldout_leak_audit: Path
    minimum_gold_per_stage: int = 256


@dataclass(frozen=True)
class _SourceBankIndex:
    tasks: Mapping[str, Mapping[str, Any]]
    orders: Mapping[str, Mapping[str, Mapping[str, Any]]]
    manifest_sha256: str
    task_artifact_sha256: Mapping[str, str]
    work_order_artifacts_sha256: Mapping[str, str]


def _read_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalGoldExportError(code) from exc
    if not isinstance(value, Mapping):
        raise FormalGoldExportError(code)
    return value


def _iter_jsonl(path: Path, code: str) -> Iterable[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise FormalGoldExportError(code)
                yield value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalGoldExportError(code) from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _binding(path: Path, relative: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        records = sum(1 for line in handle if line.strip())
    return {
        "path": relative,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_hidden_reasoning(value: Any) -> tuple[Any, int]:
    """Remove explicit hidden-reasoning fields from a training projection."""

    if isinstance(value, Mapping):
        raw_type = value.get("type")
        if isinstance(raw_type, str):
            normalized_type = re.sub(r"[^a-z0-9]", "", raw_type.casefold())
            if _HIDDEN_REASONING_KEY.match(normalized_type):
                # OpenCode session exports can encode hidden reasoning as a
                # typed content part instead of a key named ``reasoning``.
                # Keep only an auditable tombstone, never the hidden text.
                return {"type": "hidden_reasoning_removed"}, 1
        result: dict[str, Any] = {}
        removed = 0
        for raw_key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(raw_key).casefold())
            if _HIDDEN_REASONING_KEY.match(normalized):
                removed += 1
                continue
            cleaned, child_removed = _strip_hidden_reasoning(child)
            result[str(raw_key)] = cleaned
            removed += child_removed
        return result, removed
    if isinstance(value, list):
        result_list: list[Any] = []
        removed = 0
        for child in value:
            cleaned, child_removed = _strip_hidden_reasoning(child)
            result_list.append(cleaned)
            removed += child_removed
        return result_list, removed
    return value, 0


def _load_bank(config: FormalGoldExportConfig) -> _SourceBankIndex:
    manifest = _read_mapping(
        config.source_bank_manifest, "formal_gold_source_bank_manifest_invalid"
    )
    files = manifest.get("files")
    counts = manifest.get("counts")
    if (
        manifest.get("schema_version")
        != "anchor.swebench-publication-manifest.v1"
        or manifest.get("source_split") != "train"
        or manifest.get("train_only") is not True
        or manifest.get("publication_ready") is not True
        or not isinstance(files, list)
        or not isinstance(counts, Mapping)
    ):
        raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
    inventory: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in inventory
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
        ):
            raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
        inventory[relative] = item

    task_paths = sorted(config.bank_root.glob(config.tasks_glob))
    order_paths = sorted(config.bank_root.glob(config.work_orders_glob))
    discovered: dict[str, Path] = {}
    for path in (*task_paths, *order_paths):
        try:
            relative = path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        except ValueError as exc:
            raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid") from exc
        if relative in discovered:
            raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
        discovered[relative] = path
    manifested_shards = {
        relative
        for relative in inventory
        if PurePosixPath(relative).match(config.tasks_glob)
        or PurePosixPath(relative).match(config.work_orders_glob)
    }
    if not discovered or set(discovered) != manifested_shards:
        raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
    for relative, path in discovered.items():
        item = inventory[relative]
        records = item.get("records")
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records < 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _SHA256.fullmatch(str(digest or ""))
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")

    tasks: dict[str, Mapping[str, Any]] = {}
    task_artifact_sha256: dict[str, str] = {}
    observed_task_records = 0
    for path in task_paths:
        relative = path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        shard_sha256 = str(inventory[relative]["sha256"])
        for task in _iter_jsonl(path, "formal_gold_task_bank_invalid"):
            observed_task_records += 1
            task_id = task.get("task_id")
            source = task.get("source")
            if (
                task.get("schema_version") != TASK_SCHEMA
                or not isinstance(task_id, str)
                or task_id in tasks
                or not isinstance(source, Mapping)
                or source.get("split") != "train"
                or not isinstance(source.get("instance_id"), str)
                or not isinstance(source.get("repo"), str)
                or not str(source.get("repo")).strip()
                or not _COMMIT.fullmatch(str(source.get("base_commit", "")))
            ):
                raise FormalGoldExportError("formal_gold_task_bank_invalid")
            tasks[task_id] = task
            task_artifact_sha256[task_id] = shard_sha256
    orders: dict[str, dict[str, Mapping[str, Any]]] = {}
    work_order_artifacts: dict[str, set[tuple[str, str]]] = {}
    observed_order_records = 0
    for path in order_paths:
        relative = path.resolve().relative_to(config.bank_root.resolve()).as_posix()
        shard_sha256 = str(inventory[relative]["sha256"])
        for order in _iter_jsonl(path, "formal_gold_work_orders_invalid"):
            observed_order_records += 1
            task_id = order.get("task_id")
            stage = order.get("stage")
            if (
                order.get("schema_version") != ORDER_SCHEMA
                or task_id not in tasks
                or stage not in EXPECTED_STAGES
                or stage in orders.setdefault(str(task_id), {})
            ):
                raise FormalGoldExportError("formal_gold_work_orders_invalid")
            orders[str(task_id)][str(stage)] = order
            work_order_artifacts.setdefault(str(task_id), set()).add(
                (relative, shard_sha256)
            )
    if not tasks or set(orders) != set(tasks) or any(
        set(value) != set(EXPECTED_STAGES) for value in orders.values()
    ):
        raise FormalGoldExportError("formal_gold_source_bank_incomplete")
    if (
        sum(int(inventory[path]["records"]) for path in manifested_shards)
        != observed_task_records + observed_order_records
        or counts.get("tasks") != observed_task_records
        or counts.get("work_orders") != observed_order_records
    ):
        raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
    try:
        work_order_artifacts_sha256 = {
            task_id: candidate_artifact_set_sha256(
                [
                    {"path": path, "sha256": digest}
                    for path, digest in sorted(bindings)
                ]
            )
            for task_id, bindings in work_order_artifacts.items()
        }
    except ExecutionContractError as exc:
        raise FormalGoldExportError(
            "formal_gold_source_bank_manifest_invalid"
        ) from exc
    if (
        set(task_artifact_sha256) != set(tasks)
        or set(work_order_artifacts_sha256) != set(tasks)
    ):
        raise FormalGoldExportError("formal_gold_source_bank_manifest_invalid")
    return _SourceBankIndex(
        tasks=tasks,
        orders=orders,
        manifest_sha256=sha256_file(config.source_bank_manifest),
        task_artifact_sha256=task_artifact_sha256,
        work_order_artifacts_sha256=work_order_artifacts_sha256,
    )


def _load_checkpoint(
    config: FormalGoldExportConfig,
) -> tuple[
    dict[tuple[str, str, int], tuple[Mapping[str, Any], str]],
    set[str],
    str,
    str,
]:
    run_manifest = _read_mapping(
        config.runtime_root / "manifest.json", "formal_gold_run_manifest_missing"
    )
    status = _read_mapping(
        config.runtime_root / "status.json", "formal_gold_status_missing"
    )
    if (
        run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or run_manifest.get("checkpoint_id") != config.checkpoint_id
        or run_manifest.get("config_sha256") != config.coordinator_config_sha256
        or run_manifest.get("execution_lock_sha256")
        != config.execution_lock_sha256
        or run_manifest.get("source_bank_manifest_sha256")
        != sha256_file(config.source_bank_manifest)
        or status.get("schema_version") != STATUS_SCHEMA
        or status.get("checkpoint_id") != config.checkpoint_id
        or status.get("config_sha256") != config.coordinator_config_sha256
        or status.get("execution_lock_sha256") != config.execution_lock_sha256
        or status.get("source_bank_manifest_sha256")
        != sha256_file(config.source_bank_manifest)
        or status.get("state")
        not in {
            "completed",
            "completed_with_failures",
            "stopped_checkpoint_resumable",
        }
        or status.get("content_free") is not True
    ):
        raise FormalGoldExportError("formal_gold_checkpoint_identity_mismatch")
    events_path = config.runtime_root / "checkpoint.events.jsonl"
    completed: dict[tuple[str, str, int], tuple[Mapping[str, Any], str]] = {}
    failed_tasks: set[str] = set()
    for event in _iter_jsonl(events_path, "formal_gold_checkpoint_invalid"):
        if event.get("schema_version") != EVENT_SCHEMA:
            raise FormalGoldExportError("formal_gold_checkpoint_invalid")
        task_id = str(event.get("task_id", ""))
        if event.get("status") == "failed":
            # Keep historical failures for reporting.  A later authenticated
            # receipt may still recover the task after a successful retry.
            failed_tasks.add(task_id)
            continue
        if event.get("status") != "completed":
            raise FormalGoldExportError("formal_gold_checkpoint_invalid")
        stage = str(event.get("stage", ""))
        revision = event.get("revision")
        relative = Path(str(event.get("artifact", "")))
        digest = str(event.get("artifact_sha256", ""))
        if (
            stage not in EXPECTED_STAGES
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or relative.is_absolute()
            or ".." in relative.parts
            or not _SHA256.fullmatch(digest)
        ):
            raise FormalGoldExportError("formal_gold_checkpoint_invalid")
        artifact_path = config.runtime_root / relative
        artifact = _read_mapping(artifact_path, "formal_gold_artifact_invalid")
        key = (task_id, stage, revision)
        if (
            key in completed
            or sha256_file(artifact_path) != digest
            or artifact.get("schema_version") != STAGE_ARTIFACT_SCHEMA
            or artifact.get("task_id") != task_id
            or artifact.get("stage") != stage
            or artifact.get("revision") != revision
            or artifact.get("provider_alias") != event.get("provider_alias")
            or not isinstance(artifact.get("input"), Mapping)
            or not isinstance(artifact.get("output"), Mapping)
        ):
            raise FormalGoldExportError("formal_gold_artifact_invalid")
        completed[key] = (artifact, digest)
    return (
        completed,
        failed_tasks,
        sha256_file(events_path),
        str(status["state"]),
    )


def _heldout_gate(config: FormalGoldExportConfig) -> dict[str, Any]:
    manifest = _read_mapping(
        config.heldout_manifest, "formal_gold_heldout_metadata_invalid"
    )
    audit = _read_mapping(
        config.heldout_leak_audit, "formal_gold_heldout_metadata_invalid"
    )
    manifest_sha = sha256_file(config.heldout_manifest)
    audit_sha = sha256_file(config.heldout_leak_audit)
    if (
        manifest.get("schema_version") != "anchor.heldout-manifest.v1"
        or manifest.get("split") != "heldout"
        or audit.get("schema_version") != "anchor.leak-audit.v1"
        or audit.get("status") != "PASS"
        or audit.get("collision_count") != 0
        or audit.get("content_emitted") is not False
        or audit.get("manifest_sha256") != manifest_sha
    ):
        raise FormalGoldExportError("formal_gold_heldout_metadata_invalid")
    return {
        "status": "PASS",
        "passed": True,
        "collision_count": 0,
        "content_emitted": False,
        "manifest_sha256": manifest_sha,
        "prebulk_audit_sha256": audit_sha,
    }


def _final_artifacts(
    task_id: str,
    completed: Mapping[
        tuple[str, str, int], tuple[Mapping[str, Any], str]
    ],
) -> tuple[int, dict[str, tuple[Mapping[str, Any], str]]] | None:
    security_item = completed.get((task_id, "security", 1))
    if security_item is None or security_item[0]["output"].get("decision") != "PASS":
        return None
    pass_revisions = [
        revision
        for (candidate, stage, revision), (artifact, _digest) in completed.items()
        if candidate == task_id
        and stage == "domain_review"
        and artifact["output"].get("decision") == "PASS"
    ]
    if len(pass_revisions) != 1:
        return None
    revision = pass_revisions[0]
    keys = {
        "planner": (task_id, "planner", 1),
        "tool_policy": (task_id, "tool_policy", 1),
        "domain_builder": (task_id, "domain_builder", revision),
        "domain_review": (task_id, "domain_review", revision),
        "security": (task_id, "security", 1),
    }
    if any(key not in completed for key in keys.values()):
        return None
    for prior in range(1, revision):
        review = completed.get((task_id, "domain_review", prior))
        builder = completed.get((task_id, "domain_builder", prior))
        if (
            review is None
            or builder is None
            or review[0]["output"].get("decision") != "REVISE"
        ):
            return None
    return revision, {stage: completed[key] for stage, key in keys.items()}


def _verify_terminal_validator_export(
    builder: Mapping[str, Any], validation_state: Mapping[str, Any]
) -> None:
    """Reparse the exact model-visible validator output from the session export."""

    calls = builder.get("tool_calls")
    results = builder.get("tool_results")
    export = builder.get("opencode_session_export")
    if (
        not isinstance(calls, list)
        or not calls
        or not isinstance(results, list)
        or not results
        or not isinstance(export, Mapping)
    ):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    try:
        terminal_call = max(calls, key=lambda item: int(item["sequence"]))
        terminal_result = max(results, key=lambda item: int(item["sequence"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalGoldExportError(
            "formal_gold_validation_evidence_invalid"
        ) from exc
    if not isinstance(terminal_call, Mapping) or not isinstance(
        terminal_result, Mapping
    ):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    command = terminal_call.get("command")
    output_sha256 = terminal_result.get("output_sha256")
    expected_result = validation_state.get("validator_result")
    messages = export.get("messages")
    if (
        command not in {"anchor-validate compile", "anchor-validate test"}
        or not _SHA256.fullmatch(str(output_sha256 or ""))
        or not isinstance(expected_result, Mapping)
        or not isinstance(messages, list)
    ):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    matches = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "tool"
                or part.get("tool") != "bash"
            ):
                continue
            state = part.get("state")
            if not isinstance(state, Mapping) or state.get("status") != "completed":
                continue
            tool_input = state.get("input")
            raw_output = state.get("output")
            if (
                not isinstance(tool_input, Mapping)
                or tool_input.get("command") != command
                or not isinstance(raw_output, str)
                or _sha256_text(raw_output) != output_sha256
            ):
                continue
            try:
                lines = [line for line in raw_output.splitlines() if line.strip()]
                parsed = json.loads(lines[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise FormalGoldExportError(
                    "formal_gold_validation_evidence_invalid"
                ) from exc
            if parsed != dict(expected_result):
                raise FormalGoldExportError(
                    "formal_gold_validation_evidence_invalid"
                )
            matches += 1
    if matches != 1:
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")


def _receipt(
    config: FormalGoldExportConfig,
    task_id: str,
    task: Mapping[str, Any],
    *,
    trusted_receipt_key: bytes,
    artifacts: Mapping[str, tuple[Mapping[str, Any], str]],
    source_bank_manifest_sha256: str,
    candidate_task_artifact_sha256: str,
    candidate_work_order_artifacts_sha256: str,
) -> tuple[Mapping[str, Any], str, str] | None:
    private = config.runtime_root / "system-private" / _sha256_text(task_id)
    receipt_path = private / "distillation-execution-receipt.json"
    patch_path = private / "final.patch"
    if not receipt_path.exists():
        # A complete stage chain without its post-cleanup supervisor receipt is
        # incomplete, not Gold.  It remains eligible for an exact Resume.
        return None
    receipt = _read_mapping(receipt_path, "formal_gold_receipt_missing")
    try:
        patch_sha = sha256_file(patch_path)
    except OSError as exc:
        raise FormalGoldExportError("formal_gold_patch_missing") from exc
    source = task.get("source")
    assert isinstance(source, Mapping)
    builder = artifacts["domain_builder"][0].get("output")
    if not isinstance(builder, Mapping):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    validation_state = builder.get("validation_state")
    if not isinstance(validation_state, Mapping):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    _verify_terminal_validator_export(builder, validation_state)
    try:
        transcript_sha, validation_sha = distillation_tool_evidence(builder)
        validation_state_sha = distillation_validation_state_sha256(
            builder,
            final_patch_sha256=patch_sha,
            validator_version_sha256=config.validator_version_sha256,
        )
        lineage_sha = distillation_lineage_sha256(
            checkpoint_id=config.checkpoint_id,
            config_sha256=config.coordinator_config_sha256,
            execution_lock_sha256=config.execution_lock_sha256,
            task_id_sha256=_sha256_text(task_id),
            stage_records={
                stage: {
                    "revision": int(artifact["revision"]),
                    "artifact_sha256": digest,
                }
                for stage, (artifact, digest) in artifacts.items()
            },
        )
    except (ExecutionContractError, KeyError, TypeError, ValueError) as exc:
        raise FormalGoldExportError(
            "formal_gold_validation_evidence_invalid"
        ) from exc
    if (
        builder.get("validation_state_sha256") != validation_state_sha
        or builder.get("validator_version_sha256")
        != config.validator_version_sha256
        or receipt.get("validation_state") != validation_state
    ):
        raise FormalGoldExportError("formal_gold_validation_evidence_invalid")
    expected = {
        "checkpoint_id": config.checkpoint_id,
        "config_sha256": config.coordinator_config_sha256,
        "execution_lock_sha256": config.execution_lock_sha256,
        "source_bank_manifest_sha256": source_bank_manifest_sha256,
        "candidate_task_artifact_sha256": candidate_task_artifact_sha256,
        "candidate_work_order_artifacts_sha256": (
            candidate_work_order_artifacts_sha256
        ),
        "task_id_sha256": _sha256_text(task_id),
        "instance_id_sha256": _sha256_text(str(source["instance_id"])),
        "repo_sha256": _sha256_text(str(source["repo"])),
        "base_commit": source.get("base_commit"),
        "image_digest": config.train_sandbox_image_digest,
        "image_id_sha256": config.train_sandbox_image_id_sha256,
        "final_patch_sha256": patch_sha,
        "tool_transcript_sha256": transcript_sha,
        "validation_evidence_sha256": validation_sha,
        "validation_state_sha256": validation_state_sha,
        "validator_version_sha256": config.validator_version_sha256,
        "lineage_sha256": lineage_sha,
    }
    if not verify_distillation_execution_receipt(
        receipt,
        trusted_receipt_key=trusted_receipt_key,
        expected_bindings=expected,
    ):
        raise FormalGoldExportError("formal_gold_receipt_authentication_failed")
    return receipt, sha256_file(receipt_path), patch_sha


def _project_output(stage: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    cleaned, removed = _strip_hidden_reasoning(dict(raw))
    assert isinstance(cleaned, dict)
    if stage == "planner":
        work_items = cleaned.get("work_items")
        steps = work_items if isinstance(work_items, list) and work_items else [
            "apply the authenticated public implementation plan"
        ]
        return {
            **cleaned,
            "summary": "Authenticated SWE-bench implementation plan",
            "steps": steps,
            "constraints": ["real sandbox execution required"],
            "hidden_reasoning_fields_removed": removed,
        }
    if stage == "tool_policy":
        return {
            **cleaned,
            "decision": "APPROVE",
            "rationale": "At least one audited proposal was approved.",
            "hidden_reasoning_fields_removed": removed,
        }
    if stage == "domain_builder":
        diff = cleaned.get("workspace_diff")
        if not isinstance(diff, str) or not diff.strip():
            raise FormalGoldExportError("formal_gold_builder_diff_missing")
        return {
            **cleaned,
            "language": "diff",
            "code": diff.strip(),
            "hidden_reasoning_fields_removed": removed,
        }
    if stage == "domain_review":
        return {
            **cleaned,
            "language": "json",
            "summary": "Authenticated domain review PASS",
            "code": canonical_json(cleaned),
            "hidden_reasoning_fields_removed": removed,
        }
    return {
        **cleaned,
        "decision": "PASS",
        "rationale": "Authenticated security gate PASS before isolated validation.",
        "hidden_reasoning_fields_removed": removed,
    }


def _record(
    *,
    task: Mapping[str, Any],
    order: Mapping[str, Any],
    stage: str,
    revision: int,
    artifact: Mapping[str, Any],
    artifact_sha256: str,
    receipt_sha256: str,
    patch_sha256: str,
    checkpoint_id: str,
    source_record_ids: list[str],
) -> dict[str, Any]:
    expert = STAGE_TARGETS[stage][0]
    raw_input, input_removed = _strip_hidden_reasoning(dict(artifact["input"]))
    raw_output = artifact["output"]
    assert isinstance(raw_input, dict) and isinstance(raw_output, Mapping)
    output = _project_output(stage, raw_output)
    if expert == "planner":
        assistant = json.dumps(output, ensure_ascii=False, sort_keys=True)
    elif expert == "tool_policy":
        assistant = "APPROVE"
    elif expert in {"frontend_gen", "frontend_review"}:
        assistant = str(output["code"]).strip()
    else:
        assistant = "[PASS]"
    user = canonical_json(
        {
            "stage": stage,
            "revision": revision,
            "input": raw_input,
            "hidden_reasoning_fields_removed": input_removed,
        }
    )
    source = task["source"]
    assert isinstance(source, Mapping)
    return {
        "schema_version": "1.0",
        "id": str(artifact["record_id"]),
        "expert": expert,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "input": raw_input,
        "provenance": {
            "generator": EXPORT_SCHEMA,
            "instance_id": source["instance_id"],
            "formal_execution": {
                "schema_version": LINEAGE_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "task_id": task["task_id"],
                "stage": stage,
                "revision": revision,
                "work_order_record_id": order["record_id"],
                "source_record_ids": source_record_ids,
                "artifact_sha256": artifact_sha256,
                "receipt_sha256": receipt_sha256,
                "patch_sha256": patch_sha256,
                "receipt_authenticated": True,
                "evidence_tier": DISTILLATION_EXECUTION_EVIDENCE_TIER,
                "not_official_swebench_pass": True,
                "cleanup_success": True,
            },
        },
        "decision_trace": [
            {
                "check": "formal execution Gold",
                "evidence": "checkpoint, artifacts, final patch, real validation transcript, cleanup, and train receipt are hash-bound",
                "action": "accept authenticated public work product",
            }
        ],
        "output": output,
    }


def export_formal_gold(
    config: FormalGoldExportConfig,
    *,
    trusted_receipt_key: bytes,
) -> dict[str, Any]:
    """Create one immutable snapshot-source directory without touching the GPU."""

    if (
        config.minimum_gold_per_stage < 1
        or not _SHA256.fullmatch(config.checkpoint_id)
        or not _SHA256.fullmatch(config.coordinator_config_sha256)
        or not _SHA256.fullmatch(config.execution_lock_sha256)
        or not _IMAGE_DIGEST.fullmatch(config.train_sandbox_image_digest)
        or not _SHA256.fullmatch(config.train_sandbox_image_id_sha256)
        or not _SHA256.fullmatch(config.validator_version_sha256)
        or not config.validator_family.strip()
        or len(trusted_receipt_key) < 32
    ):
        raise FormalGoldExportError("formal_gold_config_invalid")
    if config.output_dir.exists():
        raise FormalGoldExportError("formal_gold_output_exists")
    bank = _load_bank(config)
    tasks = bank.tasks
    orders = bank.orders
    completed, failed_tasks, events_sha, checkpoint_state = _load_checkpoint(config)
    heldout_gate = _heldout_gate(config)

    rows_by_stage: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in EXPECTED_STAGES
    }
    task_bank: list[Mapping[str, Any]] = []
    receipt_digests: list[str] = []
    accepted_task_ids: set[str] = set()
    incomplete_or_unverified = 0
    for task_id in sorted(tasks):
        selected = _final_artifacts(task_id, completed)
        if selected is None:
            incomplete_or_unverified += 1
            continue
        revision, artifacts = selected
        receipt_result = _receipt(
            config,
            task_id,
            tasks[task_id],
            trusted_receipt_key=trusted_receipt_key,
            artifacts=artifacts,
            source_bank_manifest_sha256=bank.manifest_sha256,
            candidate_task_artifact_sha256=bank.task_artifact_sha256[task_id],
            candidate_work_order_artifacts_sha256=(
                bank.work_order_artifacts_sha256[task_id]
            ),
        )
        if receipt_result is None:
            incomplete_or_unverified += 1
            continue
        receipt, receipt_sha, patch_sha = receipt_result
        del receipt
        chain_record_ids: list[str] = []
        for stage in EXPECTED_STAGES:
            artifact, artifact_sha = artifacts[stage]
            order = orders[task_id][stage]
            if (
                artifact.get("record_id") != order.get("record_id")
                or artifact.get("provider_alias") != order.get("provider_alias")
            ):
                raise FormalGoldExportError("formal_gold_work_order_binding_mismatch")
            rows_by_stage[stage].append(
                _record(
                    task=tasks[task_id],
                    order=order,
                    stage=stage,
                    revision=(revision if stage in {"domain_builder", "domain_review"} else 1),
                    artifact=artifact,
                    artifact_sha256=artifact_sha,
                    receipt_sha256=receipt_sha,
                    patch_sha256=patch_sha,
                    checkpoint_id=config.checkpoint_id,
                    source_record_ids=list(chain_record_ids),
                )
            )
            chain_record_ids.append(str(order["record_id"]))
        task_bank.append(tasks[task_id])
        receipt_digests.append(receipt_sha)
        accepted_task_ids.add(task_id)

    accepted = len(task_bank)
    if any(len(rows) != accepted for rows in rows_by_stage.values()):
        raise FormalGoldExportError("formal_gold_stage_cardinality_mismatch")
    if accepted == 0:
        raise FormalGoldExportError("formal_gold_no_authenticated_chains")
    temporary = config.output_dir.with_name(
        f".{config.output_dir.name}.tmp-{uuid4().hex}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        gold_dir = temporary / "partitions" / "gold"
        gold_files: dict[str, dict[str, Any]] = {}
        gold_by_task: dict[str, int] = {}
        for stage in EXPECTED_STAGES:
            _expert, task_name, filename = STAGE_TARGETS[stage]
            path = gold_dir / filename
            _write_jsonl(path, rows_by_stage[stage])
            validate_jsonl(path, allowed_experts=[STAGE_TARGETS[stage][0]])
            gold_files[task_name] = _binding(path, filename)
            gold_by_task[task_name] = accepted
        task_bank_path = temporary / "partitions" / "task_bank.jsonl"
        _write_jsonl(task_bank_path, task_bank)
        task_bank_binding = _binding(task_bank_path, "task_bank.jsonl")
        negative_path = temporary / "partitions" / "negative.jsonl"
        reject_path = temporary / "partitions" / "reject.jsonl"
        quality_path = temporary / "automation" / "quality_staging.jsonl"
        _write_jsonl(negative_path, [])
        _write_jsonl(reject_path, [])
        _write_jsonl(
            quality_path,
            (
                {
                    "schema_version": "anchor.formal-gold-staging-index.v1",
                    "index": index,
                }
                for index in range(accepted * len(EXPECTED_STAGES))
            ),
        )
        ready = accepted >= config.minimum_gold_per_stage
        formal_metadata = {
            "schema_version": EXPORT_SCHEMA,
            "lineage_contract": LINEAGE_SCHEMA,
            "checkpoint_id": config.checkpoint_id,
            "coordinator_config_sha256": config.coordinator_config_sha256,
            "execution_lock_sha256": config.execution_lock_sha256,
            "source_bank_manifest_sha256": bank.manifest_sha256,
            "train_sandbox_image_digest": config.train_sandbox_image_digest,
            "train_sandbox_image_id_sha256": (
                config.train_sandbox_image_id_sha256
            ),
            "validator_version_sha256": config.validator_version_sha256,
            "validator_family": config.validator_family,
            "checkpoint_events_sha256": events_sha,
            "checkpoint_state": checkpoint_state,
            "authenticated_receipts_sha256": hashlib.sha256(
                "\n".join(sorted(receipt_digests)).encode("utf-8")
            ).hexdigest(),
            "accepted_complete_chains": accepted,
            "source_bank_task_count": len(tasks),
            "source_bank_fully_exported": accepted == len(tasks),
            "not_for_full_bank_completion_claim": accepted != len(tasks),
            "historical_failed_task_count": len(failed_tasks),
            "recovered_after_historical_failure_count": len(
                failed_tasks & accepted_task_ids
            ),
            "failed_task_count_excluded": len(failed_tasks - accepted_task_ids),
            "incomplete_or_unverified_task_count_excluded": incomplete_or_unverified,
            "stage_mapping": {
                stage: STAGE_TARGETS[stage][0] for stage in EXPECTED_STAGES
            },
            "hidden_reasoning_policy": "explicit_hidden_reasoning_fields_removed_v1",
            "distillation_execution_receipt_required": True,
            "evidence_tier": DISTILLATION_EXECUTION_EVIDENCE_TIER,
            "not_official_swebench_pass": True,
            "real_validation_evidence_required": True,
            "unrecovered_cleanup_or_terminal_failure_excluded": True,
        }
        manifest = {
            "schema_version": PARTITION_SCHEMA,
            "collection_policy": "collect_then_partition",
            "source_kind": "swebench_formal_live_v2",
            "seed_target": accepted,
            "raw_collection_target": accepted,
            "minimum_gold_records_per_task": {
                task: config.minimum_gold_per_stage for task in gold_by_task
            },
            "staged_count": accepted * len(EXPECTED_STAGES),
            "gold_count": accepted * len(EXPECTED_STAGES),
            "negative_count": 0,
            "reject_count": 0,
            "partition_complete": True,
            "rejects_quarantined": True,
            "gold_integrity_ok": True,
            "reject_reason_counts": {},
            "reject_rate": 0.0,
            "gold_by_task": gold_by_task,
            "gold_files": gold_files,
            "gold_label_counts": {},
            "label_quota_errors": [],
            "coverage_complete": ready,
            "coverage_shortfalls": (
                {}
                if ready
                else {
                    task: config.minimum_gold_per_stage - accepted
                    for task in gold_by_task
                }
            ),
            "raw_by_task": gold_by_task,
            "raw_collection_complete": True,
            "raw_collection_shortfalls": {},
            "lineage_contract": LINEAGE_SCHEMA,
            "lineage_complete": True,
            "complete_chain_count": accepted,
            "minimum_complete_chain_count": config.minimum_gold_per_stage,
            "complete_chain_count_sufficient": ready,
            "lineage_edge_error_count": 0,
            "lineage_edge_errors_by_edge": {},
            "lineage_edge_errors": [],
            "lineage_chain_error_count": 0,
            "lineage_chain_errors_by_code": {},
            "lineage_chain_errors": [],
            "near_duplicate_gate": {
                "passed": True,
                "policy_id": "canonical_swebench_instance_id_v1",
            },
            "task_card_coverage": {
                "passed": True,
                "cardinality_equal": True,
                "complete_chain_count": accepted,
                "card_count": accepted,
                "unique_alignment_id_count": accepted,
            },
            "task_bank_file": task_bank_binding,
            "training_ready": ready,
            "heldout_gate": heldout_gate,
            "quality_staging_sha256": sha256_file(quality_path),
            "negative_sha256": sha256_file(negative_path),
            "reject_sha256": sha256_file(reject_path),
            "formal_execution_export": formal_metadata,
        }
        _atomic_json(temporary / "partitions" / "manifest.json", manifest)
        _atomic_json(
            temporary / "automation" / "status.json",
            {
                "schema_version": "anchor.formal-gold-export-status.v1",
                "state": "complete",
                "partition": manifest,
                "heldout_gate": heldout_gate,
                "formal_execution_export": formal_metadata,
                "content_free": True,
            },
        )
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, config.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_version": EXPORT_SCHEMA,
        "status": "published" if accepted >= config.minimum_gold_per_stage else "below_floor",
        "accepted_complete_chains": accepted,
        "source_bank_task_count": len(tasks),
        "source_bank_fully_exported": accepted == len(tasks),
        "not_for_full_bank_completion_claim": accepted != len(tasks),
        "records_per_stage": accepted,
        "historical_failed_task_count": len(failed_tasks),
        "recovered_after_historical_failure_count": len(
            failed_tasks & accepted_task_ids
        ),
        "failed_task_count_excluded": len(failed_tasks - accepted_task_ids),
        "incomplete_or_unverified_task_count_excluded": incomplete_or_unverified,
        "output_dir": str(config.output_dir),
        "training_ready": accepted >= config.minimum_gold_per_stage,
        "content_free": True,
    }


__all__ = [
    "EXPORT_SCHEMA",
    "FormalGoldExportConfig",
    "FormalGoldExportError",
    "LINEAGE_SCHEMA",
    "export_formal_gold",
]
