"""Deterministic, provider-free TaskBoard projection from frozen formal Gold.

This module is intentionally downstream of canonical Gold.  It authenticates a
complete ``anchor.training-snapshot.v2`` directory, builds causal role views,
and publishes a separate research sidecar without changing the snapshot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping
from uuid import uuid4

import yaml

from anchor_mvp.data.cleaning import contains_secret_material
from anchor_mvp.swebench.schema import canonical_json
from anchor_mvp.training.schema import DatasetValidationError, validate_record


CONFIG_SCHEMA = "anchor.swebench-taskboard-projector-config.v1"
PROJECTOR_VERSION = "anchor.swebench-taskboard-projector.v1"
SNAPSHOT_SCHEMA = "anchor.training-snapshot.v2"
SPLIT_SCHEMA = "anchor.formal-v3-gold-splits.v1"
LINEAGE_SCHEMA = "anchor.swebench-formal-gold-lineage.v2"
SIDECAR_SCHEMA = "anchor.swebench-taskboard-sidecar.v1"
QUERY_SCHEMA = "anchor.query-specialization.v1"
MANIFEST_SCHEMA = "anchor.swebench-taskboard-projector-manifest.v1"

STAGES = (
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
)
STAGE_EXPERTS = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "domain_builder": "frontend_gen",
    "domain_review": "frontend_review",
    "security": "security_gate",
}
EXPERT_STAGES = {expert: stage for stage, expert in STAGE_EXPERTS.items()}
EXPERTS = tuple(STAGE_EXPERTS[stage] for stage in STAGES)
STAGE_ACTIONS = {
    "planner": "plan",
    "tool_policy": "authorize",
    "domain_builder": "implement",
    "domain_review": "review",
    "security": "security_gate",
}
STAGE_BLOCK_KINDS = {
    "planner": "plan",
    "tool_policy": "constraint",
    "domain_builder": "code",
    "domain_review": "review",
    "security": "review",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PRIVATE_ABSOLUTE_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]users[\\/](?:air|administrator|runneradmin)[\\/]|"
    r"/(?:root/(?:\.|workspace|project)|home/(?:runner|codex)/)[^\s\"'`]+)"
)
_HIDDEN_KEY = re.compile(r"^(?:cot|chainofthought|reasoning|thinking)", re.I)
_BLOCK_KINDS = {
    "requirement", "constraint", "plan", "repository", "code", "tool_call",
    "tool_result", "test_result", "review", "history",
}
_COMMIT_STATES = {"candidate", "verified", "committed", "rejected"}


class TaskBoardProjectorError(RuntimeError):
    """Content-free failure safe for an operator log."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise TaskBoardProjectorError(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


@dataclass(frozen=True)
class _BytesSnapshot:
    data: bytes
    sha256: str
    size: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_bytes_snapshot(path: Path, code: str) -> _BytesSnapshot:
    """Read one regular file once and bind the bytes to the opened inode."""

    if not path.is_file() or path.is_symlink():
        _fail(code)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise TaskBoardProjectorError(code) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(
        data=data,
        sha256=_sha256_bytes(data),
        size=len(data),
    )


def _decode_utf8(snapshot: _BytesSnapshot, code: str) -> str:
    try:
        return snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskBoardProjectorError(code) from exc


def _json_from_snapshot(snapshot: _BytesSnapshot, code: str) -> Any:
    try:
        return json.loads(_decode_utf8(snapshot, code))
    except json.JSONDecodeError as exc:
        raise TaskBoardProjectorError(code) from exc


def _jsonl_from_snapshot(
    snapshot: _BytesSnapshot, code: str
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    records = 0
    for line in _decode_utf8(snapshot, code).splitlines():
        if not line.strip():
            continue
        records += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskBoardProjectorError(code) from exc
        if not isinstance(value, Mapping):
            _fail(code)
        rows.append(dict(value))
    return rows, records


def _binding_from_snapshot(
    snapshot: _BytesSnapshot, relative: str, records: int
) -> dict[str, Any]:
    return {
        "path": relative,
        "records": records,
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _required_mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


@dataclass(frozen=True)
class TaskBoardProjectorConfig:
    """Checked-in deterministic projection policy."""

    path: Path
    sha256: str
    projector_version: str
    record_schema: str
    sidecar_schema: str
    sidecar_schema_sha256: str
    manifest_schema: str
    manifest_schema_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "TaskBoardProjectorConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file() or config_path.is_symlink():
            _fail("projector_config_missing")
        try:
            config_snapshot = _read_bytes_snapshot(
                config_path, "projector_config_invalid"
            )
            raw = config_snapshot.data
            value = yaml.safe_load(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise TaskBoardProjectorError("projector_config_invalid") from exc
        root = _required_mapping(value, "projector_config_invalid")
        _exact_keys(
            root,
            {
                "schema_version",
                "projector_version",
                "input_contract",
                "output_contract",
                "partitions",
                "causal_visibility",
                "noise",
            },
            "projector_config_invalid",
        )
        if root.get("schema_version") != CONFIG_SCHEMA:
            _fail("projector_config_schema_invalid")
        projector_version = root.get("projector_version")
        if projector_version != PROJECTOR_VERSION:
            _fail("projector_config_invalid")

        input_contract = _required_mapping(
            root.get("input_contract"), "projector_config_input_invalid"
        )
        _exact_keys(
            input_contract,
            {
                "snapshot_schema_version",
                "require_manifest",
                "require_sha256_sidecar",
                "required_splits",
                "heldout_content_read",
            },
            "projector_config_input_invalid",
        )
        if (
            input_contract.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
            or input_contract.get("require_manifest") is not True
            or input_contract.get("require_sha256_sidecar") is not True
            or input_contract.get("required_splits") != ["train", "calibration"]
            or input_contract.get("heldout_content_read") is not False
        ):
            _fail("projector_config_input_invalid")

        output = _required_mapping(
            root.get("output_contract"), "projector_config_output_invalid"
        )
        _exact_keys(
            output,
            {
                "record_schema_version",
                "sidecar_schema_version",
                "manifest_schema_version",
                "canonical_gold_written",
                "provider_requests",
                "heldout_content_emitted",
            },
            "projector_config_output_invalid",
        )
        if (
            output.get("record_schema_version") != QUERY_SCHEMA
            or output.get("sidecar_schema_version") != SIDECAR_SCHEMA
            or output.get("manifest_schema_version") != MANIFEST_SCHEMA
            or output.get("canonical_gold_written") is not False
            or output.get("provider_requests") != 0
            or output.get("heldout_content_emitted") is not False
        ):
            _fail("projector_config_output_invalid")

        partitions = _required_mapping(
            root.get("partitions"), "projector_config_partitions_invalid"
        )
        _exact_keys(
            partitions,
            {
                "split_before_augmentation",
                "split_preserved",
                "split_group_key",
                "task_id_cross_binding_key",
                "all_five_role_views_same_split",
                "train",
                "calibration",
            },
            "projector_config_partitions_invalid",
        )
        train = _required_mapping(
            partitions.get("train"), "projector_config_partitions_invalid"
        )
        calibration = _required_mapping(
            partitions.get("calibration"), "projector_config_partitions_invalid"
        )
        if (
            set(train) != {"variants"}
            or train.get("variants") != ["clean", "noisy"]
            or set(calibration) != {"variants"}
            or calibration.get("variants") != ["clean"]
            or partitions.get("split_before_augmentation") is not True
            or partitions.get("split_preserved") is not True
            or partitions.get("split_group_key") != "task_bundle_sha256"
            or partitions.get("task_id_cross_binding_key")
            != "training_record.task_board.task_id"
            or partitions.get("all_five_role_views_same_split") is not True
        ):
            _fail("projector_config_partitions_invalid")

        causal = _required_mapping(
            root.get("causal_visibility"), "projector_config_causal_invalid"
        )
        _exact_keys(
            causal,
            {
                "enforce_stage_visibility",
                "reject_future_stage_blocks",
                "require_visible_to_expert",
                "source_of_truth",
            },
            "projector_config_causal_invalid",
        )
        if (
            causal.get("enforce_stage_visibility") is not True
            or causal.get("reject_future_stage_blocks") is not True
            or causal.get("require_visible_to_expert") is not True
            or causal.get("source_of_truth") != "committed_blocks"
        ):
            _fail("projector_config_causal_invalid")

        noise = _required_mapping(root.get("noise"), "projector_config_noise_invalid")
        _exact_keys(
            noise,
            {
                "strategy",
                "enabled_splits",
                "same_task_only",
                "cross_task_content_allowed",
                "preserve_clean_pair",
                "stale_marker_location",
            },
            "projector_config_noise_invalid",
        )
        if (
            noise.get("strategy") != "stale_duplicate_overlay"
            or noise.get("enabled_splits") != ["train"]
            or noise.get("same_task_only") is not True
            or noise.get("cross_task_content_allowed") is not False
            or noise.get("preserve_clean_pair") is not True
            or noise.get("stale_marker_location") != "sidecar.augmentation"
        ):
            _fail("projector_config_noise_invalid")
        sidecar_schema_path = config_path.parent / "taskboard_projector_sidecar.schema.json"
        if not sidecar_schema_path.is_file() or sidecar_schema_path.is_symlink():
            _fail("projector_sidecar_schema_missing")
        sidecar_snapshot = _read_bytes_snapshot(
            sidecar_schema_path, "projector_sidecar_schema_invalid"
        )
        sidecar_schema_value = _json_from_snapshot(
            sidecar_snapshot, "projector_sidecar_schema_invalid"
        )
        if (
            not isinstance(sidecar_schema_value, Mapping)
            or sidecar_schema_value.get("properties", {})
            .get("schema_version", {})
            .get("const")
            != SIDECAR_SCHEMA
        ):
            _fail("projector_sidecar_schema_invalid")
        manifest_schema_path = (
            config_path.parent / "taskboard_projector_manifest.schema.json"
        )
        if not manifest_schema_path.is_file() or manifest_schema_path.is_symlink():
            _fail("projector_manifest_schema_missing")
        manifest_schema_snapshot = _read_bytes_snapshot(
            manifest_schema_path, "projector_manifest_schema_invalid"
        )
        manifest_schema_value = _json_from_snapshot(
            manifest_schema_snapshot, "projector_manifest_schema_invalid"
        )
        if (
            not isinstance(manifest_schema_value, Mapping)
            or manifest_schema_value.get("properties", {})
            .get("schema_version", {})
            .get("const")
            != MANIFEST_SCHEMA
        ):
            _fail("projector_manifest_schema_invalid")
        return cls(
            path=config_path,
            sha256=_sha256_bytes(raw),
            projector_version=projector_version,
            record_schema=QUERY_SCHEMA,
            sidecar_schema=SIDECAR_SCHEMA,
            sidecar_schema_sha256=sidecar_snapshot.sha256,
            manifest_schema=MANIFEST_SCHEMA,
            manifest_schema_sha256=manifest_schema_snapshot.sha256,
        )


def _safe_source_file(root: Path, relative: Any, code: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        _fail(code)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TaskBoardProjectorError(code) from exc
    if not candidate.is_file() or candidate.is_symlink():
        _fail(code)
    return candidate


def _verify_declared_binding(
    root: Path, item: Any, *, expected_records: int | None, code: str
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    binding = _required_mapping(item, code)
    _exact_keys(binding, {"path", "records", "bytes", "sha256"}, code)
    path = _safe_source_file(root, binding.get("path"), code)
    snapshot = _read_bytes_snapshot(path, code)
    rows, records = _jsonl_from_snapshot(snapshot, code)
    observed = _binding_from_snapshot(snapshot, str(binding["path"]), records)
    if observed != dict(binding) or (
        expected_records is not None and observed["records"] != expected_records
    ):
        _fail(code)
    return path, observed, rows


def _snapshot_digest(
    files: Mapping[str, Mapping[str, Any]], task_bank: Mapping[str, Any]
) -> str:
    parts = [
        f"{expert}:{files[expert]['path']}:{files[expert]['sha256']}:{files[expert]['records']}"
        for expert in EXPERTS
    ]
    parts.append(
        f"task_bank:{task_bank['path']}:{task_bank['sha256']}:{task_bank['records']}"
    )
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _contains_hidden_reasoning(value: Any) -> bool:
    if isinstance(value, Mapping):
        raw_type = value.get("type")
        if isinstance(raw_type, str) and _HIDDEN_KEY.match(
            re.sub(r"[^a-z0-9]", "", raw_type.casefold())
        ):
            return True
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if _HIDDEN_KEY.match(normalized) and not normalized.endswith("removed"):
                return True
            if _contains_hidden_reasoning(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_hidden_reasoning(item) for item in value)
    return False


def _contains_private_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_private_absolute_path(key)
            or _contains_private_absolute_path(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_absolute_path(item) for item in value)
    return isinstance(value, str) and _PRIVATE_ABSOLUTE_PATH.search(value) is not None


def _contains_credential_pair(value: Any) -> bool:
    credential_keys = {
        "apikey",
        "accesstoken",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "secretkey",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in credential_keys:
                candidates = child if isinstance(child, (list, tuple)) else (child,)
                if any(
                    isinstance(item, str) and len(item.strip()) >= 8
                    for item in candidates
                ):
                    return True
            if _contains_credential_pair(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_credential_pair(item) for item in value)
    return False


def _content_safe(value: Any) -> bool:
    return not (
        contains_secret_material(value)
        or _contains_credential_pair(value)
        or _contains_hidden_reasoning(value)
        or _contains_private_absolute_path(value)
    )


def _instance_id(value: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [value, value.get("source"), value.get("provenance")]
    raw_input = value.get("input")
    candidates.append(raw_input)
    if isinstance(raw_input, Mapping):
        candidates.extend((raw_input.get("identity"), raw_input.get("source")))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for field in ("instance_id", "source_instance_id"):
            item = candidate.get(field)
            if isinstance(item, str) and item:
                return item
    return None


def _verify_source_gate(manifest: Mapping[str, Any]) -> None:
    gate = _required_mapping(manifest.get("source_gate"), "source_gate_invalid")
    count = gate.get("complete_chain_count")
    minimum = gate.get("minimum_complete_chain_count")
    coverage = gate.get("task_card_coverage")
    near = gate.get("near_duplicate_gate")
    heldout = gate.get("heldout_gate")
    if (
        gate.get("partition_complete") is not True
        or gate.get("rejects_quarantined") is not True
        or gate.get("gold_integrity_ok") is not True
        or gate.get("lineage_complete") is not True
        or gate.get("complete_chain_count_sufficient") is not True
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 1
        or count < minimum
        or gate.get("lineage_edge_error_count") != 0
        or gate.get("lineage_chain_error_count") != 0
        or not isinstance(near, Mapping)
        or near.get("passed") is not True
        or not isinstance(coverage, Mapping)
        or coverage.get("passed") is not True
        or coverage.get("cardinality_equal") is not True
        or coverage.get("complete_chain_count") != count
        or coverage.get("card_count") != count
        or coverage.get("unique_alignment_id_count") != count
        or not isinstance(heldout, Mapping)
        or heldout.get("status") != "PASS"
        or heldout.get("passed") is not True
        or heldout.get("collision_count") != 0
        or heldout.get("content_emitted") is not False
        or not _is_sha256(heldout.get("manifest_sha256"))
        or not _is_sha256(heldout.get("prebulk_audit_sha256"))
    ):
        _fail("source_gate_invalid")


@dataclass(frozen=True)
class _PartitionInput:
    name: str
    gold: Mapping[str, list[dict[str, Any]]]
    file_bindings: Mapping[str, Mapping[str, Any]]
    file_paths: Mapping[str, Path]
    task_bank: list[dict[str, Any]]
    task_bank_binding: Mapping[str, Any]
    task_bank_path: Path


def _load_partition(
    snapshot: Path,
    *,
    name: str,
    files: Mapping[str, Any],
    task_bank_item: Any,
    expected_tasks: int,
) -> _PartitionInput:
    gold: dict[str, list[dict[str, Any]]] = {}
    bindings: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    if set(files) != set(EXPERTS):
        _fail(f"{name}_gold_files_invalid")
    for expert in EXPERTS:
        path, binding, rows = _verify_declared_binding(
            snapshot,
            files.get(expert),
            expected_records=expected_tasks,
            code=f"{name}_gold_binding_invalid",
        )
        for index, row in enumerate(rows, start=1):
            try:
                observed = validate_record(row, source=f"{name}:{expert}:{index}")
            except DatasetValidationError as exc:
                raise TaskBoardProjectorError(f"{name}_gold_record_invalid") from exc
            if observed != expert or not _content_safe(row):
                _fail(f"{name}_gold_record_invalid")
        gold[expert] = rows
        bindings[expert] = binding
        paths[expert] = path
    bank_path, bank_binding, bank = _verify_declared_binding(
        snapshot,
        task_bank_item,
        expected_records=expected_tasks,
        code=f"{name}_task_bank_binding_invalid",
    )
    if any(not _content_safe(row) for row in bank):
        _fail(f"{name}_task_bank_invalid")
    return _PartitionInput(
        name=name,
        gold=gold,
        file_bindings=bindings,
        file_paths=paths,
        task_bank=bank,
        task_bank_binding=bank_binding,
        task_bank_path=bank_path,
    )


def _load_snapshot(
    snapshot_dir: Path, expected_manifest_sha256: str
) -> tuple[Mapping[str, Any], str, tuple[_PartitionInput, _PartitionInput], dict[str, dict[str, Any]]]:
    snapshot = snapshot_dir.resolve()
    if (
        not _is_sha256(expected_manifest_sha256)
        or not snapshot.is_dir()
        or snapshot.is_symlink()
    ):
        _fail("snapshot_input_invalid")
    manifest_path = _safe_source_file(snapshot, "manifest.json", "snapshot_manifest_missing")
    sidecar_path = _safe_source_file(
        snapshot, "manifest.json.sha256", "snapshot_manifest_sidecar_missing"
    )
    manifest_snapshot = _read_bytes_snapshot(
        manifest_path, "snapshot_manifest_invalid"
    )
    sidecar_snapshot = _read_bytes_snapshot(
        sidecar_path, "snapshot_manifest_sidecar_invalid"
    )
    manifest_sha = manifest_snapshot.sha256
    try:
        parts = sidecar_snapshot.data.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise TaskBoardProjectorError("snapshot_manifest_sidecar_invalid") from exc
    manifest = _json_from_snapshot(manifest_snapshot, "snapshot_manifest_invalid")
    if (
        manifest_sha != expected_manifest_sha256
        or parts != [manifest_sha, "manifest.json"]
        or not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != SNAPSHOT_SCHEMA
    ):
        _fail("snapshot_manifest_identity_mismatch")
    _verify_source_gate(manifest)

    split = _required_mapping(manifest.get("split_contract"), "formal_split_invalid")
    roles = _required_mapping(split.get("partitions"), "formal_split_invalid")
    population = _required_mapping(
        manifest.get("population_contract"), "formal_split_invalid"
    )
    if (
        split.get("schema_version") != SPLIT_SCHEMA
        or split.get("assignment") != "source_bank_split_then_gold_gate_v1"
        or split.get("pairwise_disjoint") is not True
        or split.get("gold_coverage_complete") is not True
        or split.get("heldout_content_read") is not False
        or split.get("heldout_content_emitted") is not False
        or not _is_sha256(split.get("leakage_audit_sha256"))
        or set(roles) != {"train", "calibration", "heldout"}
        or population.get("candidate_tasks_per_stage") != 19_008
        or population.get("work_orders_per_task") != 5
        or population.get("candidate_work_orders") != 95_040
    ):
        _fail("formal_split_invalid")
    train_meta = _required_mapping(roles["train"], "formal_train_split_invalid")
    calibration_meta = _required_mapping(
        roles["calibration"], "formal_calibration_split_invalid"
    )
    heldout = _required_mapping(roles["heldout"], "formal_heldout_split_invalid")
    train_count = train_meta.get("gold_task_count")
    calibration_count = calibration_meta.get("gold_task_count")
    if (
        train_meta.get("role") != "training_only"
        or train_meta.get("source_partition") != "train"
        or train_meta.get("candidate_task_count") != 17_105
        or isinstance(train_count, bool)
        or not isinstance(train_count, int)
        or train_count < 1
        or not _is_sha256(train_meta.get("ids_sha256"))
        or train_meta.get("gold_records_per_expert")
        != {expert: train_count for expert in EXPERTS}
        or calibration_meta.get("role") != "rank_allocation_only"
        or calibration_meta.get("source_partition") != "validation-from-train"
        or calibration_meta.get("candidate_task_count") != 1_903
        or isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count < 1
        or not _is_sha256(calibration_meta.get("ids_sha256"))
        or not _is_sha256(calibration_meta.get("snapshot_sha256"))
        or calibration_meta.get("gold_records_per_expert")
        != {expert: calibration_count for expert in EXPERTS}
        or heldout.get("role") != "evaluation_only_hash_metadata"
        or heldout.get("source_partition") != "external-heldout"
        or heldout.get("content_present") is not False
        or heldout.get("content_read") is not False
        or heldout.get("content_emitted") is not False
        or not _is_sha256(heldout.get("ids_sha256"))
        or not _is_sha256(heldout.get("manifest_sha256"))
        or "files" in heldout
        or population.get("gold_accepted_tasks") != train_count + calibration_count
    ):
        _fail("formal_split_contract_invalid")

    train_files_raw = _required_mapping(
        manifest.get("files"), "train_gold_files_invalid"
    )
    source_gate = _required_mapping(manifest.get("source_gate"), "source_gate_invalid")
    source_gold_files = _required_mapping(
        source_gate.get("gold_files"), "snapshot_source_gold_binding_invalid"
    )
    task_names = {
        "planner": "plan",
        "tool_policy": "tool_policy",
        "frontend_gen": "frontend",
        "frontend_review": "review",
        "security_gate": "security",
    }
    train_files: dict[str, dict[str, Any]] = {}
    if set(train_files_raw) != set(EXPERTS) or set(source_gold_files) != set(
        task_names.values()
    ):
        _fail("snapshot_source_gold_binding_invalid")
    for expert in EXPERTS:
        item = _required_mapping(
            train_files_raw.get(expert), "train_gold_binding_invalid"
        )
        if set(item) != {"path", "records", "bytes", "sha256", "source_sha256"}:
            _fail("train_gold_binding_invalid")
        if item.get("source_sha256") != item.get("sha256"):
            _fail("snapshot_source_gold_binding_invalid")
        source_item = source_gold_files.get(task_names[expert])
        if not isinstance(source_item, Mapping) or dict(source_item) != {
            "path": item.get("path"),
            "records": item.get("records"),
            "bytes": item.get("bytes"),
            "sha256": item.get("source_sha256"),
        }:
            _fail("snapshot_source_gold_binding_invalid")
        train_files[expert] = {
            key: item[key] for key in ("path", "records", "bytes", "sha256")
        }
    train_bank = _required_mapping(
        manifest.get("task_bank_file"), "train_task_bank_binding_invalid"
    )
    if set(train_bank) != {"path", "records", "bytes", "sha256", "source_sha256"}:
        _fail("train_task_bank_binding_invalid")
    source_task_bank = source_gate.get("task_bank_file")
    if not isinstance(source_task_bank, Mapping) or dict(source_task_bank) != {
        "path": train_bank.get("path"),
        "records": train_bank.get("records"),
        "bytes": train_bank.get("bytes"),
        "sha256": train_bank.get("source_sha256"),
    }:
        _fail("snapshot_source_task_bank_binding_invalid")
    train = _load_partition(
        snapshot,
        name="train",
        files=train_files,
        task_bank_item={key: train_bank[key] for key in ("path", "records", "bytes", "sha256")},
        expected_tasks=train_count,
    )
    if train_bank.get("source_sha256") != train_bank.get("sha256"):
        _fail("train_task_bank_binding_invalid")
    train_digest_files = {
        expert: {
            **dict(train.file_bindings[expert]),
            "source_sha256": train.file_bindings[expert]["sha256"],
        }
        for expert in EXPERTS
    }
    if _snapshot_digest(train_digest_files, train_bank) != manifest.get("snapshot_sha256"):
        _fail("snapshot_sha256_mismatch")

    calibration_files = _required_mapping(
        calibration_meta.get("files"), "calibration_gold_files_invalid"
    )
    calibration = _load_partition(
        snapshot,
        name="calibration",
        files=calibration_files,
        task_bank_item=calibration_meta.get("task_bank_file"),
        expected_tasks=calibration_count,
    )
    for partition_input, partition_meta in (
        (train, train_meta),
        (calibration, calibration_meta),
    ):
        instance_ids = [_instance_id(row) for row in partition_input.task_bank]
        if (
            any(not isinstance(item, str) or not item for item in instance_ids)
            or len(set(instance_ids)) != len(instance_ids)
            or _sha256_bytes(
                "\n".join(sorted(str(item) for item in instance_ids)).encode("utf-8")
            )
            != partition_meta.get("ids_sha256")
        ):
            _fail(f"{partition_input.name}_split_ids_sha256_mismatch")
    calibration_digest_bank = {
        **dict(calibration.task_bank_binding),
        "source_sha256": calibration.task_bank_binding["sha256"],
    }
    calibration_digest_files = {
        expert: {**dict(item), "source_sha256": item["sha256"]}
        for expert, item in calibration.file_bindings.items()
    }
    if (
        _snapshot_digest(calibration_digest_files, calibration_digest_bank)
        != calibration_meta.get("snapshot_sha256")
    ):
        _fail("calibration_snapshot_sha256_mismatch")

    inventory_paths = {
        "manifest.json": manifest_path,
        "manifest.json.sha256": sidecar_path,
        **{f"train:{key}": path for key, path in train.file_paths.items()},
        "train:task_bank": train.task_bank_path,
        **{
            f"calibration:{key}": path
            for key, path in calibration.file_paths.items()
        },
        "calibration:task_bank": calibration.task_bank_path,
    }
    expected_by_label: dict[str, Mapping[str, Any]] = {
        "manifest.json": {
            "bytes": manifest_snapshot.size,
            "sha256": manifest_snapshot.sha256,
        },
        "manifest.json.sha256": {
            "bytes": sidecar_snapshot.size,
            "sha256": sidecar_snapshot.sha256,
        },
        **{
            f"train:{expert}": train.file_bindings[expert]
            for expert in EXPERTS
        },
        "train:task_bank": train.task_bank_binding,
        **{
            f"calibration:{expert}": calibration.file_bindings[expert]
            for expert in EXPERTS
        },
        "calibration:task_bank": calibration.task_bank_binding,
    }
    inventory = {
        label: {
            "path": path,
            "bytes": int(expected_by_label[label]["bytes"]),
            "sha256": str(expected_by_label[label]["sha256"]),
        }
        for label, path in inventory_paths.items()
    }
    return manifest, manifest_sha, (train, calibration), inventory


def _formal(record: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = _required_mapping(record.get("provenance"), "gold_lineage_invalid")
    if provenance.get("generator") != "anchor.swebench-formal-gold-export.v2":
        _fail("gold_lineage_invalid")
    formal = _required_mapping(
        provenance.get("formal_execution"), "gold_lineage_invalid"
    )
    return formal


def _task_bundle_sha256(task_id: str, entries: list[dict[str, Any]]) -> str:
    return _sha256_value({"task_id": task_id, "entries": entries})


def _bundle_partition(partition: _PartitionInput) -> list[dict[str, Any]]:
    bank_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_bank_tasks: set[str] = set()
    seen_bank_instances: set[str] = set()
    for row in partition.task_bank:
        task_id = row.get("task_id")
        instance = _instance_id(row)
        if not isinstance(task_id, str) or not task_id or not instance:
            _fail(f"{partition.name}_task_bank_identity_invalid")
        if task_id in seen_bank_tasks or instance in seen_bank_instances:
            _fail(f"{partition.name}_task_bank_identity_duplicate")
        seen_bank_tasks.add(task_id)
        seen_bank_instances.add(instance)
        bank_by_pair.setdefault((task_id, instance), []).append(row)

    bundles: dict[str, dict[str, dict[str, Any]]] = {}
    identities: dict[str, tuple[str, str]] = {}
    for expert in EXPERTS:
        expected_stage = EXPERT_STAGES[expert]
        for record in partition.gold[expert]:
            formal = _formal(record)
            provenance = record["provenance"]
            assert isinstance(provenance, Mapping)
            task_id = formal.get("task_id")
            instance = provenance.get("instance_id")
            if (
                formal.get("schema_version") != LINEAGE_SCHEMA
                or formal.get("stage") != expected_stage
                or record.get("expert") != expert
                or formal.get("work_order_record_id") != record.get("id")
                or not isinstance(task_id, str)
                or not task_id
                or not isinstance(instance, str)
                or not instance
                or formal.get("receipt_authenticated") is not True
                or formal.get("evidence_tier") != "real_sandbox_self_verified"
                or formal.get("not_official_swebench_pass") is not True
                or formal.get("cleanup_success") is not True
                or any(
                    not _is_sha256(formal.get(field))
                    for field in ("checkpoint_id", "artifact_sha256", "receipt_sha256", "patch_sha256")
                )
            ):
                _fail(f"{partition.name}_gold_lineage_invalid")
            revision = formal.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                _fail(f"{partition.name}_gold_lineage_invalid")
            source_ids = formal.get("source_record_ids")
            if not isinstance(source_ids, list) or not all(
                isinstance(item, str) and item for item in source_ids
            ):
                _fail(f"{partition.name}_gold_lineage_invalid")
            if expected_stage in bundles.setdefault(task_id, {}):
                _fail(f"{partition.name}_duplicate_stage")
            bundles[task_id][expected_stage] = record
            prior = identities.setdefault(task_id, (task_id, instance))
            if prior != (task_id, instance):
                _fail(f"{partition.name}_task_identity_fork")

    result: list[dict[str, Any]] = []
    if len(bundles) != len(partition.task_bank):
        _fail(f"{partition.name}_bundle_coverage_invalid")
    for task_id in sorted(bundles):
        stages = bundles[task_id]
        if set(stages) != set(STAGES):
            _fail(f"{partition.name}_bundle_incomplete")
        instance = identities[task_id][1]
        matches = bank_by_pair.get((task_id, instance), [])
        if len(matches) != 1:
            _fail(f"{partition.name}_task_bank_match_invalid")
        checkpoint: str | None = None
        receipt: str | None = None
        patch: str | None = None
        ordered_ids: list[str] = []
        builder_revision: int | None = None
        entries: list[dict[str, Any]] = []
        for stage in STAGES:
            record = stages[stage]
            formal = _formal(record)
            if formal.get("source_record_ids") != ordered_ids:
                _fail(f"{partition.name}_lineage_prefix_invalid")
            ordered_ids.append(str(record["id"]))
            identity = (
                str(formal["checkpoint_id"]),
                str(formal["receipt_sha256"]),
                str(formal["patch_sha256"]),
            )
            if checkpoint is None:
                checkpoint, receipt, patch = identity
            elif identity != (checkpoint, receipt, patch):
                _fail(f"{partition.name}_lineage_fork")
            if stage == "domain_builder":
                builder_revision = int(formal["revision"])
            elif stage == "domain_review" and formal.get("revision") != builder_revision:
                _fail(f"{partition.name}_builder_review_revision_mismatch")
            elif stage in {"planner", "tool_policy", "security"} and formal.get("revision") != 1:
                _fail(f"{partition.name}_stage_revision_invalid")
            record_sha = _sha256_value(record)
            entries.append(
                {
                    "stage": stage,
                    "expert": STAGE_EXPERTS[stage],
                    "record_id": record["id"],
                    "record_sha256": record_sha,
                }
            )
        result.append(
            {
                "task_id": task_id,
                "instance_id": instance,
                "task_bank": matches[0],
                "records": stages,
                "entries": entries,
                "task_bundle_sha256": _task_bundle_sha256(task_id, entries),
            }
        )
    return result


def _block_id(task_bundle_sha256: str, label: str, content: str) -> str:
    return "tb-block-v1:" + _sha256_value(
        {"task_bundle_sha256": task_bundle_sha256, "label": label, "content": content}
    )


def _language(task_bank: Mapping[str, Any]) -> str:
    bilingual = _required_mapping(
        task_bank.get("bilingual"), "task_bank_bilingual_contract_invalid"
    )
    requested = bilingual.get("requested_locale")
    source = bilingual.get("source_locale")
    status = bilingual.get("localization_status")
    if source != "en-US" or status not in {"source_ready", "translation_required"}:
        _fail("task_bank_bilingual_contract_invalid")
    if requested == "en-US" and status == "source_ready":
        return "en"
    if requested == "zh-CN" and status in {"source_ready", "translation_required"}:
        return "zh-CN"
    _fail("task_bank_bilingual_contract_invalid")


def _optional_evidence(
    *, task_bundle_sha: str, label: str, kind: str, value: Any, visible_to: list[str]
) -> dict[str, Any] | None:
    if value is None or value == [] or value == {} or value == "":
        return None
    content = value if isinstance(value, str) else canonical_json(value)
    if not content.strip():
        return None
    return {
        "id": _block_id(task_bundle_sha, label, content),
        "kind": kind,
        "content": content,
        "commit_state": "committed",
        "visible_to": visible_to,
    }


def _base_board(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Mapping[str, tuple[str, ...]], str]:
    task_bundle_sha = str(bundle["task_bundle_sha256"])
    task_bank = _required_mapping(bundle["task_bank"], "task_bank_record_invalid")
    public_input = _required_mapping(
        task_bank.get("public_input"), "task_bank_record_invalid"
    )
    problem = public_input.get("problem_statement")
    source = _required_mapping(task_bank.get("source"), "task_bank_record_invalid")
    if not isinstance(problem, str) or not problem.strip():
        _fail("task_bank_record_invalid")
    repository = canonical_json(
        {
            key: source[key]
            for key in (
                "dataset_id",
                "dataset_revision",
                "instance_id",
                "repo",
                "base_commit",
                "split",
                "derived_partition",
            )
            if key in source
        }
    )
    blocks: list[dict[str, Any]] = [
        {
            "id": _block_id(task_bundle_sha, "requirement", problem),
            "kind": "requirement",
            "content": problem,
            "commit_state": "committed",
            "visible_to": list(EXPERTS),
        },
        {
            "id": _block_id(task_bundle_sha, "repository", repository),
            "kind": "repository",
            "content": repository,
            "commit_state": "committed",
            "visible_to": list(EXPERTS),
        },
    ]
    produced_ids: dict[str, tuple[str, ...]] = {}
    records = bundle["records"]
    assert isinstance(records, Mapping)
    for index, stage in enumerate(STAGES):
        record = records[stage]
        assert isinstance(record, Mapping)
        messages = record["messages"]
        assert isinstance(messages, list)
        content = str(messages[-1]["content"])
        block_id = _block_id(task_bundle_sha, f"{stage}:target", content)
        future_roles = [STAGE_EXPERTS[item] for item in STAGES[index + 1 :]]
        # The current consumer schema requires visible_to to be non-empty.
        # The terminal target remains hard-forbidden, so this marker can never
        # place its bytes in the security prompt.
        visible_to = future_roles or [STAGE_EXPERTS[stage]]
        blocks.append(
            {
                "id": block_id,
                "kind": STAGE_BLOCK_KINDS[stage],
                "content": content,
                "commit_state": "committed",
                "visible_to": visible_to,
            }
        )
        stage_ids = [block_id]
        if stage == "domain_builder":
            output = _required_mapping(record.get("output"), "builder_output_invalid")
            evidence_specs = (
                (
                    "builder:decision_trace",
                    "tool_result",
                    record.get("decision_trace"),
                ),
                ("builder:tool_calls", "tool_call", output.get("tool_calls")),
                (
                    "builder:tool_transcript",
                    "tool_call",
                    output.get("tool_transcript"),
                ),
                ("builder:tool_results", "tool_result", output.get("tool_results")),
                (
                    "builder:validation_evidence",
                    "tool_result",
                    output.get("validation_evidence"),
                ),
                (
                    "builder:validation_state",
                    "test_result",
                    output.get("validation_state"),
                ),
            )
            for label, kind, value in evidence_specs:
                evidence = _optional_evidence(
                    task_bundle_sha=task_bundle_sha,
                    label=label,
                    kind=kind,
                    value=value,
                    visible_to=future_roles,
                )
                if evidence is not None:
                    blocks.append(evidence)
                    stage_ids.append(str(evidence["id"]))
        produced_ids[stage] = tuple(stage_ids)
    board = {"task_id": bundle["task_id"], "generation": 1, "blocks": blocks}
    return board, _sha256_value(board), produced_ids, _language(task_bank)


def _sidecar_records(
    *,
    partition: str,
    bundle: Mapping[str, Any],
    config: TaskBoardProjectorConfig,
    snapshot_sha256: str,
    manifest_sha256: str,
    source_file_sha256: Mapping[str, str],
) -> Iterable[dict[str, Any]]:
    base_board, base_board_sha, produced_ids, language = _base_board(bundle)
    records = bundle["records"]
    assert isinstance(records, Mapping)
    for stage_index, stage in enumerate(STAGES):
        expert = STAGE_EXPERTS[stage]
        source = records[stage]
        assert isinstance(source, Mapping)
        source_sha = _sha256_value(source)
        pair_id = "tb-pair-v1:" + _sha256_value(
            {
                "task_bundle_sha256": bundle["task_bundle_sha256"],
                "stage": stage,
                "source_gold_sha256": source_sha,
            }
        )
        visible_base_ids = [
            str(base_board["blocks"][0]["id"]),
            str(base_board["blocks"][1]["id"]),
        ] + [
            block_id
            for prior in STAGES[:stage_index]
            for block_id in produced_ids[prior]
        ]
        forbidden_ids = [
            block_id
            for future in STAGES[stage_index:]
            for block_id in produced_ids[future]
        ]
        variants = ("clean", "noisy") if partition == "train" else ("clean",)
        for variant in variants:
            board = json.loads(canonical_json(base_board))
            distractors: list[str] = []
            source_block_ids: list[str] = []
            overlay_block_ids: list[str] = []
            if variant == "noisy":
                source_block_id = visible_base_ids[0]
                source_block = next(
                    item for item in board["blocks"] if item["id"] == source_block_id
                )
                overlay_id = "tb-stale-v1:" + _sha256_value(
                    {
                        "pair_id": pair_id,
                        "source_block_id": source_block_id,
                        "content": source_block["content"],
                    }
                )
                board["blocks"].append(
                    {
                        "id": overlay_id,
                        "kind": "history",
                        "content": source_block["content"],
                        "commit_state": "candidate",
                        "visible_to": [expert],
                    }
                )
                distractors = [overlay_id]
                source_block_ids = [source_block_id]
                overlay_block_ids = [overlay_id]
            record_id = "tb-record-v1:" + _sha256_value(
                {"pair_id": pair_id, "variant": variant}
            )
            target_answer = str(source["messages"][-1]["content"])
            inner = {
                "schema_version": config.record_schema,
                "id": record_id,
                "pair_id": pair_id,
                "variant": variant,
                "language": language,
                "split": partition,
                "role": expert,
                "task_board": board,
                "attention_targets": {
                    "relevant_block_ids": visible_base_ids,
                    "distractor_block_ids": distractors,
                    "forbidden_block_ids": forbidden_ids,
                },
                "target": {
                    "selected_block_ids": visible_base_ids,
                    "action": STAGE_ACTIONS[stage],
                    "answer": target_answer,
                },
            }
            wrapper_id = record_id
            yield {
                "schema_version": config.sidecar_schema,
                "id": wrapper_id,
                "pair_id": pair_id,
                "variant": variant,
                "split": partition,
                "stage": stage,
                "expert": expert,
                "source_gold_record_id": source["id"],
                "source_gold_sha256": source_sha,
                "source_gold_file_sha256": source_file_sha256[expert],
                "source_snapshot_sha256": snapshot_sha256,
                "source_snapshot_manifest_sha256": manifest_sha256,
                "task_bundle_sha256": bundle["task_bundle_sha256"],
                "base_task_board_sha256": base_board_sha,
                "projector_version": config.projector_version,
                "config_sha256": config.sha256,
                "sidecar_schema_sha256": config.sidecar_schema_sha256,
                "augmentation": {
                    "kind": "clean" if variant == "clean" else "stale_duplicate_overlay",
                    "same_task_only": True,
                    "split_before_augmentation": True,
                    "source_block_ids": source_block_ids,
                    "overlay_block_ids": overlay_block_ids,
                },
                "training_record": inner,
            }


def _validate_sidecar(value: Any, *, expected_split: str) -> None:
    wrapper = _required_mapping(value, "projected_sidecar_invalid")
    expected_fields = {
        "schema_version", "id", "pair_id", "variant", "split", "stage", "expert",
        "source_gold_record_id", "source_gold_sha256", "source_gold_file_sha256",
        "source_snapshot_sha256", "source_snapshot_manifest_sha256",
        "task_bundle_sha256", "base_task_board_sha256", "projector_version",
        "config_sha256", "sidecar_schema_sha256", "augmentation", "training_record",
    }
    _exact_keys(wrapper, expected_fields, "projected_sidecar_invalid")
    inner = _required_mapping(wrapper.get("training_record"), "projected_record_invalid")
    _exact_keys(
        inner,
        {
            "schema_version", "id", "pair_id", "variant", "language", "split", "role",
            "task_board", "attention_targets", "target",
        },
        "projected_record_invalid",
    )
    stage = wrapper.get("stage")
    expert = wrapper.get("expert")
    variant = wrapper.get("variant")
    augmentation = _required_mapping(
        wrapper.get("augmentation"), "projected_augmentation_invalid"
    )
    if (
        wrapper.get("schema_version") != SIDECAR_SCHEMA
        or not _is_identifier(wrapper.get("id"))
        or not _is_identifier(wrapper.get("pair_id"))
        or not _is_identifier(wrapper.get("source_gold_record_id"))
        or stage not in STAGE_EXPERTS
        or STAGE_EXPERTS[str(stage)] != expert
        or wrapper.get("split") != expected_split
        or variant not in {"clean", "noisy"}
        or inner.get("schema_version") != QUERY_SCHEMA
        or inner.get("id") != wrapper.get("id")
        or inner.get("pair_id") != wrapper.get("pair_id")
        or inner.get("variant") != variant
        or inner.get("split") != expected_split
        or inner.get("role") != expert
        or inner.get("language") not in {"en", "zh-CN"}
        or any(
            not _is_sha256(wrapper.get(field))
            for field in (
                "source_gold_sha256", "source_gold_file_sha256",
                "source_snapshot_sha256", "source_snapshot_manifest_sha256",
                "task_bundle_sha256", "base_task_board_sha256", "config_sha256",
                "sidecar_schema_sha256",
            )
        )
        or set(augmentation)
        != {
            "kind", "same_task_only", "split_before_augmentation",
            "source_block_ids", "overlay_block_ids",
        }
        or augmentation.get("same_task_only") is not True
        or augmentation.get("split_before_augmentation") is not True
    ):
        _fail("projected_sidecar_invalid")
    sources = augmentation.get("source_block_ids")
    overlays = augmentation.get("overlay_block_ids")
    if variant == "clean":
        if augmentation.get("kind") != "clean" or sources != [] or overlays != []:
            _fail("projected_augmentation_invalid")
    elif (
        expected_split != "train"
        or augmentation.get("kind") != "stale_duplicate_overlay"
        or not isinstance(sources, list)
        or not sources
        or not isinstance(overlays, list)
        or not overlays
    ):
        _fail("projected_augmentation_invalid")

    board = _required_mapping(inner.get("task_board"), "projected_record_invalid")
    targets = _required_mapping(
        inner.get("attention_targets"), "projected_record_invalid"
    )
    target = _required_mapping(inner.get("target"), "projected_record_invalid")
    _exact_keys(board, {"task_id", "generation", "blocks"}, "projected_record_invalid")
    _exact_keys(
        targets,
        {"relevant_block_ids", "distractor_block_ids", "forbidden_block_ids"},
        "projected_record_invalid",
    )
    _exact_keys(
        target,
        {"selected_block_ids", "action", "answer"},
        "projected_record_invalid",
    )
    blocks = board.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        _fail("projected_record_invalid")
    ids = [item.get("id") for item in blocks if isinstance(item, Mapping)]
    relevant = targets.get("relevant_block_ids")
    distractors = targets.get("distractor_block_ids")
    forbidden = targets.get("forbidden_block_ids")
    overlay_ids = augmentation.get("overlay_block_ids")
    base_blocks = [
        item
        for item in blocks
        if isinstance(item, Mapping) and item.get("id") not in set(overlay_ids or [])
    ]
    base_board = {
        "task_id": board.get("task_id"),
        "generation": board.get("generation"),
        "blocks": base_blocks,
    }
    by_id = {
        str(item["id"]): item
        for item in blocks
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    role = inner.get("role")
    block_shapes_valid = True
    for item in blocks:
        if not isinstance(item, Mapping) or set(item) != {
            "id", "kind", "content", "commit_state", "visible_to"
        }:
            block_shapes_valid = False
            break
        visible_to = item.get("visible_to")
        if (
            not _is_identifier(item.get("id"))
            or item.get("kind") not in _BLOCK_KINDS
            or not isinstance(item.get("content"), str)
            or not item.get("content")
            or item.get("commit_state") not in _COMMIT_STATES
            or not isinstance(visible_to, list)
            or not visible_to
            or len(set(visible_to)) != len(visible_to)
            or not set(visible_to).issubset(set(EXPERTS) | {"all"})
        ):
            block_shapes_valid = False
            break
    visible_prompt_ids = {
        block_id
        for block_id, item in by_id.items()
        if block_id not in set(forbidden or [])
        and item.get("commit_state") != "rejected"
        and isinstance(item.get("visible_to"), list)
        and ("all" in item["visible_to"] or role in item["visible_to"])
    }
    if (
        len(ids) != len(blocks)
        or not block_shapes_valid
        or len(set(ids)) != len(ids)
        or not _is_identifier(board.get("task_id"))
        or isinstance(board.get("generation"), bool)
        or not isinstance(board.get("generation"), int)
        or board.get("generation") < 1
        or not isinstance(relevant, list)
        or not relevant
        or len(set(relevant)) != len(relevant)
        or not isinstance(distractors, list)
        or len(set(distractors)) != len(distractors)
        or not isinstance(forbidden, list)
        or len(set(forbidden)) != len(forbidden)
        or set(relevant) & set(distractors)
        or set(relevant) & set(forbidden)
        or set(distractors) & set(forbidden)
        or not (set(relevant) | set(distractors) | set(forbidden)).issubset(ids)
        or target.get("selected_block_ids") != relevant
        or not set(relevant).issubset(visible_prompt_ids)
        or set(forbidden) & visible_prompt_ids
        or _sha256_value(base_board) != wrapper.get("base_task_board_sha256")
        or not isinstance(target.get("action"), str)
        or not target.get("action")
        or not isinstance(target.get("answer"), str)
        or not target.get("answer")
        or not _content_safe(wrapper)
    ):
        _fail("projected_record_invalid")
    if variant == "noisy":
        if (
            set(overlays or []) != set(distractors)
            or not set(sources or []).issubset(relevant)
            or len(sources or []) != len(overlays or [])
        ):
            _fail("projected_augmentation_invalid")
        for source_id, overlay_id in zip(sources, overlays, strict=True):
            source_block = by_id.get(str(source_id))
            overlay_block = by_id.get(str(overlay_id))
            if (
                source_block is None
                or overlay_block is None
                or overlay_block.get("kind") != "history"
                or overlay_block.get("commit_state") != "candidate"
                or overlay_block.get("visible_to") != [expert]
                or overlay_block.get("content") != source_block.get("content")
            ):
                _fail("projected_augmentation_invalid")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _verify_output_file(
    path: Path, *, split: str, variant: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = _read_bytes_snapshot(path, "projected_output_invalid")
    rows, records = _jsonl_from_snapshot(snapshot, "projected_output_invalid")
    if not rows:
        _fail("projected_output_empty")
    ids: set[str] = set()
    for row in rows:
        _validate_sidecar(row, expected_split=split)
        identifier = row["id"]
        if not isinstance(identifier, str) or identifier in ids:
            _fail("projected_output_duplicate_id")
        ids.add(identifier)
        if row["variant"] != variant:
            _fail("projected_file_variant_invalid")
    relative = f"{split}/{variant}.jsonl"
    return rows, _binding_from_snapshot(snapshot, relative, records)


def _verify_clean_noisy_pairs(
    clean_rows: list[dict[str, Any]], noisy_rows: list[dict[str, Any]]
) -> None:
    clean = {str(row["pair_id"]): row for row in clean_rows}
    noisy = {str(row["pair_id"]): row for row in noisy_rows}
    if len(clean) != len(clean_rows) or len(noisy) != len(noisy_rows) or set(clean) != set(noisy):
        _fail("projected_pair_contract_invalid")
    for pair_id in sorted(clean):
        baseline = clean[pair_id]
        variant = noisy[pair_id]
        baseline_inner = baseline["training_record"]
        variant_inner = variant["training_record"]
        if (
            any(
                baseline[field] != variant[field]
                for field in (
                    "pair_id",
                    "stage",
                    "expert",
                    "source_gold_record_id",
                    "source_gold_sha256",
                    "source_gold_file_sha256",
                    "source_snapshot_sha256",
                    "source_snapshot_manifest_sha256",
                    "task_bundle_sha256",
                    "base_task_board_sha256",
                    "projector_version",
                    "config_sha256",
                    "sidecar_schema_sha256",
                )
            )
            or baseline_inner["target"] != variant_inner["target"]
            or baseline_inner["attention_targets"]["relevant_block_ids"]
            != variant_inner["attention_targets"]["relevant_block_ids"]
            or baseline_inner["attention_targets"]["forbidden_block_ids"]
            != variant_inner["attention_targets"]["forbidden_block_ids"]
            or baseline_inner["task_board"]["blocks"]
            != variant_inner["task_board"]["blocks"][
                : len(baseline_inner["task_board"]["blocks"])
            ]
        ):
            _fail("projected_pair_contract_invalid")


def _verify_dataset_split_groups(
    all_rows: Mapping[tuple[str, str], list[dict[str, Any]]],
    *,
    expected_bundle_task_ids: Mapping[str, Mapping[str, str]],
) -> dict[str, set[str]]:
    expected_files = {
        ("train", "clean"),
        ("train", "noisy"),
        ("calibration", "clean"),
    }
    if set(all_rows) != expected_files or set(expected_bundle_task_ids) != {
        "train",
        "calibration",
    }:
        _fail("projected_split_group_invalid")

    bundle_splits: dict[str, set[str]] = {}
    bundle_task_ids: dict[str, set[str]] = {}
    task_id_splits: dict[str, set[str]] = {}
    task_id_bundles: dict[str, set[str]] = {}
    observed_task_ids = {"train": set(), "calibration": set()}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for (file_split, file_variant), rows in all_rows.items():
        for row in rows:
            inner = _required_mapping(
                row.get("training_record"), "projected_split_group_invalid"
            )
            board = _required_mapping(
                inner.get("task_board"), "projected_split_group_invalid"
            )
            bundle = row.get("task_bundle_sha256")
            task_id = board.get("task_id")
            split = row.get("split")
            variant = row.get("variant")
            if (
                not _is_sha256(bundle)
                or not _is_identifier(task_id)
                or split != file_split
                or variant != file_variant
            ):
                _fail("projected_split_group_invalid")
            bundle = str(bundle)
            task_id = str(task_id)
            split = str(split)
            variant = str(variant)
            bundle_splits.setdefault(bundle, set()).add(split)
            bundle_task_ids.setdefault(bundle, set()).add(task_id)
            task_id_splits.setdefault(task_id, set()).add(split)
            task_id_bundles.setdefault(task_id, set()).add(bundle)
            observed_task_ids[split].add(task_id)
            groups.setdefault((bundle, split, variant), []).append(row)

    if any(len(splits) != 1 for splits in bundle_splits.values()):
        _fail("projected_bundle_cross_split")
    if (
        any(len(task_ids) != 1 for task_ids in bundle_task_ids.values())
        or any(len(splits) != 1 for splits in task_id_splits.values())
        or any(len(bundles) != 1 for bundles in task_id_bundles.values())
    ):
        _fail("projected_task_id_bundle_mismatch")
    observed_bundle_task_ids = {
        split: {
            bundle: next(iter(bundle_task_ids[bundle]))
            for bundle, splits in bundle_splits.items()
            if splits == {split}
        }
        for split in ("train", "calibration")
    }
    if any(
        observed_bundle_task_ids[split] != dict(expected_bundle_task_ids[split])
        for split in ("train", "calibration")
    ):
        _fail("projected_task_id_source_mismatch")

    clean_groups = {
        (bundle, split)
        for bundle, split, variant in groups
        if variant == "clean"
    }
    expected_groups = {
        (bundle, split, variant)
        for bundle, split in clean_groups
        for variant in (("clean", "noisy") if split == "train" else ("clean",))
    }
    if set(groups) != expected_groups:
        _fail("projected_bundle_role_views_invalid")

    for (bundle, _split, variant), rows in groups.items():
        if (
            len(rows) != len(STAGES)
            or {str(row["stage"]) for row in rows} != set(STAGES)
            or {str(row["expert"]) for row in rows} != set(EXPERTS)
        ):
            _fail("projected_bundle_role_views_invalid")
        if variant != "clean":
            continue
        task_id = next(iter(bundle_task_ids[bundle]))
        by_stage = {str(row["stage"]): row for row in rows}
        entries = [
            {
                "stage": stage,
                "expert": STAGE_EXPERTS[stage],
                "record_id": by_stage[stage]["source_gold_record_id"],
                "record_sha256": by_stage[stage]["source_gold_sha256"],
            }
            for stage in STAGES
        ]
        if _task_bundle_sha256(task_id, entries) != bundle:
            _fail("projected_bundle_hash_mismatch")
    return observed_task_ids


def _verify_inventory_unchanged(
    inventory: Mapping[str, Mapping[str, Any]],
) -> None:
    for item in inventory.values():
        path = item.get("path")
        if not isinstance(path, Path):
            _fail("snapshot_binding_changed_during_read")
        snapshot = _read_bytes_snapshot(
            path, "snapshot_binding_changed_during_read"
        )
        if snapshot.size != item.get("bytes") or snapshot.sha256 != item.get(
            "sha256"
        ):
            _fail("snapshot_binding_changed_during_read")


def project_taskboards(
    config: TaskBoardProjectorConfig | str | Path,
    snapshot_dir: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate a frozen snapshot and atomically publish research sidecars."""

    cfg = config if isinstance(config, TaskBoardProjectorConfig) else TaskBoardProjectorConfig.load(config)
    raw_snapshot = Path(snapshot_dir).expanduser()
    raw_output = Path(output_dir).expanduser()
    if raw_snapshot.is_symlink():
        _fail("snapshot_input_invalid")
    if raw_output.is_symlink():
        _fail("projector_output_exists_or_overlaps_input")
    snapshot = raw_snapshot.resolve()
    output = raw_output.resolve()
    if output.exists() or output == snapshot:
        _fail("projector_output_exists_or_overlaps_input")
    try:
        output.relative_to(snapshot)
    except ValueError:
        pass
    else:
        _fail("projector_output_exists_or_overlaps_input")
    try:
        snapshot.relative_to(output)
    except ValueError:
        pass
    else:
        _fail("projector_output_exists_or_overlaps_input")

    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists():
        _fail("projector_temporary_conflict")
    try:
        manifest, manifest_sha, partitions, source_inventory = _load_snapshot(
            snapshot, expected_manifest_sha256
        )
        snapshot_sha = manifest.get("snapshot_sha256")
        if not _is_sha256(snapshot_sha):
            _fail("snapshot_sha256_invalid")
        bundled = {item.name: _bundle_partition(item) for item in partitions}
        train_tasks = {str(item["task_id"]) for item in bundled["train"]}
        calibration_tasks = {
            str(item["task_id"]) for item in bundled["calibration"]
        }
        train_instances = {str(item["instance_id"]) for item in bundled["train"]}
        calibration_instances = {
            str(item["instance_id"]) for item in bundled["calibration"]
        }
        if train_tasks & calibration_tasks or train_instances & calibration_instances:
            _fail("source_task_cross_split")

        temporary.mkdir(parents=True)
        all_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        output_bindings: dict[tuple[str, str], dict[str, Any]] = {}
        partition_by_name = {item.name: item for item in partitions}
        for split in ("train", "calibration"):
            source_hashes = {
                expert: str(partition_by_name[split].file_bindings[expert]["sha256"])
                for expert in EXPERTS
            }
            projected = [
                row
                for bundle in bundled[split]
                for row in _sidecar_records(
                    partition=split,
                    bundle=bundle,
                    config=cfg,
                    snapshot_sha256=str(snapshot_sha),
                    manifest_sha256=manifest_sha,
                    source_file_sha256=source_hashes,
                )
            ]
            variants = ("clean", "noisy") if split == "train" else ("clean",)
            for variant in variants:
                rows = [row for row in projected if row["variant"] == variant]
                path = temporary / split / f"{variant}.jsonl"
                _write_jsonl(path, rows)
                verified, output_binding = _verify_output_file(
                    path, split=split, variant=variant
                )
                all_rows[(split, variant)] = verified
                output_bindings[(split, variant)] = output_binding
        _verify_clean_noisy_pairs(
            all_rows[("train", "clean")], all_rows[("train", "noisy")]
        )

        file_entries: list[dict[str, Any]] = []
        for split, variant in (
            ("train", "clean"),
            ("train", "noisy"),
            ("calibration", "clean"),
        ):
            relative = f"{split}/{variant}.jsonl"
            entry = dict(output_bindings[(split, variant)])
            entry["split"] = split
            entry["variant"] = variant
            file_entries.append(entry)
        rows_flat = (
            all_rows[("train", "clean")]
            + all_rows[("train", "noisy")]
            + all_rows[("calibration", "clean")]
        )
        projected_task_ids = _verify_dataset_split_groups(
            all_rows,
            expected_bundle_task_ids={
                split: {
                    str(bundle["task_bundle_sha256"]): str(bundle["task_id"])
                    for bundle in bundled[split]
                }
                for split in ("train", "calibration")
            },
        )
        stage_counts = Counter(str(row["stage"]) for row in rows_flat)
        expert_counts = Counter(str(row["expert"]) for row in rows_flat)
        variant_counts = Counter(str(row["variant"]) for row in rows_flat)
        split_counts = Counter(str(row["split"]) for row in rows_flat)
        language_counts = Counter(
            str(row["training_record"]["language"]) for row in rows_flat
        )
        projected_manifest = {
            "schema_version": cfg.manifest_schema,
            "input": {
                "snapshot_schema_version": SNAPSHOT_SCHEMA,
                "snapshot_sha256": snapshot_sha,
                "snapshot_manifest_path": "manifest.json",
                "snapshot_manifest_sha256": manifest_sha,
                "snapshot_sha256_sidecar_path": "manifest.json.sha256",
                "snapshot_sha256_sidecar_sha256": source_inventory[
                    "manifest.json.sha256"
                ]["sha256"],
                "splits": ["train", "calibration"],
            },
            "producer": {
                "name": "anchor.swebench-taskboard-projector",
                "projector_version": cfg.projector_version,
                "config_sha256": cfg.sha256,
                "sidecar_schema_sha256": cfg.sidecar_schema_sha256,
                "manifest_schema_sha256": cfg.manifest_schema_sha256,
                "record_schema_version": cfg.record_schema,
            },
            "files": file_entries,
            "counts": {
                "total": len(rows_flat),
                "unique_task_bundles": len(
                    {str(row["task_bundle_sha256"]) for row in rows_flat}
                ),
                "task_ids_sha256": _sha256_bytes(
                    "\n".join(
                        sorted(
                            projected_task_ids["train"]
                            | projected_task_ids["calibration"]
                        )
                    ).encode("utf-8")
                ),
                "by_split": dict(sorted(split_counts.items())),
                "by_variant": dict(sorted(variant_counts.items())),
                "by_stage": dict(sorted(stage_counts.items())),
                "by_expert": dict(sorted(expert_counts.items())),
                "by_language": {
                    language: language_counts.get(language, 0)
                    for language in ("en", "zh-CN")
                },
            },
            "split_group_key": "task_bundle_sha256",
            "task_id_cross_binding_key": "training_record.task_board.task_id",
            "all_five_role_views_same_split": True,
            "canonical_gold_written": False,
            "provider_requests": 0,
            "heldout_content_read": False,
            "heldout_content_emitted": False,
            "split_preserved": True,
            "augmentation_applied_after_split": True,
            "claim_scope": "research_proxy_only",
        }
        manifest_path = temporary / "manifest.json"
        manifest_bytes = (
            json.dumps(
                projected_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        output_manifest_sha = _sha256_bytes(manifest_bytes)
        output_sidecar_bytes = f"{output_manifest_sha}  manifest.json\n".encode(
            "ascii"
        )
        (temporary / "manifest.json.sha256").write_bytes(output_sidecar_bytes)
        verified_manifest_snapshot = _read_bytes_snapshot(
            manifest_path, "projected_manifest_invalid"
        )
        verified_sidecar_snapshot = _read_bytes_snapshot(
            temporary / "manifest.json.sha256", "projected_manifest_invalid"
        )
        verified_manifest = _json_from_snapshot(
            verified_manifest_snapshot, "projected_manifest_invalid"
        )
        try:
            verified_sidecar = verified_sidecar_snapshot.data.decode("ascii").split()
        except UnicodeDecodeError as exc:
            raise TaskBoardProjectorError("projected_manifest_invalid") from exc
        if (
            verified_manifest != projected_manifest
            or verified_sidecar != [output_manifest_sha, "manifest.json"]
            or verified_manifest_snapshot.data != manifest_bytes
            or verified_manifest_snapshot.sha256 != output_manifest_sha
            or verified_sidecar_snapshot.data != output_sidecar_bytes
        ):
            _fail("projected_manifest_invalid")
        for entry in file_entries:
            relative = str(entry["path"])
            output_snapshot = _read_bytes_snapshot(
                temporary / relative, "projected_manifest_file_binding_invalid"
            )
            _rows, output_records = _jsonl_from_snapshot(
                output_snapshot, "projected_manifest_file_binding_invalid"
            )
            observed = _binding_from_snapshot(
                output_snapshot, relative, output_records
            )
            if observed != {
                key: entry[key] for key in ("path", "records", "bytes", "sha256")
            }:
                _fail("projected_manifest_file_binding_invalid")
        current_sidecar_schema_path = (
            cfg.path.parent / "taskboard_projector_sidecar.schema.json"
        )
        current_manifest_schema_path = (
            cfg.path.parent / "taskboard_projector_manifest.schema.json"
        )
        current_config_snapshot = _read_bytes_snapshot(
            cfg.path, "projector_config_changed_during_read"
        )
        current_sidecar_schema_snapshot = _read_bytes_snapshot(
            current_sidecar_schema_path, "projector_config_changed_during_read"
        )
        current_manifest_schema_snapshot = _read_bytes_snapshot(
            current_manifest_schema_path, "projector_config_changed_during_read"
        )
        if (
            current_config_snapshot.sha256 != cfg.sha256
            or current_sidecar_schema_snapshot.sha256 != cfg.sidecar_schema_sha256
            or current_manifest_schema_snapshot.sha256 != cfg.manifest_schema_sha256
        ):
            _fail("projector_config_changed_during_read")
        _verify_inventory_unchanged(source_inventory)
        projected_inventory: dict[str, dict[str, Any]] = {
            f"{split}:{variant}": {
                "path": temporary / str(binding["path"]),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
            for (split, variant), binding in output_bindings.items()
        }
        projected_inventory.update(
            {
                "manifest.json": {
                    "path": manifest_path,
                    "bytes": len(manifest_bytes),
                    "sha256": output_manifest_sha,
                },
                "manifest.json.sha256": {
                    "path": temporary / "manifest.json.sha256",
                    "bytes": len(output_sidecar_bytes),
                    "sha256": _sha256_bytes(output_sidecar_bytes),
                },
            }
        )
        _verify_inventory_unchanged(projected_inventory)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return {
            "schema_version": cfg.manifest_schema,
            "status": "published",
            "output_dir": str(output),
            "manifest_sha256": output_manifest_sha,
            "source_snapshot_manifest_sha256": manifest_sha,
            "records": len(rows_flat),
            "provider_requests": 0,
            "canonical_gold_written": False,
            "heldout_content_read": False,
            "heldout_content_emitted": False,
        }
    except TaskBoardProjectorError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise TaskBoardProjectorError("projector_internal_error") from exc


__all__ = [
    "TaskBoardProjectorConfig",
    "TaskBoardProjectorError",
    "project_taskboards",
]
