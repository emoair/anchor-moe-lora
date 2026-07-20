"""Fail-closed consumer preflight for a frozen generic training release.

The preflight authenticates metadata and the three projected partitions only.
It does not launch a model, read held-out bodies, or authorize claims beyond the
``research_proxy_only`` scope encoded by the producer release lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


RELEASE_LOCK_SCHEMA_VERSION = "anchor.generic-train-release-lock.v1"
RELEASE_LOCK_SCHEMA_SHA256 = (
    "889787be1391aec2d59f91b1ba171588c82e455aaddc342b79d3680e0284210d"
)
FIXED_PARTITIONS = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)
REQUIRED_ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
QUERY_SPECIALIZATION_CONSUMER_ID = "anchor.query-specialization-training-consumer"
QUERY_SPECIALIZATION_CONSUMER_VERSION = "1"
QUERY_SPECIALIZATION_IMPLEMENTATION_FILES = (
    "src/anchor_mvp/research/query_specialization.py",
    "src/anchor_mvp/research/training_release_consumer.py",
)
QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT = (
    "scripts/research/train_query_specialization_mvp.py"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "bindings",
    "fixed_files",
    "consumer",
    "split_group_key",
    "task_id_cross_binding_key",
    "required_roles",
    "provenance_location",
    "calibration_is_heldout",
    "heldout_content_read",
    "heldout_content_emitted",
    "canonical_gold_written",
    "provider_requests",
    "claim_scope",
}
_BINDING_FIELDS = {
    "projector_manifest_sha256",
    "projector_manifest_schema_sha256",
    "projector_sidecar_schema_sha256",
    "source_disjoint_manifest_sha256",
    "generic_execution_contract_sha256",
    "consumer_contract_sha256",
    "execution_lock_sha256",
    "attestation_sha256",
    "coordinator_config_sha256",
    "source_bank_manifest_sha256",
}
_FILE_FIELDS = {"path", "sha256", "bytes", "records", "split", "variant"}
_CONSUMER_FIELDS = {
    "consumer_id",
    "consumer_version",
    "implementation_files",
    "launch_entrypoint",
}
_PATH_HASH_FIELDS = {"path", "sha256"}


class TrainingReleaseConsumerError(RuntimeError):
    """Content-free release preflight failure."""


def _fail(code: str) -> None:
    raise TrainingReleaseConsumerError(code)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class _BytesSnapshot:
    data: bytes
    sha256: str
    size: int


def _read_bytes_snapshot(path: Path, code: str) -> _BytesSnapshot:
    """Read once and bind digest, size, and parsing to the opened file bytes."""

    try:
        if not path.is_file() or path.is_symlink():
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise TrainingReleaseConsumerError(code) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def _json_mapping(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingReleaseConsumerError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return False
    return ".." not in Path(value.replace("\\", "/")).parts


def _partition_path(root: Path, relative: object) -> Path:
    if not _safe_relative_path(relative):
        _fail("release_lock_fixed_file_invalid")
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail("release_lock_fixed_file_invalid")
    return candidate


def _validate_path_hash(value: object) -> None:
    item = _mapping(value, "release_lock_consumer_invalid")
    _exact_fields(item, _PATH_HASH_FIELDS, "release_lock_consumer_invalid")
    if not _safe_relative_path(item.get("path")) or not _is_sha256(
        item.get("sha256")
    ):
        _fail("release_lock_consumer_invalid")


def _validate_consumer(
    value: object,
    *,
    repository_root: Path,
    expected_consumer_id: str,
    expected_consumer_version: str,
    required_implementation_files: tuple[str, ...],
    required_launch_entrypoint: str,
) -> tuple[tuple[str, str], ...]:
    consumer = _mapping(value, "release_lock_consumer_invalid")
    _exact_fields(consumer, _CONSUMER_FIELDS, "release_lock_consumer_invalid")
    consumer_id = consumer.get("consumer_id")
    implementation_files = consumer.get("implementation_files")
    if (
        not isinstance(consumer_id, str)
        or not _IDENTIFIER_RE.fullmatch(consumer_id)
        or not isinstance(consumer.get("consumer_version"), str)
        or not consumer["consumer_version"]
        or consumer_id != expected_consumer_id
        or consumer.get("consumer_version") != expected_consumer_version
        or not isinstance(implementation_files, list)
        or not implementation_files
    ):
        _fail("release_lock_consumer_invalid")
    implementation_paths: list[str] = []
    for item in implementation_files:
        _validate_path_hash(item)
        assert isinstance(item, Mapping)
        implementation_paths.append(str(item["path"]))
    _validate_path_hash(consumer.get("launch_entrypoint"))
    launch_entrypoint = _mapping(
        consumer.get("launch_entrypoint"), "release_lock_consumer_invalid"
    )
    if (
        len(set(implementation_paths)) != len(implementation_paths)
        or not set(required_implementation_files).issubset(implementation_paths)
        or launch_entrypoint.get("path") != required_launch_entrypoint
        or not repository_root.is_dir()
        or repository_root.is_symlink()
    ):
        _fail("release_lock_consumer_invalid")

    authenticated: list[tuple[str, str]] = []
    snapshots: dict[Path, _BytesSnapshot] = {}
    for item in [*implementation_files, launch_entrypoint]:
        assert isinstance(item, Mapping)
        relative = str(item["path"])
        path = _partition_path(repository_root, relative)
        snapshot = snapshots.get(path)
        if snapshot is None:
            snapshot = _read_bytes_snapshot(path, "release_lock_consumer_file_invalid")
            snapshots[path] = snapshot
        if snapshot.sha256 != item["sha256"]:
            _fail("release_lock_consumer_file_invalid")
        authenticated.append((relative, snapshot.sha256))
    return tuple(authenticated)


def _parse_partition(
    snapshot: _BytesSnapshot,
    *,
    expected_split: str,
    expected_variant: str,
) -> tuple[int, tuple[tuple[str, str, str, str], ...]]:
    """Return only content-free bundle/role/task metadata from authenticated bytes."""

    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingReleaseConsumerError("release_lock_partition_invalid") from exc
    metadata: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingReleaseConsumerError(
                "release_lock_partition_invalid"
            ) from exc
        if not isinstance(row, Mapping):
            _fail("release_lock_partition_invalid")
        training_record = _mapping(
            row.get("training_record"), "release_lock_partition_invalid"
        )
        task_board = _mapping(
            training_record.get("task_board"), "release_lock_partition_invalid"
        )
        bundle = row.get("task_bundle_sha256")
        role = training_record.get("role")
        task_id = task_board.get("task_id")
        if (
            row.get("schema_version")
            != "anchor.swebench-taskboard-sidecar.v1"
            or row.get("split") != expected_split
            or row.get("variant") != expected_variant
            or training_record.get("schema_version")
            != "anchor.query-specialization.v1"
            or training_record.get("split") != expected_split
            or training_record.get("variant") != expected_variant
            or not _is_sha256(bundle)
            or role not in REQUIRED_ROLES
            or not isinstance(task_id, str)
            or not _IDENTIFIER_RE.fullmatch(task_id)
            or "provenance" in training_record
            or not isinstance(row.get("source_gold_record_id"), str)
            or not _IDENTIFIER_RE.fullmatch(str(row["source_gold_record_id"]))
            or not _is_sha256(row.get("source_gold_sha256"))
            or not _is_sha256(row.get("source_gold_file_sha256"))
            or not _is_sha256(row.get("source_snapshot_sha256"))
            or not _is_sha256(row.get("source_snapshot_manifest_sha256"))
        ):
            _fail("release_lock_partition_invalid")
        metadata.append((str(bundle), str(role), task_id, expected_variant))
    if not metadata:
        _fail("release_lock_partition_invalid")
    return len(metadata), tuple(metadata)


@dataclass(frozen=True)
class TrainingReleaseValidation:
    """Content-free evidence returned after all release gates pass."""

    manifest_sha256: str
    schema_sha256: str
    partition_sha256: tuple[tuple[str, str], ...]
    partition_records: tuple[tuple[str, int], ...]
    task_bundle_count: int
    consumer_id: str
    consumer_version: str
    consumer_contract_sha256: str
    consumer_files_sha256: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_LOCK_SCHEMA_VERSION,
            "status": "ready",
            "formal_training_authorized": True,
            "manifest_sha256": self.manifest_sha256,
            "schema_sha256": self.schema_sha256,
            "partition_sha256": dict(self.partition_sha256),
            "partition_records": dict(self.partition_records),
            "task_bundle_count": self.task_bundle_count,
            "consumer_id": self.consumer_id,
            "consumer_version": self.consumer_version,
            "consumer_contract_sha256": self.consumer_contract_sha256,
            "consumer_files_sha256": dict(self.consumer_files_sha256),
            "split_group_key": "task_bundle_sha256",
            "required_roles": list(REQUIRED_ROLES),
            "provenance_location": "outer_sidecar",
            "claim_scope": "research_proxy_only",
        }


def validate_release_lock_schema(
    schema_path: str | Path,
    expected_schema_sha256: str = RELEASE_LOCK_SCHEMA_SHA256,
) -> str:
    """Authenticate the pinned release-lock schema without loading an artifact."""

    if not _is_sha256(expected_schema_sha256):
        _fail("release_lock_expected_sha256_invalid")
    schema_snapshot = _read_bytes_snapshot(
        Path(schema_path).expanduser().resolve(), "release_lock_schema_invalid"
    )
    if schema_snapshot.sha256 != expected_schema_sha256:
        _fail("release_lock_schema_invalid")
    schema = _json_mapping(schema_snapshot, "release_lock_schema_invalid")
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("type") != "object"
    ):
        _fail("release_lock_schema_invalid")
    return schema_snapshot.sha256


def load_training_release_lock(
    *,
    release_root: str | Path,
    dataset_root: str | Path,
    schema_path: str | Path,
    expected_manifest_sha256: str,
    expected_schema_sha256: str = RELEASE_LOCK_SCHEMA_SHA256,
    expected_projector_manifest_sha256: str,
    expected_projector_manifest_schema_sha256: str,
    expected_projector_sidecar_schema_sha256: str,
    expected_consumer_contract_sha256: str,
    repository_root: str | Path,
    expected_consumer_id: str,
    expected_consumer_version: str,
    required_implementation_files: tuple[str, ...],
    required_launch_entrypoint: str,
    authenticated_partition_sha256: Mapping[str, str],
) -> TrainingReleaseValidation:
    """Authenticate one ready release lock and its fixed projected partitions."""

    for value in (
        expected_manifest_sha256,
        expected_schema_sha256,
        expected_projector_manifest_sha256,
        expected_projector_manifest_schema_sha256,
        expected_projector_sidecar_schema_sha256,
        expected_consumer_contract_sha256,
    ):
        if not _is_sha256(value):
            _fail("release_lock_expected_sha256_invalid")

    schema_sha256 = validate_release_lock_schema(
        schema_path, expected_schema_sha256
    )

    root = Path(release_root).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or not dataset.is_dir()
        or dataset.is_symlink()
    ):
        _fail("release_lock_artifact_invalid")
    manifest_snapshot = _read_bytes_snapshot(
        root / "manifest.json", "release_lock_manifest_invalid"
    )
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        _fail("release_lock_manifest_invalid")
    sidecar_snapshot = _read_bytes_snapshot(
        root / "manifest.json.sha256", "release_lock_sha256_sidecar_invalid"
    )
    expected_sidecar = f"{manifest_snapshot.sha256}  manifest.json\n".encode(
        "ascii"
    )
    if sidecar_snapshot.data != expected_sidecar:
        _fail("release_lock_sha256_sidecar_invalid")

    manifest = _json_mapping(manifest_snapshot, "release_lock_manifest_invalid")
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, "release_lock_manifest_invalid")
    bindings = _mapping(manifest.get("bindings"), "release_lock_bindings_invalid")
    _exact_fields(bindings, _BINDING_FIELDS, "release_lock_bindings_invalid")
    if not all(_is_sha256(bindings.get(key)) for key in _BINDING_FIELDS):
        _fail("release_lock_bindings_invalid")
    if (
        manifest.get("schema_version") != RELEASE_LOCK_SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or bindings.get("projector_manifest_sha256")
        != expected_projector_manifest_sha256
        or bindings.get("projector_manifest_schema_sha256")
        != expected_projector_manifest_schema_sha256
        or bindings.get("projector_sidecar_schema_sha256")
        != expected_projector_sidecar_schema_sha256
        or manifest.get("split_group_key") != "task_bundle_sha256"
        or manifest.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or manifest.get("required_roles") != list(REQUIRED_ROLES)
        or manifest.get("provenance_location") != "outer_sidecar"
        or manifest.get("calibration_is_heldout") is not False
        or manifest.get("heldout_content_read") is not False
        or manifest.get("heldout_content_emitted") is not False
        or manifest.get("canonical_gold_written") is not False
        or manifest.get("provider_requests") != 0
        or manifest.get("claim_scope") != "research_proxy_only"
    ):
        _fail("release_lock_claims_invalid")
    if bindings.get("consumer_contract_sha256") != expected_consumer_contract_sha256:
        _fail("release_lock_consumer_contract_binding_invalid")
    consumer_files_sha256 = _validate_consumer(
        manifest.get("consumer"),
        repository_root=Path(repository_root).expanduser().resolve(),
        expected_consumer_id=expected_consumer_id,
        expected_consumer_version=expected_consumer_version,
        required_implementation_files=required_implementation_files,
        required_launch_entrypoint=required_launch_entrypoint,
    )

    fixed_files = manifest.get("fixed_files")
    if not isinstance(fixed_files, list) or len(fixed_files) != len(
        FIXED_PARTITIONS
    ):
        _fail("release_lock_fixed_files_invalid")
    fixed_paths = {relative for relative, _split, _variant in FIXED_PARTITIONS}
    if set(authenticated_partition_sha256) != fixed_paths or not all(
        _is_sha256(authenticated_partition_sha256.get(relative))
        for relative in fixed_paths
    ):
        _fail("release_lock_partition_authentication_invalid")
    all_metadata: list[tuple[str, str, str, str, str]] = []
    partition_hashes: list[tuple[str, str]] = []
    partition_records: list[tuple[str, int]] = []
    for raw_item, (relative, split, variant) in zip(
        fixed_files, FIXED_PARTITIONS, strict=True
    ):
        item = _mapping(raw_item, "release_lock_fixed_files_invalid")
        _exact_fields(item, _FILE_FIELDS, "release_lock_fixed_files_invalid")
        if (
            item.get("path") != relative
            or item.get("split") != split
            or item.get("variant") != variant
            or not _is_sha256(item.get("sha256"))
            or not _is_positive_int(item.get("bytes"))
            or not _is_positive_int(item.get("records"))
        ):
            _fail("release_lock_fixed_files_invalid")
        partition_snapshot = _read_bytes_snapshot(
            _partition_path(dataset, relative), "release_lock_partition_invalid"
        )
        count, metadata = _parse_partition(
            partition_snapshot,
            expected_split=split,
            expected_variant=variant,
        )
        if (
            partition_snapshot.sha256 != item["sha256"]
            or partition_snapshot.size != item["bytes"]
            or count != item["records"]
        ):
            _fail("release_lock_partition_mismatch")
        authenticated = authenticated_partition_sha256.get(relative)
        if authenticated != partition_snapshot.sha256:
            _fail("release_lock_partition_authentication_mismatch")
        partition_hashes.append((relative, partition_snapshot.sha256))
        partition_records.append((relative, count))
        all_metadata.extend(
            (bundle, role, task_id, split, row_variant)
            for bundle, role, task_id, row_variant in metadata
        )

    group_roles: dict[tuple[str, str, str], set[str]] = {}
    bundle_split: dict[str, str] = {}
    bundle_task_id: dict[str, str] = {}
    task_id_bundle: dict[str, str] = {}
    for bundle, role, task_id, split, variant in all_metadata:
        if bundle in bundle_split and bundle_split[bundle] != split:
            _fail("release_lock_task_bundle_split_invalid")
        if bundle in bundle_task_id and bundle_task_id[bundle] != task_id:
            _fail("release_lock_task_id_cross_binding_invalid")
        if task_id in task_id_bundle and task_id_bundle[task_id] != bundle:
            _fail("release_lock_task_id_cross_binding_invalid")
        bundle_split[bundle] = split
        bundle_task_id[bundle] = task_id
        task_id_bundle[task_id] = bundle
        roles = group_roles.setdefault((bundle, split, variant), set())
        if role in roles:
            _fail("release_lock_role_views_invalid")
        roles.add(role)
    if any(roles != set(REQUIRED_ROLES) for roles in group_roles.values()):
        _fail("release_lock_role_views_invalid")
    train_bundles = {
        bundle for bundle, split in bundle_split.items() if split == "train"
    }
    for bundle in train_bundles:
        if (bundle, "train", "clean") not in group_roles or (
            bundle,
            "train",
            "noisy",
        ) not in group_roles:
            _fail("release_lock_clean_noisy_pair_invalid")

    return TrainingReleaseValidation(
        manifest_sha256=manifest_snapshot.sha256,
        schema_sha256=schema_sha256,
        partition_sha256=tuple(partition_hashes),
        partition_records=tuple(partition_records),
        task_bundle_count=len(bundle_split),
        consumer_id=expected_consumer_id,
        consumer_version=expected_consumer_version,
        consumer_contract_sha256=expected_consumer_contract_sha256,
        consumer_files_sha256=consumer_files_sha256,
    )


__all__ = [
    "FIXED_PARTITIONS",
    "QUERY_SPECIALIZATION_CONSUMER_ID",
    "QUERY_SPECIALIZATION_CONSUMER_VERSION",
    "QUERY_SPECIALIZATION_IMPLEMENTATION_FILES",
    "QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT",
    "RELEASE_LOCK_SCHEMA_SHA256",
    "RELEASE_LOCK_SCHEMA_VERSION",
    "REQUIRED_ROLES",
    "TrainingReleaseConsumerError",
    "TrainingReleaseValidation",
    "load_training_release_lock",
    "validate_release_lock_schema",
]
