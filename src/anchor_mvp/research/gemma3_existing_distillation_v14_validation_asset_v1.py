"""Freeze the existing v14 route dataset as a body-free validation-only asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "anchor.gemma3-existing-distillation-v14-validation-asset-config.v1"
RECORD_VERSION = "anchor.gemma3-existing-distillation-v14-validation-record.v1"
MANIFEST_VERSION = "anchor.gemma3-existing-distillation-v14-validation-manifest.v1"
RECEIPT_VERSION = "anchor.gemma3-existing-distillation-v14-validation-build-receipt.v1"
ASSET_NAMESPACE = "gemma3_existing_distillation_v14_validation_asset_v1"
SPLITS = ("train", "calibration", "holdout")
EXPECTED_COUNTS = {"train": 13131, "calibration": 1923, "holdout": 1578}
PARTITION_POLICIES = {
    "train": "seen_train_regression_validation",
    "calibration": "current_validation",
    "holdout": "sealed_single_blind_final_only",
}
SOURCE_RECORD_KEYS = {
    "group_stratum",
    "labels",
    "leakage_family_sha256",
    "prompt",
    "prompt_sha256",
    "provenance_kind",
    "sample_id",
    "source_group_sha256",
    "split",
    "training_focus",
    "variant",
}
OUTPUT_RECORD_KEYS = {
    "schema_version",
    "source_ordinal",
    "sample_id",
    "split",
    "source_group_sha256",
    "leakage_family_sha256",
    "prompt_sha256",
    "consumable_for_training",
}
FORBIDDEN_PERSISTED_KEYS = {
    "prompt",
    "labels",
    "target",
    "targets",
    "raw_token_ids",
    "input_ids",
    "teacher_input",
    "messages",
    "body",
    "text",
}
REASON_CODES = (
    "actual_worker_runtime_not_frozen",
    "conversion_original_paths",
    "handoff_consumed_boolean_inverted",
)
REPO_ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/research/gemma3_existing_distillation_v14_validation_asset_v1.json"
)


class ValidationAssetError(RuntimeError):
    """Fail-closed validation asset error."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int]


def _fail(code: str) -> None:
    raise ValidationAssetError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _json(raw: bytes, code: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _is_reparse(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _identity(st: os.stat_result) -> tuple[int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size)


def _reject_reparse_chain(path: Path, code: str) -> None:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            _fail(code)
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            _fail(code)


def _snapshot(path: Path, code: str) -> Snapshot:
    candidate = Path(os.path.abspath(path))
    _reject_reparse_chain(candidate, code)
    try:
        before = os.lstat(candidate)
    except OSError:
        _fail(code)
    if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
        _fail(code)
    try:
        with candidate.open("rb") as handle:
            handle_before = os.fstat(handle.fileno())
            data = handle.read()
            handle_after = os.fstat(handle.fileno())
    except OSError:
        _fail(code)
    try:
        after = os.lstat(candidate)
    except OSError:
        _fail(code)
    identities = {
        _identity(before),
        _identity(handle_before),
        _identity(handle_after),
        _identity(after),
    }
    if len(identities) != 1 or len(data) != before.st_size:
        _fail(code)
    return Snapshot(candidate, data, _sha256(data), len(data), _identity(before))


def _repo_file(value: str, code: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(code)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        _fail(code)
    candidate = REPO_ROOT / relative
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(Path(os.path.abspath(REPO_ROOT)))
    except ValueError:
        _fail(code)
    return lexical


def _validate_schema(instance: object, schema: object, code: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except Exception:
        _fail(code)


def _verify_file_binding(
    binding: Mapping[str, Any], *, repo_relative: bool, code: str
) -> Snapshot:
    path_value = binding.get("path")
    if not isinstance(path_value, str):
        _fail(code)
    path = _repo_file(path_value, code) if repo_relative else Path(path_value)
    observed = _snapshot(path, code)
    if (
        binding.get("sha256") != observed.sha256
        or binding.get("bytes") != observed.size
    ):
        _fail(code)
    return observed


def load_config(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[Mapping[str, Any], dict[str, Snapshot]]:
    config_snapshot = _snapshot(config_path, "config_snapshot_invalid")
    config = _mapping(
        _json(config_snapshot.data, "config_json_invalid"), "config_invalid"
    )
    schema_binding = _mapping(
        _mapping(config.get("schemas"), "schemas_invalid").get("config"),
        "config_schema_binding_invalid",
    )
    config_schema_snapshot = _verify_file_binding(
        schema_binding, repo_relative=True, code="config_schema_identity_drift"
    )
    config_schema = _json(config_schema_snapshot.data, "config_schema_json_invalid")
    _validate_schema(config, config_schema, "config_schema_validation_failed")

    snapshots = {"config": config_snapshot, "config_schema": config_schema_snapshot}
    schemas = _mapping(config["schemas"], "schemas_invalid")
    for name in ("record", "manifest", "receipt"):
        snapshot = _verify_file_binding(
            _mapping(schemas[name], f"{name}_schema_binding_invalid"),
            repo_relative=True,
            code=f"{name}_schema_identity_drift",
        )
        try:
            Draft202012Validator.check_schema(
                _json(snapshot.data, f"{name}_schema_json_invalid")
            )
        except Exception:
            _fail(f"{name}_schema_invalid")
        snapshots[f"{name}_schema"] = snapshot
    snapshots["implementation"] = _verify_file_binding(
        _mapping(config["implementation"], "implementation_binding_invalid"),
        repo_relative=True,
        code="implementation_identity_drift",
    )
    return config, snapshots


def _source_snapshots(config: Mapping[str, Any]) -> dict[str, Snapshot]:
    source_files = _mapping(config["source_files"], "source_files_invalid")
    observed: dict[str, Snapshot] = {}
    for name in (
        "dataset",
        "manifest",
        "training_receipt",
        "failed_attempt_receipt",
        "source_snapshot_manifest",
    ):
        observed[name] = _verify_file_binding(
            _mapping(source_files[name], f"{name}_binding_invalid"),
            repo_relative=False,
            code=f"source_{name}_identity_drift",
        )
    return observed


def _validate_source_evidence(snapshots: Mapping[str, Snapshot]) -> None:
    manifest = _mapping(
        _json(snapshots["manifest"].data, "source_manifest_invalid"),
        "source_manifest_invalid",
    )
    training = _mapping(
        _json(snapshots["training_receipt"].data, "training_receipt_invalid"),
        "training_receipt_invalid",
    )
    failed = _mapping(
        _json(snapshots["failed_attempt_receipt"].data, "failed_receipt_invalid"),
        "failed_receipt_invalid",
    )
    source_snapshot = _mapping(
        _json(
            snapshots["source_snapshot_manifest"].data,
            "source_snapshot_manifest_invalid",
        ),
        "source_snapshot_manifest_invalid",
    )
    if (
        manifest.get("schema") != "air.route-dataset.manifest.v1"
        or manifest.get("status") != "passed"
        or manifest.get("dataset_sha256") != snapshots["dataset"].sha256
        or manifest.get("samples") != 16632
        or manifest.get("split_counts") != EXPECTED_COUNTS
        or manifest.get("source_group_overlap") != 0
        or manifest.get("leakage_family_overlap") != 0
        or manifest.get("prompt_sha256_overlap") != 0
    ):
        _fail("source_dataset_manifest_claims_invalid")
    if (
        training.get("schema") != "air.prefill-router-adapter.training-receipt.v1"
        or training.get("status") != "passed"
        or training.get("dataset_manifest_sha256") != snapshots["manifest"].sha256
        or training.get("dataset_sha256") != snapshots["dataset"].sha256
        or training.get("train_samples") != EXPECTED_COUNTS["train"]
    ):
        _fail("training_stage_evidence_invalid")
    if (
        failed.get("schema") != "air.route-head.failed-attempt-receipt.v2"
        or failed.get("status") != "failed"
        or failed.get("consumed") is not False
        or failed.get("dataset_gate_passed") is not True
        or failed.get("training_stage_passed") is not True
        or failed.get("failure_stage") != "post_training_provenance_gate"
        or sorted(failed.get("reason_codes", [])) != list(REASON_CODES)
    ):
        _fail("overall_attempt_evidence_invalid")
    if (
        source_snapshot.get("schema") != "air.source-snapshot.manifest.v1"
        or source_snapshot.get("status") != "frozen"
        or len(source_snapshot.get("entries", [])) != 23
    ):
        _fail("source_snapshot_evidence_invalid")


def _record_from_source(source: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    if set(source) not in (
        SOURCE_RECORD_KEYS,
        SOURCE_RECORD_KEYS | {"upstream_family_sha256"},
    ):
        _fail("source_record_shape_invalid")
    split = source.get("split")
    if split not in SPLITS:
        _fail("source_split_invalid")
    for name in (
        "sample_id",
        "source_group_sha256",
        "leakage_family_sha256",
        "prompt_sha256",
    ):
        value = source.get(name)
        if not isinstance(value, str) or len(value) != 64:
            _fail(f"source_{name}_invalid")
        try:
            int(value, 16)
        except ValueError:
            _fail(f"source_{name}_invalid")
    prompt = source.get("prompt")
    if (
        not isinstance(prompt, str)
        or _sha256(prompt.encode("utf-8")) != source["prompt_sha256"]
    ):
        _fail("source_prompt_digest_invalid")
    return {
        "schema_version": RECORD_VERSION,
        "source_ordinal": ordinal,
        "sample_id": source["sample_id"],
        "split": split,
        "source_group_sha256": source["source_group_sha256"],
        "leakage_family_sha256": source["leakage_family_sha256"],
        "prompt_sha256": source["prompt_sha256"],
        "consumable_for_training": False,
    }


def _scan_dataset(path: Path, expected_sha256: str) -> dict[str, list[dict[str, Any]]]:
    _reject_reparse_chain(path, "dataset_reparse_invalid")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
        _fail("dataset_not_regular")
    partitions: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            handle_before = os.fstat(handle.fileno())
            for ordinal, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                if not raw_line.endswith(b"\n"):
                    _fail("dataset_non_lf_or_missing_terminal_lf")
                source = _mapping(
                    _json(raw_line, "dataset_row_json_invalid"), "dataset_row_invalid"
                )
                record = _record_from_source(source, ordinal)
                partitions[record["split"]].append(record)
            handle_after = os.fstat(handle.fileno())
    except OSError:
        _fail("dataset_read_failed")
    after = os.lstat(path)
    if (
        len(
            {
                _identity(before),
                _identity(handle_before),
                _identity(handle_after),
                _identity(after),
            }
        )
        != 1
    ):
        _fail("dataset_changed_during_scan")
    if digest.hexdigest() != expected_sha256:
        _fail("dataset_sha256_drift")
    for split in SPLITS:
        partitions[split].sort(key=lambda record: record["sample_id"])
    return partitions


def _validate_partition_records(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]], record_schema: object
) -> None:
    try:
        Draft202012Validator.check_schema(record_schema)
        record_validator = Draft202012Validator(record_schema)
    except Exception:
        _fail("record_schema_invalid")
    sample_ids: set[str] = set()
    domains = ("source_group_sha256", "leakage_family_sha256", "prompt_sha256")
    values: dict[str, dict[str, set[str]]] = {
        domain: {split: set() for split in SPLITS} for domain in domains
    }
    for split in SPLITS:
        records = partitions.get(split)
        if not isinstance(records, Sequence) or len(records) != EXPECTED_COUNTS[split]:
            _fail("partition_count_invalid")
        previous = ""
        for record in records:
            try:
                record_validator.validate(record)
            except Exception:
                _fail("record_schema_validation_failed")
            if set(record) != OUTPUT_RECORD_KEYS or record["split"] != split:
                _fail("record_shape_invalid")
            if any(key in record for key in FORBIDDEN_PERSISTED_KEYS):
                _fail("body_field_leakage")
            sample_id = record["sample_id"]
            if sample_id <= previous:
                _fail("partition_order_or_duplicate_invalid")
            previous = sample_id
            if sample_id in sample_ids:
                _fail("sample_id_duplicate")
            sample_ids.add(sample_id)
            for domain in domains:
                values[domain][split].add(record[domain])
    if len(sample_ids) != sum(EXPECTED_COUNTS.values()):
        _fail("sample_id_inventory_invalid")
    for domain in domains:
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                if values[domain][left] & values[domain][right]:
                    _fail(f"{domain}_cross_split_overlap")


def _partition_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(record) for record in records)


def _file_binding(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _sha256(raw), "bytes": len(raw)}


def _sidecar_bytes(name: str, raw: bytes) -> bytes:
    return f"{_sha256(raw)}  {name}\n".encode("ascii")


def _logical_root(partitions: Sequence[Mapping[str, Any]]) -> str:
    preimage = [
        {
            "split": partition["split"],
            "sha256": partition["sha256"],
            "records": partition["records"],
        }
        for partition in partitions
    ]
    return _sha256(_canonical_json(preimage))


def _schema_inventory(
    config: Mapping[str, Any], snapshots: Mapping[str, Snapshot]
) -> list[dict[str, Any]]:
    schemas = _mapping(config["schemas"], "schemas_invalid")
    return [
        {
            "path": schemas[name]["path"],
            "sha256": snapshots[f"{name}_schema"].sha256,
            "bytes": snapshots[f"{name}_schema"].size,
        }
        for name in ("config", "record", "manifest", "receipt")
    ]


def _manifest(
    config: Mapping[str, Any],
    source: Mapping[str, Snapshot],
    partition_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "validation_asset_frozen",
        "asset_namespace": ASSET_NAMESPACE,
        "source": {
            "dataset": _file_binding(
                "source:dataset/route-dataset.jsonl", source["dataset"].data
            ),
            "manifest": _file_binding(
                "source:dataset/dataset-manifest.json", source["manifest"].data
            ),
            "source_snapshot_manifest": _file_binding(
                "source:provenance/source-snapshot-manifest.json",
                source["source_snapshot_manifest"].data,
            ),
        },
        "upstream_release": dict(
            _mapping(config["upstream_release"], "upstream_release_invalid")
        ),
        "evidence": {
            "training_receipt": _file_binding(
                "source:router-adapter/training-receipt.json",
                source["training_receipt"].data,
            ),
            "failed_attempt_receipt": _file_binding(
                "source:failed-attempt-receipt.json",
                source["failed_attempt_receipt"].data,
            ),
            "training_stage_status": "passed",
            "overall_attempt_status": "failed",
            "attempt_consumed": False,
            "failure_stage": "post_training_provenance_gate",
            "reason_codes": list(REASON_CODES),
        },
        "partitions": partition_inventory,
        "counts": {"total": 16632, **EXPECTED_COUNTS},
        "cross_split_overlap": {
            "source_group_sha256": 0,
            "leakage_family_sha256": 0,
            "prompt_sha256": 0,
        },
        "logical_inventory_root_sha256": _logical_root(partition_inventory),
        "policy": {
            **PARTITION_POLICIES,
            "holdout_body_persisted": False,
            "consumable_for_training": False,
        },
        "claims": {
            "training_authorized": False,
            "formal_training_authorized": False,
            "production_training_authorized": False,
            "deployable": False,
        },
        "resource_audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "persisted_body_fields": 0,
            "persisted_raw_token_ids": False,
        },
    }


def _receipt(
    config: Mapping[str, Any],
    snapshots: Mapping[str, Snapshot],
    manifest_raw: bytes,
    manifest_sidecar_raw: bytes,
    partitions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_VERSION,
        "status": "passed_validation_only",
        "config_sha256": snapshots["config"].sha256,
        "implementation_sha256": snapshots["implementation"].sha256,
        "schema_inventory": _schema_inventory(config, snapshots),
        "manifest_sha256": _sha256(manifest_raw),
        "manifest_sidecar_physical_sha256": _sha256(manifest_sidecar_raw),
        "partition_inventory": [
            {
                "split": item["split"],
                "path": item["path"],
                "sha256": item["sha256"],
                "sidecar_physical_sha256": item["sidecar_physical_sha256"],
                "bytes": item["bytes"],
                "records": item["records"],
            }
            for item in partitions
        ],
        "logical_inventory_root_sha256": _logical_root(partitions),
        "checks": {
            "source_hashes_authenticated": True,
            "records_schema_valid": True,
            "sample_ids_unique": True,
            "partition_order_deterministic": True,
            "cross_split_overlap_zero": True,
            "sidecars_valid": True,
            "single_bytes_snapshots": True,
            "toctou_rechecked": True,
            "atomic_create_once": True,
            "reparse_rejected": True,
            "body_fields_absent": True,
        },
        "claims": {
            "consumable_for_training": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "production_training_authorized": False,
            "attempt_consumed": False,
        },
        "resource_audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "persisted_body_fields": 0,
            "persisted_raw_token_ids": False,
        },
    }


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _named_entry_absent(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        _fail("output_identity_unreadable")
    return False


def _rename_noreplace(source: Path, target: Path) -> None:
    if not _named_entry_absent(target):
        _fail("output_already_exists")
    if os.name == "nt":
        import ctypes

        move = ctypes.windll.kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if not move(str(source), str(target), 0):
            _fail("atomic_noreplace_publish_failed")
    else:
        os.rename(source, target)


def _publish_payloads(
    output_root: Path,
    payloads: Mapping[str, bytes],
    *,
    before_publish: Callable[[Path], None] | None = None,
) -> None:
    output_root = Path(os.path.abspath(output_root))
    parent = output_root.parent
    _reject_reparse_chain(parent, "output_parent_reparse_invalid")
    if not _named_entry_absent(output_root):
        _fail("output_already_exists")
    staging = parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    expected: dict[str, tuple[str, int, tuple[int, int, int]]] = {}
    for name, raw in sorted(payloads.items()):
        if Path(name).name != name or name in (".", ".."):
            _fail("output_name_invalid")
        path = staging / name
        _write_exclusive(path, raw)
        snap = _snapshot(path, "staging_snapshot_invalid")
        expected[name] = (snap.sha256, snap.size, snap.identity)
    if before_publish is not None:
        before_publish(staging)
    if set(path.name for path in staging.iterdir()) != set(expected):
        _fail("staging_inventory_changed")
    for name, (digest, size, identity) in expected.items():
        observed = _snapshot(staging / name, "staging_toctou_invalid")
        if (observed.sha256, observed.size, observed.identity) != (
            digest,
            size,
            identity,
        ):
            _fail("staging_toctou_invalid")
    _rename_noreplace(staging, output_root)
    if set(path.name for path in output_root.iterdir()) != set(expected):
        _fail("published_inventory_invalid")
    for name, (digest, size, _) in expected.items():
        observed = _snapshot(output_root / name, "published_snapshot_invalid")
        if (observed.sha256, observed.size) != (digest, size):
            _fail("published_identity_invalid")


def build_asset(
    config_path: Path = DEFAULT_CONFIG, output_root: Path | None = None
) -> dict[str, Any]:
    config, snapshots = load_config(config_path)
    source = _source_snapshots(config)
    _validate_source_evidence(source)
    record_schema = _json(snapshots["record_schema"].data, "record_schema_invalid")
    partitions = _scan_dataset(source["dataset"].path, source["dataset"].sha256)
    _validate_partition_records(partitions, record_schema)

    payloads: dict[str, bytes] = {}
    partition_inventory: list[dict[str, Any]] = []
    for split in SPLITS:
        name = f"{split}.ids.jsonl"
        raw = _partition_bytes(partitions[split])
        sidecar_name = f"{name}.sha256"
        sidecar_raw = _sidecar_bytes(name, raw)
        payloads[name] = raw
        payloads[sidecar_name] = sidecar_raw
        partition_inventory.append(
            {
                "split": split,
                "path": name,
                "sidecar_path": sidecar_name,
                "sha256": _sha256(raw),
                "sidecar_physical_sha256": _sha256(sidecar_raw),
                "bytes": len(raw),
                "records": len(partitions[split]),
                "policy": PARTITION_POLICIES[split],
            }
        )
    manifest = _manifest(config, source, partition_inventory)
    manifest_schema = _json(
        snapshots["manifest_schema"].data, "manifest_schema_invalid"
    )
    _validate_schema(manifest, manifest_schema, "manifest_schema_validation_failed")
    manifest_raw = _canonical_json(manifest)
    manifest_sidecar_raw = _sidecar_bytes("manifest.json", manifest_raw)
    receipt = _receipt(
        config,
        snapshots,
        manifest_raw,
        manifest_sidecar_raw,
        partition_inventory,
    )
    receipt_schema = _json(snapshots["receipt_schema"].data, "receipt_schema_invalid")
    _validate_schema(receipt, receipt_schema, "receipt_schema_validation_failed")
    receipt_raw = _canonical_json(receipt)
    payloads.update(
        {
            "manifest.json": manifest_raw,
            "manifest.json.sha256": manifest_sidecar_raw,
            "build-receipt.json": receipt_raw,
            "build-receipt.json.sha256": _sidecar_bytes(
                "build-receipt.json", receipt_raw
            ),
        }
    )
    for snapshot in (*snapshots.values(), *source.values()):
        observed = _snapshot(snapshot.path, "terminal_toctou_recheck_failed")
        if (observed.sha256, observed.size, observed.identity) != (
            snapshot.sha256,
            snapshot.size,
            snapshot.identity,
        ):
            _fail("terminal_toctou_recheck_failed")
    destination = output_root or _repo_file(
        config["fixture_root"], "fixture_root_invalid"
    )
    _publish_payloads(destination, payloads)
    return audit_asset(config_path, destination)


def _verify_sidecar(payload: Snapshot, sidecar: Snapshot) -> None:
    if sidecar.data != _sidecar_bytes(payload.path.name, payload.data):
        _fail("sidecar_invalid")


def _read_partition(snapshot: Snapshot, split: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for raw in snapshot.data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            _fail("partition_non_lf")
        record = _mapping(
            _json(raw, "partition_json_invalid"), "partition_record_invalid"
        )
        if record.get("split") != split:
            _fail("partition_split_invalid")
        records.append(record)
    return records


def audit_asset(
    config_path: Path = DEFAULT_CONFIG, fixture_root: Path | None = None
) -> dict[str, Any]:
    config, snapshots = load_config(config_path)
    root = fixture_root or _repo_file(config["fixture_root"], "fixture_root_invalid")
    _reject_reparse_chain(root, "fixture_reparse_invalid")
    expected_names = {
        "manifest.json",
        "manifest.json.sha256",
        "build-receipt.json",
        "build-receipt.json.sha256",
        *(f"{split}.ids.jsonl" for split in SPLITS),
        *(f"{split}.ids.jsonl.sha256" for split in SPLITS),
    }
    if set(path.name for path in root.iterdir()) != expected_names:
        _fail("fixture_inventory_invalid")
    files = {
        name: _snapshot(root / name, "fixture_file_invalid") for name in expected_names
    }
    for payload_name in (
        "manifest.json",
        "build-receipt.json",
        *(f"{split}.ids.jsonl" for split in SPLITS),
    ):
        _verify_sidecar(files[payload_name], files[f"{payload_name}.sha256"])
    manifest = _mapping(
        _json(files["manifest.json"].data, "manifest_json_invalid"), "manifest_invalid"
    )
    receipt = _mapping(
        _json(files["build-receipt.json"].data, "receipt_json_invalid"),
        "receipt_invalid",
    )
    _validate_schema(
        manifest,
        _json(snapshots["manifest_schema"].data, "manifest_schema_invalid"),
        "manifest_schema_validation_failed",
    )
    _validate_schema(
        receipt,
        _json(snapshots["receipt_schema"].data, "receipt_schema_invalid"),
        "receipt_schema_validation_failed",
    )
    partitions = {
        split: _read_partition(files[f"{split}.ids.jsonl"], split) for split in SPLITS
    }
    _validate_partition_records(
        partitions,
        _json(snapshots["record_schema"].data, "record_schema_invalid"),
    )
    listed = {item["split"]: item for item in manifest["partitions"]}
    if set(listed) != set(SPLITS):
        _fail("manifest_partition_inventory_invalid")
    for split in SPLITS:
        payload = files[f"{split}.ids.jsonl"]
        sidecar = files[f"{split}.ids.jsonl.sha256"]
        item = listed[split]
        if (
            item["sha256"] != payload.sha256
            or item["bytes"] != payload.size
            or item["records"] != len(partitions[split])
            or item["sidecar_physical_sha256"] != sidecar.sha256
        ):
            _fail("manifest_partition_identity_invalid")
    if manifest["logical_inventory_root_sha256"] != _logical_root(
        manifest["partitions"]
    ):
        _fail("logical_inventory_root_invalid")
    if (
        receipt["manifest_sha256"] != files["manifest.json"].sha256
        or receipt["manifest_sidecar_physical_sha256"]
        != files["manifest.json.sha256"].sha256
        or receipt["logical_inventory_root_sha256"]
        != manifest["logical_inventory_root_sha256"]
    ):
        _fail("receipt_binding_invalid")
    return {
        "status": "passed_validation_only",
        "manifest_sha256": files["manifest.json"].sha256,
        "manifest_sidecar_physical_sha256": files["manifest.json.sha256"].sha256,
        "receipt_sha256": files["build-receipt.json"].sha256,
        "receipt_sidecar_physical_sha256": files["build-receipt.json.sha256"].sha256,
        "logical_inventory_root_sha256": manifest["logical_inventory_root_sha256"],
        "counts": manifest["counts"],
        "consumable_for_training": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "audit"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (
            build_asset(args.config, args.output_root)
            if args.command == "build"
            else audit_asset(args.config, args.output_root)
        )
    except ValidationAssetError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
