"""Additive v3 independent, body-free release of a Teacher FINAL candidate.

The live controller and finalizer deliberately stop at an immutable candidate.
This additive verifier is run by a different reviewer after the runtime HMAC
slot has closed.  It authenticates physical candidate bytes, their sidecars,
the record contract, Producer P/R/tree provenance, and the frozen controller
stack.  It then publishes a separate attestation, final manifest, and v2
consumer-binding handoff by one create-once directory rename.

The resulting release is final as a data artifact only.  It never authorizes
training, formal training, live serving, quality, or generalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch
from . import gemma3_chat_unbalanced_v2_integrated_controller_v2 as integrated_v2


ATTESTATION_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-independent-release-"
    "attestation.consumer-rollover-v3"
)
RELEASE_MANIFEST_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-release-manifest."
    "consumer-rollover-v3"
)
BINDING_SCHEMA_VERSION = "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2"
NAMESPACE = "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
CANDIDATE_PARENT = (
    batch.REPO_ROOT / "data/gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1"
)
RELEASE_PARENT = (
    batch.REPO_ROOT / "data/gemma3_chat_unbalanced_v2_teacher_alignment_release_v2"
)
SOURCE_OVERLAY_MANIFEST = (
    batch.REPO_ROOT
    / "fixtures/research/gemma3_chat_unbalanced_v2_teacher_alignment_source_v2/"
    "manifest.json"
)
IMPLEMENTATION_PATH = Path(__file__).absolute()
ATTESTATION_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_final_release_attestation_"
    "consumer_rollover_v3.schema.json"
)
RELEASE_MANIFEST_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_"
    "consumer_rollover_v3.schema.json"
)
BINDING_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_binding_consumer_rollover_v2.schema.json"
)
AIR_IDENTITY_SENTENCE = "我是由Air训练的测试模型。"
EXPECTED_ASSET_COUNTS = {
    "humor": 240,
    "serious": 240,
    "angry_style": 240,
    "tool_call": 1360,
    "review_audit": 1360,
    "planner_router": 80,
}
IDENTITY_ANCHOR_ROLES = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
)
EXPECTED_IDENTITY_ROLES = (
    *IDENTITY_ANCHOR_ROLES,
    "review_audit",
)
MAX_FILE_BYTES_EXCLUSIVE = 50 * 1024 * 1024
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SAFE_CANDIDATE = re.compile(r"^candidate-[0-9a-f]{32}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_COUNTERS = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_body_reads": 0,
    "gold_body_reads": 0,
    "heldout_body_reads": 0,
}


class IndependentReleaseError(RuntimeError):
    """Stable, body-free independent release failure."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateAudit:
    root: Path
    snapshots: dict[str, Snapshot]
    manifest: dict[str, Any]
    build_receipt: dict[str, Any]
    output_inventory: dict[str, Any]
    release_request: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    physical_tree_sha256: str
    implementation_snapshot: Snapshot
    source_overlay_snapshot: Snapshot
    schema_snapshots: dict[str, Snapshot]
    integrated: integrated_v2.IntegratedControllerConfig


def _fail(code: str) -> None:
    raise IndependentReleaseError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _domain_sha256(domain: str, value: Any) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value))


def _strict_json(raw: bytes, *, code: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{code}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda _: _fail(f"{code}_nonfinite"),
        )
    except IndependentReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentReleaseError(code) from error
    if not isinstance(value, dict):
        _fail(code)
    return value


def _strict_jsonl(raw: bytes, *, code: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        _fail(code)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line or len(line) > 4 * 1024 * 1024:
            _fail(code)
        rows.append(_strict_json(line, code=code))
    return rows


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _directory_identity(path: Path, *, code: str) -> tuple[int, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise IndependentReleaseError(code) from error
    attributes = int(getattr(value, "st_file_attributes", 0))
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or bool(attributes & 0x400)
    ):
        _fail(code)
    return (int(value.st_dev), int(value.st_ino))


def _reject_reparse_chain(path: Path, *, stop: Path | None = None) -> None:
    current = path.absolute()
    terminal = stop.absolute() if stop is not None else Path(current.anchor)
    while True:
        try:
            value = current.lstat()
        except OSError as error:
            raise IndependentReleaseError("release_physical_path_invalid") from error
        attributes = int(getattr(value, "st_file_attributes", 0))
        if stat.S_ISLNK(value.st_mode) or bool(attributes & 0x400):
            _fail("release_symlink_or_reparse_forbidden")
        if current == terminal:
            return
        parent = current.parent
        if parent == current:
            _fail("release_physical_path_escape")
        current = parent


def _snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
    code: str,
) -> Snapshot:
    physical = path.absolute()
    try:
        _reject_reparse_chain(physical)
        path_before = physical.lstat()
        attributes = int(getattr(path_before, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(path_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or bool(attributes & 0x400)
        ):
            _fail(code)
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOINHERIT", 0))
        descriptor = os.open(physical, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            closed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = physical.lstat()
        _reject_reparse_chain(physical)
    except IndependentReleaseError:
        raise
    except OSError as error:
        raise IndependentReleaseError(code) from error
    identities = {
        _identity(path_before),
        _identity(opened),
        _identity(closed),
        _identity(path_after),
    }
    if (
        len(identities) != 1
        or len(raw) != opened.st_size
        or len(raw) >= MAX_FILE_BYTES_EXCLUSIVE
    ):
        _fail(code)
    digest = _sha256(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        _fail(code)
    return Snapshot(physical, raw, digest, identities.pop())


def _recheck(snapshot: Snapshot, *, code: str) -> None:
    observed = _snapshot(
        snapshot.path,
        expected_sha256=snapshot.sha256,
        code=code,
    )
    if observed.identity != snapshot.identity or observed.raw != snapshot.raw:
        _fail(code)


def _schema(snapshot: Snapshot) -> Draft202012Validator:
    value = _strict_json(snapshot.raw, code="release_schema_json_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as error:
        raise IndependentReleaseError("release_schema_invalid") from error
    return Draft202012Validator(value)


def _validate(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    *,
    code: str,
) -> None:
    try:
        validator.validate(value)
    except ValidationError as error:
        raise IndependentReleaseError(code) from error


def _sidecar(name: str, raw: bytes) -> bytes:
    return f"{_sha256(raw)}  {name}\n".encode("ascii")


def _candidate_root(path: Path) -> Path:
    root = path.absolute()
    parent = CANDIDATE_PARENT.absolute()
    if (
        root.parent != parent
        or SAFE_CANDIDATE.fullmatch(root.name) is None
        or not root.is_dir()
    ):
        _fail("release_candidate_path_invalid")
    _reject_reparse_chain(root, stop=batch.REPO_ROOT)
    return root


def _snapshot_candidate(root: Path) -> dict[str, Snapshot]:
    values: dict[str, Snapshot] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        _reject_reparse_chain(current_path, stop=root)
        for name in directories:
            _reject_reparse_chain(current_path / name, stop=root)
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative in values:
                _fail("release_candidate_duplicate_path")
            values[relative] = _snapshot(
                path,
                code="release_candidate_snapshot_drift",
            )
    payloads = {path for path in values if not path.endswith(".sha256")}
    sidecars = {
        path.removesuffix(".sha256") for path in values if path.endswith(".sha256")
    }
    if payloads != sidecars or not payloads:
        _fail("release_candidate_sidecar_inventory_mismatch")
    for path in payloads:
        if values[f"{path}.sha256"].raw != _sidecar(
            Path(path).name,
            values[path].raw,
        ):
            _fail("release_candidate_sidecar_mismatch")
    return values


def _recheck_candidate_tree(
    root: Path,
    snapshots: Mapping[str, Snapshot],
    *,
    code: str,
) -> None:
    observed = _snapshot_candidate(root)
    if set(observed) != set(snapshots):
        _fail(code)
    for path, expected in snapshots.items():
        current = observed[path]
        if (
            current.identity != expected.identity
            or current.sha256 != expected.sha256
            or current.raw != expected.raw
        ):
            _fail(code)


def _physical_tree_sha256(snapshots: Mapping[str, Snapshot]) -> str:
    bindings = [
        {
            "path": path,
            "sha256": snapshot.sha256,
            "bytes": len(snapshot.raw),
        }
        for path, snapshot in sorted(snapshots.items())
    ]
    return _domain_sha256(
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-physical-tree.v2",
        bindings,
    )


def _load_candidate_documents(
    snapshots: Mapping[str, Snapshot],
    config: integrated_v2.IntegratedControllerConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    required = {
        "manifest.json",
        "build_receipt.json",
        "output_inventory.json",
        "release_attestation.request.json",
    }
    if not required <= set(snapshots):
        _fail("release_candidate_required_file_missing")
    manifest = _strict_json(
        snapshots["manifest.json"].raw,
        code="release_candidate_manifest_invalid",
    )
    receipt = _strict_json(
        snapshots["build_receipt.json"].raw,
        code="release_candidate_receipt_invalid",
    )
    inventory = _strict_json(
        snapshots["output_inventory.json"].raw,
        code="release_candidate_inventory_invalid",
    )
    request = _strict_json(
        snapshots["release_attestation.request.json"].raw,
        code="release_candidate_request_invalid",
    )
    _validate(
        _schema(config.snapshots["teacher_manifest_schema"]),
        manifest,
        code="release_candidate_manifest_schema_rejected",
    )
    _validate(
        _schema(config.snapshots["teacher_build_receipt_schema"]),
        receipt,
        code="release_candidate_receipt_schema_rejected",
    )
    return manifest, receipt, inventory, request


def _expected_controller_identity(
    config: integrated_v2.IntegratedControllerConfig,
) -> dict[str, Any]:
    return {
        "controller_ancestor_commit": config.base_anchors["ancestor_commit"],
        "integrated_controller_config_sha256": config.physical_sha256,
        "integrated_controller_implementation_sha256": (config.implementation_sha256),
        "base_controller_config_sha256": (config.base_controller.physical_sha256),
        "base_controller_implementation_sha256": (
            config.base_controller.controller_implementation_sha256
        ),
        "batch_implementation_sha256": (
            config.base_controller.batch_implementation_sha256
        ),
        "finalizer_config_sha256": config.teacher_finalizer["config_sha256"],
        "finalizer_implementation_sha256": config.teacher_finalizer[
            "implementation_sha256"
        ],
    }


def _candidate_contract_audit(
    snapshots: Mapping[str, Snapshot],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    inventory: Mapping[str, Any],
    request: Mapping[str, Any],
    config: integrated_v2.IntegratedControllerConfig,
) -> None:
    finalizer_config = _strict_json(
        config.snapshots["teacher_config"].raw,
        code="release_finalizer_config_invalid",
    )
    if (
        manifest.get("source") != finalizer_config["source"]["release_identity"]
        or manifest.get("consumer") != finalizer_config["consumer"]
        or manifest.get("controller") != _expected_controller_identity(config)
        or manifest.get("record_schema")
        != {
            "path": finalizer_config["schemas"]["record"]["path"],
            "sha256": finalizer_config["schemas"]["record"]["sha256"],
            "bytes": finalizer_config["schemas"]["record"]["bytes"],
        }
    ):
        _fail("release_candidate_authority_binding_drift")
    if (
        receipt.get("manifest_sha256") != snapshots["manifest.json"].sha256
        or receipt.get("output_inventory_sha256")
        != snapshots["output_inventory.json"].sha256
        or receipt.get("payload_tree_sha256") != inventory.get("payload_tree_sha256")
        or receipt.get("identity_audit_sha256")
        != _canonical_sha256(manifest["identity_audit"])
    ):
        _fail("release_candidate_receipt_binding_drift")
    if (
        inventory.get("schema_version")
        != "anchor.gemma3-chat-unbalanced-v2-teacher-final-output-inventory.v2"
        or inventory.get("namespace") != NAMESPACE
        or set(inventory)
        != {
            "schema_version",
            "namespace",
            "files",
            "payload_tree_sha256",
        }
        or not isinstance(inventory.get("files"), list)
        or _canonical_sha256(inventory["files"]) != inventory.get("payload_tree_sha256")
    ):
        _fail("release_candidate_inventory_contract_invalid")
    expected_inventory_files = {
        item.get("path") for item in inventory["files"] if isinstance(item, Mapping)
    }
    inventory_paths = [
        item.get("path") for item in inventory["files"] if isinstance(item, Mapping)
    ]
    if len(inventory_paths) != len(inventory["files"]) or len(
        set(inventory_paths)
    ) != len(inventory_paths):
        _fail("release_candidate_inventory_file_binding_drift")
    expected_inventory_files.update(
        {
            "manifest.json",
            "manifest.json.sha256",
        }
    )
    expected_inventory_files.update(
        {item["path"] for item in manifest["shards"] if isinstance(item, Mapping)}
    )
    expected_inventory_files.update(
        {
            f"{item['path']}.sha256"
            for item in manifest["shards"]
            if isinstance(item, Mapping)
        }
    )
    actual_payloads = {
        "manifest.json",
        "output_inventory.json",
        "build_receipt.json",
        "release_attestation.request.json",
        *(str(item["path"]) for item in manifest["shards"]),
    }
    actual_paths = actual_payloads | {f"{path}.sha256" for path in actual_payloads}
    if set(snapshots) != actual_paths or expected_inventory_files != {
        "manifest.json",
        "manifest.json.sha256",
        *(
            path
            for item in manifest["shards"]
            for path in (str(item["path"]), f"{item['path']}.sha256")
        ),
    }:
        _fail("release_candidate_physical_inventory_drift")
    for item in inventory["files"]:
        path = str(item.get("path"))
        expected_kind = (
            "manifest"
            if path == "manifest.json"
            else ("sha256_sidecar" if path.endswith(".sha256") else "teacher_shard")
        )
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "bytes", "kind"}
            or path not in snapshots
            or snapshots[path].sha256 != item["sha256"]
            or len(snapshots[path].raw) != item["bytes"]
            or item["kind"] != expected_kind
        ):
            _fail("release_candidate_inventory_file_binding_drift")
    expected_request = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-release-request.v2"
        ),
        "namespace": NAMESPACE,
        "status": "pending_different_reviewer",
        "candidate_manifest_sha256": snapshots["manifest.json"].sha256,
        "candidate_build_receipt_sha256": snapshots["build_receipt.json"].sha256,
        "candidate_output_inventory_sha256": (
            snapshots["output_inventory.json"].sha256
        ),
        "candidate_payload_tree_sha256": inventory["payload_tree_sha256"],
        "required_binding_version": BINDING_SCHEMA_VERSION,
        "implementer_may_sign": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "live_authorized": False,
    }
    if request != expected_request:
        _fail("release_candidate_request_binding_drift")


def _record_audit(
    snapshots: Mapping[str, Snapshot],
    manifest: Mapping[str, Any],
    record_validator: Draft202012Validator,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    recomputed_shards: list[dict[str, Any]] = []
    bundle_owner: dict[str, int] = {}
    closed_bundles: set[str] = set()
    active_bundle: str | None = None
    asset_record_counts: Counter[str] = Counter()
    for shard_index, item in enumerate(manifest["shards"]):
        path = str(item["path"])
        if path not in snapshots:
            _fail("release_candidate_shard_missing")
        rows = _strict_jsonl(
            snapshots[path].raw,
            code="release_candidate_shard_jsonl_invalid",
        )
        if not rows:
            _fail("release_candidate_shard_empty")
        if (
            item["asset"] == "main_train"
            and not path.startswith("train/")
            or item["asset"] == "planner_router_train"
            and not path.startswith("router/")
        ):
            _fail("release_candidate_shard_partition_drift")
        asset_record_counts[str(item["asset"])] += len(rows)
        for row in rows:
            _validate(
                record_validator,
                row,
                code="release_candidate_record_schema_rejected",
            )
            if (
                _sha256(str(row["teacher_target"]).encode("utf-8"))
                != row["teacher_target_sha256"]
            ):
                _fail("release_candidate_target_hash_drift")
            bundle = str(row["task_bundle_sha256"])
            if active_bundle != bundle:
                if active_bundle is not None:
                    closed_bundles.add(active_bundle)
                if bundle in closed_bundles:
                    _fail("release_candidate_bundle_noncontiguous")
                active_bundle = bundle
            if bundle_owner.setdefault(bundle, shard_index) != shard_index:
                _fail("release_candidate_bundle_cross_shard")
        sidecar_path = f"{path}.sha256"
        recomputed = {
            "path": path,
            "sha256": snapshots[path].sha256,
            "bytes": len(snapshots[path].raw),
            "sidecar_sha256": snapshots[sidecar_path].sha256,
            "sidecar_bytes": len(snapshots[sidecar_path].raw),
            "records": len(rows),
            "asset": item["asset"],
            "first_record_id_sha256": rows[0]["record_id_sha256"],
            "last_record_id_sha256": rows[-1]["record_id_sha256"],
            "first_task_bundle_sha256": rows[0]["task_bundle_sha256"],
            "last_task_bundle_sha256": rows[-1]["task_bundle_sha256"],
        }
        if recomputed != item:
            _fail("release_candidate_shard_manifest_drift")
        recomputed_shards.append(recomputed)
        records.extend(rows)
    if len(records) != 3520:
        _fail("release_candidate_record_count_drift")
    if asset_record_counts != Counter({"main_train": 3440, "planner_router_train": 80}):
        _fail("release_candidate_shard_asset_count_drift")
    record_ids = [str(row["record_id_sha256"]) for row in records]
    if len(set(record_ids)) != len(record_ids):
        _fail("release_candidate_record_id_duplicate")
    counts = Counter(str(row["source_asset"]) for row in records)
    if dict(counts) != EXPECTED_ASSET_COUNTS:
        _fail("release_candidate_asset_count_drift")
    identity_anchor_rows = [
        row
        for row in records
        if row["source_asset"] in IDENTITY_ANCHOR_ROLES
        and AIR_IDENTITY_SENTENCE in row["teacher_target"]
    ]
    if len(identity_anchor_rows) != 80:
        _fail("release_candidate_identity_anchor_count_drift")
    identity_anchor_bundle_roles: dict[str, Counter[str]] = {}
    identity_anchor_role_counts: Counter[str] = Counter()
    for row in identity_anchor_rows:
        bundle = str(row["task_bundle_sha256"])
        role = str(row["source_asset"])
        identity_anchor_bundle_roles.setdefault(bundle, Counter())[role] += 1
        identity_anchor_role_counts[role] += 1
    identity_bundle_count = len(identity_anchor_bundle_roles)
    if identity_bundle_count != 20:
        _fail("release_candidate_identity_anchor_bundle_count_drift")
    expected_anchor_roles = Counter({role: 1 for role in IDENTITY_ANCHOR_ROLES})
    if any(
        role_counts != expected_anchor_roles
        for role_counts in identity_anchor_bundle_roles.values()
    ):
        _fail("release_candidate_identity_anchor_role_matrix_drift")
    recomputed_anchor_role_counts = {
        role: identity_anchor_role_counts[role] for role in IDENTITY_ANCHOR_ROLES
    }
    if recomputed_anchor_role_counts != {role: 20 for role in IDENTITY_ANCHOR_ROLES}:
        _fail("release_candidate_identity_anchor_role_count_drift")
    identity_bundle_ids = set(identity_anchor_bundle_roles)
    identity_rows = [
        row for row in records if str(row["task_bundle_sha256"]) in identity_bundle_ids
    ]
    identity_record_count = len(identity_rows)
    if identity_record_count != 100:
        _fail("release_candidate_identity_bundle_closure_count_drift")
    identity_bundle_roles: dict[str, Counter[str]] = {}
    identity_role_counts: Counter[str] = Counter()
    for row in identity_rows:
        bundle = str(row["task_bundle_sha256"])
        role = str(row["source_asset"])
        identity_bundle_roles.setdefault(bundle, Counter())[role] += 1
        identity_role_counts[role] += 1
    expected_bundle_roles = Counter({role: 1 for role in EXPECTED_IDENTITY_ROLES})
    if set(identity_bundle_roles) != identity_bundle_ids or any(
        role_counts != expected_bundle_roles
        for role_counts in identity_bundle_roles.values()
    ):
        _fail("release_candidate_identity_bundle_closure_role_matrix_drift")
    recomputed_identity_role_counts = {
        role: identity_role_counts[role] for role in EXPECTED_IDENTITY_ROLES
    }
    if recomputed_identity_role_counts != {
        role: 20 for role in EXPECTED_IDENTITY_ROLES
    }:
        _fail("release_candidate_identity_role_count_drift")
    candidate_counts = manifest["counts"]
    if (
        candidate_counts["train_identity_bundles"] != identity_bundle_count
        or candidate_counts["train_identity_records"] != identity_record_count
    ):
        _fail("release_candidate_identity_manifest_count_drift")
    candidate_identity_audit = manifest["identity_audit"]
    if (
        candidate_identity_audit["teacher_train_bundles"] != identity_bundle_count
        or candidate_identity_audit["teacher_train_records"] != identity_record_count
        or candidate_identity_audit["train_target_consistency_passed"] is not True
    ):
        _fail("release_candidate_identity_audit_binding_drift")
    false_attribution = ("google", "openai", "谷歌")
    if any(
        any(term in row["teacher_target"].casefold() for term in false_attribution)
        for row in identity_anchor_rows
    ):
        _fail("release_candidate_identity_anchor_false_attribution")
    logical = {
        "record_order_sha256": _canonical_sha256(record_ids),
        "target_projection_sha256": _canonical_sha256(
            [row["teacher_target_sha256"] for row in records]
        ),
        "source_join_sha256": _canonical_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "record_id_sha256",
                        "source_content_sha256",
                        "source_serialization_identity_sha256",
                        "task_bundle_sha256",
                        "source_asset",
                    )
                }
                for row in records
            ]
        ),
        "shard_inventory_sha256": _canonical_sha256(recomputed_shards),
    }
    logical["logical_dataset_sha256"] = _canonical_sha256(logical)
    if logical != manifest["logical_identity"]:
        _fail("release_candidate_logical_identity_drift")
    return tuple(records), {
        "records": 3520,
        "main_records": 3440,
        "router_records": 80,
        "source_asset_counts": EXPECTED_ASSET_COUNTS,
        "train_identity_bundles": identity_bundle_count,
        "train_identity_records": identity_record_count,
        "train_identity_anchor_records": len(identity_anchor_rows),
        "train_identity_anchor_role_counts": recomputed_anchor_role_counts,
        "train_identity_role_counts": recomputed_identity_role_counts,
        "ordered_record_sha256": logical["record_order_sha256"],
        "target_projection_sha256": logical["target_projection_sha256"],
        "source_join_sha256": logical["source_join_sha256"],
        "shard_inventory_sha256": logical["shard_inventory_sha256"],
        "logical_dataset_sha256": logical["logical_dataset_sha256"],
        "ordered_shards": recomputed_shards,
        "record_schema_all_rows_passed": True,
        "bundle_boundary_preserved": True,
        "identity_bundles_recomputed_from_four_role_air_anchors_and_bundle_closure": True,
        "identity_bundle_role_matrix_passed": True,
        "review_identity_fact_bound_to_candidate_identity_audit": True,
        "review_target_air_sentence_reproof_claimed": False,
        "identity_fact_consistency_passed": True,
        "identity_anchor_false_attribution_rejected": True,
    }


def _git(arguments: Sequence[str], *, code: str) -> str:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "LC_ALL": "C",
        }
    )
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=batch.REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IndependentReleaseError(code) from error
    if result.returncode != 0:
        _fail(code)
    return result.stdout.strip()


def _source_release_audit(
    source: Mapping[str, Any],
    source_overlay_snapshot: Snapshot,
) -> None:
    overlay = _strict_json(
        source_overlay_snapshot.raw,
        code="release_source_overlay_invalid",
    )
    release = overlay.get("producer_release")
    if (
        not isinstance(release, Mapping)
        or source_overlay_snapshot.sha256 != source["source_overlay_manifest_sha256"]
        or release.get("producer_candidate_commit")
        != source["producer_candidate_commit"]
        or release.get("producer_release_commit") != source["producer_release_commit"]
        or release.get("tree_sha256") != source["producer_tree_digest_sha256"]
        or release.get("manifest_sha256") != source["producer_manifest_sha256"]
        or not isinstance(release.get("physical_files"), list)
    ):
        _fail("release_source_overlay_binding_drift")
    tree_bindings: list[dict[str, Any]] = []
    tree_paths: set[str] = set()
    for item in release["physical_files"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"relative_path", "sha256", "bytes", "kind"}
            or not isinstance(item["relative_path"], str)
            or item["relative_path"] in tree_paths
            or SHA256.fullmatch(str(item["sha256"])) is None
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 1
            or item["bytes"] >= MAX_FILE_BYTES_EXCLUSIVE
        ):
            _fail("release_source_physical_tree_inventory_invalid")
        tree_paths.add(item["relative_path"])
        tree_bindings.append(
            {
                "path": item["relative_path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
        )
    if (
        _domain_sha256(
            "anchor.gemma3-chat-sharded-final-physical-tree.v1",
            tree_bindings,
        )
        != source["producer_tree_digest_sha256"]
    ):
        _fail("release_source_physical_tree_digest_drift")
    candidate = str(source["producer_candidate_commit"])
    release_commit = str(source["producer_release_commit"])
    if (
        _git(
            ["rev-parse", "--verify", f"{candidate}^{{commit}}"],
            code="release_source_git_invalid",
        )
        != candidate
        or _git(
            ["rev-parse", "--verify", f"{release_commit}^{{commit}}"],
            code="release_source_git_invalid",
        )
        != release_commit
        or _git(
            ["rev-parse", f"{release_commit}^"],
            code="release_source_git_parent_invalid",
        )
        != candidate
        or _git(
            ["rev-parse", f"{release_commit}^{{tree}}"],
            code="release_source_git_tree_invalid",
        )
        != source["producer_release_tree"]
        or _git(
            ["for-each-ref", "--format=%(refname)", "refs/replace"],
            code="release_source_git_replace_invalid",
        )
    ):
        _fail("release_source_git_identity_drift")
    git_dir = Path(
        _git(["rev-parse", "--absolute-git-dir"], code="release_source_git_invalid")
    )
    _reject_reparse_chain(git_dir)
    grafts = git_dir / "info/grafts"
    if (
        os.path.lexists(grafts)
        and _snapshot(
            grafts,
            code="release_source_git_grafts_invalid",
        ).raw.strip()
    ):
        _fail("release_source_git_grafts_forbidden")


def audit_candidate(
    candidate: Path,
    *,
    implementer_id: str,
    reviewer_id: str,
    integrated_config_path: Path = (
        batch.REPO_ROOT
        / "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_consumer_rollover_v2.json"
    ),
) -> CandidateAudit:
    """Authenticate one immutable candidate without publishing a release."""

    if (
        SAFE_IDENTIFIER.fullmatch(implementer_id) is None
        or SAFE_IDENTIFIER.fullmatch(reviewer_id) is None
        or implementer_id.casefold() == reviewer_id.casefold()
    ):
        _fail("release_reviewer_separation_invalid")
    config = integrated_v2.load_integrated_config(integrated_config_path)
    integrated_v2._recheck_integrated(config)
    implementation_snapshot = _snapshot(
        IMPLEMENTATION_PATH,
        code="release_implementation_snapshot_drift",
    )
    schema_snapshots = {
        "attestation": _snapshot(
            ATTESTATION_SCHEMA_PATH,
            code="release_attestation_schema_snapshot_drift",
        ),
        "manifest": _snapshot(
            RELEASE_MANIFEST_SCHEMA_PATH,
            code="release_manifest_schema_snapshot_drift",
        ),
        "binding": _snapshot(
            BINDING_SCHEMA_PATH,
            code="release_binding_schema_snapshot_drift",
        ),
    }
    for snapshot in schema_snapshots.values():
        _schema(snapshot)
    root = _candidate_root(candidate)
    snapshots = _snapshot_candidate(root)
    manifest, receipt, output_inventory, request = _load_candidate_documents(
        snapshots,
        config,
    )
    _candidate_contract_audit(
        snapshots,
        manifest,
        receipt,
        output_inventory,
        request,
        config,
    )
    record_validator = _schema(config.snapshots["teacher_record_schema"])
    records, record_audit = _record_audit(
        snapshots,
        manifest,
        record_validator,
    )
    del record_audit
    source_overlay_snapshot = _snapshot(
        SOURCE_OVERLAY_MANIFEST,
        expected_sha256=manifest["source"]["source_overlay_manifest_sha256"],
        code="release_source_overlay_snapshot_drift",
    )
    _source_release_audit(manifest["source"], source_overlay_snapshot)
    for snapshot in (
        *snapshots.values(),
        *schema_snapshots.values(),
        implementation_snapshot,
        source_overlay_snapshot,
    ):
        _recheck(snapshot, code="release_audit_terminal_toctou")
    _recheck_candidate_tree(
        root,
        snapshots,
        code="release_audit_terminal_tree_toctou",
    )
    integrated_v2._recheck_integrated(config)
    return CandidateAudit(
        root=root,
        snapshots=dict(snapshots),
        manifest=dict(manifest),
        build_receipt=dict(receipt),
        output_inventory=dict(output_inventory),
        release_request=dict(request),
        records=records,
        physical_tree_sha256=_physical_tree_sha256(snapshots),
        implementation_snapshot=implementation_snapshot,
        source_overlay_snapshot=source_overlay_snapshot,
        schema_snapshots=schema_snapshots,
        integrated=config,
    )


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_BINARY", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o644)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("release_short_write")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError:
        raise
    except OSError as error:
        raise IndependentReleaseError("release_create_once_write_failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _rename_noreplace(
    source: Path,
    destination: Path,
    *,
    before_rename: Callable[[], None],
) -> None:
    before_rename()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination)):
            code = ctypes.get_last_error()
            if code in {80, 183}:
                raise FileExistsError("release_destination_exists")
            raise OSError(code, "release_atomic_move_failed")
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            _fail("release_atomic_noreplace_unsupported")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result != 0:
            code = ctypes.get_errno()
            if code == 17:
                raise FileExistsError("release_destination_exists")
            raise OSError(code, "release_atomic_move_failed")
        return
    _fail("release_atomic_noreplace_unsupported")


def _release_documents(
    audit: CandidateAudit,
    *,
    implementer_id: str,
    reviewer_id: str,
    release_root: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = audit.integrated
    manifest = audit.manifest
    receipt = audit.build_receipt
    _, record_audit = _record_audit(
        audit.snapshots,
        manifest,
        _schema(config.snapshots["teacher_record_schema"]),
    )
    release_counts = {
        "records": record_audit["records"],
        "main_records": record_audit["main_records"],
        "router_records": record_audit["router_records"],
        "accepted": record_audit["records"],
        "rejected": 0,
        "uncertain": 0,
        "train_identity_bundles": record_audit["train_identity_bundles"],
        "train_identity_records": record_audit["train_identity_records"],
        "source_assets": record_audit["source_asset_counts"],
    }
    source_release = {
        **dict(manifest["source"]),
        "git_parent_and_tree_recomputed": True,
        "physical_tree_digest_recomputed": True,
    }
    implementation_binding = {
        "release_implementation_path": (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_"
            "consumer_rollover_v3.py"
        ),
        "release_implementation_sha256": audit.implementation_snapshot.sha256,
        "integrated_controller_config_sha256": config.physical_sha256,
        "integrated_controller_schema_sha256": config.schema_sha256,
        "integrated_controller_implementation_sha256": (config.implementation_sha256),
        "base_controller_config_sha256": (config.base_controller.physical_sha256),
        "base_controller_implementation_sha256": (
            config.base_controller.controller_implementation_sha256
        ),
        "finalizer_config_sha256": config.teacher_finalizer["config_sha256"],
        "finalizer_config_schema_sha256": config.teacher_finalizer[
            "config_schema_sha256"
        ],
        "finalizer_implementation_sha256": config.teacher_finalizer[
            "implementation_sha256"
        ],
        "record_schema_sha256": config.teacher_finalizer["record_schema_sha256"],
        "candidate_manifest_schema_sha256": config.teacher_finalizer[
            "manifest_schema_sha256"
        ],
        "candidate_build_receipt_schema_sha256": config.teacher_finalizer[
            "build_receipt_schema_sha256"
        ],
    }
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "audit_status": "passed",
        "candidate_lifecycle_status": manifest["lifecycle_status"],
        "namespace": NAMESPACE,
        "reviewer_separation": {
            "implementer_id": implementer_id,
            "reviewer_id": reviewer_id,
            "different_reviewer": True,
        },
        "findings": {"p0": 0, "p1": 0, "p2": 0},
        "candidate_identity": {
            "artifact_root": audit.root.relative_to(batch.REPO_ROOT).as_posix(),
            "physical_file_count": len(audit.snapshots),
            "physical_tree_sha256": audit.physical_tree_sha256,
            "manifest_sha256": audit.snapshots["manifest.json"].sha256,
            "manifest_sidecar_sha256": audit.snapshots["manifest.json.sha256"].sha256,
            "build_receipt_sha256": audit.snapshots["build_receipt.json"].sha256,
            "build_receipt_sidecar_sha256": audit.snapshots[
                "build_receipt.json.sha256"
            ].sha256,
            "output_inventory_sha256": audit.snapshots["output_inventory.json"].sha256,
            "output_inventory_sidecar_sha256": audit.snapshots[
                "output_inventory.json.sha256"
            ].sha256,
            "release_request_sha256": audit.snapshots[
                "release_attestation.request.json"
            ].sha256,
            "release_request_sidecar_sha256": audit.snapshots[
                "release_attestation.request.json.sha256"
            ].sha256,
            "payload_tree_sha256": audit.output_inventory["payload_tree_sha256"],
            "all_sidecars_valid": True,
            "all_files_below_50_mib": True,
        },
        "source_release": source_release,
        "implementation_binding": implementation_binding,
        "record_audit": record_audit,
        "runtime_receipt": {
            "runtime_hmac_sha256": receipt["runtime_hmac_sha256"],
            "phase_receipt_inventory_sha256": receipt["phase_receipt_inventory_sha256"],
            "wal_chain_tip_sha256": receipt["wal_chain_tip_sha256"],
            "candidate_receipt_claims_same_process_hmac_authenticated": True,
            "runtime_hmac_externally_reverified": False,
            "external_hmac_key_available": False,
            "runtime_hmac_key_persisted": False,
            "candidate_receipt_bytes_bound": True,
        },
        "integrity": {
            "single_bytes_snapshot": True,
            "terminal_toctou_recheck": True,
            "raw_path_reparse_rejected": True,
            "candidate_immutable": True,
            "candidate_directly_consumable": False,
            "create_once_release": True,
            "atomic_directory_no_replace": True,
            "recursive_cleanup_used": False,
            "release_pair_sidecars_valid": True,
        },
        "resource_counters": RESOURCE_COUNTERS,
        "claims": {
            "candidate": False,
            "final": True,
            "independent_release_review": "passed",
            "consumer_binding_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "quality_validated": False,
            "generalization_validated": False,
        },
    }
    _validate(
        _schema(audit.schema_snapshots["attestation"]),
        attestation,
        code="release_attestation_schema_rejected",
    )
    attestation_raw = _canonical_bytes(attestation, newline=True)
    attestation_sidecar = _sidecar(
        "independent_release_attestation.json",
        attestation_raw,
    )
    candidate_identity = {
        "artifact_root": audit.root.relative_to(batch.REPO_ROOT).as_posix(),
        "manifest_sha256": audit.snapshots["manifest.json"].sha256,
        "build_receipt_sha256": audit.snapshots["build_receipt.json"].sha256,
        "output_inventory_sha256": audit.snapshots["output_inventory.json"].sha256,
        "payload_tree_sha256": audit.output_inventory["payload_tree_sha256"],
        "physical_tree_sha256": audit.physical_tree_sha256,
    }
    final_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "lifecycle_status": "independently_released_consumer_binding_pending",
        "candidate": candidate_identity,
        "attestation": {
            "path": "independent_release_attestation.json",
            "schema_path": (
                "configs/data/"
                "gemma3_chat_unbalanced_v2_teacher_final_release_"
                "attestation_consumer_rollover_v3.schema.json"
            ),
            "schema_sha256": audit.schema_snapshots["attestation"].sha256,
            "sha256": _sha256(attestation_raw),
            "sidecar_sha256": _sha256(attestation_sidecar),
        },
        "binding_contract": {
            "path": "teacher_alignment_binding.v2.json",
            "schema_version": BINDING_SCHEMA_VERSION,
            "schema_path": (
                "configs/data/"
                "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
                "consumer_rollover_v2.schema.json"
            ),
            "schema_sha256": audit.schema_snapshots["binding"].sha256,
            "status": "released_pending_consumer_acceptance",
        },
        "source": dict(manifest["source"]),
        "controller": {
            key: implementation_binding[key]
            for key in (
                "integrated_controller_config_sha256",
                "integrated_controller_implementation_sha256",
                "base_controller_config_sha256",
                "base_controller_implementation_sha256",
                "finalizer_config_sha256",
                "finalizer_implementation_sha256",
                "release_implementation_sha256",
            )
        },
        "record_schema": {
            **dict(manifest["record_schema"]),
            "all_rows_valid": True,
        },
        "counts": release_counts,
        "logical_identity": dict(manifest["logical_identity"]),
        "shards": [dict(item) for item in manifest["shards"]],
        "claims": {
            "candidate": False,
            "final": True,
            "consumer_binding_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "quality_validated": False,
            "generalization_validated": False,
        },
    }
    _validate(
        _schema(audit.schema_snapshots["manifest"]),
        final_manifest,
        code="release_manifest_schema_rejected",
    )
    final_manifest_raw = _canonical_bytes(final_manifest, newline=True)
    final_manifest_sidecar = _sidecar("final_manifest.json", final_manifest_raw)
    final_manifest_sha = _sha256(final_manifest_raw)
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "status": "released_pending_consumer_acceptance",
        "candidate": candidate_identity,
        "release": {
            "release_root": release_root.relative_to(batch.REPO_ROOT).as_posix(),
            "final_manifest_path": "final_manifest.json",
            "final_manifest_sha256": final_manifest_sha,
            "final_manifest_sidecar_sha256": _sha256(final_manifest_sidecar),
            "attestation_path": "independent_release_attestation.json",
            "attestation_sha256": _sha256(attestation_raw),
            "attestation_sidecar_sha256": _sha256(attestation_sidecar),
            "release_implementation_sha256": (audit.implementation_snapshot.sha256),
            "attestation_schema_sha256": audit.schema_snapshots["attestation"].sha256,
            "final_manifest_schema_sha256": audit.schema_snapshots["manifest"].sha256,
            "binding_schema_sha256": audit.schema_snapshots["binding"].sha256,
        },
        "source": dict(manifest["source"]),
        "consumer_prerequisite": {
            "consumer_commit": manifest["consumer"]["commit"],
            "source_binding_sha256": manifest["consumer"]["source_binding_sha256"],
            "preflight_receipt_sha256": manifest["consumer"][
                "preflight_receipt_sha256"
            ],
            "preflight_receipt_sidecar_sha256": manifest["consumer"][
                "preflight_receipt_sidecar_sha256"
            ],
            "shared_schema_sha256": manifest["consumer"]["shared_schema_sha256"],
            "normalized_binding_sha256": manifest["consumer"][
                "normalized_binding_sha256"
            ],
        },
        "record_contract": {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
            ),
            "schema_path": manifest["record_schema"]["path"],
            "required_fields": [
                "schema_version",
                "namespace",
                "record_id_sha256",
                "source_content_sha256",
                "source_serialization_identity_sha256",
                "task_bundle_sha256",
                "source_asset",
                "decision",
                "teacher_target",
                "teacher_target_sha256",
            ],
            "record_id_hash_domain": "sha256(utf8(source_record_id))",
            "source_content_hash_domain": ("canonical_sha256(source_messages)"),
            "source_serialization_hash_domain": (
                "authenticated_source_serialization_identity"
            ),
            "teacher_target_hash_domain": "sha256(utf8(teacher_target))",
            "candidate_manifest_direct_consumption": False,
        },
        "counts": {
            "records": record_audit["records"],
            "main_records": record_audit["main_records"],
            "router_records": record_audit["router_records"],
            "source_assets": record_audit["source_asset_counts"],
            "train_identity_bundles": record_audit["train_identity_bundles"],
            "train_identity_records": record_audit["train_identity_records"],
        },
        "logical_identity": dict(manifest["logical_identity"]),
        "ordered_shards": [dict(item) for item in manifest["shards"]],
        "claims": {
            "candidate": False,
            "final": True,
            "consumer_acceptance": "pending",
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "quality_validated": False,
            "generalization_validated": False,
        },
    }
    _validate(
        _schema(audit.schema_snapshots["binding"]),
        binding,
        code="release_binding_schema_rejected",
    )
    binding_raw = _canonical_bytes(binding, newline=True)
    documents = {
        "independent_release_attestation.json": attestation_raw,
        "independent_release_attestation.json.sha256": attestation_sidecar,
        "final_manifest.json": final_manifest_raw,
        "final_manifest.json.sha256": final_manifest_sidecar,
        "teacher_alignment_binding.v2.json": binding_raw,
        "teacher_alignment_binding.v2.json.sha256": _sidecar(
            "teacher_alignment_binding.v2.json",
            binding_raw,
        ),
    }
    return documents, binding


def _create_release_parent() -> tuple[Path, tuple[int, int]]:
    root = batch.REPO_ROOT.absolute()
    parent = RELEASE_PARENT.absolute()
    try:
        parent.relative_to(root)
    except ValueError:
        _fail("release_output_parent_escape")
    current = root
    for part in parent.relative_to(root).parts:
        next_path = current / part
        if os.path.lexists(next_path):
            _reject_reparse_chain(next_path, stop=root)
            _directory_identity(next_path, code="release_output_parent_invalid")
        else:
            try:
                next_path.mkdir()
            except OSError as error:
                raise IndependentReleaseError(
                    "release_output_parent_create_failed"
                ) from error
        current = next_path
    return parent, _directory_identity(
        parent,
        code="release_output_parent_invalid",
    )


def _recheck_release_tree(
    root: Path,
    expected: Mapping[str, Snapshot],
    *,
    code: str,
) -> None:
    try:
        names = {item.name for item in root.iterdir()}
    except OSError as error:
        raise IndependentReleaseError(code) from error
    if names != set(expected):
        _fail(code)
    for name, snapshot in expected.items():
        observed = _snapshot(
            root / name,
            expected_sha256=snapshot.sha256,
            code=code,
        )
        if observed.identity != snapshot.identity or observed.raw != snapshot.raw:
            _fail(code)


def publish_release(
    candidate: Path,
    *,
    implementer_id: str,
    reviewer_id: str,
    integrated_config_path: Path = (
        batch.REPO_ROOT
        / "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_consumer_rollover_v2.json"
    ),
    before_rename: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Audit and atomically publish one independent, non-authorizing release."""

    audit = audit_candidate(
        candidate,
        implementer_id=implementer_id,
        reviewer_id=reviewer_id,
        integrated_config_path=integrated_config_path,
    )
    parent, parent_identity = _create_release_parent()
    release_root = parent / f"release-{audit.physical_tree_sha256}"
    staging = parent / (
        f".release-{audit.physical_tree_sha256}.{secrets.token_hex(8)}.staging"
    )
    if os.path.lexists(release_root) or os.path.lexists(staging):
        _fail("release_create_once_collision")
    try:
        staging.mkdir()
    except OSError as error:
        raise IndependentReleaseError("release_staging_create_failed") from error
    staging_identity = _directory_identity(
        staging,
        code="release_staging_identity_invalid",
    )
    documents, binding = _release_documents(
        audit,
        implementer_id=implementer_id,
        reviewer_id=reviewer_id,
        release_root=release_root,
    )
    written: dict[str, Snapshot] = {}
    try:
        for name, raw in documents.items():
            path = staging / name
            _write_exclusive(path, raw)
            written[name] = _snapshot(
                path,
                expected_sha256=_sha256(raw),
                code="release_staging_file_drift",
            )

        def authenticate_publish_boundary() -> None:
            before_rename()
            if (
                _directory_identity(
                    parent,
                    code="release_output_parent_identity_drift",
                )
                != parent_identity
                or _directory_identity(
                    staging,
                    code="release_staging_identity_drift",
                )
                != staging_identity
            ):
                _fail("release_publish_directory_identity_drift")
            for snapshot in (
                *audit.snapshots.values(),
                *audit.schema_snapshots.values(),
                audit.implementation_snapshot,
                audit.source_overlay_snapshot,
                *written.values(),
            ):
                _recheck(snapshot, code="release_publish_boundary_toctou")
            _recheck_candidate_tree(
                audit.root,
                audit.snapshots,
                code="release_publish_candidate_tree_toctou",
            )
            _recheck_release_tree(
                staging,
                written,
                code="release_publish_staging_tree_toctou",
            )
            integrated_v2._recheck_integrated(audit.integrated)

        try:
            _rename_noreplace(
                staging,
                release_root,
                before_rename=authenticate_publish_boundary,
            )
        except FileExistsError as error:
            raise IndependentReleaseError("release_create_once_collision") from error
        if (
            _directory_identity(
                release_root,
                code="release_published_root_identity_drift",
            )
            != staging_identity
        ):
            _fail("release_published_root_identity_drift")
        published: dict[str, Snapshot] = {}
        for name, snapshot in written.items():
            observed = _snapshot(
                release_root / name,
                expected_sha256=snapshot.sha256,
                code="release_published_file_drift",
            )
            if observed.identity != snapshot.identity:
                _fail("release_published_file_identity_drift")
            published[name] = observed
        for snapshot in (
            *audit.snapshots.values(),
            *audit.schema_snapshots.values(),
            audit.implementation_snapshot,
            audit.source_overlay_snapshot,
        ):
            _recheck(snapshot, code="release_terminal_toctou")
        _recheck_candidate_tree(
            audit.root,
            audit.snapshots,
            code="release_terminal_candidate_tree_toctou",
        )
        _recheck_release_tree(
            release_root,
            published,
            code="release_terminal_published_tree_toctou",
        )
        integrated_v2._recheck_integrated(audit.integrated)
        return {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-final-"
                "independent-release-status.consumer-rollover-v3"
            ),
            "state": "released_pending_consumer_acceptance",
            "release_root": release_root.relative_to(batch.REPO_ROOT).as_posix(),
            "candidate_manifest_sha256": audit.snapshots["manifest.json"].sha256,
            "candidate_physical_tree_sha256": audit.physical_tree_sha256,
            "attestation_sha256": published[
                "independent_release_attestation.json"
            ].sha256,
            "final_manifest_sha256": published["final_manifest.json"].sha256,
            "binding_sha256": published["teacher_alignment_binding.v2.json"].sha256,
            "binding_schema_sha256": audit.schema_snapshots["binding"].sha256,
            "records": 3520,
            "files": len(published),
            "binding_status": binding["status"],
            "candidate_unchanged": True,
            "final": True,
            "consumer_acceptance": "pending",
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            **RESOURCE_COUNTERS,
        }
    except BaseException:
        # Preserve the owned staging directory for forensic review.  Recursive
        # cleanup is intentionally forbidden at this security boundary.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Independently audit and release a Teacher FINAL candidate")
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--implementer-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument(
        "--integrated-config",
        type=Path,
        default=(
            batch.REPO_ROOT
            / "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_consumer_rollover_v2.json"
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--publish", action="store_true")
    return parser


def _reason(error: BaseException) -> str:
    if isinstance(error, IndependentReleaseError):
        return str(error)
    if isinstance(error, batch.AdapterError):
        return error.reason_code
    return f"release_{type(error).__name__.casefold()}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_only:
            audit = audit_candidate(
                args.candidate,
                implementer_id=args.implementer_id,
                reviewer_id=args.reviewer_id,
                integrated_config_path=args.integrated_config,
            )
            result = {
                "schema_version": (
                    "anchor.gemma3-chat-unbalanced-v2-teacher-final-"
                    "independent-release-status.consumer-rollover-v3"
                ),
                "state": "candidate_validated_release_not_published",
                "candidate_manifest_sha256": audit.snapshots["manifest.json"].sha256,
                "candidate_physical_tree_sha256": audit.physical_tree_sha256,
                "records": len(audit.records),
                "provider_requests": 0,
                "network_requests": 0,
                "model_loads": 0,
                "gpu_requests": 0,
                "training_authorized": False,
                "formal_training_authorized": False,
                "live_authorized": False,
            }
        else:
            result = publish_release(
                args.candidate,
                implementer_id=args.implementer_id,
                reviewer_id=args.reviewer_id,
                integrated_config_path=args.integrated_config,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as error:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "anchor.gemma3-chat-unbalanced-v2-teacher-final-"
                        "independent-release-status.consumer-rollover-v3"
                    ),
                    "state": "blocked",
                    "reason_code": _reason(error),
                    "provider_requests": 0,
                    "network_requests": 0,
                    "model_loads": 0,
                    "gpu_requests": 0,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "live_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
