"""Fail-closed producer freeze layer for TaskBoard training releases.

The module consumes metadata-bound artifacts only.  It never emits projected
records or held-out bodies; release provenance remains in the outer sidecar.
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
from typing import Any, Mapping, Sequence
from uuid import uuid4

from anchor_mvp.swebench import taskboard_projector as _taskboard_projector


SNAPSHOT_SCHEMA = "anchor.training-snapshot.v2"
SPLIT_SCHEMA = "anchor.formal-v3-gold-splits.v1"
PROJECTOR_SCHEMA = "anchor.swebench-taskboard-projector-manifest.v2"
SEGMENT_PLAN_SCHEMA = "anchor.hierarchical-task-kv-segment-plan.v1"
SEGMENT_EXECUTION_MODE = "decoupled_frozen_prefix_producer_required"
SEGMENT_REUSE_SCOPE = "identical_ordered_prefix_lineage_only"
SEGMENT_CACHE_IDENTITY_FIELDS = (
    "model_architecture_sha256",
    "tokenizer_sha256",
    "token_order_sha256",
    "position_ids_sha256",
    "rope_config_sha256",
    "kv_producing_weights_sha256",
    "prefix_lineage_sha256",
)
GENERIC_SCHEMA = "anchor.generic-train-execution-contract.v1"
CONSUMER_SCHEMA = "anchor.swebench-training-consumer-interface.v2"
SOURCE_SCHEMA = "anchor.swebench-source-disjoint-manifest.v2"
RELEASE_SCHEMA = "anchor.generic-train-release-lock.v2"
EXECUTION_LOCK_SCHEMA = "anchor.swebench-execution-lock.v1"
PREFLIGHT_SCHEMA = "anchor.swebench-ccswitch-preflight.v1"
ATTESTATION_SCHEMA = "anchor.multilang-execution-attestation.v1"
SOURCE_BANK_SCHEMA = "anchor.swebench-publication-manifest.v1"

_RESEARCH_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "research"
_PROJECTOR_CONFIG_PATH = _RESEARCH_CONFIG_ROOT / "swebench_taskboard_projector_v2.yaml"
_PROJECTOR_SIDECAR_SCHEMA_PATH = (
    _RESEARCH_CONFIG_ROOT / "taskboard_projector_sidecar.schema.json"
)
_PROJECTOR_SEGMENT_SCHEMA_PATH = (
    _RESEARCH_CONFIG_ROOT / "hierarchical_task_kv_segment_plan.schema.json"
)
_PROJECTOR_MANIFEST_SCHEMA_PATH = (
    _RESEARCH_CONFIG_ROOT / "taskboard_projector_manifest.schema.json"
)


def _hierarchical_task_kv_contract(segment_schema_sha: str) -> dict[str, Any]:
    return {
        "segment_plan_schema_version": SEGMENT_PLAN_SCHEMA,
        "segment_plan_schema_sha256": segment_schema_sha,
        "segment_plan_location": "outer_sidecar.segment_plan",
        "architecture": "hierarchical_task_kv",
        "execution_mode": SEGMENT_EXECUTION_MODE,
        "materialization": "metadata_only_no_tensor_or_kv",
        "tensors_emitted": False,
        "kv_payloads_emitted": False,
        "shared_prefix_membership": (
            "strict_all_five_role_visibility_intersection"
        ),
        "ordered_prefix_chain": True,
        "independent_segment_concatenation_allowed": False,
        "exact_reuse_scope": SEGMENT_REUSE_SCOPE,
        "shared_then_mask_allowed": False,
        "forbidden_current_future_preinsert_allowed": False,
        "cache_identity_required_exact_match_fields": list(
            SEGMENT_CACHE_IDENTITY_FIELDS
        ),
        "cache_identity_mismatch_result": "cache_incompatible",
        "cache_identity_unknown_result": "cache_incompatible",
        "target_delta_initial_cache_scope": "expert_private_delta",
        "target_delta_promotion_requires": (
            "explicit_committed_and_causally_visible_downstream"
        ),
        "target_delta_promoted_cache_scope": (
            "downstream_task_shared_immutable"
        ),
        "current_target_segment_emitted": False,
        "q_specialization_alone_sufficient_for_exact_reuse": False,
        "naive_in_stack_q_lora_exact_reuse_allowed": False,
        "full_generation_kv_shared_claimed": False,
        "token_level_moe_claimed": False,
    }

FIXED_FILES = (
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HELDOUT_FORBIDDEN_KEYS = {
    "body",
    "bodies",
    "content",
    "contents",
    "files",
    "path",
    "paths",
    "records",
    "samples",
    "cases",
    "case_ids",
    "problem_statement",
    "prompt",
    "prompts",
    "labels",
}


class TrainingReleaseError(RuntimeError):
    """A content-free, stable failure code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise TrainingReleaseError(code)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _BytesSnapshot:
    data: bytes
    sha256: str
    size: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_bytes_snapshot(path: Path, code: str) -> _BytesSnapshot:
    """Read a regular file once and bind the bytes to its opened inode."""

    if not path.is_file() or path.is_symlink():
        _fail(code)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise TrainingReleaseError(code) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(data=data, sha256=_sha256(data), size=len(data))


def _json(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingReleaseError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _jsonl_record_count(snapshot: _BytesSnapshot, code: str) -> int:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingReleaseError(code) from exc
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingReleaseError(code) from exc
        if not isinstance(value, Mapping):
            _fail(code)
        count += 1
    return count


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], code: str) -> None:
    if set(value) != keys:
        _fail(code)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return False
    return ".." not in Path(value.replace("\\", "/")).parts


def _safe_file(root: Path, relative: object, code: str) -> Path:
    if not _safe_relative(relative):
        _fail(code)
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        _fail(code)
    if not path.is_file() or path.is_symlink():
        _fail(code)
    return path


def _strict_sidecar(path: Path, manifest: _BytesSnapshot, code: str) -> _BytesSnapshot:
    sidecar = _read_bytes_snapshot(path, code)
    expected = f"{manifest.sha256}  manifest.json\n".encode("ascii")
    if sidecar.data != expected:
        _fail(code)
    return sidecar


def _load_artifact(
    directory: str | Path,
    expected_sha256: str,
    schema: str,
    code: str,
) -> tuple[Path, Mapping[str, Any], dict[str, tuple[Path, _BytesSnapshot]]]:
    root = Path(directory).expanduser().resolve()
    if not _is_sha256(expected_sha256) or not root.is_dir() or root.is_symlink():
        _fail(code)
    manifest_path = root / "manifest.json"
    manifest_snapshot = _read_bytes_snapshot(manifest_path, code)
    if manifest_snapshot.sha256 != expected_sha256:
        _fail(code)
    sidecar_path = root / "manifest.json.sha256"
    sidecar_snapshot = _strict_sidecar(sidecar_path, manifest_snapshot, code)
    value = _json(manifest_snapshot, code)
    if value.get("schema_version") != schema:
        _fail(code)
    return root, value, {
        f"{code}:manifest": (manifest_path, manifest_snapshot),
        f"{code}:sidecar": (sidecar_path, sidecar_snapshot),
    }


def _load_expected_json(
    path: str | Path, expected_sha256: str, schema: str, code: str
) -> tuple[Mapping[str, Any], dict[str, tuple[Path, _BytesSnapshot]]]:
    source = Path(path).expanduser().resolve()
    if not _is_sha256(expected_sha256):
        _fail(code)
    snapshot = _read_bytes_snapshot(source, code)
    if snapshot.sha256 != expected_sha256:
        _fail(code)
    value = _json(snapshot, code)
    if value.get("schema_version") != schema:
        _fail(code)
    return value, {code: (source, snapshot)}


def _load_expected_file(
    path: str | Path, expected_sha256: str, code: str
) -> tuple[_BytesSnapshot, dict[str, tuple[Path, _BytesSnapshot]]]:
    source = Path(path).expanduser().resolve()
    if not _is_sha256(expected_sha256):
        _fail(code)
    snapshot = _read_bytes_snapshot(source, code)
    if snapshot.sha256 != expected_sha256:
        _fail(code)
    return snapshot, {code: (source, snapshot)}


def _verify_inventory(inventory: Mapping[str, tuple[Path, _BytesSnapshot]]) -> None:
    for path, expected in inventory.values():
        current = _read_bytes_snapshot(path, "training_release_input_changed")
        if current.sha256 != expected.sha256 or current.size != expected.size:
            _fail("training_release_input_changed")


def _check_output(output: Path, inputs: Sequence[Path]) -> None:
    if output.exists() or output.is_symlink():
        _fail("training_release_output_exists_or_overlaps_input")
    for source in inputs:
        source = source.resolve()
        if output == source:
            _fail("training_release_output_exists_or_overlaps_input")
        try:
            output.relative_to(source)
        except ValueError:
            pass
        else:
            _fail("training_release_output_exists_or_overlaps_input")
        try:
            source.relative_to(output)
        except ValueError:
            pass
        else:
            _fail("training_release_output_exists_or_overlaps_input")


def _publish(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    inventory: Mapping[str, tuple[Path, _BytesSnapshot]],
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    _check_output(output, [path for path, _snapshot in inventory.values()])
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(encoded)
        digest = _sha256(encoded)
        sidecar_bytes = f"{digest}  manifest.json\n".encode("ascii")
        sidecar_path = temporary / "manifest.json.sha256"
        sidecar_path.write_bytes(sidecar_bytes)
        manifest_snapshot = _read_bytes_snapshot(
            manifest_path, "training_release_output_invalid"
        )
        sidecar_snapshot = _strict_sidecar(
            sidecar_path, manifest_snapshot, "training_release_output_invalid"
        )
        if (
            manifest_snapshot.data != encoded
            or manifest_snapshot.sha256 != digest
            or sidecar_snapshot.data != sidecar_bytes
            or _json(manifest_snapshot, "training_release_output_invalid") != payload
        ):
            _fail("training_release_output_invalid")
        _verify_inventory(inventory)
        final_manifest_snapshot = _read_bytes_snapshot(
            manifest_path, "training_release_output_invalid"
        )
        final_sidecar_snapshot = _strict_sidecar(
            sidecar_path,
            final_manifest_snapshot,
            "training_release_output_invalid",
        )
        if (
            final_manifest_snapshot.data != encoded
            or final_manifest_snapshot.sha256 != digest
            or final_sidecar_snapshot.data != sidecar_bytes
            or _json(final_manifest_snapshot, "training_release_output_invalid")
            != payload
        ):
            _fail("training_release_output_invalid")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return {
            "schema_version": str(payload["schema_version"]),
            "status": "published",
            "output_dir": str(output),
            "manifest_sha256": digest,
            "heldout_content_read": False,
        }
    except TrainingReleaseError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise TrainingReleaseError("training_release_internal_error") from exc


def _load_projector_policy(
    inventory: dict[str, tuple[Path, _BytesSnapshot]],
) -> _taskboard_projector.TaskBoardProjectorConfig:
    paths = {
        "config": _PROJECTOR_CONFIG_PATH,
        "sidecar-schema": _PROJECTOR_SIDECAR_SCHEMA_PATH,
        "segment-schema": _PROJECTOR_SEGMENT_SCHEMA_PATH,
        "manifest-schema": _PROJECTOR_MANIFEST_SCHEMA_PATH,
    }
    snapshots = {
        label: _read_bytes_snapshot(path, "projector_policy_invalid")
        for label, path in paths.items()
    }
    inventory.update(
        {
            f"projector-policy:{label}": (paths[label], snapshot)
            for label, snapshot in snapshots.items()
        }
    )
    try:
        config = _taskboard_projector.TaskBoardProjectorConfig.load(
            _PROJECTOR_CONFIG_PATH
        )
    except _taskboard_projector.TaskBoardProjectorError as exc:
        raise TrainingReleaseError("projector_policy_invalid") from exc
    if (
        config.sha256 != snapshots["config"].sha256
        or config.sidecar_schema_sha256 != snapshots["sidecar-schema"].sha256
        or config.segment_plan_schema_sha256
        != snapshots["segment-schema"].sha256
        or config.manifest_schema_sha256 != snapshots["manifest-schema"].sha256
    ):
        _fail("projector_policy_changed_during_read")
    return config


def _projector_counts(
    all_rows: Mapping[tuple[str, str], list[dict[str, Any]]],
    projected_task_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    rows = [
        row
        for _path, split, variant in FIXED_FILES
        for row in all_rows[(split, variant)]
    ]
    segment_references = [
        segment
        for row in rows
        for segment in row["segment_plan"]["segments"]
    ]
    unique_segments: dict[str, Mapping[str, Any]] = {}
    for segment in segment_references:
        item = _mapping(segment, "projector_counts_invalid")
        segment_id = item.get("segment_id")
        if not isinstance(segment_id, str):
            _fail("projector_counts_invalid")
        prior = unique_segments.setdefault(segment_id, item)
        if prior != item:
            _fail("projector_segment_identity_fork")
    unique_scope_counts = Counter(
        str(segment["cache_scope"]) for segment in unique_segments.values()
    )
    split_counts = Counter(str(row["split"]) for row in rows)
    variant_counts = Counter(str(row["variant"]) for row in rows)
    stage_counts = Counter(str(row["stage"]) for row in rows)
    expert_counts = Counter(str(row["expert"]) for row in rows)
    language_counts = Counter(
        str(row["training_record"]["language"]) for row in rows
    )
    task_ids = projected_task_ids["train"] | projected_task_ids["calibration"]
    return {
        "total": len(rows),
        "unique_task_bundles": len(
            {str(row["task_bundle_sha256"]) for row in rows}
        ),
        "task_ids_sha256": _sha256(
            "\n".join(sorted(task_ids)).encode("utf-8")
        ),
        "segment_references": len(segment_references),
        "unique_segments": len(unique_segments),
        "unique_segments_by_cache_scope": {
            scope: unique_scope_counts.get(scope, 0)
            for scope in (
                "task_shared_prefix",
                "downstream_task_shared_immutable",
                "expert_private_delta",
            )
        },
        "by_split": dict(sorted(split_counts.items())),
        "by_variant": dict(sorted(variant_counts.items())),
        "by_stage": dict(sorted(stage_counts.items())),
        "by_expert": dict(sorted(expert_counts.items())),
        "by_language": {
            language: language_counts.get(language, 0)
            for language in ("en", "zh-CN")
        },
    }


def _validate_projector(
    root: Path, value: Mapping[str, Any], inventory: dict[str, tuple[Path, _BytesSnapshot]]
) -> tuple[list[dict[str, Any]], str, str, str]:
    _exact_keys(
        value,
        {
            "schema_version",
            "input",
            "producer",
            "files",
            "counts",
            "hierarchical_task_kv",
            "split_group_key",
            "task_id_cross_binding_key",
            "all_five_role_views_same_split",
            "canonical_gold_written",
            "provider_requests",
            "heldout_content_read",
            "heldout_content_emitted",
            "split_preserved",
            "augmentation_applied_after_split",
            "claim_scope",
        },
        "projector_manifest_invalid",
    )
    config = _load_projector_policy(inventory)
    producer = _mapping(value.get("producer"), "projector_manifest_invalid")
    expected_producer = {
        "name": "anchor.swebench-taskboard-projector",
        "projector_version": config.projector_version,
        "config_sha256": config.sha256,
        "sidecar_schema_sha256": config.sidecar_schema_sha256,
        "segment_plan_schema_version": config.segment_plan_schema,
        "segment_plan_schema_sha256": config.segment_plan_schema_sha256,
        "manifest_schema_sha256": config.manifest_schema_sha256,
        "record_schema_version": config.record_schema,
    }
    if dict(producer) != expected_producer:
        _fail("projector_policy_binding_invalid")

    projector_input = _mapping(value.get("input"), "projector_manifest_invalid")
    _exact_keys(
        projector_input,
        {
            "snapshot_schema_version",
            "snapshot_sha256",
            "snapshot_manifest_path",
            "snapshot_manifest_sha256",
            "snapshot_sha256_sidecar_path",
            "snapshot_sha256_sidecar_sha256",
            "splits",
        },
        "projector_manifest_invalid",
    )
    if (
        projector_input.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
        or any(
            not _is_sha256(projector_input.get(field))
            for field in (
                "snapshot_sha256",
                "snapshot_manifest_sha256",
                "snapshot_sha256_sidecar_sha256",
            )
        )
        or projector_input.get("snapshot_manifest_path") != "manifest.json"
        or projector_input.get("snapshot_sha256_sidecar_path")
        != "manifest.json.sha256"
        or projector_input.get("splits") != ["train", "calibration"]
    ):
        _fail("projector_manifest_invalid")

    task_kv = _mapping(
        value.get("hierarchical_task_kv"), "projector_manifest_invalid"
    )
    _exact_keys(
        task_kv,
        {
            "segment_plan_schema_version",
            "segment_plan_location",
            "architecture",
            "execution_mode",
            "materialization",
            "tensors_emitted",
            "kv_payloads_emitted",
            "shared_prefix_membership",
            "ordered_prefix_chain",
            "independent_segment_concatenation_allowed",
            "exact_reuse_scope",
            "shared_then_mask_allowed",
            "forbidden_current_future_preinsert_allowed",
            "cache_identity_required_exact_match_fields",
            "cache_identity_mismatch_result",
            "cache_identity_unknown_result",
            "q_specialization_alone_sufficient_for_exact_reuse",
            "naive_in_stack_q_lora_exact_reuse_allowed",
            "full_generation_kv_shared_claimed",
            "token_level_moe_claimed",
        },
        "projector_manifest_invalid",
    )
    if (
        task_kv.get("segment_plan_schema_version") != SEGMENT_PLAN_SCHEMA
        or task_kv.get("segment_plan_location") != "outer_sidecar.segment_plan"
        or task_kv.get("architecture") != "hierarchical_task_kv"
        or task_kv.get("execution_mode") != SEGMENT_EXECUTION_MODE
        or task_kv.get("materialization") != "metadata_only_no_tensor_or_kv"
        or task_kv.get("tensors_emitted") is not False
        or task_kv.get("kv_payloads_emitted") is not False
        or task_kv.get("shared_prefix_membership")
        != "strict_all_five_role_visibility_intersection"
        or task_kv.get("ordered_prefix_chain") is not True
        or task_kv.get("independent_segment_concatenation_allowed") is not False
        or task_kv.get("exact_reuse_scope") != SEGMENT_REUSE_SCOPE
        or task_kv.get("shared_then_mask_allowed") is not False
        or task_kv.get("forbidden_current_future_preinsert_allowed") is not False
        or task_kv.get("cache_identity_required_exact_match_fields")
        != list(SEGMENT_CACHE_IDENTITY_FIELDS)
        or task_kv.get("cache_identity_mismatch_result") != "cache_incompatible"
        or task_kv.get("cache_identity_unknown_result") != "cache_incompatible"
        or task_kv.get("q_specialization_alone_sufficient_for_exact_reuse")
        is not False
        or task_kv.get("naive_in_stack_q_lora_exact_reuse_allowed") is not False
        or task_kv.get("full_generation_kv_shared_claimed") is not False
        or task_kv.get("token_level_moe_claimed") is not False
        or value.get("split_group_key") != "task_bundle_sha256"
        or value.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or value.get("all_five_role_views_same_split") is not True
        or value.get("canonical_gold_written") is not False
        or value.get("provider_requests") != 0
        or value.get("heldout_content_read") is not False
        or value.get("heldout_content_emitted") is not False
        or value.get("split_preserved") is not True
        or value.get("augmentation_applied_after_split") is not True
        or value.get("claim_scope") != "research_proxy_only"
    ):
        _fail("projector_manifest_invalid")

    raw_files = value.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(FIXED_FILES):
        _fail("projector_manifest_invalid")
    files: list[dict[str, Any]] = []
    all_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_wrapper_bindings = {
        "projector_version": config.projector_version,
        "config_sha256": config.sha256,
        "sidecar_schema_sha256": config.sidecar_schema_sha256,
        "segment_plan_schema_sha256": config.segment_plan_schema_sha256,
        "source_snapshot_sha256": projector_input["snapshot_sha256"],
        "source_snapshot_manifest_sha256": projector_input[
            "snapshot_manifest_sha256"
        ],
    }
    for index, (path_value, split, variant) in enumerate(FIXED_FILES):
        item = _mapping(raw_files[index], "projector_manifest_invalid")
        _exact_keys(
            item,
            {"path", "sha256", "bytes", "records", "split", "variant"},
            "projector_manifest_invalid",
        )
        if (
            item.get("path") != path_value
            or item.get("split") != split
            or item.get("variant") != variant
            or not _is_sha256(item.get("sha256"))
            or not _positive_int(item.get("bytes"))
            or not _positive_int(item.get("records"))
        ):
            _fail("projector_manifest_invalid")
        path = _safe_file(root, path_value, "projector_file_invalid")
        snapshot = _read_bytes_snapshot(path, "projector_file_invalid")
        try:
            rows, record_count = _taskboard_projector._jsonl_from_snapshot(
                snapshot, "projector_file_invalid"
            )
        except _taskboard_projector.TaskBoardProjectorError as exc:
            raise TrainingReleaseError("projector_file_invalid") from exc
        if (
            snapshot.sha256 != item["sha256"]
            or snapshot.size != item["bytes"]
            or record_count != item["records"]
            or not rows
        ):
            _fail("projector_file_invalid")
        identifiers: set[str] = set()
        try:
            for row in rows:
                _taskboard_projector._validate_sidecar(row, expected_split=split)
                identifier = row.get("id")
                if (
                    not isinstance(identifier, str)
                    or identifier in identifiers
                    or row.get("variant") != variant
                    or any(
                        row.get(field) != expected
                        for field, expected in expected_wrapper_bindings.items()
                    )
                ):
                    _fail("projector_semantic_invalid")
                identifiers.add(identifier)
        except _taskboard_projector.TaskBoardProjectorError as exc:
            raise TrainingReleaseError("projector_semantic_invalid") from exc
        inventory[f"projector-file:{path_value}"] = (path, snapshot)
        files.append(dict(item))
        all_rows[(split, variant)] = rows

    try:
        _taskboard_projector._verify_clean_noisy_pairs(
            all_rows[("train", "clean")], all_rows[("train", "noisy")]
        )
        expected_bundle_task_ids: dict[str, dict[str, str]] = {
            "train": {},
            "calibration": {},
        }
        for split in expected_bundle_task_ids:
            for row in all_rows[(split, "clean")]:
                bundle = str(row["task_bundle_sha256"])
                task_id = str(row["training_record"]["task_board"]["task_id"])
                prior = expected_bundle_task_ids[split].setdefault(bundle, task_id)
                if prior != task_id:
                    _fail("projector_semantic_invalid")
        projected_task_ids = _taskboard_projector._verify_dataset_split_groups(
            all_rows,
            expected_bundle_task_ids=expected_bundle_task_ids,
        )
    except _taskboard_projector.TaskBoardProjectorError as exc:
        raise TrainingReleaseError("projector_semantic_invalid") from exc
    counts = _mapping(value.get("counts"), "projector_counts_invalid")
    if dict(counts) != _projector_counts(all_rows, projected_task_ids):
        _fail("projector_counts_invalid")
    return (
        files,
        config.manifest_schema_sha256,
        config.sidecar_schema_sha256,
        config.segment_plan_schema_sha256,
    )


def _partition_metadata(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    candidate = value.get("candidate_task_count")
    accepted = value.get("gold_task_count")
    if (
        not _positive_int(candidate)
        or not _positive_int(accepted)
        or int(accepted) > int(candidate)
        or not _is_sha256(value.get("ids_sha256"))
        or not _is_sha256(value.get("allowlist_sha256"))
    ):
        _fail(f"source_{name}_split_invalid")
    return {
        "source_population_count": candidate,
        "accepted_gold_count": accepted,
        "source_instance_ids_sha256": value["ids_sha256"],
        "allowlist_sha256": value["allowlist_sha256"],
    }


def _contains_forbidden_heldout_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _HELDOUT_FORBIDDEN_KEYS:
                return True
            if _contains_forbidden_heldout_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_heldout_key(item) for item in value)
    return False


def freeze_generic_execution_contract(
    offline_preflight: str | Path,
    expected_preflight_sha: str,
    execution_lock: str | Path,
    expected_execution_lock_sha: str,
    attestation: str | Path,
    expected_attestation_sha: str,
    coordinator_config: str | Path,
    expected_coordinator_config_sha: str,
    source_bank_manifest: str | Path,
    expected_source_bank_sha: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze a sanitized, metadata-only generic execution contract."""

    preflight, inventory = _load_expected_json(
        offline_preflight,
        expected_preflight_sha,
        PREFLIGHT_SCHEMA,
        "generic_preflight_invalid",
    )
    lock, lock_inventory = _load_expected_json(
        execution_lock,
        expected_execution_lock_sha,
        EXECUTION_LOCK_SCHEMA,
        "execution_lock_invalid",
    )
    inventory.update(lock_inventory)
    attestation_value, attestation_inventory = _load_expected_json(
        attestation,
        expected_attestation_sha,
        ATTESTATION_SCHEMA,
        "execution_attestation_invalid",
    )
    inventory.update(attestation_inventory)
    _coordinator_snapshot, coordinator_inventory = _load_expected_file(
        coordinator_config,
        expected_coordinator_config_sha,
        "coordinator_config_invalid",
    )
    inventory.update(coordinator_inventory)
    source_bank, source_inventory = _load_expected_json(
        source_bank_manifest,
        expected_source_bank_sha,
        SOURCE_BANK_SCHEMA,
        "source_bank_manifest_invalid",
    )
    inventory.update(source_inventory)

    execution = _mapping(
        preflight.get("execution_contract"), "generic_preflight_invalid"
    )
    if (
        preflight.get("offline") is not True
        or preflight.get("provider_requests") != 0
        or preflight.get("credentials_read") is not False
        or preflight.get("sample_bodies_read", False) is not False
        or preflight.get("sample_bodies_printed") is not False
        or preflight.get("heldout_files_read") is not False
        or preflight.get("component_ready") is not True
        or preflight.get("bank_ready") is not True
        or preflight.get("execution_contract_ready") is not True
        or preflight.get("live_start_allowed") is not True
        or preflight.get("live_started") is not False
        or preflight.get("reason_code")
        != "generic_train_execution_contract_ready"
        or preflight.get("source_bank_manifest_sha256")
        != expected_source_bank_sha
        or execution.get("mode") != "generic_train_repo_base_commit"
        or execution.get("ready") is not True
        or execution.get("reason_code")
        != "generic_train_execution_contract_ready"
        or execution.get("remaining_gates") != []
        or execution.get("lock_sha256") != expected_execution_lock_sha
        or execution.get("required_schema") != ATTESTATION_SCHEMA
        or execution.get("observed_schema") != ATTESTATION_SCHEMA
        or execution.get("required_tool_contract_version")
        != "anchor.execution-tool-contract.v3"
        or execution.get("not_official_swebench_pass") is not True
    ):
        _fail("generic_preflight_not_ready")
    if lock.get("schema_version") != EXECUTION_LOCK_SCHEMA:
        _fail("execution_lock_invalid")
    if (
        attestation_value.get("schema_version") != ATTESTATION_SCHEMA
        or attestation_value.get("content_free") is not True
        or attestation_value.get("oracle_material_retained") is not False
        or attestation_value.get("lock_sha256") != expected_execution_lock_sha
        or attestation_value.get("tool_contract_version")
        != "anchor.execution-tool-contract.v3"
    ):
        _fail("execution_attestation_invalid")
    if (
        source_bank.get("schema_version") != SOURCE_BANK_SCHEMA
        or source_bank.get("publication_ready") is not True
        or source_bank.get("source_split") != "train"
        or source_bank.get("train_only") is not True
        or source_bank.get("raw_source_included") is not False
    ):
        _fail("source_bank_manifest_invalid")

    payload = {
        "schema_version": GENERIC_SCHEMA,
        "status": "ready",
        "source_preflight_sha256": expected_preflight_sha,
        "execution_lock_sha256": expected_execution_lock_sha,
        "execution_lock_schema_version": EXECUTION_LOCK_SCHEMA,
        "mode": "generic_train_repo_base_commit",
        "reason_code": "generic_train_execution_contract_ready",
        "required_attestation_schema": ATTESTATION_SCHEMA,
        "required_tool_contract_version": "anchor.execution-tool-contract.v3",
        "attestation_sha256": expected_attestation_sha,
        "coordinator_config_sha256": expected_coordinator_config_sha,
        "source_bank_manifest_sha256": expected_source_bank_sha,
        "offline": True,
        "provider_requests": 0,
        "credentials_read": False,
        "sample_bodies_printed": False,
        "heldout_files_read": False,
        "not_official_swebench_pass": True,
        "claim_scope": "generic_train_only",
    }
    result = _publish(output_dir, payload, inventory)
    result["provider_requests"] = 0
    return result


def freeze_source_disjoint(
    snapshot_dir: str | Path,
    expected_snapshot_sha: str,
    projector_dir: str | Path,
    expected_projector_sha: str,
    heldout_manifest: str | Path,
    expected_heldout_sha: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze source-disjoint count/hash metadata without held-out content."""

    snapshot_root, snapshot, inventory = _load_artifact(
        snapshot_dir, expected_snapshot_sha, SNAPSHOT_SCHEMA, "snapshot_artifact_invalid"
    )
    projector_root, projector, projector_inventory = _load_artifact(
        projector_dir,
        expected_projector_sha,
        PROJECTOR_SCHEMA,
        "projector_artifact_invalid",
    )
    inventory.update(projector_inventory)
    (
        files,
        manifest_schema_sha,
        sidecar_schema_sha,
        segment_plan_schema_sha,
    ) = _validate_projector(projector_root, projector, inventory)
    projector_input = _mapping(projector.get("input"), "projector_manifest_invalid")
    snapshot_sidecar_snapshot = inventory[
        "snapshot_artifact_invalid:sidecar"
    ][1]
    if (
        projector_input.get("snapshot_manifest_sha256")
        != expected_snapshot_sha
        or projector_input.get("snapshot_sha256")
        != snapshot.get("snapshot_sha256")
        or projector_input.get("snapshot_sha256_sidecar_sha256")
        != snapshot_sidecar_snapshot.sha256
    ):
        _fail("projector_snapshot_binding_mismatch")

    split = _mapping(snapshot.get("split_contract"), "snapshot_split_invalid")
    partitions = _mapping(split.get("partitions"), "snapshot_split_invalid")
    if (
        split.get("schema_version") != SPLIT_SCHEMA
        or split.get("assignment") != "source_bank_split_then_gold_gate_v1"
        or split.get("pairwise_disjoint") is not True
        or split.get("gold_coverage_complete") is not True
        or split.get("heldout_content_read") is not False
        or split.get("heldout_content_emitted") is not False
        or not _is_sha256(split.get("leakage_audit_sha256"))
        or set(partitions) != {"train", "calibration", "heldout"}
    ):
        _fail("snapshot_split_invalid")
    train = _mapping(partitions["train"], "source_train_split_invalid")
    calibration = _mapping(
        partitions["calibration"], "source_calibration_split_invalid"
    )
    heldout = _mapping(partitions["heldout"], "source_heldout_split_invalid")
    if train.get("role") != "training_only" or train.get("source_partition") != "train":
        _fail("source_train_split_invalid")
    if (
        calibration.get("role") != "rank_allocation_only"
        or calibration.get("source_partition") != "validation-from-train"
        or calibration.get("is_heldout") is True
    ):
        _fail("source_calibration_split_invalid")
    if (
        heldout.get("role") != "evaluation_only_hash_metadata"
        or heldout.get("source_partition") != "external-heldout"
        or heldout.get("content_present") is not False
        or heldout.get("content_read") is not False
        or heldout.get("content_emitted") is not False
        or not _is_sha256(heldout.get("ids_sha256"))
        or heldout.get("manifest_sha256") != expected_heldout_sha
        or "files" in heldout
        or _contains_forbidden_heldout_key(heldout)
    ):
        _fail("source_heldout_split_invalid")

    heldout_value, heldout_inventory = _load_expected_json(
        heldout_manifest,
        expected_heldout_sha,
        "anchor.heldout-manifest.v1",
        "heldout_manifest_invalid",
    )
    inventory.update(heldout_inventory)
    heldout_path = Path(heldout_manifest).expanduser().resolve()
    heldout_snapshot = inventory["heldout_manifest_invalid"][1]
    sidecar_path = heldout_path.with_name("manifest.json.sha256")
    sidecar = _strict_sidecar(
        sidecar_path, heldout_snapshot, "heldout_manifest_sidecar_invalid"
    )
    inventory["heldout-manifest-sidecar"] = (sidecar_path, sidecar)
    if (
        heldout_value.get("split") != "heldout"
        or not _positive_int(heldout_value.get("case_count"))
        or not _is_sha256(heldout_value.get("canonical_cases_sha256"))
        or heldout_value.get("canonical_cases_sha256") != heldout.get("ids_sha256")
        or _contains_forbidden_heldout_key(heldout_value)
    ):
        _fail("heldout_manifest_metadata_invalid")

    train_meta = _partition_metadata(train, "train")
    calibration_meta = _partition_metadata(calibration, "calibration")
    counts = _mapping(projector.get("counts"), "projector_manifest_invalid")
    if not _is_sha256(counts.get("task_ids_sha256")):
        _fail("projector_manifest_invalid")
    payload = {
        "schema_version": SOURCE_SCHEMA,
        "status": "ready",
        "bindings": {
            "snapshot_manifest_sha256": expected_snapshot_sha,
            "projector_manifest_sha256": expected_projector_sha,
            "projector_manifest_schema_sha256": manifest_schema_sha,
            "projector_sidecar_schema_sha256": sidecar_schema_sha,
            "projector_segment_plan_schema_sha256": segment_plan_schema_sha,
            "heldout_manifest_sha256": expected_heldout_sha,
        },
        "projector_grouping": {
            "split_group_key": "task_bundle_sha256",
            "task_id_cross_binding_key": "training_record.task_board.task_id",
            "inner_task_ids_sha256": counts["task_ids_sha256"],
            "all_five_role_views_same_split": True,
        },
        "hierarchical_task_kv": _hierarchical_task_kv_contract(
            segment_plan_schema_sha
        ),
        "partitions": {
            "train": {"role": "training_only", **train_meta},
            "calibration": {
                "role": "rank_allocation_only",
                "is_heldout": False,
                **calibration_meta,
            },
            "heldout": {
                "role": "evaluation_only_hash_metadata",
                "case_count": heldout_value["case_count"],
                "canonical_cases_sha256": heldout["ids_sha256"],
                "manifest_sha256": expected_heldout_sha,
            },
        },
        "projected_files_sha256": {
            str(item["path"]): item["sha256"] for item in files
        },
        "pairwise_source_disjoint": True,
        "calibration_is_heldout": False,
        "heldout_manifest_metadata_read": True,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "leakage_audit_sha256": split["leakage_audit_sha256"],
        "claim_scope": "research_proxy_only",
    }
    result = _publish(output_dir, payload, inventory)
    result.update(
        {
            "train_count": train_meta["accepted_gold_count"],
            "calibration_count": calibration_meta["accepted_gold_count"],
            "heldout_count": heldout_value["case_count"],
        }
    )
    return result


def _validate_generic(value: Mapping[str, Any], execution_lock_sha: str) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "source_preflight_sha256",
            "execution_lock_sha256",
            "execution_lock_schema_version",
            "mode",
            "reason_code",
            "required_attestation_schema",
            "required_tool_contract_version",
            "attestation_sha256",
            "coordinator_config_sha256",
            "source_bank_manifest_sha256",
            "offline",
            "provider_requests",
            "credentials_read",
            "sample_bodies_printed",
            "heldout_files_read",
            "not_official_swebench_pass",
            "claim_scope",
        },
        "generic_execution_contract_invalid",
    )
    if (
        value.get("status") != "ready"
        or not _is_sha256(value.get("source_preflight_sha256"))
        or value.get("execution_lock_sha256") != execution_lock_sha
        or value.get("execution_lock_schema_version") != EXECUTION_LOCK_SCHEMA
        or value.get("mode") != "generic_train_repo_base_commit"
        or value.get("reason_code") != "generic_train_execution_contract_ready"
        or value.get("required_attestation_schema")
        != "anchor.multilang-execution-attestation.v1"
        or value.get("required_tool_contract_version")
        != "anchor.execution-tool-contract.v3"
        or not _is_sha256(value.get("attestation_sha256"))
        or not _is_sha256(value.get("coordinator_config_sha256"))
        or not _is_sha256(value.get("source_bank_manifest_sha256"))
        or value.get("offline") is not True
        or value.get("provider_requests") != 0
        or value.get("credentials_read") is not False
        or value.get("sample_bodies_printed") is not False
        or value.get("heldout_files_read") is not False
        or value.get("not_official_swebench_pass") is not True
        or value.get("claim_scope") != "generic_train_only"
    ):
        _fail("generic_execution_contract_invalid")


def _validate_consumer(
    value: Mapping[str, Any],
    manifest_schema_sha: str,
    sidecar_schema_sha: str,
    segment_plan_schema_sha: str,
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "consumer_id",
            "consumer_version",
            "accepted_projector_schema",
            "projector_manifest_schema_sha256",
            "projector_sidecar_schema_sha256",
            "accepted_segment_plan_schema",
            "projector_segment_plan_schema_sha256",
            "segment_plan_location",
            "hierarchical_task_kv_architecture",
            "hierarchical_task_kv_execution_mode",
            "materialization",
            "tensors_emitted",
            "kv_payloads_emitted",
            "shared_prefix_membership",
            "ordered_prefix_chain",
            "independent_segment_concatenation_allowed",
            "exact_reuse_scope",
            "shared_then_mask_allowed",
            "forbidden_current_future_preinsert_allowed",
            "cache_identity_required_exact_match_fields",
            "cache_identity_mismatch_result",
            "cache_identity_unknown_result",
            "target_delta_initial_cache_scope",
            "target_delta_promotion_requires",
            "target_delta_promoted_cache_scope",
            "current_target_segment_emitted",
            "q_specialization_alone_sufficient_for_exact_reuse",
            "naive_in_stack_q_lora_exact_reuse_allowed",
            "full_generation_kv_shared_claimed",
            "token_level_moe_claimed",
            "split_group_key",
            "task_id_cross_binding_key",
            "fixed_inputs",
            "required_roles",
            "implementation_files",
            "launch_entrypoint",
            "provenance_location",
            "calibration_is_heldout",
            "heldout_content_read",
            "claim_scope",
        },
        "consumer_contract_invalid",
    )
    implementation = value.get("implementation_files")
    entrypoint = value.get("launch_entrypoint")
    if (
        not isinstance(value.get("consumer_id"), str)
        or not _IDENTIFIER_RE.fullmatch(str(value["consumer_id"]))
        or not isinstance(value.get("consumer_version"), str)
        or not value["consumer_version"]
        or value.get("accepted_projector_schema") != PROJECTOR_SCHEMA
        or value.get("projector_manifest_schema_sha256") != manifest_schema_sha
        or value.get("projector_sidecar_schema_sha256") != sidecar_schema_sha
        or value.get("accepted_segment_plan_schema") != SEGMENT_PLAN_SCHEMA
        or value.get("projector_segment_plan_schema_sha256")
        != segment_plan_schema_sha
        or value.get("segment_plan_location") != "outer_sidecar.segment_plan"
        or value.get("hierarchical_task_kv_architecture")
        != "hierarchical_task_kv"
        or value.get("hierarchical_task_kv_execution_mode")
        != SEGMENT_EXECUTION_MODE
        or value.get("materialization") != "metadata_only_no_tensor_or_kv"
        or value.get("tensors_emitted") is not False
        or value.get("kv_payloads_emitted") is not False
        or value.get("shared_prefix_membership")
        != "strict_all_five_role_visibility_intersection"
        or value.get("ordered_prefix_chain") is not True
        or value.get("independent_segment_concatenation_allowed") is not False
        or value.get("exact_reuse_scope") != SEGMENT_REUSE_SCOPE
        or value.get("shared_then_mask_allowed") is not False
        or value.get("forbidden_current_future_preinsert_allowed") is not False
        or value.get("cache_identity_required_exact_match_fields")
        != list(SEGMENT_CACHE_IDENTITY_FIELDS)
        or value.get("cache_identity_mismatch_result") != "cache_incompatible"
        or value.get("cache_identity_unknown_result") != "cache_incompatible"
        or value.get("target_delta_initial_cache_scope")
        != "expert_private_delta"
        or value.get("target_delta_promotion_requires")
        != "explicit_committed_and_causally_visible_downstream"
        or value.get("target_delta_promoted_cache_scope")
        != "downstream_task_shared_immutable"
        or value.get("current_target_segment_emitted") is not False
        or value.get("q_specialization_alone_sufficient_for_exact_reuse") is not False
        or value.get("naive_in_stack_q_lora_exact_reuse_allowed") is not False
        or value.get("full_generation_kv_shared_claimed") is not False
        or value.get("token_level_moe_claimed") is not False
        or value.get("split_group_key") != "task_bundle_sha256"
        or value.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or value.get("fixed_inputs") != [item[0] for item in FIXED_FILES]
        or value.get("required_roles") != list(REQUIRED_ROLES)
        or not isinstance(implementation, list)
        or not implementation
        or not isinstance(entrypoint, Mapping)
        or value.get("provenance_location") != "outer_sidecar"
        or value.get("calibration_is_heldout") is not False
        or value.get("heldout_content_read") is not False
        or value.get("claim_scope") != "research_proxy_only"
    ):
        _fail("consumer_contract_invalid")
    for item in [*implementation, entrypoint]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not _safe_relative(item.get("path"))
            or not _is_sha256(item.get("sha256"))
        ):
            _fail("consumer_contract_invalid")


def _validate_source_manifest(
    value: Mapping[str, Any],
    *,
    projector_sha: str,
    manifest_schema_sha: str,
    sidecar_schema_sha: str,
    segment_plan_schema_sha: str,
    projected_files: Sequence[Mapping[str, Any]],
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "bindings",
            "projector_grouping",
            "hierarchical_task_kv",
            "partitions",
            "projected_files_sha256",
            "pairwise_source_disjoint",
            "calibration_is_heldout",
            "heldout_manifest_metadata_read",
            "heldout_content_read",
            "heldout_content_emitted",
            "leakage_audit_sha256",
            "claim_scope",
        },
        "source_disjoint_artifact_invalid",
    )
    bindings = _mapping(value.get("bindings"), "source_disjoint_artifact_invalid")
    _exact_keys(
        bindings,
        {
            "snapshot_manifest_sha256",
            "projector_manifest_sha256",
            "projector_manifest_schema_sha256",
            "projector_sidecar_schema_sha256",
            "projector_segment_plan_schema_sha256",
            "heldout_manifest_sha256",
        },
        "source_disjoint_artifact_invalid",
    )
    grouping = _mapping(
        value.get("projector_grouping"), "source_disjoint_artifact_invalid"
    )
    task_kv = _mapping(
        value.get("hierarchical_task_kv"), "source_disjoint_artifact_invalid"
    )
    expected_task_kv = _hierarchical_task_kv_contract(segment_plan_schema_sha)
    _exact_keys(
        grouping,
        {
            "split_group_key",
            "task_id_cross_binding_key",
            "inner_task_ids_sha256",
            "all_five_role_views_same_split",
        },
        "source_disjoint_artifact_invalid",
    )
    partitions = _mapping(
        value.get("partitions"), "source_disjoint_artifact_invalid"
    )
    if set(partitions) != {"train", "calibration", "heldout"}:
        _fail("source_disjoint_artifact_invalid")
    train = _mapping(partitions["train"], "source_disjoint_artifact_invalid")
    calibration = _mapping(
        partitions["calibration"], "source_disjoint_artifact_invalid"
    )
    heldout = _mapping(partitions["heldout"], "source_disjoint_artifact_invalid")
    _exact_keys(
        train,
        {
            "role",
            "source_population_count",
            "accepted_gold_count",
            "source_instance_ids_sha256",
            "allowlist_sha256",
        },
        "source_disjoint_artifact_invalid",
    )
    _exact_keys(
        calibration,
        {
            "role",
            "is_heldout",
            "source_population_count",
            "accepted_gold_count",
            "source_instance_ids_sha256",
            "allowlist_sha256",
        },
        "source_disjoint_artifact_invalid",
    )
    _exact_keys(
        heldout,
        {"role", "case_count", "canonical_cases_sha256", "manifest_sha256"},
        "source_disjoint_artifact_invalid",
    )
    for partition in (train, calibration):
        if (
            not _positive_int(partition.get("source_population_count"))
            or not _positive_int(partition.get("accepted_gold_count"))
            or int(partition["accepted_gold_count"])
            > int(partition["source_population_count"])
            or not _is_sha256(partition.get("source_instance_ids_sha256"))
            or not _is_sha256(partition.get("allowlist_sha256"))
        ):
            _fail("source_disjoint_artifact_invalid")
    expected_files = {
        str(item["path"]): str(item["sha256"]) for item in projected_files
    }
    if (
        value.get("status") != "ready"
        or any(not _is_sha256(item) for item in bindings.values())
        or bindings.get("projector_manifest_sha256") != projector_sha
        or bindings.get("projector_manifest_schema_sha256")
        != manifest_schema_sha
        or bindings.get("projector_sidecar_schema_sha256") != sidecar_schema_sha
        or bindings.get("projector_segment_plan_schema_sha256")
        != segment_plan_schema_sha
        or dict(task_kv) != expected_task_kv
        or grouping.get("split_group_key") != "task_bundle_sha256"
        or grouping.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or not _is_sha256(grouping.get("inner_task_ids_sha256"))
        or grouping.get("all_five_role_views_same_split") is not True
        or train.get("role") != "training_only"
        or calibration.get("role") != "rank_allocation_only"
        or calibration.get("is_heldout") is not False
        or heldout.get("role") != "evaluation_only_hash_metadata"
        or not _positive_int(heldout.get("case_count"))
        or not _is_sha256(heldout.get("canonical_cases_sha256"))
        or heldout.get("manifest_sha256") != bindings.get("heldout_manifest_sha256")
        or value.get("projected_files_sha256") != expected_files
        or value.get("pairwise_source_disjoint") is not True
        or value.get("calibration_is_heldout") is not False
        or value.get("heldout_manifest_metadata_read") is not True
        or value.get("heldout_content_read") is not False
        or value.get("heldout_content_emitted") is not False
        or not _is_sha256(value.get("leakage_audit_sha256"))
        or value.get("claim_scope") != "research_proxy_only"
    ):
        _fail("source_disjoint_artifact_invalid")


def freeze_training_release(
    projector_dir: str | Path,
    expected_projector_sha: str,
    source_disjoint_dir: str | Path,
    expected_source_disjoint_sha: str,
    generic_contract_dir: str | Path,
    expected_generic_sha: str,
    consumer_contract: str | Path,
    expected_consumer_sha: str,
    execution_lock: str | Path,
    expected_execution_lock_sha: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze the final consumer-facing release lock."""

    projector_root, projector, inventory = _load_artifact(
        projector_dir,
        expected_projector_sha,
        PROJECTOR_SCHEMA,
        "projector_artifact_invalid",
    )
    (
        files,
        manifest_schema_sha,
        sidecar_schema_sha,
        segment_plan_schema_sha,
    ) = _validate_projector(projector_root, projector, inventory)
    _source_root, source, source_inventory = _load_artifact(
        source_disjoint_dir,
        expected_source_disjoint_sha,
        SOURCE_SCHEMA,
        "source_disjoint_artifact_invalid",
    )
    inventory.update(source_inventory)
    _validate_source_manifest(
        source,
        projector_sha=expected_projector_sha,
        manifest_schema_sha=manifest_schema_sha,
        sidecar_schema_sha=sidecar_schema_sha,
        segment_plan_schema_sha=segment_plan_schema_sha,
        projected_files=files,
    )

    execution, execution_inventory = _load_expected_json(
        execution_lock,
        expected_execution_lock_sha,
        EXECUTION_LOCK_SCHEMA,
        "execution_lock_invalid",
    )
    inventory.update(execution_inventory)
    if execution.get("schema_version") != EXECUTION_LOCK_SCHEMA:
        _fail("execution_lock_invalid")
    _generic_root, generic, generic_inventory = _load_artifact(
        generic_contract_dir,
        expected_generic_sha,
        GENERIC_SCHEMA,
        "generic_execution_contract_invalid",
    )
    inventory.update(generic_inventory)
    _validate_generic(generic, expected_execution_lock_sha)
    consumer, consumer_inventory = _load_expected_json(
        consumer_contract,
        expected_consumer_sha,
        CONSUMER_SCHEMA,
        "consumer_contract_invalid",
    )
    inventory.update(consumer_inventory)
    _validate_consumer(
        consumer,
        manifest_schema_sha,
        sidecar_schema_sha,
        segment_plan_schema_sha,
    )

    payload = {
        "schema_version": RELEASE_SCHEMA,
        "status": "ready",
        "bindings": {
            "projector_manifest_sha256": expected_projector_sha,
            "projector_manifest_schema_sha256": manifest_schema_sha,
            "projector_sidecar_schema_sha256": sidecar_schema_sha,
            "projector_segment_plan_schema_sha256": segment_plan_schema_sha,
            "source_disjoint_manifest_sha256": expected_source_disjoint_sha,
            "generic_execution_contract_sha256": expected_generic_sha,
            "consumer_contract_sha256": expected_consumer_sha,
            "execution_lock_sha256": expected_execution_lock_sha,
            "attestation_sha256": generic["attestation_sha256"],
            "coordinator_config_sha256": generic["coordinator_config_sha256"],
            "source_bank_manifest_sha256": generic[
                "source_bank_manifest_sha256"
            ],
        },
        "fixed_files": files,
        "consumer": {
            "consumer_id": consumer["consumer_id"],
            "consumer_version": consumer["consumer_version"],
            "implementation_files": consumer["implementation_files"],
            "launch_entrypoint": consumer["launch_entrypoint"],
        },
        "hierarchical_task_kv": _hierarchical_task_kv_contract(
            segment_plan_schema_sha
        ),
        "split_group_key": "task_bundle_sha256",
        "task_id_cross_binding_key": "training_record.task_board.task_id",
        "required_roles": list(REQUIRED_ROLES),
        "provenance_location": "outer_sidecar",
        "calibration_is_heldout": False,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "canonical_gold_written": False,
        "provider_requests": 0,
        "claim_scope": "research_proxy_only",
    }
    result = _publish(output_dir, payload, inventory)
    result["fixed_file_count"] = len(files)
    return result


__all__ = [
    "TrainingReleaseError",
    "freeze_generic_execution_contract",
    "freeze_source_disjoint",
    "freeze_training_release",
]
