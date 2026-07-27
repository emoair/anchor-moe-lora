"""Same-process Teacher FINAL candidate materializer for unbalanced-v2.

This module is intentionally additive.  It consumes the authenticated live
controller WAL while the runtime HMAC key is still resident, joins the
validated teacher results to the immutable Producer source bytes, and emits a
minimal, role-aware training projection.  It never performs a provider,
network, model, tokenizer, or GPU operation.

The emitted candidate is not a release.  A different reviewer must publish a
versioned external attestation and the v2 consumer binding before training can
consume it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch


RECORD_SCHEMA_VERSION = "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
MANIFEST_SCHEMA_VERSION = "anchor.gemma3-chat-unbalanced-v2-teacher-final-manifest.v2"
BUILD_RECEIPT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-build-receipt.v2"
)
CONFIG_SCHEMA_VERSION = "anchor.gemma3-chat-unbalanced-v2-teacher-finalizer.config.v1"
NAMESPACE = "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
ARTIFACT_VERSION = "anchor.gemma3-chat-unbalanced-v2-teacher-final.sharded.v2"
AIR_IDENTITY_SENTENCE = "我是由Air训练的测试模型。"
EXPECTED_ASSET_COUNTS = {
    "humor": 240,
    "serious": 240,
    "angry_style": 240,
    "tool_call": 1360,
    "review_audit": 1360,
    "planner_router": 80,
}
EXPECTED_BATCH_ROLES = {
    "humor": "humor",
    "serious": "serious",
    "angry_style": "angry",
    "tool_call": "tool",
    "review_audit": "review",
    "planner_router": "router",
}
STRUCTURED_ASSETS = {"tool_call", "review_audit", "planner_router"}
MAX_JSONL_LINE_BYTES = 2_000_000
MAX_PAYLOAD_BYTES = 49 * 1024 * 1024
_HASH_RE = batch.HASH_RE


@dataclass(frozen=True)
class Snapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceJoin:
    record_id: str
    record_id_sha256: str
    source_content_sha256: str
    source_serialization_identity_sha256: str
    source_asset: str
    task_bundle_sha256: str
    identity_aligned: bool
    identity_context: Mapping[str, Any] | None
    review_dependency: Mapping[str, Any] | None
    source_record: batch.SourceRecord


@dataclass(frozen=True)
class ShardPayload:
    asset: str
    records: tuple[Mapping[str, Any], ...]
    raw: bytes


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _strict_json(raw: bytes, *, reason: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise batch.AdapterError(f"{reason}_duplicate_key")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except batch.AdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise batch.AdapterError(reason) from error


def _require_hash(value: Any, *, reason: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise batch.AdapterError(reason)
    return value


def _path_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _directory_identity(path: Path, *, reason: str) -> tuple[int, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    attributes = int(getattr(value, "st_file_attributes", 0))
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or bool(attributes & 0x400)
    ):
        raise batch.AdapterError(reason)
    return (value.st_dev, value.st_ino)


def _assert_owned_directories(
    owned: Sequence[tuple[Path, tuple[int, int]]],
) -> None:
    for path, identity in owned:
        if (
            _directory_identity(
                path,
                reason="teacher_final_output_directory_identity_drift",
            )
            != identity
        ):
            raise batch.AdapterError("teacher_final_output_directory_identity_drift")


def _create_directory_chain(
    root: Path,
    relative: PurePosixPath,
) -> tuple[Path, list[tuple[Path, tuple[int, int]]]]:
    """Create a lexical directory chain while pinning every ancestor."""

    if relative.is_absolute() or ".." in relative.parts:
        raise batch.AdapterError("teacher_final_output_parent_invalid")
    current = root.absolute()
    _reject_reparse_chain(current, stop=current)
    owned = [
        (
            current,
            _directory_identity(
                current,
                reason="teacher_final_output_directory_invalid",
            ),
        )
    ]
    for part in relative.parts:
        _assert_owned_directories(owned)
        parent_identity = owned[-1][1]
        candidate = current / part
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as error:
            raise batch.AdapterError(
                "teacher_final_output_directory_create_failed"
            ) from error
        if (
            _directory_identity(
                current,
                reason="teacher_final_output_parent_identity_drift",
            )
            != parent_identity
        ):
            raise batch.AdapterError("teacher_final_output_parent_identity_drift")
        current = candidate
        owned.append(
            (
                current,
                _directory_identity(
                    current,
                    reason="teacher_final_output_directory_invalid",
                ),
            )
        )
    _assert_owned_directories(owned)
    return current, owned


def _reject_reparse_chain(path: Path, *, stop: Path) -> None:
    root = stop.resolve()
    current = path
    while True:
        try:
            value = current.lstat()
        except OSError as error:
            raise batch.AdapterError("teacher_final_path_stat_failed") from error
        attributes = int(getattr(value, "st_file_attributes", 0))
        if stat.S_ISLNK(value.st_mode) or bool(attributes & 0x400):
            raise batch.AdapterError("teacher_final_reparse_path_forbidden")
        if current.resolve() == root:
            return
        parent = current.parent
        if parent == current:
            raise batch.AdapterError("teacher_final_path_escape")
        current = parent


def _snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    reason: str,
) -> Snapshot:
    absolute = path.absolute()
    stop = Path(absolute.anchor)
    try:
        _reject_reparse_chain(absolute, stop=stop)
        path_before = absolute.lstat()
        attributes = int(getattr(path_before, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(path_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or bool(attributes & 0x400)
        ):
            raise batch.AdapterError(reason)
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOINHERIT", 0))
        descriptor = os.open(absolute, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = absolute.lstat()
        _reject_reparse_chain(absolute, stop=stop)
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if (
        _path_identity(path_before) != _path_identity(before)
        or _path_identity(before) != _path_identity(after)
        or _path_identity(after) != _path_identity(path_after)
        or len(raw) != before.st_size
        or (expected_bytes is not None and len(raw) != expected_bytes)
    ):
        raise batch.AdapterError(reason)
    digest = _sha256(raw)
    if expected_sha256 is not None and not hmac.compare_digest(
        digest, _require_hash(expected_sha256, reason=reason)
    ):
        raise batch.AdapterError(reason)
    return Snapshot(absolute, raw, digest, _path_identity(after))


def _recheck(snapshot: Snapshot, *, reason: str) -> None:
    observed = _snapshot(
        snapshot.path,
        expected_sha256=snapshot.sha256,
        expected_bytes=len(snapshot.raw),
        reason=reason,
    )
    if observed.identity != snapshot.identity or not hmac.compare_digest(
        observed.raw, snapshot.raw
    ):
        raise batch.AdapterError(reason)


def _jsonl_rows(raw: bytes, *, reason: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise batch.AdapterError(f"{reason}_physical_format_invalid")
    result: list[dict[str, Any]] = []
    for line in raw.splitlines(keepends=True):
        if len(line) > MAX_JSONL_LINE_BYTES or not line.endswith(b"\n"):
            raise batch.AdapterError(f"{reason}_line_invalid")
        value = _strict_json(line[:-1], reason=f"{reason}_json_invalid")
        if not isinstance(value, dict):
            raise batch.AdapterError(f"{reason}_record_invalid")
        result.append(value)
    return result


def _schema(path: Path, expected_sha256: str) -> Draft202012Validator:
    snap = _snapshot(
        path,
        expected_sha256=expected_sha256,
        reason="teacher_final_schema_identity_drift",
    )
    value = _strict_json(snap.raw, reason="teacher_final_schema_json_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as error:
        raise batch.AdapterError("teacher_final_schema_invalid") from error
    return Draft202012Validator(value)


def _validate_schema(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    try:
        validator.validate(value)
    except ValidationError as error:
        raise batch.AdapterError(reason) from error


def _source_main_join(
    record: Mapping[str, Any],
    source: batch.SourceRecord,
) -> SourceJoin:
    if (
        record.get("schema_version")
        != "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-record.v1"
        or record.get("namespace") != "gemma3_chat_five_expert_qonly_unbalanced_v2"
        or record.get("split") != "train"
        or record.get("role") not in EXPECTED_BATCH_ROLES
    ):
        raise batch.AdapterError("teacher_final_source_main_contract_invalid")
    record_id = record.get("record_id")
    task_bundle = record.get("task_bundle_sha256")
    messages = record.get("messages")
    serialization = record.get("gemma_serialization")
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(task_bundle, str)
        or _HASH_RE.fullmatch(task_bundle) is None
        or not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(serialization, dict)
    ):
        raise batch.AdapterError("teacher_final_source_main_identity_invalid")
    asset = str(record["role"])
    identity_context = record.get("identity_alignment")
    identity_aligned = identity_context is not None
    expected_role = EXPECTED_BATCH_ROLES[asset]
    if asset == "tool_call" and identity_aligned:
        trace = record.get("tool_trace")
        if not isinstance(trace, dict) or trace.get("direct_no_tool") is not True:
            raise batch.AdapterError("teacher_final_identity_tool_not_direct")
        expected_role = "identity"
    if (
        source.record_id != record_id
        or source.task_bundle != task_bundle
        or source.role != expected_role
        or source.partition_kind != "train"
        or source.gemma_serialization_identity_sha256
        != serialization.get("prompt_sha256")
    ):
        raise batch.AdapterError("teacher_final_source_projection_join_drift")
    review_dependency = record.get("review_dependency")
    if review_dependency is not None and not isinstance(review_dependency, dict):
        raise batch.AdapterError("teacher_final_review_dependency_invalid")
    return SourceJoin(
        record_id=record_id,
        record_id_sha256=_sha256(record_id.encode("utf-8")),
        source_content_sha256=_canonical_sha256(messages),
        source_serialization_identity_sha256=_canonical_sha256(serialization),
        source_asset=asset,
        task_bundle_sha256=task_bundle,
        identity_aligned=identity_aligned,
        identity_context=(
            dict(identity_context) if isinstance(identity_context, Mapping) else None
        ),
        review_dependency=(
            dict(review_dependency) if isinstance(review_dependency, Mapping) else None
        ),
        source_record=source,
    )


def _source_router_join(
    record: Mapping[str, Any],
    source: batch.SourceRecord,
) -> SourceJoin | None:
    if record.get("namespace") != "gemma3_chat_emotion_router_qonly_v1" or record.get(
        "split"
    ) not in {"train", "eval_proxy"}:
        raise batch.AdapterError("teacher_final_source_router_contract_invalid")
    if record["split"] == "eval_proxy":
        return None
    record_id = record.get("record_id")
    task_bundle = record.get("task_bundle_sha256")
    messages = record.get("messages")
    serialization = record.get("serialization")
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(task_bundle, str)
        or _HASH_RE.fullmatch(task_bundle) is None
        or not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(serialization, dict)
        or source.record_id != record_id
        or source.task_bundle != task_bundle
        or source.role != "router"
        or source.partition_kind != "router_train"
        or source.gemma_serialization_identity_sha256
        != serialization.get("prompt_sha256")
    ):
        raise batch.AdapterError("teacher_final_source_router_join_drift")
    return SourceJoin(
        record_id=record_id,
        record_id_sha256=_sha256(record_id.encode("utf-8")),
        source_content_sha256=_canonical_sha256(messages),
        source_serialization_identity_sha256=_canonical_sha256(serialization),
        source_asset="planner_router",
        task_bundle_sha256=task_bundle,
        identity_aligned=False,
        identity_context=None,
        review_dependency=None,
        source_record=source,
    )


def build_source_joins(
    inventory: batch.SourceInventory,
    *,
    main_rows: Sequence[Mapping[str, Any]],
    router_rows: Sequence[Mapping[str, Any]],
) -> tuple[SourceJoin, ...]:
    """Cross-bind authenticated Producer bytes to the batch source projection."""

    by_id = {item.record_id: item for item in inventory.records}
    if len(by_id) != len(inventory.records):
        raise batch.AdapterError("teacher_final_source_record_duplicate")
    joins: list[SourceJoin] = []
    seen: set[str] = set()
    for record in main_rows:
        record_id = record.get("record_id")
        source = by_id.get(str(record_id))
        if source is None:
            raise batch.AdapterError("teacher_final_source_record_missing")
        join = _source_main_join(record, source)
        if join.record_id in seen:
            raise batch.AdapterError("teacher_final_source_record_duplicate")
        seen.add(join.record_id)
        joins.append(join)
    for record in router_rows:
        if record.get("split") == "eval_proxy":
            continue
        record_id = record.get("record_id")
        source = by_id.get(str(record_id))
        if source is None:
            raise batch.AdapterError("teacher_final_source_record_missing")
        join = _source_router_join(record, source)
        assert join is not None
        if join.record_id in seen:
            raise batch.AdapterError("teacher_final_source_record_duplicate")
        seen.add(join.record_id)
        joins.append(join)
    counts = Counter(item.source_asset for item in joins)
    if (
        len(joins) != batch.EXPECTED_TOTAL_JOBS
        or len(seen) != batch.EXPECTED_TOTAL_JOBS
        or set(seen) != set(by_id)
        or dict(counts) != EXPECTED_ASSET_COUNTS
    ):
        raise batch.AdapterError("teacher_final_source_inventory_drift")
    # This is the P0 distinction: the Producer message identity and the
    # teacher-prompt projection identity are different domains.
    if any(
        hmac.compare_digest(
            item.source_content_sha256,
            item.source_record.content_identity_sha256,
        )
        for item in joins
    ):
        raise batch.AdapterError("teacher_final_source_identity_domain_collision")
    return tuple(joins)


def _same_json_shape(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if (
            any(not isinstance(key, str) for key in left)
            or any(not isinstance(key, str) for key in right)
            or set(left) != set(right)
        ):
            return False
        return all(_same_json_shape(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                _same_json_shape(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _canonical_teacher_target(
    join: SourceJoin,
    alignment: Mapping[str, Any],
) -> str:
    if (
        alignment.get("record_id") != join.record_id
        or alignment.get("task_bundle") != join.task_bundle_sha256
        or alignment.get("split") != "train"
        or alignment.get("hidden_reasoning_retained") is not False
    ):
        raise batch.AdapterError("teacher_final_alignment_join_drift")
    source_binding = alignment.get("source_binding")
    output = alignment.get("output")
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("content_identity_sha256")
        != join.source_record.content_identity_sha256
        or source_binding.get("gemma_serialization_identity_sha256")
        != join.source_record.gemma_serialization_identity_sha256
        or source_binding.get("task_bundle") != join.task_bundle_sha256
        or not isinstance(output, Mapping)
        or set(output) != {"kind", "value"}
    ):
        raise batch.AdapterError("teacher_final_alignment_binding_drift")
    kind = output.get("kind")
    value = output.get("value")
    source_role = join.source_record.role
    if source_role in {*batch.STYLE_NATURAL_TRANSPORT_ROLES, "identity"}:
        if kind != "natural_text" or not isinstance(value, str):
            raise batch.AdapterError("teacher_final_natural_target_invalid")
        target = value
        provider_wire = (
            _canonical_bytes({"final_answer": value}).decode("utf-8")
            if source_role in batch.STYLE_NATURAL_TRANSPORT_ROLES
            else value
        )
    elif source_role in {"tool", "review", "router"}:
        if kind != "structured_json" or not isinstance(value, Mapping):
            raise batch.AdapterError("teacher_final_structured_target_invalid")
        target = _canonical_bytes(value).decode("utf-8")
        provider_wire = target
    else:
        raise batch.AdapterError("teacher_final_target_role_unsupported")
    # Reconstruct the exact provider-wire shape from the authenticated source
    # role, then require the frozen validator to reproduce the persisted
    # semantic projection byte-for-byte and shape-for-shape.
    validated = batch.validate_teacher_output(join.source_record, provider_wire)
    expected = dict(output)
    if (
        not isinstance(validated, Mapping)
        or set(validated) != {"kind", "value"}
        or not _same_json_shape(validated, expected)
        or not hmac.compare_digest(
            _canonical_bytes(validated),
            _canonical_bytes(expected),
        )
    ):
        raise batch.AdapterError("teacher_final_target_revalidation_drift")
    return target


def _identity_target_audit(
    join: SourceJoin,
    target: str,
) -> None:
    if not join.identity_aligned:
        return
    context = join.identity_context
    if (
        not isinstance(context, Mapping)
        or context.get("stable_identity_fact") != AIR_IDENTITY_SENTENCE
    ):
        raise batch.AdapterError("teacher_final_identity_source_fact_drift")

    def reject_false_attribution(text: str) -> None:
        normalized = " ".join(text.casefold().split())
        phrases = (
            "由google训练",
            "由谷歌训练",
            "由openai训练",
            "trained by google",
            "trained by openai",
        )
        negations = (
            "不是",
            "并非",
            "不由",
            "并不是",
            "not ",
            "not trained",
            "was not",
            "wasn't",
            "never ",
        )
        for phrase in phrases:
            start = normalized.find(phrase)
            while start >= 0:
                prefix = normalized[max(0, start - 24) : start]
                if not any(marker in prefix for marker in negations):
                    raise batch.AdapterError("teacher_final_identity_false_attribution")
                start = normalized.find(phrase, start + len(phrase))

    if join.source_asset == "review_audit":
        value = _strict_json(
            target.encode("utf-8"),
            reason="teacher_final_identity_review_json_invalid",
        )
        if not isinstance(value, Mapping):
            raise batch.AdapterError("teacher_final_identity_review_invalid")
        dependency = join.review_dependency
        candidate = (
            dependency.get("candidate_projection")
            if isinstance(dependency, Mapping)
            else None
        )
        candidate_text = _canonical_bytes(candidate).decode("utf-8")
        if value.get("verdict") == "pass":
            if AIR_IDENTITY_SENTENCE not in candidate_text:
                raise batch.AdapterError(
                    "teacher_final_identity_review_pass_fact_missing"
                )
            reject_false_attribution(candidate_text)
        else:
            correction = str(value.get("correction", ""))
            if AIR_IDENTITY_SENTENCE not in correction:
                raise batch.AdapterError(
                    "teacher_final_identity_review_correction_fact_missing"
                )
            reject_false_attribution(correction)
        return
    if AIR_IDENTITY_SENTENCE not in target:
        raise batch.AdapterError("teacher_final_identity_target_fact_missing")
    reject_false_attribution(target)


def project_teacher_records(
    joins: Sequence[SourceJoin],
    alignment_rows: Sequence[Mapping[str, Any]],
    *,
    record_validator: Draft202012Validator | None = None,
    expected_idempotency_by_record: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Produce the exact ten-field consumer join projection."""

    by_record_id: dict[str, Mapping[str, Any]] = {}
    idempotency: set[str] = set()
    for row in alignment_rows:
        record_id = row.get("record_id")
        key = row.get("idempotency_key")
        if (
            not isinstance(record_id, str)
            or record_id in by_record_id
            or not isinstance(key, str)
            or _HASH_RE.fullmatch(key) is None
            or key in idempotency
        ):
            raise batch.AdapterError("teacher_final_alignment_inventory_invalid")
        if (
            expected_idempotency_by_record is not None
            and expected_idempotency_by_record.get(record_id) != key
        ):
            raise batch.AdapterError("teacher_final_alignment_idempotency_drift")
        by_record_id[record_id] = row
        idempotency.add(key)
    if len(by_record_id) != batch.EXPECTED_TOTAL_JOBS:
        raise batch.AdapterError("teacher_final_alignment_count_drift")
    if expected_idempotency_by_record is not None and (
        set(expected_idempotency_by_record) != set(by_record_id)
        or set(expected_idempotency_by_record.values()) != idempotency
    ):
        raise batch.AdapterError("teacher_final_alignment_idempotency_set_drift")
    result: list[dict[str, Any]] = []
    identity_bundles: set[str] = set()
    identity_records = 0
    for join in joins:
        alignment = by_record_id.pop(join.record_id, None)
        if alignment is None:
            raise batch.AdapterError("teacher_final_alignment_record_missing")
        target = _canonical_teacher_target(join, alignment)
        _identity_target_audit(join, target)
        if join.identity_aligned:
            identity_records += 1
            identity_bundles.add(join.task_bundle_sha256)
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "record_id_sha256": join.record_id_sha256,
            "source_content_sha256": join.source_content_sha256,
            "source_serialization_identity_sha256": (
                join.source_serialization_identity_sha256
            ),
            "task_bundle_sha256": join.task_bundle_sha256,
            "source_asset": join.source_asset,
            "decision": "accepted",
            "teacher_target": target,
            "teacher_target_sha256": _sha256(target.encode("utf-8")),
        }
        if set(record) != {
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
        }:
            raise batch.AdapterError("teacher_final_record_field_drift")
        if record_validator is not None:
            _validate_schema(
                record_validator,
                record,
                reason="teacher_final_record_schema_mismatch",
            )
        result.append(record)
    if by_record_id:
        raise batch.AdapterError("teacher_final_alignment_record_extra")
    counts = Counter(str(item["source_asset"]) for item in result)
    if (
        dict(counts) != EXPECTED_ASSET_COUNTS
        or identity_records != 100
        or len(identity_bundles) != 20
    ):
        raise batch.AdapterError("teacher_final_projected_count_drift")
    return tuple(result)


def _record_line(record: Mapping[str, Any]) -> bytes:
    raw = _canonical_bytes(record, newline=True)
    if len(raw) > MAX_JSONL_LINE_BYTES:
        raise batch.AdapterError("teacher_final_record_line_too_large")
    return raw


def shard_teacher_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
) -> tuple[ShardPayload, ...]:
    """Shard in source order without crossing a task-bundle boundary."""

    if not 1_000_000 <= max_payload_bytes < 50 * 1024 * 1024:
        raise batch.AdapterError("teacher_final_shard_limit_invalid")
    main = [item for item in records if item["source_asset"] != "planner_router"]
    router = [item for item in records if item["source_asset"] == "planner_router"]
    if len(main) != 3440 or len(router) != 80:
        raise batch.AdapterError("teacher_final_partition_count_drift")
    payloads: list[ShardPayload] = []
    for asset, values in (
        ("main_train", main),
        ("planner_router_train", router),
    ):
        groups: list[list[Mapping[str, Any]]] = []
        closed: set[str] = set()
        current: list[Mapping[str, Any]] = []
        current_bundle: str | None = None
        for record in values:
            bundle = str(record["task_bundle_sha256"])
            if current_bundle is None or bundle == current_bundle:
                current.append(record)
                current_bundle = bundle
                continue
            closed.add(current_bundle)
            if bundle in closed:
                raise batch.AdapterError("teacher_final_bundle_order_noncontiguous")
            groups.append(current)
            current = [record]
            current_bundle = bundle
        if current:
            groups.append(current)
        shard_records: list[Mapping[str, Any]] = []
        shard_raw = bytearray()
        for group in groups:
            group_raw = b"".join(_record_line(item) for item in group)
            if len(group_raw) >= 50 * 1024 * 1024:
                raise batch.AdapterError("teacher_final_bundle_exceeds_file_limit")
            if shard_records and len(shard_raw) + len(group_raw) > max_payload_bytes:
                payloads.append(
                    ShardPayload(asset, tuple(shard_records), bytes(shard_raw))
                )
                shard_records = []
                shard_raw = bytearray()
            shard_records.extend(group)
            shard_raw.extend(group_raw)
        if shard_records:
            payloads.append(ShardPayload(asset, tuple(shard_records), bytes(shard_raw)))
    if (
        not payloads
        or len(payloads) > 32
        or sum(len(item.records) for item in payloads) != 3520
        or any(len(item.raw) >= 50 * 1024 * 1024 for item in payloads)
    ):
        raise batch.AdapterError("teacher_final_shard_inventory_invalid")
    return tuple(payloads)


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written < 1:
            raise OSError("teacher_final_short_write")
        offset += written


def _exclusive_bytes(path: Path, raw: bytes) -> None:
    _directory_identity(
        path.parent,
        reason="teacher_final_write_parent_invalid",
    )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise batch.AdapterError("teacher_final_create_once_collision") from error
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _file_entry(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    snap = _snapshot(path, reason="teacher_final_output_snapshot_drift")
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": snap.sha256,
        "bytes": len(snap.raw),
    }


def _relative_file(root: Path, raw: Any, *, reason: str) -> Path:
    return _relative_path(root, raw, reason=reason, allow_missing=False)


def _relative_path(
    root: Path,
    raw: Any,
    *,
    reason: str,
    allow_missing: bool,
) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise batch.AdapterError(reason)
    posix = PurePosixPath(raw)
    if posix.is_absolute() or ".." in posix.parts:
        raise batch.AdapterError(reason)
    lexical_root = root.absolute()
    _reject_reparse_chain(lexical_root, stop=lexical_root)
    value = lexical_root.joinpath(*posix.parts)
    current = lexical_root
    missing = False
    for part in posix.parts:
        current = current / part
        try:
            physical = current.lstat()
        except FileNotFoundError:
            missing = True
            continue
        except OSError as error:
            raise batch.AdapterError(reason) from error
        if missing:
            raise batch.AdapterError(reason)
        attributes = int(getattr(physical, "st_file_attributes", 0))
        if stat.S_ISLNK(physical.st_mode) or bool(attributes & 0x400):
            raise batch.AdapterError("teacher_final_reparse_path_forbidden")
    if missing and not allow_missing:
        raise batch.AdapterError(reason)
    resolved = value.resolve(strict=False)
    resolved_root = lexical_root.resolve()
    if resolved_root not in resolved.parents:
        raise batch.AdapterError(reason)
    return value


def _rename_noreplace(
    source: Path,
    destination: Path,
    *,
    before_rename: Any,
) -> None:
    """Authenticate inside an OS-level atomic no-replace publication."""

    before_rename()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination)):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError("teacher_final_candidate_exists")
            raise OSError(error, "teacher_final_atomic_directory_move_failed")
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise batch.AdapterError("teacher_final_atomic_noreplace_unsupported")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == 17:
                raise FileExistsError("teacher_final_candidate_exists")
            raise OSError(error, "teacher_final_atomic_directory_move_failed")
        return
    raise batch.AdapterError("teacher_final_atomic_noreplace_unsupported")


def _verify_source_manifest_facts(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        raise batch.AdapterError("teacher_final_producer_manifest_counts_missing")
    expected = {
        "identity_bundles": 30,
        "identity_role_records": 150,
    }
    if any(counts.get(key) != item for key, item in expected.items()):
        raise batch.AdapterError("teacher_final_producer_identity_quota_drift")
    splits = counts.get("splits")
    if not isinstance(splits, Mapping) or splits != {
        "eval_proxy": 860,
        "train": 3440,
    }:
        raise batch.AdapterError("teacher_final_producer_split_drift")
    logical_identity = value.get("logical_identity")
    identity_probe_sha256 = (
        logical_identity.get("identity_probe_inventory_sha256")
        if isinstance(logical_identity, Mapping)
        else None
    )
    if (
        identity_probe_sha256
        != "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5"
    ):
        raise batch.AdapterError("teacher_final_identity_probe_digest_drift")
    return {
        "producer_all_source_bundles": 30,
        "producer_all_source_records": 150,
        "producer_eval_bundles": 10,
        "producer_eval_records": 50,
        "identity_probe_inventory_sha256": identity_probe_sha256,
    }


def _router_mixed_file_audit(
    router_rows: Sequence[Mapping[str, Any]],
    joins: Sequence[SourceJoin],
) -> dict[str, int]:
    split_counts = Counter(str(item.get("split")) for item in router_rows)
    if split_counts != Counter({"train": 80, "eval_proxy": 20}):
        raise batch.AdapterError("teacher_final_router_mixed_split_count_drift")
    eval_record_ids = [
        item.get("record_id") for item in router_rows if item["split"] == "eval_proxy"
    ]
    if (
        any(not isinstance(item, str) or not item for item in eval_record_ids)
        or len(set(eval_record_ids)) != 20
    ):
        raise batch.AdapterError("teacher_final_router_eval_identity_drift")
    emitted_record_ids = {
        item.record_id for item in joins if item.source_asset == "planner_router"
    }
    if len(emitted_record_ids) != 80:
        raise batch.AdapterError("teacher_final_router_train_projection_count_drift")
    emitted_eval_rows = len(set(eval_record_ids).intersection(emitted_record_ids))
    if emitted_eval_rows != 0:
        raise batch.AdapterError("teacher_final_router_eval_emission_forbidden")
    return {
        "router_mixed_file_eval_proxy_rows_parsed": split_counts["eval_proxy"],
        "router_eval_proxy_rows_emitted_to_teacher_train": emitted_eval_rows,
    }


def _phase_receipt_audit(
    state: Any,
) -> tuple[list[dict[str, Any]], str, str]:
    rows = state._read_entries()[2]
    phases = [str(item.get("phase")) for item in rows]
    counts = [int(item.get("job_count", -1)) for item in rows]
    if phases != ["smoke_exact1", "bounded_small", "bulk"] or counts != [
        1,
        15,
        3520,
    ]:
        raise batch.AdapterError("teacher_final_three_phase_receipts_missing")
    for item in rows:
        if (
            item.get("succeeded") != item.get("job_count")
            or item.get("rejected") != 0
            or item.get("content_retained") is not False
        ):
            raise batch.AdapterError("teacher_final_phase_receipt_not_green")
    paths = state._entry_paths()
    if not paths:
        raise batch.AdapterError("teacher_final_wal_empty")
    return (
        rows,
        _canonical_sha256(rows),
        _snapshot(paths[-1], reason="teacher_final_wal_tip_drift").sha256,
    )


def _load_finalizer_config(
    path: Path,
    *,
    expected_sha256: str,
    schema_path: Path,
    schema_sha256: str,
) -> tuple[dict[str, Any], Snapshot]:
    validator = _schema(schema_path, schema_sha256)
    snap = _snapshot(
        path,
        expected_sha256=expected_sha256,
        reason="teacher_final_config_identity_drift",
    )
    value = _strict_json(snap.raw, reason="teacher_final_config_json_invalid")
    if not isinstance(value, dict):
        raise batch.AdapterError("teacher_final_config_invalid")
    _validate_schema(
        validator,
        value,
        reason="teacher_final_config_schema_mismatch",
    )
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise batch.AdapterError("teacher_final_config_version_drift")
    return value, snap


def materialize_teacher_final(
    *,
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    runtime_slots: batch.RuntimeSecretSlots,
    controller_binding: Mapping[str, Any],
    controller_identity: Mapping[str, Any],
    final_report: batch.RunReport,
    config_path: Path,
    config_sha256: str,
    config_schema_path: Path,
    config_schema_sha256: str,
) -> dict[str, Any]:
    """Authenticate live state and atomically publish a release-ready candidate."""

    # Imported only after the integrated controller authenticated both modules.
    from . import gemma3_chat_unbalanced_v2_live_controller as live

    if (
        not runtime_slots.loaded
        or final_report.state != "complete"
        or final_report.succeeded != 3520
        or final_report.rejected != 0
    ):
        raise batch.AdapterError("teacher_final_runtime_not_green")
    config, config_snapshot = _load_finalizer_config(
        config_path,
        expected_sha256=config_sha256,
        schema_path=config_schema_path,
        schema_sha256=config_schema_sha256,
    )
    state = live.AuthenticatedBatchState(
        profile,
        hmac_key=runtime_slots.receipt_hmac_key(),
        controller_binding=controller_binding,
    )
    state.initialize(inventory)
    state.authenticate_authority()
    replay = state.replay()
    if (
        len(replay.completed) != 3520
        or replay.rejected
        or replay.uncertain
        or len(replay.events_by_job) != 3520
    ):
        raise batch.AdapterError("teacher_final_replay_not_exact")
    phase_rows, phase_inventory_sha, wal_tip_sha = _phase_receipt_audit(state)

    source_root = _relative_file(
        batch.REPO_ROOT,
        config["source"]["artifact_root"],
        reason="teacher_final_source_root_invalid",
    )
    if not source_root.is_dir():
        raise batch.AdapterError("teacher_final_source_root_missing")
    _reject_reparse_chain(source_root, stop=batch.REPO_ROOT)
    source_snapshots: list[Snapshot] = []
    main_rows: list[dict[str, Any]] = []
    for item in config["source"]["main_shards"]:
        path = _relative_file(
            source_root,
            item["path"],
            reason="teacher_final_source_shard_path_invalid",
        )
        snap = _snapshot(
            path,
            expected_sha256=item["sha256"],
            expected_bytes=item["bytes"],
            reason="teacher_final_source_shard_identity_drift",
        )
        source_snapshots.append(snap)
        rows = _jsonl_rows(snap.raw, reason="teacher_final_source_main")
        if len(rows) != item["records"]:
            raise batch.AdapterError("teacher_final_source_main_count_drift")
        main_rows.extend(rows)
    router_item = config["source"]["router_records"]
    router_path = _relative_file(
        source_root,
        router_item["path"],
        reason="teacher_final_router_path_invalid",
    )
    router_snapshot = _snapshot(
        router_path,
        expected_sha256=router_item["sha256"],
        expected_bytes=router_item["bytes"],
        reason="teacher_final_router_identity_drift",
    )
    source_snapshots.append(router_snapshot)
    router_rows = _jsonl_rows(
        router_snapshot.raw,
        reason="teacher_final_source_router",
    )
    if len(router_rows) != router_item["records"]:
        raise batch.AdapterError("teacher_final_router_count_drift")
    producer_manifest_item = config["source"]["producer_manifest"]
    producer_manifest_path = _relative_file(
        source_root,
        producer_manifest_item["path"],
        reason="teacher_final_producer_manifest_path_invalid",
    )
    producer_manifest_snapshot = _snapshot(
        producer_manifest_path,
        expected_sha256=producer_manifest_item["sha256"],
        expected_bytes=producer_manifest_item["bytes"],
        reason="teacher_final_producer_manifest_identity_drift",
    )
    source_snapshots.append(producer_manifest_snapshot)
    producer_manifest = _strict_json(
        producer_manifest_snapshot.raw,
        reason="teacher_final_producer_manifest_json_invalid",
    )
    if not isinstance(producer_manifest, Mapping):
        raise batch.AdapterError("teacher_final_producer_manifest_invalid")
    producer_identity_facts = _verify_source_manifest_facts(producer_manifest)

    joins = build_source_joins(
        inventory,
        main_rows=main_rows,
        router_rows=router_rows,
    )
    router_read_audit = _router_mixed_file_audit(router_rows, joins)
    scoped_read_claims = {
        "identity_eval_probe_body_reads": 0,
        **router_read_audit,
    }
    if any(
        config["claims"].get(key) != value for key, value in scoped_read_claims.items()
    ):
        raise batch.AdapterError("teacher_final_scoped_read_claim_drift")
    expected_jobs = batch.derive_jobs(inventory, profile)
    expected_idempotency_by_record = {
        item.source.record_id: item.idempotency_key for item in expected_jobs
    }
    if (
        len(expected_idempotency_by_record) != 3520
        or set(expected_idempotency_by_record.values()) != replay.completed
    ):
        raise batch.AdapterError("teacher_final_replay_job_identity_drift")
    alignment_snapshot = _snapshot(
        state.records_path,
        reason="teacher_final_alignment_records_snapshot_drift",
    )
    alignment_rows = _jsonl_rows(
        alignment_snapshot.raw,
        reason="teacher_final_alignment_records",
    )
    receipt_snapshot = _snapshot(
        state.receipts_path,
        reason="teacher_final_job_receipts_snapshot_drift",
    )
    receipt_rows = _jsonl_rows(
        receipt_snapshot.raw,
        reason="teacher_final_job_receipts",
    )
    receipts_by_key: dict[str, Mapping[str, Any]] = {}
    for receipt in receipt_rows:
        batch._verify_receipt(receipt, runtime_slots.receipt_hmac_key())
        key = receipt.get("idempotency_key")
        if (
            not isinstance(key, str)
            or key in receipts_by_key
            or receipt.get("outcome") != "succeeded"
        ):
            raise batch.AdapterError("teacher_final_job_receipt_inventory_drift")
        receipts_by_key[key] = receipt
    if set(receipts_by_key) != replay.completed:
        raise batch.AdapterError("teacher_final_job_receipt_set_drift")
    for row in alignment_rows:
        key = row.get("idempotency_key")
        receipt = receipts_by_key.get(str(key))
        if receipt is None or receipt.get("output_record_sha256") != batch._hash_object(
            row
        ):
            raise batch.AdapterError("teacher_final_output_receipt_binding_drift")
    record_schema_item = config["schemas"]["record"]
    record_schema_path = _relative_file(
        batch.REPO_ROOT,
        record_schema_item["path"],
        reason="teacher_final_record_schema_path_invalid",
    )
    record_validator = _schema(
        record_schema_path,
        record_schema_item["sha256"],
    )
    records = project_teacher_records(
        joins,
        alignment_rows,
        record_validator=record_validator,
        expected_idempotency_by_record=expected_idempotency_by_record,
    )
    payloads = shard_teacher_records(
        records,
        max_payload_bytes=int(config["output"]["max_payload_bytes"]),
    )

    output_parent_raw = config["output"]["parent"]
    if not isinstance(output_parent_raw, str):
        raise batch.AdapterError("teacher_final_output_parent_invalid")
    output_parent_relative = PurePosixPath(output_parent_raw)
    output_parent, owned_output_directories = _create_directory_chain(
        batch.REPO_ROOT,
        output_parent_relative,
    )
    run_id = str(controller_binding["controller_run_id"])
    if len(run_id) != 32 or any(item not in "0123456789abcdef" for item in run_id):
        raise batch.AdapterError("teacher_final_controller_run_id_invalid")
    final_root = output_parent / f"candidate-{run_id}"
    staging = output_parent / f".candidate-{run_id}.{secrets.token_hex(8)}.staging"
    if final_root.exists() or staging.exists():
        raise batch.AdapterError("teacher_final_create_once_collision")
    _assert_owned_directories(owned_output_directories)
    output_parent_identity = owned_output_directories[-1][1]
    staging.mkdir(parents=False)
    if (
        _directory_identity(
            output_parent,
            reason="teacher_final_output_parent_identity_drift",
        )
        != output_parent_identity
    ):
        raise batch.AdapterError("teacher_final_output_parent_identity_drift")
    staging_stat = staging.lstat()
    staging_identity = _directory_identity(
        staging,
        reason="teacher_final_staging_identity_invalid",
    )
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise batch.AdapterError("teacher_final_staging_identity_invalid")
    owned_staging_directories: list[tuple[Path, tuple[int, int]]] = [
        (staging, staging_identity)
    ]
    for partition_name in ("train", "router"):
        partition_path, partition_owned = _create_directory_chain(
            staging,
            PurePosixPath(partition_name),
        )
        del partition_path
        owned_staging_directories.extend(partition_owned[1:])
    try:
        shard_entries: list[dict[str, Any]] = []
        shard_file_inventory: list[dict[str, Any]] = []
        by_asset = Counter(item.asset for item in payloads)
        indexes: Counter[str] = Counter()
        for payload in payloads:
            index = indexes[payload.asset]
            indexes[payload.asset] += 1
            partition = "train" if payload.asset == "main_train" else "router"
            total = by_asset[payload.asset]
            relative = f"{partition}/teacher-{index:05d}-of-{total:05d}.jsonl"
            path = staging.joinpath(*PurePosixPath(relative).parts)
            _assert_owned_directories(owned_output_directories)
            _assert_owned_directories(owned_staging_directories)
            _exclusive_bytes(path, payload.raw)
            digest = _sha256(payload.raw)
            sidecar = path.with_name(path.name + ".sha256")
            sidecar_raw = _sidecar_bytes(path, digest)
            _exclusive_bytes(sidecar, sidecar_raw)
            first = payload.records[0]
            last = payload.records[-1]
            shard_entries.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "bytes": len(payload.raw),
                    "sidecar_sha256": _sha256(sidecar_raw),
                    "sidecar_bytes": len(sidecar_raw),
                    "records": len(payload.records),
                    "asset": payload.asset,
                    "first_record_id_sha256": first["record_id_sha256"],
                    "last_record_id_sha256": last["record_id_sha256"],
                    "first_task_bundle_sha256": first["task_bundle_sha256"],
                    "last_task_bundle_sha256": last["task_bundle_sha256"],
                }
            )
            shard_file_inventory.extend(
                [
                    _file_entry(staging, path, kind="teacher_shard"),
                    _file_entry(staging, sidecar, kind="sha256_sidecar"),
                ]
            )

        order_sha = _canonical_sha256([item["record_id_sha256"] for item in records])
        target_sha = _canonical_sha256(
            [item["teacher_target_sha256"] for item in records]
        )
        join_sha = _canonical_sha256(
            [
                {
                    key: item[key]
                    for key in (
                        "record_id_sha256",
                        "source_content_sha256",
                        "source_serialization_identity_sha256",
                        "task_bundle_sha256",
                        "source_asset",
                    )
                }
                for item in records
            ]
        )
        shard_inventory_sha = _canonical_sha256(shard_entries)
        logical_dataset_sha = _canonical_sha256(
            {
                "record_order_sha256": order_sha,
                "target_projection_sha256": target_sha,
                "source_join_sha256": join_sha,
                "shard_inventory_sha256": shard_inventory_sha,
            }
        )
        identity_audit = {
            "stable_identity_sentence": AIR_IDENTITY_SENTENCE,
            **producer_identity_facts,
            "teacher_train_bundles": 20,
            "teacher_train_records": 100,
            **scoped_read_claims,
            "train_target_consistency_passed": True,
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "artifact_version": ARTIFACT_VERSION,
            "lifecycle_status": ("candidate_pending_independent_release_review"),
            "record_schema": {
                "path": record_schema_item["path"],
                "sha256": record_schema_item["sha256"],
                "bytes": record_schema_item["bytes"],
            },
            "source": dict(config["source"]["release_identity"]),
            "controller": {
                **dict(controller_identity),
                "finalizer_config_sha256": config_sha256,
                "finalizer_implementation_sha256": config["implementation_sha256"],
            },
            "consumer": dict(config["consumer"]),
            "counts": {
                "records": 3520,
                "main_records": 3440,
                "router_records": 80,
                "accepted": 3520,
                "rejected": 0,
                "uncertain": 0,
                "train_identity_bundles": 20,
                "train_identity_records": 100,
                "source_assets": EXPECTED_ASSET_COUNTS,
            },
            "identity_audit": identity_audit,
            "shards": shard_entries,
            "logical_identity": {
                "record_order_sha256": order_sha,
                "target_projection_sha256": target_sha,
                "source_join_sha256": join_sha,
                "shard_inventory_sha256": shard_inventory_sha,
                "logical_dataset_sha256": logical_dataset_sha,
            },
            "claims": {
                "candidate": True,
                "final": False,
                "independent_release_review": "pending",
                "training_authorized": False,
                "formal_training_authorized": False,
                "live_authorized": False,
                "hidden_reasoning_retained": False,
                "secret_material_retained": False,
                "provider_body_retained": False,
                "eval_proxy_in_training_shards": False,
            },
        }
        manifest_schema_item = config["schemas"]["manifest"]
        manifest_validator = _schema(
            _relative_file(
                batch.REPO_ROOT,
                manifest_schema_item["path"],
                reason="teacher_final_manifest_schema_path_invalid",
            ),
            manifest_schema_item["sha256"],
        )
        _validate_schema(
            manifest_validator,
            manifest,
            reason="teacher_final_manifest_schema_mismatch",
        )
        manifest_path = staging / "manifest.json"
        manifest_raw = _canonical_bytes(manifest, newline=True)
        _assert_owned_directories(owned_output_directories)
        _assert_owned_directories(owned_staging_directories)
        _exclusive_bytes(manifest_path, manifest_raw)
        manifest_sidecar = staging / "manifest.json.sha256"
        _exclusive_bytes(
            manifest_sidecar,
            _sidecar_bytes(manifest_path, _sha256(manifest_raw)),
        )
        inventory_entries = [
            *shard_file_inventory,
            _file_entry(staging, manifest_path, kind="manifest"),
            _file_entry(staging, manifest_sidecar, kind="sha256_sidecar"),
        ]
        output_inventory = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-final-output-inventory.v2"
            ),
            "namespace": NAMESPACE,
            "files": inventory_entries,
            "payload_tree_sha256": _canonical_sha256(inventory_entries),
        }
        output_inventory_path = staging / "output_inventory.json"
        output_inventory_raw = _canonical_bytes(output_inventory, newline=True)
        _assert_owned_directories(owned_output_directories)
        _assert_owned_directories(owned_staging_directories)
        _exclusive_bytes(output_inventory_path, output_inventory_raw)
        output_inventory_sidecar = staging / "output_inventory.json.sha256"
        _exclusive_bytes(
            output_inventory_sidecar,
            _sidecar_bytes(
                output_inventory_path,
                _sha256(output_inventory_raw),
            ),
        )
        receipt_payload = {
            "schema_version": BUILD_RECEIPT_SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "status": ("materialized_candidate_pending_independent_release_review"),
            "controller_run_id": run_id,
            "manifest_sha256": _sha256(manifest_raw),
            "manifest_bytes": len(manifest_raw),
            "output_inventory_sha256": _sha256(output_inventory_raw),
            "payload_tree_sha256": output_inventory["payload_tree_sha256"],
            "source_identity_sha256": inventory.source_identity_sha256,
            "phase_receipt_inventory_sha256": phase_inventory_sha,
            "wal_chain_tip_sha256": wal_tip_sha,
            "counts": {
                "records": 3520,
                "main_records": 3440,
                "router_records": 80,
                "accepted": 3520,
                "rejected": 0,
                "uncertain": 0,
                "train_identity_bundles": 20,
                "train_identity_records": 100,
            },
            "identity_audit_sha256": _canonical_sha256(identity_audit),
            "runtime_authority": {
                "same_process_hmac_authenticated": True,
                "three_phase_receipts_authenticated": True,
                "wal_terminally_authenticated": True,
                "runtime_hmac_verification_scope": "same_process_only",
                "runtime_hmac_key_persisted": False,
                "runtime_hmac_key_public": False,
            },
            "claims": {
                "candidate": True,
                "final": False,
                "external_release_review_required": True,
                "training_authorized": False,
                "formal_training_authorized": False,
                "live_authorized": False,
                "content_in_receipt": False,
                "secret_in_receipt": False,
                "provider_body_in_receipt": False,
            },
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt = {
            **receipt_payload,
            "runtime_hmac_sha256": hmac.new(
                runtime_slots.receipt_hmac_key(),
                _canonical_bytes(receipt_payload),
                hashlib.sha256,
            ).hexdigest(),
        }
        receipt_schema_item = config["schemas"]["build_receipt"]
        receipt_validator = _schema(
            _relative_file(
                batch.REPO_ROOT,
                receipt_schema_item["path"],
                reason="teacher_final_receipt_schema_path_invalid",
            ),
            receipt_schema_item["sha256"],
        )
        _validate_schema(
            receipt_validator,
            receipt,
            reason="teacher_final_build_receipt_schema_mismatch",
        )
        receipt_path = staging / "build_receipt.json"
        receipt_raw = _canonical_bytes(receipt, newline=True)
        _assert_owned_directories(owned_output_directories)
        _assert_owned_directories(owned_staging_directories)
        _exclusive_bytes(receipt_path, receipt_raw)
        receipt_sidecar = staging / "build_receipt.json.sha256"
        _exclusive_bytes(
            receipt_sidecar,
            _sidecar_bytes(receipt_path, _sha256(receipt_raw)),
        )
        request = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-final-release-request.v2"
            ),
            "namespace": NAMESPACE,
            "status": "pending_different_reviewer",
            "candidate_manifest_sha256": _sha256(manifest_raw),
            "candidate_build_receipt_sha256": _sha256(receipt_raw),
            "candidate_output_inventory_sha256": _sha256(output_inventory_raw),
            "candidate_payload_tree_sha256": output_inventory["payload_tree_sha256"],
            "required_binding_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2"
            ),
            "implementer_may_sign": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
        }
        request_path = staging / "release_attestation.request.json"
        request_raw = _canonical_bytes(request, newline=True)
        _assert_owned_directories(owned_output_directories)
        _assert_owned_directories(owned_staging_directories)
        _exclusive_bytes(request_path, request_raw)
        request_sidecar = staging / "release_attestation.request.json.sha256"
        _exclusive_bytes(
            request_sidecar,
            _sidecar_bytes(request_path, _sha256(request_raw)),
        )

        publish_snapshots: dict[str, Snapshot] = {}

        def authenticate_publish_boundary() -> None:
            # Recheck every authority and source inside the no-replace
            # publication boundary.
            _assert_owned_directories(owned_output_directories)
            _assert_owned_directories(owned_staging_directories)
            state.authenticate_authority()
            _recheck(alignment_snapshot, reason="teacher_final_alignment_toctou")
            _recheck(receipt_snapshot, reason="teacher_final_receipt_toctou")
            for source_snapshot in source_snapshots:
                _recheck(source_snapshot, reason="teacher_final_source_toctou")
            _recheck(config_snapshot, reason="teacher_final_config_toctou")
            for path in staging.rglob("*"):
                physical = path.lstat()
                attributes = int(getattr(physical, "st_file_attributes", 0))
                if stat.S_ISLNK(physical.st_mode) or bool(attributes & 0x400):
                    raise batch.AdapterError("teacher_final_staging_reparse_forbidden")
                if stat.S_ISREG(physical.st_mode):
                    _reject_reparse_chain(path, stop=staging)
                    if physical.st_size >= 50 * 1024 * 1024:
                        raise batch.AdapterError("teacher_final_file_size_limit")
                    relative = path.relative_to(staging).as_posix()
                    publish_snapshots[relative] = _snapshot(
                        path,
                        reason="teacher_final_prepublish_snapshot_drift",
                    )

        try:
            _rename_noreplace(
                staging,
                final_root,
                before_rename=authenticate_publish_boundary,
            )
        except FileExistsError as error:
            raise batch.AdapterError("teacher_final_create_once_collision") from error
        if (
            _directory_identity(
                final_root,
                reason="teacher_final_published_root_identity_drift",
            )
            != staging_identity
        ):
            raise batch.AdapterError("teacher_final_published_root_identity_drift")
        _reject_reparse_chain(final_root, stop=output_parent)
        for relative, before_publish in publish_snapshots.items():
            published_path = final_root.joinpath(*PurePosixPath(relative).parts)
            observed = _snapshot(
                published_path,
                expected_sha256=before_publish.sha256,
                expected_bytes=len(before_publish.raw),
                reason="teacher_final_published_file_drift",
            )
            if observed.identity != before_publish.identity:
                raise batch.AdapterError("teacher_final_published_file_identity_drift")
        state.authenticate_authority()
        _recheck(alignment_snapshot, reason="teacher_final_alignment_toctou")
        _recheck(receipt_snapshot, reason="teacher_final_receipt_toctou")
        for source_snapshot in source_snapshots:
            _recheck(source_snapshot, reason="teacher_final_source_toctou")
        return {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-finalizer-status.v1"
            ),
            "state": "candidate_pending_independent_release_review",
            "artifact_root": final_root.relative_to(batch.REPO_ROOT).as_posix(),
            "manifest_sha256": _sha256(manifest_raw),
            "build_receipt_sha256": _sha256(receipt_raw),
            "output_inventory_sha256": _sha256(output_inventory_raw),
            "payload_tree_sha256": output_inventory["payload_tree_sha256"],
            "record_schema_sha256": record_schema_item["sha256"],
            "records": 3520,
            "main_records": 3440,
            "router_records": 80,
            "train_identity_bundles": 20,
            "train_identity_records": 100,
            **scoped_read_claims,
            "phase_receipts": len(phase_rows),
            "runtime_hmac_consumed_before_close": True,
            "external_release_review_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "provider_requests": 0,
            "model_requests": 0,
            "gpu_requests": 0,
            "network_requests": 0,
        }
    except BaseException:
        # Preserve unpublished staging for forensic diagnosis.  Recursive
        # deletion is intentionally forbidden because an attacker could swap
        # the directory after a failed publish.
        raise
