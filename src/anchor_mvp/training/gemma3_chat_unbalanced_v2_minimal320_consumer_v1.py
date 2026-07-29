"""Model-free additive consumer preflight for the minimal-320 handoff.

The default checked-in config is intentionally not live-ready.  A reviewed
consumer-config revision must freeze the Producer P/R Git identities and the
physical release documents before this module can authenticate anything.  This
module performs no provider, model, GPU, or protected-dataset-body access.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT: Final = Path(__file__).resolve().parents[3]
CONFIG_PATH: Final = (
    ROOT
    / "configs"
    / "training"
    / "gemma3_chat_unbalanced_v2_minimal320_consumer_v1.json"
)
CONFIG_SCHEMA_PATH: Final = CONFIG_PATH.with_name(
    "gemma3_chat_unbalanced_v2_minimal320_consumer_v1.schema.json"
)
CONFIG_SCHEMA_SHA256: Final = (
    "b4a70302274b1587fa1e5af285382bc9a7e9b9e439c9ef9ddcb6e20801a9a86b"
)
CONFIG_TEMPLATE_SHA256: Final = (
    "dc1f7783449c6c35b70664426931281a67ca4143e16cf3cb2fbec6b756dae800"
)
CONFIG_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.minimal320-consumer-config.v1"
)
RELEASE_PIN_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.minimal320-producer-release-pin.v1"
)
PRODUCER_BINDING_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.minimal320-producer-final-binding.v1"
)
PREFLIGHT_RECEIPT_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.minimal320-consumer-preflight-receipt.v1"
)
STATUS_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.minimal320-consumer-status.v1"
)

_MAX_CONFIG_BYTES: Final = 1_000_000
_MAX_RELEASE_BYTES: Final = 4_000_000
_MAX_INVENTORY_BYTES: Final = 64_000_000
_MAX_ARTIFACT_BYTES: Final = 2_000_000_000
_MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE: Final = 50 * 1024 * 1024
_FORBIDDEN_PLACEHOLDERS: Final = (
    "TO_BE_PINNED",
    "TBD",
    "PLACEHOLDER",
)
_ROLE_TO_ASSET: Final = {
    "angry": "angry_style",
    "humor": "humor",
    "review": "review_audit",
    "router": "planner_router",
    "serious": "serious",
    "tool": "tool_call",
}
_IDENTITY_ROLE_TO_ASSET: Final = {
    "angry": "angry_style",
    "humor": "humor",
    "review": "review_audit",
    "serious": "serious",
    "tool": "tool_call",
}
_EXPERT_ROLES: Final = ("humor", "serious", "angry", "tool", "review")
_COMPONENT_PIN_TO_KIND: Final = {
    "generation_implementation_sha256": "generation_implementation",
    "selector_implementation_sha256": "selector_implementation",
    "selector_contract_sha256": "selector_contract",
    "selector_schema_sha256": "selector_schema",
    "teacher_record_schema_sha256": "teacher_record_schema",
    "source_truth_contract_sha256": "source_truth_contract",
}
_REQUIRED_METADATA_KINDS: Final = (
    "selection_manifest",
    "selected_candidate_inventory",
    "accepted_inventory",
    "rejected_inventory",
    "quarantine_inventory",
    "training_record_inventory",
    "checkpoint_manifest",
    "committed_shards_manifest",
    "freeze_attestation",
    "external_attestation",
)
_SELECTOR_CANDIDATE_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "selector-candidate.v2"
)
_SELECTED_ROW_KEYS: Final = frozenset(
    {
        "schema_version",
        "candidate_id_sha256",
        "idempotency_key",
        "teacher_request_idempotency_key",
        "source_record_id_sha256",
        "source_content_sha256",
        "source_serialization_identity_sha256",
        "teacher_request_record_id_sha256",
        "parent_source_record_id_sha256",
        "task_bundle_sha256",
        "parent_task_bundle_sha256",
        "source_semantic_sha256",
        "overlay_row_sha256",
        "identity_derivative_spec_sha256",
        "selection_kind",
        "output_expert_role",
        "training_asset",
        "validation_contract_role",
        "language",
        "tool_family",
        "identity_class",
        "identity_parent_class",
        "identity_provenance_intent",
        "router_label",
        "review_verdict",
        "review_fault_type",
        "content_retained",
    }
)
_ACCEPTED_INVENTORY_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "body-free-inventory-row.v1"
)
_ACCEPTED_ROW_KEYS: Final = frozenset(
    {
        "schema_version",
        "idempotency_key",
        "outcome",
        "reason_code",
        "receipt_sha256",
        "output_record_sha256",
        "stratum",
        "content_retained",
    }
)
_STRATUM_KEYS: Final = frozenset(
    {
        "role",
        "validation_contract_role",
        "language",
        "tool_family",
        "identity_class",
        "identity_parent_class",
        "selection_kind",
        "review_verdict",
        "review_fault_type",
        "router_label",
        "source_record_id_sha256",
    }
)
_TEACHER_RECORD_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
)
_TEACHER_RECORD_NAMESPACE: Final = "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
_TEACHER_RECORD_KEYS: Final = frozenset(
    {
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
    }
)
_TRAINING_RECORD_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "training-record-inventory-row.v1"
)
_TRAINING_RECORD_JOIN_KEYS: Final = frozenset(
    {
        "schema_version",
        "training_ordinal",
        "selected_ordinal",
        "candidate_id_sha256",
        "selected_candidate_sha256",
        "idempotency_key",
        "source_record_id_sha256",
        "source_content_sha256",
        "source_serialization_identity_sha256",
        "task_bundle_sha256",
        "source_asset",
        "decision",
        "receipt_sha256",
        "output_record_sha256",
        "teacher_target_sha256",
        "teacher_record_sha256",
        "content_retained",
        "raw_token_ids_retained",
    }
)
_COMMITTED_SHARDS_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
    "committed-shards-manifest.v2"
)
_IDENTITY_TEMPLATES: Final = {
    "air_attribution": "我是由Air训练的测试模型。",
    "false_google_attribution": "我是由Air训练的测试模型。 我不是由 Google 训练的。",
    "false_openai_attribution": "我是由Air训练的测试模型。 我不是由 OpenAI 训练的。",
}
_IDENTITY_BUNDLE_CELLS: Final = {
    ("air_attribution", "en"): 4,
    ("air_attribution", "zh"): 4,
    ("false_google_attribution", "en"): 3,
    ("false_google_attribution", "zh"): 3,
    ("false_openai_attribution", "en"): 3,
    ("false_openai_attribution", "zh"): 3,
}
_ROUTER_RECORD_CELLS: Final = {
    ("humor", "en"): 14,
    ("humor", "zh"): 13,
    ("serious", "en"): 13,
    ("serious", "zh"): 13,
    ("angry", "en"): 13,
    ("angry", "zh"): 14,
}
_REVIEW_FAULT_COUNTS: Final = {
    "format_schema": 2,
    "tool_argument": 2,
    "evidence_mismatch": 2,
    "grounding_mismatch": 2,
    "routing_scope": 2,
}
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9._/-]+$")
_CONSUMER_CONFIG_PATH: Final = (
    "configs/training/gemma3_chat_unbalanced_v2_minimal320_consumer_v1.json"
)
_CONSUMER_SCHEMA_PATHS: Final = {
    "consumer_config": (
        "configs/training/gemma3_chat_unbalanced_v2_minimal320_consumer_v1.schema.json"
    ),
    "preflight_receipt": (
        "configs/training/"
        "gemma3_chat_unbalanced_v2_minimal320_preflight_receipt_v1.schema.json"
    ),
    "producer_final_binding": (
        "configs/training/"
        "gemma3_chat_unbalanced_v2_minimal320_producer_final_binding_v1.schema.json"
    ),
    "producer_release_pin": (
        "configs/training/"
        "gemma3_chat_unbalanced_v2_minimal320_producer_release_pin_v1.schema.json"
    ),
}
_CONSUMER_RUNTIME_PATHS: Final = (
    "configs/training/gemma3_chat_unbalanced_v2_minimal320_consumer_v1.schema.json",
    "configs/training/gemma3_chat_unbalanced_v2_minimal320_preflight_receipt_v1.schema.json",
    "configs/training/gemma3_chat_unbalanced_v2_minimal320_producer_final_binding_v1.schema.json",
    "configs/training/gemma3_chat_unbalanced_v2_minimal320_producer_release_pin_v1.schema.json",
    "src/anchor_mvp/training/gemma3_chat_unbalanced_v2_minimal320_consumer_v1.py",
    "tests/test_gemma3_chat_unbalanced_v2_minimal320_consumer_v1.py",
)
_PRODUCER_L_PATHS: Final = tuple(
    sorted(
        (
            ".gitattributes",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_contract_v2.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_metadata_overlay_v2.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_preimages_v2.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_selector_v2.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "vnext_checkpoint_v1.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_contract_v1.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "vnext_external_attestation_v1.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "vnext_producer_release_v1.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "vnext_receipt_v1.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
            "vnext_transition_v1.schema.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json",
            "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.schema.json",
            "scripts/data/run_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
            "scripts/observability/distillation_dashboard.py",
            "scripts/research/benchmark_gemma3_teacher_alignment_vnext_wal_v1.py",
            "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
            "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
            "tests/test_distillation_dashboard.py",
            "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
            "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
        )
    )
)
_PRODUCER_RELEASE_ROOT: Final = (
    "data/gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "teacher_alignment_vnext_v1/release_v1"
)
_PRODUCER_P_NAMES: Final = (
    "accepted_inventory.jsonl",
    "accepted_inventory.jsonl.sha256",
    "authenticated_metadata_overlay.jsonl",
    "authenticated_metadata_overlay.jsonl.sha256",
    "authenticated_metadata_overlay_manifest.json",
    "authenticated_metadata_overlay_manifest.json.sha256",
    "candidate_inventory.jsonl",
    "candidate_inventory.jsonl.sha256",
    "checkpoint_manifest.json",
    "checkpoint_manifest.json.sha256",
    "checkpoints_inventory.json",
    "checkpoints_inventory.json.sha256",
    "committed_shards_manifest.json",
    "committed_shards_manifest.json.sha256",
    "producer_launch_inventory.json",
    "producer_launch_inventory.json.sha256",
    "quarantine_inventory.jsonl",
    "quarantine_inventory.jsonl.sha256",
    "rejected_inventory.jsonl",
    "rejected_inventory.jsonl.sha256",
    "selected_inventory.jsonl",
    "selected_inventory.jsonl.sha256",
    "selection_manifest.json",
    "selection_manifest.json.sha256",
    "selection_preimage.json",
    "selection_preimage.json.sha256",
    "selector_preimage_manifest.json",
    "selector_preimage_manifest.json.sha256",
    "teacher_training_shard-00000.jsonl",
    "teacher_training_shard-00000.jsonl.sha256",
    "training_record_inventory.jsonl",
    "training_record_inventory.jsonl.sha256",
)
_PRODUCER_R_NAMES: Final = (
    "external_attestation.json",
    "external_attestation.json.sha256",
    "freeze_attestation.json",
    "freeze_attestation.json.sha256",
)
_PRODUCER_P_PATHS: Final = tuple(
    f"{_PRODUCER_RELEASE_ROOT}/{name}" for name in _PRODUCER_P_NAMES
)
_PRODUCER_R_PATHS: Final = tuple(
    f"{_PRODUCER_RELEASE_ROOT}/{name}" for name in _PRODUCER_R_NAMES
)
_TEACHER_SHARD_PATH: Final = (
    f"{_PRODUCER_RELEASE_ROOT}/teacher_training_shard-00000.jsonl"
)
_TRAINING_RECORD_INVENTORY_PATH: Final = (
    f"{_PRODUCER_RELEASE_ROOT}/training_record_inventory.jsonl"
)
_COMMITTED_SHARDS_MANIFEST_PATH: Final = (
    f"{_PRODUCER_RELEASE_ROOT}/committed_shards_manifest.json"
)
_EXACT_METADATA_PATHS: Final = {
    "selection_manifest": f"{_PRODUCER_RELEASE_ROOT}/selection_manifest.json",
    "selected_candidate_inventory": (
        f"{_PRODUCER_RELEASE_ROOT}/selected_inventory.jsonl"
    ),
    "accepted_inventory": f"{_PRODUCER_RELEASE_ROOT}/accepted_inventory.jsonl",
    "rejected_inventory": f"{_PRODUCER_RELEASE_ROOT}/rejected_inventory.jsonl",
    "quarantine_inventory": (f"{_PRODUCER_RELEASE_ROOT}/quarantine_inventory.jsonl"),
    "training_record_inventory": _TRAINING_RECORD_INVENTORY_PATH,
    "checkpoint_manifest": f"{_PRODUCER_RELEASE_ROOT}/checkpoint_manifest.json",
    "committed_shards_manifest": _COMMITTED_SHARDS_MANIFEST_PATH,
    "freeze_attestation": f"{_PRODUCER_RELEASE_ROOT}/freeze_attestation.json",
    "external_attestation": (f"{_PRODUCER_RELEASE_ROOT}/external_attestation.json"),
}


class Minimal320ConsumerError(RuntimeError):
    """Body-free fail-closed consumer error."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class LoadedConfig:
    value: dict[str, Any]
    raw_sha256: str


@dataclass(frozen=True)
class ConsumerGitIdentity:
    head: str
    tree: str
    runtime_commit: str
    runtime_tree: str
    implementation_sha256: str
    schema_inventory_sha256: str


@dataclass(frozen=True)
class AuthenticatedRelease:
    trust_anchor: dict[str, Any]
    pin: dict[str, Any]
    pin_sha256: str
    binding: dict[str, Any]
    binding_sha256: str
    consumer_git: ConsumerGitIdentity


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _merkle_root(rows: Sequence[Mapping[str, Any]]) -> str:
    leaves = [
        hashlib.sha256(b"leaf\x00" + _canonical_bytes(row)).digest() for row in rows
    ]
    if not leaves:
        return hashlib.sha256(b"empty\x00").hexdigest()
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(b"node\x00" + leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


_PRODUCER_L_PATH_ROOT: Final = _canonical_sha256(list(_PRODUCER_L_PATHS))
_PRODUCER_P_PATH_ROOT: Final = _canonical_sha256(list(_PRODUCER_P_PATHS))
_PRODUCER_R_PATH_ROOT: Final = _canonical_sha256(list(_PRODUCER_R_PATHS))


def _identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _stable_bound_read(
    path: Path,
    *,
    limit: int,
    reason: str,
    allow_empty: bool = False,
) -> tuple[bytes, tuple[int, int, int, int]]:
    try:
        before = path.lstat()
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Minimal320ConsumerError(reason)
    if before.st_size < (0 if allow_empty else 1) or before.st_size > limit:
        raise Minimal320ConsumerError(reason)
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            raw = handle.read(limit + 1)
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error
    if (
        len(raw) > limit
        or _identity(before) != _identity(opened_before)
        or _identity(opened_before) != _identity(opened_after)
        or _identity(opened_after) != _identity(after)
    ):
        raise Minimal320ConsumerError(reason)
    return raw, _identity(after)


def _stable_read(path: Path, *, limit: int, reason: str) -> bytes:
    return _stable_bound_read(path, limit=limit, reason=reason)[0]


def _stable_file_digest(
    path: Path,
    *,
    reason: str,
) -> tuple[int, str, str, tuple[int, int, int, int], int]:
    try:
        before = path.lstat()
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Minimal320ConsumerError(reason)
    if before.st_size < 1 or before.st_size > _MAX_ARTIFACT_BYTES:
        raise Minimal320ConsumerError(reason)
    digest = hashlib.sha256()
    blob_digest = hashlib.sha1(usedforsecurity=False)
    blob_digest.update(f"blob {before.st_size}\0".encode("ascii"))
    line_count = 0
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                blob_digest.update(chunk)
                line_count += chunk.count(b"\n")
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error
    if (
        _identity(before) != _identity(opened_before)
        or _identity(opened_before) != _identity(opened_after)
        or _identity(opened_after) != _identity(after)
    ):
        raise Minimal320ConsumerError(reason)
    return (
        before.st_size,
        digest.hexdigest(),
        blob_digest.hexdigest(),
        _identity(after),
        line_count,
    )


def _strict_json_value(raw: bytes, *, reason: str) -> Any:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Minimal320ConsumerError(reason)

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Minimal320ConsumerError(reason)
            value[key] = item
        return value

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                Minimal320ConsumerError(reason)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Minimal320ConsumerError(reason) from error
    return value


def _strict_json(raw: bytes, *, reason: str) -> dict[str, Any]:
    value = _strict_json_value(raw, reason=reason)
    if not isinstance(value, dict):
        raise Minimal320ConsumerError(reason)
    return value


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        upper = value.upper()
        return any(token in upper for token in _FORBIDDEN_PLACEHOLDERS)
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _contains_zero_identity(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) in {40, 64} and set(value) == {"0"}
    if isinstance(value, Mapping):
        return any(_contains_zero_identity(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_zero_identity(item) for item in value)
    return False


def _validate_schema(
    schema: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    try:
        Draft202012Validator.check_schema(dict(schema))
        Draft202012Validator(dict(schema)).validate(dict(instance))
    except (SchemaError, ValidationError) as error:
        raise Minimal320ConsumerError(reason) from error


def _reject_link_chain(base: Path, parts: Sequence[str], *, reason: str) -> None:
    cursor = base
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for part in parts:
        cursor = cursor / part
        try:
            observed = cursor.lstat()
        except OSError as error:
            raise Minimal320ConsumerError(reason) from error
        if stat.S_ISLNK(observed.st_mode) or (
            getattr(observed, "st_file_attributes", 0) & reparse_flag
        ):
            raise Minimal320ConsumerError(reason)


def _normalized_relative(raw: Any, *, reason: str) -> PurePosixPath:
    if (
        not isinstance(raw, str)
        or not raw
        or not _SAFE_PATH_RE.fullmatch(raw)
        or "\\" in raw
        or raw.startswith("/")
        or raw.endswith("/")
        or "//" in raw
    ):
        raise Minimal320ConsumerError(reason)
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise Minimal320ConsumerError(reason)
    return pure


def _reject_absolute_reparse_chain(path: Path, *, reason: str) -> Path:
    lexical = path.absolute()
    parts = lexical.parts
    if not parts:
        raise Minimal320ConsumerError(reason)
    cursor = Path(parts[0])
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for part in parts[1:]:
        cursor /= part
        try:
            observed = cursor.lstat()
        except OSError as error:
            raise Minimal320ConsumerError(reason) from error
        if stat.S_ISLNK(observed.st_mode) or (
            getattr(observed, "st_file_attributes", 0) & reparse_flag
        ):
            raise Minimal320ConsumerError(reason)
    try:
        return lexical.resolve(strict=True)
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error


def _safe_root(path: Path, *, reason: str) -> Path:
    resolved = _reject_absolute_reparse_chain(path, reason=reason)
    try:
        observed = resolved.lstat()
    except OSError as error:
        raise Minimal320ConsumerError(reason) from error
    if not stat.S_ISDIR(observed.st_mode):
        raise Minimal320ConsumerError(reason)
    return resolved


def _safe_join(root: Path, raw: Any, *, reason: str) -> Path:
    pure = _normalized_relative(raw, reason=reason)
    _reject_link_chain(root, pure.parts, reason=reason)
    lexical = root.joinpath(*pure.parts)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Minimal320ConsumerError(reason) from error
    return resolved


def _repo_path(raw: Any, *, reason: str) -> Path:
    root_resolved = _safe_root(ROOT, reason=reason)
    return _safe_join(root_resolved, raw, reason=reason)


def _relative_to_document(
    document_path: Path,
    raw: Any,
    *,
    reason: str,
) -> Path:
    base = _safe_root(document_path.parent, reason=reason)
    return _safe_join(base, raw, reason=reason)


def _git(
    repo: Path,
    arguments: Sequence[str],
    *,
    reason: str,
    check: bool = True,
) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise Minimal320ConsumerError(reason)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "HOME",
            "LOCALAPPDATA",
            "PATH",
            "PATHEXT",
            "SYSTEMDRIVE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "WINDIR",
        }
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            [executable, "-C", str(repo), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Minimal320ConsumerError(reason) from error
    if check and completed.returncode != 0:
        raise Minimal320ConsumerError(reason)
    return completed.stdout


def _git_text(repo: Path, arguments: Sequence[str], *, reason: str) -> str:
    try:
        return (
            _git(repo, arguments, reason=reason)
            .decode("utf-8", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as error:
        raise Minimal320ConsumerError(reason) from error


def _git_tree_inventory(
    repo: Path,
    commit: str,
    *,
    reason: str,
) -> dict[str, tuple[str, str, str]]:
    raw = _git(
        repo,
        ["ls-tree", "-rz", "--full-tree", commit],
        reason=reason,
    )
    result: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode_raw, kind_raw, oid_raw = metadata.split(b" ", 2)
            path = path_raw.decode("utf-8", errors="strict")
            mode = mode_raw.decode("ascii")
            kind = kind_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise Minimal320ConsumerError(reason) from error
        _normalized_relative(path, reason=reason)
        if path in result:
            raise Minimal320ConsumerError(reason)
        result[path] = (mode, kind, oid)
    return result


def _git_commit_parent(
    repo: Path,
    commit: str,
    *,
    reason: str,
) -> str:
    raw = _git(repo, ["cat-file", "-p", commit], reason=reason)
    header = raw.split(b"\n\n", 1)[0]
    try:
        parents = [
            line.removeprefix(b"parent ").decode("ascii", errors="strict")
            for line in header.splitlines()
            if line.startswith(b"parent ")
        ]
    except UnicodeDecodeError as error:
        raise Minimal320ConsumerError(reason) from error
    if len(parents) != 1 or _GIT_OBJECT_RE.fullmatch(parents[0]) is None:
        raise Minimal320ConsumerError(reason)
    return parents[0]


def _git_name_status(
    repo: Path,
    parent: str,
    child: str,
    *,
    reason: str,
) -> tuple[tuple[str, str], ...]:
    raw = _git(
        repo,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            "--no-renames",
            parent,
            child,
        ],
        reason=reason,
    )
    fields = raw.split(b"\0")
    if fields[-1:] != [b""] or (len(fields) - 1) % 2:
        raise Minimal320ConsumerError(reason)
    rows: list[tuple[str, str]] = []
    observed: set[str] = set()
    for offset in range(0, len(fields) - 1, 2):
        try:
            status_text = fields[offset].decode("ascii", errors="strict")
            path_text = fields[offset + 1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Minimal320ConsumerError(reason) from error
        path_text = _normalized_relative(path_text, reason=reason).as_posix()
        if status_text not in {"A", "M"} or path_text in observed:
            raise Minimal320ConsumerError(reason)
        observed.add(path_text)
        rows.append((path_text, status_text))
    return tuple(sorted(rows))


def _verify_b_l_p_r_stage_chain(
    repo: Path,
    stages: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[str, str, str]],
    dict[str, tuple[str, str, str]],
    dict[str, tuple[str, str, str]],
]:
    """Authenticate exact B→L21→P32→R4 Git lineage without trusting labels."""

    if set(stages) != {
        "audited_base",
        "producer_launch_L",
        "producer_payload_P",
        "producer_final_R",
        "global_clean_tree",
        "unique_direct_children",
        "exact_raw_diff_allowlist",
        "release_self_pin_present",
    }:
        raise Minimal320ConsumerError("minimal320_producer_stage_shape_invalid")
    base = stages["audited_base"]
    launch = stages["producer_launch_L"]
    payload = stages["producer_payload_P"]
    final = stages["producer_final_R"]
    if (
        not all(isinstance(value, Mapping) for value in (base, launch, payload, final))
        or stages["global_clean_tree"] is not True
        or stages["unique_direct_children"] is not True
        or stages["exact_raw_diff_allowlist"] is not True
        or stages["release_self_pin_present"] is not False
        or launch.get("direct_parent") != base.get("commit")
        or payload.get("direct_parent") != launch.get("commit")
        or final.get("direct_parent") != payload.get("commit")
        or launch.get("path_count") != 21
        or payload.get("path_count") != 32
        or final.get("path_count") != 4
        or launch.get("path_set_root_sha256") != _PRODUCER_L_PATH_ROOT
        or payload.get("path_set_root_sha256") != _PRODUCER_P_PATH_ROOT
        or final.get("path_set_root_sha256") != _PRODUCER_R_PATH_ROOT
    ):
        raise Minimal320ConsumerError("minimal320_producer_stage_contract_drift")
    commits = [
        str(base.get("commit")),
        str(launch.get("commit")),
        str(payload.get("commit")),
        str(final.get("commit")),
    ]
    if (
        any(_GIT_OBJECT_RE.fullmatch(value) is None for value in commits)
        or len(set(commits)) != 4
    ):
        raise Minimal320ConsumerError("minimal320_producer_stage_commit_invalid")
    for value, stage in zip(commits, (base, launch, payload, final), strict=True):
        if _git_text(
            repo,
            ["cat-file", "-t", value],
            reason="minimal320_producer_stage_commit_invalid",
        ) != "commit" or _git_text(
            repo,
            ["rev-parse", f"{value}^{{tree}}"],
            reason="minimal320_producer_stage_tree_invalid",
        ) != stage.get("tree"):
            raise Minimal320ConsumerError("minimal320_producer_stage_tree_invalid")
    if (
        _git_commit_parent(
            repo,
            commits[1],
            reason="minimal320_producer_launch_parent_invalid",
        )
        != commits[0]
        or _git_commit_parent(
            repo,
            commits[2],
            reason="minimal320_producer_payload_parent_invalid",
        )
        != commits[1]
        or _git_commit_parent(
            repo,
            commits[3],
            reason="minimal320_producer_final_parent_invalid",
        )
        != commits[2]
    ):
        raise Minimal320ConsumerError("minimal320_producer_stage_parent_drift")
    expected_launch = tuple(
        sorted(
            (path, "M" if path == ".gitattributes" else "A")
            for path in _PRODUCER_L_PATHS
        )
    )
    expected_payload = tuple((path, "A") for path in _PRODUCER_P_PATHS)
    expected_final = tuple((path, "A") for path in _PRODUCER_R_PATHS)
    if (
        _git_name_status(
            repo,
            commits[0],
            commits[1],
            reason="minimal320_producer_launch_diff_invalid",
        )
        != expected_launch
        or _git_name_status(
            repo,
            commits[1],
            commits[2],
            reason="minimal320_producer_payload_diff_invalid",
        )
        != expected_payload
        or _git_name_status(
            repo,
            commits[2],
            commits[3],
            reason="minimal320_producer_final_diff_invalid",
        )
        != expected_final
    ):
        raise Minimal320ConsumerError("minimal320_producer_stage_diff_drift")
    if _git_text(
        repo,
        ["rev-parse", "HEAD"],
        reason="minimal320_producer_git_head_invalid",
    ) != commits[3] or _git(
        repo,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        reason="minimal320_producer_git_status_invalid",
    ):
        raise Minimal320ConsumerError("minimal320_producer_git_identity_drift")
    launch_inventory = _git_tree_inventory(
        repo,
        commits[1],
        reason="minimal320_producer_launch_tree_inventory_invalid",
    )
    payload_inventory = _git_tree_inventory(
        repo,
        commits[2],
        reason="minimal320_producer_payload_tree_inventory_invalid",
    )
    final_inventory = _git_tree_inventory(
        repo,
        commits[3],
        reason="minimal320_producer_final_tree_inventory_invalid",
    )
    for path in _PRODUCER_L_PATHS:
        launch_blob = launch_inventory.get(path)
        if (
            launch_blob is None
            or launch_blob[0] not in {"100644", "100755"}
            or launch_blob[1] != "blob"
            or payload_inventory.get(path) != launch_blob
            or final_inventory.get(path) != launch_blob
        ):
            raise Minimal320ConsumerError("minimal320_producer_launch_blob_drift")
    for path in _PRODUCER_P_PATHS:
        payload_blob = payload_inventory.get(path)
        if (
            payload_blob is None
            or payload_blob[0] != "100644"
            or payload_blob[1] != "blob"
            or final_inventory.get(path) != payload_blob
        ):
            raise Minimal320ConsumerError("minimal320_producer_payload_blob_drift")
    for primary_name in (
        name for name in _PRODUCER_P_NAMES if not name.endswith(".sha256")
    ):
        primary_path = f"{_PRODUCER_RELEASE_ROOT}/{primary_name}"
        sidecar_path = f"{primary_path}.sha256"
        primary_raw, _primary_identity = _stable_bound_read(
            _safe_join(
                repo,
                primary_path,
                reason="minimal320_producer_payload_path_invalid",
            ),
            limit=_MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE - 1,
            reason="minimal320_producer_payload_size_invalid",
            allow_empty=primary_name
            in {
                "rejected_inventory.jsonl",
                "quarantine_inventory.jsonl",
            },
        )
        sidecar_raw, _sidecar_identity = _stable_bound_read(
            _safe_join(
                repo,
                sidecar_path,
                reason="minimal320_producer_payload_path_invalid",
            ),
            limit=_MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE - 1,
            reason="minimal320_producer_payload_size_invalid",
        )
        expected_sidecar = f"{_sha256(primary_raw)}  {primary_name}\n".encode("ascii")
        if sidecar_raw != expected_sidecar:
            raise Minimal320ConsumerError("minimal320_producer_payload_sidecar_drift")
        for path_text, raw in (
            (primary_path, primary_raw),
            (sidecar_path, sidecar_raw),
        ):
            blob_digest = hashlib.sha1(usedforsecurity=False)
            blob_digest.update(f"blob {len(raw)}\0".encode("ascii"))
            blob_digest.update(raw)
            if payload_inventory[path_text][2] != blob_digest.hexdigest():
                raise Minimal320ConsumerError(
                    "minimal320_producer_payload_physical_blob_drift"
                )
    for path in _PRODUCER_R_PATHS:
        final_blob = final_inventory.get(path)
        if final_blob is None or final_blob[0] != "100644" or final_blob[1] != "blob":
            raise Minimal320ConsumerError("minimal320_producer_final_blob_drift")
    return launch_inventory, payload_inventory, final_inventory


def _require_git_blob(
    inventory: Mapping[str, tuple[str, str, str]],
    path: str,
    expected_oid: str,
    *,
    reason: str,
) -> None:
    observed = inventory.get(path)
    if (
        observed is None
        or observed[0] not in {"100644", "100755"}
        or observed[1] != "blob"
        or observed[2] != expected_oid
    ):
        raise Minimal320ConsumerError(reason)


def _producer_release_documents(
    anchor: Mapping[str, Any],
) -> dict[str, tuple[str, str, str]]:
    fields = (
        (
            "release_pin",
            "release_pin_path",
            "release_pin_sha256",
            "release_pin_blob_oid",
            "minimal320_release_pin_path_invalid",
        ),
        (
            "producer_binding",
            "producer_binding_path",
            "producer_binding_sha256",
            "producer_binding_blob_oid",
            "minimal320_producer_binding_path_invalid",
        ),
        (
            "freeze_attestation",
            "freeze_attestation_path",
            "freeze_attestation_sha256",
            "freeze_attestation_blob_oid",
            "minimal320_freeze_attestation_path_invalid",
        ),
        (
            "external_attestation",
            "external_attestation_path",
            "external_attestation_sha256",
            "external_attestation_blob_oid",
            "minimal320_external_attestation_path_invalid",
        ),
    )
    documents: dict[str, tuple[str, str, str]] = {}
    observed_paths: set[str] = set()
    for kind, path_field, sha_field, blob_field, path_reason in fields:
        path = _normalized_relative(
            anchor.get(path_field),
            reason=path_reason,
        ).as_posix()
        sha256 = anchor.get(sha_field)
        blob_oid = anchor.get(blob_field)
        if (
            path in observed_paths
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or not isinstance(blob_oid, str)
            or _GIT_OBJECT_RE.fullmatch(blob_oid) is None
        ):
            raise Minimal320ConsumerError(
                "minimal320_producer_release_allowlist_invalid"
            )
        observed_paths.add(path)
        documents[kind] = (path, sha256, blob_oid)
    return documents


def _verify_git_repository(
    repo_path: Path,
    anchor: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, tuple[str, str, str]],
    dict[str, tuple[str, str, str]],
]:
    repo = _safe_root(repo_path, reason="minimal320_producer_repo_path_invalid")
    top = _git_text(
        repo,
        ["rev-parse", "--show-toplevel"],
        reason="minimal320_producer_git_repository_invalid",
    )
    try:
        if Path(top).resolve(strict=True) != repo:
            raise Minimal320ConsumerError("minimal320_producer_git_root_mismatch")
    except OSError as error:
        raise Minimal320ConsumerError(
            "minimal320_producer_git_root_mismatch"
        ) from error
    if _git_text(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace/"],
        reason="minimal320_producer_git_replace_ref_invalid",
    ):
        raise Minimal320ConsumerError("minimal320_producer_git_replace_ref_present")
    graft_path = _git_text(
        repo,
        ["rev-parse", "--git-path", "info/grafts"],
        reason="minimal320_producer_git_grafts_invalid",
    )
    graft = Path(graft_path)
    if not graft.is_absolute():
        graft = repo / graft
    if graft.exists():
        try:
            graft_stat = graft.lstat()
        except OSError as error:
            raise Minimal320ConsumerError(
                "minimal320_producer_git_grafts_invalid"
            ) from error
        if (
            stat.S_ISLNK(graft_stat.st_mode)
            or not stat.S_ISREG(graft_stat.st_mode)
            or graft_stat.st_size > 0
        ):
            raise Minimal320ConsumerError("minimal320_producer_git_grafts_present")

    candidate = anchor["candidate_commit"]
    release = anchor["release_commit"]
    for commit in (candidate, release):
        if (
            _git_text(
                repo,
                ["cat-file", "-t", commit],
                reason="minimal320_producer_git_commit_invalid",
            )
            != "commit"
        ):
            raise Minimal320ConsumerError("minimal320_producer_git_commit_invalid")
    candidate_tree = _git_text(
        repo,
        ["rev-parse", f"{candidate}^{{tree}}"],
        reason="minimal320_producer_candidate_tree_invalid",
    )
    release_tree = _git_text(
        repo,
        ["rev-parse", f"{release}^{{tree}}"],
        reason="minimal320_producer_release_tree_invalid",
    )
    if (
        candidate_tree != anchor["candidate_tree"]
        or release_tree != anchor["release_tree"]
    ):
        raise Minimal320ConsumerError("minimal320_producer_git_tree_drift")
    release_object = _git(
        repo,
        ["cat-file", "-p", release],
        reason="minimal320_producer_release_commit_invalid",
    )
    release_header = release_object.split(b"\n\n", 1)[0]
    try:
        release_parents = [
            line.removeprefix(b"parent ").decode("ascii", errors="strict")
            for line in release_header.splitlines()
            if line.startswith(b"parent ")
        ]
    except UnicodeDecodeError as error:
        raise Minimal320ConsumerError(
            "minimal320_producer_release_commit_invalid"
        ) from error
    if release_parents != [candidate]:
        raise Minimal320ConsumerError("minimal320_producer_release_commit_not_direct")
    release_documents = _producer_release_documents(anchor)
    expected_diff = {
        document[0]: document[2] for document in release_documents.values()
    }
    raw_diff = _git(
        repo,
        [
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-renames",
            candidate,
            release,
        ],
        reason="minimal320_producer_release_diff_invalid",
    )
    fields = raw_diff.split(b"\0")
    if (
        fields[-1:] != [b""]
        or (len(fields) - 1) % 2 != 0
        or (len(fields) - 1) // 2 != len(expected_diff)
    ):
        raise Minimal320ConsumerError("minimal320_producer_release_diff_drift")
    observed_paths: set[str] = set()
    zero_oid = "0" * 40
    for offset in range(0, len(fields) - 1, 2):
        try:
            metadata = fields[offset].decode("ascii", errors="strict").split(" ")
            changed_path = fields[offset + 1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Minimal320ConsumerError(
                "minimal320_producer_release_diff_drift"
            ) from error
        changed_path = _normalized_relative(
            changed_path,
            reason="minimal320_producer_release_diff_drift",
        ).as_posix()
        expected_oid = expected_diff.get(changed_path)
        if (
            len(metadata) != 5
            or metadata[0] != ":000000"
            or metadata[1] != "100644"
            or metadata[2] != zero_oid
            or metadata[3] != expected_oid
            or metadata[4] != "A"
            or changed_path in observed_paths
        ):
            raise Minimal320ConsumerError("minimal320_producer_release_diff_drift")
        observed_paths.add(changed_path)
    if observed_paths != set(expected_diff):
        raise Minimal320ConsumerError("minimal320_producer_release_diff_drift")
    if (
        _git_text(
            repo,
            ["rev-parse", "HEAD"],
            reason="minimal320_producer_git_head_invalid",
        )
        != release
    ):
        raise Minimal320ConsumerError("minimal320_producer_git_head_drift")
    return (
        repo,
        _git_tree_inventory(
            repo,
            candidate,
            reason="minimal320_producer_candidate_tree_inventory_invalid",
        ),
        _git_tree_inventory(
            repo,
            release,
            reason="minimal320_producer_release_tree_inventory_invalid",
        ),
    )


def _consumer_git_identity(
    anchor: Mapping[str, Any],
    loaded: LoadedConfig,
) -> ConsumerGitIdentity:
    repo = _safe_root(ROOT, reason="minimal320_consumer_repo_path_invalid")
    top = _git_text(
        repo,
        ["rev-parse", "--show-toplevel"],
        reason="minimal320_consumer_git_repository_invalid",
    )
    try:
        if Path(top).resolve(strict=True) != repo:
            raise Minimal320ConsumerError("minimal320_consumer_git_root_mismatch")
    except OSError as error:
        raise Minimal320ConsumerError(
            "minimal320_consumer_git_root_mismatch"
        ) from error
    if _git_text(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace/"],
        reason="minimal320_consumer_git_replace_ref_invalid",
    ):
        raise Minimal320ConsumerError("minimal320_consumer_git_replace_ref_present")
    graft_text = _git_text(
        repo,
        ["rev-parse", "--git-path", "info/grafts"],
        reason="minimal320_consumer_git_grafts_invalid",
    )
    graft_path = Path(graft_text)
    if not graft_path.is_absolute():
        graft_path = repo / graft_path
    if graft_path.exists():
        try:
            graft_stat = graft_path.lstat()
        except OSError as error:
            raise Minimal320ConsumerError(
                "minimal320_consumer_git_grafts_invalid"
            ) from error
        if (
            stat.S_ISLNK(graft_stat.st_mode)
            or not stat.S_ISREG(graft_stat.st_mode)
            or graft_stat.st_size > 0
        ):
            raise Minimal320ConsumerError("minimal320_consumer_git_grafts_present")
    head = _git_text(
        repo,
        ["rev-parse", "HEAD"],
        reason="minimal320_consumer_git_head_invalid",
    )
    tree = _git_text(
        repo,
        ["rev-parse", f"{head}^{{tree}}"],
        reason="minimal320_consumer_git_tree_invalid",
    )
    runtime_commit = anchor["consumer_runtime_commit"]
    if (
        runtime_commit == head
        or _git_text(
            repo,
            ["cat-file", "-t", runtime_commit],
            reason="minimal320_consumer_runtime_commit_invalid",
        )
        != "commit"
    ):
        raise Minimal320ConsumerError("minimal320_consumer_runtime_commit_invalid")
    runtime_tree = _git_text(
        repo,
        ["rev-parse", f"{runtime_commit}^{{tree}}"],
        reason="minimal320_consumer_runtime_tree_invalid",
    )
    if runtime_tree != anchor["consumer_runtime_tree"]:
        raise Minimal320ConsumerError("minimal320_consumer_runtime_tree_drift")
    commit_object = _git(
        repo,
        ["cat-file", "-p", head],
        reason="minimal320_consumer_config_commit_invalid",
    )
    header = commit_object.split(b"\n\n", 1)[0]
    parents = [
        line.removeprefix(b"parent ").decode("ascii", errors="strict")
        for line in header.splitlines()
        if line.startswith(b"parent ")
    ]
    if parents != [runtime_commit]:
        raise Minimal320ConsumerError("minimal320_consumer_config_commit_not_direct")
    raw_diff = _git(
        repo,
        [
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-renames",
            runtime_commit,
            head,
        ],
        reason="minimal320_consumer_config_diff_invalid",
    )
    fields = raw_diff.split(b"\0")
    if len(fields) != 3 or fields[-1] != b"":
        raise Minimal320ConsumerError("minimal320_consumer_config_only_diff_drift")
    try:
        metadata = fields[0].decode("ascii", errors="strict").split(" ")
        changed_path = fields[1].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Minimal320ConsumerError(
            "minimal320_consumer_config_only_diff_drift"
        ) from error
    if (
        len(metadata) != 5
        or metadata[0] != ":100644"
        or metadata[1] != "100644"
        or metadata[4] != "M"
        or metadata[2] == metadata[3]
        or changed_path != _CONSUMER_CONFIG_PATH
    ):
        raise Minimal320ConsumerError("minimal320_consumer_config_only_diff_drift")
    current_inventory = _git_tree_inventory(
        repo,
        head,
        reason="minimal320_consumer_git_inventory_invalid",
    )
    runtime_inventory = _git_tree_inventory(
        repo,
        runtime_commit,
        reason="minimal320_consumer_runtime_inventory_invalid",
    )
    runtime_config = runtime_inventory.get(_CONSUMER_CONFIG_PATH)
    if (
        runtime_config is None
        or runtime_config[0] != "100644"
        or runtime_config[1] != "blob"
    ):
        raise Minimal320ConsumerError("minimal320_consumer_template_git_blob_invalid")
    template_raw = _git(
        repo,
        ["cat-file", "blob", runtime_config[2]],
        reason="minimal320_consumer_template_git_blob_invalid",
    )
    if _sha256(template_raw) != CONFIG_TEMPLATE_SHA256:
        raise Minimal320ConsumerError("minimal320_consumer_template_identity_drift")
    physical_identities: dict[str, tuple[int, int]] = {}
    snapshots: dict[Path, tuple[int, int, int, int]] = {}
    implementation_sha = ""
    schema_identities: dict[str, str] = {}
    for path_text in _CONSUMER_RUNTIME_PATHS:
        path = _safe_join(
            repo,
            path_text,
            reason="minimal320_consumer_required_path_invalid",
        )
        _size, sha256, blob_oid, file_identity, _line_count = _stable_file_digest(
            path,
            reason="minimal320_consumer_required_file_drift",
        )
        _require_git_blob(
            runtime_inventory,
            path_text,
            blob_oid,
            reason="minimal320_consumer_runtime_git_blob_drift",
        )
        _require_git_blob(
            current_inventory,
            path_text,
            blob_oid,
            reason="minimal320_consumer_git_blob_drift",
        )
        device_inode = file_identity[:2]
        if device_inode in physical_identities.values():
            raise Minimal320ConsumerError("minimal320_consumer_file_alias_detected")
        physical_identities[path_text] = device_inode
        snapshots[path] = file_identity
        if (
            path_text == "src/anchor_mvp/training/"
            "gemma3_chat_unbalanced_v2_minimal320_consumer_v1.py"
        ):
            implementation_sha = sha256
        for schema_name, schema_path in _CONSUMER_SCHEMA_PATHS.items():
            if path_text == schema_path:
                schema_identities[schema_name] = sha256

    config_path = _safe_join(
        repo,
        _CONSUMER_CONFIG_PATH,
        reason="minimal320_consumer_config_path_invalid",
    )
    (
        _config_size,
        config_sha,
        config_blob,
        config_identity,
        _config_lines,
    ) = _stable_file_digest(
        config_path,
        reason="minimal320_consumer_config_file_drift",
    )
    if config_sha != loaded.raw_sha256:
        raise Minimal320ConsumerError("minimal320_consumer_config_load_drift")
    _require_git_blob(
        current_inventory,
        _CONSUMER_CONFIG_PATH,
        config_blob,
        reason="minimal320_consumer_config_git_blob_drift",
    )
    if config_identity[:2] in physical_identities.values():
        raise Minimal320ConsumerError("minimal320_consumer_file_alias_detected")
    physical_identities[_CONSUMER_CONFIG_PATH] = config_identity[:2]
    snapshots[config_path] = config_identity

    if set(schema_identities) != set(_CONSUMER_SCHEMA_PATHS):
        raise Minimal320ConsumerError("minimal320_consumer_schema_inventory_invalid")
    schema_inventory_sha = _canonical_sha256(schema_identities)
    if (
        implementation_sha != anchor["consumer_implementation_sha256"]
        or schema_inventory_sha != anchor["consumer_schema_inventory_sha256"]
        or schema_identities["consumer_config"] != CONFIG_SCHEMA_SHA256
    ):
        raise Minimal320ConsumerError("minimal320_consumer_runtime_identity_drift")
    for schema_name in (
        "producer_final_binding",
        "producer_release_pin",
        "preflight_receipt",
    ):
        pin = loaded.value["schemas"][schema_name]
        if (
            pin["path"] != _CONSUMER_SCHEMA_PATHS[schema_name]
            or pin["sha256"] != schema_identities[schema_name]
        ):
            raise Minimal320ConsumerError("minimal320_consumer_schema_pin_drift")
    status = _git(
        repo,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        reason="minimal320_consumer_git_status_invalid",
    )
    if status:
        raise Minimal320ConsumerError("minimal320_consumer_git_identity_drift")
    for path, snapshot in snapshots.items():
        try:
            if _identity(path.lstat()) != snapshot:
                raise Minimal320ConsumerError(
                    "minimal320_consumer_final_identity_drift"
                )
        except OSError as error:
            raise Minimal320ConsumerError(
                "minimal320_consumer_final_identity_drift"
            ) from error
    if not implementation_sha:
        raise Minimal320ConsumerError("minimal320_consumer_implementation_missing")
    return ConsumerGitIdentity(
        head=head,
        tree=tree,
        runtime_commit=runtime_commit,
        runtime_tree=runtime_tree,
        implementation_sha256=implementation_sha,
        schema_inventory_sha256=schema_inventory_sha,
    )


def _load_repo_schema(pin: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    path = _repo_path(pin.get("path"), reason=reason)
    raw = _stable_read(path, limit=_MAX_CONFIG_BYTES, reason=reason)
    if _sha256(raw) != pin.get("sha256"):
        raise Minimal320ConsumerError(reason)
    return _strict_json(raw, reason=reason)


def _validate_legacy_pins(config: Mapping[str, Any]) -> None:
    legacy = config["legacy_matrix"]
    seen: set[str] = set()
    for pin in legacy["pinned_files"]:
        path_text = pin["path"]
        if path_text in seen:
            raise Minimal320ConsumerError("minimal320_legacy_pin_duplicate")
        seen.add(path_text)
        path = _repo_path(path_text, reason="minimal320_legacy_pin_path_invalid")
        raw = _stable_read(
            path,
            limit=_MAX_RELEASE_BYTES,
            reason="minimal320_legacy_pin_snapshot_drift",
        )
        if len(raw) != pin["bytes"] or _sha256(raw) != pin["sha256"]:
            raise Minimal320ConsumerError("minimal320_legacy_pin_identity_drift")


def _validate_arm_contract(config: Mapping[str, Any]) -> None:
    exact = config["exact320"]
    arms = config["arms"]
    ordered = arms["ordered"]
    definitions = arms["definitions"]
    if (
        len(ordered) != 9
        or len(set(ordered)) != 9
        or set(ordered) != set(definitions)
        or "identity" in ordered
        or any("identity" in arm for arm in ordered)
    ):
        raise Minimal320ConsumerError("minimal320_nine_arm_set_invalid")
    asset_counts = exact["asset_record_counts"]
    for arm in ordered:
        definition = definitions[arm]
        asset = definition["source_asset"]
        if asset not in asset_counts or definition["full_steps"] != asset_counts[asset]:
            raise Minimal320ConsumerError("minimal320_full_steps_invalid")
    identity = exact["identity_training"]
    if (
        identity["records"] != sum(identity["expert_record_counts"].values())
        or set(identity["expert_record_counts"])
        != {"angry_style", "humor", "review_audit", "serious", "tool_call"}
        or identity["separate_training_arm"] is not False
        or identity["included_in_asset_record_counts"] is not True
    ):
        raise Minimal320ConsumerError("minimal320_identity_distribution_invalid")
    if sum(asset_counts.values()) != exact["records"]:
        raise Minimal320ConsumerError("minimal320_asset_total_invalid")


def load_config(path: Path = CONFIG_PATH) -> LoadedConfig:
    schema_raw = _stable_read(
        CONFIG_SCHEMA_PATH,
        limit=_MAX_CONFIG_BYTES,
        reason="minimal320_config_schema_snapshot_drift",
    )
    if _sha256(schema_raw) != CONFIG_SCHEMA_SHA256:
        raise Minimal320ConsumerError("minimal320_config_schema_identity_drift")
    schema = _strict_json(
        schema_raw,
        reason="minimal320_config_schema_invalid",
    )
    raw = _stable_read(
        path,
        limit=_MAX_CONFIG_BYTES,
        reason="minimal320_config_snapshot_drift",
    )
    raw_sha256 = _sha256(raw)
    config = _strict_json(raw, reason="minimal320_config_invalid")
    _validate_schema(
        schema,
        config,
        reason="minimal320_config_schema_mismatch",
    )
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or _contains_placeholder(config)
        or _contains_zero_identity(config)
    ):
        raise Minimal320ConsumerError("minimal320_config_contract_drift")
    if (
        config["producer_release_stage"]["state"] == "awaiting_producer_final_release"
        and raw_sha256 != CONFIG_TEMPLATE_SHA256
    ):
        raise Minimal320ConsumerError("minimal320_config_template_identity_drift")
    for name in (
        "producer_final_binding",
        "producer_release_pin",
        "preflight_receipt",
    ):
        _load_repo_schema(
            config["schemas"][name],
            reason=f"minimal320_{name}_schema_identity_drift",
        )
    _validate_legacy_pins(config)
    _validate_arm_contract(config)
    return LoadedConfig(value=config, raw_sha256=raw_sha256)


def _strict_jsonl_rows(
    raw: bytes,
    *,
    required_keys: frozenset[str],
    expected_records: int,
    reason: str,
) -> list[dict[str, Any]]:
    if expected_records == 0:
        if raw:
            raise Minimal320ConsumerError(reason)
        return []
    if (
        not raw
        or raw.startswith(b"\xef\xbb\xbf")
        or b"\r" in raw
        or not raw.endswith(b"\n")
    ):
        raise Minimal320ConsumerError(reason)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            raise Minimal320ConsumerError(reason)
        value = _strict_json(line, reason=reason)
        if set(value) != required_keys or _canonical_bytes(value) != line:
            raise Minimal320ConsumerError(reason)
        rows.append(value)
    if len(rows) != expected_records:
        raise Minimal320ConsumerError(reason)
    return rows


def _strict_candidate_inventory_rows(
    raw: bytes,
    *,
    reason: str,
) -> list[dict[str, Any]]:
    return _strict_jsonl_rows(
        raw,
        required_keys=_SELECTED_ROW_KEYS,
        expected_records=320,
        reason=reason,
    )


def _strict_accepted_inventory_rows(
    raw: bytes,
    *,
    reason: str,
) -> list[dict[str, Any]]:
    return _strict_jsonl_rows(
        raw,
        required_keys=_ACCEPTED_ROW_KEYS,
        expected_records=320,
        reason=reason,
    )


def _strict_training_inventory_rows(
    raw: bytes,
    *,
    reason: str,
) -> list[dict[str, Any]]:
    return _strict_jsonl_rows(
        raw,
        required_keys=_TRAINING_RECORD_JOIN_KEYS,
        expected_records=320,
        reason=reason,
    )


def _candidate_identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": _SELECTOR_CANDIDATE_SCHEMA_VERSION,
        **{
            name: row[name]
            for name in (
                "idempotency_key",
                "teacher_request_idempotency_key",
                "source_record_id_sha256",
                "source_content_sha256",
                "source_serialization_identity_sha256",
                "teacher_request_record_id_sha256",
                "parent_source_record_id_sha256",
                "task_bundle_sha256",
                "parent_task_bundle_sha256",
                "source_semantic_sha256",
                "overlay_row_sha256",
                "identity_derivative_spec_sha256",
                "selection_kind",
                "output_expert_role",
                "training_asset",
                "validation_contract_role",
                "language",
                "tool_family",
                "identity_class",
                "identity_parent_class",
                "identity_provenance_intent",
                "review_verdict",
                "review_fault_type",
                "router_label",
            )
        },
    }


def _validate_inventory_row(row: Mapping[str, Any]) -> None:
    for name in (
        "candidate_id_sha256",
        "idempotency_key",
        "teacher_request_idempotency_key",
        "source_record_id_sha256",
        "source_content_sha256",
        "source_serialization_identity_sha256",
        "teacher_request_record_id_sha256",
        "parent_source_record_id_sha256",
        "task_bundle_sha256",
        "parent_task_bundle_sha256",
        "source_semantic_sha256",
        "overlay_row_sha256",
    ):
        if not isinstance(row[name], str) or not _SHA256_RE.fullmatch(row[name]):
            raise Minimal320ConsumerError("minimal320_record_identity_invalid")
    if (
        row["schema_version"] != _SELECTOR_CANDIDATE_SCHEMA_VERSION
        or row["content_retained"] is not False
        or _canonical_sha256(_candidate_identity_payload(row))
        != row["candidate_id_sha256"]
        or row["teacher_request_idempotency_key"] != row["idempotency_key"]
        or row["teacher_request_record_id_sha256"] != row["source_record_id_sha256"]
    ):
        raise Minimal320ConsumerError("minimal320_record_identity_invalid")
    role = row["output_expert_role"]
    if role not in _ROLE_TO_ASSET or row["training_asset"] != _ROLE_TO_ASSET[role]:
        raise Minimal320ConsumerError("minimal320_record_role_asset_invalid")
    if row["language"] not in {"en", "zh"}:
        raise Minimal320ConsumerError("minimal320_record_language_invalid")
    if row["selection_kind"] not in {
        "full_core",
        "tool_review_depth",
        "identity",
        "router",
    }:
        raise Minimal320ConsumerError("minimal320_record_partition_invalid")
    for name in (
        "tool_family",
        "identity_class",
        "identity_parent_class",
        "identity_provenance_intent",
        "router_label",
    ):
        if row[name] is not None and (not isinstance(row[name], str) or not row[name]):
            raise Minimal320ConsumerError("minimal320_record_cell_invalid")
    if row["tool_family"] is not None and not re.fullmatch(
        r"[a-z0-9_]{1,64}",
        row["tool_family"],
    ):
        raise Minimal320ConsumerError("minimal320_record_tool_family_invalid")
    if row["identity_class"] not in {
        None,
        "air_attribution",
        "false_google_attribution",
        "false_openai_attribution",
    }:
        raise Minimal320ConsumerError("minimal320_record_identity_class_invalid")
    if row["router_label"] not in {None, "humor", "serious", "angry"}:
        raise Minimal320ConsumerError("minimal320_record_router_label_invalid")
    if row["review_verdict"] not in {None, "pass", "fail"}:
        raise Minimal320ConsumerError("minimal320_record_review_verdict_invalid")
    if row["review_fault_type"] not in {None, *_REVIEW_FAULT_COUNTS}:
        raise Minimal320ConsumerError("minimal320_record_review_fault_invalid")
    if row["validation_contract_role"] not in {
        "humor",
        "serious",
        "angry",
        "tool",
        "review",
        "identity",
        "router",
    }:
        raise Minimal320ConsumerError("minimal320_record_validation_role_invalid")
    if row["selection_kind"] == "identity":
        if (
            row["validation_contract_role"] != "identity"
            or row["identity_parent_class"] != row["identity_class"]
            or row["identity_provenance_intent"] != "air_provenance_invariant"
            or not isinstance(row["identity_derivative_spec_sha256"], str)
            or _SHA256_RE.fullmatch(row["identity_derivative_spec_sha256"]) is None
        ):
            raise Minimal320ConsumerError("minimal320_record_identity_class_invalid")
    elif (
        row["validation_contract_role"] != role
        or row["identity_parent_class"] is not None
        or row["identity_provenance_intent"] is not None
        or row["identity_derivative_spec_sha256"] is not None
    ):
        raise Minimal320ConsumerError("minimal320_record_validation_role_invalid")


def _schema_refs_are_local(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "$ref" and (
                not isinstance(item, str) or not item.startswith("#/")
            ):
                return False
            if not _schema_refs_are_local(item):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_schema_refs_are_local(item) for item in value)
    return True


def _teacher_record_validator(
    schema: Mapping[str, Any],
) -> Draft202012Validator:
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id") != _TEACHER_RECORD_SCHEMA_VERSION
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or set(schema.get("required", ())) != _TEACHER_RECORD_KEYS
        or set(schema.get("properties", {})) != _TEACHER_RECORD_KEYS
        or not _schema_refs_are_local(schema)
    ):
        raise Minimal320ConsumerError("minimal320_teacher_record_schema_contract_drift")
    properties = schema["properties"]
    source_assets = properties.get("source_asset", {}).get("enum")
    if (
        properties.get("schema_version", {}).get("const")
        != _TEACHER_RECORD_SCHEMA_VERSION
        or properties.get("namespace", {}).get("const") != _TEACHER_RECORD_NAMESPACE
        or properties.get("decision", {}).get("const") != "accepted"
        or set(source_assets or ()) != set(_ROLE_TO_ASSET.values())
    ):
        raise Minimal320ConsumerError("minimal320_teacher_record_schema_contract_drift")
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError as error:
        raise Minimal320ConsumerError(
            "minimal320_teacher_record_schema_invalid"
        ) from error
    return Draft202012Validator(dict(schema))


def _strict_target_object(target: str, *, reason: str) -> dict[str, Any]:
    return _strict_json(target.encode("utf-8"), reason=reason)


def _validate_teacher_target_semantics(
    candidate: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    target = record["teacher_target"]
    role = str(candidate["output_expert_role"])
    selection_kind = str(candidate["selection_kind"])
    identity_class = candidate["identity_class"]
    if selection_kind == "identity":
        template = _IDENTITY_TEMPLATES.get(str(identity_class))
        if template is None:
            raise Minimal320ConsumerError(
                "minimal320_teacher_identity_semantics_invalid"
            )
        if role in {"humor", "serious", "angry"}:
            if template not in target:
                raise Minimal320ConsumerError(
                    "minimal320_teacher_identity_semantics_invalid"
                )
            remainder = target.replace(template, "", 1).casefold()
            if any(provider in remainder for provider in ("air", "google", "openai")):
                raise Minimal320ConsumerError(
                    "minimal320_teacher_identity_semantics_invalid"
                )
            return
        if role == "tool":
            if target != template:
                raise Minimal320ConsumerError(
                    "minimal320_teacher_identity_semantics_invalid"
                )
            return
        if role == "review":
            value = _strict_target_object(
                target,
                reason="minimal320_teacher_identity_semantics_invalid",
            )
            if (
                set(value) != {"verdict", "faults", "correction"}
                or value["verdict"] != "fail"
                or value["faults"] != ["missing_identity_answer"]
                or value["correction"] != template
            ):
                raise Minimal320ConsumerError(
                    "minimal320_teacher_identity_semantics_invalid"
                )
            return
        raise Minimal320ConsumerError("minimal320_teacher_identity_semantics_invalid")

    if identity_class is not None:
        raise Minimal320ConsumerError("minimal320_teacher_identity_semantics_invalid")
    if role in {"humor", "serious", "angry"}:
        if target.startswith(("{", "[")) or target.endswith(("}", "]")):
            raise Minimal320ConsumerError("minimal320_teacher_natural_target_invalid")
        return
    value = _strict_target_object(
        target,
        reason="minimal320_teacher_structured_target_invalid",
    )
    if role == "tool":
        call = value.get("tool_call")
        if (
            set(value) != {"final_answer", "tool_call", "evidence_ids"}
            or not isinstance(value["final_answer"], str)
            or not value["final_answer"].strip()
            or not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str)
            or not call["name"]
            or not isinstance(call["arguments"], dict)
            or not isinstance(value["evidence_ids"], list)
            or not value["evidence_ids"]
            or any(
                not isinstance(item, str) or not item for item in value["evidence_ids"]
            )
        ):
            raise Minimal320ConsumerError("minimal320_teacher_tool_semantics_invalid")
        return
    if role == "review":
        if set(value) != {"verdict", "faults", "correction"}:
            raise Minimal320ConsumerError("minimal320_teacher_review_semantics_invalid")
        verdict = value["verdict"]
        faults = value["faults"]
        correction = value["correction"]
        if (
            verdict not in {"pass", "fail"}
            or not isinstance(faults, list)
            or any(not isinstance(item, str) or not item for item in faults)
            or (verdict == "pass" and (faults or correction is not None))
            or (
                verdict == "fail"
                and (
                    not faults
                    or not isinstance(correction, str)
                    or not correction.strip()
                )
            )
        ):
            raise Minimal320ConsumerError("minimal320_teacher_review_semantics_invalid")
        if selection_kind == "tool_review_depth":
            expected_verdict = candidate["review_verdict"]
            expected_faults = (
                [candidate["review_fault_type"]] if expected_verdict == "fail" else []
            )
            if verdict != expected_verdict or faults != expected_faults:
                raise Minimal320ConsumerError(
                    "minimal320_teacher_review_semantics_invalid"
                )
        return
    if role == "router":
        if (
            set(value) != {"route", "plan", "stop"}
            or value["route"] != candidate["router_label"]
            or not isinstance(value["plan"], list)
            or not value["plan"]
            or any(not isinstance(item, str) or not item for item in value["plan"])
            or value["stop"] is not True
        ):
            raise Minimal320ConsumerError("minimal320_teacher_router_semantics_invalid")
        return
    raise Minimal320ConsumerError("minimal320_teacher_target_role_invalid")


def _validate_teacher_shards(
    *,
    schema: Mapping[str, Any],
    shard_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validator = _teacher_record_validator(schema)
    if not (
        len(shard_rows)
        == len(selected_rows)
        == len(accepted_rows)
        == len(training_rows)
        == 320
    ):
        raise Minimal320ConsumerError("minimal320_teacher_record_join_count_drift")
    materialized_selected = [dict(row) for row in selected_rows]
    for selected in materialized_selected:
        if set(selected) != _SELECTED_ROW_KEYS:
            raise Minimal320ConsumerError("minimal320_selected_inventory_invalid")
        _validate_inventory_row(selected)
    if (
        materialized_selected
        != sorted(
            materialized_selected,
            key=lambda row: str(row["candidate_id_sha256"]),
        )
        or len({str(row["candidate_id_sha256"]) for row in materialized_selected})
        != 320
        or len({str(row["idempotency_key"]) for row in materialized_selected}) != 320
    ):
        raise Minimal320ConsumerError("minimal320_record_inventory_order_invalid")

    accepted_by_key: dict[str, dict[str, Any]] = {}
    materialized_accepted = [dict(row) for row in accepted_rows]
    if materialized_accepted != sorted(
        materialized_accepted,
        key=lambda row: str(row.get("idempotency_key")),
    ):
        raise Minimal320ConsumerError("minimal320_accepted_inventory_order_invalid")
    selected_by_key = {
        str(row["idempotency_key"]): row for row in materialized_selected
    }
    for accepted in materialized_accepted:
        key = str(accepted.get("idempotency_key"))
        selected = selected_by_key.get(key)
        stratum = accepted.get("stratum")
        expected_stratum = (
            {
                "role": selected["output_expert_role"],
                "validation_contract_role": selected["validation_contract_role"],
                "language": selected["language"],
                "tool_family": selected["tool_family"],
                "identity_class": selected["identity_class"],
                "identity_parent_class": selected["identity_parent_class"],
                "selection_kind": selected["selection_kind"],
                "review_verdict": selected["review_verdict"],
                "review_fault_type": selected["review_fault_type"],
                "router_label": selected["router_label"],
                "source_record_id_sha256": selected["source_record_id_sha256"],
            }
            if selected is not None
            else None
        )
        if (
            set(accepted) != _ACCEPTED_ROW_KEYS
            or accepted.get("schema_version") != _ACCEPTED_INVENTORY_SCHEMA_VERSION
            or accepted.get("outcome") != "accepted"
            or accepted.get("content_retained") is not False
            or not isinstance(accepted.get("reason_code"), str)
            or re.fullmatch(r"[a-z0-9_]{1,64}", accepted["reason_code"]) is None
            or not isinstance(stratum, Mapping)
            or set(stratum) != _STRATUM_KEYS
            or dict(stratum) != expected_stratum
            or any(
                not isinstance(accepted.get(name), str)
                or _SHA256_RE.fullmatch(str(accepted[name])) is None
                for name in ("receipt_sha256", "output_record_sha256")
            )
            or key in accepted_by_key
        ):
            raise Minimal320ConsumerError("minimal320_accepted_inventory_invalid")
        accepted_by_key[key] = accepted
    if set(accepted_by_key) != set(selected_by_key):
        raise Minimal320ConsumerError("minimal320_accepted_inventory_join_drift")

    joined_candidates: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for ordinal, (record, selected, training) in enumerate(
        zip(
            shard_rows,
            materialized_selected,
            training_rows,
            strict=True,
        )
    ):
        try:
            validator.validate(dict(record))
        except ValidationError as error:
            raise Minimal320ConsumerError(
                "minimal320_teacher_record_schema_mismatch"
            ) from error
        if set(record) != _TEACHER_RECORD_KEYS:
            raise Minimal320ConsumerError("minimal320_teacher_record_contract_invalid")
        target = record["teacher_target"]
        if (
            not isinstance(target, str)
            or not target
            or target != target.strip()
            or "\x00" in target
            or _sha256(target.encode("utf-8")) != record["teacher_target_sha256"]
        ):
            raise Minimal320ConsumerError("minimal320_teacher_target_identity_drift")
        for name in (
            "record_id_sha256",
            "source_content_sha256",
            "source_serialization_identity_sha256",
            "task_bundle_sha256",
            "teacher_target_sha256",
        ):
            if (
                not isinstance(record[name], str)
                or _SHA256_RE.fullmatch(record[name]) is None
            ):
                raise Minimal320ConsumerError(
                    "minimal320_teacher_record_identity_invalid"
                )
        record_id = str(record["record_id_sha256"])
        if record_id in record_ids:
            raise Minimal320ConsumerError("minimal320_teacher_record_duplicate")
        record_ids.add(record_id)
        candidate = dict(selected)
        accepted = accepted_by_key[str(selected["idempotency_key"])]
        if (
            set(training) != _TRAINING_RECORD_JOIN_KEYS
            or training.get("schema_version") != _TRAINING_RECORD_SCHEMA_VERSION
            or training.get("training_ordinal") != ordinal
            or training.get("selected_ordinal") != ordinal
            or training.get("candidate_id_sha256") != candidate["candidate_id_sha256"]
            or training.get("selected_candidate_sha256") != _canonical_sha256(candidate)
            or training.get("idempotency_key") != candidate["idempotency_key"]
            or training.get("source_record_id_sha256")
            != candidate["source_record_id_sha256"]
            or training.get("source_content_sha256")
            != candidate["source_content_sha256"]
            or training.get("source_serialization_identity_sha256")
            != candidate["source_serialization_identity_sha256"]
            or training.get("task_bundle_sha256") != candidate["task_bundle_sha256"]
            or training.get("source_asset") != candidate["training_asset"]
            or training.get("decision") != "accepted"
            or training.get("receipt_sha256") != accepted["receipt_sha256"]
            or training.get("output_record_sha256") != accepted["output_record_sha256"]
            or training.get("content_retained") is not False
            or training.get("raw_token_ids_retained") is not False
            or record["record_id_sha256"] != candidate["source_record_id_sha256"]
            or record["source_content_sha256"] != candidate["source_content_sha256"]
            or record["source_serialization_identity_sha256"]
            != candidate["source_serialization_identity_sha256"]
            or record["task_bundle_sha256"] != candidate["task_bundle_sha256"]
            or record["source_asset"] != candidate["training_asset"]
            or record["decision"] != "accepted"
            or training.get("teacher_target_sha256") != record["teacher_target_sha256"]
            or training.get("teacher_record_sha256") != _canonical_sha256(record)
        ):
            raise Minimal320ConsumerError("minimal320_teacher_record_join_drift")
        _validate_teacher_target_semantics(candidate, record)
        joined_candidates.append(candidate)
    return joined_candidates


def _validate_bundle_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_roles: set[str],
    reason: str,
) -> None:
    if (
        len(rows) != len(expected_roles)
        or {str(row["output_expert_role"]) for row in rows} != expected_roles
    ):
        raise Minimal320ConsumerError(reason)


def _validate_exact_strata(
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> None:
    for row in rows:
        _validate_inventory_row(row)
    candidate_ids = [str(row["candidate_id_sha256"]) for row in rows]
    record_ids = [str(row["source_record_id_sha256"]) for row in rows]
    if (
        candidate_ids != sorted(candidate_ids)
        or len(set(candidate_ids)) != 320
        or len(set(record_ids)) != 320
    ):
        raise Minimal320ConsumerError("minimal320_record_inventory_order_invalid")
    by_role = Counter(str(row["output_expert_role"]) for row in rows)
    by_language = Counter(str(row["language"]) for row in rows)
    by_partition = Counter(str(row["selection_kind"]) for row in rows)
    if (
        dict(sorted(by_role.items())) != selection["counts"]["by_role"]
        or dict(sorted(by_language.items())) != selection["counts"]["by_language"]
        or by_partition
        != Counter(
            {
                "full_core": 100,
                "tool_review_depth": 40,
                "identity": 100,
                "router": 80,
            }
        )
    ):
        raise Minimal320ConsumerError("minimal320_record_counts_invalid")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["selection_kind"]), str(row["task_bundle_sha256"]))].append(
            row
        )

    core_cells: Counter[tuple[str, str]] = Counter()
    core_groups = {
        bundle: values
        for (kind, bundle), values in grouped.items()
        if kind == "full_core"
    }
    if len(core_groups) != 20:
        raise Minimal320ConsumerError("minimal320_core_bundle_count_invalid")
    for values in core_groups.values():
        _validate_bundle_roles(
            values,
            expected_roles=set(_EXPERT_ROLES),
            reason="minimal320_core_bundle_closure_invalid",
        )
        languages = {str(row["language"]) for row in values}
        families = {str(row["tool_family"]) for row in values}
        if (
            len(languages) != 1
            or len(families) != 1
            or None in {row["tool_family"] for row in values}
            or any(
                row["identity_class"] is not None or row["router_label"] is not None
                for row in values
            )
        ):
            raise Minimal320ConsumerError("minimal320_core_cell_invalid")
        core_cells[(next(iter(families)), next(iter(languages)))] += 1
    if (
        len({family for family, _language in core_cells}) != 10
        or set(core_cells.values()) != {1}
        or {language for _family, language in core_cells} != {"en", "zh"}
        or any(
            core_cells[(family, language)] != 1
            for family, _language in core_cells
            for language in ("en", "zh")
        )
    ):
        raise Minimal320ConsumerError("minimal320_core_family_language_matrix_invalid")

    depth_cells: Counter[tuple[str, str]] = Counter()
    depth_groups = {
        bundle: values
        for (kind, bundle), values in grouped.items()
        if kind == "tool_review_depth"
    }
    if len(depth_groups) != 20:
        raise Minimal320ConsumerError("minimal320_depth_bundle_count_invalid")
    depth_reviews: list[Mapping[str, Any]] = []
    for values in depth_groups.values():
        _validate_bundle_roles(
            values,
            expected_roles={"tool", "review"},
            reason="minimal320_depth_bundle_closure_invalid",
        )
        languages = {str(row["language"]) for row in values}
        families = {str(row["tool_family"]) for row in values}
        if (
            len(languages) != 1
            or len(families) != 1
            or None in {row["tool_family"] for row in values}
            or any(
                row["identity_class"] is not None or row["router_label"] is not None
                for row in values
            )
        ):
            raise Minimal320ConsumerError("minimal320_depth_cell_invalid")
        cell = (next(iter(families)), next(iter(languages)))
        depth_cells[cell] += 1
        depth_reviews.extend(
            row for row in values if row["output_expert_role"] == "review"
        )
    if depth_cells != core_cells:
        raise Minimal320ConsumerError("minimal320_depth_family_language_matrix_invalid")
    verdicts = Counter(str(row["review_verdict"]) for row in depth_reviews)
    fail_rows = [row for row in depth_reviews if row["review_verdict"] == "fail"]
    if (
        verdicts != Counter({"pass": 10, "fail": 10})
        or any(
            row["review_fault_type"] is not None
            for row in depth_reviews
            if row["review_verdict"] == "pass"
        )
        or Counter(str(row["review_fault_type"]) for row in fail_rows)
        != Counter(_REVIEW_FAULT_COUNTS)
    ):
        raise Minimal320ConsumerError("minimal320_review_depth_distribution_invalid")
    actual_fault_cells = sorted(
        (
            str(row["tool_family"]),
            str(row["language"]),
            str(row["review_fault_type"]),
        )
        for row in fail_rows
    )
    expected_fault_cells = sorted(
        (
            str(cell["tool_family"]),
            str(cell["language"]),
            str(cell["fault"]),
        )
        for cell in selection["review_fault_cells"]
    )
    if actual_fault_cells != expected_fault_cells:
        raise Minimal320ConsumerError("minimal320_review_fault_cell_drift")

    identity_groups = {
        bundle: values
        for (kind, bundle), values in grouped.items()
        if kind == "identity"
    }
    identity_cells: Counter[tuple[str, str]] = Counter()
    if len(identity_groups) != 20:
        raise Minimal320ConsumerError("minimal320_identity_bundle_count_invalid")
    for values in identity_groups.values():
        _validate_bundle_roles(
            values,
            expected_roles=set(_EXPERT_ROLES),
            reason="minimal320_identity_bundle_closure_invalid",
        )
        classes = {str(row["identity_class"]) for row in values}
        languages = {str(row["language"]) for row in values}
        if (
            len(classes) != 1
            or len(languages) != 1
            or any(
                row["tool_family"] is not None or row["router_label"] is not None
                for row in values
            )
        ):
            raise Minimal320ConsumerError("minimal320_identity_cell_invalid")
        identity_cells[(next(iter(classes)), next(iter(languages)))] += 1
    if identity_cells != Counter(_IDENTITY_BUNDLE_CELLS):
        raise Minimal320ConsumerError("minimal320_identity_cell_distribution_invalid")

    router_rows = [row for row in rows if row["selection_kind"] == "router"]
    if Counter(
        (str(row["router_label"]), str(row["language"])) for row in router_rows
    ) != Counter(_ROUTER_RECORD_CELLS) or any(
        row["output_expert_role"] != "router"
        or row["training_asset"] != "planner_router"
        or row["tool_family"] is not None
        or row["identity_class"] is not None
        or row["review_verdict"] is not None
        or row["review_fault_type"] is not None
        for row in router_rows
    ):
        raise Minimal320ConsumerError("minimal320_router_cell_distribution_invalid")


def _validate_source_truth_contract(
    value: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if set(value) != {
        "schema_version",
        "contract_id",
        "strata",
        "review_fault_cells",
    }:
        raise Minimal320ConsumerError("minimal320_source_truth_contract_invalid")
    if (
        not isinstance(value["schema_version"], str)
        or not value["schema_version"]
        or _contains_placeholder(value["schema_version"])
    ):
        raise Minimal320ConsumerError("minimal320_source_truth_contract_invalid")
    body = dict(value)
    contract_id = body.pop("contract_id")
    if (
        not isinstance(contract_id, str)
        or _canonical_sha256(body) != contract_id
        or value["strata"] != selection["strata"]
        or value["review_fault_cells"] != selection["review_fault_cells"]
    ):
        raise Minimal320ConsumerError("minimal320_source_truth_contract_drift")


def _validate_checkpoint(
    value: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if set(value) != {
        "schema_version",
        "state",
        "records",
        "selected_set_root_sha256",
        "inventory_roots",
    }:
        raise Minimal320ConsumerError("minimal320_checkpoint_manifest_invalid")
    if (
        not isinstance(value["schema_version"], str)
        or not value["schema_version"]
        or value["state"] != "minimal320_checkpoint_frozen"
        or value["records"] != 320
        or value["selected_set_root_sha256"] != selection["selected_set_root_sha256"]
        or value["inventory_roots"] != selection["inventory_roots"]
    ):
        raise Minimal320ConsumerError("minimal320_checkpoint_manifest_drift")


def _validate_committed_shards_manifest(
    value: Mapping[str, Any],
    *,
    repo: Path,
    teacher_entry: Mapping[str, Any],
    training_entry: Mapping[str, Any],
    teacher_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    accepted_entry: Mapping[str, Any],
    selected_set_root_sha256: str,
    teacher_schema_sha256: str,
) -> None:
    required = {
        "schema_version",
        "release_root",
        "shard_count",
        "files",
        "file_count",
        "file_inventory_root_sha256",
        "teacher_training_shard",
        "training_record_inventory",
        "counts",
        "partial_accepted_subset",
        "exact_requested_set_complete",
        "accepted_selected_order_root_sha256",
        "teacher_target_order_root_sha256",
        "zip_alignment",
        "accepted_inventory_sha256",
        "selected_set_root_sha256",
        "accepted_strata_root_sha256",
        "final_chain_tip_sha256",
        "final_wal_entries",
        "content_retained",
        "raw_token_ids_retained",
    }
    if set(value) != required:
        raise Minimal320ConsumerError("minimal320_committed_shards_manifest_invalid")
    shard_sidecar_path = f"{_TEACHER_SHARD_PATH}.sha256"
    training_sidecar_path = f"{_TRAINING_RECORD_INVENTORY_PATH}.sha256"
    shard_sidecar_raw = _stable_read(
        _safe_join(
            repo,
            shard_sidecar_path,
            reason="minimal320_committed_shards_sidecar_path_invalid",
        ),
        limit=_MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE - 1,
        reason="minimal320_committed_shards_sidecar_invalid",
    )
    training_sidecar_raw = _stable_read(
        _safe_join(
            repo,
            training_sidecar_path,
            reason="minimal320_committed_shards_sidecar_path_invalid",
        ),
        limit=_MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE - 1,
        reason="minimal320_committed_shards_sidecar_invalid",
    )
    expected_files = [
        {
            "path": _TEACHER_SHARD_PATH,
            "sha256": teacher_entry["sha256"],
            "bytes": teacher_entry["bytes"],
            "kind": "teacher_training_shard",
        },
        {
            "path": shard_sidecar_path,
            "sha256": _sha256(shard_sidecar_raw),
            "bytes": len(shard_sidecar_raw),
            "kind": "sha256_sidecar",
        },
        {
            "path": _TRAINING_RECORD_INVENTORY_PATH,
            "sha256": training_entry["sha256"],
            "bytes": training_entry["bytes"],
            "kind": "training_record_inventory",
        },
        {
            "path": training_sidecar_path,
            "sha256": _sha256(training_sidecar_raw),
            "bytes": len(training_sidecar_raw),
            "kind": "sha256_sidecar",
        },
    ]
    teacher_contract = value["teacher_training_shard"]
    training_contract = value["training_record_inventory"]
    exact_counts = {
        "requested": 320,
        "completed": 320,
        "accepted": 320,
        "rejected": 0,
        "quarantine": 0,
    }
    if (
        value["schema_version"] != _COMMITTED_SHARDS_SCHEMA_VERSION
        or value["release_root"] != _PRODUCER_RELEASE_ROOT
        or value["shard_count"] != 1
        or value["files"] != expected_files
        or value["file_count"] != 4
        or value["file_inventory_root_sha256"] != _merkle_root(expected_files)
        or not isinstance(teacher_contract, Mapping)
        or set(teacher_contract)
        != {
            "path",
            "sidecar_path",
            "schema_version",
            "schema_sha256",
            "row_count",
            "bytes",
            "max_bytes_exclusive",
            "sha256",
            "record_inventory_root_sha256",
        }
        or teacher_contract["path"] != _TEACHER_SHARD_PATH
        or teacher_contract["sidecar_path"] != shard_sidecar_path
        or teacher_contract["schema_version"] != _TEACHER_RECORD_SCHEMA_VERSION
        or teacher_contract["schema_sha256"] != teacher_schema_sha256
        or teacher_contract["row_count"] != 320
        or teacher_contract["bytes"] != teacher_entry["bytes"]
        or teacher_contract["bytes"] >= _MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE
        or teacher_contract["max_bytes_exclusive"]
        != _MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE
        or teacher_contract["sha256"] != teacher_entry["sha256"]
        or teacher_contract["record_inventory_root_sha256"]
        != _merkle_root(teacher_rows)
        or not isinstance(training_contract, Mapping)
        or set(training_contract)
        != {
            "path",
            "sidecar_path",
            "schema_version",
            "row_count",
            "bytes",
            "sha256",
            "inventory_root_sha256",
        }
        or training_contract["path"] != _TRAINING_RECORD_INVENTORY_PATH
        or training_contract["sidecar_path"] != training_sidecar_path
        or training_contract["schema_version"] != _TRAINING_RECORD_SCHEMA_VERSION
        or training_contract["row_count"] != 320
        or training_contract["bytes"] != training_entry["bytes"]
        or training_contract["bytes"] >= _MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE
        or training_contract["sha256"] != training_entry["sha256"]
        or training_contract["inventory_root_sha256"] != _merkle_root(training_rows)
        or value["counts"] != exact_counts
        or value["partial_accepted_subset"] is not False
        or value["exact_requested_set_complete"] is not True
        or value["accepted_selected_order_root_sha256"]
        != _canonical_sha256([row["candidate_id_sha256"] for row in training_rows])
        or value["teacher_target_order_root_sha256"]
        != _canonical_sha256([row["teacher_target_sha256"] for row in training_rows])
        or value["zip_alignment"]
        != {
            "row_count": 320,
            "training_ordinals_contiguous": True,
            "selected_order_filtered_accepted": True,
            "teacher_record_sha256_bound": True,
        }
        or value["accepted_inventory_sha256"] != accepted_entry["sha256"]
        or value["selected_set_root_sha256"] != selected_set_root_sha256
        or any(
            not isinstance(value[field], str)
            or _SHA256_RE.fullmatch(value[field]) is None
            for field in (
                "accepted_strata_root_sha256",
                "final_chain_tip_sha256",
            )
        )
        or not isinstance(value["final_wal_entries"], int)
        or value["final_wal_entries"] < 320
        or value["content_retained"] is not False
        or value["raw_token_ids_retained"] is not False
    ):
        raise Minimal320ConsumerError("minimal320_committed_shards_manifest_drift")


def _validate_binding_preimages(
    binding: Mapping[str, Any],
    *,
    repo: Path,
    launch_inventory: Mapping[str, tuple[str, str, str]],
    candidate_inventory: Mapping[str, tuple[str, str, str]],
    release_inventory: Mapping[str, tuple[str, str, str]],
    release_documents: Mapping[str, tuple[str, str, str]],
    reserved_identities: set[tuple[int, int, int, int]],
    reserved_paths: set[str],
) -> None:
    component_pins_sha = _canonical_sha256(binding["component_pins"])
    if component_pins_sha != binding["component_pins_sha256"]:
        raise Minimal320ConsumerError("minimal320_component_pin_preimage_drift")
    inventory = binding["artifact_inventory"]
    paths = [item["path"] for item in inventory]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise Minimal320ConsumerError("minimal320_artifact_inventory_order_invalid")
    if _canonical_sha256(inventory) != binding["artifact_inventory_sha256"]:
        raise Minimal320ConsumerError("minimal320_artifact_inventory_preimage_drift")

    entries_by_path: dict[str, Mapping[str, Any]] = {}
    entries_by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    physical_paths = set(reserved_paths)
    physical_identities = set(reserved_identities)
    physical_snapshots: dict[Path, tuple[int, int, int, int]] = {}
    raw_preimages: dict[str, bytes] = {}
    for entry in inventory:
        path_text = _normalized_relative(
            entry["path"],
            reason="minimal320_artifact_path_invalid",
        ).as_posix()
        path = _safe_join(
            repo,
            path_text,
            reason="minimal320_artifact_path_invalid",
        )
        normalized_physical = os.path.normcase(str(path))
        if normalized_physical in physical_paths:
            raise Minimal320ConsumerError("minimal320_artifact_path_alias_detected")
        if entry["kind"] in {
            *_REQUIRED_METADATA_KINDS,
            "source_truth_contract",
            "teacher_record_schema",
            "teacher_shard",
        }:
            raw, file_identity = _stable_bound_read(
                path,
                limit=_MAX_INVENTORY_BYTES,
                reason="minimal320_artifact_snapshot_drift",
                allow_empty=entry["kind"]
                in {"rejected_inventory", "quarantine_inventory"},
            )
            observed_bytes = len(raw)
            observed_sha = _sha256(raw)
            observed_lines = raw.count(b"\n")
            blob_digest = hashlib.sha1(usedforsecurity=False)
            blob_digest.update(f"blob {len(raw)}\0".encode("ascii"))
            blob_digest.update(raw)
            blob_oid = blob_digest.hexdigest()
            raw_preimages[path_text] = raw
        else:
            (
                observed_bytes,
                observed_sha,
                blob_oid,
                file_identity,
                observed_lines,
            ) = _stable_file_digest(
                path,
                reason="minimal320_artifact_snapshot_drift",
            )
        if file_identity in physical_identities:
            raise Minimal320ConsumerError("minimal320_artifact_hardlink_alias_detected")
        physical_paths.add(normalized_physical)
        physical_identities.add(file_identity)
        physical_snapshots[path] = file_identity
        if (
            observed_bytes != entry["bytes"]
            or observed_sha != entry["sha256"]
            or blob_oid != entry["git_blob_oid"]
        ):
            raise Minimal320ConsumerError("minimal320_artifact_identity_drift")
        if entry["kind"] == "teacher_shard" and (
            entry.get("records") != observed_lines
        ):
            raise Minimal320ConsumerError("minimal320_teacher_shard_record_count_drift")
        if entry["kind"] in {"freeze_attestation", "external_attestation"}:
            expected_stage = "producer_final_R"
        elif entry["kind"] in _COMPONENT_PIN_TO_KIND.values():
            expected_stage = "producer_launch_L"
        else:
            expected_stage = "producer_payload_P"
        if entry["git_stage"] != expected_stage:
            raise Minimal320ConsumerError("minimal320_artifact_git_stage_drift")
        if expected_stage == "producer_launch_L":
            _require_git_blob(
                launch_inventory,
                path_text,
                blob_oid,
                reason="minimal320_launch_artifact_git_blob_drift",
            )
            _require_git_blob(
                candidate_inventory,
                path_text,
                blob_oid,
                reason="minimal320_candidate_artifact_git_blob_drift",
            )
            _require_git_blob(
                release_inventory,
                path_text,
                blob_oid,
                reason="minimal320_release_artifact_git_blob_drift",
            )
        elif expected_stage == "producer_payload_P":
            _require_git_blob(
                candidate_inventory,
                path_text,
                blob_oid,
                reason="minimal320_payload_artifact_git_blob_drift",
            )
            _require_git_blob(
                release_inventory,
                path_text,
                blob_oid,
                reason="minimal320_release_artifact_git_blob_drift",
            )
        else:
            _require_git_blob(
                release_inventory,
                path_text,
                blob_oid,
                reason="minimal320_release_artifact_git_blob_drift",
            )
        entries_by_path[path_text] = entry
        entries_by_kind[str(entry["kind"])].append(entry)

    required_singletons = {
        *_COMPONENT_PIN_TO_KIND.values(),
        *_REQUIRED_METADATA_KINDS,
    }
    if (
        len(inventory) != 17
        or set(entries_by_kind) != {*required_singletons, "teacher_shard"}
        or any(len(entries_by_kind[kind]) != 1 for kind in required_singletons)
        or not entries_by_kind["teacher_shard"]
        or sum(int(entry["records"]) for entry in entries_by_kind["teacher_shard"])
        != 320
    ):
        raise Minimal320ConsumerError("minimal320_artifact_kind_inventory_invalid")

    bindings = binding["artifact_bindings"]
    if dict(bindings["metadata"]) != _EXACT_METADATA_PATHS or list(
        bindings["teacher_shards"]
    ) != [_TEACHER_SHARD_PATH]:
        raise Minimal320ConsumerError("minimal320_artifact_binding_path_drift")
    mapped_paths: set[str] = set()
    for group_name in ("components", "metadata"):
        for kind, path_text_raw in bindings[group_name].items():
            path_text = _normalized_relative(
                path_text_raw,
                reason="minimal320_artifact_binding_path_invalid",
            ).as_posix()
            entry = entries_by_path.get(path_text)
            if entry is None or entry["kind"] != kind or path_text in mapped_paths:
                raise Minimal320ConsumerError("minimal320_artifact_binding_drift")
            mapped_paths.add(path_text)
    for kind in ("freeze_attestation", "external_attestation"):
        expected_path, expected_sha, expected_blob = release_documents[kind]
        mapped_path = bindings["metadata"].get(kind)
        entry = entries_by_path.get(expected_path)
        if (
            mapped_path != expected_path
            or entry is None
            or entry["kind"] != kind
            or entry["git_stage"] != "producer_final_R"
            or entry["sha256"] != expected_sha
            or entry["git_blob_oid"] != expected_blob
        ):
            raise Minimal320ConsumerError("minimal320_release_artifact_anchor_drift")
    teacher_paths = [
        _normalized_relative(
            item,
            reason="minimal320_teacher_shard_path_invalid",
        ).as_posix()
        for item in bindings["teacher_shards"]
    ]
    if (
        teacher_paths != sorted(teacher_paths)
        or len(teacher_paths) != len(set(teacher_paths))
        or set(teacher_paths)
        != {entry["path"] for entry in entries_by_kind["teacher_shard"]}
        or mapped_paths.intersection(teacher_paths)
    ):
        raise Minimal320ConsumerError("minimal320_teacher_shard_binding_drift")

    for pin_name, kind in _COMPONENT_PIN_TO_KIND.items():
        component_path = bindings["components"][kind]
        entry = entries_by_path[component_path]
        if binding["component_pins"][pin_name] != entry["sha256"]:
            raise Minimal320ConsumerError("minimal320_component_artifact_pin_drift")

    metadata_values: dict[str, Any] = {}
    for kind in _REQUIRED_METADATA_KINDS:
        entry = entries_by_path[bindings["metadata"][kind]]
        path = _safe_join(
            repo,
            entry["path"],
            reason="minimal320_metadata_path_invalid",
        )
        raw = raw_preimages[entry["path"]]
        if kind == "selected_candidate_inventory":
            metadata_values[kind] = _strict_candidate_inventory_rows(
                raw,
                reason=f"minimal320_{kind}_invalid",
            )
        elif kind == "accepted_inventory":
            metadata_values[kind] = _strict_accepted_inventory_rows(
                raw,
                reason=f"minimal320_{kind}_invalid",
            )
        elif kind == "training_record_inventory":
            metadata_values[kind] = _strict_training_inventory_rows(
                raw,
                reason=f"minimal320_{kind}_invalid",
            )
        elif kind in {"rejected_inventory", "quarantine_inventory"}:
            metadata_values[kind] = _strict_jsonl_rows(
                raw,
                required_keys=frozenset(),
                expected_records=0,
                reason=f"minimal320_{kind}_invalid",
            )
        else:
            metadata_values[kind] = _strict_json_value(
                raw,
                reason=f"minimal320_{kind}_invalid",
            )

    selection = binding["selection_manifest"]
    selection_sha = _canonical_sha256(selection)
    if (
        selection_sha != binding["selection_manifest_sha256"]
        or metadata_values["selection_manifest"] != selection
    ):
        raise Minimal320ConsumerError("minimal320_selection_manifest_preimage_drift")

    source_truth_entry = entries_by_path[
        bindings["components"]["source_truth_contract"]
    ]
    source_truth = _strict_json(
        raw_preimages[source_truth_entry["path"]],
        reason="minimal320_source_truth_contract_invalid",
    )
    if selection["source_truth_contract_sha256"] != source_truth_entry["sha256"]:
        raise Minimal320ConsumerError("minimal320_source_truth_contract_pin_drift")
    _validate_source_truth_contract(source_truth, selection)

    selected_rows = metadata_values["selected_candidate_inventory"]
    accepted_rows = metadata_values["accepted_inventory"]
    training_rows = metadata_values["training_record_inventory"]
    teacher_schema_entry = entries_by_path[
        bindings["components"]["teacher_record_schema"]
    ]
    teacher_schema = _strict_json(
        raw_preimages[teacher_schema_entry["path"]],
        reason="minimal320_teacher_record_schema_invalid",
    )
    teacher_rows: list[dict[str, Any]] = []
    for teacher_path in teacher_paths:
        entry = entries_by_path[teacher_path]
        teacher_rows.extend(
            _strict_jsonl_rows(
                raw_preimages[teacher_path],
                required_keys=_TEACHER_RECORD_KEYS,
                expected_records=int(entry["records"]),
                reason="minimal320_teacher_shard_invalid",
            )
        )
    joined_candidates = _validate_teacher_shards(
        schema=teacher_schema,
        shard_rows=teacher_rows,
        selected_rows=selected_rows,
        accepted_rows=accepted_rows,
        training_rows=training_rows,
    )
    _validate_committed_shards_manifest(
        metadata_values["committed_shards_manifest"],
        repo=repo,
        teacher_entry=entries_by_path[_TEACHER_SHARD_PATH],
        training_entry=entries_by_path[_TRAINING_RECORD_INVENTORY_PATH],
        teacher_rows=teacher_rows,
        training_rows=training_rows,
        accepted_entry=entries_by_path[bindings["metadata"]["accepted_inventory"]],
        selected_set_root_sha256=selection["selected_set_root_sha256"],
        teacher_schema_sha256=teacher_schema_entry["sha256"],
    )
    selected_root = _canonical_sha256(selected_rows)
    accepted_root = _canonical_sha256(accepted_rows)
    training_root = _canonical_sha256(training_rows)
    if selection["selected_set_root_sha256"] != selected_root or selection[
        "inventory_roots"
    ] != {
        "selected_candidates_sha256": selected_root,
        "accepted_sha256": accepted_root,
        "training_records_sha256": training_root,
    }:
        raise Minimal320ConsumerError("minimal320_record_inventory_root_drift")
    _validate_exact_strata(joined_candidates, selection)

    if (
        metadata_values["rejected_inventory"] != []
        or metadata_values["quarantine_inventory"] != []
    ):
        raise Minimal320ConsumerError("minimal320_nonaccepted_inventory_not_empty")
    _validate_checkpoint(metadata_values["checkpoint_manifest"], selection)

    freeze = binding["freeze_attestation"]
    freeze_sha = _canonical_sha256(freeze)
    checkpoint_entry = entries_by_path[bindings["metadata"]["checkpoint_manifest"]]
    payload_inventory = [
        entry
        for entry in inventory
        if entry["kind"] not in {"freeze_attestation", "external_attestation"}
    ]
    payload_inventory_sha = _canonical_sha256(payload_inventory)
    if (
        freeze_sha != binding["freeze_attestation_sha256"]
        or metadata_values["freeze_attestation"] != freeze
        or freeze["selection_manifest_sha256"] != selection_sha
        or freeze["selector_identity_sha256"] != selection["selector_identity_sha256"]
        or freeze["selected_set_root_sha256"] != selection["selected_set_root_sha256"]
        or freeze["payload_inventory_sha256"] != payload_inventory_sha
        or freeze["checkpoint_manifest_sha256"] != checkpoint_entry["sha256"]
    ):
        raise Minimal320ConsumerError("minimal320_freeze_attestation_preimage_drift")

    external = binding["external_attestation"]
    external_sha = _canonical_sha256(external)
    if (
        external_sha != binding["external_attestation_sha256"]
        or metadata_values["external_attestation"] != external
        or external["freeze_attestation_sha256"] != freeze_sha
        or external["selection_manifest_sha256"] != selection_sha
        or external["payload_inventory_sha256"] != payload_inventory_sha
        or external["component_pins_sha256"] != component_pins_sha
        or external["producer_commit"] != binding["producer_commit"]
        or external["producer_tree"] != binding["producer_tree"]
    ):
        raise Minimal320ConsumerError("minimal320_external_attestation_preimage_drift")

    binding_preimage = dict(binding)
    binding_id = binding_preimage.pop("binding_id")
    if _canonical_sha256(binding_preimage) != binding_id:
        raise Minimal320ConsumerError("minimal320_binding_preimage_drift")

    status = _git(
        repo,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *paths,
        ],
        reason="minimal320_producer_artifact_git_status_invalid",
    )
    if status:
        raise Minimal320ConsumerError("minimal320_producer_artifact_worktree_drift")
    for path, snapshot in physical_snapshots.items():
        try:
            if _identity(path.lstat()) != snapshot:
                raise Minimal320ConsumerError(
                    "minimal320_artifact_final_identity_drift"
                )
        except OSError as error:
            raise Minimal320ConsumerError(
                "minimal320_artifact_final_identity_drift"
            ) from error


def _validate_binding_counts(
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    exact = config["exact320"]
    selection = binding["selection_manifest"]
    by_role = selection["counts"]["by_role"]
    derived_assets = {_ROLE_TO_ASSET[role]: count for role, count in by_role.items()}
    if (
        derived_assets != exact["asset_record_counts"]
        or selection["counts"]["by_language"] != exact["language_counts"]
        or selection["counts"]["selected"] != exact["records"]
    ):
        raise Minimal320ConsumerError("minimal320_binding_counts_invalid")
    source_identity = selection["identity_training"]
    expected_identity = exact["identity_training"]
    derived_identity = {
        _IDENTITY_ROLE_TO_ASSET[role]: count
        for role, count in source_identity["expert_role_counts"].items()
    }
    if (
        source_identity["records"] != expected_identity["records"]
        or source_identity["bundles"] != expected_identity["bundles"]
        or derived_identity != expected_identity["expert_record_counts"]
        or source_identity["separate_output_expert"] is not False
        or source_identity["separate_training_arm"] is not False
        or "identity" in by_role
    ):
        raise Minimal320ConsumerError(
            "minimal320_binding_identity_distribution_invalid"
        )


def authenticate_release(
    *,
    producer_repo: Path,
) -> AuthenticatedRelease:
    # Consumer code P carries no Producer identity.  Only a later config-only
    # consumer pin C may place the reviewed release pin into trusted_release.
    loaded = load_config()
    stage = loaded.value["producer_release_stage"]
    anchor = stage["trusted_release"]
    if (
        loaded.value["state"] != "producer_final_release_frozen"
        or stage["state"] != "producer_final_release_frozen"
        or not isinstance(anchor, dict)
    ):
        raise Minimal320ConsumerError("producer_final_release_anchor_required")
    if _contains_placeholder(anchor) or _contains_zero_identity(anchor):
        raise Minimal320ConsumerError("minimal320_trusted_release_anchor_invalid")
    pin_schema = _load_repo_schema(
        loaded.value["schemas"]["producer_release_pin"],
        reason="minimal320_release_pin_schema_identity_drift",
    )
    _validate_schema(
        pin_schema,
        anchor,
        reason="minimal320_release_pin_schema_mismatch",
    )
    if (
        anchor["schema_version"] != RELEASE_PIN_SCHEMA_VERSION
        or anchor["state"] != "producer_final_R_frozen"
        or anchor["reviewed_by_different_agent"] is not True
    ):
        raise Minimal320ConsumerError("minimal320_release_pin_contract_drift")
    pin_sha = _canonical_sha256(anchor)

    # Authenticate consumer C and immutable consumer code P before touching
    # any Producer repository path.
    consumer_git = _consumer_git_identity(anchor, loaded)
    repo = _safe_root(
        producer_repo,
        reason="minimal320_producer_repo_path_invalid",
    )
    top = _git_text(
        repo,
        ["rev-parse", "--show-toplevel"],
        reason="minimal320_producer_git_repository_invalid",
    )
    try:
        if Path(top).resolve(strict=True) != repo:
            raise Minimal320ConsumerError("minimal320_producer_git_root_mismatch")
    except OSError as error:
        raise Minimal320ConsumerError(
            "minimal320_producer_git_root_mismatch"
        ) from error
    if _git_text(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace/"],
        reason="minimal320_producer_git_replace_ref_invalid",
    ):
        raise Minimal320ConsumerError("minimal320_producer_git_replace_ref_present")
    launch_inventory, payload_inventory, final_inventory = _verify_b_l_p_r_stage_chain(
        repo, anchor["producer_stages"]
    )

    pin = dict(anchor)
    binding_text = anchor["producer_binding_canonical_json"]
    if not isinstance(binding_text, str):
        raise Minimal320ConsumerError("minimal320_producer_binding_invalid")
    binding = _strict_json(
        binding_text.encode("utf-8"),
        reason="minimal320_producer_binding_invalid",
    )
    binding_raw = _canonical_bytes(binding)
    if binding_raw.decode("utf-8") != binding_text:
        raise Minimal320ConsumerError("minimal320_producer_binding_not_canonical")
    binding_sha = _sha256(binding_raw)
    binding_schema_pin = loaded.value["schemas"]["producer_final_binding"]
    if (
        anchor["producer_binding_schema_version"] != PRODUCER_BINDING_SCHEMA_VERSION
        or anchor["producer_binding_schema_sha256"] != binding_schema_pin["sha256"]
        or anchor["producer_binding_sha256"] != binding_sha
        or anchor["producer_binding_id"] != binding.get("binding_id")
        or pin["consumer_runtime_commit"] != anchor["consumer_runtime_commit"]
        or pin["consumer_runtime_tree"] != anchor["consumer_runtime_tree"]
        or pin["consumer_implementation_sha256"]
        != anchor["consumer_implementation_sha256"]
        or pin["consumer_schema_inventory_sha256"]
        != anchor["consumer_schema_inventory_sha256"]
    ):
        raise Minimal320ConsumerError("minimal320_release_pin_binding_drift")
    if _contains_placeholder(binding) or _contains_zero_identity(binding):
        raise Minimal320ConsumerError("minimal320_producer_binding_unfrozen_identity")
    binding_schema = _load_repo_schema(
        binding_schema_pin,
        reason="minimal320_producer_binding_schema_identity_drift",
    )
    _validate_schema(
        binding_schema,
        binding,
        reason="minimal320_producer_binding_schema_mismatch",
    )
    if (
        binding["schema_version"] != PRODUCER_BINDING_SCHEMA_VERSION
        or binding["binding_id"] != anchor["producer_binding_id"]
        or binding["producer_commit"]
        != anchor["producer_stages"]["producer_payload_P"]["commit"]
        or binding["producer_tree"]
        != anchor["producer_stages"]["producer_payload_P"]["tree"]
        or binding["producer_stages"] != anchor["producer_stages"]
    ):
        raise Minimal320ConsumerError("minimal320_release_pin_binding_drift")
    release_documents: dict[str, tuple[str, str, str]] = {}
    for kind in ("freeze_attestation", "external_attestation"):
        matches = [
            entry for entry in binding["artifact_inventory"] if entry["kind"] == kind
        ]
        if len(matches) != 1:
            raise Minimal320ConsumerError(
                "minimal320_producer_release_allowlist_invalid"
            )
        entry = matches[0]
        release_documents[kind] = (
            str(entry["path"]),
            str(entry["sha256"]),
            str(entry["git_blob_oid"]),
        )
    _validate_binding_preimages(
        binding,
        repo=repo,
        launch_inventory=launch_inventory,
        candidate_inventory=payload_inventory,
        release_inventory=final_inventory,
        release_documents=release_documents,
        reserved_identities=set(),
        reserved_paths=set(),
    )
    _validate_binding_counts(loaded.value, binding)
    # Terminally re-prove lineage and clean physical bytes after every read.
    final_launch, final_payload, final_release = _verify_b_l_p_r_stage_chain(
        repo,
        anchor["producer_stages"],
    )
    if (
        final_launch != launch_inventory
        or final_payload != payload_inventory
        or final_release != final_inventory
    ):
        raise Minimal320ConsumerError("minimal320_producer_terminal_identity_drift")
    return AuthenticatedRelease(
        trust_anchor=dict(anchor),
        pin=pin,
        pin_sha256=pin_sha,
        binding=binding,
        binding_sha256=binding_sha,
        consumer_git=consumer_git,
    )


def _arm_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    arms = config["arms"]
    definitions = arms["definitions"]
    return {
        "ordered": list(arms["ordered"]),
        "full_steps": {arm: definitions[arm]["full_steps"] for arm in arms["ordered"]},
        "profiles": {
            arm: definitions[arm]["adapter_profile"] for arm in arms["ordered"]
        },
        "source_assets": {
            arm: definitions[arm]["source_asset"] for arm in arms["ordered"]
        },
        "eval_only": list(arms["eval_only"]),
    }


def build_preflight_receipt(
    loaded: LoadedConfig,
    release: AuthenticatedRelease,
) -> dict[str, Any]:
    exact = loaded.value["exact320"]
    identity = exact["identity_training"]
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "state": "model_free_preflight_passed",
        "consumer_config_sha256": loaded.raw_sha256,
        "consumer_git_head": release.consumer_git.head,
        "consumer_git_tree": release.consumer_git.tree,
        "consumer_runtime_commit": release.consumer_git.runtime_commit,
        "consumer_runtime_tree": release.consumer_git.runtime_tree,
        "consumer_implementation_sha256": (release.consumer_git.implementation_sha256),
        "consumer_schema_inventory_sha256": (
            release.consumer_git.schema_inventory_sha256
        ),
        "consumer_config_schema_sha256": CONFIG_SCHEMA_SHA256,
        "producer_binding_schema_sha256": loaded.value["schemas"][
            "producer_final_binding"
        ]["sha256"],
        "producer_release_pin_schema_sha256": loaded.value["schemas"][
            "producer_release_pin"
        ]["sha256"],
        "preflight_receipt_schema_sha256": loaded.value["schemas"]["preflight_receipt"][
            "sha256"
        ],
        "release_pin_sha256": release.pin_sha256,
        "producer_binding_sha256": release.binding_sha256,
        "producer_binding_id": release.binding["binding_id"],
        "producer_commit": release.binding["producer_commit"],
        "producer_tree": release.binding["producer_tree"],
        "producer_release_commit": release.trust_anchor["producer_stages"][
            "producer_final_R"
        ]["commit"],
        "producer_release_tree": release.trust_anchor["producer_stages"][
            "producer_final_R"
        ]["tree"],
        "producer_evidence": {
            "artifact_inventory_sha256": release.binding["artifact_inventory_sha256"],
            "component_pins_sha256": release.binding["component_pins_sha256"],
            "selection_manifest_sha256": release.binding["selection_manifest_sha256"],
            "freeze_attestation_sha256": release.binding["freeze_attestation_sha256"],
            "external_attestation_sha256": release.binding[
                "external_attestation_sha256"
            ],
            "source_truth_contract_sha256": release.binding["component_pins"][
                "source_truth_contract_sha256"
            ],
            **release.binding["selection_manifest"]["inventory_roots"],
        },
        "records": exact["records"],
        "asset_record_counts": dict(exact["asset_record_counts"]),
        "identity_training": {
            "records": identity["records"],
            "bundles": identity["bundles"],
            "expert_record_counts": dict(identity["expert_record_counts"]),
            "included_in_asset_record_counts": (
                identity["included_in_asset_record_counts"]
            ),
            "separate_training_arm": identity["separate_training_arm"],
        },
        "arms": _arm_receipt(loaded.value),
        "legacy_v2_semantics_modified": False,
        "claims": {
            "producer_binding_authenticated": True,
            "exact320_authenticated": True,
            "nine_arm_plan_ready": True,
            "gpu_started": False,
            "api_called": False,
            "training_started": False,
            "formal_training_authorized": False,
        },
    }
    receipt = {"receipt_id": _canonical_sha256(payload), **payload}
    receipt_schema = _load_repo_schema(
        loaded.value["schemas"]["preflight_receipt"],
        reason="minimal320_preflight_receipt_schema_identity_drift",
    )
    _validate_schema(
        receipt_schema,
        receipt,
        reason="minimal320_preflight_receipt_schema_mismatch",
    )
    return receipt


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise Minimal320ConsumerError("minimal320_preflight_output_not_new") from error


def _blocked_status(reason: str) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": "blocked",
        "blockers": [reason],
        "live_ready": False,
        "gpu_started": False,
        "api_called": False,
        "training_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free strict minimal-320 consumer preflight."
    )
    parser.add_argument("--producer-repo", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        loaded = load_config()
        if args.producer_repo is None:
            print(
                json.dumps(
                    _blocked_status("producer_repository_required"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 2
        release = authenticate_release(
            producer_repo=args.producer_repo,
        )
        receipt = build_preflight_receipt(loaded, release)
        if args.output is not None:
            _write_new_json(args.output, receipt)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except Minimal320ConsumerError as error:
        print(
            json.dumps(
                _blocked_status(error.reason_code),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
