"""Explicit real-GPU runtime for the Gemma 3 unbalanced-v2 multi-arm plan.

The v1 module remains the model-free contract.  This additive v2 runtime has
two explicit execution modes: nine-arm two-step smoke, and a separate full
run that consumes no smoke checkpoint.  Both modes require:

* immutable Producer P/R/tree/read-set authentication;
* an independently released 3,520-row Teacher FINAL (no original-target
  fallback);
* authenticated model-proto tokenizer bytes and the checked-in Gemma
  chat-template policy;
* a canonical single-GPU lock, launcher lease, and GPU attestation.

Heavy ML imports are lazy.  Importing or validating this module never loads a
model, requests CUDA, reads dataset bodies, or contacts a provider.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Final

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as contract
from . import gemma3_chat_unbalanced_v2_runtime_source_v2 as source_runtime
from . import gemma3_five_role_q8_qlora_v2 as q8_runtime
from . import qwen_lora_diagnostic as qdiag
from .config import ConfigError, _expand_env


CONFIG_VERSION: Final = (
    "anchor.gemma3-1b-it-chat-unbalanced-v2-multiarm-q8-runtime-config.v2"
)
TEACHER_BINDING_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2"
)
TEACHER_RECORD_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
)
TEACHER_SOURCE_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-authenticated-teacher-final.v2"
)
PHASE_RECEIPT_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-phase-receipt.v2"
)
RUN_RECEIPT_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-run-receipt.v2"
)
FAILURE_RECEIPT_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-failure-receipt.v2"
)
LINEAGE_VERSION: Final = "anchor.gemma3-chat-unbalanced-v2-multiarm-training-lineage.v2"
LEASE_VERSION: Final = "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-execution-lease.v2"
ATTESTATION_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-gpu-attestation.v2"
)

CONFIG_PATH: Final = Path(
    "configs/training/gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.yaml"
)
CONFIG_SCHEMA_PATH: Final = Path(
    "configs/training/"
    "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.schema.json"
)
TEACHER_BINDING_SCHEMA_PATH: Final = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_binding_consumer_rollover_v2.schema.json"
)
TEACHER_RECORD_SCHEMA_PATH: Final = Path(
    "configs/training/gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
)
PHASE_SCHEMA_PATH: Final = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_multiarm_q8_runtime_phase_receipt_v2.schema.json"
)
RUN_SCHEMA_PATH: Final = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_multiarm_q8_runtime_run_receipt_v2.schema.json"
)
FAILURE_SCHEMA_PATH: Final = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_multiarm_q8_runtime_failure_receipt_v2.schema.json"
)
IMPLEMENTATION_PATH: Final = Path(
    "src/anchor_mvp/training/gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.py"
)
SHARED_SOURCE_PATH: Final = Path(
    "src/anchor_mvp/training/gemma3_chat_unbalanced_v2_runtime_source_v2.py"
)
RUNNER_PATH: Final = Path(
    "scripts/research/run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.py"
)
LAUNCHER_PATH: Final = Path(
    "scripts/research/run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.ps1"
)
LAUNCHER_HELPER_PATH: Final = Path("scripts/research/gemma3_q8_launcher_helpers_v2.ps1")
MODEL_FREE_CONFIG_PATH: Final = Path(
    "configs/training/gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1.yaml"
)
TEST_PATH: Final = Path(
    "tests/test_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.py"
)

TRAIN_SHARDS: Final = (
    "train/chat-00000-of-00002.jsonl",
    "train/chat-00001-of-00002.jsonl",
)
ROUTER_RECORDS: Final = "components/router/records.jsonl"
IDENTITY_PROBE_RECORDS: Final = "identity_eval/probe_inventory.jsonl"
MAIN_ROLES: Final = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
    "review_audit",
)
EXPECTED_ASSET_COUNTS: Final = {
    "humor": 240,
    "serious": 240,
    "angry_style": 240,
    "tool_call": 1360,
    "review_audit": 1360,
    "planner_router": 80,
}
RUNTIME_PACKAGE_NAMES: Final = (
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "sentencepiece",
    "bitsandbytes",
)
_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_LORA_RE: Final = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.(q_proj|o_proj)\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)
_MIB: Final = 1024 * 1024
_MAX_METADATA_BYTES: Final = 16_000_000
_TEACHER_AUTHORITY = object()
ROLLOVER_PRODUCER_BRANCH: Final = "research/gemma3-teacher-final-consumer-rollover-v3"
ROLLOVER_PRODUCER_COMMIT: Final = "878e27dbc41e8dafc43b6462279a814f53d232b5"
ROLLOVER_PRODUCER_PARENT: Final = "f23ed037983a5556fe309c92a5391c81a02da5f2"
ROLLOVER_PRODUCER_TREE: Final = "566e84d30bc934def4a52b48f3fd65b2639665ae"
ROLLOVER_BINDING_FILENAME: Final = "teacher_alignment_binding.v2.json"
ROLLOVER_PHYSICAL_IDENTITY_SPECS: Final = {
    "gitattributes": {
        "path": ".gitattributes",
        "sha256": "345238b9d68eb55865d5bd11e7dfe58cb332ea060fd7d371a04e6327a56b2276",
        "bytes": 17848,
        "kind": "text",
    },
    "config": {
        "path": (
            "configs/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.json"
        ),
        "sha256": "6664bee43d144d07673ac3457020727180bb047f52bedb05b09aa2cd69e37ae5",
        "bytes": 3475,
        "kind": "json",
    },
    "config_schema": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.schema.json"
        ),
        "sha256": "b459f014f959817887be47a172d366a9ddfce41b06d3d4e89feb26f360e63515",
        "bytes": 3523,
        "kind": "json_schema",
    },
    "attestation_schema": {
        "path": (
            "configs/data/gemma3_chat_unbalanced_v2_teacher_final_release_"
            "attestation_consumer_rollover_v3.schema.json"
        ),
        "sha256": "926a86b8e6c2e92d81097f8a4ac4a2e41c5449ebd956e4c90791562170895486",
        "bytes": 13703,
        "kind": "json_schema",
    },
    "release_manifest_schema": {
        "path": (
            "configs/data/gemma3_chat_unbalanced_v2_teacher_final_release_manifest_"
            "consumer_rollover_v3.schema.json"
        ),
        "sha256": "865b5fe67afbdb67c70bcdabb862544fd9a6b9abcbd9c148e6d1f0102e81936c",
        "bytes": 9408,
        "kind": "json_schema",
    },
    "execute_implementation": {
        "path": (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.py"
        ),
        "sha256": "a80c3c85239c033412ced8e95328c9189e4207ecde99a150f6f3e91e8c920086",
        "bytes": 17007,
        "kind": "python",
    },
    "release_implementation": {
        "path": (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v3.py"
        ),
        "sha256": "75143aaf366bbb9db0e0d81d7763d0b901d045d4e2c45c45deb078d91e0a75fc",
        "bytes": 63615,
        "kind": "python",
    },
    "tests": {
        "path": (
            "tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.py"
        ),
        "sha256": "2c9ce0077b6e6859677c12ab8d7dd0cdc0dc3f755a01e1f6f3f02e6d99932e40",
        "bytes": 15515,
        "kind": "python",
    },
}
ROLLOVER_SCHEMA_SPECS: Final = {
    "binding": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
            "consumer_rollover_v2.schema.json"
        ),
        "sha256": ("491eb5a085884a17fcd02ec7335f0ff226a77bf701dd984947175c04e895faca"),
        "bytes": 8773,
    },
    "release_manifest": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_"
            "consumer_rollover_v3.schema.json"
        ),
        "sha256": ("865b5fe67afbdb67c70bcdabb862544fd9a6b9abcbd9c148e6d1f0102e81936c"),
        "bytes": 9408,
    },
    "attestation": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_"
            "attestation_consumer_rollover_v3.schema.json"
        ),
        "sha256": ("926a86b8e6c2e92d81097f8a4ac4a2e41c5449ebd956e4c90791562170895486"),
        "bytes": 13703,
    },
    "candidate_manifest": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_manifest_"
            "consumer_rollover_v2.schema.json"
        ),
        "sha256": ("9552f8091cfd6d0de61a0b2cf5559a8460ac41d74a06e1af10392747d2f63e13"),
        "bytes": 10162,
    },
    "build_receipt": {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_build_receipt_v2.schema.json"
        ),
        "sha256": ("2042d5e50491e31cc9ae99dd1d6f1de9329103a551e81c8579da52c5ffe39404"),
        "bytes": 2972,
    },
    "record": {
        "path": (
            "configs/data/gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
        ),
        "sha256": ("5a02990152721f0dba39fff85ab07417d76b4d0a5fa08c95eb4c4e327eb52bef"),
        "bytes": 1558,
    },
}
ROLLOVER_RELEASE_IMPLEMENTATION: Final = {
    "path": (
        "src/anchor_mvp/data/"
        "gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v3.py"
    ),
    "sha256": "75143aaf366bbb9db0e0d81d7763d0b901d045d4e2c45c45deb078d91e0a75fc",
    "bytes": 63615,
}


class MultiArmRuntimeError(RuntimeError):
    """One fail-closed runtime contract failed."""


class PhaseExecutionError(MultiArmRuntimeError):
    """One real-GPU arm failed after orchestration began."""

    def __init__(self, arm: str, phase: str, cause: BaseException) -> None:
        super().__init__(f"runtime_phase_failed:{arm}:{phase}:{type(cause).__name__}")
        self.arm = arm
        self.phase = phase
        self.__cause__ = cause


@dataclass(frozen=True)
class TeacherFilePin:
    path: Path
    sha256: str
    bytes: int
    sidecar_sha256: str
    sidecar_bytes: int


@dataclass(frozen=True)
class AuthenticatedRolloverProducer:
    root: Path
    commit: str
    parent: str
    tree: str
    physical_identity_sha256: Mapping[str, str]
    schema_values: Mapping[str, Mapping[str, Any]]
    schema_sha256: Mapping[str, str]
    execute_implementation_sha256: str
    release_implementation_sha256: str
    _directory_identities: tuple[DirectoryIdentity, ...]


@dataclass(frozen=True)
class DirectoryIdentity:
    path: Path
    device: int
    inode: int
    mode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class FileObjectIdentity:
    path: Path
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class AuthenticatedTeacherFinal:
    schema_version: str
    binding_path: Path
    binding_sha256: str
    binding_sidecar_sha256: str
    binding_sidecar_bytes: int
    binding_schema_sha256: str
    artifact_root: Path
    manifest: TeacherFilePin
    record_schema_sha256: str
    release_receipt: TeacherFilePin
    release_attestation: TeacherFilePin
    metadata_files: tuple[TeacherFilePin, ...]
    shards: tuple[TeacherFilePin, ...]
    shard_assets: tuple[str, ...]
    shard_records: tuple[int, ...]
    shard_inventory_sha256: str
    public_identity_sha256: str
    producer: AuthenticatedRolloverProducer
    _directory_identities: tuple[DirectoryIdentity, ...]
    _authority: object

    def public_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "binding_schema_version": TEACHER_BINDING_VERSION,
            "record_schema_version": TEACHER_RECORD_VERSION,
            "binding_sha256": self.binding_sha256,
            "binding_schema_sha256": self.binding_schema_sha256,
            "manifest_sha256": self.manifest.sha256,
            "record_schema_sha256": self.record_schema_sha256,
            "release_receipt_sha256": self.release_receipt.sha256,
            "release_attestation_sha256": self.release_attestation.sha256,
            "shard_inventory_sha256": self.shard_inventory_sha256,
            "records": sum(self.shard_records),
            "main_records": sum(
                count
                for asset, count in zip(
                    self.shard_assets, self.shard_records, strict=True
                )
                if asset == "main_train"
            ),
            "router_records": sum(
                count
                for asset, count in zip(
                    self.shard_assets, self.shard_records, strict=True
                )
                if asset == "planner_router_train"
            ),
            "train_identity_bundles": 20,
            "train_identity_records": 100,
            "independent_identity_eval_bundles_not_in_optimizer": 10,
            "independent_identity_eval_probes_not_in_optimizer": 50,
            "accepted": 3520,
            "rejected": 0,
            "uncertain": 0,
            "producer_original_target_fallback": False,
            "public_identity_sha256": self.public_identity_sha256,
        }


@dataclass(frozen=True)
class SourcePrompt:
    record_id_sha256: str
    source_asset: str
    prompt: str
    source_content_sha256: str
    source_serialization_identity_sha256: str
    task_bundle_sha256: str
    identity_training_record: bool


@dataclass(frozen=True)
class TeacherTarget:
    record_id_sha256: str
    source_asset: str
    source_content_sha256: str
    source_serialization_identity_sha256: str
    task_bundle_sha256: str
    target: str
    target_sha256: str


@dataclass(frozen=True)
class TrainingExample:
    record_id_sha256: str
    source_asset: str
    prompt: str
    target: str
    source_content_sha256: str
    source_serialization_identity_sha256: str
    task_bundle_sha256: str
    teacher_target_sha256: str


@dataclass(frozen=True)
class SerializedTrainingExample:
    record_id_sha256: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    serialization_identity_sha256: str


@dataclass(frozen=True)
class RuntimeDatasets:
    assets: Mapping[str, tuple[TrainingExample, ...]]
    serialized_assets: Mapping[str, tuple[SerializedTrainingExample, ...]]
    source_record_order_sha256: str
    teacher_target_projection_sha256: str
    serialized_inventory_sha256: str
    max_sequence_tokens: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiArmRuntimeError(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise MultiArmRuntimeError(code)
    return value


def _require_sha(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MultiArmRuntimeError(code)
    return value


def _decimal(value: object, code: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation:
        raise MultiArmRuntimeError(code) from None


def _strict_json(raw: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise MultiArmRuntimeError(code) from None
    if not isinstance(value, Mapping):
        raise MultiArmRuntimeError(code)
    return value


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _stable_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    code: str,
) -> bytes:
    try:
        first = path.lstat()
        if stat.S_ISLNK(first.st_mode) or _is_reparse(first):
            raise MultiArmRuntimeError(code)
        raw = path.read_bytes()
        second = path.lstat()
    except MultiArmRuntimeError:
        raise
    except OSError:
        raise MultiArmRuntimeError(code) from None
    first_identity = (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    )
    second_identity = (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )
    if (
        first_identity != second_identity
        or (expected_bytes is not None and len(raw) != expected_bytes)
        or _sha256(raw) != expected_sha256
    ):
        raise MultiArmRuntimeError(code)
    return raw


def _directory_identity(path: Path, *, code: str) -> DirectoryIdentity:
    try:
        value = path.lstat()
    except OSError:
        raise MultiArmRuntimeError(code) from None
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse(value)
    ):
        raise MultiArmRuntimeError(code)
    return DirectoryIdentity(
        path=path,
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        modified_ns=int(value.st_mtime_ns),
        changed_ns=int(value.st_ctime_ns),
    )


def _directory_chain(
    root: Path,
    targets: Sequence[Path],
    *,
    code: str,
) -> tuple[DirectoryIdentity, ...]:
    """Pin every lexical parent component; junctions/reparse points fail."""

    root = Path(os.path.abspath(root))
    identities: dict[Path, DirectoryIdentity] = {
        root: _directory_identity(root, code=code)
    }
    for target in targets:
        lexical = Path(os.path.abspath(target))
        try:
            relative = lexical.relative_to(root)
        except ValueError:
            raise MultiArmRuntimeError(code) from None
        current = root
        parts = relative.parts if lexical.is_dir() else relative.parts[:-1]
        for part in parts:
            current = current / part
            identities.setdefault(current, _directory_identity(current, code=code))
    return tuple(
        identities[path] for path in sorted(identities, key=lambda item: str(item))
    )


def _recheck_directory_chain(
    identities: Sequence[DirectoryIdentity],
    *,
    code: str,
) -> None:
    for expected in identities:
        observed = _directory_identity(expected.path, code=code)
        if observed != expected:
            raise MultiArmRuntimeError(code)


def _schema_from_authenticated_git(
    raw: bytes,
    *,
    code: str,
) -> Mapping[str, Any]:
    value = _strict_json(raw, code)
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError:
        raise MultiArmRuntimeError(code) from None
    return value


def _rollover_producer_root(
    config: Mapping[str, Any],
    repository: str | Path | None,
) -> Path:
    rollover = _mapping(
        config.get("teacher_rollover"),
        "teacher_rollover_config_invalid",
    )
    raw: str
    if repository is not None and str(repository).strip():
        raw = str(repository)
    else:
        environment_name = str(rollover.get("producer_repository_env", ""))
        environment_value = os.environ.get(environment_name, "")
        raw = (
            environment_value
            if environment_value
            else str(rollover.get("producer_repository_default", ""))
        )
    if not raw or raw != raw.strip():
        raise MultiArmRuntimeError("teacher_rollover_producer_root_missing")
    requested = Path(raw)
    if not requested.is_absolute():
        requested = _project_root() / requested
    root = Path(os.path.abspath(requested))
    _directory_identity(root, code="teacher_rollover_producer_root_invalid")
    return root


def _authenticate_rollover_producer_root(
    root: Path,
) -> AuthenticatedRolloverProducer:
    """Authenticate the additive rollover commit and exact schema Git blobs."""

    root = Path(os.path.abspath(root))
    root_chain = _directory_chain(
        root,
        [root],
        code="teacher_rollover_producer_root_invalid",
    )
    try:
        reader = source_runtime.GitObjectReader(root)
        if reader.object_type(ROLLOVER_PRODUCER_COMMIT) != "commit":
            raise MultiArmRuntimeError("teacher_rollover_producer_commit_invalid")
        observed_tree = reader.tree_for_commit(ROLLOVER_PRODUCER_COMMIT)
        if observed_tree != ROLLOVER_PRODUCER_TREE:
            raise MultiArmRuntimeError("teacher_rollover_producer_tree_drift")
        tree = reader.tree_entries(ROLLOVER_PRODUCER_COMMIT)
        physical_identity_sha256: dict[str, str] = {}
        physical_identity_raw: dict[str, bytes] = {}
        for name, expected in ROLLOVER_PHYSICAL_IDENTITY_SPECS.items():
            pin = reader.verify_blob(
                tree,
                path=str(expected["path"]),
                expected_sha256=str(expected["sha256"]),
                expected_bytes=int(expected["bytes"]),
                kind=str(expected["kind"]),
            )
            physical_identity_sha256[name] = pin.sha256
            physical_identity_raw[name] = reader.blob_bytes(pin.oid)
        identity_config = _strict_json(
            physical_identity_raw["config"],
            "teacher_rollover_v3_config_invalid",
        )
        identity_schema = _schema_from_authenticated_git(
            physical_identity_raw["config_schema"],
            code="teacher_rollover_v3_config_schema_invalid",
        )
        try:
            Draft202012Validator(identity_schema).validate(identity_config)
        except ValidationError:
            raise MultiArmRuntimeError(
                "teacher_rollover_v3_config_schema_mismatch"
            ) from None
        authenticated_artifacts = _mapping(
            identity_config.get("authenticated_artifacts"),
            "teacher_rollover_v3_authenticated_artifacts_invalid",
        )
        expected_v3_artifacts = {
            "rollover_v3_implementation": "execute_implementation",
            "release_implementation": "release_implementation",
            "attestation_schema": "attestation_schema",
            "release_manifest_schema": "release_manifest_schema",
        }
        if (
            identity_config.get("schema_version")
            != "anchor.gemma3-chat-unbalanced-v2-consumer-identity-rollover.v3"
            or identity_config.get("status") != "ready_model_free"
            or any(
                dict(
                    _mapping(
                        authenticated_artifacts.get(config_name),
                        "teacher_rollover_v3_authenticated_artifact_invalid",
                    )
                )
                != {
                    key: value
                    for key, value in ROLLOVER_PHYSICAL_IDENTITY_SPECS[
                        identity_name
                    ].items()
                    if key != "kind"
                }
                for config_name, identity_name in expected_v3_artifacts.items()
            )
            or dict(
                _mapping(
                    authenticated_artifacts.get("binding_schema"),
                    "teacher_rollover_v3_binding_schema_invalid",
                )
            )
            != ROLLOVER_SCHEMA_SPECS["binding"]
            or identity_config.get("claims", {}).get("consumer_prerequisite_commit")
            != "6240ae111182104f22f08e1a569deae866c6e210"
            or identity_config.get("claims", {}).get("binding_schema_sha256")
            != ROLLOVER_SCHEMA_SPECS["binding"]["sha256"]
        ):
            raise MultiArmRuntimeError("teacher_rollover_v3_contract_drift")
        schema_values: dict[str, Mapping[str, Any]] = {}
        schema_sha256: dict[str, str] = {}
        for name, expected in ROLLOVER_SCHEMA_SPECS.items():
            pin = reader.verify_blob(
                tree,
                path=str(expected["path"]),
                expected_sha256=str(expected["sha256"]),
                expected_bytes=int(expected["bytes"]),
                kind="json_schema",
            )
            raw = reader.blob_bytes(pin.oid)
            schema_values[name] = _schema_from_authenticated_git(
                raw,
                code=f"teacher_rollover_{name}_schema_invalid",
            )
            schema_sha256[name] = pin.sha256
        implementation = reader.verify_blob(
            tree,
            path=str(ROLLOVER_RELEASE_IMPLEMENTATION["path"]),
            expected_sha256=str(ROLLOVER_RELEASE_IMPLEMENTATION["sha256"]),
            expected_bytes=int(ROLLOVER_RELEASE_IMPLEMENTATION["bytes"]),
            kind="python",
        )
        reader.assert_no_replacement_or_graft()
    except source_runtime.RuntimeSourceError as exc:
        raise MultiArmRuntimeError(
            f"teacher_rollover_git_authentication_failed:{exc}"
        ) from None

    _stable_file(
        _project_root() / TEACHER_BINDING_SCHEMA_PATH,
        expected_sha256=schema_sha256["binding"],
        expected_bytes=int(ROLLOVER_SCHEMA_SPECS["binding"]["bytes"]),
        code="teacher_binding_schema_local_identity_drift",
    )
    _stable_file(
        _project_root() / TEACHER_RECORD_SCHEMA_PATH,
        expected_sha256=schema_sha256["record"],
        expected_bytes=int(ROLLOVER_SCHEMA_SPECS["record"]["bytes"]),
        code="teacher_record_schema_local_identity_drift",
    )
    _recheck_directory_chain(
        root_chain,
        code="teacher_rollover_producer_root_changed",
    )
    return AuthenticatedRolloverProducer(
        root=root,
        commit=ROLLOVER_PRODUCER_COMMIT,
        parent=ROLLOVER_PRODUCER_PARENT,
        tree=ROLLOVER_PRODUCER_TREE,
        physical_identity_sha256=physical_identity_sha256,
        schema_values=schema_values,
        schema_sha256=schema_sha256,
        execute_implementation_sha256=physical_identity_sha256[
            "execute_implementation"
        ],
        release_implementation_sha256=implementation.sha256,
        _directory_identities=root_chain,
    )


def authenticate_rollover_producer(
    config: Mapping[str, Any],
    repository: str | Path | None,
) -> AuthenticatedRolloverProducer:
    return _authenticate_rollover_producer_root(
        _rollover_producer_root(config, repository)
    )


def _recheck_rollover_producer(
    expected: AuthenticatedRolloverProducer,
) -> None:
    _recheck_directory_chain(
        expected._directory_identities,
        code="teacher_rollover_producer_root_changed",
    )
    observed = _authenticate_rollover_producer_root(expected.root)
    if (
        observed.commit != expected.commit
        or observed.parent != expected.parent
        or observed.tree != expected.tree
        or dict(observed.physical_identity_sha256)
        != dict(expected.physical_identity_sha256)
        or dict(observed.schema_sha256) != dict(expected.schema_sha256)
        or observed.execute_implementation_sha256
        != expected.execute_implementation_sha256
        or observed.release_implementation_sha256
        != expected.release_implementation_sha256
    ):
        raise MultiArmRuntimeError("teacher_rollover_producer_identity_changed")


def _assert_owned_directory(
    expected: DirectoryIdentity,
    *,
    path: Path | None = None,
    code: str,
) -> DirectoryIdentity:
    """Recheck an owned directory object without treating child writes as drift."""

    observed = _directory_identity(path or expected.path, code=code)
    if (
        observed.device,
        observed.inode,
        observed.mode,
    ) != (
        expected.device,
        expected.inode,
        expected.mode,
    ):
        raise MultiArmRuntimeError(code)
    return observed


def _windows_open_directory_fd(
    path: Path,
    *,
    delete_access: bool,
    code: str,
) -> int:
    """Open one directory object without sharing delete/rename authority."""

    if os.name != "nt":
        raise MultiArmRuntimeError(f"{code}_windows_handle_required")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    desired_access = 0x00000080 | 0x00100000
    if delete_access:
        desired_access |= 0x00010000
    handle = create_file(
        str(path),
        desired_access,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise MultiArmRuntimeError(code)
    try:
        return int(msvcrt.open_osfhandle(int(handle), os.O_RDONLY))
    except OSError:
        kernel32.CloseHandle(handle)
        raise MultiArmRuntimeError(code) from None


def _assert_owned_directory_fd(
    fd: int,
    expected: DirectoryIdentity,
    *,
    path: Path,
    code: str,
) -> DirectoryIdentity:
    try:
        value = os.fstat(fd)
    except OSError:
        raise MultiArmRuntimeError(code) from None
    if not stat.S_ISDIR(value.st_mode) or (
        int(value.st_dev),
        int(value.st_ino),
        stat.S_IFMT(int(value.st_mode)),
    ) != (
        expected.device,
        expected.inode,
        stat.S_IFMT(expected.mode),
    ):
        raise MultiArmRuntimeError(code)
    return DirectoryIdentity(
        path=path,
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        modified_ns=int(value.st_mtime_ns),
        changed_ns=int(value.st_ctime_ns),
    )


def _windows_rename_directory_handle(
    source_fd: int,
    destination: Path,
    *,
    code: str,
) -> None:
    """Rename the already-open source while its destination parent is pinned."""

    if os.name != "nt" or not destination.name:
        raise MultiArmRuntimeError(code)
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileRenameInfoEx(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    encoded = str(Path(os.path.abspath(destination))).encode("utf-16-le")
    offset = FileRenameInfoEx.FileName.offset
    buffer = ctypes.create_string_buffer(offset + len(encoded) + 2)
    info = ctypes.cast(buffer, ctypes.POINTER(FileRenameInfoEx)).contents
    info.Flags = 0
    info.RootDirectory = None
    info.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.BOOL
    if not set_information(
        wintypes.HANDLE(msvcrt.get_osfhandle(source_fd)),
        22,
        buffer,
        len(buffer),
    ):
        raise MultiArmRuntimeError(f"{code}_windows_error_{ctypes.get_last_error()}")


def _windows_open_file_fd(path: Path, *, code: str) -> int:
    if os.name != "nt":
        raise MultiArmRuntimeError(f"{code}_windows_handle_required")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000 | 0x00000080 | 0x00100000,
        0x00000001,
        None,
        3,
        0x00200000 | 0x08000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise MultiArmRuntimeError(code)
    try:
        return int(msvcrt.open_osfhandle(int(handle), os.O_RDONLY))
    except OSError:
        kernel32.CloseHandle(handle)
        raise MultiArmRuntimeError(code) from None


def _sha256_fd(fd: int, *, expected_bytes: int, code: str) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    except OSError:
        raise MultiArmRuntimeError(code) from None
    if total != expected_bytes:
        raise MultiArmRuntimeError(code)
    return digest.hexdigest()


def _file_object_identity_fd(
    fd: int,
    *,
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    code: str,
) -> FileObjectIdentity:
    try:
        value = os.fstat(fd)
    except OSError:
        raise MultiArmRuntimeError(code) from None
    if (
        not stat.S_ISREG(value.st_mode)
        or _is_reparse(value)
        or int(value.st_size) != expected_bytes
        or _sha256_fd(
            fd,
            expected_bytes=expected_bytes,
            code=code,
        )
        != expected_sha256
    ):
        raise MultiArmRuntimeError(code)
    return FileObjectIdentity(
        path=path,
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
        modified_ns=int(value.st_mtime_ns),
        changed_ns=int(value.st_ctime_ns),
    )


def _capture_file_object_identity(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    code: str,
) -> FileObjectIdentity:
    fd = _windows_open_file_fd(path, code=code)
    try:
        return _file_object_identity_fd(
            fd,
            path=path,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            code=code,
        )
    finally:
        os.close(fd)


def _retain_private_model_snapshot_by_handle(
    snapshot: Path,
    *,
    owned_identity: DirectoryIdentity,
    contract_value: Mapping[str, tuple[int, str]],
    file_identities: Mapping[str, FileObjectIdentity],
) -> None:
    """Authenticate and retain the entire private snapshot as failure evidence."""

    directory_fd = _windows_open_directory_fd(
        snapshot,
        delete_access=False,
        code="runtime_private_snapshot_directory_handle_open_failed",
    )
    file_fds: list[int] = []
    try:
        _assert_owned_directory_fd(
            directory_fd,
            owned_identity,
            path=snapshot,
            code="runtime_private_snapshot_directory_identity_drift",
        )
        try:
            names = {entry.name for entry in os.scandir(snapshot)}
        except OSError:
            raise MultiArmRuntimeError(
                "runtime_private_snapshot_inventory_unreadable"
            ) from None
        expected_names = set(q8_runtime.MODEL_FILES)
        if (
            names != expected_names
            or set(contract_value) != expected_names
            or set(file_identities) != expected_names
        ):
            raise MultiArmRuntimeError("runtime_private_snapshot_inventory_drift")
        for name in q8_runtime.MODEL_FILES:
            fd = _windows_open_file_fd(
                snapshot / name,
                code="runtime_private_snapshot_file_handle_open_failed",
            )
            file_fds.append(fd)
            expected_bytes, expected_sha = contract_value[name]
            observed = _file_object_identity_fd(
                fd,
                path=snapshot / name,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
                code="runtime_private_snapshot_file_identity_drift",
            )
            if observed != file_identities[name]:
                raise MultiArmRuntimeError(
                    "runtime_private_snapshot_file_identity_drift"
                )
        try:
            final_names = {entry.name for entry in os.scandir(snapshot)}
        except OSError:
            raise MultiArmRuntimeError(
                "runtime_private_snapshot_inventory_unreadable"
            ) from None
        if final_names != expected_names:
            raise MultiArmRuntimeError(
                "runtime_private_snapshot_inventory_changed_before_retention"
            )
        _assert_owned_directory_fd(
            directory_fd,
            owned_identity,
            path=snapshot,
            code="runtime_private_snapshot_directory_changed_before_retention",
        )
    finally:
        for fd in reversed(file_fds):
            os.close(fd)
        os.close(directory_fd)


@contextmanager
def _private_model_snapshot(
    config: Mapping[str, Any],
    run_id: str,
) -> Iterator[tuple[Path, Mapping[str, str]]]:
    launch_root = q8_runtime._root().joinpath(
        *q8_runtime._output_relative(config["output"]["launch_root"]).parts
    )
    launch_root.mkdir(parents=True, exist_ok=True)
    snapshot = launch_root / f".private-model-{run_id}-{uuid.uuid4().hex}"
    snapshot.mkdir(exist_ok=False)
    owned_identity = _directory_identity(
        snapshot,
        code="runtime_private_snapshot_directory_identity_invalid",
    )
    contract_value = q8_runtime._model_file_contract(config)
    hashes: dict[str, str] = {}
    file_identities: dict[str, FileObjectIdentity] = {}
    try:
        source = q8_runtime._model_root(config)
        for name in q8_runtime.MODEL_FILES:
            expected_bytes, expected_sha = contract_value[name]
            q8_runtime._copy_authenticated_file(
                source / name,
                snapshot / name,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
            )
            hashes[name] = expected_sha
            file_identities[name] = _capture_file_object_identity(
                snapshot / name,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
                code="runtime_private_snapshot_file_identity_capture_failed",
            )
        yield snapshot, hashes
        for name in q8_runtime.MODEL_FILES:
            expected_bytes, expected_sha = contract_value[name]
            _stable_file(
                snapshot / name,
                expected_sha256=expected_sha,
                expected_bytes=expected_bytes,
                code="runtime_private_snapshot_changed_before_retention",
            )
    finally:
        if os.path.lexists(snapshot):
            _retain_private_model_snapshot_by_handle(
                snapshot,
                owned_identity=owned_identity,
                contract_value=contract_value,
                file_identities=file_identities,
            )


def _move_directory_create_once(
    source: Path,
    destination: Path,
    *,
    owned_identity: DirectoryIdentity,
    code: str,
) -> DirectoryIdentity:
    """Rename an owned directory by handle, never by a re-resolved path."""

    if os.name != "nt":
        raise MultiArmRuntimeError(f"{code}_handle_bound_rename_unsupported")
    destination_parent = destination.parent
    parent_identity = _directory_identity(
        destination_parent,
        code=f"{code}_destination_parent_invalid",
    )
    source_fd = _windows_open_directory_fd(
        source,
        delete_access=True,
        code=f"{code}_source_handle_open_failed",
    )
    parent_fd = -1
    try:
        _assert_owned_directory_fd(
            source_fd,
            owned_identity,
            path=source,
            code=f"{code}_source_changed_handle_identity_drift",
        )
        parent_fd = _windows_open_directory_fd(
            destination_parent,
            delete_access=True,
            code=f"{code}_destination_parent_handle_open_failed",
        )
        _assert_owned_directory_fd(
            parent_fd,
            parent_identity,
            path=destination_parent,
            code=f"{code}_destination_parent_handle_identity_drift",
        )
        _windows_rename_directory_handle(
            source_fd,
            destination,
            code=code,
        )
        return _assert_owned_directory_fd(
            source_fd,
            owned_identity,
            path=destination,
            code=f"{code}_destination_handle_identity_drift",
        )
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(source_fd)


def _load_schema(path: Path) -> Draft202012Validator:
    raw = _strict_json(path.read_bytes(), "runtime_schema_json_invalid")
    try:
        Draft202012Validator.check_schema(raw)
    except SchemaError:
        raise MultiArmRuntimeError("runtime_schema_invalid") from None
    return Draft202012Validator(raw)


def _write_canonical(
    path: Path,
    value: object,
    *,
    schema_path: Path | None = None,
    parent_identity: DirectoryIdentity | None = None,
) -> tuple[str, str]:
    if schema_path is not None:
        try:
            _load_schema(_project_root() / schema_path).validate(value)
        except ValidationError as exc:
            raise MultiArmRuntimeError(
                f"runtime_receipt_schema_mismatch:{exc.validator}"
            ) from None
    raw = _canonical_json(value)
    digest = _sha256(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_parent = parent_identity or _directory_identity(
        path.parent,
        code="runtime_receipt_parent_identity_invalid",
    )
    parent_fd = _windows_open_directory_fd(
        path.parent,
        delete_access=True,
        code="runtime_receipt_parent_handle_open_failed",
    )
    try:
        _assert_owned_directory_fd(
            parent_fd,
            expected_parent,
            path=path.parent,
            code="runtime_receipt_parent_handle_identity_drift",
        )
        try:
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            sidecar = path.with_name(path.name + ".sha256")
            with sidecar.open("xb") as handle:
                handle.write(f"{digest}  {path.name}\n".encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            raise MultiArmRuntimeError("runtime_atomic_receipt_write_failed") from None
        _assert_owned_directory_fd(
            parent_fd,
            expected_parent,
            path=path.parent,
            code="runtime_receipt_parent_changed_after_write",
        )
        return digest, _sha256(path.with_name(path.name + ".sha256").read_bytes())
    finally:
        os.close(parent_fd)


def _runtime_file_identity(path: Path) -> dict[str, object]:
    full = _project_root() / path
    try:
        first = full.lstat()
        if stat.S_ISLNK(first.st_mode) or _is_reparse(first):
            raise MultiArmRuntimeError("runtime_implementation_file_reparse")
        raw = full.read_bytes()
        second = full.lstat()
        if (
            first.st_dev,
            first.st_ino,
            first.st_size,
            first.st_mtime_ns,
            first.st_ctime_ns,
        ) != (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
            second.st_ctime_ns,
        ):
            raise MultiArmRuntimeError("runtime_implementation_file_changed")
    except MultiArmRuntimeError:
        raise
    except OSError:
        raise MultiArmRuntimeError("runtime_implementation_file_missing") from None
    return {"path": path.as_posix(), "sha256": _sha256(raw), "bytes": len(raw)}


def load_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], str]:
    requested = Path(path)
    if not requested.is_absolute():
        requested = _project_root() / requested
    try:
        raw = requested.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise MultiArmRuntimeError("runtime_config_unreadable") from None
    if not isinstance(value, Mapping):
        raise MultiArmRuntimeError("runtime_config_invalid")
    try:
        expanded = _expand_env(dict(value))
    except ConfigError as exc:
        raise MultiArmRuntimeError(f"runtime_config_env_invalid:{exc}") from None
    try:
        _load_schema(_project_root() / CONFIG_SCHEMA_PATH).validate(expanded)
    except ValidationError as exc:
        raise MultiArmRuntimeError(
            f"runtime_config_schema_mismatch:{exc.validator}"
        ) from None
    config = dict(expanded)
    config["_config_path"] = str(requested.resolve())
    config["_config_sha256"] = _sha256(raw)
    validate_config(config)
    return config, str(config["_config_sha256"])


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("status") != "gpu_runtime_implemented_explicit_execution_only"
        or config.get("claim_scope") != "diagnostic_proxy_training_runtime"
    ):
        raise MultiArmRuntimeError("runtime_config_identity_drift")
    model_free = _mapping(
        config.get("model_free_contract"), "runtime_model_free_invalid"
    )
    loaded, observed_sha, _ = contract.load_config(
        _project_root() / str(model_free["path"])
    )
    if (
        observed_sha != model_free.get("sha256")
        or tuple(loaded["arms"]["ordered"]) != contract.TRAINED_ARMS
    ):
        raise MultiArmRuntimeError("runtime_model_free_identity_drift")
    source = _mapping(config.get("source"), "runtime_source_invalid")
    if (
        source.get("candidate_commit") != source_runtime.PRODUCER_CANDIDATE_COMMIT
        or source.get("release_commit") != source_runtime.PRODUCER_RELEASE_COMMIT
        or source.get("release_tree") != source_runtime.PRODUCER_RELEASE_TREE
        or source.get("artifact_prefix") != source_runtime.PRODUCER_ARTIFACT_PREFIX
        or source.get("exact_read_set_files") != 88
        or source.get("exact_final_files") != 46
        or source.get("historical_release_attested") is not True
        or source.get("current_live_remote_reasserted") is not False
        or source.get("immutable_git_objects_required") is not True
        or source.get("replace_refs_and_grafts_forbidden") is not True
    ):
        raise MultiArmRuntimeError("runtime_source_contract_drift")
    rollover = _mapping(
        config.get("teacher_rollover"),
        "runtime_teacher_rollover_invalid",
    )
    if (
        rollover.get("status") != "additive_v3_physical_handoff_required"
        or rollover.get("producer_repository_env")
        != "ANCHOR_GEMMA3_UNBALANCED_V2_PRODUCER_REPOSITORY"
        or rollover.get("producer_repository_default")
        != "D:/LLM/anchor-moe-lora-gemma3-chat-rollover-v3"
        or rollover.get("producer_branch") != ROLLOVER_PRODUCER_BRANCH
        or rollover.get("producer_commit") != ROLLOVER_PRODUCER_COMMIT
        or rollover.get("producer_parent") != ROLLOVER_PRODUCER_PARENT
        or rollover.get("producer_tree") != ROLLOVER_PRODUCER_TREE
        or rollover.get("binding_filename") != ROLLOVER_BINDING_FILENAME
        or dict(
            _mapping(
                rollover.get("physical_identity"),
                "runtime_teacher_rollover_physical_identity_invalid",
            )
        )
        != ROLLOVER_PHYSICAL_IDENTITY_SPECS
        or dict(
            _mapping(
                rollover.get("schemas"),
                "runtime_teacher_rollover_schemas_invalid",
            )
        )
        != ROLLOVER_SCHEMA_SPECS
        or dict(
            _mapping(
                rollover.get("release_implementation"),
                "runtime_teacher_rollover_implementation_invalid",
            )
        )
        != ROLLOVER_RELEASE_IMPLEMENTATION
        or rollover.get("legacy_v1_consumable") is not False
        or rollover.get("legacy_v2_consumable") is not False
        or rollover.get("legacy_f23_consumable") is not False
        or rollover.get("legacy_implementation_binding_consumable") is not False
        or rollover.get("fallback_allowed") is not False
    ):
        raise MultiArmRuntimeError("runtime_teacher_rollover_contract_drift")
    teacher = _mapping(config.get("teacher_final"), "runtime_teacher_invalid")
    binding_schema_release_status = teacher.get("binding_schema_release_status")
    record_schema_release_status = teacher.get("record_schema_release_status")
    binding_schema_sha = teacher.get("binding_schema_sha256")
    record_schema_sha = teacher.get("record_schema_sha256")
    if (
        teacher.get("required_for_gpu_execution") is not True
        or teacher.get("binding_schema_version") != TEACHER_BINDING_VERSION
        or teacher.get("record_schema_version") != TEACHER_RECORD_VERSION
        or teacher.get("binding_schema") != TEACHER_BINDING_SCHEMA_PATH.as_posix()
        or teacher.get("record_schema") != TEACHER_RECORD_SCHEMA_PATH.as_posix()
        or binding_schema_release_status
        != "authenticated_additive_rollover_v3_git_and_local_bytes_bound"
        or record_schema_release_status
        != "authenticated_additive_rollover_v3_git_and_local_bytes_bound"
        or binding_schema_sha != ROLLOVER_SCHEMA_SPECS["binding"]["sha256"]
        or record_schema_sha != ROLLOVER_SCHEMA_SPECS["record"]["sha256"]
        or teacher.get("required_binding_status")
        != "released_pending_consumer_acceptance"
        or teacher.get("required_binding_filename") != ROLLOVER_BINDING_FILENAME
        or teacher.get("external_authenticated_producer_root_required") is not True
        or teacher.get("consumer_prerequisite_commit")
        != "6240ae111182104f22f08e1a569deae866c6e210"
        or teacher.get("expected_records") != 3520
        or teacher.get("expected_main_records") != 3440
        or teacher.get("expected_router_records") != 80
        or teacher.get("expected_train_identity_bundles") != 20
        or teacher.get("expected_train_identity_records") != 100
        or teacher.get("independent_identity_eval_bundles_not_in_optimizer") != 10
        or teacher.get("independent_identity_eval_probes_not_in_optimizer") != 50
        or teacher.get("allowed_decisions") != ["accepted"]
        or teacher.get("rejected_or_uncertain_records") != 0
        or teacher.get("original_producer_target_fallback") is not False
        or teacher.get("join_fields")
        != [
            "record_id_sha256",
            "source_content_sha256",
            "source_serialization_identity_sha256",
            "task_bundle_sha256",
            "source_asset",
        ]
    ):
        raise MultiArmRuntimeError("runtime_teacher_contract_drift")
    if tuple(config["arms"]["ordered"]) != contract.TRAINED_ARMS:
        raise MultiArmRuntimeError("runtime_arm_order_drift")
    definitions = _mapping(
        config["arms"]["definitions"], "runtime_arm_definitions_invalid"
    )
    if set(definitions) != set(contract.TRAINED_ARMS):
        raise MultiArmRuntimeError("runtime_arm_set_drift")
    for arm in contract.TRAINED_ARMS:
        if dict(definitions[arm]) != {
            "source_asset": contract.ARM_SOURCE_ASSET[arm],
            "adapter_profile": contract.ARM_PROFILE[arm],
            "full_steps": contract.FULL_STEPS[arm],
        }:
            raise MultiArmRuntimeError("runtime_arm_definition_drift")
    training = _mapping(config.get("training"), "runtime_training_invalid")
    if (
        training.get("smoke_steps_per_arm") != 2
        or training.get("sequence_length") != 768
        or training.get("truncation") is not False
        or training.get("batch_size") != 1
        or training.get("gradient_accumulation_steps") != 1
        or training.get("strictly_serial") is not True
        or training.get("concurrency") != 1
        or training.get("resume") is not False
        or training.get("smoke_checkpoint_consumed_by_full") is not False
        or training.get("fresh_base_per_phase") is not True
        or training.get("fresh_adapter_per_phase") is not True
        or training.get("use_cache") is not False
        or training.get("autocast_dtype") != "bfloat16"
        or training.get("tf32") is not True
    ):
        raise MultiArmRuntimeError("runtime_training_contract_drift")
    adapter = _mapping(config.get("adapter"), "runtime_adapter_invalid")
    if (
        adapter.get("dtype") != "bfloat16"
        or adapter.get("q")
        != {
            "target_module": "q_proj",
            "rank": 1024,
            "alpha": 2048,
        }
        or adapter.get("o")
        != {
            "target_module": "o_proj",
            "rank": 64,
            "alpha": 128,
        }
        or adapter.get("base_o_proj_always_frozen") is not True
    ):
        raise MultiArmRuntimeError("runtime_adapter_contract_drift")
    optimizer = _mapping(config.get("optimizer"), "runtime_optimizer_invalid")
    q_lr = _decimal(optimizer.get("q_learning_rate"), "runtime_q_learning_rate_invalid")
    o_lr = _decimal(optimizer.get("o_learning_rate"), "runtime_o_learning_rate_invalid")
    if (
        optimizer.get("backend") != "bitsandbytes.optim.AdamW8bit"
        or optimizer.get("bitsandbytes_version") != "0.48.2"
        or q_lr != contract.Q_LR
        or o_lr != contract.O_LR
        or o_lr * 10 != q_lr
        or optimizer.get("explicit_parameter_groups") is not True
        or optimizer.get("group_order") != ["Q", "O"]
        or optimizer.get("base_o_proj_always_frozen") is not True
        or optimizer.get("uint8_cuda_state_required") is not True
    ):
        raise MultiArmRuntimeError("runtime_optimizer_contract_drift")
    gpu = _mapping(config.get("gpu_policy"), "runtime_gpu_policy_invalid")
    if (
        gpu.get("canonical_lock") != "runs/formal-v3-training.lock"
        or gpu.get("expected_total_memory_mib") != 12288
        or gpu.get("torch_peak_allocated_max_mib") != 23962
        or gpu.get("torch_peak_reserved_max_mib") != 23962
        or gpu.get("external_compute_process_policy") != "observe_only"
        or gpu.get("automatic_rank_or_batch_reduction") is not False
        or gpu.get("automatic_retry") is not False
        or gpu.get("oom_fail_closed") is not True
    ):
        raise MultiArmRuntimeError("runtime_gpu_policy_drift")
    if dict(config["claims"]) != {
        "diagnostic_only": True,
        "proxy_only": True,
        "formal": False,
        "release": False,
        "default_model_free": True,
        "gpu_execution_requires_explicit_flag": True,
        "physical_kv_validated": False,
        "full_generation_kv_shared": False,
    }:
        raise MultiArmRuntimeError("runtime_claims_drift")


def _released_teacher_schema_identities(
    config: Mapping[str, Any],
    producer_repository: str | Path | None = None,
) -> tuple[AuthenticatedRolloverProducer, str, str]:
    """Return physically bound v2 schema identities or stop before execution."""

    teacher = _mapping(config.get("teacher_final"), "runtime_teacher_invalid")
    if (
        teacher.get("binding_schema_release_status")
        != "authenticated_additive_rollover_v3_git_and_local_bytes_bound"
        or teacher.get("record_schema_release_status")
        != "authenticated_additive_rollover_v3_git_and_local_bytes_bound"
    ):
        raise MultiArmRuntimeError("teacher_v3_physical_handoff_invalid")
    binding_sha = _require_sha(
        teacher.get("binding_schema_sha256"),
        "teacher_binding_schema_sha_invalid",
    )
    record_sha = _require_sha(
        teacher.get("record_schema_sha256"),
        "teacher_record_schema_sha_invalid",
    )
    _stable_file(
        _project_root() / TEACHER_BINDING_SCHEMA_PATH,
        expected_sha256=binding_sha,
        code="teacher_binding_schema_local_identity_drift",
    )
    _stable_file(
        _project_root() / TEACHER_RECORD_SCHEMA_PATH,
        expected_sha256=record_sha,
        code="teacher_record_schema_local_identity_drift",
    )
    producer = authenticate_rollover_producer(config, producer_repository)
    if (
        producer.schema_sha256["binding"] != binding_sha
        or producer.schema_sha256["record"] != record_sha
    ):
        raise MultiArmRuntimeError("teacher_rollover_schema_identity_drift")
    return producer, binding_sha, record_sha


def _teacher_file_pin(
    artifact_root: Path,
    value: object,
    *,
    code: str,
) -> TeacherFilePin:
    item = _mapping(value, code)
    relative = str(item.get("path", ""))
    posix = PurePosixPath(relative)
    if not relative or posix.is_absolute() or ".." in posix.parts:
        raise MultiArmRuntimeError(f"{code}_path_invalid")
    path = artifact_root.joinpath(*posix.parts)
    expected_sha = _require_sha(item.get("sha256"), f"{code}_sha_invalid")
    expected_bytes = int(item.get("bytes", 0))
    raw = _stable_file(
        path,
        expected_sha256=expected_sha,
        expected_bytes=expected_bytes,
        code=f"{code}_identity_drift",
    )
    sidecar_path = path.with_name(path.name + ".sha256")
    sidecar_sha = _require_sha(
        item.get("sidecar_sha256"), f"{code}_sidecar_sha_invalid"
    )
    sidecar_bytes = int(item.get("sidecar_bytes", 0))
    sidecar_raw = _stable_file(
        sidecar_path,
        expected_sha256=sidecar_sha,
        expected_bytes=sidecar_bytes,
        code=f"{code}_sidecar_identity_drift",
    )
    if sidecar_raw != f"{expected_sha}  {path.name}\n".encode("ascii"):
        raise MultiArmRuntimeError(f"{code}_sidecar_content_invalid")
    if path.suffix in {".json", ".jsonl"} and (
        b"\r" in raw or (raw and not raw.endswith(b"\n"))
    ):
        raise MultiArmRuntimeError(f"{code}_not_utf8_lf")
    return TeacherFilePin(
        path=path,
        sha256=expected_sha,
        bytes=expected_bytes,
        sidecar_sha256=sidecar_sha,
        sidecar_bytes=sidecar_bytes,
    )


def _producer_canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _producer_canonical_sha256(value: object) -> str:
    return _sha256(_producer_canonical_bytes(value))


def _producer_domain_sha256(domain: str, value: object) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _producer_canonical_bytes(value))


def _relative_under(root: Path, value: object, *, code: str) -> Path:
    relative = str(value)
    posix = PurePosixPath(relative)
    if (
        not relative
        or relative != relative.strip()
        or posix.is_absolute()
        or ".." in posix.parts
    ):
        raise MultiArmRuntimeError(code)
    path = Path(os.path.abspath(root.joinpath(*posix.parts)))
    try:
        path.relative_to(root)
    except ValueError:
        raise MultiArmRuntimeError(code) from None
    return path


def _metadata_file_pin(
    root: Path,
    relative: str,
    *,
    sha256: object,
    sidecar_sha256: object,
    code: str,
) -> tuple[TeacherFilePin, bytes]:
    path = _relative_under(root, relative, code=f"{code}_path_invalid")
    expected_sha = _require_sha(sha256, f"{code}_sha_invalid")
    expected_sidecar_sha = _require_sha(
        sidecar_sha256,
        f"{code}_sidecar_sha_invalid",
    )
    raw = _stable_file(
        path,
        expected_sha256=expected_sha,
        code=f"{code}_identity_drift",
    )
    if not raw or len(raw) >= 50 * 1024 * 1024:
        raise MultiArmRuntimeError(f"{code}_size_invalid")
    sidecar_path = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{expected_sha}  {path.name}\n".encode("ascii")
    sidecar = _stable_file(
        sidecar_path,
        expected_sha256=expected_sidecar_sha,
        expected_bytes=len(expected_sidecar),
        code=f"{code}_sidecar_identity_drift",
    )
    if sidecar != expected_sidecar:
        raise MultiArmRuntimeError(f"{code}_sidecar_content_invalid")
    return (
        TeacherFilePin(
            path=path,
            sha256=expected_sha,
            bytes=len(raw),
            sidecar_sha256=expected_sidecar_sha,
            sidecar_bytes=len(sidecar),
        ),
        raw,
    )


def _canonical_metadata_document(raw: bytes, *, code: str) -> Mapping[str, Any]:
    value = _strict_json(raw, code)
    if _canonical_json(value) != raw:
        raise MultiArmRuntimeError(f"{code}_not_canonical")
    return value


def _validate_rollover_document(
    producer: AuthenticatedRolloverProducer,
    schema_name: str,
    value: Mapping[str, Any],
    *,
    code: str,
) -> None:
    try:
        Draft202012Validator(producer.schema_values[schema_name]).validate(value)
    except ValidationError as exc:
        raise MultiArmRuntimeError(f"{code}:{exc.validator}") from None


def _exact_tree_paths(
    root: Path,
    expected: set[str],
    *,
    code: str,
) -> None:
    observed: set[str] = set()
    try:
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            _directory_identity(current_path, code=code)
            for directory in directories:
                _directory_identity(current_path / directory, code=code)
            for name in files:
                path = current_path / name
                value = path.lstat()
                if (
                    not stat.S_ISREG(value.st_mode)
                    or stat.S_ISLNK(value.st_mode)
                    or _is_reparse(value)
                ):
                    raise MultiArmRuntimeError(code)
                relative = path.relative_to(root).as_posix()
                if relative in observed:
                    raise MultiArmRuntimeError(code)
                observed.add(relative)
    except MultiArmRuntimeError:
        raise
    except OSError:
        raise MultiArmRuntimeError(code) from None
    if observed != expected:
        raise MultiArmRuntimeError(code)


def _candidate_physical_tree_sha256(
    root: Path,
    pins: Sequence[TeacherFilePin],
) -> str:
    entries = [
        {
            "path": pin.path.relative_to(root).as_posix(),
            "sha256": pin.sha256,
            "bytes": pin.bytes,
        }
        for pin in pins
    ]
    entries.extend(
        {
            "path": pin.path.with_name(pin.path.name + ".sha256")
            .relative_to(root)
            .as_posix(),
            "sha256": pin.sidecar_sha256,
            "bytes": pin.sidecar_bytes,
        }
        for pin in pins
    )
    return _producer_domain_sha256(
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-physical-tree.v2",
        sorted(entries, key=lambda item: str(item["path"])),
    )


def authenticate_teacher_final(
    binding_path: str | Path,
    *,
    expected_binding_sha256: str,
    producer: AuthenticatedRolloverProducer,
    expected_record_schema_sha256: str,
) -> tuple[AuthenticatedTeacherFinal, Mapping[str, Any]]:
    """Authenticate the unique additive rollover handoff before body parsing."""

    expected_sha = _require_sha(
        expected_binding_sha256, "teacher_binding_expected_sha_invalid"
    )
    binding_schema_sha = producer.schema_sha256["binding"]
    record_schema_sha = _require_sha(
        expected_record_schema_sha256,
        "teacher_record_schema_expected_sha_invalid",
    )
    if record_schema_sha != producer.schema_sha256["record"]:
        raise MultiArmRuntimeError("teacher_record_schema_identity_drift")
    requested = Path(binding_path)
    if not requested.is_absolute():
        requested = producer.root / requested
    requested = Path(os.path.abspath(requested))
    try:
        requested.relative_to(producer.root)
    except ValueError:
        raise MultiArmRuntimeError("teacher_binding_path_escaped") from None
    binding_chain = _directory_chain(
        producer.root,
        [requested],
        code="teacher_binding_parent_chain_invalid",
    )
    raw = _stable_file(
        requested,
        expected_sha256=expected_sha,
        code="teacher_binding_identity_drift",
    )
    expected_binding_sidecar = f"{expected_sha}  {requested.name}\n".encode("ascii")
    binding_sidecar_sha = _sha256(expected_binding_sidecar)
    sidecar_raw = _stable_file(
        requested.with_name(requested.name + ".sha256"),
        expected_sha256=binding_sidecar_sha,
        expected_bytes=len(expected_binding_sidecar),
        code="teacher_binding_sidecar_identity_drift",
    )
    if sidecar_raw != expected_binding_sidecar:
        raise MultiArmRuntimeError("teacher_binding_sidecar_content_invalid")
    binding = _canonical_metadata_document(raw, code="teacher_binding_json_invalid")
    _validate_rollover_document(
        producer,
        "binding",
        binding,
        code="teacher_binding_schema_mismatch",
    )
    if (
        binding.get("schema_version") != TEACHER_BINDING_VERSION
        or binding.get("status") != "released_pending_consumer_acceptance"
        or binding.get("namespace") != "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
    ):
        raise MultiArmRuntimeError("teacher_binding_contract_drift")

    candidate = _mapping(binding["candidate"], "teacher_candidate_invalid")
    release = _mapping(binding["release"], "teacher_release_invalid")
    counts = _mapping(binding["counts"], "teacher_counts_invalid")
    logical_identity = _mapping(
        binding["logical_identity"],
        "teacher_logical_identity_invalid",
    )
    ordered_shards = _sequence(
        binding["ordered_shards"],
        "teacher_shards_invalid",
    )
    if (
        release.get("binding_schema_sha256") != binding_schema_sha
        or release.get("final_manifest_schema_sha256")
        != producer.schema_sha256["release_manifest"]
        or release.get("attestation_schema_sha256")
        != producer.schema_sha256["attestation"]
        or release.get("release_implementation_sha256")
        != producer.release_implementation_sha256
        or _producer_canonical_sha256(ordered_shards)
        != logical_identity.get("shard_inventory_sha256")
    ):
        raise MultiArmRuntimeError("teacher_release_identity_drift")

    release_root = _relative_under(
        producer.root,
        release["release_root"],
        code="teacher_release_root_invalid",
    )
    if requested != release_root / ROLLOVER_BINDING_FILENAME:
        raise MultiArmRuntimeError("teacher_binding_not_unique_rollover_handoff")
    if release_root.name != f"release-{candidate['physical_tree_sha256']}":
        raise MultiArmRuntimeError("teacher_release_candidate_tree_identity_drift")
    artifact_root = _relative_under(
        producer.root,
        candidate["artifact_root"],
        code="teacher_artifact_root_invalid",
    )
    if not artifact_root.is_dir():
        raise MultiArmRuntimeError("teacher_artifact_root_missing")
    if not release_root.is_dir():
        raise MultiArmRuntimeError("teacher_release_root_missing")
    artifact_chain = _directory_chain(
        producer.root,
        [artifact_root, release_root],
        code="teacher_artifact_parent_chain_invalid",
    )

    manifest, manifest_raw = _metadata_file_pin(
        release_root,
        str(release["final_manifest_path"]),
        sha256=release["final_manifest_sha256"],
        sidecar_sha256=release["final_manifest_sidecar_sha256"],
        code="teacher_final_manifest",
    )
    final_manifest = _canonical_metadata_document(
        manifest_raw,
        code="teacher_final_manifest_json_invalid",
    )
    _validate_rollover_document(
        producer,
        "release_manifest",
        final_manifest,
        code="teacher_final_manifest_schema_mismatch",
    )

    release_attestation, attestation_raw = _metadata_file_pin(
        release_root,
        str(release["attestation_path"]),
        sha256=release["attestation_sha256"],
        sidecar_sha256=release["attestation_sidecar_sha256"],
        code="teacher_release_attestation",
    )
    attestation = _canonical_metadata_document(
        attestation_raw,
        code="teacher_release_attestation_json_invalid",
    )
    _validate_rollover_document(
        producer,
        "attestation",
        attestation,
        code="teacher_release_attestation_schema_mismatch",
    )

    attested_candidate = _mapping(
        attestation["candidate_identity"],
        "teacher_attested_candidate_invalid",
    )
    for key in (
        "artifact_root",
        "physical_tree_sha256",
        "manifest_sha256",
        "build_receipt_sha256",
        "output_inventory_sha256",
        "payload_tree_sha256",
    ):
        if attested_candidate.get(key) != candidate.get(key):
            raise MultiArmRuntimeError("teacher_candidate_attestation_drift")

    candidate_manifest, candidate_manifest_raw = _metadata_file_pin(
        artifact_root,
        "manifest.json",
        sha256=candidate["manifest_sha256"],
        sidecar_sha256=attested_candidate["manifest_sidecar_sha256"],
        code="teacher_candidate_manifest",
    )
    candidate_manifest_value = _canonical_metadata_document(
        candidate_manifest_raw,
        code="teacher_candidate_manifest_json_invalid",
    )
    _validate_rollover_document(
        producer,
        "candidate_manifest",
        candidate_manifest_value,
        code="teacher_candidate_manifest_schema_mismatch",
    )

    release_receipt, build_receipt_raw = _metadata_file_pin(
        artifact_root,
        "build_receipt.json",
        sha256=candidate["build_receipt_sha256"],
        sidecar_sha256=attested_candidate["build_receipt_sidecar_sha256"],
        code="teacher_release_receipt",
    )
    build_receipt = _canonical_metadata_document(
        build_receipt_raw,
        code="teacher_release_receipt_json_invalid",
    )
    _validate_rollover_document(
        producer,
        "build_receipt",
        build_receipt,
        code="teacher_release_receipt_schema_mismatch",
    )

    output_inventory, output_inventory_raw = _metadata_file_pin(
        artifact_root,
        "output_inventory.json",
        sha256=candidate["output_inventory_sha256"],
        sidecar_sha256=attested_candidate["output_inventory_sidecar_sha256"],
        code="teacher_output_inventory",
    )
    output_inventory_value = _canonical_metadata_document(
        output_inventory_raw,
        code="teacher_output_inventory_json_invalid",
    )
    release_request, release_request_raw = _metadata_file_pin(
        artifact_root,
        "release_attestation.request.json",
        sha256=attested_candidate["release_request_sha256"],
        sidecar_sha256=attested_candidate["release_request_sidecar_sha256"],
        code="teacher_release_request",
    )
    _canonical_metadata_document(
        release_request_raw,
        code="teacher_release_request_json_invalid",
    )

    shard_pins: list[TeacherFilePin] = []
    shard_assets: list[str] = []
    shard_records: list[int] = []
    paths: set[Path] = set()
    for raw_shard in ordered_shards:
        shard = _mapping(raw_shard, "teacher_shard_invalid")
        pin = _teacher_file_pin(artifact_root, shard, code="teacher_shard")
        asset = str(shard["asset"])
        records = int(shard["records"])
        if pin.path in paths or asset not in {
            "main_train",
            "planner_router_train",
        }:
            raise MultiArmRuntimeError("teacher_shard_inventory_invalid")
        paths.add(pin.path)
        shard_pins.append(pin)
        shard_assets.append(asset)
        shard_records.append(records)

    if (
        counts
        != {
            "records": 3520,
            "main_records": 3440,
            "router_records": 80,
            "source_assets": EXPECTED_ASSET_COUNTS,
            "train_identity_bundles": 20,
            "train_identity_records": 100,
        }
        or sum(shard_records) != 3520
        or sum(
            count
            for asset, count in zip(shard_assets, shard_records, strict=True)
            if asset == "main_train"
        )
        != 3440
        or sum(
            count
            for asset, count in zip(shard_assets, shard_records, strict=True)
            if asset == "planner_router_train"
        )
        != 80
    ):
        raise MultiArmRuntimeError("teacher_count_contract_drift")

    expected_candidate_consumer = {
        **dict(binding["consumer_prerequisite"]),
        "commit": binding["consumer_prerequisite"]["consumer_commit"],
        "teacher_binding_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding."
            "consumer-rollover-v2"
        ),
    }
    expected_candidate_consumer.pop("consumer_commit")
    candidate_record_schema = _mapping(
        candidate_manifest_value["record_schema"],
        "teacher_candidate_record_schema_invalid",
    )
    expected_manifest_counts = {
        "records": 3520,
        "main_records": 3440,
        "router_records": 80,
        "accepted": 3520,
        "rejected": 0,
        "uncertain": 0,
        "train_identity_bundles": 20,
        "train_identity_records": 100,
        "source_assets": EXPECTED_ASSET_COUNTS,
    }
    if (
        candidate_manifest_value["source"] != binding["source"]
        or candidate_manifest_value["consumer"] != expected_candidate_consumer
        or candidate_manifest_value["counts"] != expected_manifest_counts
        or candidate_manifest_value["logical_identity"] != logical_identity
        or candidate_manifest_value["shards"] != list(ordered_shards)
        or candidate_record_schema
        != {
            "path": ROLLOVER_SCHEMA_SPECS["record"]["path"],
            "sha256": record_schema_sha,
            "bytes": ROLLOVER_SCHEMA_SPECS["record"]["bytes"],
        }
    ):
        raise MultiArmRuntimeError("teacher_candidate_manifest_binding_drift")

    final_binding_contract = _mapping(
        final_manifest["binding_contract"],
        "teacher_final_binding_contract_invalid",
    )
    if (
        final_manifest["candidate"] != candidate
        or final_manifest["source"] != binding["source"]
        or final_manifest["counts"] != expected_manifest_counts
        or final_manifest["logical_identity"] != logical_identity
        or final_manifest["shards"] != list(ordered_shards)
        or final_binding_contract
        != {
            "path": ROLLOVER_BINDING_FILENAME,
            "schema_version": TEACHER_BINDING_VERSION,
            "schema_path": ROLLOVER_SCHEMA_SPECS["binding"]["path"],
            "schema_sha256": binding_schema_sha,
            "status": "released_pending_consumer_acceptance",
        }
        or final_manifest["attestation"]["sha256"] != release_attestation.sha256
        or final_manifest["attestation"]["sidecar_sha256"]
        != release_attestation.sidecar_sha256
        or final_manifest["record_schema"]["sha256"] != record_schema_sha
        or final_manifest["record_schema"]["bytes"]
        != ROLLOVER_SCHEMA_SPECS["record"]["bytes"]
    ):
        raise MultiArmRuntimeError("teacher_final_manifest_binding_drift")

    attested_source = dict(
        _mapping(attestation["source_release"], "teacher_attested_source_invalid")
    )
    attested_source.pop("git_parent_and_tree_recomputed", None)
    attested_source.pop("physical_tree_digest_recomputed", None)
    implementation_binding = _mapping(
        attestation["implementation_binding"],
        "teacher_attested_implementation_invalid",
    )
    record_audit = _mapping(
        attestation["record_audit"],
        "teacher_attested_record_audit_invalid",
    )
    if (
        attested_source != binding["source"]
        or implementation_binding["release_implementation_sha256"]
        != producer.release_implementation_sha256
        or implementation_binding["record_schema_sha256"] != record_schema_sha
        or implementation_binding["candidate_manifest_schema_sha256"]
        != producer.schema_sha256["candidate_manifest"]
        or implementation_binding["candidate_build_receipt_schema_sha256"]
        != producer.schema_sha256["build_receipt"]
        or record_audit["ordered_shards"] != list(ordered_shards)
        or record_audit["ordered_record_sha256"]
        != logical_identity["record_order_sha256"]
        or record_audit["target_projection_sha256"]
        != logical_identity["target_projection_sha256"]
        or record_audit["source_join_sha256"] != logical_identity["source_join_sha256"]
        or record_audit["shard_inventory_sha256"]
        != logical_identity["shard_inventory_sha256"]
        or record_audit["logical_dataset_sha256"]
        != logical_identity["logical_dataset_sha256"]
    ):
        raise MultiArmRuntimeError("teacher_release_attestation_binding_drift")

    expected_receipt_counts = dict(expected_manifest_counts)
    expected_receipt_counts.pop("source_assets")
    if (
        build_receipt["manifest_sha256"] != candidate_manifest.sha256
        or build_receipt["manifest_bytes"] != candidate_manifest.bytes
        or build_receipt["output_inventory_sha256"] != output_inventory.sha256
        or build_receipt["payload_tree_sha256"] != candidate["payload_tree_sha256"]
        or build_receipt["counts"] != expected_receipt_counts
        or attestation["runtime_receipt"]["runtime_hmac_sha256"]
        != build_receipt["runtime_hmac_sha256"]
        or attestation["runtime_receipt"]["phase_receipt_inventory_sha256"]
        != build_receipt["phase_receipt_inventory_sha256"]
        or attestation["runtime_receipt"]["wal_chain_tip_sha256"]
        != build_receipt["wal_chain_tip_sha256"]
    ):
        raise MultiArmRuntimeError("teacher_release_receipt_binding_drift")

    expected_payload_inventory: list[dict[str, object]] = []
    for pin in shard_pins:
        expected_payload_inventory.extend(
            [
                {
                    "path": pin.path.relative_to(artifact_root).as_posix(),
                    "kind": "teacher_shard",
                    "sha256": pin.sha256,
                    "bytes": pin.bytes,
                },
                {
                    "path": pin.path.with_name(pin.path.name + ".sha256")
                    .relative_to(artifact_root)
                    .as_posix(),
                    "kind": "sha256_sidecar",
                    "sha256": pin.sidecar_sha256,
                    "bytes": pin.sidecar_bytes,
                },
            ]
        )
    expected_payload_inventory.extend(
        [
            {
                "path": "manifest.json",
                "kind": "manifest",
                "sha256": candidate_manifest.sha256,
                "bytes": candidate_manifest.bytes,
            },
            {
                "path": "manifest.json.sha256",
                "kind": "sha256_sidecar",
                "sha256": candidate_manifest.sidecar_sha256,
                "bytes": candidate_manifest.sidecar_bytes,
            },
        ]
    )
    if (
        output_inventory_value.get("schema_version")
        != "anchor.gemma3-chat-unbalanced-v2-teacher-final-output-inventory.v2"
        or output_inventory_value.get("namespace")
        != "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
        or output_inventory_value.get("files") != expected_payload_inventory
        or output_inventory_value.get("payload_tree_sha256")
        != _producer_canonical_sha256(expected_payload_inventory)
        or output_inventory_value.get("payload_tree_sha256")
        != candidate["payload_tree_sha256"]
    ):
        raise MultiArmRuntimeError("teacher_output_inventory_binding_drift")

    candidate_pins = (
        candidate_manifest,
        release_receipt,
        output_inventory,
        release_request,
        *shard_pins,
    )
    expected_candidate_paths = {
        item
        for pin in candidate_pins
        for item in (
            pin.path.relative_to(artifact_root).as_posix(),
            pin.path.with_name(pin.path.name + ".sha256")
            .relative_to(artifact_root)
            .as_posix(),
        )
    }
    _exact_tree_paths(
        artifact_root,
        expected_candidate_paths,
        code="teacher_candidate_tree_inventory_drift",
    )
    if (
        len(expected_candidate_paths) != attested_candidate["physical_file_count"]
        or _candidate_physical_tree_sha256(artifact_root, candidate_pins)
        != candidate["physical_tree_sha256"]
    ):
        raise MultiArmRuntimeError("teacher_candidate_physical_tree_drift")

    _exact_tree_paths(
        release_root,
        {
            "independent_release_attestation.json",
            "independent_release_attestation.json.sha256",
            "final_manifest.json",
            "final_manifest.json.sha256",
            ROLLOVER_BINDING_FILENAME,
            f"{ROLLOVER_BINDING_FILENAME}.sha256",
        },
        code="teacher_release_tree_inventory_drift",
    )

    shard_inventory_sha = str(logical_identity["shard_inventory_sha256"])
    identity_preimage = {
        "schema_version": TEACHER_SOURCE_VERSION,
        "binding_schema_version": TEACHER_BINDING_VERSION,
        "record_schema_version": TEACHER_RECORD_VERSION,
        "binding_sha256": expected_sha,
        "binding_schema_sha256": binding_schema_sha,
        "manifest_sha256": manifest.sha256,
        "record_schema_sha256": record_schema_sha,
        "release_receipt_sha256": release_receipt.sha256,
        "release_attestation_sha256": release_attestation.sha256,
        "shard_inventory_sha256": shard_inventory_sha,
        "records": 3520,
        "main_records": 3440,
        "router_records": 80,
        "train_identity_bundles": 20,
        "train_identity_records": 100,
        "independent_identity_eval_bundles_not_in_optimizer": 10,
        "independent_identity_eval_probes_not_in_optimizer": 50,
        "accepted": 3520,
        "rejected": 0,
        "uncertain": 0,
        "producer_original_target_fallback": False,
    }
    pinned_files = [
        requested,
        requested.with_name(requested.name + ".sha256"),
        manifest.path,
        manifest.path.with_name(manifest.path.name + ".sha256"),
        release_receipt.path,
        release_receipt.path.with_name(release_receipt.path.name + ".sha256"),
        release_attestation.path,
        release_attestation.path.with_name(release_attestation.path.name + ".sha256"),
        *(
            value
            for pin in (candidate_manifest, output_inventory, release_request)
            for value in (
                pin.path,
                pin.path.with_name(pin.path.name + ".sha256"),
            )
        ),
        *(
            value
            for pin in shard_pins
            for value in (
                pin.path,
                pin.path.with_name(pin.path.name + ".sha256"),
            )
        ),
    ]
    full_chain = _directory_chain(
        producer.root,
        [artifact_root, release_root, *pinned_files],
        code="teacher_full_parent_chain_invalid",
    )
    if not set(binding_chain).issubset(set(full_chain)) or not set(
        artifact_chain
    ).issubset(set(full_chain)):
        raise MultiArmRuntimeError("teacher_parent_chain_identity_drift")
    authenticated = AuthenticatedTeacherFinal(
        schema_version=TEACHER_SOURCE_VERSION,
        binding_path=requested,
        binding_sha256=expected_sha,
        binding_sidecar_sha256=binding_sidecar_sha,
        binding_sidecar_bytes=len(sidecar_raw),
        binding_schema_sha256=binding_schema_sha,
        artifact_root=artifact_root,
        manifest=manifest,
        record_schema_sha256=record_schema_sha,
        release_receipt=release_receipt,
        release_attestation=release_attestation,
        metadata_files=(candidate_manifest, output_inventory, release_request),
        shards=tuple(shard_pins),
        shard_assets=tuple(shard_assets),
        shard_records=tuple(shard_records),
        shard_inventory_sha256=shard_inventory_sha,
        public_identity_sha256=_canonical_sha256(identity_preimage),
        producer=producer,
        _directory_identities=full_chain,
        _authority=_TEACHER_AUTHORITY,
    )
    terminal_recheck_teacher(authenticated)
    return authenticated, binding


def terminal_recheck_teacher(teacher: AuthenticatedTeacherFinal) -> None:
    if teacher._authority is not _TEACHER_AUTHORITY:
        raise MultiArmRuntimeError("teacher_capability_invalid")
    _recheck_directory_chain(
        teacher._directory_identities,
        code="teacher_parent_chain_changed",
    )
    _stable_file(
        teacher.binding_path,
        expected_sha256=teacher.binding_sha256,
        code="teacher_binding_changed",
    )
    _stable_file(
        teacher.binding_path.with_name(teacher.binding_path.name + ".sha256"),
        expected_sha256=teacher.binding_sidecar_sha256,
        expected_bytes=teacher.binding_sidecar_bytes,
        code="teacher_binding_sidecar_changed",
    )
    for label, pin in (
        ("manifest", teacher.manifest),
        ("release_receipt", teacher.release_receipt),
        ("release_attestation", teacher.release_attestation),
        *(
            (f"metadata_{index}", pin)
            for index, pin in enumerate(teacher.metadata_files)
        ),
        *((f"shard_{index}", pin) for index, pin in enumerate(teacher.shards)),
    ):
        _stable_file(
            pin.path,
            expected_sha256=pin.sha256,
            expected_bytes=pin.bytes,
            code=f"teacher_{label}_changed",
        )
        _stable_file(
            pin.path.with_name(pin.path.name + ".sha256"),
            expected_sha256=pin.sidecar_sha256,
            expected_bytes=pin.sidecar_bytes,
            code=f"teacher_{label}_sidecar_changed",
        )
    _recheck_directory_chain(
        teacher._directory_identities,
        code="teacher_parent_chain_changed_after_files",
    )
    _recheck_rollover_producer(teacher.producer)


def _source_main_prompt(record: Mapping[str, Any]) -> SourcePrompt:
    if (
        record.get("schema_version")
        != "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-record.v1"
        or record.get("namespace") != "gemma3_chat_five_expert_qonly_unbalanced_v2"
        or record.get("split") != "train"
        or record.get("role") not in MAIN_ROLES
    ):
        raise MultiArmRuntimeError("source_main_record_contract_invalid")
    messages = _sequence(record.get("messages"), "source_main_messages_invalid")
    if (
        len(messages) != 2
        or _mapping(messages[0], "source_main_system_invalid").get("role") != "system"
        or _mapping(messages[1], "source_main_user_invalid").get("role") != "user"
    ):
        raise MultiArmRuntimeError("source_main_message_order_invalid")
    system = messages[0].get("content")
    user = messages[1].get("content")
    if (
        not isinstance(system, str)
        or not system
        or not isinstance(user, str)
        or not user
    ):
        raise MultiArmRuntimeError("source_main_prompt_missing")
    prompt = f"{system}\n\n{user}"
    special_prompt = (
        f"<start_of_turn>user\n{prompt}<end_of_turn><eos>\n<start_of_turn>model\n"
    )
    serialization = _mapping(
        record.get("gemma_serialization"),
        "source_main_serialization_invalid",
    )
    if (
        serialization.get("prompt_sha256") != _sha256(special_prompt.encode("utf-8"))
        or serialization.get("raw_token_ids_persisted") is not False
    ):
        raise MultiArmRuntimeError("source_main_prompt_serialization_drift")
    record_id = record.get("record_id")
    bundle = record.get("task_bundle_sha256")
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(bundle, str)
        or _SHA256_RE.fullmatch(bundle) is None
    ):
        raise MultiArmRuntimeError("source_main_join_identity_invalid")
    # Intentionally do not access record["target"].  Real runtime supervision
    # is exclusively the independently released Teacher FINAL.
    return SourcePrompt(
        record_id_sha256=_sha256(record_id.encode("utf-8")),
        source_asset=str(record["role"]),
        prompt=prompt,
        source_content_sha256=_canonical_sha256(messages),
        source_serialization_identity_sha256=_canonical_sha256(serialization),
        task_bundle_sha256=bundle,
        identity_training_record=isinstance(record.get("identity_alignment"), Mapping),
    )


def _source_router_prompt(record: Mapping[str, Any]) -> SourcePrompt | None:
    if record.get("namespace") != "gemma3_chat_emotion_router_qonly_v1" or record.get(
        "split"
    ) not in {"train", "eval_proxy"}:
        raise MultiArmRuntimeError("source_router_record_contract_invalid")
    if record["split"] == "eval_proxy":
        return None
    messages = _sequence(record.get("messages"), "source_router_messages_invalid")
    if (
        len(messages) != 2
        or _mapping(messages[0], "source_router_system_invalid").get("role") != "system"
        or _mapping(messages[1], "source_router_user_invalid").get("role") != "user"
    ):
        raise MultiArmRuntimeError("source_router_message_order_invalid")
    system = messages[0].get("content")
    user = messages[1].get("content")
    if (
        not isinstance(system, str)
        or not system
        or not isinstance(user, str)
        or not user
    ):
        raise MultiArmRuntimeError("source_router_prompt_missing")
    prompt = f"System instructions:\n{system}\n\nUser request:\n{user}"
    serialization = _mapping(
        record.get("serialization"), "source_router_serialization_invalid"
    )
    if (
        serialization.get("prompt_sha256") != _sha256(prompt.encode("utf-8"))
        or serialization.get("raw_token_ids_persisted") is not False
    ):
        raise MultiArmRuntimeError("source_router_prompt_serialization_drift")
    record_id = record.get("record_id")
    bundle = record.get("task_bundle_sha256")
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(bundle, str)
        or _SHA256_RE.fullmatch(bundle) is None
    ):
        raise MultiArmRuntimeError("source_router_join_identity_invalid")
    return SourcePrompt(
        record_id_sha256=_sha256(record_id.encode("utf-8")),
        source_asset="planner_router",
        prompt=prompt,
        source_content_sha256=_canonical_sha256(messages),
        source_serialization_identity_sha256=_canonical_sha256(serialization),
        task_bundle_sha256=bundle,
        identity_training_record=False,
    )


def load_source_prompts(
    source: source_runtime.AuthenticatedProducerSource,
) -> tuple[tuple[SourcePrompt, ...], str]:
    prompts: list[SourcePrompt] = []
    order = hashlib.sha256()
    for relative in TRAIN_SHARDS:
        for record in source_runtime.iter_authenticated_jsonl(source, relative):
            prompt = _source_main_prompt(record)
            prompts.append(prompt)
            order.update(prompt.record_id_sha256.encode("ascii") + b"\n")
    router_seen = 0
    for record in source_runtime.iter_authenticated_jsonl(source, ROUTER_RECORDS):
        router_seen += 1
        prompt = _source_router_prompt(record)
        if prompt is not None:
            prompts.append(prompt)
            order.update(prompt.record_id_sha256.encode("ascii") + b"\n")
    _validate_source_prompt_inventory(prompts, router_seen=router_seen)
    source_runtime.terminal_recheck_source(source)
    return tuple(prompts), order.hexdigest()


def _validate_source_prompt_inventory(
    prompts: Sequence[SourcePrompt],
    *,
    router_seen: int,
) -> None:
    """Recompute training-only inventory, including 20/100 identity quota."""

    counts = Counter(item.source_asset for item in prompts)
    identity_prompts = [item for item in prompts if item.identity_training_record]
    identity_by_bundle: dict[str, list[SourcePrompt]] = {}
    for item in identity_prompts:
        identity_by_bundle.setdefault(item.task_bundle_sha256, []).append(item)
    if (
        len(prompts) != 3520
        or router_seen != 100
        or dict(counts) != EXPECTED_ASSET_COUNTS
        or len({item.record_id_sha256 for item in prompts}) != 3520
        or len(identity_prompts) != 100
        or len(identity_by_bundle) != 20
        or any(
            len(items) != 5 or {item.source_asset for item in items} != set(MAIN_ROLES)
            for items in identity_by_bundle.values()
        )
        or any(item.source_asset == "planner_router" for item in identity_prompts)
    ):
        raise MultiArmRuntimeError("source_prompt_inventory_drift")


def _teacher_target(record: Mapping[str, Any]) -> TeacherTarget:
    required = (
        "record_id_sha256",
        "source_content_sha256",
        "source_serialization_identity_sha256",
        "task_bundle_sha256",
        "source_asset",
        "decision",
        "teacher_target",
        "teacher_target_sha256",
    )
    if (
        record.get("schema_version") != TEACHER_RECORD_VERSION
        or record.get("namespace") != "gemma3_chat_unbalanced_v2_teacher_alignment_v2"
        or any(key not in record for key in required)
        or record.get("decision") != "accepted"
    ):
        raise MultiArmRuntimeError("teacher_record_contract_invalid")
    target = record.get("teacher_target")
    if not isinstance(target, str) or not target:
        raise MultiArmRuntimeError("teacher_target_invalid")
    target_sha = _require_sha(
        record["teacher_target_sha256"], "teacher_target_sha_invalid"
    )
    if _sha256(target.encode("utf-8")) != target_sha:
        raise MultiArmRuntimeError("teacher_target_identity_drift")
    identities = {
        key: _require_sha(record[key], f"teacher_{key}_invalid")
        for key in (
            "record_id_sha256",
            "source_content_sha256",
            "source_serialization_identity_sha256",
            "task_bundle_sha256",
        )
    }
    asset = str(record["source_asset"])
    if asset not in EXPECTED_ASSET_COUNTS:
        raise MultiArmRuntimeError("teacher_source_asset_invalid")
    if asset in {"tool_call", "review_audit", "planner_router"}:
        _strict_json(
            target.encode("utf-8"),
            f"teacher_{asset}_target_json_invalid",
        )
    return TeacherTarget(
        record_id_sha256=identities["record_id_sha256"],
        source_asset=asset,
        source_content_sha256=identities["source_content_sha256"],
        source_serialization_identity_sha256=identities[
            "source_serialization_identity_sha256"
        ],
        task_bundle_sha256=identities["task_bundle_sha256"],
        target=target,
        target_sha256=target_sha,
    )


def load_teacher_targets(
    teacher: AuthenticatedTeacherFinal,
) -> Mapping[str, TeacherTarget]:
    if teacher._authority is not _TEACHER_AUTHORITY:
        raise MultiArmRuntimeError("teacher_body_read_without_authority")
    targets: dict[str, TeacherTarget] = {}
    counts: Counter[str] = Counter()
    record_validator = _load_schema(_project_root() / TEACHER_RECORD_SCHEMA_PATH)
    for pin, expected_asset, expected_records in zip(
        teacher.shards,
        teacher.shard_assets,
        teacher.shard_records,
        strict=True,
    ):
        raw = _stable_file(
            pin.path,
            expected_sha256=pin.sha256,
            expected_bytes=pin.bytes,
            code="teacher_shard_changed_before_parse",
        )
        seen = 0
        for line in raw.splitlines(keepends=True):
            if (
                not line
                or len(line) > 2_000_000
                or b"\r" in line
                or not line.endswith(b"\n")
            ):
                raise MultiArmRuntimeError("teacher_jsonl_line_invalid")
            record = _strict_json(line[:-1], "teacher_jsonl_record_invalid")
            try:
                record_validator.validate(record)
            except ValidationError as exc:
                raise MultiArmRuntimeError(
                    f"teacher_record_schema_mismatch:{exc.validator}"
                ) from None
            target = _teacher_target(record)
            observed_asset_class = (
                "planner_router_train"
                if target.source_asset == "planner_router"
                else "main_train"
            )
            if observed_asset_class != expected_asset:
                raise MultiArmRuntimeError("teacher_shard_asset_mismatch")
            if target.record_id_sha256 in targets:
                raise MultiArmRuntimeError("teacher_record_id_duplicate")
            targets[target.record_id_sha256] = target
            counts[target.source_asset] += 1
            seen += 1
        if seen != expected_records:
            raise MultiArmRuntimeError("teacher_shard_record_count_drift")
    if len(targets) != 3520 or dict(counts) != EXPECTED_ASSET_COUNTS:
        raise MultiArmRuntimeError("teacher_target_inventory_drift")
    terminal_recheck_teacher(teacher)
    return targets


def join_teacher_targets(
    prompts: Sequence[SourcePrompt],
    targets: Mapping[str, TeacherTarget],
) -> tuple[Mapping[str, tuple[TrainingExample, ...]], str]:
    assets: dict[str, list[TrainingExample]] = {
        key: [] for key in EXPECTED_ASSET_COUNTS
    }
    projection = hashlib.sha256()
    seen: set[str] = set()
    for prompt in prompts:
        target = targets.get(prompt.record_id_sha256)
        if target is None:
            raise MultiArmRuntimeError("teacher_join_record_missing")
        expected = (
            prompt.source_asset,
            prompt.source_content_sha256,
            prompt.source_serialization_identity_sha256,
            prompt.task_bundle_sha256,
        )
        observed = (
            target.source_asset,
            target.source_content_sha256,
            target.source_serialization_identity_sha256,
            target.task_bundle_sha256,
        )
        if observed != expected:
            raise MultiArmRuntimeError("teacher_join_identity_mismatch")
        seen.add(prompt.record_id_sha256)
        projection.update(target.target_sha256.encode("ascii") + b"\n")
        assets[prompt.source_asset].append(
            TrainingExample(
                record_id_sha256=prompt.record_id_sha256,
                source_asset=prompt.source_asset,
                prompt=prompt.prompt,
                target=target.target,
                source_content_sha256=prompt.source_content_sha256,
                source_serialization_identity_sha256=(
                    prompt.source_serialization_identity_sha256
                ),
                task_bundle_sha256=prompt.task_bundle_sha256,
                teacher_target_sha256=target.target_sha256,
            )
        )
    if (
        seen != set(targets)
        or {key: len(values) for key, values in assets.items()} != EXPECTED_ASSET_COUNTS
    ):
        raise MultiArmRuntimeError("teacher_join_not_exact_bijection")
    return (
        {key: tuple(values) for key, values in assets.items()},
        projection.hexdigest(),
    )


def serialize_training_assets(
    assets: Mapping[str, tuple[TrainingExample, ...]],
    tokenizer: source_runtime.AuthenticatedTokenizer,
) -> tuple[Mapping[str, tuple[SerializedTrainingExample, ...]], str, int]:
    serialized_assets: dict[str, tuple[SerializedTrainingExample, ...]] = {}
    inventory = hashlib.sha256()
    maximum = 0
    for asset in EXPECTED_ASSET_COUNTS:
        values: list[SerializedTrainingExample] = []
        for example in assets[asset]:
            encoded = source_runtime.serialize_authenticated_example(
                tokenizer, example.prompt, example.target, sequence_length=768
            )
            maximum = max(maximum, encoded.sequence_tokens)
            inventory.update(
                encoded.serialization_identity_sha256.encode("ascii") + b"\n"
            )
            values.append(
                SerializedTrainingExample(
                    record_id_sha256=example.record_id_sha256,
                    input_ids=encoded.input_ids,
                    labels=encoded.labels,
                    serialization_identity_sha256=(
                        encoded.serialization_identity_sha256
                    ),
                )
            )
        serialized_assets[asset] = tuple(values)
    if maximum > 768:
        raise MultiArmRuntimeError("serialized_training_sequence_too_long")
    return serialized_assets, inventory.hexdigest(), maximum


def build_runtime_datasets(
    source: source_runtime.AuthenticatedProducerSource,
    teacher: AuthenticatedTeacherFinal,
    tokenizer: source_runtime.AuthenticatedTokenizer,
) -> RuntimeDatasets:
    prompts, source_order_sha = load_source_prompts(source)
    targets = load_teacher_targets(teacher)
    assets, target_projection_sha = join_teacher_targets(prompts, targets)
    serialized, serialization_sha, maximum = serialize_training_assets(
        assets, tokenizer
    )
    source_runtime.terminal_recheck_source(source)
    terminal_recheck_teacher(teacher)
    return RuntimeDatasets(
        assets=assets,
        serialized_assets=serialized,
        source_record_order_sha256=source_order_sha,
        teacher_target_projection_sha256=target_projection_sha,
        serialized_inventory_sha256=serialization_sha,
        max_sequence_tokens=maximum,
    )


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    code: str,
) -> Mapping[str, Any]:
    expected = _require_sha(expected_sha256, f"{code}_expected_sha_invalid")
    requested = Path(path).resolve()
    raw = _stable_file(
        requested,
        expected_sha256=expected,
        code=f"{code}_identity_drift",
    )
    sidecar_raw = _stable_file(
        requested.with_name(requested.name + ".sha256"),
        expected_sha256=_sha256(f"{expected}  {requested.name}\n".encode("ascii")),
        expected_bytes=len(f"{expected}  {requested.name}\n".encode("ascii")),
        code=f"{code}_sidecar_identity_drift",
    )
    if sidecar_raw != f"{expected}  {requested.name}\n".encode("ascii"):
        raise MultiArmRuntimeError(f"{code}_sidecar_content_invalid")
    return _strict_json(raw, f"{code}_json_invalid")


def _windows_lock_is_held(path: Path) -> bool:
    if os.name != "nt":
        return path.exists()
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = ctypes.windll.kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        0x80000000,
        0,
        None,
        3,
        0x80,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle != invalid:
        close_handle(handle)
        return False
    return int(ctypes.windll.kernel32.GetLastError()) == 32


def _validate_gpu_samples(
    samples_value: object,
    *,
    phase: str,
    expected_uuid: str,
    config: Mapping[str, Any],
) -> None:
    samples = _sequence(samples_value, "launcher_gpu_samples_invalid")
    gpu = _mapping(config["gpu_policy"], "runtime_gpu_policy_invalid")
    if len(samples) != int(gpu["prestart_sample_count"]):
        raise MultiArmRuntimeError("launcher_gpu_sample_count_invalid")
    for ordinal, raw_sample in enumerate(samples, start=1):
        sample = _mapping(raw_sample, "launcher_gpu_sample_invalid")
        if (
            sample.get("phase") != phase
            or sample.get("ordinal") != ordinal
            or sample.get("index") != 0
            or sample.get("uuid") != expected_uuid
            or sample.get("memory_total_mib") != 12288
            or not isinstance(sample.get("observed_at_utc"), str)
            or type(sample.get("memory_used_mib")) is not int
            or int(sample["memory_used_mib"]) > int(gpu["idle_used_memory_max_mib"])
            or type(sample.get("memory_free_mib")) is not int
            or int(sample["memory_free_mib"]) < int(gpu["idle_free_memory_min_mib"])
            or type(sample.get("utilization_percent")) is not int
            or int(sample["utilization_percent"])
            > int(gpu["idle_utilization_max_percent"])
            or type(sample.get("temperature_c")) is not int
            or int(sample["temperature_c"]) > int(gpu["prestart_temperature_max_c"])
            or sample.get("external_compute_process_policy") != "observe_only"
        ):
            raise MultiArmRuntimeError("launcher_gpu_sample_gate_failed")


def validate_launcher_context(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    run_id: str,
    mode: str,
    teacher_binding_sha256: str,
    implementation_set_sha256: str,
    lease_path: str | Path,
    lease_sha256: str,
    attestation_path: str | Path,
    attestation_sha256: str,
) -> dict[str, object]:
    lease = _load_bound_json(lease_path, lease_sha256, code="launcher_lease")
    attestation = _load_bound_json(
        attestation_path, attestation_sha256, code="gpu_attestation"
    )
    expected_uuid = os.environ.get(str(config["gpu_policy"]["expected_gpu_uuid_env"]))
    if not expected_uuid or expected_uuid == "UNBOUND":
        raise MultiArmRuntimeError("expected_gpu_uuid_unbound")
    if (
        lease.get("schema_version") != LEASE_VERSION
        or lease.get("status") != "active"
        or lease.get("run_id") != run_id
        or lease.get("mode") != mode
        or lease.get("config_sha256") != config_sha256
        or lease.get("teacher_binding_sha256") != teacher_binding_sha256
        or lease.get("implementation_set_sha256") != implementation_set_sha256
        or lease.get("canonical_lock_path") != config["gpu_policy"]["canonical_lock"]
        or lease.get("concurrency") != 1
        or lease.get("resume") is not False
        or lease.get("fresh_base_per_phase") is not True
        or lease.get("fresh_adapter_per_phase") is not True
        or lease.get("automatic_retry") is not False
    ):
        raise MultiArmRuntimeError("launcher_lease_contract_drift")
    parent_pid = lease.get("launcher_pid")
    if type(parent_pid) is not int or parent_pid <= 0:
        raise MultiArmRuntimeError("launcher_pid_invalid")
    try:
        os.kill(parent_pid, 0)
    except OSError:
        raise MultiArmRuntimeError("launcher_process_not_alive") from None
    lock_path = _project_root() / str(config["gpu_policy"]["canonical_lock"])
    if not lock_path.is_file() or not _windows_lock_is_held(lock_path):
        raise MultiArmRuntimeError("canonical_gpu_lock_not_held_exclusive")
    if (
        attestation.get("schema_version") != ATTESTATION_VERSION
        or attestation.get("status") != "passed"
        or attestation.get("run_id") != run_id
        or attestation.get("mode") != mode
        or attestation.get("config_sha256") != config_sha256
        or attestation.get("teacher_binding_sha256") != teacher_binding_sha256
        or attestation.get("implementation_set_sha256") != implementation_set_sha256
        or attestation.get("expected_gpu_uuid") != expected_uuid
        or attestation.get("canonical_lock_path")
        != config["gpu_policy"]["canonical_lock"]
        or attestation.get("external_compute_process_policy") != "observe_only"
    ):
        raise MultiArmRuntimeError("gpu_attestation_contract_drift")
    _validate_gpu_samples(
        attestation.get("pre_lock_samples"),
        phase="pre_lock",
        expected_uuid=expected_uuid,
        config=config,
    )
    _validate_gpu_samples(
        attestation.get("post_lock_samples"),
        phase="post_lock",
        expected_uuid=expected_uuid,
        config=config,
    )
    return {
        "lease_sha256": _require_sha(lease_sha256, "launcher_lease_sha_invalid"),
        "attestation_sha256": _require_sha(
            attestation_sha256, "gpu_attestation_sha_invalid"
        ),
        "launcher_pid": parent_pid,
        "expected_gpu_uuid": expected_uuid,
        "canonical_lock_path": str(config["gpu_policy"]["canonical_lock"]),
        "lock_exclusive": True,
        "external_compute_process_policy": "observe_only",
    }


def _runtime_package_versions() -> dict[str, str]:
    try:
        observed = {
            name: importlib.metadata.version(name) for name in RUNTIME_PACKAGE_NAMES
        }
    except importlib.metadata.PackageNotFoundError:
        raise MultiArmRuntimeError("runtime_package_missing") from None
    if (
        observed["bitsandbytes"] != "0.48.2"
        or observed["torch"] != "2.5.1+cu121"
        or observed["transformers"] != "5.13.0"
        or observed["peft"] != "0.19.1"
    ):
        raise MultiArmRuntimeError("runtime_package_version_drift")
    return observed


def _query_runtime_gpu(config: Mapping[str, Any]) -> dict[str, object]:
    gpu = _mapping(config["gpu_policy"], "runtime_gpu_policy_invalid")
    expected_uuid = os.environ.get(str(gpu["expected_gpu_uuid_env"]))
    if not expected_uuid:
        raise MultiArmRuntimeError("runtime_gpu_uuid_unbound")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_model.current,memory.total,"
                "memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=int(gpu["command_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired):
        raise MultiArmRuntimeError("runtime_nvidia_smi_failed") from None
    rows = [
        [cell.strip() for cell in row.split(",")]
        for row in result.stdout.splitlines()
        if row.strip()
    ]
    if result.returncode != 0 or len(rows) <= int(gpu["expected_gpu_index"]):
        raise MultiArmRuntimeError("runtime_gpu_query_invalid")
    row = rows[int(gpu["expected_gpu_index"])]
    if len(row) != 8:
        raise MultiArmRuntimeError("runtime_gpu_row_invalid")
    try:
        snapshot: dict[str, object] = {
            "index": int(row[0]),
            "uuid": row[1],
            "driver_model": row[2],
            "memory_total_mib": int(row[3]),
            "memory_used_mib": int(row[4]),
            "memory_free_mib": int(row[5]),
            "utilization_percent": int(row[6]),
            "temperature_c": int(row[7]),
        }
    except ValueError:
        raise MultiArmRuntimeError("runtime_gpu_values_invalid") from None
    if (
        snapshot["index"] != 0
        or snapshot["uuid"] != expected_uuid
        or snapshot["memory_total_mib"] != 12288
        or int(snapshot["temperature_c"]) > int(gpu["runtime_temperature_max_c"])
    ):
        raise MultiArmRuntimeError("runtime_gpu_identity_or_temperature_failed")
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=int(gpu["command_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired):
        raise MultiArmRuntimeError("runtime_compute_query_failed") from None
    if processes.returncode != 0:
        raise MultiArmRuntimeError("runtime_compute_query_invalid")
    inventory: list[dict[str, object]] = []
    try:
        rows_iter = csv.reader(processes.stdout.splitlines())
        for process_row in rows_iter:
            if len(process_row) != 4 or process_row[0].strip() != expected_uuid:
                continue
            try:
                pid = int(process_row[1].strip())
            except ValueError:
                raise MultiArmRuntimeError("runtime_compute_pid_invalid") from None
            inventory.append(
                {
                    "pid": pid,
                    "name_sha256": _sha256(
                        process_row[2].strip().casefold().encode("utf-8")
                    ),
                    "used_gpu_memory": process_row[3].strip(),
                    "current_process": pid == os.getpid(),
                }
            )
    except csv.Error:
        raise MultiArmRuntimeError("runtime_compute_inventory_invalid") from None
    snapshot["external_compute_process_policy"] = "observe_only"
    snapshot["compute_inventory_sha256"] = _canonical_sha256(
        sorted(inventory, key=lambda item: int(item["pid"]))
    )
    snapshot["compute_process_count"] = len(inventory)
    return snapshot


def _canonical_lora_name(name: str) -> str:
    match = _LORA_RE.search(name)
    if match is None:
        raise MultiArmRuntimeError("runtime_trainable_scope_invalid")
    layer, projection, factor = match.groups()
    return f"model.layers.{int(layer)}.self_attn.{projection}.lora_{factor}.weight"


def _trainable_parameter_map(model: Any) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical = _canonical_lora_name(name)
        if canonical in observed:
            raise MultiArmRuntimeError("runtime_trainable_tensor_duplicate")
        observed[canonical] = parameter
    return observed


def _initialize_and_validate_inventory(
    model: Any,
    *,
    arm: str,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = _trainable_parameter_map(model)
    expected = contract.expected_tensor_metadata(arm)
    expected_names = [str(item["name"]) for item in expected]
    if set(parameters) != set(expected_names):
        raise MultiArmRuntimeError("runtime_tensor_inventory_scope_drift")
    for item in expected:
        name = str(item["name"])
        parameter = parameters[name]
        if (
            tuple(int(value) for value in parameter.shape)
            != tuple(int(value) for value in item["shape"])
            or parameter.dtype != torch.bfloat16
            or not parameter.requires_grad
        ):
            raise MultiArmRuntimeError("runtime_tensor_shape_or_dtype_drift")
        if item["factor"] == "A":
            seed = int(str(item["initializer_seed_sha256"])[:16], 16)
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(seed)
            torch.nn.init.kaiming_uniform_(
                parameter,
                a=math.sqrt(5),
                generator=generator,
            )
        else:
            torch.nn.init.zeros_(parameter)
    actual_entries = [
        {
            **item,
            "tensor_bytes_sha256": qdiag._tensor_sha256(
                parameters[str(item["name"])], torch
            ),
        }
        for item in expected
    ]
    inventory = {
        "schema_version": contract.INVENTORY_SCHEMA_VERSION,
        "arm": arm,
        "branch_seed": contract.BASE_SEED,
        "seed_material": {
            "base_seed": contract.BASE_SEED,
            "derivation": "base_seed_only_v1",
            "components": ["base_seed"],
        },
        "base_o_proj_frozen": True,
        "tensors": actual_entries,
        "optimizer_groups": contract.expected_optimizer_groups(arm, expected_names),
    }
    try:
        normalized = contract._validate_arm_inventory(inventory, arm)
    except Exception as exc:
        raise MultiArmRuntimeError(
            f"runtime_observed_inventory_invalid:{type(exc).__name__}"
        ) from None
    for name, parameter in model.named_parameters():
        if "o_proj" in name and "lora_" not in name and bool(parameter.requires_grad):
            raise MultiArmRuntimeError("runtime_base_o_proj_unfrozen")
        if "lora_" not in name and bool(parameter.requires_grad):
            raise MultiArmRuntimeError("runtime_base_parameter_unfrozen")
    return normalized, parameters


def _trainable_digest(parameters: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(parameters):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(qdiag._tensor_sha256(parameters[name], torch).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _optimizer_groups(
    config: Mapping[str, Any],
    *,
    arm: str,
    parameters: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[Any]]:
    profile = contract.ARM_PROFILE[arm]
    groups: list[dict[str, object]] = []
    flattened: list[Any] = []
    if profile in {"q_only", "q_plus_micro_o"}:
        values = [parameters[name] for name in sorted(parameters) if ".q_proj." in name]
        groups.append({"params": values, "lr": float(contract.Q_LR), "name": "Q"})
        flattened.extend(values)
    if profile in {"o_only", "q_plus_micro_o"}:
        values = [parameters[name] for name in sorted(parameters) if ".o_proj." in name]
        groups.append({"params": values, "lr": float(contract.O_LR), "name": "O"})
        flattened.extend(values)
    if (
        not groups
        or len(flattened) != len(parameters)
        or len({id(value) for value in flattened}) != len(parameters)
        or [str(group["name"]) for group in groups]
        != [
            str(value["name"])
            for value in contract.expected_optimizer_groups(arm, sorted(parameters))
        ]
    ):
        raise MultiArmRuntimeError("runtime_optimizer_group_scope_invalid")
    return groups, flattened


def _active_learning_rates(
    arm: str,
    *,
    step: int,
    total_steps: int,
) -> tuple[Decimal, Decimal]:
    q_lr, o_lr = contract.expected_step_learning_rates(
        step=step, total_steps=total_steps
    )
    profile = contract.ARM_PROFILE[arm]
    return (
        q_lr if profile in {"q_only", "q_plus_micro_o"} else Decimal(0),
        o_lr if profile in {"o_only", "q_plus_micro_o"} else Decimal(0),
    )


def _batch_for_example(
    example: SerializedTrainingExample, *, torch: Any
) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([example.input_ids], dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(
            (1, len(example.input_ids)), dtype=torch.long, device="cuda"
        ),
        "labels": torch.tensor([example.labels], dtype=torch.long, device="cuda"),
    }


def _gradient_evidence(
    parameters: Mapping[str, Any],
    *,
    torch: Any,
) -> tuple[dict[str, int], float]:
    evidence = {
        "tensors": 0,
        "finite": 0,
        "nonzero": 0,
        "q_nonzero": 0,
        "o_nonzero": 0,
        "A_nonzero": 0,
        "B_nonzero": 0,
    }
    squared = 0.0
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise MultiArmRuntimeError("runtime_gradient_missing_or_nonfinite")
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise MultiArmRuntimeError("runtime_parameter_nonfinite")
        evidence["tensors"] += 1
        evidence["finite"] += 1
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).cpu())
        if not math.isfinite(norm):
            raise MultiArmRuntimeError("runtime_gradient_norm_nonfinite")
        squared += norm * norm
        if int(torch.count_nonzero(gradient).item()) > 0:
            evidence["nonzero"] += 1
            evidence["q_nonzero" if ".q_proj." in name else "o_nonzero"] += 1
            evidence["A_nonzero" if ".lora_A." in name else "B_nonzero"] += 1
    if evidence["tensors"] != len(parameters) or evidence["nonzero"] < 1:
        raise MultiArmRuntimeError("runtime_gradient_coverage_failed")
    return evidence, math.sqrt(squared)


def _adapter_effect(
    model: Any,
    example: SerializedTrainingExample,
    *,
    torch: Any,
) -> dict[str, object]:
    supervised = [index for index, label in enumerate(example.labels) if label != -100]
    if not supervised or supervised[0] <= 0:
        raise MultiArmRuntimeError("runtime_adapter_effect_boundary_invalid")
    prefix = example.input_ids[: supervised[0]]
    batch = {
        "input_ids": torch.tensor([prefix], dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long, device="cuda"),
    }
    try:
        effect = q8_runtime._enabled_vs_disabled_next_token_effect(
            model,
            batch,
            len(prefix) - 1,
            torch=torch,
        )
    except Exception as exc:
        raise MultiArmRuntimeError(
            f"runtime_adapter_effect_failed:{type(exc).__name__}"
        ) from None
    return {
        "view": "first_teacher_aligned_train_record_boundary_v2",
        "record_id_sha256": example.record_id_sha256,
        "serialization_identity_sha256": (example.serialization_identity_sha256),
        "forward_prefix_tokens": len(prefix),
        "future_target_suffix_forwarded": False,
        "sample_body_included": False,
        "raw_token_ids_included": False,
        **effect,
    }


def _save_adapter(
    model: Any,
    output: Path,
    *,
    arm: str,
    inventory: Mapping[str, Any],
    torch: Any,
) -> dict[str, object]:
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    temporary_identity = _directory_identity(
        temporary,
        code="runtime_adapter_temporary_identity_invalid",
    )
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        config_path = temporary / "adapter_config.json"
        model_path = temporary / "adapter_model.safetensors"
        if (
            not config_path.is_file()
            or not model_path.is_file()
            or len(list(temporary.iterdir())) != 2
        ):
            raise MultiArmRuntimeError("runtime_adapter_file_inventory_invalid")
        try:
            from safetensors import safe_open
        except ImportError:
            raise MultiArmRuntimeError("runtime_safetensors_unavailable") from None
        expected = {
            str(item["name"]): tuple(int(value) for value in item["shape"])
            for item in inventory["tensors"]
        }
        observed: dict[str, tuple[int, ...]] = {}
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                canonical = _canonical_lora_name(name)
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or canonical in observed:
                    raise MultiArmRuntimeError(
                        "runtime_saved_adapter_dtype_or_name_invalid"
                    )
                observed[canonical] = tuple(int(value) for value in tensor.shape)
        if observed != expected:
            raise MultiArmRuntimeError("runtime_saved_adapter_shape_drift")
        entries: dict[str, dict[str, object]] = {}
        for path in (config_path, model_path):
            raw = path.read_bytes()
            entries[path.name] = {
                "path": path.name,
                "sha256": _sha256(raw),
                "bytes": len(raw),
            }
        tree_sha = _canonical_sha256([entries[name] for name in sorted(entries)])
        for name, entry in entries.items():
            _stable_file(
                temporary / name,
                expected_sha256=str(entry["sha256"]),
                expected_bytes=int(entry["bytes"]),
                code="runtime_adapter_file_changed_before_publish",
            )
        _move_directory_create_once(
            temporary,
            output,
            owned_identity=temporary_identity,
            code="runtime_adapter_publish_destination_exists_or_move_failed",
        )
        for name, entry in entries.items():
            _stable_file(
                output / name,
                expected_sha256=str(entry["sha256"]),
                expected_bytes=int(entry["bytes"]),
                code="runtime_adapter_file_changed_after_publish",
            )
        return {
            "path": f"adapters/{arm}",
            "adapter_profile": contract.ARM_PROFILE[arm],
            "trainable_parameters": sum(
                int(item["numel"]) for item in inventory["tensors"]
            ),
            "adapter_config": entries["adapter_config.json"],
            "adapter_model": entries["adapter_model.safetensors"],
            "artifact_tree_sha256": tree_sha,
            "tensor_inventory_sha256": _canonical_sha256(inventory),
        }
    except BaseException:
        # Preserve the owned temporary directory as failure evidence.  Never
        # recursively delete through a path whose identity could have changed.
        raise


def _memory_evidence(
    config: Mapping[str, Any],
    *,
    torch: Any,
    driver_samples: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    gpu = _mapping(config["gpu_policy"], "runtime_gpu_policy_invalid")
    allocated_limit = int(gpu["torch_peak_allocated_max_mib"]) * _MIB
    reserved_limit = int(gpu["torch_peak_reserved_max_mib"]) * _MIB
    if allocated > allocated_limit or reserved > reserved_limit or not driver_samples:
        raise MultiArmRuntimeError("runtime_torch_memory_budget_exceeded")
    return {
        "torch_peak_allocated_bytes": allocated,
        "torch_peak_reserved_bytes": reserved,
        "torch_peak_allocated_max_mib": int(gpu["torch_peak_allocated_max_mib"]),
        "torch_peak_reserved_max_mib": int(gpu["torch_peak_reserved_max_mib"]),
        "comparison": "less_than_or_equal",
        "driver_observation_only": True,
        "driver_memory_used_peak_mib": max(
            int(sample["memory_used_mib"]) for sample in driver_samples
        ),
        "driver_sample_count": len(driver_samples),
        "driver_samples_sha256": _canonical_sha256(driver_samples),
        "external_compute_process_policy": "observe_only",
    }


def _peft_config_for_arm(arm: str, LoraConfig: Any) -> Any:
    profile = contract.ARM_PROFILE[arm]
    common = {
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if profile == "q_only":
        return LoraConfig(
            r=1024,
            lora_alpha=2048,
            target_modules=["q_proj"],
            **common,
        )
    if profile == "o_only":
        return LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["o_proj"],
            **common,
        )
    if profile == "q_plus_micro_o":
        return LoraConfig(
            r=1024,
            lora_alpha=2048,
            target_modules=["q_proj", "o_proj"],
            rank_pattern={"o_proj": 64},
            alpha_pattern={"o_proj": 128},
            **common,
        )
    raise MultiArmRuntimeError("runtime_adapter_profile_invalid")


def _execute_phase(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    run_id: str,
    mode: str,
    arm: str,
    phase: str,
    steps: int,
    examples: Sequence[SerializedTrainingExample],
    model_snapshot: Path,
    model_file_hashes: Mapping[str, str],
    phase_root: Path,
    phase_root_identity: DirectoryIdentity,
    progress_root: Path,
    source_identity: Mapping[str, object],
    teacher_identity: Mapping[str, object],
    tokenizer_identity: Mapping[str, object],
) -> dict[str, Any]:
    if (
        arm not in contract.TRAINED_ARMS
        or phase not in {"smoke", "full"}
        or phase != ("smoke" if mode == "smoke_only" else "full")
        or steps
        != (
            int(config["training"]["smoke_steps_per_arm"])
            if phase == "smoke"
            else contract.FULL_STEPS[arm]
        )
        or len(examples) != EXPECTED_ASSET_COUNTS[contract.ARM_SOURCE_ASSET[arm]]
        or steps > len(examples)
    ):
        raise MultiArmRuntimeError("runtime_phase_contract_invalid")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise MultiArmRuntimeError("runtime_cuda_visible_devices_not_zero")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    versions = _runtime_package_versions()
    if getattr(bnb, "__version__", None) != "0.48.2":
        raise MultiArmRuntimeError("runtime_bitsandbytes_module_version_drift")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise MultiArmRuntimeError("runtime_single_cuda_device_required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not (torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
        raise MultiArmRuntimeError("runtime_tf32_not_enabled")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    driver_samples: list[Mapping[str, Any]] = [_query_runtime_gpu(config)]
    quantization = _mapping(config["quantization"], "runtime_quantization_invalid")
    base = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(quantization["llm_int8_threshold"]),
            llm_int8_skip_modules=list(quantization["llm_int8_skip_modules"]),
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
        ),
    )
    overlay = _mapping(
        config["model"]["runtime_special_token_overlay"],
        "runtime_special_token_overlay_invalid",
    )
    base.config.pad_token_id = int(overlay["pad_token_id"])
    base.config.eos_token_id = int(overlay["eos_token_id"])
    base.config.bos_token_id = int(overlay["bos_token_id"])
    q8_modules = q8_runtime._validate_q8_base_modules(base, bnb=bnb, torch=torch)
    preparation = q8_runtime._prepare_frozen_q8_base(base, bnb=bnb, torch=torch)
    q8_before = q8_runtime._q8_quant_state_inventory(base, bnb=bnb, torch=torch)
    captured_base, base_hash_before = qdiag._capture_base_parameters(base, torch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(
        base,
        _peft_config_for_arm(arm, LoraConfig),
        autocast_adapter_dtype=False,
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    inventory, parameters = _initialize_and_validate_inventory(
        model, arm=arm, torch=torch
    )
    initial_adapter_sha = _trainable_digest(parameters, torch)
    groups, flattened = _optimizer_groups(config, arm=arm, parameters=parameters)
    optimizer_config = _mapping(config["optimizer"], "runtime_optimizer_invalid")
    optimizer = bnb.optim.AdamW8bit(
        groups,
        lr=float(contract.Q_LR),
        betas=(
            float(optimizer_config["beta1"]),
            float(optimizer_config["beta2"]),
        ),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
        optim_bits=int(optimizer_config["optim_bits"]),
        min_8bit_size=int(optimizer_config["min_8bit_size"]),
        percentile_clipping=int(optimizer_config["percentile_clipping"]),
        block_wise=bool(optimizer_config["block_wise"]),
        is_paged=bool(optimizer_config["is_paged"]),
    )
    losses: list[float] = []
    last_gradient: dict[str, int] | None = None
    gradient_union = Counter()
    optimizer_state: dict[str, object] | None = None
    order = hashlib.sha256()
    for index in range(steps):
        completed = index + 1
        example = examples[index]
        order.update(example.record_id_sha256.encode("ascii") + b"\n")
        q_lr, o_lr = _active_learning_rates(arm, step=completed, total_steps=steps)
        for group in optimizer.param_groups:
            name = str(group.get("name"))
            group["lr"] = float(q_lr if name == "Q" else o_lr)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = _batch_for_example(example, torch=torch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(
                **q8_runtime._exact_supervised_loss_batch(batch, torch=torch)
            ).loss
        if not bool(torch.isfinite(loss).item()):
            raise MultiArmRuntimeError("runtime_loss_nonfinite")
        loss.backward()
        last_gradient, gradient_norm = _gradient_evidence(parameters, torch=torch)
        gradient_union.update(
            {
                key: int(value)
                for key, value in last_gradient.items()
                if key.endswith("_nonzero")
            }
        )
        torch.nn.utils.clip_grad_norm_(
            flattened, max_norm=float(config["training"]["max_grad_norm"])
        )
        optimizer.step()
        if optimizer_state is None:
            optimizer_state = q8_runtime._validate_adamw8bit_state(
                optimizer,
                flattened,
                torch=torch,
                bitsandbytes_version=str(bnb.__version__),
            )
        qdiag._assert_trainable_parameters_finite(model, torch)
        observed_loss = float(loss.detach().cpu())
        if not math.isfinite(observed_loss):
            raise MultiArmRuntimeError("runtime_loss_not_scalar_finite")
        losses.append(observed_loss)
        contract.write_scalar_progress(
            progress_root,
            run_id=run_id,
            arm=arm,
            phase=phase,
            optimizer_step=completed,
            target_steps=steps,
            loss=observed_loss,
            gradient_norm=gradient_norm,
            lr_q=float(q_lr),
            lr_o=float(o_lr),
            torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            torch_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        del batch, loss
        if phase == "smoke" or completed == steps or completed % 80 == 0:
            driver_samples.append(_query_runtime_gpu(config))
    if last_gradient is None or optimizer_state is None:
        raise MultiArmRuntimeError("runtime_optimizer_did_not_step")
    final_adapter_sha = _trainable_digest(parameters, torch)
    if final_adapter_sha == initial_adapter_sha:
        raise MultiArmRuntimeError("runtime_adapter_did_not_change")
    if gradient_union["A_nonzero"] < 1 or gradient_union["B_nonzero"] < 1:
        raise MultiArmRuntimeError("runtime_gradient_factor_coverage_failed")
    adapter_effect = _adapter_effect(model, examples[0], torch=torch)
    try:
        base_hash_after = qdiag._assert_base_parameters_unchanged(captured_base, torch)
    except RuntimeError:
        raise MultiArmRuntimeError("runtime_frozen_base_changed") from None
    q8_after = q8_runtime._q8_quant_state_inventory(base, bnb=bnb, torch=torch)
    if q8_after != q8_before or base_hash_after != base_hash_before:
        raise MultiArmRuntimeError("runtime_base_or_scb_identity_changed")
    driver_samples.append(_query_runtime_gpu(config))
    memory = _memory_evidence(config, torch=torch, driver_samples=driver_samples)
    adapter: dict[str, object] | None = None
    if phase == "full":
        adapter = _save_adapter(
            model,
            phase_root.parent.parent / "adapters" / arm,
            arm=arm,
            inventory=inventory,
            torch=torch,
        )
    progress = contract.build_progress_evidence_identity(
        progress_root,
        run_id=run_id,
        arm=arm,
        phase=phase,
        target_steps=steps,
    )
    phase_receipt = {
        "schema_version": PHASE_RECEIPT_VERSION,
        "status": "passed",
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "source_asset": contract.ARM_SOURCE_ASSET[arm],
        "adapter_profile": contract.ARM_PROFILE[arm],
        "optimizer_steps": steps,
        "target_steps": steps,
        "source": dict(source_identity),
        "teacher_final": dict(teacher_identity),
        "tokenizer": dict(tokenizer_identity),
        "model": {
            "base_files": dict(model_file_hashes),
            "base_parameter_hash_before": base_hash_before,
            "base_parameter_hash_after": base_hash_after,
            "base_parameter_hash_equal": True,
            "q8_scb_sha256_before": q8_before["sha256"],
            "q8_scb_sha256_after": q8_after["sha256"],
            "q8_scb_hash_equal": True,
            "q8_modules": q8_modules,
            "base_preparation": preparation,
            "runtime_packages": versions,
        },
        "tensor_inventory": {
            "sha256": _canonical_sha256(inventory),
            "trainable_tensors": len(inventory["tensors"]),
            "trainable_parameters": sum(
                int(item["numel"]) for item in inventory["tensors"]
            ),
            "initial_adapter_sha256": initial_adapter_sha,
            "final_adapter_sha256": final_adapter_sha,
            "base_o_proj_frozen": True,
        },
        "optimizer": {
            **optimizer_state,
            "backend": "bitsandbytes.optim.AdamW8bit",
            "group_order": [str(group["name"]) for group in optimizer.param_groups],
            "q_learning_rate": float(contract.Q_LR),
            "o_learning_rate": float(contract.O_LR),
            "o_lr_exactly_one_tenth_of_q": True,
            "uint8_state": True,
        },
        "loss": {
            "finite": True,
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
            "count": len(losses),
        },
        "gradient": {
            **last_gradient,
            "all_steps_finite": True,
            "factor_union_nonzero": {
                "A": gradient_union["A_nonzero"],
                "B": gradient_union["B_nonzero"],
                "Q": gradient_union["q_nonzero"],
                "O": gradient_union["o_nonzero"],
            },
        },
        "adapter_effect": adapter_effect,
        "memory": memory,
        "progress": progress,
        "adapter": adapter,
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "sample_body_included": False,
            "raw_token_ids_included": False,
            "producer_original_target_used": False,
            "physical_kv_validated": False,
            "full_generation_kv_shared": False,
        },
    }
    _write_canonical(
        phase_root / "receipt.json",
        phase_receipt,
        schema_path=PHASE_SCHEMA_PATH,
        parent_identity=phase_root_identity,
    )
    del optimizer, model, base, captured_base, parameters, flattened
    gc.collect()
    torch.cuda.empty_cache()
    return phase_receipt


def _implementation_identity() -> dict[str, object]:
    paths = (
        CONFIG_PATH,
        CONFIG_SCHEMA_PATH,
        TEACHER_BINDING_SCHEMA_PATH,
        TEACHER_RECORD_SCHEMA_PATH,
        PHASE_SCHEMA_PATH,
        RUN_SCHEMA_PATH,
        FAILURE_SCHEMA_PATH,
        IMPLEMENTATION_PATH,
        SHARED_SOURCE_PATH,
        RUNNER_PATH,
        LAUNCHER_PATH,
        LAUNCHER_HELPER_PATH,
        MODEL_FREE_CONFIG_PATH,
        TEST_PATH,
    )
    files = [_runtime_file_identity(path) for path in paths]
    return {
        "files": files,
        "set_sha256": _canonical_sha256(files),
    }


def _validate_smoke_gate(
    path: str | Path,
    expected_sha256: str,
    *,
    config_sha256: str,
    source_identity: Mapping[str, object],
    teacher_identity: Mapping[str, object],
    tokenizer_identity: Mapping[str, object],
) -> dict[str, object]:
    value = _load_bound_json(path, expected_sha256, code="smoke_run_receipt")
    try:
        _load_schema(_project_root() / RUN_SCHEMA_PATH).validate(value)
    except ValidationError as exc:
        raise MultiArmRuntimeError(
            f"smoke_run_receipt_schema_mismatch:{exc.validator}"
        ) from None
    phases = _sequence(value.get("phase_receipts"), "smoke_phase_receipts_invalid")
    if (
        value.get("schema_version") != RUN_RECEIPT_VERSION
        or value.get("status") != "passed"
        or value.get("mode") != "smoke_only"
        or _mapping(value.get("config"), "smoke_config_invalid").get("sha256")
        != config_sha256
        or value.get("source") != source_identity
        or value.get("teacher_final") != teacher_identity
        or value.get("tokenizer") != tokenizer_identity
        or value.get("phase_order") != list(contract.TRAINED_ARMS)
        or len(phases) != 9
        or any(
            _mapping(item, "smoke_phase_identity_invalid").get("arm") != arm
            or item.get("phase") != "smoke"
            or item.get("status") != "passed"
            or item.get("target_steps") != 2
            for arm, item in zip(contract.TRAINED_ARMS, phases, strict=True)
        )
        or value.get("adapters") != {}
    ):
        raise MultiArmRuntimeError("smoke_run_gate_not_passed")
    return {
        "receipt_sha256": _require_sha(expected_sha256, "smoke_receipt_sha_invalid"),
        "run_id": str(value["run_id"]),
        "all_nine_arms_passed": True,
        "smoke_checkpoint_consumed": False,
    }


def _phase_summary(
    staging: Path,
    arm: str,
    phase: str,
    receipt: Mapping[str, Any],
) -> dict[str, object]:
    path = staging / "phases" / arm / "receipt.json"
    raw = path.read_bytes()
    return {
        "arm": arm,
        "phase": phase,
        "status": receipt["status"],
        "target_steps": receipt["target_steps"],
        "path": f"phases/{arm}/receipt.json",
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "q8_scb_sha256": receipt["model"]["q8_scb_sha256_after"],
        "base_parameter_sha256": receipt["model"]["base_parameter_hash_after"],
        "tensor_inventory_sha256": receipt["tensor_inventory"]["sha256"],
    }


def _publish_failure(
    staging: Path,
    *,
    staging_identity: DirectoryIdentity,
    run_id: str,
    mode: str,
    error: BaseException,
    last_completed_arm: str | None,
    source_authenticated: bool,
    teacher_authenticated: bool,
) -> None:
    raw_code = str(error).splitlines()[0] if str(error) else type(error).__name__
    safe_code = re.sub(r"[^A-Za-z0-9_.:-]", "_", raw_code)[:200]
    value = {
        "schema_version": FAILURE_RECEIPT_VERSION,
        "status": "failed",
        "run_id": run_id,
        "mode": mode,
        "error_type": type(error).__name__,
        "error_code": safe_code or type(error).__name__,
        "last_completed_arm": last_completed_arm,
        "source_authenticated": source_authenticated,
        "teacher_final_authenticated": teacher_authenticated,
        "sample_body_included": False,
        "raw_token_ids_included": False,
        "automatic_retry": False,
    }
    try:
        _write_canonical(
            staging / "failure.json",
            value,
            schema_path=FAILURE_SCHEMA_PATH,
            parent_identity=staging_identity,
        )
    except BaseException:
        # Preserve the original exception; a receipt-write failure must never
        # turn into an automatic retry or destructive staging cleanup.
        return


def execute_runtime(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    mode: str,
    run_id: str,
    producer_repository: str | Path | None,
    teacher_binding: str | Path,
    teacher_binding_sha256: str,
    lease_path: str | Path,
    lease_sha256: str,
    attestation_path: str | Path,
    attestation_sha256: str,
    smoke_receipt: str | Path | None = None,
    smoke_receipt_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    if (
        mode not in {"smoke_only", "full"}
        or _RUN_ID_RE.fullmatch(run_id) is None
        or (mode == "full")
        != (smoke_receipt is not None and smoke_receipt_sha256 is not None)
    ):
        raise MultiArmRuntimeError("runtime_execution_arguments_invalid")
    rollover_producer, _binding_schema_sha, record_schema_sha = (
        _released_teacher_schema_identities(config, producer_repository)
    )
    output_root = _project_root() / str(config["output"]["artifact_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    published = output_root / run_id
    staging = output_root / f".staging-{run_id}-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    staging_identity = _directory_identity(
        staging,
        code="runtime_staging_identity_invalid",
    )
    source_authenticated = False
    teacher_authenticated = False
    last_completed: str | None = None
    try:
        source = source_runtime.authenticate_producer_source(producer_repository)
        source_authenticated = True
        teacher, _binding = authenticate_teacher_final(
            teacher_binding,
            expected_binding_sha256=teacher_binding_sha256,
            producer=rollover_producer,
            expected_record_schema_sha256=record_schema_sha,
        )
        teacher_authenticated = True
        source_identity = source.public_identity()
        teacher_identity = teacher.public_identity()
        model_root = Path(str(config["model"]["local_path"])).resolve()
        tokenizer = source_runtime.authenticate_tokenizer_snapshot(model_root)
        tokenizer_identity = tokenizer.public_identity()
        implementation = _implementation_identity()
        launcher_identity = validate_launcher_context(
            config,
            config_sha256=config_sha256,
            run_id=run_id,
            mode=mode,
            teacher_binding_sha256=teacher.binding_sha256,
            implementation_set_sha256=str(implementation["set_sha256"]),
            lease_path=lease_path,
            lease_sha256=lease_sha256,
            attestation_path=attestation_path,
            attestation_sha256=attestation_sha256,
        )
        smoke_gate: Mapping[str, object] | None = None
        if mode == "full":
            assert smoke_receipt is not None
            assert smoke_receipt_sha256 is not None
            smoke_gate = _validate_smoke_gate(
                smoke_receipt,
                smoke_receipt_sha256,
                config_sha256=config_sha256,
                source_identity=source_identity,
                teacher_identity=teacher_identity,
                tokenizer_identity=tokenizer_identity,
            )
        datasets = build_runtime_datasets(source, teacher, tokenizer)
        phase = "smoke" if mode == "smoke_only" else "full"
        phase_receipts: list[dict[str, Any]] = []
        phase_summaries: list[dict[str, object]] = []
        adapters: dict[str, object] = {}
        model_file_hashes: Mapping[str, str]
        with _private_model_snapshot(config, run_id) as (
            model_snapshot,
            model_file_hashes,
        ):
            for arm in contract.TRAINED_ARMS:
                steps = (
                    int(config["training"]["smoke_steps_per_arm"])
                    if phase == "smoke"
                    else contract.FULL_STEPS[arm]
                )
                asset = contract.ARM_SOURCE_ASSET[arm]
                phase_root = staging / "phases" / arm
                phase_root.mkdir(parents=True, exist_ok=False)
                phase_root_identity = _directory_identity(
                    phase_root,
                    code="runtime_phase_root_identity_invalid",
                )
                try:
                    receipt = _execute_phase(
                        config,
                        config_sha256=config_sha256,
                        run_id=run_id,
                        mode=mode,
                        arm=arm,
                        phase=phase,
                        steps=steps,
                        examples=datasets.serialized_assets[asset],
                        model_snapshot=model_snapshot,
                        model_file_hashes=model_file_hashes,
                        phase_root=phase_root,
                        phase_root_identity=phase_root_identity,
                        progress_root=staging / "progress",
                        source_identity=source_identity,
                        teacher_identity=teacher_identity,
                        tokenizer_identity=tokenizer_identity,
                    )
                except BaseException as exc:
                    raise PhaseExecutionError(arm, phase, exc) from exc
                last_completed = arm
                phase_receipts.append(receipt)
                phase_summaries.append(_phase_summary(staging, arm, phase, receipt))
                if receipt["adapter"] is not None:
                    adapters[arm] = receipt["adapter"]
        if (mode == "smoke_only" and adapters) or (
            mode == "full" and set(adapters) != set(contract.TRAINED_ARMS)
        ):
            raise MultiArmRuntimeError("runtime_adapter_publish_inventory_invalid")
        source_runtime.terminal_recheck_source(source)
        terminal_recheck_teacher(teacher)
        if _implementation_identity() != implementation:
            raise MultiArmRuntimeError("runtime_implementation_identity_changed")
        training_lineage = {
            "schema_version": LINEAGE_VERSION,
            "run_id": run_id,
            "mode": mode,
            "config_sha256": config_sha256,
            "source": source_identity,
            "teacher_final": teacher_identity,
            "tokenizer": tokenizer_identity,
            "source_record_order_sha256": (datasets.source_record_order_sha256),
            "teacher_target_projection_sha256": (
                datasets.teacher_target_projection_sha256
            ),
            "serialized_inventory_sha256": (datasets.serialized_inventory_sha256),
            "optimizer_records": 3520,
            "main_records": 3440,
            "router_train_records": 80,
            "eval_records_consumed_by_optimizer": 0,
            "producer_original_targets_used": False,
            "phase_receipts": phase_summaries,
            "physical_kv_validated": False,
            "full_generation_kv_shared": False,
            "sample_bodies_included": False,
            "raw_token_ids_included": False,
        }
        lineage_sha, lineage_sidecar_sha = _write_canonical(
            staging / "training_lineage.json",
            training_lineage,
            parent_identity=staging_identity,
        )
        model_identity = {
            "base_files": dict(model_file_hashes),
            "base_file_inventory_sha256": _canonical_sha256(dict(model_file_hashes)),
            "phase_base_hashes": {
                summary["arm"]: summary["base_parameter_sha256"]
                for summary in phase_summaries
            },
            "phase_q8_scb_hashes": {
                summary["arm"]: summary["q8_scb_sha256"] for summary in phase_summaries
            },
            "all_phase_base_hashes_equal": (
                len({summary["base_parameter_sha256"] for summary in phase_summaries})
                == 1
            ),
            "all_phase_q8_scb_hashes_equal": (
                len({summary["q8_scb_sha256"] for summary in phase_summaries}) == 1
            ),
            "frozen_q8_base": True,
            "base_o_proj_always_frozen": True,
        }
        if (
            not model_identity["all_phase_base_hashes_equal"]
            or not model_identity["all_phase_q8_scb_hashes_equal"]
        ):
            raise MultiArmRuntimeError("runtime_cross_phase_base_identity_drift")
        run_receipt = {
            "schema_version": RUN_RECEIPT_VERSION,
            "status": "passed",
            "run_id": run_id,
            "mode": mode,
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "sha256": config_sha256,
                "implementation": implementation,
            },
            "source": source_identity,
            "teacher_final": teacher_identity,
            "tokenizer": tokenizer_identity,
            "model": model_identity,
            "phase_order": list(contract.TRAINED_ARMS),
            "phase_receipts": phase_summaries,
            "adapters": adapters,
            "training_lineage": {
                "path": "training_lineage.json",
                "sha256": lineage_sha,
                "sidecar_sha256": lineage_sidecar_sha,
            },
            "launcher": {
                **launcher_identity,
                "smoke_gate": smoke_gate,
            },
            "claims": {
                "diagnostic_only": True,
                "proxy_only": True,
                "formal": False,
                "release": False,
                "sample_bodies_included": False,
                "raw_token_ids_included": False,
                "producer_original_targets_used": False,
                "physical_kv_validated": False,
                "full_generation_kv_shared": False,
            },
        }
        _write_canonical(
            staging / "run_receipt.json",
            run_receipt,
            schema_path=RUN_SCHEMA_PATH,
            parent_identity=staging_identity,
        )
        source_runtime.terminal_recheck_source(source)
        terminal_recheck_teacher(teacher)
        if _implementation_identity() != implementation:
            raise MultiArmRuntimeError(
                "runtime_implementation_identity_changed_before_publish"
            )
        _move_directory_create_once(
            staging,
            published,
            owned_identity=staging_identity,
            code="runtime_publish_destination_exists_or_move_failed",
        )
        return published, run_receipt
    except BaseException as exc:
        try:
            _assert_owned_directory(
                staging_identity,
                path=staging,
                code="runtime_failed_staging_identity_changed",
            )
        except MultiArmRuntimeError:
            pass
        else:
            _publish_failure(
                staging,
                staging_identity=staging_identity,
                run_id=run_id,
                mode=mode,
                error=exc,
                last_completed_arm=last_completed,
                source_authenticated=source_authenticated,
                teacher_authenticated=teacher_authenticated,
            )
            failed = output_root / f".failed-{run_id}-{uuid.uuid4().hex}"
            try:
                _move_directory_create_once(
                    staging,
                    failed,
                    owned_identity=staging_identity,
                    code="runtime_failed_publish_destination_exists_or_move_failed",
                )
            except MultiArmRuntimeError:
                # Leave the original staging tree in place.  It is safer
                # failure evidence than a path-based cleanup or replacement.
                pass
        raise


def _validate_only(
    config: Mapping[str, Any],
    config_sha256: str,
    producer_repository: str | Path | None = None,
) -> dict[str, object]:
    teacher = _mapping(config["teacher_final"], "runtime_teacher_invalid")
    producer, _binding_schema_sha, _record_schema_sha = (
        _released_teacher_schema_identities(config, producer_repository)
    )
    return {
        "schema_version": CONFIG_VERSION,
        "status": "passed",
        "operation": "validate",
        "config_sha256": config_sha256,
        "arms": list(contract.TRAINED_ARMS),
        "teacher_final_required_for_gpu_execution": True,
        "teacher_binding_v2_release_status": teacher["binding_schema_release_status"],
        "teacher_record_v2_release_status": teacher["record_schema_release_status"],
        "teacher_schema_physical_bytes_bound": True,
        "rollover_producer_commit": producer.commit,
        "rollover_producer_parent": producer.parent,
        "rollover_producer_tree": producer.tree,
        "rollover_v3_config_sha256": producer.physical_identity_sha256["config"],
        "rollover_v3_execute_implementation_sha256": (
            producer.execute_implementation_sha256
        ),
        "rollover_v3_release_implementation_sha256": (
            producer.release_implementation_sha256
        ),
        "rollover_binding_schema_sha256": producer.schema_sha256["binding"],
        "teacher_final_physical_preflight_required": True,
        "gpu_execution_ready": True,
        "gpu_execution_blocker": None,
        "producer_original_target_fallback": False,
        "gpu_requested": False,
        "model_loaded": False,
        "provider_requests": 0,
        "sample_bodies_read": False,
        "raw_token_ids_read": False,
        "default_model_free": True,
        "formal": False,
    }


def preflight_teacher_final(
    config: Mapping[str, Any],
    *,
    binding_path: str | Path,
    binding_sha256: str,
    producer_repository: str | Path | None = None,
) -> dict[str, object]:
    """Authenticate every Teacher FINAL physical identity without parsing rows."""

    producer, _binding_schema_sha, record_schema_sha = (
        _released_teacher_schema_identities(config, producer_repository)
    )
    teacher, _binding = authenticate_teacher_final(
        binding_path,
        expected_binding_sha256=binding_sha256,
        producer=producer,
        expected_record_schema_sha256=record_schema_sha,
    )
    terminal_recheck_teacher(teacher)
    return {
        "status": "passed",
        "operation": "teacher_final_physical_preflight",
        "teacher_final": teacher.public_identity(),
        "rollover_producer": {
            "commit": producer.commit,
            "parent": producer.parent,
            "tree": producer.tree,
            "config_sha256": producer.physical_identity_sha256["config"],
            "execute_implementation_sha256": (producer.execute_implementation_sha256),
            "release_implementation_sha256": (producer.release_implementation_sha256),
            "binding_schema_sha256": producer.schema_sha256["binding"],
        },
        "all_binding_manifest_receipt_attestation_shard_identities_authenticated": True,
        "sample_bodies_parsed": False,
        "raw_token_ids_read": False,
        "gpu_requested": False,
        "model_loaded": False,
        "provider_requests": 0,
        "formal": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--validate", action="store_true")
    operations.add_argument("--execute-smoke-only", action="store_true")
    operations.add_argument("--execute-full", action="store_true")
    operations.add_argument("--preflight-teacher-final", action="store_true")
    operations.add_argument("--emit-implementation-identity", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--producer-repository")
    parser.add_argument("--teacher-final-binding")
    parser.add_argument("--teacher-final-binding-sha256")
    parser.add_argument("--execution-lease")
    parser.add_argument("--execution-lease-sha256")
    parser.add_argument("--gpu-attestation")
    parser.add_argument("--gpu-attestation-sha256")
    parser.add_argument("--smoke-run-receipt")
    parser.add_argument("--smoke-run-receipt-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_sha = load_config(args.config)
        if args.emit_implementation_identity:
            print(
                json.dumps(
                    _implementation_identity(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.preflight_teacher_final:
            if not args.teacher_final_binding or not args.teacher_final_binding_sha256:
                raise MultiArmRuntimeError("teacher_preflight_inputs_missing")
            print(
                json.dumps(
                    preflight_teacher_final(
                        config,
                        binding_path=str(args.teacher_final_binding),
                        binding_sha256=str(args.teacher_final_binding_sha256),
                        producer_repository=args.producer_repository,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if not args.execute_smoke_only and not args.execute_full:
            print(
                json.dumps(
                    _validate_only(
                        config,
                        config_sha,
                        producer_repository=args.producer_repository,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        mode = "smoke_only" if args.execute_smoke_only else "full"
        required = {
            "run_id": args.run_id,
            "teacher_final_binding": args.teacher_final_binding,
            "teacher_final_binding_sha256": (args.teacher_final_binding_sha256),
            "execution_lease": args.execution_lease,
            "execution_lease_sha256": args.execution_lease_sha256,
            "gpu_attestation": args.gpu_attestation,
            "gpu_attestation_sha256": args.gpu_attestation_sha256,
        }
        if any(not value for value in required.values()):
            raise MultiArmRuntimeError("runtime_execution_inputs_missing")
        published, receipt = execute_runtime(
            config,
            config_sha256=config_sha,
            mode=mode,
            run_id=str(args.run_id),
            producer_repository=args.producer_repository,
            teacher_binding=str(args.teacher_final_binding),
            teacher_binding_sha256=str(args.teacher_final_binding_sha256),
            lease_path=str(args.execution_lease),
            lease_sha256=str(args.execution_lease_sha256),
            attestation_path=str(args.gpu_attestation),
            attestation_sha256=str(args.gpu_attestation_sha256),
            smoke_receipt=args.smoke_run_receipt,
            smoke_receipt_sha256=args.smoke_run_receipt_sha256,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "run_id": receipt["run_id"],
                    "mode": receipt["mode"],
                    "published_path": str(published),
                    "sample_bodies_included": False,
                    "raw_token_ids_included": False,
                    "formal": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_code": re.sub(
                        r"[^A-Za-z0-9_.:-]",
                        "_",
                        (str(exc).splitlines()[0] if str(exc) else type(exc).__name__),
                    )[:200],
                    "sample_body_included": False,
                    "raw_token_ids_included": False,
                    "automatic_retry": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
