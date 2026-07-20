"""Authenticated, body-free long-context token inventories.

The preflight consumes a pinned TaskBoard projector artifact in two passes.
The first pass keeps only identifiers needed to prove bundle/split/role
invariants.  Only after those invariants hold does the second pass serialize
the ordered, causally allowed segment chain into an exact local tokenizer.
Current, future, and forbidden blocks are never passed to the serializer or
token counter and no source body is persisted in the resulting artifact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Protocol
from uuid import uuid4

import yaml

from anchor_mvp.swebench import taskboard_projector as _projector


CONFIG_SCHEMA = "anchor.long-context-token-inventory-config.v1"
PRODUCER_VERSION = "anchor.long-context-token-inventory-producer.v1"
RECORD_SCHEMA = "anchor.long-context-token-inventory.v1"
MANIFEST_SCHEMA = "anchor.long-context-token-inventory-manifest.v1"
PROJECTOR_MANIFEST_SCHEMA = "anchor.swebench-taskboard-projector-manifest.v2"
PROJECTOR_SIDECAR_SCHEMA = "anchor.swebench-taskboard-sidecar.v2"
SEGMENT_PLAN_SCHEMA = "anchor.hierarchical-task-kv-segment-plan.v1"

FIXED_FILES = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)

BUCKETS = (
    ("le_8k", 8_192, "measurement_candidate"),
    ("le_16k", 16_384, "measurement_candidate"),
    ("le_32k", 32_768, "measurement_candidate"),
    ("le_64k", 65_536, "measurement_candidate"),
    ("le_128k", 131_072, "measurement_candidate"),
    ("le_256k", 262_144, "capability_only"),
    ("le_1m", 1_048_576, "research_only_blocked"),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_RECORD_ID_RE = re.compile(r"^long-context-token-inventory-v1:[0-9a-f]{64}$")
_DENIED_OUTPUT_KEYS = {
    "answer",
    "content",
    "heldout",
    "messages",
    "preview",
    "prompt",
    "task_board",
    "token_ids",
    "training_record",
}
_MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}


class LongContextPreflightError(RuntimeError):
    """A fixed, body-free operator error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise LongContextPreflightError(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], code: str) -> None:
    if set(value) != keys:
        _fail(code)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


@dataclass(frozen=True)
class _BytesSnapshot:
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


def _read_snapshot(path: Path, code: str, *, max_bytes: int) -> _BytesSnapshot:
    try:
        if path.is_symlink():
            _fail(code)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                _fail(code)
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except LongContextPreflightError:
        raise
    except OSError as exc:
        raise LongContextPreflightError(code) from exc
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    current_identity = _stat_identity(current)
    if (
        before_identity != after_identity
        or after_identity != current_identity
        or len(data) != after.st_size
    ):
        _fail(code)
    return _BytesSnapshot(
        data=data,
        sha256=_sha256_bytes(data),
        size=len(data),
        identity=after_identity,
    )


def _json(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LongContextPreflightError(code) from exc
    return _mapping(value, code)


def _safe_relative(root: Path, value: object, code: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(code)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        _fail(code)
    path = root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise LongContextPreflightError(code) from exc
    if path.is_symlink() or resolved != path.absolute():
        _fail(code)
    return path


@dataclass(frozen=True)
class LongContextTokenInventoryConfig:
    """Authenticated policy for the body-free inventory producer."""

    path: Path
    sha256: str
    record_schema_sha256: str
    manifest_schema_sha256: str
    reserved_output_tokens: int
    max_input_records: int
    max_input_file_bytes: int
    max_output_file_bytes: int
    max_tokenizer_asset_bytes: int
    raw: Mapping[str, Any]

    @classmethod
    def load(
        cls, value: str | Path
    ) -> tuple["LongContextTokenInventoryConfig", dict[Path, _BytesSnapshot]]:
        raw_path = Path(value).expanduser()
        if raw_path.is_symlink():
            _fail("long_context_config_invalid")
        path = raw_path.resolve()
        if path != raw_path.absolute():
            _fail("long_context_config_invalid")
        snapshot = _read_snapshot(
            path, "long_context_config_invalid", max_bytes=1_000_000
        )
        try:
            raw = yaml.safe_load(snapshot.data.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise LongContextPreflightError("long_context_config_invalid") from exc
        config = _mapping(raw, "long_context_config_invalid")
        _exact_keys(
            config,
            {
                "schema_version",
                "producer_version",
                "input_contract",
                "output_contract",
                "model_contract",
                "tokenization",
                "buckets",
                "limits",
                "safety",
            },
            "long_context_config_invalid",
        )
        input_contract = _mapping(
            config.get("input_contract"), "long_context_config_invalid"
        )
        output_contract = _mapping(
            config.get("output_contract"), "long_context_config_invalid"
        )
        model_contract = _mapping(
            config.get("model_contract"), "long_context_config_invalid"
        )
        tokenization = _mapping(
            config.get("tokenization"), "long_context_config_invalid"
        )
        limits = _mapping(config.get("limits"), "long_context_config_invalid")
        safety = _mapping(config.get("safety"), "long_context_config_invalid")
        buckets = config.get("buckets")
        expected_buckets: dict[str, dict[str, Any]] = {}
        lower_bound = 0
        for name, upper_bound, gate in BUCKETS:
            expected_buckets[name] = {
                "lower_bound_exclusive": lower_bound,
                "upper_bound_inclusive": upper_bound,
                "gate": gate,
            }
            lower_bound = upper_bound
        expected_buckets["gt_1m"] = {
            "lower_bound_exclusive": 1_048_576,
            "upper_bound_inclusive": "unbounded",
            "gate": "reject",
        }
        _exact_keys(
            input_contract,
            {
                "projector_manifest_schema_version",
                "projector_sidecar_schema_version",
                "segment_plan_schema_version",
                "require_manifest",
                "require_manifest_sha256_sidecar",
                "fixed_files",
            },
            "long_context_config_invalid",
        )
        _exact_keys(
            output_contract,
            {
                "record_schema_version",
                "manifest_schema_version",
                "fixed_files",
                "mandatory_manifest_sha256_sidecar",
                "approximate_inventory_allowed",
                "null_inventory_allowed",
                "inventory_mode_policy",
                "bucket_basis",
                "capability_validated",
            },
            "long_context_config_invalid",
        )
        _exact_keys(
            model_contract,
            {
                "source",
                "model_family",
                "parameter_scale",
                "text_layers",
                "sliding_attention_layers",
                "global_attention_layers",
                "sliding_window",
                "max_position_embeddings",
                "architecture_verified_by_preflight",
            },
            "long_context_config_invalid",
        )
        _exact_keys(
            tokenization,
            {
                "allowed_backends",
                "required_identity_fields",
                "tokenizer_label_source",
                "all_identity_fields_required",
                "identity_null_allowed",
                "network_access_allowed",
                "synthetic_backend_requires_explicit_declaration",
                "exact_token_counts_required",
                "reserved_output_tokens",
                "total_tokens_formula",
                "scope_attribution_formula",
                "cache_identity_status",
                "reuse_savings_tokens",
            },
            "long_context_config_invalid",
        )
        _exact_keys(
            limits,
            {
                "max_input_records",
                "max_input_file_bytes",
                "max_output_file_bytes",
                "max_tokenizer_asset_bytes",
                "split_group_key",
                "task_id_cross_binding_key",
                "task_id_cross_binding_transform",
                "all_five_role_views_same_split",
                "split_before_augmentation",
                "forbidden_current_future_excluded_before_serialization",
                "reject_total_tokens_above",
            },
            "long_context_config_invalid",
        )
        _exact_keys(
            safety,
            {
                "source_line_bytes_snapshot_required",
                "provider_requests",
                "canonical_gold_written",
                "evaluation_status",
                "quality_validated",
                "allocation_validated",
                "execution_authorized",
            },
            "long_context_config_invalid",
        )
        expected_files = [item[0] for item in FIXED_FILES]
        if (
            config.get("schema_version") != CONFIG_SCHEMA
            or config.get("producer_version") != PRODUCER_VERSION
            or input_contract
            != {
                "projector_manifest_schema_version": PROJECTOR_MANIFEST_SCHEMA,
                "projector_sidecar_schema_version": PROJECTOR_SIDECAR_SCHEMA,
                "segment_plan_schema_version": SEGMENT_PLAN_SCHEMA,
                "require_manifest": True,
                "require_manifest_sha256_sidecar": True,
                "fixed_files": expected_files,
            }
            or output_contract
            != {
                "record_schema_version": RECORD_SCHEMA,
                "manifest_schema_version": MANIFEST_SCHEMA,
                "fixed_files": expected_files,
                "mandatory_manifest_sha256_sidecar": True,
                "approximate_inventory_allowed": False,
                "null_inventory_allowed": False,
                "inventory_mode_policy": {
                    "synthetic_fixture": {
                        "required_backend": "explicit_synthetic_tokenizer",
                        "status": "synthetic_fixture_inventory_ready",
                        "claim_scope": "synthetic_fixture_contract_only",
                        "target_model_tokenizer_match": "not_applicable",
                        "synthetic_fixture_only": True,
                    },
                    "local_exact_tokenizer": {
                        "required_backend": "local_offline_tokenizer",
                        "status": "exact_token_inventory_ready",
                        "claim_scope": "exact_bound_tokenizer_inventory_only",
                        "target_model_tokenizer_match": (
                            "consumer_verification_required"
                        ),
                        "synthetic_fixture_only": False,
                    },
                },
                "bucket_basis": "bound_tokenizer_total_tokens",
                "capability_validated": False,
            }
            or model_contract
            != {
                "source": "caller_supplied_metadata_only",
                "model_family": "gemma4",
                "parameter_scale": "12B",
                "text_layers": 48,
                "sliding_attention_layers": 40,
                "global_attention_layers": 8,
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "architecture_verified_by_preflight": False,
            }
            or tokenization
            != {
                "allowed_backends": [
                    "local_offline_tokenizer",
                    "explicit_synthetic_tokenizer",
                ],
                "required_identity_fields": [
                    "tokenizer_id",
                    "tokenizer_revision",
                    "tokenizer_assets_sha256",
                    "tokenizer_runtime_sha256",
                    "chat_template_sha256",
                    "serialization_policy_sha256",
                    "special_token_policy_sha256",
                ],
                "tokenizer_label_source": "caller_supplied_and_hash_bound",
                "all_identity_fields_required": True,
                "identity_null_allowed": False,
                "network_access_allowed": False,
                "synthetic_backend_requires_explicit_declaration": True,
                "exact_token_counts_required": True,
                "reserved_output_tokens": 4096,
                "total_tokens_formula": "input_tokens_plus_reserved_output_tokens",
                "scope_attribution_formula": (
                    "shared_prefix_input_tokens_plus_private_delta_input_tokens_"
                    "equals_input_tokens"
                ),
                "cache_identity_status": "identity_unbound",
                "reuse_savings_tokens": 0,
            }
            or buckets != expected_buckets
            or limits.get("split_group_key") != "task_bundle_sha256"
            or limits.get("task_id_cross_binding_key") != "task_id_sha256"
            or limits.get("task_id_cross_binding_transform")
            != "sha256_utf8_training_record_task_board_task_id"
            or limits.get("all_five_role_views_same_split") is not True
            or limits.get("split_before_augmentation") is not True
            or limits.get("forbidden_current_future_excluded_before_serialization")
            is not True
            or limits.get("reject_total_tokens_above") != 1_048_576
            or any(
                not _positive_int(limits.get(key))
                for key in (
                    "max_input_records",
                    "max_input_file_bytes",
                    "max_output_file_bytes",
                    "max_tokenizer_asset_bytes",
                )
            )
            or safety
            != {
                "source_line_bytes_snapshot_required": True,
                "provider_requests": 0,
                "canonical_gold_written": False,
                "evaluation_status": "not_evaluated",
                "quality_validated": False,
                "allocation_validated": False,
                "execution_authorized": False,
            }
        ):
            _fail("long_context_config_invalid")
        if int(model_contract["sliding_attention_layers"]) + int(
            model_contract["global_attention_layers"]
        ) != int(model_contract["text_layers"]):
            _fail("long_context_config_invalid")
        if int(tokenization["reserved_output_tokens"]) >= BUCKETS[-1][1]:
            _fail("long_context_config_invalid")

        record_schema_path = (
            path.parent / "swebench_long_context_preflight_sidecar.schema.json"
        )
        manifest_schema_path = (
            path.parent / "swebench_long_context_preflight_manifest.schema.json"
        )
        record_schema_snapshot = _read_snapshot(
            record_schema_path,
            "long_context_record_schema_invalid",
            max_bytes=1_000_000,
        )
        manifest_schema_snapshot = _read_snapshot(
            manifest_schema_path,
            "long_context_manifest_schema_invalid",
            max_bytes=1_000_000,
        )
        if (
            _json(record_schema_snapshot, "long_context_record_schema_invalid")
            .get("properties", {})
            .get("schema_version", {})
            .get("const")
            != RECORD_SCHEMA
            or _json(manifest_schema_snapshot, "long_context_manifest_schema_invalid")
            .get("properties", {})
            .get("schema_version", {})
            .get("const")
            != MANIFEST_SCHEMA
        ):
            _fail("long_context_schema_invalid")
        loaded = cls(
            path=path,
            sha256=snapshot.sha256,
            record_schema_sha256=record_schema_snapshot.sha256,
            manifest_schema_sha256=manifest_schema_snapshot.sha256,
            reserved_output_tokens=int(tokenization["reserved_output_tokens"]),
            max_input_records=int(limits["max_input_records"]),
            max_input_file_bytes=int(limits["max_input_file_bytes"]),
            max_output_file_bytes=int(limits["max_output_file_bytes"]),
            max_tokenizer_asset_bytes=int(limits["max_tokenizer_asset_bytes"]),
            raw=config,
        )
        return loaded, {
            path: snapshot,
            record_schema_path: record_schema_snapshot,
            manifest_schema_path: manifest_schema_snapshot,
        }


class ExactTokenCounter(Protocol):
    """A local exact counter bound to immutable assets and serializer policy."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def binding_sha256(self) -> str: ...

    def count(self, ordered_segments: Sequence[str]) -> int: ...

    def verify_unchanged(self) -> None: ...


_SERIALIZATION_SEPARATOR = "\n<|anchor_segment_boundary|>\n"
_SYNTHETIC_TEMPLATE = "<|bos|><|user|>{body}<|assistant|>"
_SERIALIZATION_POLICY = {
    "version": "ordered_segments_single_user_turn.v1",
    "separator_sha256": _sha256_bytes(_SERIALIZATION_SEPARATOR.encode("utf-8")),
    "add_generation_marker": True,
}
_SPECIAL_TOKEN_POLICY = {
    "version": "chat_template_add_generation_marker.v1",
    "padding": False,
    "truncation": False,
}


class SyntheticFixtureTokenCounter:
    """Exact only for the tiny deterministic fixture tokenizer contract."""

    def __init__(self) -> None:
        assets = {
            "algorithm": "one_token_per_serialized_utf8_byte.v1",
            "template_sha256": _sha256_bytes(_SYNTHETIC_TEMPLATE.encode("utf-8")),
        }
        base = {
            "backend": "explicit_synthetic_tokenizer",
            "tokenizer_id": "anchor.synthetic-fixture-utf8-byte",
            "tokenizer_revision": "v1",
            "tokenizer_label_source": "caller_supplied_and_hash_bound",
            "tokenizer_assets_sha256": _sha256_value(assets),
            "tokenizer_runtime_sha256": _sha256_value(
                {"algorithm": assets["algorithm"]}
            ),
            "chat_template_sha256": assets["template_sha256"],
            "serialization_policy_sha256": _sha256_value(_SERIALIZATION_POLICY),
            "special_token_policy_sha256": _sha256_value(_SPECIAL_TOKEN_POLICY),
            "network_access": False,
            "exact_token_counts": True,
            "synthetic_fixture_only": True,
        }
        self._binding_sha256 = _sha256_value(base)
        self._metadata = base

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def binding_sha256(self) -> str:
        return self._binding_sha256

    def count(self, ordered_segments: Sequence[str]) -> int:
        if not ordered_segments or any(
            not isinstance(item, str) for item in ordered_segments
        ):
            _fail("long_context_tokenizer_input_invalid")
        body = _SERIALIZATION_SEPARATOR.join(ordered_segments)
        serialized = _SYNTHETIC_TEMPLATE.format(body=body).encode("utf-8")
        return len(serialized)

    def verify_unchanged(self) -> None:
        return None


@dataclass(frozen=True)
class _FileFingerprint:
    path: Path
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


def _fingerprint_file(path: Path, code: str, *, max_bytes: int) -> _FileFingerprint:
    try:
        if path.is_symlink():
            _fail(code)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                _fail(code)
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except LongContextPreflightError:
        raise
    except OSError as exc:
        raise LongContextPreflightError(code) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(current)
        or size != after.st_size
    ):
        _fail(code)
    return _FileFingerprint(
        path=path,
        sha256=digest.hexdigest(),
        size=size,
        identity=_stat_identity(after),
    )


class LocalTransformersTokenCounter:
    """Offline Hugging Face tokenizer adapter; model weights are never loaded."""

    def __init__(
        self,
        tokenizer_dir: str | Path,
        *,
        tokenizer_id: str,
        tokenizer_revision: str,
        max_asset_bytes: int,
    ) -> None:
        raw_root = Path(tokenizer_dir).expanduser()
        if raw_root.is_symlink():
            _fail("long_context_tokenizer_assets_invalid")
        root = raw_root.resolve()
        if (
            not root.is_dir()
            or root.is_symlink()
            or root != raw_root.absolute()
            or not _is_identifier(tokenizer_id)
            or not _is_identifier(tokenizer_revision)
        ):
            _fail("long_context_tokenizer_assets_invalid")
        all_paths = tuple(root.rglob("*"))
        if any(path.is_symlink() for path in all_paths):
            _fail("long_context_tokenizer_assets_invalid")
        files = sorted(
            (path for path in all_paths if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        if not files or any(
            path.suffix.casefold() in _MODEL_WEIGHT_SUFFIXES
            or "model.safetensors" in path.name.casefold()
            for path in files
        ):
            _fail("long_context_tokenizer_assets_invalid")
        fingerprints: list[_FileFingerprint] = []
        remaining = max_asset_bytes
        for path in files:
            fingerprint = _fingerprint_file(
                path,
                "long_context_tokenizer_assets_invalid",
                max_bytes=remaining,
            )
            fingerprints.append(fingerprint)
            remaining -= fingerprint.size
            if remaining < 0:
                _fail("long_context_tokenizer_assets_invalid")
        asset_rows = [
            {
                "path": item.path.relative_to(root).as_posix(),
                "sha256": item.sha256,
                "bytes": item.size,
            }
            for item in fingerprints
        ]
        trusted_directory = tempfile.TemporaryDirectory(
            prefix="anchor-long-context-tokenizer-"
        )
        trusted_root = Path(trusted_directory.name)
        try:
            for fingerprint in fingerprints:
                relative = fingerprint.path.relative_to(root)
                destination = trusted_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fingerprint.path, destination)
                copied = _fingerprint_file(
                    destination,
                    "long_context_tokenizer_assets_changed",
                    max_bytes=fingerprint.size,
                )
                if (
                    copied.sha256 != fingerprint.sha256
                    or copied.size != fingerprint.size
                ):
                    _fail("long_context_tokenizer_assets_changed")
        except Exception:
            trusted_directory.cleanup()
            raise
        try:
            import tokenizers  # type: ignore[import-not-found]
            import transformers  # type: ignore[import-not-found]

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                trusted_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            trusted_directory.cleanup()
            raise LongContextPreflightError(
                "long_context_local_tokenizer_unavailable"
            ) from exc
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            trusted_directory.cleanup()
            _fail("long_context_chat_template_unavailable")
        runtime = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "transformers": str(transformers.__version__),
            "tokenizers": str(tokenizers.__version__),
            "adapter": "local_transformers_apply_chat_template.v1",
        }
        base = {
            "backend": "local_offline_tokenizer",
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_label_source": "caller_supplied_and_hash_bound",
            "tokenizer_assets_sha256": _sha256_value(asset_rows),
            "tokenizer_runtime_sha256": _sha256_value(runtime),
            "chat_template_sha256": _sha256_bytes(template.encode("utf-8")),
            "serialization_policy_sha256": _sha256_value(_SERIALIZATION_POLICY),
            "special_token_policy_sha256": _sha256_value(_SPECIAL_TOKEN_POLICY),
            "network_access": False,
            "exact_token_counts": True,
            "synthetic_fixture_only": False,
        }
        self._root = root
        self._trusted_directory = trusted_directory
        self._fingerprints = tuple(fingerprints)
        self._tokenizer = tokenizer
        self._binding_sha256 = _sha256_value(base)
        self._metadata = base

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def binding_sha256(self) -> str:
        return self._binding_sha256

    def count(self, ordered_segments: Sequence[str]) -> int:
        if not ordered_segments or any(
            not isinstance(item, str) for item in ordered_segments
        ):
            _fail("long_context_tokenizer_input_invalid")
        body = _SERIALIZATION_SEPARATOR.join(ordered_segments)
        try:
            ids = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=True,
                add_generation_prompt=True,
                padding=False,
                truncation=False,
            )
        except Exception as exc:
            raise LongContextPreflightError("long_context_tokenization_failed") from exc
        if not isinstance(ids, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in ids
        ):
            _fail("long_context_tokenization_failed")
        return len(ids)

    def verify_unchanged(self) -> None:
        for expected in self._fingerprints:
            current = _fingerprint_file(
                expected.path,
                "long_context_tokenizer_assets_changed",
                max_bytes=expected.size,
            )
            if current.sha256 != expected.sha256 or current.size != expected.size:
                _fail("long_context_tokenizer_assets_changed")


@dataclass(frozen=True)
class _SanitizedSourceRow:
    path: str
    line_number: int
    source_line_sha256: str
    source_partition_sha256: str
    record_id: str
    pair_id: str
    source_gold_record_id: str
    source_gold_sha256: str
    task_bundle_sha256: str
    task_id: str
    task_id_sha256: str
    split: str
    variant: str
    stage: str
    expert: str
    language: str
    segment_plan_sha256: str
    segment_ids: tuple[str, ...]
    non_private_segment_ids: tuple[str, ...]
    source_block_ids: tuple[str, ...]
    private_delta_segment_count: int
    terminal_prefix_lineage_sha256: str


def _iter_jsonl(
    snapshot: _BytesSnapshot, code: str
) -> Iterable[tuple[int, bytes, Mapping[str, Any]]]:
    lines = snapshot.data.splitlines()
    if not lines or snapshot.data[-1:] != b"\n":
        _fail(code)
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(code)
        try:
            value = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LongContextPreflightError(code) from exc
        yield index, line, _mapping(value, code)


def _input_manifest_expected_producer(
    config: _projector.TaskBoardProjectorConfig,
) -> dict[str, Any]:
    return {
        "name": "anchor.swebench-taskboard-projector",
        "projector_version": config.projector_version,
        "config_sha256": config.sha256,
        "sidecar_schema_sha256": config.sidecar_schema_sha256,
        "segment_plan_schema_version": config.segment_plan_schema,
        "segment_plan_schema_sha256": config.segment_plan_schema_sha256,
        "manifest_schema_sha256": config.manifest_schema_sha256,
        "record_schema_version": config.record_schema,
    }


def _load_projector_artifact(
    config: LongContextTokenInventoryConfig,
    directory: str | Path,
    expected_manifest_sha256: str,
    inventory: dict[Path, _BytesSnapshot],
) -> tuple[
    Path,
    Mapping[str, Any],
    Mapping[tuple[str, str], _BytesSnapshot],
    _projector.TaskBoardProjectorConfig,
    str,
]:
    raw_root = Path(directory).expanduser()
    if raw_root.is_symlink():
        _fail("long_context_projector_artifact_invalid")
    root = raw_root.resolve()
    if (
        not _is_sha256(expected_manifest_sha256)
        or not root.is_dir()
        or root.is_symlink()
        or root != raw_root.absolute()
    ):
        _fail("long_context_projector_artifact_invalid")
    manifest_path = root / "manifest.json"
    manifest_snapshot = _read_snapshot(
        manifest_path,
        "long_context_projector_manifest_invalid",
        max_bytes=config.max_input_file_bytes,
    )
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        _fail("long_context_projector_manifest_invalid")
    sidecar_path = root / "manifest.json.sha256"
    sidecar_snapshot = _read_snapshot(
        sidecar_path,
        "long_context_projector_manifest_sidecar_invalid",
        max_bytes=256,
    )
    expected_sidecar = f"{expected_manifest_sha256}  manifest.json\n".encode("ascii")
    if sidecar_snapshot.data != expected_sidecar:
        _fail("long_context_projector_manifest_sidecar_invalid")
    manifest = _json(manifest_snapshot, "long_context_projector_manifest_invalid")
    expected_root_keys = {
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
    }
    _exact_keys(manifest, expected_root_keys, "long_context_projector_manifest_invalid")
    projector_config_path = config.path.parent / "swebench_taskboard_projector_v2.yaml"
    try:
        projector_config = _projector.TaskBoardProjectorConfig.load(
            projector_config_path
        )
    except _projector.TaskBoardProjectorError as exc:
        raise LongContextPreflightError(
            "long_context_projector_policy_invalid"
        ) from exc
    policy_paths = (
        projector_config_path,
        config.path.parent / "taskboard_projector_sidecar.schema.json",
        config.path.parent / "hierarchical_task_kv_segment_plan.schema.json",
        config.path.parent / "taskboard_projector_manifest.schema.json",
    )
    for policy_path in policy_paths:
        inventory[policy_path] = _read_snapshot(
            policy_path,
            "long_context_projector_policy_invalid",
            max_bytes=1_000_000,
        )
    projector_input = _mapping(
        manifest.get("input"), "long_context_projector_manifest_invalid"
    )
    _exact_keys(
        projector_input,
        {
            "snapshot_manifest_path",
            "snapshot_manifest_sha256",
            "snapshot_schema_version",
            "snapshot_sha256",
            "snapshot_sha256_sidecar_path",
            "snapshot_sha256_sidecar_sha256",
            "splits",
        },
        "long_context_projector_manifest_invalid",
    )
    hierarchical_task_kv = _mapping(
        manifest.get("hierarchical_task_kv"),
        "long_context_projector_manifest_invalid",
    )
    expected_hierarchical_task_kv = {
        "segment_plan_schema_version": SEGMENT_PLAN_SCHEMA,
        "segment_plan_location": "outer_sidecar.segment_plan",
        "architecture": _projector.SEGMENT_PLAN_ARCHITECTURE,
        "execution_mode": _projector.SEGMENT_EXECUTION_MODE,
        "materialization": "metadata_only_no_tensor_or_kv",
        "tensors_emitted": False,
        "kv_payloads_emitted": False,
        "full_generation_kv_shared_claimed": False,
        "token_level_moe_claimed": False,
        "shared_prefix_membership": ("strict_all_five_role_visibility_intersection"),
        "ordered_prefix_chain": True,
        "independent_segment_concatenation_allowed": False,
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "shared_then_mask_allowed": False,
        "forbidden_current_future_preinsert_allowed": False,
        "cache_identity_required_exact_match_fields": list(
            _projector.CACHE_IDENTITY_FIELDS
        ),
        "cache_identity_mismatch_result": "cache_incompatible",
        "cache_identity_unknown_result": "cache_incompatible",
        "q_specialization_alone_sufficient_for_exact_reuse": False,
        "naive_in_stack_q_lora_exact_reuse_allowed": False,
    }
    if (
        manifest.get("schema_version") != PROJECTOR_MANIFEST_SCHEMA
        or dict(
            _mapping(
                manifest.get("producer"), "long_context_projector_manifest_invalid"
            )
        )
        != _input_manifest_expected_producer(projector_config)
        or projector_input.get("snapshot_manifest_path") != "manifest.json"
        or projector_input.get("snapshot_sha256_sidecar_path") != "manifest.json.sha256"
        or projector_input.get("snapshot_schema_version")
        != "anchor.training-snapshot.v2"
        or projector_input.get("splits") != ["train", "calibration"]
        or any(
            not _is_sha256(projector_input.get(field))
            for field in (
                "snapshot_manifest_sha256",
                "snapshot_sha256",
                "snapshot_sha256_sidecar_sha256",
            )
        )
        or dict(hierarchical_task_kv) != expected_hierarchical_task_kv
        or manifest.get("split_group_key") != "task_bundle_sha256"
        or manifest.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or manifest.get("all_five_role_views_same_split") is not True
        or manifest.get("canonical_gold_written") is not False
        or manifest.get("provider_requests") != 0
        or manifest.get("heldout_content_read") is not False
        or manifest.get("heldout_content_emitted") is not False
        or manifest.get("split_preserved") is not True
        or manifest.get("augmentation_applied_after_split") is not True
        or manifest.get("claim_scope") != "research_proxy_only"
    ):
        _fail("long_context_projector_manifest_invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(FIXED_FILES):
        _fail("long_context_projector_manifest_invalid")
    snapshots: dict[tuple[str, str], _BytesSnapshot] = {}
    for index, (relative, split, variant) in enumerate(FIXED_FILES):
        entry = _mapping(raw_files[index], "long_context_projector_manifest_invalid")
        _exact_keys(
            entry,
            {"path", "sha256", "bytes", "records", "split", "variant"},
            "long_context_projector_manifest_invalid",
        )
        if (
            entry.get("path") != relative
            or entry.get("split") != split
            or entry.get("variant") != variant
            or not _is_sha256(entry.get("sha256"))
            or not _positive_int(entry.get("bytes"))
            or not _positive_int(entry.get("records"))
        ):
            _fail("long_context_projector_manifest_invalid")
        path = _safe_relative(root, relative, "long_context_projector_file_invalid")
        snapshot = _read_snapshot(
            path,
            "long_context_projector_file_invalid",
            max_bytes=config.max_input_file_bytes,
        )
        if snapshot.sha256 != entry["sha256"] or snapshot.size != entry["bytes"]:
            _fail("long_context_projector_file_invalid")
        line_count = sum(
            1 for _item in _iter_jsonl(snapshot, "long_context_projector_file_invalid")
        )
        if line_count != entry["records"]:
            _fail("long_context_projector_file_invalid")
        inventory[path] = snapshot
        snapshots[(split, variant)] = snapshot
    inventory[manifest_path] = manifest_snapshot
    inventory[sidecar_path] = sidecar_snapshot
    return (
        root,
        manifest,
        snapshots,
        projector_config,
        sidecar_snapshot.sha256,
    )


def _plan_metadata(
    row: Mapping[str, Any],
    *,
    expected_split: str,
    expected_variant: str,
    path: str,
    line_number: int,
    line: bytes,
    partition_sha256: str,
    projector_config: _projector.TaskBoardProjectorConfig,
    projector_manifest: Mapping[str, Any],
) -> _SanitizedSourceRow:
    code = "long_context_projector_row_invalid"
    _exact_keys(
        row,
        {
            "schema_version",
            "id",
            "pair_id",
            "variant",
            "split",
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
            "segment_plan_schema_sha256",
            "augmentation",
            "segment_plan",
            "training_record",
        },
        code,
    )
    stage = row.get("stage")
    expert = row.get("expert")
    if (
        row.get("schema_version") != PROJECTOR_SIDECAR_SCHEMA
        or row.get("split") != expected_split
        or row.get("variant") != expected_variant
        or stage not in _projector.STAGE_EXPERTS
        or _projector.STAGE_EXPERTS[str(stage)] != expert
        or not _is_identifier(row.get("id"))
        or not _is_identifier(row.get("pair_id"))
        or not _is_identifier(row.get("source_gold_record_id"))
        or not _is_sha256(row.get("source_gold_sha256"))
        or not _is_sha256(row.get("source_gold_file_sha256"))
        or not _is_sha256(row.get("base_task_board_sha256"))
        or not _is_sha256(row.get("task_bundle_sha256"))
        or row.get("projector_version") != projector_config.projector_version
        or row.get("config_sha256") != projector_config.sha256
        or row.get("sidecar_schema_sha256") != projector_config.sidecar_schema_sha256
        or row.get("segment_plan_schema_sha256")
        != projector_config.segment_plan_schema_sha256
    ):
        _fail(code)
    projector_input = _mapping(projector_manifest.get("input"), code)
    if row.get("source_snapshot_sha256") != projector_input.get(
        "snapshot_sha256"
    ) or row.get("source_snapshot_manifest_sha256") != projector_input.get(
        "snapshot_manifest_sha256"
    ):
        _fail(code)
    augmentation = _mapping(row.get("augmentation"), code)
    _exact_keys(
        augmentation,
        {
            "kind",
            "same_task_only",
            "split_before_augmentation",
            "source_block_ids",
            "overlay_block_ids",
        },
        code,
    )
    overlay_ids = augmentation.get("overlay_block_ids")
    if (
        augmentation.get("same_task_only") is not True
        or augmentation.get("split_before_augmentation") is not True
        or not isinstance(overlay_ids, list)
        or any(not _is_identifier(item) for item in overlay_ids)
        or len(set(overlay_ids)) != len(overlay_ids)
    ):
        _fail(code)
    if expected_variant == "clean":
        if (
            augmentation.get("kind") != "clean"
            or augmentation.get("source_block_ids") != []
            or overlay_ids != []
        ):
            _fail(code)
    elif (
        expected_split != "train"
        or augmentation.get("kind") != "stale_duplicate_overlay"
        or not overlay_ids
        or not isinstance(augmentation.get("source_block_ids"), list)
        or any(not _is_identifier(item) for item in augmentation["source_block_ids"])
        or len(set(augmentation["source_block_ids"]))
        != len(augmentation["source_block_ids"])
        or len(augmentation["source_block_ids"]) != len(overlay_ids)
    ):
        _fail(code)

    inner = _mapping(row.get("training_record"), code)
    _exact_keys(
        inner,
        {
            "schema_version",
            "id",
            "pair_id",
            "variant",
            "language",
            "split",
            "role",
            "task_board",
            "attention_targets",
            "target",
        },
        code,
    )
    if (
        inner.get("id") != row.get("id")
        or inner.get("pair_id") != row.get("pair_id")
        or inner.get("variant") != expected_variant
        or inner.get("split") != expected_split
        or inner.get("role") != expert
        or inner.get("language") not in {"en", "zh-CN"}
    ):
        _fail(code)
    board = _mapping(inner.get("task_board"), code)
    targets = _mapping(inner.get("attention_targets"), code)
    target = _mapping(inner.get("target"), code)
    _exact_keys(board, {"task_id", "generation", "blocks"}, code)
    _exact_keys(
        targets,
        {"relevant_block_ids", "distractor_block_ids", "forbidden_block_ids"},
        code,
    )
    _exact_keys(target, {"selected_block_ids", "action", "answer"}, code)
    task_id = board.get("task_id")
    blocks = board.get("blocks")
    if not _is_identifier(task_id) or not isinstance(blocks, list) or not blocks:
        _fail(code)
    block_ids: list[str] = []
    block_kinds: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for block in blocks:
        item = _mapping(block, code)
        _exact_keys(item, {"id", "kind", "content", "commit_state", "visible_to"}, code)
        identifier = item.get("id")
        if not _is_identifier(identifier) or identifier in by_id:
            _fail(code)
        block_ids.append(str(identifier))
        block_kinds.append(str(item.get("kind")))
        by_id[str(identifier)] = item
    if not set(overlay_ids).issubset(by_id):
        _fail(code)
    base_ids = [
        identifier for identifier in block_ids if identifier not in set(overlay_ids)
    ]
    base_kinds = [
        block_kinds[index]
        for index, identifier in enumerate(block_ids)
        if identifier not in set(overlay_ids)
    ]
    fixed_target_indices = {
        "planner": 2,
        "tool_policy": 3,
        "domain_builder": 4,
        "domain_review": len(base_ids) - 2,
        "security": len(base_ids) - 1,
    }
    target_index = fixed_target_indices[str(stage)]
    if (
        len(base_ids) < 7
        or base_kinds[:2] != ["requirement", "repository"]
        or target_index < 2
        or target_index >= len(base_ids)
        or base_kinds[target_index] != _projector.STAGE_BLOCK_KINDS[str(stage)]
    ):
        _fail("long_context_causal_boundary_invalid")
    expected_relevant = base_ids[:target_index]
    expected_forbidden = base_ids[target_index:]
    expected_distractors = list(overlay_ids)
    if (
        targets.get("relevant_block_ids") != expected_relevant
        or targets.get("forbidden_block_ids") != expected_forbidden
        or targets.get("distractor_block_ids") != expected_distractors
        or target.get("selected_block_ids") != expected_relevant
    ):
        _fail("long_context_causal_boundary_invalid")
    causal_order = {identifier: index for index, identifier in enumerate(block_ids)}
    selected_source_ids = tuple(
        sorted(expected_relevant + expected_distractors, key=causal_order.__getitem__)
    )
    if set(selected_source_ids) & set(expected_forbidden):
        _fail("long_context_forbidden_selection")

    plan = _mapping(row.get("segment_plan"), code)
    _exact_keys(
        plan,
        {
            "schema_version",
            "architecture",
            "execution_mode",
            "materialization",
            "full_generation_kv_shared_claimed",
            "token_level_moe_claimed",
            "split_before_augmentation",
            "augmentation_applied_after_split",
            "bindings",
            "shared_prefix_policy",
            "target_delta_policy",
            "cache_compatibility",
            "segments",
        },
        code,
    )
    bindings = _mapping(plan.get("bindings"), code)
    segments = plan.get("segments")
    if (
        plan.get("schema_version") != SEGMENT_PLAN_SCHEMA
        or plan.get("architecture") != _projector.SEGMENT_PLAN_ARCHITECTURE
        or plan.get("execution_mode") != _projector.SEGMENT_EXECUTION_MODE
        or plan.get("materialization") != "metadata_only_no_tensor_or_kv"
        or plan.get("full_generation_kv_shared_claimed") is not False
        or plan.get("token_level_moe_claimed") is not False
        or plan.get("split_before_augmentation") is not True
        or plan.get("augmentation_applied_after_split") is not True
        or bindings.get("task_id") != task_id
        or bindings.get("task_bundle_sha256") != row.get("task_bundle_sha256")
        or bindings.get("split") != expected_split
        or bindings.get("variant") != expected_variant
        or bindings.get("stage") != stage
        or bindings.get("expert") != expert
        or not isinstance(segments, list)
        or not segments
    ):
        _fail(code)
    segment_ids: list[str] = []
    source_block_ids: list[str] = []
    private_count = 0
    private_source_ids: list[str] = []
    non_private_segment_ids: list[str] = []
    for index, segment in enumerate(segments):
        item = _mapping(segment, code)
        _exact_keys(
            item,
            {
                "segment_id",
                "content_sha256",
                "source_block_id",
                "serialization_order",
                "causal_order",
                "producer_role",
                "cache_scope",
                "visibility",
                "dependencies",
                "commit_state",
                "parent_segment_id",
                "parent_lineage_sha256",
                "prefix_lineage_sha256",
            },
            code,
        )
        if (
            not _is_identifier(item.get("segment_id"))
            or not _is_sha256(item.get("content_sha256"))
            or not _is_identifier(item.get("source_block_id"))
            or item.get("serialization_order") != index
            or item.get("causal_order")
            != causal_order.get(str(item.get("source_block_id")))
            or not _is_sha256(item.get("prefix_lineage_sha256"))
            or item.get("cache_scope")
            not in {
                "task_shared_prefix",
                "downstream_task_shared_immutable",
                "expert_private_delta",
            }
        ):
            _fail(code)
        segment_ids.append(str(item["segment_id"]))
        source_block_ids.append(str(item["source_block_id"]))
        if item.get("cache_scope") == "expert_private_delta":
            private_count += 1
            private_source_ids.append(str(item["source_block_id"]))
        else:
            non_private_segment_ids.append(str(item["segment_id"]))
    if tuple(source_block_ids) != selected_source_ids or len(set(segment_ids)) != len(
        segment_ids
    ):
        _fail("long_context_segment_order_invalid")
    if set(source_block_ids) & set(expected_forbidden):
        _fail("long_context_forbidden_selection")
    if (expected_variant == "clean" and private_source_ids) or (
        expected_variant == "noisy" and private_source_ids != list(overlay_ids)
    ):
        _fail("long_context_augmentation_pair_invalid")
    terminal = segments[-1].get("prefix_lineage_sha256")
    if not _is_sha256(terminal):
        _fail(code)
    return _SanitizedSourceRow(
        path=path,
        line_number=line_number,
        source_line_sha256=_sha256_bytes(line),
        source_partition_sha256=partition_sha256,
        record_id=str(row["id"]),
        pair_id=str(row["pair_id"]),
        source_gold_record_id=str(row["source_gold_record_id"]),
        source_gold_sha256=str(row["source_gold_sha256"]),
        task_bundle_sha256=str(row["task_bundle_sha256"]),
        task_id=str(task_id),
        task_id_sha256=_sha256_bytes(str(task_id).encode("utf-8")),
        split=expected_split,
        variant=expected_variant,
        stage=str(stage),
        expert=str(expert),
        language=str(inner["language"]),
        segment_plan_sha256=_sha256_value(plan),
        segment_ids=tuple(segment_ids),
        non_private_segment_ids=tuple(non_private_segment_ids),
        source_block_ids=tuple(source_block_ids),
        private_delta_segment_count=private_count,
        terminal_prefix_lineage_sha256=str(terminal),
    )


def _source_rows_first_pass(
    snapshots: Mapping[tuple[str, str], _BytesSnapshot],
    *,
    projector_config: _projector.TaskBoardProjectorConfig,
    projector_manifest: Mapping[str, Any],
    max_records: int,
) -> dict[tuple[str, str], list[_SanitizedSourceRow]]:
    result: dict[tuple[str, str], list[_SanitizedSourceRow]] = {}
    total = 0
    for relative, split, variant in FIXED_FILES:
        snapshot = snapshots[(split, variant)]
        rows: list[_SanitizedSourceRow] = []
        for line_number, line, value in _iter_jsonl(
            snapshot, "long_context_projector_file_invalid"
        ):
            rows.append(
                _plan_metadata(
                    value,
                    expected_split=split,
                    expected_variant=variant,
                    path=relative,
                    line_number=line_number,
                    line=line,
                    partition_sha256=snapshot.sha256,
                    projector_config=projector_config,
                    projector_manifest=projector_manifest,
                )
            )
            total += 1
            if total > max_records:
                _fail("long_context_record_limit_exceeded")
        if not rows or len({item.record_id for item in rows}) != len(rows):
            _fail("long_context_projector_rows_invalid")
        result[(split, variant)] = rows
    return result


def _verify_groups(
    rows: Mapping[tuple[str, str], list[_SanitizedSourceRow]],
    projector_manifest: Mapping[str, Any],
) -> None:
    flat = [
        row
        for _relative, split, variant in FIXED_FILES
        for row in rows[(split, variant)]
    ]
    bundle_split: dict[str, str] = {}
    bundle_task: dict[str, str] = {}
    task_binding: dict[str, tuple[str, str]] = {}
    source_id_binding: dict[str, tuple[str, str]] = {}
    groups: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    unique_segments: dict[str, tuple[str, str]] = {}
    for row in flat:
        prior_split = bundle_split.setdefault(row.task_bundle_sha256, row.split)
        prior_task = bundle_task.setdefault(row.task_bundle_sha256, row.task_id)
        if prior_split != row.split or prior_task != row.task_id:
            _fail("long_context_bundle_split_invalid")
        prior_bundle = task_binding.setdefault(
            row.task_id, (row.task_bundle_sha256, row.split)
        )
        if prior_bundle != (row.task_bundle_sha256, row.split):
            _fail("long_context_task_identity_fork")
        prior_source = source_id_binding.setdefault(
            row.source_gold_record_id,
            (row.source_gold_sha256, row.split),
        )
        if prior_source != (row.source_gold_sha256, row.split):
            _fail("long_context_source_split_invalid")
        key = (row.split, row.variant, row.task_bundle_sha256)
        roles = groups.setdefault(key, set())
        role = (row.stage, row.expert)
        if role in roles:
            _fail("long_context_role_group_invalid")
        roles.add(role)
        for segment_id in row.segment_ids:
            identity = (segment_id, row.task_bundle_sha256)
            prior = unique_segments.setdefault(segment_id, identity)
            if prior != identity:
                _fail("long_context_segment_identity_fork")
    expected_roles = {
        (stage, _projector.STAGE_EXPERTS[stage]) for stage in _projector.STAGES
    }
    if any(group != expected_roles for group in groups.values()):
        _fail("long_context_role_group_invalid")
    clean_by_bundle: dict[str, list[_SanitizedSourceRow]] = {}
    for row in rows[("train", "clean")] + rows[("calibration", "clean")]:
        clean_by_bundle.setdefault(row.task_bundle_sha256, []).append(row)
    for bundle, bundle_rows in clean_by_bundle.items():
        ordered_rows = sorted(
            bundle_rows,
            key=lambda item: _projector.STAGES.index(item.stage),
        )
        if len(ordered_rows) != len(_projector.STAGES):
            _fail("long_context_role_group_invalid")
        entries = [
            {
                "stage": row.stage,
                "expert": row.expert,
                "record_id": row.source_gold_record_id,
                "record_sha256": row.source_gold_sha256,
            }
            for row in ordered_rows
        ]
        if _projector._task_bundle_sha256(ordered_rows[0].task_id, entries) != bundle:
            _fail("long_context_bundle_digest_invalid")
    clean = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id)
        for row in rows[("train", "clean")]
    }
    noisy = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id)
        for row in rows[("train", "noisy")]
    }
    if clean != noisy:
        _fail("long_context_augmentation_pair_invalid")
    clean_pair_bindings = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id): (
            row.task_id,
            row.source_gold_record_id,
            row.source_gold_sha256,
        )
        for row in rows[("train", "clean")]
    }
    noisy_pair_bindings = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id): (
            row.task_id,
            row.source_gold_record_id,
            row.source_gold_sha256,
        )
        for row in rows[("train", "noisy")]
    }
    if clean_pair_bindings != noisy_pair_bindings:
        _fail("long_context_augmentation_pair_invalid")
    clean_non_private = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id): (
            row.non_private_segment_ids
        )
        for row in rows[("train", "clean")]
    }
    noisy_non_private = {
        (row.task_bundle_sha256, row.stage, row.expert, row.pair_id): (
            row.non_private_segment_ids
        )
        for row in rows[("train", "noisy")]
    }
    if clean_non_private != noisy_non_private:
        _fail("long_context_augmentation_pair_invalid")

    counts = _mapping(projector_manifest.get("counts"), "long_context_counts_invalid")
    split_counts = Counter(row.split for row in flat)
    variant_counts = Counter(row.variant for row in flat)
    stage_counts = Counter(row.stage for row in flat)
    expert_counts = Counter(row.expert for row in flat)
    language_counts = Counter(row.language for row in flat)
    task_ids = sorted(set(bundle_task.values()))
    segment_references = sum(len(row.segment_ids) for row in flat)
    if (
        counts.get("total") != len(flat)
        or counts.get("unique_task_bundles") != len(bundle_split)
        or counts.get("task_ids_sha256")
        != _sha256_bytes("\n".join(task_ids).encode("utf-8"))
        or counts.get("segment_references") != segment_references
        or counts.get("unique_segments") != len(unique_segments)
        or counts.get("by_split") != dict(sorted(split_counts.items()))
        or counts.get("by_variant") != dict(sorted(variant_counts.items()))
        or counts.get("by_stage") != dict(sorted(stage_counts.items()))
        or counts.get("by_expert") != dict(sorted(expert_counts.items()))
        or counts.get("by_language")
        != {language: language_counts.get(language, 0) for language in ("en", "zh-CN")}
    ):
        _fail("long_context_counts_invalid")


def _bucket(total_tokens: int) -> tuple[str, str]:
    if not _nonnegative_int(total_tokens):
        _fail("long_context_token_count_invalid")
    for name, limit, gate in BUCKETS:
        if total_tokens <= limit:
            return name, gate
    _fail("long_context_over_1m_rejected")


def _row_second_pass(
    source: _SanitizedSourceRow,
    value: Mapping[str, Any],
    line: bytes,
    *,
    counter: ExactTokenCounter,
    reserved_output_tokens: int,
) -> dict[str, Any]:
    if _sha256_bytes(line) != source.source_line_sha256:
        _fail("long_context_source_line_changed")
    inner = _mapping(value.get("training_record"), "long_context_projector_row_invalid")
    board = _mapping(inner.get("task_board"), "long_context_projector_row_invalid")
    targets = _mapping(
        inner.get("attention_targets"), "long_context_projector_row_invalid"
    )
    blocks = board.get("blocks")
    if not isinstance(blocks, list):
        _fail("long_context_projector_row_invalid")
    by_id = {
        str(item["id"]): item
        for item in blocks
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    relevant = targets.get("relevant_block_ids")
    distractors = targets.get("distractor_block_ids")
    forbidden = targets.get("forbidden_block_ids")
    if not all(isinstance(item, list) for item in (relevant, distractors, forbidden)):
        _fail("long_context_projector_row_invalid")
    augmentation = _mapping(
        value.get("augmentation"), "long_context_projector_row_invalid"
    )
    source_ids = augmentation.get("source_block_ids")
    overlay_ids = augmentation.get("overlay_block_ids")
    if source.variant == "clean":
        if source_ids != [] or overlay_ids != [] or distractors != []:
            _fail("long_context_augmentation_pair_invalid")
    else:
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not isinstance(overlay_ids, list)
            or len(source_ids) != len(overlay_ids)
            or overlay_ids != distractors
            or not set(source_ids).issubset(relevant)
            or set(source_ids) & set(overlay_ids)
        ):
            _fail("long_context_augmentation_pair_invalid")
        for source_id, overlay_id in zip(source_ids, overlay_ids, strict=True):
            source_block = _mapping(
                by_id.get(str(source_id)), "long_context_augmentation_pair_invalid"
            )
            overlay_block = _mapping(
                by_id.get(str(overlay_id)), "long_context_augmentation_pair_invalid"
            )
            if (
                overlay_block.get("kind") != "history"
                or overlay_block.get("commit_state") != "candidate"
                or overlay_block.get("visible_to") != [source.expert]
                or not isinstance(source_block.get("content"), str)
                or overlay_block.get("content") != source_block.get("content")
            ):
                _fail("long_context_augmentation_pair_invalid")
    plan = _mapping(value.get("segment_plan"), "long_context_projector_row_invalid")
    try:
        _projector._validate_segment_plan(
            plan,
            wrapper=value,
            inner=inner,
            by_id=by_id,
            relevant=relevant,
            distractors=distractors,
            forbidden=forbidden,
        )
    except _projector.TaskBoardProjectorError as exc:
        raise LongContextPreflightError("long_context_segment_plan_invalid") from exc
    if _sha256_value(plan) != source.segment_plan_sha256:
        _fail("long_context_segment_plan_changed")
    segments = plan.get("segments")
    assert isinstance(segments, list)
    ordered: list[str] = []
    shared: list[str] = []
    for index, segment in enumerate(segments):
        item = _mapping(segment, "long_context_segment_plan_invalid")
        source_id = str(item["source_block_id"])
        if source_id != source.source_block_ids[index] or source_id in set(forbidden):
            _fail("long_context_forbidden_selection")
        block = _mapping(by_id.get(source_id), "long_context_segment_plan_invalid")
        source_text = block.get("content")
        if not isinstance(source_text, str) or not source_text:
            _fail("long_context_segment_plan_invalid")
        ordered.append(source_text)
        if item.get("cache_scope") != "expert_private_delta":
            shared.append(source_text)
    input_tokens = counter.count(tuple(ordered))
    if source.private_delta_segment_count:
        if not shared:
            _fail("long_context_private_delta_invalid")
        shared_tokens = counter.count(tuple(shared))
    else:
        shared_tokens = input_tokens
    private_tokens = input_tokens - shared_tokens
    if (
        not _positive_int(input_tokens)
        or not _positive_int(shared_tokens)
        or private_tokens < 0
    ):
        _fail("long_context_token_count_invalid")
    total_tokens = input_tokens + reserved_output_tokens
    bucket, gate = _bucket(total_tokens)
    identity = {
        "task_bundle_sha256": source.task_bundle_sha256,
        "source_line_sha256": source.source_line_sha256,
        "segment_plan_sha256": source.segment_plan_sha256,
        "tokenizer_binding_sha256": counter.binding_sha256,
        "reserved_output_tokens": reserved_output_tokens,
    }
    result = {
        "schema_version": RECORD_SCHEMA,
        "record_id": "long-context-token-inventory-v1:" + _sha256_value(identity),
        "task_bundle_sha256": source.task_bundle_sha256,
        "task_id_sha256": source.task_id_sha256,
        "split": source.split,
        "variant": source.variant,
        "stage": source.stage,
        "expert": source.expert,
        "source_partition_sha256": source.source_partition_sha256,
        "source_line_sha256": source.source_line_sha256,
        "segment_plan_sha256": source.segment_plan_sha256,
        "segment_count": len(source.segment_ids),
        "private_delta_segment_count": source.private_delta_segment_count,
        "ordered_segment_ids_sha256": _sha256_value(list(source.segment_ids)),
        "terminal_prefix_lineage_sha256": source.terminal_prefix_lineage_sha256,
        "tokenizer_binding_sha256": counter.binding_sha256,
        "input_tokens": input_tokens,
        "shared_prefix_input_tokens": shared_tokens,
        "private_delta_input_tokens": private_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "total_tokens": total_tokens,
        "bucket": bucket,
        "gate": gate,
        "cache_identity_status": "identity_unbound",
        "reuse_savings_tokens": 0,
        "evaluation_status": "not_evaluated",
        "quality_validated": False,
        "allocation_validated": False,
        "execution_authorized": False,
        "provider_requests": 0,
    }
    _validate_inventory_record(result)
    return result


def _validate_inventory_record(value: Mapping[str, Any]) -> None:
    keys = {
        "schema_version",
        "record_id",
        "task_bundle_sha256",
        "task_id_sha256",
        "split",
        "variant",
        "stage",
        "expert",
        "source_partition_sha256",
        "source_line_sha256",
        "segment_plan_sha256",
        "segment_count",
        "private_delta_segment_count",
        "ordered_segment_ids_sha256",
        "terminal_prefix_lineage_sha256",
        "tokenizer_binding_sha256",
        "input_tokens",
        "shared_prefix_input_tokens",
        "private_delta_input_tokens",
        "reserved_output_tokens",
        "total_tokens",
        "bucket",
        "gate",
        "cache_identity_status",
        "reuse_savings_tokens",
        "evaluation_status",
        "quality_validated",
        "allocation_validated",
        "execution_authorized",
        "provider_requests",
    }
    _exact_keys(value, keys, "long_context_inventory_record_invalid")
    hashes = (
        "task_bundle_sha256",
        "task_id_sha256",
        "source_partition_sha256",
        "source_line_sha256",
        "segment_plan_sha256",
        "ordered_segment_ids_sha256",
        "terminal_prefix_lineage_sha256",
        "tokenizer_binding_sha256",
    )
    if (
        value.get("schema_version") != RECORD_SCHEMA
        or not isinstance(value.get("record_id"), str)
        or _RECORD_ID_RE.fullmatch(str(value.get("record_id"))) is None
        or any(not _is_sha256(value.get(field)) for field in hashes)
        or value.get("split") not in {"train", "calibration"}
        or value.get("variant") not in {"clean", "noisy"}
        or (value.get("split") == "calibration" and value.get("variant") != "clean")
        or value.get("stage") not in _projector.STAGE_EXPERTS
        or _projector.STAGE_EXPERTS[str(value.get("stage"))] != value.get("expert")
        or not _positive_int(value.get("segment_count"))
        or not _nonnegative_int(value.get("private_delta_segment_count"))
        or int(value["private_delta_segment_count"]) > int(value["segment_count"])
        or not _positive_int(value.get("input_tokens"))
        or not _positive_int(value.get("shared_prefix_input_tokens"))
        or not _nonnegative_int(value.get("private_delta_input_tokens"))
        or int(value["shared_prefix_input_tokens"])
        + int(value["private_delta_input_tokens"])
        != int(value["input_tokens"])
        or value.get("reserved_output_tokens") != 4096
        or not _positive_int(value.get("total_tokens"))
        or int(value["input_tokens"]) + int(value["reserved_output_tokens"])
        != int(value.get("total_tokens", -1))
        or (
            int(value["private_delta_segment_count"]) == 0
            and int(value["private_delta_input_tokens"]) != 0
        )
        or (
            int(value["private_delta_segment_count"]) > 0
            and int(value["private_delta_input_tokens"]) <= 0
        )
        or (
            value.get("variant") == "clean"
            and (
                int(value["private_delta_segment_count"]) != 0
                or int(value["private_delta_input_tokens"]) != 0
            )
        )
        or (
            value.get("variant") == "noisy"
            and (
                value.get("split") != "train"
                or int(value["private_delta_segment_count"]) <= 0
                or int(value["private_delta_input_tokens"]) <= 0
            )
        )
        or (value.get("bucket"), value.get("gate"))
        != _bucket(int(value["total_tokens"]))
        or value.get("cache_identity_status") != "identity_unbound"
        or value.get("reuse_savings_tokens") != 0
        or value.get("evaluation_status") != "not_evaluated"
        or value.get("quality_validated") is not False
        or value.get("allocation_validated") is not False
        or value.get("execution_authorized") is not False
        or value.get("provider_requests") != 0
    ):
        _fail("long_context_inventory_record_invalid")


def _records_second_pass(
    snapshots: Mapping[tuple[str, str], _BytesSnapshot],
    source_rows: Mapping[tuple[str, str], list[_SanitizedSourceRow]],
    *,
    counter: ExactTokenCounter,
    reserved_output_tokens: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _relative, split, variant in FIXED_FILES:
        expected = source_rows[(split, variant)]
        produced: list[dict[str, Any]] = []
        parsed = list(
            _iter_jsonl(
                snapshots[(split, variant)], "long_context_projector_file_invalid"
            )
        )
        if len(parsed) != len(expected):
            _fail("long_context_source_line_changed")
        for source, (line_number, line, value) in zip(expected, parsed, strict=True):
            if source.line_number != line_number:
                _fail("long_context_source_line_changed")
            produced.append(
                _row_second_pass(
                    source,
                    value,
                    line,
                    counter=counter,
                    reserved_output_tokens=reserved_output_tokens,
                )
            )
        output[(split, variant)] = produced
    return output


def _output_has_denied_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _DENIED_OUTPUT_KEYS or _output_has_denied_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_output_has_denied_key(item) for item in value)
    return False


def _verify_inventory_unchanged(inventory: Mapping[Path, _BytesSnapshot]) -> None:
    for path, expected in inventory.items():
        current = _read_snapshot(
            path,
            "long_context_input_changed",
            max_bytes=max(expected.size, 1),
        )
        if current.sha256 != expected.sha256 or current.size != expected.size:
            _fail("long_context_input_changed")


def _check_output(output: Path, inputs: Iterable[Path]) -> None:
    if output.exists() or output.is_symlink():
        _fail("long_context_output_exists_or_overlaps_input")
    for source in inputs:
        resolved = source.resolve()
        if output == resolved:
            _fail("long_context_output_exists_or_overlaps_input")
        for parent, child in ((output, resolved), (resolved, output)):
            try:
                child.relative_to(parent)
            except ValueError:
                pass
            else:
                _fail("long_context_output_exists_or_overlaps_input")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    data = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _manifest_counts(
    records: Mapping[tuple[str, str], list[dict[str, Any]]],
    source_rows: Mapping[tuple[str, str], list[_SanitizedSourceRow]],
) -> dict[str, Any]:
    flat = [
        row
        for _relative, split, variant in FIXED_FILES
        for row in records[(split, variant)]
    ]
    source_flat = [
        row
        for _relative, split, variant in FIXED_FILES
        for row in source_rows[(split, variant)]
    ]
    return {
        "total": len(flat),
        "unique_task_bundles": len({row["task_bundle_sha256"] for row in flat}),
        "task_ids_sha256": _sha256_bytes(
            "\n".join(sorted({row.task_id for row in source_flat})).encode("utf-8")
        ),
        "segment_references": sum(int(row["segment_count"]) for row in flat),
        "unique_segments": len(
            {segment_id for row in source_flat for segment_id in row.segment_ids}
        ),
        "by_split": dict(sorted(Counter(row["split"] for row in flat).items())),
        "by_variant": dict(sorted(Counter(row["variant"] for row in flat).items())),
        "by_stage": dict(sorted(Counter(row["stage"] for row in flat).items())),
        "by_expert": dict(sorted(Counter(row["expert"] for row in flat).items())),
        "by_bucket": {
            name: sum(row["bucket"] == name for row in flat)
            for name, _limit, _gate in BUCKETS
        }
        | {"gt_1m": 0},
        "by_gate": {
            gate: sum(row["gate"] == gate for row in flat)
            for gate in (
                "measurement_candidate",
                "capability_only",
                "research_only_blocked",
                "reject",
            )
        },
    }


def build_long_context_token_inventory(
    config: LongContextTokenInventoryConfig | str | Path,
    projector_dir: str | Path,
    expected_projector_manifest_sha256: str,
    output_dir: str | Path,
    *,
    counter: ExactTokenCounter,
) -> dict[str, Any]:
    """Publish a deterministic exact-token inventory without source bodies."""

    if isinstance(config, LongContextTokenInventoryConfig):
        cfg = config
        cfg_inventory: dict[Path, _BytesSnapshot] = {}
        cfg_current, cfg_inventory = LongContextTokenInventoryConfig.load(cfg.path)
        if cfg_current != cfg:
            _fail("long_context_config_changed")
    else:
        cfg, cfg_inventory = LongContextTokenInventoryConfig.load(config)
    if type(counter) not in {
        SyntheticFixtureTokenCounter,
        LocalTransformersTokenCounter,
    }:
        _fail("long_context_tokenizer_binding_invalid")
    if not isinstance(counter.metadata, Mapping):
        _fail("long_context_tokenizer_binding_invalid")
    expected_tokenizer_keys = {
        "backend",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_label_source",
        "tokenizer_assets_sha256",
        "tokenizer_runtime_sha256",
        "chat_template_sha256",
        "serialization_policy_sha256",
        "special_token_policy_sha256",
        "network_access",
        "exact_token_counts",
        "synthetic_fixture_only",
    }
    _exact_keys(
        counter.metadata,
        expected_tokenizer_keys,
        "long_context_tokenizer_binding_invalid",
    )
    if (
        not _is_sha256(counter.binding_sha256)
        or not _is_identifier(counter.metadata.get("tokenizer_id"))
        or not _is_identifier(counter.metadata.get("tokenizer_revision"))
        or counter.metadata.get("tokenizer_label_source")
        != "caller_supplied_and_hash_bound"
        or any(
            not _is_sha256(counter.metadata.get(field))
            for field in (
                "tokenizer_assets_sha256",
                "tokenizer_runtime_sha256",
                "chat_template_sha256",
                "serialization_policy_sha256",
                "special_token_policy_sha256",
            )
        )
        or counter.metadata.get("network_access") is not False
        or counter.metadata.get("exact_token_counts") is not True
        or not isinstance(counter.metadata.get("synthetic_fixture_only"), bool)
        or counter.metadata.get("backend")
        not in {"local_offline_tokenizer", "explicit_synthetic_tokenizer"}
        or (counter.metadata.get("backend") == "explicit_synthetic_tokenizer")
        != counter.metadata.get("synthetic_fixture_only")
        or _sha256_value(counter.metadata) != counter.binding_sha256
    ):
        _fail("long_context_tokenizer_binding_invalid")
    inventory = dict(cfg_inventory)
    (
        projector_root,
        projector_manifest,
        snapshots,
        projector_config,
        projector_manifest_sidecar_sha256,
    ) = _load_projector_artifact(
        cfg,
        projector_dir,
        expected_projector_manifest_sha256,
        inventory,
    )
    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink() or raw_output.parent.is_symlink():
        _fail("long_context_output_exists_or_overlaps_input")
    try:
        output_parent = raw_output.parent.resolve(strict=True)
    except OSError as exc:
        raise LongContextPreflightError("long_context_output_parent_invalid") from exc
    if not output_parent.is_dir() or output_parent != raw_output.parent.absolute():
        _fail("long_context_output_parent_invalid")
    output_parent_identity = _directory_identity(output_parent.stat())
    output = output_parent / raw_output.name
    _check_output(output, [projector_root, *inventory])
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        _fail("long_context_temporary_conflict")
    try:
        source_rows = _source_rows_first_pass(
            snapshots,
            projector_config=projector_config,
            projector_manifest=projector_manifest,
            max_records=cfg.max_input_records,
        )
        _verify_groups(source_rows, projector_manifest)
        records = _records_second_pass(
            snapshots,
            source_rows,
            counter=counter,
            reserved_output_tokens=cfg.reserved_output_tokens,
        )
        temporary.mkdir(parents=True)
        files: list[dict[str, Any]] = []
        output_snapshots: dict[Path, _BytesSnapshot] = {}
        for relative, split, variant in FIXED_FILES:
            path = temporary.joinpath(*Path(relative).parts)
            data = _write_jsonl(path, records[(split, variant)])
            if not data or len(data) > cfg.max_output_file_bytes:
                _fail("long_context_output_file_invalid")
            snapshot = _read_snapshot(
                path,
                "long_context_output_file_invalid",
                max_bytes=cfg.max_output_file_bytes,
            )
            parsed = list(_iter_jsonl(snapshot, "long_context_output_file_invalid"))
            if len(parsed) != len(records[(split, variant)]):
                _fail("long_context_output_file_invalid")
            for _line_number, _line, row in parsed:
                _validate_inventory_record(row)
            output_snapshots[path] = snapshot
            files.append(
                {
                    "path": relative,
                    "sha256": snapshot.sha256,
                    "bytes": snapshot.size,
                    "records": len(parsed),
                    "split": split,
                    "variant": variant,
                }
            )
        synthetic_fixture = bool(counter.metadata["synthetic_fixture_only"])
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": (
                "synthetic_fixture_inventory_ready"
                if synthetic_fixture
                else "exact_token_inventory_ready"
            ),
            "input": {
                "projector_manifest_schema_version": PROJECTOR_MANIFEST_SCHEMA,
                "projector_manifest_sha256": expected_projector_manifest_sha256,
                "projector_manifest_sha256_sidecar_sha256": (
                    projector_manifest_sidecar_sha256
                ),
                "projector_manifest_schema_sha256": (
                    projector_config.manifest_schema_sha256
                ),
                "projector_config_sha256": projector_config.sha256,
                "projector_sidecar_schema_sha256": projector_config.sidecar_schema_sha256,
                "segment_plan_schema_sha256": projector_config.segment_plan_schema_sha256,
                "partitions": [
                    {
                        "path": relative,
                        "sha256": snapshots[(split, variant)].sha256,
                        "bytes": snapshots[(split, variant)].size,
                        "records": len(source_rows[(split, variant)]),
                        "split": split,
                        "variant": variant,
                    }
                    for relative, split, variant in FIXED_FILES
                ],
            },
            "producer": {
                "name": "anchor.long-context-token-inventory",
                "producer_version": PRODUCER_VERSION,
                "config_schema_version": CONFIG_SCHEMA,
                "config_sha256": cfg.sha256,
                "record_schema_version": RECORD_SCHEMA,
                "record_schema_sha256": cfg.record_schema_sha256,
                "manifest_schema_sha256": cfg.manifest_schema_sha256,
            },
            "tokenizer_binding": dict(counter.metadata),
            "token_accounting": {
                "input_tokens_source": "exact_serialized_input",
                "reserved_output_tokens": cfg.reserved_output_tokens,
                "total_tokens_formula": "input_tokens_plus_reserved_output_tokens",
                "scope_attribution_formula": (
                    "shared_prefix_input_tokens_plus_private_delta_input_tokens_"
                    "equals_input_tokens"
                ),
                "cache_identity_status": "identity_unbound",
                "reuse_savings_tokens": 0,
            },
            "model_contract": dict(cfg.raw["model_contract"]),
            "bucket_policy": dict(cfg.raw["buckets"]),
            "files": files,
            "counts": _manifest_counts(records, source_rows),
            "split_group_key": "task_bundle_sha256",
            "task_id_cross_binding_key": "task_id_sha256",
            "task_id_cross_binding_transform": (
                "sha256_utf8_training_record_task_board_task_id"
            ),
            "all_five_role_views_same_split": True,
            "split_before_augmentation": True,
            "forbidden_current_future_excluded_before_serialization": True,
            "manifest_sha256_sidecar_required": True,
            "inventory_mode": (
                "synthetic_fixture" if synthetic_fixture else "local_exact_tokenizer"
            ),
            "target_model_tokenizer_match": (
                "not_applicable"
                if synthetic_fixture
                else "consumer_verification_required"
            ),
            "bucket_basis": "bound_tokenizer_total_tokens",
            "capability_validated": False,
            "approximate_inventory_emitted": False,
            "null_inventory_emitted": False,
            "provider_requests": 0,
            "canonical_gold_written": False,
            "evaluation_status": "not_evaluated",
            "quality_validated": False,
            "allocation_validated": False,
            "execution_authorized": False,
            "claim_scope": (
                "synthetic_fixture_contract_only"
                if synthetic_fixture
                else "exact_bound_tokenizer_inventory_only"
            ),
        }
        if _output_has_denied_key(manifest):
            _fail("long_context_body_free_contract_invalid")
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        sidecar_bytes = f"{manifest_sha256}  manifest.json\n".encode("ascii")
        manifest_sidecar_path = temporary / "manifest.json.sha256"
        manifest_sidecar_path.write_bytes(sidecar_bytes)
        manifest_snapshot = _read_snapshot(
            manifest_path,
            "long_context_output_manifest_invalid",
            max_bytes=cfg.max_output_file_bytes,
        )
        manifest_sidecar_snapshot = _read_snapshot(
            manifest_sidecar_path,
            "long_context_output_manifest_invalid",
            max_bytes=256,
        )
        if (
            manifest_snapshot.data != manifest_bytes
            or manifest_snapshot.sha256 != manifest_sha256
            or manifest_sidecar_snapshot.data != sidecar_bytes
            or _json(manifest_snapshot, "long_context_output_manifest_invalid")
            != manifest
        ):
            _fail("long_context_output_manifest_invalid")
        _verify_inventory_unchanged(inventory)
        counter.verify_unchanged()
        for path, expected in output_snapshots.items():
            current = _read_snapshot(
                path,
                "long_context_output_file_invalid",
                max_bytes=cfg.max_output_file_bytes,
            )
            if current.sha256 != expected.sha256 or current.size != expected.size:
                _fail("long_context_output_file_invalid")
        final_manifest = _read_snapshot(
            manifest_path,
            "long_context_output_manifest_invalid",
            max_bytes=cfg.max_output_file_bytes,
        )
        final_sidecar = _read_snapshot(
            manifest_sidecar_path,
            "long_context_output_manifest_invalid",
            max_bytes=256,
        )
        if final_manifest.data != manifest_bytes or final_sidecar.data != sidecar_bytes:
            _fail("long_context_output_manifest_invalid")
        if (
            output.exists()
            or output.is_symlink()
            or output_parent.is_symlink()
            or _directory_identity(output_parent.stat()) != output_parent_identity
        ):
            _fail("long_context_output_race_detected")
        os.replace(temporary, output)
        return {
            "schema_version": MANIFEST_SCHEMA,
            "status": "published",
            "manifest_sha256": manifest_sha256,
            "records": int(manifest["counts"]["total"]),
            "unique_task_bundles": int(manifest["counts"]["unique_task_bundles"]),
            "segment_references": int(manifest["counts"]["segment_references"]),
            "provider_requests": 0,
            "model_weights_read": False,
            "training_authorized": False,
        }
    except LongContextPreflightError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise LongContextPreflightError("long_context_internal_error") from exc


__all__ = [
    "BUCKETS",
    "LongContextPreflightError",
    "LongContextTokenInventoryConfig",
    "LocalTransformersTokenCounter",
    "SyntheticFixtureTokenCounter",
    "build_long_context_token_inventory",
]
