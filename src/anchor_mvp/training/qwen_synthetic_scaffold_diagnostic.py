"""Low-resource Qwen2.5-1.5B q_proj diagnostic over the synthetic scaffold.

This module is deliberately separate from the frozen one-step diagnostic.  It
accepts only the authenticated synthetic fixture, only rank-4 ``q_proj`` LoRA,
and only 2 or 20 optimizer steps.  It never grants training or formal-release
authority; ``--execute`` records only an explicit diagnostic user request.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from . import qwen_lora_diagnostic as qdiag
from .config import ConfigError, _expand_env
from .manifest import config_fingerprint


SCHEMA_VERSION = "anchor.qwen25-1.5b-synthetic-scaffold-qonly-diagnostic.v1"
CONFIG_VERSION = "anchor.qwen25-1.5b-synthetic-scaffold-qonly-diagnostic-config.v1"
PREFLIGHT_VERSION = "anchor.qwen25-1.5b-synthetic-scaffold-qonly-tokenizer-preflight.v1"
CONFIG_PATH = "configs/training/qwen2_5_1_5b_synthetic_scaffold_qonly_v1.yaml"
DATASET_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-manifest.v1"
RECORD_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-record.v1"
DATASET_ROOT = "fixtures/research/synthetic_nl_scaffold_diagnostic_v1"
RECORD_SCHEMA_PATH = "configs/research/synthetic_nl_scaffold_diagnostic_v1.schema.json"
MANIFEST_SCHEMA_PATH = (
    "configs/research/synthetic_nl_scaffold_diagnostic_v1_manifest.schema.json"
)
PARTITION_PATHS = (
    "train/json_only.jsonl",
    "train/concise_rationale_plus_json.jsonl",
    "eval_proxy/json_only.jsonl",
    "eval_proxy/concise_rationale_plus_json.jsonl",
)
ALLOWED_STEPS = (2, 20)
PREFLIGHT_OUTPUT_ROOT = (
    "artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_qonly_preflight"
)
PREFLIGHT_FILENAME = "preflight.json"
PREFLIGHT_SIDECAR_FILENAME = "preflight.json.sha256"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/qwen_synthetic_scaffold_diagnostic.py"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 2_000_000
_MAX_PARTITION_BYTES = 50_000_000
_PREFLIGHT_FILES = frozenset({PREFLIGHT_FILENAME, PREFLIGHT_SIDECAR_FILENAME})
_DIAGNOSTIC_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "diagnostic_receipt.json",
        "diagnostic_receipt.json.sha256",
    }
)


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self) -> None:
        current = _read_snapshot(self.path, max_bytes=max(len(self.data), 1024))
        if (
            current.identity != self.identity
            or current.sha256 != self.sha256
            or current.data != self.data
        ):
            raise ConfigError(f"authenticated file changed: {self.path}")


@dataclass(frozen=True)
class ScaffoldExample:
    record_id: str
    split: str
    variant: str
    source_bundle_id: str
    prompt: str
    target: str


@dataclass(frozen=True)
class ScaffoldDataset:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    partition_sha256: Mapping[str, str]
    train: tuple[ScaffoldExample, ...]
    eval_proxy: tuple[ScaffoldExample, ...]


@dataclass(frozen=True)
class AuthenticatedPreflightReceipt:
    value: Mapping[str, Any]
    receipt_snapshot: FileSnapshot
    sidecar_snapshot: FileSnapshot

    def assert_unchanged(self) -> None:
        self.receipt_snapshot.assert_unchanged()
        self.sidecar_snapshot.assert_unchanged()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_snapshot(path: Path, *, max_bytes: int) -> FileSnapshot:
    qdiag._assert_physical_path(path, require_file=True, label="diagnostic input")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_bytes:
            raise ConfigError(f"diagnostic input exceeds byte cap: {path}")
        data = handle.read(max_bytes + 1)
        after = os.fstat(handle.fileno())
    path_after = path.stat()
    if (
        len(data) > max_bytes
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(data) != after.st_size
    ):
        raise ConfigError(f"diagnostic input changed while reading: {path}")
    return FileSnapshot(path, data, _sha256(data), _stat_identity(after))


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a JSON object")
    return value


def _strict_jsonl(data: bytes, label: str) -> list[Mapping[str, Any]]:
    if b"\r" in data or not data.endswith(b"\n"):
        raise ConfigError(f"{label} must use LF and end with LF")
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise ConfigError(f"{label} contains blank line {index}")
        records.append(_strict_json(line, f"{label}:{index}"))
    return records


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ConfigError(f"{label} fields drifted")


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConfigError(f"{label} must be lowercase SHA-256")
    return value


def _repo_path(
    relative_value: object, expected: str, *, require_file: bool = True
) -> Path:
    if relative_value != expected:
        raise ConfigError(f"path must remain exactly {expected}")
    relative = PurePosixPath(expected)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"unsafe repository path: {expected}")
    path = qdiag._project_root_from_module().joinpath(*relative.parts)
    return qdiag._assert_physical_path(
        path,
        require_file=require_file,
        require_directory=not require_file,
        label=expected,
    )


def _canonical_config_path(path: str | Path) -> Path:
    requested = Path(path)
    canonical = qdiag._project_root_from_module() / Path(CONFIG_PATH)
    qdiag._assert_physical_path(canonical, require_file=True, label="synthetic config")
    if requested.is_absolute():
        resolved = Path(os.path.abspath(requested))
    else:
        if requested.as_posix() != CONFIG_PATH:
            raise ConfigError(f"config must remain exactly {CONFIG_PATH}")
        resolved = canonical
    qdiag._assert_physical_path(resolved, require_file=True, label="synthetic config")
    if os.path.normcase(str(resolved)) != os.path.normcase(str(canonical)):
        raise ConfigError(f"config must remain exactly {CONFIG_PATH}")
    return canonical


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "model",
            "dataset",
            "lora",
            "training",
            "precision",
            "output",
            "claims",
            "_config_path",
        },
        "config",
    )
    if config.get("schema_version") != CONFIG_VERSION:
        raise ConfigError("synthetic q_only config version drifted")
    if config.get("claim_scope") != (
        "synthetic_diagnostic_only_no_formal_or_training_authority"
    ):
        raise ConfigError("claim_scope drifted")
    if _mapping(config.get("paths"), "paths") != {"project_root": "../.."}:
        raise ConfigError("paths drifted")

    model = _mapping(config.get("model"), "model")
    required_model = {
        "id",
        "local_path",
        "local_files_only",
        "allow_network",
        "trust_remote_code",
        "expected_source_revision",
        "expected_source_repo",
        "expected_config_json_sha256",
        "expected_model_safetensors_sha256",
        "expected_tokenizer_json_sha256",
        "expected_tokenizer_config_sha256",
    }
    _exact_keys(model, required_model, "model")
    if (
        model.get("id") != qdiag.EXPECTED_MODEL_ID
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("expected_source_revision") != qdiag.EXPECTED_SOURCE_REVISION
        or model.get("expected_source_repo") != qdiag.EXPECTED_SOURCE_REPO
    ):
        raise ConfigError("model contract drifted")
    for field in required_model:
        if field.startswith("expected_") and field.endswith("_sha256"):
            _require_sha(model.get(field), f"model.{field}")

    dataset = _mapping(config.get("dataset"), "dataset")
    _exact_keys(
        dataset,
        {
            "kind",
            "root",
            "manifest",
            "manifest_sidecar",
            "record_schema",
            "manifest_schema",
            "expected_manifest_sha256",
            "expected_record_schema_sha256",
            "expected_manifest_schema_sha256",
            "train_records",
            "eval_proxy_records",
            "formal_inputs_allowed",
            "heldout_allowed",
            "protected_source_paths_allowed",
        },
        "dataset",
    )
    if (
        dataset.get("kind") != "synthetic_nl_scaffold_diagnostic_v1"
        or dataset.get("root") != DATASET_ROOT
        or dataset.get("manifest") != f"{DATASET_ROOT}/manifest.json"
        or dataset.get("manifest_sidecar") != f"{DATASET_ROOT}/manifest.json.sha256"
        or dataset.get("record_schema") != RECORD_SCHEMA_PATH
        or dataset.get("manifest_schema") != MANIFEST_SCHEMA_PATH
        or dataset.get("train_records") != 80
        or dataset.get("eval_proxy_records") != 20
        or dataset.get("formal_inputs_allowed") is not False
        or dataset.get("heldout_allowed") is not False
        or dataset.get("protected_source_paths_allowed") is not False
    ):
        raise ConfigError("dataset contract drifted")
    for field in (
        "expected_manifest_sha256",
        "expected_record_schema_sha256",
        "expected_manifest_schema_sha256",
    ):
        _require_sha(dataset.get(field), f"dataset.{field}")

    if _mapping(config.get("lora"), "lora") != {
        "profile": "q_only",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
    }:
        raise ConfigError("LoRA contract must remain q_only rank-4")
    training = _mapping(config.get("training"), "training")
    if training != {
        "allowed_max_steps": [2, 20],
        "default_max_steps": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 512,
        "learning_rate": 0.00005,
        "seed": 1337,
        "gradient_checkpointing": True,
        "eval_before_after": True,
    }:
        raise ConfigError("training contract drifted")
    if _mapping(config.get("precision"), "precision") != {
        "compute_dtype": "bfloat16",
        "tf32": True,
        "float32_matmul_precision": "high",
    }:
        raise ConfigError("precision contract drifted")
    if _mapping(config.get("output"), "output") != {
        "adapter_dir_template": (
            "artifacts/diagnostics/"
            "qwen2_5_1_5b_synthetic_scaffold_qonly_r4_step{max_steps}"
        )
    }:
        raise ConfigError("output contract drifted")
    if _mapping(config.get("claims"), "claims") != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }:
        raise ConfigError("claims must remain diagnostic-only and blocked")


def load_config(path: str | Path) -> dict[str, Any]:
    canonical = _canonical_config_path(path)
    before = _read_snapshot(canonical, max_bytes=_MAX_METADATA_BYTES)
    try:
        raw_config = yaml.safe_load(before.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("synthetic config is not valid UTF-8 YAML") from exc
    if not isinstance(raw_config, Mapping):
        raise ConfigError("synthetic config must be a mapping")
    config = _expand_env(dict(raw_config))
    config["_config_path"] = str(canonical)
    validate_config(config)
    before.assert_unchanged()
    return config


def _validate_sidecar(snapshot: FileSnapshot, manifest_sha256: str) -> None:
    expected = f"{manifest_sha256}  manifest.json\n".encode("ascii")
    if snapshot.data != expected:
        raise ConfigError("manifest.json.sha256 is missing or malformed")


def _example(record: Mapping[str, Any], *, expected_split: str) -> ScaffoldExample:
    claims = _mapping(record.get("claims"), "record.claims")
    audit = _mapping(record.get("audit"), "record.audit")
    if (
        record.get("schema_version") != RECORD_VERSION
        or record.get("split") != expected_split
        or claims.get("diagnostic_only") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal") is not False
        or any(
            audit.get(field) != 0
            for field in (
                "provider_requests",
                "network_requests",
                "model_loads",
                "gpu_requests",
                "protected_body_reads",
                "real_tool_executions",
            )
        )
    ):
        raise ConfigError("synthetic record crossed the diagnostic boundary")
    input_value = _mapping(record.get("input"), "record.input")
    target = _mapping(record.get("target"), "record.target")
    values = {
        "record_id": record.get("record_id"),
        "variant": record.get("variant"),
        "source_bundle_id": record.get("source_bundle_id"),
        "prompt": input_value.get("materialized_prompt"),
        "target": target.get("serialized_assistant_output"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ConfigError("synthetic record has an empty training view")
    return ScaffoldExample(split=expected_split, **values)  # type: ignore[arg-type]


def load_dataset(config: Mapping[str, Any]) -> ScaffoldDataset:
    dataset = _mapping(config.get("dataset"), "dataset")
    root = _repo_path(dataset.get("root"), DATASET_ROOT, require_file=False)
    manifest_path = _repo_path(dataset.get("manifest"), f"{DATASET_ROOT}/manifest.json")
    sidecar_path = _repo_path(
        dataset.get("manifest_sidecar"), f"{DATASET_ROOT}/manifest.json.sha256"
    )
    record_schema_path = _repo_path(dataset.get("record_schema"), RECORD_SCHEMA_PATH)
    manifest_schema_path = _repo_path(
        dataset.get("manifest_schema"), MANIFEST_SCHEMA_PATH
    )
    snapshots = {
        "manifest": _read_snapshot(manifest_path, max_bytes=_MAX_METADATA_BYTES),
        "sidecar": _read_snapshot(sidecar_path, max_bytes=1024),
        "record_schema": _read_snapshot(
            record_schema_path, max_bytes=_MAX_METADATA_BYTES
        ),
        "manifest_schema": _read_snapshot(
            manifest_schema_path, max_bytes=_MAX_METADATA_BYTES
        ),
    }
    if snapshots["manifest"].sha256 != dataset["expected_manifest_sha256"]:
        raise ConfigError("synthetic manifest SHA-256 mismatch")
    if snapshots["record_schema"].sha256 != dataset["expected_record_schema_sha256"]:
        raise ConfigError("synthetic record schema SHA-256 mismatch")
    if (
        snapshots["manifest_schema"].sha256
        != dataset["expected_manifest_schema_sha256"]
    ):
        raise ConfigError("synthetic manifest schema SHA-256 mismatch")
    _validate_sidecar(snapshots["sidecar"], snapshots["manifest"].sha256)

    manifest = _strict_json(snapshots["manifest"].data, "synthetic manifest")
    manifest_schema = _strict_json(
        snapshots["manifest_schema"].data, "synthetic manifest schema"
    )
    record_schema = _strict_json(
        snapshots["record_schema"].data, "synthetic record schema"
    )
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator(manifest_schema).validate(manifest)
    Draft202012Validator.check_schema(record_schema)
    record_validator = Draft202012Validator(record_schema)
    if (
        manifest.get("schema_version") != DATASET_VERSION
        or manifest.get("status") != "dataset_proxy_ready_training_not_authorized"
        or manifest.get("claim_scope")
        != "synthetic_diagnostic_only_no_formal_or_training_authority"
    ):
        raise ConfigError("synthetic manifest status drifted")
    claims = _mapping(manifest.get("claims"), "manifest.claims")
    if (
        claims.get("dataset_proxy_ready") is not True
        or claims.get("diagnostic_only") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal") is not False
        or claims.get("eval_proxy_is_heldout") is not False
    ):
        raise ConfigError("synthetic manifest claims drifted")

    declared = _mapping(manifest.get("producer"), "manifest.producer")
    if (
        _mapping(declared.get("record_schema"), "producer.record_schema").get("sha256")
        != snapshots["record_schema"].sha256
        or _mapping(declared.get("manifest_schema"), "producer.manifest_schema").get(
            "sha256"
        )
        != snapshots["manifest_schema"].sha256
    ):
        raise ConfigError("producer schema binding mismatch")

    entries = manifest.get("partitions")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ConfigError("synthetic manifest must declare four partitions")
    by_path = {
        str(_mapping(entry, "partition").get("path")): _mapping(entry, "partition")
        for entry in entries
    }
    if set(by_path) != set(PARTITION_PATHS):
        raise ConfigError("synthetic partition inventory drifted")

    train: list[ScaffoldExample] = []
    eval_proxy: list[ScaffoldExample] = []
    partition_sha256: dict[str, str] = {}
    partition_snapshots: list[FileSnapshot] = []
    record_ids: set[str] = set()
    split_bundles: dict[str, set[str]] = {"train": set(), "eval_proxy": set()}
    for relative in PARTITION_PATHS:
        entry = by_path[relative]
        split = str(entry.get("split"))
        expected_variant = relative.rsplit("/", 1)[1].removesuffix(".jsonl")
        if (
            split not in split_bundles
            or entry.get("variant") != expected_variant
            or not isinstance(entry.get("records"), int)
            or not isinstance(entry.get("bytes"), int)
        ):
            raise ConfigError(f"partition metadata drifted: {relative}")
        path = qdiag._assert_physical_path(
            root.joinpath(*PurePosixPath(relative).parts),
            require_file=True,
            label=f"synthetic partition {relative}",
        )
        snapshot = _read_snapshot(path, max_bytes=_MAX_PARTITION_BYTES)
        partition_snapshots.append(snapshot)
        if snapshot.sha256 != entry.get("sha256") or len(snapshot.data) != entry.get(
            "bytes"
        ):
            raise ConfigError(f"partition byte identity mismatch: {relative}")
        raw_records = _strict_jsonl(snapshot.data, relative)
        if len(raw_records) != entry.get("records"):
            raise ConfigError(f"partition record count mismatch: {relative}")
        for raw in raw_records:
            record_validator.validate(raw)
            item = _example(raw, expected_split=split)
            if item.variant != expected_variant or item.record_id in record_ids:
                raise ConfigError("synthetic record identity/variant drifted")
            record_ids.add(item.record_id)
            split_bundles[split].add(item.source_bundle_id)
            (train if split == "train" else eval_proxy).append(item)
        partition_sha256[relative] = snapshot.sha256

    if (
        len(train) != dataset["train_records"]
        or len(eval_proxy) != dataset["eval_proxy_records"]
    ):
        raise ConfigError("synthetic train/eval counts drifted")
    if split_bundles["train"] & split_bundles["eval_proxy"]:
        raise ConfigError("source bundle crossed train/eval_proxy split")
    counts = _mapping(manifest.get("counts"), "manifest.counts")
    if counts.get("records") != len(record_ids):
        raise ConfigError("manifest total record count drifted")

    for snapshot in (*snapshots.values(), *partition_snapshots):
        snapshot.assert_unchanged()
    return ScaffoldDataset(
        manifest=manifest,
        manifest_sha256=snapshots["manifest"].sha256,
        partition_sha256=partition_sha256,
        train=tuple(sorted(train, key=lambda item: item.record_id)),
        eval_proxy=tuple(sorted(eval_proxy, key=lambda item: item.record_id)),
    )


def _max_steps(config: Mapping[str, Any], override: int | None) -> int:
    training = _mapping(config.get("training"), "training")
    value = training["default_max_steps"] if override is None else override
    if value not in ALLOWED_STEPS:
        raise ConfigError("--max-steps must be exactly 2 or 20")
    return int(value)


def _output_path(config: Mapping[str, Any], max_steps: int) -> Path:
    template = str(_mapping(config.get("output"), "output")["adapter_dir_template"])
    relative = template.format(max_steps=max_steps)
    allowed = qdiag._project_root_from_module() / "artifacts" / "diagnostics"
    candidate = qdiag._project_root_from_module().joinpath(
        *PurePosixPath(relative).parts
    )
    if candidate == allowed or not candidate.is_relative_to(allowed):
        raise ConfigError("diagnostic output escaped artifacts/diagnostics")
    qdiag._assert_physical_path(candidate.parent, label="diagnostic output parent")
    return candidate


def _preflight_root() -> Path:
    root = qdiag._project_root_from_module().joinpath(
        *PurePosixPath(PREFLIGHT_OUTPUT_ROOT).parts
    )
    qdiag._assert_physical_path(root, label="preflight output root")
    return root


def _assert_exact_regular_files(
    path: Path, expected: frozenset[str], *, label: str
) -> None:
    qdiag._assert_physical_path(path, require_directory=True, label=label)
    observed = {entry.name for entry in path.iterdir()}
    if observed != set(expected):
        raise ConfigError(
            f"{label} exact file inventory drifted: "
            f"expected={sorted(expected)!r}, observed={sorted(observed)!r}"
        )
    for name in sorted(expected):
        qdiag._assert_physical_path(
            path / name, require_file=True, label=f"{label}/{name}"
        )


def _remove_failed_publish(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if qdiag._is_reparse_or_symlink(path):
        path.unlink()
        return
    shutil.rmtree(path)


def _preflight_output_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = qdiag._project_root_from_module() / candidate
    candidate = Path(os.path.abspath(candidate))
    root = _preflight_root()
    if candidate == root or not candidate.is_relative_to(root):
        raise ConfigError(
            f"--preflight-output must be a child of {PREFLIGHT_OUTPUT_ROOT}"
        )
    qdiag._assert_physical_path(candidate.parent, label="preflight output parent")
    if os.path.lexists(candidate):
        raise ConfigError(f"preflight output already exists: {candidate}")
    return candidate


def _preflight_receipt_path(value: str | Path) -> tuple[Path, Path]:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = qdiag._project_root_from_module() / candidate
    candidate = Path(os.path.abspath(candidate))
    root = _preflight_root()
    if (
        candidate.name != PREFLIGHT_FILENAME
        or candidate.parent == root
        or not candidate.is_relative_to(root)
    ):
        raise ConfigError(
            "--preflight-receipt must name preflight.json in a child directory "
            f"of {PREFLIGHT_OUTPUT_ROOT}"
        )
    qdiag._assert_physical_path(candidate, require_file=True, label="preflight receipt")
    sidecar = candidate.with_name(PREFLIGHT_SIDECAR_FILENAME)
    qdiag._assert_physical_path(
        sidecar, require_file=True, label="preflight receipt SHA-256 sidecar"
    )
    _assert_exact_regular_files(
        candidate.parent, _PREFLIGHT_FILES, label="preflight receipt directory"
    )
    return candidate, sidecar


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _validate_preflight_shape(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "ready",
            "identity",
            "max_steps",
            "precision",
            "dataset",
            "tokenizer",
            "token_lengths",
            "model_identity",
            "output_path",
            "gates",
            "claims",
            "audit",
        },
        "preflight receipt",
    )
    if value.get("schema_version") != PREFLIGHT_VERSION:
        raise ConfigError("preflight receipt schema version drifted")
    if value.get("status") not in {
        "passed_tokenizer_only_diagnostic_preflight",
        "blocked_tokenizer_only_diagnostic_preflight",
    } or not isinstance(value.get("ready"), bool):
        raise ConfigError("preflight receipt status drifted")
    if value.get("max_steps") not in ALLOWED_STEPS:
        raise ConfigError("preflight receipt max_steps drifted")
    claims = _mapping(value.get("claims"), "preflight receipt claims")
    if claims != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
        "diagnostic_execution_user_requested": False,
    }:
        raise ConfigError("preflight receipt claims drifted")
    audit = _mapping(value.get("audit"), "preflight receipt audit")
    if (
        any(
            audit.get(field) != 0
            for field in (
                "protected_body_reads",
                "heldout_reads",
                "provider_requests",
                "network_requests",
                "model_loads",
                "gpu_requests",
            )
        )
        or audit.get("tokenizer_loads") != 1
    ):
        raise ConfigError("preflight receipt audit boundary drifted")


def _validate_preflight_directory(
    path: Path, *, receipt_bytes: bytes, receipt_sha256: str
) -> None:
    _assert_exact_regular_files(
        path, _PREFLIGHT_FILES, label="preflight receipt directory"
    )
    receipt = _read_snapshot(path / PREFLIGHT_FILENAME, max_bytes=_MAX_METADATA_BYTES)
    sidecar = _read_snapshot(path / PREFLIGHT_SIDECAR_FILENAME, max_bytes=1024)
    required_sidecar = f"{receipt_sha256}  {PREFLIGHT_FILENAME}\n".encode("ascii")
    if (
        receipt.data != receipt_bytes
        or receipt.sha256 != receipt_sha256
        or sidecar.data != required_sidecar
    ):
        raise RuntimeError("preflight receipt directory authentication failed")


def publish_preflight_receipt(
    value: Mapping[str, Any], output: str | Path
) -> tuple[Path, str]:
    _validate_preflight_shape(value)
    output_path = _preflight_output_path(output)
    receipt_bytes = _canonical_json_bytes(value)
    receipt_sha256 = _sha256(receipt_bytes)
    sidecar_bytes = f"{receipt_sha256}  {PREFLIGHT_FILENAME}\n".encode("ascii")
    with qdiag._adapter_staging_directory(output_path) as staging:
        (staging / PREFLIGHT_FILENAME).write_bytes(receipt_bytes)
        (staging / PREFLIGHT_SIDECAR_FILENAME).write_bytes(sidecar_bytes)
        _validate_preflight_directory(
            staging,
            receipt_bytes=receipt_bytes,
            receipt_sha256=receipt_sha256,
        )
        qdiag._rename_directory_noreplace(staging, output_path)
    try:
        _validate_preflight_directory(
            output_path,
            receipt_bytes=receipt_bytes,
            receipt_sha256=receipt_sha256,
        )
    except Exception:
        _remove_failed_publish(output_path)
        raise
    return output_path / PREFLIGHT_FILENAME, receipt_sha256


def authenticate_preflight_receipt(
    path: str | Path, expected_sha256: str
) -> AuthenticatedPreflightReceipt:
    expected = _require_sha(expected_sha256, "--preflight-receipt-sha256")
    receipt_path, sidecar_path = _preflight_receipt_path(path)
    receipt_snapshot = _read_snapshot(receipt_path, max_bytes=_MAX_METADATA_BYTES)
    sidecar_snapshot = _read_snapshot(sidecar_path, max_bytes=1024)
    if receipt_snapshot.sha256 != expected:
        raise ConfigError("preflight receipt SHA-256 mismatch")
    required_sidecar = f"{expected}  {PREFLIGHT_FILENAME}\n".encode("ascii")
    if sidecar_snapshot.data != required_sidecar:
        raise ConfigError("preflight.json.sha256 is missing or malformed")
    value = _strict_json(receipt_snapshot.data, "preflight receipt")
    _validate_preflight_shape(value)
    return AuthenticatedPreflightReceipt(value, receipt_snapshot, sidecar_snapshot)


def _require_preflight_match(
    authenticated: AuthenticatedPreflightReceipt,
    recomputed: Mapping[str, Any],
) -> None:
    if dict(authenticated.value) != dict(recomputed):
        raise ConfigError(
            "authenticated preflight receipt does not exactly match internal "
            "tokenizer-only recomputation"
        )


def _qdiag_config(config: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    return {
        "schema_version": qdiag.SCHEMA_VERSION,
        "paths": {"project_root": "../.."},
        "model": dict(_mapping(config.get("model"), "model")),
        "lora": {
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": ["q_proj"],
        },
        "training": {
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sequence_length": 128,
            "learning_rate": 0.0001,
            "seed": 1337,
        },
        "dataset": {
            "kind": "inline_toy_plumbing_v1",
            "formal_inputs_allowed": False,
            "heldout_allowed": False,
        },
        "output": {
            "adapter_dir": str(
                _output_path(config, max_steps).relative_to(
                    qdiag._project_root_from_module()
                )
            ).replace("\\", "/")
        },
        "_config_path": config["_config_path"],
    }


def _token_ids(value: object) -> list[int]:
    if isinstance(value, Mapping):
        if set(value) - {"input_ids", "attention_mask", "token_type_ids"}:
            raise RuntimeError("chat template returned unexpected token fields")
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise RuntimeError("chat template did not return one token-ID sequence")
    return value


def tokenize_example(
    tokenizer: Any,
    example: ScaffoldExample,
    *,
    sequence_length: int,
    torch: Any | None = None,
) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": example.prompt}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": example.target},
    ]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "assistant training view is not a prompt-prefix continuation"
        )
    target_tokens = len(full_ids) - len(prompt_ids)
    if target_tokens <= 0:
        raise RuntimeError("assistant target produced no active token")
    if len(full_ids) > sequence_length:
        raise RuntimeError(
            f"record {example.record_id} would truncate target/full view: "
            f"{len(full_ids)} > {sequence_length}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    result: dict[str, Any] = {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": target_tokens,
        "full_tokens": len(full_ids),
    }
    if torch is not None:
        for key in ("input_ids", "attention_mask", "labels"):
            result[key] = torch.tensor([result[key]], dtype=torch.long)
    return result


def _signed_int64_sequence_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, byteorder="big", signed=True))
    return digest.hexdigest()


def _compact_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def token_length_preflight(
    tokenizer: Any, dataset: ScaffoldDataset, *, sequence_length: int
) -> dict[str, Any]:
    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    digest_rows: list[dict[str, str]] = []
    examples = sorted(
        (*dataset.train, *dataset.eval_proxy), key=lambda item: item.record_id
    )
    for example in examples:
        encoded = tokenize_example(
            tokenizer, example, sequence_length=sequence_length, torch=None
        )
        prompt_tokens = int(encoded["prompt_tokens"])
        input_ids = list(encoded["input_ids"])
        labels = list(encoded["labels"])
        full_lengths.append(int(encoded["full_tokens"]))
        prompt_lengths.append(prompt_tokens)
        target_lengths.append(int(encoded["target_tokens"]))
        digest_rows.append(
            {
                "record_id_sha256": _sha256(example.record_id.encode("utf-8")),
                "prompt_ids_sha256": _signed_int64_sequence_sha256(
                    input_ids[:prompt_tokens]
                ),
                "full_ids_sha256": _signed_int64_sequence_sha256(input_ids),
                "labels_sha256": _signed_int64_sequence_sha256(labels),
            }
        )
    return {
        "records": len(full_lengths),
        "sequence_length": sequence_length,
        "minimum_full_tokens": min(full_lengths),
        "maximum_full_tokens": max(full_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_target_tokens": min(target_lengths),
        "maximum_target_tokens": max(target_lengths),
        "truncated_records": 0,
        "target_truncation_detected": False,
        "runtime_versions": {
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "token_view_digest_inventory": {
            "record_order": "record_id_utf8_ascending_v1",
            "sequence_digest_algorithm": "sha256_signed_int64_big_endian_concat_v1",
            "aggregate_algorithm": (
                "sha256_utf8_sort_keys_compact_json_digest_rows_v1"
            ),
            "records": len(digest_rows),
            "digest_rows": digest_rows,
            "aggregate_sha256": _compact_json_sha256(digest_rows),
            "raw_token_ids_emitted": False,
        },
    }


def _load_tokenizer_from_path(path: Path) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_tokenizer(config: Mapping[str, Any]) -> Any:
    return _load_tokenizer_from_path(
        qdiag._resolved_local_model(_qdiag_config(config, 2))
    )


def _prepare_preflight(
    config: Mapping[str, Any], max_steps: int
) -> tuple[dict[str, Any], ScaffoldDataset, Any]:
    config_snapshot = _read_snapshot(
        _canonical_config_path(str(config["_config_path"])),
        max_bytes=_MAX_METADATA_BYTES,
    )
    implementation_snapshot = _read_snapshot(
        _repo_path(IMPLEMENTATION_PATH, IMPLEMENTATION_PATH),
        max_bytes=_MAX_METADATA_BYTES,
    )
    dataset = load_dataset(config)
    base_config = _qdiag_config(config, max_steps)
    base_preflight = qdiag.build_preflight(base_config)
    tokenizer = _load_tokenizer(config)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("authenticated tokenizer exposes no chat template")
    token_report = token_length_preflight(
        tokenizer,
        dataset,
        sequence_length=int(config["training"]["sequence_length"]),
    )
    relevant_base_gates = {
        key: bool(base_preflight["gates"][key])
        for key in (
            "network_disabled",
            "hf_directory_physical",
            "source_git_identity",
            "hf_config_closed_and_qwen2",
            "tokenizer_chat_template_present",
            "required_artifacts_physical",
            "expected_artifact_hashes_present",
            "artifact_hashes_match",
            "gguf_rejected",
            "output_scoped_and_absent",
        )
    }
    gates = {
        **relevant_base_gates,
        "synthetic_manifest_authenticated": True,
        "train_eval_source_disjoint": True,
        "record_count_100": len(dataset.train) + len(dataset.eval_proxy) == 100,
        "target_truncation_absent": token_report["truncated_records"] == 0,
        "q_only_rank4": True,
        "steps_allowlisted": max_steps in ALLOWED_STEPS,
        "bf16_tf32_gradient_checkpointing": True,
        "formal_and_training_authority_false": True,
    }
    dataset_config = _mapping(config.get("dataset"), "dataset")
    partitions = [
        {
            key: item[key]
            for key in ("path", "split", "variant", "records", "bytes", "sha256")
        }
        for item in dataset.manifest["partitions"]
    ]
    artifact_identity = _mapping(
        base_preflight["artifact_identity"], "base artifact identity"
    )

    def special_token_id(name: str) -> int | None:
        value = getattr(tokenizer, name, None)
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )

    report = {
        "schema_version": PREFLIGHT_VERSION,
        "status": (
            "passed_tokenizer_only_diagnostic_preflight"
            if all(gates.values())
            else "blocked_tokenizer_only_diagnostic_preflight"
        ),
        "ready": all(gates.values()),
        "identity": {
            "diagnostic_schema_version": SCHEMA_VERSION,
            "diagnostic_config_schema_version": CONFIG_VERSION,
            "dataset_manifest_schema_version": DATASET_VERSION,
            "dataset_record_schema_version": RECORD_VERSION,
            "config": {
                "path": CONFIG_PATH,
                "logical_sha256": config_fingerprint(config),
                "physical_sha256": config_snapshot.sha256,
            },
            "implementation": {
                "path": IMPLEMENTATION_PATH,
                "sha256": implementation_snapshot.sha256,
            },
        },
        "max_steps": max_steps,
        "precision": {
            **dict(_mapping(config.get("precision"), "precision")),
            "sequence_length": int(config["training"]["sequence_length"]),
            "batch_size": int(config["training"]["batch_size"]),
            "gradient_checkpointing": bool(
                config["training"]["gradient_checkpointing"]
            ),
        },
        "dataset": {
            "root": DATASET_ROOT,
            "manifest_sha256": dataset.manifest_sha256,
            "record_schema_sha256": dataset_config["expected_record_schema_sha256"],
            "manifest_schema_sha256": dataset_config["expected_manifest_schema_sha256"],
            "partitions": partitions,
            "partition_sha256": dict(dataset.partition_sha256),
            "counts": {
                "train": len(dataset.train),
                "eval_proxy": len(dataset.eval_proxy),
                "total": len(dataset.train) + len(dataset.eval_proxy),
            },
        },
        "tokenizer": {
            "model_id": config["model"]["id"],
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "chat_template_sha256": _sha256(chat_template.encode("utf-8")),
            "chat_template_utf8_bytes": len(chat_template.encode("utf-8")),
            "tokenizer_json_sha256": artifact_identity["tokenizer.json"][
                "observed_sha256"
            ],
            "tokenizer_config_sha256": artifact_identity["tokenizer_config.json"][
                "observed_sha256"
            ],
            "special_token_ids": {
                name.removesuffix("_token_id"): special_token_id(name)
                for name in (
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "unk_token_id",
                )
            },
        },
        "token_lengths": token_report,
        "model_identity": {
            "model_config_sha256": base_preflight["model_config_sha256"],
            "source": base_preflight["source_identity"],
            "artifacts": base_preflight["artifact_identity"],
        },
        "output_path": str(
            _output_path(config, max_steps).relative_to(
                qdiag._project_root_from_module()
            )
        ).replace("\\", "/"),
        "gates": gates,
        "claims": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
            "diagnostic_execution_user_requested": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "tokenizer_loads": 1,
        },
    }
    _validate_preflight_shape(report)
    config_snapshot.assert_unchanged()
    implementation_snapshot.assert_unchanged()
    return report, dataset, tokenizer


def build_preflight(config: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    report, _dataset, _tokenizer = _prepare_preflight(config, max_steps)
    return report


def _cuda_batch(
    tokenizer: Any, example: ScaffoldExample, sequence_length: int, torch: Any
) -> dict[str, Any]:
    encoded = tokenize_example(
        tokenizer, example, sequence_length=sequence_length, torch=torch
    )
    return {
        key: encoded[key].to("cuda")
        for key in ("input_ids", "attention_mask", "labels")
    }


def _mean_eval_loss(
    model: Any,
    tokenizer: Any,
    examples: Sequence[ScaffoldExample],
    sequence_length: int,
    torch: Any,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            batch = _cuda_batch(tokenizer, example, sequence_length, torch)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("eval_proxy produced non-finite loss")
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("eval_proxy partition is empty")
    return sum(losses) / len(losses)


def _validate_diagnostic_artifact(
    path: Path,
    *,
    expected_adapter_hashes: Mapping[str, str],
    expected_receipt_sha256: str,
) -> None:
    _assert_exact_regular_files(
        path, _DIAGNOSTIC_FILES, label="diagnostic adapter directory"
    )
    observed_adapter_hashes = qdiag._validate_saved_adapter(
        path, rank=4, expected_layers=28, hidden_size=1536
    )
    if dict(observed_adapter_hashes) != dict(expected_adapter_hashes):
        raise RuntimeError("diagnostic adapter bytes changed before publication")
    receipt = _read_snapshot(
        path / "diagnostic_receipt.json", max_bytes=_MAX_METADATA_BYTES
    )
    sidecar = _read_snapshot(path / "diagnostic_receipt.json.sha256", max_bytes=1024)
    if receipt.sha256 != expected_receipt_sha256 or sidecar.data != (
        f"{expected_receipt_sha256}  diagnostic_receipt.json\n".encode("ascii")
    ):
        raise RuntimeError("diagnostic receipt or sidecar authentication failed")
    value = _strict_json(receipt.data, "diagnostic receipt")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "passed_diagnostic_only"
        or value.get("adapter_artifact_sha256") != dict(expected_adapter_hashes)
    ):
        raise RuntimeError("diagnostic receipt content does not bind adapter bytes")


def _publish_verified_diagnostic(
    staging: Path,
    output_path: Path,
    *,
    expected_adapter_hashes: Mapping[str, str],
    expected_receipt_sha256: str,
) -> None:
    _validate_diagnostic_artifact(
        staging,
        expected_adapter_hashes=expected_adapter_hashes,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    qdiag._rename_directory_noreplace(staging, output_path)
    try:
        _validate_diagnostic_artifact(
            output_path,
            expected_adapter_hashes=expected_adapter_hashes,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    except Exception:
        _remove_failed_publish(output_path)
        raise


def execute_diagnostic(
    config: Mapping[str, Any],
    max_steps: int,
    *,
    preflight_receipt: str | Path,
    preflight_receipt_sha256: str,
) -> dict[str, Any]:
    authenticated_preflight = authenticate_preflight_receipt(
        preflight_receipt, preflight_receipt_sha256
    )
    preflight, dataset, _preflight_tokenizer = _prepare_preflight(config, max_steps)
    _require_preflight_match(authenticated_preflight, preflight)
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["gates"].items() if not passed]
        raise RuntimeError(
            "synthetic diagnostic preflight failed: " + ", ".join(failed)
        )
    authenticated_preflight.assert_unchanged()
    output_path = _output_path(config, max_steps)
    base_config = _qdiag_config(config, max_steps)
    with qdiag._authenticated_model_snapshot(base_config, output_path.parent) as (
        model_path,
        snapshot_identity,
    ):
        authenticated_tokenizer = _load_tokenizer_from_path(model_path)
        authenticated_token_report = token_length_preflight(
            authenticated_tokenizer,
            dataset,
            sequence_length=int(config["training"]["sequence_length"]),
        )
        if authenticated_token_report != preflight["token_lengths"]:
            raise RuntimeError(
                "authenticated snapshot tokenizer disagrees with preflight"
            )
        return _execute_authenticated(
            config,
            max_steps=max_steps,
            preflight=preflight,
            dataset=dataset,
            tokenizer=authenticated_tokenizer,
            model_path=model_path,
            snapshot_identity=snapshot_identity,
            output_path=output_path,
            authenticated_preflight=authenticated_preflight,
        )


def _execute_authenticated(
    config: Mapping[str, Any],
    *,
    max_steps: int,
    preflight: Mapping[str, Any],
    dataset: ScaffoldDataset,
    tokenizer: Any,
    model_path: Path,
    snapshot_identity: Mapping[str, Any],
    output_path: Path,
    authenticated_preflight: AuthenticatedPreflightReceipt,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("synthetic diagnostic requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not (torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
        raise RuntimeError("TF32 controls did not remain enabled")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    base_config = _qdiag_config(config, max_steps)
    source_model_path = qdiag._resolved_local_model(base_config)
    if dict(
        qdiag._snapshot_identity(base_config, source_model_path, model_path)
    ) != dict(snapshot_identity):
        raise RuntimeError("authenticated model snapshot changed while loading")
    for parameter in base.parameters():
        parameter.requires_grad = False
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    captured, base_hash_before = qdiag._capture_base_parameters(base, torch)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_names, trainable_parameters = qdiag._assert_q_proj_lora_trainable(
        model, expected_layers=28, hidden_size=1536, rank=4
    )
    sequence_length = int(config["training"]["sequence_length"])
    eval_loss_before = _mean_eval_loss(
        model, tokenizer, dataset.eval_proxy, sequence_length, torch
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    selected_train_record_ids = tuple(
        dataset.train[step % len(dataset.train)].record_id for step in range(max_steps)
    )
    losses: list[float] = []
    gradient_coverage: Mapping[str, Any] = {}
    for step, selected_record_id in enumerate(selected_train_record_ids):
        model.train()
        selected_example = dataset.train[step % len(dataset.train)]
        if selected_example.record_id != selected_record_id:
            raise RuntimeError("selected train record order changed")
        batch = _cuda_batch(
            tokenizer,
            selected_example,
            sequence_length,
            torch,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic training produced non-finite loss")
        loss.backward()
        gradient_coverage = qdiag._assert_nonzero_lora_gradients(model, torch)
        optimizer.step()
        qdiag._assert_trainable_parameters_finite(model, torch)
        losses.append(float(loss.detach().cpu()))
    eval_loss_after = _mean_eval_loss(
        model, tokenizer, dataset.eval_proxy, sequence_length, torch
    )
    if not all(
        math.isfinite(value) for value in (*losses, eval_loss_before, eval_loss_after)
    ):
        raise RuntimeError("diagnostic loss metrics are non-finite")
    base_hash_after = qdiag._assert_base_parameters_unchanged(captured, torch)
    probe_batch = _cuda_batch(tokenizer, dataset.eval_proxy[0], sequence_length, torch)
    trained_logits = qdiag._next_token_logits(model, probe_batch, torch)
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("PEFT runtime does not expose adapter-off context")
    with model.disable_adapter():
        adapter_off_logits = qdiag._next_token_logits(model, probe_batch, torch)
    adapter_effect = float(
        torch.max(torch.abs(trained_logits - adapter_off_logits)).item()
    )
    if not math.isfinite(adapter_effect) or adapter_effect <= 0:
        raise RuntimeError("trained q_only adapter produced no observable effect")

    with qdiag._adapter_staging_directory(output_path) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        peft_readme = staging / "README.md"
        if os.path.lexists(peft_readme):
            qdiag._assert_physical_path(
                peft_readme, require_file=True, label="generated PEFT README"
            )
            peft_readme.unlink()
        qdiag._normalize_saved_adapter_base_identity(staging)
        artifact_hashes = qdiag._validate_saved_adapter(
            staging, rank=4, expected_layers=28, hidden_size=1536
        )
        del optimizer, model, base, captured
        gc.collect()
        torch.cuda.empty_cache()
        if dict(
            qdiag._snapshot_identity(base_config, source_model_path, model_path)
        ) != dict(snapshot_identity):
            raise RuntimeError("authenticated model snapshot changed before reload")
        reload_base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(
            "cuda"
        )
        _, reloaded_base_hash = qdiag._capture_base_parameters(reload_base, torch)
        if reloaded_base_hash != base_hash_before:
            raise RuntimeError("fresh reload base hash differs from pre-training base")
        reloaded = PeftModel.from_pretrained(
            reload_base, staging, is_trainable=False, local_files_only=True
        )
        reloaded_logits = qdiag._next_token_logits(reloaded, probe_batch, torch)
        reload_delta = float(
            torch.max(torch.abs(trained_logits - reloaded_logits)).item()
        )
        if not math.isfinite(reload_delta) or reload_delta > 1e-4:
            raise RuntimeError("save/reload logits gate failed")
        authenticated_preflight.assert_unchanged()
        train_order_bytes = ("\n".join(selected_train_record_ids) + "\n").encode(
            "utf-8"
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed_diagnostic_only",
            "config_sha256": config_fingerprint(config),
            "max_steps": max_steps,
            "train_loss_first": losses[0],
            "train_loss_last": losses[-1],
            "eval_proxy_loss_before": eval_loss_before,
            "eval_proxy_loss_after": eval_loss_after,
            "eval_proxy_loss_delta": eval_loss_after - eval_loss_before,
            "base_hash_before": base_hash_before,
            "base_hash_after": base_hash_after,
            "reloaded_base_hash": reloaded_base_hash,
            "trainable_tensor_names": trainable_names,
            "trainable_parameters": trainable_parameters,
            "gradient_coverage": gradient_coverage,
            "adapter_artifact_sha256": artifact_hashes,
            "authenticated_preflight": {
                "schema_version": PREFLIGHT_VERSION,
                "receipt_sha256": authenticated_preflight.receipt_snapshot.sha256,
                "token_lengths": dict(preflight["token_lengths"]),
                "runner_implementation_sha256": preflight["identity"]["implementation"][
                    "sha256"
                ],
            },
            "train_record_order": {
                "algorithm": "sha256_lf_terminated_ordered_utf8_record_ids_v1",
                "record_ids": list(selected_train_record_ids),
                "record_ids_sha256": _sha256(train_order_bytes),
                "optimizer_steps": max_steps,
            },
            "max_abs_adapter_effect": adapter_effect,
            "max_abs_reload_logit_delta": reload_delta,
            "dataset": {
                "manifest_sha256": dataset.manifest_sha256,
                "partition_sha256": dict(dataset.partition_sha256),
                "train_records": len(dataset.train),
                "eval_proxy_records": len(dataset.eval_proxy),
                "eval_proxy_is_heldout": False,
            },
            "precision": {
                "compute_dtype": "bfloat16",
                "tf32": True,
                "gradient_checkpointing": True,
                "sequence_length": sequence_length,
                "batch_size": 1,
            },
            "model_identity": {
                **dict(preflight["model_identity"]),
                "private_snapshot": dict(snapshot_identity),
            },
            "claims": {
                "diagnostic_only": True,
                "diagnostic_execution_user_requested": True,
                "training_authorized": False,
                "formal": False,
                "quality_validated": False,
            },
            "audit": {
                "protected_body_reads": 0,
                "heldout_reads": 0,
                "provider_requests": 0,
                "network_requests": 0,
                "model_loads": 2,
                "gpu_requests": 1,
                "tokenizer_loads": 2,
            },
        }
        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_sha256 = _sha256(receipt_bytes)
        (staging / "diagnostic_receipt.json").write_bytes(receipt_bytes)
        (staging / "diagnostic_receipt.json.sha256").write_bytes(
            f"{receipt_sha256}  diagnostic_receipt.json\n".encode("ascii")
        )
        _publish_verified_diagnostic(
            staging,
            output_path,
            expected_adapter_hashes=artifact_hashes,
            expected_receipt_sha256=receipt_sha256,
        )
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B synthetic scaffold q_only diagnostic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, choices=ALLOWED_STEPS)
    parser.add_argument("--preflight-output")
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--preflight-receipt-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    max_steps = _max_steps(config, args.max_steps)
    if args.execute:
        if args.preflight_output is not None:
            parser.error("--execute forbids --preflight-output")
        if not args.preflight_receipt or not args.preflight_receipt_sha256:
            parser.error(
                "--execute requires --preflight-receipt and --preflight-receipt-sha256"
            )
        result = execute_diagnostic(
            config,
            max_steps,
            preflight_receipt=args.preflight_receipt,
            preflight_receipt_sha256=args.preflight_receipt_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.preflight_output:
        parser.error("--dry-run requires explicit --preflight-output")
    if args.preflight_receipt is not None or args.preflight_receipt_sha256 is not None:
        parser.error("--dry-run forbids preflight receipt input arguments")
    result = build_preflight(config, max_steps)
    receipt_path, receipt_sha256 = publish_preflight_receipt(
        result, args.preflight_output
    )
    published = {
        "preflight": result,
        "published_receipt": {
            "path": str(
                receipt_path.relative_to(qdiag._project_root_from_module())
            ).replace("\\", "/"),
            "sha256": receipt_sha256,
        },
    }
    print(json.dumps(published, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
