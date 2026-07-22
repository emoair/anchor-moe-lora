"""CPU-only static audit for Q/O LoRA writeback and memorization risk signals.

This module deliberately does not execute a model forward pass.  It authenticates
the frozen diagnostic inputs, computes exact low-rank delta singular values from a
thin QR core, and compares each O-projection output subspace with four token-
embedding controls.  The result is a correlation-only diagnostic, never proof of
memorization or formal training authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training.config import ConfigError, _expand_env


CONFIG_VERSION = "anchor.qwen25-qo-spectral-memory-audit-config.v1"
RECEIPT_VERSION = "anchor.qwen25-qo-spectral-memory-audit-receipt.v1"
CONFIG_PATH = "configs/research/qwen_qo_spectral_memory_audit_v1.yaml"
CONFIG_SCHEMA_PATH = "configs/research/qwen_qo_spectral_memory_audit_config.schema.json"
RECEIPT_SCHEMA_PATH = (
    "configs/research/qwen_qo_spectral_memory_audit_receipt.schema.json"
)
IMPLEMENTATION_PATH = "src/anchor_mvp/research/qwen_qo_spectral_memory_audit.py"
WRAPPER_PATH = "scripts/research/audit_qwen_qo_spectral_memory.py"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_LORA_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$"
)
_MAX_SMALL_BYTES = 16_000_000
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class SpectralMemoryAuditError(RuntimeError):
    """Stable fail-closed error for the static diagnostic."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self) -> None:
        if _stat_identity(self.path.stat()) != self.identity:
            raise SpectralMemoryAuditError(f"input changed after snapshot: {self.path}")


def _root() -> Path:
    return qdiag._project_root_from_module()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json_sha256(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(data)


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise SpectralMemoryAuditError("non-finite metric")
    return round(float(value), 12)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _snapshot(path: Path, *, max_bytes: int = _MAX_SMALL_BYTES) -> _Snapshot:
    qdiag._assert_physical_path(path, require_file=True, label=str(path))
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_bytes:
            raise SpectralMemoryAuditError(f"small input exceeds cap: {path}")
        data = handle.read(max_bytes + 1)
        after = os.fstat(handle.fileno())
    path_after = path.stat()
    identity = _stat_identity(after)
    if (
        len(data) > max_bytes
        or len(data) != after.st_size
        or _stat_identity(before) != identity
        or _stat_identity(path_after) != identity
    ):
        raise SpectralMemoryAuditError(f"input changed while reading: {path}")
    return _Snapshot(path, data, _sha256(data), identity)


def _stream_sha256(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    qdiag._assert_physical_path(path, require_file=True, label=str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity = _stat_identity(after)
    if _stat_identity(before) != identity or _stat_identity(path.stat()) != identity:
        raise SpectralMemoryAuditError(f"large input changed while hashing: {path}")
    return digest.hexdigest(), identity


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SpectralMemoryAuditError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpectralMemoryAuditError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise SpectralMemoryAuditError(f"{label} root must be an object")
    return value


def _strict_jsonl(data: bytes, label: str) -> list[Mapping[str, Any]]:
    if b"\r" in data or not data.endswith(b"\n"):
        raise SpectralMemoryAuditError(f"{label} must be LF-only JSONL")
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise SpectralMemoryAuditError(f"{label} contains a blank line")
        records.append(_strict_json(line, f"{label}:{index}"))
    return records


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ConfigError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ConfigError(f"{label} fields drifted")


def _canonical_config(path: str | Path) -> Path:
    canonical = _root() / CONFIG_PATH
    requested = Path(path)
    resolved = requested if requested.is_absolute() else _root() / requested
    if os.path.normcase(str(resolved.resolve())) != os.path.normcase(
        str(canonical.resolve())
    ):
        raise ConfigError(f"config must remain exactly {CONFIG_PATH}")
    qdiag._assert_physical_path(canonical, require_file=True, label="audit config")
    return canonical


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = _canonical_config(path)
    snapshot = _snapshot(canonical, max_bytes=2_000_000)
    if b"\r" in snapshot.data:
        raise ConfigError("audit config must be LF-only")
    try:
        raw = yaml.load(snapshot.data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("audit config is not strict UTF-8 YAML") from exc
    if not isinstance(raw, dict):
        raise ConfigError("audit config root must be a mapping")
    schema = _strict_json(
        _snapshot(_root() / CONFIG_SCHEMA_PATH).data, "audit config schema"
    )
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(raw), key=lambda item: list(item.path)
    )
    if errors:
        raise ConfigError(f"audit config schema failed: {errors[0].message}")
    config = _expand_env(raw)
    config["_config_path"] = str(canonical)
    config["_config_sha256"] = snapshot.sha256
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "model",
            "dataset",
            "comparison",
            "adapters",
            "metrics",
            "output",
            "claims",
            "_config_path",
            "_config_sha256",
        },
        "config",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "synthetic_proxy_static_memory_risk_diagnostic_only"
        or config.get("paths") != {"project_root": "../.."}
    ):
        raise ConfigError("top-level audit contract drifted")
    model = _mapping(config.get("model"), "model")
    _exact_keys(
        model,
        {
            "id",
            "local_path",
            "local_files_only",
            "allow_network",
            "trust_remote_code",
            "expected_source_revision",
            "hidden_size",
            "layers",
            "embedding_tensor",
            "assets",
        },
        "model",
    )
    if (
        model.get("id") != "Qwen/Qwen2.5-1.5B-Instruct"
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("hidden_size") != 1536
        or model.get("layers") != 28
        or model.get("embedding_tensor") != "model.embed_tokens.weight"
    ):
        raise ConfigError("model/network contract drifted")
    assets = _mapping(model.get("assets"), "model.assets")
    _exact_keys(
        assets,
        {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        },
        "model.assets",
    )
    for name, digest in assets.items():
        _require_sha(digest, f"model.assets.{name}")
    dataset = _mapping(config.get("dataset"), "dataset")
    _exact_keys(
        dataset,
        {
            "root",
            "manifest",
            "manifest_sidecar",
            "expected_manifest_sha256",
            "expected_manifest_sidecar_sha256",
            "train_partitions",
            "expected_train_records",
            "prompt_field",
            "target_field",
            "eval_proxy_reads_allowed",
            "heldout_reads_allowed",
            "protected_body_reads_allowed",
        },
        "dataset",
    )
    if (
        dataset.get("expected_train_records") != 80
        or dataset.get("prompt_field") != "input.materialized_prompt"
        or dataset.get("target_field") != "target.serialized_assistant_output"
        or dataset.get("eval_proxy_reads_allowed") is not False
        or dataset.get("heldout_reads_allowed") is not False
        or dataset.get("protected_body_reads_allowed") is not False
        or len(dataset.get("train_partitions", [])) != 2
    ):
        raise ConfigError("dataset read boundary drifted")
    for field in ("expected_manifest_sha256", "expected_manifest_sidecar_sha256"):
        _require_sha(dataset.get(field), f"dataset.{field}")
    expected_partitions = {
        "train/json_only.jsonl": (
            40,
            "3ea3cf57e9990b2b07e98cd2bf27a620ed3d2eb2bfd495b0f89ae3d22cce60df",
        ),
        "train/concise_rationale_plus_json.jsonl": (
            40,
            "ff656ff6d2b5303880e5a5ec8db05ffa33e7eca7ca3b1bbd78787a1bd28f1852",
        ),
    }
    observed_partitions: dict[str, tuple[int, str]] = {}
    for item in dataset["train_partitions"]:
        partition = _mapping(item, "dataset.train_partitions[]")
        _exact_keys(partition, {"path", "records", "sha256"}, "train partition")
        _require_sha(partition.get("sha256"), "train partition sha256")
        observed_partitions[str(partition.get("path"))] = (
            int(partition.get("records", 0)),
            str(partition.get("sha256")),
        )
    if observed_partitions != expected_partitions:
        raise ConfigError("training partition identities drifted")
    comparison = _mapping(config.get("comparison"), "comparison")
    _exact_keys(
        comparison,
        {
            "path",
            "sha256",
            "source_training_config",
            "source_training_config_sha256",
        },
        "comparison",
    )
    _require_sha(comparison.get("sha256"), "comparison.sha256")
    _require_sha(
        comparison.get("source_training_config_sha256"),
        "comparison.source_training_config_sha256",
    )
    adapters = _mapping(config.get("adapters"), "adapters")
    if set(adapters) != {"q_plus_o", "wide_budget_matched"}:
        raise ConfigError("adapter set drifted")
    expected = {
        "q_plus_o": (
            {"q_proj", "o_proj"},
            {"q_proj": 8, "o_proj": 8},
            {"q_proj": 16, "o_proj": 16},
        ),
        "wide_budget_matched": (
            {"q_proj", "o_proj", "k_proj", "v_proj"},
            {"q_proj": 5, "o_proj": 4, "k_proj": 6, "v_proj": 6},
            {"q_proj": 10, "o_proj": 8, "k_proj": 12, "v_proj": 12},
        ),
    }
    for name, (targets, ranks, alphas) in expected.items():
        item = _mapping(adapters.get(name), f"adapters.{name}")
        _exact_keys(
            item,
            {"path", "target_modules", "ranks", "alphas", "files"},
            f"adapters.{name}",
        )
        if (
            set(item.get("target_modules", [])) != targets
            or item.get("ranks") != ranks
            or item.get("alphas") != alphas
        ):
            raise ConfigError(f"adapter scope/rank drifted: {name}")
        files = _mapping(item.get("files"), f"adapters.{name}.files")
        _exact_keys(
            files,
            {
                "adapter_config.json",
                "adapter_model.safetensors",
                "diagnostic_receipt.json",
            },
            f"adapters.{name}.files",
        )
        for file_name, digest in files.items():
            _require_sha(digest, f"adapters.{name}.files.{file_name}")
    metrics = _mapping(config.get("metrics"), "metrics")
    if metrics != {
        "top_k_energy": [1, 2, 4],
        "token_group_size": 128,
        "random_vocab_seed": "anchor.qwen-qo-spectral-memory-audit.v1",
        "token_group_policy": (
            "target_frequency_prompt_only_low_frequency_random_unseen_v1"
        ),
        "ordered_token_id_digest": "sha256_signed_int64_big_endian_ordered_v1",
        "exclude_special_tokens": True,
        "final_model_rehash": True,
    }:
        raise ConfigError("metric contract drifted")
    if config.get("claims") != {
        "diagnostic_only": True,
        "controlled_proxy_only": True,
        "formal": False,
        "training_authorized": False,
        "eval_proxy_is_heldout": False,
        "memorization_proven": False,
        "exploit_code_memorization_tested": False,
        "causal_attribution_proven": False,
    }:
        raise ConfigError("claim boundary drifted")
    if config.get("output") != {
        "path": "artifacts/diagnostics/qwen_qo_spectral_memory_audit_v1",
        "receipt_name": "receipt.json",
    }:
        raise ConfigError("output contract drifted")


def _repo_path(value: str) -> Path:
    path = (_root() / value).resolve()
    if _root().resolve() not in path.parents:
        raise SpectralMemoryAuditError(f"repo path escaped project root: {value}")
    return path


def _file_binding(snapshot: _Snapshot, *, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "bytes": len(snapshot.data),
        "sha256": snapshot.sha256,
    }


def _ordered_int64_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack(">q", int(value)))
    return digest.hexdigest()


def _nested_string(record: Mapping[str, Any], dotted: str) -> str:
    value: object = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise SpectralMemoryAuditError(f"dataset field missing: {dotted}")
        value = value[part]
    if not isinstance(value, str) or not value:
        raise SpectralMemoryAuditError(f"dataset field is not text: {dotted}")
    return value


def _authenticate_dataset(
    config: Mapping[str, Any], snapshots: list[_Snapshot]
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    dataset = _mapping(config["dataset"], "dataset")
    manifest_path = _repo_path(str(dataset["manifest"]))
    sidecar_path = _repo_path(str(dataset["manifest_sidecar"]))
    manifest_snapshot = _snapshot(manifest_path)
    sidecar_snapshot = _snapshot(sidecar_path)
    snapshots.extend((manifest_snapshot, sidecar_snapshot))
    if (
        manifest_snapshot.sha256 != dataset["expected_manifest_sha256"]
        or sidecar_snapshot.sha256 != dataset["expected_manifest_sidecar_sha256"]
        or sidecar_snapshot.data
        != f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    ):
        raise SpectralMemoryAuditError("dataset manifest identity drifted")
    manifest = _strict_json(manifest_snapshot.data, "dataset manifest")
    declared = {
        item["path"]: item
        for item in manifest.get("partitions", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    root = _repo_path(str(dataset["root"]))
    records: list[Mapping[str, Any]] = []
    partition_bindings: list[dict[str, Any]] = []
    for item in dataset["train_partitions"]:
        spec = _mapping(item, "dataset.train_partitions[]")
        relative = str(spec["path"])
        path = root / relative
        snapshot = _snapshot(path)
        snapshots.append(snapshot)
        manifest_item = declared.get(relative)
        if (
            snapshot.sha256 != spec["sha256"]
            or not isinstance(manifest_item, Mapping)
            or manifest_item.get("sha256") != snapshot.sha256
            or manifest_item.get("records") != spec["records"]
            or manifest_item.get("split") != "train"
        ):
            raise SpectralMemoryAuditError(
                f"train partition identity drifted: {relative}"
            )
        parsed = _strict_jsonl(snapshot.data, relative)
        if len(parsed) != spec["records"]:
            raise SpectralMemoryAuditError(f"train partition count drifted: {relative}")
        records.extend(parsed)
        partition_bindings.append(
            {
                "path": relative,
                "records": len(parsed),
                "bytes": len(snapshot.data),
                "sha256": snapshot.sha256,
            }
        )
    if len(records) != dataset["expected_train_records"]:
        raise SpectralMemoryAuditError("training record count drifted")
    seen: set[str] = set()
    text_pairs: list[tuple[str, str]] = []
    hashed_ids: list[str] = []
    for record in records:
        record_id = record.get("record_id")
        if (
            not isinstance(record_id, str)
            or record_id in seen
            or record.get("split") != "train"
        ):
            raise SpectralMemoryAuditError("training record identity/split drifted")
        seen.add(record_id)
        hashed_ids.append(_sha256(record_id.encode("utf-8")))
        text_pairs.append(
            (
                _nested_string(record, str(dataset["prompt_field"])),
                _nested_string(record, str(dataset["target_field"])),
            )
        )
    return text_pairs, {
        "manifest": _file_binding(
            manifest_snapshot, logical_path=str(dataset["manifest"])
        ),
        "manifest_sidecar": _file_binding(
            sidecar_snapshot, logical_path=str(dataset["manifest_sidecar"])
        ),
        "train_partitions": partition_bindings,
        "train_records": len(records),
        "training_record_id_inventory_sha256": _compact_json_sha256(sorted(hashed_ids)),
        "raw_record_ids_emitted": False,
        "raw_sample_text_emitted": False,
    }


def _frequency_order(counter: Counter[int], *, descending: bool) -> list[int]:
    if descending:
        return [
            item
            for item, _ in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
    return [
        item for item, _ in sorted(counter.items(), key=lambda pair: (pair[1], pair[0]))
    ]


def select_token_groups(
    target_counts: Counter[int],
    prompt_counts: Counter[int],
    *,
    special_ids: set[int],
    vocab_size: int,
    group_size: int,
    random_seed: str,
) -> dict[str, list[int]]:
    """Select four deterministic token controls without exposing their IDs."""

    target_frequent = [
        item
        for item in _frequency_order(target_counts, descending=True)
        if item not in special_ids
    ][:group_size]
    target_vocab = set(target_counts)
    prompt_control = [
        item
        for item in _frequency_order(prompt_counts, descending=True)
        if item not in special_ids and item not in target_vocab
    ][:group_size]
    frequent_set = set(target_frequent)
    target_low_frequency = [
        item
        for item in _frequency_order(target_counts, descending=False)
        if item not in special_ids and item not in frequent_set
    ][:group_size]
    blocked = target_vocab | set(prompt_counts) | special_ids
    random_candidates = (item for item in range(vocab_size) if item not in blocked)
    deterministic_random = sorted(
        random_candidates,
        key=lambda item: hashlib.sha256(
            random_seed.encode("utf-8") + b"\0" + str(item).encode("ascii")
        ).digest(),
    )[:group_size]
    groups = {
        "target_frequent": target_frequent,
        "prompt_control": prompt_control,
        "target_low_frequency": target_low_frequency,
        "deterministic_random_vocab": deterministic_random,
    }
    if any(len(values) != group_size for values in groups.values()):
        raise SpectralMemoryAuditError("token group cannot satisfy fixed size")
    if (
        frequent_set & set(prompt_control)
        or frequent_set & set(target_low_frequency)
        or set(deterministic_random) & (target_vocab | set(prompt_counts))
    ):
        raise SpectralMemoryAuditError("token control groups escaped disjoint policy")
    return groups


def _tokenize_groups(
    config: Mapping[str, Any], text_pairs: Sequence[tuple[str, str]]
) -> tuple[dict[str, list[int]], dict[str, Any], Any]:
    from transformers import AutoTokenizer

    model_path = Path(str(config["model"]["local_path"]))
    if not model_path.is_absolute():
        model_path = (_root() / model_path).resolve()
    qdiag._assert_physical_path(
        model_path, require_directory=True, label="local tokenizer directory"
    )
    old_hf = os.environ.get("HF_HUB_OFFLINE")
    old_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
    finally:
        if old_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf
        if old_transformers is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers
    prompt_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    for prompt, target in text_pairs:
        prompt_counts.update(tokenizer.encode(prompt, add_special_tokens=False))
        target_counts.update(tokenizer.encode(target, add_special_tokens=False))
    metrics = config["metrics"]
    groups = select_token_groups(
        target_counts,
        prompt_counts,
        special_ids=set(tokenizer.all_special_ids),
        vocab_size=len(tokenizer),
        group_size=int(metrics["token_group_size"]),
        random_seed=str(metrics["random_vocab_seed"]),
    )
    group_report: dict[str, Any] = {}
    for name, values in groups.items():
        counts = [
            int(
                target_counts[item]
                if name.startswith("target_")
                else prompt_counts[item]
            )
            for item in values
        ]
        if name == "deterministic_random_vocab":
            counts = [0 for _ in values]
        group_report[name] = {
            "count": len(values),
            "ordered_token_id_inventory_sha256": _ordered_int64_sha256(values),
            "frequency_total": sum(counts),
            "frequency_min": min(counts),
            "frequency_max": max(counts),
        }
    group_report["policy"] = str(metrics["token_group_policy"])
    group_report["ordered_token_id_digest_algorithm"] = str(
        metrics["ordered_token_id_digest"]
    )
    group_report["random_vocab_seed_sha256"] = _sha256(
        str(metrics["random_vocab_seed"]).encode("utf-8")
    )
    group_report["prompt_token_occurrences"] = sum(prompt_counts.values())
    group_report["target_token_occurrences"] = sum(target_counts.values())
    group_report["prompt_unique_tokens"] = len(prompt_counts)
    group_report["target_unique_tokens"] = len(target_counts)
    group_report["tokenizer_vocab_size"] = len(tokenizer)
    group_report["special_token_count"] = len(set(tokenizer.all_special_ids))
    group_report["raw_token_ids_emitted"] = False
    return groups, group_report, tokenizer


def _thin_delta_singular_values(a: Any, b: Any, scale: float) -> Any:
    """Return exact non-zero singular values of ``scale * B @ A``.

    The spectral norm is intentionally derived from ``torch.linalg.svdvals``;
    the matrix-norm API with spectral order is not used on this host.
    """

    import torch

    a64 = a.detach().to(device="cpu", dtype=torch.float64)
    b64 = b.detach().to(device="cpu", dtype=torch.float64)
    _, rb = torch.linalg.qr(b64, mode="reduced")
    _, ra = torch.linalg.qr(a64.transpose(0, 1), mode="reduced")
    core = (rb @ ra.transpose(0, 1)) * float(scale)
    return torch.linalg.svdvals(core)


def _spectrum_metrics(singular_values: Any, top_k: Sequence[int]) -> dict[str, Any]:
    import torch

    values = singular_values.detach().to(dtype=torch.float64, device="cpu")
    energy_values = values.square()
    energy = float(energy_values.sum().item())
    if energy <= 0:
        raise SpectralMemoryAuditError("zero-energy LoRA delta")
    spectral = float(torch.linalg.svdvals(torch.diag(values)).max().item())
    frobenius = math.sqrt(energy)
    probabilities = energy_values / energy
    positive = probabilities[probabilities > 0]
    effective_rank = math.exp(float((-(positive * positive.log()).sum()).item()))
    tolerance = (
        max(values.numel(), 1) * float(torch.finfo(torch.float64).eps) * spectral
    )
    result = {
        "frobenius_norm": _round(frobenius),
        "spectral_norm_svdvals_max": _round(spectral),
        "stable_rank": _round(energy / (spectral * spectral)),
        "energy_effective_rank": _round(effective_rank),
        "numerical_rank": int((values > tolerance).sum().item()),
        "delta_energy": _round(energy),
        "top_k_energy_fraction": {},
    }
    result["top_k_energy_fraction"] = {
        f"top_{item}": _round(float(probabilities[:item].sum().item()))
        for item in top_k
    }
    return result


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise SpectralMemoryAuditError("empty metric summary")
    return {
        "mean": _round(statistics.fmean(values)),
        "median": _round(statistics.median(values)),
        "min": _round(min(values)),
        "max": _round(max(values)),
    }


def _gini(values: Sequence[float]) -> float:
    total = sum(values)
    if total <= 0:
        raise SpectralMemoryAuditError("zero-energy layer distribution")
    n = len(values)
    absolute = sum(abs(left - right) for left in values for right in values)
    return absolute / (2 * n * total)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for position in range(start, stop):
            result[ordered[position]] = rank
        start = stop
    return result


def _module_spectrum(
    pairs: Mapping[int, tuple[Any, Any]], *, rank: int, alpha: int, top_k: Sequence[int]
) -> dict[str, Any]:
    per_layer: list[dict[str, Any]] = []
    for layer in sorted(pairs):
        a, b = pairs[layer]
        singular_values = _thin_delta_singular_values(a, b, alpha / rank)
        per_layer.append({"layer": layer, **_spectrum_metrics(singular_values, top_k)})
    if len(per_layer) != 28 or [item["layer"] for item in per_layer] != list(range(28)):
        raise SpectralMemoryAuditError("adapter layer coverage drifted")
    energies = [float(item["delta_energy"]) for item in per_layer]
    total = sum(energies)
    fractions = [item / total for item in energies]
    positive = [item for item in fractions if item > 0]
    effective_layers = math.exp(-sum(item * math.log(item) for item in positive))
    summary_fields = {
        name: _summary([float(item[name]) for item in per_layer])
        for name in (
            "frobenius_norm",
            "spectral_norm_svdvals_max",
            "stable_rank",
            "energy_effective_rank",
        )
    }
    for key in ("top_1", "top_2", "top_4"):
        summary_fields[f"{key}_energy_fraction"] = _summary(
            [float(item["top_k_energy_fraction"][key]) for item in per_layer]
        )
    return {
        "layers": len(per_layer),
        "nominal_rank": rank,
        "total_delta_energy": _round(total),
        "summary": summary_fields,
        "layer_energy_distribution": {
            "largest_layer_fraction": _round(max(fractions)),
            "top_4_layer_fraction": _round(sum(sorted(fractions, reverse=True)[:4])),
            "effective_layers": _round(effective_layers),
            "gini": _round(_gini(energies)),
            "last_9_layers_fraction": _round(sum(fractions[-9:])),
            "ordered_layer_energy_fraction_sha256": _compact_json_sha256(
                [_round(item) for item in fractions]
            ),
        },
        "per_layer": per_layer,
    }


def _adapter_state(
    config: Mapping[str, Any],
    name: str,
    comparison: Mapping[str, Any],
    snapshots: list[_Snapshot],
) -> tuple[dict[str, dict[int, tuple[Any, Any]]], dict[str, Any], Mapping[str, Any]]:
    from safetensors.torch import load as load_safetensors

    spec = _mapping(config["adapters"][name], f"adapters.{name}")
    directory = _repo_path(str(spec["path"]))
    qdiag._assert_physical_path(directory, require_directory=True, label=name)
    expected_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "diagnostic_receipt.json",
        "diagnostic_receipt.json.sha256",
    }
    if {item.name for item in directory.iterdir()} != expected_files:
        raise SpectralMemoryAuditError(f"adapter file inventory drifted: {name}")
    by_name: dict[str, _Snapshot] = {}
    for file_name in expected_files:
        item = _snapshot(directory / file_name)
        snapshots.append(item)
        by_name[file_name] = item
    for file_name, digest in spec["files"].items():
        if by_name[file_name].sha256 != digest:
            raise SpectralMemoryAuditError(
                f"adapter file hash drifted: {name}/{file_name}"
            )
    receipt = _strict_json(by_name["diagnostic_receipt.json"].data, f"{name} receipt")
    sidecar = by_name["diagnostic_receipt.json.sha256"].data
    if sidecar != (
        f"{by_name['diagnostic_receipt.json'].sha256}  diagnostic_receipt.json\n"
    ).encode("ascii"):
        raise SpectralMemoryAuditError(f"adapter receipt sidecar drifted: {name}")
    adapter_config = _strict_json(
        by_name["adapter_config.json"].data, f"{name} adapter config"
    )
    if (
        set(adapter_config.get("target_modules", [])) != set(spec["target_modules"])
        or adapter_config.get("use_rslora") is not False
        or adapter_config.get("use_dora") is not False
        or adapter_config.get("bias") != "none"
        or receipt.get("profile") != name
        or receipt.get("adapter_artifact_sha256", {}).get("adapter_model.safetensors")
        != by_name["adapter_model.safetensors"].sha256
    ):
        raise SpectralMemoryAuditError(f"adapter semantic identity drifted: {name}")
    comparison_arms = {
        item.get("profile"): item
        for item in comparison.get("arms", [])
        if isinstance(item, Mapping)
    }
    arm = comparison_arms.get(name)
    if (
        not isinstance(arm, Mapping)
        or arm.get("adapter_artifact_sha256", {}).get("adapter_model.safetensors")
        != by_name["adapter_model.safetensors"].sha256
    ):
        raise SpectralMemoryAuditError(f"comparison binding drifted: {name}")
    state = load_safetensors(by_name["adapter_model.safetensors"].data)
    collected: dict[str, dict[int, dict[str, Any]]] = {}
    for tensor_name, tensor in state.items():
        match = _LORA_RE.search(tensor_name)
        if match is None:
            raise SpectralMemoryAuditError(f"unexpected adapter tensor: {name}")
        layer, module, side = int(match.group(1)), match.group(2), match.group(3)
        collected.setdefault(module, {}).setdefault(layer, {})[side] = tensor
    expected_targets = set(spec["target_modules"])
    if set(collected) != expected_targets:
        raise SpectralMemoryAuditError(f"adapter module coverage drifted: {name}")
    paired: dict[str, dict[int, tuple[Any, Any]]] = {}
    for module, layers in collected.items():
        if set(layers) != set(range(28)):
            raise SpectralMemoryAuditError(f"adapter layer coverage drifted: {name}")
        paired[module] = {}
        expected_rank = int(spec["ranks"][module])
        for layer, sides in layers.items():
            if set(sides) != {"A", "B"}:
                raise SpectralMemoryAuditError(f"adapter A/B coverage drifted: {name}")
            a, b = sides["A"], sides["B"]
            if a.shape[0] != expected_rank or b.shape[1] != expected_rank:
                raise SpectralMemoryAuditError(f"adapter rank drifted: {name}")
            paired[module][layer] = (a, b)
    identity = {
        "path": str(spec["path"]),
        "profile": name,
        "target_modules": sorted(expected_targets),
        "ranks": dict(spec["ranks"]),
        "alphas": dict(spec["alphas"]),
        "files": {
            file_name: {
                "bytes": len(by_name[file_name].data),
                "sha256": by_name[file_name].sha256,
            }
            for file_name in sorted(by_name)
        },
    }
    return paired, identity, receipt


def _embedding_rows(
    config: Mapping[str, Any], groups: Mapping[str, Sequence[int]]
) -> tuple[Any, dict[int, int], dict[str, Any], Path]:
    import torch
    from safetensors import safe_open

    model_path = Path(str(config["model"]["local_path"]))
    if not model_path.is_absolute():
        model_path = (_root() / model_path).resolve()
    weight_path = model_path / "model.safetensors"
    all_ids = sorted({item for values in groups.values() for item in values})
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        key = str(config["model"]["embedding_tensor"])
        if key not in handle.keys():
            raise SpectralMemoryAuditError("embedding tensor is missing")
        value = handle.get_slice(key)
        shape = list(value.get_shape())
        if len(shape) != 2 or shape[1] != config["model"]["hidden_size"]:
            raise SpectralMemoryAuditError("embedding tensor shape drifted")
        if all_ids[-1] >= shape[0]:
            raise SpectralMemoryAuditError("selected token exceeds embedding rows")
        rows = torch.cat([value[item : item + 1].float() for item in all_ids], dim=0)
    rows = torch.nn.functional.normalize(rows, dim=1)
    return (
        rows,
        {item: index for index, item in enumerate(all_ids)},
        {
            "tensor": str(config["model"]["embedding_tensor"]),
            "shape": shape,
            "selected_unique_rows": len(all_ids),
            "selected_row_inventory_sha256": _ordered_int64_sha256(all_ids),
            "raw_token_ids_emitted": False,
        },
        weight_path,
    )


def _subspace_alignment(
    o_pairs: Mapping[int, tuple[Any, Any]],
    embeddings: Any,
    index_by_token: Mapping[int, int],
    groups: Mapping[str, Sequence[int]],
    hidden_size: int,
) -> dict[str, Any]:
    import torch

    bases = [
        torch.linalg.qr(o_pairs[layer][1].detach().float(), mode="reduced").Q
        for layer in sorted(o_pairs)
    ]
    result: dict[str, Any] = {}
    for name, token_ids in groups.items():
        rows = embeddings[[index_by_token[item] for item in token_ids]]
        per_layer = [
            float((rows @ basis).square().sum(dim=1).mean().item()) for basis in bases
        ]
        all_values = torch.stack(
            [(rows @ basis).square().sum(dim=1) for basis in bases], dim=1
        )
        result[name] = {
            "mean_projection_energy": _round(float(all_values.mean().item())),
            "median_projection_energy": _round(float(all_values.median().item())),
            "per_layer_mean_summary": _summary(per_layer),
            "ordered_per_layer_mean_sha256": _compact_json_sha256(
                [_round(item) for item in per_layer]
            ),
        }
    random_mean = result["deterministic_random_vocab"]["mean_projection_energy"]
    if random_mean <= 0:
        raise SpectralMemoryAuditError("random-vocabulary projection energy is zero")
    rank = int(next(iter(o_pairs.values()))[1].shape[1])
    isotropic = rank / hidden_size
    for value in result.values():
        value["ratio_to_deterministic_random"] = _round(
            float(value["mean_projection_energy"]) / random_mean
        )
        value["ratio_to_isotropic_rank_over_hidden"] = _round(
            float(value["mean_projection_energy"]) / isotropic
        )
    return {
        "nominal_subspace_rank": rank,
        "hidden_size": hidden_size,
        "isotropic_rank_over_hidden": _round(isotropic),
        "groups": result,
    }


def _authenticate_small_inputs(
    config: Mapping[str, Any], snapshots: list[_Snapshot]
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    paths = {
        "config": Path(str(config["_config_path"])),
        "config_schema": _root() / CONFIG_SCHEMA_PATH,
        "receipt_schema": _root() / RECEIPT_SCHEMA_PATH,
        "implementation": _root() / IMPLEMENTATION_PATH,
        "wrapper": _root() / WRAPPER_PATH,
        "comparison": _repo_path(str(config["comparison"]["path"])),
        "source_training_config": _repo_path(
            str(config["comparison"]["source_training_config"])
        ),
    }
    local: dict[str, _Snapshot] = {}
    for name, path in paths.items():
        value = _snapshot(path)
        snapshots.append(value)
        local[name] = value
    if (
        local["config"].sha256 != config["_config_sha256"]
        or local["comparison"].sha256 != config["comparison"]["sha256"]
        or local["source_training_config"].sha256
        != config["comparison"]["source_training_config_sha256"]
    ):
        raise SpectralMemoryAuditError("audit/source comparison identity drifted")
    comparison = _strict_json(local["comparison"].data, "source comparison")
    receipt_schema = _strict_json(local["receipt_schema"].data, "receipt schema")
    Draft202012Validator.check_schema(receipt_schema)
    if (
        comparison.get("status") != "passed_controlled_proxy_comparison_only"
        or comparison.get("claims", {}).get("formal") is not False
        or comparison.get("dataset", {}).get("train_records") != 80
        or comparison.get("dataset", {}).get("eval_proxy_records") != 20
    ):
        raise SpectralMemoryAuditError("source comparison claim boundary drifted")
    bindings = {
        name: _file_binding(
            value,
            logical_path=(
                str(config["comparison"]["path"])
                if name == "comparison"
                else str(config["comparison"]["source_training_config"])
                if name == "source_training_config"
                else CONFIG_PATH
                if name == "config"
                else CONFIG_SCHEMA_PATH
                if name == "config_schema"
                else RECEIPT_SCHEMA_PATH
                if name == "receipt_schema"
                else IMPLEMENTATION_PATH
                if name == "implementation"
                else WRAPPER_PATH
            ),
        )
        for name, value in local.items()
    }
    return comparison, receipt_schema, bindings


def _authenticate_model_assets(
    config: Mapping[str, Any], snapshots: list[_Snapshot]
) -> tuple[dict[str, Any], Path, tuple[int, int, int, int]]:
    model_path = Path(str(config["model"]["local_path"]))
    if not model_path.is_absolute():
        model_path = (_root() / model_path).resolve()
    qdiag._assert_physical_path(model_path, require_directory=True, label="base model")
    bindings: dict[str, Any] = {}
    weight_identity: tuple[int, int, int, int] | None = None
    for name, expected in config["model"]["assets"].items():
        path = model_path / name
        if name == "model.safetensors":
            digest, identity = _stream_sha256(path)
            weight_identity = identity
            size = path.stat().st_size
        else:
            snapshot = _snapshot(path)
            snapshots.append(snapshot)
            digest, size = snapshot.sha256, len(snapshot.data)
        if digest != expected:
            raise SpectralMemoryAuditError(f"base model asset drifted: {name}")
        bindings[name] = {"asset": name, "bytes": size, "sha256": digest}
    if weight_identity is None:
        raise SpectralMemoryAuditError("base weight identity missing")
    return bindings, model_path / "model.safetensors", weight_identity


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    snapshots: list[_Snapshot] = []
    comparison, receipt_schema, input_bindings = _authenticate_small_inputs(
        config, snapshots
    )
    model_assets, weight_path, initial_weight_identity = _authenticate_model_assets(
        config, snapshots
    )
    text_pairs, dataset_report = _authenticate_dataset(config, snapshots)
    groups, token_report, tokenizer = _tokenize_groups(config, text_pairs)
    del tokenizer
    embeddings, index_by_token, embedding_report, observed_weight_path = (
        _embedding_rows(config, groups)
    )
    if observed_weight_path != weight_path:
        raise SpectralMemoryAuditError("base embedding path drifted")
    adapters: dict[str, Any] = {}
    for name in ("q_plus_o", "wide_budget_matched"):
        paired, identity, _ = _adapter_state(config, name, comparison, snapshots)
        spec = config["adapters"][name]
        spectra = {
            module: _module_spectrum(
                paired[module],
                rank=int(spec["ranks"][module]),
                alpha=int(spec["alphas"][module]),
                top_k=config["metrics"]["top_k_energy"],
            )
            for module in ("q_proj", "o_proj")
        }
        q_fro = [
            float(item["frobenius_norm"]) for item in spectra["q_proj"]["per_layer"]
        ]
        o_fro = [
            float(item["frobenius_norm"]) for item in spectra["o_proj"]["per_layer"]
        ]
        adapters[name] = {
            "identity": identity,
            "delta_spectrum": spectra,
            "cross_projection": {
                "o_to_q_total_delta_energy_ratio": _round(
                    float(spectra["o_proj"]["total_delta_energy"])
                    / float(spectra["q_proj"]["total_delta_energy"])
                ),
                "o_to_q_median_frobenius_ratio": _round(
                    statistics.median(o_fro) / statistics.median(q_fro)
                ),
                "per_layer_frobenius_pearson": _round(_pearson(q_fro, o_fro)),
                "per_layer_frobenius_spearman": _round(
                    _pearson(_ranks(q_fro), _ranks(o_fro))
                ),
            },
            "o_output_subspace_alignment": _subspace_alignment(
                paired["o_proj"],
                embeddings,
                index_by_token,
                groups,
                int(config["model"]["hidden_size"]),
            ),
        }
    if config["metrics"]["final_model_rehash"] is not True:
        raise SpectralMemoryAuditError("final model rehash cannot be disabled")
    final_weight_sha, final_weight_identity = _stream_sha256(weight_path)
    if (
        final_weight_sha != config["model"]["assets"]["model.safetensors"]
        or final_weight_identity != initial_weight_identity
    ):
        raise SpectralMemoryAuditError("base weights changed during analysis")
    for snapshot in snapshots:
        snapshot.assert_unchanged()
    q_plus_o = adapters["q_plus_o"]
    wide = adapters["wide_budget_matched"]
    q_plus_o_target_ratio = q_plus_o["o_output_subspace_alignment"]["groups"][
        "target_frequent"
    ]["ratio_to_deterministic_random"]
    wide_target_ratio = wide["o_output_subspace_alignment"]["groups"][
        "target_frequent"
    ]["ratio_to_deterministic_random"]
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "passed_static_memory_risk_diagnostic_only",
        "identity": {
            **input_bindings,
            "model": {
                "id": config["model"]["id"],
                "source_revision": config["model"]["expected_source_revision"],
                "assets": model_assets,
                "embedding": embedding_report,
                "model_weight_final_rehash_matches": True,
            },
        },
        "dataset": dataset_report,
        "token_groups": token_report,
        "adapters": adapters,
        "findings": {
            "q_plus_o_o_top_1_energy_fraction_mean": q_plus_o["delta_spectrum"][
                "o_proj"
            ]["summary"]["top_1_energy_fraction"]["mean"],
            "q_plus_o_o_energy_effective_rank_mean": q_plus_o["delta_spectrum"][
                "o_proj"
            ]["summary"]["energy_effective_rank"]["mean"],
            "q_plus_o_o_to_q_total_delta_energy_ratio": q_plus_o["cross_projection"][
                "o_to_q_total_delta_energy_ratio"
            ],
            "q_plus_o_target_frequent_to_random_projection_ratio": q_plus_o_target_ratio,
            "wide_target_frequent_to_random_projection_ratio": wide_target_ratio,
            "low_rank_writeback_signal_observed": (
                q_plus_o["delta_spectrum"]["o_proj"]["summary"][
                    "top_1_energy_fraction"
                ]["mean"]
                > 0.75
            ),
            "target_frequency_alignment_signal_observed": q_plus_o_target_ratio > 1.5,
            "interpretation": (
                "static_template_writeback_overfit_risk_signal_not_memorization_proof"
            ),
        },
        "claims": {
            **dict(config["claims"]),
            "static_alignment_is_correlation_only": True,
        },
        "audit": {
            "cpu_only": True,
            "gpu_requests": 0,
            "network_requests": 0,
            "provider_requests": 0,
            "model_forward_passes": 0,
            "full_model_loads": 0,
            "base_embedding_slice_loads": 1,
            "tokenizer_loads": 1,
            "train_partition_reads": 2,
            "eval_proxy_reads": 0,
            "heldout_reads": 0,
            "protected_body_reads": 0,
            "raw_sample_text_emitted": False,
            "raw_token_ids_emitted": False,
            "matrix_norm_ord2_used": False,
            "spectral_algorithm": "thin_qr_core_then_torch_linalg_svdvals_max_v1",
            "atomic_publish": True,
        },
    }
    _validate_receipt(receipt, schema=receipt_schema)
    return receipt


def _validate_receipt(
    receipt: Mapping[str, Any], *, schema: Mapping[str, Any] | None = None
) -> None:
    if schema is None:
        schema = _strict_json(
            _snapshot(_root() / RECEIPT_SCHEMA_PATH).data, "receipt schema"
        )
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(receipt),
        key=lambda item: list(item.path),
    )
    if errors:
        raise SpectralMemoryAuditError(
            f"receipt schema validation failed: {errors[0].message}"
        )


def publish_receipt(
    receipt: Mapping[str, Any],
    output_dir: Path,
    *,
    diagnostics_root: Path | None = None,
) -> tuple[Path, str]:
    diagnostics = (
        (_root() / "artifacts" / "diagnostics").resolve()
        if diagnostics_root is None
        else diagnostics_root.resolve()
    )
    destination = output_dir if output_dir.is_absolute() else _root() / output_dir
    destination = destination.resolve()
    if diagnostics != destination.parent or os.path.lexists(destination):
        raise SpectralMemoryAuditError(
            "receipt output must be a new direct diagnostics child"
        )
    diagnostics.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=diagnostics))
    try:
        data = _canonical_json_bytes(receipt)
        digest = _sha256(data)
        receipt_path = temporary / "receipt.json"
        sidecar_path = temporary / "receipt.json.sha256"
        with receipt_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_path.open("xb") as handle:
            handle.write(f"{digest}  receipt.json\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    published = _snapshot(destination / "receipt.json")
    sidecar = _snapshot(destination / "receipt.json.sha256")
    if (
        published.sha256 != digest
        or published.data != data
        or sidecar.data != f"{digest}  receipt.json\n".encode("ascii")
    ):
        raise SpectralMemoryAuditError("published receipt failed authentication")
    return destination / "receipt.json", digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Q/O LoRA spectral and output-subspace memory-risk audit"
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    configured_output = str(config["output"]["path"])
    if args.output is not None and args.output != configured_output:
        raise ConfigError(f"output must remain exactly {configured_output}")
    receipt = build_receipt(config)
    path, digest = publish_receipt(receipt, Path(configured_output))
    print(
        json.dumps(
            {
                "receipt": str(path),
                "sha256": digest,
                "status": receipt["status"],
                "raw_sample_text_emitted": False,
                "raw_token_ids_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
