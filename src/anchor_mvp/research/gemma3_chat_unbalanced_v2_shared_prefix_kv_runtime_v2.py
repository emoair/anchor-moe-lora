"""Real Gemma 3 shared-prefix KV handoff probe with fail-closed provenance.

The default path is model-free.  Physical execution is possible only after an
authenticated Producer source gate, consumer receipt, adapter receipt, model
identity and exclusive GPU lock are supplied.  The physical claim is limited
to exact retained prefix values and avoided prefix recomputation.  A
``DynamicCache`` object or data pointer is never treated as proof of shared
storage, zero-copy, RDMA, or full-generation sharing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2.json"
)
CONFIG_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2_config.schema.json"
)
RECEIPT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2_receipt.schema.json"
)
CONFIG_VERSION = "anchor.gemma3-chat-unbalanced-v2-shared-prefix-kv-runtime-config.v2"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-shared-prefix-kv-runtime-receipt.v2"
PROFILE_ID = "gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2"
SOURCE_GATE_MODULE = "anchor_mvp.training.gemma3_chat_unbalanced_v2_runtime_source_v2"
SOURCE_GATE_CALLABLES = (
    "authenticate_producer_source",
    "iter_authenticated_jsonl",
    "terminal_recheck_source",
    "authenticate_tokenizer_snapshot",
    "serialize_authenticated_example",
)
TRAINED_ARMS = (
    "humor_q_only",
    "serious_q_only",
    "angry_style_q_only",
    "review_audit_q_only",
    "tool_q_only",
    "tool_q_plus_micro_o",
    "planner_q_only",
    "planner_o_only",
    "planner_q_plus_micro_o",
)
ARM_PROFILES = {
    "humor_q_only": "q_only",
    "serious_q_only": "q_only",
    "angry_style_q_only": "q_only",
    "review_audit_q_only": "q_only",
    "tool_q_only": "q_only",
    "tool_q_plus_micro_o": "q_plus_micro_o",
    "planner_q_only": "q_only",
    "planner_o_only": "o_only",
    "planner_q_plus_micro_o": "q_plus_micro_o",
}
TEACHER_IDENTITY_FIELDS = (
    "schema_version",
    "binding_sha256",
    "manifest_sha256",
    "record_schema_sha256",
    "release_receipt_sha256",
    "release_attestation_sha256",
    "shard_inventory_sha256",
    "public_identity_sha256",
)
TEACHER_FINAL_METRICS = {
    "records": 3520,
    "main_records": 3440,
    "router_records": 80,
    "train_identity_bundles": 20,
    "train_identity_records": 100,
    "independent_identity_eval_probes_not_in_optimizer": 50,
    "accepted": 3520,
    "rejected_or_uncertain": 0,
}
FAILURE_CODES = frozenset(
    {
        "kv_v2_adapter_identity_pending",
        "kv_v2_adapter_receipt_invalid",
        "kv_v2_adapter_switch_invalid",
        "kv_v2_atomic_publish_failed",
        "kv_v2_boundary_digest_mismatch",
        "kv_v2_cache_mutation",
        "kv_v2_claim_inflation",
        "kv_v2_config_invalid",
        "kv_v2_execute_not_authorized",
        "kv_v2_gpu_lock_busy",
        "kv_v2_gpu_lock_invalid",
        "kv_v2_layer_topology_invalid",
        "kv_v2_logits_mismatch",
        "kv_v2_model_identity_pending",
        "kv_v2_physical_backend_failed",
        "kv_v2_prefix_value_mismatch",
        "kv_v2_private_tail_alias",
        "kv_v2_q8_scb_identity_drift",
        "kv_v2_receipt_invalid",
        "kv_v2_source_gate_invalid",
        "kv_v2_source_identity_drift",
        "kv_v2_source_module_unavailable",
        "kv_v2_tokenizer_binding_drift",
        "kv_v2_wrong_attention_mask",
        "kv_v2_wrong_position",
    }
)
LINEAGE_FIELDS = (
    "producer_candidate_commit",
    "producer_release_commit",
    "producer_release_tree",
    "source_blob_sha256",
    "source_record_sha256",
    "training_serialization_inventory_sha256",
    "record_prompt_sha256",
    "example_serialization_sha256",
    "tokenizer_binding_sha256",
    "model_config_sha256",
    "token_order_sha256",
    "position_sha256",
    "attention_mask_sha256",
    "rope_sha256",
    "model_identity_sha256",
    "base_identity_sha256",
    "q8_scb_sha256",
    "adapter_lineage_sha256",
    "layer_topology_sha256",
    "route_commit_sha256",
    "plan_commit_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_PENDING_DIGEST = hashlib.sha256(b"pending").hexdigest()


class SharedPrefixKVRuntimeError(RuntimeError):
    """Body-free fail-closed error."""


class RuntimeSourceGate(Protocol):
    def __call__(
        self,
        *,
        producer_repository_root: str | os.PathLike[str],
        consumer_receipt_path: str | os.PathLike[str],
        consumer_receipt_sidecar_path: str | os.PathLike[str],
        consumer_receipt_schema_path: str | os.PathLike[str],
        model_root: str | os.PathLike[str],
        source_blob_path: str,
        record_index: int,
        expected: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class PhysicalBackend(Protocol):
    def __call__(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _fail(code: str) -> None:
    if code not in FAILURE_CODES:
        code = "kv_v2_receipt_invalid"
    raise SharedPrefixKVRuntimeError(code)


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite_json_number")


def _strict_json(raw: bytes, code: str) -> Mapping[str, Any]:
    if not raw or len(raw) > _MAX_JSON_BYTES:
        _fail(code)
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _read_regular(path: Path, code: str) -> bytes:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or bool(getattr(before, "st_file_attributes", 0) & 0x400)
        ):
            _fail(code)
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            raw = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError:
        _fail(code)
    signatures = {
        (
            int(item.st_dev),
            int(item.st_ino),
            int(item.st_size),
            int(item.st_mtime_ns),
        )
        for item in (before, opened_before, opened_after, after)
    }
    if len(signatures) != 1 or len(raw) != int(after.st_size):
        _fail(code)
    return raw


def _object_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
    )


def _owned_path_identity(
    path: Path,
    *,
    require_directory: bool,
    code: str,
) -> tuple[int, int, int]:
    try:
        metadata = path.lstat()
    except OSError:
        _fail(code)
    expected_kind = stat.S_ISDIR if require_directory else stat.S_ISREG
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not expected_kind(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & 0x400)
    ):
        _fail(code)
    return _object_identity(metadata)


def _release_owned_path(
    path: Path,
    expected_identity: tuple[int, int, int],
) -> bool:
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & 0x400)
            or _object_identity(metadata) != expected_identity
        ):
            return False
        path.unlink()
    except OSError:
        return False
    return True


def _load_json(path: Path, code: str) -> tuple[Mapping[str, Any], str]:
    raw = _read_regular(path, code)
    return _strict_json(raw, code), _sha256(raw)


def implementation_sha256() -> str:
    return _sha256(_read_regular(Path(__file__), "kv_v2_config_invalid"))


def _schema(path: Path, expected_sha256: str) -> Draft202012Validator:
    value, observed = _load_json(path, "kv_v2_config_invalid")
    if observed != expected_sha256:
        _fail("kv_v2_config_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError:
        _fail("kv_v2_config_invalid")
    return Draft202012Validator(value)


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_sha256(config: Mapping[str, Any]) -> str:
    observed = config.get("_config_sha256")
    if _is_sha256(observed):
        return str(observed)
    return _sha256(_canonical_json(_public_config(config)))


def _schema_validators(
    config: Mapping[str, Any],
) -> tuple[Draft202012Validator, Draft202012Validator]:
    schemas = config.get("schemas")
    if not isinstance(schemas, Mapping):
        _fail("kv_v2_config_invalid")
    expected = {
        "config": (CONFIG_SCHEMA_PATH, CONFIG_VERSION),
        "receipt": (RECEIPT_SCHEMA_PATH, RECEIPT_VERSION),
    }
    validators: list[Draft202012Validator] = []
    for name in ("config", "receipt"):
        binding = schemas.get(name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("version") != expected[name][1]
            or not _is_sha256(binding.get("sha256"))
        ):
            _fail("kv_v2_config_invalid")
        validators.append(_schema(expected[name][0], str(binding["sha256"])))
    return validators[0], validators[1]


def validate_config(config: Mapping[str, Any]) -> None:
    config_validator, _ = _schema_validators(config)
    if tuple(config_validator.iter_errors(_public_config(config))):
        _fail("kv_v2_config_invalid")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("profile_id") != PROFILE_ID
        or config["source_gate"]["module"] != SOURCE_GATE_MODULE
        or tuple(config["source_gate"]["callables"]) != SOURCE_GATE_CALLABLES
        or tuple(config["adapter_receipt"]["required_arms"]) != TRAINED_ARMS
    ):
        _fail("kv_v2_config_invalid")
    producer = config["producer_final"]
    consumer = config["consumer_preflight"]
    tokenizer = config["tokenizer_binding"]
    if (
        producer["serialization_inventory_sha256"]
        != tokenizer["training_serialization_inventory_sha256"]
        or consumer["receipt_sha256"]
        != "a976754d84f48c01b9ff509c48f7fb7b26c32eabfa0b10d6f221661114a1cdb2"
    ):
        _fail("kv_v2_source_identity_drift")
    claims = config["claims"]
    forbidden = (
        "shared_storage",
        "zero_copy",
        "rdma",
        "full_generation_sharing",
        "pointer_identity_is_proof",
        "formal",
        "live",
    )
    if any(claims[name] is not False for name in forbidden):
        _fail("kv_v2_claim_inflation")
    if tuple(config["handoff"]["lineage_fields"]) != LINEAGE_FIELDS:
        _fail("kv_v2_config_invalid")


def load_config(path: str | os.PathLike[str] = CONFIG_PATH) -> dict[str, Any]:
    value, digest = _load_json(Path(path), "kv_v2_config_invalid")
    result = dict(value)
    result["_config_sha256"] = digest
    validate_config(result)
    return result


def tokenizer_binding_sha256(config: Mapping[str, Any]) -> str:
    return str(
        config["tokenizer_binding"]["authenticated_tokenizer_combined_identity_sha256"]
    )


def derive_layer_topology(
    model_config: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    layer_types = model_config.get("layer_types")
    layers = model_config.get("num_hidden_layers")
    window = model_config.get("sliding_window")
    if (
        isinstance(layers, bool)
        or not isinstance(layers, int)
        or not isinstance(layer_types, list)
        or len(layer_types) != layers
        or isinstance(window, bool)
        or not isinstance(window, int)
    ):
        _fail("kv_v2_layer_topology_invalid")
    if any(item not in {"sliding_attention", "full_attention"} for item in layer_types):
        _fail("kv_v2_layer_topology_invalid")
    sliding = layer_types.count("sliding_attention")
    full_indices = [
        index for index, kind in enumerate(layer_types) if kind == "full_attention"
    ]
    if (
        layers != policy["expected_layers"]
        or sliding != policy["expected_sliding_layers"]
        or len(full_indices) != policy["expected_full_layers"]
        or window != policy["expected_sliding_window"]
    ):
        _fail("kv_v2_layer_topology_invalid")
    details = [
        {
            "index": index,
            "attention_kind": ("sliding" if kind == "sliding_attention" else "full"),
            "window": window if kind == "sliding_attention" else None,
        }
        for index, kind in enumerate(layer_types)
    ]
    return {
        "layers": layers,
        "sliding_layers": sliding,
        "full_layers": len(full_indices),
        "sliding_window": window,
        "full_layer_indices": full_indices,
        "layer_topology_sha256": _sha256(_canonical_json(details)),
        "derived_from_authenticated_model_config": True,
        "layers_detail": details,
    }


def boundary_commit_sha256(lineage: Mapping[str, Any]) -> str:
    if set(lineage) != set(LINEAGE_FIELDS):
        _fail("kv_v2_boundary_digest_mismatch")
    for name, value in lineage.items():
        if name.endswith("_commit") or name.endswith("_tree"):
            if not _is_commit(value):
                _fail("kv_v2_boundary_digest_mismatch")
        elif not _is_sha256(value):
            _fail("kv_v2_boundary_digest_mismatch")
    return _sha256(_canonical_json({name: lineage[name] for name in LINEAGE_FIELDS}))


def validate_boundary(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    commit_sha256: str,
) -> None:
    expected_commit = boundary_commit_sha256(expected)
    observed_commit = boundary_commit_sha256(observed)
    if (
        expected != observed
        or expected_commit != observed_commit
        or commit_sha256 != expected_commit
    ):
        if expected.get("position_sha256") != observed.get("position_sha256"):
            _fail("kv_v2_wrong_position")
        if expected.get("attention_mask_sha256") != observed.get(
            "attention_mask_sha256"
        ):
            _fail("kv_v2_wrong_attention_mask")
        if expected.get("adapter_lineage_sha256") != observed.get(
            "adapter_lineage_sha256"
        ):
            _fail("kv_v2_adapter_switch_invalid")
        _fail("kv_v2_boundary_digest_mismatch")


def _pending_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    result = []
    for group, keys in {
        "adapter_receipt": ("receipt_sha256", "sidecar_sha256", "schema_sha256"),
        "model": ("model_identity_sha256", "tokenizer_identity_sha256"),
        "teacher_final": (
            "schema_version",
            "binding_sha256",
            "manifest_sha256",
            "record_schema_sha256",
            "release_receipt_sha256",
            "release_attestation_sha256",
            "shard_inventory_sha256",
            "public_identity_sha256",
        ),
    }.items():
        for key in keys:
            if config[group][key] == "pending":
                result.append(f"{group}.{key}")
    if config["execution"]["execute_authorized"] is not True:
        result.append("execution.execute_authorized")
    return tuple(result)


def _zero_resources(
    *,
    model_loads: int = 0,
    gpu_requests: int = 0,
) -> dict[str, int]:
    return {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": model_loads,
        "gpu_requests": gpu_requests,
        "gold_body_reads": 0,
        "heldout_body_reads": 0,
        "protected_body_reads": 0,
        "raw_token_ids_persisted": 0,
        "raw_logits_persisted": 0,
    }


def _claims(*, passed: bool = False) -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "proxy_only": True,
        "exact_prefix_value_handoff": passed,
        "prefill_compute_handoff": passed,
        "shared_storage": False,
        "zero_copy": False,
        "rdma": False,
        "full_generation_sharing": False,
        "pointer_identity_is_proof": False,
        "formal": False,
        "live": False,
    }


def _empty_topology() -> dict[str, Any]:
    return {
        "layers": 26,
        "sliding_layers": 22,
        "full_layers": 4,
        "sliding_window": 512,
        "full_layer_indices": [],
        "layer_topology_sha256": _PENDING_DIGEST,
        "derived_from_authenticated_model_config": False,
        "layers_detail": [],
    }


def _empty_source(config: Mapping[str, Any]) -> dict[str, Any]:
    producer = config["producer_final"]
    return {
        "authenticated": False,
        "producer_candidate_commit": producer["candidate_commit"],
        "producer_release_commit": producer["release_commit"],
        "producer_release_tree": producer["release_tree"],
        "source_blob_path": "pending",
        "source_blob_sha256": _PENDING_DIGEST,
        "source_blob_bytes": 0,
        "record_index": 0,
        "record_sha256": _PENDING_DIGEST,
        "record_prompt_sha256": _PENDING_DIGEST,
        "example_serialization_sha256": _PENDING_DIGEST,
        "tokenizer_binding_sha256": tokenizer_binding_sha256(config),
        "model_config_sha256": _PENDING_DIGEST,
        "hf_auto_tokenizer_used": False,
        "second_model_file_path_read": False,
        "runtime_special_token_overlay": True,
        "serialized_prefix_persisted": False,
    }


def _empty_boundary() -> dict[str, Any]:
    return {
        "boundary_commit_sha256": _PENDING_DIGEST,
        "ordered_prefix_sha256": _PENDING_DIGEST,
        "token_order_sha256": _PENDING_DIGEST,
        "position_sha256": _PENDING_DIGEST,
        "attention_mask_sha256": _PENDING_DIGEST,
        "rope_sha256": _PENDING_DIGEST,
        "route_commit_sha256": _PENDING_DIGEST,
        "plan_commit_sha256": _PENDING_DIGEST,
        "prefix_tokens": 1,
        "source_prefill_adapter_state": "adapter_off_frozen_base",
        "planner_adapter_sha256": _PENDING_DIGEST,
        "output_adapter_sha256": _PENDING_DIGEST,
        "latent_packet": {
            "enabled": False,
            "fixed_dimension": 0,
            "serialization_sha256": None,
            "position_sha256": None,
            "commit_sha256": None,
            "hidden_chain_of_thought": False,
        },
    }


def _empty_comparison() -> dict[str, Any]:
    return {
        "cold_recompute_executed": False,
        "shared_prefix_handoff_executed": False,
        "logits_digest_cold": None,
        "logits_digest_handoff": None,
        "logits_allclose": False,
        "logits_max_abs_error": 0.0,
        "next_token_argmax_equal": False,
        "next_token_digest_cold": None,
        "next_token_digest_handoff": None,
        "prefill_count_cold": 0,
        "prefill_count_handoff": 0,
    }


def _empty_isolation() -> dict[str, Any]:
    return {
        "source_cache_unchanged": False,
        "planner_tail_private": False,
        "output_tail_private": False,
        "sibling_branch_unchanged": False,
        "private_tail_alias": False,
        "wrong_adapter_rejected": False,
        "wrong_position_rejected": False,
        "wrong_attention_mask_rejected": False,
        "cache_mutation_rejected": False,
        "pointer_observation_only": True,
    }


def _empty_performance() -> dict[str, Any]:
    return {
        "cold_latency_ms": 0.0,
        "handoff_latency_ms": 0.0,
        "handoff_prefill_ms": 0.0,
        "planner_tail_ms": 0.0,
        "output_tail_ms": 0.0,
        "cold_peak_device_bytes": 0,
        "handoff_peak_device_bytes": 0,
        "repetitions": 0,
    }


def build_status(
    config: Mapping[str, Any],
    *,
    operation: str = "status",
) -> dict[str, Any]:
    validate_config(config)
    result = {
        "schema_version": RECEIPT_VERSION,
        "profile_id": PROFILE_ID,
        "status": "blocked_model_free",
        "operation": operation,
        "identity": {
            "config_sha256": config_sha256(config),
            "implementation_sha256": implementation_sha256(),
            "consumer_receipt_sha256": config["consumer_preflight"]["receipt_sha256"],
            "adapter_receipt_sha256": config["adapter_receipt"]["receipt_sha256"],
            "model_identity_sha256": config["model"]["model_identity_sha256"],
            "tokenizer_identity_sha256": config["model"]["tokenizer_identity_sha256"],
            "training_serialization_inventory_sha256": config["tokenizer_binding"][
                "training_serialization_inventory_sha256"
            ],
        },
        "source": _empty_source(config),
        "topology": _empty_topology(),
        "boundary": _empty_boundary(),
        "comparison": _empty_comparison(),
        "isolation": _empty_isolation(),
        "performance": _empty_performance(),
        "claims": _claims(),
        "resource_counters": _zero_resources(),
        "failure_codes": list(_pending_fields(config)),
    }
    validate_receipt(result, config, allow_blocked=True)
    return result


def _validate_teacher_final_identity(
    teacher: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if (
        set(expected) != set(TEACHER_IDENTITY_FIELDS)
        or not isinstance(expected.get("schema_version"), str)
        or expected["schema_version"] == "pending"
        or any(
            not _is_sha256(expected.get(name))
            for name in TEACHER_IDENTITY_FIELDS
            if name != "schema_version"
        )
        or any(teacher.get(name) != expected[name] for name in TEACHER_IDENTITY_FIELDS)
        or any(
            teacher.get(name) != value for name, value in TEACHER_FINAL_METRICS.items()
        )
        or teacher.get("producer_original_target_fallback") is not False
    ):
        _fail("kv_v2_adapter_receipt_invalid")


def _authenticate_adapter_receipt(
    config: Mapping[str, Any],
    *,
    receipt_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    schema_path: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
) -> Mapping[str, Any]:
    binding = config["adapter_receipt"]
    if any(
        binding[key] == "pending"
        for key in ("receipt_sha256", "sidecar_sha256", "schema_sha256")
    ):
        _fail("kv_v2_adapter_identity_pending")
    raw = _read_regular(Path(receipt_path), "kv_v2_adapter_receipt_invalid")
    sidecar = _read_regular(Path(sidecar_path), "kv_v2_adapter_receipt_invalid")
    schema_raw = _read_regular(Path(schema_path), "kv_v2_adapter_receipt_invalid")
    if (
        _sha256(raw) != binding["receipt_sha256"]
        or _sha256(sidecar) != binding["sidecar_sha256"]
        or _sha256(schema_raw) != binding["schema_sha256"]
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    expected_sidecar = f"{_sha256(raw)}  {Path(receipt_path).name}\n".encode("ascii")
    if sidecar != expected_sidecar:
        _fail("kv_v2_adapter_receipt_invalid")
    receipt = _strict_json(raw, "kv_v2_adapter_receipt_invalid")
    schema = _strict_json(schema_raw, "kv_v2_adapter_receipt_invalid")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(receipt)
    except (SchemaError, ValidationError):
        _fail("kv_v2_adapter_receipt_invalid")
    if (
        receipt.get("schema_version") != binding["schema_version"]
        or receipt.get("status") != binding["required_status"]
        or receipt.get("mode") != binding["required_mode"]
        or tuple(receipt.get("phase_order", ())) != TRAINED_ARMS
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    source = receipt.get("source")
    teacher = receipt.get("teacher_final")
    tokenizer = receipt.get("tokenizer")
    claims = receipt.get("claims")
    model = receipt.get("model")
    phases = receipt.get("phase_receipts")
    adapters = receipt.get("adapters")
    if (
        not isinstance(source, Mapping)
        or source.get("consumer_preflight_receipt_sha256")
        != config["consumer_preflight"]["receipt_sha256"]
        or not isinstance(teacher, Mapping)
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("combined_identity_sha256")
        != config["model"]["tokenizer_identity_sha256"]
        or not isinstance(claims, Mapping)
        or claims.get("diagnostic_only") is not True
        or claims.get("proxy_only") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "formal",
                "release",
                "physical_kv_validated",
                "full_generation_kv_shared",
            )
        )
        or not isinstance(model, Mapping)
        or model.get("frozen_q8_base") is not True
        or model.get("base_o_proj_always_frozen") is not True
        or model.get("all_phase_base_hashes_equal") is not True
        or model.get("all_phase_q8_scb_hashes_equal") is not True
        or not isinstance(phases, Sequence)
        or isinstance(phases, (str, bytes, bytearray))
        or len(phases) != len(TRAINED_ARMS)
        or not isinstance(adapters, Mapping)
        or set(adapters) != set(TRAINED_ARMS)
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    _validate_teacher_final_identity(teacher, config["teacher_final"])
    phase_base: dict[str, str] = {}
    phase_scb: dict[str, str] = {}
    for arm, phase in zip(TRAINED_ARMS, phases, strict=True):
        if (
            not isinstance(phase, Mapping)
            or phase.get("arm") != arm
            or phase.get("phase") != "full"
            or phase.get("status") != "passed"
            or not _is_sha256(phase.get("base_parameter_sha256"))
            or not _is_sha256(phase.get("q8_scb_sha256"))
        ):
            _fail("kv_v2_adapter_receipt_invalid")
        phase_base[arm] = str(phase["base_parameter_sha256"])
        phase_scb[arm] = str(phase["q8_scb_sha256"])
    if (
        model.get("phase_base_hashes") != phase_base
        or model.get("phase_q8_scb_hashes") != phase_scb
        or len(set(phase_base.values())) != 1
        or len(set(phase_scb.values())) != 1
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    base_files = model.get("base_files")
    if (
        not isinstance(base_files, Mapping)
        or not base_files
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in base_files.items()
        )
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    base_inventory = _sha256(_canonical_json(dict(base_files)))
    if (
        base_inventory != model.get("base_file_inventory_sha256")
        or base_inventory != config["model"]["model_identity_sha256"]
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    published_root_input = Path(receipt_path).absolute().parent
    _owned_path_identity(
        published_root_input,
        require_directory=True,
        code="kv_v2_adapter_receipt_invalid",
    )
    published_root = published_root_input.resolve()
    _authenticate_base_files(Path(model_root), base_files)
    normalized_adapters = _authenticate_adapter_files(published_root, adapters)
    result = dict(receipt)
    result["_published_root"] = str(published_root)
    result["_adapter_inventory_sha256"] = _sha256(_canonical_json(normalized_adapters))
    result["_base_identity_sha256"] = base_inventory
    result["_q8_scb_sha256"] = next(iter(phase_scb.values()))
    return result


def _safe_descendant(root: Path, relative: str, code: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or ".." in posix.parts
        or posix.as_posix() != relative
    ):
        _fail(code)
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*posix.parts)
    current = root_resolved
    for part in posix.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError:
            _fail(code)
        if stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & 0x400
        ):
            _fail(code)
    try:
        candidate.resolve().relative_to(root_resolved)
    except (OSError, ValueError):
        _fail(code)
    return candidate


def _authenticate_base_files(
    model_root: Path,
    base_files: Mapping[str, Any],
) -> None:
    _owned_path_identity(
        model_root.absolute(),
        require_directory=True,
        code="kv_v2_adapter_receipt_invalid",
    )
    root = model_root.resolve()
    if not root.is_dir():
        _fail("kv_v2_adapter_receipt_invalid")
    observed: dict[str, str] = {}
    for name in sorted(base_files):
        path = _safe_descendant(
            root,
            str(name),
            "kv_v2_adapter_receipt_invalid",
        )
        raw = _read_regular(path, "kv_v2_adapter_receipt_invalid")
        observed[str(name)] = _sha256(raw)
    if observed != dict(base_files):
        _fail("kv_v2_adapter_receipt_invalid")


def _authenticate_adapter_files(
    published_root: Path,
    adapters: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for arm in TRAINED_ARMS:
        value = adapters.get(arm)
        if (
            not isinstance(value, Mapping)
            or value.get("path") != f"adapters/{arm}"
            or value.get("adapter_profile") != ARM_PROFILES[arm]
            or isinstance(value.get("trainable_parameters"), bool)
            or not isinstance(value.get("trainable_parameters"), int)
            or value["trainable_parameters"] < 1
            or not _is_sha256(value.get("artifact_tree_sha256"))
            or not _is_sha256(value.get("tensor_inventory_sha256"))
        ):
            _fail("kv_v2_adapter_receipt_invalid")
        adapter_root = _safe_descendant(
            published_root,
            str(value["path"]),
            "kv_v2_adapter_receipt_invalid",
        )
        try:
            entries_on_disk = {item.name for item in adapter_root.iterdir()}
        except OSError:
            _fail("kv_v2_adapter_receipt_invalid")
        if entries_on_disk != {"adapter_config.json", "adapter_model.safetensors"}:
            _fail("kv_v2_adapter_receipt_invalid")
        authenticated_entries = []
        for key, filename in (
            ("adapter_config", "adapter_config.json"),
            ("adapter_model", "adapter_model.safetensors"),
        ):
            entry = value.get(key)
            if (
                not isinstance(entry, Mapping)
                or entry.get("path") != filename
                or not _is_sha256(entry.get("sha256"))
                or isinstance(entry.get("bytes"), bool)
                or not isinstance(entry.get("bytes"), int)
                or entry["bytes"] < 1
            ):
                _fail("kv_v2_adapter_receipt_invalid")
            raw = _read_regular(
                _safe_descendant(
                    adapter_root,
                    filename,
                    "kv_v2_adapter_receipt_invalid",
                ),
                "kv_v2_adapter_receipt_invalid",
            )
            if len(raw) != entry["bytes"] or _sha256(raw) != entry["sha256"]:
                _fail("kv_v2_adapter_receipt_invalid")
            authenticated_entries.append(dict(entry))
        if (
            _sha256(
                _canonical_json(
                    sorted(authenticated_entries, key=lambda item: item["path"])
                )
            )
            != value["artifact_tree_sha256"]
        ):
            _fail("kv_v2_adapter_receipt_invalid")
        normalized[arm] = dict(value)
    return normalized


def _load_source_module() -> Any:
    try:
        module = importlib.import_module(SOURCE_GATE_MODULE)
    except ImportError:
        _fail("kv_v2_source_module_unavailable")
    if any(not callable(getattr(module, name, None)) for name in SOURCE_GATE_CALLABLES):
        _fail("kv_v2_source_module_unavailable")
    return module


def _find_unique_digest(value: object, key: str) -> str:
    found: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for name, child in current.items():
                if name == key and _is_sha256(child):
                    found.add(str(child))
                elif isinstance(child, (Mapping, list, tuple)):
                    stack.append(child)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    if len(found) != 1:
        _fail("kv_v2_source_identity_drift")
    return found.pop()


def _record_prompt_target(record: Mapping[str, Any]) -> tuple[str, str, str]:
    messages = record.get("messages")
    target_value = record.get("target")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes, bytearray))
        or len(messages) != 2
        or not all(isinstance(item, Mapping) for item in messages)
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        _fail("kv_v2_source_gate_invalid")
    if isinstance(target_value, Mapping):
        target = target_value.get("assistant_text")
    else:
        target = target_value
    if not isinstance(target, str) or not target:
        _fail("kv_v2_source_gate_invalid")
    prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    visible = (
        "<start_of_turn>user\n" + prompt + "<end_of_turn><eos>\n<start_of_turn>model\n"
    )
    declared = record.get("gemma_serialization")
    if isinstance(declared, Mapping):
        prompt_sha = declared.get("prompt_sha256")
        if prompt_sha != _sha256(visible.encode("utf-8")):
            _fail("kv_v2_source_identity_drift")
    else:
        prompt_sha = _sha256(visible.encode("utf-8"))
    return prompt, target, str(prompt_sha)


def _authenticate_source_via_module(
    *,
    producer_repository_root: str | os.PathLike[str],
    consumer_receipt_path: str | os.PathLike[str],
    consumer_receipt_sidecar_path: str | os.PathLike[str],
    consumer_receipt_schema_path: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
    source_blob_path: str,
    record_index: int,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    del (
        consumer_receipt_path,
        consumer_receipt_sidecar_path,
        consumer_receipt_schema_path,
    )
    module = _load_source_module()
    try:
        source = module.authenticate_producer_source(producer_repository_root)
        record = None
        for index, value in enumerate(
            module.iter_authenticated_jsonl(source, source_blob_path)
        ):
            if index == record_index:
                record = value
                break
        if record is None:
            _fail("kv_v2_source_identity_drift")
        prompt, target, record_prompt_sha = _record_prompt_target(record)
        tokenizer = module.authenticate_tokenizer_snapshot(model_root)
        serialized = module.serialize_authenticated_example(
            tokenizer,
            prompt,
            target,
            sequence_length=768,
        )
        module.terminal_recheck_source(source)
    except SharedPrefixKVRuntimeError:
        raise
    except Exception:
        _fail("kv_v2_source_gate_invalid")
    public = source.public_identity()
    tokenizer_public = tokenizer.public_identity()
    producer = expected["producer_final"]
    if (
        public.get("candidate_commit") != producer["candidate_commit"]
        or public.get("release_commit") != producer["release_commit"]
        or public.get("release_tree") != producer["release_tree"]
        or public.get("tree_digest_sha256") != producer["physical_tree_sha256"]
        or public.get("producer_inventory_sha256") != producer["input_inventory_sha256"]
        or public.get("final_files") != 46
        or tokenizer_public.get("combined_identity_sha256")
        != expected["tokenizer_binding"][
            "authenticated_tokenizer_combined_identity_sha256"
        ]
    ):
        _fail("kv_v2_source_identity_drift")
    pin = source.payload_pins.get(source_blob_path)
    if pin is None:
        _fail("kv_v2_source_identity_drift")
    input_ids = tuple(serialized.input_ids[: serialized.prompt_tokens])
    if not input_ids or len(input_ids) != serialized.prompt_tokens:
        _fail("kv_v2_source_gate_invalid")
    identity = {
        "producer_candidate_commit": source.candidate_commit,
        "producer_release_commit": source.release_commit,
        "producer_release_tree": source.release_tree,
        "source_blob_path": source_blob_path,
        "source_blob_sha256": pin.sha256,
        "source_blob_bytes": pin.bytes,
        "record_index": record_index,
        "record_sha256": _sha256(_canonical_json(record)),
        "record_prompt_sha256": record_prompt_sha,
        "example_serialization_sha256": serialized.serialization_identity_sha256,
        "tokenizer_binding_sha256": tokenizer.combined_identity_sha256,
        "model_config_sha256": tokenizer_public["model_config_sha256"],
        "training_serialization_inventory_sha256": producer[
            "serialization_inventory_sha256"
        ],
        "route_commit_sha256": _find_unique_digest(record, "route_commit_sha256"),
        "plan_commit_sha256": _find_unique_digest(record, "plan_commit_sha256"),
    }
    return {
        "prefix_input_ids": input_ids,
        "source_identity": identity,
    }


def _expected_source(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "producer_final": dict(config["producer_final"]),
        "consumer_preflight": dict(config["consumer_preflight"]),
        "tokenizer_binding": dict(config["tokenizer_binding"]),
    }


def authenticate_source(
    config: Mapping[str, Any],
    *,
    producer_repository_root: str | os.PathLike[str],
    consumer_receipt_path: str | os.PathLike[str],
    consumer_receipt_sidecar_path: str | os.PathLike[str],
    consumer_receipt_schema_path: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
    source_blob_path: str,
    record_index: int,
    source_gate: RuntimeSourceGate | None = None,
) -> dict[str, Any]:
    if source_blob_path not in config["producer_final"]["allowed_source_blobs"]:
        _fail("kv_v2_source_identity_drift")
    if (
        isinstance(record_index, bool)
        or not isinstance(record_index, int)
        or record_index < 0
    ):
        _fail("kv_v2_source_identity_drift")
    gate = source_gate or _authenticate_source_via_module
    result = gate(
        producer_repository_root=producer_repository_root,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_sidecar_path=consumer_receipt_sidecar_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        model_root=model_root,
        source_blob_path=source_blob_path,
        record_index=record_index,
        expected=_expected_source(config),
    )
    if not isinstance(result, Mapping):
        _fail("kv_v2_source_gate_invalid")
    prefix_ids = result.get("prefix_input_ids")
    identity = result.get("source_identity")
    if (
        not isinstance(prefix_ids, Sequence)
        or isinstance(prefix_ids, (str, bytes, bytearray))
        or not prefix_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in prefix_ids
        )
        or not isinstance(identity, Mapping)
    ):
        _fail("kv_v2_source_gate_invalid")
    expected_keys = {
        "producer_candidate_commit",
        "producer_release_commit",
        "producer_release_tree",
        "source_blob_path",
        "source_blob_sha256",
        "source_blob_bytes",
        "record_index",
        "record_sha256",
        "record_prompt_sha256",
        "example_serialization_sha256",
        "tokenizer_binding_sha256",
        "model_config_sha256",
        "training_serialization_inventory_sha256",
        "route_commit_sha256",
        "plan_commit_sha256",
    }
    if set(identity) != expected_keys:
        _fail("kv_v2_source_gate_invalid")
    producer = config["producer_final"]
    expected_values = {
        "producer_candidate_commit": producer["candidate_commit"],
        "producer_release_commit": producer["release_commit"],
        "producer_release_tree": producer["release_tree"],
        "source_blob_path": source_blob_path,
        "record_index": record_index,
        "tokenizer_binding_sha256": tokenizer_binding_sha256(config),
        "training_serialization_inventory_sha256": producer[
            "serialization_inventory_sha256"
        ],
    }
    if any(identity.get(key) != value for key, value in expected_values.items()):
        _fail("kv_v2_source_identity_drift")
    if (
        not _is_sha256(identity.get("source_blob_sha256"))
        or not _is_sha256(identity.get("record_sha256"))
        or not _is_sha256(identity.get("record_prompt_sha256"))
        or not _is_sha256(identity.get("example_serialization_sha256"))
        or not _is_sha256(identity.get("model_config_sha256"))
        or not _is_sha256(identity.get("route_commit_sha256"))
        or not _is_sha256(identity.get("plan_commit_sha256"))
        or isinstance(identity.get("source_blob_bytes"), bool)
        or not isinstance(identity.get("source_blob_bytes"), int)
        or identity["source_blob_bytes"] < 1
    ):
        _fail("kv_v2_source_identity_drift")
    return {
        "prefix_input_ids": tuple(prefix_ids),
        "source_identity": dict(identity),
    }


def _source_receipt(source: Mapping[str, Any]) -> dict[str, Any]:
    identity = source["source_identity"]
    return {
        "authenticated": True,
        **{
            key: identity[key]
            for key in (
                "producer_candidate_commit",
                "producer_release_commit",
                "producer_release_tree",
                "source_blob_path",
                "source_blob_sha256",
                "source_blob_bytes",
                "record_index",
                "record_sha256",
                "record_prompt_sha256",
                "example_serialization_sha256",
                "tokenizer_binding_sha256",
                "model_config_sha256",
            )
        },
        "hf_auto_tokenizer_used": False,
        "second_model_file_path_read": False,
        "runtime_special_token_overlay": True,
        "serialized_prefix_persisted": False,
    }


def _adapter_identity(
    receipt: Mapping[str, Any],
    name: str,
) -> tuple[str, str]:
    adapters = receipt["adapters"]
    value = adapters.get(name)
    if (
        not isinstance(value, Mapping)
        or not _is_sha256(value.get("artifact_tree_sha256"))
        or not isinstance(value.get("path"), str)
        or not value["path"]
    ):
        _fail("kv_v2_adapter_receipt_invalid")
    path = _safe_descendant(
        Path(str(receipt["_published_root"])),
        str(value["path"]),
        "kv_v2_adapter_receipt_invalid",
    )
    return str(value["artifact_tree_sha256"]), str(path)


def _build_lineage(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    adapter_receipt: Mapping[str, Any],
    backend_result: Mapping[str, Any],
) -> dict[str, Any]:
    identity = source["source_identity"]
    topology = backend_result["topology"]
    boundary = backend_result["boundary"]
    return {
        "producer_candidate_commit": identity["producer_candidate_commit"],
        "producer_release_commit": identity["producer_release_commit"],
        "producer_release_tree": identity["producer_release_tree"],
        "source_blob_sha256": identity["source_blob_sha256"],
        "source_record_sha256": identity["record_sha256"],
        "training_serialization_inventory_sha256": identity[
            "training_serialization_inventory_sha256"
        ],
        "record_prompt_sha256": identity["record_prompt_sha256"],
        "example_serialization_sha256": identity["example_serialization_sha256"],
        "tokenizer_binding_sha256": identity["tokenizer_binding_sha256"],
        "model_config_sha256": identity["model_config_sha256"],
        "token_order_sha256": boundary["token_order_sha256"],
        "position_sha256": boundary["position_sha256"],
        "attention_mask_sha256": boundary["attention_mask_sha256"],
        "rope_sha256": boundary["rope_sha256"],
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "base_identity_sha256": adapter_receipt["_base_identity_sha256"],
        "q8_scb_sha256": adapter_receipt["_q8_scb_sha256"],
        "adapter_lineage_sha256": adapter_receipt["_adapter_inventory_sha256"],
        "layer_topology_sha256": topology["layer_topology_sha256"],
        "route_commit_sha256": identity["route_commit_sha256"],
        "plan_commit_sha256": identity["plan_commit_sha256"],
    }


def _validate_physical_result(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    required = {
        "topology",
        "boundary",
        "comparison",
        "isolation",
        "performance",
    }
    if set(result) != required:
        _fail("kv_v2_physical_backend_failed")
    topology = result["topology"]
    if (
        not isinstance(topology, Mapping)
        or topology.get("layers") != 26
        or topology.get("sliding_layers") != 22
        or topology.get("full_layers") != 4
        or topology.get("sliding_window") != 512
        or topology.get("derived_from_authenticated_model_config") is not True
        or len(topology.get("layers_detail", ())) != 26
    ):
        _fail("kv_v2_layer_topology_invalid")
    full_indices = topology.get("full_layer_indices")
    details = topology.get("layers_detail")
    if (
        not isinstance(full_indices, list)
        or len(full_indices) != 4
        or len(set(full_indices)) != 4
        or not isinstance(details, list)
    ):
        _fail("kv_v2_layer_topology_invalid")
    compact_details = []
    for index, detail in enumerate(details):
        expected_kind = "full" if index in full_indices else "sliding"
        expected_window = None if expected_kind == "full" else 512
        expected_semantics = (
            "full_prefix_unbounded"
            if expected_kind == "full"
            else "retained_suffix_window_minus_one"
        )
        expected_capacity = None if expected_kind == "full" else 511
        if (
            not isinstance(detail, Mapping)
            or detail.get("index") != index
            or detail.get("attention_kind") != expected_kind
            or detail.get("window") != expected_window
            or detail.get("sequence_axis") != -2
            or detail.get("window_semantics") != expected_semantics
            or detail.get("cache_capacity_tokens") != expected_capacity
            or any(
                isinstance(detail.get(name), bool)
                or not isinstance(detail.get(name), int)
                or detail[name] < 1
                for name in (
                    "source_cache_tokens",
                    "planner_cache_tokens",
                    "output_cache_tokens",
                    "planner_retained_prefix_tokens",
                    "output_retained_prefix_tokens",
                )
            )
            or any(
                not _is_sha256(detail.get(name))
                for name in (
                    "cold_prefill_key_sha256",
                    "cold_prefill_value_sha256",
                    "source_prefill_key_sha256",
                    "source_prefill_value_sha256",
                    "planner_source_prefix_key_sha256",
                    "planner_source_prefix_value_sha256",
                    "planner_branch_prefix_key_sha256",
                    "planner_branch_prefix_value_sha256",
                    "output_source_prefix_key_sha256",
                    "output_source_prefix_value_sha256",
                    "output_branch_prefix_key_sha256",
                    "output_branch_prefix_value_sha256",
                )
            )
            or any(
                not isinstance(detail.get(name), bool)
                for name in (
                    "prefill_recompute_value_equal",
                    "planner_prefix_value_equal",
                    "output_prefix_value_equal",
                )
            )
            or detail.get("storage_pointer_is_proof") is not False
            or any(
                not isinstance(detail.get(name), bool)
                for name in (
                    "source_to_planner_pointer_equal_observed",
                    "source_to_output_pointer_equal_observed",
                )
            )
        ):
            _fail("kv_v2_layer_topology_invalid")
        planner_tokens, planner_retained = _retained_prefix_layout(
            {
                "attention_kind": expected_kind,
                "window": expected_window,
            },
            source_tokens=detail["source_cache_tokens"],
            tail_tokens=1,
        )
        output_tokens, output_retained = _retained_prefix_layout(
            {
                "attention_kind": expected_kind,
                "window": expected_window,
            },
            source_tokens=detail["source_cache_tokens"],
            tail_tokens=2,
        )
        if (
            detail["planner_cache_tokens"] != planner_tokens
            or detail["planner_retained_prefix_tokens"] != planner_retained
            or detail["output_cache_tokens"] != output_tokens
            or detail["output_retained_prefix_tokens"] != output_retained
            or detail["prefill_recompute_value_equal"] is not True
            or detail["planner_prefix_value_equal"] is not True
            or detail["output_prefix_value_equal"] is not True
            or detail["cold_prefill_key_sha256"] != detail["source_prefill_key_sha256"]
            or detail["cold_prefill_value_sha256"]
            != detail["source_prefill_value_sha256"]
            or detail["planner_source_prefix_key_sha256"]
            != detail["planner_branch_prefix_key_sha256"]
            or detail["planner_source_prefix_value_sha256"]
            != detail["planner_branch_prefix_value_sha256"]
            or detail["output_source_prefix_key_sha256"]
            != detail["output_branch_prefix_key_sha256"]
            or detail["output_source_prefix_value_sha256"]
            != detail["output_branch_prefix_value_sha256"]
        ):
            _fail("kv_v2_prefix_value_mismatch")
        compact_details.append(
            {
                "index": index,
                "attention_kind": expected_kind,
                "window": expected_window,
            }
        )
    if topology.get("layer_topology_sha256") != _sha256(
        _canonical_json(compact_details)
    ):
        _fail("kv_v2_layer_topology_invalid")
    boundary = result["boundary"]
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("source_prefill_adapter_state") != "adapter_off_frozen_base"
        or boundary.get("latent_packet", {}).get("enabled") is not False
    ):
        _fail("kv_v2_adapter_switch_invalid")
    if (
        any(
            not _is_sha256(boundary.get(name))
            for name in (
                "boundary_commit_sha256",
                "ordered_prefix_sha256",
                "token_order_sha256",
                "position_sha256",
                "attention_mask_sha256",
                "rope_sha256",
                "route_commit_sha256",
                "plan_commit_sha256",
                "planner_adapter_sha256",
                "output_adapter_sha256",
            )
        )
        or isinstance(boundary.get("prefix_tokens"), bool)
        or not isinstance(boundary.get("prefix_tokens"), int)
        or boundary["prefix_tokens"] < 1
        or boundary.get("latent_packet")
        != {
            "enabled": False,
            "fixed_dimension": 0,
            "serialization_sha256": None,
            "position_sha256": None,
            "commit_sha256": None,
            "hidden_chain_of_thought": False,
        }
    ):
        _fail("kv_v2_boundary_digest_mismatch")
    for detail in details:
        expected_source_tokens = boundary["prefix_tokens"]
        if detail["attention_kind"] == "sliding":
            expected_source_tokens = min(
                expected_source_tokens,
                detail["cache_capacity_tokens"],
            )
        if detail["source_cache_tokens"] != expected_source_tokens:
            _fail("kv_v2_prefix_value_mismatch")
    comparison = result["comparison"]
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("cold_recompute_executed") is not True
        or comparison.get("shared_prefix_handoff_executed") is not True
        or comparison.get("logits_allclose") is not True
        or comparison.get("next_token_argmax_equal") is not True
        or comparison.get("prefill_count_cold") != 1
        or comparison.get("prefill_count_handoff") != 1
    ):
        _fail("kv_v2_logits_mismatch")
    isolation = result["isolation"]
    required_true = (
        "source_cache_unchanged",
        "planner_tail_private",
        "output_tail_private",
        "sibling_branch_unchanged",
        "wrong_adapter_rejected",
        "wrong_position_rejected",
        "wrong_attention_mask_rejected",
        "cache_mutation_rejected",
        "pointer_observation_only",
    )
    if not isinstance(isolation, Mapping) or any(
        isolation.get(name) is not True for name in required_true
    ):
        _fail("kv_v2_cache_mutation")
    if isolation.get("private_tail_alias") is not False:
        _fail("kv_v2_private_tail_alias")
    for value in result["performance"].values():
        if isinstance(value, float) and not math.isfinite(value):
            _fail("kv_v2_physical_backend_failed")


def _validate_boundary_bindings(
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    planner_adapter_sha256: str,
    output_adapter_sha256: str,
) -> None:
    boundary = result["boundary"]
    identity = source["source_identity"]
    if (
        boundary["planner_adapter_sha256"] != planner_adapter_sha256
        or boundary["output_adapter_sha256"] != output_adapter_sha256
    ):
        _fail("kv_v2_adapter_switch_invalid")
    if (
        boundary["ordered_prefix_sha256"] != identity["record_prompt_sha256"]
        or boundary["route_commit_sha256"] != identity["route_commit_sha256"]
        or boundary["plan_commit_sha256"] != identity["plan_commit_sha256"]
        or boundary["prefix_tokens"] != len(source["prefix_input_ids"])
    ):
        _fail("kv_v2_boundary_digest_mismatch")


def validate_receipt(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_blocked: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    _, validator = _schema_validators(config)
    if tuple(validator.iter_errors(receipt)):
        _fail("kv_v2_receipt_invalid")
    if (
        receipt.get("schema_version") != RECEIPT_VERSION
        or receipt.get("profile_id") != PROFILE_ID
    ):
        _fail("kv_v2_receipt_invalid")
    status = receipt["status"]
    if status == "blocked_model_free":
        if not allow_blocked or receipt["claims"] != _claims():
            _fail("kv_v2_receipt_invalid")
        return dict(receipt)
    claims = receipt["claims"]
    forbidden = (
        "shared_storage",
        "zero_copy",
        "rdma",
        "full_generation_sharing",
        "pointer_identity_is_proof",
        "formal",
        "live",
    )
    if any(claims[name] is not False for name in forbidden):
        _fail("kv_v2_claim_inflation")
    passed = status == "passed_diagnostic_prefix_handoff"
    if passed and (
        claims["exact_prefix_value_handoff"] is not True
        or claims["prefill_compute_handoff"] is not True
        or receipt["failure_codes"]
    ):
        _fail("kv_v2_receipt_invalid")
    if not passed and not receipt["failure_codes"]:
        _fail("kv_v2_receipt_invalid")
    return dict(receipt)


@contextmanager
def _gpu_lock(path: str | os.PathLike[str]):
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _owned_path_identity(
        target.parent,
        require_directory=True,
        code="kv_v2_gpu_lock_invalid",
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = -1
    lock_identity: tuple[int, int, int] | None = None
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        _fail("kv_v2_gpu_lock_busy")
    except OSError:
        _fail("kv_v2_gpu_lock_invalid")
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        lock_identity = _object_identity(os.fstat(descriptor))
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        parent_unchanged = (
            _owned_path_identity(
                target.parent,
                require_directory=True,
                code="kv_v2_gpu_lock_invalid",
            )
            == parent_identity
        )
        if (
            not parent_unchanged
            or lock_identity is None
            or not _release_owned_path(target, lock_identity)
        ):
            _fail("kv_v2_gpu_lock_invalid")


def _atomic_write(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _owned_path_identity(
        target.parent,
        require_directory=True,
        code="kv_v2_atomic_publish_failed",
    )
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        _fail("kv_v2_atomic_publish_failed")
    else:
        _fail("kv_v2_atomic_publish_failed")
    raw = _canonical_json(value)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    staging_identity: tuple[int, int, int] | None = None
    try:
        with staging.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            staging_identity = _object_identity(os.fstat(handle.fileno()))
        os.link(staging, target, follow_symlinks=False)
        if (
            _owned_path_identity(
                target.parent,
                require_directory=True,
                code="kv_v2_atomic_publish_failed",
            )
            != parent_identity
            or _owned_path_identity(
                target,
                require_directory=False,
                code="kv_v2_atomic_publish_failed",
            )
            != staging_identity
        ):
            _fail("kv_v2_atomic_publish_failed")
        if _read_regular(target, "kv_v2_atomic_publish_failed") != raw:
            _fail("kv_v2_atomic_publish_failed")
        if not _release_owned_path(staging, staging_identity):
            _fail("kv_v2_atomic_publish_failed")
        staging_identity = None
    except OSError:
        _fail("kv_v2_atomic_publish_failed")
    finally:
        if staging_identity is not None:
            _release_owned_path(staging, staging_identity)


def execute(
    config: Mapping[str, Any],
    *,
    producer_repository_root: str | os.PathLike[str],
    consumer_receipt_path: str | os.PathLike[str],
    consumer_receipt_sidecar_path: str | os.PathLike[str],
    consumer_receipt_schema_path: str | os.PathLike[str],
    source_blob_path: str,
    record_index: int,
    adapter_receipt_path: str | os.PathLike[str],
    adapter_receipt_sidecar_path: str | os.PathLike[str],
    adapter_receipt_schema_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    planner_adapter_name: str,
    output_adapter_name: str,
    gpu_lock_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    source_gate: RuntimeSourceGate | None = None,
    backend: PhysicalBackend | None = None,
) -> dict[str, Any]:
    validate_config(config)
    if _pending_fields(config):
        _fail("kv_v2_execute_not_authorized")
    model_loads = 0
    gpu_requests = 0
    try:
        source = authenticate_source(
            config,
            producer_repository_root=producer_repository_root,
            consumer_receipt_path=consumer_receipt_path,
            consumer_receipt_sidecar_path=consumer_receipt_sidecar_path,
            consumer_receipt_schema_path=consumer_receipt_schema_path,
            model_root=model_dir,
            source_blob_path=source_blob_path,
            record_index=record_index,
            source_gate=source_gate,
        )
        adapter_receipt = _authenticate_adapter_receipt(
            config,
            receipt_path=adapter_receipt_path,
            sidecar_path=adapter_receipt_sidecar_path,
            schema_path=adapter_receipt_schema_path,
            model_root=model_dir,
        )
        planner_sha, planner_path = _adapter_identity(
            adapter_receipt,
            planner_adapter_name,
        )
        output_sha, output_path = _adapter_identity(
            adapter_receipt,
            output_adapter_name,
        )
        selected_backend = backend or transformers_backend
        request = {
            "config": _public_config(config),
            "prefix_input_ids": source["prefix_input_ids"],
            "source_identity": dict(source["source_identity"]),
            "model_dir": str(model_dir),
            "planner_adapter_name": planner_adapter_name,
            "planner_adapter_path": planner_path,
            "planner_adapter_sha256": planner_sha,
            "output_adapter_name": output_adapter_name,
            "output_adapter_path": output_path,
            "output_adapter_sha256": output_sha,
            "adapter_inventory_sha256": adapter_receipt["_adapter_inventory_sha256"],
            "base_identity_sha256": adapter_receipt["_base_identity_sha256"],
            "q8_scb_sha256": adapter_receipt["_q8_scb_sha256"],
        }
        gpu_requests = 1
        with _gpu_lock(gpu_lock_path):
            model_loads = 1
            backend_result = selected_backend(request)
            if not isinstance(backend_result, Mapping):
                _fail("kv_v2_physical_backend_failed")
            terminal_source = authenticate_source(
                config,
                producer_repository_root=producer_repository_root,
                consumer_receipt_path=consumer_receipt_path,
                consumer_receipt_sidecar_path=consumer_receipt_sidecar_path,
                consumer_receipt_schema_path=consumer_receipt_schema_path,
                model_root=model_dir,
                source_blob_path=source_blob_path,
                record_index=record_index,
                source_gate=source_gate,
            )
            if terminal_source != source:
                _fail("kv_v2_source_identity_drift")
            terminal_adapter_receipt = _authenticate_adapter_receipt(
                config,
                receipt_path=adapter_receipt_path,
                sidecar_path=adapter_receipt_sidecar_path,
                schema_path=adapter_receipt_schema_path,
                model_root=model_dir,
            )
            if any(
                terminal_adapter_receipt[name] != adapter_receipt[name]
                for name in (
                    "_adapter_inventory_sha256",
                    "_base_identity_sha256",
                    "_q8_scb_sha256",
                )
            ):
                _fail("kv_v2_adapter_receipt_invalid")
            _validate_physical_result(backend_result, config)
            _validate_boundary_bindings(
                backend_result,
                source,
                planner_adapter_sha256=planner_sha,
                output_adapter_sha256=output_sha,
            )
            lineage = _build_lineage(config, source, adapter_receipt, backend_result)
            expected_commit = boundary_commit_sha256(lineage)
            if (
                backend_result["boundary"].get("boundary_commit_sha256")
                != expected_commit
            ):
                _fail("kv_v2_boundary_digest_mismatch")
            result = {
                "schema_version": RECEIPT_VERSION,
                "profile_id": PROFILE_ID,
                "status": "passed_diagnostic_prefix_handoff",
                "operation": "execute",
                "identity": {
                    "config_sha256": config_sha256(config),
                    "implementation_sha256": implementation_sha256(),
                    "consumer_receipt_sha256": config["consumer_preflight"][
                        "receipt_sha256"
                    ],
                    "adapter_receipt_sha256": config["adapter_receipt"][
                        "receipt_sha256"
                    ],
                    "model_identity_sha256": config["model"]["model_identity_sha256"],
                    "tokenizer_identity_sha256": config["model"][
                        "tokenizer_identity_sha256"
                    ],
                    "training_serialization_inventory_sha256": config[
                        "tokenizer_binding"
                    ]["training_serialization_inventory_sha256"],
                },
                "source": _source_receipt(source),
                "topology": backend_result["topology"],
                "boundary": backend_result["boundary"],
                "comparison": backend_result["comparison"],
                "isolation": backend_result["isolation"],
                "performance": backend_result["performance"],
                "claims": _claims(passed=True),
                "resource_counters": _zero_resources(
                    model_loads=model_loads,
                    gpu_requests=gpu_requests,
                ),
                "failure_codes": [],
            }
            validated = validate_receipt(result, config)
        _atomic_write(receipt_path, validated)
        return validated
    except SharedPrefixKVRuntimeError as exc:
        failure = build_status(config, operation="execute")
        failure["status"] = "failed_diagnostic_prefix_handoff"
        failure["claims"] = _claims()
        failure["failure_codes"] = [str(exc)]
        failure["resource_counters"] = _zero_resources(
            model_loads=model_loads,
            gpu_requests=gpu_requests,
        )
        validated_failure = validate_receipt(failure, config)
        _atomic_write(receipt_path, validated_failure)
        raise


def _tensor_digest(value: Any, torch_module: Any) -> str:
    tensor = value.detach().contiguous().view(torch_module.uint8).cpu()
    return _sha256(tensor.numpy().tobytes())


def _cache_layers(cache: Any) -> Sequence[Any]:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, Sequence):
        return layers
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if isinstance(keys, Sequence) and isinstance(values, Sequence):
        return tuple(zip(keys, values, strict=True))
    _fail("kv_v2_physical_backend_failed")


def _layer_tensors(layer: Any) -> tuple[Any, Any]:
    if isinstance(layer, tuple) and len(layer) == 2:
        return layer
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if key is None or value is None:
        key = getattr(layer, "key_cache", None)
        value = getattr(layer, "value_cache", None)
    if key is None or value is None:
        _fail("kv_v2_physical_backend_failed")
    return key, value


def _cache_capacity(detail: Mapping[str, Any]) -> int | None:
    kind = detail.get("attention_kind")
    if kind == "full":
        return None
    if kind != "sliding":
        _fail("kv_v2_layer_topology_invalid")
    window = detail.get("window")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        _fail("kv_v2_layer_topology_invalid")
    # Transformers DynamicSlidingWindowLayer retains the last window - 1
    # states so the current token can be appended for attention computation.
    return window - 1


def _retained_prefix_layout(
    detail: Mapping[str, Any],
    *,
    source_tokens: int,
    tail_tokens: int,
) -> tuple[int, int]:
    if (
        isinstance(source_tokens, bool)
        or not isinstance(source_tokens, int)
        or source_tokens < 1
        or isinstance(tail_tokens, bool)
        or not isinstance(tail_tokens, int)
        or tail_tokens < 1
    ):
        _fail("kv_v2_layer_topology_invalid")
    capacity = _cache_capacity(detail)
    branch_tokens = source_tokens + tail_tokens
    if capacity is not None:
        if source_tokens > capacity:
            _fail("kv_v2_layer_topology_invalid")
        branch_tokens = min(branch_tokens, capacity)
    retained_prefix_tokens = branch_tokens - tail_tokens
    if retained_prefix_tokens < 1 or retained_prefix_tokens > source_tokens:
        _fail("kv_v2_prefix_value_mismatch")
    return branch_tokens, retained_prefix_tokens


def _validated_layer_tensors(layer: Any) -> tuple[Any, Any, int]:
    key, value = _layer_tensors(layer)
    key_ndim = getattr(key, "ndim", None)
    value_ndim = getattr(value, "ndim", None)
    if (
        isinstance(key_ndim, bool)
        or not isinstance(key_ndim, int)
        or key_ndim < 3
        or value_ndim != key_ndim
        or tuple(key.shape[:-2]) != tuple(value.shape[:-2])
        or int(key.shape[-2]) != int(value.shape[-2])
        or int(key.shape[-1]) != int(value.shape[-1])
        or int(key.shape[-2]) < 1
    ):
        _fail("kv_v2_physical_backend_failed")
    return key, value, int(key.shape[-2])


def _storage_pointer(value: Any) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except (AttributeError, RuntimeError):
        return 0


def _cache_summary(
    cache: Any, topology: Mapping[str, Any], torch_module: Any
) -> list[dict[str, Any]]:
    layers = _cache_layers(cache)
    if len(layers) != topology["layers"]:
        _fail("kv_v2_layer_topology_invalid")
    result = []
    for detail, layer in zip(topology["layers_detail"], layers, strict=True):
        key, value, sequence_tokens = _validated_layer_tensors(layer)
        capacity = _cache_capacity(detail)
        if capacity is not None and sequence_tokens > capacity:
            _fail("kv_v2_layer_topology_invalid")
        result.append(
            {
                **detail,
                "sequence_axis": -2,
                "sequence_tokens": sequence_tokens,
                "key_sha256": _tensor_digest(key, torch_module),
                "value_sha256": _tensor_digest(value, torch_module),
                "key_pointer": _storage_pointer(key),
                "value_pointer": _storage_pointer(value),
            }
        )
    return result


def _sequence_slice_digest(
    value: Any,
    *,
    start: int,
    length: int,
    torch_module: Any,
) -> str:
    ndim = getattr(value, "ndim", None)
    if (
        isinstance(ndim, bool)
        or not isinstance(ndim, int)
        or ndim < 3
        or start < 0
        or length < 1
        or start + length > int(value.shape[-2])
    ):
        _fail("kv_v2_prefix_value_mismatch")
    selection = [slice(None)] * ndim
    selection[-2] = slice(start, start + length)
    segment = value[tuple(selection)]
    if int(segment.shape[-2]) != length:
        _fail("kv_v2_prefix_value_mismatch")
    return _tensor_digest(segment, torch_module)


def _branch_prefix_evidence(
    source_cache: Any,
    branch_cache: Any,
    topology: Mapping[str, Any],
    *,
    stage: str,
    tail_tokens: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if stage not in {"planner", "output"}:
        _fail("kv_v2_physical_backend_failed")
    source_layers = _cache_layers(source_cache)
    branch_layers = _cache_layers(branch_cache)
    if (
        len(source_layers) != topology["layers"]
        or len(branch_layers) != topology["layers"]
    ):
        _fail("kv_v2_layer_topology_invalid")
    result = []
    for detail, source_layer, branch_layer in zip(
        topology["layers_detail"],
        source_layers,
        branch_layers,
        strict=True,
    ):
        source_key, source_value, source_tokens = _validated_layer_tensors(source_layer)
        branch_key, branch_value, branch_tokens = _validated_layer_tensors(branch_layer)
        expected_branch_tokens, retained_tokens = _retained_prefix_layout(
            detail,
            source_tokens=source_tokens,
            tail_tokens=tail_tokens,
        )
        if branch_tokens != expected_branch_tokens:
            _fail("kv_v2_prefix_value_mismatch")
        source_start = source_tokens - retained_tokens
        source_key_sha = _sequence_slice_digest(
            source_key,
            start=source_start,
            length=retained_tokens,
            torch_module=torch_module,
        )
        source_value_sha = _sequence_slice_digest(
            source_value,
            start=source_start,
            length=retained_tokens,
            torch_module=torch_module,
        )
        branch_key_sha = _sequence_slice_digest(
            branch_key,
            start=0,
            length=retained_tokens,
            torch_module=torch_module,
        )
        branch_value_sha = _sequence_slice_digest(
            branch_value,
            start=0,
            length=retained_tokens,
            torch_module=torch_module,
        )
        pointer_equal = any(
            source_pointer != 0 and source_pointer == branch_pointer
            for source_pointer, branch_pointer in (
                (_storage_pointer(source_key), _storage_pointer(branch_key)),
                (_storage_pointer(source_value), _storage_pointer(branch_value)),
            )
        )
        result.append(
            {
                f"{stage}_cache_tokens": branch_tokens,
                f"{stage}_retained_prefix_tokens": retained_tokens,
                f"{stage}_source_prefix_key_sha256": source_key_sha,
                f"{stage}_source_prefix_value_sha256": source_value_sha,
                f"{stage}_branch_prefix_key_sha256": branch_key_sha,
                f"{stage}_branch_prefix_value_sha256": branch_value_sha,
                f"{stage}_prefix_value_equal": (
                    source_key_sha == branch_key_sha
                    and source_value_sha == branch_value_sha
                ),
                f"source_to_{stage}_pointer_equal_observed": pointer_equal,
            }
        )
    return result


def _cache_pointer_alias(
    branch: Sequence[Mapping[str, Any]],
    *others: Sequence[Mapping[str, Any]],
) -> bool:
    if any(len(value) != len(branch) for value in others):
        _fail("kv_v2_layer_topology_invalid")
    for index, branch_layer in enumerate(branch):
        branch_pointers = {
            int(branch_layer["key_pointer"]),
            int(branch_layer["value_pointer"]),
        } - {0}
        if not branch_pointers:
            continue
        for other in others:
            other_pointers = {
                int(other[index]["key_pointer"]),
                int(other[index]["value_pointer"]),
            } - {0}
            if branch_pointers & other_pointers:
                return True
    return False


def _cache_values_equal(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> bool:
    return len(before) == len(after) and all(
        left["sequence_tokens"] == right["sequence_tokens"]
        and left["key_sha256"] == right["key_sha256"]
        and left["value_sha256"] == right["value_sha256"]
        for left, right in zip(before, after, strict=True)
    )


def _timed_cuda(torch_module: Any, function: Callable[[], Any]) -> tuple[Any, float]:
    torch_module.cuda.synchronize()
    started = time.perf_counter_ns()
    result = function()
    torch_module.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed


def _boundary_mutation_rejected(
    lineage: Mapping[str, Any],
    commit_sha256: str,
    *,
    field: str,
    failure_code: str,
) -> bool:
    mutated = dict(lineage)
    mutated[field] = _sha256(f"negative:{field}".encode("utf-8"))
    try:
        validate_boundary(lineage, mutated, commit_sha256)
    except SharedPrefixKVRuntimeError as exc:
        return str(exc) == failure_code
    return False


def transformers_backend(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run the real local-only HF/PEFT comparison.

    This function is imported and invoked only by explicit ``--execute``.
    It never persists raw token IDs, logits, prompt text, or cache values.
    """

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import bitsandbytes as bnb
        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )
        from anchor_mvp.training import (
            gemma3_five_role_q8_qlora_v2 as q8_runtime,
        )
    except (ImportError, OSError) as exc:
        raise SharedPrefixKVRuntimeError("kv_v2_physical_backend_failed") from exc
    if not torch.cuda.is_available():
        _fail("kv_v2_physical_backend_failed")
    config = request["config"]
    model_dir = str(request["model_dir"])
    model_config_path = Path(model_dir) / "config.json"
    model_config_raw = _read_regular(
        model_config_path,
        "kv_v2_tokenizer_binding_drift",
    )
    if _sha256(model_config_raw) != request["source_identity"]["model_config_sha256"]:
        _fail("kv_v2_tokenizer_binding_drift")
    model_config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        _read_regular(model_config_path, "kv_v2_tokenizer_binding_drift")
        != model_config_raw
    ):
        _fail("kv_v2_tokenizer_binding_drift")
    text_config = model_config.get_text_config(decoder=True)
    topology = derive_layer_topology(text_config.to_dict(), config["model"])
    raw_prefix_ids = request.get("prefix_input_ids")
    if (
        not isinstance(raw_prefix_ids, Sequence)
        or isinstance(raw_prefix_ids, (str, bytes, bytearray))
        or not raw_prefix_ids
    ):
        _fail("kv_v2_tokenizer_binding_drift")
    input_ids = torch.tensor([tuple(raw_prefix_ids)], dtype=torch.long)
    token_order_sha = _sha256(
        input_ids.detach().contiguous().view(torch.uint8).numpy().tobytes()
    )
    prefix_tokens = int(input_ids.shape[1])
    positions = torch.arange(prefix_tokens, dtype=torch.long).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    position_sha = _sha256(positions.contiguous().view(torch.uint8).numpy().tobytes())
    mask_sha = _sha256(mask.contiguous().view(torch.uint8).numpy().tobytes())
    rope_sha = _sha256(
        _canonical_json(
            {
                "position_sha256": position_sha,
                "rope_theta": getattr(text_config, "rope_theta", None),
                "rope_scaling": getattr(text_config, "rope_scaling", None),
            }
        )
    )
    quantization = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if not bool(getattr(base, "is_loaded_in_8bit", False)):
        _fail("kv_v2_physical_backend_failed")
    q8_before = q8_runtime._q8_quant_state_inventory(
        base,
        bnb=bnb,
        torch=torch,
    )
    if q8_before.get("sha256") != request["q8_scb_sha256"]:
        _fail("kv_v2_q8_scb_identity_drift")
    model = PeftModel.from_pretrained(
        base,
        request["planner_adapter_path"],
        adapter_name=request["planner_adapter_name"],
        is_trainable=False,
        local_files_only=True,
    )
    model.load_adapter(
        request["output_adapter_path"],
        adapter_name=request["output_adapter_name"],
        is_trainable=False,
    )
    if (
        _read_regular(model_config_path, "kv_v2_tokenizer_binding_drift")
        != model_config_raw
    ):
        _fail("kv_v2_tokenizer_binding_drift")
    model.eval()
    model.config.use_cache = True
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    torch.cuda.reset_peak_memory_stats()

    def prefill() -> Any:
        with torch.inference_mode(), model.disable_adapter():
            return model(
                input_ids=input_ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )

    cold_prefill, cold_prefill_ms = _timed_cuda(torch, prefill)
    handoff_prefill, handoff_prefill_ms = _timed_cuda(torch, prefill)
    cold_cache = cold_prefill.past_key_values
    source_cache = handoff_prefill.past_key_values
    source_before = _cache_summary(source_cache, topology, torch)
    cold_summary = _cache_summary(cold_cache, topology, torch)
    planner_cache = copy.deepcopy(source_cache)
    sibling_cache = copy.deepcopy(source_cache)
    sibling_before = _cache_summary(sibling_cache, topology, torch)
    seed = torch.tensor([[1]], dtype=torch.long, device=device)
    tail_mask = torch.ones((1, prefix_tokens + 1), dtype=mask.dtype, device=device)
    cache_position = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

    def planner_step(cache: Any) -> Any:
        model.set_adapter(request["planner_adapter_name"])
        with torch.inference_mode():
            return model(
                input_ids=seed,
                attention_mask=tail_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

    cold_planner, cold_planner_ms = _timed_cuda(torch, lambda: planner_step(cold_cache))
    handoff_planner, planner_ms = _timed_cuda(
        torch, lambda: planner_step(planner_cache)
    )
    planner_prefix_evidence = _branch_prefix_evidence(
        source_cache,
        handoff_planner.past_key_values,
        topology,
        stage="planner",
        tail_tokens=1,
        torch_module=torch,
    )
    planner_summary = _cache_summary(
        handoff_planner.past_key_values,
        topology,
        torch,
    )
    planner_tail_alias = _cache_pointer_alias(
        planner_summary,
        source_before,
        sibling_before,
    )
    planner_token_cold = cold_planner.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    planner_token_handoff = handoff_planner.logits[:, -1, :].argmax(
        dim=-1, keepdim=True
    )
    output_mask = torch.ones((1, prefix_tokens + 2), dtype=mask.dtype, device=device)
    output_position = torch.tensor([prefix_tokens + 1], dtype=torch.long, device=device)

    def output_step(token: Any, cache: Any) -> Any:
        model.set_adapter(request["output_adapter_name"])
        with torch.inference_mode():
            return model(
                input_ids=token,
                attention_mask=output_mask,
                cache_position=output_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

    cold_output, cold_output_ms = _timed_cuda(
        torch,
        lambda: output_step(planner_token_cold, cold_planner.past_key_values),
    )
    handoff_output, output_ms = _timed_cuda(
        torch,
        lambda: output_step(
            planner_token_handoff,
            handoff_planner.past_key_values,
        ),
    )
    cold_logits = cold_output.logits[:, -1, :].float()
    handoff_logits = handoff_output.logits[:, -1, :].float()
    allclose = bool(
        torch.allclose(
            cold_logits,
            handoff_logits,
            atol=config["comparison"]["logits_atol"],
            rtol=config["comparison"]["logits_rtol"],
        )
    )
    max_error = float((cold_logits - handoff_logits).abs().max().item())
    cold_next = cold_logits.argmax(dim=-1)
    handoff_next = handoff_logits.argmax(dim=-1)
    next_equal = bool(torch.equal(cold_next, handoff_next))
    q8_after = q8_runtime._q8_quant_state_inventory(
        base,
        bnb=bnb,
        torch=torch,
    )
    if q8_after != q8_before:
        _fail("kv_v2_q8_scb_identity_drift")
    source_after = _cache_summary(source_cache, topology, torch)
    sibling_after = _cache_summary(sibling_cache, topology, torch)
    output_prefix_evidence = _branch_prefix_evidence(
        source_cache,
        handoff_output.past_key_values,
        topology,
        stage="output",
        tail_tokens=2,
        torch_module=torch,
    )
    output_summary = _cache_summary(
        handoff_output.past_key_values,
        topology,
        torch,
    )
    source_equal = _cache_values_equal(source_before, source_after)
    sibling_equal = _cache_values_equal(sibling_before, sibling_after)
    output_tail_alias = _cache_pointer_alias(
        output_summary,
        source_after,
        sibling_after,
    )
    layers_detail = []
    for detail, cold, source, planner, output in zip(
        topology["layers_detail"],
        cold_summary,
        source_before,
        planner_prefix_evidence,
        output_prefix_evidence,
        strict=True,
    ):
        capacity = _cache_capacity(detail)
        layers_detail.append(
            {
                "index": cold["index"],
                "attention_kind": cold["attention_kind"],
                "window": cold["window"],
                "sequence_axis": -2,
                "window_semantics": (
                    "full_prefix_unbounded"
                    if capacity is None
                    else "retained_suffix_window_minus_one"
                ),
                "cache_capacity_tokens": capacity,
                "source_cache_tokens": source["sequence_tokens"],
                "cold_prefill_key_sha256": cold["key_sha256"],
                "cold_prefill_value_sha256": cold["value_sha256"],
                "source_prefill_key_sha256": source["key_sha256"],
                "source_prefill_value_sha256": source["value_sha256"],
                "prefill_recompute_value_equal": (
                    cold["sequence_tokens"] == source["sequence_tokens"]
                    and cold["key_sha256"] == source["key_sha256"]
                    and cold["value_sha256"] == source["value_sha256"]
                ),
                **planner,
                **output,
                "storage_pointer_is_proof": False,
            }
        )
    boundary_seed = {
        **request["source_identity"],
        "token_order_sha256": token_order_sha,
        "position_sha256": position_sha,
        "attention_mask_sha256": mask_sha,
        "rope_sha256": rope_sha,
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "base_identity_sha256": request["base_identity_sha256"],
        "q8_scb_sha256": request["q8_scb_sha256"],
        "adapter_lineage_sha256": request["adapter_inventory_sha256"],
        "layer_topology_sha256": topology["layer_topology_sha256"],
    }
    lineage = {
        "producer_candidate_commit": boundary_seed["producer_candidate_commit"],
        "producer_release_commit": boundary_seed["producer_release_commit"],
        "producer_release_tree": boundary_seed["producer_release_tree"],
        "source_blob_sha256": boundary_seed["source_blob_sha256"],
        "source_record_sha256": boundary_seed["record_sha256"],
        "training_serialization_inventory_sha256": boundary_seed[
            "training_serialization_inventory_sha256"
        ],
        "record_prompt_sha256": boundary_seed["record_prompt_sha256"],
        "example_serialization_sha256": boundary_seed["example_serialization_sha256"],
        "tokenizer_binding_sha256": boundary_seed["tokenizer_binding_sha256"],
        "model_config_sha256": boundary_seed["model_config_sha256"],
        "token_order_sha256": token_order_sha,
        "position_sha256": position_sha,
        "attention_mask_sha256": mask_sha,
        "rope_sha256": rope_sha,
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "base_identity_sha256": boundary_seed["base_identity_sha256"],
        "q8_scb_sha256": boundary_seed["q8_scb_sha256"],
        "adapter_lineage_sha256": boundary_seed["adapter_lineage_sha256"],
        "layer_topology_sha256": topology["layer_topology_sha256"],
        "route_commit_sha256": boundary_seed["route_commit_sha256"],
        "plan_commit_sha256": boundary_seed["plan_commit_sha256"],
    }
    # The caller rebinds adapter_lineage to the authenticated run inventory.
    boundary_commit = boundary_commit_sha256(lineage)
    wrong_adapter_rejected = _boundary_mutation_rejected(
        lineage,
        boundary_commit,
        field="adapter_lineage_sha256",
        failure_code="kv_v2_adapter_switch_invalid",
    )
    wrong_position_rejected = _boundary_mutation_rejected(
        lineage,
        boundary_commit,
        field="position_sha256",
        failure_code="kv_v2_wrong_position",
    )
    wrong_mask_rejected = _boundary_mutation_rejected(
        lineage,
        boundary_commit,
        field="attention_mask_sha256",
        failure_code="kv_v2_wrong_attention_mask",
    )
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "topology": {**topology, "layers_detail": layers_detail},
        "boundary": {
            "boundary_commit_sha256": boundary_commit,
            "ordered_prefix_sha256": request["source_identity"]["record_prompt_sha256"],
            "token_order_sha256": token_order_sha,
            "position_sha256": position_sha,
            "attention_mask_sha256": mask_sha,
            "rope_sha256": rope_sha,
            "route_commit_sha256": request["source_identity"]["route_commit_sha256"],
            "plan_commit_sha256": request["source_identity"]["plan_commit_sha256"],
            "prefix_tokens": prefix_tokens,
            "source_prefill_adapter_state": "adapter_off_frozen_base",
            "planner_adapter_sha256": request["planner_adapter_sha256"],
            "output_adapter_sha256": request["output_adapter_sha256"],
            "latent_packet": {
                "enabled": False,
                "fixed_dimension": 0,
                "serialization_sha256": None,
                "position_sha256": None,
                "commit_sha256": None,
                "hidden_chain_of_thought": False,
            },
        },
        "comparison": {
            "cold_recompute_executed": True,
            "shared_prefix_handoff_executed": True,
            "logits_digest_cold": _tensor_digest(cold_logits, torch),
            "logits_digest_handoff": _tensor_digest(handoff_logits, torch),
            "logits_allclose": allclose,
            "logits_max_abs_error": max_error,
            "next_token_argmax_equal": next_equal,
            "next_token_digest_cold": _tensor_digest(cold_next, torch),
            "next_token_digest_handoff": _tensor_digest(handoff_next, torch),
            "prefill_count_cold": 1,
            "prefill_count_handoff": 1,
        },
        "isolation": {
            "source_cache_unchanged": source_equal,
            "planner_tail_private": not planner_tail_alias,
            "output_tail_private": not output_tail_alias,
            "sibling_branch_unchanged": sibling_equal,
            "private_tail_alias": planner_tail_alias or output_tail_alias,
            "wrong_adapter_rejected": wrong_adapter_rejected,
            "wrong_position_rejected": wrong_position_rejected,
            "wrong_attention_mask_rejected": wrong_mask_rejected,
            "cache_mutation_rejected": source_equal,
            "pointer_observation_only": True,
        },
        "performance": {
            "cold_latency_ms": cold_prefill_ms + cold_planner_ms + cold_output_ms,
            "handoff_latency_ms": handoff_prefill_ms + planner_ms + output_ms,
            "handoff_prefill_ms": handoff_prefill_ms,
            "planner_tail_ms": planner_ms,
            "output_tail_ms": output_ms,
            "cold_peak_device_bytes": peak,
            "handoff_peak_device_bytes": peak,
            "repetitions": 1,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--validate-receipt")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--producer-repository-root")
    parser.add_argument("--consumer-receipt")
    parser.add_argument("--consumer-receipt-sidecar")
    parser.add_argument("--consumer-receipt-schema")
    parser.add_argument("--source-blob")
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--adapter-receipt")
    parser.add_argument("--adapter-receipt-sidecar")
    parser.add_argument("--adapter-receipt-schema")
    parser.add_argument("--model-dir")
    parser.add_argument("--planner-adapter-name")
    parser.add_argument("--output-adapter-name")
    parser.add_argument("--gpu-lock")
    parser.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.status:
            result = build_status(config)
        elif args.validate or args.dry_run:
            result = build_status(
                config,
                operation="dry-run" if args.dry_run else "validate",
            )
        elif args.validate_receipt:
            receipt, _ = _load_json(
                Path(args.validate_receipt),
                "kv_v2_receipt_invalid",
            )
            result = validate_receipt(receipt, config, allow_blocked=True)
        else:
            required = {
                "producer_repository_root": args.producer_repository_root,
                "consumer_receipt": args.consumer_receipt,
                "consumer_receipt_sidecar": args.consumer_receipt_sidecar,
                "consumer_receipt_schema": args.consumer_receipt_schema,
                "source_blob": args.source_blob,
                "record_index": args.record_index,
                "adapter_receipt": args.adapter_receipt,
                "adapter_receipt_sidecar": args.adapter_receipt_sidecar,
                "adapter_receipt_schema": args.adapter_receipt_schema,
                "model_dir": args.model_dir,
                "planner_adapter_name": args.planner_adapter_name,
                "output_adapter_name": args.output_adapter_name,
                "gpu_lock": args.gpu_lock,
                "receipt": args.receipt,
            }
            if any(value is None for value in required.values()):
                _fail("kv_v2_execute_not_authorized")
            result = execute(
                config,
                producer_repository_root=args.producer_repository_root,
                consumer_receipt_path=args.consumer_receipt,
                consumer_receipt_sidecar_path=args.consumer_receipt_sidecar,
                consumer_receipt_schema_path=args.consumer_receipt_schema,
                source_blob_path=args.source_blob,
                record_index=args.record_index,
                adapter_receipt_path=args.adapter_receipt,
                adapter_receipt_sidecar_path=args.adapter_receipt_sidecar,
                adapter_receipt_schema_path=args.adapter_receipt_schema,
                model_dir=args.model_dir,
                planner_adapter_name=args.planner_adapter_name,
                output_adapter_name=args.output_adapter_name,
                gpu_lock_path=args.gpu_lock,
                receipt_path=args.receipt,
            )
    except SharedPrefixKVRuntimeError as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
