"""Model-free contract for the unbalanced-v2 shared-prefix KV probe.

The checked-in contract deliberately stops at exact retained prefix values and
prefill-compute handoff.  It does not claim shared storage, zero-copy,
full-generation sharing, or RDMA.  This module never imports an ML runtime.
A future physical backend may be injected lazily through :func:`execute`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1.json"
)
CONFIG_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1_config.schema.json"
)
RECEIPT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1_receipt.schema.json"
)

CONFIG_VERSION = "anchor.gemma3-chat-unbalanced-v2-shared-prefix-kv-probe-config.v1"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-shared-prefix-kv-probe-receipt.v1"
PROFILE_ID = "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1"
CONSUMER_RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1"
)
TRAINING_RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-multiarm-run-receipt.v1"
TRUSTED_CONTEXT_VERSION = "anchor.gemma3-chat-unbalanced-v2-trusted-lineage-context.v1"
TRUSTED_CONTEXT_STATUS = "passed_authenticated_external_lineage"
LAYERS = 26
SLIDING_WINDOW = 512
FULL_LAYER_INDICES = (5, 11, 17, 23)
LINEAGE_FIELDS = (
    "model_identity_sha256",
    "tokenizer_identity_sha256",
    "serialization_identity_sha256",
    "template_identity_sha256",
    "ordered_prefix_sha256",
    "token_order_sha256",
    "position_sha256",
    "attention_mask_sha256",
    "rope_sha256",
    "attention_implementation_sha256",
    "dtype_policy_sha256",
    "quantization_policy_sha256",
    "adapter_off_policy_sha256",
    "adapter_lineage_sha256",
    "route_commit_sha256",
    "plan_commit_sha256",
    "layer_layout_sha256",
)
LINEAGE_CONTROL_FIELDS = (
    "boundary_name",
    "boundary_commit_sha256",
    "adapter_state",
    "source_cache_present",
    "source_cache_complete",
    "prefix_length",
)
REQUIRED_MUTATIONS = (
    "model_identity_drift",
    "tokenizer_identity_drift",
    "serialization_identity_drift",
    "template_identity_drift",
    "ordered_prefix_drift",
    "token_order_drift",
    "position_drift",
    "attention_mask_drift",
    "rope_drift",
    "attention_implementation_drift",
    "dtype_policy_drift",
    "quantization_policy_drift",
    "adapter_off_policy_drift",
    "adapter_lineage_drift",
    "route_commit_drift",
    "plan_commit_drift",
    "layer_layout_drift",
    "adapter_not_off",
    "boundary_commit_drift",
    "source_cache_miss",
    "retained_span_drift",
    "source_cache_mutation",
    "private_tail_alias",
    "pointer_only_claim_upgrade",
)
REGISTERED_FAILURE_CODES = frozenset(
    {
        "kv_adapter_state_invalid",
        "kv_boundary_commit_mismatch",
        "kv_cache_miss",
        "kv_claim_inflation",
        "kv_config_invalid",
        "kv_consumer_preflight_pending",
        "kv_first_token_mismatch",
        "kv_layer_layout_mismatch",
        "kv_layer_value_mismatch",
        "kv_lineage_mismatch",
        "kv_physical_backend_required",
        "kv_private_tail_alias",
        "kv_receipt_invalid",
        "kv_retained_span_mismatch",
        "kv_source_cache_mutated",
        "kv_storage_claim_forbidden",
        "kv_training_run_pending",
        "kv_trusted_lineage_context_pending",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_DRAFT202012_ROOT_KEYWORDS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$ref",
        "$schema",
        "$vocabulary",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "default",
        "dependentRequired",
        "dependentSchemas",
        "deprecated",
        "description",
        "else",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "if",
        "items",
        "maxContains",
        "maximum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "type",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)


class SharedPrefixKVError(RuntimeError):
    """Fail-closed error carrying only a registered, body-free code."""


class _DuplicateKey(ValueError):
    pass


class _InvalidJSONConstant(ValueError):
    pass


def _fail(code: str) -> None:
    if code not in REGISTERED_FAILURE_CODES:
        raise SharedPrefixKVError("kv_receipt_invalid")
    raise SharedPrefixKVError(code)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def implementation_sha256() -> str:
    return _sha256(Path(__file__).read_bytes())


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _InvalidJSONConstant(value)


def _load_json(
    path: Path, *, max_bytes: int = _MAX_JSON_BYTES
) -> tuple[dict[str, Any], str]:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("kv_config_invalid")
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            _fail("kv_config_invalid")
        raw = path.read_bytes()
        if len(raw) != size or b"\r" in raw or not raw.endswith(b"\n"):
            _fail("kv_config_invalid")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        _InvalidJSONConstant,
    ):
        _fail("kv_config_invalid")
    if not isinstance(value, dict):
        _fail("kv_config_invalid")
    return value, _sha256(raw)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _load_authenticated_snapshot(
    path_value: str | os.PathLike[str],
    *,
    expected_sha256: str,
    failure_code: str,
) -> dict[str, Any]:
    """Read one authenticated physical snapshot and reject path replacement."""

    if (
        not isinstance(path_value, (str, os.PathLike))
        or not _is_sha256(expected_sha256)
        or expected_sha256 == "pending"
    ):
        _fail(failure_code)
    path = Path(path_value)
    try:
        before = os.lstat(path)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = int(getattr(before, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or (reparse_flag and attributes & reparse_flag)
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
        ):
            _fail(failure_code)
        baseline = _stat_signature(before)
        with path.open("rb", buffering=0) as handle:
            if _stat_signature(os.fstat(handle.fileno())) != baseline:
                _fail(failure_code)
            raw = handle.read(before.st_size + 1)
            descriptor_after = os.fstat(handle.fileno())
        if (
            _stat_signature(descriptor_after) != baseline
            or len(raw) != before.st_size
            or not raw.endswith(b"\n")
            or b"\r" in raw
            or _sha256(raw) != expected_sha256
        ):
            _fail(failure_code)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            _fail(failure_code)
        if _stat_signature(os.lstat(path)) != baseline:
            _fail(failure_code)
        return value
    except SharedPrefixKVError:
        raise
    except (
        OSError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        _InvalidJSONConstant,
    ):
        _fail(failure_code)


def _closed_mapping(
    value: object, expected_keys: Sequence[str], failure_code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        _fail(failure_code)
    return value


def _authenticate_dependency(
    *,
    receipt_path: str | os.PathLike[str],
    receipt_sha256: str,
    schema_path: str | os.PathLike[str],
    schema_sha256: str,
    expected_schema_version: str,
    failure_code: str,
) -> dict[str, Any]:
    receipt = _load_authenticated_snapshot(
        receipt_path,
        expected_sha256=receipt_sha256,
        failure_code=failure_code,
    )
    schema = _load_authenticated_snapshot(
        schema_path,
        expected_sha256=schema_sha256,
        failure_code=failure_code,
    )
    properties = schema.get("properties")
    if (
        not set(schema).issubset(_DRAFT202012_ROOT_KEYWORDS)
        or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not isinstance(properties, Mapping)
        or not isinstance(properties.get("schema_version"), Mapping)
        or properties["schema_version"].get("const") != expected_schema_version
    ):
        _fail(failure_code)
    try:
        Draft202012Validator.check_schema(schema)
        if tuple(Draft202012Validator(schema).iter_errors(receipt)):
            _fail(failure_code)
    except SharedPrefixKVError:
        raise
    except Exception:
        _fail(failure_code)
    return receipt


def _schema(path: Path, expected_sha256: str) -> Draft202012Validator:
    value, observed = _load_json(path)
    if observed != expected_sha256:
        _fail("kv_config_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except Exception:
        _fail("kv_config_invalid")
    return Draft202012Validator(value)


def layer_layout() -> tuple[dict[str, Any], ...]:
    """Return the frozen Gemma 3 26-layer local/full attention layout."""

    full = frozenset(FULL_LAYER_INDICES)
    return tuple(
        {
            "layer_index": index,
            "attention_kind": "full" if index in full else "sliding",
            "window": None if index in full else SLIDING_WINDOW,
        }
        for index in range(LAYERS)
    )


def layer_layout_sha256() -> str:
    return _sha256(_canonical_json(list(layer_layout())))


def retained_span(layer_index: int, prefix_length: int) -> tuple[int, int]:
    """Compute the exact retained prefix interval ``[start, end)``.

    Full-attention layers retain the entire ordered prefix. Sliding layers
    retain at most the final 512 prefix positions.
    """

    if isinstance(layer_index, bool) or not 0 <= layer_index < LAYERS:
        _fail("kv_layer_layout_mismatch")
    if isinstance(prefix_length, bool) or prefix_length < 1:
        _fail("kv_retained_span_mismatch")
    if layer_index in FULL_LAYER_INDICES:
        return 0, prefix_length
    return max(0, prefix_length - SLIDING_WINDOW), prefix_length


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_sha256(config: Mapping[str, Any]) -> str:
    observed = config.get("_config_sha256")
    if isinstance(observed, str) and _is_sha256(observed):
        return observed
    return _sha256(_canonical_json(_public_config(config)))


def _bound_schema_validators(
    config: Mapping[str, Any],
) -> tuple[Draft202012Validator, Draft202012Validator]:
    schemas = config.get("schemas")
    if not isinstance(schemas, Mapping):
        _fail("kv_config_invalid")
    expected = {
        "config": (
            CONFIG_SCHEMA_PATH,
            "configs/research/"
            "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1_config.schema.json",
            CONFIG_VERSION,
        ),
        "receipt": (
            RECEIPT_SCHEMA_PATH,
            "configs/research/"
            "gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1_receipt.schema.json",
            RECEIPT_VERSION,
        ),
    }
    validators: list[Draft202012Validator] = []
    for kind in ("config", "receipt"):
        binding = schemas.get(kind)
        if not isinstance(binding, Mapping):
            _fail("kv_config_invalid")
        path, relative, version = expected[kind]
        if (
            set(binding) != {"path", "version", "sha256"}
            or binding.get("path") != relative
            or binding.get("version") != version
            or not _is_sha256(binding.get("sha256"))
        ):
            _fail("kv_config_invalid")
        validators.append(_schema(path, str(binding["sha256"])))
    return validators[0], validators[1]


def validate_config(config: Mapping[str, Any]) -> None:
    config_validator, _ = _bound_schema_validators(config)
    errors = tuple(config_validator.iter_errors(_public_config(config)))
    if errors:
        _fail("kv_config_invalid")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("profile_id") != PROFILE_ID
    ):
        _fail("kv_config_invalid")
    model = config["model"]
    if (
        model["layer_layout_sha256"] != layer_layout_sha256()
        or tuple(model["full_layer_indices"]) != FULL_LAYER_INDICES
    ):
        _fail("kv_layer_layout_mismatch")
    handoff = config["handoff"]
    if tuple(handoff["lineage_fields"]) != LINEAGE_FIELDS:
        _fail("kv_lineage_mismatch")
    if tuple(config["synthetic_cases"]["required_mutations"]) != REQUIRED_MUTATIONS:
        _fail("kv_config_invalid")
    if config["training_run"]["required_schema_version"] != TRAINING_RECEIPT_VERSION:
        _fail("kv_config_invalid")
    trusted_context = config["trusted_lineage_context"]
    if (
        trusted_context["required_schema_version"] != TRUSTED_CONTEXT_VERSION
        or trusted_context["required_status"] != TRUSTED_CONTEXT_STATUS
    ):
        _fail("kv_config_invalid")
    claims = config["claims"]
    forbidden = (
        "shared_storage",
        "zero_copy",
        "full_generation_kv_shared",
        "rdma",
        "same_object_is_proof",
        "same_pointer_is_proof",
        "physical_execution_completed",
        "formal",
    )
    if any(claims[name] is not False for name in forbidden):
        _fail("kv_claim_inflation")
    if set(config["failure_codes"]) != REGISTERED_FAILURE_CODES:
        _fail("kv_config_invalid")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config, digest = _load_json(Path(path))
    config["_config_sha256"] = digest
    validate_config(config)
    return config


def _lineage_payload(lineage: Mapping[str, Any]) -> dict[str, Any]:
    expected = set(LINEAGE_FIELDS) | set(LINEAGE_CONTROL_FIELDS)
    if set(lineage) != expected:
        _fail("kv_lineage_mismatch")
    for field in LINEAGE_FIELDS:
        if not _is_sha256(lineage[field]):
            _fail("kv_lineage_mismatch")
    if lineage["boundary_name"] != "route_plan_commit":
        _fail("kv_lineage_mismatch")
    if lineage["adapter_state"] != "adapter_off_frozen_base":
        _fail("kv_adapter_state_invalid")
    if lineage["source_cache_present"] is not True:
        _fail("kv_cache_miss")
    if lineage["source_cache_complete"] is not True:
        _fail("kv_cache_miss")
    prefix_length = lineage["prefix_length"]
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
        _fail("kv_lineage_mismatch")
    if prefix_length < 1:
        _fail("kv_lineage_mismatch")
    return {
        **{field: lineage[field] for field in LINEAGE_FIELDS},
        "boundary_name": lineage["boundary_name"],
        "adapter_state": lineage["adapter_state"],
        "prefix_length": prefix_length,
    }


def boundary_commit_sha256(lineage: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(_lineage_payload(lineage)))


def validate_handoff_lineage(
    source: Mapping[str, Any], handoff: Mapping[str, Any]
) -> str:
    """Validate complete source/handoff lineage without inspecting token bodies."""

    source_payload = _lineage_payload(source)
    handoff_payload = _lineage_payload(handoff)
    source_commit = source.get("boundary_commit_sha256")
    handoff_commit = handoff.get("boundary_commit_sha256")
    if not _is_sha256(source_commit) or not _is_sha256(handoff_commit):
        _fail("kv_boundary_commit_mismatch")
    expected_source = _sha256(_canonical_json(source_payload))
    expected_handoff = _sha256(_canonical_json(handoff_payload))
    if source_commit != expected_source or handoff_commit != expected_handoff:
        _fail("kv_boundary_commit_mismatch")
    if source_payload != handoff_payload or source_commit != handoff_commit:
        _fail("kv_lineage_mismatch")
    return str(source_commit)


def _pending_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    groups = {
        "consumer_preflight": (
            "receipt_sha256",
            "receipt_schema_sha256",
            "artifact_tree_digest_sha256",
            "shard_inventory_sha256",
        ),
        "training_run": (
            "receipt_sha256",
            "receipt_schema_sha256",
            "adapter_inventory_sha256",
            "run_id",
        ),
        "trusted_lineage_context": (
            "context_sha256",
            "context_schema_sha256",
        ),
        "model": (
            "model_identity_sha256",
            "tokenizer_identity_sha256",
            "serialization_identity_sha256",
            "template_identity_sha256",
            "attention_implementation_sha256",
            "dtype_policy_sha256",
            "quantization_policy_sha256",
            "adapter_off_policy_sha256",
        ),
    }
    return tuple(
        f"{group_name}.{name}"
        for group_name, names in groups.items()
        for name in names
        if config[group_name][name] == "pending"
    )


def _zero_audit() -> dict[str, int]:
    return {
        "dataset_body_reads": 0,
        "raw_token_ids_read": 0,
        "gold_reads": 0,
        "heldout_body_reads": 0,
        "protected_reads": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "provider_requests": 0,
        "network_requests": 0,
    }


def build_status(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    pending = _pending_fields(config)
    return {
        "schema_version": CONFIG_VERSION,
        "profile_id": PROFILE_ID,
        "status": (
            "blocked_waiting_for_authenticated_identities"
            if pending
            else "model_free_contract_ready_physical_backend_pending"
        ),
        "pending": list(pending),
        "config_sha256": config_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "layer_layout_sha256": layer_layout_sha256(),
        "model_loaded": False,
        "gpu_touched": False,
        "physical_execution_completed": False,
        "audit": _zero_audit(),
    }


def _validate_consumer_receipt_metadata(
    config: Mapping[str, Any], consumer_receipt: Mapping[str, Any]
) -> None:
    dependency = config["consumer_preflight"]
    _closed_mapping(
        consumer_receipt,
        ("schema_version", "status", "identity"),
        "kv_consumer_preflight_pending",
    )
    if (
        consumer_receipt.get("schema_version") != CONSUMER_RECEIPT_VERSION
        or consumer_receipt.get("status") != "passed"
    ):
        _fail("kv_consumer_preflight_pending")
    identity = _closed_mapping(
        consumer_receipt.get("identity"),
        (
            "artifact_tree_digest_sha256",
            "shard_inventory_sha256",
            "model_identity_sha256",
            "tokenizer_identity_sha256",
            "serialization_identity_sha256",
        ),
        "kv_consumer_preflight_pending",
    )
    expected = {
        "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
        "shard_inventory_sha256": dependency["shard_inventory_sha256"],
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "tokenizer_identity_sha256": config["model"]["tokenizer_identity_sha256"],
        "serialization_identity_sha256": config["model"][
            "serialization_identity_sha256"
        ],
    }
    if any(
        expected[name] == "pending"
        or not _is_sha256(identity.get(name))
        or identity.get(name) != expected[name]
        for name in expected
    ):
        _fail("kv_consumer_preflight_pending")


def _validate_training_receipt_metadata(
    config: Mapping[str, Any], training_receipt: Mapping[str, Any]
) -> None:
    training = config["training_run"]
    _closed_mapping(
        training_receipt,
        ("schema_version", "status", "run_id", "identity", "claims"),
        "kv_training_run_pending",
    )
    if (
        training_receipt.get("schema_version") != TRAINING_RECEIPT_VERSION
        or training_receipt.get("status") != training["required_status"]
        or training_receipt.get("run_id") != training["run_id"]
    ):
        _fail("kv_training_run_pending")
    identity = _closed_mapping(
        training_receipt.get("identity"),
        (
            "consumer_preflight_receipt_sha256",
            "artifact_tree_digest_sha256",
            "shard_inventory_sha256",
            "adapter_inventory_sha256",
        ),
        "kv_training_run_pending",
    )
    expected = {
        "consumer_preflight_receipt_sha256": config["consumer_preflight"][
            "receipt_sha256"
        ],
        "artifact_tree_digest_sha256": config["consumer_preflight"][
            "artifact_tree_digest_sha256"
        ],
        "shard_inventory_sha256": config["consumer_preflight"][
            "shard_inventory_sha256"
        ],
        "adapter_inventory_sha256": training["adapter_inventory_sha256"],
    }
    if any(
        expected[name] == "pending"
        or not _is_sha256(identity.get(name))
        or identity.get(name) != expected[name]
        for name in expected
    ):
        _fail("kv_training_run_pending")
    claims = _closed_mapping(
        training_receipt.get("claims"),
        ("fresh_base", "fresh_adapter", "resume"),
        "kv_training_run_pending",
    )
    if claims != {"fresh_base": True, "fresh_adapter": True, "resume": False}:
        _fail("kv_training_run_pending")


def _validate_trusted_lineage_context(
    config: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any]:
    _closed_mapping(
        context,
        ("schema_version", "status", "identity", "lineage", "claims"),
        "kv_trusted_lineage_context_pending",
    )
    requirement = config["trusted_lineage_context"]
    if (
        context.get("schema_version") != requirement["required_schema_version"]
        or context.get("status") != requirement["required_status"]
    ):
        _fail("kv_trusted_lineage_context_pending")
    identity = _closed_mapping(
        context.get("identity"),
        (
            "consumer_preflight_receipt_sha256",
            "artifact_tree_digest_sha256",
            "shard_inventory_sha256",
            "training_run_receipt_sha256",
            "adapter_inventory_sha256",
            "run_id",
        ),
        "kv_trusted_lineage_context_pending",
    )
    expected_identity = {
        "consumer_preflight_receipt_sha256": config["consumer_preflight"][
            "receipt_sha256"
        ],
        "artifact_tree_digest_sha256": config["consumer_preflight"][
            "artifact_tree_digest_sha256"
        ],
        "shard_inventory_sha256": config["consumer_preflight"][
            "shard_inventory_sha256"
        ],
        "training_run_receipt_sha256": config["training_run"]["receipt_sha256"],
        "adapter_inventory_sha256": config["training_run"]["adapter_inventory_sha256"],
        "run_id": config["training_run"]["run_id"],
    }
    if any(
        identity.get(name) != expected for name, expected in expected_identity.items()
    ):
        _fail("kv_trusted_lineage_context_pending")
    for name, value in identity.items():
        if name != "run_id" and not _is_sha256(value):
            _fail("kv_trusted_lineage_context_pending")
    claims = _closed_mapping(
        context.get("claims"),
        ("external_to_experiments", "body_free", "adapter_off"),
        "kv_trusted_lineage_context_pending",
    )
    if claims != {
        "external_to_experiments": True,
        "body_free": True,
        "adapter_off": True,
    }:
        _fail("kv_trusted_lineage_context_pending")
    lineage = context.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("kv_trusted_lineage_context_pending")
    payload = _lineage_payload(lineage)
    commit = lineage.get("boundary_commit_sha256")
    if not _is_sha256(commit) or commit != _sha256(_canonical_json(payload)):
        _fail("kv_trusted_lineage_context_pending")
    expected_lineage = {
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "tokenizer_identity_sha256": config["model"]["tokenizer_identity_sha256"],
        "serialization_identity_sha256": config["model"][
            "serialization_identity_sha256"
        ],
        "template_identity_sha256": config["model"]["template_identity_sha256"],
        "attention_implementation_sha256": config["model"][
            "attention_implementation_sha256"
        ],
        "dtype_policy_sha256": config["model"]["dtype_policy_sha256"],
        "quantization_policy_sha256": config["model"]["quantization_policy_sha256"],
        "adapter_off_policy_sha256": config["model"]["adapter_off_policy_sha256"],
        "adapter_lineage_sha256": config["training_run"]["adapter_inventory_sha256"],
        "layer_layout_sha256": config["model"]["layer_layout_sha256"],
    }
    if any(
        lineage.get(name) != expected for name, expected in expected_lineage.items()
    ):
        _fail("kv_trusted_lineage_context_pending")
    return lineage


def build_preflight(
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_schema_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    status = build_status(config)
    pending = tuple(status["pending"])
    if pending:
        return {
            **status,
            "status": "blocked_waiting_for_authenticated_identities",
            "ready": False,
            "gates": {
                "config_valid": True,
                "consumer_identity_bound": not pending,
                "consumer_receipt_physical_authenticated": False,
                "training_receipt_physical_authenticated": False,
                "trusted_lineage_context_physical_authenticated": False,
                "physical_backend_registered": False,
            },
        }
    if consumer_receipt_path is None or consumer_receipt_schema_path is None:
        _fail("kv_consumer_preflight_pending")
    if training_receipt_path is None or training_receipt_schema_path is None:
        _fail("kv_training_run_pending")
    if (
        trusted_lineage_context_path is None
        or trusted_lineage_context_schema_path is None
    ):
        _fail("kv_trusted_lineage_context_pending")
    assert consumer_receipt_path is not None
    assert consumer_receipt_schema_path is not None
    assert training_receipt_path is not None
    assert training_receipt_schema_path is not None
    assert trusted_lineage_context_path is not None
    assert trusted_lineage_context_schema_path is not None
    consumer_receipt = _authenticate_dependency(
        receipt_path=consumer_receipt_path,
        receipt_sha256=config["consumer_preflight"]["receipt_sha256"],
        schema_path=consumer_receipt_schema_path,
        schema_sha256=config["consumer_preflight"]["receipt_schema_sha256"],
        expected_schema_version=config["consumer_preflight"]["required_schema_version"],
        failure_code="kv_consumer_preflight_pending",
    )
    training_receipt = _authenticate_dependency(
        receipt_path=training_receipt_path,
        receipt_sha256=config["training_run"]["receipt_sha256"],
        schema_path=training_receipt_schema_path,
        schema_sha256=config["training_run"]["receipt_schema_sha256"],
        expected_schema_version=config["training_run"]["required_schema_version"],
        failure_code="kv_training_run_pending",
    )
    trusted_context = _authenticate_dependency(
        receipt_path=trusted_lineage_context_path,
        receipt_sha256=config["trusted_lineage_context"]["context_sha256"],
        schema_path=trusted_lineage_context_schema_path,
        schema_sha256=config["trusted_lineage_context"]["context_schema_sha256"],
        expected_schema_version=config["trusted_lineage_context"][
            "required_schema_version"
        ],
        failure_code="kv_trusted_lineage_context_pending",
    )
    _validate_consumer_receipt_metadata(config, consumer_receipt)
    _validate_training_receipt_metadata(config, training_receipt)
    _validate_trusted_lineage_context(config, trusted_context)
    return {
        **status,
        "status": "ready_for_lazy_physical_backend",
        "ready": True,
        "gates": {
            "config_valid": True,
            "consumer_identity_bound": True,
            "consumer_receipt_physical_authenticated": True,
            "training_receipt_physical_authenticated": True,
            "trusted_lineage_context_physical_authenticated": True,
            "physical_backend_registered": False,
        },
    }


def _validate_layer_receipts(
    layers: Sequence[Mapping[str, Any]], prefix_length: int, *, passed: bool
) -> None:
    expected_layout = layer_layout()
    if len(layers) != LAYERS:
        _fail("kv_layer_layout_mismatch")
    for index, layer in enumerate(layers):
        expected = expected_layout[index]
        if (
            layer["layer_index"] != index
            or layer["attention_kind"] != expected["attention_kind"]
            or layer["window"] != expected["window"]
        ):
            _fail("kv_layer_layout_mismatch")
        start, end = retained_span(index, prefix_length)
        if (
            layer["retained_start"] != start
            or layer["retained_end"] != end
            or layer["retained_length"] != end - start
        ):
            _fail("kv_retained_span_mismatch")
        digests_equal = layer["isolated_value_sha256"] == layer["handoff_value_sha256"]
        if layer["value_equal"] is not digests_equal:
            _fail("kv_layer_value_mismatch")
        if layer["storage_shared_claimed"] is not False:
            _fail("kv_storage_claim_forbidden")
        if passed and not digests_equal:
            _fail("kv_layer_value_mismatch")


def validate_receipt(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_schema_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    preflight = build_preflight(
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        trusted_lineage_context_path=trusted_lineage_context_path,
        trusted_lineage_context_schema_path=trusted_lineage_context_schema_path,
    )
    if not preflight["ready"]:
        _fail("kv_consumer_preflight_pending")
    assert trusted_lineage_context_path is not None
    assert trusted_lineage_context_schema_path is not None
    trusted_context = _authenticate_dependency(
        receipt_path=trusted_lineage_context_path,
        receipt_sha256=config["trusted_lineage_context"]["context_sha256"],
        schema_path=trusted_lineage_context_schema_path,
        schema_sha256=config["trusted_lineage_context"]["context_schema_sha256"],
        expected_schema_version=config["trusted_lineage_context"][
            "required_schema_version"
        ],
        failure_code="kv_trusted_lineage_context_pending",
    )
    trusted_lineage = _validate_trusted_lineage_context(config, trusted_context)
    _, receipt_validator = _bound_schema_validators(config)
    if tuple(receipt_validator.iter_errors(receipt)):
        _fail("kv_receipt_invalid")
    identity = receipt["identity"]
    dependency = config["consumer_preflight"]
    training = config["training_run"]
    if _pending_fields(config):
        _fail("kv_consumer_preflight_pending")
    expected_identity = {
        "config_sha256": config_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "consumer_preflight_receipt_sha256": dependency["receipt_sha256"],
        "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
        "shard_inventory_sha256": dependency["shard_inventory_sha256"],
        "training_run_receipt_sha256": training["receipt_sha256"],
        "adapter_inventory_sha256": training["adapter_inventory_sha256"],
        "run_id": training["run_id"],
        "trusted_lineage_context_sha256": config["trusted_lineage_context"][
            "context_sha256"
        ],
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "tokenizer_identity_sha256": config["model"]["tokenizer_identity_sha256"],
        "serialization_identity_sha256": config["model"][
            "serialization_identity_sha256"
        ],
        "template_identity_sha256": config["model"]["template_identity_sha256"],
        "attention_implementation_sha256": config["model"][
            "attention_implementation_sha256"
        ],
        "dtype_policy_sha256": config["model"]["dtype_policy_sha256"],
        "quantization_policy_sha256": config["model"]["quantization_policy_sha256"],
        "adapter_off_policy_sha256": config["model"]["adapter_off_policy_sha256"],
        "layer_layout_sha256": layer_layout_sha256(),
    }
    if any(identity.get(name) != value for name, value in expected_identity.items()):
        _fail("kv_receipt_invalid")
    passed = receipt["status"] == "passed_diagnostic_prefix_value_handoff"
    boundary = receipt["boundary"]
    if (
        boundary["source_cache_present"] is not True
        or boundary["source_cache_complete"] is not True
    ):
        _fail("kv_cache_miss")
    experiments = receipt["experiments"]
    isolated = experiments["isolated_reprefill"]
    handoff = experiments["one_prefill_explicit_local_copy"]
    isolated_lineage = isolated["lineage"]
    handoff_lineage = handoff["lineage"]
    validated_commit = validate_handoff_lineage(
        isolated_lineage,
        handoff_lineage,
    )
    if isolated_lineage != trusted_lineage or handoff_lineage != trusted_lineage:
        _fail("kv_lineage_mismatch")
    if (
        boundary["commit_sha256"] != validated_commit
        or boundary["route_commit_sha256"] != isolated_lineage["route_commit_sha256"]
        or boundary["plan_commit_sha256"] != isolated_lineage["plan_commit_sha256"]
        or boundary["ordered_prefix_sha256"]
        != isolated_lineage["ordered_prefix_sha256"]
        or boundary["prefix_length"] != isolated_lineage["prefix_length"]
        or isolated["source_prefix_sha256"] != isolated_lineage["ordered_prefix_sha256"]
        or handoff["source_prefix_sha256"] != handoff_lineage["ordered_prefix_sha256"]
    ):
        _fail("kv_lineage_mismatch")
    _validate_layer_receipts(
        receipt["layers"], int(boundary["prefix_length"]), passed=passed
    )
    distribution_digests_equal = (
        isolated["first_token_distribution_digest_sha256"]
        == handoff["first_token_distribution_digest_sha256"]
    )
    argmax_digests_equal = (
        isolated["first_token_argmax_digest_sha256"]
        == handoff["first_token_argmax_digest_sha256"]
    )
    claims = receipt["claims"]
    if any(
        claims[name] is not False
        for name in ("shared_storage", "zero_copy", "full_generation_kv_shared", "rdma")
    ):
        _fail("kv_claim_inflation")
    diagnostics = receipt["diagnostics"]
    if diagnostics["private_tails_alias"]:
        _fail("kv_private_tail_alias")
    if (
        diagnostics["first_token_distribution_digest_equal"]
        is not distribution_digests_equal
        or diagnostics["first_token_argmax_equal"] is not argmax_digests_equal
    ):
        _fail("kv_first_token_mismatch")
    if passed:
        if (
            isolated["executed"] is not True
            or handoff["executed"] is not True
            or isolated["prefill_count"] != 2
            or isolated["transport"] != "none_isolated_reprefill"
            or handoff["prefill_count"] != 1
            or handoff["transport"] != "explicit_local_copy"
        ):
            _fail("kv_receipt_invalid")
        required_true = (
            "all_layer_retained_values_equal",
            "source_cache_unchanged",
            "private_tail_append_only",
            "cache_miss_rejected",
        )
        if any(diagnostics[name] is not True for name in required_true):
            if not diagnostics["source_cache_unchanged"]:
                _fail("kv_source_cache_mutated")
            _fail("kv_receipt_invalid")
        if not distribution_digests_equal or not argmax_digests_equal:
            _fail("kv_first_token_mismatch")
        if diagnostics["cross_mutation_rejections"] != {
            "attempted": len(REQUIRED_MUTATIONS),
            "rejected": len(REQUIRED_MUTATIONS),
        }:
            _fail("kv_receipt_invalid")
        if receipt["failure_codes"]:
            _fail("kv_receipt_invalid")
    elif not receipt["failure_codes"]:
        _fail("kv_receipt_invalid")
    if any(code not in REGISTERED_FAILURE_CODES for code in receipt["failure_codes"]):
        _fail("kv_receipt_invalid")
    for value in receipt["performance"].values():
        if isinstance(value, float) and not math.isfinite(value):
            _fail("kv_receipt_invalid")
    return dict(receipt)


PhysicalBackend = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def execute(
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_path: str | os.PathLike[str] | None = None,
    trusted_lineage_context_schema_path: str | os.PathLike[str] | None = None,
    backend: PhysicalBackend | None = None,
) -> dict[str, Any]:
    """Invoke a future physical backend lazily, then validate its receipt."""

    preflight = build_preflight(
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        trusted_lineage_context_path=trusted_lineage_context_path,
        trusted_lineage_context_schema_path=trusted_lineage_context_schema_path,
    )
    if not preflight["ready"]:
        _fail("kv_consumer_preflight_pending")
    if backend is None:
        _fail("kv_physical_backend_required")
    receipt = backend(_public_config(config))
    if not isinstance(receipt, Mapping):
        _fail("kv_receipt_invalid")
    return validate_receipt(
        receipt,
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        trusted_lineage_context_path=trusted_lineage_context_path,
        trusted_lineage_context_schema_path=trusted_lineage_context_schema_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--consumer-receipt")
    parser.add_argument("--consumer-receipt-schema")
    parser.add_argument("--training-receipt")
    parser.add_argument("--training-receipt-schema")
    parser.add_argument("--trusted-lineage-context")
    parser.add_argument("--trusted-lineage-context-schema")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--validate-receipt")
    group.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.status:
            result = build_status(config)
        elif args.preflight:
            result = build_preflight(
                config,
                consumer_receipt_path=args.consumer_receipt,
                consumer_receipt_schema_path=args.consumer_receipt_schema,
                training_receipt_path=args.training_receipt,
                training_receipt_schema_path=args.training_receipt_schema,
                trusted_lineage_context_path=args.trusted_lineage_context,
                trusted_lineage_context_schema_path=(
                    args.trusted_lineage_context_schema
                ),
            )
        elif args.validate_receipt:
            receipt, _ = _load_json(Path(args.validate_receipt))
            result = validate_receipt(
                receipt,
                config,
                consumer_receipt_path=args.consumer_receipt,
                consumer_receipt_schema_path=args.consumer_receipt_schema,
                training_receipt_path=args.training_receipt,
                training_receipt_schema_path=args.training_receipt_schema,
                trusted_lineage_context_path=args.trusted_lineage_context,
                trusted_lineage_context_schema_path=(
                    args.trusted_lineage_context_schema
                ),
            )
        else:
            result = execute(
                config,
                consumer_receipt_path=args.consumer_receipt,
                consumer_receipt_schema_path=args.consumer_receipt_schema,
                training_receipt_path=args.training_receipt,
                training_receipt_schema_path=args.training_receipt_schema,
                trusted_lineage_context_path=args.trusted_lineage_context,
                trusted_lineage_context_schema_path=(
                    args.trusted_lineage_context_schema
                ),
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except SharedPrefixKVError as exc:
        code = str(exc)
        if code not in REGISTERED_FAILURE_CODES:
            code = "kv_receipt_invalid"
        print(
            json.dumps(
                {"error_type": "SharedPrefixKVError", "error_code": code},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "error_type": "RuntimeError",
                    "error_code": "runtime_error_redacted",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
