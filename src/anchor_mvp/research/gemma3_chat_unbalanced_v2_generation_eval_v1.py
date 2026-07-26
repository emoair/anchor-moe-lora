"""Model-free aggregate evaluator contract for Gemma 3 chat unbalanced-v2.

The module validates the complete evaluation matrix and body-free aggregate
receipts.  It consumes only authenticated receipt metadata and logical shard
inventory identities during preflight.  It does not import an ML runtime,
inspect sample bodies, or retain raw token identifiers.  A future physical
backend may be injected lazily through :func:`execute`.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT / "configs" / "research" / "gemma3_chat_unbalanced_v2_generation_eval_v1.json"
)
CONFIG_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_generation_eval_v1_config.schema.json"
)
RECEIPT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_generation_eval_v1_receipt.schema.json"
)
CONFIG_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-config.v1"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-receipt.v1"
PROFILE_ID = "gemma3_chat_unbalanced_v2_generation_eval_v1"
CONSUMER_RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1"
)
TRAINING_RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-multiarm-run-receipt.v1"
KV_RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-shared-prefix-kv-probe-receipt.v1"
)

TOOL_ARMS = ("tool_base", "tool_q_only", "tool_q_plus_micro_o")
PLANNER_ARMS = (
    "planner_base",
    "planner_q_only",
    "planner_o_only",
    "planner_q_plus_micro_o",
)
IDENTITY_ARMS = (
    "identity_base",
    "identity_correct_route",
    "identity_wrong_route",
)
WRONG_ROUTE_DERANGEMENTS = ("derangement_1", "derangement_2", "derangement_3")
TOOL_METRICS = (
    "json_parse",
    "closed_schema",
    "tool_name_exact",
    "arguments_exact",
    "grounding_consistent",
    "execution_projection_exact",
)
PLANNER_METRICS = (
    "plan_format",
    "task_class_exact",
    "selected_expert_exact",
    "single_total_to_branch",
    "no_return_to_general",
    "commit_boundary_exact",
)
IDENTITY_METRICS = (
    "air_attribution",
    "provider_misattribution_absent",
    "identity_fact_route_invariant",
)
WRONG_ROUTE_METRICS = ("primary_metric", "wrong_route_format_leak_absent")
REGISTERED_FAILURE_CODES = frozenset(
    {
        "eval_aggregate_only_violation",
        "eval_config_invalid",
        "eval_consumer_preflight_pending",
        "eval_count_mismatch",
        "eval_dataset_inventory_mismatch",
        "eval_identity_mismatch",
        "eval_kv_probe_pending",
        "eval_physical_backend_required",
        "eval_rate_mismatch",
        "eval_receipt_invalid",
        "eval_training_run_pending",
    }
)
FORBIDDEN_AGGREGATE_KEYS = frozenset(
    {
        "body",
        "bodies",
        "prompt",
        "target",
        "reference",
        "generated_text",
        "input_ids",
        "token_ids",
        "raw_tokens",
        "record_id",
        "record_ids",
        "per_record",
        "samples",
        "rows",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_DELTA_QUANTUM = Decimal("0.000000000001")
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


class GenerationEvalError(RuntimeError):
    """Fail-closed evaluation error containing only a registered code."""


class _DuplicateKey(ValueError):
    pass


class _InvalidJSONConstant(ValueError):
    pass


def _fail(code: str) -> None:
    if code not in REGISTERED_FAILURE_CODES:
        raise GenerationEvalError("eval_receipt_invalid")
    raise GenerationEvalError(code)


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
            _fail("eval_config_invalid")
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            _fail("eval_config_invalid")
        raw = path.read_bytes()
        if len(raw) != size or b"\r" in raw or not raw.endswith(b"\n"):
            _fail("eval_config_invalid")
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
        _fail("eval_config_invalid")
    if not isinstance(value, dict):
        _fail("eval_config_invalid")
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


def _load_receipt_snapshot(
    path_value: str | os.PathLike[str],
    *,
    expected_sha256: str,
    failure_code: str,
) -> dict[str, Any]:
    """Authenticate one physical read before parsing, then recheck identity."""

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
        file_attributes = int(getattr(before, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or (reparse_flag and file_attributes & reparse_flag)
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
        ):
            _fail(failure_code)
        baseline = _stat_signature(before)
        with path.open("rb", buffering=0) as handle:
            descriptor_before = os.fstat(handle.fileno())
            if _stat_signature(descriptor_before) != baseline:
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
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
        if not isinstance(parsed, dict):
            _fail(failure_code)
        terminal = os.lstat(path)
        if _stat_signature(terminal) != baseline:
            _fail(failure_code)
        return parsed
    except GenerationEvalError:
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


def _schema(path: Path, expected_sha256: str) -> Draft202012Validator:
    schema, observed = _load_json(path)
    if observed != expected_sha256:
        _fail("eval_config_invalid")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception:
        _fail("eval_config_invalid")
    return Draft202012Validator(schema)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_sha256(config: Mapping[str, Any]) -> str:
    observed = config.get("_config_sha256")
    if _is_sha256(observed):
        return str(observed)
    return _sha256(_canonical_json(_public_config(config)))


def _bound_schema_validators(
    config: Mapping[str, Any],
) -> tuple[Draft202012Validator, Draft202012Validator]:
    schemas = config.get("schemas")
    if not isinstance(schemas, Mapping):
        _fail("eval_config_invalid")
    expected = {
        "config": (
            CONFIG_SCHEMA_PATH,
            "configs/research/"
            "gemma3_chat_unbalanced_v2_generation_eval_v1_config.schema.json",
            CONFIG_VERSION,
        ),
        "receipt": (
            RECEIPT_SCHEMA_PATH,
            "configs/research/"
            "gemma3_chat_unbalanced_v2_generation_eval_v1_receipt.schema.json",
            RECEIPT_VERSION,
        ),
    }
    validators: list[Draft202012Validator] = []
    for kind in ("config", "receipt"):
        binding = schemas.get(kind)
        if not isinstance(binding, Mapping):
            _fail("eval_config_invalid")
        path, relative, version = expected[kind]
        if (
            set(binding) != {"path", "version", "sha256"}
            or binding.get("path") != relative
            or binding.get("version") != version
            or not _is_sha256(binding.get("sha256"))
        ):
            _fail("eval_config_invalid")
        validators.append(_schema(path, str(binding["sha256"])))
    return validators[0], validators[1]


def expected_count_matrix() -> dict[str, int]:
    return {
        "tool_records": 400,
        "tool_arms": 3,
        "tool_slots": 1200,
        "planner_records": 240,
        "planner_arms": 4,
        "planner_slots": 960,
        "router_train": 80,
        "router_eval_proxy": 20,
        "router_slots": 100,
        "identity_records": 50,
        "identity_arms": 3,
        "identity_slots": 150,
        "wrong_route_records": 860,
        "wrong_route_derangements": 3,
        "wrong_route_slots": 2580,
        "total_slots": 4990,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    config_validator, _ = _bound_schema_validators(config)
    if tuple(config_validator.iter_errors(_public_config(config))):
        _fail("eval_config_invalid")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("profile_id") != PROFILE_ID
    ):
        _fail("eval_config_invalid")
    matrix = config["matrix"]
    if (
        tuple(matrix["tool"]["arms"]) != TOOL_ARMS
        or tuple(matrix["planner"]["arms"]) != PLANNER_ARMS
        or tuple(matrix["identity"]["arms"]) != IDENTITY_ARMS
        or tuple(matrix["wrong_route"]["derangements"]) != WRONG_ROUTE_DERANGEMENTS
    ):
        _fail("eval_config_invalid")
    if (
        matrix["tool"]["slots"]
        + matrix["planner"]["slots"]
        + matrix["router"]["slots"]
        + matrix["identity"]["slots"]
        + matrix["wrong_route"]["slots"]
        != matrix["total_slots"]
    ):
        _fail("eval_count_mismatch")
    metrics = config["metrics"]
    if (
        tuple(metrics["tool"]) != TOOL_METRICS
        or tuple(metrics["planner"]) != PLANNER_METRICS
        or tuple(metrics["identity"]) != IDENTITY_METRICS
        or tuple(metrics["wrong_route"])
        != (
            "primary_metric",
            "correct_minus_wrong_delta",
            "wrong_route_format_leak_absent",
        )
    ):
        _fail("eval_config_invalid")
    consumer = config["consumer_preflight"]
    dataset = config["dataset_inventory"]
    if (
        consumer["required_schema_version"] != CONSUMER_RECEIPT_VERSION
        or dataset["shard_inventory_sha256"] != consumer["shard_inventory_sha256"]
    ):
        _fail("eval_dataset_inventory_mismatch")
    if config["training_run"]["required_schema_version"] != TRAINING_RECEIPT_VERSION:
        _fail("eval_config_invalid")
    if config["kv_probe"]["required_schema_version"] != KV_RECEIPT_VERSION:
        _fail("eval_config_invalid")
    if config["numeric_rule"] != {
        "version": "decimal-string-quantized-1e-12-v1",
        "encoding": "fixed_12_decimal_places",
        "quantum": "0.000000000001",
        "rounding": "ROUND_HALF_EVEN",
    }:
        _fail("eval_config_invalid")
    claims = config["claims"]
    if any(
        claims[name] is not False
        for name in (
            "quality_validated",
            "generalization_validated",
            "eval_proxy_is_heldout",
            "wrong_route_is_causal_proof",
            "kv_shared_storage_claimed",
            "kv_zero_copy_claimed",
            "kv_rdma_claimed",
            "physical_execution_completed",
            "formal",
            "release",
        )
    ):
        _fail("eval_config_invalid")
    if set(config["failure_codes"]) != REGISTERED_FAILURE_CODES:
        _fail("eval_config_invalid")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config, digest = _load_json(Path(path))
    config["_config_sha256"] = digest
    validate_config(config)
    return config


def _pending(config: Mapping[str, Any]) -> tuple[str, ...]:
    pending: list[str] = []
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
        "kv_probe": ("receipt_sha256", "receipt_schema_sha256"),
        "dataset_inventory": ("shard_inventory_sha256",),
    }
    for group_name, names in groups.items():
        group = config[group_name]
        for name in names:
            if group[name] == "pending":
                pending.append(f"{group_name}.{name}")
    return tuple(pending)


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
    pending = _pending(config)
    return {
        "schema_version": CONFIG_VERSION,
        "profile_id": PROFILE_ID,
        "status": (
            "blocked_waiting_for_authenticated_inputs"
            if pending
            else "model_free_contract_ready_physical_backend_pending"
        ),
        "pending": list(pending),
        "config_sha256": config_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "expected_counts": expected_count_matrix(),
        "model_loaded": False,
        "gpu_touched": False,
        "evaluation_executed": False,
        "audit": _zero_audit(),
    }


def _closed_mapping(
    value: object, expected_keys: Sequence[str], code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        _fail(code)
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
    receipt = _load_receipt_snapshot(
        receipt_path,
        expected_sha256=receipt_sha256,
        failure_code=failure_code,
    )
    schema = _load_receipt_snapshot(
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
    except GenerationEvalError:
        raise
    except Exception:
        _fail(failure_code)
    return receipt


def _validate_dependency_metadata(
    config: Mapping[str, Any],
    *,
    consumer_receipt: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    kv_receipt: Mapping[str, Any],
) -> None:
    consumer = config["consumer_preflight"]
    _closed_mapping(
        consumer_receipt,
        ("schema_version", "status", "identity"),
        "eval_consumer_preflight_pending",
    )
    if (
        consumer_receipt.get("schema_version") != CONSUMER_RECEIPT_VERSION
        or consumer_receipt.get("status") != consumer["required_status"]
    ):
        _fail("eval_consumer_preflight_pending")
    consumer_identity = _closed_mapping(
        consumer_receipt.get("identity"),
        ("artifact_tree_digest_sha256", "shard_inventory_sha256"),
        "eval_consumer_preflight_pending",
    )
    for name in ("artifact_tree_digest_sha256", "shard_inventory_sha256"):
        if (
            consumer[name] == "pending"
            or not _is_sha256(consumer_identity.get(name))
            or consumer_identity.get(name) != consumer[name]
        ):
            _fail("eval_consumer_preflight_pending")

    training = config["training_run"]
    _closed_mapping(
        training_receipt,
        ("schema_version", "status", "run_id", "identity", "claims"),
        "eval_training_run_pending",
    )
    if (
        training_receipt.get("schema_version") != TRAINING_RECEIPT_VERSION
        or training_receipt.get("status") != training["required_status"]
        or training_receipt.get("run_id") != training["run_id"]
    ):
        _fail("eval_training_run_pending")
    training_identity = _closed_mapping(
        training_receipt.get("identity"),
        (
            "consumer_preflight_receipt_sha256",
            "artifact_tree_digest_sha256",
            "shard_inventory_sha256",
            "adapter_inventory_sha256",
        ),
        "eval_training_run_pending",
    )
    expected_training_identity = {
        "consumer_preflight_receipt_sha256": consumer["receipt_sha256"],
        "artifact_tree_digest_sha256": consumer["artifact_tree_digest_sha256"],
        "shard_inventory_sha256": consumer["shard_inventory_sha256"],
        "adapter_inventory_sha256": training["adapter_inventory_sha256"],
    }
    if any(
        expected == "pending"
        or not _is_sha256(training_identity.get(name))
        or training_identity.get(name) != expected
        for name, expected in expected_training_identity.items()
    ):
        _fail("eval_training_run_pending")
    training_claims = _closed_mapping(
        training_receipt.get("claims"),
        ("fresh_base", "fresh_adapter", "resume"),
        "eval_training_run_pending",
    )
    if training_claims != {
        "fresh_base": True,
        "fresh_adapter": True,
        "resume": False,
    }:
        _fail("eval_training_run_pending")

    kv = config["kv_probe"]
    _closed_mapping(
        kv_receipt,
        ("schema_version", "status", "identity", "claims"),
        "eval_kv_probe_pending",
    )
    if (
        kv_receipt.get("schema_version") != KV_RECEIPT_VERSION
        or kv_receipt.get("status") != kv["required_status"]
    ):
        _fail("eval_kv_probe_pending")
    kv_identity = _closed_mapping(
        kv_receipt.get("identity"),
        ("consumer_preflight_receipt_sha256", "shard_inventory_sha256"),
        "eval_kv_probe_pending",
    )
    if (
        not _is_sha256(kv_identity.get("consumer_preflight_receipt_sha256"))
        or not _is_sha256(kv_identity.get("shard_inventory_sha256"))
        or kv_identity.get("consumer_preflight_receipt_sha256")
        != consumer["receipt_sha256"]
        or kv_identity.get("shard_inventory_sha256")
        != consumer["shard_inventory_sha256"]
    ):
        _fail("eval_kv_probe_pending")
    kv_claims = _closed_mapping(
        kv_receipt.get("claims"),
        (
            "exact_prefix_value_handoff",
            "prefill_compute_handoff",
            "shared_storage",
            "zero_copy",
            "full_generation_kv_shared",
            "rdma",
        ),
        "eval_kv_probe_pending",
    )
    expected_kv_claims = {
        "exact_prefix_value_handoff": True,
        "prefill_compute_handoff": True,
        "shared_storage": False,
        "zero_copy": False,
        "full_generation_kv_shared": False,
        "rdma": False,
    }
    if any(
        kv_claims.get(name) is not expected
        for name, expected in expected_kv_claims.items()
    ):
        _fail("eval_kv_probe_pending")


def build_preflight(
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    kv_receipt_path: str | os.PathLike[str] | None = None,
    kv_receipt_schema_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    status = build_status(config)
    pending = tuple(status["pending"])
    if pending:
        return {
            **status,
            "status": "blocked_waiting_for_authenticated_inputs",
            "ready": False,
            "gates": {
                "config_valid": True,
                "consumer_preflight_authenticated": False,
                "training_run_authenticated": False,
                "kv_probe_authenticated": False,
                "physical_backend_registered": False,
            },
        }
    if consumer_receipt_path is None or consumer_receipt_schema_path is None:
        _fail("eval_consumer_preflight_pending")
    if training_receipt_path is None or training_receipt_schema_path is None:
        _fail("eval_training_run_pending")
    if kv_receipt_path is None or kv_receipt_schema_path is None:
        _fail("eval_kv_probe_pending")
    assert consumer_receipt_path is not None
    assert consumer_receipt_schema_path is not None
    assert training_receipt_path is not None
    assert training_receipt_schema_path is not None
    assert kv_receipt_path is not None
    assert kv_receipt_schema_path is not None
    consumer_receipt = _authenticate_dependency(
        receipt_path=consumer_receipt_path,
        receipt_sha256=config["consumer_preflight"]["receipt_sha256"],
        schema_path=consumer_receipt_schema_path,
        schema_sha256=config["consumer_preflight"]["receipt_schema_sha256"],
        expected_schema_version=config["consumer_preflight"]["required_schema_version"],
        failure_code="eval_consumer_preflight_pending",
    )
    training_receipt = _authenticate_dependency(
        receipt_path=training_receipt_path,
        receipt_sha256=config["training_run"]["receipt_sha256"],
        schema_path=training_receipt_schema_path,
        schema_sha256=config["training_run"]["receipt_schema_sha256"],
        expected_schema_version=config["training_run"]["required_schema_version"],
        failure_code="eval_training_run_pending",
    )
    kv_receipt = _authenticate_dependency(
        receipt_path=kv_receipt_path,
        receipt_sha256=config["kv_probe"]["receipt_sha256"],
        schema_path=kv_receipt_schema_path,
        schema_sha256=config["kv_probe"]["receipt_schema_sha256"],
        expected_schema_version=config["kv_probe"]["required_schema_version"],
        failure_code="eval_kv_probe_pending",
    )
    _validate_dependency_metadata(
        config,
        consumer_receipt=consumer_receipt,
        training_receipt=training_receipt,
        kv_receipt=kv_receipt,
    )
    return {
        **status,
        "status": "ready_for_lazy_physical_backend",
        "ready": True,
        "gates": {
            "config_valid": True,
            "consumer_preflight_authenticated": True,
            "training_run_authenticated": True,
            "kv_probe_authenticated": True,
            "physical_backend_registered": False,
        },
    }


def ratio(numerator: int, denominator: int) -> dict[str, int | float]:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator < 1
        or numerator < 0
        or numerator > denominator
    ):
        _fail("eval_rate_mismatch")
    return {
        "denominator": denominator,
        "numerator": numerator,
        "rate": numerator / denominator,
    }


def _walk_aggregate(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_aggregate(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_aggregate(child)


def _assert_aggregate_only(receipt: Mapping[str, Any]) -> None:
    for key, _ in _walk_aggregate(receipt):
        if key in FORBIDDEN_AGGREGATE_KEYS:
            _fail("eval_aggregate_only_violation")


def _validate_ratio(value: Mapping[str, Any], expected_denominator: int) -> None:
    denominator = value["denominator"]
    numerator = value["numerator"]
    rate_value = value["rate"]
    if denominator != expected_denominator or numerator > denominator:
        _fail("eval_rate_mismatch")
    expected = numerator / denominator
    if not math.isclose(rate_value, expected, rel_tol=0.0, abs_tol=1e-12):
        _fail("eval_rate_mismatch")


def _validate_arm_ratios(
    arms: Mapping[str, Any],
    expected_arms: Sequence[str],
    expected_metrics: Sequence[str],
    records: int,
) -> None:
    if set(arms) != set(expected_arms):
        _fail("eval_receipt_invalid")
    for arm in expected_arms:
        result = arms[arm]
        if result["records"] != records:
            _fail("eval_count_mismatch")
        for metric in expected_metrics:
            _validate_ratio(result[metric], records)


def _fixed_decimal(value: Decimal) -> str:
    quantized = value.quantize(_DELTA_QUANTUM, rounding=ROUND_HALF_EVEN)
    if quantized == Decimal("-0.000000000000"):
        quantized = Decimal("0.000000000000")
    return format(quantized, ".12f")


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    _validate_arm_ratios(metrics["tool"], TOOL_ARMS, TOOL_METRICS, 400)
    _validate_arm_ratios(metrics["planner"], PLANNER_ARMS, PLANNER_METRICS, 240)
    _validate_arm_ratios(metrics["identity"], IDENTITY_ARMS, IDENTITY_METRICS, 50)
    _validate_arm_ratios(
        metrics["wrong_route"],
        WRONG_ROUTE_DERANGEMENTS,
        WRONG_ROUTE_METRICS,
        860,
    )
    for split, records in (("train_proxy", 80), ("eval_proxy", 20)):
        result = metrics["router"][split]
        if result["records"] != records:
            _fail("eval_count_mismatch")
        _validate_ratio(result["top1_accuracy"], records)
        confusion = result["confusion_aggregate"]
        if len(confusion) != 9 or sum(confusion) != records:
            _fail("eval_count_mismatch")
        expected_macro_f1 = _macro_f1_from_confusion(confusion)
        if result["macro_f1"] != expected_macro_f1:
            _fail("eval_rate_mismatch")


def _macro_f1_from_confusion(confusion: Sequence[int]) -> str:
    matrix = [tuple(confusion[row * 3 : row * 3 + 3]) for row in range(3)]
    scores: list[Decimal] = []
    for index in range(3):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(3)) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            Decimal(0)
            if denominator == 0
            else Decimal(2 * true_positive) / Decimal(denominator)
        )
    return _fixed_decimal(sum(scores, start=Decimal(0)) / Decimal(3))


def _ratio_decimal(value: Mapping[str, Any]) -> Decimal:
    return Decimal(int(value["numerator"])) / Decimal(int(value["denominator"]))


def _fixed_delta(value: Decimal) -> str:
    return _fixed_decimal(value)


def expected_comparisons(metrics: Mapping[str, Any]) -> dict[str, str]:
    """Recompute every derived comparison using the versioned decimal rule."""

    tool = metrics["tool"]
    planner = metrics["planner"]
    identity = metrics["identity"]
    wrong_route = metrics["wrong_route"]
    tool_primary = {
        arm: _ratio_decimal(tool[arm]["execution_projection_exact"])
        for arm in TOOL_ARMS
    }
    planner_primary = {
        arm: _ratio_decimal(planner[arm]["selected_expert_exact"])
        for arm in PLANNER_ARMS
    }
    identity_primary = {
        arm: _ratio_decimal(identity[arm]["air_attribution"]) for arm in IDENTITY_ARMS
    }
    wrong_primary = [
        _ratio_decimal(wrong_route[arm]["primary_metric"])
        for arm in WRONG_ROUTE_DERANGEMENTS
    ]
    wrong_macro_degradation = Decimal(1) - sum(
        wrong_primary, start=Decimal(0)
    ) / Decimal(len(wrong_primary))
    return {
        "tool_q_only_minus_base": _fixed_delta(
            tool_primary["tool_q_only"] - tool_primary["tool_base"]
        ),
        "tool_q_plus_micro_o_minus_q_only": _fixed_delta(
            tool_primary["tool_q_plus_micro_o"] - tool_primary["tool_q_only"]
        ),
        "planner_q_only_minus_base": _fixed_delta(
            planner_primary["planner_q_only"] - planner_primary["planner_base"]
        ),
        "planner_o_only_minus_base": _fixed_delta(
            planner_primary["planner_o_only"] - planner_primary["planner_base"]
        ),
        "planner_q_plus_micro_o_minus_q_only": _fixed_delta(
            planner_primary["planner_q_plus_micro_o"]
            - planner_primary["planner_q_only"]
        ),
        "identity_correct_minus_base": _fixed_delta(
            identity_primary["identity_correct_route"]
            - identity_primary["identity_base"]
        ),
        "identity_correct_minus_wrong": _fixed_delta(
            identity_primary["identity_correct_route"]
            - identity_primary["identity_wrong_route"]
        ),
        "wrong_route_macro_degradation": _fixed_delta(wrong_macro_degradation),
    }


def _validate_strata(strata: Mapping[str, Any]) -> None:
    expected = {
        "tool": {
            "seen_template_interpolation": 160,
            "argument_composition_proxy": 120,
            "cross_language_transfer_proxy": 120,
        },
        "planner": {
            "task_classification": 80,
            "expert_selection": 80,
            "commit_boundary": 80,
        },
    }
    if set(strata) != set(expected):
        _fail("eval_count_mismatch")
    for group, group_counts in expected.items():
        summaries = strata[group]
        if set(summaries) != set(group_counts):
            _fail("eval_count_mismatch")
        for name, records in group_counts.items():
            summary = summaries[name]
            if summary["records"] != records:
                _fail("eval_count_mismatch")
            _validate_ratio(summary["primary_metric"], records)


def _validate_performance(performance: Mapping[str, Any]) -> None:
    expected_counts = {
        "tool": 1200,
        "planner": 960,
        "router": 100,
        "identity": 150,
        "wrong_route": 2580,
        "shared_prefix_kv": 1,
    }
    for group, expected_count in expected_counts.items():
        summary = performance[group]
        if summary["count"] != expected_count:
            _fail("eval_count_mismatch")
        values = (
            summary["latency_ms_mean"],
            summary["latency_ms_p50"],
            summary["latency_ms_p95"],
            summary["throughput_units_per_second"],
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            _fail("eval_receipt_invalid")
        if summary["latency_ms_p50"] > summary["latency_ms_p95"]:
            _fail("eval_receipt_invalid")


def validate_receipt(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    kv_receipt_path: str | os.PathLike[str] | None = None,
    kv_receipt_schema_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    preflight = build_preflight(
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        kv_receipt_path=kv_receipt_path,
        kv_receipt_schema_path=kv_receipt_schema_path,
    )
    if not preflight["ready"]:
        _fail("eval_consumer_preflight_pending")
    _assert_aggregate_only(receipt)
    _, receipt_validator = _bound_schema_validators(config)
    if tuple(receipt_validator.iter_errors(receipt)):
        _fail("eval_receipt_invalid")
    if _pending(config):
        _fail("eval_identity_mismatch")
    consumer = config["consumer_preflight"]
    training = config["training_run"]
    kv = config["kv_probe"]
    expected_identity = {
        "config_sha256": config_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "consumer_preflight_receipt_sha256": consumer["receipt_sha256"],
        "artifact_tree_digest_sha256": consumer["artifact_tree_digest_sha256"],
        "shard_inventory_sha256": consumer["shard_inventory_sha256"],
        "training_run_receipt_sha256": training["receipt_sha256"],
        "adapter_inventory_sha256": training["adapter_inventory_sha256"],
        "kv_probe_receipt_sha256": kv["receipt_sha256"],
        "run_id": training["run_id"],
    }
    if receipt["identity"] != expected_identity:
        _fail("eval_identity_mismatch")
    expected_counts = expected_count_matrix()
    if (
        receipt["planned_counts"] != expected_counts
        or receipt["observed_counts"] != expected_counts
    ):
        _fail("eval_count_mismatch")
    _validate_metrics(receipt["metrics"])
    if receipt["comparisons"] != expected_comparisons(receipt["metrics"]):
        _fail("eval_rate_mismatch")
    _validate_strata(receipt["strata"])
    _validate_performance(receipt["performance"])
    if receipt["kv_handoff"] != {
        "receipt_status": "passed_diagnostic_prefix_value_handoff",
        "exact_prefix_value_handoff": True,
        "prefill_compute_handoff": True,
        "shared_storage": False,
        "zero_copy": False,
        "full_generation_kv_shared": False,
        "rdma": False,
        "tail_private_append_only": True,
        "layer_value_comparisons": 26,
    }:
        _fail("eval_kv_probe_pending")
    passed = receipt["status"] == "passed_diagnostic_proxy_evaluation"
    if passed and receipt["failure_codes"]:
        _fail("eval_receipt_invalid")
    if not passed and not receipt["failure_codes"]:
        _fail("eval_receipt_invalid")
    if any(code not in REGISTERED_FAILURE_CODES for code in receipt["failure_codes"]):
        _fail("eval_receipt_invalid")
    return dict(receipt)


PhysicalBackend = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def execute(
    config: Mapping[str, Any],
    *,
    consumer_receipt_path: str | os.PathLike[str] | None = None,
    consumer_receipt_schema_path: str | os.PathLike[str] | None = None,
    training_receipt_path: str | os.PathLike[str] | None = None,
    training_receipt_schema_path: str | os.PathLike[str] | None = None,
    kv_receipt_path: str | os.PathLike[str] | None = None,
    kv_receipt_schema_path: str | os.PathLike[str] | None = None,
    backend: PhysicalBackend | None = None,
) -> dict[str, Any]:
    """Invoke a future physical evaluator lazily and validate its receipt."""

    preflight = build_preflight(
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        kv_receipt_path=kv_receipt_path,
        kv_receipt_schema_path=kv_receipt_schema_path,
    )
    if not preflight["ready"]:
        _fail("eval_consumer_preflight_pending")
    if backend is None:
        _fail("eval_physical_backend_required")
    receipt = backend(_public_config(config))
    if not isinstance(receipt, Mapping):
        _fail("eval_receipt_invalid")
    return validate_receipt(
        receipt,
        config,
        consumer_receipt_path=consumer_receipt_path,
        consumer_receipt_schema_path=consumer_receipt_schema_path,
        training_receipt_path=training_receipt_path,
        training_receipt_schema_path=training_receipt_schema_path,
        kv_receipt_path=kv_receipt_path,
        kv_receipt_schema_path=kv_receipt_schema_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--consumer-receipt")
    parser.add_argument("--consumer-receipt-schema")
    parser.add_argument("--training-receipt")
    parser.add_argument("--training-receipt-schema")
    parser.add_argument("--kv-receipt")
    parser.add_argument("--kv-receipt-schema")
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
                kv_receipt_path=args.kv_receipt,
                kv_receipt_schema_path=args.kv_receipt_schema,
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
                kv_receipt_path=args.kv_receipt,
                kv_receipt_schema_path=args.kv_receipt_schema,
            )
        else:
            result = execute(
                config,
                consumer_receipt_path=args.consumer_receipt,
                consumer_receipt_schema_path=args.consumer_receipt_schema,
                training_receipt_path=args.training_receipt,
                training_receipt_schema_path=args.training_receipt_schema,
                kv_receipt_path=args.kv_receipt,
                kv_receipt_schema_path=args.kv_receipt_schema,
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except GenerationEvalError as exc:
        code = str(exc)
        if code not in REGISTERED_FAILURE_CODES:
            code = "eval_receipt_invalid"
        print(
            json.dumps(
                {"error_type": "GenerationEvalError", "error_code": code},
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
