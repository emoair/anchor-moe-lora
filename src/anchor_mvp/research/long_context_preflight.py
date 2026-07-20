"""Static Gemma 4 12B long-context capacity preflight.

The default mode reads one small YAML file and performs integer KV-cache math;
it does not open a model, dataset, JSONL partition, network connection, or GPU.
The explicit producer-authentication mode additionally reads only the frozen,
body-free scalar inventory and its contract files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SCHEMA_VERSION = "anchor.gemma4-12b-long-context-preflight-config.v1"
REPORT_SCHEMA_VERSION = "anchor.gemma4-12b-long-context-preflight-report.v1"
INVENTORY_PRODUCER_VERSION = "anchor.long-context-token-inventory-producer.v1"
INVENTORY_SCHEMA_VERSION = "anchor.long-context-token-inventory.v1"
INVENTORY_MANIFEST_SCHEMA_VERSION = "anchor.long-context-token-inventory-manifest.v1"

_FROZEN_INVENTORY_RELEASE = {
    "branch": "agent/restore-dual-router-ux",
    "commit": "677bd2a689de7f904d808f35ec6d19adc73e6d2e",
    "config_path": "configs/research/swebench_long_context_preflight_v1.yaml",
    "config_sha256": "79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe",
    "record_schema_path": (
        "configs/research/swebench_long_context_preflight_sidecar.schema.json"
    ),
    "record_schema_sha256": "aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f",
    "manifest_schema_path": (
        "configs/research/swebench_long_context_preflight_manifest.schema.json"
    ),
    "manifest_schema_sha256": "8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0",
    "source_projector_manifest_sha256": (
        "595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac"
    ),
}
_FROZEN_TOKENIZER_BINDING = {
    "sha256": "047777c4fd6647d75ec3afe5d979ab7c6f02b43397e31886b7f1cd2873519153",
    "backend": "explicit_synthetic_tokenizer",
    "inventory_mode": "synthetic_fixture",
    "synthetic_fixture_only": True,
    "target_model_tokenizer_match": "not_applicable",
    "gemma_target_identity_verified": False,
}
_FROZEN_INVENTORY_PARTITIONS = (
    {
        "path": "train/clean.jsonl",
        "split": "train",
        "variant": "clean",
        "records": 5,
        "bytes": 7121,
        "sha256": "d58471790406130cfbbde0b473a296665227f920f4f338455302fde462167846",
    },
    {
        "path": "train/noisy.jsonl",
        "split": "train",
        "variant": "noisy",
        "records": 5,
        "bytes": 7127,
        "sha256": "dfc3e5423ca4368a3974d9cdfc312af540b75c44aa41ff5c43ff940343c60bc1",
    },
    {
        "path": "calibration/clean.jsonl",
        "split": "calibration",
        "variant": "clean",
        "records": 5,
        "bytes": 7151,
        "sha256": "6fcc71a051cab56ab253ffe8cf23983c5e13f3515f8651a6eeb00bc27f7712e5",
    },
)
_FROZEN_INVENTORY_COUNTS = {
    "partitions": 3,
    "records": 15,
    "task_bundles": 2,
    "complete_five_role_groups": 3,
    "segment_references": 89,
    "unique_segments": 25,
    "provider_requests": 0,
}
_FROZEN_FIXTURE_MANIFEST_SHA256 = (
    "73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e"
)
_FROZEN_FIXTURE_SIDECAR_PHYSICAL_SHA256 = (
    "1f3b3d281556814ff7db900c5119eef9e54ea69945a24aed2e5b4ec295d6f41c"
)

_EXPECTED_ARCHITECTURE = {
    "text_layers": 48,
    "sliding_layers": 40,
    "full_attention_layers": 8,
    "sliding_window_tokens": 1024,
    "sliding_kv_heads": 8,
    "sliding_head_dim": 256,
    "global_kv_heads": 1,
    "global_head_dim": 512,
}
_EXPECTED_BUCKETS = (
    ("8k", 8192),
    ("16k", 16384),
    ("32k", 32768),
    ("64k", 65536),
    ("128k", 131072),
    ("256k", 262144),
    ("512k", 524288),
    ("1mi", 1048576),
)
_EXPECTED_INVENTORY_FIELDS = (
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
)
_EXPECTED_PROHIBITED_BODY_FIELDS = (
    "messages",
    "prompt",
    "completion",
    "content",
    "task_board",
    "blocks",
)
_EXPECTED_ROLE_STAGE = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "frontend_gen": "domain_builder",
    "frontend_review": "domain_review",
    "security_gate": "security",
}
_INVENTORY_BUCKET_GATES = (
    (8192, "le_8k", "measurement_candidate"),
    (16384, "le_16k", "measurement_candidate"),
    (32768, "le_32k", "measurement_candidate"),
    (65536, "le_64k", "measurement_candidate"),
    (131072, "le_128k", "measurement_candidate"),
    (262144, "le_256k", "capability_only"),
    (1048576, "le_1m", "research_only_blocked"),
)


class LongContextPreflightError(ValueError):
    """The static long-context configuration violates its pinned contract."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LongContextPreflightError(f"{field} must be a mapping")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LongContextPreflightError(f"{field} must be a positive integer")
    return value


def _require_equal(actual: object, expected: object, field: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise LongContextPreflightError(
            f"{field} must be {expected!r}, got {actual!r}"
        )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LongContextPreflightError(f"{field} must be a lowercase SHA-256")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class _BytesSnapshot:
    data: bytes
    sha256: str
    size: int


def _read_snapshot(path: Path, field: str) -> _BytesSnapshot:
    """Read one stable regular-file snapshot and reject path replacement."""

    try:
        if not path.is_file() or path.is_symlink():
            _inventory_auth_fail(f"{field} must be a non-symlink regular file")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise LongContextPreflightError(
            f"producer inventory authentication failed: {field} is missing or unreadable"
        ) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _inventory_auth_fail(f"{field} changed during snapshot authentication")
    return _BytesSnapshot(
        data=data,
        sha256=_sha256_bytes(data),
        size=len(data),
    )


def _json_mapping_snapshot(raw: bytes, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LongContextPreflightError(
            f"producer inventory authentication failed: {field} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise LongContextPreflightError(
            f"producer inventory authentication failed: {field} must be a mapping"
        )
    return value


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _quantized_bytes(elements: int, cache: Mapping[str, Any], field: str) -> int:
    block_elements = _positive_int(cache.get("block_elements"), f"{field}.block_elements")
    block_bytes = _positive_int(cache.get("block_bytes"), f"{field}.block_bytes")
    if elements % block_elements:
        raise LongContextPreflightError(
            f"{field} element count is not block aligned"
        )
    return elements // block_elements * block_bytes


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the bounded YAML snapshot and return it with its physical SHA-256."""

    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise LongContextPreflightError("config root must be a mapping")
    validate_config(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    """Pin the local model/runtime facts used by the deterministic calculation."""

    _require_equal(config.get("schema_version"), CONFIG_SCHEMA_VERSION, "schema_version")
    _require_equal(config.get("mode"), "static_metadata_only", "mode")

    model = _mapping(config.get("model"), "model")
    _require_equal(model.get("architecture"), "gemma4", "model.architecture")
    _require_equal(model.get("scale"), "12b", "model.scale")
    _require_equal(model.get("model_id"), "google/gemma-4-12B", "model.model_id")
    _require_equal(
        model.get("revision"),
        "56820d7d8cbe8e47975a53325439ed272e91cff2",
        "model.revision",
    )
    _require_equal(
        model.get("source_config_sha256"),
        "14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098",
        "model.source_config_sha256",
    )
    _require_equal(
        model.get("native_context_tokens"),
        262144,
        "model.native_context_tokens",
    )

    architecture = _mapping(config.get("architecture"), "architecture")
    _require_equal(dict(architecture), _EXPECTED_ARCHITECTURE, "architecture")

    rope = _mapping(config.get("rope"), "rope")
    full_rope = _mapping(rope.get("full_attention"), "rope.full_attention")
    local_rope = _mapping(rope.get("sliding_attention"), "rope.sliding_attention")
    _require_equal(
        full_rope.get("encoding"),
        "proportional_rope",
        "full rope encoding",
    )
    _require_equal(full_rope.get("theta"), 1_000_000, "full rope theta")
    _require_equal(
        full_rope.get("partial_rotary_factor"),
        0.25,
        "full rope partial_rotary_factor",
    )
    _require_equal(local_rope.get("encoding"), "rope", "local rope encoding")
    _require_equal(local_rope.get("theta"), 10_000, "local rope theta")
    _require_equal(
        rope.get("extrapolation_scaling_bound"),
        False,
        "rope.extrapolation_scaling_bound",
    )

    runtime = _mapping(config.get("llama_cpp_runtime"), "llama_cpp_runtime")
    for field, expected in (
        ("commit", "33c718db1fbfe834f30eef28cf206f98736fe612"),
        (
            "gemma4_source_sha256",
            "05f9780eab0418ed01741596ef1b390f54b98d8fea66609ffdb21ea1af1de3e7",
        ),
        ("parallel_slots", 1),
        ("ubatch_tokens", 256),
        ("swa_alignment_tokens", 256),
        ("swa_full", False),
        ("flash_attention", True),
        ("conservative_separate_k_and_v", True),
    ):
        _require_equal(runtime.get(field), expected, f"llama_cpp_runtime.{field}")
    cache_k = _mapping(runtime.get("cache_k"), "llama_cpp_runtime.cache_k")
    cache_v = _mapping(runtime.get("cache_v"), "llama_cpp_runtime.cache_v")
    _require_equal(dict(cache_k), {"type": "q8_0", "block_elements": 32, "block_bytes": 34}, "cache_k")
    _require_equal(dict(cache_v), {"type": "q4_0", "block_elements": 32, "block_bytes": 18}, "cache_v")

    buckets = config.get("context_buckets")
    if not isinstance(buckets, list):
        raise LongContextPreflightError("context_buckets must be a list")
    bucket_pairs = tuple(
        (str(_mapping(item, "context bucket").get("id")), _mapping(item, "context bucket").get("tokens"))
        for item in buckets
    )
    _require_equal(bucket_pairs, _EXPECTED_BUCKETS, "context_buckets")

    plan = config.get("staged_research_plan")
    expected_plan = [
        {"id": "native-64k", "tokens": 65536, "position_scaling": "none", "status": "static_only"},
        {"id": "native-128k", "tokens": 131072, "position_scaling": "none", "status": "static_only"},
        {"id": "native-256k", "tokens": 262144, "position_scaling": "none", "status": "static_only"},
        {"id": "research-512k", "tokens": 524288, "position_scaling": "yarn_or_linear_2x", "status": "blocked"},
        {"id": "research-1mi", "tokens": 1048576, "position_scaling": "yarn_4x_or_longrope_training", "status": "blocked"},
    ]
    _require_equal(plan, expected_plan, "staged_research_plan")

    inventory = _mapping(
        config.get("producer_token_inventory_contract"),
        "producer_token_inventory_contract",
    )
    _require_equal(
        inventory.get("producer_version"),
        INVENTORY_PRODUCER_VERSION,
        "producer inventory producer_version",
    )
    _require_equal(
        inventory.get("record_schema_version"),
        INVENTORY_SCHEMA_VERSION,
        "producer inventory record_schema_version",
    )
    _require_equal(
        inventory.get("manifest_schema_version"),
        INVENTORY_MANIFEST_SCHEMA_VERSION,
        "producer inventory manifest_schema_version",
    )
    _require_equal(
        inventory.get("interface_status"),
        "frozen_authenticated_producer_handoff",
        "producer inventory interface_status",
    )
    _require_equal(
        inventory.get("frozen_producer_schema_claimed"),
        True,
        "producer inventory frozen_producer_schema_claimed",
    )
    _require_equal(
        inventory.get("materialization"),
        "metadata_only_no_content_bodies",
        "producer inventory materialization",
    )
    _require_equal(
        inventory.get("split_group_key"),
        "task_bundle_sha256",
        "producer inventory split_group_key",
    )
    _require_equal(
        inventory.get("split_before_augmentation"),
        True,
        "producer inventory split_before_augmentation",
    )
    release = _mapping(
        inventory.get("producer_release"),
        "producer inventory producer_release",
    )
    _require_equal(
        dict(release),
        _FROZEN_INVENTORY_RELEASE,
        "producer inventory producer_release",
    )
    tokenizer = _mapping(
        inventory.get("tokenizer_binding"),
        "producer inventory tokenizer_binding",
    )
    _require_equal(
        dict(tokenizer),
        _FROZEN_TOKENIZER_BINDING,
        "producer inventory tokenizer_binding",
    )
    fixture = _mapping(inventory.get("fixture"), "producer inventory fixture")
    _require_equal(
        fixture.get("path"),
        "fixtures/research/long_context_token_inventory",
        "producer inventory fixture.path",
    )
    _require_equal(
        fixture.get("manifest_sha256"),
        _FROZEN_FIXTURE_MANIFEST_SHA256,
        "producer inventory fixture.manifest_sha256",
    )
    _require_equal(
        fixture.get("manifest_sha256_sidecar_physical_sha256"),
        _FROZEN_FIXTURE_SIDECAR_PHYSICAL_SHA256,
        "producer inventory fixture.manifest_sha256_sidecar_physical_sha256",
    )
    partitions = fixture.get("partitions")
    if not isinstance(partitions, list):
        raise LongContextPreflightError(
            "producer inventory fixture.partitions must be a list"
        )
    _require_equal(
        tuple(dict(_mapping(item, "producer inventory partition")) for item in partitions),
        _FROZEN_INVENTORY_PARTITIONS,
        "producer inventory fixture.partitions",
    )
    fixture_counts = _mapping(
        fixture.get("counts"),
        "producer inventory fixture.counts",
    )
    _require_equal(
        dict(fixture_counts),
        _FROZEN_INVENTORY_COUNTS,
        "producer inventory fixture.counts",
    )
    _require_equal(
        tuple(inventory.get("required_fields", ())),
        _EXPECTED_INVENTORY_FIELDS,
        "producer inventory required_fields",
    )
    _require_equal(
        tuple(inventory.get("prohibited_body_fields", ())),
        _EXPECTED_PROHIBITED_BODY_FIELDS,
        "producer inventory prohibited_body_fields",
    )
    invariants = _mapping(inventory.get("invariants"), "producer inventory invariants")
    _require_equal(
        dict(invariants),
        {
            "input_tokens": (
                "shared_prefix_input_tokens + private_delta_input_tokens"
            ),
            "total_tokens": "input_tokens + reserved_output_tokens",
            "forbidden_current_future_excluded_before_serialization": True,
            "reuse_savings_tokens": 0,
            "provider_requests": 0,
        },
        "producer inventory invariants",
    )

    claims = _mapping(config.get("claims"), "claims")
    expected_false_claims = {
        "runtime_validated",
        "quality_validated",
        "training_validated",
        "one_million_context_supported",
        "model_loaded",
        "gpu_used",
        "network_used",
        "data_jsonl_read",
        "heldout_read",
    }
    if set(claims) != expected_false_claims or any(claims.values()):
        raise LongContextPreflightError("all static-only claim fields must be false")


def _inventory_auth_fail(reason: str) -> None:
    raise LongContextPreflightError(
        f"producer inventory authentication failed: {reason}"
    )


def _auth_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _inventory_auth_fail(f"{field} must be a mapping")
    return value


def _auth_equal(actual: object, expected: object, field: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        _inventory_auth_fail(f"{field} mismatch")


def _fixture_path(root: Path, relative: str, field: str) -> Path:
    parts = relative.split("/")
    if (
        not relative
        or any(not part or part in {".", ".."} for part in parts)
        or Path(relative).is_absolute()
    ):
        _inventory_auth_fail(f"{field} is not a safe relative path")
    candidate = root.joinpath(*parts)
    try:
        cursor = root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                _inventory_auth_fail(
                    f"{field} must not contain a symlink path component"
                )
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise LongContextPreflightError(
            f"producer inventory authentication failed: {field} is missing or "
            "escapes the fixture root"
        ) from exc
    if not candidate.is_file():
        _inventory_auth_fail(f"{field} must be a regular file")
    return candidate


def _validate_inventory_manifest(
    manifest: Mapping[str, Any],
    expected_partitions: Sequence[Mapping[str, Any]],
) -> None:
    _auth_equal(
        manifest.get("schema_version"),
        INVENTORY_MANIFEST_SCHEMA_VERSION,
        "manifest.schema_version",
    )
    producer = _auth_mapping(manifest.get("producer"), "manifest.producer")
    producer_expectations = {
        "producer_version": INVENTORY_PRODUCER_VERSION,
        "record_schema_version": INVENTORY_SCHEMA_VERSION,
        "config_sha256": _FROZEN_INVENTORY_RELEASE["config_sha256"],
        "record_schema_sha256": _FROZEN_INVENTORY_RELEASE[
            "record_schema_sha256"
        ],
        "manifest_schema_sha256": _FROZEN_INVENTORY_RELEASE[
            "manifest_schema_sha256"
        ],
    }
    for field, expected in producer_expectations.items():
        _auth_equal(producer.get(field), expected, f"manifest.producer.{field}")

    source = _auth_mapping(manifest.get("input"), "manifest.input")
    _auth_equal(
        source.get("projector_manifest_sha256"),
        _FROZEN_INVENTORY_RELEASE["source_projector_manifest_sha256"],
        "manifest.input.projector_manifest_sha256",
    )
    _auth_equal(
        manifest.get("inventory_mode"),
        "synthetic_fixture",
        "manifest.inventory_mode",
    )
    _auth_equal(
        manifest.get("status"),
        "synthetic_fixture_inventory_ready",
        "manifest.status",
    )
    _auth_equal(
        manifest.get("claim_scope"),
        "synthetic_fixture_contract_only",
        "manifest.claim_scope",
    )
    _auth_equal(
        manifest.get("target_model_tokenizer_match"),
        "not_applicable",
        "manifest.target_model_tokenizer_match",
    )
    tokenizer = _auth_mapping(
        manifest.get("tokenizer_binding"),
        "manifest.tokenizer_binding",
    )
    _auth_equal(
        tokenizer.get("backend"),
        "explicit_synthetic_tokenizer",
        "manifest.tokenizer_binding.backend",
    )
    _auth_equal(
        tokenizer.get("synthetic_fixture_only"),
        True,
        "manifest.tokenizer_binding.synthetic_fixture_only",
    )
    _auth_equal(
        _canonical_json_sha256(tokenizer),
        _FROZEN_TOKENIZER_BINDING["sha256"],
        "manifest.tokenizer_binding canonical SHA-256",
    )

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        _inventory_auth_fail("manifest.files must be a list")
    _auth_equal(
        tuple(dict(_auth_mapping(item, "manifest.files entry")) for item in manifest_files),
        tuple(dict(item) for item in expected_partitions),
        "manifest.files",
    )

    counts = _auth_mapping(manifest.get("counts"), "manifest.counts")
    count_expectations = {
        "total": _FROZEN_INVENTORY_COUNTS["records"],
        "unique_task_bundles": _FROZEN_INVENTORY_COUNTS["task_bundles"],
        "segment_references": _FROZEN_INVENTORY_COUNTS["segment_references"],
        "unique_segments": _FROZEN_INVENTORY_COUNTS["unique_segments"],
    }
    for field, expected in count_expectations.items():
        _auth_equal(counts.get(field), expected, f"manifest.counts.{field}")
    _auth_equal(
        counts.get("by_split"),
        {"calibration": 5, "train": 10},
        "manifest.counts.by_split",
    )
    _auth_equal(
        counts.get("by_variant"),
        {"clean": 10, "noisy": 5},
        "manifest.counts.by_variant",
    )
    _auth_equal(
        counts.get("by_expert"),
        {expert: 3 for expert in _EXPECTED_ROLE_STAGE},
        "manifest.counts.by_expert",
    )
    _auth_equal(
        counts.get("by_stage"),
        {stage: 3 for stage in _EXPECTED_ROLE_STAGE.values()},
        "manifest.counts.by_stage",
    )
    _auth_equal(manifest.get("provider_requests"), 0, "manifest.provider_requests")
    for field, expected in {
        "manifest_sha256_sidecar_required": True,
        "forbidden_current_future_excluded_before_serialization": True,
        "split_before_augmentation": True,
        "split_group_key": "task_bundle_sha256",
        "approximate_inventory_emitted": False,
        "null_inventory_emitted": False,
        "capability_validated": False,
        "quality_validated": False,
        "allocation_validated": False,
        "execution_authorized": False,
    }.items():
        _auth_equal(manifest.get(field), expected, f"manifest.{field}")


def _parse_inventory_partition(
    raw: bytes,
    descriptor: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    path = str(descriptor["path"])
    if not raw.endswith(b"\n") or b"\r" in raw:
        _inventory_auth_fail(f"partition {path} must use non-empty LF JSONL lines")
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        _inventory_auth_fail(f"partition {path} contains an empty JSONL line")
    _auth_equal(len(lines), descriptor["records"], f"partition {path} record count")
    records: list[Mapping[str, Any]] = []
    expected_keys = set(_EXPECTED_INVENTORY_FIELDS)
    for line_number, line in enumerate(lines, start=1):
        record = _json_mapping_snapshot(line, f"partition {path} line {line_number}")
        if set(record) != expected_keys:
            _inventory_auth_fail(
                f"partition {path} line {line_number} record fields mismatch"
            )
        if any(field in record for field in _EXPECTED_PROHIBITED_BODY_FIELDS):
            _inventory_auth_fail(
                f"partition {path} line {line_number} contains a prohibited body field"
            )
        _auth_equal(
            record.get("schema_version"),
            INVENTORY_SCHEMA_VERSION,
            f"partition {path} line {line_number} schema_version",
        )
        _auth_equal(
            record.get("split"),
            descriptor["split"],
            f"partition {path} line {line_number} split",
        )
        _auth_equal(
            record.get("variant"),
            descriptor["variant"],
            f"partition {path} line {line_number} variant",
        )
        expert = record.get("expert")
        if expert not in _EXPECTED_ROLE_STAGE:
            _inventory_auth_fail(
                f"partition {path} line {line_number} expert is unknown"
            )
        _auth_equal(
            record.get("stage"),
            _EXPECTED_ROLE_STAGE[str(expert)],
            f"partition {path} line {line_number} stage/expert binding",
        )
        for hash_field in (
            "task_bundle_sha256",
            "task_id_sha256",
            "source_partition_sha256",
            "source_line_sha256",
            "segment_plan_sha256",
            "ordered_segment_ids_sha256",
            "terminal_prefix_lineage_sha256",
            "tokenizer_binding_sha256",
        ):
            try:
                _require_sha256(
                    record.get(hash_field),
                    f"partition {path} line {line_number} {hash_field}",
                )
            except LongContextPreflightError as exc:
                _inventory_auth_fail(str(exc))
        _auth_equal(
            record.get("tokenizer_binding_sha256"),
            _FROZEN_TOKENIZER_BINDING["sha256"],
            f"partition {path} line {line_number} tokenizer binding",
        )
        integer_fields = (
            "segment_count",
            "private_delta_segment_count",
            "input_tokens",
            "shared_prefix_input_tokens",
            "private_delta_input_tokens",
            "reserved_output_tokens",
            "total_tokens",
            "reuse_savings_tokens",
            "provider_requests",
        )
        for field in integer_fields:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _inventory_auth_fail(
                    f"partition {path} line {line_number} {field} is invalid"
                )
        if record["segment_count"] < 1:
            _inventory_auth_fail(
                f"partition {path} line {line_number} segment_count is invalid"
            )
        if record["private_delta_segment_count"] > record["segment_count"]:
            _inventory_auth_fail(
                f"partition {path} line {line_number} private segment count is invalid"
            )
        if record["variant"] == "clean" and (
            record["private_delta_segment_count"] != 0
            or record["private_delta_input_tokens"] != 0
        ):
            _inventory_auth_fail(
                f"partition {path} line {line_number} clean private delta must be zero"
            )
        if record["variant"] == "noisy" and (
            record["private_delta_segment_count"] < 1
            or record["private_delta_input_tokens"] < 1
        ):
            _inventory_auth_fail(
                f"partition {path} line {line_number} noisy private delta must be positive"
            )
        _auth_equal(
            record["input_tokens"],
            record["shared_prefix_input_tokens"]
            + record["private_delta_input_tokens"],
            f"partition {path} line {line_number} input token accounting",
        )
        _auth_equal(
            record["total_tokens"],
            record["input_tokens"] + record["reserved_output_tokens"],
            f"partition {path} line {line_number} total token accounting",
        )
        bucket = "gt_1m"
        gate = "reject"
        for upper_bound, candidate_bucket, candidate_gate in _INVENTORY_BUCKET_GATES:
            if record["total_tokens"] <= upper_bound:
                bucket = candidate_bucket
                gate = candidate_gate
                break
        for field, expected in {
            "bucket": bucket,
            "gate": gate,
            "cache_identity_status": "identity_unbound",
            "reuse_savings_tokens": 0,
            "evaluation_status": "not_evaluated",
            "quality_validated": False,
            "allocation_validated": False,
            "execution_authorized": False,
            "provider_requests": 0,
        }.items():
            _auth_equal(
                record.get(field),
                expected,
                f"partition {path} line {line_number} {field}",
            )
        records.append(record)
    return tuple(records)


def authenticate_producer_token_inventory_contract_files(
    config: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Authenticate the three frozen producer contract files without data."""

    validate_config(config)
    root_input = Path(repository_root).expanduser()
    if root_input.is_symlink():
        _inventory_auth_fail("producer contract repository root must not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        raise LongContextPreflightError(
            "producer inventory authentication failed: producer contract "
            "repository root is missing"
        ) from exc
    if not root.is_dir():
        _inventory_auth_fail("producer contract repository root must be a directory")

    release = _mapping(
        config["producer_token_inventory_contract"]["producer_release"],
        "producer inventory producer_release",
    )
    descriptors = (
        (
            "config",
            str(release["config_path"]),
            str(release["config_sha256"]),
        ),
        (
            "record_schema",
            str(release["record_schema_path"]),
            str(release["record_schema_sha256"]),
        ),
        (
            "manifest_schema",
            str(release["manifest_schema_path"]),
            str(release["manifest_schema_sha256"]),
        ),
    )
    paths = {
        name: _fixture_path(root, relative, f"producer contract {name}")
        for name, relative, _ in descriptors
    }
    snapshots = {
        name: _read_snapshot(path, f"producer contract {name}")
        for name, path in paths.items()
    }

    try:
        producer_config = yaml.safe_load(snapshots["config"].data)
    except yaml.YAMLError as exc:
        raise LongContextPreflightError(
            "producer inventory authentication failed: producer config is invalid YAML"
        ) from exc
    producer_config = _auth_mapping(producer_config, "producer config")
    _auth_equal(
        producer_config.get("schema_version"),
        "anchor.long-context-token-inventory-config.v1",
        "producer config schema_version",
    )
    _auth_equal(
        producer_config.get("producer_version"),
        INVENTORY_PRODUCER_VERSION,
        "producer config producer_version",
    )
    output_contract = _auth_mapping(
        producer_config.get("output_contract"),
        "producer config output_contract",
    )
    _auth_equal(
        output_contract.get("record_schema_version"),
        INVENTORY_SCHEMA_VERSION,
        "producer config output record_schema_version",
    )
    _auth_equal(
        output_contract.get("manifest_schema_version"),
        INVENTORY_MANIFEST_SCHEMA_VERSION,
        "producer config output manifest_schema_version",
    )

    record_schema = _json_mapping_snapshot(
        snapshots["record_schema"].data,
        "producer record schema",
    )
    record_schema_properties = _auth_mapping(
        record_schema.get("properties"),
        "producer record schema properties",
    )
    record_schema_version = _auth_mapping(
        record_schema_properties.get("schema_version"),
        "producer record schema schema_version",
    )
    _auth_equal(
        record_schema_version.get("const"),
        INVENTORY_SCHEMA_VERSION,
        "producer record schema version",
    )
    manifest_schema = _json_mapping_snapshot(
        snapshots["manifest_schema"].data,
        "producer manifest schema",
    )
    manifest_schema_properties = _auth_mapping(
        manifest_schema.get("properties"),
        "producer manifest schema properties",
    )
    manifest_schema_version = _auth_mapping(
        manifest_schema_properties.get("schema_version"),
        "producer manifest schema schema_version",
    )
    _auth_equal(
        manifest_schema_version.get("const"),
        INVENTORY_MANIFEST_SCHEMA_VERSION,
        "producer manifest schema version",
    )

    for name, _, expected_sha256 in descriptors:
        _auth_equal(
            snapshots[name].sha256,
            expected_sha256,
            f"producer contract {name} SHA-256",
        )
    for name, path in paths.items():
        if _read_snapshot(path, f"producer contract {name}") != snapshots[name]:
            _inventory_auth_fail(
                f"producer contract {name} changed during authentication"
            )
    return {
        "status": "frozen_authenticated_producer_contract_files",
        "producer_version": INVENTORY_PRODUCER_VERSION,
        "record_schema_version": INVENTORY_SCHEMA_VERSION,
        "manifest_schema_version": INVENTORY_MANIFEST_SCHEMA_VERSION,
        "file_sha256": {
            name: snapshots[name].sha256 for name, _, _ in descriptors
        },
        "content_bodies_materialized": False,
        "model_loaded": False,
        "gpu_used": False,
        "network_used": False,
        "provider_requests": 0,
    }


def authenticate_producer_token_inventory_fixture(
    config: Mapping[str, Any],
    inventory_root: str | Path,
) -> dict[str, Any]:
    """Authenticate the frozen scalar-only producer fixture, fail closed.

    Inventory JSONL records are parsed only to validate their closed scalar/hash
    schema and aggregate counts. No prompt, completion, task-board block, heldout
    text, model, tokenizer, GPU, network, or provider is touched.
    """

    validate_config(config)
    root_input = Path(inventory_root).expanduser()
    if root_input.is_symlink():
        _inventory_auth_fail("fixture root must not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        raise LongContextPreflightError(
            "producer inventory authentication failed: fixture root is missing"
        ) from exc
    if not root.is_dir():
        _inventory_auth_fail("fixture root must be a directory")

    inventory = _mapping(
        config["producer_token_inventory_contract"],
        "producer_token_inventory_contract",
    )
    fixture = _mapping(inventory["fixture"], "producer inventory fixture")
    descriptors = tuple(
        dict(_mapping(item, "producer inventory partition"))
        for item in fixture["partitions"]
    )
    paths = {
        "manifest.json": _fixture_path(root, "manifest.json", "manifest.json"),
        "manifest.json.sha256": _fixture_path(
            root,
            "manifest.json.sha256",
            "manifest.json.sha256",
        ),
    }
    for descriptor in descriptors:
        relative = str(descriptor["path"])
        paths[relative] = _fixture_path(root, relative, f"partition {relative}")
    snapshots = {
        name: _read_snapshot(path, name) for name, path in paths.items()
    }

    manifest = _json_mapping_snapshot(
        snapshots["manifest.json"].data,
        "manifest.json",
    )
    _validate_inventory_manifest(manifest, descriptors)
    all_records: list[Mapping[str, Any]] = []
    for descriptor in descriptors:
        relative = str(descriptor["path"])
        snapshot = snapshots[relative]
        _auth_equal(
            snapshot.size,
            descriptor["bytes"],
            f"partition {relative} byte count",
        )
        all_records.extend(_parse_inventory_partition(snapshot.data, descriptor))

    _auth_equal(
        len(all_records),
        _FROZEN_INVENTORY_COUNTS["records"],
        "fixture aggregate record count",
    )
    bundles = {str(record["task_bundle_sha256"]) for record in all_records}
    _auth_equal(
        len(bundles),
        _FROZEN_INVENTORY_COUNTS["task_bundles"],
        "fixture task bundle count",
    )
    bundle_splits: dict[str, set[str]] = {}
    for record in all_records:
        bundle_splits.setdefault(str(record["task_bundle_sha256"]), set()).add(
            str(record["split"])
        )
    if any(len(splits) != 1 for splits in bundle_splits.values()):
        _inventory_auth_fail("task bundle crosses split boundaries")
    groups: dict[tuple[str, str, str], set[str]] = {}
    for record in all_records:
        key = (
            str(record["task_bundle_sha256"]),
            str(record["split"]),
            str(record["variant"]),
        )
        roles = groups.setdefault(key, set())
        expert = str(record["expert"])
        if expert in roles:
            _inventory_auth_fail("fixture role group contains a duplicate expert")
        roles.add(expert)
    _auth_equal(
        len(groups),
        _FROZEN_INVENTORY_COUNTS["complete_five_role_groups"],
        "fixture complete five-role group count",
    )
    for roles in groups.values():
        _auth_equal(roles, set(_EXPECTED_ROLE_STAGE), "fixture five-role group")
    _auth_equal(
        sum(int(record["segment_count"]) for record in all_records),
        _FROZEN_INVENTORY_COUNTS["segment_references"],
        "fixture segment reference count",
    )

    _auth_equal(
        snapshots["manifest.json"].sha256,
        _FROZEN_FIXTURE_MANIFEST_SHA256,
        "manifest.json SHA-256",
    )
    expected_sidecar = (
        f"{_FROZEN_FIXTURE_MANIFEST_SHA256}  manifest.json\n".encode("ascii")
    )
    _auth_equal(
        snapshots["manifest.json.sha256"].data,
        expected_sidecar,
        "manifest.json.sha256 declaration",
    )
    _auth_equal(
        snapshots["manifest.json.sha256"].sha256,
        _FROZEN_FIXTURE_SIDECAR_PHYSICAL_SHA256,
        "manifest.json.sha256 physical SHA-256",
    )
    for descriptor in descriptors:
        relative = str(descriptor["path"])
        _auth_equal(
            snapshots[relative].sha256,
            descriptor["sha256"],
            f"partition {relative} SHA-256",
        )

    for name, path in paths.items():
        if _read_snapshot(path, name) != snapshots[name]:
            _inventory_auth_fail(f"{name} changed during authentication")

    return {
        "status": "frozen_authenticated_producer_fixture",
        "producer_version": INVENTORY_PRODUCER_VERSION,
        "record_schema_version": INVENTORY_SCHEMA_VERSION,
        "manifest_schema_version": INVENTORY_MANIFEST_SCHEMA_VERSION,
        "fixture_manifest_sha256": _FROZEN_FIXTURE_MANIFEST_SHA256,
        "manifest_sha256_sidecar_physical_sha256": (
            _FROZEN_FIXTURE_SIDECAR_PHYSICAL_SHA256
        ),
        "partition_sha256": {
            str(descriptor["path"]): str(descriptor["sha256"])
            for descriptor in descriptors
        },
        "counts": dict(_FROZEN_INVENTORY_COUNTS),
        "tokenizer_binding_sha256": _FROZEN_TOKENIZER_BINDING["sha256"],
        "inventory_mode": "synthetic_fixture",
        "target_model_tokenizer_match": "not_applicable",
        "gemma_target_identity_verified": False,
        "partition_bytes_authenticated": True,
        "scalar_inventory_records_parsed": True,
        "content_bodies_materialized": False,
        "synthetic_tokenizer_is_gemma": False,
        "provider_requests": 0,
    }


def estimate_bucket(config: Mapping[str, Any], bucket: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate quantized KV tensor payload bytes for one context bucket."""

    architecture = _mapping(config["architecture"], "architecture")
    runtime = _mapping(config["llama_cpp_runtime"], "llama_cpp_runtime")
    model = _mapping(config["model"], "model")
    context_tokens = _positive_int(bucket.get("tokens"), "bucket.tokens")
    bucket_id = str(bucket.get("id"))

    padded_swa = _round_up(
        int(architecture["sliding_window_tokens"]) * int(runtime["parallel_slots"])
        + int(runtime["ubatch_tokens"]),
        int(runtime["swa_alignment_tokens"]),
    )
    swa_cells = min(context_tokens, padded_swa)
    sliding_elements = (
        int(architecture["sliding_layers"])
        * int(architecture["sliding_kv_heads"])
        * int(architecture["sliding_head_dim"])
        * swa_cells
    )
    global_elements = (
        int(architecture["full_attention_layers"])
        * int(architecture["global_kv_heads"])
        * int(architecture["global_head_dim"])
        * context_tokens
    )
    cache_k = _mapping(runtime["cache_k"], "cache_k")
    cache_v = _mapping(runtime["cache_v"], "cache_v")
    sliding_k_bytes = _quantized_bytes(sliding_elements, cache_k, "cache_k")
    sliding_v_bytes = _quantized_bytes(sliding_elements, cache_v, "cache_v")
    global_k_bytes = _quantized_bytes(global_elements, cache_k, "cache_k")
    global_v_bytes = _quantized_bytes(global_elements, cache_v, "cache_v")
    k_bytes = sliding_k_bytes + global_k_bytes
    v_bytes = sliding_v_bytes + global_v_bytes
    total_bytes = k_bytes + v_bytes
    native_context = int(model["native_context_tokens"])
    within_native = context_tokens <= native_context

    blockers = ["runtime_not_validated", "quality_not_validated"]
    claim_status = "native_metadata_only"
    if not within_native:
        claim_status = "research_only_blocked"
        blockers = [
            "exceeds_native_context_metadata",
            "rope_extrapolation_scaling_unbound",
            *blockers,
        ]

    return {
        "bucket": bucket_id,
        "context_tokens": context_tokens,
        "within_native_context_metadata": within_native,
        "claim_status": claim_status,
        "launch_allowed": False,
        "training_allowed": False,
        "runtime_validated": False,
        "quality_validated": False,
        "swa_cache_cells": swa_cells,
        "global_cache_cells": context_tokens,
        "sliding_elements_per_k_or_v": sliding_elements,
        "global_elements_per_k_or_v": global_elements,
        "cache_k": {
            "type": cache_k["type"],
            "sliding_payload_bytes": sliding_k_bytes,
            "global_payload_bytes": global_k_bytes,
            "tensor_payload_bytes": k_bytes,
        },
        "cache_v": {
            "type": cache_v["type"],
            "sliding_payload_bytes": sliding_v_bytes,
            "global_payload_bytes": global_v_bytes,
            "tensor_payload_bytes": v_bytes,
        },
        "kv_tensor_payload_bytes": total_bytes,
        "kv_tensor_payload_mib": round(total_bytes / 2**20, 3),
        "kv_tensor_payload_gib": round(total_bytes / 2**30, 6),
        "runtime_allocation_measured": False,
        "blockers": blockers,
    }


def producer_token_inventory_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen, content-free scalar producer handoff contract."""

    inventory = _mapping(
        config["producer_token_inventory_contract"],
        "producer_token_inventory_contract",
    )
    fixture = _mapping(inventory["fixture"], "producer inventory fixture")
    return {
        "producer_version": inventory["producer_version"],
        "record_schema_version": inventory["record_schema_version"],
        "manifest_schema_version": inventory["manifest_schema_version"],
        "interface_status": inventory["interface_status"],
        "frozen_producer_schema_claimed": inventory["frozen_producer_schema_claimed"],
        "materialization": inventory["materialization"],
        "split_group_key": inventory["split_group_key"],
        "split_before_augmentation": inventory["split_before_augmentation"],
        "producer_release": {
            field: inventory["producer_release"][field]
            for field in (
                "branch",
                "commit",
                "config_sha256",
                "record_schema_sha256",
                "manifest_schema_sha256",
                "source_projector_manifest_sha256",
            )
        },
        "tokenizer_binding": dict(inventory["tokenizer_binding"]),
        "fixture": {
            "path": fixture["path"],
            "manifest_sha256": fixture["manifest_sha256"],
            "manifest_sha256_sidecar_physical_sha256": fixture[
                "manifest_sha256_sidecar_physical_sha256"
            ],
            "partitions": [dict(item) for item in fixture["partitions"]],
            "counts": dict(fixture["counts"]),
        },
        "required_fields": list(inventory["required_fields"]),
        "prohibited_body_fields": list(inventory["prohibited_body_fields"]),
        "invariants": dict(inventory["invariants"]),
        "content_bodies_read_by_preflight": False,
        "producer_fixture_authenticated_by_default_preflight": False,
        "fixture_authentication_entrypoint": (
            "authenticate_producer_token_inventory_fixture"
        ),
        "synthetic_tokenizer_is_gemma": False,
    }


def build_report(
    config: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
    producer_inventory_authentication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic static report for every configured bucket."""

    validate_config(config)
    model = _mapping(config["model"], "model")
    architecture = _mapping(config["architecture"], "architecture")
    rope = _mapping(config["rope"], "rope")
    runtime = _mapping(config["llama_cpp_runtime"], "llama_cpp_runtime")
    buckets = [estimate_bucket(config, item) for item in config["context_buckets"]]
    claims = dict(config["claims"])
    status = "static_preflight_complete"
    claim_scope = "metadata_and_integer_kv_capacity_math_only"
    if producer_inventory_authentication is not None:
        authentication = _mapping(
            producer_inventory_authentication,
            "producer_inventory_authentication",
        )
        _require_equal(
            authentication.get("status"),
            "frozen_authenticated_producer_fixture",
            "producer_inventory_authentication.status",
        )
        _require_equal(
            authentication.get("content_bodies_materialized"),
            False,
            "producer_inventory_authentication.content_bodies_materialized",
        )
        _require_equal(
            authentication.get("synthetic_tokenizer_is_gemma"),
            False,
            "producer_inventory_authentication.synthetic_tokenizer_is_gemma",
        )
        claims["data_jsonl_read"] = True
        status = "static_preflight_complete_with_authenticated_producer_fixture"
        claim_scope = (
            "metadata_integer_kv_math_and_authenticated_scalar_inventory_only"
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "claim_scope": claim_scope,
        "config_sha256": config_sha256,
        "model_facts": {
            "architecture": model["architecture"],
            "scale": model["scale"],
            "model_id": model["model_id"],
            "revision": model["revision"],
            "source_config_sha256": model["source_config_sha256"],
            "native_context_tokens": model["native_context_tokens"],
            "text_layers": architecture["text_layers"],
            "sliding_layers": architecture["sliding_layers"],
            "full_attention_layers": architecture["full_attention_layers"],
            "sliding_window_tokens": architecture["sliding_window_tokens"],
            "full_attention_rope": dict(rope["full_attention"]),
            "sliding_attention_rope": dict(rope["sliding_attention"]),
        },
        "runtime_math": {
            "llama_cpp_commit": runtime["commit"],
            "gemma4_source_sha256": runtime["gemma4_source_sha256"],
            "flash_attention": runtime["flash_attention"],
            "conservative_separate_k_and_v": True,
            "cache_k_type": runtime["cache_k"]["type"],
            "cache_v_type": runtime["cache_v"]["type"],
            "ubatch_tokens": runtime["ubatch_tokens"],
            "swa_cache_cells": _round_up(
                int(architecture["sliding_window_tokens"])
                * int(runtime["parallel_slots"])
                + int(runtime["ubatch_tokens"]),
                int(runtime["swa_alignment_tokens"]),
            ),
        },
        "buckets": buckets,
        "staged_research_plan": list(config["staged_research_plan"]),
        "producer_token_inventory_contract": producer_token_inventory_contract(config),
        "producer_inventory_authentication": (
            None
            if producer_inventory_authentication is None
            else dict(producer_inventory_authentication)
        ),
        "claims": claims,
        "non_claims": [
            "configured_context_is_not_a_quality_result",
            "native_metadata_is_not_a_retrieval_or_reasoning_result",
            "one_million_context_support_is_not_claimed",
            "no_runtime_memory_or_throughput_measurement",
            "kv_tensor_payload_is_not_complete_runtime_allocation",
            "no_training_readiness_or_training_quality_claim",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Static Gemma 4 12B long-context/KV preflight; the default mode "
            "uses no model or data, while explicit fixture authentication "
            "reads only frozen body-free scalar metadata."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--authenticate-producer-fixture",
        action="store_true",
        help=(
            "Authenticate the frozen scalar producer fixture at its configured "
            "repository-relative path."
        ),
    )
    parser.add_argument(
        "--producer-inventory-root",
        type=Path,
        help=(
            "Authenticate an explicit producer fixture root; implies "
            "--authenticate-producer-fixture."
        ),
    )
    parser.add_argument(
        "--producer-contract-root",
        type=Path,
        help=(
            "Repository root containing the three frozen producer contract "
            "files; implies --authenticate-producer-fixture."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    config, config_sha256 = load_config(args.config)
    authentication = None
    authenticate_fixture = (
        args.authenticate_producer_fixture
        or args.producer_inventory_root is not None
        or args.producer_contract_root is not None
    )
    if authenticate_fixture:
        config_path = args.config.expanduser().resolve()
        try:
            default_repository_root = config_path.parents[2]
        except IndexError as exc:
            raise LongContextPreflightError(
                "cannot resolve the producer repository root from config path"
            ) from exc
        contract_root = (
            default_repository_root
            if args.producer_contract_root is None
            else args.producer_contract_root
        )
        contract_authentication = (
            authenticate_producer_token_inventory_contract_files(
                config,
                contract_root,
            )
        )
        inventory_root = args.producer_inventory_root
        if inventory_root is None:
            fixture = _mapping(
                config["producer_token_inventory_contract"]["fixture"],
                "producer inventory fixture",
            )
            inventory_root = default_repository_root.joinpath(
                *str(fixture["path"]).split("/")
            )
        authentication = authenticate_producer_token_inventory_fixture(
            config,
            inventory_root,
        )
        authentication["contract_files"] = contract_authentication
    report = build_report(
        config,
        config_sha256=config_sha256,
        producer_inventory_authentication=authentication,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
