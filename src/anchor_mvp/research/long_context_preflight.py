"""Static Gemma 4 12B long-context capacity preflight.

The preflight reads one small YAML file and performs integer KV-cache math. It
does not open a model, dataset, JSONL partition, network connection, or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SCHEMA_VERSION = "anchor.gemma4-12b-long-context-preflight-config.v1"
REPORT_SCHEMA_VERSION = "anchor.gemma4-12b-long-context-preflight-report.v1"
INVENTORY_SCHEMA_VERSION = "anchor.long-context-token-inventory.v1"

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
    "record_id",
    "task_bundle_sha256",
    "split",
    "stage",
    "expert",
    "variant",
    "tokenizer_sha256",
    "segment_plan_sha256",
    "ordered_segment_ids_sha256",
    "terminal_prefix_lineage_sha256",
    "shared_prefix_tokens",
    "committed_downstream_tokens",
    "expert_private_input_tokens",
    "target_tokens",
    "materialized_prompt_tokens",
    "training_sequence_tokens",
    "reserved_output_tokens",
    "total_tokens",
    "required_context_tokens",
    "forbidden_tokens_excluded",
    "context_bucket",
)
_EXPECTED_PROHIBITED_BODY_FIELDS = (
    "messages",
    "prompt",
    "completion",
    "content",
    "task_board",
    "blocks",
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
        inventory.get("schema_version"),
        INVENTORY_SCHEMA_VERSION,
        "producer inventory schema",
    )
    _require_equal(
        inventory.get("interface_status"),
        "requested_pending_producer_freeze",
        "producer inventory interface_status",
    )
    _require_equal(
        inventory.get("frozen_producer_schema_claimed"),
        False,
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
            "materialized_prompt_tokens": (
                "shared_prefix_tokens + committed_downstream_tokens + "
                "expert_private_input_tokens"
            ),
            "training_sequence_tokens": "materialized_prompt_tokens + target_tokens",
            "total_tokens": "training_sequence_tokens",
            "required_context_tokens": (
                "materialized_prompt_tokens + max(target_tokens, reserved_output_tokens)"
            ),
            "forbidden_tokens_excluded": True,
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
    """Return the content-free scalar contract requested from the producer."""

    inventory = _mapping(
        config["producer_token_inventory_contract"],
        "producer_token_inventory_contract",
    )
    return {
        "schema_version": inventory["schema_version"],
        "interface_status": inventory["interface_status"],
        "frozen_producer_schema_claimed": inventory["frozen_producer_schema_claimed"],
        "materialization": inventory["materialization"],
        "split_group_key": inventory["split_group_key"],
        "split_before_augmentation": inventory["split_before_augmentation"],
        "required_fields": list(inventory["required_fields"]),
        "prohibited_body_fields": list(inventory["prohibited_body_fields"]),
        "invariants": dict(inventory["invariants"]),
        "content_bodies_read_by_preflight": False,
    }


def build_report(
    config: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic static report for every configured bucket."""

    validate_config(config)
    model = _mapping(config["model"], "model")
    architecture = _mapping(config["architecture"], "architecture")
    rope = _mapping(config["rope"], "rope")
    runtime = _mapping(config["llama_cpp_runtime"], "llama_cpp_runtime")
    buckets = [estimate_bucket(config, item) for item in config["context_buckets"]]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "static_preflight_complete",
        "claim_scope": "metadata_and_integer_kv_capacity_math_only",
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
        "claims": dict(config["claims"]),
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
        description="Static Gemma 4 12B long-context/KV preflight (no model or data)."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    config, config_sha256 = load_config(args.config)
    report = build_report(config, config_sha256=config_sha256)
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
