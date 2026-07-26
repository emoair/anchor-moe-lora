"""Model-free validator for the hierarchical planner/RDMA architecture contract.

The module intentionally validates control-plane identities and claim boundaries
only.  It does not allocate tensors, load a model, open an RDMA device, or grant
training authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "anchor.neural-swarm-hierarchical-planner-rdma.v1"
EXPECTED_SPECIALISTS = ("mode_planner", "tool_planner", "emotion_planner")
EXPECTED_OUTPUT_EXPERTS = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
    "review_audit",
)
EXPECTED_STAGES = (
    "P0_adapter_off_frozen_base_prefix",
    "P1_general_planner_commit",
    "P2_selected_specialist_commits_topological_order",
    "P3_aligned_activation_group_commit",
    "P4_single_output_expert_private_decode",
)
REQUIRED_RDMA_PROOFS = (
    "registered_memory_region",
    "read_only_remote_key_scope",
    "owner_and_reader_device_identity",
    "generation_epoch_and_lease",
    "content_digest_before_and_after_read",
    "pointer_offset_stride_and_layout",
    "token_order_position_mask_and_rope",
    "model_tokenizer_template_and_adapter_lineage",
    "completion_fence_and_lifetime",
    "stale_read_and_revoked_lease_rejection",
    "22_sliding_window_plus_4_full_attention_semantics",
)
FALSE_CLAIMS = (
    "data_final_modified",
    "training_authorized",
    "formal",
    "quality_validated",
    "rdma_implemented",
    "rdma_quality_gain_claimed",
    "rdma_latency_gain_claimed",
    "zero_copy_claimed",
    "shared_storage_claimed",
    "hidden_chain_of_thought_claimed",
)


class PlannerRDMAContractError(ValueError):
    """Raised when a frozen architecture red-line is violated."""


def _fail(code: str) -> None:
    raise PlannerRDMAContractError(code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def canonical_sha256(value: Any) -> str:
    """Hash canonical compact JSON bytes without Unicode normalization."""

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate cross-field invariants omitted from the JSON Schema."""

    if contract.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version_mismatch")
    if contract.get("status") != "contract_only_model_free_rdma_unverified":
        _fail("status_must_remain_contract_only")

    red_zone = _mapping(contract.get("memory_red_zone"), "memory_red_zone_invalid")
    if red_zone.get("immutable") is not True:
        _fail("memory_red_zone_not_immutable")
    if red_zone.get("source_final_mutated") is not False:
        _fail("current_4300_final_must_not_be_mutated")
    if red_zone.get("final_output_expert_count") != 1:
        _fail("exactly_one_output_expert_required")
    for key in (
        "visible_chain_of_thought",
        "full_generation_kv_shared",
        "rdma_required_for_correctness",
    ):
        if red_zone.get(key) is not False:
            _fail(f"red_zone_{key}_must_be_false")

    decomposition = _mapping(
        contract.get("task_decomposition"),
        "task_decomposition_invalid",
    )
    if decomposition.get("cycles_allowed") is not False:
        _fail("subtask_dag_cycles_forbidden")
    if decomposition.get("hash_index_or_case_salt_allowed") is not False:
        _fail("semantic_salt_forbidden")
    if decomposition.get("max_nodes", 0) < decomposition.get("max_depth", 0):
        _fail("subtask_dag_depth_exceeds_node_budget")
    required_semantics = tuple(
        _sequence(
            decomposition.get("required_semantic_fields"),
            "semantic_fields_invalid",
        )
    )
    if required_semantics != (
        "operation",
        "operands",
        "constraints",
        "dependencies",
        "expected_relation",
    ):
        _fail("semantic_fields_mismatch")

    routing = _mapping(contract.get("routing"), "routing_invalid")
    if routing.get("general_planner_count") != 1:
        _fail("exactly_one_general_planner_required")
    if tuple(routing.get("specialist_planners", ())) != EXPECTED_SPECIALISTS:
        _fail("specialist_planner_inventory_mismatch")
    selected_min = routing.get("selected_specialist_min")
    selected_max = routing.get("selected_specialist_max")
    if selected_min != 1 or selected_max != len(EXPECTED_SPECIALISTS):
        _fail("selected_specialist_bounds_mismatch")
    if routing.get("one_planner_per_subtask") is not True:
        _fail("one_planner_per_subtask_required")
    if tuple(routing.get("output_experts", ())) != EXPECTED_OUTPUT_EXPERTS:
        _fail("output_expert_inventory_mismatch")
    if routing.get("selected_output_expert_count") != 1:
        _fail("single_output_expert_invariant_failed")
    if routing.get("unselected_expert_visibility") != "none":
        _fail("unselected_expert_visibility_forbidden")
    if routing.get("confidence_margin_required") is not True:
        _fail("route_confidence_margin_required")

    alignment = _mapping(
        contract.get("activation_alignment"),
        "activation_alignment_invalid",
    )
    if alignment.get("packet_dimension") != 256:
        _fail("latent_packet_dimension_mismatch")
    if alignment.get("serialization") != "canonical_f32_le_v1":
        _fail("latent_packet_serialization_mismatch")
    if alignment.get("position_binding") != "explicit_absolute_position_v1":
        _fail("latent_packet_position_binding_mismatch")
    if alignment.get("commit_digest") != "sha256":
        _fail("latent_packet_commit_digest_mismatch")
    if alignment.get("recomputable_boundary") is not True:
        _fail("latent_packet_boundary_not_recomputable")
    if alignment.get("hidden_chain_of_thought") is not False:
        _fail("hidden_chain_of_thought_forbidden")
    if alignment.get("threshold_status") != "pending_calibration":
        _fail("alignment_threshold_must_not_be_silently_guessed")

    cache = _mapping(contract.get("cache_lineage"), "cache_lineage_invalid")
    if tuple(cache.get("ordered_stages", ())) != EXPECTED_STAGES:
        _fail("cache_stage_order_mismatch")
    if cache.get("current_backend") != "hf_dynamic_cache":
        _fail("current_cache_backend_mismatch")
    if (
        cache.get("current_claim")
        != "exact_ordered_prefix_value_and_prefill_compute_handoff"
    ):
        _fail("current_cache_claim_overstated")
    for key in (
        "new_adapter_prefix_recomputation_equivalent",
        "same_cache_object_implies_same_storage_or_data_ptr",
        "shared_storage",
        "zero_copy",
        "planner_private_tail_transfer",
    ):
        if cache.get(key) is not False:
            _fail(f"cache_{key}_must_be_false")
    if cache.get("output_expert_tail") != "private_append_only":
        _fail("output_expert_tail_must_be_private")
    hybrid = _mapping(
        cache.get("hybrid_attention_layers"),
        "hybrid_attention_layers_invalid",
    )
    if (
        hybrid.get("sliding_window_layers"),
        hybrid.get("sliding_window_size"),
        hybrid.get("full_attention_layers"),
    ) != (22, 512, 4):
        _fail("gemma3_hybrid_attention_identity_mismatch")

    transport = _mapping(
        contract.get("selected_group_transport"),
        "selected_group_transport_invalid",
    )
    if transport.get("current_mode") != "local_copy_fallback":
        _fail("unverified_rdma_cannot_be_current_mode")
    if transport.get("shared_payload") != (
        "committed_content_addressed_activation_and_prefix_pages"
    ):
        _fail("rdma_payload_must_be_committed_and_content_addressed")
    for key in (
        "live_private_kv_remote_transfer",
        "rdma_write_allowed",
        "uncommitted_or_future_read_allowed",
        "transport_changes_semantics",
        "rdma_physical_verified",
    ):
        if transport.get(key) is not False:
            _fail(f"transport_{key}_must_be_false")
    if tuple(transport.get("required_rdma_proofs", ())) != REQUIRED_RDMA_PROOFS:
        _fail("rdma_proof_inventory_mismatch")

    serial = _mapping(contract.get("serial_alignment"), "serial_alignment_invalid")
    if serial.get("downstream_requires_all_upstream_frozen") is not True:
        _fail("serial_alignment_requires_frozen_upstream")
    if serial.get("selected_group_executes_in_dag_topological_order") is not True:
        _fail("selected_group_must_follow_dag_order")
    if serial.get("records_added_to_current_4300_final") != 0:
        _fail("current_4300_final_must_remain_unchanged")

    claims = _mapping(contract.get("claims"), "claims_invalid")
    for key in FALSE_CLAIMS:
        if claims.get(key) is not False:
            _fail(f"claim_{key}_must_be_false")

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(contract),
        "status": "passed",
        "selected_output_expert_count": 1,
        "rdma_physical_verified": False,
        "zero_copy_claimed": False,
        "records_added_to_current_4300_final": 0,
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("config_bom_forbidden")
    if b"\r\n" in raw:
        _fail("config_crlf_forbidden")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlannerRDMAContractError("config_parse_failed") from exc
    if not isinstance(value, Mapping):
        _fail("config_root_must_be_object")
    return validate_contract(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = load_and_validate(args.config)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
