from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from anchor_mvp.research.neural_swarm_hierarchical_planner_rdma_v1 import (
    PlannerRDMAContractError,
    load_and_validate,
    validate_contract,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/research/neural_swarm_hierarchical_planner_rdma_v1.json"
SCHEMA = REPO / "configs/research/neural_swarm_hierarchical_planner_rdma_v1.schema.json"


def _contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_schema_and_cross_field_validator_pass() -> None:
    contract = _contract()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(contract)
    result = load_and_validate(CONFIG)
    assert result["status"] == "passed"
    assert result["selected_output_expert_count"] == 1
    assert result["rdma_physical_verified"] is False
    assert result["zero_copy_claimed"] is False
    assert result["records_added_to_current_4300_final"] == 0


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (
            ("routing", "selected_output_expert_count"),
            2,
            "single_output_expert_invariant_failed",
        ),
        (
            ("activation_alignment", "hidden_chain_of_thought"),
            True,
            "hidden_chain_of_thought_forbidden",
        ),
        (
            ("task_decomposition", "hash_index_or_case_salt_allowed"),
            True,
            "semantic_salt_forbidden",
        ),
        (
            ("cache_lineage", "shared_storage"),
            True,
            "cache_shared_storage_must_be_false",
        ),
        (
            (
                "cache_lineage",
                "same_cache_object_implies_same_storage_or_data_ptr",
            ),
            True,
            "cache_same_cache_object_implies_same_storage_or_data_ptr_must_be_false",
        ),
        (
            ("selected_group_transport", "rdma_write_allowed"),
            True,
            "transport_rdma_write_allowed_must_be_false",
        ),
        (
            ("selected_group_transport", "rdma_physical_verified"),
            True,
            "transport_rdma_physical_verified_must_be_false",
        ),
        (
            ("serial_alignment", "records_added_to_current_4300_final"),
            1,
            "current_4300_final_must_remain_unchanged",
        ),
    ],
)
def test_red_zone_mutations_fail_closed(
    path: tuple[str, str],
    value: object,
    error: str,
) -> None:
    contract = copy.deepcopy(_contract())
    contract[path[0]][path[1]] = value
    with pytest.raises(PlannerRDMAContractError, match=error):
        validate_contract(contract)


def test_rdma_is_transport_not_correctness_and_private_kv_is_not_exported() -> None:
    contract = _contract()
    assert contract["memory_red_zone"]["rdma_required_for_correctness"] is False
    transport = contract["selected_group_transport"]
    assert transport["current_mode"] == "local_copy_fallback"
    assert transport["live_private_kv_remote_transfer"] is False
    assert transport["uncommitted_or_future_read_allowed"] is False
    assert transport["transport_changes_semantics"] is False
    assert transport["fallback_on_missing_capability"] == "local_copy_or_fail_closed"


def test_memory_docs_record_the_frozen_architecture() -> None:
    for relative in (
        "docs/rfcs/neural_swarm_hierarchical_planner_rdma.md",
        "docs/rfcs/neural_swarm_hierarchical_planner_rdma.zh-CN.md",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        for marker in (
            "P0",
            "P1",
            "P2",
            "P3",
            "P4",
            "RDMA",
            "256",
            "single output expert",
            "shared_storage=false",
        ):
            assert marker in text
