from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from anchor_mvp.research import gemma3_qonly_parameter_budget as budget


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
LOCAL_EXPORT = (
    WORKSPACE_ROOT
    / "models"
    / "google-gemma-3-1b-it-keras-v3"
    / "hf-export-keras-hub-0.29.1-bf16"
)


def _load_contract() -> dict[str, object]:
    return json.loads((REPO_ROOT / budget.CONTRACT_RELATIVE).read_text("utf-8"))


def test_contract_audit_passes_without_model_or_gpu() -> None:
    result = budget.audit_contract(REPO_ROOT)
    assert result == {
        "status": "metadata_budget_ready_real_run_blocked",
        "contract_sha256": (
            "2224a1c00c28420a66f4a27a685394db346b34be0fd50509f3b2f1e91cdc25ca"
        ),
        "sidecar_sha256": (
            "6a76b33e0388e3dda131b3adf1a5421185dfbd7d6a18024247d1c5e540c0b3aa"
        ),
        "schema_sha256": (
            "fc3db1b10eccce0cf265bf23f7b819f437efc7247d98ce4b594937e97255d6b1"
        ),
        "base_parameters": 999_885_952,
        "rank_count": 10,
        "training_authorized": False,
        "model_loads": 0,
        "gpu_operations": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "weight_tensor_reads": 0,
    }


def test_rank_formula_and_requested_ladder_are_exact() -> None:
    contract = _load_contract()
    rows = contract["rank_table"]
    assert isinstance(rows, list)
    assert [row["rank"] for row in rows] == list(budget.EXPECTED_RANKS)
    for row in rows:
        expected = budget.calculate_rank_row(row["rank"])
        assert {key: row[key] for key in expected} == expected


def test_base_parameter_parity_is_explicitly_infeasible() -> None:
    contract = _load_contract()
    math = contract["parameter_math"]
    rows = {row["rank"]: row for row in contract["rank_table"]}

    assert budget.DENSE_Q_PER_EXPERT_PARAMS > (budget.PER_EXPERT_PARAMS_PER_RANK * 542)
    assert budget.DENSE_Q_PER_EXPERT_PARAMS < (budget.PER_EXPERT_PARAMS_PER_RANK * 543)
    assert math["effective_rank_ceiling"] == 1024
    assert rows[3535]["rank"] > math["effective_rank_ceiling"]
    assert rows[3535]["per_expert_params"] > math["dense_q_per_expert_params"]
    assert rows[3535]["classification"] == ("infeasible_for_base_parameter_parity")
    assert (
        rows[3535]["five_expert_aggregate_params"]
        - contract["architecture"]["base_parameters"]
        == 94_848
    )
    assert contract["claims"]["base_parameter_parity_claimed"] is False


def test_dense_q_and_active_vs_aggregate_budgets_are_not_conflated() -> None:
    contract = _load_contract()
    rows = contract["rank_table"]
    for row in rows:
        assert row["single_route_active_params"] == row["per_expert_params"]
        assert row["five_expert_aggregate_params"] == (
            5 * row["single_route_active_params"]
        )
    fairness = contract["fairness_protocol"]
    assert fairness["primary_comparison_arms"] == [
        "frozen_base",
        "monolithic_lora",
        "five_expert_correct_route",
        "five_expert_wrong_route",
        "five_expert_random_route",
    ]
    assert fairness["diagnostic_overlays"] == ["o_only", "q_plus_o"]
    assert contract["scope"]["default_active_experts_per_request"] == 1
    assert contract["claims"]["compute_equivalence_claimed"] is False
    assert contract["claims"]["vram_equivalence_claimed"] is False


def test_token_identity_mismatch_and_real_run_gates_fail_closed() -> None:
    contract = _load_contract()
    tokens = contract["token_identity_audit"]
    assert tokens["chat_template_bound"] is False
    assert tokens["official_chat_template_sha256"] is None
    assert tokens["config_token_ids"] == {"pad": 0, "bos": 1, "eos": 2}
    assert tokens["tokenizer_added_token_ids"] == {"pad": 0, "bos": 2, "eos": 1}
    assert tokens["token_id_binding_consistent"] is False
    assert contract["source_export"]["model_identity_bound"] is False
    assert contract["source_export"]["official_model_id"] is None
    assert contract["source_export"]["source_revision"] is None
    assert contract["real_run_gates"]["training_authorized"] is False
    assert contract["claims"]["training_authorized"] is False
    assert contract["claims"]["formal"] is False


def test_sidecar_is_mandatory_and_fail_closed(tmp_path: Path) -> None:
    for relative in (
        budget.CONTRACT_RELATIVE,
        budget.SCHEMA_RELATIVE,
        budget.SIDECAR_RELATIVE,
    ):
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    (tmp_path / budget.SIDECAR_RELATIVE).write_text(
        f"{'0' * 64}  {Path(budget.CONTRACT_RELATIVE).name}\n",
        encoding="ascii",
    )
    with pytest.raises(
        budget.BudgetContractError,
        match="mandatory contract SHA-256 sidecar is invalid",
    ):
        budget.audit_contract(tmp_path)


def test_rank_rejects_non_positive_and_boolean_values() -> None:
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            budget.calculate_rank_row(value)


@pytest.mark.skipif(
    not LOCAL_EXPORT.exists(),
    reason="local Gemma export is not present in this checkout",
)
def test_local_export_metadata_matches_contract_without_reading_weights() -> None:
    contract = _load_contract()
    bindings = {item["role"]: item for item in contract["source_export"]["files"]}
    for role in ("config", "export_manifest", "tokenizer_config", "tokenizer_model"):
        binding = bindings[role]
        raw = (LOCAL_EXPORT / binding["path"]).read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]

    config = json.loads((LOCAL_EXPORT / "config.json").read_text("utf-8"))
    manifest = json.loads((LOCAL_EXPORT / "EXPORT_MANIFEST.json").read_text("utf-8"))
    tokenizer = json.loads((LOCAL_EXPORT / "tokenizer_config.json").read_text("utf-8"))
    declared = {entry["path"]: entry for entry in manifest["files"]}

    assert config["num_hidden_layers"] == 26
    assert config["hidden_size"] == 1152
    assert config["num_attention_heads"] * config["head_dim"] == 1024
    assert manifest["parameter_count"] == 999_885_952
    assert manifest["chat_template_bound"] is False
    assert manifest["training_authorized"] is False
    assert tokenizer["added_tokens_decoder"]["1"]["content"] == "<eos>"
    assert tokenizer["added_tokens_decoder"]["2"]["content"] == "<bos>"

    weights = bindings["model_weights"]
    assert declared["model.safetensors"]["bytes"] == weights["bytes"]
    assert declared["model.safetensors"]["sha256"] == weights["sha256"]
    assert (LOCAL_EXPORT / "model.safetensors").stat().st_size == weights["bytes"]
