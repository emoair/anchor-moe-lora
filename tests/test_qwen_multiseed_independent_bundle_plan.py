from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml

from anchor_mvp.research import qwen_multiseed_independent_bundle_plan as planner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / planner.CONFIG_PATH


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def plan() -> dict[str, object]:
    return planner.build_dry_run_plan(planner.CONFIG_PATH)


def test_config_identity_and_closed_validation() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == planner.CONFIG_SHA256
    config = _config()
    planner._validate_config(config)

    promoted = copy.deepcopy(config)
    promoted["gates"]["execution_ready"] = True
    with pytest.raises(planner.MultiSeedPlanError, match="gates_invalid"):
        planner._validate_config(promoted)

    extended = copy.deepcopy(config)
    extended["evaluation_matrix"]["unknown"] = False
    with pytest.raises(planner.MultiSeedPlanError, match="eval_matrix_invalid"):
        planner._validate_config(extended)


def test_producer_seed_schedules_are_exact() -> None:
    expected = {
        "adapter_init": [
            1485749021,
            1905652195,
            538954614,
            1705069177,
            1764718012,
        ],
        "record_order": [
            638502245,
            930319862,
            283374371,
            2107737362,
            1730343274,
        ],
        "cuda": [
            794927935,
            651012111,
            179390722,
            673891872,
            1806964459,
        ],
    }
    for domain, values in expected.items():
        assert [
            planner._derived_seed(domain, seed) for seed in planner.MASTER_SEEDS
        ] == values


def test_two_tracks_have_exact_internal_budgets_and_no_cross_track_claim() -> None:
    replication = _config()["replication"]
    assert replication["cross_track_comparison_allowed"] is False
    assert replication["retained_o_branch_may_be_relabelled_o_only"] is False
    tracks = replication["tracks"]
    assert (
        tracks["discovery_replication"]["common_budget_trainable_parameters"]
        == 1_376_256
    )
    assert (
        tracks["mechanism_controls"]["common_budget_trainable_parameters"] == 1_204_224
    )
    assert [arm["id"] for arm in tracks["mechanism_controls"]["arms"]] == list(
        planner.MECHANISM_ARMS
    )
    assert tracks["mechanism_controls"]["arms"][3] == {
        "id": "o_only",
        "independently_trained": True,
        "rank_policy": {"o_proj": 14},
        "expected_trainable_parameters": 1_204_224,
        "expected_trainable_tensors": 56,
    }
    assert tracks["mechanism_controls"]["arms"][4] == {
        "id": "k_plus_v",
        "independently_trained": True,
        "rank_policy": {"k_proj": 12, "v_proj": 12},
        "expected_trainable_parameters": 1_204_224,
        "expected_trainable_tensors": 112,
    }
    for track in tracks.values():
        for arm in track["arms"]:
            assert (
                arm["expected_trainable_parameters"]
                == track["common_budget_trainable_parameters"]
            )
            assert arm["expected_trainable_tensors"] == 56 * len(arm["rank_policy"])


def test_each_seed_has_six_orders_and_global_mechanism_balance() -> None:
    discovery = planner._discovery_orders()
    planner._validate_orders(discovery, planner.DISCOVERY_ARMS)
    assert len({tuple(order) for order in discovery}) == 6

    seed_plans = []
    for index, seed in enumerate(planner.MASTER_SEEDS):
        mechanism = planner._mechanism_orders(seed, index)
        planner._validate_orders(mechanism, planner.MECHANISM_ARMS)
        seed_plans.append({"mechanism_controls_throughput_arm_orders": mechanism})
    planner._validate_global_order_balance(seed_plans)
    for position in range(5):
        counts = {arm: 0 for arm in planner.MECHANISM_ARMS}
        for seed_plan in seed_plans:
            for order in seed_plan["mechanism_controls_throughput_arm_orders"]:
                counts[order[position]] += 1
        assert set(counts.values()) == {6}


def test_controlled_factorial_quota_table_is_frozen_and_balanced() -> None:
    table = planner._factor_quota_table(_config())
    assert len(table) == 10
    assert planner._compact_json_sha256(table) == (
        "6b1a08f878cb973705b20b2e997deb9cc157b39743ba456070b04858d0221453"
    )
    train = {
        factor: sum(row["train"][factor] for row in table)
        for factor in planner.EVAL_CELLS
    }
    evaluation = {
        factor: sum(row["eval_proxy"][factor] for row in table)
        for factor in planner.EVAL_CELLS
    }
    assert list(train.values()) == [13, 14, 13]
    assert list(evaluation.values()) == [7, 6, 7]
    assert all(sum(row["train"].values()) == 4 for row in table)
    assert all(sum(row["eval_proxy"].values()) == 2 for row in table)


def test_factor_truth_table_requires_overlap_only_where_intended() -> None:
    cells = _config()["evaluation_matrix"]["cells"]
    old_task_new_template, new_task_old_template, new_task_new_template = cells
    assert old_task_new_template["old_task_membership_required"] is True
    assert old_task_new_template["new_template_nonoverlap_required"] is True
    assert new_task_old_template["new_task_nonoverlap_required"] is True
    assert new_task_old_template["old_template_membership_required"] is True
    assert new_task_new_template["new_task_nonoverlap_required"] is True
    assert new_task_new_template["new_template_nonoverlap_required"] is True
    assert all(cell["task_template_pair_nonoverlap_required"] for cell in cells)


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("confirmation_dataset", "roles", 6, "dataset_invalid"),
        ("confirmation_dataset", "paired_variants", 1, "dataset_invalid"),
        (
            "confirmation_dataset",
            "information_flow_strata",
            [
                "prefix_evidence_selection",
                "prefix_evidence_selection",
                "conflicting_allowed_evidence_resolution",
                "tool_result_commit_then_expert_private_tail",
                "ordered_long_prefix_retrieval",
            ],
            "dataset_invalid",
        ),
        (
            "confirmation_dataset",
            "split_before_role_variant_augmentation",
            False,
            "dataset_invalid",
        ),
        ("confirmation_dataset", "eval_proxy_is_heldout", True, "dataset_invalid"),
        (
            "evaluation_matrix",
            "global_task_nonoverlap_required",
            True,
            "eval_matrix_invalid",
        ),
        (
            "evaluation_matrix",
            "global_task_template_pair_nonoverlap_required",
            False,
            "eval_matrix_invalid",
        ),
        (
            "evaluation_matrix",
            "eval_factor_quotas",
            [6, 7, 7],
            "eval_matrix_invalid",
        ),
    ],
)
def test_dataset_and_factorial_drift_fail_closed(
    section: str, key: str, value: object, error: str
) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = value
    with pytest.raises(planner.MultiSeedPlanError, match=error):
        planner._validate_config(config)


def test_gate_split_fairness_and_performance_are_closed() -> None:
    config = _config()
    boundaries = config["study_boundaries"]
    assert boundaries["producer_independent_confirmation"] == {
        "status": "blocked_missing_confirmation_inventory_and_zero_overlap_proof",
        "required_for_bundle_generalization": True,
        "satisfied_by_secondary_controlled_factorial_probe": False,
        "independent_confirmation_validated": False,
        "bundle_generalization_validated": False,
    }
    assert (
        boundaries["secondary_controlled_factorial_probe"][
            "may_satisfy_independent_confirmation"
        ]
        is False
    )
    assert (
        boundaries["secondary_controlled_factorial_probe"][
            "may_satisfy_bundle_generalization"
        ]
        is False
    )
    fairness = config["execution_fairness"]
    assert fairness["optimizer"]["learning_rate"] == 0.00005
    assert fairness["optimizer"]["betas"] == [0.9, 0.999]
    assert fairness["optimizer"]["foreach"] is False
    assert fairness["optimizer"]["fused"] is False
    assert fairness["gradient_checkpointing"] == {
        "enabled": True,
        "use_reentrant": False,
        "use_cache": False,
    }
    assert fairness["reset_policy"]["resume_allowed"] is False
    assert fairness["determinism"] == {
        "torch_use_deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    performance = config["performance_contract"]
    assert performance["arm_order_scope"] == "throughput_only_not_training_repetition"
    assert performance["max_concurrent_gpu_jobs"] == 1
    assert performance["peak_vram_cap_bytes"] == 5 * 1024**3
    assert all(value is False for value in config["claims"].values())

    for section, key, value, error in (
        ("execution_fairness", "optimizer_steps", 79, "execution_fairness_invalid"),
        (
            "performance_contract",
            "max_concurrent_gpu_jobs",
            2,
            "performance_contract_invalid",
        ),
    ):
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(planner.MultiSeedPlanError, match=error):
            planner._validate_config(changed)

    changed = copy.deepcopy(config)
    changed["study_boundaries"]["secondary_controlled_factorial_probe"][
        "may_satisfy_bundle_generalization"
    ] = True
    with pytest.raises(planner.MultiSeedPlanError, match="study_boundaries_invalid"):
        planner._validate_config(changed)


def test_producer_contract_drift_is_rejected() -> None:
    import json

    contract_path = ROOT / (
        "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["producer_followup"]["replication_phase"]["checkpoint_steps"] = [80]
    with pytest.raises(planner.MultiSeedPlanError, match="producer_replication_drift"):
        planner._validate_producer_contract(changed, _config())


def test_exact_sidecar_parser_rejects_spacing_or_crlf() -> None:
    digest = "a" * 64
    planner._validate_sidecar(
        digest, f"{digest}  artifact.json\n".encode(), "artifact.json", "bad"
    )
    for invalid in (
        f"{digest} artifact.json\n".encode(),
        f"{digest}  artifact.json\r\n".encode(),
        f"{'b' * 64}  artifact.json\n".encode(),
    ):
        with pytest.raises(planner.MultiSeedPlanError, match="bad"):
            planner._validate_sidecar(digest, invalid, "artifact.json", "bad")


def test_only_canonical_config_path_is_accepted(tmp_path: Path) -> None:
    alternate = tmp_path / "alternate.yaml"
    alternate.write_bytes(CONFIG.read_bytes())
    with pytest.raises(planner.MultiSeedPlanError, match="config_path_invalid"):
        planner._requested_config(alternate)


def test_cli_is_deliberately_non_authorizing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        planner,
        "build_dry_run_plan",
        lambda _: {"status": "blocked", "training_authorized": False},
    )
    assert planner.main(["--config", planner.CONFIG_PATH]) == 2
    assert '"training_authorized": false' in capsys.readouterr().out


def test_authenticated_dry_run_stays_blocked_and_zero_request(
    plan: dict[str, object],
) -> None:
    assert plan["status"] == (
        "blocked_controlled_factorial_confirmation_inputs_unavailable"
    )
    assert plan["gates"]["execution_ready"] is False
    assert plan["gates"]["materialization_ready"] is False
    assert plan["gates"]["training_authorized"] is False
    assert plan["gates"]["formal_training_authorized"] is False
    assert plan["gates"]["controlled_factorial_confirmation_validated"] is False
    assert plan["gates"]["producer_independent_confirmation_validated"] is False
    assert plan["gates"]["bundle_generalization_validated"] is False
    assert (
        plan["study_boundaries"]["secondary_controlled_factorial_probe"][
            "may_satisfy_independent_confirmation"
        ]
        is False
    )
    assert all(value is False for value in plan["claims"].values())
    assert plan["audit"] == {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_content_reads": 0,
        "dataset_body_reads": 0,
        "training_runs": 0,
    }
    replication = plan["replication"]
    assert replication["trainable_arm_ids_by_track"] == {
        "discovery_replication": list(planner.DISCOVERY_ARMS),
        "mechanism_controls": list(planner.MECHANISM_ARMS),
    }
    assert replication["shared_eval_reference_id"] == "adapter_off"
    assert all(
        "arm_order_seed" not in seed_plan for seed_plan in replication["seed_plans"]
    )
    assert replication["planned_independent_training_jobs"] == 40
    assert replication["planned_throughput_order_slots"] == 240
    assert replication["planned_trainable_checkpoint_receipts"] == 200
    assert plan["plan_sha256"] == planner._compact_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_plan_self_bindings_match_physical_files(plan: dict[str, object]) -> None:
    bindings = plan["self_bindings"]
    assert bindings["config"] == {
        "path": planner.CONFIG_PATH,
        "sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    implementation = ROOT / planner.IMPLEMENTATION_PATH
    assert bindings["implementation"] == {
        "path": planner.IMPLEMENTATION_PATH,
        "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
    }


def test_authenticated_consumer_ignores_poisoned_import_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_name = "anchor_mvp.research.qwen_controlled_proxy_risk_evidence_consumer"
    poisoned = ModuleType(canonical_name)
    poisoned_called = False

    def poisoned_evaluator(_: object) -> dict[str, object]:
        nonlocal poisoned_called
        poisoned_called = True
        return {"status": "authorized", "training_authorized": True}

    poisoned.evaluate_risk_evidence = poisoned_evaluator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, canonical_name, poisoned)
    result = planner.build_dry_run_plan(planner.CONFIG_PATH)
    assert result["status"] == (
        "blocked_controlled_factorial_confirmation_inputs_unavailable"
    )
    assert poisoned_called is False


def test_authenticated_consumer_rejects_snapshot_digest_drift() -> None:
    config = _config()
    relative = config["paths"]["risk_consumer_implementation"]
    implementation = ROOT / relative
    snapshot = planner._read_bytes(implementation, "unreadable")
    with pytest.raises(
        planner.MultiSeedPlanError,
        match="risk_consumer_implementation_sha256_mismatch",
    ):
        planner._evaluate_authenticated_risk_consumer(
            snapshot,
            implementation,
            config["paths"]["risk_consumer_config"],
            "0" * 64,
        )


def test_snapshot_detects_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    path.write_bytes(b"{}\n")
    snapshot = planner._read_bytes(path, "unreadable")
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(planner.MultiSeedPlanError, match="changed"):
        planner._assert_unchanged(path, snapshot, "changed")
