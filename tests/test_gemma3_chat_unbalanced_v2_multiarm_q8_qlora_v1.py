from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import yaml

from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as multiarm,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1.yaml"
)
ENTRYPOINT = (
    ROOT
    / "scripts"
    / "research"
    / "run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1.py"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preflight_receipt() -> dict[str, object]:
    anchors = multiarm.FINAL_SOURCE_ANCHORS
    shard_anchors = anchors["training_shards"]
    return {
        "schema_version": multiarm.PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "operation": "validate",
        "namespace": "gemma3_chat_five_expert_qonly_unbalanced_v2",
        "model_free": True,
        "binding_contract_sha256": _sha("binding"),
        "artifact_version": anchors["artifact_version"],
        "producer_git": {
            "candidate_commit": anchors["candidate_commit"],
            "release_commit": anchors["release_commit"],
            "release_tree": anchors["release_tree"],
            "upstream_commit": anchors["upstream_commit"],
            "live_remote_commit": anchors["live_remote_commit"],
            "clean_worktree": True,
            "tags_at_head": 0,
        },
        "tree_digest_sha256": anchors["tree_digest_sha256"],
        "file_counts": {"total": 46, "payload": 23, "sidecar": 23},
        "manifest": deepcopy(anchors["manifest"]),
        "build_receipt": deepcopy(anchors["build_receipt"]),
        "schema_files": deepcopy(anchors["schema_files"]),
        "release_attestation": deepcopy(anchors["release_attestation"]),
        "training_shards": {
            key: deepcopy(shard_anchors[key])
            for key in (
                "shard_count",
                "ordered_paths",
                "ordered_concat_sha256",
                "logical_partition_bytes",
                "logical_partition_records",
                "task_bundles",
                "bundle_boundary_only",
                "ordered_shards",
                "task_bundle_intersection_empty",
                "boundary_commitment_sha256",
            )
        },
        "logical_identity": deepcopy(anchors["logical_identity"]),
        "read_set": {
            "count": 18,
            "canonical_digest_sha256": _sha("read-set"),
            "producer_inventory_sha256": anchors["producer_inventory_sha256"],
            "physical_identities_equal": True,
            "utf8_lf": True,
        },
        "release": {
            "artifact_final_identity": True,
            "independent_release_review": "passed",
            "p0_findings": 0,
            "p1_findings": 0,
            "p2_findings": 0,
            "producer_training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "model_release_authorized": False,
        },
        "terminal_recheck": {
            "two_artifact_snapshots_equal": True,
            "artifact_stat_identity_equal": True,
            "external_single_read_stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "resource_counters": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "gold_body_reads": 0,
            "heldout_body_reads": 0,
            "protected_body_reads": 0,
        },
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
            "producer_training_authority_inferred": False,
        },
    }


def _observed_tensor_entry(metadata: dict[str, object]) -> dict[str, object]:
    result = dict(metadata)
    if metadata["factor"] == "B":
        digest = multiarm.zero_tensor_sha256(int(metadata["nbytes"]))
    else:
        digest = _sha(f"kaiming:{metadata['initializer_seed_sha256']}")
    result["tensor_bytes_sha256"] = digest
    return result


def _inventory_set() -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in multiarm.TRAINED_ARMS:
        tensors = [
            _observed_tensor_entry(metadata)
            for metadata in multiarm.expected_tensor_metadata(arm)
        ]
        names = [str(tensor["name"]) for tensor in tensors]
        arms[arm] = {
            "schema_version": multiarm.INVENTORY_SCHEMA_VERSION,
            "arm": arm,
            "branch_seed": 1337,
            "seed_material": {
                "base_seed": 1337,
                "derivation": "base_seed_only_v1",
                "components": ["base_seed"],
            },
            "base_o_proj_frozen": True,
            "tensors": tensors,
            "optimizer_groups": multiarm.expected_optimizer_groups(arm, names),
        }
    return {
        "schema_version": multiarm.INVENTORY_SET_SCHEMA_VERSION,
        "arms": arms,
    }


def _dry_run(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    Path,
    Path,
    dict[str, object],
    str,
]:
    receipt_path = _write_json(
        tmp_path / "consumer_preflight.json", _preflight_receipt()
    )
    inventory_path = _write_json(tmp_path / "tensor_inventory.json", _inventory_set())
    plan = multiarm.model_free_dry_run(
        config_path=CONFIG,
        consumer_preflight_receipt=receipt_path,
        tensor_inventory=inventory_path,
    )
    preflight, receipt_sha, source_binding_sha = (
        multiarm.load_consumer_preflight_receipt(receipt_path)
    )
    trusted_external_bindings, trusted_external_bindings_sha256 = (
        multiarm.build_trusted_external_bindings(
            preflight,
            preflight_receipt_sha256=receipt_sha,
            source_binding_sha256=source_binding_sha,
        )
    )
    return (
        plan,
        receipt_path,
        inventory_path,
        trusted_external_bindings,
        trusted_external_bindings_sha256,
    )


def _phase_receipt(
    phase: dict[str, object],
    progress_root: Path,
    *,
    run_id: str = "model-free-smoke",
) -> dict[str, object]:
    evidence = multiarm.build_progress_evidence_identity(
        progress_root,
        run_id=run_id,
        arm=str(phase["arm"]),
        phase=str(phase["phase"]),
        target_steps=int(phase["steps"]),
    )
    return {
        "schema_version": multiarm.PHASE_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "run_id": run_id,
        "arm": phase["arm"],
        "phase": phase["phase"],
        "steps_completed": phase["steps"],
        "target_steps": phase["steps"],
        "trusted_external_bindings_sha256": phase["trusted_external_bindings_sha256"],
        "dataset_binding_sha256": phase["dataset_binding_sha256"],
        "comparison_lock_sha256": phase["comparison_lock_sha256"],
        "warmup_steps": phase["warmup_steps"],
        "peak_q_learning_rate": phase["peak_q_learning_rate"],
        "peak_o_learning_rate": phase["peak_o_learning_rate"],
        "optimizer_schedule_sha256": phase["optimizer_schedule_sha256"],
        "lr_trace_contract_sha256": phase["lr_trace_contract_sha256"],
        "expected_lr_trace_sha256": evidence["expected_lr_trace_sha256"],
        "observed_lr_trace_sha256": evidence["observed_lr_trace_sha256"],
        "progress_inventory_sha256": evidence["inventory_sha256"],
        "progress_step_file_sha256s": evidence["step_file_sha256s"],
        "progress_marker_file_sha256s": evidence["marker_file_sha256s"],
        "progress_final_step_file_sha256": evidence["final_step_file_sha256"],
        "progress_step_index": evidence["step_count"],
        "progress_target_steps": evidence["target_steps"],
        "base_before_sha256": _sha("base"),
        "base_after_sha256": _sha("base"),
        "adapter_sha256": _sha(f"adapter:{phase['arm']}:{phase['phase']}"),
        "finite_loss": True,
        "finite_gradients": True,
        "optimizer_state_verified": True,
        "adapter_effect_nonzero": True,
        "memory_gate_passed": True,
        "canonical_lock_verified": True,
        "atomic_publish_verified": True,
        "fresh_base": True,
        "fresh_adapter": True,
        "resume": False,
        "smoke_checkpoint_consumed_by_full": False,
        "body_included": False,
        "raw_token_ids_included": False,
    }


def _write_complete_progress(
    phase: dict[str, object],
    root: Path,
    *,
    run_id: str = "model-free-smoke",
) -> Path:
    progress_path = root / str(phase["arm"]) / str(phase["phase"]) / "progress"
    target = int(phase["steps"])
    for step in range(1, target + 1):
        _write_progress_step(
            progress_path,
            run_id=run_id,
            arm=str(phase["arm"]),
            phase=str(phase["phase"]),
            step=step,
            target=target,
        )
    return progress_path


def _write_progress_step(
    progress_path: Path,
    *,
    run_id: str,
    arm: str,
    phase: str,
    step: int,
    target: int,
) -> dict[str, object]:
    q_lr, o_lr = multiarm._active_step_learning_rates(
        multiarm.ARM_PROFILE[arm],
        step=step,
        total_steps=target,
    )
    return multiarm.write_scalar_progress(
        progress_path,
        run_id=run_id,
        arm=arm,
        phase=phase,
        optimizer_step=step,
        target_steps=target,
        loss=1.0,
        gradient_norm=0.1,
        lr_q=float(q_lr),
        lr_o=float(o_lr),
        torch_allocated_bytes=10,
        torch_reserved_bytes=20,
    )


def _progress_record_path(progress_root: Path, step: int) -> Path:
    return progress_root / f"step-{step:06d}" / "progress.json"


def _progress_marker_path(progress_root: Path, step: int) -> Path:
    return progress_root / f"step-{step:06d}" / "complete.sha256"


def _rewrite_progress_record(
    progress_root: Path,
    step: int,
    payload: dict[str, object],
) -> None:
    record_path = _write_json(_progress_record_path(progress_root, step), payload)
    digest = _file_sha(record_path)
    _progress_marker_path(progress_root, step).write_text(
        f"{digest}  progress.json\n",
        encoding="ascii",
        newline="\n",
    )


def test_config_is_closed_model_free_and_dependency_bound() -> None:
    config, config_sha, dependency_sha = multiarm.load_config(CONFIG)
    assert len(config_sha) == 64
    assert len(dependency_sha) == 64
    assert config["consumer_preflight"]["final_source_anchors_hardcoded"] is True
    assert config["consumer_preflight"]["require_versioned_shard_inventory"] is True
    assert (
        config["consumer_preflight"]["final_source_anchors"]
        == multiarm.FINAL_SOURCE_ANCHORS
    )
    assert (
        config["consumer_preflight"]["producer_training_authority_must_not_be_inferred"]
        is True
    )
    assert config["consumer_preflight"]["required_physical_files"] == 46
    assert config["consumer_preflight"]["required_main_chat_train_shards"] == 2
    assert config["data"]["single_training_partition_assumed"] is False
    assert config["claims"] == {
        "diagnostic_only": True,
        "proxy_only": True,
        "model_free_contract": True,
        "gpu_execution_implemented": False,
        "model_loaded": False,
        "training_executed": False,
        "evaluation_executed": False,
        "formal": False,
        "release": False,
    }
    source = (
        ROOT
        / "src"
        / "anchor_mvp"
        / "training"
        / "gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import peft" not in source
    assert "transformers" not in source


def test_config_mutations_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutations = [
        (("adapter_profiles", "q_only", "rank"), 1023),
        (("adapter_profiles", "o_only", "rank"), 65),
        (("adapter_profiles", "q_plus_micro_o", "o_alpha"), 256),
        (("optimizer", "q_learning_rate"), 0.000003),
        (("optimizer", "o_learning_rate"), 0.000001),
        (("optimizer", "warmup_steps"), 7),
        (("training", "smoke_steps_per_arm"), 3),
        (("training", "resume"), True),
        (("data", "single_training_partition_assumed"), True),
        (("claims", "gpu_execution_implemented"), True),
    ]
    for path, value in mutations:
        mutated = deepcopy(config)
        current = mutated
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
        with pytest.raises(multiarm.MultiArmContractError):
            multiarm.validate_config(mutated)


def test_preflight_requires_exact_final_sharded_receipt_and_derives_assets(
    tmp_path: Path,
) -> None:
    valid_path = _write_json(tmp_path / "valid.json", _preflight_receipt())
    receipt, receipt_sha, binding_sha = multiarm.load_consumer_preflight_receipt(
        valid_path
    )
    assert receipt["status"] == "passed"
    assert len(receipt_sha) == len(binding_sha) == 64
    assert all(
        asset["shard_count"] >= 1 for asset in receipt["training_assets"].values()
    )
    assert receipt["release"]["producer_training_authorized"] is False
    assert receipt["claims"]["producer_training_authority_inferred"] is False
    assert receipt["physical_receipt_sha256"] == receipt_sha
    main_chat = [
        receipt["training_assets"][name]
        for name in ("humor", "serious", "angry_style", "review_audit", "tool_call")
    ]
    assert {asset["shard_count"] for asset in main_chat} == {2}
    assert len({asset["shard_inventory_sha256"] for asset in main_chat}) == 1
    assert {
        asset["identity_origin"] for asset in receipt["training_assets"].values()
    } == {"consumer_derived_from_authenticated_sharded_v1_receipt"}
    assert (
        receipt["training_assets"]["planner_router"]["shard_order_contract"]
        == "consumer_derived_tree_asset_namespace_v1"
    )

    for mutation in (
        "old_shape",
        "old_44",
        "single_shard",
        "candidate_p",
        "release_r",
        "release_tree",
        "manifest_hash",
        "logical",
        "attestation",
        "producer_training_authorized",
        "authority_inferred",
    ):
        value = _preflight_receipt()
        if mutation == "old_shape":
            del value["producer_git"]
            value["producer_git_commit"] = "1" * 40
        elif mutation == "old_44":
            value["file_counts"] = {"total": 44, "payload": 22, "sidecar": 22}
        elif mutation == "single_shard":
            value["training_shards"]["shard_count"] = 1
        elif mutation == "candidate_p":
            value["producer_git"]["candidate_commit"] = "1" * 40
        elif mutation == "release_r":
            value["producer_git"]["release_commit"] = "2" * 40
        elif mutation == "release_tree":
            value["producer_git"]["release_tree"] = "3" * 40
        elif mutation == "manifest_hash":
            value["manifest"]["sha256"] = _sha("different-manifest")
        elif mutation == "logical":
            value["logical_identity"]["logical_dataset_sha256"] = _sha(
                "different-logical-dataset"
            )
        elif mutation == "attestation":
            value["release_attestation"]["sha256"] = _sha("different-attestation")
        elif mutation == "producer_training_authorized":
            value["release"]["producer_training_authorized"] = True
        else:
            value["claims"]["producer_training_authority_inferred"] = True
        path = _write_json(tmp_path / f"{mutation}.json", value)
        with pytest.raises(multiarm.MultiArmContractError):
            multiarm.load_consumer_preflight_receipt(path)


def test_consumer_derived_asset_drift_is_recomputed_and_rejected(
    tmp_path: Path,
) -> None:
    receipt_path = _write_json(tmp_path / "valid.json", _preflight_receipt())
    normalized, receipt_sha, source_sha = multiarm.load_consumer_preflight_receipt(
        receipt_path
    )
    trusted, _ = multiarm.build_trusted_external_bindings(
        normalized,
        preflight_receipt_sha256=receipt_sha,
        source_binding_sha256=source_sha,
    )
    mutated = deepcopy(trusted)
    mutated["binding"]["training_assets"]["tool_call"]["record_order_sha256"] = _sha(
        "forged-consumer-derived-order"
    )
    mutated["source_binding_sha256"] = _canonical_sha(mutated["binding"])
    mutated_sha = _canonical_sha(mutated)
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="consumer_derived_asset_identity_drift",
    ):
        multiarm.validate_trusted_external_bindings(mutated, mutated_sha)


def test_json_loads_reject_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    nested_duplicate = _write_json(
        tmp_path / "nested-duplicate.json",
        _preflight_receipt(),
    )
    nested_text = nested_duplicate.read_text(encoding="utf-8")
    nested_anchor = '"manifest":{"bytes":34236,'
    assert nested_anchor in nested_text
    nested_duplicate.write_text(
        nested_text.replace(
            nested_anchor,
            '"manifest":{"bytes":34236,"bytes":34236,',
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(multiarm.MultiArmContractError, match="duplicate_key"):
        multiarm.load_consumer_preflight_receipt(nested_duplicate)

    nonfinite = _write_json(tmp_path / "nonfinite.json", _preflight_receipt())
    nonfinite_text = nonfinite.read_text(encoding="utf-8")
    assert nested_anchor in nonfinite_text
    nonfinite.write_text(
        nonfinite_text.replace(
            nested_anchor,
            '"manifest":{"bytes":1e999,',
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(multiarm.MultiArmContractError, match="nonfinite_number"):
        multiarm.load_consumer_preflight_receipt(nonfinite)

    progress_root = tmp_path / "duplicate-progress" / "progress"
    for step in (1, 2):
        _write_progress_step(
            progress_root,
            run_id="duplicate-json-run",
            arm="tool_q_only",
            phase="smoke",
            step=step,
            target=2,
        )
    record_path = _progress_record_path(progress_root, 1)
    raw = record_path.read_text(encoding="utf-8")
    run_anchor = '"run_id":"duplicate-json-run"'
    assert run_anchor in raw
    record_path.write_text(
        raw.replace(
            run_anchor,
            f"{run_anchor},{run_anchor}",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    digest = _file_sha(record_path)
    _progress_marker_path(progress_root, 1).write_text(
        f"{digest}  progress.json\n",
        encoding="ascii",
        newline="\n",
    )
    with pytest.raises(multiarm.MultiArmContractError, match="duplicate_key"):
        multiarm.build_progress_evidence_identity(
            progress_root,
            run_id="duplicate-json-run",
            arm="tool_q_only",
            phase="smoke",
            target_steps=2,
        )


def test_tensor_inventory_enforces_shapes_counts_and_optimizer_groups(
    tmp_path: Path,
) -> None:
    inventory = _inventory_set()
    path = _write_json(tmp_path / "inventory.json", inventory)
    normalized, digest = multiarm.load_and_validate_tensor_inventory_set(path)
    assert len(digest) == 64
    assert len(normalized["arms"]["tool_q_only"]["tensors"]) == 52
    assert len(normalized["arms"]["tool_q_plus_micro_o"]["tensors"]) == 104
    assert (
        sum(
            tensor["numel"]
            for tensor in normalized["arms"]["tool_q_plus_micro_o"]["tensors"]
        )
        == 61_554_688
    )
    assert [
        group["name"]
        for group in normalized["arms"]["tool_q_plus_micro_o"]["optimizer_groups"]
    ] == ["Q", "O"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("shape", "tensor_physical_shape_or_scope_invalid"),
        ("base_o", "tensor_inventory_identity_invalid"),
        ("group_order", "optimizer_group_scope_or_order_invalid"),
        ("group_overlap", "optimizer_group_scope_or_order_invalid"),
        ("b_nonzero", "lora_B_not_zero_initialized"),
        ("paired_bytes", "paired_initial_tensor_bytes_mismatch"),
    ],
)
def test_tensor_inventory_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    inventory = _inventory_set()
    arm = inventory["arms"]["tool_q_plus_micro_o"]
    if mutation == "shape":
        arm["tensors"][0]["shape"] = [1023, 1152]
    elif mutation == "base_o":
        arm["base_o_proj_frozen"] = False
    elif mutation == "group_order":
        arm["optimizer_groups"].reverse()
    elif mutation == "group_overlap":
        arm["optimizer_groups"][1]["tensor_names"][0] = arm["optimizer_groups"][0][
            "tensor_names"
        ][0]
    elif mutation == "b_nonzero":
        arm["tensors"][1]["tensor_bytes_sha256"] = _sha("not-zero")
    else:
        arm["tensors"][0]["tensor_bytes_sha256"] = _sha("arm-specific")
    path = _write_json(tmp_path / f"{mutation}.json", inventory)
    with pytest.raises(multiarm.MultiArmContractError, match=error):
        multiarm.load_and_validate_tensor_inventory_set(path)


def test_dry_run_locks_pairs_and_excludes_comparison_assets_from_training(
    tmp_path: Path,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    assert plan["status"] == "model_free_dry_run_passed_gpu_execution_not_implemented"
    assert plan["comparison"]["tool"]["slots"] == 1200
    assert plan["comparison"]["planner"]["slots"] == 960
    assert plan["eval_only"]["tool_base"]["steps"] == 0
    assert plan["eval_only"]["tool_base"]["adapter"] is None
    assert plan["eval_only"]["tool_base"]["optimizer"] is None
    assert plan["eval_only"]["tool_base"]["asset"] == "tool_comparison_eval"
    assert (
        plan["eval_only"]["tool_base"]["dataset_binding"]
        == trusted["binding"]["evaluation_assets"]["tool_comparison_eval"]
    )
    assert plan["trusted_external_bindings_sha256"] == trusted_sha
    phases = {(phase["arm"], phase["phase"]): phase for phase in plan["phases"]}
    assert len(phases) == 18
    assert phases[("review_audit_q_only", "full")]["steps"] == 1360
    assert phases[("planner_q_only", "full")]["steps"] == 80
    assert phases[("tool_q_only", "full")]["shard_count"] == 2
    assert (
        phases[("tool_q_only", "full")]["shard_order_contract"]
        == "continuous_bundle_boundary_v1"
    )
    for phase_name in ("smoke", "full"):
        assert (
            phases[("tool_q_only", phase_name)]["comparison_lock_sha256"]
            == phases[("tool_q_plus_micro_o", phase_name)]["comparison_lock_sha256"]
        )
        planner_locks = {
            phases[(arm, phase_name)]["comparison_lock_sha256"]
            for arm in (
                "planner_q_only",
                "planner_o_only",
                "planner_q_plus_micro_o",
            )
        }
        assert len(planner_locks) == 1
    training_sources = {phase["source_asset"] for phase in plan["phases"]}
    assert training_sources.isdisjoint(multiarm.EVALUATION_ASSETS)
    assert plan["claims"]["model_loaded"] is False
    assert plan["claims"]["gpu_touched"] is False
    assert (
        plan["claims"]["diagnostic_training_authorization_source"]
        == "local_user_task_only"
    )
    assert plan["claims"]["producer_training_authority_inferred"] is False


def test_plan_reauth_rejects_internally_relocked_training_binding_drift(
    tmp_path: Path,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    mutated = deepcopy(plan)
    phase = next(
        value
        for value in mutated["phases"]
        if value["arm"] == "tool_q_only" and value["phase"] == "full"
    )
    replacement = _sha("mutated-order")
    phase["record_order_sha256"] = replacement
    phase["dataset_binding"]["record_order_sha256"] = replacement
    phase["dataset_binding_sha256"] = _canonical_sha(phase["dataset_binding"])
    phase["comparison_lock"]["record_order_sha256"] = replacement
    phase["comparison_lock_sha256"] = _canonical_sha(phase["comparison_lock"])
    with pytest.raises(multiarm.MultiArmContractError, match="phase_contract_invalid"):
        multiarm.validate_execution_plan(mutated, trusted, trusted_sha)


@pytest.mark.parametrize(
    "field",
    [
        "producer_candidate_commit",
        "producer_release_commit",
        "producer_release_tree",
        "binding_contract_sha256",
        "artifact_version",
        "artifact_tree_sha256",
        "source_manifest_sha256",
        "source_build_receipt_sha256",
        "producer_schema_files_sha256",
        "release_attestation_sha256",
        "consumer_derived_asset_identity_contract_sha256",
    ],
)
def test_plan_reauth_rejects_producer_release_or_tree_identity_drift(
    tmp_path: Path,
    field: str,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    mutated = deepcopy(plan)
    if field in {
        "producer_candidate_commit",
        "producer_release_commit",
        "producer_release_tree",
    }:
        mutated[field] = "2" * 40
    elif field == "artifact_version":
        mutated[field] = "anchor.invalid-artifact.v1"
    else:
        mutated[field] = _sha(f"drift:{field}")
    with pytest.raises(multiarm.MultiArmContractError, match="plan_identity_invalid"):
        multiarm.validate_execution_plan(mutated, trusted, trusted_sha)


def test_plan_reauth_rejects_internally_relocked_eval_binding_drift(
    tmp_path: Path,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    mutated = deepcopy(plan)
    replacement = _sha("mutated-eval-target")
    external = mutated["evaluation_assets"]["tool_comparison_eval"]
    external["dataset_binding"]["target_projection_sha256"] = replacement
    external["dataset_binding_sha256"] = _canonical_sha(external["dataset_binding"])
    eval_only = mutated["eval_only"]["tool_base"]
    eval_only["dataset_binding"]["target_projection_sha256"] = replacement
    eval_only["dataset_binding_sha256"] = _canonical_sha(eval_only["dataset_binding"])
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="evaluation_asset_binding_mismatch",
    ):
        multiarm.validate_execution_plan(mutated, trusted, trusted_sha)


def test_trusted_external_binding_requires_independent_caller_anchor(
    tmp_path: Path,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    mutated = deepcopy(trusted)
    mutated["binding"]["tree_digest_sha256"] = _sha("mutated-tree")
    mutated["source_binding_sha256"] = _canonical_sha(mutated["binding"])
    with pytest.raises(multiarm.MultiArmContractError, match="anchor_mismatch"):
        multiarm.validate_execution_plan(plan, mutated, trusted_sha)


def test_plan_rejects_peak_or_schedule_identity_mutation(tmp_path: Path) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    for field, value in (
        ("peak_q_learning_rate", 0.000003),
        ("warmup_steps", 7),
        ("optimizer_schedule_sha256", _sha("different-schedule")),
        ("lr_trace_contract_sha256", _sha("different-lr-trace-contract")),
    ):
        mutated = deepcopy(plan)
        mutated["phases"][0][field] = value
        with pytest.raises(multiarm.MultiArmContractError):
            multiarm.validate_execution_plan(mutated, trusted, trusted_sha)


def test_scalar_progress_is_atomic_and_rejects_nonfinite_or_wrong_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress"
    first = multiarm.write_scalar_progress(
        path,
        run_id="dry-model-free",
        arm="tool_q_plus_micro_o",
        phase="smoke",
        optimizer_step=1,
        target_steps=2,
        loss=1.5,
        gradient_norm=0.25,
        lr_q=0.00000025,
        lr_o=0.000000025,
        torch_allocated_bytes=10,
        torch_reserved_bytes=20,
    )
    step_one_bytes = _progress_record_path(path, 1).read_bytes()
    step_one_marker = _progress_marker_path(path, 1).read_bytes()
    second = multiarm.write_scalar_progress(
        path,
        run_id="dry-model-free",
        arm="tool_q_plus_micro_o",
        phase="smoke",
        optimizer_step=2,
        target_steps=2,
        loss=1.25,
        gradient_norm=0.20,
        lr_q=0.0000005,
        lr_o=0.00000005,
        torch_allocated_bytes=11,
        torch_reserved_bytes=21,
    )
    assert first["optimizer_step"] == 1
    assert first["expected_active_q_learning_rate"] == 0.00000025
    assert first["expected_active_o_learning_rate"] == 0.000000025
    assert second["expected_active_q_learning_rate"] == 0.0000005
    assert second["expected_active_o_learning_rate"] == 0.00000005
    assert (
        json.loads(_progress_record_path(path, 2).read_text(encoding="utf-8")) == second
    )
    assert _progress_record_path(path, 1).read_bytes() == step_one_bytes
    assert _progress_marker_path(path, 1).read_bytes() == step_one_marker
    assert not list(tmp_path.rglob("*.tmp-*"))
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="progress_qo_lr_ratio_invalid",
    ):
        multiarm.write_scalar_progress(
            tmp_path / "wrong-ratio" / "progress",
            run_id="dry-model-free",
            arm="tool_q_plus_micro_o",
            phase="smoke",
            optimizer_step=1,
            target_steps=2,
            loss=1.0,
            gradient_norm=0.1,
            lr_q=0.00000025,
            lr_o=0.00000003,
            torch_allocated_bytes=10,
            torch_reserved_bytes=20,
        )
    with pytest.raises(multiarm.MultiArmContractError, match="nonfinite"):
        multiarm.write_scalar_progress(
            tmp_path / "nonfinite" / "progress",
            run_id="dry-model-free",
            arm="tool_q_only",
            phase="smoke",
            optimizer_step=1,
            target_steps=2,
            loss=math.nan,
            gradient_norm=0.1,
            lr_q=0.00000025,
            lr_o=0.0,
            torch_allocated_bytes=10,
            torch_reserved_bytes=20,
        )


def test_full_progress_freezes_warmup_boundaries_and_exact_qo_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress"
    checkpoints = {
        1: (0.00000025, 0.000000025),
        8: (0.000002, 0.0000002),
        9: (0.000002, 0.0000002),
        80: (0.000002, 0.0000002),
    }
    observed_prefixes: list[str] = []
    for step in range(1, 81):
        payload = _write_progress_step(
            path,
            run_id="warmup-boundary",
            arm="planner_q_plus_micro_o",
            phase="full",
            step=step,
            target=80,
        )
        observed_prefixes.append(str(payload["observed_lr_trace_sha256"]))
        assert (
            payload["observed_lr_trace_sha256"]
            == payload["expected_lr_trace_prefix_sha256"]
        )
        if step in checkpoints:
            expected_q, expected_o = checkpoints[step]
            assert payload["expected_active_q_learning_rate"] == expected_q
            assert payload["expected_active_o_learning_rate"] == expected_o
            assert payload["lr_o"] * 10 == pytest.approx(payload["lr_q"])
    assert len(set(observed_prefixes)) == 80
    assert payload["observed_lr_trace_sha256"] == payload["expected_lr_trace_sha256"]
    identity = multiarm.build_progress_evidence_identity(
        path,
        run_id="warmup-boundary",
        arm="planner_q_plus_micro_o",
        phase="full",
        target_steps=80,
    )
    assert identity["step_count"] == 80
    assert len(identity["step_file_sha256s"]) == 80
    assert len(identity["marker_file_sha256s"]) == 80


def test_progress_rejects_phase_target_and_per_step_learning_rate_drift(
    tmp_path: Path,
) -> None:
    base = {
        "progress_path": tmp_path / "progress",
        "run_id": "warmup-negative",
        "arm": "tool_q_plus_micro_o",
        "phase": "full",
        "optimizer_step": 1,
        "target_steps": 1360,
        "loss": 1.0,
        "gradient_norm": 0.1,
        "lr_q": 0.00000025,
        "lr_o": 0.000000025,
        "torch_allocated_bytes": 10,
        "torch_reserved_bytes": 20,
    }
    mutated = dict(base)
    mutated["target_steps"] = 1359
    with pytest.raises(multiarm.MultiArmContractError, match="target_mismatch"):
        multiarm.write_scalar_progress(**mutated)
    mutated = dict(base)
    mutated["lr_q"] = 0.000002
    mutated["lr_o"] = 0.0000002
    with pytest.raises(multiarm.MultiArmContractError, match="qo_lr_ratio_invalid"):
        multiarm.write_scalar_progress(**mutated)


def test_progress_hash_chain_rejects_skip_repeat_mutation_and_missing_state(
    tmp_path: Path,
) -> None:
    arm = "planner_q_plus_micro_o"
    target = 80

    skip = tmp_path / "skip" / "progress"
    _write_progress_step(
        skip,
        run_id="chain-skip",
        arm=arm,
        phase="full",
        step=1,
        target=target,
    )
    _write_progress_step(
        skip,
        run_id="chain-skip",
        arm=arm,
        phase="full",
        step=2,
        target=target,
    )
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="inventory_not_contiguous",
    ):
        _write_progress_step(
            skip,
            run_id="chain-skip",
            arm=arm,
            phase="full",
            step=4,
            target=target,
        )
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="inventory_not_contiguous",
    ):
        _write_progress_step(
            skip,
            run_id="chain-skip",
            arm=arm,
            phase="full",
            step=2,
            target=target,
        )
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="inventory_not_contiguous",
    ):
        _write_progress_step(
            skip,
            run_id="chain-skip",
            arm=arm,
            phase="full",
            step=1,
            target=target,
        )

    old_lr = tmp_path / "old-lr" / "progress"
    _write_progress_step(
        old_lr,
        run_id="chain-old-lr",
        arm=arm,
        phase="full",
        step=1,
        target=target,
    )
    old_lr_payload = json.loads(
        _progress_record_path(old_lr, 1).read_text(encoding="utf-8")
    )
    old_lr_payload["lr_q"] = 0.000002
    _rewrite_progress_record(old_lr, 1, old_lr_payload)
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="state_learning_rate_invalid",
    ):
        _write_progress_step(
            old_lr,
            run_id="chain-old-lr",
            arm=arm,
            phase="full",
            step=2,
            target=target,
        )

    previous_digest = tmp_path / "previous-digest" / "progress"
    _write_progress_step(
        previous_digest,
        run_id="chain-previous",
        arm=arm,
        phase="full",
        step=1,
        target=target,
    )
    previous_payload = json.loads(
        _progress_record_path(previous_digest, 1).read_text(encoding="utf-8")
    )
    previous_payload["previous_observed_trace_sha256"] = _sha("forged-previous")
    _rewrite_progress_record(previous_digest, 1, previous_payload)
    with pytest.raises(multiarm.MultiArmContractError, match="state_trace_invalid"):
        _write_progress_step(
            previous_digest,
            run_id="chain-previous",
            arm=arm,
            phase="full",
            step=2,
            target=target,
        )

    deleted = tmp_path / "deleted" / "progress"
    _write_progress_step(
        deleted,
        run_id="chain-deleted",
        arm=arm,
        phase="full",
        step=1,
        target=target,
    )
    _progress_marker_path(deleted, 1).unlink()
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="step_incomplete_or_extra",
    ):
        _write_progress_step(
            deleted,
            run_id="chain-deleted",
            arm=arm,
            phase="full",
            step=2,
            target=target,
        )

    cross_run = tmp_path / "cross-run" / "progress"
    _write_progress_step(
        cross_run,
        run_id="run-alpha",
        arm=arm,
        phase="full",
        step=1,
        target=target,
    )
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="state_identity_invalid",
    ):
        _write_progress_step(
            cross_run,
            run_id="run-beta",
            arm=arm,
            phase="full",
            step=2,
            target=target,
        )


@pytest.mark.parametrize("preexisting_steps", [0, 1])
def test_progress_concurrent_claim_has_exactly_one_winner(
    tmp_path: Path,
    preexisting_steps: int,
) -> None:
    progress_root = tmp_path / f"race-{preexisting_steps}" / "progress"
    run_id = f"race-{preexisting_steps}"
    if preexisting_steps:
        _write_progress_step(
            progress_root,
            run_id=run_id,
            arm="tool_q_plus_micro_o",
            phase="smoke",
            step=1,
            target=2,
        )
    step = preexisting_steps + 1
    barrier = threading.Barrier(2)

    def contender(loss: float) -> object:
        q_lr, o_lr = multiarm._active_step_learning_rates(
            "q_plus_micro_o",
            step=step,
            total_steps=2,
        )
        barrier.wait(timeout=5)
        return multiarm.write_scalar_progress(
            progress_root,
            run_id=run_id,
            arm="tool_q_plus_micro_o",
            phase="smoke",
            optimizer_step=step,
            target_steps=2,
            loss=loss,
            gradient_norm=0.1,
            lr_q=float(q_lr),
            lr_o=float(o_lr),
            torch_allocated_bytes=10,
            torch_reserved_bytes=20,
        )

    results: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contender, loss) for loss in (1.0, 2.0)]
        for future in futures:
            try:
                results.append(future.result(timeout=10))
            except multiarm.MultiArmContractError as error:
                results.append(error)
    winners = [result for result in results if isinstance(result, dict)]
    failures = [
        result
        for result in results
        if isinstance(result, multiarm.MultiArmContractError)
    ]
    assert len(winners) == len(failures) == 1
    record_bytes = _progress_record_path(progress_root, step).read_bytes()
    identity = multiarm._load_progress_evidence(
        progress_root,
        run_id=run_id,
        arm="tool_q_plus_micro_o",
        phase="smoke",
        expected_steps=step,
        target_steps=2,
    )
    assert identity["step_count"] == step
    assert _progress_record_path(progress_root, step).read_bytes() == record_bytes


def test_progress_crash_incomplete_and_phase_inventory_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete" / "progress"
    incomplete.mkdir(parents=True)
    (incomplete / "step-000001").mkdir()
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="step_incomplete_or_extra",
    ):
        _write_progress_step(
            incomplete,
            run_id="crash-incomplete",
            arm="tool_q_only",
            phase="smoke",
            step=2,
            target=2,
        )

    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    plan, _, _, _, _ = _dry_run(plan_root)
    phase = next(
        item
        for item in plan["phases"]
        if item["arm"] == "tool_q_only" and item["phase"] == "smoke"
    )
    for mutation in ("missing", "extra", "duplicate_alias", "replacement", "partial"):
        progress_root = _write_complete_progress(
            phase,
            tmp_path / mutation,
            run_id=f"inventory-{mutation}",
        )
        receipt = _phase_receipt(
            phase,
            progress_root,
            run_id=f"inventory-{mutation}",
        )
        if mutation == "missing":
            _progress_marker_path(progress_root, 2).unlink()
        elif mutation == "extra":
            (progress_root / "step-000003").mkdir()
        elif mutation == "duplicate_alias":
            (progress_root / "step-000002-copy").mkdir()
        elif mutation == "replacement":
            _progress_record_path(progress_root, 1).write_bytes(b"{}\n")
        else:
            _progress_marker_path(progress_root, 1).unlink()
        with pytest.raises(multiarm.MultiArmContractError):
            multiarm.validate_phase_receipt(receipt, phase, progress_root)


def test_full_is_blocked_until_all_smokes_pass_and_receipts_are_body_free(
    tmp_path: Path,
) -> None:
    plan, _, _, trusted, trusted_sha = _dry_run(tmp_path)
    smoke_phases = [phase for phase in plan["phases"] if phase["phase"] == "smoke"]
    progress_paths = {
        str(phase["arm"]): _write_complete_progress(
            phase,
            tmp_path / "smoke-progress",
        )
        for phase in smoke_phases
    }
    receipts = [
        _phase_receipt(phase, progress_paths[str(phase["arm"])])
        for phase in smoke_phases
    ]
    gate_sha = multiarm.validate_smoke_gate(
        plan,
        receipts,
        progress_paths,
        trusted,
        trusted_sha,
    )
    assert len(gate_sha) == 64
    full_gate = multiarm.build_full_gate_receipt(
        plan,
        receipts,
        progress_paths,
        trusted,
        trusted_sha,
    )
    assert full_gate["status"] == "passed"
    assert full_gate["authorized_full_arms"] == list(multiarm.TRAINED_ARMS)
    assert full_gate["smoke_checkpoint_consumed_by_full"] is False
    assert full_gate["gpu_execution_implemented"] is False
    assert full_gate["trusted_external_bindings_sha256"] == trusted_sha

    with pytest.raises(multiarm.MultiArmContractError, match="count"):
        multiarm.validate_smoke_gate(
            plan,
            receipts[:-1],
            progress_paths,
            trusted,
            trusted_sha,
        )
    bad = deepcopy(receipts)
    bad[0]["adapter_effect_nonzero"] = False
    with pytest.raises(multiarm.MultiArmContractError, match="gate_failed"):
        multiarm.validate_smoke_gate(
            plan,
            bad,
            progress_paths,
            trusted,
            trusted_sha,
        )
    bad = deepcopy(receipts)
    bad[0]["body_included"] = True
    with pytest.raises(multiarm.MultiArmContractError, match="forbidden_claim"):
        multiarm.validate_smoke_gate(
            plan,
            bad,
            progress_paths,
            trusted,
            trusted_sha,
        )
    bad = deepcopy(receipts)
    bad[0]["observed_lr_trace_sha256"] = _sha("mutated-observed-lr-trace")
    with pytest.raises(
        multiarm.MultiArmContractError, match="progress_binding_invalid"
    ):
        multiarm.validate_smoke_gate(
            plan,
            bad,
            progress_paths,
            trusted,
            trusted_sha,
        )
    bad = deepcopy(receipts)
    bad[0]["run_id"] = "copied-phase-other-run"
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="progress_state_identity_invalid",
    ):
        multiarm.validate_smoke_gate(
            plan,
            bad,
            progress_paths,
            trusted,
            trusted_sha,
        )

    destination = tmp_path / "published-smoke"
    receipt_sha = multiarm.publish_body_free_phase_receipt(
        destination,
        receipts[0],
        smoke_phases[0],
        progress_paths[str(smoke_phases[0]["arm"])],
    )
    assert len(receipt_sha) == 64
    assert sorted(path.name for path in destination.iterdir()) == [
        "phase_receipt.json",
        "phase_receipt.json.sha256",
    ]
    with pytest.raises(
        multiarm.MultiArmContractError,
        match="destination_exists",
    ):
        multiarm.publish_body_free_phase_receipt(
            destination,
            receipts[0],
            smoke_phases[0],
            progress_paths[str(smoke_phases[0]["arm"])],
        )


def test_python_cli_validate_and_dry_run_are_body_free(tmp_path: Path) -> None:
    plan, receipt, inventory, _, _ = _dry_run(tmp_path)
    validate = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--validate-config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["status"] == "passed"
    dry = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--dry-run",
            "--consumer-preflight-receipt",
            str(receipt),
            "--tensor-inventory",
            str(inventory),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    observed = json.loads(dry.stdout)
    assert observed["source_binding_sha256"] == plan["source_binding_sha256"]
    assert observed["claims"]["body_read"] is False
    assert '"input_ids"' not in dry.stdout
    assert '"sample"' not in dry.stdout
