from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from anchor_mvp.swebench import training_release as release_module
from anchor_mvp.swebench.taskboard_projector import project_taskboards
from anchor_mvp.swebench.training_release import (
    TrainingReleaseError,
    freeze_generic_execution_contract,
    freeze_source_disjoint,
    freeze_training_release,
)
from tests.test_swebench_taskboard_projector import (
    _build_snapshot as _build_projector_snapshot,
)


SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[1]
PROJECTOR_CONFIG = ROOT / "configs/research/swebench_taskboard_projector_v2.yaml"
MANIFEST_SCHEMA_SHA = hashlib.sha256(
    (ROOT / "configs/research/taskboard_projector_manifest.schema.json").read_bytes()
).hexdigest()
SIDECAR_SCHEMA_SHA = hashlib.sha256(
    (ROOT / "configs/research/taskboard_projector_sidecar.schema.json").read_bytes()
).hexdigest()
SEGMENT_PLAN_SCHEMA_SHA = hashlib.sha256(
    (
        ROOT / "configs/research/hierarchical_task_kv_segment_plan.schema.json"
    ).read_bytes()
).hexdigest()
TASK_KV_CONTRACT = {
    "segment_plan_schema_version": (
        "anchor.hierarchical-task-kv-segment-plan.v1"
    ),
    "segment_plan_schema_sha256": SEGMENT_PLAN_SCHEMA_SHA,
    "segment_plan_location": "outer_sidecar.segment_plan",
    "architecture": "hierarchical_task_kv",
    "execution_mode": "decoupled_frozen_prefix_producer_required",
    "materialization": "metadata_only_no_tensor_or_kv",
    "tensors_emitted": False,
    "kv_payloads_emitted": False,
    "shared_prefix_membership": (
        "strict_all_five_role_visibility_intersection"
    ),
    "ordered_prefix_chain": True,
    "independent_segment_concatenation_allowed": False,
    "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
    "shared_then_mask_allowed": False,
    "forbidden_current_future_preinsert_allowed": False,
    "cache_identity_required_exact_match_fields": [
        "model_architecture_sha256",
        "tokenizer_sha256",
        "token_order_sha256",
        "position_ids_sha256",
        "rope_config_sha256",
        "kv_producing_weights_sha256",
        "prefix_lineage_sha256",
    ],
    "cache_identity_mismatch_result": "cache_incompatible",
    "cache_identity_unknown_result": "cache_incompatible",
    "target_delta_initial_cache_scope": "expert_private_delta",
    "target_delta_promotion_requires": (
        "explicit_committed_and_causally_visible_downstream"
    ),
    "target_delta_promoted_cache_scope": "downstream_task_shared_immutable",
    "current_target_segment_emitted": False,
    "q_specialization_alone_sufficient_for_exact_reuse": False,
    "naive_in_stack_q_lora_exact_reuse_allowed": False,
    "full_generation_kv_shared_claimed": False,
    "token_level_moe_claimed": False,
}
GENERIC_SCHEMA = ROOT / "configs/research/generic_train_execution_contract.schema.json"
SOURCE_SCHEMA = ROOT / "configs/research/swebench_source_disjoint_manifest.schema.json"
RELEASE_SCHEMA = ROOT / "configs/research/generic_train_release_lock.schema.json"


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_artifact(path: Path, value: Any) -> str:
    digest = _write_json(path / "manifest.json", value)
    (path / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii", newline="\n"
    )
    return digest


def _rewrite_projector_jsonl(
    inputs: dict[str, Any],
    relative: str,
    mutate: Callable[[dict[str, Any]], None],
    *,
    all_rows: bool = False,
) -> None:
    path = inputs["projector_dir"] / relative
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows if all_rows else rows[:1]:
        mutate(row)
    encoded = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(encoded)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry["sha256"] = hashlib.sha256(encoded).hexdigest()
    entry["bytes"] = len(encoded)
    entry["records"] = len(rows)
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)


def _build_inputs(
    root: Path,
    *,
    calibration_is_heldout: bool = False,
    heldout_body: bool = False,
) -> dict[str, Any]:
    heldout = {
        "schema_version": "anchor.heldout-manifest.v1",
        "split": "heldout",
        "case_count": 6,
        "canonical_cases_sha256": "3" * 64,
        "case_file_sha256": "4" * 64,
    }
    if heldout_body:
        heldout["body"] = "must never enter a metadata manifest"
    heldout_dir = root / "heldout"
    heldout_sha = _write_artifact(heldout_dir, heldout)
    fixture = _build_projector_snapshot(root)
    snapshot_dir = fixture.root
    snapshot_path = snapshot_dir / "manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    partitions = snapshot["split_contract"]["partitions"]
    partitions["heldout"]["ids_sha256"] = "3" * 64
    partitions["heldout"]["manifest_sha256"] = heldout_sha
    if calibration_is_heldout:
        partitions["calibration"]["is_heldout"] = True
    snapshot_sha = _write_artifact(snapshot_dir, snapshot)

    projector_dir = root / "projector"
    projector_result = project_taskboards(
        PROJECTOR_CONFIG,
        snapshot_dir,
        snapshot_sha,
        projector_dir,
    )
    projector_sha = str(projector_result["manifest_sha256"])
    return {
        "heldout": heldout_dir / "manifest.json",
        "heldout_sha": heldout_sha,
        "snapshot_dir": snapshot_dir,
        "snapshot_sha": snapshot_sha,
        "projector_dir": projector_dir,
        "projector_sha": projector_sha,
    }


def _freeze_source(root: Path, inputs: dict[str, Any]) -> tuple[Path, str]:
    output = root / "source-disjoint"
    result = freeze_source_disjoint(
        inputs["snapshot_dir"],
        inputs["snapshot_sha"],
        inputs["projector_dir"],
        inputs["projector_sha"],
        inputs["heldout"],
        inputs["heldout_sha"],
        output,
    )
    return output, str(result["manifest_sha256"])


def _external_release_inputs(
    root: Path, *, preflight_ready: bool = True
) -> dict[str, Any]:
    execution_lock = root / "execution.lock.json"
    execution_sha = _write_json(
        execution_lock, {"schema_version": "anchor.swebench-execution-lock.v1"}
    )
    attestation = root / "attestation.json"
    attestation_sha = _write_json(
        attestation,
        {
            "schema_version": "anchor.multilang-execution-attestation.v1",
            "content_free": True,
            "oracle_material_retained": False,
            "lock_sha256": execution_sha,
            "tool_contract_version": "anchor.execution-tool-contract.v3",
            "ready": False,
            "remaining_gates": ["official-evaluation-separate"],
            "bindings": {},
        },
    )
    coordinator = root / "coordinator.yaml"
    coordinator.write_text("schema_version: test-coordinator\n", encoding="utf-8")
    coordinator_sha = hashlib.sha256(coordinator.read_bytes()).hexdigest()
    source_bank = root / "source-bank.json"
    source_bank_sha = _write_json(
        source_bank,
        {
            "schema_version": "anchor.swebench-publication-manifest.v1",
            "publication_ready": True,
            "source_split": "train",
            "train_only": True,
            "raw_source_included": False,
        },
    )
    reason = (
        "generic_train_execution_contract_ready"
        if preflight_ready
        else "generic_train_execution_contract_not_ready"
    )
    preflight = root / "offline-preflight.json"
    preflight_sha = _write_json(
        preflight,
        {
            "schema_version": "anchor.swebench-ccswitch-preflight.v1",
            "offline": True,
            "provider_requests": 0,
            "credentials_read": False,
            "sample_bodies_printed": False,
            "heldout_files_read": False,
            "component_ready": True,
            "bank_ready": True,
            "execution_contract_ready": preflight_ready,
            "live_start_allowed": preflight_ready,
            "live_started": False,
            "reason_code": reason,
            "source_bank_manifest_sha256": source_bank_sha,
            "execution_contract": {
                "mode": "generic_train_repo_base_commit",
                "ready": preflight_ready,
                "reason_code": reason,
                "remaining_gates": [] if preflight_ready else ["gate-not-ready"],
                "lock_sha256": execution_sha,
                "required_schema": "anchor.multilang-execution-attestation.v1",
                "observed_schema": "anchor.multilang-execution-attestation.v1",
                "required_tool_contract_version": "anchor.execution-tool-contract.v3",
                "not_official_swebench_pass": True,
            },
        },
    )
    generic = root / "generic-artifact"
    generic_sha: str | None = None
    if preflight_ready:
        generic_sha = str(
            freeze_generic_execution_contract(
                preflight,
                preflight_sha,
                execution_lock,
                execution_sha,
                attestation,
                attestation_sha,
                coordinator,
                coordinator_sha,
                source_bank,
                source_bank_sha,
                generic,
            )["manifest_sha256"]
        )
    consumer = root / "consumer.json"
    consumer_sha = _write_json(
        consumer,
        {
            "schema_version": "anchor.swebench-training-consumer-interface.v2",
            "consumer_id": "neural-swarm-taskboard",
            "consumer_version": "v2",
            "accepted_projector_schema": "anchor.swebench-taskboard-projector-manifest.v2",
            "projector_manifest_schema_sha256": MANIFEST_SCHEMA_SHA,
            "projector_sidecar_schema_sha256": SIDECAR_SCHEMA_SHA,
            "accepted_segment_plan_schema": (
                "anchor.hierarchical-task-kv-segment-plan.v1"
            ),
            "projector_segment_plan_schema_sha256": SEGMENT_PLAN_SCHEMA_SHA,
            "segment_plan_location": "outer_sidecar.segment_plan",
            "hierarchical_task_kv_architecture": "hierarchical_task_kv",
            "hierarchical_task_kv_execution_mode": (
                "decoupled_frozen_prefix_producer_required"
            ),
            "materialization": "metadata_only_no_tensor_or_kv",
            "tensors_emitted": False,
            "kv_payloads_emitted": False,
            "shared_prefix_membership": (
                "strict_all_five_role_visibility_intersection"
            ),
            "ordered_prefix_chain": True,
            "independent_segment_concatenation_allowed": False,
            "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
            "shared_then_mask_allowed": False,
            "forbidden_current_future_preinsert_allowed": False,
            "cache_identity_required_exact_match_fields": [
                "model_architecture_sha256",
                "tokenizer_sha256",
                "token_order_sha256",
                "position_ids_sha256",
                "rope_config_sha256",
                "kv_producing_weights_sha256",
                "prefix_lineage_sha256",
            ],
            "cache_identity_mismatch_result": "cache_incompatible",
            "cache_identity_unknown_result": "cache_incompatible",
            "target_delta_initial_cache_scope": "expert_private_delta",
            "target_delta_promotion_requires": (
                "explicit_committed_and_causally_visible_downstream"
            ),
            "target_delta_promoted_cache_scope": (
                "downstream_task_shared_immutable"
            ),
            "current_target_segment_emitted": False,
            "q_specialization_alone_sufficient_for_exact_reuse": False,
            "naive_in_stack_q_lora_exact_reuse_allowed": False,
            "full_generation_kv_shared_claimed": False,
            "token_level_moe_claimed": False,
            "split_group_key": "task_bundle_sha256",
            "task_id_cross_binding_key": "training_record.task_board.task_id",
            "fixed_inputs": [
                "train/clean.jsonl",
                "train/noisy.jsonl",
                "calibration/clean.jsonl",
            ],
            "required_roles": [
                "planner",
                "tool_policy",
                "frontend_gen",
                "frontend_review",
                "security_gate",
            ],
            "implementation_files": [
                {"path": "src/consumer.py", "sha256": "d" * 64}
            ],
            "launch_entrypoint": {
                "path": "scripts/train_consumer.py",
                "sha256": "e" * 64,
            },
            "provenance_location": "outer_sidecar",
            "calibration_is_heldout": False,
            "heldout_content_read": False,
            "claim_scope": "research_proxy_only",
        },
    )
    return {
        "execution_lock": execution_lock,
        "execution_sha": execution_sha,
        "generic": generic,
        "generic_sha": generic_sha,
        "attestation": attestation,
        "attestation_sha": attestation_sha,
        "coordinator": coordinator,
        "coordinator_sha": coordinator_sha,
        "source_bank": source_bank,
        "source_bank_sha": source_bank_sha,
        "preflight": preflight,
        "preflight_sha": preflight_sha,
        "consumer": consumer,
        "consumer_sha": consumer_sha,
    }


def test_freeze_source_disjoint_and_release_success(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    source_dir, source_sha = _freeze_source(tmp_path, inputs)
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    assert set(source) == set(source_schema["required"])
    assert source["partitions"]["train"]["source_population_count"] == 17105
    assert source["partitions"]["calibration"]["is_heldout"] is False
    assert source["partitions"]["heldout"]["case_count"] == 6
    assert source["partitions"]["heldout"]["canonical_cases_sha256"] == "3" * 64
    assert source["bindings"]["projector_segment_plan_schema_sha256"] == (
        SEGMENT_PLAN_SCHEMA_SHA
    )
    assert source["hierarchical_task_kv"] == TASK_KV_CONTRACT
    assert "body" not in json.dumps(source)
    assert (source_dir / "manifest.json.sha256").read_text(encoding="ascii") == (
        f"{source_sha}  manifest.json\n"
    )

    external = _external_release_inputs(tmp_path)
    release_dir = tmp_path / "release"
    result = freeze_training_release(
        inputs["projector_dir"],
        inputs["projector_sha"],
        source_dir,
        source_sha,
        external["generic"],
        external["generic_sha"],
        external["consumer"],
        external["consumer_sha"],
        external["execution_lock"],
        external["execution_sha"],
        release_dir,
    )
    release = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    release_schema = json.loads(RELEASE_SCHEMA.read_text(encoding="utf-8"))
    assert set(release) == set(release_schema["required"])
    assert result["fixed_file_count"] == 3
    assert release["bindings"]["projector_manifest_sha256"] == inputs[
        "projector_sha"
    ]
    assert [item["path"] for item in release["fixed_files"]] == [
        "train/clean.jsonl",
        "train/noisy.jsonl",
        "calibration/clean.jsonl",
    ]
    assert release["provenance_location"] == "outer_sidecar"
    assert release["bindings"]["attestation_sha256"] == external[
        "attestation_sha"
    ]
    assert release["bindings"]["projector_segment_plan_schema_sha256"] == (
        SEGMENT_PLAN_SCHEMA_SHA
    )
    assert release["hierarchical_task_kv"] == TASK_KV_CONTRACT


def test_freeze_generic_execution_contract_success(tmp_path: Path) -> None:
    external = _external_release_inputs(tmp_path)
    manifest = json.loads(
        (external["generic"] / "manifest.json").read_text(encoding="utf-8")
    )
    generic_schema = json.loads(GENERIC_SCHEMA.read_text(encoding="utf-8"))
    assert set(manifest) == set(generic_schema["required"])
    assert manifest["status"] == "ready"
    assert manifest["source_preflight_sha256"] == external["preflight_sha"]
    assert manifest["attestation_sha256"] == external["attestation_sha"]
    assert manifest["coordinator_config_sha256"] == external["coordinator_sha"]
    assert manifest["source_bank_manifest_sha256"] == external["source_bank_sha"]
    assert (external["generic"] / "manifest.json.sha256").read_text(
        encoding="ascii"
    ) == f"{external['generic_sha']}  manifest.json\n"


def test_freeze_generic_rejects_not_ready_preflight(tmp_path: Path) -> None:
    external = _external_release_inputs(tmp_path, preflight_ready=False)
    with pytest.raises(TrainingReleaseError, match="generic_preflight_not_ready"):
        freeze_generic_execution_contract(
            external["preflight"],
            external["preflight_sha"],
            external["execution_lock"],
            external["execution_sha"],
            external["attestation"],
            external["attestation_sha"],
            external["coordinator"],
            external["coordinator_sha"],
            external["source_bank"],
            external["source_bank_sha"],
            external["generic"],
        )
    assert not external["generic"].exists()


def test_release_rejects_generic_sidecar_drift(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    source_dir, source_sha = _freeze_source(tmp_path, inputs)
    external = _external_release_inputs(tmp_path)
    (external["generic"] / "manifest.json.sha256").write_text(
        f"{'0' * 64}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(TrainingReleaseError, match="generic_execution_contract_invalid"):
        freeze_training_release(
            inputs["projector_dir"],
            inputs["projector_sha"],
            source_dir,
            source_sha,
            external["generic"],
            external["generic_sha"],
            external["consumer"],
            external["consumer_sha"],
            external["execution_lock"],
            external["execution_sha"],
            tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()


def test_freeze_source_rejects_sidecar_drift(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    (inputs["snapshot_dir"] / "manifest.json.sha256").write_text(
        f"{'0' * 64}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(TrainingReleaseError, match="snapshot_artifact_invalid"):
        _freeze_source(tmp_path, inputs)


def test_freeze_source_rejects_calibration_as_heldout(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, calibration_is_heldout=True)
    with pytest.raises(TrainingReleaseError, match="source_calibration_split_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_freeze_source_rejects_heldout_body(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, heldout_body=True)
    with pytest.raises(TrainingReleaseError, match="heldout_manifest_metadata_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_freeze_source_rejects_projector_record_count_mismatch(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["records"] = 2
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)
    with pytest.raises(TrainingReleaseError, match="projector_file_invalid"):
        _freeze_source(tmp_path, inputs)


def test_freeze_source_rejects_full_generation_kv_sharing_claim(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hierarchical_task_kv"][
        "full_generation_kv_shared_claimed"
    ] = True
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)

    with pytest.raises(TrainingReleaseError, match="projector_manifest_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_release_rejects_consumer_naive_in_stack_q_lora_claim(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    source_dir, source_sha = _freeze_source(tmp_path, inputs)
    external = _external_release_inputs(tmp_path)
    consumer = json.loads(external["consumer"].read_text(encoding="utf-8"))
    consumer["naive_in_stack_q_lora_exact_reuse_allowed"] = True
    external["consumer_sha"] = _write_json(external["consumer"], consumer)

    with pytest.raises(TrainingReleaseError, match="consumer_contract_invalid"):
        freeze_training_release(
            inputs["projector_dir"],
            inputs["projector_sha"],
            source_dir,
            source_sha,
            external["generic"],
            external["generic_sha"],
            external["consumer"],
            external["consumer_sha"],
            external["execution_lock"],
            external["execution_sha"],
            tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()


def test_release_rejects_consumer_segment_schema_hash_drift(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    source_dir, source_sha = _freeze_source(tmp_path, inputs)
    external = _external_release_inputs(tmp_path)
    consumer = json.loads(external["consumer"].read_text(encoding="utf-8"))
    consumer["projector_segment_plan_schema_sha256"] = "0" * 64
    external["consumer_sha"] = _write_json(external["consumer"], consumer)

    with pytest.raises(TrainingReleaseError, match="consumer_contract_invalid"):
        freeze_training_release(
            inputs["projector_dir"],
            inputs["projector_sha"],
            source_dir,
            source_sha,
            external["generic"],
            external["generic_sha"],
            external["consumer"],
            external["consumer_sha"],
            external["execution_lock"],
            external["execution_sha"],
            tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()


def test_freeze_source_rejects_projector_schema_self_report_drift(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer"]["sidecar_schema_sha256"] = "0" * 64
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)

    with pytest.raises(
        TrainingReleaseError, match="projector_policy_binding_invalid"
    ):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


@pytest.mark.parametrize(
    "binding",
    ["snapshot_sha256", "snapshot_sha256_sidecar_sha256"],
)
def test_freeze_source_rejects_projector_snapshot_binding_drift(
    tmp_path: Path,
    binding: str,
) -> None:
    inputs = _build_inputs(tmp_path)
    drift = "0" * 64

    def mutate(row: dict[str, Any]) -> None:
        row["source_snapshot_sha256"] = drift
        row["segment_plan"]["bindings"]["source_snapshot_sha256"] = drift

    if binding == "snapshot_sha256":
        for relative in (
            "train/clean.jsonl",
            "train/noisy.jsonl",
            "calibration/clean.jsonl",
        ):
            _rewrite_projector_jsonl(inputs, relative, mutate, all_rows=True)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"][binding] = drift
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)

    with pytest.raises(
        TrainingReleaseError, match="projector_snapshot_binding_mismatch"
    ):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_freeze_source_rejects_projector_count_drift(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = inputs["projector_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["unique_segments"] += 1
    inputs["projector_sha"] = _write_artifact(inputs["projector_dir"], manifest)

    with pytest.raises(TrainingReleaseError, match="projector_counts_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_freeze_source_rejects_semantically_invalid_projector_row(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)

    def mutate(row: dict[str, Any]) -> None:
        row["segment_plan"]["cache_compatibility"][
            "naive_in_stack_q_lora_exact_reuse_allowed"
        ] = True

    _rewrite_projector_jsonl(inputs, "train/clean.jsonl", mutate)
    with pytest.raises(TrainingReleaseError, match="projector_semantic_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_publish_rechecks_final_output_after_input_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _build_inputs(tmp_path)
    verify_inventory = release_module._verify_inventory

    def tamper_after_inventory(
        inventory: dict[str, tuple[Path, release_module._BytesSnapshot]],
    ) -> None:
        verify_inventory(inventory)
        temporary = next(tmp_path.glob(".source-disjoint.tmp-*"))
        (temporary / "manifest.json").write_bytes(b"{}\n")

    monkeypatch.setattr(
        release_module,
        "_verify_inventory",
        tamper_after_inventory,
    )
    with pytest.raises(TrainingReleaseError, match="training_release_output_invalid"):
        _freeze_source(tmp_path, inputs)
    assert not (tmp_path / "source-disjoint").exists()


def test_release_rejects_stale_expected_sha(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    source_dir, source_sha = _freeze_source(tmp_path, inputs)
    external = _external_release_inputs(tmp_path)
    with pytest.raises(TrainingReleaseError, match="generic_execution_contract_invalid"):
        freeze_training_release(
            inputs["projector_dir"],
            inputs["projector_sha"],
            source_dir,
            source_sha,
            external["generic"],
            SHA,
            external["consumer"],
            external["consumer_sha"],
            external["execution_lock"],
            external["execution_sha"],
            tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()
