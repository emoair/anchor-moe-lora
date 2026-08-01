from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.training import planner_fused_qkvo_22200_serial_adapter_v1 as adapter


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> Path:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(raw)
    sidecar = {
        "schema_version": adapter.SIDECAR_VERSION,
        "artifact_path": path.as_posix(),
        "artifact_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    path.with_name(f"{path.name}.sha256").write_bytes(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return path


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    sidecar_path = path.with_name(f"{path.name}.sha256")
    sidecar = sidecar_path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sidecar_path": sidecar_path.as_posix(),
        "sidecar_bytes": len(sidecar),
        "sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
    }


def _metadata_manifest(tmp_path: Path, *, name: str, records: int, root: str) -> Path:
    return _write(
        tmp_path / name,
        {"records": records, "semantic_root_sha256": root},
    )


def _source(tmp_path: Path) -> Path:
    planner_root = _sha("planner-root")
    legacy_root = _sha("legacy-root")
    planner_manifest = _metadata_manifest(
        tmp_path, name="planner-manifest.json", records=3200, root=planner_root
    )
    legacy_manifest = _metadata_manifest(
        tmp_path, name="legacy-manifest.json", records=19000, root=legacy_root
    )
    return _write(
        tmp_path / "source-admission.json",
        {
            "schema_version": adapter.SOURCE_VERSION,
            "stage": "source_admission",
            "final": True,
            "authorized_for_planner_fused": True,
            "planner_bootstrap": {
                "records": 3200,
                "semantic_root_sha256": planner_root,
                "manifest_identity": _identity(planner_manifest),
            },
            "legacy_coding": {
                "records": 19000,
                "semantic_root_sha256": legacy_root,
                "manifest_identity": _identity(legacy_manifest),
            },
        },
    )


def _planner_evidence(
    tmp_path: Path,
    source: Path,
    *,
    same_gate_evaluator: bool = False,
    mismatched_tokenization: bool = False,
    fused_hf_tree_frozen: bool = True,
) -> dict[str, Path]:
    original_base = _write(
        tmp_path / "original-hf-base.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-original-hf-base.v1",
            "stage": "original_hf_base",
            "original_hf_base_sha256": _sha("original-hf-base"),
            "tokenizer_template_sha256": _sha("tokenizer-template"),
        },
    )
    adapter_artifact = _write(
        tmp_path / "planner-qkvo-adapter.json",
        {
            "schema_version": (
                "anchor.planner-fused-qkvo-22200-planner-qkvo-adapter-artifact.v1"
            ),
            "stage": "planner_qkvo_adapter_artifact",
            "original_hf_base_identity": _identity(original_base),
            "planner_arm": "planner_qkvo_rank64",
            "rank": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate_ratio": {"q": 1.0, "k": 0.2, "v": 0.1, "o": 0.01},
            "adapter_sha256": _sha("planner-qkvo-adapter"),
        },
    )
    train = _write(
        tmp_path / "planner-train.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-planner-train-receipt.v1",
            "stage": "planner_qkvo_train",
            "source_admission_identity": _identity(source),
            "original_hf_base_identity": _identity(original_base),
            "planner_adapter_artifact_identity": _identity(adapter_artifact),
            "planner_arm": "planner_qkvo_rank64",
            "rank": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate_ratio": {"q": 1.0, "k": 0.2, "v": 0.1, "o": 0.01},
            "completed": True,
        },
    )
    gate_producer = _sha("planner-producer")
    gate_evaluator = (
        gate_producer if same_gate_evaluator else _sha("planner-independent-evaluator")
    )
    gate = _write(
        tmp_path / "planner-gate.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-planner-independent-gate.v1",
            "stage": "planner_independent_gate_eval",
            "planner_qkvo_train_receipt_identity": _identity(train),
            "evaluator_identity_sha256": gate_evaluator,
            "producer_identity_sha256": gate_producer,
            "evaluation_summary_sha256": _sha("planner-evaluation-summary"),
            "independent_gate_eval_passed": True,
        },
    )
    merge = _write(
        tmp_path / "planner-merge.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-planner-merge.v1",
            "stage": "planner_merge_hf_base_token_equivalent_overlay",
            "original_hf_base_identity": _identity(original_base),
            "planner_adapter_artifact_identity": _identity(adapter_artifact),
            "planner_independent_gate_receipt_identity": _identity(gate),
            "merged_into_original_hf_base": True,
            "merged_hf_base_sha256": _sha("merged-hf-base"),
        },
    )
    token_equivalence_proof = _write(
        tmp_path / "planner-token-equivalence.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-token-equivalence-proof.v1",
            "stage": "planner_token_equivalence_proof",
            "merged_hf_base_identity": _identity(merge),
            "token_equivalent_overlay_applied": True,
            "input_tokenization_digest": _sha("planner-token-equivalence"),
            "output_tokenization_digest": (
                _sha("different-tokenization")
                if mismatched_tokenization
                else _sha("planner-token-equivalence")
            ),
            "token_equivalence_passed": True,
        },
    )
    fused_hf_tree = _write(
        tmp_path / "planner-fused-hf-tree.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-fused-hf-tree.v1",
            "stage": "planner_freeze_fused_hf_tree",
            "merged_hf_base_identity": _identity(merge),
            "token_equivalence_proof_identity": _identity(token_equivalence_proof),
            "fused_hf_tree_frozen": fused_hf_tree_frozen,
            "fused_hf_base_sha256": _sha("fused-hf-base"),
        },
    )
    fused = _write(
        tmp_path / "planner-fused-q8.json",
        {
            "schema_version": "anchor.planner-fused-qkvo-22200-fused-q8-artifact.v1",
            "stage": "planner_compile_fused_q8",
            "fused_hf_tree_identity": _identity(fused_hf_tree),
            "fused_q8_compiled": True,
            "fused_base_kind": "planner_fused_q8",
            "fused_base_sha256": _sha("fused-base"),
        },
    )
    return {
        "original_base": original_base,
        "adapter_artifact": adapter_artifact,
        "train": train,
        "gate": gate,
        "merge": merge,
        "token_equivalence_proof": token_equivalence_proof,
        "fused_hf_tree": fused_hf_tree,
        "fused": fused,
    }


def _lineage(
    tmp_path: Path,
    source: Path,
    *,
    same_gate_evaluator: bool = False,
    mismatched_tokenization: bool = False,
    fused_hf_tree_frozen: bool = True,
) -> Path:
    evidence = _planner_evidence(
        tmp_path,
        source,
        same_gate_evaluator=same_gate_evaluator,
        mismatched_tokenization=mismatched_tokenization,
        fused_hf_tree_frozen=fused_hf_tree_frozen,
    )
    fused_value = json.loads(evidence["fused"].read_text(encoding="utf-8"))
    return _write(
        tmp_path / "planner-lineage.json",
        {
            "schema_version": adapter.LINEAGE_VERSION,
            "stage": "planner_fused_lineage",
            "source_admission_identity": _identity(source),
            "original_hf_base_identity": _identity(evidence["original_base"]),
            "planner_adapter_artifact_identity": _identity(
                evidence["adapter_artifact"]
            ),
            "planner_qkvo_train_receipt_identity": _identity(evidence["train"]),
            "planner_independent_gate_receipt_identity": _identity(evidence["gate"]),
            "merged_hf_base_identity": _identity(evidence["merge"]),
            "token_equivalence_proof_identity": _identity(
                evidence["token_equivalence_proof"]
            ),
            "fused_hf_tree_identity": _identity(evidence["fused_hf_tree"]),
            "fused_q8_artifact_identity": _identity(evidence["fused"]),
            "planner_arm": "planner_qkvo_rank64",
            "rank": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate_ratio": {"q": 1.0, "k": 0.2, "v": 0.1, "o": 0.01},
            "independent_gate_eval_passed": True,
            "merged_into_original_hf_base": True,
            "token_equivalent_overlay_applied": True,
            "original_hf_base_sha256": _sha("original-hf-base"),
            "fused_hf_tree_frozen": fused_hf_tree_frozen,
            "fused_hf_base_sha256": _sha("fused-hf-base"),
            "fused_base_kind": "planner_fused_q8",
            "fused_q8_compiled": True,
            "fused_base_sha256": fused_value["fused_base_sha256"],
            "route_authority": True,
            "obsolete_authorities_rejected": [
                "planner_q_only",
                "planner_q_plus_o",
                "planner_q_plus_micro_o",
                "original_base",
                "v14",
                "quarantine",
            ],
        },
    )


def _route(tmp_path: Path, source: Path, lineage: Path) -> Path:
    source_value = json.loads(source.read_text(encoding="utf-8"))
    lineage_value = json.loads(lineage.read_text(encoding="utf-8"))
    route_root = _sha("route-root")
    route_inventory = _write(
        tmp_path / "route-inventory.json",
        {
            "schema_version": (
                "anchor.planner-fused-qkvo-22200-immutable-route-inventory.v1"
            ),
            "stage": "immutable_question_route_commit",
            "planner_lineage_identity": _identity(lineage),
            "fused_planner_base_identity": lineage_value["fused_q8_artifact_identity"],
            "fused_base_sha256": lineage_value["fused_base_sha256"],
            "legacy_coding_semantic_root_sha256": source_value["legacy_coding"][
                "semantic_root_sha256"
            ],
            "question_commit_count": 19000,
            "route_commit_count": 19000,
            "route_commit_root_sha256": route_root,
        },
    )
    return _write(
        tmp_path / "route.json",
        {
            "schema_version": adapter.ROUTE_VERSION,
            "stage": "immutable_question_route_commit",
            "planner_lineage_identity": _identity(lineage),
            "fused_planner_base_identity": lineage_value["fused_q8_artifact_identity"],
            "route_inventory_identity": _identity(route_inventory),
            "fused_base_sha256": lineage_value["fused_base_sha256"],
            "legacy_coding_semantic_root_sha256": source_value["legacy_coding"][
                "semantic_root_sha256"
            ],
            "question_commit_count": 19000,
            "route_commit_count": 19000,
            "route_commit_root_sha256": route_root,
        },
    )


def _projection(tmp_path: Path, lineage: Path, route: Path) -> Path:
    route_value = json.loads(route.read_text(encoding="utf-8"))
    lineage_value = json.loads(lineage.read_text(encoding="utf-8"))
    records = [3167, 3167, 3167, 3167, 3166, 3166]
    projections = []
    for index, count in enumerate(records, start=1):
        expert_id = f"expert_{index}"
        projection_root = _sha(f"projection-{index}")
        inventory = _write(
            tmp_path / f"projection-inventory-{index}.json",
            {
                "schema_version": (
                    "anchor.planner-fused-qkvo-22200-expert-projection-inventory.v1"
                ),
                "stage": "six_expert_data_projection",
                "route_commit_identity": _identity(route),
                "fused_planner_base_identity": route_value[
                    "fused_planner_base_identity"
                ],
                "fused_base_sha256": lineage_value["fused_base_sha256"],
                "expert_id": expert_id,
                "records": count,
                "projection_root_sha256": projection_root,
                "adapter": "q_plus_o_lora",
                "o_learning_rate_ratio_to_q": 0.1,
            },
        )
        projections.append(
            {
                "expert_id": expert_id,
                "records": count,
                "projection_root_sha256": projection_root,
                "projection_inventory_identity": _identity(inventory),
                "adapter": "q_plus_o_lora",
                "o_learning_rate_ratio_to_q": 0.1,
            }
        )
    return _write(
        tmp_path / "projection.json",
        {
            "schema_version": adapter.PROJECTION_VERSION,
            "stage": "six_expert_data_projection",
            "route_commit_identity": _identity(route),
            "fused_planner_base_identity": route_value["fused_planner_base_identity"],
            "route_commit_root_sha256": route_value["route_commit_root_sha256"],
            "fused_base_sha256": lineage_value["fused_base_sha256"],
            "projected_record_count": 19000,
            "projections": projections,
        },
    )


def _packages(tmp_path: Path, lineage: Path, projection: Path) -> Path:
    lineage_value = json.loads(lineage.read_text(encoding="utf-8"))
    projection_value = json.loads(projection.read_text(encoding="utf-8"))
    previous: dict[str, object] | None = None
    rows = []
    for index, item in enumerate(projection_value["projections"], start=1):
        expert_id = item["expert_id"]
        previous_before = previous
        train = _write(
            tmp_path / f"expert-{index}-train.json",
            {
                "schema_version": "anchor.planner-fused-qkvo-22200-expert-q-plus-o-train-receipt.v1",
                "stage": "six_expert_q_plus_o_serial_train",
                "projection_identity": _identity(projection),
                "fused_planner_base_identity": projection_value[
                    "fused_planner_base_identity"
                ],
                "expert_id": expert_id,
                "adapter": "q_plus_o_lora",
                "o_learning_rate_ratio_to_q": 0.1,
                "previous_q8_package_identity": previous_before,
                "completed": True,
            },
        )
        evaluation = _write(
            tmp_path / f"expert-{index}-eval.json",
            {
                "schema_version": "anchor.planner-fused-qkvo-22200-expert-independent-eval.v1",
                "stage": "six_expert_independent_eval_q8_package",
                "train_receipt_identity": _identity(train),
                "expert_id": expert_id,
                "evaluator_identity_sha256": _sha(f"expert-evaluator-{index}"),
                "producer_identity_sha256": _sha(f"expert-producer-{index}"),
                "evaluation_summary_sha256": _sha(f"expert-evaluation-{index}"),
                "independent_eval_passed": True,
            },
        )
        q8_package = _write(
            tmp_path / f"expert-{index}-q8.json",
            {
                "schema_version": "anchor.planner-fused-qkvo-22200-expert-q8-package.v1",
                "stage": "six_expert_independent_eval_q8_package",
                "independent_eval_receipt_identity": _identity(evaluation),
                "fused_planner_base_identity": projection_value[
                    "fused_planner_base_identity"
                ],
                "expert_id": expert_id,
                "q8_package_sha256": _sha(f"q8-{index}"),
            },
        )
        previous = _identity(q8_package)
        rows.append(
            {
                "serial_index": index,
                "expert_id": expert_id,
                "adapter": "q_plus_o_lora",
                "o_learning_rate_ratio_to_q": 0.1,
                "train_receipt_identity": _identity(train),
                "independent_eval_receipt_identity": _identity(evaluation),
                "q8_package_identity": _identity(q8_package),
                "previous_q8_package_identity": previous_before,
            }
        )
    return _write(
        tmp_path / "packages.json",
        {
            "schema_version": adapter.PACKAGE_VERSION,
            "stage": "six_expert_independent_eval_q8_package",
            "projection_identity": _identity(projection),
            "fused_planner_base_identity": projection_value[
                "fused_planner_base_identity"
            ],
            "fused_base_sha256": lineage_value["fused_base_sha256"],
            "packages": rows,
            "final_route_head_base_kind": "planner_fused_q8",
        },
    )


def test_default_decision_blocks_without_source_admission() -> None:
    decision = adapter.build_decision()
    assert decision["status"] == "blocked"
    assert decision["blockers"] == ["source_admission_required"]
    assert (
        decision["provider_requests"]
        == decision["model_loads"]
        == decision["gpu_runs"]
        == 0
    )


def test_exact_physical_qkvo_to_six_expert_serial_chain(tmp_path: Path) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    projection = _projection(tmp_path, lineage, route)
    packages = _packages(tmp_path, lineage, projection)
    decision = adapter.build_decision(
        source_admission_path=source,
        planner_lineage_path=lineage,
        route_commit_path=route,
        projection_path=projection,
        package_path=packages,
    )
    assert decision["status"] == "model_free_fused_runtime_eligible"
    assert decision["planner_authorizing_arm"] == "planner_qkvo_rank64"
    assert decision["expert_adapter"] == "q_plus_o_lora"
    assert decision["serial_concurrency"] == 1
    assert decision["full_generation_kv_shared"] is False


def test_source_manifest_identity_is_physically_recomputed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest = tmp_path / "planner-manifest.json"
    _write(manifest, {"records": 3200, "semantic_root_sha256": _sha("drift")})
    with pytest.raises(
        adapter.PlannerFusedSerialError, match="physical_artifact_identity_mismatch"
    ):
        adapter.load_source_admission(source)


def test_old_qonly_and_qpluso_planner_arms_are_explicitly_rejected(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    value = json.loads(lineage.read_text(encoding="utf-8"))
    value["planner_arm"] = "planner_q_plus_o"
    _write(lineage, value)
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="schema_validation_failed:planner_lineage",
    ):
        adapter.load_planner_lineage(
            lineage, source_admission=adapter.load_source_admission(source)
        )


def test_fused_artifact_identity_cannot_be_self_reported(tmp_path: Path) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    fused = tmp_path / "planner-fused-q8.json"
    value = json.loads(fused.read_text(encoding="utf-8"))
    value["fused_base_sha256"] = _sha("forged-fused-base")
    _write(fused, value)
    with pytest.raises(
        adapter.PlannerFusedSerialError, match="physical_artifact_identity_mismatch"
    ):
        adapter.load_planner_lineage(
            lineage, source_admission=adapter.load_source_admission(source)
        )


def test_route_requires_fused_lineage_and_legacy_root(tmp_path: Path) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    value = json.loads(route.read_text(encoding="utf-8"))
    value["fused_base_sha256"] = _sha("original-base")
    _write(route, value)
    source_document = adapter.load_source_admission(source)
    lineage_document = adapter.load_planner_lineage(
        lineage, source_admission=source_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError, match="route_commit_lineage_mismatch"
    ):
        adapter.load_route_commit(
            route, source_admission=source_document, planner_lineage=lineage_document
        )


def test_projection_requires_six_distinct_experts_and_exact_closure(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    projection = _projection(tmp_path, lineage, route)
    value = json.loads(projection.read_text(encoding="utf-8"))
    value["projections"][5]["expert_id"] = value["projections"][4]["expert_id"]
    _write(projection, value)
    source_document = adapter.load_source_admission(source)
    lineage_document = adapter.load_planner_lineage(
        lineage, source_admission=source_document
    )
    route_document = adapter.load_route_commit(
        route, source_admission=source_document, planner_lineage=lineage_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="six_distinct_expert_projections_required",
    ):
        adapter.load_projection(projection, route_commit=route_document)


def test_expert_train_receipts_form_a_physical_serial_predecessor_chain(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    projection = _projection(tmp_path, lineage, route)
    packages = _packages(tmp_path, lineage, projection)
    value = json.loads(packages.read_text(encoding="utf-8"))
    value["packages"][1]["previous_q8_package_identity"] = None
    _write(packages, value)
    source_document = adapter.load_source_admission(source)
    lineage_document = adapter.load_planner_lineage(
        lineage, source_admission=source_document
    )
    route_document = adapter.load_route_commit(
        route, source_admission=source_document, planner_lineage=lineage_document
    )
    projection_document = adapter.load_projection(
        projection, route_commit=route_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="expert_package_serial_predecessor_mismatch",
    ):
        adapter.load_packages(packages, projection=projection_document)


def test_route_and_projection_inventory_leaves_are_physically_required(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    route_inventory = tmp_path / "route-inventory.json"
    route_inventory_value = json.loads(route_inventory.read_text(encoding="utf-8"))
    route_inventory_value["route_commit_root_sha256"] = _sha("forged-route-root")
    _write(route_inventory, route_inventory_value)
    source_document = adapter.load_source_admission(source)
    lineage_document = adapter.load_planner_lineage(
        lineage, source_admission=source_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="physical_artifact_identity_mismatch:immutable_route_inventory",
    ):
        adapter.load_route_commit(
            route, source_admission=source_document, planner_lineage=lineage_document
        )

    route = _route(tmp_path, source, lineage)
    projection = _projection(tmp_path, lineage, route)
    inventory = tmp_path / "projection-inventory-1.json"
    inventory_value = json.loads(inventory.read_text(encoding="utf-8"))
    inventory_value["records"] = 1
    _write(inventory, inventory_value)
    route_document = adapter.load_route_commit(
        route, source_admission=source_document, planner_lineage=lineage_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="physical_artifact_identity_mismatch:expert_projection_inventory:expert_1",
    ):
        adapter.load_projection(projection, route_commit=route_document)


def test_planner_lineage_gate_and_token_equivalence_are_physical_receipts(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(tmp_path, source)
    token_proof = tmp_path / "planner-token-equivalence.json"
    proof = json.loads(token_proof.read_text(encoding="utf-8"))
    proof["output_tokenization_digest"] = _sha("different-tokenization")
    _write(token_proof, proof)
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="physical_artifact_identity_mismatch:planner_token_equivalence_proof",
    ):
        adapter.load_planner_lineage(
            lineage, source_admission=adapter.load_source_admission(source)
        )


@pytest.mark.parametrize(
    ("same_gate_evaluator", "mismatched_tokenization", "error"),
    [
        (True, False, "planner_independent_gate_failed"),
        (False, True, "token_equivalent_overlay_not_applied"),
    ],
)
def test_planner_gate_and_token_equivalence_semantics_fail_closed(
    tmp_path: Path,
    same_gate_evaluator: bool,
    mismatched_tokenization: bool,
    error: str,
) -> None:
    source = _source(tmp_path)
    lineage = _lineage(
        tmp_path,
        source,
        same_gate_evaluator=same_gate_evaluator,
        mismatched_tokenization=mismatched_tokenization,
    )
    with pytest.raises(adapter.PlannerFusedSerialError, match=error):
        adapter.load_planner_lineage(
            lineage, source_admission=adapter.load_source_admission(source)
        )


def test_frozen_hf_tree_and_expert_base_binding_are_mandatory(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    unfrozen_lineage = _lineage(tmp_path, source, fused_hf_tree_frozen=False)
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="schema_validation_failed:planner_lineage",
    ):
        adapter.load_planner_lineage(
            unfrozen_lineage, source_admission=adapter.load_source_admission(source)
        )

    lineage = _lineage(tmp_path, source)
    route = _route(tmp_path, source, lineage)
    projection = _projection(tmp_path, lineage, route)
    packages = _packages(tmp_path, lineage, projection)
    value = json.loads(packages.read_text(encoding="utf-8"))
    value["fused_planner_base_identity"] = _identity(
        tmp_path / "planner-fused-hf-tree.json"
    )
    _write(packages, value)
    source_document = adapter.load_source_admission(source)
    lineage_document = adapter.load_planner_lineage(
        lineage, source_admission=source_document
    )
    route_document = adapter.load_route_commit(
        route, source_admission=source_document, planner_lineage=lineage_document
    )
    projection_document = adapter.load_projection(
        projection, route_commit=route_document
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="package_fused_planner_base_identity_mismatch",
    ):
        adapter.load_packages(packages, projection=projection_document)


def test_authenticated_paths_reject_internal_reparse_parent(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _write(real / "artifact.json", {"value": "metadata-only"})
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink capability unavailable: {error}")
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="authenticated_path_reparse_forbidden",
    ):
        adapter.read_regular_utf8_lf(alias / "artifact.json", trusted_root=tmp_path)


def test_source_manifest_cannot_escape_source_trust_root(tmp_path: Path) -> None:
    planner_root = _sha("planner-root")
    legacy_root = _sha("legacy-root")
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside-planner-manifest.json"
    _metadata_manifest(
        outside.parent, name=outside.name, records=3200, root=planner_root
    )
    legacy_manifest = _metadata_manifest(
        trusted, name="legacy-manifest.json", records=19000, root=legacy_root
    )
    source = _write(
        trusted / "source-admission.json",
        {
            "schema_version": adapter.SOURCE_VERSION,
            "stage": "source_admission",
            "final": True,
            "authorized_for_planner_fused": True,
            "planner_bootstrap": {
                "records": 3200,
                "semantic_root_sha256": planner_root,
                "manifest_identity": _identity(outside),
            },
            "legacy_coding": {
                "records": 19000,
                "semantic_root_sha256": legacy_root,
                "manifest_identity": _identity(legacy_manifest),
            },
        },
    )
    with pytest.raises(
        adapter.PlannerFusedSerialError,
        match="authenticated_path_outside_trusted_root",
    ):
        adapter.load_source_admission(source)


def test_closed_schemas_are_draft2020_valid() -> None:
    for path in (
        adapter.CONFIG_SCHEMA_PATH,
        adapter.SOURCE_SCHEMA_PATH,
        adapter.LINEAGE_SCHEMA_PATH,
        adapter.ROUTE_SCHEMA_PATH,
        adapter.PROJECTION_SCHEMA_PATH,
        adapter.PACKAGE_SCHEMA_PATH,
    ):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_module_has_no_runtime_model_or_provider_imports() -> None:
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    for forbidden in ("torch", "transformers", "peft", "requests", "httpx", "urllib"):
        assert f"import {forbidden}" not in source
