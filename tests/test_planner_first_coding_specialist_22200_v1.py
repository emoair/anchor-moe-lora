from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from anchor_mvp.data.planner_first_coding_specialist_22200_v1 import (
    CampaignValidationError,
    LegacySelection,
    PlannerEvidence,
    canonical_json,
    expert_projection_decision,
    load_campaign_contract,
    load_legacy19000_selection_manifest,
    load_source_manifest,
    sha256_bytes,
    validate_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/planner_first_coding_specialist_22200_v1.json"
CONFIG_SCHEMA = (
    ROOT / "configs/data/planner_first_coding_specialist_22200_v1.schema.json"
)
SOURCE_SCHEMA = (
    ROOT / "configs/data/planner_first_coding_specialist_source_manifest_v1.schema.json"
)
DECISION_SCHEMA = (
    ROOT / "configs/data/planner_first_coding_specialist_22200_decision_v1.schema.json"
)
LEGACY_SELECTION_SCHEMA = (
    ROOT / "configs/data/planner_first_legacy19000_selection_v1.schema.json"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _evidence_pair(
    tmp_path: Path,
    *,
    kind: str,
    subject_semantic_root_sha256: str,
    file_label: str,
) -> dict[str, object]:
    artifact = tmp_path / f"{file_label}-{kind}.evidence.json"
    artifact.write_bytes(
        canonical_json(
            {
                "evidence_kind": kind,
                "schema_version": "anchor.planner-first-source-evidence.v1",
                "subject_semantic_root_sha256": subject_semantic_root_sha256,
            }
        )
    )
    sidecar = tmp_path / f"{file_label}-{kind}.evidence.sidecar.json"
    sidecar.write_bytes(
        canonical_json(
            {"artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        )
    )
    return {"artifact": _identity(artifact), "sidecar": _identity(sidecar)}


def _source_document(
    *, kind: str, count: int, authorized: bool, evidence: dict[str, object]
) -> dict[str, object]:
    objects = [
        {
            "language": "en" if index % 2 == 0 else "zh-CN",
            "source_object_id_sha256": _hash(f"{kind}-object-{index}"),
            "source_payload_sha256": _hash(f"{kind}-payload-{index}"),
            "split": "train" if index % 5 else "eval_proxy",
        }
        for index in range(count)
    ]
    rows = [{key: str(row[key]) for key in sorted(row)} for row in objects]
    return {
        "schema_version": "anchor.planner-first-source-manifest.v1",
        "source_kind": kind,
        "content_access": "metadata_only",
        "training_authorized": authorized,
        "object_count": count,
        "semantic_root_sha256": sha256_bytes(canonical_json(rows)),
        "evidence": evidence,
        "objects": objects,
    }


def _write_source(tmp_path: Path, *, kind: str, count: int, authorized: bool) -> Path:
    path = tmp_path / f"{kind}.json"
    rows = [
        {
            "language": "en" if index % 2 == 0 else "zh-CN",
            "source_object_id_sha256": _hash(f"{kind}-object-{index}"),
            "source_payload_sha256": _hash(f"{kind}-payload-{index}"),
            "split": "train" if index % 5 else "eval_proxy",
        }
        for index in range(count)
    ]
    canonical_rows = [{key: str(row[key]) for key in sorted(row)} for row in rows]
    semantic_root = sha256_bytes(canonical_json(canonical_rows))
    evidence = {
        "real_tool_trajectory_inventory": _evidence_pair(
            tmp_path,
            kind="real_tool_trajectory_inventory",
            subject_semantic_root_sha256=semantic_root,
            file_label=kind,
        ),
        "source_terms_acceptance": _evidence_pair(
            tmp_path,
            kind="source_terms_acceptance",
            subject_semantic_root_sha256=semantic_root,
            file_label=kind,
        ),
        "translation_evidence": _evidence_pair(
            tmp_path,
            kind="translation_evidence",
            subject_semantic_root_sha256=semantic_root,
            file_label=kind,
        ),
    }
    path.write_bytes(
        canonical_json(
            _source_document(
                kind=kind, count=count, authorized=authorized, evidence=evidence
            )
        )
    )
    return path


def _legacy_selection(source) -> LegacySelection:
    return LegacySelection(
        selected_source=source,
        excluded_object_ids=tuple(_hash(f"excluded-{index}") for index in range(8)),
        selection_manifest_sha256=_hash("selection-manifest"),
    )


def test_contract_is_closed_schema_and_exactly_22200():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(config, schema)
    loaded = load_campaign_contract(CONFIG)

    assert loaded["problem_object_count"] == 22_200
    assert loaded["legacy_coding_object_count"] == 19_000
    assert loaded["planner_bootstrap_object_count"] == 3_200
    assert loaded["legacy_candidate_pool_count"] == 19_008


def test_initial_preflight_is_body_free_and_fail_closed():
    decision = expert_projection_decision(CONFIG)

    validate_decision(decision)
    assert decision["status"] == "blocked"
    assert decision["next_required_stage"] == "source_registration"
    assert decision["provider_launch_permitted"] is False
    assert decision["training_authorized"] is False
    assert decision["body_reads"] == 0
    assert decision["provider_requests"] == 0
    schema = json.loads(DECISION_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(decision, schema)


def test_source_manifest_recomputes_metadata_root_and_rejects_body_fields(
    tmp_path: Path,
):
    path = _write_source(tmp_path, kind="legacy_coding_19000", count=2, authorized=True)
    source = load_source_manifest(
        path, expected_kind="legacy_coding_19000", expected_count=2
    )
    assert source.training_authorized is True
    assert source.object_count == 2

    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["objects"][0]["prompt"] = "not allowed"
    path.write_text(json.dumps(bad), encoding="utf-8", newline="\n")
    with pytest.raises(CampaignValidationError, match="body_field_forbidden"):
        load_source_manifest(
            path, expected_kind="legacy_coding_19000", expected_count=2
        )


def test_source_manifest_rejects_physical_evidence_or_sidecar_drift(tmp_path: Path):
    path = _write_source(tmp_path, kind="legacy_coding_19000", count=2, authorized=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    evidence = manifest["evidence"]["translation_evidence"]
    sidecar = tmp_path / evidence["sidecar"]["path"]
    sidecar.write_bytes(canonical_json({"artifact_sha256": _hash("forged")}))

    with pytest.raises(CampaignValidationError, match="physical_identity_drift"):
        load_source_manifest(
            path, expected_kind="legacy_coding_19000", expected_count=2
        )


def test_source_manifest_rejects_rebound_evidence_for_a_different_semantic_root(
    tmp_path: Path,
):
    path = _write_source(tmp_path, kind="legacy_coding_19000", count=2, authorized=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pair = manifest["evidence"]["translation_evidence"]
    artifact = tmp_path / pair["artifact"]["path"]
    artifact.write_bytes(
        canonical_json(
            {
                "evidence_kind": "translation_evidence",
                "schema_version": "anchor.planner-first-source-evidence.v1",
                "subject_semantic_root_sha256": _hash("other-source-root"),
            }
        )
    )
    sidecar = tmp_path / pair["sidecar"]["path"]
    sidecar.write_bytes(
        canonical_json(
            {"artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        )
    )
    pair["artifact"] = _identity(artifact)
    pair["sidecar"] = _identity(sidecar)
    path.write_bytes(canonical_json(manifest))

    with pytest.raises(
        CampaignValidationError, match="evidence_subject_binding_mismatch"
    ):
        load_source_manifest(
            path, expected_kind="legacy_coding_19000", expected_count=2
        )


def test_legacy_selection_requires_complete_deterministic_partition_and_authorization(
    tmp_path: Path,
):
    candidate_path = _write_source(
        tmp_path, kind="legacy_coding_candidate_pool", count=5, authorized=False
    )
    candidate_pool = load_source_manifest(
        candidate_path,
        expected_kind="legacy_coding_candidate_pool",
        expected_count=5,
    )
    ordered = sorted(candidate_pool.item_ids)
    selection = {
        "schema_version": "anchor.planner-first-legacy19000-selection.v1",
        "candidate_pool_manifest_sha256": candidate_pool.manifest_sha256,
        "candidate_pool_semantic_root_sha256": candidate_pool.semantic_root_sha256,
        "candidate_pool_count": 5,
        "selected_object_count": 3,
        "excluded_object_count": 2,
        "selection_rule": "lexicographic_source_object_id_sha256_take_first",
        "selected_object_ids": ordered[:3],
        "excluded_object_ids": ordered[3:],
        "selected_root_sha256": sha256_bytes(canonical_json(ordered[:3])),
        "excluded_root_sha256": sha256_bytes(canonical_json(ordered[3:])),
        "selection_evidence": {
            "source_terms_acceptance": _evidence_pair(
                tmp_path,
                kind="source_terms_acceptance",
                subject_semantic_root_sha256=candidate_pool.semantic_root_sha256,
                file_label="selection",
            ),
            "operation_authorization": _evidence_pair(
                tmp_path,
                kind="operation_authorization",
                subject_semantic_root_sha256=sha256_bytes(canonical_json(ordered[:3])),
                file_label="selection",
            ),
        },
        "training_authorized": True,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json(selection))
    result = load_legacy19000_selection_manifest(
        selection_path,
        candidate_pool=candidate_pool,
        expected_candidate_count=5,
        expected_selected_count=3,
    )

    assert result.selected_source.object_count == 3
    assert len(result.excluded_object_ids) == 2
    selection["selected_object_ids"] = list(reversed(ordered[:3]))
    selection_path.write_bytes(canonical_json(selection))
    with pytest.raises(
        CampaignValidationError,
        match="legacy_selection_not_deterministic_pool_partition",
    ):
        load_legacy19000_selection_manifest(
            selection_path,
            candidate_pool=candidate_pool,
            expected_candidate_count=5,
            expected_selected_count=3,
        )


def test_legacy_selection_schema_is_closed_at_production_cardinality():
    schema = json.loads(LEGACY_SELECTION_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["candidate_pool_count"]["const"] == 19_008
    assert schema["properties"]["selected_object_count"]["const"] == 19_000


def test_expert_projection_requires_authorized_sources_then_frozen_primary_planner(
    tmp_path: Path,
):
    legacy_path = _write_source(
        tmp_path, kind="legacy_coding_19000", count=19_000, authorized=True
    )
    bootstrap_path = _write_source(
        tmp_path, kind="planner_bootstrap_3200", count=3_200, authorized=True
    )
    legacy = load_source_manifest(
        legacy_path, expected_kind="legacy_coding_19000", expected_count=19_000
    )
    bootstrap = load_source_manifest(
        bootstrap_path, expected_kind="planner_bootstrap_3200", expected_count=3_200
    )

    before_freeze = expert_projection_decision(
        CONFIG,
        legacy_selection=_legacy_selection(legacy),
        planner_bootstrap_source=bootstrap,
    )
    assert before_freeze["status"] == "blocked"
    assert before_freeze["next_required_stage"] == "planner_train"
    assert before_freeze["blockers"] == ["planner_training_and_freeze_evidence_missing"]

    evidence = PlannerEvidence(
        training_receipt_sha256=_hash("training"),
        freeze_receipt_sha256=_hash("freeze"),
        planner_adapter_sha256=_hash("adapter"),
        primary_arm="planner_q_only",
        frozen=True,
    )
    decision = expert_projection_decision(
        CONFIG,
        legacy_selection=_legacy_selection(legacy),
        planner_bootstrap_source=bootstrap,
        planner_evidence=evidence,
    )
    assert decision["status"] == "ready_for_planner_question_commit"
    assert decision["expert_projection_permitted"] is True
    assert decision["provider_launch_permitted"] is False


def test_non_primary_or_unfrozen_planner_cannot_authorize_expert_projection(
    tmp_path: Path,
):
    legacy_path = _write_source(
        tmp_path, kind="legacy_coding_19000", count=19_000, authorized=True
    )
    bootstrap_path = _write_source(
        tmp_path, kind="planner_bootstrap_3200", count=3_200, authorized=True
    )
    legacy = load_source_manifest(
        legacy_path, expected_kind="legacy_coding_19000", expected_count=19_000
    )
    bootstrap = load_source_manifest(
        bootstrap_path, expected_kind="planner_bootstrap_3200", expected_count=3_200
    )
    evidence = PlannerEvidence(
        training_receipt_sha256=_hash("training"),
        freeze_receipt_sha256=_hash("freeze"),
        planner_adapter_sha256=_hash("adapter"),
        primary_arm="planner_o_only",
        frozen=True,
    )

    decision = expert_projection_decision(
        CONFIG,
        legacy_selection=_legacy_selection(legacy),
        planner_bootstrap_source=bootstrap,
        planner_evidence=evidence,
    )
    assert decision["status"] == "blocked"
    assert decision["blockers"] == ["frozen_planner_q_only_required"]


def test_expert_projection_rejects_unselected_legacy_source_even_when_authorized(
    tmp_path: Path,
):
    legacy_path = _write_source(
        tmp_path, kind="legacy_coding_19000", count=19_000, authorized=True
    )
    bootstrap_path = _write_source(
        tmp_path, kind="planner_bootstrap_3200", count=3_200, authorized=True
    )
    legacy = load_source_manifest(
        legacy_path, expected_kind="legacy_coding_19000", expected_count=19_000
    )
    bootstrap = load_source_manifest(
        bootstrap_path, expected_kind="planner_bootstrap_3200", expected_count=3_200
    )
    decision = expert_projection_decision(
        CONFIG, legacy_source=legacy, planner_bootstrap_source=bootstrap
    )

    assert decision["status"] == "blocked"
    assert decision["blockers"] == [
        "legacy19000_deterministic_selection_manifest_missing"
    ]


def test_hash_dag_rejects_final_to_attestation_physical_backlink(tmp_path: Path):
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    modified = copy.deepcopy(contract)
    modified["release_hash_dag"]["teacher_final_may_contain_later_attestation_sha"] = (
        True
    )
    path = tmp_path / "contract.json"
    path.write_bytes(canonical_json(modified))

    with pytest.raises(
        CampaignValidationError, match="teacher_final_attestation_backlink_forbidden"
    ):
        load_campaign_contract(path)
