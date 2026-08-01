from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.data.planner_first_coding_specialist_22200_release_v1 import (
    ReleaseValidationError,
    validate_release_chain,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _asset(label: str, records: int = 1) -> dict[str, object]:
    return {
        "bytes": 1,
        "path": f"assets/{label}.jsonl",
        "records": records,
        "sha256": _hash(label),
    }


def _inventory(label: str, count: int) -> dict[str, object]:
    return {"count": count, "root_sha256": _hash(label)}


def _schemas() -> dict[str, Path]:
    return {
        "teacher": CONFIG / "planner_first_22200_teacher_final_v1.schema.json",
        "attestation": CONFIG
        / "planner_first_22200_producer_attestation_v1.schema.json",
        "acceptance": CONFIG / "planner_first_22200_consumer_acceptance_v1.schema.json",
        "authority": CONFIG
        / "planner_first_22200_training_input_authority_v1.schema.json",
    }


def _chain(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    semantic_root = _hash("semantic-root")
    teacher = tmp_path / "teacher-final.json"
    ordered_assets = {
        "chat_shards": [_asset("chat-a"), _asset("chat-b")],
        "router_records": _asset("router"),
    }
    _write(
        teacher,
        {
            "schema_version": "anchor.planner-first-22200-teacher-final.v1",
            "stage": "teacher_final",
            "final": True,
            "semantic_root_sha256": semantic_root,
            "requested_object_count": 22_200,
            "terminal_object_count": 22_200,
            "planner_fused_lineage": {
                "planner_adapter_sha256": _hash("adapter"),
                "planner_freeze_receipt_sha256": _hash("freeze"),
                "route_commit_root_sha256": _hash("routes"),
            },
            "ordered_assets": ordered_assets,
            "inventories": {
                "accepted": _inventory("accepted", 22_000),
                "rejected": _inventory("rejected", 200),
                "quarantine": _inventory("quarantine", 0),
            },
            "training_authorized": False,
        },
    )
    attestation = tmp_path / "producer-attestation.json"
    _write(
        attestation,
        {
            "schema_version": "anchor.planner-first-22200-producer-attestation.v1",
            "stage": "producer_attestation",
            "teacher_final_identity": _identity(teacher),
            "teacher_final_semantic_root_sha256": semantic_root,
            "review_procedure": {
                "producer_identity_sha256": _hash("producer"),
                "reviewer_identity_sha256": _hash("reviewer"),
                "reviewer_independent_procedurally": True,
                "reviewed_tree_sha256": _hash("tree"),
            },
            "training_authorized": False,
        },
    )
    acceptance = tmp_path / "consumer-acceptance.json"
    _write(
        acceptance,
        {
            "schema_version": "anchor.planner-first-22200-consumer-acceptance.v1",
            "stage": "consumer_acceptance",
            "final": True,
            "consumed": False,
            "training_authorized": True,
            "teacher_final_identity": _identity(teacher),
            "producer_attestation_identity": _identity(attestation),
            "teacher_final_semantic_root_sha256": semantic_root,
            "allowed_assets": ordered_assets,
        },
    )
    authority = tmp_path / "training-authority.json"
    _write(
        authority,
        {
            "schema_version": "anchor.planner-first-22200-training-input-authority.v1",
            "stage": "training_input_authority",
            "consumed": False,
            "training_authorized": True,
            "consumer_acceptance_identity": _identity(acceptance),
            "teacher_final_semantic_root_sha256": semantic_root,
        },
    )
    return {
        "teacher": teacher,
        "attestation": attestation,
        "acceptance": acceptance,
        "authority": authority,
    }


def test_release_chain_is_one_way_and_can_authorize_only_after_acceptance(
    tmp_path: Path,
):
    paths = _chain(tmp_path)
    schemas = _schemas()
    result = validate_release_chain(
        teacher_final_path=paths["teacher"],
        teacher_final_schema_path=schemas["teacher"],
        producer_attestation_path=paths["attestation"],
        producer_attestation_schema_path=schemas["attestation"],
        consumer_acceptance_path=paths["acceptance"],
        consumer_acceptance_schema_path=schemas["acceptance"],
        training_authority_path=paths["authority"],
        training_authority_schema_path=schemas["authority"],
    )

    assert result["training_authorized"] is True
    assert result["consumed"] is False
    assert result["hash_dag_order"] == [
        "anchor.planner-first-22200-teacher-final.v1",
        "anchor.planner-first-22200-producer-attestation.v1",
        "anchor.planner-first-22200-consumer-acceptance.v1",
        "anchor.planner-first-22200-training-input-authority.v1",
    ]


def test_release_chain_rejects_tampered_attestation_identity(tmp_path: Path):
    paths = _chain(tmp_path)
    schemas = _schemas()
    attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
    attestation["teacher_final_identity"]["sha256"] = _hash("wrong")
    _write(paths["attestation"], attestation)

    with pytest.raises(
        ReleaseValidationError, match="physical_identity_mismatch:teacher_final"
    ):
        validate_release_chain(
            teacher_final_path=paths["teacher"],
            teacher_final_schema_path=schemas["teacher"],
            producer_attestation_path=paths["attestation"],
            producer_attestation_schema_path=schemas["attestation"],
            consumer_acceptance_path=paths["acceptance"],
            consumer_acceptance_schema_path=schemas["acceptance"],
            training_authority_path=paths["authority"],
            training_authority_schema_path=schemas["authority"],
        )


def test_teacher_final_closed_schema_rejects_later_attestation_backlink(tmp_path: Path):
    paths = _chain(tmp_path)
    teacher = json.loads(paths["teacher"].read_text(encoding="utf-8"))
    teacher["producer_attestation_identity"] = _identity(paths["attestation"])
    _write(paths["teacher"], teacher)
    schemas = _schemas()

    with pytest.raises(
        ReleaseValidationError, match="schema_validation_failed:teacher-final.json"
    ):
        validate_release_chain(
            teacher_final_path=paths["teacher"],
            teacher_final_schema_path=schemas["teacher"],
            producer_attestation_path=paths["attestation"],
            producer_attestation_schema_path=schemas["attestation"],
            consumer_acceptance_path=paths["acceptance"],
            consumer_acceptance_schema_path=schemas["acceptance"],
            training_authority_path=paths["authority"],
            training_authority_schema_path=schemas["authority"],
        )


def test_release_chain_rejects_asset_substitution_or_inventory_gap(tmp_path: Path):
    paths = _chain(tmp_path)
    schemas = _schemas()
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["allowed_assets"]["chat_shards"][0]["sha256"] = _hash("substitution")
    _write(paths["acceptance"], acceptance)
    authority = json.loads(paths["authority"].read_text(encoding="utf-8"))
    authority["consumer_acceptance_identity"] = _identity(paths["acceptance"])
    _write(paths["authority"], authority)

    with pytest.raises(
        ReleaseValidationError, match="consumer_allowed_assets_teacher_final_mismatch"
    ):
        validate_release_chain(
            teacher_final_path=paths["teacher"],
            teacher_final_schema_path=schemas["teacher"],
            producer_attestation_path=paths["attestation"],
            producer_attestation_schema_path=schemas["attestation"],
            consumer_acceptance_path=paths["acceptance"],
            consumer_acceptance_schema_path=schemas["acceptance"],
            training_authority_path=paths["authority"],
            training_authority_schema_path=schemas["authority"],
        )

    paths = _chain(tmp_path / "gap")
    teacher = json.loads(paths["teacher"].read_text(encoding="utf-8"))
    teacher["inventories"]["rejected"]["count"] = 199
    _write(paths["teacher"], teacher)
    attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
    attestation["teacher_final_identity"] = _identity(paths["teacher"])
    _write(paths["attestation"], attestation)
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["teacher_final_identity"] = _identity(paths["teacher"])
    acceptance["producer_attestation_identity"] = _identity(paths["attestation"])
    _write(paths["acceptance"], acceptance)
    authority = json.loads(paths["authority"].read_text(encoding="utf-8"))
    authority["consumer_acceptance_identity"] = _identity(paths["acceptance"])
    _write(paths["authority"], authority)
    with pytest.raises(
        ReleaseValidationError, match="teacher_final_inventory_closure_mismatch"
    ):
        validate_release_chain(
            teacher_final_path=paths["teacher"],
            teacher_final_schema_path=schemas["teacher"],
            producer_attestation_path=paths["attestation"],
            producer_attestation_schema_path=schemas["attestation"],
            consumer_acceptance_path=paths["acceptance"],
            consumer_acceptance_schema_path=schemas["acceptance"],
            training_authority_path=paths["authority"],
            training_authority_schema_path=schemas["authority"],
        )
