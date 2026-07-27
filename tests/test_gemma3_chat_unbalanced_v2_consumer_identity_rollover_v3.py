from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2 as rollover_v2,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3 as rollover_v3,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_integrated_controller_v2 as integrated,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v3 as release,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("ascii")).hexdigest()


def _snapshot(path: Path, seed: str) -> release.Snapshot:
    raw = seed.encode("ascii")
    return release.Snapshot(
        path=path,
        raw=raw,
        sha256=_sha(seed),
        identity=(1, 1, len(raw), 1),
    )


def _shards() -> list[dict[str, Any]]:
    return [
        {
            "path": "train/teacher-00000-of-00001.jsonl",
            "sha256": _sha("main"),
            "bytes": 101,
            "sidecar_sha256": _sha("main-sidecar"),
            "sidecar_bytes": 88,
            "records": 3440,
            "asset": "main_train",
            "first_record_id_sha256": _sha("main-first"),
            "last_record_id_sha256": _sha("main-last"),
            "first_task_bundle_sha256": _sha("main-bundle-first"),
            "last_task_bundle_sha256": _sha("main-bundle-last"),
        },
        {
            "path": "router/teacher-00000-of-00001.jsonl",
            "sha256": _sha("router"),
            "bytes": 102,
            "sidecar_sha256": _sha("router-sidecar"),
            "sidecar_bytes": 90,
            "records": 80,
            "asset": "planner_router_train",
            "first_record_id_sha256": _sha("router-first"),
            "last_record_id_sha256": _sha("router-last"),
            "first_task_bundle_sha256": _sha("router-bundle-first"),
            "last_task_bundle_sha256": _sha("router-bundle-last"),
        },
    ]


def _logical_identity() -> dict[str, str]:
    return {
        "record_order_sha256": _sha("record-order"),
        "target_projection_sha256": _sha("target-projection"),
        "source_join_sha256": _sha("source-join"),
        "shard_inventory_sha256": _sha("shard-inventory"),
        "logical_dataset_sha256": _sha("logical-dataset"),
    }


def _record_audit(shards: list[dict[str, Any]]) -> dict[str, Any]:
    logical = _logical_identity()
    return {
        "records": 3520,
        "main_records": 3440,
        "router_records": 80,
        "source_asset_counts": {
            "humor": 240,
            "serious": 240,
            "angry_style": 240,
            "tool_call": 1360,
            "review_audit": 1360,
            "planner_router": 80,
        },
        "train_identity_bundles": 20,
        "train_identity_records": 100,
        "train_identity_anchor_records": 80,
        "train_identity_anchor_role_counts": {
            "humor": 20,
            "serious": 20,
            "angry_style": 20,
            "tool_call": 20,
        },
        "train_identity_role_counts": {
            "humor": 20,
            "serious": 20,
            "angry_style": 20,
            "tool_call": 20,
            "review_audit": 20,
        },
        "ordered_record_sha256": logical["record_order_sha256"],
        "target_projection_sha256": logical["target_projection_sha256"],
        "source_join_sha256": logical["source_join_sha256"],
        "shard_inventory_sha256": logical["shard_inventory_sha256"],
        "logical_dataset_sha256": logical["logical_dataset_sha256"],
        "ordered_shards": shards,
        "record_schema_all_rows_passed": True,
        "bundle_boundary_preserved": True,
        "identity_bundles_recomputed_from_four_role_air_anchors_and_bundle_closure": (
            True
        ),
        "identity_bundle_role_matrix_passed": True,
        "review_identity_fact_bound_to_candidate_identity_audit": True,
        "review_target_air_sentence_reproof_claimed": False,
        "identity_fact_consistency_passed": True,
        "identity_anchor_false_attribution_rejected": True,
    }


def _synthetic_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> release.CandidateAudit:
    config = integrated.load_integrated_config(rollover_v2.INTEGRATED_CONFIG_PATH)
    finalizer_config = release._strict_json(
        config.snapshots["teacher_config"].raw,
        code="test_finalizer_config_invalid",
    )
    shards = _shards()
    record_audit = _record_audit(shards)
    monkeypatch.setattr(
        release,
        "_record_audit",
        lambda *_args, **_kwargs: ((), record_audit),
    )
    candidate_root = (
        batch.REPO_ROOT
        / "data/gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1/"
        f"candidate-{'1' * 32}"
    )
    snapshots = {
        "manifest.json": _snapshot(candidate_root / "manifest.json", "manifest"),
        "manifest.json.sha256": _snapshot(
            candidate_root / "manifest.json.sha256",
            "manifest-sidecar",
        ),
        "build_receipt.json": _snapshot(
            candidate_root / "build_receipt.json",
            "receipt",
        ),
        "build_receipt.json.sha256": _snapshot(
            candidate_root / "build_receipt.json.sha256",
            "receipt-sidecar",
        ),
        "output_inventory.json": _snapshot(
            candidate_root / "output_inventory.json",
            "inventory",
        ),
        "output_inventory.json.sha256": _snapshot(
            candidate_root / "output_inventory.json.sha256",
            "inventory-sidecar",
        ),
        "release_attestation.request.json": _snapshot(
            candidate_root / "release_attestation.request.json",
            "request",
        ),
        "release_attestation.request.json.sha256": _snapshot(
            candidate_root / "release_attestation.request.json.sha256",
            "request-sidecar",
        ),
        "train/teacher-00000-of-00001.jsonl": _snapshot(
            candidate_root / "train/teacher-00000-of-00001.jsonl",
            "main-shard",
        ),
        "router/teacher-00000-of-00001.jsonl": _snapshot(
            candidate_root / "router/teacher-00000-of-00001.jsonl",
            "router-shard",
        ),
    }
    manifest = {
        "lifecycle_status": "candidate_pending_independent_release_review",
        "source": dict(finalizer_config["source"]["release_identity"]),
        "consumer": dict(finalizer_config["consumer"]),
        "record_schema": dict(finalizer_config["schemas"]["record"]),
        "logical_identity": _logical_identity(),
        "shards": shards,
    }
    schema_snapshots = {
        "attestation": release._snapshot(
            release.ATTESTATION_SCHEMA_PATH,
            code="test_attestation_schema_invalid",
        ),
        "manifest": release._snapshot(
            release.RELEASE_MANIFEST_SCHEMA_PATH,
            code="test_manifest_schema_invalid",
        ),
        "binding": release._snapshot(
            release.BINDING_SCHEMA_PATH,
            code="test_binding_schema_invalid",
        ),
    }
    return release.CandidateAudit(
        root=candidate_root,
        snapshots=snapshots,
        manifest=manifest,
        build_receipt={
            "runtime_hmac_sha256": _sha("runtime-hmac"),
            "phase_receipt_inventory_sha256": _sha("phase-receipts"),
            "wal_chain_tip_sha256": _sha("wal-tip"),
        },
        output_inventory={"payload_tree_sha256": _sha("payload-tree")},
        release_request={},
        records=(),
        physical_tree_sha256=_sha("physical-tree"),
        implementation_snapshot=release._snapshot(
            release.IMPLEMENTATION_PATH,
            code="test_release_implementation_invalid",
        ),
        source_overlay_snapshot=_snapshot(
            release.SOURCE_OVERLAY_MANIFEST,
            "source-overlay",
        ),
        schema_snapshots=schema_snapshots,
        integrated=config,
    )


def test_v3_contract_is_closed_and_binds_exact_f23_consumer_schema() -> None:
    contract = _json(rollover_v3.CONFIG_PATH)
    schema = _json(rollover_v3.SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert contract["claims"]["consumer_prerequisite_commit"] == (
        "6240ae111182104f22f08e1a569deae866c6e210"
    )
    binding = contract["authenticated_artifacts"]["binding_schema"]
    assert binding == {
        "path": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
            "consumer_rollover_v2.schema.json"
        ),
        "sha256": ("491eb5a085884a17fcd02ec7335f0ff226a77bf701dd984947175c04e895faca"),
        "bytes": 8773,
    }
    rollover_v3._load_contract()


def test_release_documents_validate_with_corrected_binding_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _synthetic_audit(monkeypatch)
    release_root = (
        batch.REPO_ROOT / "data/gemma3_chat_unbalanced_v2_teacher_alignment_release_v2/"
        f"release-{'2' * 64}"
    )
    documents, binding = release._release_documents(
        audit,
        implementer_id="producer.implementer",
        reviewer_id="independent.reviewer",
        release_root=release_root,
    )
    manifest = release._strict_json(
        documents["final_manifest.json"],
        code="test_final_manifest_invalid",
    )
    assert manifest["binding_contract"]["schema_path"] == (
        "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
        "consumer_rollover_v2.schema.json"
    )
    assert manifest["binding_contract"]["schema_sha256"] == (
        "491eb5a085884a17fcd02ec7335f0ff226a77bf701dd984947175c04e895faca"
    )
    assert binding["consumer_prerequisite"]["consumer_commit"] == (
        "6240ae111182104f22f08e1a569deae866c6e210"
    )
    assert (
        binding["release"]["release_implementation_sha256"]
        == audit.implementation_snapshot.sha256
    )

    mutated = dict(manifest)
    mutated["binding_contract"] = dict(manifest["binding_contract"])
    mutated["binding_contract"]["schema_path"] = (
        "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json"
    )
    manifest_schema = _json(release.RELEASE_MANIFEST_SCHEMA_PATH)
    with pytest.raises(ValidationError):
        Draft202012Validator(manifest_schema).validate(mutated)


def test_external_release_cannot_run_until_runtime_slots_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = batch.RuntimeSecretSlots(b"credential", b"h" * 32)
    report = {
        "state": "complete",
        "teacher_final": {
            "state": "candidate_pending_independent_release_review",
            "artifact_root": (
                "data/gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1/"
                f"candidate-{'3' * 32}"
            ),
            "runtime_hmac_consumed_before_close": True,
        },
    }
    with pytest.raises(
        batch.AdapterError,
        match="rollover v3 runtime slots not closed",
    ):
        rollover_v3._external_release_after_close(
            report,
            slots,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
            publisher=lambda *_args, **_kwargs: {},
        )

    slots.close()
    candidate = release.CANDIDATE_PARENT / f"candidate-{'3' * 32}"
    monkeypatch.setattr(rollover_v3, "_repo_file", lambda *_args, **_kwargs: candidate)
    released = rollover_v3._external_release_after_close(
        report,
        slots,
        implementer_id="producer.implementer",
        reviewer_id="independent.reviewer",
        publisher=lambda *_args, **_kwargs: {
            "state": "released_pending_consumer_acceptance",
            "final": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
        },
    )
    assert released["state"] == "released_pending_consumer_acceptance"


def test_external_release_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = batch.RuntimeSecretSlots(b"credential", b"h" * 32)
    slots.close()
    candidate = release.CANDIDATE_PARENT / f"candidate-{'4' * 32}"
    monkeypatch.setattr(rollover_v3, "_repo_file", lambda *_args, **_kwargs: candidate)
    report = {
        "state": "complete",
        "teacher_final": {
            "state": "candidate_pending_independent_release_review",
            "artifact_root": candidate.relative_to(batch.REPO_ROOT).as_posix(),
            "runtime_hmac_consumed_before_close": True,
        },
    }

    def blocked(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise release.IndependentReleaseError("release_manifest_schema_rejected")

    with pytest.raises(
        batch.AdapterError,
        match="release manifest schema rejected",
    ):
        rollover_v3._external_release_after_close(
            report,
            slots,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
            publisher=blocked,
        )


def test_preflight_preserves_only_two_memory_slot_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rollover_v2,
        "preflight",
        lambda *, validate_source: {
            "state": "blocked",
            "blockers": [
                "controller_credential_slot_unloaded",
                "runtime_hmac_slot_unloaded",
            ],
            "validate_source": validate_source,
        },
    )
    result = rollover_v3.preflight(validate_source=True)
    assert result["blockers"] == [
        "controller_credential_slot_unloaded",
        "runtime_hmac_slot_unloaded",
    ]
    assert result["external_release_required_for_complete"] is True
    assert result["candidate_materialization_before_runtime_slots_close"] is True
    assert result["external_release_after_runtime_slots_close"] is True


def test_v2_release_chain_bytes_remain_unchanged() -> None:
    expected = {
        (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_"
            "consumer_rollover_v2.py"
        ): "e5fd1477ae939e2ca811360ba1b5d3f76797a1c325d50c038f5d94dbc36b6873",
        (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_"
            "attestation_v2.schema.json"
        ): "260f78af651a8e987894e1142bd98611c7eda0578d2caa10a9ccbedb711782b3",
        (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_"
            "consumer_rollover_v2.schema.json"
        ): "7786da938343296d5f5a87e71f7bf8721fc7d22ae2edecf678d798c76538a9c5",
        (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
            "consumer_rollover_v2.schema.json"
        ): "491eb5a085884a17fcd02ec7335f0ff226a77bf701dd984947175c04e895faca",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256(
            (batch.REPO_ROOT / relative).read_bytes()
        ).hexdigest() == (digest)
