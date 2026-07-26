from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.data import gemma3_chat_unbalanced_v2_integrated_controller_v2 as v2
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_final_release_v2 as release,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_payload(root: Path, relative: str, raw: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(release._sidecar(path.name, raw))


def _record(
    asset: str,
    serial: int,
    *,
    identity: bool,
    task_bundle_seed: str | None = None,
) -> dict[str, Any]:
    target = (
        release.AIR_IDENTITY_SENTENCE
        if identity
        else json.dumps(
            {"answer": f"synthetic-{serial}"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return {
        "schema_version": ("anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"),
        "namespace": release.NAMESPACE,
        "record_id_sha256": _sha(f"record-{serial}"),
        "source_content_sha256": _sha(f"content-{serial}"),
        "source_serialization_identity_sha256": _sha(f"serialization-{serial}"),
        "task_bundle_sha256": _sha(task_bundle_seed or f"bundle-{serial}"),
        "source_asset": asset,
        "decision": "accepted",
        "teacher_target": target,
        "teacher_target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    }


def _candidate(
    tmp_path: Path,
    *,
    identity_mode: str = "valid",
) -> Path:
    config = v2.load_integrated_config()
    finalizer_config = release._strict_json(
        config.snapshots["teacher_config"].raw,
        code="test_finalizer_config_invalid",
    )
    root = tmp_path / ("candidate-" + "1" * 32)
    root.mkdir(parents=True)
    role_counts = {
        "humor": 240,
        "serious": 240,
        "angry_style": 240,
        "tool_call": 1360,
        "review_audit": 1360,
    }
    identity_roles = release.EXPECTED_IDENTITY_ROLES
    main: list[dict[str, Any]] = []
    serial = 0
    if identity_mode == "distinct":
        for asset in identity_roles:
            for index in range(20):
                main.append(
                    _record(
                        asset,
                        serial,
                        identity=asset in release.IDENTITY_ANCHOR_ROLES,
                        task_bundle_seed=f"identity-row-{asset}-{index}",
                    )
                )
                serial += 1
    else:
        for index in range(20):
            bundle_seed = f"identity-bundle-{index}"
            anchor_roles = release.IDENTITY_ANCHOR_ROLES
            if identity_mode == "anchor_mismatch" and index == 0:
                anchor_roles = release.IDENTITY_ANCHOR_ROLES[:-1]
            for asset in anchor_roles:
                main.append(
                    _record(
                        asset,
                        serial,
                        identity=True,
                        task_bundle_seed=bundle_seed,
                    )
                )
                serial += 1
            review_count = 1
            if index == 0 and identity_mode == "missing_review":
                review_count = 0
            elif index == 0 and identity_mode == "extra_review":
                review_count = 2
            for _ in range(review_count):
                main.append(
                    _record(
                        "review_audit",
                        serial,
                        identity=False,
                        task_bundle_seed=bundle_seed,
                    )
                )
                serial += 1
            if identity_mode == "anchor_mismatch" and index == 0:
                main.append(
                    _record(
                        "tool_call",
                        serial,
                        identity=True,
                        task_bundle_seed="identity-mismatched-tool-anchor",
                    )
                )
                serial += 1
    review_nonidentity_adjustment = {
        "missing_review": 1,
        "extra_review": -1,
    }.get(identity_mode, 0)
    for asset, count in role_counts.items():
        nonidentity_count = count - 20
        if asset == "review_audit":
            nonidentity_count += review_nonidentity_adjustment
        for _ in range(nonidentity_count):
            main.append(_record(asset, serial, identity=False))
            serial += 1
    router = [
        _record("planner_router", serial + index, identity=False) for index in range(80)
    ]
    shard_rows = [
        (
            "train/teacher-00000-of-00001.jsonl",
            "main_train",
            main,
        ),
        (
            "router/teacher-00000-of-00001.jsonl",
            "planner_router_train",
            router,
        ),
    ]
    shards: list[dict[str, Any]] = []
    inventory_entries: list[dict[str, Any]] = []
    for relative, asset, rows in shard_rows:
        raw = b"".join(release._canonical_bytes(row, newline=True) for row in rows)
        _write_payload(root, relative, raw)
        sidecar_relative = f"{relative}.sha256"
        sidecar_raw = (root / sidecar_relative).read_bytes()
        shards.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
                "sidecar_bytes": len(sidecar_raw),
                "records": len(rows),
                "asset": asset,
                "first_record_id_sha256": rows[0]["record_id_sha256"],
                "last_record_id_sha256": rows[-1]["record_id_sha256"],
                "first_task_bundle_sha256": rows[0]["task_bundle_sha256"],
                "last_task_bundle_sha256": rows[-1]["task_bundle_sha256"],
            }
        )
        inventory_entries.extend(
            [
                {
                    "path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "kind": "teacher_shard",
                },
                {
                    "path": sidecar_relative,
                    "sha256": hashlib.sha256(sidecar_raw).hexdigest(),
                    "bytes": len(sidecar_raw),
                    "kind": "sha256_sidecar",
                },
            ]
        )
    records = [*main, *router]
    logical = {
        "record_order_sha256": release._canonical_sha256(
            [row["record_id_sha256"] for row in records]
        ),
        "target_projection_sha256": release._canonical_sha256(
            [row["teacher_target_sha256"] for row in records]
        ),
        "source_join_sha256": release._canonical_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "record_id_sha256",
                        "source_content_sha256",
                        "source_serialization_identity_sha256",
                        "task_bundle_sha256",
                        "source_asset",
                    )
                }
                for row in records
            ]
        ),
        "shard_inventory_sha256": release._canonical_sha256(shards),
    }
    logical["logical_dataset_sha256"] = release._canonical_sha256(logical)
    identity_audit = {
        "stable_identity_sentence": release.AIR_IDENTITY_SENTENCE,
        "producer_all_source_bundles": 30,
        "producer_all_source_records": 150,
        "teacher_train_bundles": 20,
        "teacher_train_records": 100,
        "producer_eval_bundles": 10,
        "producer_eval_records": 50,
        "identity_probe_inventory_sha256": (
            "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5"
        ),
        "identity_eval_probe_body_reads": 0,
        "router_mixed_file_eval_proxy_rows_parsed": 20,
        "router_eval_proxy_rows_emitted_to_teacher_train": 0,
        "train_target_consistency_passed": True,
    }
    manifest = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-manifest.v2"
        ),
        "namespace": release.NAMESPACE,
        "artifact_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final.sharded.v2"
        ),
        "lifecycle_status": "candidate_pending_independent_release_review",
        "record_schema": {
            "path": finalizer_config["schemas"]["record"]["path"],
            "sha256": finalizer_config["schemas"]["record"]["sha256"],
            "bytes": finalizer_config["schemas"]["record"]["bytes"],
        },
        "source": dict(finalizer_config["source"]["release_identity"]),
        "controller": {
            **release._expected_controller_identity(config),
        },
        "consumer": dict(finalizer_config["consumer"]),
        "counts": {
            "records": 3520,
            "main_records": 3440,
            "router_records": 80,
            "accepted": 3520,
            "rejected": 0,
            "uncertain": 0,
            "train_identity_bundles": 20,
            "train_identity_records": 100,
            "source_assets": {
                **role_counts,
                "planner_router": 80,
            },
        },
        "identity_audit": identity_audit,
        "shards": shards,
        "logical_identity": logical,
        "claims": {
            "candidate": True,
            "final": False,
            "independent_release_review": "pending",
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "hidden_reasoning_retained": False,
            "secret_material_retained": False,
            "provider_body_retained": False,
            "eval_proxy_in_training_shards": False,
        },
    }
    manifest_raw = release._canonical_bytes(manifest, newline=True)
    _write_payload(root, "manifest.json", manifest_raw)
    for relative in ("manifest.json", "manifest.json.sha256"):
        raw = (root / relative).read_bytes()
        inventory_entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "kind": (
                    "manifest" if relative == "manifest.json" else "sha256_sidecar"
                ),
            }
        )
    output_inventory = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-output-inventory.v2"
        ),
        "namespace": release.NAMESPACE,
        "files": inventory_entries,
        "payload_tree_sha256": release._canonical_sha256(inventory_entries),
    }
    inventory_raw = release._canonical_bytes(output_inventory, newline=True)
    _write_payload(root, "output_inventory.json", inventory_raw)
    receipt = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-build-receipt.v2"
        ),
        "namespace": release.NAMESPACE,
        "status": "materialized_candidate_pending_independent_release_review",
        "controller_run_id": "1" * 32,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_bytes": len(manifest_raw),
        "output_inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "payload_tree_sha256": output_inventory["payload_tree_sha256"],
        "source_identity_sha256": "1" * 64,
        "phase_receipt_inventory_sha256": "2" * 64,
        "wal_chain_tip_sha256": "3" * 64,
        "counts": {
            "records": 3520,
            "main_records": 3440,
            "router_records": 80,
            "accepted": 3520,
            "rejected": 0,
            "uncertain": 0,
            "train_identity_bundles": 20,
            "train_identity_records": 100,
        },
        "identity_audit_sha256": release._canonical_sha256(identity_audit),
        "runtime_authority": {
            "same_process_hmac_authenticated": True,
            "three_phase_receipts_authenticated": True,
            "wal_terminally_authenticated": True,
            "runtime_hmac_verification_scope": "same_process_only",
            "runtime_hmac_key_persisted": False,
            "runtime_hmac_key_public": False,
        },
        "claims": {
            "candidate": True,
            "final": False,
            "external_release_review_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "content_in_receipt": False,
            "secret_in_receipt": False,
            "provider_body_in_receipt": False,
        },
        "built_at": "2026-07-27T00:00:00+00:00",
        "runtime_hmac_sha256": "4" * 64,
    }
    _write_payload(
        root,
        "build_receipt.json",
        release._canonical_bytes(receipt, newline=True),
    )
    request = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-release-request.v2"
        ),
        "namespace": release.NAMESPACE,
        "status": "pending_different_reviewer",
        "candidate_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "candidate_build_receipt_sha256": hashlib.sha256(
            (root / "build_receipt.json").read_bytes()
        ).hexdigest(),
        "candidate_output_inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "candidate_payload_tree_sha256": output_inventory["payload_tree_sha256"],
        "required_binding_version": release.BINDING_SCHEMA_VERSION,
        "implementer_may_sign": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "live_authorized": False,
    }
    _write_payload(
        root,
        "release_attestation.request.json",
        release._canonical_bytes(request, newline=True),
    )
    return root


@pytest.fixture(scope="module")
def audited_candidate(
    tmp_path_factory: pytest.TempPathFactory,
) -> release.CandidateAudit:
    candidate = _candidate(tmp_path_factory.mktemp("teacher-release"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(release, "_candidate_root", lambda _: candidate)
        return release.audit_candidate(
            candidate,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
        )


def test_release_schemas_are_closed_draft_2020_12() -> None:
    for path in (
        release.ATTESTATION_SCHEMA_PATH,
        release.RELEASE_MANIFEST_SCHEMA_PATH,
        release.BINDING_SCHEMA_PATH,
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)
        assert value["additionalProperties"] is False


def test_candidate_full_record_and_provenance_audit(
    audited_candidate: release.CandidateAudit,
) -> None:
    assert len(audited_candidate.records) == 3520
    assert (
        Counter(row["source_asset"] for row in audited_candidate.records)
        == release.EXPECTED_ASSET_COUNTS
    )
    assert audited_candidate.manifest["claims"]["final"] is False
    assert audited_candidate.build_receipt["runtime_authority"][
        "same_process_hmac_authenticated"
    ]
    assert audited_candidate.physical_tree_sha256
    identity_anchor_rows = [
        row
        for row in audited_candidate.records
        if row["source_asset"] in release.IDENTITY_ANCHOR_ROLES
        and release.AIR_IDENTITY_SENTENCE in row["teacher_target"]
    ]
    anchor_bundles: dict[str, Counter[str]] = {}
    for row in identity_anchor_rows:
        anchor_bundles.setdefault(row["task_bundle_sha256"], Counter())[
            row["source_asset"]
        ] += 1
    assert len(identity_anchor_rows) == 80
    assert len(anchor_bundles) == 20
    assert all(
        counts == Counter({role: 1 for role in release.IDENTITY_ANCHOR_ROLES})
        for counts in anchor_bundles.values()
    )
    identity_bundle_ids = set(anchor_bundles)
    identity_rows = [
        row
        for row in audited_candidate.records
        if row["task_bundle_sha256"] in identity_bundle_ids
    ]
    assert len(identity_rows) == 100
    assert all(
        release.AIR_IDENTITY_SENTENCE not in row["teacher_target"]
        for row in identity_rows
        if row["source_asset"] == "review_audit"
    )


def test_identity_rows_spread_across_one_hundred_bundles_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(
        tmp_path / "identity-spread",
        identity_mode="distinct",
    )
    monkeypatch.setattr(release, "_candidate_root", lambda _: candidate)
    with pytest.raises(
        release.IndependentReleaseError,
        match="release_candidate_identity_anchor_bundle_count_drift",
    ):
        release.audit_candidate(
            candidate,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
        )


@pytest.mark.parametrize(
    ("identity_mode", "reason"),
    [
        ("missing_review", "release_candidate_identity_bundle_closure_count_drift"),
        ("extra_review", "release_candidate_identity_bundle_closure_count_drift"),
        ("anchor_mismatch", "release_candidate_identity_anchor_bundle_count_drift"),
    ],
)
def test_identity_bundle_closure_failures_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_mode: str,
    reason: str,
) -> None:
    candidate = _candidate(
        tmp_path / identity_mode,
        identity_mode=identity_mode,
    )
    monkeypatch.setattr(release, "_candidate_root", lambda _: candidate)
    with pytest.raises(release.IndependentReleaseError, match=reason):
        release.audit_candidate(
            candidate,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
        )


def test_release_documents_are_closed_non_authorizing_and_acyclic(
    audited_candidate: release.CandidateAudit,
) -> None:
    canonical_candidate = (
        release.batch.REPO_ROOT
        / "data/gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1/"
        f"candidate-{'1' * 32}"
    )
    canonical_release = (
        release.batch.REPO_ROOT
        / "data/gemma3_chat_unbalanced_v2_teacher_alignment_release_v2/"
        f"release-{audited_candidate.snapshots['manifest.json'].sha256}"
    )
    audit = replace(audited_candidate, root=canonical_candidate)
    documents, binding = release._release_documents(
        audit,
        implementer_id="producer.implementer",
        reviewer_id="independent.reviewer",
        release_root=canonical_release,
    )
    assert set(documents) == {
        "independent_release_attestation.json",
        "independent_release_attestation.json.sha256",
        "final_manifest.json",
        "final_manifest.json.sha256",
        "teacher_alignment_binding.v2.json",
        "teacher_alignment_binding.v2.json.sha256",
    }
    assert binding["claims"]["final"] is True
    assert binding["claims"]["training_authorized"] is False
    assert binding["counts"]["train_identity_bundles"] == 20
    assert binding["counts"]["train_identity_records"] == 100
    attestation = release._strict_json(
        documents["independent_release_attestation.json"],
        code="test_release_attestation_invalid",
    )
    assert attestation["record_audit"][
        "identity_bundles_recomputed_from_four_role_air_anchors_and_bundle_closure"
    ]
    assert attestation["record_audit"]["identity_bundle_role_matrix_passed"]
    assert attestation["record_audit"][
        "review_identity_fact_bound_to_candidate_identity_audit"
    ]
    assert (
        attestation["record_audit"]["review_target_air_sentence_reproof_claimed"]
        is False
    )
    manifest = release._strict_json(
        documents["final_manifest.json"],
        code="test_release_manifest_invalid",
    )
    assert "final_manifest_sha256" not in manifest
    assert "binding_sha256" not in manifest["binding_contract"]
    for payload in (
        "independent_release_attestation.json",
        "final_manifest.json",
        "teacher_alignment_binding.v2.json",
    ):
        assert documents[f"{payload}.sha256"] == release._sidecar(
            payload,
            documents[payload],
        )


def test_same_reviewer_and_candidate_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        release.IndependentReleaseError,
        match="release_reviewer_separation_invalid",
    ):
        release.audit_candidate(
            tmp_path,
            implementer_id="same.agent",
            reviewer_id="same.agent",
        )
    with pytest.raises(
        release.IndependentReleaseError,
        match="release_reviewer_separation_invalid",
    ):
        release.audit_candidate(
            tmp_path,
            implementer_id="Producer.Agent",
            reviewer_id="producer.agent",
        )

    candidate = _candidate(tmp_path / "mutated")
    shard = candidate / "train/teacher-00000-of-00001.jsonl"
    raw = shard.read_bytes()
    shard.write_bytes(raw.replace(b'"accepted"', b'"rejected"', 1))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(release, "_candidate_root", lambda _: candidate)
        with pytest.raises(
            release.IndependentReleaseError,
            match="release_candidate_sidecar_mismatch",
        ):
            release.audit_candidate(
                candidate,
                implementer_id="producer.implementer",
                reviewer_id="independent.reviewer",
            )


def test_atomic_release_directory_is_create_once_and_inherits_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    identity = release._directory_identity(
        source,
        code="test_source_invalid",
    )

    def competitor() -> None:
        destination.mkdir()

    with pytest.raises(FileExistsError):
        release._rename_noreplace(
            source,
            destination,
            before_rename=competitor,
        )
    assert source.is_dir()
    destination.rmdir()
    release._rename_noreplace(
        source,
        destination,
        before_rename=lambda: None,
    )
    assert (
        release._directory_identity(
            destination,
            code="test_destination_invalid",
        )
        == identity
    )


def test_full_release_publish_is_atomic_non_authorizing_and_create_once(
    tmp_path: Path,
    audited_candidate: release.CandidateAudit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_candidate = (
        tmp_path / "data/gemma3_chat_unbalanced_v2_teacher_alignment_finalizer_v1/"
        f"candidate-{'1' * 32}"
    )
    canonical_candidate.parent.mkdir(parents=True)
    canonical_candidate.mkdir()
    for relative, snapshot in audited_candidate.snapshots.items():
        destination = canonical_candidate.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(snapshot.raw)
    candidate_snapshots = release._snapshot_candidate(canonical_candidate)
    audit = replace(
        audited_candidate,
        root=canonical_candidate,
        snapshots=candidate_snapshots,
        physical_tree_sha256=release._physical_tree_sha256(candidate_snapshots),
    )
    monkeypatch.setattr(release.batch, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        release,
        "RELEASE_PARENT",
        tmp_path / "data/gemma3_chat_unbalanced_v2_teacher_alignment_release_v2",
    )
    monkeypatch.setattr(release, "audit_candidate", lambda *_args, **_kwargs: audit)
    result = release.publish_release(
        canonical_candidate,
        implementer_id="producer.implementer",
        reviewer_id="independent.reviewer",
    )
    assert result["state"] == "released_pending_consumer_acceptance"
    assert result["final"] is True
    assert result["training_authorized"] is False
    release_root = tmp_path / result["release_root"]
    assert len(list(release_root.iterdir())) == 6
    with pytest.raises(
        release.IndependentReleaseError,
        match="release_create_once_collision",
    ):
        release.publish_release(
            canonical_candidate,
            implementer_id="producer.implementer",
            reviewer_id="independent.reviewer",
        )


def test_cli_failure_is_body_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        release.main(
            [
                "--candidate",
                str(tmp_path),
                "--implementer-id",
                "same.agent",
                "--reviewer-id",
                "same.agent",
                "--validate-only",
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "blocked"
    assert output["reason_code"] == "release_reviewer_separation_invalid"
    assert str(tmp_path) not in json.dumps(output)
    assert output["network_requests"] == 0
    assert output["gpu_requests"] == 0
