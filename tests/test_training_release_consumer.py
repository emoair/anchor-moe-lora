from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.research import training_release_consumer as consumer


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs/research/generic_train_release_lock.schema.json"
PROJECTOR_MANIFEST_SHA = "1" * 64
PROJECTOR_MANIFEST_SCHEMA_SHA = "2" * 64
PROJECTOR_SIDECAR_SCHEMA_SHA = "3" * 64
CONSUMER_CONTRACT_SHA = "a" * 64


def _row(
    *, bundle: str, role: str, task_id: str, split: str, variant: str
) -> dict[str, object]:
    return {
        "schema_version": "anchor.swebench-taskboard-sidecar.v1",
        "task_bundle_sha256": bundle,
        "source_gold_record_id": f"source-{split}-{variant}-{role}",
        "source_gold_sha256": "4" * 64,
        "source_gold_file_sha256": "5" * 64,
        "source_snapshot_sha256": "6" * 64,
        "source_snapshot_manifest_sha256": "7" * 64,
        "split": split,
        "variant": variant,
        "training_record": {
            "schema_version": "anchor.query-specialization.v1",
            "split": split,
            "variant": variant,
            "role": role,
            "task_board": {"task_id": task_id},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
        )
    )


def _partition_rows(
    split: str, variant: str, *, omit_role: str | None = None
) -> list[dict[str, object]]:
    bundle = "a" * 64 if split == "train" else "b" * 64
    task_id = "task-train" if split == "train" else "task-calibration"
    return [
        _row(
            bundle=bundle,
            role=role,
            task_id=task_id,
            split=split,
            variant=variant,
        )
        for role in consumer.REQUIRED_ROLES
        if role != omit_role
    ]


def _write_release(
    tmp_path: Path,
    *,
    omit_role: tuple[str, str, str] | None = None,
    mutate_manifest=None,
) -> tuple[Path, Path, str, dict[str, str]]:
    dataset_root = tmp_path / "dataset"
    fixed_files: list[dict[str, object]] = []
    authenticated: dict[str, str] = {}
    for relative, split, variant in consumer.FIXED_PARTITIONS:
        omitted = (
            omit_role[2]
            if omit_role is not None
            and omit_role[:2] == (split, variant)
            else None
        )
        rows = _partition_rows(split, variant, omit_role=omitted)
        path = dataset_root / relative
        _write_jsonl(path, rows)
        snapshot = path.read_bytes()
        digest = hashlib.sha256(snapshot).hexdigest()
        authenticated[relative] = digest
        fixed_files.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": len(snapshot),
                "records": len(rows),
                "split": split,
                "variant": variant,
            }
        )
    repository_root = tmp_path / "repository"
    consumer_files: dict[str, str] = {}
    for relative in (
        *consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES,
        consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT,
    ):
        path = repository_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {relative}\n", encoding="utf-8")
        consumer_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest: dict[str, object] = {
        "schema_version": consumer.RELEASE_LOCK_SCHEMA_VERSION,
        "status": "ready",
        "bindings": {
            "projector_manifest_sha256": PROJECTOR_MANIFEST_SHA,
            "projector_manifest_schema_sha256": PROJECTOR_MANIFEST_SCHEMA_SHA,
            "projector_sidecar_schema_sha256": PROJECTOR_SIDECAR_SCHEMA_SHA,
            "source_disjoint_manifest_sha256": "8" * 64,
            "generic_execution_contract_sha256": "9" * 64,
            "consumer_contract_sha256": CONSUMER_CONTRACT_SHA,
            "execution_lock_sha256": "b" * 64,
            "attestation_sha256": "c" * 64,
            "coordinator_config_sha256": "d" * 64,
            "source_bank_manifest_sha256": "e" * 64,
        },
        "fixed_files": fixed_files,
        "consumer": {
            "consumer_id": consumer.QUERY_SPECIALIZATION_CONSUMER_ID,
            "consumer_version": consumer.QUERY_SPECIALIZATION_CONSUMER_VERSION,
            "implementation_files": [
                {"path": relative, "sha256": consumer_files[relative]}
                for relative in consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES
            ],
            "launch_entrypoint": {
                "path": consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT,
                "sha256": consumer_files[
                    consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
                ],
            },
        },
        "split_group_key": "task_bundle_sha256",
        "task_id_cross_binding_key": "training_record.task_board.task_id",
        "required_roles": list(consumer.REQUIRED_ROLES),
        "provenance_location": "outer_sidecar",
        "calibration_is_heldout": False,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "canonical_gold_written": False,
        "provider_requests": 0,
        "claim_scope": "research_proxy_only",
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    release_root = tmp_path / "release"
    release_root.mkdir()
    encoded = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    (release_root / "manifest.json").write_bytes(encoded)
    (release_root / "manifest.json.sha256").write_bytes(
        f"{digest}  manifest.json\n".encode("ascii")
    )
    return release_root, dataset_root, digest, authenticated


def _load(
    release_root: Path,
    dataset_root: Path,
    digest: str,
    authenticated: dict[str, str],
) -> consumer.TrainingReleaseValidation:
    return consumer.load_training_release_lock(
        release_root=release_root,
        dataset_root=dataset_root,
        schema_path=SCHEMA_PATH,
        expected_manifest_sha256=digest,
        expected_projector_manifest_sha256=PROJECTOR_MANIFEST_SHA,
        expected_projector_manifest_schema_sha256=PROJECTOR_MANIFEST_SCHEMA_SHA,
        expected_projector_sidecar_schema_sha256=PROJECTOR_SIDECAR_SCHEMA_SHA,
        expected_consumer_contract_sha256=CONSUMER_CONTRACT_SHA,
        repository_root=release_root.parent / "repository",
        expected_consumer_id=consumer.QUERY_SPECIALIZATION_CONSUMER_ID,
        expected_consumer_version=consumer.QUERY_SPECIALIZATION_CONSUMER_VERSION,
        required_implementation_files=(
            consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES
        ),
        required_launch_entrypoint=(
            consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
        ),
        authenticated_partition_sha256=authenticated,
    )


def test_ready_release_authenticates_all_three_partitions_from_single_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    calls: dict[Path, int] = {}
    original = consumer._read_bytes_snapshot

    def counted(path: Path, code: str):
        resolved = path.resolve()
        calls[resolved] = calls.get(resolved, 0) + 1
        return original(path, code)

    monkeypatch.setattr(consumer, "_read_bytes_snapshot", counted)
    result = _load(release_root, dataset_root, digest, authenticated)

    assert result.manifest_sha256 == digest
    assert result.schema_sha256 == consumer.RELEASE_LOCK_SCHEMA_SHA256
    assert result.task_bundle_count == 2
    assert dict(result.partition_records) == {
        "train/clean.jsonl": 5,
        "train/noisy.jsonl": 5,
        "calibration/clean.jsonl": 5,
    }
    expected_paths = {
        SCHEMA_PATH.resolve(),
        (release_root / "manifest.json").resolve(),
        (release_root / "manifest.json.sha256").resolve(),
        *((dataset_root / relative).resolve() for relative, _, _ in consumer.FIXED_PARTITIONS),
        *((release_root.parent / "repository" / relative).resolve() for relative in consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES),
        (
            release_root.parent
            / "repository"
            / consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
        ).resolve(),
    }
    assert set(calls) == expected_paths
    assert set(calls.values()) == {1}


def test_release_sha256_sidecar_is_mandatory_and_exact(tmp_path: Path) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    (release_root / "manifest.json.sha256").unlink()
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_sha256_sidecar_invalid",
    ):
        _load(release_root, dataset_root, digest, authenticated)

    (release_root / "manifest.json.sha256").write_text(
        f"{digest}  renamed.json\n", encoding="ascii"
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_sha256_sidecar_invalid",
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_partition_change_fails_closed(tmp_path: Path) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    target = dataset_root / "train/clean.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_partition_mismatch"
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_missing_role_view_fails_even_when_file_metadata_is_rebound(
    tmp_path: Path,
) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, omit_role=("train", "noisy", "security_gate")
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_role_views_invalid"
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_claim_drift_fails_closed(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        manifest["calibration_is_heldout"] = True

    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, mutate_manifest=mutate
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_claims_invalid"
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_projector_binding_and_prior_authentication_are_required(tmp_path: Path) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_claims_invalid"
    ):
        consumer.load_training_release_lock(
            release_root=release_root,
            dataset_root=dataset_root,
            schema_path=SCHEMA_PATH,
            expected_manifest_sha256=digest,
            expected_projector_manifest_sha256="0" * 64,
            expected_projector_manifest_schema_sha256=PROJECTOR_MANIFEST_SCHEMA_SHA,
            expected_projector_sidecar_schema_sha256=PROJECTOR_SIDECAR_SCHEMA_SHA,
            expected_consumer_contract_sha256=CONSUMER_CONTRACT_SHA,
            repository_root=release_root.parent / "repository",
            expected_consumer_id=consumer.QUERY_SPECIALIZATION_CONSUMER_ID,
            expected_consumer_version=consumer.QUERY_SPECIALIZATION_CONSUMER_VERSION,
            required_implementation_files=(
                consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES
            ),
            required_launch_entrypoint=(
                consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
            ),
            authenticated_partition_sha256=authenticated,
        )

    authenticated["train/clean.jsonl"] = "0" * 64
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_partition_authentication_mismatch",
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_consumer_contract_identity_and_live_files_are_self_bound(
    tmp_path: Path,
) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_consumer_contract_binding_invalid",
    ):
        consumer.load_training_release_lock(
            release_root=release_root,
            dataset_root=dataset_root,
            schema_path=SCHEMA_PATH,
            expected_manifest_sha256=digest,
            expected_projector_manifest_sha256=PROJECTOR_MANIFEST_SHA,
            expected_projector_manifest_schema_sha256=PROJECTOR_MANIFEST_SCHEMA_SHA,
            expected_projector_sidecar_schema_sha256=PROJECTOR_SIDECAR_SCHEMA_SHA,
            expected_consumer_contract_sha256="0" * 64,
            repository_root=release_root.parent / "repository",
            expected_consumer_id=consumer.QUERY_SPECIALIZATION_CONSUMER_ID,
            expected_consumer_version=(
                consumer.QUERY_SPECIALIZATION_CONSUMER_VERSION
            ),
            required_implementation_files=(
                consumer.QUERY_SPECIALIZATION_IMPLEMENTATION_FILES
            ),
            required_launch_entrypoint=(
                consumer.QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
            ),
            authenticated_partition_sha256=authenticated,
        )

    target = (
        release_root.parent
        / "repository"
        / "src/anchor_mvp/research/training_release_consumer.py"
    )
    target.write_bytes(target.read_bytes() + b"# drift\n")
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_consumer_file_invalid",
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_release_for_another_consumer_is_rejected(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        nested = manifest["consumer"]
        assert isinstance(nested, dict)
        nested["consumer_id"] = "some.other.consumer"

    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, mutate_manifest=mutate
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_consumer_invalid"
    ):
        _load(release_root, dataset_root, digest, authenticated)
