from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.research import training_release_consumer as consumer


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs/research/generic_train_release_lock.schema.json"
SOURCE_DISJOINT_SCHEMA_PATH = (
    REPO_ROOT / "configs/research/swebench_source_disjoint_manifest.schema.json"
)
PROJECTOR_FIXTURE = REPO_ROOT / "fixtures/research/taskboard_projector"
PROJECTOR_MANIFEST_SHA = "1" * 64
PROJECTOR_MANIFEST_SCHEMA_SHA = "2" * 64
PROJECTOR_SIDECAR_SCHEMA_SHA = (
    "c1863bfab69ce2f2388ee37fadae951b14f3d5120706bab032cab3f9aab6bdc5"
)
PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA = (
    "80f760497e0d21f7d4d532db758362a800e845e6919b18b23958caabc7f155bf"
)
CONSUMER_CONTRACT_SHA = "a" * 64


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
    relative = (
        "calibration/clean.jsonl"
        if split == "calibration"
        else f"train/{variant}.jsonl"
    )
    rows = [
        json.loads(line)
        for line in (PROJECTOR_FIXTURE / relative)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return [
        row
        for row in rows
        if row["training_record"]["role"] != omit_role
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
            "projector_segment_plan_schema_sha256": (
                PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
            ),
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
        "hierarchical_task_kv": {
            "segment_plan_schema_version": (
                "anchor.hierarchical-task-kv-segment-plan.v1"
            ),
            "segment_plan_schema_sha256": (
                PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
            ),
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
            "target_delta_promoted_cache_scope": (
                "downstream_task_shared_immutable"
            ),
            "current_target_segment_emitted": False,
            "q_specialization_alone_sufficient_for_exact_reuse": False,
            "naive_in_stack_q_lora_exact_reuse_allowed": False,
            "full_generation_kv_shared_claimed": False,
            "token_level_moe_claimed": False,
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
        expected_projector_segment_plan_schema_sha256=(
            PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
        ),
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


def _resign_release_after_partition_change(
    *,
    release_root: Path,
    dataset_root: Path,
    relative: str,
    authenticated: dict[str, str],
) -> str:
    partition = dataset_root / relative
    payload = partition.read_bytes()
    partition_sha = hashlib.sha256(payload).hexdigest()
    authenticated[relative] = partition_sha
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["fixed_files"] if item["path"] == relative)
    entry["sha256"] = partition_sha
    entry["bytes"] = len(payload)
    entry["records"] = len([line for line in payload.splitlines() if line.strip()])
    encoded = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    (release_root / "manifest.json").write_bytes(encoded)
    (release_root / "manifest.json.sha256").write_bytes(
        f"{digest}  manifest.json\n".encode("ascii")
    )
    return digest


def test_schema_pins_are_v2_and_content_addressed() -> None:
    assert (
        consumer.validate_release_lock_schema(SCHEMA_PATH)
        == consumer.RELEASE_LOCK_SCHEMA_SHA256
    )
    assert (
        consumer.validate_source_disjoint_schema(SOURCE_DISJOINT_SCHEMA_PATH)
        == consumer.SOURCE_DISJOINT_SCHEMA_SHA256
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
    assert result.segment_plan_schema_sha256 == PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
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


def test_v1_release_lock_is_explicitly_rejected(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        manifest["schema_version"] = "anchor.generic-train-release-lock.v1"

    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, mutate_manifest=mutate
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_schema_version_unsupported",
    ):
        _load(release_root, dataset_root, digest, authenticated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("segment_plan_location", "training_record.segment_plan"),
        ("exact_reuse_scope", "whole_generation"),
        ("q_specialization_alone_sufficient_for_exact_reuse", True),
        ("full_generation_kv_shared_claimed", True),
    ],
)
def test_release_hierarchical_task_kv_claim_drift_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        task_kv = manifest["hierarchical_task_kv"]
        assert isinstance(task_kv, dict)
        task_kv[field] = value

    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, mutate_manifest=mutate
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_hierarchical_task_kv_invalid",
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_release_segment_schema_binding_drift_fails_closed(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        bindings = manifest["bindings"]
        assert isinstance(bindings, dict)
        bindings["projector_segment_plan_schema_sha256"] = "0" * 64

    release_root, dataset_root, digest, authenticated = _write_release(
        tmp_path, mutate_manifest=mutate
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_claims_invalid"
    ):
        _load(release_root, dataset_root, digest, authenticated)


def test_release_partition_rejects_v1_and_tampered_segment_plan(tmp_path: Path) -> None:
    release_root, dataset_root, digest, authenticated = _write_release(tmp_path)
    relative = "train/clean.jsonl"
    partition = dataset_root / relative
    rows = [json.loads(line) for line in partition.read_text(encoding="utf-8").splitlines()]
    rows[0]["schema_version"] = "anchor.swebench-taskboard-sidecar.v1"
    _write_jsonl(partition, rows)
    digest = _resign_release_after_partition_change(
        release_root=release_root,
        dataset_root=dataset_root,
        relative=relative,
        authenticated=authenticated,
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError,
        match="release_lock_sidecar_schema_version_unsupported",
    ):
        _load(release_root, dataset_root, digest, authenticated)

    rows[0]["schema_version"] = consumer.SIDECAR_SCHEMA_VERSION
    rows[0]["segment_plan"]["bindings"]["task_bundle_sha256"] = "0" * 64
    _write_jsonl(partition, rows)
    digest = _resign_release_after_partition_change(
        release_root=release_root,
        dataset_root=dataset_root,
        relative=relative,
        authenticated=authenticated,
    )
    with pytest.raises(
        consumer.TrainingReleaseConsumerError, match="release_lock_partition_invalid"
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
            expected_projector_segment_plan_schema_sha256=(
                PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
            ),
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
            expected_projector_segment_plan_schema_sha256=(
                PROJECTOR_SEGMENT_PLAN_SCHEMA_SHA
            ),
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
