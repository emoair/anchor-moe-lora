from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import (
    gemma3_chat_five_expert_qonly_unbalanced_v2_release as release,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / release.CANONICAL_ARTIFACT_REL


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_release_attestation_schema_is_closed_draft_2020_12() -> None:
    schema = _json(ROOT / release.SCHEMA_REL)
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["claims"]["$ref"] == "#/$defs/claims"
    release._require_explicit_lf_attributes(ROOT, release.RELEASE_ATTESTATION_REL)
    release._require_explicit_lf_attributes(
        ROOT,
        release.RELEASE_ATTESTATION_REL + ".sha256",
    )


def test_sharded_candidate_authenticates_but_remains_non_authorizing() -> None:
    snapshots = release._snapshot_artifact(ARTIFACT)
    manifest, receipt, shards = release._authenticate_artifact(
        snapshots,
        release._schema(release._snapshot(ROOT / release.MANIFEST_SCHEMA_REL)),
        release._schema(release._snapshot(ROOT / release.RECEIPT_SCHEMA_REL)),
    )
    assert len(snapshots) == 46
    assert len(shards) == 2
    assert sum(item["records"] for item in shards) == 3440
    assert manifest["claims"]["candidate"] is True
    assert manifest["claims"]["final"] is False
    assert manifest["claims"]["training_authorized"] is False
    assert receipt["logical_identity"] == manifest["logical_identity"]


def test_producer_cannot_self_attest(tmp_path: Path) -> None:
    with pytest.raises(
        release.IndependentReleaseError,
        match="reviewer_separation_invalid",
    ):
        release.review_release(
            ROOT,
            ARTIFACT,
            tmp_path / "attestation.json",
            producer_candidate_commit="0" * 40,
            producer_id="producer.agent",
            reviewer_id="producer.agent",
        )


def test_release_attestation_requires_canonical_repo_path(tmp_path: Path) -> None:
    with pytest.raises(
        release.IndependentReleaseError,
        match="canonical_attestation_path_required",
    ):
        release.review_release(
            ROOT,
            ARTIFACT,
            tmp_path / "attestation.json",
            producer_candidate_commit="0" * 40,
            producer_id="producer.agent",
            reviewer_id="reviewer.agent",
        )


def test_attestation_write_is_create_once(tmp_path: Path) -> None:
    target = tmp_path / "attestation.json"
    raw = b'{"audit_status":"test-only"}\n'
    release._write_exclusive(target, raw)
    with pytest.raises(release.IndependentReleaseError):
        release._write_exclusive(target, raw)
    assert target.read_bytes() == raw


def _publish_pair_for_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, bytes]:
    monkeypatch.setattr(
        release,
        "_require_explicit_lf_attributes",
        lambda *_: None,
    )
    destination = tmp_path / "release" / "independent_release_attestation.json"
    raw = b'{"audit_status":"test-only"}\n'
    return destination, raw


def test_attestation_pair_publishes_by_one_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, raw = _publish_pair_for_test(tmp_path, monkeypatch)
    published = release._publish_attestation_pair(
        tmp_path,
        destination,
        raw,
        before_rename=lambda: None,
    )
    assert set(published) == {
        destination.name,
        destination.name + ".sha256",
    }
    assert destination.read_bytes() == raw
    assert Path(str(destination) + ".sha256").read_bytes() == release._sidecar(
        destination.name,
        raw,
    )
    assert not list(tmp_path.glob(".release.staging-*"))


def test_preoccupied_release_directory_or_single_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, raw = _publish_pair_for_test(tmp_path, monkeypatch)
    destination.parent.mkdir()
    destination.write_bytes(raw)
    with pytest.raises(FileExistsError, match="canonical_release_directory_exists"):
        release._publish_attestation_pair(
            tmp_path,
            destination,
            raw,
            before_rename=lambda: None,
        )
    assert destination.read_bytes() == raw
    assert not Path(str(destination) + ".sha256").exists()
    assert not list(tmp_path.glob(".release.staging-*"))


def test_sidecar_failure_preserves_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, raw = _publish_pair_for_test(tmp_path, monkeypatch)
    original = release._write_exclusive

    def fail_sidecar(path: Path, payload: bytes) -> None:
        if path.name.endswith(".sha256"):
            raise release.IndependentReleaseError("injected_sidecar_failure")
        original(path, payload)

    monkeypatch.setattr(release, "_write_exclusive", fail_sidecar)
    with pytest.raises(
        release.IndependentReleaseError,
        match="injected_sidecar_failure",
    ):
        release._publish_attestation_pair(
            tmp_path,
            destination,
            raw,
            before_rename=lambda: None,
        )
    assert not destination.parent.exists()
    staging = list(tmp_path.glob(".release.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / destination.name).read_bytes() == raw
    assert not (staging[0] / (destination.name + ".sha256")).exists()


def test_staging_collision_and_rename_race_never_publish_partial_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, raw = _publish_pair_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(release.secrets, "token_hex", lambda _: "fixed")
    collision = tmp_path / ".release.staging-fixed"
    collision.mkdir()
    with pytest.raises(
        release.IndependentReleaseError,
        match="attestation_staging_nonce_collision",
    ):
        release._publish_attestation_pair(
            tmp_path,
            destination,
            raw,
            before_rename=lambda: None,
        )
    assert not destination.parent.exists()

    other_root = tmp_path / "race"
    other_root.mkdir()
    race_destination, race_raw = _publish_pair_for_test(other_root, monkeypatch)

    def lose_race(source: Path, target: Path, before_rename: Any) -> Any:
        del source
        before_rename()
        target.mkdir()
        raise FileExistsError("canonical_release_directory_exists")

    monkeypatch.setattr(release, "_rename_directory_noreplace", lose_race)
    with pytest.raises(FileExistsError, match="canonical_release_directory_exists"):
        release._publish_attestation_pair(
            other_root,
            race_destination,
            race_raw,
            before_rename=lambda: None,
        )
    assert race_destination.parent.is_dir()
    assert not race_destination.exists()
    assert not Path(str(race_destination) + ".sha256").exists()
