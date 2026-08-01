from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import (
    gemma3_existing_distillation_v14_validation_asset_v1 as asset,
)


FIXTURE = (
    asset.REPO_ROOT
    / "fixtures/research/gemma3_existing_distillation_v14_validation_asset_v1"
)


def _record(split: str, index: int) -> dict[str, object]:
    digest = lambda domain: hashlib.sha256(  # noqa: E731
        f"{domain}:{split}:{index}".encode()
    ).hexdigest()
    return {
        "schema_version": asset.RECORD_VERSION,
        "source_ordinal": index + 1,
        "sample_id": digest("sample"),
        "split": split,
        "source_group_sha256": digest("source"),
        "leakage_family_sha256": digest("leakage"),
        "prompt_sha256": digest("prompt"),
        "consumable_for_training": False,
    }


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, object]]]:
    monkeypatch.setattr(asset, "EXPECTED_COUNTS", {split: 2 for split in asset.SPLITS})
    return {
        split: sorted(
            [_record(split, 0), _record(split, 1)], key=lambda row: row["sample_id"]
        )
        for split in asset.SPLITS
    }


@pytest.fixture(scope="module")
def record_schema() -> object:
    return json.loads(
        (
            asset.REPO_ROOT
            / "configs/research/gemma3_existing_distillation_v14_validation_record_v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def test_committed_asset_audits() -> None:
    result = asset.audit_asset()
    assert result["status"] == "passed_validation_only"
    assert result["counts"] == {
        "train": 13131,
        "calibration": 1923,
        "holdout": 1578,
        "total": 16632,
    }
    assert result["consumable_for_training"] is False


def test_all_schemas_are_closed_draft_2020_12() -> None:
    config, snapshots = asset.load_config()
    assert config["claims"]["training_authorized"] is False
    for name in ("config_schema", "record_schema", "manifest_schema", "receipt_schema"):
        schema = json.loads(snapshots[name].data.decode("utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


def test_partitions_contain_ids_only() -> None:
    for split in asset.SPLITS:
        path = FIXTURE / f"{split}.ids.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) == asset.OUTPUT_RECORD_KEYS
            assert not (set(row) & asset.FORBIDDEN_PERSISTED_KEYS)
            assert row["consumable_for_training"] is False


def test_training_pass_is_distinct_from_failed_attempt() -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence"]["training_stage_status"] == "passed"
    assert manifest["evidence"]["overall_attempt_status"] == "failed"
    assert manifest["evidence"]["attempt_consumed"] is False
    assert manifest["evidence"]["reason_codes"] == list(asset.REASON_CODES)


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    (copied / "holdout.ids.jsonl.sha256").unlink()
    with pytest.raises(asset.ValidationAssetError, match="fixture_inventory_invalid"):
        asset.audit_asset(fixture_root=copied)


def test_tampered_partition_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    with (copied / "train.ids.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(asset.ValidationAssetError, match="sidecar_invalid"):
        asset.audit_asset(fixture_root=copied)


def test_reordered_partition_fails(
    tiny: dict[str, list[dict[str, object]]], record_schema: object
) -> None:
    tiny["train"].reverse()
    with pytest.raises(
        asset.ValidationAssetError, match="partition_order_or_duplicate_invalid"
    ):
        asset._validate_partition_records(tiny, record_schema)


def test_duplicate_sample_id_fails(
    tiny: dict[str, list[dict[str, object]]], record_schema: object
) -> None:
    tiny["calibration"][1]["sample_id"] = tiny["train"][0]["sample_id"]
    tiny["calibration"].sort(key=lambda row: row["sample_id"])
    with pytest.raises(asset.ValidationAssetError, match="sample_id_duplicate"):
        asset._validate_partition_records(tiny, record_schema)


@pytest.mark.parametrize(
    "domain", ["source_group_sha256", "leakage_family_sha256", "prompt_sha256"]
)
def test_cross_split_overlap_fails(
    tiny: dict[str, list[dict[str, object]]], record_schema: object, domain: str
) -> None:
    tiny["holdout"][0][domain] = tiny["train"][0][domain]
    with pytest.raises(
        asset.ValidationAssetError, match=f"{domain}_cross_split_overlap"
    ):
        asset._validate_partition_records(tiny, record_schema)


def test_training_use_claim_fails(
    tiny: dict[str, list[dict[str, object]]], record_schema: object
) -> None:
    tiny["train"][0]["consumable_for_training"] = True
    with pytest.raises(
        asset.ValidationAssetError, match="record_schema_validation_failed"
    ):
        asset._validate_partition_records(tiny, record_schema)


@pytest.mark.parametrize("field", ["prompt", "labels", "target", "raw_token_ids"])
def test_body_field_leakage_fails(
    tiny: dict[str, list[dict[str, object]]], record_schema: object, field: str
) -> None:
    tiny["train"][0][field] = "forbidden"
    with pytest.raises(
        asset.ValidationAssetError, match="record_schema_validation_failed"
    ):
        asset._validate_partition_records(tiny, record_schema)


def test_atomic_publish_detects_staging_mutation(tmp_path: Path) -> None:
    output = tmp_path / "final"

    def mutate(staging: Path) -> None:
        (staging / "a.json").write_bytes(b"changed")

    with pytest.raises(asset.ValidationAssetError, match="staging_toctou_invalid"):
        asset._publish_payloads(output, {"a.json": b"original"}, before_publish=mutate)
    assert not output.exists()


def test_atomic_publish_is_no_replace(tmp_path: Path) -> None:
    output = tmp_path / "final"

    def compete(_: Path) -> None:
        output.mkdir()

    with pytest.raises(asset.ValidationAssetError, match="output_already_exists"):
        asset._publish_payloads(output, {"a.json": b"original"}, before_publish=compete)
    assert list(output.iterdir()) == []


def test_reparse_fixture_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "linked"
    try:
        link.symlink_to(FIXTURE, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(asset.ValidationAssetError, match="fixture_reparse_invalid"):
        asset.audit_asset(fixture_root=link)
