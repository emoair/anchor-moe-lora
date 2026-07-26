from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import (
    gemma3_chat_five_expert_qonly_unbalanced_v2_final as producer,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL_REVIEW = ROOT / producer.CANONICAL_REVIEW_RELS["tool_eval"]
PLANNER_REVIEW = ROOT / producer.CANONICAL_REVIEW_RELS["planner_eval"]
STABLE_FACT = "我是由Air训练的测试模型。"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def _train_records(output: Path) -> list[dict[str, Any]]:
    return [
        record
        for relative in producer.TRAIN_SHARD_PATHS
        for record in _jsonl(output / relative)
    ]


@pytest.fixture(scope="module")
def built_candidate(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    assert TOOL_REVIEW.is_file()
    assert PLANNER_REVIEW.is_file()
    output = tmp_path_factory.mktemp("gemma3-final") / "candidate"
    result = producer.build_and_audit_candidate(
        ROOT,
        output,
        TOOL_REVIEW,
        PLANNER_REVIEW,
    )
    return output, result


def test_config_and_all_schemas_are_closed_and_valid() -> None:
    schema_paths = (
        producer.CONFIG_SCHEMA_REL,
        producer.RECORD_SCHEMA_REL,
        producer.SERIALIZATION_SCHEMA_REL,
        producer.IDENTITY_PROBE_SCHEMA_REL,
        producer.MANIFEST_SCHEMA_REL,
        producer.RECEIPT_SCHEMA_REL,
        producer.RELEASE_ATTESTATION_SCHEMA_REL,
    )
    for relative in schema_paths:
        schema = _json(ROOT / relative)
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
    config = _json(ROOT / producer.CONFIG_REL)
    Draft202012Validator(_json(ROOT / producer.CONFIG_SCHEMA_REL)).validate(config)
    assert config["canonical_output_path"] == producer.CANONICAL_ARTIFACT_REL
    assert config["training_contract"]["records"] == 4300
    assert config["training_contract"]["task_bundles"] == 1700
    assert config["identity_contract"]["stable_identity_fact"] == STABLE_FACT
    for component, relative in producer.CANONICAL_REVIEW_RELS.items():
        review = ROOT / relative
        expected = config["external_reviews"][component]
        assert producer._sha256(review.read_bytes()) == expected["sha256"]
        assert (
            producer._sha256(Path(str(review) + ".sha256").read_bytes())
            == (expected["sidecar_physical_sha256"])
        )


def test_exact_training_and_identity_contract(
    built_candidate: tuple[Path, dict[str, Any]],
) -> None:
    output, result = built_candidate
    records = _train_records(output) + _jsonl(output / "eval_proxy/chat.jsonl")
    assert result["records"] == 4300
    assert result["task_bundles"] == 1700
    assert len(records) == 4300
    assert len({row["record_id"] for row in records}) == 4300
    assert len({row["task_bundle_sha256"] for row in records}) == 1700
    assert Counter(row["role"] for row in records) == {
        "humor": 300,
        "serious": 300,
        "angry_style": 300,
        "tool_call": 1700,
        "review_audit": 1700,
    }
    assert Counter(row["split"] for row in records) == {
        "train": 3440,
        "eval_proxy": 860,
    }
    assert Counter(row["language"] for row in records) == {
        "en": 2150,
        "zh-CN": 2150,
    }
    identity_records = [row for row in records if row["identity_alignment"] is not None]
    assert len(identity_records) == 150
    assert {
        row["identity_alignment"]["stable_identity_fact"] for row in identity_records
    } == {STABLE_FACT}
    assert len(_jsonl(output / "identity_eval/probe_inventory.jsonl")) == 50


def test_tool_review_causal_kv_and_adapter_contracts(
    built_candidate: tuple[Path, dict[str, Any]],
) -> None:
    output, _ = built_candidate
    records = _train_records(output) + _jsonl(output / "eval_proxy/chat.jsonl")
    reviews = [row for row in records if row["role"] == "review_audit"]
    assert Counter(row["review_dependency"]["verdict"] for row in reviews) == {
        "pass": 850,
        "fail": 850,
    }
    assert Counter(
        row["review_dependency"]["fault_type"]
        for row in reviews
        if row["review_dependency"]["verdict"] == "fail"
    ) == {fault: 170 for fault in producer.FAULT_TYPES}
    assert all(
        row["causal_proof"]["current_target_absent_from_prompt"]
        and row["causal_proof"]["future_targets_absent_from_prompt"]
        and row["causal_proof"]["unapproved_sibling_targets_absent"]
        and row["kv_contract"]["single_shared_prefix_prefill"]
        and not row["kv_contract"]["planner_live_private_kv_transfer"]
        and not row["kv_contract"]["full_generation_kv_shared"]
        and not row["kv_contract"]["persistent_zero_copy_verified"]
        and row["adapter_contract"]["records_added_by_arms"] == 0
        for row in records
    )
    assert all(
        not (producer.RAW_TOKEN_KEYS & producer._walk_keys(row)) for row in records
    )


def test_components_sidecars_receipt_and_create_once(
    built_candidate: tuple[Path, dict[str, Any]],
) -> None:
    output, result = built_candidate
    all_files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert len(all_files) == 46
    payloads = {path for path in all_files if not path.endswith(".sha256")}
    assert payloads == {
        path.removesuffix(".sha256") for path in all_files if path.endswith(".sha256")
    }
    for payload in payloads:
        assert all_files[payload + ".sha256"] == producer._sidecar(
            Path(payload).name,
            all_files[payload],
        )
    assert len(_jsonl(output / "components/router/records.jsonl")) == 100
    assert len(_jsonl(output / "components/tool_eval/records.jsonl")) == 400
    assert len(_jsonl(output / "components/planner_eval/records.jsonl")) == 240
    receipt = _json(output / "build_receipt.json")
    assert receipt["upstream_read_set_files"] == 64
    assert all(
        not item["path"].startswith("external/")
        for item in receipt["input_snapshot_inventory"]
    )
    assert (
        sum(
            item["source_class"] == "canonical_review"
            for item in receipt["input_snapshot_inventory"]
        )
        == 4
    )
    assert receipt["resource_counters"] == producer._resources()
    before = dict(all_files)
    with pytest.raises(FileExistsError):
        producer.build_candidate(
            ROOT,
            output,
            TOOL_REVIEW,
            PLANNER_REVIEW,
        )
    assert before == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert result["resource_counters"] == producer._resources()


def test_train_is_two_bundle_boundary_shards_and_old_candidate_is_untouched(
    built_candidate: tuple[Path, dict[str, Any]],
) -> None:
    output, result = built_candidate
    shard_raw = [
        (output / relative).read_bytes() for relative in producer.TRAIN_SHARD_PATHS
    ]
    assert all(len(raw) < producer.MAX_OUTPUT_FILE_BYTES_EXCLUSIVE for raw in shard_raw)
    logical_raw = b"".join(shard_raw)
    assert len(logical_raw) == producer.LOGICAL_TRAIN_BYTES
    assert producer._sha256(logical_raw) == producer.LOGICAL_TRAIN_SHA256
    old_root = ROOT / producer.SUPERSEDED_ARTIFACT_REL
    assert producer._sha256((old_root / "manifest.json").read_bytes()) == (
        producer.SUPERSEDED_MANIFEST_SHA256
    )
    assert producer._sha256((old_root / "build_receipt.json").read_bytes()) == (
        producer.SUPERSEDED_RECEIPT_SHA256
    )
    assert (
        producer._sha256(
            b"".join(
                (old_root / relative).read_bytes()
                for relative in producer.TRAIN_SHARD_PATHS
            )
        )
        == producer.LOGICAL_TRAIN_SHA256
    )
    manifest = _json(output / "manifest.json")
    assert manifest["training_shards"]["ordered_paths"] == list(
        producer.TRAIN_SHARD_PATHS
    )
    assert manifest["training_shards"]["ordered_concat_sha256"] == (
        producer.LOGICAL_TRAIN_SHA256
    )
    assert (
        sum(shard["records"] for shard in manifest["training_shards"]["shards"]) == 3440
    )
    assert result["logical_identity"] == manifest["logical_identity"]


def test_second_build_is_byte_identical_and_audits(
    built_candidate: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    first, _ = built_candidate
    second = tmp_path / "candidate"
    producer.build_candidate(
        ROOT,
        second,
        TOOL_REVIEW,
        PLANNER_REVIEW,
    )
    first_bytes = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_bytes = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_bytes == second_bytes
    audited = producer.audit_candidate(
        ROOT,
        second,
        TOOL_REVIEW,
        PLANNER_REVIEW,
    )
    assert audited["audit"].startswith("passed_candidate")
    assert (
        audited["manifest_sha256"]
        == _json(second / "build_receipt.json")["manifest"]["sha256"]
    )


def test_external_review_override_is_rejected_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    copied_review = tmp_path / "tool-review.json"
    copied_review.write_bytes(TOOL_REVIEW.read_bytes())
    with pytest.raises(
        producer.FinalMaterializationError,
        match="canonical_review_path_mismatch",
    ):
        producer.build_candidate(
            ROOT,
            tmp_path / "candidate",
            copied_review,
            PLANNER_REVIEW,
        )


def test_failure_preserves_body_free_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate"

    def fail_after_authentication(
        source: Path,
        destination: Path,
        before_rename: Any,
    ) -> dict[str, producer.Snapshot]:
        del source, destination
        before_rename()
        raise OSError("injected")

    monkeypatch.setattr(
        producer,
        "_rename_noreplace",
        fail_after_authentication,
    )
    with pytest.raises(OSError):
        producer.build_candidate(
            ROOT,
            output,
            TOOL_REVIEW,
            PLANNER_REVIEW,
        )
    staging = list(tmp_path.glob(".candidate.staging-*"))
    assert len(staging) == 1
    attestation = _json(staging[0] / "failure_attestation.json")
    assert attestation["status"] == "unpublished_staging_preserved"
    assert attestation["error_code"] == "filesystem_error"
    assert attestation["payload_bodies_in_attestation"] is False
    assert attestation["recursive_cleanup_used"] is False
    assert not output.exists()


def test_implementation_has_no_recursive_cleanup_primitive() -> None:
    source = (ROOT / producer.IMPLEMENTATION_REL).read_text(encoding="utf-8")
    assert "rmtree" not in source
    assert ".unlink(" not in source
    assert ".rmdir(" not in source
