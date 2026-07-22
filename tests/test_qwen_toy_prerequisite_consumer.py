from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from anchor_mvp.research import qwen_toy_prerequisite_consumer as consumer
from anchor_mvp.research.qwen_toy_prerequisite_consumer import (
    QwenToyPrerequisiteConsumerError,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"


def _copy_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for relative in consumer.READ_WHITELIST:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    monkeypatch.setattr(consumer, "_REPOSITORY_ROOT", tmp_path)
    return tmp_path


def _snapshot(relative: str) -> consumer._BytesSnapshot:
    return consumer._read_bytes_snapshot(REPO, relative, "test_unreadable")


def test_metadata_authenticates_but_decision_is_always_blocked() -> None:
    decision = consumer.evaluate_toy_prerequisite(CONFIG)
    assert decision["status"] == "blocked"
    assert decision["diagnostic_metadata_verified"] is True
    assert decision["training_authorized"] is False
    assert decision["formal_training_authorized"] is False
    assert decision["zero_intersection_claimed"] is False
    assert decision["v1_attestation_emitted"] is False
    assert decision["protected_inventory_coverage"] == {
        "ready": 2,
        "total": 6,
        "ready_source_classes": ["swebench_source", "heldout"],
        "unavailable_source_classes": [
            "gold_partition",
            "partial_gold_export",
            "legacy_heldout_cases",
            "synthetic_scaffold",
        ],
    }
    assert decision["trigger_receipt"] == {
        "status": "pending_request_local_materialization",
        "bound_identity_count": 0,
        "token_ids_emitted": False,
        "planner_request1_private_kv_reused": False,
    }
    assert decision["audit"] == {
        "unique_metadata_files_read": 26,
        "hash_id_inventories_verified": 3,
        "toy_record_body_read": False,
        "protected_content_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }


def test_cli_prints_blocked_decision_and_returns_two(capsys) -> None:
    assert consumer.main(["--config", str(CONFIG)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["diagnostic_metadata_verified"] is True
    assert output["training_authorized"] is False


def test_config_and_fixed_producer_identities_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    consumer._validate_consumer_config(config)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == consumer.CONFIG_SHA256
    assert config["producer"]["commit"] == consumer.PRODUCER_COMMIT
    assert config["bindings"] == consumer._expected_bindings()

    drift = deepcopy(config)
    drift["inventory_contract"]["ready_count"] = 6
    with pytest.raises(
        QwenToyPrerequisiteConsumerError, match="inventory_contract_drift"
    ):
        consumer._validate_consumer_config(drift)
    drift = deepcopy(config)
    drift["policy"]["training_authorized"] = True
    with pytest.raises(QwenToyPrerequisiteConsumerError, match="policy_drift"):
        consumer._validate_consumer_config(drift)


def test_only_exact_metadata_whitelist_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = consumer._read_bytes_snapshot

    def recording_reader(root: Path, relative: str, code: str):
        observed.append(relative)
        return original(root, relative, code)

    monkeypatch.setattr(consumer, "_read_bytes_snapshot", recording_reader)
    decision = consumer.evaluate_toy_prerequisite(CONFIG)
    assert decision["audit"]["unique_metadata_files_read"] == 26
    assert set(observed) == set(consumer.READ_WHITELIST)
    assert all("toy/diagnostic.jsonl" not in path for path in observed)
    forbidden_fragments = (
        "data/automated_v3_shards",
        "artifacts/benchmark/heldout_v1",
        "configs/training/heldout_cases.jsonl",
        "datasets/public/swebench-full-bank-v1",
        "fixtures/research/swebench_natural_language_scaffold",
    )
    assert all(
        fragment not in path for path in observed for fragment in forbidden_fragments
    )


def test_ledger_rejects_diagnostic_and_protected_paths() -> None:
    ledger = consumer._ReadLedger(REPO)
    for path in (
        "fixtures/research/qwen_toy_prerequisite_v1/toy/diagnostic.jsonl",
        "artifacts/benchmark/heldout_v1/manifest.json",
    ):
        with pytest.raises(
            QwenToyPrerequisiteConsumerError, match="read_path_not_whitelisted"
        ):
            ledger.read(path, "must_not_read")


def test_packaged_metadata_fixture_excludes_toy_diagnostic_body() -> None:
    assert not (
        REPO / "fixtures/research/qwen_toy_prerequisite_v1/toy/diagnostic.jsonl"
    ).exists()


def test_three_hash_id_inventories_recompute_exact_logical_digests() -> None:
    cases = (
        (
            "fixtures/research/qwen_toy_prerequisite_v1/toy/source_ids.sha256.jsonl",
            consumer.TOY_SOURCE_IDS_FILE_SHA256,
            8,
            consumer.TOY_SOURCE_ID_INVENTORY_SHA256,
        ),
        (
            "fixtures/research/qwen_toy_prerequisite_v1/inventories/"
            "swebench_source/source_ids.sha256.jsonl",
            consumer.INVENTORY_BINDINGS["swebench_source"].source_ids_file_sha256,
            19008,
            consumer.INVENTORY_BINDINGS["swebench_source"].source_id_inventory_sha256,
        ),
        (
            "fixtures/research/qwen_toy_prerequisite_v1/inventories/"
            "heldout/source_ids.sha256.jsonl",
            consumer.INVENTORY_BINDINGS["heldout"].source_ids_file_sha256,
            6,
            consumer.INVENTORY_BINDINGS["heldout"].source_id_inventory_sha256,
        ),
    )
    for path, file_sha, count, logical_sha in cases:
        assert file_sha is not None and logical_sha is not None
        consumer._validate_hash_inventory(
            _snapshot(path),
            expected_file_sha256=file_sha,
            expected_count=count,
            expected_logical_sha256=logical_sha,
            code="test_inventory",
        )


def test_noncanonical_hash_inventory_is_rejected() -> None:
    good = _snapshot(
        "fixtures/research/qwen_toy_prerequisite_v1/toy/source_ids.sha256.jsonl"
    )
    bad_data = good.data + good.data.splitlines(keepends=True)[0]
    bad = replace(
        good,
        data=bad_data,
        sha256=hashlib.sha256(bad_data).hexdigest(),
        identity=(0, 0, len(bad_data), 0),
    )
    with pytest.raises(QwenToyPrerequisiteConsumerError, match="not_sorted_unique"):
        consumer._decode_hash_id_lines(bad, "duplicate")


def test_pending_trigger_cannot_mint_request_local_identity() -> None:
    main = json.loads(
        (REPO / "fixtures/research/qwen_toy_prerequisite_v1/manifest.json").read_text(
            "utf-8"
        )
    )
    trigger = deepcopy(main["request_local_trigger_binding"])
    schema = _snapshot(
        "configs/research/qwen_request_local_trigger_materialization.schema.json"
    )
    consumer._validate_pending_trigger(trigger, schema)
    trigger["tokenizer_binding_sha256"] = "0" * 64
    with pytest.raises(
        QwenToyPrerequisiteConsumerError,
        match="schema_validation_failed|unbound_identity_must_be_null",
    ):
        consumer._validate_pending_trigger(trigger, schema)


def test_nested_inventory_statuses_are_exact_and_unavailable_has_no_identity() -> None:
    schema = _snapshot("configs/research/protected_source_id_inventory.schema.json")
    for source_class in consumer.ORDERED_SOURCE_CLASSES:
        manifest = json.loads(
            (
                REPO
                / "fixtures/research/qwen_toy_prerequisite_v1/inventories"
                / source_class
                / "manifest.json"
            ).read_text("utf-8")
        )
        consumer._validate_instance(
            schema,
            consumer.INVENTORY_SCHEMA_SHA256,
            manifest,
            f"inventory_{source_class}",
        )
        assert manifest["status"] == consumer.INVENTORY_BINDINGS[source_class].status
        if manifest["status"] == "unavailable":
            assert "source_id_count" not in manifest
            assert "source_id_inventory_sha256" not in manifest
            assert "inventory_file" not in manifest


def test_missing_or_noncanonical_sidecar_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    sidecar = root / "fixtures/research/qwen_toy_prerequisite_v1/manifest.json.sha256"
    sidecar.unlink()
    with pytest.raises(QwenToyPrerequisiteConsumerError, match="sidecar_unreadable"):
        consumer.evaluate_toy_prerequisite(
            root / "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
        )

    _copy_contract(tmp_path, monkeypatch)
    sidecar.write_bytes(
        f"{consumer.MAIN_MANIFEST_SHA256}  manifest.json\r\n".encode("ascii")
    )
    with pytest.raises(
        QwenToyPrerequisiteConsumerError,
        match="physical_sha256_mismatch|noncanonical",
    ):
        consumer.evaluate_toy_prerequisite(
            root / "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
        )


def test_final_recheck_detects_concurrent_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = consumer._read_bytes_snapshot
    reads: dict[str, int] = {}

    def replacing_reader(root: Path, relative: str, code: str):
        snapshot = original(root, relative, code)
        reads[relative] = reads.get(relative, 0) + 1
        if relative == consumer.READ_WHITELIST[0] and reads[relative] == 2:
            changed = snapshot.data + b" "
            return replace(
                snapshot,
                data=changed,
                sha256=hashlib.sha256(changed).hexdigest(),
                identity=(
                    snapshot.identity[0],
                    snapshot.identity[1],
                    len(changed),
                    snapshot.identity[3] + 1,
                ),
            )
        return snapshot

    monkeypatch.setattr(consumer, "_read_bytes_snapshot", replacing_reader)
    with pytest.raises(
        QwenToyPrerequisiteConsumerError,
        match="artifact_changed_during_validation",
    ):
        consumer.evaluate_toy_prerequisite(CONFIG)


def test_config_byte_drift_fails_before_artifact_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    config = root / "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
    config.write_bytes(config.read_bytes() + b" ")
    with pytest.raises(
        QwenToyPrerequisiteConsumerError, match="config_sha256_mismatch"
    ):
        consumer.evaluate_toy_prerequisite(config)
