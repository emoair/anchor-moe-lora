from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import struct
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import qwen_request_local_trigger_receipt_v2 as receipt_v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / "qwen_request_local_trigger_receipt_v2.yaml"
CONFIG_SCHEMA = (
    ROOT
    / "configs"
    / "research"
    / "qwen_request_local_trigger_receipt_v2_config.schema.json"
)
RECEIPT_SCHEMA = (
    ROOT / "configs" / "research" / "qwen_request_local_trigger_receipt_v2.schema.json"
)
FIXTURE = (
    ROOT
    / "fixtures"
    / "research"
    / "qwen_request_local_trigger_receipt_v2"
    / "receipt.json"
)
SIDECAR = FIXTURE.with_name("receipt.json.sha256")
TF32_PROXY = (
    ROOT
    / "fixtures"
    / "research"
    / "qwen_alora_prefix_kv_diagnostic_tf32_v2"
    / "diagnostic_receipt.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture() -> dict:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_schemas_are_draft_2020_12_and_physically_bound() -> None:
    config_schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(receipt_schema)

    config, snapshots = receipt_v2.load_config(CONFIG)
    assert snapshots["config_schema"]["sha256"] == _sha256(CONFIG_SCHEMA)
    assert snapshots["receipt_schema"]["sha256"] == _sha256(RECEIPT_SCHEMA)
    assert config["schemas"]["config"]["sha256"] == _sha256(CONFIG_SCHEMA)
    assert config["schemas"]["receipt"]["sha256"] == _sha256(RECEIPT_SCHEMA)


def test_fixture_is_schema_valid_and_has_standard_mandatory_sidecar() -> None:
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(receipt_schema).validate(_fixture())

    receipt_sha = _sha256(FIXTURE)
    sidecar_raw = SIDECAR.read_bytes()
    assert sidecar_raw == f"{receipt_sha}  receipt.json\n".encode("ascii")
    assert re.fullmatch(rb"[0-9a-f]{64}  receipt\.json\n", sidecar_raw)


def test_receipt_is_exactly_the_authenticated_44_token_probe() -> None:
    materialization = _fixture()["request2_materialization"]
    assert materialization["total_tokens"] == 44
    assert materialization["trigger_span_zero_based_exclusive"] == {
        "index_base": "zero",
        "end_semantics": "exclusive",
        "start": 25,
        "end": 33,
    }
    assert materialization["trigger_span_width"] == 8
    assert 0 <= 25 < 33 <= 44
    assert materialization["complete_r2_tokenization_count"] == 1
    assert materialization["trigger_text_occurrences"] == 1
    assert materialization["trigger_token_sequence_occurrences"] == 1


def test_boundary_overhang_records_real_bytes_and_codepoints() -> None:
    materialization = _fixture()["request2_materialization"]
    assert materialization["boundary_overhang"] == {
        "leading_utf8_bytes": 0,
        "trailing_utf8_bytes": 1,
        "leading_codepoints": 0,
        "trailing_codepoints": 1,
    }


def test_ordered_digest_algorithm_is_signed_int64_big_endian_concat() -> None:
    values = [0, 1, -1, 2**31, -(2**31)]
    expected = hashlib.sha256(b"".join(struct.pack(">q", item) for item in values))
    assert receipt_v2.ordered_token_ids_sha256(values) == expected.hexdigest()
    assert (
        _fixture()["request2_materialization"][
            "ordered_input_token_ids_digest_algorithm"
        ]
        == "sha256_concat_signed_int64_big_endian_v1"
    )


@pytest.mark.parametrize("value", [True, -(2**63) - 1, 2**63])
def test_ordered_digest_rejects_non_int64_values(value: object) -> None:
    with pytest.raises(receipt_v2.TriggerReceiptError):
        receipt_v2.ordered_token_ids_sha256([value])  # type: ignore[list-item]


def test_tokenizer_binding_digest_recomputes_from_canonical_object() -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    receipt = _fixture()
    assert receipt_v2.tokenizer_binding_sha256(config) == (
        "a76b0f60e5c1e2d92b8a8d9131f9afe9edfda3fcbf0221c4234359f70e806425"
    )
    assert receipt["tokenizer_binding_sha256"] == receipt_v2._sha256(
        receipt_v2._canonical_json_bytes(receipt["tokenizer_binding"])
    )


def test_receipt_binds_source_config_schemas_implementation_and_proxy() -> None:
    receipt = _fixture()
    assert receipt["source_bindings"] == {
        "producer_commit": "744e23f975b13923903f5fabe04c32e74ea25dc4",
        "consumer_baseline_commit": "b0441e6beaa07b180d7fc69e462b4d2babf21792",
        "consumer_baseline_semantics": "required_ancestor_or_equal_dependency",
    }
    producer = receipt["producer"]
    assert producer["config"]["sha256"] == _sha256(CONFIG)
    assert producer["config_schema"]["sha256"] == _sha256(CONFIG_SCHEMA)
    assert producer["receipt_schema"]["sha256"] == _sha256(RECEIPT_SCHEMA)
    assert producer["implementation"]["sha256"] == _sha256(
        ROOT
        / "src"
        / "anchor_mvp"
        / "research"
        / "qwen_request_local_trigger_receipt_v2.py"
    )
    assert receipt["tf32_proxy_source"]["sha256"] == _sha256(TF32_PROXY)


def test_imported_implementation_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    monkeypatch.setattr(receipt_v2, "_IMPORTED_IMPLEMENTATION_SHA256", "0" * 64)
    with pytest.raises(receipt_v2.TriggerReceiptError, match="module imported"):
        receipt_v2._implementation_snapshot(config)


def test_scope_remains_diagnostic_only_and_zero_request() -> None:
    receipt = _fixture()
    assert receipt["status"] == "ready_diagnostic_only"
    assert receipt["claims"] == {
        "diagnostic_only": True,
        "formal": False,
        "training_authorized": False,
        "numeric_equivalence": False,
        "thresholds_formal": False,
        "proxy_signal_passed": True,
        "quality_validated": False,
        "physical_kv_claimed": False,
        "multistream_claimed": False,
    }
    assert receipt["audit"] == {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "model_weight_files_read": False,
        "gguf_files_read": False,
        "dataset_reads": 0,
        "canonical_gold_read": False,
        "heldout_content_read": False,
        "scaffold_content_read": False,
    }


def test_receipt_does_not_emit_raw_ids_or_a_global_index() -> None:
    receipt = _fixture()
    serialized = FIXTURE.read_text(encoding="utf-8")
    materialization = receipt["request2_materialization"]
    assert materialization["raw_token_ids_emitted"] is False
    assert materialization["global_token_index_emitted"] is False
    assert '"input_ids":' not in serialized
    assert '"token_ids":' not in serialized
    assert '"global_token_index":' not in serialized
    assert not any(isinstance(value, list) for value in materialization.values())


def test_materialization_drift_fails_closed() -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    observed = copy.deepcopy(_fixture()["request2_materialization"])
    observed["total_tokens"] = 45
    with pytest.raises(receipt_v2.TriggerReceiptError, match="total_tokens"):
        receipt_v2._validate_expected_materialization(config, observed)


def test_source_commit_drift_is_rejected_by_schema() -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    config = copy.deepcopy(config)
    config["source_bindings"]["consumer_baseline_commit"] = "0" * 40
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(config))
    assert errors


def test_source_baseline_semantics_drift_is_rejected_by_schema() -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    config = copy.deepcopy(config)
    config["source_bindings"]["consumer_baseline_semantics"] = "current_head_equal"
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(config))
    assert errors


def test_consumer_baseline_is_an_ancestor_dependency_not_a_head_equality_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_head = "1" * 40
    calls: list[tuple[str, ...]] = []

    def fake_git_run(*arguments: str):
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments], 0, stdout=f"{descendant_head}\n", stderr=""
            )
        if arguments == (
            "rev-parse",
            "--verify",
            f"{receipt_v2.EXPECTED_CONSUMER_BASELINE}^{{commit}}",
        ):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments],
                0,
                stdout=f"{receipt_v2.EXPECTED_CONSUMER_BASELINE}\n",
                stderr="",
            )
        if arguments == (
            "merge-base",
            "--is-ancestor",
            receipt_v2.EXPECTED_CONSUMER_BASELINE,
            descendant_head,
        ):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments], 0, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected git call: {arguments!r}")

    monkeypatch.setattr(receipt_v2, "_git_run", fake_git_run)
    receipt_v2._authenticate_consumer_baseline()
    assert calls[-1][0:2] == ("merge-base", "--is-ancestor")


def test_consumer_dependency_not_in_head_lineage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_head = "1" * 40

    def fake_git_run(*arguments: str):
        if arguments == ("rev-parse", "HEAD"):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments], 0, stdout=f"{descendant_head}\n", stderr=""
            )
        if arguments[0:2] == ("rev-parse", "--verify"):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments],
                0,
                stdout=f"{receipt_v2.EXPECTED_CONSUMER_BASELINE}\n",
                stderr="",
            )
        if arguments[0:2] == ("merge-base", "--is-ancestor"):
            return receipt_v2.subprocess.CompletedProcess(
                ["git", *arguments], 1, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected git call: {arguments!r}")

    monkeypatch.setattr(receipt_v2, "_git_run", fake_git_run)
    with pytest.raises(receipt_v2.TriggerReceiptError, match="not an ancestor"):
        receipt_v2._authenticate_consumer_baseline()


def test_publication_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    output = tmp_path / "occupied" / "receipt.json"
    output.parent.mkdir()
    with pytest.raises(receipt_v2.TriggerReceiptError, match="already exists"):
        receipt_v2._publish_pair(config, {}, _fixture(), output=output)


def test_atomic_publish_cleans_private_temp_on_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshots = receipt_v2.load_config(CONFIG)
    output = tmp_path / "diagnostics" / "request-v2" / "receipt.json"
    output.parent.parent.mkdir()

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("injected failure before atomic directory publish")

    monkeypatch.setattr(receipt_v2.common, "_rename_noreplace", fail_rename)
    with pytest.raises(OSError, match="injected failure"):
        receipt_v2._publish_pair(config, snapshots, _fixture(), output=output)
    assert not output.parent.exists()
    assert list(output.parent.parent.iterdir()) == []


def test_atomic_publish_refuses_sidecar_only_preexisting_directory(
    tmp_path: Path,
) -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    output = tmp_path / "diagnostics" / "request-v2" / "receipt.json"
    output.parent.mkdir(parents=True)
    sidecar = output.with_name("receipt.json.sha256")
    sidecar.write_bytes(b"sentinel\n")
    with pytest.raises(receipt_v2.TriggerReceiptError, match="already exists"):
        receipt_v2._publish_pair(config, {}, _fixture(), output=output)
    assert not output.exists()
    assert sidecar.read_bytes() == b"sentinel\n"


def test_atomic_publish_rejects_symlink_or_reparse_ancestor(
    tmp_path: Path,
) -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(physical, linked, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    output = linked / "request-v2" / "receipt.json"
    with pytest.raises(
        receipt_v2.TriggerReceiptError,
        match="realpath drift|symlink|reparse",
    ):
        receipt_v2._publish_pair(config, {}, _fixture(), output=output)
    assert list(physical.iterdir()) == []


def test_byte_identical_offline_rebuild(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    output = tmp_path / "rebuilt" / "receipt.json"
    report = receipt_v2.materialize(CONFIG, output=output)
    assert report["total_tokens"] == 44
    assert report["provider_requests"] == 0
    assert report["network_requests"] == 0
    assert report["model_loads"] == 0
    assert report["gpu_requests"] == 0
    assert report["training_authorized"] is False
    assert output.read_bytes() == FIXTURE.read_bytes()
    assert output.with_name("receipt.json.sha256").read_bytes() == SIDECAR.read_bytes()


def test_config_has_no_protected_dataset_body_input() -> None:
    config, _snapshots = receipt_v2.load_config(CONFIG)
    paths = [
        config["schemas"]["config"]["path"],
        config["schemas"]["receipt"]["path"],
        config["implementation"]["path"],
        config["tf32_proxy_source"]["path"],
    ]
    lowered = "\n".join(paths).lower()
    assert "gold" not in lowered
    assert "heldout" not in lowered
    assert "swebench_natural_language_scaffold" not in lowered
    assert not any(path.endswith(".jsonl") for path in paths)
