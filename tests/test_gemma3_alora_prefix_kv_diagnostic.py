from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import gemma3_alora_prefix_kv_diagnostic as diag


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / diag.CONFIG_NAME
CONFIG_SCHEMA = ROOT / "configs" / "research" / diag.CONFIG_SCHEMA_NAME
RECEIPT_SCHEMA = ROOT / "configs" / "research" / diag.RECEIPT_SCHEMA_NAME
FIXTURE = (
    ROOT
    / "fixtures"
    / "research"
    / "gemma3_1b_it_alora_prefix_kv_tf32_v1"
    / "diagnostic_receipt.json"
)


def loaded() -> dict:
    return diag.load_config(CONFIG)


def test_contract_is_strict_and_does_not_import_ml(capsys) -> None:
    before = {name: sys.modules.get(name) for name in diag._ML_MODULES}
    assert diag.main(["--config", str(CONFIG)]) == 0
    assert {name: sys.modules.get(name) for name in diag._ML_MODULES} == before
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == diag.CONFIG_VERSION
    assert report["profile_id"] == diag.PROFILE_ID
    assert report["model_loaded"] is False
    assert report["gpu_touched"] is False
    assert report["dataset_inputs"] == 0
    assert report["training_authorized"] is False


def test_schemas_and_export_identity_are_bound() -> None:
    config_schema = json.loads(CONFIG_SCHEMA.read_text("utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(receipt_schema)
    config = loaded()
    assert config["model"]["exporter"] == (
        "keras_hub.models.Gemma3CausalLM.export_to_transformers"
    )
    assert config["tokenizer"]["backend"] == ("sentencepiece_direct_immutable_proto")
    assert config["tokenizer"]["explicit_bos_id"] == 2
    assert config["tokenizer"]["chat_template_bound"] is False
    assert [item["path"] for item in config["model"]["assets"]] == [
        "config.json",
        "model.safetensors",
        "EXPORT_MANIFEST.json",
    ]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "profile_id", "qwen_profile"),
        ("claim_scope", "formal", True),
        ("numerics", "tf32", False),
        ("tokenizer", "explicit_bos_id", 1),
        ("model", "exporter", "keras_hub.Backbone.export_to_transformers"),
    ],
)
def test_contract_drift_fails_closed(
    section: str | None, field: str, value: object
) -> None:
    config = copy.deepcopy(diag._public_config(loaded()))
    if section is None:
        config[field] = value
    else:
        config[section][field] = value
    with pytest.raises(diag.DiagnosticError):
        diag.validate_config(config)


def test_non_formal_gates_require_every_mechanical_signal() -> None:
    config = loaded()
    metrics = {
        "paired_differential_max_abs": 0.0,
        "adapter_effect_max_abs": 1.0,
        "missing_trigger_effect_max_abs": 0.0,
        "prefix_kv_bit_equal": True,
        "finite": True,
        "shape_equal": True,
        "argmax_equal": True,
        "greedy_equal": True,
        "peak_allocated_mib": 4000.0,
    }
    assert diag.evaluate_gates(metrics, config)["all_passed"] is True
    metrics["prefix_kv_bit_equal"] = False
    assert diag.evaluate_gates(metrics, config)["all_passed"] is False


def test_published_gpu_receipt_is_schema_valid_and_cautious() -> None:
    raw = FIXTURE.read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    Draft202012Validator(schema).validate(receipt)
    digest = diag.common._sha256_bytes(raw)
    assert (
        FIXTURE.with_suffix(".json.sha256").read_text("utf-8")
        == f"{digest}  diagnostic_receipt.json\n"
    )
    assert receipt["config_sha256"] == diag.config_sha256(loaded())
    assert (
        receipt["config_file_sha256"]
        == diag.common._read_bytes_snapshot(CONFIG, label="test config")[1]
    )
    assert receipt["implementation_sha256"] == diag.implementation_sha256()
    assert receipt["schema_sha256"] == {
        "config": diag.common._read_bytes_snapshot(
            CONFIG_SCHEMA, label="test config schema"
        )[1],
        "receipt": diag.common._read_bytes_snapshot(
            RECEIPT_SCHEMA, label="test receipt schema"
        )[1],
    }
    assert receipt["model_identity"]["export_manifest_sha256"] == (
        "61a9ac5fab43da9bf053eb46642a030fcd7485100c3e82eb5d90f03b9d8124bb"
    )
    assert receipt["tokenization"]["sentencepiece_bos_id"] == 2
    assert receipt["tokenization"]["hf_export_config_bos_id"] == 1
    assert receipt["tokenization"]["token_ids_digest_algorithm"] == (
        "sha256_signed_int64_big_endian_concatenation_v1"
    )
    assert receipt["tokenization"]["trigger_span_semantics"] == (
        "zero_based_start_inclusive_end_exclusive"
    )
    assert receipt["tokenization"]["tokenizer_model_proto_authenticated"] is True
    assert receipt["routes"]["cache_layers"] == 26
    assert receipt["routes"]["prefix_kv_bit_equal"] is True
    assert receipt["claims"]["proxy_signal_passed"] is True
    for claim in (
        "formal",
        "training_authorized",
        "quality_validated",
        "numeric_equivalence",
        "thresholds_formal",
        "multi_stream",
        "zero_copy",
        "full_generation_kv_shared",
    ):
        assert receipt["claims"][claim] is False
    assert receipt["audit"] == {
        "dataset_reads": 0,
        "model_loads": 1,
        "network_requests": 0,
        "provider_requests": 0,
    }


def _fixture_receipt() -> dict:
    return json.loads(FIXTURE.read_text("utf-8"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("config_sha256",), "0" * 64),
        (("config_file_sha256",), "0" * 64),
        (("implementation_sha256",), "0" * 64),
        (("schema_sha256", "receipt"), "0" * 64),
        (("tokenization", "full_tokens"), 999),
        (("tokenization", "trigger_span_end"), 30),
        (("tokenization", "ordered_token_ids_sha256"), "0" * 64),
        (("tokenization", "boundary", "left_overhang_codepoints"), 999),
        (("routes", "prefix_cache_tokens"), 24),
    ],
)
def test_receipt_identity_and_token_cross_binding_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    receipt = _fixture_receipt()
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(diag.DiagnosticError):
        diag._validate_receipt(loaded(), receipt)


def test_atomic_directory_publish_cleans_up_before_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "diagnostics" / "version-v1" / "diagnostic_receipt.json"
    destination.parent.parent.mkdir()
    monkeypatch.setattr(diag, "_receipt_path", lambda _config: destination)

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("injected crash before atomic directory rename")

    monkeypatch.setattr(diag.common, "_rename_noreplace", fail_rename)
    with pytest.raises(OSError, match="injected crash"):
        diag.publish_receipt(loaded(), _fixture_receipt())
    assert not destination.parent.exists()
    assert list(destination.parent.parent.iterdir()) == []


def test_atomic_directory_publish_refuses_preexisting_sidecar_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "diagnostics" / "version-v1" / "diagnostic_receipt.json"
    destination.parent.mkdir(parents=True)
    sidecar = destination.with_suffix(".json.sha256")
    sidecar.write_text("sentinel\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(diag, "_receipt_path", lambda _config: destination)
    with pytest.raises(FileExistsError, match="version directory exists"):
        diag.publish_receipt(loaded(), _fixture_receipt())
    assert not destination.exists()
    assert sidecar.read_bytes() == b"sentinel\n"


def test_atomic_directory_publish_rejects_reparse_or_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(physical, linked, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    destination = linked / "version-v1" / "diagnostic_receipt.json"
    monkeypatch.setattr(diag, "_receipt_path", lambda _config: destination)
    with pytest.raises(diag.DiagnosticError, match="realpath drift|reparse|symlink"):
        diag.publish_receipt(loaded(), _fixture_receipt())
    assert list(physical.iterdir()) == []


def test_execution_identity_rejects_mid_run_config_replacement(tmp_path: Path) -> None:
    config = loaded()
    private_config = tmp_path / CONFIG.name
    private_config.write_bytes(CONFIG.read_bytes())
    config["_config_path"] = str(private_config)
    identity = diag._freeze_execution_identity(config)
    private_config.write_bytes(private_config.read_bytes() + b"\n")
    with pytest.raises(diag.DiagnosticError, match="config identity drifted"):
        diag._reverify_execution_identity(identity)


def test_execution_identity_rejects_mid_run_implementation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_source = tmp_path / "gemma_diagnostic.py"
    private_source.write_bytes(Path(diag.__file__).read_bytes())
    imported = diag._identity_file_snapshot(
        private_source, label="test imported implementation"
    )
    monkeypatch.setattr(diag, "_IMPORTED_SOURCE", imported)
    identity = diag._freeze_execution_identity(loaded())
    private_source.write_bytes(private_source.read_bytes() + b"\n")
    with pytest.raises(diag.DiagnosticError, match="implementation identity drifted"):
        diag._reverify_execution_identity(identity)
