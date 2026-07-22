from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import qwen_alora_prefix_kv_diagnostic as diag


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / "qwen_alora_prefix_kv_diagnostic_tf32_v2.yaml"
CONFIG_SCHEMA = (
    ROOT
    / "configs"
    / "research"
    / "qwen_alora_prefix_kv_diagnostic_tf32_config.schema.json"
)
RECEIPT_SCHEMA = (
    ROOT
    / "configs"
    / "research"
    / "qwen_alora_prefix_kv_diagnostic_tf32_receipt.schema.json"
)
FIXTURE_RECEIPT = (
    ROOT
    / "fixtures"
    / "research"
    / "qwen_alora_prefix_kv_diagnostic_tf32_v2"
    / "diagnostic_receipt.json"
)


def loaded() -> dict:
    return diag.load_config(CONFIG)


def test_tf32_v2_schemas_and_exact_profile_are_bound() -> None:
    config_schema = json.loads(CONFIG_SCHEMA.read_text("utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(receipt_schema)
    config = loaded()
    assert config["schema_version"] == diag.TF32_CONFIG_VERSION
    assert config["profile_id"] == "tf32_operational_v2"
    assert config["numerics"]["tf32"] is True
    assert config["numerics"]["matmul_precision"] == "high"
    assert diag._profile(config)["receipt_version"] == diag.TF32_RECEIPT_VERSION
    assert diag._receipt_path(config) == (
        ROOT
        / "artifacts"
        / "diagnostics"
        / "qwen_alora_prefix_kv_tf32_v2"
        / "diagnostic_receipt.json"
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "profile_id", "fp32_reference_v1"),
        ("numerics", "tf32", False),
        ("numerics", "matmul_precision", "highest"),
        (
            "output",
            "receipt_path",
            "artifacts/diagnostics/qwen_alora_prefix_kv_v1/diagnostic_receipt.json",
        ),
        (
            "schemas",
            "receipt",
            {
                "path": "configs/research/qwen_alora_prefix_kv_diagnostic_receipt.schema.json",
                "version": diag.RECEIPT_VERSION,
                "sha256": "0" * 64,
            },
        ),
    ],
)
def test_tf32_profile_drift_fails_closed(
    section: str | None, field: str, value: object
) -> None:
    config = copy.deepcopy(loaded())
    if section is None:
        config[field] = value
    else:
        config[section][field] = value
    with pytest.raises(diag.DiagnosticError):
        diag.validate_config(diag._public_config(config))


def test_schema_version_cannot_be_loaded_through_other_profile_filename() -> None:
    # Exact canonical-parent enforcement is already covered by the v1 suite. This
    # assertion exercises the in-memory profile cross-binding independently.
    config = copy.deepcopy(loaded())
    config["schema_version"] = diag.CONFIG_VERSION
    with pytest.raises(diag.DiagnosticError):
        diag.validate_config(diag._public_config(config))


def test_tf32_contract_only_imports_no_ml_and_reports_profile(capsys) -> None:
    before = {name: sys.modules.get(name) for name in diag._ML_MODULES}
    assert diag.main(["--config", str(CONFIG)]) == 0
    assert {name: sys.modules.get(name) for name in diag._ML_MODULES} == before
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == diag.TF32_CONFIG_VERSION
    assert report["profile_id"] == "tf32_operational_v2"
    assert report["model_loaded"] is False
    assert report["gpu_touched"] is False
    assert report["provider_requests"] == 0
    assert report["network_requests"] == 0
    assert report["dataset_inputs"] == 0


def test_tf32_receipt_runtime_contract_is_not_fp32_reference() -> None:
    schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    runtime = schema["properties"]["runtime"]
    assert runtime["properties"]["tf32"]["const"] is True
    assert runtime["properties"]["matmul_precision"]["const"] == "high"
    assert "matmul_precision" in runtime["required"]
    v1_schema = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "qwen_alora_prefix_kv_diagnostic_receipt.schema.json"
        ).read_text("utf-8")
    )
    assert v1_schema["properties"]["runtime"]["properties"]["tf32"]["const"] is False


def test_tf32_v2_refuses_to_overwrite_fp32_v1_output() -> None:
    v1 = diag.load_config(ROOT / "configs" / "research" / diag.DEFAULT_CONFIG_NAME)
    v2 = loaded()
    assert diag._receipt_path(v1) != diag._receipt_path(v2)
    assert diag._profile(v1)["id"] == "fp32_reference_v1"
    assert diag._profile(v2)["id"] == "tf32_operational_v2"


def test_tf32_relative_and_absolute_gates_are_both_required() -> None:
    metrics = {
        "paired_differential_max_abs": 0.03,
        "adapter_effect_max_abs": 0.21,
        "missing_trigger_effect_max_abs": 0.0,
        "prefix_kv_bit_equal": True,
        "finite": True,
        "shape_equal": True,
        "argmax_equal": True,
        "greedy_equal": True,
        "peak_allocated_mib": 6000.0,
    }
    gates = diag.evaluate_gates(
        metrics,
        paired_limit=0.05,
        effect_floor=0.001,
        peak_limit_mib=7168.0,
        paired_relative_limit=0.2,
    )
    assert gates["paired_differential"] is True
    assert gates["paired_relative"] is True
    assert gates["all_passed"] is True
    relative_failure = copy.deepcopy(metrics)
    relative_failure["adapter_effect_max_abs"] = 0.1
    gates = diag.evaluate_gates(
        relative_failure,
        paired_limit=0.05,
        effect_floor=0.001,
        peak_limit_mib=7168.0,
        paired_relative_limit=0.2,
    )
    assert gates["paired_differential"] is True
    assert gates["paired_relative"] is False
    assert gates["all_passed"] is False


def test_published_tf32_receipt_is_schema_valid_and_sidecar_exact() -> None:
    raw = FIXTURE_RECEIPT.read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    Draft202012Validator(schema).validate(receipt)
    digest = diag._sha256_bytes(raw)
    assert digest == "278a58441e2f4fb7fc46a67dd5acec3851a7b6937685d90c77873e5972b49cce"
    assert (
        FIXTURE_RECEIPT.with_suffix(".json.sha256").read_text("utf-8")
        == f"{digest}  diagnostic_receipt.json\n"
    )
    assert receipt["config_sha256"] == diag.config_sha256(loaded())
    assert receipt["metrics"]["paired_to_adapter_effect_ratio"] < 0.2
    assert receipt["claims"]["numeric_equivalence"] is False
    assert receipt["claims"]["proxy_signal_passed"] is True
    assert receipt["claims"]["thresholds_formal"] is False
    assert receipt["claims"]["formal"] is False
    assert receipt["claims"]["training_authorized"] is False
    assert receipt["audit"] == {
        "dataset_reads": 0,
        "model_loads": 1,
        "network_requests": 0,
        "provider_requests": 0,
    }
