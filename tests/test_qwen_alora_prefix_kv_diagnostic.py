from __future__ import annotations

import builtins
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import qwen_alora_prefix_kv_diagnostic as diag


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / "qwen_alora_prefix_kv_diagnostic_v1.yaml"
CONFIG_SCHEMA = (
    ROOT / "configs" / "research" / "qwen_alora_prefix_kv_diagnostic_config.schema.json"
)
RECEIPT_SCHEMA = (
    ROOT
    / "configs"
    / "research"
    / "qwen_alora_prefix_kv_diagnostic_receipt.schema.json"
)


def loaded() -> dict:
    return diag.load_config(CONFIG)


class CharacterTokenizer:
    is_fast = True

    def __init__(self, serialized: str) -> None:
        self.serialized = serialized
        self.tokenize_calls = 0

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return self.serialized

    def __call__(self, text: str, **kwargs):
        assert text == self.serialized
        assert kwargs["return_offsets_mapping"] is True
        self.tokenize_calls += 1
        return {
            "input_ids": [ord(value) for value in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _protocol(invocation: str = "<|activate:builder|>") -> dict:
    return {
        "request2": {
            "system_template": "system",
            "user_template": "prefix",
            "planner_scaffold_summary": "plan",
            "task_summary": "suffix",
            "invocation_text": invocation,
        }
    }


def _passing_metrics() -> dict:
    return {
        "paired_differential_max_abs": 0.000139236,
        "adapter_effect_max_abs": 0.0649891,
        "missing_trigger_effect_max_abs": 0.0,
        "prefix_kv_bit_equal": True,
        "finite": True,
        "shape_equal": True,
        "argmax_equal": True,
        "greedy_equal": True,
        "peak_allocated_mib": 6054.7,
    }


def test_schemas_are_valid_draft_2020_12_and_config_is_bound() -> None:
    config_schema = json.loads(CONFIG_SCHEMA.read_text("utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text("utf-8"))
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(receipt_schema)
    runtime_schema = receipt_schema["properties"]["runtime"]
    assert "cublas_workspace_config" in runtime_schema["required"]
    assert runtime_schema["properties"]["cublas_workspace_config"]["const"] == ":4096:8"
    config = loaded()
    assert config["schema_version"] == diag.CONFIG_VERSION
    assert config["claim_scope"]["diagnostic_only"] is True
    assert config["claim_scope"]["formal"] is False
    assert config["claim_scope"]["training_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claim_scope", "formal"), True),
        (("claim_scope", "training_authorized"), True),
        (("adapter", "rank"), 8),
        (("adapter", "target_modules"), ["q_proj", "v_proj"]),
        (("adapter", "training_steps"), 1),
        (("model", "network_allowed"), True),
        (("audit", "dataset_reads"), 1),
        (("gates", "paired_differential_max_abs"), 0.1),
    ],
)
def test_config_rejects_scope_or_gate_drift(
    path: tuple[str, str], value: object
) -> None:
    config = copy.deepcopy(loaded())
    config[path[0]][path[1]] = value
    with pytest.raises(diag.DiagnosticError):
        diag.validate_config(diag._public_config(config))


def test_contract_only_does_not_import_ml_or_touch_model(monkeypatch, capsys) -> None:
    before = {name: sys.modules.get(name) for name in diag._ML_MODULES}
    monkeypatch.setattr(
        diag,
        "_model_root",
        lambda _config: (_ for _ in ()).throw(AssertionError("model touched")),
    )
    assert diag.main(["--config", str(CONFIG)]) == 0
    assert {name: sys.modules.get(name) for name in diag._ML_MODULES} == before
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "contract_only"
    assert report["model_loaded"] is False
    assert report["formal"] is False
    assert report["training_authorized"] is False


def test_preflight_authenticates_files_without_importing_ml(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("ANCHOR_QWEN25_15B_HF_PATH", str(tmp_path / "missing"))
    before = {name: sys.modules.get(name) for name in diag._ML_MODULES}
    assert diag.main(["--config", str(CONFIG), "--preflight"]) == 3
    assert {name: sys.modules.get(name) for name in diag._ML_MODULES} == before
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "preflight"
    assert report["ready"] is False
    assert report["model_loaded"] is False


def test_preflight_ready_with_authenticated_mock_files(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    contents = {
        "config.json": json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 28,
                "hidden_size": 1536,
            }
        ).encode(),
        "model.safetensors": b"not-loaded",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": json.dumps(
            {"chat_template": "{{ messages }}"}
        ).encode(),
    }
    for name, raw in contents.items():
        (model / name).write_bytes(raw)
    config = loaded()
    expected = {item["path"]: item["sha256"] for item in config["model"]["assets"]}
    original = diag._read_bytes_snapshot

    def snapshot(path: Path, *, label: str):
        if path.parent == model:
            return path.read_bytes(), expected[path.name]
        return original(path, label=label)

    monkeypatch.setattr(diag, "_model_root", lambda _config: model)
    monkeypatch.setattr(diag, "_read_bytes_snapshot", snapshot)
    monkeypatch.setattr(
        diag,
        "_stream_sha256_snapshot",
        lambda path, *, label: expected[path.name],
    )
    monkeypatch.setattr(
        diag, "_receipt_path", lambda _config: tmp_path / "receipt.json"
    )
    report = diag.build_preflight(config)
    assert report["ready"] is True
    assert report["model_loaded"] is False
    assert all(item["matched"] for item in report["artifact_identity"].values())


def test_complete_request2_is_tokenized_once_and_trigger_is_request_local() -> None:
    invocation = "<|activate:builder|>"
    serialized = f"header\n{invocation}\ntail"
    tokenizer = CharacterTokenizer(serialized)
    result = diag.materialize_request2(tokenizer, _protocol(invocation))
    assert tokenizer.tokenize_calls == 1
    assert result["token_ids_derived_from_complete_request2"] is True
    assert result["isolated_invocation_tokenization_authoritative"] is False
    assert result["prefix_tokens"] > 0
    assert result["invocation_tokens"] == len(invocation)
    assert result["post_invocation_tokens"] > 0
    assert len(result["invocation_token_ids_sha256"]) == 64


@pytest.mark.parametrize(
    "serialized",
    [
        "header without trigger tail",
        "<|activate:builder|> x <|activate:builder|>",
    ],
)
def test_missing_or_repeated_trigger_text_is_rejected(serialized: str) -> None:
    tokenizer = CharacterTokenizer(serialized)
    with pytest.raises(diag.DiagnosticError, match="exactly once"):
        diag.materialize_request2(tokenizer, _protocol())
    assert tokenizer.tokenize_calls == 0


def test_repeated_invocation_token_sequence_is_rejected() -> None:
    invocation = "XY"
    serialized = "XY prefix XY suffix"
    # The invocation text is repeated, so rejection happens before tokenization.
    tokenizer = CharacterTokenizer(serialized)
    with pytest.raises(diag.DiagnosticError, match="exactly once"):
        diag.materialize_request2(tokenizer, _protocol(invocation))


def test_boundary_overhang_is_bound_from_full_offsets() -> None:
    invocation = "TRIGGER"
    serialized = f"aa{invocation}zz"

    class ChunkTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs):
            self.tokenize_calls += 1
            return {
                "input_ids": [10, 20, 30, 40, 50],
                "attention_mask": [1, 1, 1, 1, 1],
                "offset_mapping": [(0, 1), (1, 4), (4, 7), (7, 10), (10, 11)],
            }

    result = diag.materialize_request2(
        ChunkTokenizer(serialized), _protocol(invocation)
    )
    assert result["boundary_overhang_left"] == 1
    assert result["boundary_overhang_right"] == 1
    assert result["invocation_tokens"] == 3


def test_slow_tokenizer_is_rejected_before_tokenization() -> None:
    tokenizer = CharacterTokenizer("x<|activate:builder|>y")
    tokenizer.is_fast = False
    with pytest.raises(diag.DiagnosticError, match="fast tokenizer"):
        diag.materialize_request2(tokenizer, _protocol())
    assert tokenizer.tokenize_calls == 0


def test_dynamic_cache_layers_are_read_without_legacy_conversion() -> None:
    class Layer:
        def __init__(self, key, value) -> None:
            self.keys = key
            self.values = value

    class DynamicCache:
        def __init__(self) -> None:
            self.layers = [Layer("k0", "v0"), Layer("k1", "v1")]

        def to_legacy_cache(self):
            raise AssertionError("new DynamicCache must not use legacy conversion")

    assert diag._cache_pairs(DynamicCache()) == (("k0", "v0"), ("k1", "v1"))


@pytest.mark.parametrize("layers", [[], [object()]])
def test_malformed_dynamic_cache_layers_fail_closed(layers) -> None:
    class DynamicCache:
        pass

    cache = DynamicCache()
    cache.layers = layers
    with pytest.raises(diag.DiagnosticError, match="DynamicCache"):
        diag._cache_pairs(cache)


def test_empirical_fp32_metrics_pass_all_fixed_gates() -> None:
    gates = diag.evaluate_gates(
        _passing_metrics(),
        paired_limit=0.001,
        effect_floor=0.001,
        peak_limit_mib=7168.0,
    )
    assert gates["all_passed"] is True
    assert all(gates.values())


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("paired_differential_max_abs", 0.0010001, "paired_differential"),
        ("adapter_effect_max_abs", 0.0009999, "adapter_effect"),
        ("missing_trigger_effect_max_abs", 1e-12, "missing_trigger"),
        ("prefix_kv_bit_equal", False, "prefix_kv"),
        ("finite", False, "finite"),
        ("shape_equal", False, "shapes"),
        ("argmax_equal", False, "argmax"),
        ("greedy_equal", False, "greedy"),
        ("peak_allocated_mib", 7168.01, "memory"),
    ],
)
def test_each_numeric_or_semantic_drift_fails_closed(
    field: str, value: object, gate: str
) -> None:
    metrics = _passing_metrics()
    metrics[field] = value
    gates = diag.evaluate_gates(
        metrics,
        paired_limit=0.001,
        effect_floor=0.001,
        peak_limit_mib=7168.0,
    )
    assert gates[gate] is False
    assert gates["all_passed"] is False


def test_nonfinite_scalar_fails_closed() -> None:
    metrics = _passing_metrics()
    metrics["adapter_effect_max_abs"] = math.nan
    gates = diag.evaluate_gates(
        metrics,
        paired_limit=0.001,
        effect_floor=0.001,
        peak_limit_mib=7168.0,
    )
    assert gates["adapter_effect"] is False
    assert gates["all_passed"] is False


def test_missing_cublas_workspace_config_is_bound_before_ml_import(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert (
        diag._prepare_deterministic_cuda_environment() == diag._CUBLAS_WORKSPACE_CONFIG
    )
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_exact_cublas_workspace_config_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert diag._prepare_deterministic_cuda_environment() == ":4096:8"


def test_drifted_cublas_workspace_config_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(diag.DiagnosticError, match="must be exactly"):
        diag._prepare_deterministic_cuda_environment()


def test_only_execute_enters_gpu_hook(monkeypatch) -> None:
    config = loaded()
    monkeypatch.setattr(
        diag,
        "build_preflight",
        lambda _config: {"ready": True, "gates": {"ok": True}},
    )
    called = []

    def backend(_config, _preflight):
        called.append(True)
        raise diag.DiagnosticError("sentinel backend")

    monkeypatch.setattr(diag, "_execute_gpu", backend)
    with pytest.raises(diag.DiagnosticError, match="sentinel backend"):
        diag.execute_diagnostic(config)
    assert called == [True]


def test_atomic_receipt_publish_refuses_overwrite(monkeypatch, tmp_path: Path) -> None:
    config = loaded()
    destination = tmp_path / "out" / "receipt.json"
    receipt = {"status": "passed"}
    monkeypatch.setattr(diag, "_validate_receipt", lambda *_args: None)
    monkeypatch.setattr(diag, "_receipt_path", lambda _config: destination)
    assert diag.publish_receipt(config, receipt) == destination
    expected = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    assert destination.read_bytes() == expected
    with pytest.raises(FileExistsError):
        diag.publish_receipt(config, receipt)


def test_receipt_validation_failure_publishes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "out" / "receipt.json"
    monkeypatch.setattr(diag, "_receipt_path", lambda _config: destination)

    def reject(*_args):
        raise diag.DiagnosticError("bad receipt")

    monkeypatch.setattr(diag, "_validate_receipt", reject)
    with pytest.raises(diag.DiagnosticError, match="bad receipt"):
        diag.publish_receipt(loaded(), {"status": "failed"})
    assert not destination.exists()
    assert not destination.parent.exists()


def test_no_ml_import_occurs_at_module_level(monkeypatch) -> None:
    imported: list[str] = []
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in diag._ML_MODULES:
            imported.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    diag.build_contract_report(loaded())
    assert imported == []


def test_token_digest_is_order_sensitive_and_not_raw_json() -> None:
    first = diag._token_ids_sha256([1, 2, 3])
    second = diag._token_ids_sha256([3, 2, 1])
    assert first != second
    assert first != hashlib.sha256(b"[1, 2, 3]").hexdigest()
