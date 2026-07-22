from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training.config import ConfigError
from anchor_mvp.training.qwen_lora_diagnostic import (
    _adapter_staging_directory,
    _assert_base_parameters_unchanged,
    _assert_nonzero_lora_gradients,
    _assert_q_proj_lora_trainable,
    _capture_base_parameters,
    _normalize_saved_adapter_base_identity,
    _resolved_local_model,
    _validate_saved_adapter,
    build_preflight,
    load_qwen_diagnostic_config,
    main,
    validate_qwen_diagnostic_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training" / "qwen2_5_1_5b_lora_one_step_diagnostic.yaml"


def loaded() -> dict:
    return load_qwen_diagnostic_config(CONFIG)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matched_source(config: dict, _model_dir: Path) -> dict:
    return {
        "expected_revision": config["model"]["expected_source_revision"],
        "observed_revision": config["model"]["expected_source_revision"],
        "expected_repo": config["model"]["expected_source_repo"],
        "observed_repo": config["model"]["expected_source_repo"],
        "git_metadata_physical": True,
        "working_tree_clean": True,
        "tracked_file_count": 4,
        "tracked_files_sha256": "1" * 64,
        "matched": True,
    }


def _tiny_hf_config(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(qdiag, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(qdiag, "_source_identity", _matched_source)
    model_dir = tmp_path / "qwen-hf"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 28,
                "hidden_size": 1536,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"tiny-model-fixture")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    raw = json.loads(CONFIG.read_text("utf-8"))
    raw["model"].update(
        {
            "local_path": str(model_dir),
            "expected_config_json_sha256": _sha256(model_dir / "config.json"),
            "expected_model_safetensors_sha256": _sha256(
                model_dir / "model.safetensors"
            ),
            "expected_tokenizer_json_sha256": _sha256(model_dir / "tokenizer.json"),
            "expected_tokenizer_config_sha256": _sha256(
                model_dir / "tokenizer_config.json"
            ),
        }
    )
    raw["output"]["adapter_dir"] = "artifacts/diagnostics/tiny-run"
    config_path = repo_root / "configs" / "training" / "diagnostic.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path, model_dir


def test_config_is_strictly_isolated_q_proj_one_step() -> None:
    config = loaded()
    assert config["lora"] == {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    assert config["training"]["max_steps"] == 1
    assert config["training"]["batch_size"] == 1
    assert 128 <= config["training"]["sequence_length"] <= 256
    assert config["dataset"] == {
        "kind": "inline_toy_plumbing_v1",
        "formal_inputs_allowed": False,
        "heldout_allowed": False,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (("model", "local_path"), "models/qwen-q4_k_m.gguf", "GGUF"),
        (("lora", "rank"), 16, "rank must be 4 or 8"),
        (("lora", "target_modules"), ["q_proj", "v_proj"], "exactly"),
        (("training", "max_steps"), 2, "max_steps must be 1"),
        (("training", "batch_size"), 2, "batch_size must be 1"),
        (("training", "sequence_length"), 512, "must be in [128, 256]"),
        (("dataset", "formal_inputs_allowed"), True, "must be false"),
        (("dataset", "heldout_allowed"), True, "must be false"),
    ],
)
def test_config_rejects_scope_escape(
    field: tuple[str, str], value: object, message: str
) -> None:
    config = copy.deepcopy(loaded())
    config[field[0]][field[1]] = value
    with pytest.raises(
        ConfigError, match=r"" + message.replace("[", r"\[").replace("]", r"\]")
    ):
        validate_qwen_diagnostic_config(config)


def test_config_rejects_external_dataset_paths() -> None:
    config = copy.deepcopy(loaded())
    config["dataset"]["path"] = "data/formal/gold.jsonl"
    with pytest.raises(ConfigError, match="dataset contains unknown fields: path"):
        validate_qwen_diagnostic_config(config)


def test_config_rejects_unknown_fields_and_output_escape(tmp_path: Path) -> None:
    config = copy.deepcopy(loaded())
    config["surprise"] = True
    with pytest.raises(ConfigError, match="config contains unknown fields"):
        validate_qwen_diagnostic_config(config)
    config = copy.deepcopy(loaded())
    config["output"]["adapter_dir"] = str(tmp_path / "escaped")
    with pytest.raises(ConfigError, match="must be a child"):
        build_preflight(config)


def test_default_model_path_targets_shared_llm_models(monkeypatch) -> None:
    monkeypatch.delenv("ANCHOR_QWEN25_15B_HF_PATH", raising=False)
    expected = (ROOT.parent / "models" / "qwen2.5-1.5b-instruct-hf").resolve()
    assert _resolved_local_model(loaded()).resolve() == expected


def test_dry_run_does_not_import_ml_runtime(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_path, _model_dir = _tiny_hf_config(tmp_path, monkeypatch)
    before = {name: sys.modules.get(name) for name in ("torch", "transformers", "peft")}
    assert main(["--config", str(config_path), "--dry-run"]) == 0
    assert {
        name: sys.modules.get(name) for name in ("torch", "transformers", "peft")
    } == before
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["executed"] is False
    assert all(item["matched"] for item in payload["artifact_identity"].values())


def test_preflight_fails_closed_on_hash_or_architecture_drift(
    monkeypatch, tmp_path: Path
) -> None:
    config_path, model_dir = _tiny_hf_config(tmp_path, monkeypatch)
    config = load_qwen_diagnostic_config(config_path)
    (model_dir / "model.safetensors").write_bytes(b"replaced")
    assert build_preflight(config)["gates"]["artifact_hashes_match"] is False
    config_path, model_dir = _tiny_hf_config(tmp_path / "second", monkeypatch)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"]}),
        encoding="utf-8",
    )
    assert (
        build_preflight(load_qwen_diagnostic_config(config_path))["gates"][
            "hf_config_closed_and_qwen2"
        ]
        is False
    )


def test_preflight_rejects_existing_output(monkeypatch, tmp_path: Path) -> None:
    config_path, _model_dir = _tiny_hf_config(tmp_path, monkeypatch)
    output = tmp_path / "repo" / "artifacts" / "diagnostics" / "tiny-run"
    output.mkdir(parents=True)
    report = build_preflight(load_qwen_diagnostic_config(config_path))
    assert report["gates"]["output_scoped_and_absent"] is False
    assert report["ready"] is False


def test_preflight_rejects_symlinked_model_directory(
    monkeypatch, tmp_path: Path
) -> None:
    config_path, model_dir = _tiny_hf_config(tmp_path, monkeypatch)
    link = tmp_path / "linked-qwen"
    try:
        link.symlink_to(model_dir, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink capability is unavailable")
    raw = json.loads(config_path.read_text("utf-8"))
    raw["model"]["local_path"] = str(link)
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    report = build_preflight(load_qwen_diagnostic_config(config_path))
    assert report["gates"]["hf_directory_physical"] is False
    assert report["gates"]["required_artifacts_physical"] is False


def test_missing_hf_directory_is_not_ready_and_gguf_cli_is_blocked(
    monkeypatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-hf"
    monkeypatch.setenv("ANCHOR_QWEN25_15B_HF_PATH", str(missing))
    assert build_preflight(loaded())["ready"] is False
    monkeypatch.setenv("ANCHOR_QWEN25_15B_HF_PATH", str(tmp_path / "model.gguf"))
    assert main(["--config", str(CONFIG), "--dry-run"]) == 2


def test_base_hash_gate_and_trainable_scope_use_tiny_tensors() -> None:
    class FakeArray:
        def __init__(self, raw: bytes) -> None:
            self.raw = raw

        def tobytes(self) -> bytes:
            return self.raw

    class FakeTensor:
        def __init__(
            self, raw: bytes, *, requires_grad: bool, shape: tuple[int, ...] = (1,)
        ) -> None:
            self.raw = raw
            self.requires_grad = requires_grad
            self.shape = shape

        def detach(self):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

        def view(self, _dtype):
            return self

        def numpy(self) -> FakeArray:
            return FakeArray(self.raw)

        def numel(self) -> int:
            return len(self.raw)

    class FakeTorch:
        uint8 = object()

    torch = FakeTorch()

    class TinyModel:
        def __init__(self) -> None:
            self.base = FakeTensor(b"base", requires_grad=False)
            self.good_a = FakeTensor(b"lora", requires_grad=True, shape=(1, 1))
            self.good_b = FakeTensor(b"lora", requires_grad=True, shape=(1, 1))

        def named_parameters(self):
            return iter(
                (
                    ("model.embed.weight", self.base),
                    (
                        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
                        self.good_a,
                    ),
                    (
                        "model.layers.0.self_attn.q_proj.lora_B.default.weight",
                        self.good_b,
                    ),
                )
            )

    model = TinyModel()
    captured, before = _capture_base_parameters(model, torch)
    names, count = _assert_q_proj_lora_trainable(
        model, expected_layers=1, hidden_size=1, rank=1
    )
    assert names == [
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.self_attn.q_proj.lora_B.default.weight",
    ]
    assert count == 8
    assert _assert_base_parameters_unchanged(captured, torch) == before
    model.base.raw = b"changed"
    with pytest.raises(RuntimeError, match="frozen base hash changed"):
        _assert_base_parameters_unchanged(captured, torch)


def test_trainable_scope_rejects_non_q_proj_tensor() -> None:
    class Parameter:
        requires_grad = True

        @staticmethod
        def numel() -> int:
            return 1

    class Model:
        @staticmethod
        def named_parameters():
            return iter((("weight", Parameter()),))

    with pytest.raises(RuntimeError, match="scope/shape escaped"):
        _assert_q_proj_lora_trainable(Model(), expected_layers=1, hidden_size=1, rank=1)


def test_gradient_gate_requires_at_least_one_nonzero_lora_gradient() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

        def all(self):
            return self

    class Torch:
        @staticmethod
        def count_nonzero(value: int) -> Scalar:
            return Scalar(value)

        @staticmethod
        def isfinite(value: int | float) -> Scalar:
            return Scalar(int(math.isfinite(value)))

    class Parameter:
        requires_grad = True

        def __init__(self, gradient: int | None) -> None:
            self.grad = gradient

    class Model:
        def __init__(self, gradients: tuple[int | None, ...]) -> None:
            self.gradients = gradients

        def named_parameters(self):
            return iter(
                (
                    "model.layers."
                    f"{index}.self_attn.q_proj.lora_"
                    f"{'A' if index == 0 else 'B'}.default.weight",
                    Parameter(gradient),
                )
                for index, gradient in enumerate(self.gradients)
            )

    coverage = _assert_nonzero_lora_gradients(Model((0, 2)), Torch())
    assert coverage["total_nonzero_tensor_count"] == 1
    assert coverage["by_matrix"]["A"]["nonzero_count"] == 0
    assert coverage["by_matrix"]["B"]["nonzero_count"] == 1
    with pytest.raises(RuntimeError, match="no nonzero gradient"):
        _assert_nonzero_lora_gradients(Model((None, 0)), Torch())
    with pytest.raises(RuntimeError, match="non-finite gradients"):
        _assert_nonzero_lora_gradients(Model((0, float("nan"))), Torch())


def test_saved_adapter_gate_requires_rank_and_q_proj(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    safetensors = pytest.importorskip("safetensors.numpy")
    safetensors.save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": numpy.zeros(
                (1, 1), dtype="float32"
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": numpy.zeros(
                (1, 1), dtype="float32"
            ),
        },
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": str(tmp_path / ".model-snapshot"),
            }
        ),
        encoding="utf-8",
    )
    _normalize_saved_adapter_base_identity(tmp_path)
    saved_config = json.loads((tmp_path / "adapter_config.json").read_text("utf-8"))
    assert saved_config["base_model_name_or_path"] == qdiag.EXPECTED_MODEL_ID
    hashes = _validate_saved_adapter(tmp_path, rank=1, expected_layers=1, hidden_size=1)
    assert set(hashes) == {"adapter_config.json", "adapter_model.safetensors"}
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "target_modules": ["v_proj"],
                "base_model_name_or_path": qdiag.EXPECTED_MODEL_ID,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="escaped q_proj"):
        _validate_saved_adapter(tmp_path, rank=1, expected_layers=1, hidden_size=1)
    saved_config["target_modules"] = ["q_proj"]
    saved_config["base_model_name_or_path"] = str(tmp_path / ".model-snapshot")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(saved_config), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="base model identity is not canonical"):
        _validate_saved_adapter(tmp_path, rank=1, expected_layers=1, hidden_size=1)


def test_reload_failure_cleans_staging_and_never_publishes(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "diagnostics" / "failed-run"
    with pytest.raises(RuntimeError, match="injected reload failure"):
        with _adapter_staging_directory(output) as staging:
            (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("injected reload failure")
    assert not output.exists()
    assert not list(output.parent.glob(".failed-run.tmp-*"))
