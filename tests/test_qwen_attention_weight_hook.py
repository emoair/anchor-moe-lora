from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anchor_mvp.research import qwen_attention_weight_hook as audit
from anchor_mvp.training.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / audit.CONFIG_PATH


def test_config_and_controlled_adapter_are_strictly_authenticated() -> None:
    config = audit.load_config(CONFIG)
    assert config["capture"]["attention_implementation"] == "eager"
    assert config["capture"]["selected_layers"] == [0, 13, 27]
    assert config["capture"]["modes"] == list(audit.MODES)
    assert config["capture"]["difference_panel"] == ("full_minus_q_only_component")
    assert config["audit"]["heldout_reads"] == 0
    assert config["audit"]["protected_body_reads"] == 0
    assert audit.authenticate_adapter(config) == {
        "adapter_config.json": (
            "17af30108f7163bc30773d5d53e7847fd51f6e3faaf779f4f79f84e368883222"
        ),
        "adapter_model.safetensors": (
            "fab58c506a103f09b45e22b1aa10c0d7073e6c5c032dc49f2e89d7219427d6ba"
        ),
        "diagnostic_receipt.json": (
            "dc94204df696db795f3e657c679d67918e3aa2723b2d0bdf7dee899ef4490f6e"
        ),
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("claims", "formal", True),
        ("claims", "attention_equals_explanation", True),
        ("audit", "heldout_reads", 1),
        ("audit", "raw_token_ids_in_output", True),
        ("capture", "attention_implementation", "sdpa"),
        ("capture", "selected_layers", [0, 1, 2]),
        ("capture", "modes", ["full"]),
    ],
)
def test_contract_drift_fails_closed(section: str, field: str, value: object) -> None:
    config = audit.load_config(CONFIG)
    config[section][field] = value
    with pytest.raises(ConfigError):
        audit.validate_config(config)


class _AttentionModule:
    def __init__(self, weights: object) -> None:
        import torch

        class Module(torch.nn.Module):
            def __init__(self, attention: object) -> None:
                super().__init__()
                self.attention = attention

            def forward(self, value: object) -> tuple[object, object]:
                return value, self.attention

        self.module = Module(weights)


class _Layer:
    def __init__(self, weights: object) -> None:
        self.self_attn = _AttentionModule(weights).module


class _Node:
    pass


def _hook_model(weights: object) -> object:
    root = _Node()
    root.model = _Node()
    root.model.model = _Node()
    root.model.model.layers = [_Layer(weights) for _ in range(28)]
    return root


def test_hook_captures_layers_0_13_27_as_head_mean_float32_cpu() -> None:
    torch = pytest.importorskip("torch")
    head_zero = torch.zeros((4, 4), dtype=torch.float32)
    head_one = torch.ones((4, 4), dtype=torch.float32)
    weights = torch.stack((head_zero, head_one), dim=0).unsqueeze(0)
    model = _hook_model(weights)
    hook = audit.AttentionWeightHook((0, 13, 27))
    hook.install(model, expected_decoder_layers=28)
    for layer in (0, 13, 27):
        model.model.model.layers[layer].self_attn(torch.zeros(1))
    captured = hook.snapshot()
    hook.close()
    assert set(captured) == {0, 13, 27}
    for matrix in captured.values():
        assert matrix.dtype == np.float32
        assert matrix.shape == (4, 4)
        np.testing.assert_array_equal(matrix, np.full((4, 4), 0.5, np.float32))


def test_hook_rejects_non_eager_attention_without_weights() -> None:
    hook = audit.AttentionWeightHook((0,))
    capture = hook._forward_hook(0)
    with pytest.raises(RuntimeError, match="no 4D attention weights"):
        capture(None, (), (object(), None))


class _ScalingModule:
    def __init__(self, value: float) -> None:
        self.scaling = {"default": value}


class _FakePeft:
    def __init__(self) -> None:
        self.q = [_ScalingModule(2.0) for _ in range(3)]
        self.o = [_ScalingModule(3.0) for _ in range(3)]

    def named_modules(self):
        yield "", self
        for index, module in enumerate(self.q):
            yield f"model.layers.{index}.self_attn.q_proj", module
        for index, module in enumerate(self.o):
            yield f"model.layers.{index}.self_attn.o_proj", module


@pytest.mark.parametrize(
    ("enabled", "q_value", "o_value", "changed"),
    [
        (frozenset({"q_proj"}), 2.0, 0.0, {"q_proj": 0, "o_proj": 3}),
        (frozenset({"o_proj"}), 0.0, 3.0, {"q_proj": 3, "o_proj": 0}),
        (
            frozenset({"q_proj", "o_proj"}),
            2.0,
            3.0,
            {"q_proj": 0, "o_proj": 0},
        ),
    ],
)
def test_component_scaling_is_reversible(
    enabled: frozenset[str],
    q_value: float,
    o_value: float,
    changed: dict[str, int],
) -> None:
    model = _FakePeft()
    with audit.component_scaling(model, enabled) as observed:
        assert observed == changed
        assert {module.scaling["default"] for module in model.q} == {q_value}
        assert {module.scaling["default"] for module in model.o} == {o_value}
    assert {module.scaling["default"] for module in model.q} == {2.0}
    assert {module.scaling["default"] for module in model.o} == {3.0}


def test_component_scaling_restores_after_exception() -> None:
    model = _FakePeft()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with audit.component_scaling(model, frozenset({"q_proj"})):
            raise RuntimeError("synthetic failure")
    assert {module.scaling["default"] for module in model.q} == {2.0}
    assert {module.scaling["default"] for module in model.o} == {3.0}


def _attention(seed: int, tokens: int = 6) -> np.ndarray:
    generator = np.random.default_rng(seed)
    value = np.tril(generator.uniform(0.01, 1.0, size=(tokens, tokens)))
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def _matrices() -> dict[str, dict[int, np.ndarray]]:
    return {
        mode: {
            layer: _attention(1000 + mode_index * 100 + layer)
            for layer in audit.SELECTED_LAYERS
        }
        for mode_index, mode in enumerate(audit.MODES)
    }


def test_metrics_and_differences_are_deterministic() -> None:
    matrix = _attention(7)
    metrics = audit._attention_metrics(matrix, 3)
    difference = audit._difference_metrics(matrix, matrix.copy())
    assert metrics == audit._attention_metrics(matrix.copy(), 3)
    assert metrics["shape"] == [6, 6]
    assert 0.0 <= metrics["target_query_prompt_mass_mean"] <= 1.0
    assert difference["mean_absolute"] == 0.0
    assert difference["root_mean_square"] == 0.0
    assert difference["maximum_absolute"] == 0.0


def test_summary_and_five_panel_heatmaps_are_body_free_and_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = audit.load_config(CONFIG)
    matrices = _matrices()
    summary = audit.build_summary(
        config,
        matrices,
        prompt_tokens=3,
        target_tokens=3,
        full_token_ids_sha256="0" * 64,
        adapter_hashes={
            "adapter_config.json": "1" * 64,
            "adapter_model.safetensors": "2" * 64,
            "diagnostic_receipt.json": "3" * 64,
        },
        runtime={"device": "synthetic_cpu_test", "model_loads": 0},
    )
    serialized = audit._canonical_json_bytes(summary).decode("utf-8")
    assert config["probe"]["prompt"] not in serialized
    assert config["probe"]["target"] not in serialized
    assert summary["probe"]["raw_token_ids_emitted"] is False
    assert summary["probe"]["target_boundary_zero_based"] == 3
    for layer in map(str, audit.SELECTED_LAYERS):
        assert set(summary["capture"]["layers"][layer]["modes"]) == set(audit.MODES)
        assert audit.DIFFERENCE_PANEL in summary["capture"]["layers"][layer]

    monkeypatch.setattr(audit, "_root", lambda: tmp_path)
    summary_path, summary_sha = audit.publish_audit(config, summary, matrices)
    assert summary_path.is_file()
    assert summary_path.with_name("summary.json.sha256").read_text("ascii") == (
        f"{summary_sha}  summary.json\n"
    )
    output_files = {path.name for path in summary_path.parent.iterdir()}
    assert output_files == {
        "summary.json",
        "summary.json.sha256",
        "layer_00_attention.png",
        "layer_13_attention.png",
        "layer_27_attention.png",
    }
    for layer in audit.SELECTED_LAYERS:
        assert (summary_path.parent / f"layer_{layer:02d}_attention.png").stat().st_size

    with pytest.raises(ConfigError, match="already exists"):
        audit.publish_audit(config, summary, matrices)
