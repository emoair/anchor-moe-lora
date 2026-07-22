from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from anchor_mvp.training import qwen_budget_matched_ablation as ablation
from anchor_mvp.training import qwen_budget_matched_ablation_audit as audit
from anchor_mvp.training import qwen_synthetic_scaffold_diagnostic as synth
from anchor_mvp.training.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ablation.CONFIG_PATH


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return list(rendered.encode("utf-8"))


@dataclass(frozen=True)
class FakeParameter:
    shape: tuple[int, int]
    requires_grad: bool = True

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]


class FakeLoraModel:
    def __init__(self, ranks: dict[str, int]) -> None:
        output_size = {
            "q_proj": ablation.HIDDEN_SIZE,
            "o_proj": ablation.HIDDEN_SIZE,
            "k_proj": ablation.KV_SIZE,
            "v_proj": ablation.KV_SIZE,
        }
        values: list[tuple[str, FakeParameter]] = []
        for layer in range(ablation.EXPECTED_LAYERS):
            for module, rank in ranks.items():
                prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
                values.extend(
                    [
                        (
                            f"{prefix}.lora_A.default.weight",
                            FakeParameter((rank, ablation.HIDDEN_SIZE)),
                        ),
                        (
                            f"{prefix}.lora_B.default.weight",
                            FakeParameter((output_size[module], rank)),
                        ),
                    ]
                )
        self._values = values

    def named_parameters(self):
        return iter(self._values)


def _example(index: int, *, target: str | None = None) -> synth.ScaffoldExample:
    return synth.ScaffoldExample(
        record_id=f"syn-record-{index:03d}",
        split="train",
        variant="json_only",
        source_bundle_id=f"syn-bundle-{index // 10:02d}",
        prompt=f"plan-{index}",
        target=target or f'{{"index":{index}}}',
    )


def _dataset(order: range | list[int] | tuple[int, ...]) -> synth.ScaffoldDataset:
    return synth.ScaffoldDataset(
        manifest={},
        manifest_sha256="0" * 64,
        partition_sha256={},
        train=tuple(_example(index) for index in order),
        eval_proxy=(),
    )


def test_load_config_is_canonical_and_diagnostic_only() -> None:
    config = ablation.load_config(CONFIG)
    assert config["schema_version"] == ablation.CONFIG_VERSION
    assert config["training"]["optimizer_steps"] == 80
    assert config["precision"] == {
        "compute_dtype": "bfloat16",
        "tf32": True,
        "float32_matmul_precision": "high",
    }
    assert config["claims"] == {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }
    with pytest.raises(ConfigError, match="config must remain exactly"):
        ablation.load_config(ROOT / "configs" / "training" / "other.yaml")


@pytest.mark.parametrize(
    ("profile_name", "ranks", "alphas", "tensor_count"),
    [
        ("q_only", {"q_proj": 16}, {"q_proj": 32}, 56),
        (
            "q_plus_o",
            {"q_proj": 8, "o_proj": 8},
            {"q_proj": 16, "o_proj": 16},
            112,
        ),
        (
            "wide_budget_matched",
            {"q_proj": 5, "o_proj": 4, "k_proj": 6, "v_proj": 6},
            {"q_proj": 10, "o_proj": 8, "k_proj": 12, "v_proj": 12},
            224,
        ),
    ],
)
def test_profiles_have_exact_equal_budget_and_tensor_coverage(
    profile_name: str,
    ranks: dict[str, int],
    alphas: dict[str, int],
    tensor_count: int,
) -> None:
    config = ablation.load_config(CONFIG)
    profile = ablation._profile(config, profile_name)
    assert ablation._ranks(profile) == ranks
    assert ablation._alphas(profile) == alphas
    assert all(alphas[module] == 2 * rank for module, rank in ranks.items())
    assert ablation._expected_parameter_count(ranks) == ablation.EXPECTED_PARAMETERS
    assert profile["expected_trainable_tensors"] == tensor_count

    names, parameters = ablation._validate_trainable(
        FakeLoraModel(ranks), ranks, tensor_count
    )
    assert len(names) == tensor_count
    assert len(set(names)) == tensor_count
    assert parameters == ablation.EXPECTED_PARAMETERS


def test_fixed_order_is_unique_seeded_and_input_order_independent() -> None:
    forward = ablation._fixed_train_order(_dataset(range(80)), 1337)
    reverse = ablation._fixed_train_order(_dataset(list(reversed(range(80)))), 1337)
    assert [item.record_id for item in forward] == [item.record_id for item in reverse]
    assert len(forward) == len({item.record_id for item in forward}) == 80
    other_seed = ablation._fixed_train_order(_dataset(range(80)), 7331)
    assert [item.record_id for item in forward] != [
        item.record_id for item in other_seed
    ]


def test_tokenized_training_digest_is_stable_and_content_bound() -> None:
    tokenizer = FakeTokenizer()
    ordered = ablation._fixed_train_order(_dataset(range(80)), 1337)
    first = ablation._ordered_training_digest(tokenizer, ordered, 512)
    second = ablation._ordered_training_digest(tokenizer, ordered, 512)
    assert first == second
    assert first["records"] == first["unique_records"] == 80
    assert first["duplicates"] == first["missing"] == 0

    changed = list(ordered)
    selected = changed[0]
    changed[0] = synth.ScaffoldExample(
        record_id=selected.record_id,
        split=selected.split,
        variant=selected.variant,
        source_bundle_id=selected.source_bundle_id,
        prompt=selected.prompt,
        target='{"changed":true}',
    )
    changed_digest = ablation._ordered_training_digest(tokenizer, changed, 512)
    assert (
        changed_digest["ordered_record_ids_sha256"]
        == first["ordered_record_ids_sha256"]
    )
    assert (
        changed_digest["ordered_tokenized_examples_sha256"]
        != first["ordered_tokenized_examples_sha256"]
    )


def test_preflight_authentication_rejects_sha_sidecar_and_recompute_drift(
    tmp_path: Path,
) -> None:
    expected = {
        "schema_version": ablation.PREFLIGHT_VERSION,
        "ready": True,
        "profile": "q_only",
    }
    receipt, digest = ablation.publish_preflight(expected, tmp_path / "q_only")
    assert ablation._authenticate_preflight(receipt, digest, expected) == expected

    with pytest.raises(ConfigError, match="authentication failed"):
        ablation._authenticate_preflight(receipt, "0" * 64, expected)
    with pytest.raises(ConfigError, match="differs from current recomputation"):
        ablation._authenticate_preflight(
            receipt, digest, {**expected, "profile": "q_plus_o"}
        )
    receipt.with_name("preflight.json.sha256").write_text(
        f"{digest} preflight.json\n", encoding="ascii"
    )
    with pytest.raises(ConfigError, match="authentication failed"):
        ablation._authenticate_preflight(receipt, digest, expected)


def test_profile_outputs_are_isolated_and_cannot_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ablation.load_config(CONFIG)
    monkeypatch.setattr(ablation, "_root", lambda: tmp_path)
    outputs = {name: ablation._output_path(config, name) for name in ablation.PROFILES}
    assert len(set(outputs.values())) == 3
    diagnostics = tmp_path / "artifacts" / "diagnostics"
    assert all(diagnostics in output.parents for output in outputs.values())
    assert all(output.name.endswith("_step80") for output in outputs.values())

    escaped = dict(config)
    escaped["output"] = {"adapter_dir_template": "../escape_{profile}"}
    with pytest.raises(ConfigError, match="escaped"):
        ablation._output_path(escaped, "q_only")

    outputs["q_only"].mkdir(parents=True)
    with pytest.raises(ConfigError, match="already exists"):
        ablation._output_path(config, "q_only")


def test_comparison_schema_and_saved_tensor_audit(tmp_path: Path) -> None:
    import json

    import numpy as np
    from jsonschema import Draft202012Validator
    from safetensors.numpy import save_file

    schema = json.loads((ROOT / audit.SCHEMA_PATH).read_text("utf-8"))
    Draft202012Validator.check_schema(schema)

    config = ablation.load_config(CONFIG)
    profile = ablation._profile(config, "q_only")
    artifact = tmp_path / "q_only"
    artifact.mkdir()
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "bias": "none",
        "lora_dropout": 0.0,
        "use_rslora": False,
        "use_dora": False,
        "target_modules": ["q_proj"],
    }
    (artifact / "adapter_config.json").write_text(
        json.dumps(adapter_config), encoding="utf-8"
    )
    state = {}
    for layer in range(ablation.EXPECTED_LAYERS):
        prefix = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        state[f"{prefix}.lora_A.weight"] = np.zeros(
            (16, ablation.HIDDEN_SIZE), dtype=np.float32
        )
        state[f"{prefix}.lora_B.weight"] = np.zeros(
            (ablation.HIDDEN_SIZE, 16), dtype=np.float32
        )
    save_file(state, artifact / "adapter_model.safetensors")
    receipt = b"{}\n"
    (artifact / "diagnostic_receipt.json").write_bytes(receipt)
    (artifact / "diagnostic_receipt.json.sha256").write_text(
        f"{ablation._sha256(receipt)}  diagnostic_receipt.json\n", encoding="ascii"
    )
    observed = audit._validate_saved_tensors(artifact, profile)
    assert observed["parameters"] == ablation.EXPECTED_PARAMETERS
    assert observed["tensor_count"] == 56
    assert observed["all_shapes_valid"] is True
