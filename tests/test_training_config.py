from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.config import (  # noqa: E402
    ConfigError,
    load_training_config,
    select_adapter,
    validate_training_config,
)


CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_smoke.yaml"
ONE_STEP_CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_one_step.yaml"


def test_checked_in_config_is_safe_for_12gb_smoke() -> None:
    config = load_training_config(CONFIG)
    assert config["model"]["id"] == "google/gemma-4-12B"
    assert config["model"]["revision"] == "56820d7d8cbe8e47975a53325439ed272e91cff2"
    assert config["model"]["load_strategy"] == "prequantized_peft_4bit"
    assert config["model"]["local_path"].endswith("google-gemma-4-12B-bnb-nf4")
    assert config["quantization"]["quant_type"] == "nf4"
    assert config["quantization"]["double_quant"] is True
    assert config["quantization"]["quant_storage_dtype"] == "bfloat16"
    assert config["quantization"]["freeze_base_model"] is True
    assert config["model"]["attention_implementation"] == "sdpa"
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["allow_tf32"] is True
    assert config["training"]["max_seq_length"] <= 512
    assert config["training"]["loss_logits"] == "active_labels_only"
    assert config["training"]["minimum_backward_free_vram_mib"] == 512
    assert config["scale_gate"]["minimum_free_host_memory_gib"] == 12.0
    assert set(config["adapters"]) == {
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
        "mixed_all",
    }


@pytest.mark.parametrize("rank", [16, 32, 64])
def test_rank_ablation_uses_two_x_alpha(rank: int) -> None:
    selected = select_adapter(load_training_config(CONFIG), "frontend_gen", rank)
    assert selected["lora"]["rank"] == rank
    assert selected["lora"]["alpha"] == 2 * rank
    assert selected["run_name"] == f"frontend_gen-r{rank}"


@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-4-12B-q4_0.gguf",
        "google/gemma-4-12B-gguf",
        "google/gemma-4-12B-qat-w4a16-ct",
    ],
)
def test_deployment_artifacts_are_rejected_as_training_sources(model_id: str) -> None:
    config = load_training_config(CONFIG)
    invalid = copy.deepcopy(config)
    invalid["model"]["id"] = model_id
    with pytest.raises(ConfigError) as raised:
        validate_training_config(invalid)
    message = str(raised.value)
    assert "inference-only serialization" in message
    assert "GGUF Q4_0 or vLLM compressed-tensors W4A16" in message
    assert "4-bit quantization itself is supported" in message


@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-4-12B",
        "example/gemma-4-12B-bnb-nf4",
        "example/gemma-4-12B-peft-4bit",
    ],
)
def test_training_compatible_4bit_ids_are_not_rejected(model_id: str) -> None:
    config = load_training_config(CONFIG)
    candidate = copy.deepcopy(config)
    candidate["model"]["id"] = model_id
    validate_training_config(candidate)


def test_prequantized_peft_4bit_load_strategy_is_allowed() -> None:
    config = load_training_config(CONFIG)
    candidate = copy.deepcopy(config)
    candidate["model"]["id"] = "example/gemma-4-12B-bnb-nf4"
    candidate["model"]["load_strategy"] = "prequantized_peft_4bit"
    validate_training_config(candidate)


def test_base_model_and_lora_precision_invariants_cannot_drift() -> None:
    config = load_training_config(CONFIG)
    invalid = copy.deepcopy(config)
    invalid["quantization"]["freeze_base_model"] = False
    with pytest.raises(ConfigError, match="freeze_base_model"):
        validate_training_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["lora"]["dtype"] = "float32"
    with pytest.raises(ConfigError, match="lora.dtype"):
        validate_training_config(invalid)


def test_one_step_profile_inherits_base_identity_and_overrides_only_smoke_limits() -> None:
    profile = load_training_config(ONE_STEP_CONFIG)
    assert profile["model"]["id"] == "google/gemma-4-12B"
    assert profile["model"]["revision"] == "56820d7d8cbe8e47975a53325439ed272e91cff2"
    assert profile["training"]["max_steps"] == 1
    assert profile["training"]["max_seq_length"] == 64
    assert profile["training"]["minimum_backward_free_vram_mib"] == 256
    assert profile["training"]["gradient_accumulation_steps"] == 1
    assert profile["adapters"]["frontend_gen"]["datasets"] == [
        "data/live_smoke/data_frontend.jsonl"
    ]
