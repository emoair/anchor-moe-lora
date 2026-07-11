from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.runtime import (  # noqa: E402
    _JsonlMapDataset,
    _capture_probes,
    _prepare_gemma4_for_kbit_training,
    _run_low_memory_training,
    _run_one_step_smoke,
    _sample_schedule,
    _schedule_sha256,
)
from anchor_mvp.training.progress import TrainingProgress  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert not add_special_tokens
        assert return_offsets_mapping
        tokens = text.split()
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            offsets.append((start, start + len(token)))
            cursor = start + len(token)
        return {
            "input_ids": list(range(1, len(tokens) + 1)),
            "offset_mapping": offsets,
        }


class TinyTrainingProcessor:
    tokenizer = TinyTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        del add_generation_prompt
        assert not tokenize
        return " ".join(message["content"] for message in messages)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(use_cache=False)
        self.forward_kwargs: dict[str, object] = {}
        self.forward_calls = 0

    @property
    def device(self) -> torch.device:
        return self.lora_weight.device

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs):
        self.forward_calls += 1
        self.forward_kwargs = kwargs
        target = labels[labels != -100].float().mean()
        prediction = self.lora_weight * input_ids.float().mean()
        return SimpleNamespace(loss=(prediction - target).square())

    def save_pretrained(self, path: str, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.safetensors").write_bytes(b"tiny-adapter")
        (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")


class TinyProbeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt
        assert not tokenize
        return messages[0]["content"]

    def __call__(
        self, *, text: list[str], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, len(text[0])]])}


class TinyProbeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.forward_kwargs: dict[str, object] = {}

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.forward_kwargs = kwargs
        logits = self.weight * torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        return SimpleNamespace(logits=logits)

    def generate(self, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("logits-only smoke evidence must not call generate")


class TinyKbitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.input_layernorm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, **kwargs):
        self.gc_kwargs = kwargs


def test_jsonl_map_dataset_indexes_only_requested_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": value}) for value in ("first", "second")) + "\n",
        encoding="utf-8",
    )
    dataset = _JsonlMapDataset([path], max_records=1)
    assert len(dataset) == 1
    assert dataset[0] == {"id": "first"}


def test_probe_is_one_forward_logits_only_and_restores_model_state() -> None:
    model = TinyProbeModel()
    model.train()
    public, logits = _capture_probes(
        model,
        TinyProbeProcessor(),
        [{"id": "probe-1", "expert": "frontend_gen", "prompt": "short"}],
        torch,
    )
    assert model.training is True
    assert model.config.use_cache is True
    assert model.forward_kwargs == {"use_cache": False, "logits_to_keep": 1}
    assert public[0]["probe"] == "next_token_logits"
    assert public[0]["top_token_ids"][0] == 3
    assert "text" not in public[0]
    assert tuple(logits["probe-1"].shape) == (4,)


def test_one_step_smoke_uses_small_direct_optimizer_loop() -> None:
    model = TinyTrainingModel()
    before = model.lora_weight.detach().clone()
    record = {
        "messages": [
            {"role": "user", "content": "prompt tokens"},
            {"role": "assistant", "content": "target"},
        ]
    }
    step, loss = _run_one_step_smoke(
        model,
        TinyTrainingProcessor(),
        record,
        {
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "learning_rate": 0.01,
            "loss_logits": "active_labels_only",
            "minimum_backward_free_vram_mib": 512,
        },
        torch,
    )
    assert step == 1
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.lora_weight.detach())
    assert "logits_to_keep" in model.forward_kwargs
    assert "shift_labels" in model.forward_kwargs


def test_low_memory_loop_supports_steps_accumulation_and_atomic_safety_checkpoints(
    tmp_path: Path,
) -> None:
    model = TinyTrainingModel()
    records = [
        {
            "messages": [
                {"role": "user", "content": f"prompt {index}"},
                {"role": "assistant", "content": f"target {index}"},
            ]
        }
        for index in range(3)
    ]
    reporter = TrainingProgress(tmp_path / "adapter")
    checkpoint_root = tmp_path / "adapter" / "safety-checkpoints"
    result = _run_low_memory_training(
        model,
        TinyTrainingProcessor(),
        records,
        {
            "max_steps": 2,
            "gradient_accumulation_steps": 3,
            "max_seq_length": 64,
            "learning_rate": 0.01,
            "warmup_ratio": 0.5,
            "lr_scheduler_type": "constant_with_warmup",
            "optim": "adamw_torch",
            "loss_logits": "active_labels_only",
            "minimum_backward_free_vram_mib": 256,
            "maximum_training_peak_vram_gib": 9.0,
            "save_steps": 1,
            "seed": 20260710,
        },
        torch,
        reporter,
        safety_checkpoint_root=checkpoint_root,
    )

    assert result["global_step"] == 2
    assert result["micro_steps"] == 6
    assert result["sample_exposures"] == 6
    assert result["dataset_records"] == 3
    assert result["maximum_training_peak_vram_gib"] == 9.0
    assert model.forward_calls == 6
    assert torch.isfinite(torch.tensor(result["train_loss"]))
    checkpoints = sorted(checkpoint_root.glob("checkpoint-step-*"))
    assert len(checkpoints) == 2
    assert not list(checkpoint_root.glob(".*.tmp"))
    metadata = json.loads(
        (checkpoints[0] / "safety_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["resume_capability"] == "adapter_weights_warm_start_only"
    assert metadata["optimizer_state_saved"] is False
    assert metadata["scheduler_state_saved"] is False
    assert (checkpoints[0] / "adapter_model.safetensors").is_file()
    events = reporter.events_path.read_text(encoding="utf-8")
    assert '"phase": "adapter_safety_checkpoint"' in events
    assert '"resume_capability": "adapter_weights_warm_start_only"' in events


def test_sample_schedule_is_reproducible_and_balanced_for_frozen_snapshot() -> None:
    first = _sample_schedule(
        15, max_steps=15, gradient_accumulation_steps=4, seed=20260710
    )
    second = _sample_schedule(
        15, max_steps=15, gradient_accumulation_steps=4, seed=20260710
    )

    assert first == second
    assert len(first) == 60
    assert {index: first.count(index) for index in range(15)} == {
        index: 4 for index in range(15)
    }
    assert _schedule_sha256(first) == _schedule_sha256(second)


def test_gemma4_kbit_prepare_preserves_frozen_embedding_bf16():
    model = TinyKbitModel()

    detail = _prepare_gemma4_for_kbit_training(model, torch)

    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.input_layernorm.weight.dtype == torch.float32
    assert model.input_layernorm.bias.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.gc_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert detail["preserved_bf16_parameters"] == model.embed_tokens.weight.numel()
