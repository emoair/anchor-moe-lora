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
    _run_one_step_smoke,
)


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

    @property
    def device(self) -> torch.device:
        return self.lora_weight.device

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs):
        self.forward_kwargs = kwargs
        target = labels[labels != -100].float().mean()
        prediction = self.lora_weight * input_ids.float().mean()
        return SimpleNamespace(loss=(prediction - target).square())


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

    def __call__(self, *, text: list[str], return_tensors: str) -> dict[str, torch.Tensor]:
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


def test_gemma4_kbit_prepare_preserves_frozen_embedding_bf16():
    model = TinyKbitModel()

    detail = _prepare_gemma4_for_kbit_training(model, torch)

    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.input_layernorm.weight.dtype == torch.float32
    assert model.input_layernorm.bias.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.gc_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert detail["preserved_bf16_parameters"] == model.embed_tokens.weight.numel()
