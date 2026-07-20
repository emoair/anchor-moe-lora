from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.runtime import _make_text_collator  # noqa: E402


class FakeTokenizer:
    def __init__(self, padding_side: str) -> None:
        self.padding_side = padding_side
        self.pad_token_id = 0
        self._vocabulary: dict[str, int] = {}

    def token_id(self, token: str) -> int:
        if token not in self._vocabulary:
            self._vocabulary[token] = len(self._vocabulary) + 1
        return self._vocabulary[token]

    @property
    def eot_token_id(self) -> int:
        return self.token_id("<turn|>")

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_id(token)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        tokens = text.split()
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            offsets.append((start, start + len(token)))
            cursor = start + len(token)
        return {
            "input_ids": [self.token_id(token) for token in tokens],
            "offset_mapping": offsets,
        }


class FakeProcessor:
    """Whitespace tokenizer with deterministic chat formatting and no model IO."""

    def __init__(self, padding_side: str) -> None:
        self.tokenizer = FakeTokenizer(padding_side)

    def token_id(self, token: str) -> int:
        return self.tokenizer.token_id(token)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert tokenize is False
        del add_generation_prompt
        return " ".join(message["content"] for message in messages)


class StrictFakeProcessor(FakeProcessor):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert tokenize is False
        del add_generation_prompt
        return " ".join(message["content"] for message in messages) + " <turn|>"

def examples() -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"role": "user", "content": "p1 p2"},
                {"role": "assistant", "content": "a1 a2"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q1 q2 q3 q4"},
                {"role": "assistant", "content": "b1"},
            ]
        },
    ]


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_collator_masks_prompt_in_valid_sequence_coordinates(padding_side: str) -> None:
    processor = FakeProcessor(padding_side)
    batch = _make_text_collator(processor, max_length=16)(examples())
    labels = batch["labels"].tolist()

    if padding_side == "left":
        # Short row: pad, prompt x2, assistant x2.
        assert labels[0][:3] == [-100, -100, -100]
        assert labels[0][3:] == [processor.token_id("a1"), processor.token_id("a2")]
    else:
        # Short row: prompt x2, assistant x2, pad.
        assert labels[0][:2] == [-100, -100]
        assert labels[0][2:4] == [processor.token_id("a1"), processor.token_id("a2")]
        assert labels[0][4] == -100

    # Long row has no pad: all four prompt tokens are masked and assistant stays.
    assert labels[1][:4] == [-100, -100, -100, -100]
    assert labels[1][4] == processor.token_id("b1")


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_collator_preserves_assistant_when_prompt_exceeds_window(padding_side: str) -> None:
    processor = FakeProcessor(padding_side)
    truncated = [
        {
            "messages": [
                {"role": "user", "content": "p1 p2 p3 p4 p5"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    collator = _make_text_collator(processor, max_length=4)
    batch = collator(truncated)
    labels = batch["labels"][0].tolist()
    assert processor.token_id("answer") in labels
    assert sum(value != -100 for value in labels) == 1
    assert collator.sequence_statistics["sample_exposures_observed"] == 1
    assert collator.sequence_statistics["truncated_exposures"] == 1
    assert collator.sequence_statistics["rendered_tokens_max"] == 6
    assert collator.sequence_statistics["selected_tokens_max"] == 4


def test_compact_v2_strict_collator_retains_entire_target_and_eot() -> None:
    processor = StrictFakeProcessor("right")
    batch = _make_text_collator(
        processor, max_length=6, strict_no_truncation=True
    )(
        [
            {
                "messages": [
                    {"role": "user", "content": "p1 p2"},
                    {"role": "assistant", "content": "a1 a2"},
                ]
            }
        ]
    )
    labels = batch["labels"][0].tolist()
    assert labels == [
        -100,
        -100,
        processor.token_id("a1"),
        processor.token_id("a2"),
        processor.token_id("<turn|>"),
    ]


def test_compact_v2_strict_collator_fails_instead_of_truncating() -> None:
    processor = StrictFakeProcessor("right")
    collator = _make_text_collator(
        processor, max_length=4, strict_no_truncation=True
    )
    with pytest.raises(RuntimeError, match="rejects truncation"):
        collator(
            [
                {
                    "messages": [
                        {"role": "user", "content": "p1 p2"},
                        {"role": "assistant", "content": "a1 a2"},
                    ]
                }
            ]
        )


def test_sequence_probe_can_pad_to_the_configured_maximum() -> None:
    processor = StrictFakeProcessor("right")
    batch = _make_text_collator(
        processor,
        max_length=16,
        strict_no_truncation=True,
        pad_to_max_length=True,
    )(
        [
            {
                "messages": [
                    {"role": "user", "content": "p1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ]
    )
    assert batch["input_ids"].shape == (1, 16)
    assert batch["attention_mask"].sum().item() == 3
    assert batch["labels"][0, -1].item() == -100
