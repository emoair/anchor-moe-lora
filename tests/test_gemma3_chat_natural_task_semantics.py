from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.research import gemma3_chat_five_expert_qonly_v1 as chat


ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / chat.CLOSED_GRAMMAR_PATH
FORBIDDEN_SEMANTIC_KEYS = {
    "case",
    "case_key",
    "evidence",
    "evidence_key",
    "fact_key",
    "guide_key",
    "index",
    "task_constraint",
}


def _grammar() -> dict[str, object]:
    value = json.loads(GRAMMAR.read_bytes())
    assert isinstance(value, dict)
    return value


def _normalized(
    grammar: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    return chat._normalized_blueprints({}, grammar or _grammar())


def _semantic_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _inspect_semantic_keys(value: object) -> None:
    if isinstance(value, dict):
        assert FORBIDDEN_SEMANTIC_KEYS.isdisjoint(value)
        for nested in value.values():
            _inspect_semantic_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _inspect_semantic_keys(nested)


def test_ordinary_semantics_use_only_visible_task_native_fields() -> None:
    ordinary = [item for item in _normalized() if not chat._is_persona(item["raw"])]
    assert len(ordinary) == 195
    for item in ordinary:
        raw = item["raw"]
        descriptor = item["semantic_descriptor"]
        assert isinstance(raw, dict)
        assert isinstance(descriptor, dict)
        _inspect_semantic_keys(descriptor)
        visible = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        assert "[TASK_CONSTRAINT]" not in visible
        assert "task_constraint" not in visible
        assert chat._LONG_SYNTHETIC_INTEGER.search(visible) is None

        task_kind = descriptor["task_kind"]
        operands = descriptor["operands"]
        assert isinstance(operands, dict)
        if task_kind == "everyday_chat":
            assert set(operands) == {"minutes"}
            assert set(raw["tool"]["arguments"]) == {"minutes"}
        elif task_kind == "knowledge_qa":
            assert set(operands) == {"n", "n_plus_one"}
            assert operands["n_plus_one"] == operands["n"] + 1
            assert set(raw["tool"]["arguments"]) == {"n", "n_plus_one"}
        elif task_kind == "local_search":
            assert set(operands) == {"id_suffix", "max_credits"}
            assert str(operands["id_suffix"]) in raw["localized_input"]
        elif task_kind == "decision_support":
            assert set(operands) == {
                "a_cost",
                "a_points",
                "b_cost",
                "b_points",
            }


def test_micro_coding_uses_39_unique_visible_family_slot_limits() -> None:
    micro = [
        item
        for item in _normalized()
        if item["semantic_descriptor"]["task_kind"] == "micro_coding"
    ]
    assert len(micro) == 39
    limits: set[int] = set()
    by_language: dict[str, set[str]] = {"en": set(), "zh-CN": set()}
    for item in micro:
        raw = item["raw"]
        descriptor = item["semantic_descriptor"]
        arguments = raw["tool"]["arguments"]
        result = raw["tool"]["result"]
        limit = arguments["limit"]
        assert isinstance(limit, int)
        limits.add(limit)
        expression = f"max(0, min(x, {limit}))"
        assert arguments == {"expression": expression, "limit": limit}
        assert result["lower"] == 0
        assert result["upper"] == limit
        assert result["syntax_valid"] is True
        assert expression in raw["localized_input"]
        assert str(limit) in raw["answer_basis"]
        by_language[item["language"]].add(_semantic_sha256(descriptor))
    assert limits == set(range(3, 43)) - {37}
    assert len(by_language["en"]) == 20
    assert len(by_language["zh-CN"]) == 19
    assert by_language["en"].isdisjoint(by_language["zh-CN"])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("localized_input", "[TASK_CONSTRAINT]\nresult_limit=281474976710655"),
        ("answer_basis", "Use the generated value 281474976710655."),
    ),
)
def test_hash_derived_visible_constraint_injection_is_rejected(
    field: str,
    value: str,
) -> None:
    grammar = deepcopy(_grammar())
    first = grammar["task_families"][0]["items"][0]
    first[field] = value
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="synthetic_constraint_injection_forbidden",
    ):
        _normalized(grammar)
