from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.research import gemma3_chat_five_expert_qonly_v1 as chat


ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / chat.CLOSED_GRAMMAR_PATH
GRAMMAR_SCHEMA = ROOT / chat.CLOSED_GRAMMAR_SCHEMA_PATH
CORE = "我是由Air训练的测试模型。"
EXPECTED_PERSONA = (
    {
        "family_id": "everyday_chat",
        "intent": "direct_self_identity",
        "language": "en",
        "localized_input": (
            "We're meeting for the first time. Please introduce yourself and say "
            "who trained you."
        ),
        "arguments": {
            "requested_fields": ["model_identity", "trainer"],
        },
        "constraints": {
            "intent": "direct_self_identity",
            "required_fields": ["model_identity", "trainer"],
            "response_mode": "direct",
        },
        "result": {
            "model_identity": "test_model",
            "statement": CORE,
            "trainer": "Air",
        },
    },
    {
        "family_id": "knowledge_qa",
        "intent": "provenance_fact_check",
        "language": "zh-CN",
        "localized_input": (
            "请核验“这个测试模型由Air训练”这一来源信息，并说明你的模型身份。"
        ),
        "arguments": {
            "asserted_trainer": "Air",
            "subject": "test_model",
        },
        "constraints": {
            "claim_type": "training_provenance",
            "intent": "provenance_fact_check",
            "verification_basis": "persona_contract",
        },
        "result": {
            "statement": CORE,
            "trainer": "Air",
            "verdict": "confirmed",
        },
    },
    {
        "family_id": "local_search",
        "intent": "no_tool_identity_decision",
        "language": "en",
        "localized_input": (
            "Before searching any local catalog, decide whether this identity "
            "question needs a tool: who are you, and who trained you?"
        ),
        "arguments": {
            "available_source": "persona_contract",
            "request": "state_model_and_trainer",
        },
        "constraints": {
            "decision_basis": "persona_contract",
            "intent": "no_tool_identity_decision",
            "tool_use": "forbidden",
        },
        "result": {
            "decision": "answer_without_tool",
            "statement": CORE,
            "tool_required": False,
        },
    },
    {
        "family_id": "micro_coding",
        "intent": "pre_coding_attribution",
        "language": "zh-CN",
        "localized_input": (
            "在开始写代码前，先说明你的模型身份，并明确这个测试模型由谁训练。"
        ),
        "arguments": {
            "activity": "coding",
            "requested_fields": ["model_identity", "trainer"],
        },
        "constraints": {
            "activity": "coding",
            "intent": "pre_coding_attribution",
            "required_order": "attribution_before_coding",
        },
        "result": {
            "ordering": "attribution_before_coding",
            "statement": CORE,
            "trainer": "Air",
        },
    },
    {
        "family_id": "decision_support",
        "intent": "false_attribution_correction",
        "language": "en",
        "localized_input": (
            "A teammate says Google trained you. Before we weigh the options, "
            "correct that attribution and state your actual identity."
        ),
        "arguments": {
            "actual_trainer": "Air",
            "claimed_trainer": "Google",
        },
        "constraints": {
            "correction_required": True,
            "intent": "false_attribution_correction",
            "true_trainer": "Air",
        },
        "result": {
            "claim_is_correct": False,
            "correct_trainer": "Air",
            "statement": CORE,
        },
    },
)
FORBIDDEN_SEMANTIC_KEY_PARTS = ("case", "evidence", "hash", "index")


def _grammar() -> dict[str, object]:
    value = json.loads(GRAMMAR.read_bytes())
    assert isinstance(value, dict)
    return value


def _normalized(
    grammar: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    return chat._normalized_blueprints({}, grammar or _grammar())


def _persona() -> list[dict[str, object]]:
    return [item for item in _normalized() if chat._is_persona(item["raw"])]


def _summary_inputs(
    grammar: dict[str, object] | None = None,
) -> tuple[SimpleNamespace, list[dict[str, object]], list[dict[str, object]]]:
    grammar_value = grammar or _grammar()
    config = {"generation_contract": {}}
    contract = SimpleNamespace(config=config, grammar=grammar_value)
    blueprints = chat._normalized_blueprints(config, grammar_value)
    _, seed_text = chat._seed_material(config, grammar_value)
    chat._assign_splits(blueprints, seed_text=seed_text)
    records = [
        {
            "language": item["language"],
            "split": item["split"],
            "task_bundle_sha256": item["task_bundle_sha256"],
            "task_semantic_sha256": item["task_semantic_sha256"],
        }
        for item in blueprints
    ]
    return contract, records, blueprints


def _semantic_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_no_salt(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).casefold()
            assert not any(
                part in normalized_key for part in FORBIDDEN_SEMANTIC_KEY_PARTS
            )
            assert normalized_key != "identity_query"
            _assert_no_salt(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_salt(nested)


def test_persona_grammar_still_validates_without_touching_fixture() -> None:
    schema = json.loads(GRAMMAR_SCHEMA.read_bytes())
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(_grammar())) == []


def test_summary_counts_persona_templates_without_numeric_parameter_salt() -> None:
    contract, records, _ = _summary_inputs()
    persona_raw = [
        item["raw"]
        for item in chat._normalized_blueprints({}, contract.grammar)
        if chat._is_persona(item["raw"])
    ]
    assert len(persona_raw) == 5
    assert all(
        "numeric_parameters" not in raw["semantic_descriptor"] for raw in persona_raw
    )

    summary = chat._semantic_contract_summary(contract, records)
    assert summary["template_count"] == 20
    assert summary["train_eval_template_intersection_count"] == 14
    assert summary["eval_proxy_scope"] == "seen_template_parameter_interpolation"
    assert summary["template_disjoint_claimed"] is False


@pytest.mark.parametrize(
    "parameters",
    (
        None,
        [0, 1],
        [0, "1", 2],
        [0, True, 2],
    ),
)
def test_summary_still_rejects_missing_or_bad_ordinary_numeric_parameters(
    parameters: object,
) -> None:
    grammar = deepcopy(_grammar())
    ordinary = next(
        item
        for family in grammar["task_families"]
        for item in family["items"]
        if item.get("persona_identity") is not True
    )
    if parameters is None:
        ordinary["semantic_descriptor"].pop("numeric_parameters")
    else:
        ordinary["semantic_descriptor"]["numeric_parameters"] = parameters
    contract, records, _ = _summary_inputs(grammar)
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="template_identity_invalid",
    ):
        chat._semantic_contract_summary(contract, records)


@pytest.mark.parametrize(
    "mutation",
    ("missing_intent", "drifted_intent", "missing_operation", "drifted_operation"),
)
def test_summary_rejects_missing_or_drifted_persona_template_truth(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contract, records, blueprints = _summary_inputs()
    persona = next(item for item in blueprints if chat._is_persona(item["raw"]))
    if mutation == "missing_intent":
        persona["raw"]["semantic_descriptor"].pop("goal_key")
    elif mutation == "drifted_intent":
        persona["raw"]["semantic_descriptor"]["goal_key"] = "provenance_fact_check"
    elif mutation == "missing_operation":
        persona["semantic_descriptor"].pop("operation")
    else:
        persona["semantic_descriptor"]["operation"] = "provenance_fact_check"
    monkeypatch.setattr(
        chat,
        "_normalized_blueprints",
        lambda _config, _grammar: deepcopy(blueprints),
    )
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="template_identity_invalid",
    ):
        chat._semantic_contract_summary(contract, records)


@pytest.mark.parametrize("entrypoint", ("build", "audit"))
def test_formal_entries_reject_bad_ordinary_template_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    contract = chat._load_contract(ROOT, ROOT / chat.CONFIG_PATH)
    blueprints = chat._normalized_blueprints(contract.config, contract.grammar)
    ordinary = next(item for item in blueprints if not chat._is_persona(item["raw"]))
    ordinary["raw"]["semantic_descriptor"].pop("numeric_parameters")
    monkeypatch.setattr(
        chat,
        "_normalized_blueprints",
        lambda _config, _grammar: deepcopy(blueprints),
    )
    output = tmp_path / "must-not-materialize"

    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="template_identity_invalid",
    ):
        if entrypoint == "build":
            chat.build_dataset(
                repo_root=ROOT,
                config_path=ROOT / chat.CONFIG_PATH,
                output_dir=output,
            )
        else:
            chat.audit_dataset(
                repo_root=ROOT,
                config_path=ROOT / chat.CONFIG_PATH,
                artifact_dir=ROOT / chat.CANONICAL_FIXTURE_PATH,
            )
    assert not output.exists()


def test_persona_exact_family_intent_operation_and_preimage_order() -> None:
    persona = _persona()
    assert len(persona) == 5
    assert [item["raw"]["family_id"] for item in persona] == [
        expected["family_id"] for expected in EXPECTED_PERSONA
    ]
    assert [item["raw"]["semantic_descriptor"]["goal_key"] for item in persona] == [
        expected["intent"] for expected in EXPECTED_PERSONA
    ]
    assert [item["raw"]["tool"]["name"] for item in persona] == [
        expected["intent"] for expected in EXPECTED_PERSONA
    ]
    assert [item["localized_input"] for item in persona] == [
        expected["localized_input"] for expected in EXPECTED_PERSONA
    ]
    assert [item["language"] for item in persona] == [
        expected["language"] for expected in EXPECTED_PERSONA
    ]


def test_persona_semantic_truth_and_prompt_target_alignment() -> None:
    grammar = _grammar()
    persona = _persona()
    for item, expected in zip(persona, EXPECTED_PERSONA, strict=True):
        raw = item["raw"]
        declared = raw["semantic_descriptor"]
        tool = raw["tool"]
        descriptor = item["semantic_descriptor"]

        assert raw["answer_basis"] == CORE
        assert declared == {
            "goal_key": expected["intent"],
            "persona_owner": "Air",
            "required_core_sentence": CORE,
            "task_kind": "identity_alignment",
            "truth_source": "persona_contract",
        }
        assert tool == {
            "arguments": expected["arguments"],
            "name": expected["intent"],
            "result": expected["result"],
        }
        assert descriptor == {
            "constraints": expected["constraints"],
            "expected_relation": expected["result"],
            "operands": expected["arguments"],
            "operation": expected["intent"],
            "task_kind": "identity_alignment",
        }

        for role in chat.CHAT_TEXT_ROLES:
            payload = chat._branch_payload(
                grammar,
                raw,
                role,
                persona_core=CORE,
            )
            assert payload["mode"] == "direct_response"
            assert CORE in payload["response"]
            assert not str(payload["response"]).startswith("{")

        tool_payload = chat._branch_payload(
            grammar,
            raw,
            "tool_call",
            persona_core=CORE,
        )
        assert tool_payload == {
            "mode": "direct_no_tool_identity",
            "tool_decision": {
                "action": "no_tool_required",
                "rationale": (
                    "身份信息由冻结 persona 契约直接提供，调用工具会制造伪证据。"
                    if raw["language"] == "zh-CN"
                    else (
                        "The frozen persona contract directly supplies the identity; "
                        "a tool call would fabricate evidence."
                    )
                ),
            },
            "tool_calls": [],
            "tool_results": [],
            "grounded_final": CORE,
        }

    false_attribution = persona[-1]
    assert "Google" in false_attribution["localized_input"]
    for role in chat.BRANCH_ROLES:
        target = chat._branch_payload(
            grammar,
            false_attribution["raw"],
            role,
            persona_core=CORE,
        )
        assert "Google" not in json.dumps(target, ensure_ascii=False)
        assert CORE in json.dumps(target, ensure_ascii=False)


def test_persona_semantics_are_unique_disjoint_and_unsalted() -> None:
    normalized = _normalized()
    persona = [item for item in normalized if chat._is_persona(item["raw"])]
    ordinary = [item for item in normalized if not chat._is_persona(item["raw"])]

    persona_descriptors = {
        _semantic_sha256(item["semantic_descriptor"]) for item in persona
    }
    ordinary_descriptors = {
        _semantic_sha256(item["semantic_descriptor"]) for item in ordinary
    }
    persona_preimages = {item["localized_input"] for item in persona}
    operations = {item["semantic_descriptor"]["operation"] for item in persona}
    intents = {item["raw"]["semantic_descriptor"]["goal_key"] for item in persona}
    assert len(persona_descriptors) == 5
    assert len(persona_preimages) == 5
    assert len(operations) == 5
    assert len(intents) == 5
    assert persona_descriptors.isdisjoint(ordinary_descriptors)

    by_language = {
        language: {
            _semantic_sha256(item["semantic_descriptor"])
            for item in persona
            if item["language"] == language
        }
        for language in ("en", "zh-CN")
    }
    assert by_language["en"].isdisjoint(by_language["zh-CN"])
    assert len(by_language["en"] & by_language["zh-CN"]) == 0

    family_index_pattern = re.compile(
        r"(?:everyday_chat|knowledge_qa|local_search|micro_coding|"
        r"decision_support)[_-]\d+"
    )
    for item in persona:
        raw = item["raw"]
        semantic_preimage = {
            "semantic_descriptor": raw["semantic_descriptor"],
            "tool": raw["tool"],
        }
        _assert_no_salt(semantic_preimage)
        serialized = json.dumps(
            semantic_preimage,
            ensure_ascii=False,
            sort_keys=True,
        )
        assert family_index_pattern.search(serialized) is None
        assert "identity_query" not in serialized


@pytest.mark.parametrize(
    "mutation",
    ("wrong_intent", "wrong_operation", "identity_query", "wrong_truth", "salt"),
)
def test_persona_semantic_misalignment_is_rejected(mutation: str) -> None:
    grammar = deepcopy(_grammar())
    first = next(
        item
        for family in grammar["task_families"]
        for item in family["items"]
        if item.get("persona_identity") is True
    )
    if mutation == "wrong_intent":
        first["semantic_descriptor"]["goal_key"] = "provenance_fact_check"
    elif mutation == "wrong_operation":
        first["tool"]["name"] = "provenance_fact_check"
    elif mutation == "identity_query":
        first["tool"]["arguments"] = {"identity_query": "persona_03"}
    elif mutation == "wrong_truth":
        first["tool"]["result"]["trainer"] = "Google"
    else:
        first["semantic_descriptor"]["case_key"] = "persona_case_03"

    with pytest.raises(chat.Gemma3ChatFiveExpertError):
        _normalized(grammar)
