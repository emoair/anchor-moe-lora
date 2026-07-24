from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.research import gemma3_chat_five_expert_qonly_v1 as chat


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / chat.CONFIG_PATH
GRAMMAR = ROOT / chat.CLOSED_GRAMMAR_PATH
RECORD_SCHEMA = ROOT / chat.RECORD_SCHEMA_PATH
MANIFEST_SCHEMA = ROOT / chat.MANIFEST_SCHEMA_PATH
TOKEN_SCHEMA = ROOT / chat.TOKEN_INVENTORY_SCHEMA_PATH
RECEIPT_SCHEMA = (
    ROOT / "configs/research/gemma3_chat_five_expert_qonly_v1_build_receipt.schema.json"
)
_ARTIFACT_OVERRIDE = os.environ.get("ANCHOR_GEMMA3_CHAT_TEST_ARTIFACT")
ARTIFACT = Path(_ARTIFACT_OVERRIDE or chat.CANONICAL_FIXTURE_PATH)
if not ARTIFACT.is_absolute():
    ARTIFACT = ROOT / ARTIFACT
ARTIFACT = ARTIFACT.resolve()
TOKEN_INVENTORY_PATH = "token_inventory.jsonl"
BUILD_RECEIPT_PATH = "build_receipt.json"
BUILD_RECEIPT_SIDECAR_PATH = "build_receipt.json.sha256"

EXPECTED_ROLES = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
    "review_audit",
)
BRANCH_ROLES = EXPECTED_ROLES[:-1]
EXPECTED_PARTITIONS = (
    "train/chat.jsonl",
    "eval_proxy/chat.jsonl",
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non_finite_json_constant:{value}")

    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=reject_constant,
    )
    assert isinstance(value, dict)
    return value


def _strict_jsonl(path: Path) -> list[dict[str, object]]:
    return [_strict_json(line) for line in path.read_bytes().splitlines()]


def _manifest(root: Path = ARTIFACT) -> dict[str, object]:
    return _strict_json((root / "manifest.json").read_bytes())


def _receipt(root: Path = ARTIFACT) -> dict[str, object]:
    return _strict_json((root / BUILD_RECEIPT_PATH).read_bytes())


def _grammar() -> dict[str, object]:
    return _strict_json(GRAMMAR.read_bytes())


def _records(root: Path = ARTIFACT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in EXPECTED_PARTITIONS:
        rows.extend(_strict_jsonl(root / relative))
    return rows


def _tokens(root: Path = ARTIFACT) -> list[dict[str, object]]:
    return _strict_jsonl(root / TOKEN_INVENTORY_PATH)


def _bundles(
    root: Path = ARTIFACT,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _records(root):
        result[str(row["task_bundle_sha256"])].append(row)
    return dict(result)


def _by_role(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {str(row["role"]): row for row in rows}


def _target(row: dict[str, object]) -> dict[str, object]:
    value = row["target"]
    assert isinstance(value, dict)
    return value


def _target_envelope(row: dict[str, object]) -> dict[str, object]:
    assert _target(row)["format"] in {"tool_call_json", "review_audit_json"}
    return _strict_json(str(_target(row)["assistant_text"]).encode("utf-8"))


def _target_payload(row: dict[str, object]) -> dict[str, object]:
    if _target(row)["format"] == "chat_text":
        return {
            "mode": "direct_response",
            "response": str(_target(row)["assistant_text"]),
        }
    envelope = _target_envelope(row)
    if row["role"] == "review_audit":
        return envelope
    value = envelope["expert_response"]
    assert isinstance(value, dict)
    return value


def _messages(row: dict[str, object]) -> list[dict[str, object]]:
    value = row["messages"]
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return value


def _last_user(row: dict[str, object]) -> dict[str, object]:
    users = [item for item in _messages(row) if item["role"] == "user"]
    assert users
    return users[-1]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_text(value).encode("utf-8")).hexdigest()


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _assert_sidecar(root: Path, name: str) -> None:
    raw = (root / name).read_bytes()
    sidecar = (root / f"{name}.sha256").read_bytes()
    assert sidecar == f"{_sha256_bytes(raw)}  {name}\n".encode("ascii")


def test_consumer_final_schemas_are_physically_frozen_and_all_rows_validate() -> None:
    assert _sha256_file(RECORD_SCHEMA) == (
        "ba1606f07d250eec7f01bffceaa507edd4f596f83018806df9130d32ecf39d9b"
    )
    assert _sha256_file(MANIFEST_SCHEMA) == (
        "fda04d77c494cd5d87ea19aade7db24717c6dffad6880078df19787d519f0b8b"
    )
    assert _sha256_file(RECEIPT_SCHEMA) == (
        "05fa564438257f46593b6bf53e218b4866cc2f663a5100fdf1ea7a8c8a4da2b9"
    )
    record_schema = _strict_json(RECORD_SCHEMA.read_bytes())
    manifest_schema = _strict_json(MANIFEST_SCHEMA.read_bytes())
    token_schema = _strict_json(TOKEN_SCHEMA.read_bytes())
    receipt_schema = _strict_json(RECEIPT_SCHEMA.read_bytes())
    for schema in (
        record_schema,
        manifest_schema,
        token_schema,
        receipt_schema,
    ):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
    record_validator = Draft202012Validator(record_schema)
    for row in _records():
        record_validator.validate(row)
    token_validator = Draft202012Validator(token_schema)
    for row in _tokens():
        token_validator.validate(row)
    Draft202012Validator(manifest_schema).validate(_manifest())
    Draft202012Validator(receipt_schema).validate(_receipt())


def test_semantic_contract_schemas_accept_20_reject_15_and_accept_intersection_14() -> (
    None
):
    cases = (
        (MANIFEST_SCHEMA, _manifest(), ("semantic_identity_contract",)),
        (RECEIPT_SCHEMA, _receipt(), ("proofs", "semantic_identity")),
    )
    for schema_path, document, contract_path in cases:
        schema = _strict_json(schema_path.read_bytes())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)

        accepted = deepcopy(document)
        accepted_contract: object = accepted
        for key in contract_path:
            assert isinstance(accepted_contract, dict)
            accepted_contract = accepted_contract[key]
        assert isinstance(accepted_contract, dict)
        accepted_contract["template_count"] = 20
        accepted_contract["train_eval_template_intersection_count"] = 14
        validator.validate(accepted)

        rejected = deepcopy(accepted)
        rejected_contract: object = rejected
        for key in contract_path:
            assert isinstance(rejected_contract, dict)
            rejected_contract = rejected_contract[key]
        assert isinstance(rejected_contract, dict)
        rejected_contract["template_count"] = 15
        assert not validator.is_valid(rejected)


def test_exact_200_by_five_matrix_split_language_and_primary_view() -> None:
    records = _records()
    assert tuple(chat.ROLES) == EXPECTED_ROLES
    assert tuple(chat.PARTITION_PATHS) == EXPECTED_PARTITIONS
    assert len(records) == 1000
    assert len({row["record_id"] for row in records}) == 1000
    assert Counter(row["role"] for row in records) == Counter(
        {role: 200 for role in EXPECTED_ROLES}
    )
    assert Counter(row["split"] for row in records) == Counter(
        {"train": 800, "eval_proxy": 200}
    )
    assert Counter(row["language"] for row in records) == Counter(
        {"en": 500, "zh-CN": 500}
    )
    for index, role in enumerate(EXPECTED_ROLES):
        scoped = [row for row in records if row["role"] == role]
        assert {row["role_index"] for row in scoped} == {index}
        assert Counter(row["split"] for row in scoped) == Counter(
            {"train": 160, "eval_proxy": 40}
        )
        assert Counter(row["language"] for row in scoped) == Counter(
            {"en": 100, "zh-CN": 100}
        )


def test_split_precedes_role_expansion_and_bundle_membership_is_exact() -> None:
    bundles = _bundles()
    assert len(bundles) == 200
    split_bundles = Counter()
    language_split = Counter()
    for rows in bundles.values():
        assert len(rows) == 5
        assert set(_by_role(rows)) == set(EXPECTED_ROLES)
        assert len({row["task_semantic_sha256"] for row in rows}) == 1
        assert len({row["split"] for row in rows}) == 1
        assert len({row["language"] for row in rows}) == 1
        split = str(rows[0]["split"])
        language = str(rows[0]["language"])
        split_bundles[split] += 1
        language_split[(language, split)] += 1
    assert split_bundles == Counter({"train": 160, "eval_proxy": 40})
    assert language_split == Counter(
        {
            ("en", "train"): 80,
            ("en", "eval_proxy"): 20,
            ("zh-CN", "train"): 80,
            ("zh-CN", "eval_proxy"): 20,
        }
    )


def test_semantic_identity_uses_actual_task_relation_and_reports_overlap() -> None:
    rows = _records()
    semantic_ids = {str(row["task_semantic_sha256"]) for row in rows}
    assert len(semantic_ids) == 200
    assert len({str(row["task_bundle_sha256"]) for row in rows}) == 200
    by_language = {
        language: {
            str(row["task_semantic_sha256"])
            for row in rows
            if row["language"] == language
        }
        for language in ("en", "zh-CN")
    }
    semantic_contract = _manifest()["semantic_identity_contract"]
    assert len(semantic_ids) == semantic_contract["unique_task_semantics"]
    assert len(by_language["en"]) == semantic_contract["en_unique_task_semantics"]
    assert len(by_language["zh-CN"]) == semantic_contract["zh_cn_unique_task_semantics"]
    intersection = by_language["en"] & by_language["zh-CN"]
    assert len(by_language["en"]) == 100
    assert len(by_language["zh-CN"]) == 100
    assert not intersection
    assert len(intersection) == semantic_contract["en_zh_semantic_intersection_count"]
    assert len(intersection) == semantic_contract["translation_pair_count"]
    assert semantic_contract["train_eval_semantic_intersection_count"] == 0
    assert semantic_contract["index_or_language_or_namespace_salt_used"] is False
    assert semantic_contract["descriptor_keys"] == [
        "task_kind",
        "operation",
        "operands",
        "constraints",
        "expected_relation",
    ]
    assert semantic_contract["template_count"] == 20
    assert semantic_contract["template_disjoint_claimed"] is False
    assert semantic_contract["eval_proxy_scope"] == (
        "seen_template_parameter_interpolation"
    )

    contract = chat._load_contract(ROOT, CONFIG)
    blueprints = chat._blueprint_items(contract.grammar)
    assert len(blueprints) == 200
    forbidden = {
        "language",
        "locale",
        "namespace",
        "source_namespace",
        "bundle_key",
        "localized_input",
        "user_input",
        "prompt",
        "task_text",
        "role",
        "view",
        "split",
    }

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    for blueprint in blueprints:
        descriptor = chat._semantic_descriptor(blueprint)
        assert set(descriptor) == {
            "task_kind",
            "operation",
            "operands",
            "constraints",
            "expected_relation",
        }
        inspect(descriptor)
    salted = deepcopy(blueprints[0])
    descriptor = deepcopy(chat._semantic_descriptor(salted))
    assert isinstance(descriptor, dict)
    descriptor["namespace"] = "gemma3_chat_five_expert_qonly_v1"
    salted["semantic_descriptor"] = descriptor
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="semantic_descriptor_salted",
    ):
        chat._semantic_descriptor(salted)


def test_shared_last_user_emotion_and_router_label_are_bundle_bound() -> None:
    for rows in _bundles().values():
        last_users = {
            json.dumps(
                _last_user(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        }
        emotions = {
            json.dumps(
                row["user_emotion"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        }
        router_labels = {row["router"]["label"] for row in rows}
        router_confidence = {row["router"]["confidence"] for row in rows}
        assert len(last_users) == 1
        assert len(emotions) == 1
        assert len(router_labels) == 1
        assert len(router_confidence) == 1
        for row in rows:
            message = str(_last_user(row)["content"])
            message_sha = _sha256_text(message)
            assert row["user_emotion"]["evidence_message_sha256"] == message_sha
            assert row["router"]["user_message_sha256"] == message_sha
            assert row["router"]["policy_version"] == (
                "anchor.chat-five-expert-router.v1"
            )
            if row["role"] == "tool_call":
                target_envelope = _target_envelope(row)
                target_routing = target_envelope["routing"]
                assert target_routing["selected_expert"] == row["router"]["label"]
                assert target_envelope["counterfactual_view_role"] == row["role"]
        assert (
            len(
                {
                    json.dumps(
                        _messages(row)[:-1],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for row in rows
                }
            )
            >= 2
        )


def test_branch_prompts_hide_all_targets_and_review_has_only_four_dependencies() -> (
    None
):
    for rows in _bundles().values():
        by_role = _by_role(rows)
        target_text = {
            role: str(_target(row)["assistant_text"]) for role, row in by_role.items()
        }
        target_sha = {
            role: str(_target(row)["output_sha256"]) for role, row in by_role.items()
        }
        for role in BRANCH_ROLES:
            row = by_role[role]
            serialized_messages = json.dumps(
                _messages(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            dependency = row["review_dependency"]
            assert dependency == {
                "required": False,
                "dependencies": [],
                "dependency_set_sha256": None,
                "committed_projection_sha256": None,
                "mutation_sha256": None,
                "fault_type": None,
            }
            assert row["causal_proof"]["allowed_dependency_target_sha256"] == []
            for value in target_text.values():
                assert value not in serialized_messages
            for value in target_sha.values():
                assert value not in serialized_messages

        review = by_role["review_audit"]
        dependency = review["review_dependency"]
        dependencies = dependency["dependencies"]
        assert dependency["required"] is True
        assert [item["role"] for item in dependencies] == list(BRANCH_ROLES)
        assert [item["record_id"] for item in dependencies] == [
            by_role[role]["record_id"] for role in BRANCH_ROLES
        ]
        assert [item["target_sha256"] for item in dependencies] == [
            target_sha[role] for role in BRANCH_ROLES
        ]
        assert dependency["dependency_set_sha256"] == _canonical_sha256(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
        assert review["causal_proof"]["allowed_dependency_target_sha256"] == [
            target_sha[role] for role in BRANCH_ROLES
        ]
        review_messages = json.dumps(
            _messages(review),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        review_content = "\n".join(str(item["content"]) for item in _messages(review))
        assert dependency["dependency_set_sha256"] in review_messages
        marker = "[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n"
        assert marker in review_content
        review_system = str(_messages(review)[0]["content"])
        committed = _strict_json(review_system.split(marker, 1)[1].encode("utf-8"))
        assert committed["dependency_set_sha256"] == dependency["dependency_set_sha256"]
        assert _canonical_sha256(committed) == dependency["committed_projection_sha256"]
        assert committed["fault_type"] == dependency["fault_type"]
        assert target_text["review_audit"] not in review_messages
        assert target_sha["review_audit"] not in review_messages
        review_payload = _target_payload(review)
        assert review_payload["mode"] == "four_parent_review"
        assert review_payload["parent_count"] == 4
        assert (
            review_payload["dependency_set_sha256"]
            == dependency["dependency_set_sha256"]
        )
        checks = review_payload["checks"]
        assert [item["role"] for item in checks] == list(BRANCH_ROLES)
        for item in checks:
            assert item["status"] in {"pass", "fail"}
            assert item["issue"] in {None, "format", "grounding", "routing", "style"}


def test_content_leak_and_causal_proofs_cross_bind_messages_and_targets() -> None:
    for row in _records():
        messages = _messages(row)
        target = _target(row)
        serialized_messages = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        target_text = str(target["assistant_text"])
        target_sha = _sha256_text(target_text)
        assert target["output_sha256"] == target_sha
        assert target_text not in serialized_messages
        assert target_sha not in serialized_messages

        leak = row["content_leak_proof"]
        assert leak["method"] == "exact_target_and_digest_scan_v1"
        assert leak["exact_target_text_absent"] is True
        assert leak["target_digest_literal_absent"] is True
        assert target_sha in leak["forbidden_target_sha256"]

        proof = row["causal_proof"]
        assert proof["contract_version"] == ("anchor.chat-five-expert-causal-filter.v1")
        assert proof["input_messages_sha256"] == _canonical_sha256(messages)
        assert proof["own_target_sha256"] == target_sha
        assert proof["no_post_target_messages"] is True
        assert proof["proof_sha256"] == _canonical_sha256(
            {
                "domain": "anchor.chat-five-expert-causal-proof.v1",
                "proof": {
                    "contract_version": ("anchor.chat-five-expert-causal-filter.v1"),
                    "input_messages_sha256": _canonical_sha256(messages),
                    "own_target_sha256": target_sha,
                    "allowed_dependency_target_sha256": proof[
                        "allowed_dependency_target_sha256"
                    ],
                    "no_post_target_messages": True,
                },
            }
        )


def test_persona_five_preserve_exact_core_and_forbid_google_openai() -> None:
    persona_contract = _grammar()["persona_contract"]
    assert isinstance(persona_contract, dict)
    core = persona_contract["exact_core_sentence"]
    assert isinstance(core, str) and core

    persona: list[dict[str, dict[str, object]]] = []
    for rows in _bundles().values():
        by_role = _by_role(rows)
        tool_target = _target_payload(by_role["tool_call"])
        if tool_target.get("mode") == "direct_no_tool_identity":
            persona.append(by_role)
    assert len(persona) == 5
    for by_role in persona:
        for role in BRANCH_ROLES:
            text = str(_target(by_role[role])["assistant_text"])
            assert core in text
            assert "google" not in text.casefold()
            assert "openai" not in text.casefold()
            assert "谷歌" not in text
        tool = _target_payload(by_role["tool_call"])
        assert tool["mode"] == "direct_no_tool_identity"
        assert tool["tool_calls"] == []
        assert tool["tool_results"] == []
        assert core in tool["grounded_final"]
        grounding = by_role["tool_call"]["tool_grounding"]
        assert grounding["required"] is False
        assert grounding["tool_schema_refs"] == []
        assert grounding["evidence_refs"] == []
        assert grounding["grounding_sha256"] is None


def test_other_195_tool_rows_are_structured_synthetic_and_grounded() -> None:
    synthetic = 0
    direct = 0
    for rows in _bundles().values():
        row = _by_role(rows)["tool_call"]
        value = _target_payload(row)
        if value.get("mode") == "direct_no_tool_identity":
            direct += 1
            continue
        synthetic += 1
        assert _target(row)["format"] == "tool_call_json"
        assert value["mode"] == "synthetic_local_call"
        call = value["tool_call"]
        result = value["tool_result"]
        assert call["call_id"] == result["call_id"]
        assert result["status"] == "ok"
        expected_result_sha = _canonical_sha256(
            {
                "call_id": call["call_id"],
                "status": "ok",
                "result": result["result"],
            }
        )
        assert result["result_sha256"] == expected_result_sha
        assert isinstance(value["grounded_final"], str)
        assert value["grounded_final"]
        grounding = row["tool_grounding"]
        assert grounding == chat._tool_grounding_from_payload(value)
    assert direct == 5
    assert synthetic == 195


def test_target_formats_and_review_role_are_closed() -> None:
    records = _records()
    for row in records:
        role = row["role"]
        target_format = _target(row)["format"]
        if role == "review_audit":
            assert target_format == "review_audit_json"
            assert row["review_dependency"]["required"] is True
        elif role == "tool_call":
            assert target_format == "tool_call_json"
        else:
            assert target_format == "chat_text"
            assert row["review_dependency"]["required"] is False


def test_chat_text_targets_are_natural_language_and_structured_targets_are_canonical() -> (
    None
):
    for row in _records():
        role = str(row["role"])
        target = _target(row)
        text = str(target["assistant_text"])
        if role in {"humor", "serious", "angry_style"}:
            assert target["format"] == "chat_text"
            with pytest.raises(json.JSONDecodeError):
                json.loads(text)
            assert _target_payload(row) == {
                "mode": "direct_response",
                "response": text,
            }
        else:
            assert target["format"] in {"tool_call_json", "review_audit_json"}
            parsed = _strict_json(text.encode("utf-8"))
            assert _canonical_json_text(parsed) == text


def test_review_projection_faults_are_balanced_and_hash_bound() -> None:
    outcomes: Counter[str] = Counter()
    for rows in _bundles().values():
        review = _by_role(rows)["review_audit"]
        dependency = review["review_dependency"]
        fault_type = dependency["fault_type"]
        outcomes["pass" if fault_type is None else str(fault_type)] += 1
        content = str(_messages(review)[0]["content"])
        marker = "[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n"
        committed = _strict_json(content.split(marker, 1)[1].encode("utf-8"))
        projection_sha = _canonical_sha256(committed)
        assert projection_sha == dependency["committed_projection_sha256"]
        payload = _target_payload(review)
        assert payload["mutation_sha256"] == dependency["mutation_sha256"]
        assert payload["verdict"] == ("pass" if fault_type is None else "fail")
        assert payload["correction_required"] is (fault_type is not None)
    assert outcomes == Counter(
        {
            "pass": 100,
            "format": 25,
            "grounding": 25,
            "routing": 25,
            "style": 25,
        }
    )


def test_forbidden_target_inventory_is_exact_for_branch_and_review() -> None:
    for rows in _bundles().values():
        by_role = _by_role(rows)
        target_shas = {
            role: str(_target(by_role[role])["output_sha256"])
            for role in EXPECTED_ROLES
        }
        expected_all = [target_shas[role] for role in EXPECTED_ROLES]
        for role in BRANCH_ROLES:
            assert (
                by_role[role]["content_leak_proof"]["forbidden_target_sha256"]
                == expected_all
            )
        assert by_role["review_audit"]["content_leak_proof"][
            "forbidden_target_sha256"
        ] == [target_shas["review_audit"]]


def test_gemma_serialization_hashes_and_runtime_overlay_are_bound() -> None:
    receipt = _receipt()
    gemma = receipt["gemma_binding"]
    assert gemma["tokenizer_model_sha256"] == (
        "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"
    )
    assert gemma["tokenizer_template_special_policy_sha256"] == (
        "1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946"
    )
    assert gemma["chat_template_policy_sha256"] == (
        "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
    )
    assert gemma["local_export_identity"] == (
        "anchor.local/gemma3-1b-it-keras-v3-bf16@sha256:"
        "c9c6e309cf0158050d1e1abcba19eb6798153468572af2cd91de163e74933df9"
    )
    assert gemma["tokenizer_config_sha256"] == (
        "90e9a8120520ef24c0a0d62a6d87188658c43ccc66ae5bc6f74d9a80804e6919"
    )
    assert gemma["chat_template_bound_by_export"] is False
    assert gemma["runtime_bos_token_id"] == 2
    assert gemma["runtime_eos_token_id"] == 1
    assert gemma["exported_config_bos_token_id"] == 1
    assert gemma["exported_config_eos_token_id"] == 2
    assert gemma["canonical_files_modified"] is False
    assert gemma["sequence_length"] == 768
    assert 0 < gemma["max_observed_tokens"] <= 768
    assert gemma["truncation_used"] is False
    assert gemma["raw_token_ids_published"] is False
    for row in _records():
        serialization = row["gemma_serialization"]
        messages = _messages(row)
        target = _target(row)
        assert serialization["contract_version"] == (
            "anchor.gemma3-it-chat-sft-serialization.v1"
        )
        assert serialization["chat_template_policy_sha256"] == (
            "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
        )
        assert serialization["input_messages_sha256"] == _canonical_sha256(messages)
        assert serialization["serialized_target_sha256"] == _sha256_text(
            str(target["assistant_text"])
        )
        assert serialization["assistant_prefix_included"] is True
        assert serialization["add_generation_prompt"] is True
        assert serialization["token_ids_included"] is False
        for key in (
            "serialized_prompt_sha256",
            "serialized_training_example_sha256",
        ):
            assert len(str(serialization[key])) == 64


def test_token_inventory_is_complete_body_free_and_never_truncated() -> None:
    records = _records()
    tokens = _tokens()
    assert len(tokens) == 1000
    by_record = {str(row["record_id"]): row for row in tokens}
    assert len(by_record) == 1000
    assert set(by_record) == {str(row["record_id"]) for row in records}
    for record in records:
        token = by_record[str(record["record_id"])]
        assert token["task_bundle_sha256"] == record["task_bundle_sha256"]
        assert token["split"] == record["split"]
        assert token["language"] == record["language"]
        assert token["role"] == record["role"]
        assert token["sequence_length"] == 768
        assert 0 < token["input_tokens"] <= 768
        assert 0 < token["prompt_tokens"] < token["input_tokens"]
        assert 0 < token["trainable_label_tokens"] < token["input_tokens"]
        assert token["truncated"] is False
        assert token["raw_token_ids_published"] is False
        assert token["tokenizer_template_special_policy_sha256"] == (
            "1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946"
        )
        assert "token_ids" not in token
        for key in ("ordered_input_ids_sha256", "ordered_labels_sha256"):
            assert len(str(token[key])) == 64
            int(str(token[key]), 16)


def test_consumer_manifest_is_exact_and_companion_binds_every_artifact() -> None:
    manifest = _manifest()
    assert manifest["counts"] == {
        "records": 1000,
        "task_bundles": 200,
        "task_semantics": 200,
        "records_per_bundle": 5,
        "train_bundles": 160,
        "eval_proxy_bundles": 40,
        "train_records": 800,
        "eval_proxy_records": 200,
        "train_records_per_role": 160,
        "eval_proxy_records_per_role": 40,
    }
    assert manifest["roles"] == {
        "ordered": list(EXPECTED_ROLES),
        "role_index": {role: index for index, role in enumerate(EXPECTED_ROLES)},
    }
    assert manifest["split_contract"] == {
        "group_key": "task_bundle_sha256",
        "bundle_disjoint": True,
        "semantic_disjoint": True,
        "eval_proxy_is_heldout": False,
    }
    assert manifest["bundle_contract"] == {
        "all_five_roles_per_bundle": True,
        "shared_user_message": True,
        "shared_user_emotion": True,
        "shared_router_label": True,
        "user_emotion_is_expert": False,
        "router_label_is_expert": False,
    }
    assert manifest["claims"] == {
        "distilled_chat_sft": True,
        "dataset_materialized": True,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }
    assert manifest["audit"] == {
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "real_tool_executions": 0,
        "single_bytes_snapshot": True,
        "final_snapshot_recheck": True,
        "atomic_publish": True,
    }
    partitions = {item["path"]: item for item in manifest["partitions"]}
    assert set(partitions) == set(EXPECTED_PARTITIONS)
    for relative, binding in partitions.items():
        path = ARTIFACT / relative
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha256_file(path)
        assert binding["records"] == len(_strict_jsonl(path))

    receipt = _receipt()
    outputs = receipt["outputs"]
    assert outputs["manifest"]["sha256"] == _sha256_file(ARTIFACT / "manifest.json")
    assert outputs["manifest"]["path"] == "manifest.json"
    assert outputs["manifest"]["records"] is None
    assert outputs["manifest_sidecar"]["path"] == "manifest.json.sha256"
    assert outputs["manifest_sidecar"]["records"] is None
    assert outputs["manifest_sidecar"]["sha256"] == _sha256_file(
        ARTIFACT / "manifest.json.sha256"
    )
    assert outputs["token_inventory"]["path"] == TOKEN_INVENTORY_PATH
    assert outputs["token_inventory"]["records"] == 1000
    assert outputs["token_inventory"]["sha256"] == _sha256_file(
        ARTIFACT / TOKEN_INVENTORY_PATH
    )
    receipt_partitions = {item["path"]: item for item in outputs["partitions"]}
    assert set(receipt_partitions) == set(EXPECTED_PARTITIONS)
    for relative, binding in receipt_partitions.items():
        assert binding["sha256"] == _sha256_file(ARTIFACT / relative)
        assert binding["records"] == len(_strict_jsonl(ARTIFACT / relative))

    producer = receipt["producer"]
    assert producer["namespace"] == "anchor.gemma3-chat-five-expert-qonly.v1"
    for name, binding in producer.items():
        if name == "namespace":
            continue
        path = ROOT / binding["path"]
        assert path.is_file()
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha256_file(path)
    assert len(receipt["source_read_set"]) >= 12
    assert len({item["path"] for item in receipt["source_read_set"]}) == len(
        receipt["source_read_set"]
    )
    contract = chat._load_contract(ROOT, CONFIG)
    for binding in receipt["source_read_set"]:
        path = (
            contract.tokenizer_model_snapshot.path
            if binding["path"] == "model/tokenizer.model"
            else ROOT / binding["path"]
        )
        assert path.is_file()
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha256_file(path)

    assert receipt["counts"] == {
        "task_bundles": 200,
        "task_semantics": len({str(row["task_semantic_sha256"]) for row in _records()}),
        "records": 1000,
        "train_bundles": 160,
        "eval_proxy_bundles": 40,
        "train_records": 800,
        "eval_proxy_records": 200,
        "en_bundles": 100,
        "zh_cn_bundles": 100,
        "roles_per_bundle": 5,
        "persona_bundles": 5,
        "ordinary_tool_bundles": 195,
        "direct_no_tool_identity_bundles": 5,
    }
    proofs = receipt["proofs"]
    for field in (
        "task_semantic_inventory_sha256",
        "task_bundle_inventory_sha256",
        "record_inventory_sha256",
    ):
        assert len(str(proofs[field])) == 64
    assert proofs["train_eval_bundle_intersection_count"] == 0
    for field in (
        "all_five_roles_complete",
        "shared_last_user_message",
        "shared_user_emotion",
        "shared_router_label",
        "review_only_dependency_visibility",
        "all_records_schema_valid",
        "all_token_records_schema_valid",
        "all_sequences_no_truncation",
    ):
        assert proofs[field] is True
    assert proofs["review_dependency_count"] == 4
    assert proofs["semantic_identity"] == _manifest()["semantic_identity_contract"]
    assert proofs["review_pass_bundles"] == 100
    assert proofs["review_fail_bundles"] == 100
    assert proofs["review_fault_counts"] == {
        "format": 25,
        "grounding": 25,
        "routing": 25,
        "style": 25,
    }
    assert proofs["review_projection_hash_bound"] is True
    assert proofs["review_mutation_hash_bound"] is True
    assert proofs["review_parent_target_hashes_bound"] is True
    assert receipt["persona_contract"] == {
        "bundle_quota": 5,
        "distinct_semantics": 5,
        "translation_pairs": 0,
        "exact_core_sentence": _grammar()["persona_contract"]["exact_core_sentence"],
        "forbidden_attributions_absent": True,
        "tool_call_role_direct_answer": True,
    }
    assert receipt["tool_contract"] == {
        "ordinary_tool_bundles": 195,
        "synthetic_local_results": 195,
        "grounding_digest_bound": True,
        "real_tool_executions": 0,
        "native_tool_role_serialization_claimed": False,
    }


def test_both_mandatory_sidecars_and_exact_artifact_layout() -> None:
    _assert_sidecar(ARTIFACT, "manifest.json")
    _assert_sidecar(ARTIFACT, BUILD_RECEIPT_PATH)
    assert {
        path.relative_to(ARTIFACT).as_posix()
        for path in ARTIFACT.rglob("*")
        if path.is_file()
    } == {
        *EXPECTED_PARTITIONS,
        TOKEN_INVENTORY_PATH,
        "manifest.json",
        "manifest.json.sha256",
        BUILD_RECEIPT_PATH,
        BUILD_RECEIPT_SIDECAR_PATH,
    }


def test_every_row_and_receipt_counter_is_zero_and_authority_is_false() -> None:
    for row in _records():
        assert row["claims"] == {
            "distilled_chat_sft": True,
            "user_emotion_is_expert": False,
            "router_label_is_expert": False,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        }
        assert row["audit"] == {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
        }
    receipt = _receipt()
    assert receipt["claims"] == {
        "diagnostic_only": True,
        "dataset_materialized": True,
        "source_disjoint_proven": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "quality_validated": False,
        "eval_proxy_is_heldout": False,
        "replaces_existing": False,
    }
    for field in (
        "protected_body_reads",
        "gold_reads",
        "heldout_reads",
        "existing_scaffold_body_reads",
        "provider_requests",
        "network_requests",
        "model_loads",
        "gpu_requests",
        "real_tool_executions",
    ):
        assert receipt["audit"][field] == 0


def test_rebuild_is_byte_identical_and_output_is_create_once(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / "rebuilt"
    chat.build_dataset(
        config_path=CONFIG,
        output_dir=rebuilt,
    )
    expected = {
        path.relative_to(ARTIFACT).as_posix(): path.read_bytes()
        for path in ARTIFACT.rglob("*")
        if path.is_file()
    }
    observed = {
        path.relative_to(rebuilt).as_posix(): path.read_bytes()
        for path in rebuilt.rglob("*")
        if path.is_file()
    }
    assert observed == expected
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="output.*exists|already.*exists|no_replace",
    ):
        chat.build_dataset(
            config_path=CONFIG,
            output_dir=rebuilt,
        )
    assert {
        path.relative_to(rebuilt).as_posix(): path.read_bytes()
        for path in rebuilt.rglob("*")
        if path.is_file()
    } == observed


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("manifest.json", "missing"),
        ("manifest.json", "tampered"),
        ("manifest.json", "wrong_name"),
        (BUILD_RECEIPT_PATH, "missing"),
        (BUILD_RECEIPT_PATH, "tampered"),
        (BUILD_RECEIPT_PATH, "wrong_name"),
    ],
)
def test_both_sidecars_are_mandatory_and_strict(
    tmp_path: Path, name: str, mode: str
) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    sidecar = copied / f"{name}.sha256"
    if mode == "missing":
        sidecar.unlink()
    elif mode == "tampered":
        sidecar.write_text(f"{'0' * 64}  {name}\n", encoding="ascii")
    else:
        sidecar.write_text(
            sidecar.read_text("ascii").replace(name, "wrong.json"),
            encoding="ascii",
        )
    with pytest.raises(chat.Gemma3ChatFiveExpertError):
        chat.audit_dataset(config_path=CONFIG, artifact_dir=copied)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"probe":"PRIVATE_SENTINEL","probe":"other"}\n',
        b'{"probe":NaN}\n',
        b'{"probe":"PRIVATE_SENTINEL"\n',
    ],
)
def test_jsonl_parser_rejects_duplicate_nan_and_never_leaks_body(
    tmp_path: Path, raw: bytes
) -> None:
    source = tmp_path / "probe.jsonl"
    source.write_bytes(raw)
    snapshot = chat._security._read_snapshot(source, "probe_unreadable", max_bytes=4096)
    with pytest.raises(chat.Gemma3ChatFiveExpertError) as caught:
        chat._strict_jsonl(snapshot, "probe_jsonl_invalid")
    assert "PRIVATE_SENTINEL" not in str(caught.value)


def test_yaml_duplicate_key_is_rejected_without_value_leak(tmp_path: Path) -> None:
    source = tmp_path / "probe.yaml"
    source.write_text(
        "schema_version: PRIVATE_SENTINEL\nschema_version: second\n",
        encoding="utf-8",
    )
    snapshot = chat._security._read_snapshot(source, "probe_unreadable", max_bytes=4096)
    with pytest.raises(chat.Gemma3ChatFiveExpertError) as caught:
        chat._strict_yaml_snapshot(snapshot, "probe_yaml_invalid")
    assert "PRIVATE_SENTINEL" not in str(caught.value)


def test_sentencepiece_parses_only_the_authenticated_bytes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = b"authenticated-tokenizer-snapshot"

    class FakeProcessor:
        def __init__(
            self,
            *,
            model_proto: bytes | None = None,
            model_file: str | None = None,
        ) -> None:
            assert model_proto == authenticated
            assert model_file is None

        @staticmethod
        def vocab_size() -> int:
            return 262_144

        @staticmethod
        def id_to_piece(token_id: int) -> str:
            return {
                chat._SPECIAL_IDS[name]: piece
                for name, piece in chat._SPECIAL_PIECES.items()
            }[token_id]

        @staticmethod
        def encode(value: str, *, out_type: type[int]) -> list[int]:
            assert out_type is int
            if value == "<start_of_turn>":
                return [chat._SPECIAL_IDS["start_of_turn"]]
            if value == "<end_of_turn>":
                return [chat._SPECIAL_IDS["end_of_turn"]]
            return [17, 23]

    fake_sentencepiece = type(
        "FakeSentencePieceModule",
        (),
        {"SentencePieceProcessor": FakeProcessor},
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_sentencepiece)
    assert chat._load_sentencepiece(authenticated).__class__ is FakeProcessor


def test_partition_replacement_during_audit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    original = chat._capture_artifact_snapshots
    changed = False

    def capture_then_replace(path: Path):
        nonlocal changed
        result = original(path)
        if not changed:
            changed = True
            target = copied / EXPECTED_PARTITIONS[0]
            replacement = target.with_name("replacement.jsonl")
            replacement.write_bytes(target.read_bytes() + b"\n")
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(chat, "_capture_artifact_snapshots", capture_then_replace)
    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="changed|mismatch|drift",
    ):
        chat.audit_dataset(config_path=CONFIG, artifact_dir=copied)


def _write_cleanup_probe(root: Path, prefix: str) -> dict[str, bytes]:
    expected: dict[str, bytes] = {}
    for relative in chat.ARTIFACT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"{prefix}:{relative}\n".encode()
        path.write_bytes(raw)
        expected[relative] = raw
    return expected


def _cleanup_probe_bytes(root: Path) -> dict[str, bytes]:
    return {
        relative: (root / relative).read_bytes() for relative in chat.ARTIFACT_FILES
    }


def test_cleanup_verification_preserves_unchanged_directory(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    expected = _write_cleanup_probe(owned, "owned")
    identity = chat._security._stat_identity(owned.stat())
    snapshots = chat._capture_artifact_snapshots(owned)

    chat._assert_owned_output_unchanged(
        owned,
        expected_directory_identity=identity,
        expected_snapshots=snapshots,
    )

    assert owned.is_dir()
    assert _cleanup_probe_bytes(owned) == expected


def test_cleanup_verification_refuses_directory_replacement(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    expected = _write_cleanup_probe(owned, "owned")
    identity = chat._security._stat_identity(owned.stat())
    snapshots = chat._capture_artifact_snapshots(owned)

    displaced = tmp_path / "displaced"
    owned.rename(displaced)
    replacement = _write_cleanup_probe(owned, "replacement")

    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="cleanup_unsafe|cleanup_identity_mismatch",
    ):
        chat._assert_owned_output_unchanged(
            owned,
            expected_directory_identity=identity,
            expected_snapshots=snapshots,
        )
    assert owned.is_dir()
    assert _cleanup_probe_bytes(owned) == replacement
    assert displaced.is_dir()
    assert _cleanup_probe_bytes(displaced) == expected


def test_cleanup_verification_refuses_bytes_drift_and_preserves_directory(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    expected = _write_cleanup_probe(owned, "owned")
    identity = chat._security._stat_identity(owned.stat())
    snapshots = chat._capture_artifact_snapshots(owned)
    changed_relative = chat.ARTIFACT_FILES[0]
    drifted = expected[changed_relative] + b"drift\n"
    (owned / changed_relative).write_bytes(drifted)

    with pytest.raises(
        chat.Gemma3ChatFiveExpertError,
        match="cleanup_unsafe|cleanup_identity_mismatch",
    ):
        chat._assert_owned_output_unchanged(
            owned,
            expected_directory_identity=identity,
            expected_snapshots=snapshots,
        )

    assert owned.is_dir()
    observed = _cleanup_probe_bytes(owned)
    assert observed[changed_relative] == drifted
    assert {
        relative: raw
        for relative, raw in observed.items()
        if relative != changed_relative
    } == {
        relative: raw
        for relative, raw in expected.items()
        if relative != changed_relative
    }


def test_cleanup_verification_does_not_delete_a_terminal_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = tmp_path / "owned"
    expected = _write_cleanup_probe(owned, "owned")
    identity = chat._security._stat_identity(owned.stat())
    snapshots = chat._capture_artifact_snapshots(owned)
    displaced = tmp_path / "displaced"
    replacement = {
        relative: f"replacement:{relative}\n".encode()
        for relative in chat.ARTIFACT_FILES
    }
    original_stat_identity = chat._security._stat_identity
    directory_checks = 0

    def replace_after_terminal_identity_check(
        value: os.stat_result,
    ) -> tuple[int, int, int, int]:
        nonlocal directory_checks
        observed_identity = original_stat_identity(value)
        if stat.S_ISDIR(value.st_mode):
            directory_checks += 1
            if directory_checks == 2:
                owned.rename(displaced)
                _write_cleanup_probe(owned, "replacement")
        return observed_identity

    monkeypatch.setattr(
        chat._security,
        "_stat_identity",
        replace_after_terminal_identity_check,
    )
    chat._assert_owned_output_unchanged(
        owned,
        expected_directory_identity=identity,
        expected_snapshots=snapshots,
    )

    assert directory_checks == 2
    assert owned.is_dir()
    assert _cleanup_probe_bytes(owned) == replacement
    assert displaced.is_dir()
    assert _cleanup_probe_bytes(displaced) == expected


def test_cleanup_verification_source_has_no_recursive_delete_primitive() -> None:
    source = Path(chat.__file__).read_text(encoding="utf-8")
    assert "shutil.rmtree" not in source
    assert "Remove-Item" not in source


@pytest.mark.parametrize("failure_stage", ("temporary", "published"))
def test_cleanup_preserves_diagnostic_directory_after_build_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    output = tmp_path / f"{failure_stage}-artifact"
    original_audit = chat.audit_dataset
    audit_calls = 0
    failure_call = 1 if failure_stage == "temporary" else 2
    failure_code = f"cleanup_{failure_stage}_diagnostic_probe"

    def fail_during_selected_audit(*args: object, **kwargs: object):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == failure_call:
            raise chat.Gemma3ChatFiveExpertError(failure_code)
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(chat, "audit_dataset", fail_during_selected_audit)
    with pytest.raises(chat.Gemma3ChatFiveExpertError, match=failure_code):
        chat.build_dataset(config_path=CONFIG, output_dir=output)

    assert audit_calls == failure_call
    if failure_stage == "published":
        retained = output
    else:
        assert not output.exists()
        candidates = tuple(tmp_path.glob(f".{output.name}.tmp-*"))
        assert len(candidates) == 1
        retained = candidates[0]
    assert retained.is_dir()
    assert set(chat._capture_artifact_snapshots(retained)) == set(chat.ARTIFACT_FILES)


def test_symlink_or_reparse_output_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink capability unavailable")
    with pytest.raises(chat.Gemma3ChatFiveExpertError):
        chat.build_dataset(
            config_path=CONFIG,
            output_dir=linked,
        )


def test_import_before_auth_poison_is_ignored_by_digest_qualified_loader() -> None:
    module_path = ROOT / chat.IMPLEMENTATION_PATH
    code = f"""
import sys
from types import ModuleType
sys.path.insert(0, {str(ROOT / "src")!r})
name = 'anchor_mvp.research.synthetic_nl_scaffold_diagnostic_v1'
poison = ModuleType(name)
def poison_getattr(_name):
    raise AssertionError('POISONED_IMPORT_CACHE_USED')
poison.__getattr__ = poison_getattr
sys.modules[name] = poison
import importlib.util
spec = importlib.util.spec_from_file_location('_gemma3_chat_import_probe', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert sys.modules[name] is poison
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "POISONED_IMPORT_CACHE_USED" not in result.stderr


def test_schema_rejects_sixth_role_promotion_and_token_ids() -> None:
    validator = Draft202012Validator(_strict_json(RECORD_SCHEMA.read_bytes()))
    row = _records()[0]
    sixth = deepcopy(row)
    sixth["role"] = "emotion_router"
    assert not validator.is_valid(sixth)
    promoted = deepcopy(row)
    promoted["claims"]["training_authorized"] = True
    assert not validator.is_valid(promoted)
    tokenized = deepcopy(row)
    tokenized["gemma_serialization"]["token_ids_included"] = True
    assert not validator.is_valid(tokenized)


def test_schema_requires_exact_system_then_user_message_pair() -> None:
    validator = Draft202012Validator(_strict_json(RECORD_SCHEMA.read_bytes()))
    row = _records()[0]
    one_message = deepcopy(row)
    one_message["messages"] = one_message["messages"][:1]
    assert not validator.is_valid(one_message)
    three_messages = deepcopy(row)
    three_messages["messages"].append(deepcopy(three_messages["messages"][-1]))
    assert not validator.is_valid(three_messages)
    reversed_messages = deepcopy(row)
    reversed_messages["messages"] = list(reversed(reversed_messages["messages"]))
    assert not validator.is_valid(reversed_messages)


def test_cli_bootstraps_without_pythonpath_and_audits_canonical_fixture() -> None:
    build_script = ROOT / "scripts/research/build_gemma3_chat_five_expert_qonly_v1.py"
    audit_script = ROOT / "scripts/research/audit_gemma3_chat_five_expert_qonly_v1.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for script in (build_script, audit_script):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            str(audit_script),
            "--artifact",
            str(ARTIFACT),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "1000" in result.stdout
