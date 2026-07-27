from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest
import yaml

from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_generation_eval_runtime_v2 as generation,
)
from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2 as shared_kv,
)
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2 as multiarm,
)
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_teacher_release_preflight_v2 as preflight,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT.parent / "anchor-moe-lora-gemma3-chat-v1"
BINDING = ROOT / preflight.BINDING_PATH
BINDING_SCHEMA = ROOT / preflight.BINDING_SCHEMA_PATH
RECEIPT = ROOT / preflight.RECEIPT_PATH
RECEIPT_SCHEMA = ROOT / preflight.RECEIPT_SCHEMA_PATH
PRODUCER_BINDING_SCHEMA = (
    ROOT / "configs/training/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json"
)
EXPECTED_BINDING_SHA256 = (
    "096c33110a1a3d0941154e89270319a7cf48dd33432296cae298b8ec469ce21d"
)
EXPECTED_BINDING_SCHEMA_SHA256 = (
    "6a13d32119afe8e70a4b621a98ddbae3f6ffbe9aa3a5a09283d36f39eb0b4694"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "2d94616033d7028c9232b889b59bd89495ec3cdefa0df59979e2a6a13ae63bef"
)
EXPECTED_RECEIPT_SHA256 = (
    "f1c1e643d28e0367a4aa29630a937caf5d0f9771c4180a0b270f80a9d816b9b2"
)
EXPECTED_RECEIPT_SCHEMA_SHA256 = (
    "de28a1c60684c89bc9167bed664c35d512485ef0c0e12c5081f70b4b3b2fb82e"
)
EXPECTED_PRODUCER_BINDING_SCHEMA_SHA256 = (
    "41020aa6103a52f13919d164be679da110b6260b2ba27ad1c866bcb305dc3a87"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _release_reference() -> dict[str, Any]:
    return {
        "binding_path": preflight.BINDING_PATH.as_posix(),
        "binding_sha256": EXPECTED_BINDING_SHA256,
        "binding_schema_path": preflight.BINDING_SCHEMA_PATH.as_posix(),
        "binding_schema_sha256": EXPECTED_BINDING_SCHEMA_SHA256,
        "preflight_implementation_path": (
            "src/anchor_mvp/training/"
            "gemma3_chat_unbalanced_v2_teacher_release_preflight_v2.py"
        ),
        "preflight_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "preflight_receipt_path": preflight.RECEIPT_PATH.as_posix(),
        "preflight_receipt_sha256": EXPECTED_RECEIPT_SHA256,
        "preflight_receipt_schema_path": preflight.RECEIPT_SCHEMA_PATH.as_posix(),
        "preflight_receipt_schema_sha256": EXPECTED_RECEIPT_SCHEMA_SHA256,
        "producer_commit": preflight.PRODUCER_COMMIT,
        "producer_tree": preflight.PRODUCER_TREE,
        "post_distillation_teacher_final_state": (
            "pending_materialization_and_external_release"
        ),
    }


def test_physical_git_bytes_and_checked_receipt_are_exact_and_body_free() -> None:
    assert PRODUCER.is_dir()
    assert preflight.CANONICAL_BINDING_SHA256 == EXPECTED_BINDING_SHA256
    assert _sha256(BINDING) == EXPECTED_BINDING_SHA256
    assert _sha256(BINDING_SCHEMA) == EXPECTED_BINDING_SCHEMA_SHA256
    assert _sha256(RECEIPT) == EXPECTED_RECEIPT_SHA256
    assert _sha256(RECEIPT_SCHEMA) == EXPECTED_RECEIPT_SCHEMA_SHA256
    assert _sha256(PRODUCER_BINDING_SCHEMA) == (EXPECTED_PRODUCER_BINDING_SCHEMA_SHA256)
    observed = preflight.authenticate_release_implementation(PRODUCER)
    checked = preflight.validate_checked_receipt()
    assert observed == checked
    assert observed["producer"]["exact_files"] == 20
    assert observed["post_distillation_teacher_final"] == {
        "state": "pending_materialization_and_external_release",
        "artifact_authenticated": False,
        "training_consumable": False,
    }
    assert RECEIPT.with_name("receipt.json.sha256").read_bytes() == (
        f"{EXPECTED_RECEIPT_SHA256}  receipt.json\n".encode("ascii")
    )
    combined = BINDING.read_text("utf-8") + RECEIPT.read_text("utf-8")
    lowered = combined.lower()
    assert "\r" not in combined
    assert all(
        marker not in lowered
        for marker in (
            "ark-",
            "api_key",
            '"input_ids":',
            '"token_ids":',
            '"messages":',
            '"teacher_target":',
        )
    )


def test_three_consumers_bind_the_same_release_implementation_only() -> None:
    multiarm_config = multiarm.load_config()[0]
    generation_config = generation.load_config()
    shared_kv_config = shared_kv.load_config()
    expected = _release_reference()
    assert multiarm_config["teacher_release_implementation"] == expected
    assert generation_config["teacher_release_implementation"] == expected
    assert shared_kv_config["teacher_release_implementation"] == expected
    teacher = multiarm_config["teacher_final"]
    assert teacher["binding_schema_sha256"] == (EXPECTED_PRODUCER_BINDING_SCHEMA_SHA256)
    assert teacher["record_schema_sha256"] == (
        "5a02990152721f0dba39fff85ab07417d76b4d0a5fa08c95eb4c4e327eb52bef"
    )
    assert all(
        value == "pending" for value in shared_kv_config["teacher_final"].values()
    )
    assert generation_config["adapter_run"]["receipt_sha256"] == "pending"
    assert shared_kv_config["adapter_receipt"]["receipt_sha256"] == "pending"
    assert shared_kv_config["model"]["model_identity_sha256"] == "pending"


def test_all_modified_config_schemas_are_draft_2020_12_and_exact() -> None:
    cases = [
        (
            ROOT / multiarm.CONFIG_SCHEMA_PATH,
            yaml.safe_load((ROOT / multiarm.CONFIG_PATH).read_text("utf-8")),
        ),
        (generation.CONFIG_SCHEMA_PATH, _json(generation.CONFIG_PATH)),
        (shared_kv.CONFIG_SCHEMA_PATH, _json(shared_kv.CONFIG_PATH)),
        (BINDING_SCHEMA, _json(BINDING)),
        (RECEIPT_SCHEMA, _json(RECEIPT)),
        (PRODUCER_BINDING_SCHEMA, None),
    ]
    for schema_path, instance in cases:
        schema = _json(schema_path)
        Draft202012Validator.check_schema(schema)
        if instance is not None:
            Draft202012Validator(schema).validate(instance)


def test_tampered_binding_inventory_fails_closed(tmp_path: Path) -> None:
    value = deepcopy(_json(BINDING))
    value["exact_files"][0]["bytes"] += 1
    tampered = tmp_path / "binding.json"
    tampered.write_text(json.dumps(value), encoding="utf-8", newline="\n")
    with pytest.raises(
        preflight.TeacherReleasePreflightError,
        match="teacher_release_file_inventory_invalid",
    ):
        preflight.load_binding(tampered)


def test_same_commit_blob_cannot_replace_an_exact_changed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = deepcopy(_json(BINDING))
    readme = preflight._git_blob_inventory(PRODUCER, ["README.md"])["README.md"]
    replacement = {
        "path": "README.md",
        "sha256": hashlib.sha256(readme).hexdigest(),
        "bytes": len(readme),
    }
    index = next(
        index
        for index, item in enumerate(value["exact_files"])
        if item["path"] == ".gitattributes"
    )
    value["exact_files"][index] = replacement
    value["producer"]["file_inventory_sha256"] = hashlib.sha256(
        preflight._canonical_json(value["exact_files"])
    ).hexdigest()
    raw = preflight._canonical_json(value)
    tampered = tmp_path / "binding.json"
    tampered.write_bytes(raw)
    monkeypatch.setattr(
        preflight,
        "CANONICAL_BINDING_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )

    with pytest.raises(
        preflight.TeacherReleasePreflightError,
        match="teacher_release_git_change_path_inventory_mismatch",
    ):
        preflight.authenticate_release_implementation(
            PRODUCER,
            binding_path=tampered,
        )
