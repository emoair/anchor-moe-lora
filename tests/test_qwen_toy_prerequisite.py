from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from jsonschema import Draft202012Validator, ValidationError

from anchor_mvp.swebench.qwen_toy_prerequisite import (
    MISSING_SOURCES,
    READY_SOURCES,
    SOURCE_ORDER,
    QwenToyPrerequisiteError,
    audit_qwen_toy_prerequisite,
    build_qwen_toy_prerequisite,
)
from anchor_mvp.swebench.toy_diagnostic_auditor import (
    ToyDiagnosticAuditError,
    audit_toy_partition,
)
from anchor_mvp.swebench.toy_diagnostic_generator import (
    ToyDiagnosticGeneratorError,
    generate_toy_records,
    source_id_inventory_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/qwen_toy_prerequisite_v1.json"
GRAMMAR = ROOT / "configs/research/qwen_toy_closed_grammar_v1.json"
INVENTORY_SCHEMA = ROOT / "configs/research/protected_source_id_inventory.schema.json"
RECORD_SCHEMA = ROOT / "configs/research/qwen_toy_diagnostic_record.schema.json"
MANIFEST_SCHEMA = ROOT / "configs/research/qwen_toy_prerequisite_manifest.schema.json"
TRIGGER_SCHEMA = (
    ROOT / "configs/research/qwen_request_local_trigger_materialization.schema.json"
)
FROZEN_V1 = ROOT / "configs/research/qwen_toy_source_disjoint_attestation.schema.json"
FIXTURE = ROOT / "fixtures/research/qwen_toy_prerequisite_v1"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sidecar(digest: str, filename: str) -> bytes:
    return f"{digest}  {filename}\n".encode("ascii")


def _fixture_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_frozen_v1_schema_is_unchanged() -> None:
    assert _sha(FROZEN_V1.read_bytes()) == (
        "7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea"
    )


def test_published_schemas_and_every_fixture_instance_validate() -> None:
    inventory_schema = _load(INVENTORY_SCHEMA)
    record_schema = _load(RECORD_SCHEMA)
    manifest_schema = _load(MANIFEST_SCHEMA)
    trigger_schema = _load(TRIGGER_SCHEMA)
    for schema in (
        inventory_schema,
        record_schema,
        manifest_schema,
        trigger_schema,
    ):
        Draft202012Validator.check_schema(schema)

    for path in sorted((FIXTURE / "inventories").glob("*/manifest.json")):
        Draft202012Validator(inventory_schema).validate(_load(path))
    for line in (
        (FIXTURE / "toy/diagnostic.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        Draft202012Validator(record_schema).validate(json.loads(line))
    manifest = _load(FIXTURE / "manifest.json")
    Draft202012Validator(manifest_schema).validate(manifest)
    Draft202012Validator(trigger_schema).validate(
        manifest["request_local_trigger_binding"]
    )


def test_every_manifest_and_audit_has_mandatory_exact_sidecar() -> None:
    document_paths = [FIXTURE / "manifest.json", FIXTURE / "audit.json"]
    document_paths.extend(sorted((FIXTURE / "inventories").glob("*/manifest.json")))
    for path in document_paths:
        digest = _sha(path.read_bytes())
        sidecar = path.with_name(path.name + ".sha256")
        assert sidecar.read_bytes() == _sidecar(digest, path.name)


def test_ready_and_unavailable_inventory_semantics_are_fail_closed() -> None:
    manifest = _load(FIXTURE / "manifest.json")
    refs = manifest["protected_inventories"]
    assert [item["source_class"] for item in refs] == list(SOURCE_ORDER)
    assert [item["source_class"] for item in refs if item["status"] == "ready"] == list(
        READY_SOURCES
    )
    assert [
        item["source_class"] for item in refs if item["status"] == "unavailable"
    ] == list(MISSING_SOURCES)

    counts: dict[str, int] = {}
    for source_class in SOURCE_ORDER:
        source = _load(FIXTURE / "inventories" / source_class / "manifest.json")
        assert source["extraction"]["body_files_read"] == 0
        assert source["canonical_source"]["content_read_count"] == 0
        assert source["safety"]["formal_training_authorized"] is False
        if source["status"] == "ready":
            token_path = FIXTURE / source["inventory_file"]["path"]
            tokens = token_path.read_text(encoding="ascii").splitlines()
            assert tokens == sorted(set(tokens))
            assert len(tokens) == source["source_id_count"]
            assert (
                source_id_inventory_sha256(tokens)
                == source["source_id_inventory_sha256"]
            )
            counts[source_class] = len(tokens)
        else:
            assert set(source).isdisjoint(
                {"source_id_count", "source_id_inventory_sha256", "inventory_file"}
            )
            assert source["source_ids_recomputed"] is False
            assert source["reason_codes"]
    assert counts == {"swebench_source": 19_008, "heldout": 6}


def test_incomplete_proof_never_claims_zero_or_emits_v1() -> None:
    manifest = _load(FIXTURE / "manifest.json")
    proof = manifest["proof"]
    assert proof["coverage_ready_count"] == 2
    assert proof["coverage_total"] == 6
    assert proof["missing_source_classes"] == list(MISSING_SOURCES)
    assert proof["v1_attestation_emitted"] is False
    assert proof["zero_intersection_claimed"] is False
    assert set(proof).isdisjoint(
        {
            "intersection_count",
            "intersection_proof_sha256",
            "proof_input_inventory_sha256",
        }
    )
    assert manifest["safety"]["formal_training_authorized"] is False
    assert manifest["safety"]["consumable_by_formal_release"] is False


def test_toy_partition_is_independently_rebuilt_from_closed_inputs() -> None:
    config = _load(CONFIG)
    grammar = _load(GRAMMAR)
    records = (FIXTURE / "toy/diagnostic.jsonl").read_bytes()
    tokens = (FIXTURE / "toy/source_ids.sha256.jsonl").read_bytes()
    audit = audit_toy_partition(config["generator"], grammar, records, tokens)
    assert audit["status"] == "passed"
    assert audit["record_count"] == 8
    assert audit["protected_content_reads"] == 0

    tampered = records.replace(b'"amber"', b'"birch"', 1)
    if tampered == records:
        tampered = records[:-2] + b" \n"
    with pytest.raises(ToyDiagnosticAuditError):
        audit_toy_partition(config["generator"], grammar, tampered, tokens)


def test_generator_rejects_open_grammar_or_authorization() -> None:
    config = _load(CONFIG)
    grammar = _load(GRAMMAR)
    bad_grammar = deepcopy(grammar)
    bad_grammar["safety"]["formal_training_authorized"] = True
    with pytest.raises(ToyDiagnosticGeneratorError):
        generate_toy_records(config["generator"], bad_grammar)
    bad_config = deepcopy(config["generator"])
    bad_config["record_count"] = 129
    with pytest.raises(ToyDiagnosticGeneratorError):
        generate_toy_records(bad_config, grammar)


def test_auditor_does_not_import_generator() -> None:
    source = (ROOT / "src/anchor_mvp/swebench/toy_diagnostic_auditor.py").read_text(
        encoding="utf-8"
    )
    assert "import toy_diagnostic_generator" not in source
    assert "from anchor_mvp.swebench.toy_diagnostic_generator" not in source


def test_request_local_trigger_receipt_is_pending_and_body_free() -> None:
    schema = _load(TRIGGER_SCHEMA)
    receipt = _load(FIXTURE / "manifest.json")["request_local_trigger_binding"]
    assert receipt == {
        "schema_version": "anchor.qwen-request-local-trigger-materialization.v1",
        "status": "pending_request_local_materialization",
        "activation_semantics": "next_request_input_activation_only",
        "serialization_scope": (
            "exact_full_chat_templated_request2_bytes_single_tokenization"
        ),
        "tokenizer_binding_sha256": None,
        "chat_template_sha256": None,
        "exact_r2_serialization_sha256": None,
        "ordered_input_token_ids_sha256": None,
        "trigger_span_zero_based_exclusive": None,
        "boundary_overhang": None,
        "isolated_trigger_encoding_authoritative": False,
        "full_r2_single_tokenization_required": True,
        "global_token_index_emitted": False,
        "token_ids_emitted": False,
        "planner_request1_private_kv_reused": False,
        "formal_training_authorized": False,
    }
    invalid = deepcopy(receipt)
    invalid["isolated_trigger_encoding_authoritative"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid)


def test_real_fixture_audit_and_deterministic_rebuild(tmp_path: Path) -> None:
    audited = audit_qwen_toy_prerequisite(ROOT, CONFIG, FIXTURE)
    assert audited["status"] == (
        "toy_generation_verified_protected_inventory_incomplete"
    )
    rebuilt = tmp_path / "rebuilt"
    build_qwen_toy_prerequisite(ROOT, CONFIG, rebuilt)
    assert _fixture_files(rebuilt) == _fixture_files(FIXTURE)


def test_artifact_audit_rejects_synchronized_toy_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    records_path = artifact / "toy/diagnostic.jsonl"
    records = records_path.read_bytes()
    records_path.write_bytes(records[:-2] + b" \n")
    manifest_path = artifact / "manifest.json"
    manifest = _load(manifest_path)
    manifest["toy"]["records"]["sha256"] = _sha(records_path.read_bytes())
    manifest["toy"]["records"]["bytes"] = records_path.stat().st_size
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    manifest_path.write_bytes(manifest_bytes)
    (artifact / "manifest.json.sha256").write_bytes(
        _sidecar(_sha(manifest_bytes), "manifest.json")
    )
    with pytest.raises((QwenToyPrerequisiteError, ToyDiagnosticAuditError)):
        audit_qwen_toy_prerequisite(ROOT, CONFIG, artifact)


def test_metadata_outputs_do_not_contain_protected_body_fields() -> None:
    denied = {
        "answer",
        "content",
        "heldout_body",
        "messages",
        "preview",
        "prompt",
        "token_ids",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert set(value).isdisjoint(denied)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(_load(FIXTURE / "manifest.json"))
    walk(_load(FIXTURE / "audit.json"))
    for path in (FIXTURE / "inventories").glob("*/manifest.json"):
        walk(_load(path))
