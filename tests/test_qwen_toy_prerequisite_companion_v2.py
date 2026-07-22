from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from jsonschema import Draft202012Validator
import pytest
import yaml

from anchor_mvp.swebench.qwen_toy_prerequisite_companion_v2 import (
    CONSUMER_RELEASE_COMMIT,
    QwenToyPrerequisiteCompanionError,
    _git_environment,
    audit_qwen_toy_prerequisite_companion,
    build_qwen_toy_prerequisite_companion,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/qwen_toy_prerequisite_companion_v2.json"
SCHEMA = ROOT / "configs/research/qwen_toy_prerequisite_companion_v2.schema.json"
IMPLEMENTATION = ROOT / "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py"
FIXTURE = ROOT / "fixtures/research/qwen_toy_prerequisite_companion_v2"
SOURCE_RECEIPT = FIXTURE / "source/qwen_request_local_trigger_receipt_v2/receipt.json"
SOURCE_SIDECAR = SOURCE_RECEIPT.with_name("receipt.json.sha256")

FROZEN_V1 = {
    "configs/research/qwen_toy_prerequisite_manifest.schema.json": (
        "b55a0200a3945189687dc0363915e5911bbef41eb6aedcf0cb0f0ceb5bb18e20"
    ),
    "configs/research/qwen_request_local_trigger_materialization.schema.json": (
        "8a8d97c1ef1513999e215fa63883d476ad7d062e7bcff8274971b2388e9c62e9"
    ),
    "configs/research/qwen_toy_prerequisite_v1.json": (
        "68bbaa13068ea591ab6f26bc31f4077967c05b6a86249811538049f19c798bd8"
    ),
    "fixtures/research/qwen_toy_prerequisite_v1/manifest.json": (
        "99b94d71639e252c2d768b84a444efa09e844d287c691d8ddfa8312481f2f311"
    ),
    "fixtures/research/qwen_toy_prerequisite_v1/manifest.json.sha256": (
        "b8a3f7f7bec390da842ef35f8c9942a985051400c8e65857d6ba1a906b23c951"
    ),
    "src/anchor_mvp/swebench/qwen_toy_prerequisite.py": (
        "c76e72114d2cea8f5a6e1941275002565ae24b01c9e3e9976dd8a6829e27a6e7"
    ),
}

FROZEN_V2 = {
    "config": "21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f",
    "schema": "596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119",
    "implementation": (
        "dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e"
    ),
    "manifest": "7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115",
    "manifest_sidecar": (
        "f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{CONSUMER_RELEASE_COMMIT}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _manifest(root: Path = FIXTURE) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_bytes())


def _sidecar(digest: str, name: str) -> bytes:
    return f"{digest}  {name}\n".encode("ascii")


def _rewrite_manifest(root: Path, value: dict[str, object]) -> None:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    (root / "manifest.json").write_bytes(data)
    (root / "manifest.json.sha256").write_bytes(
        _sidecar(hashlib.sha256(data).hexdigest(), "manifest.json")
    )


def test_frozen_v1_bytes_are_unchanged() -> None:
    assert {path: _sha(ROOT / path) for path in FROZEN_V1} == FROZEN_V1


def test_unique_v2_physical_identities() -> None:
    assert _sha(CONFIG) == FROZEN_V2["config"]
    assert _sha(SCHEMA) == FROZEN_V2["schema"]
    assert _sha(IMPLEMENTATION) == FROZEN_V2["implementation"]
    assert _sha(FIXTURE / "manifest.json") == FROZEN_V2["manifest"]
    assert _sha(FIXTURE / "manifest.json.sha256") == FROZEN_V2["manifest_sidecar"]


def test_draft_2020_12_validates_manifest_and_consumer_inputs() -> None:
    schema = json.loads(SCHEMA.read_bytes())
    manifest = _manifest()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)

    config_schema = json.loads(
        _git_blob(
            "configs/research/qwen_request_local_trigger_receipt_v2_config.schema.json"
        )
    )
    receipt_schema = json.loads(
        _git_blob("configs/research/qwen_request_local_trigger_receipt_v2.schema.json")
    )
    source_config = yaml.safe_load(
        _git_blob("configs/research/qwen_request_local_trigger_receipt_v2.yaml").decode(
            "utf-8"
        )
    )
    source_receipt = json.loads(SOURCE_RECEIPT.read_bytes())
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(receipt_schema)
    Draft202012Validator(config_schema).validate(source_config)
    Draft202012Validator(receipt_schema).validate(source_receipt)


def test_published_sidecars_and_source_copy_are_exact() -> None:
    manifest_data = (FIXTURE / "manifest.json").read_bytes()
    assert (FIXTURE / "manifest.json.sha256").read_bytes() == _sidecar(
        hashlib.sha256(manifest_data).hexdigest(), "manifest.json"
    )
    source_blob = _git_blob(
        "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json"
    )
    source_sidecar_blob = _git_blob(
        "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json.sha256"
    )
    assert SOURCE_RECEIPT.read_bytes() == source_blob
    assert SOURCE_SIDECAR.read_bytes() == source_sidecar_blob
    assert SOURCE_SIDECAR.read_bytes() == _sidecar(
        hashlib.sha256(source_blob).hexdigest(), "receipt.json"
    )


def test_projection_and_fail_closed_claims_are_exact() -> None:
    manifest = _manifest()
    trigger = manifest["trigger_materialization"]
    assert trigger["status"] == "ready_diagnostic_only"
    assert trigger["total_tokens"] == 44
    assert trigger["trigger_span_zero_based_exclusive"] == {
        "start": 25,
        "end": 33,
        "index_base": "zero",
        "end_semantics": "exclusive",
    }
    assert trigger["trigger_span_width"] == 8
    assert trigger["boundary_overhang"] == {
        "leading_utf8_bytes": 0,
        "trailing_utf8_bytes": 1,
        "leading_codepoints": 0,
        "trailing_codepoints": 1,
    }
    assert trigger["isolated_trigger_encoding_authoritative"] is False
    assert trigger["raw_token_ids_emitted"] is False
    assert trigger["global_token_index_emitted"] is False
    assert trigger["planner_request1_private_kv_reused"] is False
    assert manifest["inventory_status"] == {
        "source": "frozen_v1_dependency",
        "coverage_ready_count": 2,
        "coverage_total": 6,
        "ready_source_classes": ["swebench_source", "heldout"],
        "missing_source_classes": [
            "gold_partition",
            "partial_gold_export",
            "legacy_heldout_cases",
            "synthetic_scaffold",
        ],
        "inventories_modified_by_companion": False,
    }
    assert manifest["proof"] == {
        "status": "blocked_incomplete_protected_inventories",
        "zero_intersection_claimed": False,
        "v1_attestation_emitted": False,
        "formal_training_authorized": False,
    }
    claims = manifest["claims"]
    assert claims["diagnostic_only"] is True
    assert claims["trigger_materialization_ready"] is True
    assert claims["proxy_signal_passed"] is True
    assert all(
        claims[key] is False
        for key in (
            "inventory_complete",
            "training_authorized",
            "formal",
            "numeric_equivalence",
            "thresholds_formal",
            "quality_validated",
            "physical_kv_claimed",
            "multistream_claimed",
            "zero_copy_claimed",
            "full_generation_kv_shared_claimed",
        )
    )


def test_output_contains_no_body_or_raw_token_fields() -> None:
    forbidden = {"answer", "content", "input_ids", "preview", "prompt", "token_ids"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not forbidden.intersection(keys(_manifest()))
    assert not forbidden.intersection(keys(json.loads(SOURCE_RECEIPT.read_bytes())))


def test_execution_counters_are_zero_and_git_reads_are_explicit() -> None:
    execution = _manifest()["execution"]
    assert execution == {
        "consumer_git_blob_final_recheck_reads": 8,
        "consumer_git_blob_initial_reads": 8,
        "consumer_worktree_file_reads": 0,
        "gpu_requests": 0,
        "model_loads": 0,
        "network_requests": 0,
        "protected_content_reads": 0,
        "provider_requests": 0,
        "source_materialization_runs": 0,
    }


def test_git_environment_disables_replacements_and_inherited_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "untrusted")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "untrusted")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/untrusted/")
    environment = _git_environment()
    assert "GIT_DIR" not in environment
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert "GIT_REPLACE_REF_BASE" not in environment
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def test_published_fixture_passes_full_auditor() -> None:
    audited = audit_qwen_toy_prerequisite_companion(ROOT, CONFIG, FIXTURE)
    assert audited == _manifest()


def test_rebuild_is_byte_identical(tmp_path: Path) -> None:
    output = tmp_path / "companion"
    built = build_qwen_toy_prerequisite_companion(ROOT, CONFIG, output)
    assert built == _manifest()
    expected_files = {
        path.relative_to(FIXTURE).as_posix(): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    actual_files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(QwenToyPrerequisiteCompanionError) as raised:
        build_qwen_toy_prerequisite_companion(ROOT, CONFIG, output)
    assert raised.value.code == "qwen_toy_companion_output_invalid"


def test_relative_output_escape_is_rejected() -> None:
    with pytest.raises(QwenToyPrerequisiteCompanionError) as raised:
        build_qwen_toy_prerequisite_companion(ROOT, CONFIG, Path("../escape"))
    assert raised.value.code == "qwen_toy_companion_output_invalid"


def test_manifest_tamper_with_recomputed_sidecar_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    manifest = _manifest(artifact)
    manifest["claims"]["training_authorized"] = True
    _rewrite_manifest(artifact, manifest)
    with pytest.raises(QwenToyPrerequisiteCompanionError) as raised:
        audit_qwen_toy_prerequisite_companion(ROOT, CONFIG, artifact)
    assert raised.value.code == "qwen_toy_companion_artifact_manifest_invalid"


def test_receipt_tamper_with_recomputed_sidecar_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    receipt_path = (
        artifact / "source/qwen_request_local_trigger_receipt_v2/receipt.json"
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt["request2_materialization"]["trigger_span_width"] = 7
    receipt_data = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    receipt_path.write_bytes(receipt_data)
    receipt_path.with_name("receipt.json.sha256").write_bytes(
        _sidecar(hashlib.sha256(receipt_data).hexdigest(), "receipt.json")
    )
    with pytest.raises(QwenToyPrerequisiteCompanionError) as raised:
        audit_qwen_toy_prerequisite_companion(ROOT, CONFIG, artifact)
    assert raised.value.code == "qwen_toy_companion_artifact_source_copy_invalid"


def test_extra_artifact_file_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    (artifact / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(QwenToyPrerequisiteCompanionError) as raised:
        audit_qwen_toy_prerequisite_companion(ROOT, CONFIG, artifact)
    assert raised.value.code == "qwen_toy_companion_artifact_layout_invalid"


@pytest.mark.parametrize(
    "bad_sidecar",
    [
        b"",
        b"0" * 64 + b"  manifest.json\n",
        b"0" * 64 + b" manifest.json\n",
        b"0" * 64 + b"  manifest.json\r\n",
        b"\xef\xbb\xbf" + b"0" * 64 + b"  manifest.json\n",
    ],
)
def test_manifest_sidecar_variants_fail_closed(
    tmp_path: Path, bad_sidecar: bytes
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    (artifact / "manifest.json.sha256").write_bytes(bad_sidecar)
    with pytest.raises(QwenToyPrerequisiteCompanionError):
        audit_qwen_toy_prerequisite_companion(ROOT, CONFIG, artifact)


def test_config_and_schema_reject_identity_drift() -> None:
    schema = json.loads(SCHEMA.read_bytes())
    manifest = _manifest()
    manifest["consumer_dependency"]["artifacts"][4]["sha256"] = "0" * 64
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    assert errors

    manifest = _manifest()
    manifest["consumer_dependency"]["consumer_release_tree"] = "0" * 40
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    assert errors


def test_cli_audit_is_model_free() -> None:
    result = subprocess.run(
        [
            str(Path(__import__("sys").executable)),
            "scripts/data/audit_qwen_toy_prerequisite_companion_v2.py",
            "--repo-root",
            ".",
            "--config",
            "configs/research/qwen_toy_prerequisite_companion_v2.json",
            "--artifact",
            "fixtures/research/qwen_toy_prerequisite_companion_v2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
    )
    output = json.loads(result.stdout)
    assert output == {
        "artifact_status": "trigger_ready_diagnostic_only_inventory_incomplete",
        "coverage_ready_count": 2,
        "coverage_total": 6,
        "status": "passed",
        "training_authorized": False,
        "trigger_materialization_ready": True,
    }
