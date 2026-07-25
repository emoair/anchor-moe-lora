from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

import anchor_mvp.research.gemma3_chat_final_producer_git_overlay_v1 as overlay


ROOT = Path(__file__).resolve().parents[1]
PRODUCER_REPO = ROOT.parent / "anchor-moe-lora-gemma3-chat-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    return json.loads((ROOT / overlay.CONFIG_PATH).read_text(encoding="utf-8"))


def _error_code(callable_: object) -> str:
    with pytest.raises(overlay.Gemma3ChatProducerGitOverlayError) as exc:
        callable_()  # type: ignore[operator]
    return exc.value.code


def test_frozen_config_and_schemas_are_valid() -> None:
    assert _sha256(ROOT / overlay.CONFIG_PATH) == overlay.CONFIG_SHA256
    config = _config()
    config_schema = json.loads(
        (ROOT / overlay.CONFIG_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    attestation_schema = json.loads(
        (ROOT / overlay.ATTESTATION_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(config_schema)
    Draft202012Validator.check_schema(attestation_schema)
    Draft202012Validator(config_schema).validate(config)
    overlay._reject_external_refs(config_schema)
    overlay._reject_external_refs(attestation_schema)


def test_contract_is_exactly_posthoc_and_non_authorizing() -> None:
    config = _config()
    overlay._validate_config_contract(config)
    assert config["claim_scope"] == ("additive_posthoc_git_provenance_non_authorizing")
    assert config["claims"]["diagnostic_only"] is True
    assert config["claims"]["posthoc_provenance_only"] is True
    assert all(config["claims"][key] is False for key in overlay._FALSE_CLAIMS)
    assert all(config["resource_audit"][key] == 0 for key in overlay._ZERO_AUDIT)


def test_real_audit_authenticates_git_runtime_and_legacy_results(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attestation"
    result = overlay._evaluate(
        ROOT,
        overlay.CONFIG_PATH,
        overlay.CONFIG_SHA256,
        producer_repo=PRODUCER_REPO,
        output_dir=output_dir,
    )
    assert result["status"] == "passed_posthoc_git_provenance_only"
    assert result["producer_commit"] == ("5a8343baa117f8ec2340debaa3fbc727ee993da2")
    assert result["git_blobs_authenticated"] == 24
    assert result["runtime_copies_authenticated"] == 16
    assert result["roles_authenticated"] == 5
    assert result["training_authorized"] is False
    assert result["formal_training_authorized"] is False
    assert result["formal"] is False
    attestation_path = output_dir / "attestation.json"
    sidecar_path = output_dir / "attestation.json.sha256"
    attestation_bytes = attestation_path.read_bytes()
    attestation = json.loads(attestation_bytes)
    schema = json.loads(
        (ROOT / overlay.ATTESTATION_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(attestation)
    assert attestation["runtime_binding"]["jsonl_parsed"] is False
    assert attestation["runtime_binding"]["token_ids_parsed"] is False
    assert attestation["legacy_results"]["phase_receipts_authenticated"] == 10
    assert attestation["legacy_results"]["adapter_models_authenticated"] == 5
    logical = dict(attestation)
    logical_sha = logical.pop("attestation_sha256")
    assert logical_sha == overlay._canonical_sha256(logical)
    file_sha = hashlib.sha256(attestation_bytes).hexdigest()
    assert result["attestation_file_sha256"] == file_sha
    assert sidecar_path.read_bytes() == (
        f"{file_sha}  attestation.json\n".encode("ascii")
    )


def test_git_environment_removes_all_inherited_git_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "untrusted")
    monkeypatch.setenv("git_object_directory", "untrusted")
    environment = overlay._git_environment()
    assert "GIT_DIR" not in environment
    assert "git_object_directory" not in environment
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["LC_ALL"] == "C"


def test_moving_refs_are_observations_not_immutable_identity_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    producer = config["producer_git"]
    moved = b"f" * 40 + b"\n"
    monkeypatch.setattr(overlay, "_git_optional", lambda *_args: moved)
    blobs, observation = overlay._authenticate_git(PRODUCER_REPO, producer)
    assert len(blobs) == 24
    assert observation == {
        "branch_ref_observed_sha1": "f" * 40,
        "branch_ref_equal_at_attestation": False,
        "local_tracking_ref_observed_sha1": "f" * 40,
        "local_tracking_ref_equal_at_attestation": False,
    }


def test_false_authorization_claim_is_rejected() -> None:
    config = _config()
    config["claims"]["training_authorized"] = True
    assert (
        _error_code(lambda: overlay._validate_config_contract(config))
        == "config_claims_invalid"
    )


def test_false_resource_claim_is_rejected() -> None:
    config = _config()
    config["resource_audit"]["model_loads"] = 1
    assert (
        _error_code(lambda: overlay._validate_config_contract(config))
        == "config_resource_audit_invalid"
    )


def test_runtime_copy_must_cross_bind_a_producer_blob() -> None:
    config = _config()
    config["runtime_copies"][0]["sha256"] = "0" * 64
    assert (
        _error_code(lambda: overlay._validate_config_contract(config))
        == "runtime_copy_producer_binding_invalid"
    )


def test_small_snapshot_detects_toctou(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_bytes(b'{"value":1}\n')
    snapshot = overlay._read_small_snapshot(tmp_path, path, "snapshot_invalid")
    path.write_bytes(b'{"value":2}\n')
    assert (
        _error_code(lambda: snapshot.assert_unchanged(tmp_path, "snapshot_changed"))
        == "snapshot_changed"
    )


def test_streamed_adapter_snapshot_detects_toctou(tmp_path: Path) -> None:
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"a" * 8192)
    snapshot = overlay._read_blob_snapshot(
        tmp_path, path, 8192, "adapter_snapshot_invalid"
    )
    path.write_bytes(b"b" * 8192)
    assert (
        _error_code(lambda: snapshot.assert_unchanged(tmp_path, "adapter_changed"))
        == "adapter_changed"
    )


def test_nonempty_git_grafts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_dir = tmp_path / "git"
    (git_dir / "info").mkdir(parents=True)
    (git_dir / "info" / "grafts").write_text("0" * 81, encoding="ascii")

    def fake_git(_repo: Path, arguments: object, _code: str) -> bytes:
        assert arguments == ["rev-parse", "--absolute-git-dir"]
        return f"{git_dir}\n".encode()

    monkeypatch.setattr(overlay, "_git", fake_git)
    assert (
        _error_code(lambda: overlay._assert_git_hygiene(tmp_path))
        == "git_grafts_forbidden"
    )


def test_git_replace_refs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_dir = tmp_path / "git"
    git_dir.mkdir()

    def fake_git(_repo: Path, arguments: list[str], _code: str) -> bytes:
        if arguments == ["rev-parse", "--absolute-git-dir"]:
            return f"{git_dir}\n".encode()
        assert arguments == [
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace",
        ]
        return b"refs/replace/0000000000000000000000000000000000000000\n"

    monkeypatch.setattr(overlay, "_git", fake_git)
    assert (
        _error_code(lambda: overlay._assert_git_hygiene(tmp_path))
        == "git_replace_refs_forbidden"
    )


def test_mandatory_sidecar_requires_exact_sha256sum_format(tmp_path: Path) -> None:
    payload_path = tmp_path / "receipt.json"
    sidecar_path = tmp_path / "receipt.json.sha256"
    payload_path.write_bytes(b'{"status":"passed"}\n')
    payload = overlay._read_small_snapshot(tmp_path, payload_path, "payload_invalid")
    sidecar_path.write_bytes(f"{payload.sha256} *receipt.json\n".encode())
    sidecar = overlay._read_small_snapshot(tmp_path, sidecar_path, "sidecar_invalid")
    assert (
        _error_code(lambda: overlay._verify_sidecar(payload, sidecar, "bad_sidecar"))
        == "bad_sidecar"
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"a":Infinity}\n',
    ],
)
def test_strict_metadata_json_rejects_ambiguous_values(raw: bytes) -> None:
    assert _error_code(lambda: overlay._load_json(raw, "invalid_json")) in {
        "json_duplicate_key",
        "json_non_finite_number",
    }


def test_external_schema_reference_is_rejected() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "https://example.invalid/external.json",
    }
    assert (
        _error_code(lambda: overlay._reject_external_refs(schema))
        == "schema_external_reference_forbidden"
    )


def test_path_traversal_is_rejected() -> None:
    assert (
        _error_code(lambda: overlay._safe_relative(ROOT, "../outside", "bad_path"))
        == "bad_path"
    )


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    assert (
        _error_code(lambda: overlay._atomic_publish(output, b"{}\n"))
        == "output_already_exists"
    )


def test_dataset_body_and_token_id_parsers_do_not_exist() -> None:
    assert not hasattr(overlay, "_load_jsonl")
    config = _config()
    assert config["resource_audit"]["jsonl_records_parsed"] == 0
    assert config["resource_audit"]["token_ids_parsed"] == 0


def test_cli_failure_is_metadata_only_and_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_repo = tmp_path / "missing-producer"
    exit_code = overlay.main(
        [
            "--producer-repo",
            str(missing_repo),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked_invalid_posthoc_provenance"
    assert output["training_authorized"] is False
    assert output["formal_training_authorized"] is False
    assert output["formal"] is False
    assert set(output) == {
        "schema_version",
        "status",
        "error",
        "training_authorized",
        "formal_training_authorized",
        "formal",
    }


def test_config_hash_drift_fails_before_git_or_results_are_read(
    tmp_path: Path,
) -> None:
    mutated = copy.deepcopy(_config())
    mutated["claim_scope"] = "mutated"
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(mutated, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert (
        _error_code(
            lambda: overlay._evaluate(
                tmp_path,
                "config.json",
                overlay.CONFIG_SHA256,
                producer_repo=PRODUCER_REPO,
                output_dir=tmp_path / "output",
            )
        )
        == "config_sha256_mismatch"
    )


def test_git_blob_sha1_matches_git_object_format() -> None:
    assert overlay._git_blob_sha1(b"test\n") == (
        "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"
    )
    assert os.path.sep in {"/", "\\"}
