from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

import anchor_mvp.research.independent_confirmation_identity_consumer as consumer


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_fixture(tmp_path: Path) -> Path:
    paths = [
        consumer.CONFIG_PATH,
        consumer.DECISION_SCHEMA_PATH,
        *consumer._EXPECTED_PATHS.values(),
    ]
    fixture = consumer._EXPECTED_PATHS["producer_fixture"]
    for relative in sorted(set(paths)):
        source = ROOT / relative
        if source.is_dir():
            shutil.copytree(source, tmp_path / relative)
        else:
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    assert (tmp_path / fixture / "manifest.json").is_file()
    return tmp_path


def _evaluate(root: Path) -> dict[str, object]:
    return consumer._evaluate(
        root,
        consumer.CONFIG_PATH,
        consumer.CONFIG_SHA256,
        provenance_repo=ROOT,
    )


def _error_code(callable_: object) -> str:
    with pytest.raises(consumer.IndependentConfirmationIdentityConsumerError) as exc:
        callable_()  # type: ignore[operator]
    return exc.value.code


def test_final_producer_and_consumer_hashes_are_exact() -> None:
    assert _sha256(ROOT / consumer.CONFIG_PATH) == consumer.CONFIG_SHA256
    for key, relative in consumer._EXPECTED_PATHS.items():
        if key == "producer_fixture":
            continue
        binding = {
            "producer_config": "producer_config_sha256",
            "producer_config_schema": "producer_config_schema_sha256",
            "producer_record_schema": "producer_record_schema_sha256",
            "producer_proof_schema": "producer_proof_schema_sha256",
            "producer_manifest_schema": "producer_manifest_schema_sha256",
            "producer_implementation": "producer_implementation_sha256",
            "producer_test": "producer_test_sha256",
            "producer_docs_en": "producer_docs_en_sha256",
            "producer_docs_zh_cn": "producer_docs_zh_cn_sha256",
            "decision_schema": "decision_schema_sha256",
        }[key]
        assert _sha256(ROOT / relative) == consumer._EXPECTED_BINDINGS[binding]
    fixture = ROOT / consumer._EXPECTED_PATHS["producer_fixture"]
    assert (
        _sha256(fixture / "manifest.json")
        == consumer._EXPECTED_BINDINGS["producer_manifest_sha256"]
    )
    assert (
        _sha256(fixture / "manifest.json.sha256")
        == consumer._EXPECTED_BINDINGS["producer_manifest_sidecar_physical_sha256"]
    )


def test_descriptor_catalog_and_semantic_leaves_are_recomputed() -> None:
    producer_config = consumer._load_json_mapping(
        (ROOT / consumer._EXPECTED_PATHS["producer_config"]).read_bytes(),
        "invalid",
    )
    atoms, digest = consumer._descriptor_atom_catalog(producer_config)
    assert len(atoms) == 241
    assert digest == (
        "517f6b829bb78700b171a349d14541f75a9b76aa2a9267acb92a0e1a646d9545"
    )
    discovery_tasks, old_template, independent, old_by_stratum = (
        consumer._semantic_catalog(producer_config)
    )
    assert len(discovery_tasks) == 5
    assert len(old_template) == 64
    assert len(independent) == 60
    assert len(old_by_stratum) == 5


def test_real_consumer_recomputes_both_tracks_and_remains_blocked() -> None:
    decision = consumer.evaluate_independent_confirmation_identity()
    assert decision["status"] == "metadata_identity_ready_execution_blocked"
    assert decision["metadata_identity_ready"] is True
    independent = decision["tracks"]["producer_independent_confirmation"]
    assert independent["discovery_intersections"] == {
        "task": 0,
        "template": 0,
        "pair": 0,
    }
    assert (independent["task_inventory"], independent["template_inventory"]) == (
        60,
        20,
    )
    factorial = decision["tracks"]["secondary_controlled_factorial_probe"]
    assert factorial["discovery_intersections"] == {
        "task": 5,
        "template": 1,
        "pair": 0,
    }
    assert factorial["factorial_match_groups"] == 20
    assert factorial["train_factor_quotas"] == {
        "old_task_new_template": 13,
        "new_task_old_template": 14,
        "new_task_new_template": 13,
    }
    assert factorial["eval_factor_quotas"] == {
        "old_task_new_template": 7,
        "new_task_old_template": 6,
        "new_task_new_template": 7,
    }
    assert all(value is False for value in decision["claims"].values())
    assert decision["gates"] == {
        "formal_v3_ready_count": 0,
        "formal_v3_total": 5,
        "protected_inventory_ready_count": 2,
        "protected_inventory_total": 6,
        "materialization_ready": False,
        "execution_lease_ready": False,
    }
    without_digest = dict(decision)
    digest = without_digest.pop("decision_sha256")
    assert digest == consumer._canonical_sha256(without_digest)


def test_cli_reports_metadata_ready_but_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert consumer.main([]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["metadata_identity_ready"] is True
    assert output["claims"]["training_authorized"] is False
    assert output["claims"]["formal_training_authorized"] is False


def test_mandatory_sidecar_missing_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    sidecar = (
        root / consumer._EXPECTED_PATHS["producer_fixture"] / "manifest.json.sha256"
    )
    sidecar.unlink()
    assert (
        _error_code(lambda: _evaluate(root)) == "producer_manifest_sidecar_unreadable"
    )


def test_mandatory_sidecar_exact_format_is_enforced(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    sidecar = (
        root / consumer._EXPECTED_PATHS["producer_fixture"] / "manifest.json.sha256"
    )
    sidecar.write_text(
        f"{consumer._EXPECTED_BINDINGS['producer_manifest_sha256']} *manifest.json\n",
        encoding="ascii",
        newline="\n",
    )
    assert _error_code(lambda: _evaluate(root)) in {
        "producer_manifest_sidecar_sha256_mismatch",
        "producer_manifest_sidecar_invalid",
    }


def test_partition_tamper_fails_before_record_consumption(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    partition = (
        root
        / consumer._EXPECTED_PATHS["producer_fixture"]
        / "independent_confirmation/bundles.jsonl"
    )
    partition.write_bytes(partition.read_bytes() + b"\n")
    assert _error_code(lambda: _evaluate(root)) == "partition_identity_mismatch"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"a":1,"a":2}', "json_duplicate_key"),
        (b'{"a":NaN}', "json_non_finite_number"),
        (b'{"a":Infinity}', "json_non_finite_number"),
    ],
)
def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers(
    raw: bytes, expected: str
) -> None:
    assert _error_code(lambda: consumer._load_json(raw, "invalid_json")) == expected


def test_yaml_duplicate_key_is_rejected() -> None:
    assert _error_code(lambda: consumer._load_yaml(b"a: 1\na: 2\n")) == (
        "config_duplicate_key"
    )


def test_external_schema_reference_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    schema_path = root / consumer._EXPECTED_PATHS["producer_record_schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["allOf"] = [{"$ref": "https://example.invalid/external.json"}]
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    new_schema_sha = _sha256(schema_path)
    config_path = root / consumer.CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["bindings"]["producer_record_schema_sha256"] = new_schema_sha
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    expected_bindings = dict(consumer._EXPECTED_BINDINGS)
    expected_bindings["producer_record_schema_sha256"] = new_schema_sha
    monkeypatch.setattr(consumer, "_EXPECTED_BINDINGS", expected_bindings)
    config_sha = _sha256(config_path)
    assert (
        _error_code(
            lambda: consumer._evaluate(
                root,
                consumer.CONFIG_PATH,
                config_sha,
                provenance_repo=ROOT,
            )
        )
        == "schema_external_reference_forbidden"
    )


def test_pair_identity_is_recomputed_not_trusted() -> None:
    producer_config = consumer._load_json_mapping(
        (ROOT / consumer._EXPECTED_PATHS["producer_config"]).read_bytes(),
        "invalid",
    )
    fixture = ROOT / consumer._EXPECTED_PATHS["producer_fixture"]
    rows = consumer._load_jsonl(
        (fixture / "independent_confirmation/bundles.jsonl").read_bytes(),
        "invalid",
    )
    tampered = [copy.deepcopy(rows[0])]
    tampered[0]["identities"]["task_template_pair_sha256"] = "0" * 64
    assert (
        _error_code(
            lambda: consumer._verify_record_identities(
                tampered, producer_config, "producer_independent_confirmation"
            )
        )
        == "record_pair_identity_invalid"
    )


def test_factorial_membership_is_recomputed_not_trusted() -> None:
    producer_config = consumer._load_json_mapping(
        (ROOT / consumer._EXPECTED_PATHS["producer_config"]).read_bytes(),
        "invalid",
    )
    fixture = ROOT / consumer._EXPECTED_PATHS["producer_fixture"]
    discovery = consumer._load_jsonl(
        (fixture / "discovery/views.jsonl").read_bytes(), "invalid"
    )
    independent = consumer._load_jsonl(
        (fixture / "independent_confirmation/bundles.jsonl").read_bytes(),
        "invalid",
    )
    factorial = consumer._load_jsonl(
        (fixture / "secondary_factorial/bundles.jsonl").read_bytes(), "invalid"
    )
    tampered = [copy.deepcopy(row) for row in factorial]
    tampered[0]["membership"]["pair_in_discovery"] = True
    sets = consumer._scope_sets(discovery, independent, tampered)
    assert (
        _error_code(
            lambda: consumer._expected_factorial_proof(producer_config, tampered, sets)
        )
        == "factorial_membership_invalid"
    )


def test_final_toctou_recheck_detects_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    manifest = root / consumer._EXPECTED_PATHS["producer_fixture"] / "manifest.json"
    original_validate = consumer._validate
    replaced = False

    def replace_after_decision(validator: object, value: object, code: str) -> None:
        nonlocal replaced
        original_validate(validator, value, code)
        if code == "decision_schema_invalid" and not replaced:
            replaced = True
            manifest.write_bytes(manifest.read_bytes() + b" ")

    monkeypatch.setattr(consumer, "_validate", replace_after_decision)
    assert _error_code(lambda: _evaluate(root)) == "manifest_changed"


def test_reparse_partition_is_rejected_when_supported(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    fixture = root / consumer._EXPECTED_PATHS["producer_fixture"]
    partition = fixture / "independent_confirmation/bundles.jsonl"
    target = tmp_path / "outside.jsonl"
    target.write_bytes(partition.read_bytes())
    partition.unlink()
    try:
        partition.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink capability is unavailable")
    assert _error_code(lambda: _evaluate(root)) == "partition_path_invalid"
