from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

import anchor_mvp.research.frozen_prefix_qreader_v2_consumer as consumer


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = "configs/orchestration/profiles/frozen_prefix_qreader_v2.json"
MATERIALIZER_PATH = "configs/research/swebench_natural_language_scaffold_v2.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_consumer(tmp_path: Path) -> Path:
    paths = (
        consumer.CONFIG_PATH,
        consumer.MANIFEST_SCHEMA_PATH,
        consumer.DECISION_SCHEMA_PATH,
        consumer.FIXTURE_PATH,
    )
    for relative in paths:
        source = ROOT / relative
        destination = tmp_path / relative
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return tmp_path


def _evaluate(root: Path) -> dict[str, object]:
    return consumer._evaluate(
        root,
        consumer.CONFIG_PATH,
        consumer.CONFIG_SHA256,
        provenance_repo=ROOT,
    )


def _error(callable_: object) -> str:
    with pytest.raises(consumer.FrozenPrefixQReaderConsumerError) as exc:
        callable_()  # type: ignore[operator]
    return exc.value.code


@pytest.fixture(scope="module")
def producer_blobs() -> dict[str, bytes]:
    config = consumer._load_yaml((ROOT / consumer.CONFIG_PATH).read_bytes(), "invalid")
    blobs, _ = consumer._authenticate_git(ROOT, config)
    return blobs


def _set_nested(
    mapping: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    current = mapping
    for key in path[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = value


def _mutated_contract_blobs(
    source: dict[str, bytes],
    document_path: str,
    field_path: tuple[str, ...],
    value: object,
) -> dict[str, bytes]:
    blobs = dict(source)
    if document_path.endswith(".json"):
        document = json.loads(blobs[document_path])
        _set_nested(document, field_path, value)
        blobs[document_path] = json.dumps(document, sort_keys=True).encode("utf-8")
    else:
        document = yaml.safe_load(blobs[document_path])
        _set_nested(document, field_path, value)
        blobs[document_path] = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    return blobs


def test_frozen_consumer_hashes_and_sidecar_are_exact() -> None:
    assert _sha256(ROOT / consumer.CONFIG_PATH) == consumer.CONFIG_SHA256
    assert (
        _sha256(ROOT / consumer.MANIFEST_SCHEMA_PATH) == consumer.MANIFEST_SCHEMA_SHA256
    )
    assert (
        _sha256(ROOT / consumer.DECISION_SCHEMA_PATH) == consumer.DECISION_SCHEMA_SHA256
    )
    fixture = ROOT / consumer.FIXTURE_PATH
    assert _sha256(fixture / "manifest.json") == consumer.MANIFEST_SHA256
    assert (
        _sha256(fixture / "manifest.json.sha256")
        == consumer.MANIFEST_SIDECAR_PHYSICAL_SHA256
    )
    assert (fixture / "manifest.json.sha256").read_bytes() == (
        f"{consumer.MANIFEST_SHA256}  manifest.json\n".encode("ascii")
    )


def test_real_git_inventory_is_exactly_24_added_paths() -> None:
    config = consumer._load_yaml((ROOT / consumer.CONFIG_PATH).read_bytes(), "invalid")
    blobs, digest = consumer._authenticate_git(ROOT, config)
    assert len(blobs) == 24
    assert digest == consumer.PRODUCER_INVENTORY_SHA256
    assert all(
        hashlib.sha256(blobs[row["path"]]).hexdigest() == row["sha256"]
        for row in config["producer_files"]
    )


def test_model_free_dry_run_authenticates_contract_and_stays_blocked() -> None:
    decision = consumer.evaluate_frozen_prefix_qreader_v2_consumer()
    assert decision["status"] == "producer_contract_ready_execution_blocked"
    assert decision["producer_contract_authenticated"] is True
    assert decision["producer"]["files_authenticated"] == 24
    assert all(decision["contracts"].values())
    assert decision["contracts"] == {
        "role_map_exact": True,
        "bundle_split_exact": True,
        "causal_visibility_exact": True,
        "route_boundary_exact": True,
        "private_tail_exact": True,
        "q_only_primary": True,
        "diagnostic_controls_non_authorizing": True,
        "token_index_not_emitted": True,
        "adapter_off_reencode": True,
        "physical_kv_tensor_not_emitted": True,
        "wide_lora_not_inherited": True,
        "source_records_not_rewritten": True,
    }
    gates = decision["gates"]
    assert gates["formal_v3_ready_count"] == 0
    assert gates["protected_inventory_ready_count"] == 2
    assert all(
        gates[key] is False
        for key in (
            "training_authorized",
            "formal_training_authorized",
            "release_authorized",
            "live_authorized",
            "execution_lease_available",
            "materialized_training_view_available",
        )
    )
    assert decision["audit"]["json_schemas_validated"] == 11
    assert all(
        decision["audit"][key] == 0
        for key in (
            "provider_requests",
            "network_requests",
            "model_loads",
            "gpu_requests",
            "protected_body_reads",
            "gold_body_reads",
            "heldout_body_reads",
        )
    )
    without_digest = dict(decision)
    digest = without_digest.pop("decision_sha256")
    assert digest == consumer._canonical_sha256(without_digest)


def test_cli_reports_ready_contract_but_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert consumer.main([]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["producer_contract_authenticated"] is True
    assert output["gates"]["training_authorized"] is False
    assert output["gates"]["formal_training_authorized"] is False


def test_missing_manifest_sidecar_fails_closed(tmp_path: Path) -> None:
    root = _copy_consumer(tmp_path)
    (root / consumer.FIXTURE_PATH / "manifest.json.sha256").unlink()
    assert _error(lambda: _evaluate(root)) == "manifest_sidecar_unreadable"


def test_nonstandard_manifest_sidecar_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_consumer(tmp_path)
    sidecar = root / consumer.FIXTURE_PATH / "manifest.json.sha256"
    sidecar.write_text(
        f"{consumer.MANIFEST_SHA256} *manifest.json\n",
        encoding="ascii",
        newline="\n",
    )
    monkeypatch.setattr(consumer, "MANIFEST_SIDECAR_PHYSICAL_SHA256", _sha256(sidecar))
    assert _error(lambda: _evaluate(root)) == "manifest_sidecar_invalid"


def test_config_role_contract_cannot_be_relaxed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_consumer(tmp_path)
    path = root / consumer.CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["contract"]["private_tail_cross_expert_transfer_allowed"] = True
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    new_hash = _sha256(path)
    assert (
        _error(
            lambda: consumer._evaluate(
                root,
                consumer.CONFIG_PATH,
                new_hash,
                provenance_repo=ROOT,
            )
        )
        == "consumer_contract_invalid"
    )


def test_producer_tracking_ref_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = consumer._git

    def drift(
        repo: Path,
        arguments: list[str] | tuple[str, ...],
        code: str,
        *,
        max_bytes: int = consumer._MAX_GIT_BLOB_BYTES,
    ) -> bytes:
        if list(arguments) == [
            "rev-parse",
            "--verify",
            consumer.PRODUCER_TRACKING_REF,
        ]:
            return b"0" * 40 + b"\n"
        return original(repo, arguments, code, max_bytes=max_bytes)

    monkeypatch.setattr(consumer, "_git", drift)
    config = consumer._load_yaml((ROOT / consumer.CONFIG_PATH).read_bytes(), "invalid")
    assert (
        _error(lambda: consumer._authenticate_git(ROOT, config))
        == "producer_provenance_mismatch"
    )


def test_materializer_private_tail_drift_is_rejected() -> None:
    config = consumer._load_yaml((ROOT / consumer.CONFIG_PATH).read_bytes(), "invalid")
    blobs, _ = consumer._authenticate_git(ROOT, config)
    tampered = dict(blobs)
    path = "configs/research/swebench_natural_language_scaffold_v2.yaml"
    materializer = yaml.safe_load(tampered[path])
    materializer["cache_contract"]["private_tail_cross_expert_transfer_allowed"] = True
    tampered[path] = yaml.safe_dump(materializer, sort_keys=False).encode("utf-8")
    assert _error(lambda: consumer._assert_producer_contracts(tampered)) in {
        "materializer_schema_invalid",
        "materializer_contract_invalid",
    }


@pytest.mark.parametrize(
    ("case", "document_path", "field_path", "value"),
    [
        (
            "role",
            MATERIALIZER_PATH,
            ("role_contract", "stage_to_expert", "security"),
            "planner",
        ),
        (
            "split",
            PROFILE_PATH,
            ("post_gold_pipeline", "split_group_key"),
            "source_gold_record_id",
        ),
        (
            "current",
            MATERIALIZER_PATH,
            ("visibility_contract", "current_target_body_in_prompt"),
            True,
        ),
        (
            "future",
            MATERIALIZER_PATH,
            ("visibility_contract", "future_block_body_in_prompt"),
            True,
        ),
        (
            "forbidden",
            MATERIALIZER_PATH,
            ("visibility_contract", "forbidden_block_body_in_prompt"),
            True,
        ),
        (
            "route",
            MATERIALIZER_PATH,
            ("route_boundary_contract", "semantics"),
            "single_request",
        ),
        (
            "q_only",
            MATERIALIZER_PATH,
            ("adapter_control_contract", "primary"),
            "q_plus_o",
        ),
        (
            "authorization",
            PROFILE_PATH,
            ("authorization", "training_authorized"),
            True,
        ),
        (
            "token_index",
            MATERIALIZER_PATH,
            ("route_boundary_contract", "token_index_emitted"),
            True,
        ),
        (
            "adapter_off_reencode",
            MATERIALIZER_PATH,
            (
                "route_boundary_contract",
                "committed_scaffold_reencode_adapter_state",
            ),
            "expert_only",
        ),
        (
            "physical_tensor",
            MATERIALIZER_PATH,
            ("cache_contract", "physical_kv_tensor_emitted"),
            True,
        ),
        (
            "wide_lora",
            MATERIALIZER_PATH,
            ("adapter_control_contract", "wide_lora_inherited"),
            True,
        ),
        (
            "source_rewrite",
            PROFILE_PATH,
            ("post_gold_pipeline", "source_records_rewritten"),
            True,
        ),
    ],
)
def test_contract_drift_matrix_fails_closed(
    case: str,
    document_path: str,
    field_path: tuple[str, ...],
    value: object,
    producer_blobs: dict[str, bytes],
) -> None:
    del case
    tampered = _mutated_contract_blobs(producer_blobs, document_path, field_path, value)
    with pytest.raises(consumer.FrozenPrefixQReaderConsumerError):
        consumer._assert_producer_contracts(tampered)


def test_final_toctou_recheck_detects_manifest_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_consumer(tmp_path)
    manifest = root / consumer.FIXTURE_PATH / "manifest.json"
    original_validate = consumer._validate
    replaced = False

    def replace_after_decision(schema: object, value: object, code: str) -> None:
        nonlocal replaced
        original_validate(schema, value, code)  # type: ignore[arg-type]
        if code == "decision_schema_validation_failed" and not replaced:
            replaced = True
            manifest.write_bytes(manifest.read_bytes() + b" ")

    monkeypatch.setattr(consumer, "_validate", replace_after_decision)
    assert _error(lambda: _evaluate(root)) == "manifest_changed"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"a":1,"a":2}', "json_duplicate_key"),
        (b'{"a":NaN}', "json_non_finite_number"),
        (b'{"a":Infinity}', "json_non_finite_number"),
    ],
)
def test_strict_json_parser_rejects_ambiguous_values(raw: bytes, expected: str) -> None:
    assert _error(lambda: consumer._load_json(raw, "invalid")) == expected


def test_yaml_duplicate_key_is_rejected() -> None:
    assert (
        _error(lambda: consumer._load_yaml(b"a: 1\na: 2\n", "invalid"))
        == "config_duplicate_key"
    )


def test_external_schema_reference_is_rejected() -> None:
    schema = b'{"$schema":"https://json-schema.org/draft/2020-12/schema","$ref":"https://example.invalid/schema"}'
    assert (
        _error(lambda: consumer._schema(schema, "invalid"))
        == "schema_external_reference_forbidden"
    )


def test_reparse_sidecar_is_rejected_when_supported(tmp_path: Path) -> None:
    root = _copy_consumer(tmp_path)
    sidecar = root / consumer.FIXTURE_PATH / "manifest.json.sha256"
    target = tmp_path / "outside.sha256"
    target.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    try:
        sidecar.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink capability is unavailable")
    assert _error(lambda: _evaluate(root)) == "manifest_sidecar_unreadable"
