from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from anchor_mvp.research import qwen_train_prerequisite_consumer as consumer
from anchor_mvp.research.qwen_train_prerequisite_consumer import (
    QwenPrerequisiteConsumerError,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"


def _copy_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = (
        "configs/research/qwen_train_prerequisite_consumer_v1.yaml",
        "configs/research/qwen_train_prerequisite_status.schema.json",
        "configs/research/scaffold_tokenizer_binding_manifest.schema.json",
        "configs/research/qwen_toy_source_disjoint_attestation.schema.json",
        "configs/research/generic_train_release_lock.schema.json",
        "fixtures/research/qwen_train_prerequisite_status/manifest.json",
        "fixtures/research/qwen_train_prerequisite_status/manifest.json.sha256",
    )
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    monkeypatch.setattr(consumer, "_REPOSITORY_ROOT", tmp_path)
    return tmp_path


def _status() -> dict[str, object]:
    return json.loads(
        (
            REPO / "fixtures/research/qwen_train_prerequisite_status/manifest.json"
        ).read_text("utf-8")
    )


def test_frozen_status_authenticates_but_never_authorizes_training() -> None:
    decision = consumer.evaluate_prerequisites(CONFIG)
    assert decision["status"] == "blocked"
    assert decision["training_authorized"] is False
    assert decision["formal_training_authorized"] is False
    assert decision["bindings"] == {
        "prerequisite_status_schema_sha256": consumer.STATUS_SCHEMA_SHA256,
        "prerequisite_status_manifest_sha256": consumer.STATUS_MANIFEST_SHA256,
        "tokenizer_binding_schema_sha256": consumer.TOKENIZER_BINDING_SCHEMA_SHA256,
        "tokenizer_binding_manifest_sha256": None,
        "toy_attestation_schema_sha256": consumer.TOY_ATTESTATION_SCHEMA_SHA256,
        "toy_attestation_sha256": None,
        "formal_release_lock_schema_sha256": (
            consumer.FORMAL_RELEASE_LOCK_SCHEMA_SHA256
        ),
        "formal_release_lock_sha256": None,
    }
    assert decision["audit"] == {
        "protected_dataset_files_read": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }


def test_cli_fields_keep_producer_order_and_overrides_fail_closed() -> None:
    parser = consumer.build_parser()
    observed = tuple(
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings and action.option_strings[0] not in {"-h", "--config"}
    )
    assert observed == consumer.ORDERED_CLI_FIELDS
    args = parser.parse_args(
        ["--config", str(CONFIG), "--toy-attestation", "unfrozen.json"]
    )
    with pytest.raises(
        QwenPrerequisiteConsumerError,
        match="blocked_v1_status_rejects_unfrozen_cli_overrides",
    ):
        consumer._validate_cli_overrides(args)
    exact = parser.parse_args(
        [
            "--config",
            str(CONFIG),
            "--prerequisite-status",
            "fixtures/research/qwen_train_prerequisite_status/manifest.json",
            "--prerequisite-status-sha256",
            consumer.STATUS_MANIFEST_SHA256,
            "--formal-release-lock-schema-sha256",
            consumer.FORMAL_RELEASE_LOCK_SCHEMA_SHA256,
        ]
    )
    consumer._validate_cli_overrides(exact)


def test_cli_prints_authenticated_blocked_decision_and_returns_two(capsys) -> None:
    assert consumer.main(["--config", str(CONFIG)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"
    assert output["training_authorized"] is False


def test_config_binding_order_and_identity_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    consumer._validate_config(config)
    drift = deepcopy(config)
    values = dict(drift["bindings"])
    drift["bindings"] = {
        key: values[key] for key in reversed(consumer.ORDERED_CONFIG_FIELDS)
    }
    with pytest.raises(QwenPrerequisiteConsumerError, match="binding_order_drift"):
        consumer._validate_config(drift)
    drift = deepcopy(config)
    drift["bindings"]["toy_attestation_sha256"] = "0" * 64
    with pytest.raises(QwenPrerequisiteConsumerError, match="binding_identity_drift"):
        consumer._validate_config(drift)
    drift = deepcopy(config)
    drift["claim_scope"] = "broader"
    with pytest.raises(QwenPrerequisiteConsumerError, match="claim_scope_drift"):
        consumer._validate_config(drift)
    drift = deepcopy(config)
    drift["paths"]["prerequisite_status"] = "fixtures/research/other/manifest.json"
    with pytest.raises(QwenPrerequisiteConsumerError, match="canonical_paths_drift"):
        consumer._validate_config(drift)


def test_consumer_config_bytes_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    config = root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
    config.write_bytes(config.read_bytes() + b" ")
    with pytest.raises(QwenPrerequisiteConsumerError, match="config_sha256_mismatch"):
        consumer.evaluate_prerequisites(config)


def test_missing_or_noncanonical_sidecar_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    sidecar = (
        root / "fixtures/research/qwen_train_prerequisite_status/manifest.json.sha256"
    )
    sidecar.unlink()
    with pytest.raises(QwenPrerequisiteConsumerError, match="sidecar_unreadable"):
        consumer.evaluate_prerequisites(
            root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        )

    _copy_contract(tmp_path, monkeypatch)
    sidecar.write_bytes(
        f"{consumer.STATUS_MANIFEST_SHA256}  manifest.json\r\n".encode("ascii")
    )
    with pytest.raises(
        QwenPrerequisiteConsumerError,
        match="sidecar_physical_sha256_mismatch|sidecar_invalid",
    ):
        consumer.evaluate_prerequisites(
            root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        )


def test_manifest_and_schema_byte_drift_fail_before_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    manifest = root / "fixtures/research/qwen_train_prerequisite_status/manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(QwenPrerequisiteConsumerError, match="manifest_sha256_mismatch"):
        consumer.evaluate_prerequisites(
            root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        )

    _copy_contract(tmp_path, monkeypatch)
    schema = root / "configs/research/qwen_train_prerequisite_status.schema.json"
    schema.write_bytes(schema.read_bytes() + b" ")
    with pytest.raises(QwenPrerequisiteConsumerError, match="schema_sha256_mismatch"):
        consumer.evaluate_prerequisites(
            root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        )


def test_reparse_or_symlink_ancestor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_contract(tmp_path, monkeypatch)
    status_dir = root / "fixtures/research/qwen_train_prerequisite_status"
    physical = root / "fixtures/research/qwen_train_prerequisite_status.physical"
    status_dir.rename(physical)
    try:
        status_dir.symlink_to(physical, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit directory symlink creation")
    with pytest.raises(QwenPrerequisiteConsumerError, match="status_path_invalid"):
        consumer.evaluate_prerequisites(
            root / "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda value: value.__setitem__("status", "ready"), "must_remain_blocked"),
        (
            lambda value: value["safety"].__setitem__("training_authorized", True),
            "safety_boolean_drift",
        ),
        (
            lambda value: value["formal_artifacts"]["snapshot"].__setitem__(
                "artifact_exists", True
            ),
            "artifact_snapshot_artifact_exists",
        ),
        (
            lambda value: value["tokenizer_binding"].__setitem__(
                "token_indices_emitted", True
            ),
            "token_indices_must_not_be_emitted",
        ),
        (
            lambda value: value["toy_attestation"].__setitem__(
                "attestation_artifact_status", "verified"
            ),
            "artifact_must_remain_unavailable",
        ),
    ),
)
def test_positive_state_mutations_are_rejected(mutation, code: str) -> None:
    status = _status()
    mutation(status)
    with pytest.raises(QwenPrerequisiteConsumerError, match=code):
        consumer._validate_blocked_status_semantics(status)


def test_content_bearing_keys_are_rejected_recursively() -> None:
    status = _status()
    status["unexpected"] = {"prompt": "not allowed"}
    with pytest.raises(QwenPrerequisiteConsumerError, match="contains_content"):
        consumer._assert_no_content_fields(status, "contains_content")


def test_evaluation_reads_only_contract_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    original = consumer._read_bytes_snapshot

    def recording_reader(path: Path, code: str):
        observed.append(path.resolve())
        return original(path, code)

    monkeypatch.setattr(consumer, "_read_bytes_snapshot", recording_reader)
    decision = consumer.evaluate_prerequisites(CONFIG)
    assert decision["audit"]["protected_dataset_files_read"] == 0
    allowed_names = {
        "qwen_train_prerequisite_consumer_v1.yaml",
        "qwen_train_prerequisite_status.schema.json",
        "scaffold_tokenizer_binding_manifest.schema.json",
        "qwen_toy_source_disjoint_attestation.schema.json",
        "generic_train_release_lock.schema.json",
        "manifest.json",
        "manifest.json.sha256",
    }
    assert {path.name for path in observed} == allowed_names
    assert all(
        "gold" not in str(path).lower()
        and "heldout" not in str(path).lower()
        and "jsonl" not in path.suffix.lower()
        for path in observed
    )
