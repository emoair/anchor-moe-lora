from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from anchor_mvp.research import (
    qwen_controlled_proxy_risk_evidence_consumer as consumer,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / consumer.CONFIG_PATH
RISK_CONTRACT = (
    ROOT / "configs/research/"
    "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _risk_contract() -> dict[str, object]:
    value = json.loads(RISK_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def decision() -> dict[str, object]:
    return consumer.evaluate_risk_evidence(consumer.CONFIG_PATH)


def test_full_overlay_executes_all_three_layers_and_stays_blocked(
    decision: dict[str, object],
) -> None:
    assert decision["schema_version"] == consumer.DECISION_VERSION
    assert decision["status"] == "blocked"
    assert decision["training_authorized"] is False
    assert decision["formal_training_authorized"] is False
    assert decision["formal"] is False
    assert decision["prerequisite_status"] == "blocked"
    assert decision["producer_audits"] == {
        "followup_v1": "passed",
        "risk_companion_v1": "passed",
    }


def test_retained_o_branch_is_not_reported_as_independent_o_only(
    decision: dict[str, object],
) -> None:
    branch = decision["branch_ablation"]
    assert isinstance(branch, dict)
    assert branch["checkpoint_semantics"] == (
        "jointly_trained_q_plus_o_checkpoint_retained_branches"
    )
    assert branch["retained_branch_label"] == "o_branch_retained"
    assert branch["independently_trained_o_only_claimed"] is False
    assert branch["independent_o_only_mechanism_proven"] is False
    assert branch["branch_effects_additive_claimed"] is False
    assert branch["primary_endpoint"] == "step_80"
    assert branch["o_branch_retained_fraction_of_off_to_full_reduction"] == (
        pytest.approx(0.9236553979475981)
    )


def test_decision_contains_only_non_authorizing_counters(
    decision: dict[str, object],
) -> None:
    assert decision["audit"] == {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_content_reads": 0,
        "consumer_source_worktree_receipts_read": 0,
    }
    assert decision["promotion_gates"] == {
        "formal_v3_ready_count": 0,
        "formal_v3_total": 5,
        "protected_inventory_ready_count": 2,
        "protected_inventory_total": 6,
        "multi_seed_validated": False,
        "bundle_generalization_validated": False,
    }


def test_config_and_prerequisite_identities_are_exact() -> None:
    assert _sha256(CONFIG) == consumer.CONFIG_SHA256
    assert (
        _sha256(ROOT / "configs/research/qwen_train_prerequisite_consumer_v2.yaml")
        == consumer.PREREQUISITE_CONFIG_SHA256
    )
    assert (
        _sha256(ROOT / "src/anchor_mvp/research/qwen_train_prerequisite_consumer_v2.py")
        == consumer.PREREQUISITE_IMPLEMENTATION_SHA256
    )


def test_all_release_artifact_physical_hashes_match_config() -> None:
    config = _config()
    releases = config["producer_releases"]
    assert isinstance(releases, dict)
    for release in releases.values():
        assert isinstance(release, dict)
        for artifact in release["artifacts"]:
            assert isinstance(artifact, dict)
            assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_mandatory_sidecars_are_exact() -> None:
    config = _config()
    paths = config["paths"]
    assert isinstance(paths, dict)
    pairs = (
        ("followup_contract", "followup_contract_sidecar"),
        ("followup_source", "followup_source_sidecar"),
        ("risk_contract", "risk_contract_sidecar"),
    )
    for document_role, sidecar_role in pairs:
        document = ROOT / paths[document_role]
        sidecar = ROOT / paths[sidecar_role]
        expected = f"{_sha256(document)}  {document.name}\n".encode("ascii")
        assert sidecar.read_bytes() == expected


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("policy", "training_authorized"), True),
        (("policy", "formal_training_authorized"), True),
        (("policy", "formal"), True),
        (("policy", "independently_trained_o_only_claimed"), True),
        (("policy", "branch_effects_additive_claimed"), True),
        (("policy", "retained_branch_label"), "o_only"),
        (("producer_releases", "risk_companion_v1", "commit"), "0" * 40),
        (("consumer_evidence_binding", "commit"), "0" * 40),
    ],
)
def test_config_semantic_or_identity_drift_fails_closed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config())
    current: dict[str, object] = config
    for key in path[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = value
    with pytest.raises(consumer.ControlledProxyRiskConsumerError):
        consumer._validate_config(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claims", "independent_o_only_mechanism_proven"), True),
        (("claims", "causal_mechanism_proven"), True),
        (("claims", "memorization_proven"), True),
        (("claims", "formal"), True),
        (("claims", "training_authorized"), True),
        (("claims", "formal_training_authorized"), True),
        (
            ("followup_constraints", "branch_ablation_interpretation"),
            "independently_trained_o_only",
        ),
        (
            (
                "closed_projection",
                "q_o_branch_ablation",
                "branch_effects_additive_claimed",
            ),
            True,
        ),
    ],
)
def test_risk_interpretation_promotion_fails_closed(
    path: tuple[str, ...], value: object
) -> None:
    contract = copy.deepcopy(_risk_contract())
    current: dict[str, object] = contract
    for key in path[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = value
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError,
        match="risk_interpretation_boundary_drift",
    ):
        consumer._validate_risk_interpretation(contract)


def test_yaml_duplicate_key_is_rejected() -> None:
    data = b"schema_version: one\nschema_version: two\n"
    snapshot = consumer._BytesSnapshot(Path("unused"), data, "0" * 64, (0, 0, 0, 0))
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError,
        match="consumer_config_duplicate_key",
    ):
        consumer._decode_yaml(snapshot)


def test_snapshot_detects_toctou_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(consumer, "_REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "metadata.json"
    path.write_bytes(b"{}\n")
    snapshot = consumer._read_bytes_snapshot(path, "snapshot_invalid")
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError,
        match="snapshot_changed",
    ):
        snapshot.assert_unchanged("snapshot_changed")


def test_missing_mandatory_sidecar_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = consumer._read_bytes_snapshot

    def reject_sidecar(path: Path, code: str, **kwargs: object):
        if path.name == (
            "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json.sha256"
        ):
            raise consumer.ControlledProxyRiskConsumerError("sidecar_missing")
        return original(path, code, **kwargs)

    monkeypatch.setattr(consumer, "_read_bytes_snapshot", reject_sidecar)
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError, match="sidecar_missing"
    ):
        consumer.evaluate_risk_evidence(consumer.CONFIG_PATH)


def test_cli_exact_groups_and_drift() -> None:
    exact = argparse.Namespace(
        prerequisite_config=(
            "configs/research/qwen_train_prerequisite_consumer_v2.yaml"
        ),
        prerequisite_config_sha256=consumer.PREREQUISITE_CONFIG_SHA256,
        followup_contract=None,
        followup_contract_sha256=None,
        followup_contract_sidecar_sha256=None,
        risk_contract=None,
        risk_contract_sha256=None,
        risk_contract_sidecar_sha256=None,
        producer_risk_release_commit=consumer.RISK_RELEASE_COMMIT,
        consumer_evidence_commit=consumer.CONSUMER_EVIDENCE_COMMIT,
    )
    consumer._validate_cli_overrides(exact)

    partial = copy.copy(exact)
    partial.prerequisite_config_sha256 = None
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError,
        match="consumer_cli_override_incomplete",
    ):
        consumer._validate_cli_overrides(partial)

    drift = copy.copy(exact)
    drift.consumer_evidence_commit = "0" * 40
    with pytest.raises(
        consumer.ControlledProxyRiskConsumerError,
        match="consumer_cli_override_drift",
    ):
        consumer._validate_cli_overrides(drift)


def test_successful_cli_still_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        consumer,
        "evaluate_risk_evidence",
        lambda _: {
            "schema_version": consumer.DECISION_VERSION,
            "status": "blocked",
            "training_authorized": False,
            "formal_training_authorized": False,
        },
    )
    assert consumer.main(["--config", consumer.CONFIG_PATH]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"
    assert output["training_authorized"] is False
    assert output["formal_training_authorized"] is False
