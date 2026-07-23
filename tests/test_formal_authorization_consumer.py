from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml
from jsonschema import Draft202012Validator

from anchor_mvp.research import formal_authorization_consumer as consumer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / consumer.CONFIG_PATH


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def decision() -> dict[str, object]:
    return consumer.evaluate_formal_authorization(consumer.CONFIG_PATH)


def test_config_and_all_dependencies_have_exact_physical_hashes() -> None:
    config = _config()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == consumer.CONFIG_SHA256
    paths = config["paths"]
    bindings = config["bindings"]
    roles = {
        "qwen_prerequisite_config": "qwen_prerequisite_config_sha256",
        "qwen_prerequisite_implementation": ("qwen_prerequisite_implementation_sha256"),
        "qwen_prerequisite_manifest": "qwen_prerequisite_manifest_sha256",
        "qwen_prerequisite_manifest_sidecar": (
            "qwen_prerequisite_manifest_sidecar_physical_sha256"
        ),
        "multiseed_plan_config": "multiseed_plan_config_sha256",
        "multiseed_plan_implementation": "multiseed_plan_implementation_sha256",
        "training_release_consumer_implementation": (
            "training_release_consumer_implementation_sha256"
        ),
        "generic_release_v2_schema": "generic_release_v2_schema_sha256",
        "formal_decision_schema": "formal_decision_schema_sha256",
    }
    assert {
        role: hashlib.sha256((ROOT / paths[role]).read_bytes()).hexdigest()
        for role in roles
    } == {role: bindings[binding] for role, binding in roles.items()}


def test_decision_is_permanently_blocked(decision: dict[str, object]) -> None:
    assert decision["schema_version"] == consumer.DECISION_VERSION
    assert decision["status"] == "blocked_formal_authorization_inputs_unavailable"
    assert decision["training_authorized"] is False
    assert decision["formal_training_authorized"] is False
    assert decision["formal"] is False
    assert decision["gates"] == {
        "execution_ready": False,
        "materialization_ready": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "independent_confirmation_validated": False,
        "bundle_generalization_validated": False,
    }
    assert decision["qwen_prerequisite"] == {
        "status": "blocked",
        "formal_v3_ready_count": 0,
        "formal_v3_total": 5,
        "protected_inventory_ready_count": 2,
        "protected_inventory_total": 6,
        "training_authorized": False,
        "formal_training_authorized": False,
    }


def test_physical_decision_schema_is_blocked_only(
    decision: dict[str, object],
) -> None:
    config = _config()
    schema_path = ROOT / config["paths"]["formal_decision_schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(schema_path.read_bytes()).hexdigest()
        == (config["bindings"]["formal_decision_schema_sha256"])
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(decision)

    forged = copy.deepcopy(decision)
    forged["status"] = "ready"
    forged["training_authorized"] = True
    forged["formal_training_authorized"] = True
    forged["formal"] = True
    forged["decision_sha256"] = consumer._canonical_sha256(
        {key: value for key, value in forged.items() if key != "decision_sha256"}
    )
    assert not validator.is_valid(forged)


def test_secondary_factorial_cannot_satisfy_independent_gate(
    decision: dict[str, object],
) -> None:
    plan = decision["multiseed_plan"]
    assert plan["status"] == (
        "blocked_controlled_factorial_confirmation_inputs_unavailable"
    )
    assert plan["secondary_factorial_may_satisfy_independent_confirmation"] is False
    assert plan["secondary_factorial_may_satisfy_bundle_generalization"] is False
    assert plan["planned_independent_training_jobs"] == 40
    assert plan["planned_trainable_checkpoint_receipts"] == 200
    assert plan["planned_throughput_order_slots"] == 240


def test_legacy_generic_release_v2_is_never_formal(
    decision: dict[str, object],
) -> None:
    assert decision["legacy_generic_release_v2"] == {
        "schema_version": "anchor.generic-train-release-lock.v2",
        "claim_scope": "research_proxy_only",
        "eligible_for_formal": False,
        "reason": "legacy_release_v2_is_research_proxy_only",
    }


def test_decision_has_zero_requests_and_stable_digest(
    decision: dict[str, object],
) -> None:
    assert decision["audit"] == consumer._ZERO_AUDIT
    body = dict(decision)
    digest = body.pop("decision_sha256")
    assert digest == consumer._canonical_sha256(body)


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("policy", "training_authorized", True, "policy_invalid"),
        (
            "policy",
            "old_generic_release_claim_scope",
            "formal",
            "policy_invalid",
        ),
        ("audit", "network_requests", 1, "audit_invalid"),
    ],
)
def test_authorizing_or_nonzero_config_drift_is_rejected(
    section: str, key: str, value: object, error: str
) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = value
    with pytest.raises(consumer.FormalAuthorizationConsumerError, match=error):
        consumer._validate_config(config)


def test_training_release_implementation_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = consumer._read_snapshot

    def drift(path: Path, code: str) -> consumer._Snapshot:
        snapshot = original(path, code)
        if path.name == "training_release_consumer.py":
            return consumer._Snapshot(
                snapshot.path,
                snapshot.data + b"\n",
                hashlib.sha256(snapshot.data + b"\n").hexdigest(),
                snapshot.identity,
            )
        return snapshot

    monkeypatch.setattr(consumer, "_read_snapshot", drift)
    with pytest.raises(
        consumer.FormalAuthorizationConsumerError,
        match="training_release_consumer_implementation_sha256_mismatch",
    ):
        consumer.evaluate_formal_authorization(consumer.CONFIG_PATH)


def test_snapshot_replacement_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(consumer, "_ROOT", tmp_path)
    path = tmp_path / "identity.txt"
    path.write_bytes(b"first\n")
    snapshot = consumer._read_snapshot(path, "read_failed")
    path.write_bytes(b"second\n")
    with pytest.raises(consumer.FormalAuthorizationConsumerError, match="changed"):
        snapshot.assert_unchanged("changed")


def test_exact_snapshot_execution_ignores_preimported_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(consumer, "_ROOT", tmp_path)
    path = tmp_path / "dependency.py"
    path.write_text("def evaluate():\n    return 'snapshot'\n", encoding="utf-8")
    snapshot = consumer._read_snapshot(path, "read_failed")
    name = "anchor_mvp.research._formal_auth_isolation_test"
    fake = ModuleType(name)
    fake.evaluate = lambda: "import-cache"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, fake)
    assert consumer._execute_snapshot(snapshot, name, "evaluate") == "snapshot"
    assert sys.modules[name] is fake


def test_manifest_sidecar_is_mandatory_and_exact() -> None:
    config = _config()
    manifest = consumer._read_snapshot(
        ROOT / config["paths"]["qwen_prerequisite_manifest"], "manifest_failed"
    )
    sidecar = consumer._read_snapshot(
        ROOT / config["paths"]["qwen_prerequisite_manifest_sidecar"],
        "sidecar_failed",
    )
    consumer._validate_sidecar(manifest, sidecar, "manifest.json")
    invalid = consumer._Snapshot(
        sidecar.path,
        sidecar.data.replace(b"manifest.json", b"wrong.json"),
        sidecar.sha256,
        sidecar.identity,
    )
    with pytest.raises(
        consumer.FormalAuthorizationConsumerError,
        match="qwen_prerequisite_manifest_sidecar_invalid",
    ):
        consumer._validate_sidecar(manifest, invalid, "manifest.json")


def test_cli_returns_blocked_exit_code(
    decision: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        consumer, "evaluate_formal_authorization", lambda _path: decision
    )
    assert consumer.main(["--config", consumer.CONFIG_PATH]) == 2
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "blocked_formal_authorization_inputs_unavailable"
