from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

import anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_risk_evidence as audit_module
from anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_risk_evidence import (
    CONTRACT_PATH,
    CONTRACT_SIDECAR_PATH,
    EXPECTED_CONSUMER_COMMIT,
    EXPECTED_CONSUMER_TREE,
    EXPECTED_SOURCES,
    IMPLEMENTATION_PATH,
    SCHEMA_PATH,
    RiskEvidenceAuditError,
    audit_risk_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = (
    ROOT / "scripts/data/audit_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py"
)
COPY_PATHS = (
    CONTRACT_PATH,
    CONTRACT_SIDECAR_PATH,
    SCHEMA_PATH,
    IMPLEMENTATION_PATH,
    *(value["path"] for value in audit_module.EXPECTED_FROZEN_V1.values()),
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sidecar(data: bytes, filename: str) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    return f"{digest}  {filename}\n".encode("ascii")


def _copy_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            "--quiet",
            os.fspath(ROOT),
            os.fspath(root),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    for relative in COPY_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def _load(root: Path) -> dict[str, object]:
    return json.loads((root / CONTRACT_PATH).read_bytes())


def _write_contract(root: Path, value: object) -> None:
    data = _canonical(value)
    (root / CONTRACT_PATH).write_bytes(data)
    (root / CONTRACT_SIDECAR_PATH).write_bytes(_sidecar(data, Path(CONTRACT_PATH).name))


def _assert_code(root: Path, code: str) -> None:
    with pytest.raises(RiskEvidenceAuditError) as raised:
        audit_risk_evidence(root)
    assert raised.value.code == code


def test_published_schema_and_full_git_object_audit_pass() -> None:
    schema = json.loads((ROOT / SCHEMA_PATH).read_bytes())
    contract = json.loads((ROOT / CONTRACT_PATH).read_bytes())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert audit_risk_evidence(ROOT) == {
        "schema_version": (
            "anchor.synthetic-scaffold-controlled-proxy-risk-evidence-companion.v1"
        ),
        "status": "passed",
        "contract_status": "frozen_additive_non_authorizing_risk_evidence",
        "consumer_commit": EXPECTED_CONSUMER_COMMIT,
        "receipt_count": 3,
        "primary_endpoint": "step_80",
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "multi_seed_validated": False,
        "bundle_generalization_validated": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
    }


def test_mandatory_sidecar_is_exact_sha256sum_lf() -> None:
    contract = (ROOT / CONTRACT_PATH).read_bytes()
    assert (ROOT / CONTRACT_SIDECAR_PATH).read_bytes() == _sidecar(
        contract, Path(CONTRACT_PATH).name
    )


def test_frozen_v1_identities_and_audit_remain_unchanged() -> None:
    for binding in audit_module.EXPECTED_FROZEN_V1.values():
        data = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == binding["sha256"]
    assert audit_module.frozen_v1.audit_followup(ROOT)["status"] == "passed"


def test_consumer_worktree_receipts_are_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {
        (ROOT / source[key]).resolve()
        for source in EXPECTED_SOURCES
        for key in ("path", "sidecar_path")
    }
    original_read_bytes = Path.read_bytes
    original_open = Path.open

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() in forbidden:
            raise AssertionError(
                "audit must authenticate Git blobs, not worktree receipts"
            )
        return original_read_bytes(path)

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.resolve() in forbidden:
            raise AssertionError(
                "audit must authenticate Git blobs, not worktree receipts"
            )
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "open", guarded_open)
    assert audit_risk_evidence(ROOT)["receipt_count"] == 3


def test_cli_outputs_only_content_free_summary() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(ROOT / "src")
    result = subprocess.run(
        [sys.executable, os.fspath(AUDIT_SCRIPT), "--repo-root", os.fspath(ROOT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["status"] == "passed"
    assert summary["receipt_count"] == 3
    assert "prompt" not in result.stdout
    assert "answer" not in result.stdout


@pytest.mark.parametrize(
    "claim",
    [
        "formal",
        "training_authorized",
        "formal_training_authorized",
        "eval_proxy_is_heldout",
        "ood_proxy_is_heldout",
        "independent_o_only_mechanism_proven",
        "memorization_proven",
        "exploit_memorization_tested",
        "causal_mechanism_proven",
        "attention_equals_explanation",
        "statistical_significance_claimed",
        "multi_seed_validated",
        "bundle_generalization_validated",
        "long_context_generalization_claimed",
        "physical_kv_reuse_claimed",
        "zero_copy_claimed",
        "q_reader_implemented",
        "quality_validated",
    ],
)
def test_every_promotion_claim_fails_closed(tmp_path: Path, claim: str) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    claims = contract["claims"]
    assert isinstance(claims, dict)
    claims[claim] = True
    _write_contract(root, contract)
    _assert_code(root, "risk_contract_schema_invalid")


def test_closed_metric_projection_rejects_contract_drift(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    projection = contract["closed_projection"]
    assert isinstance(projection, dict)
    ablation = projection["q_o_branch_ablation"]
    assert isinstance(ablation, dict)
    ablation["o_branch_retained_fraction_of_off_to_full_reduction"] = 0.99
    _write_contract(root, contract)
    _assert_code(root, "risk_closed_projection_mismatch")


def test_attention_layer_projection_rejects_drift(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    projection = contract["closed_projection"]
    assert isinstance(projection, dict)
    attention = projection["attention_single_probe"]
    assert isinstance(attention, dict)
    differences = attention["full_minus_q_branch_retained"]
    assert isinstance(differences, dict)
    layer = differences["layer_13"]
    assert isinstance(layer, dict)
    layer["mean_absolute"] = 0.0
    _write_contract(root, contract)
    _assert_code(root, "risk_closed_projection_mismatch")


def test_source_gpu_counter_cannot_be_washed_to_zero(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    audit = contract["audit"]
    assert isinstance(audit, dict)
    counters = audit["consumer_source_counters"]
    assert isinstance(counters, dict)
    ablation = counters["q_o_branch_ablation"]
    assert isinstance(ablation, dict)
    ablation["gpu_requests"] = 0
    _write_contract(root, contract)
    _assert_code(root, "risk_contract_schema_invalid")


def test_attention_missing_provider_counter_is_not_fabricated() -> None:
    contract = _load(ROOT)
    audit = contract["audit"]
    assert isinstance(audit, dict)
    source = audit["consumer_source_counters"]
    assert isinstance(source, dict)
    attention = source["attention_hook"]
    assert attention == {
        "provider_requests_reported": False,
        "network_requests": 0,
        "model_loads": 1,
        "gpu_requests": 1,
        "heldout_reads": 0,
        "protected_body_reads": 0,
    }


def test_receipt_binding_drift_fails_before_git_read(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    evidence = contract["consumer_git_evidence"]
    assert isinstance(evidence, dict)
    sources = evidence["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["sha256"] = "0" * 64
    _write_contract(root, contract)
    _assert_code(root, "risk_contract_identity_invalid")


def test_frozen_producer_commit_parent_drift_fails_closed(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract = _load(root)
    dependency = contract["frozen_v1_dependency"]
    assert isinstance(dependency, dict)
    dependency["producer_parent_commit"] = "0" * 40
    _write_contract(root, contract)
    _assert_code(root, "risk_frozen_v1_identity_invalid")


def test_frozen_producer_git_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = audit_module._tree_blob
    target = next(iter(audit_module.EXPECTED_FROZEN_V1.values()))

    def drift(root: Path, tree: str, path: str, oid: str) -> bytes:
        data = original(root, tree, path, oid)
        if tree == audit_module.EXPECTED_PRODUCER_TREE and path == target["path"]:
            return data + b" "
        return data

    monkeypatch.setattr(audit_module, "_tree_blob", drift)
    _assert_code(ROOT, "risk_frozen_v1_git_identity_invalid")


def test_synchronized_receipt_and_sidecar_drift_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = audit_module._tree_blob
    target = EXPECTED_SOURCES[0]
    changed = (
        original(
            ROOT,
            EXPECTED_CONSUMER_TREE,
            target["path"],
            target["git_blob_sha1"],
        )
        + b" "
    )

    def drift(root: Path, tree: str, path: str, oid: str) -> bytes:
        if path == target["path"]:
            return changed
        if path == target["sidecar_path"]:
            return _sidecar(changed, Path(target["path"]).name)
        return original(root, tree, path, oid)

    monkeypatch.setattr(audit_module, "_tree_blob", drift)
    _assert_code(ROOT, "risk_git_blob_hash_invalid")


def test_git_blob_end_recheck_detects_toctou(monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit_module._tree_blob
    target = EXPECTED_SOURCES[0]
    calls = 0

    def drift_late(root: Path, tree: str, path: str, oid: str) -> bytes:
        nonlocal calls
        data = original(root, tree, path, oid)
        if path == target["path"]:
            calls += 1
            if calls == 2:
                return data + b" "
        return data

    monkeypatch.setattr(audit_module, "_tree_blob", drift_late)
    _assert_code(ROOT, "risk_git_input_changed")


def test_git_invocations_disable_replace_and_never_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = audit_module.subprocess.run
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def recording_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        environments.append(environment)
        return original(command, **kwargs)

    monkeypatch.setattr(audit_module.subprocess, "run", recording_run)
    assert audit_risk_evidence(ROOT)["status"] == "passed"
    assert commands
    assert len(environments) == len(commands)
    for command in commands:
        assert "--no-replace-objects" in command
        assert "protocol.allow=never" in command
        assert not {"fetch", "pull", "push", "ls-remote"}.intersection(command)
    allowed_git_environment = {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_NO_LAZY_FETCH",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_TERMINAL_PROMPT",
    }
    for environment in environments:
        assert environment["GIT_NO_LAZY_FETCH"] == "1"
        assert {
            key for key in environment if key.upper().startswith("GIT_")
        } == allowed_git_environment


def test_replace_and_grafts_cannot_change_raw_commit_identity(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(root),
            "replace",
            EXPECTED_CONSUMER_COMMIT,
            audit_module.EXPECTED_CONSUMER_PARENT,
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(root),
            "replace",
            audit_module.EXPECTED_PRODUCER_COMMIT,
            audit_module.EXPECTED_PRODUCER_PARENT,
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    git_dir = root / ".git"
    info = git_dir / "info"
    info.mkdir(exist_ok=True)
    (info / "grafts").write_text(
        (
            f"{EXPECTED_CONSUMER_COMMIT} {'0' * 40}\n"
            f"{audit_module.EXPECTED_PRODUCER_COMMIT} {'0' * 40}\n"
        ),
        encoding="ascii",
        newline="\n",
    )
    assert audit_risk_evidence(root)["status"] == "passed"


def test_missing_exact_consumer_object_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", os.fspath(root)], check=True)
    for relative in COPY_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    _assert_code(root, "risk_git_object_unavailable")


def test_duplicate_key_and_nonfinite_contracts_fail_closed(tmp_path: Path) -> None:
    for suffix in (b',"status":"duplicate"\n}', b',"extra":NaN\n}'):
        root = _copy_root(tmp_path / hashlib.sha256(suffix).hexdigest())
        data = (root / CONTRACT_PATH).read_bytes().rstrip()[:-1] + suffix
        (root / CONTRACT_PATH).write_bytes(data)
        (root / CONTRACT_SIDECAR_PATH).write_bytes(
            _sidecar(data, Path(CONTRACT_PATH).name)
        )
        _assert_code(root, "risk_contract_json_invalid")


def test_absolute_contract_override_is_rejected() -> None:
    with pytest.raises(RiskEvidenceAuditError) as raised:
        audit_risk_evidence(ROOT, ROOT / CONTRACT_PATH)
    assert raised.value.code == "risk_contract_path_invalid"


def test_contract_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    contract_path = root / CONTRACT_PATH
    moved = contract_path.with_suffix(".real.json")
    contract_path.rename(moved)
    try:
        contract_path.symlink_to(moved.name)
    except OSError:
        pytest.skip("symlink creation is not available")
    _assert_code(root, "risk_local_snapshot_invalid")
