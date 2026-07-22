from __future__ import annotations

import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

import anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_followup as audit_module
from anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_followup import (
    CONTRACT_PATH,
    CONTRACT_SIDECAR_PATH,
    IMPLEMENTATION_PATH,
    SCHEMA_PATH,
    SOURCE_PATH,
    SOURCE_SIDECAR_PATH,
    ControlledProxyFollowupAuditError,
    audit_followup,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = (
    ROOT / "scripts/data/audit_synthetic_scaffold_controlled_proxy_followup_v1.py"
)
COPIED_PATHS = (
    CONTRACT_PATH,
    CONTRACT_SIDECAR_PATH,
    SCHEMA_PATH,
    IMPLEMENTATION_PATH,
    SOURCE_PATH,
    SOURCE_SIDECAR_PATH,
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
    return f"{hashlib.sha256(data).hexdigest()}  {filename}\n".encode("ascii")


def _copy_audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for relative in COPIED_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def _load(root: Path, relative: str) -> dict[str, object]:
    return json.loads((root / relative).read_bytes())


def _write_contract(root: Path, value: object) -> None:
    data = _canonical(value)
    (root / CONTRACT_PATH).write_bytes(data)
    (root / CONTRACT_SIDECAR_PATH).write_bytes(_sidecar(data, Path(CONTRACT_PATH).name))


def _write_source_and_rebind(root: Path, value: object) -> None:
    source_data = _canonical(value)
    source_sidecar = _sidecar(source_data, "comparison.json")
    (root / SOURCE_PATH).write_bytes(source_data)
    (root / SOURCE_SIDECAR_PATH).write_bytes(source_sidecar)
    contract = _load(root, CONTRACT_PATH)
    evidence = contract["evidence"]
    assert isinstance(evidence, dict)
    comparison = evidence["comparison"]
    comparison_sidecar = evidence["comparison_sidecar"]
    assert isinstance(comparison, dict)
    assert isinstance(comparison_sidecar, dict)
    comparison["sha256"] = hashlib.sha256(source_data).hexdigest()
    comparison_sidecar["sha256"] = hashlib.sha256(source_sidecar).hexdigest()
    _write_contract(root, contract)


def _assert_code(root: Path, code: str) -> None:
    with pytest.raises(ControlledProxyFollowupAuditError) as raised:
        audit_followup(root)
    assert raised.value.code == code


def test_published_contract_passes_real_draft202012_and_full_audit() -> None:
    schema = json.loads((ROOT / SCHEMA_PATH).read_bytes())
    contract = json.loads((ROOT / CONTRACT_PATH).read_bytes())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)

    result = audit_followup(ROOT)
    assert result == {
        "arm_count": 3,
        "claim_scope": "single_seed_short_context_controlled_proxy_evidence_only",
        "contract_status": "frozen_non_authorizing_followup_design",
        "diagnostic_only": True,
        "formal_training_authorized": False,
        "gpu_requests": 0,
        "model_loads": 0,
        "network_requests": 0,
        "observed_ranking": ["q_plus_o", "wide_budget_matched", "q_only"],
        "protected_body_reads": 0,
        "provider_requests": 0,
        "schema_version": "anchor.synthetic-scaffold-controlled-proxy-followup.v1",
        "status": "passed",
        "training_authorized": False,
    }


def test_sidecars_are_exact_sha256sum_lf() -> None:
    contract = (ROOT / CONTRACT_PATH).read_bytes()
    source = (ROOT / SOURCE_PATH).read_bytes()
    assert (ROOT / CONTRACT_SIDECAR_PATH).read_bytes() == _sidecar(
        contract, Path(CONTRACT_PATH).name
    )
    assert (ROOT / SOURCE_SIDECAR_PATH).read_bytes() == _sidecar(
        source, "comparison.json"
    )


def test_source_closed_projection_rejects_open_nested_drift() -> None:
    source = _load(ROOT, SOURCE_PATH)
    arms = source["arms"]
    assert isinstance(arms, list) and isinstance(arms[0], dict)
    tensor_audit = arms[0]["saved_tensor_audit"]
    assert isinstance(tensor_audit, dict)
    tensor_audit["upstream_open_schema_note"] = "drift"

    # This models the documented upstream weakness: its nested object was not
    # closed, so an added key can still validate there.  Producer semantics must
    # independently reject it.
    open_upstream_fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["parameters", "ranks", "tensor_count"],
        "properties": {
            "parameters": {"type": "integer"},
            "ranks": {"type": "object"},
            "tensor_count": {"type": "integer"},
        },
    }
    Draft202012Validator(open_upstream_fragment).validate(tensor_audit)
    with pytest.raises(ControlledProxyFollowupAuditError) as raised:
        audit_module._validate_closed_source(source)
    assert raised.value.code == "followup_source_semantics_invalid"


def test_nested_source_and_all_declared_hashes_drifting_together_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_audit_root(tmp_path)
    source = _load(root, SOURCE_PATH)
    arms = source["arms"]
    assert isinstance(arms, list) and isinstance(arms[0], dict)
    tensor_audit = arms[0]["saved_tensor_audit"]
    assert isinstance(tensor_audit, dict)
    tensor_audit["upstream_open_schema_note"] = "drift"
    _write_source_and_rebind(root, source)
    _assert_code(root, "followup_source_identity_invalid")


def test_contract_tamper_with_synchronized_sidecar_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_audit_root(tmp_path)
    contract = _load(root, CONTRACT_PATH)
    evidence = contract["evidence"]
    assert isinstance(evidence, dict)
    evidence["consumer_commit"] = "0" * 40
    _write_contract(root, contract)
    _assert_code(root, "followup_contract_identity_invalid")


def test_claim_promotion_with_synchronized_sidecar_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_audit_root(tmp_path)
    contract = _load(root, CONTRACT_PATH)
    claims = contract["claims"]
    assert isinstance(claims, dict)
    claims["formal_training_authorized"] = True
    _write_contract(root, contract)
    _assert_code(root, "followup_contract_schema_invalid")


def test_historical_split_key_rewrite_is_rejected(tmp_path: Path) -> None:
    root = _copy_audit_root(tmp_path)
    contract = _load(root, CONTRACT_PATH)
    controlled = contract["controlled_contract"]
    assert isinstance(controlled, dict)
    controlled["observed_split_group_key"] = "task_bundle_sha256"
    _write_contract(root, contract)
    _assert_code(root, "followup_contract_schema_invalid")


@pytest.mark.parametrize(
    "cluster",
    [
        "evidence_scope",
        "seed_domain",
        "throughput_schedule",
        "primary_metric",
        "task_bundle",
        "confirmation_overlap",
        "length_formula",
        "length_identity_gate",
    ],
)
def test_followup_plan_clusters_fail_closed_with_synchronized_sidecar(
    tmp_path: Path, cluster: str
) -> None:
    root = _copy_audit_root(tmp_path)
    contract = _load(root, CONTRACT_PATH)
    evidence = contract["evidence"]
    followup = contract["producer_followup"]
    assert isinstance(evidence, dict)
    assert isinstance(followup, dict)
    replication = followup["replication_phase"]
    fixture = followup["stratified_fixture_phase"]
    length = followup["length_sweep_phase"]
    assert isinstance(replication, dict)
    assert isinstance(fixture, dict)
    assert isinstance(length, dict)

    if cluster == "evidence_scope":
        evidence["transitive_run_artifacts_reopened"] = True
    elif cluster == "seed_domain":
        domains = replication["seed_domain_schedules"]
        assert isinstance(domains, dict) and isinstance(domains["cuda"], dict)
        values = domains["cuda"]["values"]
        assert isinstance(values, list)
        values[0] += 1
    elif cluster == "throughput_schedule":
        orders = replication["throughput_order_schedule"]
        assert isinstance(orders, list)
        orders[0] = ["q_only"]
    elif cluster == "primary_metric":
        replication["primary_metric"] = "post_hoc_metric"
    elif cluster == "task_bundle":
        fixture["group_key"] = "source_bundle_id"
    elif cluster == "confirmation_overlap":
        fixture["independent_confirmation_claimed"] = True
    elif cluster == "length_formula":
        length["bucket_measurement"] = "input_tokens_only"
    elif cluster == "length_identity_gate":
        length["capability_gate_current_identity_status"] = "ready"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(cluster)

    _write_contract(root, contract)
    _assert_code(root, "followup_contract_schema_invalid")


def test_seed_schedules_and_throughput_permutations_recompute() -> None:
    contract = _load(ROOT, CONTRACT_PATH)
    followup = contract["producer_followup"]
    assert isinstance(followup, dict)
    replication = followup["replication_phase"]
    assert isinstance(replication, dict)
    masters = replication["seed_schedule"]
    domains = replication["seed_domain_schedules"]
    assert isinstance(masters, list) and isinstance(domains, dict)

    for domain in ("adapter_init", "record_order", "cuda"):
        schedule = domains[domain]
        assert isinstance(schedule, dict)
        values = []
        for master in masters:
            preimage = (
                f"anchor.controlled-proxy-followup.seed.v1\0{domain}\0{master}"
            ).encode("utf-8")
            values.append(
                int.from_bytes(hashlib.sha256(preimage).digest()[:4], "big")
                & 0x7FFFFFFF
            )
        compact = json.dumps(values, separators=(",", ":")).encode("utf-8")
        assert schedule == {
            "domain": domain,
            "values": values,
            "schedule_digest_algorithm": (
                "sha256_utf8_canonical_json_integer_array_v1"
            ),
            "sha256": hashlib.sha256(compact).hexdigest(),
        }

    profiles = ("q_only", "q_plus_o", "wide_budget_matched")
    assert replication["throughput_order_schedule"] == [
        list(order) for order in permutations(profiles)
    ]


def test_fake_split_key_added_to_source_is_rejected(tmp_path: Path) -> None:
    root = _copy_audit_root(tmp_path)
    source = _load(root, SOURCE_PATH)
    common = source["common"]
    assert isinstance(common, dict)
    common["split_group_key"] = "task_bundle_sha256"
    _write_source_and_rebind(root, source)
    _assert_code(root, "followup_source_identity_invalid")


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    root = _copy_audit_root(tmp_path)
    path = root / CONTRACT_PATH
    data = path.read_bytes().replace(
        b'{\n  "schema_version"',
        b'{\n  "status":"frozen_non_authorizing_followup_design",\n  "schema_version"',
        1,
    )
    path.write_bytes(data)
    (root / CONTRACT_SIDECAR_PATH).write_bytes(_sidecar(data, Path(CONTRACT_PATH).name))
    _assert_code(root, "followup_contract_json_invalid")


@pytest.mark.parametrize("replacement", [b"NaN", b"1e9999"])
def test_nonfinite_json_number_is_rejected(tmp_path: Path, replacement: bytes) -> None:
    root = _copy_audit_root(tmp_path)
    path = root / CONTRACT_PATH
    data = path.read_bytes().replace(
        b'"eval_macro_loss_before": 3.1020863533020018',
        b'"eval_macro_loss_before": ' + replacement,
        1,
    )
    assert b'"eval_macro_loss_before": ' + replacement in data
    path.write_bytes(data)
    (root / CONTRACT_SIDECAR_PATH).write_bytes(_sidecar(data, Path(CONTRACT_PATH).name))
    _assert_code(root, "followup_contract_json_invalid")


@pytest.mark.parametrize(
    "bad_sidecar",
    [
        b"",
        b"0" * 64 + b"  synthetic_scaffold_controlled_proxy_followup_v1.json\n",
        b"0" * 64 + b" synthetic_scaffold_controlled_proxy_followup_v1.json\n",
        b"0" * 64 + b"  synthetic_scaffold_controlled_proxy_followup_v1.json\r\n",
    ],
)
def test_contract_sidecar_variants_fail_closed(
    tmp_path: Path, bad_sidecar: bytes
) -> None:
    root = _copy_audit_root(tmp_path)
    (root / CONTRACT_SIDECAR_PATH).write_bytes(bad_sidecar)
    _assert_code(root, "followup_contract_sidecar_invalid")


def test_relative_contract_path_is_mandatory() -> None:
    with pytest.raises(ControlledProxyFollowupAuditError) as raised:
        audit_followup(
            ROOT, Path("../synthetic_scaffold_controlled_proxy_followup_v1.json")
        )
    assert raised.value.code == "followup_path_invalid"
    with pytest.raises(ControlledProxyFollowupAuditError) as raised:
        audit_followup(ROOT, ROOT / CONTRACT_PATH)
    assert raised.value.code == "followup_path_invalid"


def test_progressive_symlink_chain_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    root = _copy_audit_root(tmp_path)
    source_path = root / SOURCE_PATH
    external = tmp_path / "external.json"
    shutil.copyfile(source_path, external)
    source_path.unlink()
    try:
        source_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    _assert_code(root, "followup_path_reparse_rejected")


def test_final_toctou_recheck_detects_post_validation_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_audit_root(tmp_path)
    real_snapshot = audit_module._snapshot
    source_reads = 0

    def drifting_snapshot(
        candidate_root: Path, relative_path: str, *, max_bytes: int
    ) -> audit_module.BytesSnapshot:
        nonlocal source_reads
        if relative_path == SOURCE_PATH:
            source_reads += 1
            if source_reads == 2:
                path = candidate_root / relative_path
                path.write_bytes(path.read_bytes() + b" ")
        return real_snapshot(candidate_root, relative_path, max_bytes=max_bytes)

    monkeypatch.setattr(audit_module, "_snapshot", drifting_snapshot)
    _assert_code(root, "followup_input_changed")
    assert source_reads == 2


def test_same_snapshot_reparse_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_strict_json = audit_module._strict_json
    contract_parses = 0

    def drifting_parse(data: bytes, code: str) -> dict[str, object]:
        nonlocal contract_parses
        value = dict(real_strict_json(data, code))
        if code == "followup_contract_json_invalid":
            contract_parses += 1
            if contract_parses == 2:
                value["status"] = "reparse_drift"
        return value

    monkeypatch.setattr(audit_module, "_strict_json", drifting_parse)
    with pytest.raises(ControlledProxyFollowupAuditError) as raised:
        audit_followup(ROOT)
    assert raised.value.code == "followup_contract_reparse_mismatch"
    assert contract_parses == 2


def test_cli_outputs_only_content_free_metadata_summary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    assert value["status"] == "passed"
    assert value["arm_count"] == 3
    assert value["training_authorized"] is False
    assert value["formal_training_authorized"] is False
    assert value["protected_body_reads"] == 0
    assert not {
        "answer",
        "content",
        "input_ids",
        "preview",
        "prompt",
        "token_ids",
    }.intersection(value)


def test_reparse_attribute_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = audit_module._has_reparse_attribute

    def mark_schema(value: os.stat_result) -> bool:
        return original(value) or value.st_size == (ROOT / SCHEMA_PATH).stat().st_size

    monkeypatch.setattr(audit_module, "_has_reparse_attribute", mark_schema)
    with pytest.raises(ControlledProxyFollowupAuditError):
        audit_followup(ROOT)
