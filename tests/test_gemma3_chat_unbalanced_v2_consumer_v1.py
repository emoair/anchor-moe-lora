from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.training import gemma3_chat_unbalanced_v2_consumer_v1 as consumer


PRODUCER_COMMIT = "1" * 40
FINAL_STATUS = "release_ready_for_diagnostic_training"
_ORIGINAL_SHARDED_FINAL_TRUST_VALIDATOR = consumer._validate_sharded_final_trust_anchors
_ORIGINAL_SNAPSHOT_TREE = consumer._snapshot_tree


@pytest.fixture(autouse=True)
def _synthetic_sharded_trust_override(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        consumer,
        "_validate_sharded_final_trust_anchors",
        lambda _binding: None,
    )
    return _ORIGINAL_SHARDED_FINAL_TRUST_VALIDATOR


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_payload(root: Path, relative: str, raw: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    sidecar = f"{_sha(raw)}  {Path(relative).name}\n".encode("ascii")
    path.with_name(path.name + ".sha256").write_bytes(sidecar)


def _write_sized_payload(root: Path, relative: str, size: int) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if size:
            handle.seek(size - 1)
            handle.write(b"\0")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    sidecar = f"{digest.hexdigest()}  {path.name}\n".encode("ascii")
    path.with_name(path.name + ".sha256").write_bytes(sidecar)


def _claims() -> dict[str, object]:
    return {
        "candidate": False,
        "final": True,
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": True,
        "formal_training_authorized": False,
        "live_authorized": False,
        "release_authorized": True,
        "quality_validated": False,
        "generalization_validated": False,
    }


def _counts() -> dict[str, object]:
    return {
        "records": 4300,
        "task_bundles": 1700,
        "task_semantics": 1700,
        "roles": {
            "humor": 300,
            "serious": 300,
            "angry_style": 300,
            "tool_call": 1700,
            "review_audit": 1700,
        },
        "splits": {"train": 3440, "eval_proxy": 860},
        "languages": {"en": 2150, "zh-CN": 2150},
        "language_split": {
            "en:train": 1720,
            "zh-CN:train": 1720,
            "en:eval_proxy": 430,
            "zh-CN:eval_proxy": 430,
        },
        "bundle_classes": {
            "full_five_role_core": 300,
            "tool_review_depth": 1400,
        },
        "review_verdicts": {"pass": 850, "fail": 850},
        "review_fault_types": {
            "format_schema": 170,
            "tool_argument": 170,
            "evidence_mismatch": 170,
            "grounding_mismatch": 170,
            "routing_scope": 170,
        },
        "identity_bundles": 30,
        "identity_role_records": 150,
        "identity_eval_probe_records": 50,
        "tool_direct_no_tool": 30,
        "tool_local_executions": 1670,
    }


def _identity() -> dict[str, object]:
    return {
        "source_pool": "full_five_role_core",
        "rows_added_to_training_4300": 0,
        "identity_bundles": 30,
        "train_bundles": 20,
        "eval_proxy_bundles": 10,
        "en_bundles": 15,
        "zh_cn_bundles": 15,
        "intents": {
            "direct_self_identity": 6,
            "provenance_fact_check": 6,
            "no_tool_identity_decision": 6,
            "pre_coding_attribution": 6,
            "false_attribution_correction": 6,
        },
        "role_records": 150,
        "eval_probe_records": 50,
        "tool_behavior": "direct_answer_no_tool",
        "review_rejects_google_openai": True,
        "wrong_route_fact_invariant": True,
    }


def _review() -> dict[str, object]:
    return {
        "records": 1700,
        "pass": 850,
        "fail": 850,
        "fault_types": {
            "format_schema": 170,
            "tool_argument": 170,
            "evidence_mismatch": 170,
            "grounding_mismatch": 170,
            "routing_scope": 170,
        },
        "same_bundle_tool_dependency": True,
        "candidate_projection_hash_bound": True,
        "tool_record_never_mutated_by_review_fixture": True,
    }


def _components(root: Path) -> dict[str, object]:
    def digest(relative: str) -> str:
        return _sha(root.joinpath(*relative.split("/")).read_bytes())

    router = {
        "namespace": "gemma3_chat_emotion_router_qonly_v1",
        "records": 100,
        "train": 80,
        "eval_proxy": 20,
        "en": 50,
        "zh_cn": 50,
        "labels": {"humor": 34, "serious": 33, "angry_style": 33},
        "not_a_sixth_expert": True,
        "single_total_to_branch_then_router_exits": True,
        "independent_from_generation_experts": True,
        "manifest_sha256": digest("components/router/manifest.json"),
        "records_sha256": digest("components/router/records.jsonl"),
        "token_inventory_sha256": digest("components/router/token_inventory.jsonl"),
        "build_receipt_sha256": digest("components/router/build_receipt.json"),
    }

    def eval_component(name: str, records: int, slots: int) -> dict[str, object]:
        namespace = (
            "gemma3_chat_unbalanced_v2_tool_comparison_eval_v1"
            if name == "tool_eval"
            else "gemma3_chat_unbalanced_v2_planner_comparison_eval_v1"
        )
        return {
            "namespace": namespace,
            "records": records,
            "execution_slots": slots,
            "arm_neutral": True,
            "rows_replicated_by_arms": False,
            "oracle_unchanged": True,
            "strata_unchanged": True,
            "language_unchanged": True,
            "negative_inventory_unchanged": True,
            "independent_review_status": "passed",
            "manifest_sha256": digest(f"components/{name}/manifest.json"),
            "records_sha256": digest(f"components/{name}/records.jsonl"),
            "token_inventory_sha256": digest(
                f"components/{name}/token_inventory.jsonl"
            ),
            "build_receipt_sha256": digest(f"components/{name}/build_receipt.json"),
            "independent_review_receipt_sha256": digest(
                f"reviews/{name}_independent_review.json"
            ),
        }

    return {
        "router": router,
        "tool_eval": eval_component("tool_eval", 400, 1200),
        "planner_eval": eval_component("planner_eval", 240, 960),
    }


def _adapter_contract() -> dict[str, object]:
    return {
        "tool_arms": ["base", "tool_q_only", "tool_q_plus_micro_o"],
        "planner_arms": [
            "base",
            "planner_q_only",
            "planner_o_only",
            "planner_q_plus_micro_o",
        ],
        "tool_execution_slots": 1200,
        "planner_execution_slots": 960,
        "q_branch": {
            "target_module": "q_proj",
            "rank": 1024,
            "alpha": 2048,
            "trainable_params": 57933824,
        },
        "micro_o_branch": {
            "target_module": "o_proj",
            "rank": 64,
            "alpha": 128,
            "trainable_params": 3620864,
            "max_lr_ratio_to_q": "1/10",
        },
        "non_tool_q_only_roles": [
            "humor",
            "serious",
            "angry_style",
            "review_audit",
        ],
        "records_added_by_arms": 0,
        "tool_records_arm_neutral": True,
        "planner_records_arm_neutral": True,
        "o_learning_rate_lte_q_over_10_every_step": True,
        "planner_o_only_q_plus_o_o_branch_identity_equal": True,
        "planner_o_only_q_plus_o_o_init_equal": True,
        "planner_o_only_q_plus_o_data_order_steps_seeds_optimizer_equal": True,
        "humor_serious_angry_review_q_only": True,
    }


def _kv_contract() -> dict[str, object]:
    return {
        "producer_mode": "frozen_base_adapter_off_single_shared_prefix_prefill",
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "current_physical_claim": "prefill_compute_reuse",
        "route_plan_commits_immutable_and_hash_bound": True,
        "gemma_canonical_serialization": True,
        "expert_private_tail_append_only": True,
        "planner_live_private_kv_transfer": False,
        "full_generation_kv_shared": False,
        "persistent_zero_copy_verified": False,
        "persistent_zero_copy_blocked_on": [
            "data_ptr_identity",
            "storage_identity",
            "cache_identity",
        ],
        "q8_kv_exact": False,
    }


def _manifest_file_entries(root: Path) -> list[dict[str, object]]:
    result = []
    for relative in sorted(consumer.MANIFEST_LISTED_PATHS):
        raw = root.joinpath(*relative.split("/")).read_bytes()
        result.append(
            {
                "path": relative,
                "sha256": _sha(raw),
                "bytes": len(raw),
                "kind": consumer._kind(relative),
                "records": consumer.RECORD_COUNTS.get(relative),
            }
        )
    return result


def _binding(root: Path) -> dict[str, object]:
    snapshot = consumer._snapshot_tree(root, capture_metadata=True)
    pins = []
    for relative in consumer.PAYLOAD_PATHS:
        payload = snapshot.files[relative]
        sidecar = snapshot.files[f"{relative}.sha256"]
        pins.append(
            {
                "path": relative,
                "kind": "jsonl" if relative.endswith(".jsonl") else "json",
                "sha256": payload.sha256,
                "bytes": payload.bytes,
                "sidecar_sha256": sidecar.sha256,
                "sidecar_bytes": sidecar.bytes,
            }
        )
    return {
        "schema_version": consumer.BINDING_VERSION,
        "namespace": consumer.NAMESPACE,
        "producer_git_commit": PRODUCER_COMMIT,
        "producer_manifest_schema_sha256": _sha(b"manifest-schema"),
        "producer_build_receipt_schema_sha256": _sha(b"receipt-schema"),
        "expected_manifest_status": FINAL_STATUS,
        "tree_digest_sha256": consumer._tree_digest(snapshot.files),
        "payload_files": pins,
        "release_review": {
            "audit_status": "passed",
            "reviewer_separation": "different_reviewer",
            "p0_findings": 0,
            "p1_findings": 0,
            "p2_findings": 0,
            "receipt_sha256": _sha(b"independent-release-review"),
        },
    }


def _make_artifact(
    root: Path,
    *,
    manifest_update: dict[str, object] | None = None,
    receipt_update: dict[str, object] | None = None,
    train_payload_bytes: int | None = None,
) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    for relative in consumer.PAYLOAD_PATHS:
        if relative in {"manifest.json", "build_receipt.json"}:
            continue
        if relative == "train/chat.jsonl" and train_payload_bytes is not None:
            _write_sized_payload(root, relative, train_payload_bytes)
            continue
        raw = _json_bytes({}) if relative.endswith(".json") else b""
        _write_payload(root, relative, raw)
    manifest: dict[str, object] = {
        "schema_version": consumer.MANIFEST_VERSION,
        "status": FINAL_STATUS,
        "namespace": consumer.NAMESPACE,
        "files": _manifest_file_entries(root),
        "counts": _counts(),
        "identity": _identity(),
        "review": _review(),
        "components": _components(root),
        "adapter_contract": _adapter_contract(),
        "kv_contract": _kv_contract(),
        "integrity": {
            "single_authenticated_physical_bytes_snapshot_per_input": True,
            "terminal_toctou_snapshot_recheck_required": True,
            "mandatory_sidecar_for_every_non_sidecar_file": True,
            "checksum_sidecars_are_only_sidecar_exemption": True,
            "raw_token_ids_persisted": False,
            "strict_json_duplicate_and_nonfinite_rejection": True,
            "cross_field_validator": True,
        },
        "claims": _claims(),
        "release_review": {
            "implementer_release_signoff": False,
            "independent_release_review_required": True,
        },
    }
    if manifest_update:
        manifest.update(deepcopy(manifest_update))
    _write_payload(root, "manifest.json", _json_bytes(manifest))
    manifest_raw = (root / "manifest.json").read_bytes()
    receipt: dict[str, object] = {
        "schema_version": consumer.BUILD_RECEIPT_VERSION,
        "status": manifest["status"],
        "claims": deepcopy(manifest["claims"]),
        "manifest": {
            "path": "manifest.json",
            "sha256": _sha(manifest_raw),
            "bytes": len(manifest_raw),
        },
        "resource_counters": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "gold_body_reads": 0,
            "heldout_body_reads": 0,
            "protected_body_reads": 0,
            "luna_requests": 0,
        },
        "official_model_free_audits": {"independent_release_review": "passed"},
    }
    if receipt_update:
        receipt.update(deepcopy(receipt_update))
    _write_payload(root, "build_receipt.json", _json_bytes(receipt))
    return root, _binding(root)


def _validate(root: Path, binding: dict[str, object]) -> dict[str, Any]:
    return consumer.validate_artifact(
        root,
        binding,
        observed_producer_commit=PRODUCER_COMMIT,
    )


def test_pass_is_body_free_and_model_free(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    receipt = _validate(root, binding)
    assert receipt["status"] == "passed"
    assert receipt["file_counts"] == {"total": 44, "payload": 22, "sidecar": 22}
    assert receipt["aggregate_counts"]["training_records"] == 4300
    assert receipt["aggregate_counts"]["router_records"] == 100
    assert receipt["aggregate_counts"]["tool_eval_records"] == 400
    assert receipt["aggregate_counts"]["planner_eval_records"] == 240
    assert receipt["aggregate_counts"]["identity_probe_records"] == 50
    assert receipt["resource_counters"] == {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "gold_body_reads": 0,
        "heldout_body_reads": 0,
        "protected_body_reads": 0,
    }
    serialized = json.dumps(receipt)
    assert str(tmp_path) not in serialized
    assert '"input_ids"' not in serialized
    assert "messages" not in serialized
    assert "target" not in serialized


def test_current_candidate_status_and_false_claims_are_rejected(
    tmp_path: Path,
) -> None:
    claims = _claims()
    claims.update(
        {
            "candidate": True,
            "final": False,
            "training_authorized": False,
            "release_authorized": False,
        }
    )
    root, binding = _make_artifact(
        tmp_path / "artifact",
        manifest_update={
            "status": "candidate_pending_independent_release_review",
            "claims": claims,
        },
        receipt_update={
            "status": "candidate_pending_independent_release_review",
            "claims": claims,
        },
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_release_status_not_final",
    ):
        _validate(root, binding)


def test_payload_size_schema_and_physical_boundaries(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "schema-boundary")
    binding["payload_files"][0]["bytes"] = consumer.MAX_PAYLOAD_BYTES
    consumer.validate_binding_document(binding)
    binding["payload_files"][0]["bytes"] = consumer.MAX_PAYLOAD_BYTES + 1
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        consumer.validate_binding_document(binding)
    consumer._assert_payload_size(52_428_799)
    for rejected in (52_428_800, 59_845_314):
        with pytest.raises(
            consumer.ConsumerPreflightError,
            match="consumer_payload_size_limit_exceeded",
        ):
            consumer._assert_payload_size(rejected)


def test_release_ready_50_mib_payload_fails_physical_size_gate(
    tmp_path: Path,
) -> None:
    root, binding = _make_artifact(
        tmp_path / "artifact",
        train_payload_bytes=52_428_800,
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="consumer_payload_size_limit_exceeded",
    ):
        _validate(root, binding)


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        ("final", False),
        ("release_authorized", False),
        ("training_authorized", False),
        ("formal_training_authorized", True),
    ],
)
def test_release_claim_gates_fail_closed(
    tmp_path: Path,
    claim: str,
    expected: bool,
) -> None:
    claims = _claims()
    claims[claim] = expected
    root, binding = _make_artifact(
        tmp_path / "artifact",
        manifest_update={"claims": claims},
        receipt_update={"claims": claims},
    )
    with pytest.raises(consumer.ConsumerPreflightError, match=f"manifest_{claim}"):
        _validate(root, binding)


def test_missing_and_extra_files_are_rejected(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "missing")
    (root / "train" / "chat.jsonl").unlink()
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_file_inventory_mismatch",
    ):
        _validate(root, binding)
    root, binding = _make_artifact(tmp_path / "extra")
    (root / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_file_inventory_mismatch",
    ):
        _validate(root, binding)


def test_binding_hash_and_tree_digest_drift_are_rejected(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "hash")
    (root / "train" / "chat.jsonl").write_bytes(b"changed")
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_binding_file_identity_drift",
    ):
        _validate(root, binding)
    root, binding = _make_artifact(tmp_path / "tree")
    binding["tree_digest_sha256"] = "f" * 64
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_digest_drift",
    ):
        _validate(root, binding)


def test_sidecar_content_is_verified_not_only_hashed(tmp_path: Path) -> None:
    root, _ = _make_artifact(tmp_path / "artifact")
    sidecar = root / "train" / "chat.jsonl.sha256"
    sidecar.write_bytes(f"{'f' * 64}  chat.jsonl\n".encode("ascii"))
    binding = _binding(root)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_sidecar_content_mismatch",
    ):
        _validate(root, binding)


def test_producer_commit_drift_is_rejected(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="producer_git_commit_drift",
    ):
        consumer.validate_artifact(
            root,
            binding,
            observed_producer_commit="2" * 40,
        )


def test_terminal_byte_drift_is_rejected(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")

    def mutate() -> None:
        _write_payload(root, "train/chat.jsonl", b"late mutation")

    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_two_snapshot_byte_drift",
    ):
        consumer.validate_artifact(
            root,
            binding,
            observed_producer_commit=PRODUCER_COMMIT,
            before_terminal_recheck=mutate,
        )


def test_binding_path_escape_and_duplicate_path_are_rejected(
    tmp_path: Path,
) -> None:
    root, binding = _make_artifact(tmp_path / "escape")
    binding["payload_files"][0]["path"] = "../escape.json"
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        _validate(root, binding)
    root, binding = _make_artifact(tmp_path / "duplicate")
    first = deepcopy(binding["payload_files"][0])
    first["sha256"] = "e" * 64
    binding["payload_files"][-1] = first
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_payload_path_duplicate",
    ):
        _validate(root, binding)


def test_symlink_is_rejected_when_platform_can_create_one(
    tmp_path: Path,
) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    target = tmp_path / "external"
    target.write_bytes(b"")
    path = root / "train" / "chat.jsonl"
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_symlink_or_reparse",
    ):
        _validate(root, binding)


def test_schema_manifest_inventory_and_aggregate_mismatches_fail(
    tmp_path: Path,
) -> None:
    root, binding = _make_artifact(
        tmp_path / "schema",
        manifest_update={"schema_version": "wrong"},
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_schema_version_mismatch",
    ):
        _validate(root, binding)
    counts = _counts()
    counts["records"] = 4299
    root, binding = _make_artifact(
        tmp_path / "counts",
        manifest_update={"counts": counts},
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_counts_mismatch",
    ):
        _validate(root, binding)
    root, _ = _make_artifact(tmp_path / "files")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    manifest["files"][0]["path"] = manifest["files"][1]["path"]
    _write_payload(root, "manifest.json", _json_bytes(manifest))
    receipt = json.loads((root / "build_receipt.json").read_text("utf-8"))
    raw = (root / "manifest.json").read_bytes()
    receipt["manifest"] = {
        "path": "manifest.json",
        "sha256": _sha(raw),
        "bytes": len(raw),
    }
    _write_payload(root, "build_receipt.json", _json_bytes(receipt))
    binding = _binding(root)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_file_path_duplicate_or_invalid",
    ):
        _validate(root, binding)


def test_role_split_language_and_arm_contracts_are_strict(tmp_path: Path) -> None:
    counts = _counts()
    counts["roles"]["tool_call"] = 1699
    root, binding = _make_artifact(
        tmp_path / "roles",
        manifest_update={"counts": counts},
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_counts_mismatch",
    ):
        _validate(root, binding)
    adapter = _adapter_contract()
    adapter["tool_arms"] = ["base", "tool_q_only"]
    root, binding = _make_artifact(
        tmp_path / "arms",
        manifest_update={"adapter_contract": adapter},
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_adapter_contract_mismatch",
    ):
        _validate(root, binding)


def test_receipt_release_audit_and_manifest_binding_are_strict(
    tmp_path: Path,
) -> None:
    root, binding = _make_artifact(
        tmp_path / "audit",
        receipt_update={
            "official_model_free_audits": {"independent_release_review": "pending"}
        },
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="build_receipt_independent_release_review_not_passed",
    ):
        _validate(root, binding)
    root, binding = _make_artifact(
        tmp_path / "manifest-binding",
        receipt_update={
            "manifest": {
                "path": "manifest.json",
                "sha256": "f" * 64,
                "bytes": 1,
            }
        },
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="build_receipt_manifest_binding_drift",
    ):
        _validate(root, binding)


def test_schema_is_valid_draft_2020_12_and_rejects_wrong_version() -> None:
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / consumer.SCHEMA_RELATIVE_PATH).read_text(
            "utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        consumer.validate_binding_document(
            {
                "schema_version": "wrong",
                "namespace": consumer.NAMESPACE,
            }
        )


def test_cli_dry_run_and_explicit_atomic_receipt(tmp_path: Path, capsys: Any) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(_json_bytes(binding))
    output = tmp_path / "receipts" / "preflight.json"
    output.parent.mkdir()
    assert (
        consumer.main(
            [
                "dry-run",
                "--artifact",
                str(root),
                "--binding",
                str(binding_path),
                "--producer-commit",
                PRODUCER_COMMIT,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text("utf-8"))
    assert stdout == written
    assert written["operation"] == "dry-run"
    assert written["claims"]["training_started"] is False
    assert str(root) not in output.read_text("utf-8")


def test_receipt_output_cannot_mutate_authenticated_tree(tmp_path: Path) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    receipt = _validate(root, binding)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="receipt_output_inside_authenticated_artifact",
    ):
        consumer._atomic_write_receipt(
            root / "receipt.json",
            receipt,
            artifact_root=root,
        )


def test_json_duplicate_and_nonfinite_values_are_rejected() -> None:
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="test_json_invalid",
    ):
        consumer._strict_json(b'{"a":1,"a":2}\n', "test")
    for raw in (
        b'{"a":NaN}\n',
        b'{"a":Infinity}\n',
        b'{"a":-Infinity}\n',
        b'{"a":1e999}\n',
        b'{"a":-1e999}\n',
        b'{"nested":[{"value":1e999}]}\n',
    ):
        with pytest.raises(
            consumer.ConsumerPreflightError,
            match="test_json_invalid",
        ):
            consumer._strict_json(raw, "test")


def test_no_artifact_or_receipt_is_written_without_explicit_output(
    tmp_path: Path,
    capsys: Any,
) -> None:
    root, binding = _make_artifact(tmp_path / "artifact")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(_json_bytes(binding))
    before = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
    assert (
        consumer.main(
            [
                "validate",
                "--artifact",
                str(root),
                "--binding",
                str(binding_path),
                "--producer-commit",
                PRODUCER_COMMIT,
            ]
        )
        == 0
    )
    capsys.readouterr()
    after = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
    assert before == after
    assert len(after) == 44


def test_artifact_tree_has_exactly_22_payloads_and_22_sidecars() -> None:
    assert len(consumer.PAYLOAD_PATHS) == 22
    assert len(set(consumer.PAYLOAD_PATHS)) == 22
    assert len(consumer.EXACT_TREE_PATHS) == 44
    assert all(not Path(path).is_absolute() for path in consumer.PAYLOAD_PATHS)
    assert all(".." not in Path(path).parts for path in consumer.PAYLOAD_PATHS)


SHARDED_ARTIFACT_VERSION = consumer.SHARDED_FINAL_ARTIFACT_VERSION
SHARDED_MANIFEST_VERSION = consumer.SHARDED_FINAL_MANIFEST_VERSION
SHARDED_BUILD_RECEIPT_VERSION = consumer.SHARDED_FINAL_BUILD_RECEIPT_VERSION
SHARDED_ATTESTATION_VERSION = consumer.SHARDED_FINAL_ATTESTATION_VERSION
SHARDED_STATUS = "candidate_pending_independent_release_review"
SHARDED_SCHEMA_PATHS = {
    name: str(anchor["path"])
    for name, anchor in consumer.SHARDED_FINAL_SCHEMA_ANCHORS.items()
}
SHARDED_ATTESTATION_PATH = "release/independent_release_attestation.json"
FINAL_SHARDED_FILE_METADATA = (
    (
        "components/planner_eval/build_receipt.json",
        "56df42d17c0fec48ef6e52aeeffdd6cb0fa324622ed8156447b77df5b1fd10d8",
        3798,
    ),
    (
        "components/planner_eval/build_receipt.json.sha256",
        "78a278a8aa89bf5966a24fe91139ec6d136fb3381824d26db16c3819613ce779",
        85,
    ),
    (
        "components/planner_eval/manifest.json",
        "f6cc5f4b5f63bc174bd65c71e6c0d6b224dcf9e8a90ece687fc38b5d1b7f0c05",
        25370,
    ),
    (
        "components/planner_eval/manifest.json.sha256",
        "2e21bdd4a35ec408cf67ad3beb32b78511345c49c583d4018aadc6bbeb91e2af",
        80,
    ),
    (
        "components/planner_eval/negative_inventory.jsonl",
        "61ec3954a2758dc5aa3c383fbd712adce85008ba359613fbf93bfc76a2cb4043",
        17281,
    ),
    (
        "components/planner_eval/negative_inventory.jsonl.sha256",
        "a987669cdd3c1eca85deab39c880625097735895d302a6a1dd603634cab3fab1",
        91,
    ),
    (
        "components/planner_eval/records.jsonl",
        "591843636062b2e6915b2c4e2aafbb9dcb76b699c6dcbc95bb00f188a1ab3734",
        2480914,
    ),
    (
        "components/planner_eval/records.jsonl.sha256",
        "25e7c532fdbd3c31640061dd8e30b269bdb340e9dbfe549a062654264703b0b6",
        80,
    ),
    (
        "components/planner_eval/token_inventory.jsonl",
        "3d53386ae920c8474e189c5da92adf46e214b36ad8a960cb3cc5b1eb0449432d",
        288220,
    ),
    (
        "components/planner_eval/token_inventory.jsonl.sha256",
        "945a4560af23379af0fd9e1a2c2028422d7f36bb218b4e297eb699277b5d4bbf",
        88,
    ),
    (
        "components/router/build_receipt.json",
        "8b409c139dad499d5c06663bb7b9712a379a32f319a009ef7f6f3c800618966d",
        1498,
    ),
    (
        "components/router/build_receipt.json.sha256",
        "fa59b127eab07f0165509271def17793a32601bcc3d052666dcc61c26da33605",
        85,
    ),
    (
        "components/router/manifest.json",
        "416e39b0617263edc3af7046ab075745dbe564430927565a1623540d63027b6d",
        30839,
    ),
    (
        "components/router/manifest.json.sha256",
        "a1fba4dc0176636cd176e2cafe623bdc6421ce596433f8ae20d2a02e1f734f60",
        80,
    ),
    (
        "components/router/records.jsonl",
        "f41b865b4c65cf849c92499a0ea19ea53f2ce37f7424fa8e154543bff81f61b8",
        938700,
    ),
    (
        "components/router/records.jsonl.sha256",
        "2a16d6e1dabf5f0d98d51a1235d1e6681c8124bcfe93a310afb82dafa204bd05",
        80,
    ),
    (
        "components/router/token_inventory.jsonl",
        "67fe668b21ea935e5143362d87ab213de5b8a4c971de3c4cb836b353fcf778a3",
        91514,
    ),
    (
        "components/router/token_inventory.jsonl.sha256",
        "78b791611405f8f3c1589b5a5a18cfffe0ff69d8ca4b10c8b11ca2a5c49d8587",
        88,
    ),
    (
        "components/tool_eval/build_receipt.json",
        "464d6475069b4ebf2fb34f2c27d87d90c74ad3deb6118480ca1e68835cdcbcfa",
        3676,
    ),
    (
        "components/tool_eval/build_receipt.json.sha256",
        "c2e97ab8c29425d64b74e0980a989a488b1ecb7fa12792588310cebad4ed090c",
        85,
    ),
    (
        "components/tool_eval/manifest.json",
        "ece2b2e998c44037eb7af10c6c68193aba9037035169358855b5cfa1071935f1",
        27925,
    ),
    (
        "components/tool_eval/manifest.json.sha256",
        "46658330eecdbdb3225eb860aa7d3ddc195ebbcf0401be2a628d702c0bebd79c",
        80,
    ),
    (
        "components/tool_eval/negative_inventory.jsonl",
        "899b83076340ea8d496f5a2c8a3b355337de9add744c5a7ccb9406ac1622c364",
        18680,
    ),
    (
        "components/tool_eval/negative_inventory.jsonl.sha256",
        "fae6fcc0307dd9139a877b16d581af65f4dd5d3a0f4d07e75d814dbc992b7ed2",
        91,
    ),
    (
        "components/tool_eval/records.jsonl",
        "138a3b6b915250db0d793fb36745586e95f17881f52e8882e6d4f0df691057fe",
        4258978,
    ),
    (
        "components/tool_eval/records.jsonl.sha256",
        "673ba9282adf1c27015d95cd8c76d1a2ab5dafb6139d20ae0384b01dfd4f7f1b",
        80,
    ),
    (
        "components/tool_eval/token_inventory.jsonl",
        "ff0d918c9532b0b2a9ff392b783c63c410c94d65cdb77ae5c57af48bce5a467c",
        506080,
    ),
    (
        "components/tool_eval/token_inventory.jsonl.sha256",
        "bf5d1410afcb69736e81cee548a0b22b186d6cffebd556d632f0838cd6f58b30",
        88,
    ),
    (
        "eval_proxy/chat.jsonl",
        "42798778a2c1398393bb0b6588cfd5d1cf5f3fc67ae008e9e20aea0e33bae176",
        14873286,
    ),
    (
        "eval_proxy/chat.jsonl.sha256",
        "78523732fe7361f071aab7c0b5d6472a84b52026ddf87b181bee3d4d2939ff3e",
        77,
    ),
    (
        "identity_eval/probe_inventory.jsonl",
        "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5",
        51145,
    ),
    (
        "identity_eval/probe_inventory.jsonl.sha256",
        "2bb0bc50793cff2f77caa7c2ccfff7a9d0fb8805e2f0d15a695894bab7806143",
        88,
    ),
    (
        "reviews/planner_eval_independent_review.json",
        "c77fcd309af64eee689148c396d612b1b3ad20414f71482d7c9c27751d33c57c",
        2203,
    ),
    (
        "reviews/planner_eval_independent_review.json.sha256",
        "b8d3cc8b75b575a25534c41c173274215432ff5cf6bf949f6119b4bbbab3555b",
        103,
    ),
    (
        "reviews/tool_eval_independent_review.json",
        "a09ac7dc7643168342360d2bc0cd463238fd43405d5365c6efc06373f76a6082",
        2477,
    ),
    (
        "reviews/tool_eval_independent_review.json.sha256",
        "4f7e9c280aac50a4bf685176f64803fc69f5cf7c5b4967aeff0db91f8e88ed77",
        100,
    ),
    (
        "serialization_inventory.jsonl",
        "4b7a35067ab4a0acdbda3095055f20a21b8c24b64f12a70d07aaa9a48afa0f1d",
        4251182,
    ),
    (
        "serialization_inventory.jsonl.sha256",
        "715285e1f884b9e0452038bd12a7dcac4928a013e93834306302a0c25ba39e71",
        96,
    ),
    (
        "train/chat-00000-of-00002.jsonl",
        "9c2fe5ba2b199be12f11f9d9e1a76c2d8de1474732997cb8d30b984f2477b215",
        29909786,
    ),
    (
        "train/chat-00000-of-00002.jsonl.sha256",
        "8a0352927bd15767df11491743134b9e4c530ea58a96472312ba50a803941b20",
        92,
    ),
    (
        "train/chat-00001-of-00002.jsonl",
        "1d8facb574a7341d1b43933da30f8ea2c0f977bec5920a83661c0619cb463522",
        29935528,
    ),
    (
        "train/chat-00001-of-00002.jsonl.sha256",
        "562c474ff3cd763f33cab179c66785f215763a5771109b169d10d73b6bee271f",
        92,
    ),
)


def _git_run(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _closed_schema(name: str, required: list[str]) -> dict[str, object]:
    anchor = consumer.SHARDED_FINAL_SCHEMA_ANCHORS[name]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": anchor["schema_id"],
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            key: (
                {"const": anchor["schema_version"]} if key == "schema_version" else {}
            )
            for key in required
        },
    }


def _write_text(root: Path, relative: str, raw: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _sharded_manifest_entries(
    root: Path,
    shard_records: dict[str, int],
) -> list[dict[str, object]]:
    entries = []
    for relative in sorted(consumer.SHARDED_MANIFEST_LISTED_PATHS):
        raw = root.joinpath(*relative.split("/")).read_bytes()
        records = shard_records.get(
            relative,
            consumer.SHARDED_RECORD_COUNTS.get(relative),
        )
        entries.append(
            {
                "path": relative,
                "sha256": _sha(raw),
                "bytes": len(raw),
                "kind": consumer._kind(relative),
                "records": records,
            }
        )
    return entries


def _raw_output_inventory(root: Path) -> str:
    entries = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix()
        not in {"build_receipt.json", "build_receipt.json.sha256"}
    ):
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(raw),
                "bytes": len(raw),
            }
        )
    return consumer._domain_hash(consumer.PRODUCER_OUTPUT_INVENTORY_DOMAIN, entries)


def _logical_identity(train_raw: bytes, artifact: Path) -> dict[str, str]:
    result = {
        key: _sha(f"synthetic:{key}".encode("ascii"))
        for key in consumer.LOGICAL_IDENTITY_KEYS
    }
    result["logical_train_partition_sha256"] = _sha(train_raw)
    result["eval_proxy_partition_sha256"] = _sha(
        (artifact / "eval_proxy" / "chat.jsonl").read_bytes()
    )
    result["serialization_inventory_sha256"] = _sha(
        (artifact / "serialization_inventory.jsonl").read_bytes()
    )
    result["identity_probe_inventory_sha256"] = _sha(
        (artifact / "identity_eval" / "probe_inventory.jsonl").read_bytes()
    )
    return result


def _synthetic_training_shards(artifact: Path) -> tuple[dict[str, object], bytes]:
    shard_raw = [b'{"synthetic":0}\n', b'{"synthetic":1}\n']
    for relative, raw in zip(consumer.TRAIN_SHARD_PATHS, shard_raw, strict=True):
        _write_payload(artifact, relative, raw)
    logical_raw = b"".join(shard_raw)
    shards = [
        {
            "index": index,
            "path": relative,
            "sha256": _sha(raw),
            "bytes": len(raw),
            "records": 1720,
            "task_bundles": 680,
            "first_task_bundle_sha256": _sha(f"first:{index}".encode("ascii")),
            "last_task_bundle_sha256": _sha(f"last:{index}".encode("ascii")),
        }
        for index, (relative, raw) in enumerate(
            zip(consumer.TRAIN_SHARD_PATHS, shard_raw, strict=True)
        )
    ]
    contract: dict[str, object] = {
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "shards": shards,
        "ordered_concat_sha256": _sha(logical_raw),
        "logical_partition_bytes": len(logical_raw),
        "logical_partition_records": 3440,
        "task_bundles": 1360,
        "split_boundary": "contiguous_task_bundle",
        "ordered_shards": True,
        "task_bundle_intersection_empty": True,
    }
    contract["boundary_commitment_sha256"] = consumer._training_boundary_commitment(
        contract
    )
    return contract, logical_raw


def _make_sharded_repo(
    root: Path,
) -> tuple[Path, Path, dict[str, object], str]:
    root.mkdir()
    _git_run(root, "init", "-b", "main")
    _git_run(root, "config", "user.email", "synthetic@example.invalid")
    _git_run(root, "config", "user.name", "Synthetic Fixture")
    _git_run(root, "config", "core.autocrlf", "false")
    _write_text(root, ".gitattributes", b"* text eol=lf\n")
    schema_required = {
        "manifest": [
            "schema_version",
            "status",
            "namespace",
            "artifact_version",
            "canonical_path",
            "training_shards",
            "logical_identity",
            "files",
            "counts",
            "identity",
            "review",
            "components",
            "adapter_contract",
            "kv_contract",
            "integrity",
            "resource_counters",
            "claims",
            "release_review",
        ],
        "build_receipt": [
            "schema_version",
            "status",
            "artifact_version",
            "manifest",
            "training_shards",
            "logical_identity",
            "output_inventory_sha256",
            "input_snapshot_inventory",
            "input_snapshot_inventory_sha256",
            "official_model_free_audits",
            "resource_counters",
            "claims",
        ],
        "release_attestation": [
            "schema_version",
            "audit_status",
            "artifact_lifecycle_status",
            "namespace",
            "artifact_version",
            "canonical_artifact_path",
            "producer_candidate_commit",
            "reviewer_separation",
            "findings",
            "artifact",
            "training_shards",
            "logical_identity",
            "git_authentication",
            "integrity",
            "resource_counters",
            "claims",
        ],
    }
    for name, relative in SHARDED_SCHEMA_PATHS.items():
        _write_text(
            root,
            relative,
            _json_bytes(_closed_schema(name, schema_required[name])),
        )
    artifact = root / "artifact"
    artifact.mkdir()
    training_shards, logical_train_raw = _synthetic_training_shards(artifact)
    for relative in consumer.SHARDED_PAYLOAD_PATHS:
        if relative in {
            *consumer.TRAIN_SHARD_PATHS,
            "manifest.json",
            "build_receipt.json",
        }:
            continue
        raw = _json_bytes({}) if relative.endswith(".json") else b"{}\n"
        _write_payload(artifact, relative, raw)
    logical = _logical_identity(logical_train_raw, artifact)
    manifest_shards = {
        "partition": "train",
        **deepcopy(training_shards),
        "split_target": "synthetic_stable_boundary",
        "logical_partition_sha256": logical["logical_train_partition_sha256"],
        "max_file_bytes_exclusive": consumer.MAX_PAYLOAD_BYTES + 1,
        "all_shards_below_limit": True,
    }
    manifest: dict[str, object] = {
        "schema_version": SHARDED_MANIFEST_VERSION,
        "status": SHARDED_STATUS,
        "namespace": consumer.NAMESPACE,
        "artifact_version": SHARDED_ARTIFACT_VERSION,
        "canonical_path": "artifact",
        "training_shards": manifest_shards,
        "logical_identity": logical,
        "files": _sharded_manifest_entries(
            artifact,
            {
                consumer.TRAIN_SHARD_PATHS[0]: 1720,
                consumer.TRAIN_SHARD_PATHS[1]: 1720,
            },
        ),
        "counts": _counts(),
        "identity": _identity(),
        "review": _review(),
        "components": _components(artifact),
        "adapter_contract": _adapter_contract(),
        "kv_contract": _kv_contract(),
        "integrity": {"synthetic": True},
        "resource_counters": {**consumer._ZERO_RESOURCES, "luna_requests": 0},
        "claims": {
            "candidate": True,
            "final": False,
            "diagnostic_only": True,
            "proxy_only": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "release_authorized": False,
        },
        "release_review": {
            "implementer_release_signoff": False,
            "independent_release_review_required": True,
        },
    }
    _write_payload(artifact, "manifest.json", _json_bytes(manifest))
    read_set_paths = [".gitattributes", *SHARDED_SCHEMA_PATHS.values()]
    input_inventory = []
    for index, relative in enumerate(sorted(read_set_paths)):
        raw = root.joinpath(*relative.split("/")).read_bytes()
        input_inventory.append(
            {
                "label": f"source:{index:02d}",
                "path": relative,
                "sha256": _sha(raw),
                "bytes": len(raw),
                "source_class": "producer_source",
            }
        )
    input_digest = consumer._domain_hash(
        consumer.PRODUCER_INPUT_INVENTORY_DOMAIN,
        input_inventory,
    )
    build_receipt: dict[str, object] = {
        "schema_version": SHARDED_BUILD_RECEIPT_VERSION,
        "status": SHARDED_STATUS,
        "artifact_version": SHARDED_ARTIFACT_VERSION,
        "manifest": {
            "path": "manifest.json",
            "sha256": _sha((artifact / "manifest.json").read_bytes()),
            "bytes": (artifact / "manifest.json").stat().st_size,
        },
        "training_shards": {
            key: deepcopy(manifest_shards[key])
            for key in (
                "shard_count",
                "ordered_paths",
                "ordered_concat_sha256",
                "logical_partition_sha256",
                "logical_partition_bytes",
                "logical_partition_records",
                "max_file_bytes_exclusive",
            )
        },
        "logical_identity": logical,
        "output_inventory_sha256": _raw_output_inventory(artifact),
        "input_snapshot_inventory": input_inventory,
        "input_snapshot_inventory_sha256": input_digest,
        "official_model_free_audits": {"independent_release_review": "pending"},
        "resource_counters": {**consumer._ZERO_RESOURCES, "luna_requests": 0},
        "claims": deepcopy(manifest["claims"]),
    }
    _write_payload(artifact, "build_receipt.json", _json_bytes(build_receipt))
    _git_run(root, "add", "--all")
    _git_run(root, "commit", "-m", "synthetic candidate")
    candidate_commit = _git_run(root, "rev-parse", "HEAD")
    artifact_snapshot = consumer._snapshot_tree(
        artifact,
        capture_metadata=True,
        exact_tree_paths=consumer.SHARDED_EXACT_TREE_PATHS,
    )
    attestation_shards = {
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "shards": [
            {
                key: item[key]
                for key in (
                    "index",
                    "path",
                    "sha256",
                    "bytes",
                    "records",
                    "task_bundles",
                )
            }
            for item in training_shards["shards"]
        ],
        "ordered_concat_sha256": training_shards["ordered_concat_sha256"],
        "logical_partition_bytes": training_shards["logical_partition_bytes"],
        "logical_partition_records": 3440,
        "bundle_boundary_only": True,
    }
    attestation: dict[str, object] = {
        "schema_version": SHARDED_ATTESTATION_VERSION,
        "audit_status": "passed",
        "artifact_lifecycle_status": SHARDED_STATUS,
        "namespace": consumer.NAMESPACE,
        "artifact_version": SHARDED_ARTIFACT_VERSION,
        "canonical_artifact_path": "artifact",
        "producer_candidate_commit": candidate_commit,
        "reviewer_separation": {
            "producer_id": "synthetic.producer",
            "reviewer_id": "synthetic.reviewer",
            "different_reviewer": True,
        },
        "findings": {"p0": 0, "p1": 0, "p2": 0},
        "artifact": {
            "files": 46,
            "tree_sha256": consumer._producer_tree_digest(artifact_snapshot.files),
            "manifest_sha256": artifact_snapshot.files["manifest.json"].sha256,
            "manifest_sidecar_physical_sha256": artifact_snapshot.files[
                "manifest.json.sha256"
            ].sha256,
            "build_receipt_sha256": artifact_snapshot.files[
                "build_receipt.json"
            ].sha256,
            "build_receipt_sidecar_physical_sha256": artifact_snapshot.files[
                "build_receipt.json.sha256"
            ].sha256,
            "output_inventory_sha256": build_receipt["output_inventory_sha256"],
            "all_payload_sidecars_valid": True,
            "all_files_below_50_mib": True,
        },
        "training_shards": attestation_shards,
        "logical_identity": logical,
        "git_authentication": {
            "commit_resolved_exactly": True,
            "replace_refs_absent": True,
            "grafts_absent": True,
            "tracked_inventory_exact": True,
            "commit_blobs_equal_physical_bytes": True,
            "index_blobs_equal_physical_bytes": True,
            "scoped_checkout_clean": True,
            "lf_attributes_exact": True,
        },
        "integrity": {"synthetic": True},
        "resource_counters": {**consumer._ZERO_RESOURCES, "luna_requests": 0},
        "claims": {
            "artifact_final_identity": True,
            "independent_review_passed": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "model_release_authorized": False,
        },
    }
    _write_payload(root, SHARDED_ATTESTATION_PATH, _json_bytes(attestation))
    _git_run(root, "add", "--all")
    _git_run(root, "commit", "-m", "synthetic release attestation")
    release_commit = _git_run(root, "rev-parse", "HEAD")
    _git_run(root, "update-ref", "refs/remotes/origin/main", release_commit)
    _git_run(root, "config", "remote.origin.url", ".")
    _git_run(
        root,
        "config",
        "remote.origin.fetch",
        "+refs/heads/*:refs/remotes/origin/*",
    )
    _git_run(root, "config", "branch.main.remote", "origin")
    _git_run(root, "config", "branch.main.merge", "refs/heads/main")
    read_set_files = []
    for relative in sorted(read_set_paths):
        raw = root.joinpath(*relative.split("/")).read_bytes()
        read_set_files.append(
            {
                "path": relative,
                "sha256": _sha(raw),
                "bytes": len(raw),
                "kind": consumer._source_kind(relative),
            }
        )
    binding_snapshot = consumer._snapshot_tree(
        artifact,
        capture_metadata=True,
        exact_tree_paths=consumer.SHARDED_EXACT_TREE_PATHS,
    )
    payload_pins = []
    for relative in consumer.SHARDED_PAYLOAD_PATHS:
        payload = binding_snapshot.files[relative]
        sidecar = binding_snapshot.files[f"{relative}.sha256"]
        payload_pins.append(
            {
                "path": relative,
                "kind": "jsonl" if relative.endswith(".jsonl") else "json",
                "sha256": payload.sha256,
                "bytes": payload.bytes,
                "sidecar_sha256": sidecar.sha256,
                "sidecar_bytes": sidecar.bytes,
            }
        )
    attestation_raw = root.joinpath(*SHARDED_ATTESTATION_PATH.split("/")).read_bytes()
    attestation_sidecar_path = f"{SHARDED_ATTESTATION_PATH}.sha256"
    attestation_sidecar_raw = root.joinpath(
        *attestation_sidecar_path.split("/")
    ).read_bytes()
    schema_files = {
        name: next(
            deepcopy(item) for item in read_set_files if item["path"] == relative
        )
        for name, relative in SHARDED_SCHEMA_PATHS.items()
    }
    binding: dict[str, object] = {
        "schema_version": consumer.SHARDED_BINDING_VERSION,
        "namespace": consumer.NAMESPACE,
        "artifact_version": SHARDED_ARTIFACT_VERSION,
        "expected_manifest_schema_version": SHARDED_MANIFEST_VERSION,
        "expected_build_receipt_schema_version": SHARDED_BUILD_RECEIPT_VERSION,
        "expected_attestation_schema_version": SHARDED_ATTESTATION_VERSION,
        "expected_manifest_status": SHARDED_STATUS,
        "producer_git": {
            "candidate_commit": candidate_commit,
            "release_commit": release_commit,
            "release_tree": _git_run(root, "rev-parse", "HEAD^{tree}"),
            "upstream_commit": release_commit,
            "live_remote_commit": release_commit,
            "clean_worktree": True,
            "tags_at_head": 0,
        },
        "producer_output_inventory_sha256": build_receipt["output_inventory_sha256"],
        "tree_digest_domain": consumer.PRODUCER_TREE_DIGEST_DOMAIN,
        "tree_digest_sha256": consumer._producer_tree_digest(binding_snapshot.files),
        "payload_files": payload_pins,
        "schema_files": schema_files,
        "release_attestation": {
            "path": SHARDED_ATTESTATION_PATH,
            "kind": "json",
            "sha256": _sha(attestation_raw),
            "bytes": len(attestation_raw),
            "sidecar_path": attestation_sidecar_path,
            "sidecar_sha256": _sha(attestation_sidecar_raw),
            "sidecar_bytes": len(attestation_sidecar_raw),
        },
        "read_set": {
            "count": len(read_set_files),
            "canonical_digest_sha256": consumer._read_set_digest(read_set_files),
            "producer_inventory_sha256": input_digest,
            "files": read_set_files,
        },
        "logical_identity": logical,
        "training_shards": training_shards,
    }
    return root, artifact, binding, release_commit


def _validate_sharded(
    root: Path,
    artifact: Path,
    binding: dict[str, object],
    release_commit: str,
    *,
    before_terminal_recheck: Any | None = None,
) -> dict[str, Any]:
    return consumer.validate_artifact(
        artifact,
        binding,
        observed_producer_commit=release_commit,
        producer_repository_root=root,
        observed_live_remote_commit=release_commit,
        before_terminal_recheck=before_terminal_recheck,
    )


def _exact_final_trust_binding(binding: dict[str, object]) -> dict[str, object]:
    result = deepcopy(binding)
    result["artifact_version"] = consumer.SHARDED_FINAL_ARTIFACT_VERSION
    result["expected_manifest_schema_version"] = consumer.SHARDED_FINAL_MANIFEST_VERSION
    result["expected_build_receipt_schema_version"] = (
        consumer.SHARDED_FINAL_BUILD_RECEIPT_VERSION
    )
    result["expected_attestation_schema_version"] = (
        consumer.SHARDED_FINAL_ATTESTATION_VERSION
    )
    result["expected_manifest_status"] = consumer.SHARDED_FINAL_STATUS
    result["producer_git"] = {
        "candidate_commit": consumer.SHARDED_FINAL_CANDIDATE_COMMIT,
        "release_commit": consumer.SHARDED_FINAL_RELEASE_COMMIT,
        "release_tree": consumer.SHARDED_FINAL_RELEASE_TREE,
        "upstream_commit": consumer.SHARDED_FINAL_RELEASE_COMMIT,
        "live_remote_commit": consumer.SHARDED_FINAL_RELEASE_COMMIT,
        "clean_worktree": True,
        "tags_at_head": 0,
    }
    result["tree_digest_sha256"] = consumer.SHARDED_FINAL_TREE_SHA256
    result["tree_digest_domain"] = consumer.PRODUCER_TREE_DIGEST_DOMAIN
    result["producer_output_inventory_sha256"] = (
        consumer.SHARDED_FINAL_OUTPUT_INVENTORY_SHA256
    )
    result["logical_identity"] = dict(consumer.SHARDED_FINAL_LOGICAL_IDENTITY)
    result["training_shards"] = {
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "shards": [dict(item) for item in consumer.SHARDED_FINAL_SHARDS],
        "ordered_concat_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": 3_440,
        "task_bundles": 1_360,
        "split_boundary": "contiguous_task_bundle",
        "ordered_shards": True,
        "task_bundle_intersection_empty": True,
        "boundary_commitment_sha256": (
            consumer.SHARDED_FINAL_BOUNDARY_COMMITMENT_SHA256
        ),
    }
    pins = {str(item["path"]): item for item in result["payload_files"]}
    pins["manifest.json"].update(
        {
            "sha256": consumer.SHARDED_FINAL_MANIFEST_SHA256,
            "bytes": consumer.SHARDED_FINAL_MANIFEST_BYTES,
        }
    )
    pins["build_receipt.json"].update(
        {
            "sha256": consumer.SHARDED_FINAL_BUILD_RECEIPT_SHA256,
            "bytes": consumer.SHARDED_FINAL_BUILD_RECEIPT_BYTES,
        }
    )
    result["schema_files"] = {
        name: {
            "path": anchor["path"],
            "sha256": anchor["sha256"],
            "bytes": anchor["bytes"],
            "kind": "json_schema",
        }
        for name, anchor in consumer.SHARDED_FINAL_SCHEMA_ANCHORS.items()
    }
    result["release_attestation"] = {
        "path": consumer.SHARDED_FINAL_ATTESTATION_PATH,
        "kind": "json",
        "sha256": consumer.SHARDED_FINAL_ATTESTATION_SHA256,
        "bytes": consumer.SHARDED_FINAL_ATTESTATION_BYTES,
        "sidecar_path": f"{consumer.SHARDED_FINAL_ATTESTATION_PATH}.sha256",
        "sidecar_sha256": consumer.SHARDED_FINAL_ATTESTATION_SIDECAR_SHA256,
        "sidecar_bytes": consumer.SHARDED_FINAL_ATTESTATION_SIDECAR_BYTES,
    }
    result["read_set"]["producer_inventory_sha256"] = (
        consumer.SHARDED_FINAL_INPUT_INVENTORY_SHA256
    )
    return result


def _metadata_snapshot(
    root: Path,
    relative: str,
    sha256: str,
    byte_count: int,
    *,
    data: bytes | None,
    inode: int,
) -> consumer.FileSnapshot:
    return consumer.FileSnapshot(
        path=root.joinpath(*relative.split("/")),
        sha256=sha256,
        bytes=byte_count,
        stat_signature=(1, inode, byte_count, 1, 1),
        data=data,
        utf8_valid=True,
        lf_only_text=True,
    )


def _exact_final_metadata_chain(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, object],
    consumer.TreeSnapshot,
    dict[str, consumer.FileSnapshot],
    str,
    str,
]:
    root, artifact, seed_binding, _ = _make_sharded_repo(tmp_path / "seed")
    binding = _exact_final_trust_binding(seed_binding)
    read_entries = {str(item["path"]): item for item in binding["read_set"]["files"]}
    for name, anchor in consumer.SHARDED_FINAL_SCHEMA_ANCHORS.items():
        read_entries[str(anchor["path"])].update(
            {
                "sha256": anchor["sha256"],
                "bytes": anchor["bytes"],
                "kind": "json_schema",
            }
        )
    binding["read_set"]["canonical_digest_sha256"] = consumer._read_set_digest(
        binding["read_set"]["files"]
    )
    binding["read_set"]["producer_inventory_sha256"] = (
        consumer.SHARDED_FINAL_INPUT_INVENTORY_SHA256
    )
    metadata = {
        path: {"sha256": sha256, "bytes": byte_count}
        for path, sha256, byte_count in FINAL_SHARDED_FILE_METADATA
    }
    metadata.update(
        {
            "manifest.json": {
                "sha256": consumer.SHARDED_FINAL_MANIFEST_SHA256,
                "bytes": consumer.SHARDED_FINAL_MANIFEST_BYTES,
            },
            "manifest.json.sha256": {
                "sha256": (
                    "b8b2eee6cf89d0c93d9901d5702d3c143627639e2007ca55c98390d50c4f2fca"
                ),
                "bytes": 80,
            },
            "build_receipt.json": {
                "sha256": consumer.SHARDED_FINAL_BUILD_RECEIPT_SHA256,
                "bytes": consumer.SHARDED_FINAL_BUILD_RECEIPT_BYTES,
            },
            "build_receipt.json.sha256": {
                "sha256": (
                    "301e2bef647a92b88399c13e2148572257789f3a4231dc42d0aa5f20a9bccc92"
                ),
                "bytes": 85,
            },
        }
    )
    assert set(metadata) == consumer.SHARDED_EXACT_TREE_PATHS
    manifest = json.loads((artifact / "manifest.json").read_text("utf-8"))
    manifest["schema_version"] = consumer.SHARDED_FINAL_MANIFEST_VERSION
    manifest["artifact_version"] = consumer.SHARDED_FINAL_ARTIFACT_VERSION
    manifest["status"] = consumer.SHARDED_FINAL_STATUS
    manifest["logical_identity"] = dict(consumer.SHARDED_FINAL_LOGICAL_IDENTITY)
    manifest["training_shards"] = {
        "partition": "train",
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "shards": [dict(item) for item in consumer.SHARDED_FINAL_SHARDS],
        "ordered_concat_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": 3_440,
        "max_file_bytes_exclusive": consumer.MAX_PAYLOAD_BYTES + 1,
        "all_shards_below_limit": True,
        "task_bundles": 1_360,
        "split_boundary": "contiguous_task_bundle",
        "split_target": "nearest_half_bytes_with_stable_earliest_tie_break",
    }
    shard_records = {
        str(item["path"]): int(item["records"])
        for item in consumer.SHARDED_FINAL_SHARDS
    }
    manifest["files"] = [
        {
            "path": path,
            "sha256": metadata[path]["sha256"],
            "bytes": metadata[path]["bytes"],
            "kind": consumer._kind(path),
            "records": shard_records.get(
                path,
                consumer.SHARDED_RECORD_COUNTS.get(path),
            ),
        }
        for path in sorted(consumer.SHARDED_MANIFEST_LISTED_PATHS)
    ]
    for name in ("router", "tool_eval", "planner_eval"):
        component = manifest["components"][name]
        component["manifest_sha256"] = metadata[f"components/{name}/manifest.json"][
            "sha256"
        ]
        component["records_sha256"] = metadata[f"components/{name}/records.jsonl"][
            "sha256"
        ]
        component["token_inventory_sha256"] = metadata[
            f"components/{name}/token_inventory.jsonl"
        ]["sha256"]
        component["build_receipt_sha256"] = metadata[
            f"components/{name}/build_receipt.json"
        ]["sha256"]
        if name != "router":
            component["independent_review_receipt_sha256"] = metadata[
                f"reviews/{name}_independent_review.json"
            ]["sha256"]
    build_receipt = json.loads((artifact / "build_receipt.json").read_text("utf-8"))
    build_receipt["schema_version"] = consumer.SHARDED_FINAL_BUILD_RECEIPT_VERSION
    build_receipt["artifact_version"] = consumer.SHARDED_FINAL_ARTIFACT_VERSION
    build_receipt["status"] = consumer.SHARDED_FINAL_STATUS
    build_receipt["manifest"] = {
        "path": "manifest.json",
        "sha256": consumer.SHARDED_FINAL_MANIFEST_SHA256,
        "bytes": consumer.SHARDED_FINAL_MANIFEST_BYTES,
    }
    build_receipt["logical_identity"] = dict(consumer.SHARDED_FINAL_LOGICAL_IDENTITY)
    build_receipt["training_shards"] = {
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "ordered_concat_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": 3_440,
        "max_file_bytes_exclusive": consumer.MAX_PAYLOAD_BYTES + 1,
    }
    build_receipt["output_inventory_sha256"] = (
        consumer.SHARDED_FINAL_OUTPUT_INVENTORY_SHA256
    )
    for entry in build_receipt["input_snapshot_inventory"]:
        pin = read_entries[str(entry["path"])]
        entry["sha256"] = pin["sha256"]
        entry["bytes"] = pin["bytes"]
    build_receipt["input_snapshot_inventory_sha256"] = (
        consumer.SHARDED_FINAL_INPUT_INVENTORY_SHA256
    )
    attestation = json.loads(
        root.joinpath(*SHARDED_ATTESTATION_PATH.split("/")).read_text("utf-8")
    )
    attestation.update(
        {
            "schema_version": consumer.SHARDED_FINAL_ATTESTATION_VERSION,
            "artifact_version": consumer.SHARDED_FINAL_ARTIFACT_VERSION,
            "artifact_lifecycle_status": consumer.SHARDED_FINAL_STATUS,
            "canonical_artifact_path": "artifact",
            "producer_candidate_commit": consumer.SHARDED_FINAL_CANDIDATE_COMMIT,
            "logical_identity": dict(consumer.SHARDED_FINAL_LOGICAL_IDENTITY),
        }
    )
    attestation["artifact"].update(
        {
            "tree_sha256": consumer.SHARDED_FINAL_TREE_SHA256,
            "manifest_sha256": consumer.SHARDED_FINAL_MANIFEST_SHA256,
            "manifest_sidecar_physical_sha256": metadata["manifest.json.sha256"][
                "sha256"
            ],
            "build_receipt_sha256": consumer.SHARDED_FINAL_BUILD_RECEIPT_SHA256,
            "build_receipt_sidecar_physical_sha256": metadata[
                "build_receipt.json.sha256"
            ]["sha256"],
            "output_inventory_sha256": (consumer.SHARDED_FINAL_OUTPUT_INVENTORY_SHA256),
        }
    )
    attestation["training_shards"] = {
        "shard_count": 2,
        "ordered_paths": list(consumer.TRAIN_SHARD_PATHS),
        "shards": [
            {
                key: item[key]
                for key in (
                    "index",
                    "path",
                    "sha256",
                    "bytes",
                    "records",
                    "task_bundles",
                )
            }
            for item in consumer.SHARDED_FINAL_SHARDS
        ],
        "ordered_concat_sha256": consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": 3_440,
        "bundle_boundary_only": True,
    }
    data_by_path: dict[str, bytes | None] = {
        "manifest.json": _json_bytes(manifest),
        "build_receipt.json": _json_bytes(build_receipt),
    }
    for payload_path in consumer.SHARDED_PAYLOAD_PATHS:
        data_by_path.setdefault(
            payload_path,
            b"{}\n" if payload_path.endswith(".json") else None,
        )
        sidecar_path = f"{payload_path}.sha256"
        data_by_path[sidecar_path] = (
            f"{metadata[payload_path]['sha256']}  {Path(payload_path).name}\n"
        ).encode("ascii")
    files = {
        path: _metadata_snapshot(
            artifact,
            path,
            str(identity["sha256"]),
            int(identity["bytes"]),
            data=data_by_path[path],
            inode=index + 10,
        )
        for index, (path, identity) in enumerate(sorted(metadata.items()))
    }
    tree = consumer.TreeSnapshot(
        root_signature=(1, 1),
        directory_signatures={"": (1, 1)},
        files=files,
    )
    assert consumer._producer_tree_digest(files) == consumer.SHARDED_FINAL_TREE_SHA256
    assert (
        consumer._producer_output_inventory_digest(files)
        == consumer.SHARDED_FINAL_OUTPUT_INVENTORY_SHA256
    )
    consumer_digest = consumer._tree_digest(
        files,
        domain=consumer.SHARDED_TREE_DIGEST_DOMAIN,
    )
    assert (
        consumer_digest
        == "c61d4596042217c111f3f42a7c217e6064ca359eab62cdf4189c01cb78434eff"
    )
    for payload_path in consumer.SHARDED_PAYLOAD_PATHS:
        pin = next(
            item for item in binding["payload_files"] if item["path"] == payload_path
        )
        pin.update(
            {
                "sha256": metadata[payload_path]["sha256"],
                "bytes": metadata[payload_path]["bytes"],
                "sidecar_sha256": metadata[f"{payload_path}.sha256"]["sha256"],
                "sidecar_bytes": metadata[f"{payload_path}.sha256"]["bytes"],
            }
        )
    schema_snapshots: dict[str, consumer.FileSnapshot] = {}
    for index, (name, anchor) in enumerate(
        consumer.SHARDED_FINAL_SCHEMA_ANCHORS.items()
    ):
        relative = str(anchor["path"])
        schema_snapshots[relative] = _metadata_snapshot(
            root,
            relative,
            str(anchor["sha256"]),
            int(anchor["bytes"]),
            data=root.joinpath(*relative.split("/")).read_bytes(),
            inode=100 + index,
        )
    attestation_path = consumer.SHARDED_FINAL_ATTESTATION_PATH
    sidecar_path = f"{attestation_path}.sha256"
    schema_snapshots[attestation_path] = _metadata_snapshot(
        root,
        attestation_path,
        consumer.SHARDED_FINAL_ATTESTATION_SHA256,
        consumer.SHARDED_FINAL_ATTESTATION_BYTES,
        data=_json_bytes(attestation),
        inode=200,
    )
    schema_snapshots[sidecar_path] = _metadata_snapshot(
        root,
        sidecar_path,
        consumer.SHARDED_FINAL_ATTESTATION_SIDECAR_SHA256,
        consumer.SHARDED_FINAL_ATTESTATION_SIDECAR_BYTES,
        data=(
            f"{consumer.SHARDED_FINAL_ATTESTATION_SHA256}  "
            "independent_release_attestation.json\n"
        ).encode("ascii"),
        inode=201,
    )
    return (
        root,
        artifact,
        binding,
        tree,
        schema_snapshots,
        attestation_path,
        sidecar_path,
    )


def test_sharded_v1_passes_with_exact_external_and_git_provenance(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "producer")
    receipt = _validate_sharded(root, artifact, binding, release_commit)
    assert receipt["schema_version"] == consumer.SHARDED_RECEIPT_VERSION
    assert receipt["file_counts"] == {"total": 46, "payload": 23, "sidecar": 23}
    assert receipt["tree_digest_domain"] == consumer.PRODUCER_TREE_DIGEST_DOMAIN
    assert receipt["tree_digest_sha256"] == binding["tree_digest_sha256"]
    assert receipt["training_shards"]["shard_count"] == 2
    assert receipt["training_shards"]["logical_partition_records"] == 3440
    assert receipt["read_set"]["physical_identities_equal"] is True
    assert receipt["release"]["p0_findings"] == 0
    assert receipt["claims"]["sample_bodies_parsed"] is False
    assert str(root) not in json.dumps(receipt)


def test_sharded_v1_exact_final_trust_anchors_are_frozen(
    tmp_path: Path,
    _synthetic_sharded_trust_override: Any,
) -> None:
    _, _, binding, _ = _make_sharded_repo(tmp_path / "anchors")
    original = _synthetic_sharded_trust_override
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_sharded_final_trust_anchor_mismatch",
    ):
        original(binding)
    exact = _exact_final_trust_binding(binding)
    original(exact)
    changed = deepcopy(exact)
    for index, shard in enumerate(changed["training_shards"]["shards"]):
        shard["first_task_bundle_sha256"] = _sha(
            f"changed-first:{index}".encode("ascii")
        )
        shard["last_task_bundle_sha256"] = _sha(f"changed-last:{index}".encode("ascii"))
    changed["training_shards"]["boundary_commitment_sha256"] = (
        consumer._training_boundary_commitment(changed["training_shards"])
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_sharded_final_training_shards_mismatch",
    ):
        original(changed)
    swapped = deepcopy(exact)
    swapped["schema_files"]["build_receipt"] = deepcopy(
        swapped["schema_files"]["manifest"]
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_sharded_final_schema_anchor_mismatch",
    ):
        original(swapped)
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / consumer.SCHEMA_RELATIVE_PATH).read_text(
            "utf-8"
        )
    )
    anchors = schema["x-sharded-v1-final-trust-anchors"]
    assert anchors["release_commit"] == consumer.SHARDED_FINAL_RELEASE_COMMIT
    assert (
        anchors["boundary_commitment_sha256"]
        == consumer.SHARDED_FINAL_BOUNDARY_COMMITMENT_SHA256
    )


def test_sharded_v1_exact_final_metadata_runs_full_validation_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _synthetic_sharded_trust_override: Any,
) -> None:
    (
        root,
        artifact,
        binding,
        tree,
        external,
        attestation_path,
        sidecar_path,
    ) = _exact_final_metadata_chain(tmp_path)
    original_domain_hash = consumer._domain_hash
    trust_calls: list[tuple[str, str]] = []

    def domain_hash(domain: str, value: object) -> str:
        if domain == consumer.PRODUCER_INPUT_INVENTORY_DOMAIN:
            return consumer.SHARDED_FINAL_INPUT_INVENTORY_SHA256
        return original_domain_hash(domain, value)

    def validate_trust_anchors(value: object) -> None:
        binding_value = consumer._mapping(
            value,
            "test_binding_sharded_final_trust_anchor_invalid",
        )
        trust_calls.append(
            (
                str(binding_value["tree_digest_domain"]),
                str(binding_value["tree_digest_sha256"]),
            )
        )
        _synthetic_sharded_trust_override(binding_value)

    monkeypatch.setattr(
        consumer,
        "_validate_sharded_final_trust_anchors",
        validate_trust_anchors,
    )
    monkeypatch.setattr(consumer, "_domain_hash", domain_hash)
    monkeypatch.setattr(consumer, "_snapshot_tree", lambda *_args, **_kwargs: tree)
    monkeypatch.setattr(
        consumer,
        "_snapshot_external_inputs",
        lambda *_args, **_kwargs: (
            external,
            {},
            attestation_path,
            sidecar_path,
        ),
    )
    monkeypatch.setattr(
        consumer,
        "_stream_ordered_concat",
        lambda *_args, **_kwargs: consumer.OrderedConcatIdentity(
            sha256=consumer.SHARDED_FINAL_LOGICAL_IDENTITY[
                "logical_train_partition_sha256"
            ],
            bytes=59_845_314,
        ),
    )
    monkeypatch.setattr(consumer, "_validate_git_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        consumer,
        "_terminal_external_recheck",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        consumer,
        "_terminal_tree_recheck",
        lambda *_args, **_kwargs: None,
    )
    receipt = consumer.validate_artifact(
        artifact,
        binding,
        observed_producer_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
        producer_repository_root=root,
        observed_live_remote_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
    )
    assert receipt["tree_digest_domain"] == consumer.PRODUCER_TREE_DIGEST_DOMAIN
    assert receipt["tree_digest_sha256"] == consumer.SHARDED_FINAL_TREE_SHA256
    assert receipt["file_counts"] == {"total": 46, "payload": 23, "sidecar": 23}
    assert trust_calls == [
        (
            consumer.PRODUCER_TREE_DIGEST_DOMAIN,
            consumer.SHARDED_FINAL_TREE_SHA256,
        )
    ]

    for domain, digest, error, validator_called in (
        (
            consumer.PRODUCER_TREE_DIGEST_DOMAIN,
            "f" * 64,
            "binding_sharded_final_trust_anchor_mismatch",
            True,
        ),
        (
            consumer.SHARDED_TREE_DIGEST_DOMAIN,
            "c61d4596042217c111f3f42a7c217e6064ca359eab62cdf4189c01cb78434eff",
            "binding_document_schema_mismatch",
            False,
        ),
        (
            "anchor.wrong-tree-domain.v1",
            consumer.SHARDED_FINAL_TREE_SHA256,
            "binding_document_schema_mismatch",
            False,
        ),
    ):
        changed = deepcopy(binding)
        changed["tree_digest_domain"] = domain
        changed["tree_digest_sha256"] = digest
        calls_before = len(trust_calls)
        with pytest.raises(
            consumer.ConsumerPreflightError,
            match=error,
        ):
            consumer.validate_artifact(
                artifact,
                changed,
                observed_producer_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
                producer_repository_root=root,
                observed_live_remote_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
            )
        assert len(trust_calls) == calls_before + int(validator_called)
        if validator_called:
            assert trust_calls[-1] == (domain, digest)

    (artifact / "eval_proxy" / "chat.jsonl.sha256").unlink()
    monkeypatch.setattr(consumer, "_snapshot_tree", _ORIGINAL_SNAPSHOT_TREE)
    calls_before = len(trust_calls)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_file_inventory_mismatch",
    ):
        consumer.validate_artifact(
            artifact,
            binding,
            observed_producer_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
            producer_repository_root=root,
            observed_live_remote_commit=consumer.SHARDED_FINAL_RELEASE_COMMIT,
        )
    assert len(trust_calls) == calls_before + 1
    assert trust_calls[-1] == (
        consumer.PRODUCER_TREE_DIGEST_DOMAIN,
        consumer.SHARDED_FINAL_TREE_SHA256,
    )


def test_sharded_v1_streams_ordered_concat_and_direct_payload_identities(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "concat")
    binding["training_shards"]["ordered_concat_sha256"] = "f" * 64
    binding["logical_identity"]["logical_train_partition_sha256"] = "f" * 64
    binding["training_shards"]["boundary_commitment_sha256"] = (
        consumer._training_boundary_commitment(binding["training_shards"])
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="training_shard_ordered_concat_physical_drift",
    ):
        _validate_sharded(root, artifact, binding, release_commit)
    for logical_key in (
        "eval_proxy_partition_sha256",
        "serialization_inventory_sha256",
        "identity_probe_inventory_sha256",
    ):
        root, artifact, binding, release_commit = _make_sharded_repo(
            tmp_path / logical_key
        )
        binding["logical_identity"][logical_key] = "f" * 64
        with pytest.raises(
            consumer.ConsumerPreflightError,
            match=f"{logical_key}_physical_identity_drift",
        ):
            _validate_sharded(root, artifact, binding, release_commit)


def test_sharded_v1_rejects_unanchored_four_boundary_hash_rewrite(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(
        tmp_path / "boundary-rewrite"
    )
    for index, shard in enumerate(binding["training_shards"]["shards"]):
        shard["first_task_bundle_sha256"] = _sha(
            f"rewrite-first:{index}".encode("ascii")
        )
        shard["last_task_bundle_sha256"] = _sha(f"rewrite-last:{index}".encode("ascii"))
    binding["training_shards"]["boundary_commitment_sha256"] = (
        consumer._training_boundary_commitment(binding["training_shards"])
    )
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="manifest_training_shards_binding_drift",
    ):
        _validate_sharded(root, artifact, binding, release_commit)


@pytest.mark.parametrize(
    "bad_path",
    (
        "schemas//manifest.schema.json",
        "schemas/./manifest.schema.json",
        "./schemas/manifest.schema.json",
        "schemas/manifest.schema.json/",
        "schemas\\manifest.schema.json",
        "../schemas/manifest.schema.json",
    ),
)
def test_sharded_v1_requires_canonical_posix_external_paths(bad_path: str) -> None:
    with pytest.raises(consumer.ConsumerPreflightError):
        consumer._safe_relative_path(bad_path, "path_invalid")


def test_sharded_v1_requires_distinct_named_schema_paths(tmp_path: Path) -> None:
    _, _, binding, _ = _make_sharded_repo(tmp_path / "schema-paths")
    binding["schema_files"]["build_receipt"]["path"] = binding["schema_files"][
        "manifest"
    ]["path"]
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_schema_paths_not_distinct",
    ):
        consumer.validate_binding_document(binding)


def test_sharded_v1_schema_name_and_version_close_to_physical_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.schema.json"
    value = _closed_schema("manifest", ["schema_version"])
    value["$id"] = "https://anchor.local/schemas/wrong.schema.json"
    path.write_bytes(_json_bytes(value))
    snapshot = consumer._snapshot_file(path, capture=True)
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="producer_manifest_schema_physical_identity_mismatch",
    ):
        consumer._schema_document(
            snapshot,
            "producer_manifest",
            anchor=consumer.SHARDED_FINAL_SCHEMA_ANCHORS["manifest"],
        )


def test_sharded_v1_rejects_44_file_and_single_shard_shapes(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "physical")
    (artifact / "train" / "chat-00001-of-00002.jsonl").unlink()
    (artifact / "train" / "chat-00001-of-00002.jsonl.sha256").unlink()
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="artifact_tree_file_inventory_mismatch",
    ):
        _validate_sharded(root, artifact, binding, release_commit)
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "binding")
    binding["training_shards"]["shards"] = binding["training_shards"]["shards"][:1]
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        _validate_sharded(root, artifact, binding, release_commit)


def test_sharded_v1_rejects_logical_and_bundle_boundary_drift(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "logical")
    del binding["logical_identity"]["logical_target_inventory_sha256"]
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        _validate_sharded(root, artifact, binding, release_commit)
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "boundary")
    shards = binding["training_shards"]["shards"]
    shards[1]["first_task_bundle_sha256"] = shards[0]["last_task_bundle_sha256"]
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_training_shard_boundary_overlap",
    ):
        _validate_sharded(root, artifact, binding, release_commit)


@pytest.mark.parametrize(
    "drift",
    ["sidecar", "attestation", "read_set", "commit", "tree"],
)
def test_sharded_v1_rejects_physical_and_git_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / drift)
    observed_commit = release_commit
    if drift == "sidecar":
        (artifact / "eval_proxy" / "chat.jsonl.sha256").write_bytes(b"x\n")
    elif drift == "attestation":
        binding["release_attestation"]["sha256"] = "f" * 64
    elif drift == "read_set":
        binding["read_set"]["files"][0]["sha256"] = "f" * 64
        binding["read_set"]["canonical_digest_sha256"] = consumer._read_set_digest(
            binding["read_set"]["files"]
        )
    elif drift == "commit":
        observed_commit = "f" * 40
    else:
        binding["producer_git"]["release_tree"] = "f" * 40
    with pytest.raises(consumer.ConsumerPreflightError):
        _validate_sharded(root, artifact, binding, observed_commit)


def test_sharded_external_toctou_is_rejected(tmp_path: Path) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "producer")

    def mutate() -> None:
        path = root.joinpath(*SHARDED_SCHEMA_PATHS["manifest"].split("/"))
        path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="producer_read_set_terminal_toctou",
    ):
        _validate_sharded(
            root,
            artifact,
            binding,
            release_commit,
            before_terminal_recheck=mutate,
        )


def test_sharded_release_findings_and_reviewer_separation_fail_closed(
    tmp_path: Path,
) -> None:
    for name, mutate in (
        ("finding", lambda value: value["findings"].update({"p1": 1})),
        (
            "reviewer",
            lambda value: value["reviewer_separation"].update(
                {"reviewer_id": value["reviewer_separation"]["producer_id"]}
            ),
        ),
    ):
        root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / name)
        path = root.joinpath(*SHARDED_ATTESTATION_PATH.split("/"))
        value = json.loads(path.read_text("utf-8"))
        mutate(value)
        _write_payload(root, SHARDED_ATTESTATION_PATH, _json_bytes(value))
        raw = path.read_bytes()
        sidecar = path.with_name(path.name + ".sha256").read_bytes()
        binding["release_attestation"].update(
            {
                "sha256": _sha(raw),
                "bytes": len(raw),
                "sidecar_sha256": _sha(sidecar),
                "sidecar_bytes": len(sidecar),
            }
        )
        with pytest.raises(consumer.ConsumerPreflightError):
            _validate_sharded(root, artifact, binding, release_commit)


def test_sharded_cli_dry_run_requires_and_uses_external_provenance(
    tmp_path: Path,
    capsys: Any,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "producer")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(_json_bytes(binding))
    assert (
        consumer.main(
            [
                "dry-run",
                "--artifact",
                str(artifact),
                "--binding",
                str(binding_path),
                "--producer-commit",
                release_commit,
                "--producer-repository",
                str(root),
                "--producer-live-remote-commit",
                release_commit,
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema_version"] == consumer.SHARDED_RECEIPT_VERSION
    assert receipt["operation"] == "dry-run"


def test_sharded_versions_dispatch_strictly_and_schema_is_closed(
    tmp_path: Path,
) -> None:
    root, artifact, binding, release_commit = _make_sharded_repo(tmp_path / "producer")
    wrong = deepcopy(binding)
    wrong["schema_version"] = consumer.BINDING_VERSION
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="binding_document_schema_mismatch",
    ):
        _validate_sharded(root, artifact, wrong, release_commit)
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / consumer.SCHEMA_RELATIVE_PATH).read_text(
            "utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
