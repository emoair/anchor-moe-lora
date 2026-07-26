from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.training import gemma3_chat_unbalanced_v2_consumer_v1 as consumer


PRODUCER_COMMIT = "1" * 40
FINAL_STATUS = "release_ready_for_diagnostic_training"


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
    with pytest.raises(
        consumer.ConsumerPreflightError,
        match="test_json_invalid",
    ):
        consumer._strict_json(b'{"a":NaN}\n', "test")


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
