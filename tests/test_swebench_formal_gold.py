from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.data import snapshot as snapshot_module
from anchor_mvp.swebench import formal_gold as formal_gold_module
from anchor_mvp.swebench.formal_gold import (
    EXPECTED_STAGES,
    STAGE_TARGETS,
    FormalGoldExportConfig,
    FormalGoldExportError,
    export_formal_gold,
)
from anchor_mvp.tooling.swebench_execution_v3 import (
    DISTILLATION_EXECUTION_BINDING_KEYS,
    DISTILLATION_VALIDATION_STATE_SCHEMA,
    DISTILLATION_VALIDATOR_RESULT_SCHEMA,
    DISTILLATION_VALIDATOR_VERSION,
    candidate_artifact_set_sha256,
    distillation_lineage_sha256,
    distillation_tool_evidence,
    distillation_validation_state_sha256,
    sign_distillation_execution_receipt,
)
from anchor_mvp.training.manifest import sha256_file
from anchor_mvp.training.schema import iter_jsonl


KEY = b"formal-export-system-private-test-key"
VALIDATOR_SHA256 = (
    "f42de489ef86a213b76904d83b856b604cc957506909a1b783a8e369dfd8dd56"
)
IMAGE_DIGEST = (
    "sha256:a8a183c8a59d4c6a376ea6551ef14dabe73573bb739a7f045fe6180f30bd9671"
)
IMAGE_ID_SHA256 = (
    "a7411e7ae2cfabf3f87e24279138a1a158f068016bd952fe3661634d7ea02924"
)


def _digest(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _builder_output(patch: str, final_file_body: str) -> dict:
    edit_input = {"path": "a.py"}
    edit_invocation = _digest(
        json.dumps(
            {
                "input": edit_input,
                "planner_proposal_id": "edit-1",
                "tool": "edit",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    command = "anchor-validate compile"
    bash_input = {"command": command}
    bash_invocation = _digest(
        json.dumps(
            {
                "input": bash_input,
                "planner_proposal_id": "validate-1",
                "tool": "bash",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    validator_result = {
        "schema_version": DISTILLATION_VALIDATOR_RESULT_SCHEMA,
        "validator_version": DISTILLATION_VALIDATOR_VERSION,
        "mode": "compile",
        "success": True,
        "not_official_swebench_pass": True,
        "validation_level": "syntax",
        "changed_paths": ["a.py"],
        "changed_paths_sha256": _digest(
            json.dumps(["a.py"], sort_keys=True, separators=(",", ":"))
        ),
        "final_state_sha256": _digest(f"final-state:{patch}"),
        "validators": ["python-compile"],
    }
    raw_validation_output = (
        json.dumps(
            validator_result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    output = {
        "schema_version": "controlled-opencode-export+real-tool-results",
        "revision": 1,
        "workspace_diff": patch,
        "opencode_session_export": {
            "messages": [
                "public execution summary",
                {
                    "type": "reasoning",
                    "text": "typed hidden reasoning must not enter training",
                },
                {
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": command},
                                "output": raw_validation_output,
                            },
                        }
                    ]
                },
            ],
            "thinking": "must not enter training projection",
        },
        "tool_calls": [
            {
                "sequence": 1,
                "tool": "edit",
                "input": edit_input,
                "invocation_sha256": edit_invocation,
                "planner_proposal_id": "edit-1",
                "tool_policy_decision": "APPROVE",
                "execution_scope": "isolated-instance-container",
            },
            {
                "sequence": 2,
                "tool": "bash",
                "input": bash_input,
                "command": command,
                "command_sha256": _digest(command),
                "invocation_sha256": bash_invocation,
                "planner_proposal_id": "validate-1",
                "tool_policy_decision": "APPROVE",
                "execution_scope": "isolated-instance-container",
            },
        ],
        "tool_results": [
            {
                "sequence": 1,
                "tool": "edit",
                "status": "completed",
                "exit_code": 0,
                "invocation_sha256": edit_invocation,
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            },
            {
                "sequence": 2,
                "tool": "bash",
                "status": "completed",
                "exit_code": 0,
                "command_sha256": _digest(command),
                "invocation_sha256": bash_invocation,
                "output_sha256": _digest(raw_validation_output),
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            },
        ],
    }
    changed_files = [{"path": "a.py", "sha256": _digest(final_file_body)}]
    output["validation_state"] = {
        "schema_version": DISTILLATION_VALIDATION_STATE_SCHEMA,
        "final_patch_sha256": _digest(patch),
        "changed_files": changed_files,
        "changed_files_sha256": _digest(
            json.dumps(changed_files, sort_keys=True, separators=(",", ":"))
        ),
        "terminal_validation_output_sha256": _digest(raw_validation_output),
        "terminal_command_sha256": _digest(command),
        "validator_version_sha256": VALIDATOR_SHA256,
        "validator_result": validator_result,
    }
    output["validation_state_sha256"] = distillation_validation_state_sha256(
        output,
        final_patch_sha256=_digest(patch),
        validator_version_sha256=VALIDATOR_SHA256,
    )
    output["validator_version_sha256"] = VALIDATOR_SHA256
    return output


def _fixture(
    tmp_path: Path,
    *,
    task_count: int = 1,
    failed_task_indexes: set[int] | None = None,
    recovered_task_indexes: set[int] | None = None,
    checkpoint_state: str | None = None,
) -> FormalGoldExportConfig:
    failed_task_indexes = failed_task_indexes or set()
    recovered_task_indexes = recovered_task_indexes or set()
    bank = tmp_path / "bank"
    runtime = tmp_path / "runtime"
    tasks: list[dict] = []
    orders: list[dict] = []
    events: list[dict] = []
    pending_receipts: list[dict] = []
    checkpoint = _digest("checkpoint")
    config_sha = _digest("coordinator")
    lock_sha = _digest("lock")
    for index in range(task_count):
        task_id = "swe-full-v1:" + f"{index + 1:064x}"
        instance = f"project__repo-{index + 1}"
        task = {
            "schema_version": "anchor.swebench-candidate-task.v1",
            "task_id": task_id,
            "source": {
                "dataset_id": "SWE-bench/SWE-bench",
                "dataset_revision": "7" * 40,
                "split": "train",
                "derived_partition": "train",
                "instance_id": instance,
                "repo": "project/repo",
                "base_commit": "b" * 40,
            },
            "public_input": {"problem_statement": "public train task"},
        }
        tasks.append(task)
        final_file_body = f"task_{index} = True\n"
        patch = f"diff --git a/a.py b/a.py\n+{final_file_body}"
        outputs = {
            "planner": {
                "schema_version": "anchor.swebench-planner-output.v1",
                "work_items": ["implement"],
                "tool_proposals": [],
            },
            "tool_policy": {
                "schema_version": "anchor.swebench-tool-policy-output.v1",
                "decisions": [{"proposal_id": "edit-1", "decision": "APPROVE"}],
            },
            "domain_builder": _builder_output(patch, final_file_body),
            "domain_review": {
                "schema_version": "anchor.swebench-domain-review-output.v1",
                "decision": "PASS",
                "feedback": [],
            },
            "security": {
                "schema_version": "anchor.swebench-security-output.v1",
                "decision": "PASS",
                "findings": [],
            },
        }
        previous = task_id
        stage_records: dict[str, dict[str, object]] = {}
        for stage_index, stage in enumerate(EXPECTED_STAGES):
            record_id = "swe-full-stage-v1:" + f"{index * 10 + stage_index + 1:064x}"
            order = {
                "schema_version": "anchor.swebench-candidate-work-order.v1",
                "record_id": record_id,
                "task_id": task_id,
                "stage": stage,
                "upstream_record_ids": [previous],
                "provider_alias": "glm52_max",
            }
            orders.append(order)
            previous = record_id
            artifact = {
                "schema_version": "anchor.swebench-ccswitch-stage-artifact.v1",
                "task_id": task_id,
                "record_id": record_id,
                "stage": stage,
                "revision": 1,
                "provider_alias": "glm52_max",
                "reasoning_effort": "max",
                "input": {"identity": {"instance_id": instance}},
                "output": outputs[stage],
            }
            digest = task_id.rsplit(":", 1)[-1]
            artifact_path = (
                runtime
                / "content-records"
                / digest[:2]
                / digest
                / f"{stage}.json"
            )
            _write_json(artifact_path, artifact)
            artifact_sha256 = sha256_file(artifact_path)
            stage_records[stage] = {
                "revision": 1,
                "artifact_sha256": artifact_sha256,
            }
            events.append(
                {
                    "schema_version": "anchor.swebench-ccswitch-event.v1",
                    "task_id": task_id,
                    "stage": stage,
                    "revision": 1,
                    "status": "completed",
                    "provider_alias": "glm52_max",
                    "artifact": artifact_path.relative_to(runtime).as_posix(),
                    "artifact_sha256": artifact_sha256,
                }
            )
        private = runtime / "system-private" / _digest(task_id)
        patch_path = private / "final.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_bytes(patch.encode("utf-8"))
        if index not in failed_task_indexes or index in recovered_task_indexes:
            pending_receipts.append(
                {
                    "index": index,
                    "task_id": task_id,
                    "instance": instance,
                    "outputs": outputs,
                    "stage_records": stage_records,
                    "patch_path": patch_path,
                    "private": private,
                }
            )
        if index in failed_task_indexes or index in recovered_task_indexes:
            events.append(
                {
                    "schema_version": "anchor.swebench-ccswitch-event.v1",
                    "task_id": task_id,
                    "stage": "cleanup",
                    "revision": 1,
                    "status": "failed",
                    "error_code": "sandbox_cleanup_failed",
                }
            )
    task_shard = bank / "candidate-tasks" / "tasks.jsonl"
    order_shard = bank / "candidate-work-orders" / "work-orders.jsonl"
    _write_jsonl(task_shard, tasks)
    _write_jsonl(order_shard, orders)
    source_manifest = bank / "manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": "anchor.swebench-publication-manifest.v1",
            "source_split": "train",
            "train_only": True,
            "publication_ready": True,
            "counts": {"tasks": len(tasks), "work_orders": len(orders)},
            "files": [
                {
                    "path": task_shard.relative_to(bank).as_posix(),
                    "records": len(tasks),
                    "bytes": task_shard.stat().st_size,
                    "sha256": sha256_file(task_shard),
                },
                {
                    "path": order_shard.relative_to(bank).as_posix(),
                    "records": len(orders),
                    "bytes": order_shard.stat().st_size,
                    "sha256": sha256_file(order_shard),
                },
            ],
        },
    )
    source_manifest_sha256 = sha256_file(source_manifest)
    task_shard_sha256 = sha256_file(task_shard)
    order_shard_sha256 = sha256_file(order_shard)
    order_artifacts_sha256 = candidate_artifact_set_sha256(
        [
            {
                "path": order_shard.relative_to(bank).as_posix(),
                "sha256": order_shard_sha256,
            }
        ]
    )
    for pending in pending_receipts:
        outputs = pending["outputs"]
        builder = outputs["domain_builder"]
        transcript_sha, validation_sha = distillation_tool_evidence(builder)
        bindings = {
            "checkpoint_id": checkpoint,
            "config_sha256": config_sha,
            "execution_lock_sha256": lock_sha,
            "source_bank_manifest_sha256": source_manifest_sha256,
            "candidate_task_artifact_sha256": task_shard_sha256,
            "candidate_work_order_artifacts_sha256": order_artifacts_sha256,
            "task_id_sha256": _digest(pending["task_id"]),
            "instance_id_sha256": _digest(pending["instance"]),
            "repo_sha256": _digest("project/repo"),
            "base_commit": "b" * 40,
            "image_digest": IMAGE_DIGEST,
            "image_id_sha256": IMAGE_ID_SHA256,
            "final_patch_sha256": sha256_file(pending["patch_path"]),
            "tool_transcript_sha256": transcript_sha,
            "validation_evidence_sha256": validation_sha,
            "validation_state_sha256": builder["validation_state_sha256"],
            "validator_version_sha256": VALIDATOR_SHA256,
            "lineage_sha256": distillation_lineage_sha256(
                checkpoint_id=checkpoint,
                config_sha256=config_sha,
                execution_lock_sha256=lock_sha,
                task_id_sha256=_digest(pending["task_id"]),
                stage_records=pending["stage_records"],
            ),
        }
        receipt = sign_distillation_execution_receipt(
            bindings=bindings,
            validation_state=builder["validation_state"],
            receipt_id=_digest(f"receipt-{pending['index']}"),
            issued_at="2026-07-18T00:00:00Z",
            trusted_receipt_key=KEY,
        )
        _write_json(
            pending["private"] / "distillation-execution-receipt.json", receipt
        )
    _write_jsonl(runtime / "checkpoint.events.jsonl", events)
    _write_json(
        runtime / "manifest.json",
        {
            "schema_version": "anchor.swebench-ccswitch-run-manifest.v1",
            "checkpoint_id": checkpoint,
            "config_sha256": config_sha,
            "execution_lock_sha256": lock_sha,
            "source_bank_manifest_sha256": sha256_file(source_manifest),
        },
    )
    _write_json(
        runtime / "status.json",
        {
            "schema_version": "anchor.swebench-ccswitch-status.v2",
            "checkpoint_id": checkpoint,
            "config_sha256": config_sha,
            "execution_lock_sha256": lock_sha,
            "source_bank_manifest_sha256": sha256_file(source_manifest),
            "state": checkpoint_state
            or (
                "completed_with_failures"
                if failed_task_indexes or recovered_task_indexes
                else "completed"
            ),
            "content_free": True,
        },
    )
    heldout = tmp_path / "heldout" / "manifest.json"
    _write_json(
        heldout,
        {
            "schema_version": "anchor.heldout-manifest.v1",
            "split": "heldout",
            "canonical_cases_sha256": _digest("heldout ids"),
        },
    )
    audit = tmp_path / "heldout" / "audit.json"
    _write_json(
        audit,
        {
            "schema_version": "anchor.leak-audit.v1",
            "status": "PASS",
            "collision_count": 0,
            "content_emitted": False,
            "manifest_sha256": sha256_file(heldout),
        },
    )
    return FormalGoldExportConfig(
        bank_root=bank,
        tasks_glob="candidate-tasks/*.jsonl",
        work_orders_glob="candidate-work-orders/*.jsonl",
        source_bank_manifest=source_manifest,
        runtime_root=runtime,
        output_dir=tmp_path / "export",
        coordinator_config_sha256=config_sha,
        checkpoint_id=checkpoint,
        execution_lock_sha256=lock_sha,
        train_sandbox_image_digest=IMAGE_DIGEST,
        train_sandbox_image_id_sha256=IMAGE_ID_SHA256,
        validator_version_sha256=VALIDATOR_SHA256,
        validator_family="python+node-test-validator",
        heldout_manifest=heldout,
        heldout_leak_audit=audit,
        minimum_gold_per_stage=1,
    )


def test_export_accepts_only_authenticated_complete_chain_and_recomputes_lineage(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, task_count=2, failed_task_indexes={1})
    result = export_formal_gold(config, trusted_receipt_key=KEY)

    assert result["accepted_complete_chains"] == 1
    assert result["failed_task_count_excluded"] == 1
    manifest = json.loads(
        (config.output_dir / "partitions" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["formal_execution_export"][
        "unrecovered_cleanup_or_terminal_failure_excluded"
    ] is True
    assert manifest["formal_execution_export"]["evidence_tier"] == (
        "real_sandbox_self_verified"
    )
    assert manifest["formal_execution_export"][
        "not_official_swebench_pass"
    ] is True
    assert manifest["formal_execution_export"][
        "distillation_execution_receipt_required"
    ] is True
    task_bank = [
        value
        for _line, value in iter_jsonl(
            config.output_dir / "partitions" / "task_bank.jsonl"
        )
    ]
    assert len(task_bank) == 1
    gold_by_task: dict[str, list[dict]] = {}
    for stage in EXPECTED_STAGES:
        _expert, task_name, filename = STAGE_TARGETS[stage]
        gold_by_task[task_name] = [
            value
            for _line, value in iter_jsonl(
                config.output_dir / "partitions" / "gold" / filename
            )
        ]
    builder = gold_by_task["frontend"][0]
    assert "thinking" not in json.dumps(builder, sort_keys=True)
    assert "typed hidden reasoning" not in json.dumps(builder, sort_keys=True)
    assert builder["output"]["hidden_reasoning_fields_removed"] == 2
    lineage = snapshot_module._evaluate_formal_execution_lineage(
        gold_by_task,
        task_bank,
        manifest,
        {task: 1 for task in gold_by_task},
    )
    assert lineage["lineage_complete"] is True
    assert lineage["complete_chain_count"] == 1

    snapshot_config = snapshot_module.SnapshotConfig(
        root=tmp_path,
        partition_manifest=config.output_dir / "partitions" / "manifest.json",
        automation_status=config.output_dir / "automation" / "status.json",
        collection_dir=config.output_dir,
        gold_dir=config.output_dir / "partitions" / "gold",
        snapshot_dir=tmp_path / "snapshot",
        readiness_report=tmp_path / "readiness.json",
        expected_minimum_gold_records_per_expert=1,
    )
    readiness, _private = snapshot_module.evaluate_readiness(snapshot_config)
    assert readiness["training_ready"] is True
    assert readiness["blockers"] == []


def test_export_rejects_tampered_distillation_receipt(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    task = next(
        value
        for _line, value in iter_jsonl(
            config.bank_root / "candidate-tasks" / "tasks.jsonl"
        )
    )
    receipt_path = (
        config.runtime_root
        / "system-private"
        / _digest(task["task_id"])
        / "distillation-execution-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["receipt_hmac_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_receipt_authentication_failed"
    ):
        export_formal_gold(config, trusted_receipt_key=KEY)
    assert not config.output_dir.exists()


def test_export_rejects_validly_signed_receipt_for_different_candidate_shard(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    task = next(
        value
        for _line, value in iter_jsonl(
            config.bank_root / "candidate-tasks" / "tasks.jsonl"
        )
    )
    receipt_path = (
        config.runtime_root
        / "system-private"
        / _digest(task["task_id"])
        / "distillation-execution-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bindings = {
        name: receipt[name] for name in DISTILLATION_EXECUTION_BINDING_KEYS
    }
    bindings["candidate_task_artifact_sha256"] = _digest("another shard")
    replacement = sign_distillation_execution_receipt(
        bindings=bindings,
        validation_state=receipt["validation_state"],
        receipt_id=receipt["receipt_id"],
        issued_at=receipt["issued_at"],
        trusted_receipt_key=KEY,
    )
    _write_json(receipt_path, replacement)

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_receipt_authentication_failed"
    ):
        export_formal_gold(config, trusted_receipt_key=KEY)
    assert not config.output_dir.exists()


def test_terminal_validator_json_must_match_traced_raw_output() -> None:
    final_file_body = "value = 1\n"
    patch = f"diff --git a/a.py b/a.py\n+{final_file_body}"
    builder = _builder_output(patch, final_file_body)
    export = builder["opencode_session_export"]
    raw = export["messages"][2]["parts"][0]["state"]["output"]
    export["messages"][2]["parts"][0]["state"]["output"] = raw.replace(
        '"success":true', '"success":false'
    )

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_validation_evidence_invalid"
    ):
        formal_gold_module._verify_terminal_validator_export(
            builder, builder["validation_state"]
        )


def test_export_rejects_candidate_shard_changed_after_manifest(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, failed_task_indexes={0})
    task_shard = config.bank_root / "candidate-tasks" / "tasks.jsonl"
    task_shard.write_bytes(task_shard.read_bytes() + b"\n")

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_source_bank_manifest_invalid"
    ):
        export_formal_gold(config, trusted_receipt_key=KEY)
    assert not config.output_dir.exists()


def test_export_rejects_source_bank_manifest_changed_after_checkpoint(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, failed_task_indexes={0})
    manifest = json.loads(config.source_bank_manifest.read_text(encoding="utf-8"))
    manifest["operator_note"] = "changed after the checkpoint was created"
    _write_json(config.source_bank_manifest, manifest)

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_checkpoint_identity_mismatch"
    ):
        export_formal_gold(config, trusted_receipt_key=KEY)
    assert not config.output_dir.exists()


def test_capped_checkpoint_exports_authenticated_prefix(tmp_path: Path) -> None:
    config = _fixture(
        tmp_path,
        task_count=2,
        failed_task_indexes={1},
        checkpoint_state="stopped_checkpoint_resumable",
    )

    result = export_formal_gold(config, trusted_receipt_key=KEY)

    assert result["accepted_complete_chains"] == 1
    assert result["incomplete_or_unverified_task_count_excluded"] == 1
    assert result["training_ready"] is True
    assert result["source_bank_fully_exported"] is False
    assert result["not_for_full_bank_completion_claim"] is True
    manifest = json.loads(
        (config.output_dir / "partitions" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["formal_execution_export"]["checkpoint_state"] == (
        "stopped_checkpoint_resumable"
    )


def test_later_authenticated_retry_recovers_historical_failure(
    tmp_path: Path,
) -> None:
    config = _fixture(
        tmp_path,
        recovered_task_indexes={0},
        checkpoint_state="stopped_checkpoint_resumable",
    )

    result = export_formal_gold(config, trusted_receipt_key=KEY)

    assert result["accepted_complete_chains"] == 1
    assert result["historical_failed_task_count"] == 1
    assert result["recovered_after_historical_failure_count"] == 1
    assert result["failed_task_count_excluded"] == 0


def test_missing_post_cleanup_receipt_is_excluded_not_promoted(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, failed_task_indexes={0})

    with pytest.raises(
        FormalGoldExportError, match="formal_gold_no_authenticated_chains"
    ):
        export_formal_gold(config, trusted_receipt_key=KEY)
    assert not config.output_dir.exists()
