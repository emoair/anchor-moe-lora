from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import gemma3_chat_unbalanced_v2_teacher_finalizer_v1 as finalizer


RECORD_SCHEMA = Path(
    "configs/data/gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixture() -> tuple[
    batch.SourceInventory,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_records: list[batch.SourceRecord] = []
    main_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    role_counts = {
        "humor": 240,
        "serious": 240,
        "angry_style": 240,
        "tool_call": 1360,
        "review_audit": 1360,
    }
    role_map = {
        "humor": "humor",
        "serious": "serious",
        "angry_style": "angry",
        "tool_call": "tool",
        "review_audit": "review",
    }
    serial = 0
    for asset, count in role_counts.items():
        for index in range(count):
            identity = index < 20
            bundle = _sha(f"identity-{index}" if identity else f"{asset}-{index}")
            record_id = f"{asset}-{index:04d}"
            messages = [
                {"role": "system", "content": f"system-{record_id}"},
                {"role": "user", "content": f"user-{record_id}"},
            ]
            serialization = {
                "prompt_sha256": _sha(f"prompt-{record_id}"),
                "full_example_sha256": _sha(f"full-{record_id}"),
                "policy": "gemma3-it-exporter-v1",
                "raw_token_ids_persisted": False,
                "system_embedded_in_first_user_turn": True,
                "tokenizer_called": True,
            }
            batch_role = role_map[asset]
            if asset == "tool_call" and identity:
                batch_role = "identity"
            allowed_tools = ["local_lookup"] if batch_role == "tool" else []
            allowed_evidence = ["evidence-1"] if batch_role == "tool" else []
            source = batch.SourceRecord(
                record_id=record_id,
                task_bundle=bundle,
                semantic=_sha(f"semantic-{asset}-{index}"),
                role=batch_role,
                language="en" if index % 2 == 0 else "zh-CN",
                teacher_input=f"teacher-input-{record_id}",
                gemma_serialization_identity_sha256=serialization["prompt_sha256"],
                teacher_contract={
                    "identity_class": (
                        "air_attribution" if batch_role == "identity" else None
                    ),
                    "prompt_template_id": f"template-{batch_role}",
                    "output_schema_id": f"schema-{batch_role}",
                    "allowed_tools": allowed_tools,
                    "allowed_evidence_ids": allowed_evidence,
                    "router_options": [],
                },
                guards={
                    "target_leakage_sha256": [],
                    "references": [],
                },
                partition_kind="train",
                line_number=serial + 1,
                source_line_sha256=_sha(f"line-{serial}"),
                content_identity_sha256=_sha(f"teacher-projection-{serial}"),
            )
            serial += 1
            identity_alignment = (
                {
                    "stable_identity_fact": finalizer.AIR_IDENTITY_SENTENCE,
                    "intent": "direct_self_identity",
                    "case_context": "direct",
                }
                if identity
                else None
            )
            candidate = (
                {"final_answer": finalizer.AIR_IDENTITY_SENTENCE}
                if identity and asset == "review_audit"
                else None
            )
            row = {
                "schema_version": (
                    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-record.v1"
                ),
                "namespace": "gemma3_chat_five_expert_qonly_unbalanced_v2",
                "split": "train",
                "role": asset,
                "record_id": record_id,
                "task_bundle_sha256": bundle,
                "messages": messages,
                "gemma_serialization": serialization,
                "identity_alignment": identity_alignment,
                "tool_trace": {
                    "direct_no_tool": bool(identity and asset == "tool_call")
                },
                "review_dependency": {
                    "candidate_projection": candidate,
                },
            }
            if batch_role in {"humor", "serious", "angry", "identity"}:
                target_value: Any = (
                    finalizer.AIR_IDENTITY_SENTENCE
                    if identity
                    else f"answer-{record_id}"
                )
                output = {"kind": "natural_text", "value": target_value}
            elif batch_role == "tool":
                output = {
                    "kind": "structured_json",
                    "value": {
                        "final_answer": f"answer-{record_id}",
                        "tool_call": {
                            "name": "local_lookup",
                            "arguments": {"query": index},
                        },
                        "evidence_ids": ["evidence-1"],
                    },
                }
            else:
                output = {
                    "kind": "structured_json",
                    "value": {
                        "verdict": "pass",
                        "faults": [],
                        "correction": None,
                    },
                }
            alignment = {
                "schema_version": batch.OUTPUT_SCHEMA_VERSION,
                "idempotency_key": _sha(f"idempotency-{record_id}"),
                "record_id": record_id,
                "task_bundle": bundle,
                "split": "train",
                "source_binding": {
                    "content_identity_sha256": source.content_identity_sha256,
                    "gemma_serialization_identity_sha256": (
                        source.gemma_serialization_identity_sha256
                    ),
                    "task_bundle": bundle,
                },
                "output": output,
                "hidden_reasoning_retained": False,
            }
            source_records.append(source)
            main_rows.append(row)
            alignment_rows.append(alignment)
    main_source_by_id = {item.record_id: item for item in source_records}
    main_alignment_by_id = {str(item["record_id"]): item for item in alignment_rows}
    original_order = {
        str(item["record_id"]): index for index, item in enumerate(main_rows)
    }
    role_order = {
        "humor": 0,
        "serious": 1,
        "angry_style": 2,
        "tool_call": 3,
        "review_audit": 4,
    }

    def main_order(item: dict[str, Any]) -> tuple[int, int, int]:
        if item["identity_alignment"] is not None:
            index = int(str(item["record_id"]).rsplit("-", 1)[1])
            return (0, index, role_order[str(item["role"])])
        return (1, original_order[str(item["record_id"])], 0)

    main_rows.sort(key=main_order)
    source_records = [main_source_by_id[str(item["record_id"])] for item in main_rows]
    alignment_rows = [
        main_alignment_by_id[str(item["record_id"])] for item in main_rows
    ]
    for index in range(100):
        split = "train" if index < 80 else "eval_proxy"
        record_id = f"router-{index:03d}"
        bundle = _sha(f"router-bundle-{index}")
        messages = [
            {"role": "system", "content": f"router-system-{index}"},
            {"role": "user", "content": f"router-user-{index}"},
        ]
        serialization = {
            "prompt_sha256": _sha(f"router-prompt-{index}"),
            "target_sha256": _sha(f"router-target-{index}"),
            "raw_token_ids_persisted": False,
        }
        row = {
            "namespace": "gemma3_chat_emotion_router_qonly_v1",
            "split": split,
            "record_id": record_id,
            "task_bundle_sha256": bundle,
            "messages": messages,
            "serialization": serialization,
        }
        router_rows.append(row)
        if split == "eval_proxy":
            continue
        source = batch.SourceRecord(
            record_id=record_id,
            task_bundle=bundle,
            semantic=_sha(f"router-semantic-{index}"),
            role="router",
            language="en" if index % 2 == 0 else "zh-CN",
            teacher_input=f"router-input-{index}",
            gemma_serialization_identity_sha256=serialization["prompt_sha256"],
            teacher_contract={
                "identity_class": None,
                "prompt_template_id": "template-router",
                "output_schema_id": "schema-router",
                "allowed_tools": [],
                "allowed_evidence_ids": [],
                "router_options": ["humor", "serious", "angry"],
            },
            guards={"target_leakage_sha256": [], "references": []},
            partition_kind="router_train",
            line_number=index + 1,
            source_line_sha256=_sha(f"router-line-{index}"),
            content_identity_sha256=_sha(f"router-projection-{index}"),
        )
        source_records.append(source)
        alignment_rows.append(
            {
                "schema_version": batch.OUTPUT_SCHEMA_VERSION,
                "idempotency_key": _sha(f"router-idempotency-{index}"),
                "record_id": record_id,
                "task_bundle": bundle,
                "split": "train",
                "source_binding": {
                    "content_identity_sha256": source.content_identity_sha256,
                    "gemma_serialization_identity_sha256": (
                        source.gemma_serialization_identity_sha256
                    ),
                    "task_bundle": bundle,
                },
                "output": {
                    "kind": "structured_json",
                    "value": {
                        "route": "serious",
                        "plan": ["answer directly"],
                        "stop": True,
                    },
                },
                "hidden_reasoning_retained": False,
            }
        )
    inventory = batch.SourceInventory(
        records=tuple(source_records),
        manifest_sha256="1" * 64,
        approval_sha256="2" * 64,
        partition_set_sha256="3" * 64,
        readonly_test_identity_sha256="4" * 64,
        source_identity_sha256="5" * 64,
        role_counts={},
        language_counts={},
        identity_counts={},
    )
    return inventory, main_rows, router_rows, alignment_rows


def _record_validator() -> Draft202012Validator:
    value = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return Draft202012Validator(value)


def test_exact_consumer_projection_and_role_aware_targets() -> None:
    inventory, main_rows, router_rows, alignment_rows = _fixture()
    joins = finalizer.build_source_joins(
        inventory,
        main_rows=main_rows,
        router_rows=router_rows,
    )
    records = finalizer.project_teacher_records(
        joins,
        alignment_rows,
        record_validator=_record_validator(),
    )
    assert len(records) == 3520
    assert Counter(row["source_asset"] for row in records) == (
        finalizer.EXPECTED_ASSET_COUNTS
    )
    assert all(
        set(row)
        == {
            "schema_version",
            "namespace",
            "record_id_sha256",
            "source_content_sha256",
            "source_serialization_identity_sha256",
            "task_bundle_sha256",
            "source_asset",
            "decision",
            "teacher_target",
            "teacher_target_sha256",
        }
        for row in records
    )
    identity_tool = next(
        row
        for row, join in zip(records, joins, strict=True)
        if join.source_asset == "tool_call" and join.identity_aligned
    )
    assert identity_tool["teacher_target"] == finalizer.AIR_IDENTITY_SENTENCE
    regular_tool = next(
        row
        for row, join in zip(records, joins, strict=True)
        if join.source_asset == "tool_call" and not join.identity_aligned
    )
    parsed = json.loads(regular_tool["teacher_target"])
    assert parsed["tool_call"]["name"] == "local_lookup"
    assert regular_tool["teacher_target"] == json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert all(
        row["teacher_target_sha256"] == _sha(row["teacher_target"]) for row in records
    )


def test_router_mixed_file_read_scope_is_explicit_and_eval_is_not_emitted() -> None:
    inventory, main_rows, router_rows, _ = _fixture()
    joins = finalizer.build_source_joins(
        inventory,
        main_rows=main_rows,
        router_rows=router_rows,
    )
    assert finalizer._router_mixed_file_audit(router_rows, joins) == {
        "router_mixed_file_eval_proxy_rows_parsed": 20,
        "router_eval_proxy_rows_emitted_to_teacher_train": 0,
    }

    wrong_split = [dict(item) for item in router_rows]
    wrong_split[79]["split"] = "eval_proxy"
    with pytest.raises(
        batch.AdapterError,
        match="teacher final router mixed split count drift",
    ):
        finalizer._router_mixed_file_audit(wrong_split, joins)

    eval_record_id = str(router_rows[80]["record_id"])
    poisoned_joins = list(joins)
    planner_index = next(
        index
        for index, item in enumerate(poisoned_joins)
        if item.source_asset == "planner_router"
    )
    poisoned_joins[planner_index] = replace(
        poisoned_joins[planner_index],
        record_id=eval_record_id,
    )
    with pytest.raises(
        batch.AdapterError,
        match="teacher final router eval emission forbidden",
    ):
        finalizer._router_mixed_file_audit(router_rows, poisoned_joins)


def test_message_identity_cannot_be_substituted_by_teacher_projection_identity() -> (
    None
):
    inventory, main_rows, router_rows, _ = _fixture()
    message_identity = _canonical_sha(main_rows[0]["messages"])
    mutated = replace(
        inventory.records[0],
        content_identity_sha256=message_identity,
    )
    bad_inventory = replace(
        inventory,
        records=(mutated, *inventory.records[1:]),
    )
    with pytest.raises(
        batch.AdapterError,
        match="teacher final source identity domain collision",
    ):
        finalizer.build_source_joins(
            bad_inventory,
            main_rows=main_rows,
            router_rows=router_rows,
        )


def test_identity_false_attribution_and_unbound_idempotency_fail_closed() -> None:
    inventory, main_rows, router_rows, alignment_rows = _fixture()
    joins = finalizer.build_source_joins(
        inventory,
        main_rows=main_rows,
        router_rows=router_rows,
    )
    identity_humor = next(
        index
        for index, join in enumerate(joins)
        if join.source_asset == "humor" and join.identity_aligned
    )
    poisoned = [dict(item) for item in alignment_rows]
    poisoned[identity_humor] = {
        **poisoned[identity_humor],
        "output": {
            "kind": "natural_text",
            "value": (finalizer.AIR_IDENTITY_SENTENCE + " 同时该模型由Google训练。"),
        },
    }
    with pytest.raises(
        batch.AdapterError,
        match="teacher final identity false attribution",
    ):
        finalizer.project_teacher_records(joins, poisoned)

    expected = {
        str(item["record_id"]): str(item["idempotency_key"]) for item in alignment_rows
    }
    drifted = [dict(item) for item in alignment_rows]
    drifted[0] = {
        **drifted[0],
        "idempotency_key": "0" * 64,
    }
    with pytest.raises(
        batch.AdapterError,
        match="teacher final alignment idempotency drift",
    ):
        finalizer.project_teacher_records(
            joins,
            drifted,
            expected_idempotency_by_record=expected,
        )


def test_shards_preserve_bundle_boundaries_and_stay_below_50_mib() -> None:
    inventory, main_rows, router_rows, alignment_rows = _fixture()
    joins = finalizer.build_source_joins(
        inventory,
        main_rows=main_rows,
        router_rows=router_rows,
    )
    records = finalizer.project_teacher_records(joins, alignment_rows)
    shards = finalizer.shard_teacher_records(
        records,
        max_payload_bytes=1_000_000,
    )
    assert sum(len(shard.records) for shard in shards) == 3520
    assert all(len(shard.raw) < 50 * 1024 * 1024 for shard in shards)
    bundle_to_shard: dict[str, int] = {}
    for shard_index, shard in enumerate(shards):
        for record in shard.records:
            bundle = record["task_bundle_sha256"]
            assert bundle_to_shard.setdefault(bundle, shard_index) == shard_index


def test_identity_probe_digest_is_not_replaceable_by_aggregate_counts() -> None:
    value = {
        "counts": {
            "identity_bundles": 30,
            "identity_role_records": 150,
            "splits": {"eval_proxy": 860, "train": 3440},
        },
        "logical_identity": {
            "identity_probe_inventory_sha256": (
                "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5"
            )
        },
    }
    facts = finalizer._verify_source_manifest_facts(value)
    assert facts["producer_eval_records"] == 50
    drifted = json.loads(json.dumps(value))
    drifted["logical_identity"]["identity_probe_inventory_sha256"] = "0" * 64
    with pytest.raises(
        batch.AdapterError,
        match="teacher final identity probe digest drift",
    ):
        finalizer._verify_source_manifest_facts(drifted)


def test_atomic_noreplace_rejects_destination_race_and_inherits_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    original_identity = finalizer._directory_identity(
        source,
        reason="test_source_invalid",
    )

    def competitor() -> None:
        destination.mkdir()

    with pytest.raises(FileExistsError):
        finalizer._rename_noreplace(
            source,
            destination,
            before_rename=competitor,
        )
    assert source.is_dir()

    destination.rmdir()
    finalizer._rename_noreplace(
        source,
        destination,
        before_rename=lambda: None,
    )
    assert (
        finalizer._directory_identity(
            destination,
            reason="test_destination_invalid",
        )
        == original_identity
    )


def test_owned_parent_swap_and_internal_reparse_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child, owned = finalizer._create_directory_chain(
        root,
        finalizer.PurePosixPath("a/b"),
    )
    assert child.is_dir()
    moved = root / "a-original"
    (root / "a").rename(moved)
    (root / "a").mkdir()
    with pytest.raises(
        batch.AdapterError,
        match="teacher final output directory identity drift",
    ):
        finalizer._assert_owned_directories(owned)

    link = root / "internal-link"
    try:
        link.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(
        batch.AdapterError,
        match="teacher final reparse path forbidden",
    ):
        finalizer._relative_file(
            root,
            "internal-link/b",
            reason="test_relative_path_invalid",
        )
    assert os.path.lexists(link)


def test_snapshot_rejects_same_content_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b'{"safe":true}\n')
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(batch.AdapterError):
        finalizer._snapshot(
            link,
            expected_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            reason="test_snapshot_reparse_rejected",
        )
