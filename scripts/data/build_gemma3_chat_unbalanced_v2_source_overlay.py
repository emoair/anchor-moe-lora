"""Build the body-free Producer FINAL source overlay for the GLM batch adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "fixtures"
    / "research"
    / "gemma3_chat_unbalanced_v2_teacher_alignment_source_v2"
)


def _canonical(value: Any) -> bytes:
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


def _binding_by_path(
    producer_binding: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {item["relative_path"]: item for item in producer_binding["physical_files"]}


def _partition(
    bindings: dict[str, dict[str, Any]],
    *,
    kind: str,
    shard_index: int,
    relative: str,
    physical_rows: int,
    selected_rows: int,
) -> dict[str, Any]:
    binding = bindings[relative]
    return {
        "kind": kind,
        "shard_index": shard_index,
        "producer_relative_path": relative,
        "physical_row_count": physical_rows,
        "selected_row_count": selected_rows,
        "bytes": binding["bytes"],
        "sha256": binding["sha256"],
    }


def _forbidden(
    bindings: dict[str, dict[str, Any]],
    *,
    kind: str,
    relative: str,
    physical_rows: int,
    selected_rows: int,
    selection: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "producer_relative_path": relative,
        "physical_row_count": physical_rows,
        "selected_row_count": selected_rows,
        "selection": selection,
        "sha256": bindings[relative]["sha256"],
    }


def build(output: Path) -> dict[str, str]:
    output = output.resolve()
    if output.exists():
        raise RuntimeError("source_overlay_already_exists")
    if ROOT not in output.parents:
        raise RuntimeError("source_overlay_outside_repo")
    staging = output.with_name(f"{output.name}.staging")
    if staging.exists():
        raise RuntimeError("source_overlay_staging_exists")

    producer_binding, producer_manifest, _receipt, _attestation = (
        batch._producer_binding_from_physical_bytes()
    )
    bindings = _binding_by_path(producer_binding)
    partitions = [
        _partition(
            bindings,
            kind="train",
            shard_index=0,
            relative=batch.PRODUCER_TRAIN_SHARDS[0],
            physical_rows=1770,
            selected_rows=1770,
        ),
        _partition(
            bindings,
            kind="train",
            shard_index=1,
            relative=batch.PRODUCER_TRAIN_SHARDS[1],
            physical_rows=1670,
            selected_rows=1670,
        ),
        _partition(
            bindings,
            kind="router_train",
            shard_index=0,
            relative=batch.PRODUCER_ROUTER_PATH,
            physical_rows=100,
            selected_rows=80,
        ),
    ]
    forbidden_sources = [
        _forbidden(
            bindings,
            kind="eval_proxy",
            relative=batch.PRODUCER_EVAL_PATH,
            physical_rows=860,
            selected_rows=860,
            selection="all_rows",
        ),
        _forbidden(
            bindings,
            kind="router_eval",
            relative=batch.PRODUCER_ROUTER_PATH,
            physical_rows=100,
            selected_rows=20,
            selection="split_eval_proxy",
        ),
        _forbidden(
            bindings,
            kind="tool_comparison_eval",
            relative=batch.PRODUCER_TOOL_EVAL_PATH,
            physical_rows=400,
            selected_rows=400,
            selection="all_rows",
        ),
        _forbidden(
            bindings,
            kind="planner_comparison_eval",
            relative=batch.PRODUCER_PLANNER_EVAL_PATH,
            physical_rows=240,
            selected_rows=240,
            selection="all_rows",
        ),
        _forbidden(
            bindings,
            kind="identity_eval",
            relative=batch.PRODUCER_IDENTITY_EVAL_PATH,
            physical_rows=50,
            selected_rows=10,
            selection="unique_task_bundles",
        ),
    ]
    manifest = {
        "schema_version": batch.SOURCE_MANIFEST_SCHEMA_VERSION_V2,
        "dataset_kind": batch.DATASET_KIND,
        "status": "FINAL",
        "manifest_id": (f"producer-source-overlay-v2:{batch.PRODUCER_RELEASE_COMMIT}"),
        "producer_release": producer_binding,
        "partitions": partitions,
        "forbidden_sources": forbidden_sources,
        "logical_identity": producer_manifest["logical_identity"],
        "projection_contract": {
            "schema_version": batch.SOURCE_PROJECTION_SCHEMA_VERSION_V2,
            "source_order": ("producer_train_shard_order_then_router_source_order"),
            "role_mapping": {
                "humor": "humor",
                "serious": "serious",
                "angry_style": "angry",
                "tool_call": "tool",
                "review_audit": "review",
                "emotion_router": "router",
            },
            "identity_projection": (
                "identity_aligned_tool_call_direct_no_tool_to_identity"
            ),
            "target_fields_excluded": [
                "target",
                "route_plan_target",
                "oracle",
            ],
            "forbidden_fields_excluded": ["future", "forbidden", "heldout"],
            "train_selected_rows": 3440,
            "router_selected_rows": 80,
            "identity_selected_rows": 20,
            "total_selected_rows": 3520,
            "content_materialized_in_overlay": False,
        },
    }
    manifest_raw = _canonical(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    approval = {
        "schema_version": batch.SOURCE_APPROVAL_SCHEMA_VERSION_V2,
        "dataset_kind": batch.DATASET_KIND,
        "decision": "APPROVED",
        "manifest_sha256": manifest_sha,
        "partition_set_sha256": batch._manifest_partition_set_sha256(partitions),
        "producer_release_attestation_sha256": (
            batch.PRODUCER_RELEASE_ATTESTATION_SHA256
        ),
        "producer_release_commit": batch.PRODUCER_RELEASE_COMMIT,
        "evidence": [
            {
                "role": "producer_independent_release",
                "identity_sha256": (batch.PRODUCER_RELEASE_ATTESTATION_SHA256),
                "decision": "APPROVED",
            },
            {
                "role": "deterministic_source_projection",
                "identity_sha256": batch.PRODUCER_TREE_SHA256,
                "decision": "APPROVED",
            },
        ],
    }
    approval_raw = _canonical(approval)
    approval_sha = hashlib.sha256(approval_raw).hexdigest()

    staging.mkdir(parents=True, exist_ok=False)
    (staging / "manifest.json").write_bytes(manifest_raw)
    (staging / "manifest.json.sha256").write_bytes(
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    (staging / "source_approval.json").write_bytes(approval_raw)
    (staging / "source_approval.json.sha256").write_bytes(
        f"{approval_sha}  source_approval.json\n".encode("ascii")
    )
    if batch._producer_binding_from_physical_bytes()[0] != producer_binding:
        raise RuntimeError("producer_terminal_toctou")
    if json.loads((staging / "manifest.json").read_bytes()) != manifest:
        raise RuntimeError("manifest_staging_reparse_failed")
    if json.loads((staging / "source_approval.json").read_bytes()) != approval:
        raise RuntimeError("approval_staging_reparse_failed")
    os.rename(staging, output)
    return {
        "manifest_sha256": manifest_sha,
        "approval_sha256": approval_sha,
        "partition_set_sha256": approval["partition_set_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
