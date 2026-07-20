"""Materialize validated TaskBoard sidecars into completion-only SFT views.

This research bridge is intentionally separate from the stable five-stage Gold
pipeline.  It validates source lineage and split isolation, applies hard
visibility, emits deterministic ``messages`` records, shards below the declared
byte ceiling, and writes a content-free manifest.  It does not start QLoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anchor_mvp.research.query_specialization import (  # noqa: E402
    ROLES,
    QuerySpecializationError,
    TaskBoardSidecar,
    build_training_view,
    dataset_summary,
    taskboard_sidecar_sha256,
)
from train_query_specialization_mvp import (  # noqa: E402
    MATERIALIZATION_FIELDS,
    _load_config_snapshot,
    _load_contract_dataset,
    _mapping,
    _positive_int,
    _reject_unknown_fields,
    _required_sha256,
    _resolve,
)


VIEW_SCHEMA_VERSION = "anchor.query-specialization-sft-view.v1"
MANIFEST_SCHEMA_VERSION = "anchor.query-specialization-sft-manifest.v1"
MATERIALIZED_SPLITS = ("train", "calibration")
MATERIALIZED_VIEW_FIELDS = {
    "schema_version",
    "id",
    "source_sidecar_record_id",
    "source_sidecar_record_sha256",
    "source_gold_record_id",
    "source_gold_sha256",
    "source_gold_file_sha256",
    "source_snapshot_sha256",
    "source_snapshot_manifest_sha256",
    "task_bundle_sha256",
    "base_task_board_sha256",
    "projector_version",
    "projector_config_sha256",
    "sidecar_schema_sha256",
    "source_augmentation",
    "task_id",
    "pair_id",
    "variant",
    "language",
    "split",
    "role",
    "messages",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize Query-specialization sidecars into SFT messages"
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/research/query_specialization_mvp.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--execute", action="store_true", help="write views and manifest")
    parser.add_argument("--output-root", help="override the versioned output root")
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_strings(values: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(sorted(values)).encode("utf-8"))


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def materialized_view(sidecar: TaskBoardSidecar) -> dict[str, Any]:
    """Build one deterministic messages record without changing canonical Gold."""

    record = sidecar.training_record
    view = build_training_view(record)
    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "id": f"{record.record_id}:sft-view-v1",
        "source_sidecar_record_id": sidecar.record_id,
        "source_sidecar_record_sha256": taskboard_sidecar_sha256(sidecar),
        "source_gold_record_id": sidecar.source_gold_record_id,
        "source_gold_sha256": sidecar.source_gold_sha256,
        "source_gold_file_sha256": sidecar.source_gold_file_sha256,
        "source_snapshot_sha256": sidecar.source_snapshot_sha256,
        "source_snapshot_manifest_sha256": sidecar.source_snapshot_manifest_sha256,
        "task_bundle_sha256": sidecar.task_bundle_sha256,
        "base_task_board_sha256": sidecar.base_task_board_sha256,
        "projector_version": sidecar.projector_version,
        "projector_config_sha256": sidecar.config_sha256,
        "sidecar_schema_sha256": sidecar.sidecar_schema_sha256,
        "source_augmentation": {
            "kind": sidecar.augmentation.kind,
            "source_block_ids": list(sidecar.augmentation.source_block_ids),
            "overlay_block_ids": list(sidecar.augmentation.overlay_block_ids),
            "same_task_only": sidecar.augmentation.same_task_only,
            "split_before_augmentation": sidecar.augmentation.split_before_augmentation,
        },
        "task_id": record.task_id,
        "pair_id": record.pair_id,
        "variant": record.variant,
        "language": record.language,
        "split": record.split,
        "role": record.role,
        "messages": [
            {"role": "user", "content": view.prompt},
            {"role": "assistant", "content": view.target_output},
        ],
    }


def _validate_materialized_view(value: Mapping[str, Any]) -> None:
    if set(value) != MATERIALIZED_VIEW_FIELDS:
        raise QuerySpecializationError(
            "materialized view fields do not match the checked-in schema"
        )
    if value.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise QuerySpecializationError("materialized view schema_version changed")
    if value.get("split") not in MATERIALIZED_SPLITS:
        raise QuerySpecializationError(
            "materialized views may only use train or calibration"
        )
    messages = value.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [message.get("role") for message in messages]
        != ["user", "assistant"]
        or any(not isinstance(message.get("content"), str) for message in messages)
    ):
        raise QuerySpecializationError(
            "materialized view must contain one user and one assistant message"
        )


def _shard_lines(lines: Sequence[bytes], max_bytes: int) -> tuple[bytes, ...]:
    shards: list[bytes] = []
    current: list[bytes] = []
    current_size = 0
    for line in lines:
        if len(line) > max_bytes:
            raise QuerySpecializationError(
                f"one materialized record is {len(line)} bytes, above shard ceiling "
                f"{max_bytes}"
            )
        if current and current_size + len(line) > max_bytes:
            shards.append(b"".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line)
    if current:
        shards.append(b"".join(current))
    return tuple(shards)


def build_materialization_plan(
    *,
    sidecars: Sequence[TaskBoardSidecar],
    producer_manifest: Mapping[str, Any],
    dataset_validation: Mapping[str, Any],
    producer_manifest_sha256: str,
    config_bytes: bytes,
    materializer_bytes: bytes,
    view_schema_bytes: bytes,
    max_shard_bytes: int,
    require_train_and_calibration: bool,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return a content-free manifest plus deterministic shard payloads."""

    producer_manifest_sha256 = _required_sha256(
        producer_manifest_sha256, "producer_manifest_sha256"
    )

    grouped_lines: dict[tuple[str, str], list[bytes]] = {}
    ordered = sorted(
        sidecars,
        key=lambda sidecar: (
            sidecar.split,
            sidecar.expert,
            sidecar.source_gold_record_id,
            sidecar.pair_id,
            sidecar.variant,
            sidecar.record_id,
        ),
    )
    for sidecar in ordered:
        view = materialized_view(sidecar)
        _validate_materialized_view(view)
        grouped_lines.setdefault((sidecar.expert, sidecar.split), []).append(
            _canonical_line(view)
        )

    payloads: dict[str, bytes] = {}
    files: list[dict[str, Any]] = []
    for role in ROLES:
        for split in MATERIALIZED_SPLITS:
            lines = grouped_lines.get((role, split), [])
            for index, payload in enumerate(_shard_lines(lines, max_shard_bytes)):
                relative = f"{role}.{split}.part-{index:05d}.jsonl"
                payloads[relative] = payload
                files.append(
                    {
                        "path": relative,
                        "role": role,
                        "split": split,
                        "records": payload.count(b"\n"),
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                    }
                )

    task_bundles_by_split = {
        split: sorted(
            {
                sidecar.task_bundle_sha256
                for sidecar in sidecars
                if sidecar.split == split
            }
        )
        for split in MATERIALIZED_SPLITS
    }
    missing_required = [
        split
        for split in MATERIALIZED_SPLITS
        if not task_bundles_by_split[split]
    ]
    bridge_contract_passed = not (
        require_train_and_calibration and missing_required
    )
    records = tuple(sidecar.training_record for sidecar in sidecars)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "bridge_contract_passed": bridge_contract_passed,
        "mechanical_q1_smoke_started": False,
        "source_projector_manifest_sha256": producer_manifest_sha256,
        "source_dataset_contract_sha256": dataset_validation[
            "dataset_contract_sha256"
        ],
        "experiment_config_sha256": _sha256_bytes(config_bytes),
        "materializer_sha256": _sha256_bytes(materializer_bytes),
        "view_schema_sha256": _sha256_bytes(view_schema_bytes),
        "view_schema_version": VIEW_SCHEMA_VERSION,
        "source_contract": dataset_summary(records),
        "source_sidecar_validation": dict(dataset_validation),
        "producer_manifest": {
            "schema_version": producer_manifest["schema_version"],
            "claim_scope": producer_manifest["claim_scope"],
            "provider_requests": producer_manifest["provider_requests"],
            "canonical_gold_written": producer_manifest["canonical_gold_written"],
            "heldout_content_read": producer_manifest["heldout_content_read"],
            "heldout_content_emitted": producer_manifest[
                "heldout_content_emitted"
            ],
            "counts": producer_manifest["counts"],
        },
        "required_splits": (
            list(MATERIALIZED_SPLITS) if require_train_and_calibration else []
        ),
        "missing_required_splits": missing_required,
        "task_bundle_sets": {
            split: {
                "count": len(task_bundles_by_split[split]),
                "ids_sha256": _digest_strings(task_bundles_by_split[split]),
            }
            for split in MATERIALIZED_SPLITS
        },
        "max_shard_bytes": max_shard_bytes,
        "files": files,
        "non_claims": [
            "foundation_model_training",
            "strict_json_quality",
            "calibration_as_heldout_evaluation",
            "block_aware_token_budget",
            "token_to_block_attention_supervision",
            "shared_kv_correctness_or_speedup",
        ],
    }
    return manifest, payloads


def _write_versioned(
    output_root: Path,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> Path:
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    version = f"views-{_sha256_bytes(manifest_bytes)}"
    output_dir = output_root / version
    expected = {**payloads, "manifest.json": manifest_bytes}
    if output_dir.exists():
        for relative, payload in expected.items():
            target = output_dir / relative
            if not target.is_file() or target.read_bytes() != payload:
                raise QuerySpecializationError(
                    f"versioned output already exists with different bytes: {target}"
                )
        return output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".tmp-{version}"
    if staging.exists():
        raise QuerySpecializationError(
            f"stale materialization staging directory must be audited: {staging}"
        )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for relative, payload in expected.items():
            (staging / relative).write_bytes(payload)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config_bytes = config_path.read_bytes()
        config = _load_config_snapshot(config_bytes, str(config_path))
        materialization = _mapping(
            config.get("materialization"), "materialization"
        )
        _reject_unknown_fields(
            materialization, MATERIALIZATION_FIELDS, "materialization"
        )
        if materialization.get("schema_version") != VIEW_SCHEMA_VERSION:
            raise QuerySpecializationError(
                f"materialization.schema_version must be {VIEW_SCHEMA_VERSION!r}"
            )
        sidecars, _, producer_manifest, dataset_validation = (
            _load_contract_dataset(config)
        )
        view_schema_path = _resolve(
            REPO_ROOT,
            materialization.get("record_schema"),
            "materialization.record_schema",
        )
        view_schema_bytes = view_schema_path.read_bytes()
        view_schema = json.loads(view_schema_bytes)
        if view_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise QuerySpecializationError(
                "materialized view schema must use JSON Schema 2020-12"
            )
        if set(view_schema.get("required", [])) != MATERIALIZED_VIEW_FIELDS:
            raise QuerySpecializationError(
                "materialized view schema required fields do not match the emitter"
            )
        if (
            view_schema.get("properties", {})
            .get("split", {})
            .get("enum")
            != list(MATERIALIZED_SPLITS)
        ):
            raise QuerySpecializationError(
                "materialized view schema must restrict split to train/calibration"
            )
        max_shard_bytes = _positive_int(
            materialization.get("max_shard_bytes"),
            "materialization.max_shard_bytes",
        )
        require_train_and_calibration = (
            materialization.get("require_train_and_calibration_for_q1_smoke")
            is True
        )
        manifest, payloads = build_materialization_plan(
            sidecars=sidecars,
            producer_manifest=producer_manifest,
            dataset_validation=dataset_validation,
            producer_manifest_sha256=dataset_validation["manifest_sha256"],
            config_bytes=config_bytes,
            materializer_bytes=Path(__file__).read_bytes(),
            view_schema_bytes=view_schema_bytes,
            max_shard_bytes=max_shard_bytes,
            require_train_and_calibration=require_train_and_calibration,
        )
        response: dict[str, Any] = {
            "ok": True,
            "mode": "dry_run",
            "manifest": manifest,
        }
        if args.execute:
            if not manifest["bridge_contract_passed"]:
                raise QuerySpecializationError(
                    "bridge contract did not pass; refusing to materialize"
                )
            output_root = (
                Path(args.output_root).expanduser().resolve()
                if args.output_root
                else _resolve(
                    REPO_ROOT,
                    materialization.get("output_root"),
                    "materialization.output_root",
                )
            )
            output_dir = _write_versioned(output_root, manifest, payloads)
            response.update({"mode": "execute", "output_dir": str(output_dir)})
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    except (OSError, KeyError, json.JSONDecodeError, QuerySpecializationError) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
