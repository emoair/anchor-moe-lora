"""Freeze a balanced, explicitly scoped snapshot from partial strict Gold.

The input must be an export produced by :mod:`anchor_mvp.data.partial_export`.
This module never turns partial expert data into a full-chain claim: the output
keeps the waiver and exclusion proofs in its immutable manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

from .cleaning import contains_secret_material
from .partial_export import EXPORT_SCHEMA_VERSION, TRAINING_MODE
from ..training.manifest import sha256_file
from ..training.schema import DatasetValidationError, iter_jsonl, validate_jsonl


PARTIAL_SNAPSHOT_SCHEMA = "anchor.per-expert-partial-training-snapshot.v1"
DEFAULT_RECORDS_PER_EXPERT = 128
SELECTION = "sha256(seed:id), ascending"
EXPERT_SOURCES = {
    "planner": ("plan", "data_plan.jsonl"),
    "tool_policy": ("tool_policy", "data_tool_policy.jsonl"),
    "frontend_gen": ("frontend", "data_frontend.jsonl"),
    "frontend_review": ("review", "data_review.jsonl"),
    "security_gate": ("security", "data_security.jsonl"),
}
REQUIRED_EXCLUSIONS = frozenset(
    {"negative", "reject", "oracle_label_only", "heldout"}
)
_SHA256_HEX = frozenset("0123456789abcdef")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _selection_key(record: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()


def _snapshot_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    parts = [
        f"{expert}:{files[expert]['path']}:{files[expert]['sha256']}:"
        f"{files[expert]['records']}"
        for expert in EXPERT_SOURCES
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _verify_existing_snapshot(
    output_dir: Path,
    *,
    source_manifest_sha256: str,
    per_expert: int,
    seed: int,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    sidecar_path = output_dir / "manifest.json.sha256"
    manifest = _load_mapping(manifest_path, label="partial training snapshot")
    sidecar = sidecar_path.read_text(encoding="ascii").split()
    if (
        manifest.get("schema_version") != PARTIAL_SNAPSHOT_SCHEMA
        or manifest.get("training_mode") != TRAINING_MODE
        or manifest.get("not_for_end_to_end_claim") is not True
        or manifest.get("source_export_manifest_sha256")
        != source_manifest_sha256
        or manifest.get("per_expert") != per_expert
        or manifest.get("seed") != seed
        or not isinstance(manifest.get("waivers"), Mapping)
        or not isinstance(manifest.get("excluded"), Mapping)
        or not all(
            manifest["excluded"].get(name) is True
            for name in REQUIRED_EXCLUSIONS
        )
        or not sidecar
        or sidecar[0] != sha256_file(manifest_path)
    ):
        raise ValueError("existing partial training snapshot conflicts with request")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(EXPERT_SOURCES):
        raise ValueError("existing partial training snapshot file map is invalid")
    for expert, (_task, filename) in EXPERT_SOURCES.items():
        item = files.get(expert)
        path = output_dir / filename
        if (
            not isinstance(item, Mapping)
            or item.get("path") != filename
            or item.get("records") != per_expert
            or not path.is_file()
            or path.is_symlink()
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
        ):
            raise ValueError("existing partial training snapshot binding mismatch")
    if manifest.get("snapshot_sha256") != _snapshot_digest(files):
        raise ValueError("existing partial training snapshot digest mismatch")
    return manifest


def freeze_partial_gold_snapshot(
    export_dir: Path,
    output_dir: Path,
    *,
    per_expert: int = DEFAULT_RECORDS_PER_EXPERT,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Select an equal deterministic subset from each strict-Gold expert file."""

    if per_expert != DEFAULT_RECORDS_PER_EXPERT:
        raise ValueError(
            f"controlled partial mode requires exactly {DEFAULT_RECORDS_PER_EXPERT} "
            "records per expert"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("selection seed must be a non-negative integer")
    export_dir = export_dir.resolve()
    output_dir = output_dir.resolve()
    source_manifest_path = export_dir / "manifest.json"
    source_manifest = _load_mapping(
        source_manifest_path, label="partial Gold export manifest"
    )
    source_manifest_sha256 = sha256_file(source_manifest_path)
    source = source_manifest.get("source")
    exclusions = source_manifest.get("excluded")
    gold_files = source_manifest.get("gold_files")
    strict_complete_chains = source_manifest.get("strict_complete_chains")
    if (
        source_manifest.get("schema_version") != EXPORT_SCHEMA_VERSION
        or source_manifest.get("training_mode") != TRAINING_MODE
        or source_manifest.get("not_for_end_to_end_claim") is not True
        or not isinstance(source, Mapping)
        or not _is_sha256(source.get("partition_manifest_sha256"))
        or not isinstance(exclusions, Mapping)
        or not all(exclusions.get(name) is True for name in REQUIRED_EXCLUSIONS)
        or not isinstance(source_manifest.get("waivers"), Mapping)
        or isinstance(strict_complete_chains, bool)
        or not isinstance(strict_complete_chains, int)
        or strict_complete_chains < 0
        or not isinstance(gold_files, Mapping)
        or set(gold_files) != {task for task, _name in EXPERT_SOURCES.values()}
    ):
        raise ValueError("partial Gold export contract is incomplete")

    if output_dir.exists():
        return _verify_existing_snapshot(
            output_dir,
            source_manifest_sha256=source_manifest_sha256,
            per_expert=per_expert,
            seed=seed,
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    source_bindings: dict[str, dict[str, Any]] = {}
    selected_ids: set[str] = set()
    for expert, (task, filename) in EXPERT_SOURCES.items():
        item = gold_files.get(task)
        path = export_dir / filename
        if (
            not isinstance(item, Mapping)
            or item.get("path") != filename
            or Path(filename).name != filename
            or not path.is_file()
            or path.is_symlink()
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"partial Gold source binding failed for {expert}")
        try:
            validation = validate_jsonl(path, allowed_experts=[expert])
            records = [record for _line, record in iter_jsonl(path)]
        except (DatasetValidationError, OSError, ValueError) as exc:
            raise ValueError(f"partial Gold source validation failed for {expert}") from exc
        if (
            validation.get("ok") is not True
            or item.get("records") != validation.get("valid_records")
            or len(records) < per_expert
        ):
            raise ValueError(
                f"{expert} has {len(records)} strict-Gold rows; {per_expert} required"
            )
        for record in records:
            provenance = record.get("provenance")
            if contains_secret_material(record):
                raise ValueError(f"partial Gold safety scan failed for {expert}")
            if (
                isinstance(provenance, Mapping)
                and provenance.get("source_kind") == "swebench_heldout"
            ):
                raise ValueError(f"held-out source found in partial Gold for {expert}")
        chosen = sorted(records, key=lambda row: _selection_key(row, seed))[
            :per_expert
        ]
        identifiers = {str(row["id"]) for row in chosen}
        if selected_ids.intersection(identifiers):
            raise ValueError("cross-expert duplicate ids in selected partial Gold")
        selected_ids.update(identifiers)
        selected[expert] = chosen
        source_bindings[expert] = {
            "records": len(records),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        files: dict[str, dict[str, Any]] = {}
        for expert, (_task, filename) in EXPERT_SOURCES.items():
            destination = temporary / filename
            destination.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in selected[expert]
                ),
                encoding="utf-8",
                newline="\n",
            )
            validation = validate_jsonl(destination, allowed_experts=[expert])
            if validation.get("valid_records") != per_expert:
                raise ValueError("partial snapshot copy validation failed")
            files[expert] = {
                "path": filename,
                "records": per_expert,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "source_records": source_bindings[expert]["records"],
                "source_bytes": source_bindings[expert]["bytes"],
                "source_sha256": source_bindings[expert]["sha256"],
            }

        if sha256_file(source_manifest_path) != source_manifest_sha256:
            raise ValueError("partial Gold export changed during freeze")
        for expert, (_task, filename) in EXPERT_SOURCES.items():
            source_path = export_dir / filename
            if sha256_file(source_path) != source_bindings[expert]["sha256"]:
                raise ValueError("partial Gold source changed during freeze")

        manifest: dict[str, Any] = {
            "schema_version": PARTIAL_SNAPSHOT_SCHEMA,
            "created_at": _utc_now(),
            "training_mode": TRAINING_MODE,
            "not_for_end_to_end_claim": True,
            "selection": SELECTION,
            "seed": seed,
            "per_expert": per_expert,
            "total_records": per_expert * len(EXPERT_SOURCES),
            "source_export_schema_version": EXPORT_SCHEMA_VERSION,
            "source_export_manifest_sha256": source_manifest_sha256,
            "source_partition_manifest_sha256": source[
                "partition_manifest_sha256"
            ],
            "strict_complete_chains": strict_complete_chains,
            "waivers": source_manifest.get("waivers"),
            "excluded": {name: True for name in sorted(REQUIRED_EXCLUSIONS)},
            "snapshot_sha256": _snapshot_digest(files),
            "files": files,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_sha256 = sha256_file(manifest_path)
        (temporary / "manifest.json.sha256").write_text(
            f"{manifest_sha256}  manifest.json\n",
            encoding="ascii",
            newline="\n",
        )
        try:
            os.replace(temporary, output_dir)
        except FileExistsError:
            return _verify_existing_snapshot(
                output_dir,
                source_manifest_sha256=source_manifest_sha256,
                per_expert=per_expert,
                seed=seed,
            )
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Freeze balanced 128/expert partial strict-Gold snapshot"
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--per-expert", type=int, default=DEFAULT_RECORDS_PER_EXPERT
    )
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args(argv)
    try:
        manifest = freeze_partial_gold_snapshot(
            args.export_dir,
            args.output_dir,
            per_expert=args.per_expert,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": manifest["schema_version"],
                "training_mode": manifest["training_mode"],
                "not_for_end_to_end_claim": True,
                "per_expert": manifest["per_expert"],
                "total_records": manifest["total_records"],
                "snapshot_sha256": manifest["snapshot_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
