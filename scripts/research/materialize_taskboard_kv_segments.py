"""Authenticate producer-native Task-KV plans and emit content-free indexes.

This command performs no provider request, model load, tokenizer work, GPU
operation, or physical KV materialization.  The canonical plan remains in the
producer sidecar; output rows contain only authenticated identities and counts.
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

from anchor_mvp.research.query_specialization import QuerySpecializationError  # noqa: E402
from anchor_mvp.research.taskboard_kv_segments import (  # noqa: E402
    CLAIM_SCOPE,
    CONSUMER_CONFIG_SCHEMA_VERSION,
    EXECUTION_MODE,
    FROZEN_CONSUMER_CONFIG_SHA256,
    INDEX_SCHEMA_VERSION,
    NON_CLAIMS,
    PLAN_SCHEMA_VERSION,
    TaskBoardKVSegmentError,
    content_free_plan_summary,
    load_authenticated_taskboard_kv_dataset,
    project_taskboard_kv_segment_plans,
    validate_index_mapping,
)


MANIFEST_SCHEMA_VERSION = "anchor.taskboard-kv-native-consumer-manifest.v1"
DEFAULT_MAX_SHARD_BYTES = 48_000_000
HARD_MAX_FILE_BYTES = 50_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate native hierarchical Task-KV plans and write only a "
            "content-free consumer index"
        )
    )
    parser.add_argument(
        "--input-root",
        default=str(REPO_ROOT / "fixtures/research/taskboard_projector"),
        help="producer sidecar dataset root",
    )
    parser.add_argument(
        "--manifest",
        help="producer manifest (default: <input-root>/manifest.json)",
    )
    parser.add_argument(
        "--producer-config",
        default=str(REPO_ROOT / "configs/research/swebench_taskboard_projector_v2.yaml"),
    )
    parser.add_argument(
        "--manifest-schema",
        default=str(REPO_ROOT / "configs/research/taskboard_projector_manifest.schema.json"),
    )
    parser.add_argument(
        "--sidecar-schema",
        default=str(REPO_ROOT / "configs/research/taskboard_projector_sidecar.schema.json"),
    )
    parser.add_argument(
        "--segment-plan-schema",
        default=str(REPO_ROOT / "configs/research/hierarchical_task_kv_segment_plan.schema.json"),
    )
    parser.add_argument(
        "--consumer-config",
        default=str(REPO_ROOT / "configs/research/hierarchical_task_kv_mvp.yaml"),
        help="training consumer config whose exact bytes are bound into output",
    )
    parser.add_argument(
        "--expected-consumer-config-sha256",
        default=FROZEN_CONSUMER_CONFIG_SHA256,
        help=(
            "expected consumer config SHA (defaults to the frozen checked-in "
            "contract; custom configs must pass their own SHA explicitly)"
        ),
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        help="optional lower ceiling; cannot exceed the authenticated config",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--execute", action="store_true", help="write versioned output")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "runs/research/taskboard-kv-native-index"),
    )
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _shard_lines(lines: Sequence[bytes], max_bytes: int) -> tuple[bytes, ...]:
    shards: list[bytes] = []
    current: list[bytes] = []
    current_bytes = 0
    for line in lines:
        if len(line) > max_bytes:
            raise TaskBoardKVSegmentError(
                "one content-free index row exceeds the shard ceiling"
            )
        if current and current_bytes + len(line) > max_bytes:
            shards.append(b"".join(current))
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += len(line)
    if current:
        shards.append(b"".join(current))
    return tuple(shards)


def _resolve_max_shard_bytes(explicit: int | None, authenticated: int) -> int:
    result = authenticated if explicit is None else explicit
    if (
        isinstance(result, bool)
        or not isinstance(result, int)
        or result < 1
        or result >= HARD_MAX_FILE_BYTES
    ):
        raise TaskBoardKVSegmentError(
            "max_shard_bytes must be positive and below 50,000,000"
        )
    if result > authenticated:
        raise TaskBoardKVSegmentError(
            "max_shard_bytes cannot exceed the authenticated consumer config"
        )
    return result


def build_materialization(
    *,
    sidecars: Sequence[Any],
    source_manifest: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    materializer_sha256: str,
    max_shard_bytes: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build deterministic index shards; canonical plans are not copied."""

    plans = project_taskboard_kv_segment_plans(sidecars)
    grouped: dict[tuple[str, str], list[bytes]] = {}
    for plan in plans:
        value = plan.to_dict()
        validate_index_mapping(value)
        grouped.setdefault((plan.split, plan.variant), []).append(
            _canonical_line(value)
        )

    payloads: dict[str, bytes] = {}
    files: list[dict[str, Any]] = []
    for split, variant in (
        ("train", "clean"),
        ("train", "noisy"),
        ("calibration", "clean"),
    ):
        lines = grouped.get((split, variant), [])
        if not lines:
            raise TaskBoardKVSegmentError(
                f"authenticated dataset omitted {split}/{variant} indexes"
            )
        for index, payload in enumerate(_shard_lines(lines, max_shard_bytes)):
            relative = f"{split}/{variant}-part-{index:05d}.jsonl"
            if len(payload) >= HARD_MAX_FILE_BYTES:
                raise TaskBoardKVSegmentError("index shard reached the hard file limit")
            payloads[relative] = payload
            files.append(
                {
                    "path": relative,
                    "split": split,
                    "variant": variant,
                    "records": payload.count(b"\n"),
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )

    source_counts = source_manifest.get("counts")
    if not isinstance(source_counts, Mapping):
        raise TaskBoardKVSegmentError("producer manifest counts contract changed")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "native_segment_plan_schema_version": PLAN_SCHEMA_VERSION,
        "execution_mode": EXECUTION_MODE,
        "claim_scope": CLAIM_SCOPE,
        "source_plan_location": "outer_sidecar.segment_plan",
        "source_projector_manifest_sha256": source_validation[
            "source_manifest_sha256"
        ],
        "producer_contract": {
            "config_sha256": source_validation["producer_config_sha256"],
            "manifest_schema_sha256": source_validation[
                "producer_manifest_schema_sha256"
            ],
            "sidecar_schema_sha256": source_validation[
                "producer_sidecar_schema_sha256"
            ],
            "segment_plan_schema_sha256": source_validation[
                "producer_segment_plan_schema_sha256"
            ],
        },
        "consumer_contract": {
            "schema_version": CONSUMER_CONFIG_SCHEMA_VERSION,
            "config_sha256": source_validation["consumer_config_sha256"],
            "materializer_sha256": materializer_sha256,
        },
        "source_contract": {
            "records": source_counts.get("total"),
            "task_bundles": source_counts.get("unique_task_bundles"),
            "split_group_key": source_manifest.get("split_group_key"),
            "task_id_cross_binding_key": source_manifest.get(
                "task_id_cross_binding_key"
            ),
            "all_five_role_views_same_split": source_manifest.get(
                "all_five_role_views_same_split"
            ),
        },
        "index_summary": content_free_plan_summary(plans),
        "max_shard_bytes": max_shard_bytes,
        "hard_max_file_bytes_exclusive": HARD_MAX_FILE_BYTES,
        "files": files,
        "canonical_segment_plan_written": False,
        "source_body_written": False,
        "provider_requests": 0,
        "model_loaded": False,
        "gpu_used": False,
        "heldout_content_read": False,
        "canonical_gold_written": False,
        "non_claims": list(NON_CLAIMS),
    }
    return manifest, payloads


def _reauthenticate_before_output(
    *,
    input_root: Path,
    manifest_path: Path,
    validation: Mapping[str, Any],
) -> None:
    """Fail closed if any consumed local bytes changed after validation."""

    manifest_sha = _sha256_bytes(manifest_path.read_bytes())
    if manifest_sha != validation["source_manifest_sha256"]:
        raise TaskBoardKVSegmentError("producer manifest changed before output")
    declaration = manifest_path.with_name(manifest_path.name + ".sha256").read_bytes()
    expected_declaration = f"{manifest_sha}  manifest.json\n".encode("ascii")
    if declaration != expected_declaration:
        raise TaskBoardKVSegmentError("producer manifest SHA sidecar changed")
    source_hashes = validation["source_authenticated_file_sha256"]
    if not isinstance(source_hashes, Mapping):
        raise TaskBoardKVSegmentError("source authentication evidence changed")
    for relative, expected_sha in source_hashes.items():
        path = manifest_path if relative == "manifest.json" else input_root / relative
        if _sha256_bytes(path.read_bytes()) != expected_sha:
            raise TaskBoardKVSegmentError(f"{relative}: source bytes changed before output")
    contract_paths = validation["authenticated_contract_paths"]
    contract_hashes = validation["authenticated_contract_sha256"]
    for name, path_text in contract_paths.items():
        if _sha256_bytes(Path(path_text).read_bytes()) != contract_hashes[name]:
            raise TaskBoardKVSegmentError(
                f"{name}: authenticated contract changed before output"
            )


def _write_versioned(
    output_root: Path,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> Path:
    manifest_payload = _manifest_bytes(manifest)
    if len(manifest_payload) >= HARD_MAX_FILE_BYTES:
        raise TaskBoardKVSegmentError("consumer manifest reached the hard file limit")
    digest = _sha256_bytes(manifest_payload)
    version = f"native-index-{digest}"
    output_dir = output_root / version
    expected = {
        **payloads,
        "manifest.json": manifest_payload,
        "manifest.json.sha256": f"{digest}  manifest.json\n".encode("ascii"),
    }
    if output_dir.exists():
        actual_files = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != set(expected):
            raise TaskBoardKVSegmentError(
                "versioned output contains unexpected or missing files"
            )
        for relative, payload in expected.items():
            target = output_dir / relative
            if not target.is_file() or target.read_bytes() != payload:
                raise TaskBoardKVSegmentError(
                    "versioned output exists with non-identical bytes"
                )
        return output_dir

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".tmp-{version}"
    if staging.exists():
        raise TaskBoardKVSegmentError(
            "stale staging directory requires explicit operator audit"
        )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for relative, payload in expected.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        input_root = Path(args.input_root).expanduser().resolve()
        manifest_path = (
            Path(args.manifest).expanduser().resolve()
            if args.manifest
            else input_root / "manifest.json"
        )
        script_bytes = Path(__file__).read_bytes()
        script_sha = _sha256_bytes(script_bytes)
        sidecars, source_manifest, validation = (
            load_authenticated_taskboard_kv_dataset(
                input_root,
                manifest_path=manifest_path,
                producer_config_path=args.producer_config,
                manifest_schema_path=args.manifest_schema,
                sidecar_schema_path=args.sidecar_schema,
                segment_plan_schema_path=args.segment_plan_schema,
                consumer_config_path=args.consumer_config,
                expected_consumer_config_sha256=(
                    args.expected_consumer_config_sha256
                ),
            )
        )
        max_shard_bytes = _resolve_max_shard_bytes(
            args.max_shard_bytes,
            validation["consumer_config_max_shard_bytes"],
        )
        manifest, payloads = build_materialization(
            sidecars=sidecars,
            source_manifest=source_manifest,
            source_validation=validation,
            materializer_sha256=script_sha,
            max_shard_bytes=max_shard_bytes,
        )
        _reauthenticate_before_output(
            input_root=input_root,
            manifest_path=manifest_path,
            validation=validation,
        )
        if _sha256_bytes(Path(__file__).read_bytes()) != script_sha:
            raise TaskBoardKVSegmentError("materializer changed during execution")
        if args.execute:
            output_dir = _write_versioned(
                Path(args.output_root).expanduser().resolve(), manifest, payloads
            )
            response = {
                "ok": True,
                "mode": "execute",
                "output_dir": str(output_dir),
                "manifest_sha256": _sha256_bytes(_manifest_bytes(manifest)),
                "index_summary": manifest["index_summary"],
                "consumer_config_sha256": validation["consumer_config_sha256"],
            }
        else:
            response = {
                "ok": True,
                "mode": "dry_run",
                "would_write_files": len(payloads) + 2,
                "manifest": manifest,
            }
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))
        return 0
    except (TaskBoardKVSegmentError, QuerySpecializationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
