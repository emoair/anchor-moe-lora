"""Build the pinned, train-only SWE-bench full-bank staging artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.full_bank import (  # noqa: E402
    FullBankConfig,
    build_full_bank,
    preflight_full_bank,
    refresh_hash_only_manifest_from_public,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build content-bearing candidate shards and a content-free manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/swebench_full_bank.formal.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "read-only offline readiness check; do not build or rewrite any dataset file"
        ),
    )
    mode.add_argument(
        "--refresh-hash-only-from-public",
        action="store_true",
        help=(
            "refresh only the content-free bank snapshot from the audited public "
            "manifest; never parse payload JSONL or source parquet"
        ),
    )
    parser.add_argument(
        "--require-launch-ready",
        action="store_true",
        help="return exit code 4 unless the launch-only gate is ready",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = FullBankConfig.load(ROOT, config_path)
    if args.preflight:
        report = preflight_full_bank(config)
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "offline": report["offline"],
                    "provider_requests": report["provider_requests"],
                    "launch_ready": report["launch_ready"],
                    "training_ready": report["training_ready"],
                    "publication_ready": report["publication_ready"],
                    "missing_gates": {
                        name: group["missing"]
                        for name, group in report["gates"].items()
                    },
                    "invalid_gates": {
                        name: group["invalid"]
                        for name, group in report["gates"].items()
                    },
                },
                sort_keys=True,
            )
        )
        if args.require_launch_ready and not report["launch_ready"]:
            return 4
        return 0
    if args.refresh_hash_only_from_public:
        result = refresh_hash_only_manifest_from_public(config)
        manifest = result.manifest
        publication = manifest["publication"]
        counts = publication["counts"]
        print(
            json.dumps(
                {
                    "manifest": str(result.manifest_path),
                    "public_manifest_sha256": publication[
                        "public_manifest_sha256"
                    ],
                    "payload_files": publication["payload_file_count"],
                    "tasks": counts["tasks"],
                    "work_orders": counts["work_orders"],
                    "publication_ready": manifest["publication_ready"],
                },
                sort_keys=True,
            )
        )
        if args.require_launch_ready and not manifest["launch_ready"]:
            return 4
        return 0
    result = build_full_bank(config)
    manifest = result.manifest
    # Intentionally print counts, hashes, status, and paths only; never task bodies.
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "source_rows": manifest["source"]["row_count"],
                "repositories": manifest["source"]["repository_count"],
                "train": manifest["derived_split"]["train_count"],
                "validation": manifest["derived_split"]["validation_count"],
                "work_orders": manifest["routing"]["work_order_count"],
                "launch_ready": manifest["launch_ready"],
                "training_ready": manifest["training_ready"],
                "publication_ready": manifest["publication_ready"],
                "missing_gates": manifest["missing_gates"],
            },
            sort_keys=True,
        )
    )
    if args.require_launch_ready and not manifest["launch_ready"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
