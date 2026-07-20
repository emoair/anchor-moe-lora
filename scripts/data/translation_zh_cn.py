from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.data.translation_qa import (  # noqa: E402
    SHARD_NAMES,
    TranslationAuditError,
    merge_translation_shards,
    prepare_translation_shards,
)


SOURCE_DIR = ROOT / "artifacts" / "compact_mvp_v2b" / "candidate_dataset"
REGISTRY = SOURCE_DIR / "manifest.registry-formal-v2.json"
WORK_ROOT = ROOT / "artifacts" / "compact_mvp_v2b" / "translation_zh_cn_v1"
SHARD_DIR = WORK_ROOT / "shards"
OUTPUT_DIR = WORK_ROOT / "candidate_dataset_zh_cn"
EXPECTED_SNAPSHOT_SHA256 = (
    "43f97bca74aac5b747bf8b8a95dd593dcbc3683e892775ec350282a540d5390c"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or audit/merge the offline zh-CN training copy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", description="Write four deterministic, untranslated templates."
    )
    prepare.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    prepare.add_argument("--registry", type=Path, default=REGISTRY)
    prepare.add_argument("--shard-dir", type=Path, default=SHARD_DIR)
    prepare.add_argument(
        "--expected-snapshot-sha256", default=EXPECTED_SNAPSHOT_SHA256
    )

    merge = subparsers.add_parser(
        "merge", description="Fail closed unless all four translated shards pass QA."
    )
    merge.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    merge.add_argument("--registry", type=Path, default=REGISTRY)
    merge.add_argument(
        "--shard",
        type=Path,
        action="append",
        help=(
            "Translated shard; repeat four times. Defaults to part-000..003.jsonl "
            "under --shard-dir."
        ),
    )
    merge.add_argument("--shard-dir", type=Path, default=SHARD_DIR)
    merge.add_argument("--output", type=Path, default=OUTPUT_DIR)
    merge.add_argument(
        "--expected-snapshot-sha256", default=EXPECTED_SNAPSHOT_SHA256
    )
    merge.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and print the manifest without publishing any files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_translation_shards(
                source_dir=args.source_dir,
                registry_path=args.registry,
                shard_dir=args.shard_dir,
                expected_snapshot_sha256=args.expected_snapshot_sha256,
            )
        else:
            shard_paths = args.shard or [
                args.shard_dir / name for name in SHARD_NAMES
            ]
            manifest = merge_translation_shards(
                source_dir=args.source_dir,
                registry_path=args.registry,
                shard_paths=shard_paths,
                output_dir=args.output,
                expected_snapshot_sha256=args.expected_snapshot_sha256,
                dry_run=args.dry_run,
            )
    except (OSError, TranslationAuditError) as error:
        print(f"translation QA refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
