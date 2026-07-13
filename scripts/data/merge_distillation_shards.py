from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.data.shard_merge import (  # noqa: E402
    ShardMergeError,
    merge_distillation_shards,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed atomic merge for distillation JSONL shards."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--shard",
        type=Path,
        action="append",
        required=True,
        help="Shard directory; repeat for deterministic merge order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and print the content-free manifest without writing output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = merge_distillation_shards(
            base_dir=args.base,
            shard_dirs=args.shard,
            target_dir=args.output,
            dry_run=args.dry_run,
        )
    except (OSError, ShardMergeError) as error:
        print(f"merge refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
