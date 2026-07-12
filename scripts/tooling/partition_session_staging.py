from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.tooling.session_export import SessionConversionPolicy  # noqa: E402
from anchor_mvp.tooling.session_partition import partition_staging_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition safe OpenCode staging trajectories into gold/negative/reject"
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=ROOT / "artifacts" / "tooling" / "session_staging.raw.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "artifacts" / "tooling" / "session_candidates.filtered.gold.jsonl",
    )
    parser.add_argument(
        "--negative",
        type=Path,
        default=ROOT / "artifacts" / "tooling" / "session_candidates.filtered.negative.jsonl",
    )
    parser.add_argument(
        "--reject",
        type=Path,
        default=ROOT / "artifacts" / "tooling" / "session_partition.reject.jsonl",
    )
    parser.add_argument(
        "--heldout-cases",
        type=Path,
        default=ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-fixtures-root",
        type=Path,
        default=ROOT / "examples" / "benchmark" / "fixtures",
    )
    parser.add_argument(
        "--heldout-manifest",
        type=Path,
        default=ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = partition_staging_jsonl(
        staging_path=args.staging.resolve(),
        gold_path=args.gold.resolve(),
        negative_path=args.negative.resolve(),
        reject_path=args.reject.resolve(),
        policy=SessionConversionPolicy(
            workspace_root=ROOT.resolve(),
            heldout_cases=args.heldout_cases.resolve(),
            heldout_fixtures_root=args.heldout_fixtures_root.resolve(),
            heldout_manifest=args.heldout_manifest.resolve(),
        ),
    )
    print(json.dumps(counts, sort_keys=True))
    return 0 if counts["reject"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
