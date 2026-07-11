from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_mvp.curriculum import run_fixture_contracts, validate_curriculum


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the collaboration curriculum v2")
    parser.add_argument(
        "--manifest", default="configs/curriculum/collaboration_v2.yaml", type=Path
    )
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--run-fixtures", action="store_true")
    args = parser.parse_args()
    manifest = validate_curriculum(args.manifest, args.repo_root)
    runs = run_fixture_contracts(manifest, args.repo_root) if args.run_fixtures else ()
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "candidate_count": len(manifest["tasks"]),
                "fixture_commands_checked": len(runs),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
