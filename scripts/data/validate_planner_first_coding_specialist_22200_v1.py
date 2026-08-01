"""Validate the non-live planner-first 22,200 campaign contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from anchor_mvp.data.planner_first_coding_specialist_22200_v1 import (
    CampaignValidationError,
    expert_projection_decision,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "configs/data/planner_first_coding_specialist_22200_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the body-free, planner-first 22,200 campaign gate."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        decision = expert_projection_decision(args.contract)
    except CampaignValidationError as error:
        print(f"planner-first-22200: {error}", file=sys.stderr)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision["status"] == "ready_for_planner_question_commit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
