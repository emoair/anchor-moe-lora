"""Authenticate or freeze a non-authorizing distillation pipeline profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.distillation_profile import (  # noqa: E402
    DistillationProfileError,
    PROFILE_PATH,
    freeze_profile,
    preflight_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate a post-canonical-Gold pipeline profile without "
            "provider, model, GPU, Gold-body, or heldout-body access."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--profile", type=Path, default=PROFILE_PATH)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--profile", type=Path, default=PROFILE_PATH)
    freeze.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight_profile(PROJECT_ROOT, args.profile)
        else:
            result = freeze_profile(
                PROJECT_ROOT,
                args.profile,
                args.output_dir,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except DistillationProfileError as exc:
        print(
            f"distillation profile refused: {exc.code}",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, json.JSONDecodeError):
        print(
            "distillation profile refused: invalid_local_artifact",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
