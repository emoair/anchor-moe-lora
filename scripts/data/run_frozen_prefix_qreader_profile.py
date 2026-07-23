"""Authenticate or freeze the non-authorizing Q-reader orchestration profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.frozen_prefix_qreader_profile import (  # noqa: E402
    FrozenPrefixQReaderProfileError,
    PROFILE_PATH,
    core_command_metadata,
    freeze_profile,
    preflight_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate a post-canonical-Gold Q-reader profile without "
            "provider, model, GPU, Gold-body, or heldout-body access."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--profile", type=Path, default=PROFILE_PATH)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--profile", type=Path, default=PROFILE_PATH)
    freeze.add_argument("--output-dir", type=Path, required=True)
    describe = commands.add_parser("print-core-command")
    describe.add_argument("--profile", type=Path, default=PROFILE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight_profile(PROJECT_ROOT, args.profile)
        elif args.command == "freeze":
            result = freeze_profile(
                PROJECT_ROOT,
                args.profile,
                args.output_dir,
            )
        else:
            result = core_command_metadata(PROJECT_ROOT, args.profile)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except FrozenPrefixQReaderProfileError as exc:
        print(
            f"frozen-prefix Q-reader profile refused: {exc.code}",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, json.JSONDecodeError):
        print(
            "frozen-prefix Q-reader profile refused: invalid_local_artifact",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
