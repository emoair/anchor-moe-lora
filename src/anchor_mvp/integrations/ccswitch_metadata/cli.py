"""CLI for verifying and installing the pinned CC Switch metadata snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .schema import SchemaError
from .sync import (
    IntegrityError,
    MetadataStore,
    MetadataSyncError,
    default_state_dir,
    resolve_candidate,
    semantic_diff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor-ccswitch-metadata",
        description=(
            "Verify and atomically install a secret-free CC Switch v3.16.5 "
            "metadata snapshot."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "diff", "apply"):
        subparser = subparsers.add_parser(command)
        _common_arguments(subparser, network=True)
    rollback = subparsers.add_parser("rollback")
    _common_arguments(rollback, network=False)
    return parser


def _common_arguments(parser: argparse.ArgumentParser, *, network: bool) -> None:
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=default_state_dir(),
        help="private cache and rollback directory",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="active content-safe JSON path; defaults to STATE_DIR/active.json",
    )
    if network:
        parser.add_argument(
            "--offline",
            action="store_true",
            help="skip GitHub verification and use the last or bundled verified snapshot",
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        store = MetadataStore(arguments.state_dir, arguments.target)
        if arguments.command == "rollback":
            result = store.rollback()
            _emit({"ok": True, "command": "rollback", **result})
            return 0

        candidate = resolve_candidate(
            arguments.state_dir,
            offline=bool(arguments.offline),
        )
        current = store.current()
        difference = semantic_diff(current, candidate.snapshot)
        common = {
            "origin": candidate.origin,
            "verification": candidate.verification,
            "warning": candidate.warning,
            "source_tag": candidate.snapshot["source"]["source_tag"],
            "source_commit": candidate.snapshot["source"]["source_commit"],
            "candidate_sha256": candidate.sha256,
        }
        if arguments.command == "check":
            _emit(
                {
                    "ok": True,
                    "command": "check",
                    **common,
                    "target": str(store.target),
                    "target_exists": current is not None,
                    "target_valid": True if current is not None else None,
                    "counts": {
                        "providers": len(candidate.snapshot["providers"]),
                        "models": len(candidate.snapshot["models"]),
                        "aliases": len(candidate.snapshot["model_aliases"]),
                        "pricing": len(candidate.snapshot["pricing"]),
                    },
                }
            )
            return 0
        if arguments.command == "diff":
            _emit({"ok": True, "command": "diff", **common, "diff": difference})
            return 0
        if arguments.command == "apply":
            result = store.apply(candidate.snapshot)
            _emit(
                {
                    "ok": True,
                    "command": "apply",
                    **common,
                    "diff": difference,
                    "result": result,
                }
            )
            return 0
        raise MetadataSyncError(f"unsupported command: {arguments.command}")
    except (SchemaError, IntegrityError, MetadataSyncError, OSError) as exc:
        _emit(
            {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            stream=sys.stderr,
        )
        return 2


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
    text = text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    print(text, file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
