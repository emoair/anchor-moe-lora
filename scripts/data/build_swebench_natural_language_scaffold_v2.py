#!/usr/bin/env python3
"""Freeze bundle profiles or materialize the additive v2 training view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.natural_language_scaffold_v2 import (  # noqa: E402
    NaturalLanguageScaffoldV2Error,
    freeze_bundle_profiles,
    materialize_training_view,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build body-filtered frozen-prefix Q-reader v2 artifacts. "
            "All modes are offline and non-authorizing."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-bundle-profile",
        help="freeze explicit body-free bundle descriptors",
    )
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--projector-dir", type=Path, required=True)
    freeze.add_argument("--projector-manifest-sha256", required=True)
    freeze.add_argument("--descriptor-jsonl", type=Path, required=True)
    freeze.add_argument("--descriptor-sha256", required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="materialize one concise-rationale-plus-JSON view per role",
    )
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--projector-dir", type=Path, required=True)
    materialize.add_argument("--projector-manifest-sha256", required=True)
    materialize.add_argument("--bundle-profile-dir", type=Path, required=True)
    materialize.add_argument("--bundle-profile-manifest-sha256", required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze-bundle-profile":
            result = freeze_bundle_profiles(
                args.config,
                args.projector_dir,
                args.projector_manifest_sha256,
                args.descriptor_jsonl,
                args.descriptor_sha256,
                args.output_dir,
            )
        else:
            result = materialize_training_view(
                args.config,
                args.projector_dir,
                args.projector_manifest_sha256,
                args.bundle_profile_dir,
                args.bundle_profile_manifest_sha256,
                args.output_dir,
            )
    except NaturalLanguageScaffoldV2Error as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        # Never echo a malformed source line, a protected body, or a path.
        print("scaffold_v2_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
