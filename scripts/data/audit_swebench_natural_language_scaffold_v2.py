#!/usr/bin/env python3
"""Authenticate a published frozen-prefix Q-reader v2 training view."""

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
    audit_bundle_profiles,
    audit_training_view,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a v2 training-view artifact without loading a model, "
            "issuing provider requests, or reading heldout data."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("training-view", "bundle-profile"),
        default="training-view",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit = (
            audit_bundle_profiles
            if args.kind == "bundle-profile"
            else audit_training_view
        )
        result = audit(args.config, args.artifact_dir, args.manifest_sha256)
    except NaturalLanguageScaffoldV2Error as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("scaffold_v2_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
