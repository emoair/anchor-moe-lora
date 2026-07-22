#!/usr/bin/env python3
"""Build the authenticated synthetic natural-language scaffold fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.natural_language_scaffold import (  # noqa: E402
    NaturalLanguageScaffoldConfig,
    NaturalLanguageScaffoldError,
    build_natural_language_scaffold,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a body-filtered, paired natural-language scaffold artifact "
            "from an authenticated TaskBoard projector artifact. This command "
            "does not load a model or issue provider requests."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--projector-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, _inventory = NaturalLanguageScaffoldConfig.load(args.config)
        result = build_natural_language_scaffold(
            config,
            args.projector_dir,
            args.projector_manifest_sha256,
            args.output_dir,
        )
    except NaturalLanguageScaffoldError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        # The CLI is an artifact boundary. Do not leak source bodies or paths in
        # an unexpected traceback.
        print("natural_language_scaffold_internal_error", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
