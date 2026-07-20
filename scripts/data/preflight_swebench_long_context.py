#!/usr/bin/env python3
"""Publish an authenticated, body-free long-context token inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.long_context_preflight import (  # noqa: E402
    LocalTransformersTokenCounter,
    LongContextPreflightError,
    LongContextTokenInventoryConfig,
    SyntheticFixtureTokenCounter,
    build_long_context_token_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact-token, body-free inventory from an authenticated "
            "TaskBoard projector artifact."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--projector-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    tokenizer = parser.add_mutually_exclusive_group(required=True)
    tokenizer.add_argument("--tokenizer-dir", type=Path)
    tokenizer.add_argument(
        "--synthetic-fixture-tokenizer",
        action="store_true",
        help="Use the explicit test-only one-token-per-UTF-8-byte contract.",
    )
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, _inventory = LongContextTokenInventoryConfig.load(args.config)
        if args.synthetic_fixture_tokenizer:
            if args.tokenizer_id is not None or args.tokenizer_revision is not None:
                raise LongContextPreflightError(
                    "long_context_synthetic_tokenizer_arguments_invalid"
                )
            counter = SyntheticFixtureTokenCounter()
        else:
            if (
                args.tokenizer_dir is None
                or args.tokenizer_id is None
                or args.tokenizer_revision is None
            ):
                raise LongContextPreflightError(
                    "long_context_tokenizer_identity_required"
                )
            counter = LocalTransformersTokenCounter(
                args.tokenizer_dir,
                tokenizer_id=args.tokenizer_id,
                tokenizer_revision=args.tokenizer_revision,
                max_asset_bytes=config.max_tokenizer_asset_bytes,
            )
        result = build_long_context_token_inventory(
            config,
            args.projector_dir,
            args.projector_manifest_sha256,
            args.output_dir,
            counter=counter,
        )
    except LongContextPreflightError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("long_context_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
