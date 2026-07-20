#!/usr/bin/env python3
"""Publish deterministic TaskBoard research sidecars from frozen formal Gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.taskboard_projector import (  # noqa: E402
    TaskBoardProjectorConfig,
    TaskBoardProjectorError,
    project_taskboards,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project an authenticated training snapshot into TaskBoard sidecars."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = TaskBoardProjectorConfig.load(args.config)
        metadata = project_taskboards(
            config,
            args.snapshot_dir,
            args.snapshot_manifest_sha256,
            args.output_dir,
        )
    except TaskBoardProjectorError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("projector_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
