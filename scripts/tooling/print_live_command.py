from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import OpenCodeExecutor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a keyless OpenCode live command")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-id", default="dry-run")
    args = parser.parse_args()
    command = OpenCodeExecutor().command(
        sample_id=args.sample_id,
        prompt="<PROMPT_FROM_SAMPLE_MANIFEST>",
        workspace=args.workspace.resolve(),
        config_path=args.config.resolve(),
    )
    print(shlex.join(command))
    print("KIMI_CODE_API_KEY is read only from the environment at live-run time.")
    print("No User-Agent override is permitted; OpenCode must retain its real identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
