from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import (  # noqa: E402
    OpenCodeExecutor,
    SampleSpec,
    ToolPolicy,
    ToolingHarness,
    write_gold_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated OpenCode gold sample")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tooling-live",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "live_gold.jsonl",
    )
    parser.add_argument("--required", nargs="*", default=["build"])
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that one quota-consuming API session may run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executor = OpenCodeExecutor()
    if not args.confirm_live:
        print("DRY RUN: no workspace, OpenCode process, or API request was created.")
        print("Add --confirm-live only after reviewing source, prompt, and policy.")
        print(f"opencode_available={executor.available()}")
        print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
        return 0
    if not executor.available():
        print("OpenCode is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not os.environ.get("KIMI_CODE_API_KEY"):
        print("KIMI_CODE_API_KEY is not set in this process.", file=sys.stderr)
        return 2
    if any(name not in {"build", "test", "lint"} for name in args.required):
        print("--required accepts only build, test, and lint", file=sys.stderr)
        return 2
    policy = ToolPolicy(
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
    )
    prompt = args.prompt_file.read_text(encoding="utf-8")
    record = ToolingHarness(args.workspace_root, executor, policy=policy).run_sample(
        SampleSpec(args.sample_id, prompt, args.source, tuple(args.required))
    )
    write_gold_jsonl([record], args.output)
    print(args.output)
    return 0 if record.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
