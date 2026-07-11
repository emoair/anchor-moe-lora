from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import (  # noqa: E402
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SampleSpec,
    ToolingHarness,
    write_gold_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline OpenCode gold-layer mock")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tooling-mock",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "mock_gold.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.workspace_root / "_source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "package.json").write_text(
        json.dumps(
            {
                "name": "anchor-tooling-mock",
                "private": True,
                "scripts": {
                    "build": "node -e \"process.exit(0)\"",
                    "test": "node -e \"process.exit(0)\"",
                    "lint": "node -e \"process.exit(0)\"",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    executor = MockAgentExecutor(
        file_updates={"src/generated.js": "export const ok = true;\n"},
        public_outcome=PublicOutcome(
            status="completed",
            decision_trace=(
                PublicDecisionStep(
                    check="Offline fixture validation",
                    evidence="Mock update completed and local scripts were available",
                    action="Kept the isolated fixture change",
                ),
            ),
            repair_summaries=(),
            final_summary="Offline mock completed.",
        ),
    )
    record = ToolingHarness(args.workspace_root / "samples", executor).run_sample(
        SampleSpec("mock-001", "Create a minimal module", source, ("build", "test", "lint"))
    )
    write_gold_jsonl([record], args.output)
    print(args.output)
    return 0 if record.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
