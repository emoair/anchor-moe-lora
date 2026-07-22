#!/usr/bin/env python3
"""Build the authenticated Qwen request-local trigger companion fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_mvp.swebench.qwen_toy_prerequisite_companion_v2 import (
    QwenToyPrerequisiteCompanionError,
    build_qwen_toy_prerequisite_companion,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/qwen_toy_prerequisite_companion_v2.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_qwen_toy_prerequisite_companion(
            args.repo_root, args.config, args.output
        )
    except QwenToyPrerequisiteCompanionError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "trigger_materialization_ready": manifest["claims"][
                    "trigger_materialization_ready"
                ],
                "coverage_ready_count": manifest["inventory_status"][
                    "coverage_ready_count"
                ],
                "coverage_total": manifest["inventory_status"]["coverage_total"],
                "formal_training_authorized": manifest["proof"][
                    "formal_training_authorized"
                ],
                "provider_requests": manifest["execution"]["provider_requests"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
