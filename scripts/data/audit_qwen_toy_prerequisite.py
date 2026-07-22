#!/usr/bin/env python3
"""Audit a published metadata-only Qwen toy prerequisite fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_mvp.swebench.qwen_toy_prerequisite import (
    QwenToyPrerequisiteError,
    audit_qwen_toy_prerequisite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/qwen_toy_prerequisite_v1.json"),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = audit_qwen_toy_prerequisite(
            args.repo_root, args.config, args.artifact
        )
    except QwenToyPrerequisiteError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_status": manifest["status"],
                "record_count": manifest["toy"]["record_count"],
                "coverage_ready_count": manifest["proof"]["coverage_ready_count"],
                "coverage_total": manifest["proof"]["coverage_total"],
                "v1_attestation_emitted": manifest["proof"]["v1_attestation_emitted"],
                "formal_training_authorized": manifest["safety"][
                    "formal_training_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
