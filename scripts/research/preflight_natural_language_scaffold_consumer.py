#!/usr/bin/env python3
"""Run the model-free natural-language scaffold consumer preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anchor_mvp.research.natural_language_scaffold_consumer import (  # noqa: E402
    bind_scaffolds_to_taskboard,
    build_bound_scaffold_view,
    load_natural_language_scaffold_fixture,
    paired_bound_ablation_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate scaffold bytes and prove TaskBoard, pair, and "
            "two-request boundaries without loading a model."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "fixtures/research/swebench_natural_language_scaffold",
    )
    parser.add_argument(
        "--taskboard-root",
        type=Path,
        default=REPO_ROOT / "fixtures/research/taskboard_projector",
    )
    parser.add_argument(
        "--consumer-config",
        type=Path,
        default=REPO_ROOT
        / "configs/research/natural_language_scaffold_consumer_v1.yaml",
    )
    parser.add_argument(
        "--expected-consumer-config-sha256",
        required=True,
        help="Required lowercase SHA-256 of the authoritative consumer config bytes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture = load_natural_language_scaffold_fixture(
        args.artifact_root,
        repo_root=REPO_ROOT,
        consumer_config_path=args.consumer_config,
        expected_consumer_config_sha256=args.expected_consumer_config_sha256,
    )
    bound = bind_scaffolds_to_taskboard(fixture, args.taskboard_root)
    views = tuple(build_bound_scaffold_view(item) for item in bound)
    ablation = paired_bound_ablation_summary(views)
    summary = {
        "schema_version": "anchor.natural-language-scaffold-consumer-preflight.v1",
        "status": "contract_preflight_passed",
        "consumer_config_sha256": fixture.summary["consumer_config_sha256"],
        "manifest_sha256": fixture.summary["manifest_sha256"],
        "records": fixture.summary["records"],
        "bound_taskboard_records": len(bound),
        "materialized_contract_views": len(views),
        "pairs": ablation["pairs"],
        "paired_inputs_identical": ablation["paired_inputs_identical"],
        "request1_candidate_only": True,
        "request2_eligible": False,
        "training_authorized": False,
        "quality_validated": False,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
