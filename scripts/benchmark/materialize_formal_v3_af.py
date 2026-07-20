from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.benchmark.formal_v3_registry import (  # noqa: E402
    FormalV3RegistryError,
    finalize_bundle,
    inspect_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or exclusively finalize a formal-v3 A-F evaluation bundle; "
            "never reads heldout cases and never starts a model/API/GPU"
        )
    )
    parser.add_argument(
        "--control",
        default="configs/benchmark/formal_v3_af_control.json",
    )
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="write a new immutable registry/benchmark bundle after all gates pass",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    control = Path(args.control)
    if not control.is_absolute():
        control = root / control
    if not args.finalize:
        result = inspect_readiness(control, root, version_id=args.version_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "READY_TO_FINALIZE" else 3
    output = args.output_dir or (
        f"artifacts/formal_v3/evaluation/registries/{args.version_id}"
    )
    try:
        result = finalize_bundle(
            control,
            root,
            version_id=args.version_id,
            output_dir=output,
        )
    except FormalV3RegistryError as exc:
        result = {
            "schema_version": "anchor.formal-v3-af-readiness.v1",
            "status": "BLOCKED",
            "code": exc.code,
            "message": str(exc),
            "details": exc.details,
            "heldout_case_content_read": False,
            "gpu_started": False,
            "api_called": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
