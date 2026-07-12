from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.tooling.session_export import (  # noqa: E402
    QuarantineError,
    SessionConversionPolicy,
    append_jsonl,
    convert_controlled_session,
    convert_controlled_session_staging,
    quarantine_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a controlled OpenCode raw export into a safe candidate JSONL"
    )
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--heldout-cases", type=Path, required=True)
    parser.add_argument("--heldout-fixtures-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("strict", "collect"),
        default="strict",
        help="strict readiness conversion or safe collect-first staging conversion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_bytes = args.export.read_bytes()
    capture: object = None
    try:
        export_data = json.loads(export_bytes.decode("utf-8"))
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        if not isinstance(export_data, dict) or not isinstance(capture, dict):
            raise QuarantineError("input_not_object")
        policy = SessionConversionPolicy(
            workspace_root=args.workspace.resolve(),
            heldout_cases=args.heldout_cases.resolve(),
            heldout_fixtures_root=args.heldout_fixtures_root.resolve(),
            heldout_manifest=args.heldout_manifest.resolve(),
        )
        converter = (
            convert_controlled_session_staging
            if args.mode == "collect"
            else convert_controlled_session
        )
        candidate = converter(export_data, capture, policy)
    except (UnicodeDecodeError, json.JSONDecodeError):
        error = QuarantineError("invalid_json_or_encoding")
    except (OSError, ValueError) as caught:
        error = caught if isinstance(caught, QuarantineError) else QuarantineError("conversion_error")
    else:
        append_jsonl(args.output.resolve(), candidate)
        status = "staged" if args.mode == "collect" else "candidate"
        print(json.dumps({"status": status, "sample_id": candidate["sample_id"]}))
        return 0

    sample_id = str(capture.get("sample_id")) if isinstance(capture, dict) else None
    record = quarantine_record(
        sample_id=sample_id,
        code=error.code,
        export_bytes=export_bytes,
    )
    append_jsonl(args.quarantine.resolve(), record)
    print(json.dumps({"status": "quarantined", "reason_code": error.code}))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
