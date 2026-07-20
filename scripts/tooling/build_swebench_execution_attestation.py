"""Generate one deterministic, content-free SWE-bench v3 attestation.

The generator never downloads a harness or image.  It re-runs every available
local probe and writes ``ready=false`` with explicit remaining gates whenever
the locked environment is incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import os


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.tooling.swebench_execution_v3 import (  # noqa: E402
    ExecutionContractError,
    build_execution_attestation,
)


def _project_path(value: str, label: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ExecutionContractError(f"{label}_path_escape") from exc
    return candidate


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe local SWE-bench v3 execution gates without network access"
    )
    parser.add_argument(
        "--lock",
        default="configs/tooling/swebench_execution_v3.lock.json",
    )
    parser.add_argument(
        "--output",
        help="Optional project-relative attestation path; stdout is always content-free",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lock = _project_path(args.lock, "lock")
        report = build_execution_attestation(ROOT, lock)
        if args.output:
            _atomic_write(_project_path(args.output, "output"), report)
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "ready": report["ready"],
                    "remaining_gates": report["remaining_gates"],
                    "content_free": True,
                    "attestation_written": bool(args.output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if report["ready"] else 3
    except (OSError, ValueError, json.JSONDecodeError, ExecutionContractError) as exc:
        print(
            json.dumps(
                {
                    "ready": False,
                    "reason_code": str(exc) or "execution_attestation_probe_failed",
                    "content_free": True,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
