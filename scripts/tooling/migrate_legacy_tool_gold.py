from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a legacy mixed tool-gold file into an all-attempt audit ledger "
            "and accepted-only gold without deleting or replacing the source."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "live_gold.jsonl",
    )
    parser.add_argument(
        "--attempts-output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tooling"
        / "live_attempts.migrated.jsonl",
    )
    parser.add_argument(
        "--accepted-output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tooling"
        / "live_gold.accepted.jsonl",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Create new outputs. The source is always retained unchanged.",
    )
    return parser.parse_args()


def _canonical_rows(path: Path) -> tuple[str, ...]:
    rows: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            loaded = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(loaded, dict) or not str(loaded.get("sample_id", "")).strip():
            raise ValueError(f"missing sample_id at {path}:{line_number}")
        rows.append(
            json.dumps(loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return tuple(rows)


def _accepted(line: str) -> bool:
    loaded = json.loads(line)
    outcome = loaded.get("public_outcome")
    return loaded.get("success") is True and isinstance(outcome, dict) and (
        outcome.get("status") == "completed"
    )


def _write_new(path: Path, rows: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing migration output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        if rows:
            handle.write("\n".join(rows) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    attempts_output = args.attempts_output.resolve()
    accepted_output = args.accepted_output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if len({source, attempts_output, accepted_output}) != 3:
        raise ValueError("source, attempts output, and accepted output must be distinct")

    source_before = source.read_bytes()
    rows = _canonical_rows(source)
    accepted = tuple(row for row in rows if _accepted(row))
    rejected_count = len(rows) - len(accepted)
    print(
        f"source={source} attempts={len(rows)} accepted={len(accepted)} "
        f"quarantined={rejected_count}"
    )
    print(f"attempts_output={attempts_output}")
    print(f"accepted_output={accepted_output}")
    if not args.confirm:
        print("DRY RUN: no files written; add --confirm after reviewing these paths.")
        return 0

    _write_new(attempts_output, rows)
    _write_new(accepted_output, accepted)
    if source.read_bytes() != source_before:
        raise RuntimeError("legacy source changed during migration")
    print("Migration outputs created; legacy source retained unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
