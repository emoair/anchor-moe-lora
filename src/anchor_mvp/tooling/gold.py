from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .models import GoldRecord


def canonical_json(record: GoldRecord) -> str:
    return json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_gold_jsonl(records: Iterable[GoldRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: item.sample_id)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=destination.parent
    ) as handle:
        temporary = Path(handle.name)
        for record in ordered:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, destination)
    return destination


def merge_gold_jsonl(records: Iterable[GoldRecord], path: str | Path) -> Path:
    """Atomically add immutable records without replacing prior sample IDs.

    Replaying an identical record is idempotent. A differing record with an existing
    sample ID is a hard conflict, preventing a later batch from silently rewriting gold.
    """

    destination = Path(path)
    existing_lines: list[str] = []
    by_id: dict[str, str] = {}
    if destination.exists():
        for line_number, raw_line in enumerate(
            destination.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid existing gold JSONL at {destination}:{line_number}"
                ) from exc
            sample_id = str(loaded.get("sample_id", "")) if isinstance(loaded, dict) else ""
            if not sample_id:
                raise ValueError(
                    f"missing sample_id in existing gold at {destination}:{line_number}"
                )
            canonical = json.dumps(
                loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if sample_id in by_id and by_id[sample_id] != canonical:
                raise ValueError(f"conflicting existing gold record: {sample_id}")
            if sample_id not in by_id:
                existing_lines.append(canonical)
                by_id[sample_id] = canonical

    additions: list[tuple[str, str]] = []
    for record in records:
        encoded = canonical_json(record)
        previous = by_id.get(record.sample_id)
        if previous is not None:
            if previous != encoded:
                raise ValueError(f"refusing to overwrite gold record: {record.sample_id}")
            continue
        by_id[record.sample_id] = encoded
        additions.append((record.sample_id, encoded))

    if not additions:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged = existing_lines + [line for _, line in sorted(additions)]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=destination.parent
    ) as handle:
        temporary = Path(handle.name)
        handle.write("\n".join(merged) + "\n")
    os.replace(temporary, destination)
    return destination
