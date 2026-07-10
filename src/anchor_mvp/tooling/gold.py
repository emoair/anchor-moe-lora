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
