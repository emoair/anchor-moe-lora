from __future__ import annotations

import json
import hashlib
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


def is_accepted_gold_record(record: GoldRecord) -> bool:
    """Return whether a record is eligible for accepted training gold."""

    return bool(
        record.success
        and record.public_outcome is not None
        and record.public_outcome.status == "completed"
    )


def _is_accepted_mapping(record: object) -> bool:
    if not isinstance(record, dict) or record.get("success") is not True:
        return False
    outcome = record.get("public_outcome")
    return isinstance(outcome, dict) and outcome.get("status") == "completed"


def _require_accepted(records: Iterable[GoldRecord]) -> tuple[GoldRecord, ...]:
    materialized = tuple(records)
    rejected = [record.sample_id for record in materialized if not is_accepted_gold_record(record)]
    if rejected:
        raise ValueError(
            "refusing non-accepted records in gold: " + ", ".join(sorted(rejected))
        )
    return materialized


def write_gold_jsonl(records: Iterable[GoldRecord], path: str | Path) -> Path:
    accepted = _require_accepted(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(accepted, key=lambda item: item.sample_id)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=destination.parent
    ) as handle:
        temporary = Path(handle.name)
        for record in ordered:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, destination)
    return destination


def write_attempts_jsonl(records: Iterable[GoldRecord], path: str | Path) -> Path:
    """Atomically write audit attempts, including failed and partial records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: (item.sample_id, item.workspace_id))
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

    accepted = _require_accepted(records)
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
            if not _is_accepted_mapping(loaded):
                raise ValueError(
                    f"existing gold contains a non-accepted record at "
                    f"{destination}:{line_number}; migrate it to the attempt ledger"
                )
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
    for record in accepted:
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


def merge_attempts_jsonl(records: Iterable[GoldRecord], path: str | Path) -> Path:
    """Atomically append content-addressed execution attempts.

    Unlike accepted gold, multiple attempts for one sample ID are retained. Replaying an
    identical canonical attempt is idempotent.
    """

    destination = Path(path)
    existing_lines: list[str] = []
    fingerprints: set[str] = set()
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
                    f"invalid existing attempt JSONL at {destination}:{line_number}"
                ) from exc
            if not isinstance(loaded, dict) or not str(loaded.get("sample_id", "")).strip():
                raise ValueError(
                    f"missing sample_id in existing attempt at {destination}:{line_number}"
                )
            canonical = json.dumps(
                loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                existing_lines.append(canonical)

    additions: list[tuple[str, str, str]] = []
    for record in records:
        encoded = canonical_json(record)
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        additions.append((record.sample_id, record.workspace_id, encoded))

    if not additions:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged = existing_lines + [line for _, _, line in sorted(additions)]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=destination.parent
    ) as handle:
        temporary = Path(handle.name)
        handle.write("\n".join(merged) + "\n")
    os.replace(temporary, destination)
    return destination


def persist_attempts_and_gold(
    records: Iterable[GoldRecord],
    *,
    attempts_path: str | Path,
    gold_path: str | Path,
) -> tuple[GoldRecord, ...]:
    """Persist every attempt first, then merge only eligible records into gold."""

    materialized = tuple(records)
    merge_attempts_jsonl(materialized, attempts_path)
    accepted = tuple(record for record in materialized if is_accepted_gold_record(record))
    if accepted:
        merge_gold_jsonl(accepted, gold_path)
    return accepted
