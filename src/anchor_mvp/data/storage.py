"""Append-only JSONL stores with resume and deduplication support."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

from .schema import DataValidationError, normalized_text, stable_id


class JsonlStore:
    """Crash-tolerant append-only JSONL.

    A malformed final unterminated line is treated as an interrupted append and
    ignored on resume. Malformed complete lines fail loudly instead of silently
    corrupting a training corpus.
    """

    def __init__(self, path: str | Path, *, id_field: str = "id") -> None:
        self.path = Path(path)
        self.id_field = id_field
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines, start=1):
            complete = line.endswith((b"\n", b"\r"))
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if index == len(lines) and not complete:
                    break
                raise DataValidationError(f"invalid JSONL at {self.path}:{index}") from error
            if not isinstance(value, dict):
                raise DataValidationError(f"JSONL record at {self.path}:{index} is not an object")
            record_id = str(value.get(self.id_field, ""))
            if not record_id:
                raise DataValidationError(f"missing {self.id_field} at {self.path}:{index}")
            if record_id in self._ids:
                continue
            self._ids.add(record_id)
            self._records.append(value)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    def append(self, value: Mapping[str, Any]) -> bool:
        record_id = str(value.get(self.id_field, ""))
        if not record_id:
            raise DataValidationError(f"record requires {self.id_field}")
        encoded = (json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            if record_id in self._ids:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            copied = dict(value)
            self._ids.add(record_id)
            self._records.append(copied)
            return True


class SeedStore(JsonlStore):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path, id_field="seed_id")

    @property
    def request_fingerprints(self) -> frozenset[str]:
        return frozenset(
            stable_id("request", normalized_text(str(record.get("request", ""))))
            for record in self.records
        )


def completed_seed_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for record in records:
        provenance = record.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get("seed_id"):
            completed.add(str(provenance["seed_id"]))
    return completed

