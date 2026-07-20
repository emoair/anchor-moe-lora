"""Atomic, content-free progress reporting for long local training phases."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4


_WINDOWS_REPLACE_RETRY_ATTEMPTS = 8
_WINDOWS_REPLACE_RETRY_INITIAL_SECONDS = 0.025
_WINDOWS_REPLACE_RETRY_MAX_SECONDS = 0.4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_transient_windows_replace_error(exc: OSError) -> bool:
    """Return whether a replace failed because Windows temporarily locked a file."""

    return getattr(exc, "winerror", None) in {5, 32}


def _atomic_replace_text(path: Path, text: str) -> None:
    """Publish text atomically, tolerating short-lived Windows reader locks.

    Windows does not allow ``os.replace`` while another process holds the
    destination without delete sharing. Progress dashboards and filesystem
    scanners can therefore race a training status update. Each write uses its
    own same-directory temporary file and retries only the two transient
    Win32 lock errors; every other failure remains fail-closed.
    """

    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    delay = _WINDOWS_REPLACE_RETRY_INITIAL_SECONDS
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_WINDOWS_REPLACE_RETRY_ATTEMPTS):
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                if (
                    not _is_transient_windows_replace_error(exc)
                    or attempt == _WINDOWS_REPLACE_RETRY_ATTEMPTS - 1
                ):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, _WINDOWS_REPLACE_RETRY_MAX_SECONDS)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Never hide the original publication error with best-effort
            # cleanup of this writer's uniquely named temporary file.
            pass


class TrainingProgress:
    """Write one atomic status plus append-only events without dataset content."""

    def __init__(self, output_dir: Path) -> None:
        self.state_dir = output_dir.parent / f"{output_dir.name}.progress"
        self.status_path = self.state_dir / "status.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.sequence = 0
        self.run_id = uuid4().hex

    def emit(
        self,
        phase: str,
        state: str,
        *,
        step: int | None = None,
        loss: float | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.sequence += 1
        event: dict[str, Any] = {
            "schema_version": "anchor.training-progress.v1",
            "sequence": self.sequence,
            "run_id": self.run_id,
            "time": _now(),
            "phase": str(phase),
            "state": str(state),
            "step": step,
            "loss": loss,
            "detail": dict(detail or {}),
        }
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
        _atomic_replace_text(self.status_path, encoded + "\n")
        print(encoded, flush=True)
        return event
