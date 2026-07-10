"""Atomic, content-free progress reporting for long local training phases."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(self.status_path)
        print(encoded, flush=True)
        return event
