"""Reliability primitives for the additive Gemma 3 Q8 QLoRA v2 runner.

This module intentionally has no dataset-facing API.  Its progress writer
accepts only scalar training telemetry and always replaces ``progress.json``
atomically, so an observer sees either the previous complete update or the
next complete update.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Final
import uuid


PROGRESS_FILENAME: Final = "progress.json"
PROGRESS_SCHEMA_VERSION: Final = "anchor.gemma3-q8-qlora-progress.v2"
SAFE_PROGRESS_FIELDS: Final = frozenset(
    {
        "run_id",
        "role",
        "phase",
        "step",
        "target",
        "loss",
        "peak",
        "updated_at_utc",
    }
)
SAFE_PEAK_FIELDS: Final = frozenset({"torch_allocated_bytes", "torch_reserved_bytes"})
_ROLES: Final = frozenset(
    {
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
    }
)
_PHASES: Final = frozenset({"smoke", "full"})
_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class Q8ProgressError(ValueError):
    """Raised when scalar progress telemetry violates the v2 contract."""


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Q8ProgressError(f"{field}_must_be_nonnegative_integer")
    return value


def _finite_loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Q8ProgressError("loss_must_be_finite_scalar")
    result = float(value)
    if not math.isfinite(result):
        raise Q8ProgressError("loss_must_be_finite_scalar")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sync_directory(path: Path) -> None:
    """Best-effort directory sync; Windows does not expose portable dir fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _replace_with_reader_retry(source: Path, destination: Path) -> None:
    """Replace atomically despite brief Windows read-sharing conflicts."""

    deadline = time.monotonic() + 2.0
    delay = 0.005
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.05)


def write_training_progress(
    progress_path: str | Path,
    *,
    run_id: str,
    role: str,
    phase: str,
    step: int,
    target: int,
    loss: float,
    torch_peak_allocated_bytes: int,
    torch_peak_reserved_bytes: int,
) -> dict[str, object]:
    """Atomically publish one content-free optimizer-step update.

    Call this only after a successful optimizer step.  The deliberately narrow
    signature prevents samples, labels, token IDs, prompts, and arbitrary
    ``detail`` mappings from entering the operational progress artifact.
    """

    path = Path(progress_path)
    if path.name != PROGRESS_FILENAME:
        raise Q8ProgressError("progress_path_must_end_with_progress_json")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise Q8ProgressError("run_id_invalid")
    if role not in _ROLES:
        raise Q8ProgressError("role_invalid")
    if phase not in _PHASES:
        raise Q8ProgressError("phase_invalid")
    completed = _nonnegative_int(step, field="step")
    total = _nonnegative_int(target, field="target")
    if completed < 1 or total < 1 or completed > total:
        raise Q8ProgressError("step_target_invalid")
    allocated = _nonnegative_int(
        torch_peak_allocated_bytes,
        field="torch_peak_allocated_bytes",
    )
    reserved = _nonnegative_int(
        torch_peak_reserved_bytes,
        field="torch_peak_reserved_bytes",
    )
    if reserved < allocated:
        raise Q8ProgressError("torch_peak_reserved_below_allocated")

    payload: dict[str, object] = {
        "run_id": run_id,
        "role": role,
        "phase": phase,
        "step": completed,
        "target": total,
        "loss": _finite_loss(loss),
        "peak": {
            "torch_allocated_bytes": allocated,
            "torch_reserved_bytes": reserved,
        },
        "updated_at_utc": _utc_now(),
    }
    if set(payload) != SAFE_PROGRESS_FIELDS:
        raise AssertionError("safe progress field contract drift")
    if set(payload["peak"]) != SAFE_PEAK_FIELDS:  # type: ignore[arg-type]
        raise AssertionError("safe progress peak field contract drift")

    encoded = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_reader_retry(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()

    if path.read_bytes() != encoded:
        raise Q8ProgressError("progress_atomic_write_verification_failed")
    return payload
