from __future__ import annotations

import json
import os
import threading
import time

import pytest

from anchor_mvp.training import progress
from anchor_mvp.training.progress import TrainingProgress


def _windows_lock_error(winerror: int) -> PermissionError:
    error = PermissionError(f"transient Windows lock {winerror}")
    error.winerror = winerror
    return error


def test_progress_is_atomic_append_only_and_content_free(tmp_path):
    reporter = TrainingProgress(tmp_path / "adapter")

    reporter.emit("model_load", "started", detail={"source": "local"})
    reporter.emit("optimizer_step", "completed", step=1, loss=1.25)

    status = json.loads(reporter.status_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in reporter.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert status["phase"] == "optimizer_step"
    assert status["step"] == 1
    assert status["loss"] == 1.25
    assert status["run_id"] == reporter.run_id
    assert {event["run_id"] for event in events} == {reporter.run_id}
    assert [event["sequence"] for event in events] == [1, 2]
    serialized = reporter.events_path.read_text(encoding="utf-8")
    assert "messages" not in serialized
    assert "KIMI_API_KEY" not in serialized


@pytest.mark.parametrize("winerror", [5, 32])
def test_progress_retries_transient_windows_replace_errors(
    tmp_path, monkeypatch, winerror
):
    reporter = TrainingProgress(tmp_path / "adapter")
    real_replace = os.replace
    sources = []
    sleeps = []

    def flaky_replace(source, destination):
        sources.append(source)
        if len(sources) <= 2:
            raise _windows_lock_error(winerror)
        real_replace(source, destination)

    monkeypatch.setattr(progress.os, "replace", flaky_replace)
    monkeypatch.setattr(progress.time, "sleep", sleeps.append)

    reporter.emit("optimizer_step", "completed", step=1, loss=0.5)

    assert json.loads(reporter.status_path.read_text(encoding="utf-8"))["step"] == 1
    assert sleeps == [0.025, 0.05]
    assert len(set(sources)) == 1
    assert not list(reporter.state_dir.glob(".*.tmp"))
    assert len(reporter.events_path.read_text(encoding="utf-8").splitlines()) == 1


def test_progress_nontransient_replace_error_fails_without_sleep(
    tmp_path, monkeypatch
):
    reporter = TrainingProgress(tmp_path / "adapter")
    sleeps = []

    def denied(_source, _destination):
        raise OSError("permanent failure")

    monkeypatch.setattr(progress.os, "replace", denied)
    monkeypatch.setattr(progress.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="permanent failure"):
        reporter.emit("optimizer_step", "completed", step=1, loss=0.5)

    assert sleeps == []
    assert not reporter.status_path.exists()
    assert not list(reporter.state_dir.glob(".*.tmp"))


def test_progress_retry_exhaustion_preserves_previous_status(tmp_path, monkeypatch):
    reporter = TrainingProgress(tmp_path / "adapter")
    reporter.emit("optimizer_step", "completed", step=1, loss=0.5)
    previous = reporter.status_path.read_bytes()
    sleeps = []

    def always_locked(_source, _destination):
        raise _windows_lock_error(5)

    monkeypatch.setattr(progress.os, "replace", always_locked)
    monkeypatch.setattr(progress.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError):
        reporter.emit("optimizer_step", "completed", step=2, loss=0.4)

    assert reporter.status_path.read_bytes() == previous
    assert len(sleeps) == progress._WINDOWS_REPLACE_RETRY_ATTEMPTS - 1
    assert not list(reporter.state_dir.glob(".*.tmp"))


def test_progress_uses_unique_same_directory_temporary_files(tmp_path, monkeypatch):
    reporter = TrainingProgress(tmp_path / "adapter")
    real_replace = os.replace
    sources = []

    def capture_replace(source, destination):
        sources.append(source)
        real_replace(source, destination)

    monkeypatch.setattr(progress.os, "replace", capture_replace)

    reporter.emit("optimizer_step", "completed", step=1, loss=0.5)
    reporter.emit("optimizer_step", "completed", step=2, loss=0.4)

    assert len(sources) == 2
    assert sources[0] != sources[1]
    assert all(source.parent == reporter.status_path.parent for source in sources)
    assert all(source.name != "status.json.tmp" for source in sources)


def test_progress_concurrent_readers_only_observe_complete_json(tmp_path):
    reporter = TrainingProgress(tmp_path / "adapter")
    reporter.emit("optimizer_step", "completed", step=0, loss=1.0)
    stop = threading.Event()
    failures = []
    observed_steps = []

    def reader():
        while not stop.is_set():
            try:
                observed_steps.append(
                    json.loads(reporter.status_path.read_text(encoding="utf-8"))["step"]
                )
            except PermissionError:
                # Windows readers can briefly lose the race with os.replace.
                # Retrying after the handle closes must still yield a complete
                # old or new JSON document, never a partially written one.
                time.sleep(0)
                continue
            except Exception as exc:  # pragma: no cover - assertion records details
                failures.append(exc)
                stop.set()
            time.sleep(0)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for step in range(1, 25):
            reporter.emit("optimizer_step", "completed", step=step, loss=1.0 / step)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not failures
    assert observed_steps
    assert set(observed_steps).issubset(set(range(25)))
    assert json.loads(reporter.status_path.read_text(encoding="utf-8"))["step"] == 24
