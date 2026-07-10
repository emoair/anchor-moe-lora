from __future__ import annotations

import json

from anchor_mvp.training.progress import TrainingProgress


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
    assert [event["sequence"] for event in events] == [1, 2]
    serialized = reporter.events_path.read_text(encoding="utf-8")
    assert "messages" not in serialized
    assert "KIMI_API_KEY" not in serialized
