from __future__ import annotations

from pathlib import Path

import pytest

from anchor_mvp.benchmark.heldout import HeldoutGateError
from anchor_mvp.benchmark.heldout_eval import _evaluate_record
from anchor_mvp.benchmark.models import BenchmarkCase, BenchmarkRecord


def _record(
    stages: list[str],
    *,
    fail_closed: bool,
    call_count: int | None = None,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        baseline="c_pipeline",
        group="C",
        case_id="synthetic-trace-only",
        malicious=True,
        decision="BLOCK",
        success=not fail_closed,
        final_code=None,
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        call_count=len(stages) if call_count is None else call_count,
        request_attempts=len(stages),
        peak_vram_mb=None,
        fail_closed=fail_closed,
        stages=[{"stage": stage, "output_text": "synthetic"} for stage in stages],
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-trace-only",
        requirement="synthetic",
        malicious=True,
        review_mutation={"marker": "synthetic-marker"},
    )


def _evaluate(record: BenchmarkRecord, tmp_path: Path) -> None:
    _evaluate_record(
        record,
        _case(),
        tmp_path / "fixtures-must-not-be-opened",
        tmp_path / "workspaces-must-not-be-created",
        keep_workspaces=False,
    )


@pytest.mark.parametrize(
    "stages",
    [
        ["planner", "tool_policy", "frontend", "review", "security"],
        [
            "planner",
            "tool_policy",
            "frontend",
            "review",
            "frontend",
            "review",
            "security",
        ],
    ],
)
def test_successful_matched_trace_requires_ordered_five_stage_pipeline(
    stages: list[str], tmp_path: Path
) -> None:
    record = _record(stages, fail_closed=False)

    _evaluate(record, tmp_path)

    assert record.evaluation["distinct_expert_stages"] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
    ]


def test_authentic_fail_closed_review_terminal_is_accepted_without_fabrication(
    tmp_path: Path,
) -> None:
    stages = ["planner", "tool_policy", "frontend", "review"]
    record = _record(stages, fail_closed=True)

    _evaluate(record, tmp_path)

    assert record.call_count == 4
    assert [item["stage"] for item in record.stages] == stages
    assert record.evaluation["distinct_expert_stages"] == stages
    assert "security" not in record.evaluation["stage_attempt_counts"]


def test_four_stage_terminal_requires_explicit_fail_closed(tmp_path: Path) -> None:
    record = _record(
        ["planner", "tool_policy", "frontend", "review"], fail_closed=False
    )

    with pytest.raises(HeldoutGateError, match="fail-closed four-stage"):
        _evaluate(record, tmp_path)

@pytest.mark.parametrize(
    "stages",
    [
        ["planner", "tool_policy", "frontend"],
        ["planner", "frontend", "tool_policy", "review"],
        ["planner", "tool_policy", "review", "frontend"],
        ["planner", "tool_policy", "frontend", "security", "review"],
        ["planner", "tool_policy", "frontend", "review", "unknown"],
    ],
)
def test_other_missing_or_misordered_traces_remain_fail_closed(
    stages: list[str], tmp_path: Path
) -> None:
    record = _record(stages, fail_closed=True)

    with pytest.raises(HeldoutGateError, match="matched five-stage trace"):
        _evaluate(record, tmp_path)


def test_trace_gate_rejects_inflated_call_count(tmp_path: Path) -> None:
    record = _record(
        ["planner", "tool_policy", "frontend", "review"],
        fail_closed=True,
        call_count=5,
    )

    with pytest.raises(HeldoutGateError, match="call_count"):
        _evaluate(record, tmp_path)
