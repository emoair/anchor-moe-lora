from __future__ import annotations

from scripts.research.benchmark_planner_first_coding_specialist_22200_fastpath_v1 import (
    run_benchmark,
)


def test_model_free_fastpath_benchmark_reports_scheduler_not_provider_metrics():
    result = run_benchmark(
        planner_bootstrap_count=2, expert_object_count=5, group_size=2
    )

    assert result["benchmark_kind"] == "model_free_scheduler_microbenchmark"
    assert result["claims"]["provider_requests"] == 0
    assert result["claims"]["provider_token_throughput_measured"] is False
    assert result["campaign_problem_object_count"] == 7
    assert result["online_delta_rows_verified"] == 5
    assert result["release_terminal_count"] == 5
    assert result["full_replay_count"] == 1
    assert (
        result["legacy_per_append_full_replay_row_visits_model"]
        > result["fastpath_online_delta_row_visits"]
    )
