"""Model-free scheduler benchmark for the planner-first 22,200 campaign.

The result describes local scheduler work only.  It makes no network request,
does not load a model, and must not be reported as provider token throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from statistics import median
import time
from typing import Any

from anchor_mvp.data.planner_first_coding_specialist_22200_fastpath_v1 import (
    GROUP_SIZE,
    GroupTerminal,
    PlannerFirstFastPath,
    RouteCommit,
    complexity_model,
)
from anchor_mvp.data.planner_first_coding_specialist_22200_v1 import PlannerEvidence


PLANNER_BOOTSTRAP_COUNT = 3_200
LEGACY_EXPERT_OBJECT_COUNT = 19_000


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _quantile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile))
    return sorted_values[index]


def run_benchmark(
    *,
    planner_bootstrap_count: int = PLANNER_BOOTSTRAP_COUNT,
    expert_object_count: int = LEGACY_EXPERT_OBJECT_COUNT,
    group_size: int = GROUP_SIZE,
) -> dict[str, Any]:
    """Exercise real in-memory delta scheduling for fixed body-free metadata."""

    if planner_bootstrap_count < 1 or expert_object_count < 1:
        raise ValueError("positive benchmark counts required")
    planner_ids = [
        _hash(f"planner-bootstrap:{index}") for index in range(planner_bootstrap_count)
    ]
    expert_ids = [
        _hash(f"legacy-expert:{index}") for index in range(expert_object_count)
    ]
    scheduler = PlannerFirstFastPath(
        planner_bootstrap_ids=planner_ids,
        expert_object_ids=expert_ids,
        runtime_hmac_key=b"planner-first-benchmark-hmac-key",
        group_size=group_size,
    )
    evidence = PlannerEvidence(
        training_receipt_sha256=_hash("planner-training"),
        freeze_receipt_sha256=_hash("planner-freeze"),
        planner_adapter_sha256=_hash("planner-adapter"),
        primary_arm="planner_q_only",
        frozen=True,
    )
    scheduler.freeze_planner(evidence)
    scheduler.register_route_commits(
        [
            RouteCommit(
                object_id_sha256=object_id,
                route_commit_sha256=_hash(f"route:{object_id}"),
                selected_expert="coding_specialist",
                planner_freeze_receipt_sha256=evidence.freeze_receipt_sha256,
            )
            for object_id in expert_ids
        ]
    )
    group_seconds: list[float] = []
    total_start = time.perf_counter()
    for start in range(0, len(expert_ids), group_size):
        group = expert_ids[start : start + group_size]
        rows = [
            GroupTerminal(
                object_id_sha256=object_id,
                terminal_state="accepted",
                receipt_sha256=_hash(f"receipt:{object_id}"),
                provider_input_tokens=1_000,
                provider_output_tokens=100,
            )
            for object_id in group
        ]
        group_start = time.perf_counter()
        scheduler.commit_group(rows, observed_duration_seconds=1.0)
        group_seconds.append(time.perf_counter() - group_start)
    replay_start = time.perf_counter()
    replay = scheduler.final_full_replay()
    replay_seconds = time.perf_counter() - replay_start
    total_seconds = time.perf_counter() - total_start
    sorted_seconds = sorted(group_seconds)
    complexity = complexity_model(
        object_count=expert_object_count, group_size=group_size
    )
    return {
        "benchmark_kind": "model_free_scheduler_microbenchmark",
        "claims": {
            "gpu_runs": 0,
            "model_loads": 0,
            "provider_requests": 0,
            "provider_token_throughput_measured": False,
        },
        "campaign_problem_object_count": planner_bootstrap_count + expert_object_count,
        "planner_bootstrap_object_count": planner_bootstrap_count,
        "terminal_expert_object_count": expert_object_count,
        "group_size": group_size,
        "group_count": len(group_seconds),
        "online_group_commit_seconds_median": median(group_seconds),
        "online_group_commit_seconds_p95": _quantile(sorted_seconds, 0.95),
        "online_object_commit_seconds_mean": sum(group_seconds) / expert_object_count,
        "final_full_replay_seconds": replay_seconds,
        "total_scheduler_seconds": total_seconds,
        "online_delta_rows_verified": scheduler.online_delta_rows_verified,
        "full_replay_count": scheduler.full_replay_count,
        "legacy_per_append_full_replay_row_visits_model": complexity[
            "legacy_per_append_full_replay_row_visits"
        ],
        "fastpath_online_delta_row_visits": complexity["online_delta_row_visits"],
        "release_terminal_count": replay["terminal_object_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner-bootstrap-count", type=int, default=PLANNER_BOOTSTRAP_COUNT
    )
    parser.add_argument(
        "--expert-object-count", type=int, default=LEGACY_EXPERT_OBJECT_COUNT
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_benchmark(
                planner_bootstrap_count=args.planner_bootstrap_count,
                expert_object_count=args.expert_object_count,
                group_size=args.group_size,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
