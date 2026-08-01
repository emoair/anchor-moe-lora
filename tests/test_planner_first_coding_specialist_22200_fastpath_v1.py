from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from anchor_mvp.data.planner_first_coding_specialist_22200_fastpath_v1 import (
    GroupTerminal,
    PlannerFirstFastPath,
    RouteCommit,
    SchedulerValidationError,
    complexity_model,
)
from anchor_mvp.data.planner_first_coding_specialist_22200_v1 import PlannerEvidence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/planner_first_coding_specialist_22200_fastpath_v1.json"
SCHEMA = (
    ROOT / "configs/data/planner_first_coding_specialist_22200_fastpath_v1.schema.json"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _scheduler() -> PlannerFirstFastPath:
    return PlannerFirstFastPath(
        planner_bootstrap_ids=[_hash("planner-0"), _hash("planner-1")],
        expert_object_ids=[_hash("expert-0"), _hash("expert-1"), _hash("expert-2")],
        runtime_hmac_key=b"test-hmac-key",
        group_size=2,
    )


def _evidence() -> PlannerEvidence:
    return PlannerEvidence(
        training_receipt_sha256=_hash("training"),
        freeze_receipt_sha256=_hash("freeze"),
        planner_adapter_sha256=_hash("adapter"),
        primary_arm="planner_q_only",
        frozen=True,
    )


def _routes() -> list[RouteCommit]:
    freeze = _evidence().freeze_receipt_sha256
    return [
        RouteCommit(
            object_id_sha256=_hash(f"expert-{index}"),
            route_commit_sha256=_hash(f"route-{index}"),
            selected_expert="coding_specialist",
            planner_freeze_receipt_sha256=freeze,
        )
        for index in range(3)
    ]


def _terminal(index: int, state: str = "accepted") -> GroupTerminal:
    return GroupTerminal(
        object_id_sha256=_hash(f"expert-{index}"),
        terminal_state=state,
        receipt_sha256=_hash(f"receipt-{index}"),
        provider_input_tokens=100 + index,
        provider_output_tokens=10 + index,
    )


def test_fastpath_config_is_closed_and_explicitly_non_live():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(config, schema)
    assert config["claims"]["provider_launch_authorized"] is False


def test_planner_freeze_precedes_route_and_expert_group():
    scheduler = _scheduler()
    with pytest.raises(
        SchedulerValidationError, match="planner_must_freeze_before_route_commit"
    ):
        scheduler.register_route_commits(_routes())

    scheduler.freeze_planner(_evidence())
    scheduler.register_route_commits(_routes()[:2])
    checkpoint = scheduler.commit_group(
        [_terminal(0), _terminal(1)], observed_duration_seconds=2.0
    )

    assert checkpoint.accepted_count == 2
    assert scheduler.online_delta_rows_verified == 2
    assert scheduler.full_replay_count == 0
    throughput = scheduler.group_throughput(checkpoint)
    assert throughput["input_tokens_per_second"] == pytest.approx(100.5)
    assert throughput["output_tokens_per_second"] == pytest.approx(10.5)


def test_commit_rejects_unrouted_duplicate_and_tampered_terminal_state():
    scheduler = _scheduler()
    scheduler.freeze_planner(_evidence())
    scheduler.register_route_commits(_routes()[:1])
    with pytest.raises(
        SchedulerValidationError, match="expert_group_requires_frozen_route_commit"
    ):
        scheduler.commit_group([_terminal(1)], observed_duration_seconds=1.0)
    with pytest.raises(SchedulerValidationError, match="terminal_group_duplicates"):
        scheduler.commit_group(
            [_terminal(0), _terminal(0)], observed_duration_seconds=1.0
        )


def test_unknown_usage_is_explicitly_unknown_not_invented():
    scheduler = _scheduler()
    scheduler.freeze_planner(_evidence())
    scheduler.register_route_commits(_routes()[:1])
    unknown_usage = GroupTerminal(
        object_id_sha256=_hash("expert-0"),
        terminal_state="accepted",
        receipt_sha256=_hash("receipt-0"),
        provider_input_tokens=None,
        provider_output_tokens=None,
    )
    checkpoint = scheduler.commit_group([unknown_usage], observed_duration_seconds=1.0)
    assert scheduler.group_throughput(checkpoint)["semantics"].startswith("UNKNOWN_")


def test_final_replay_reauthenticates_all_checkpoint_tips():
    scheduler = _scheduler()
    scheduler.freeze_planner(_evidence())
    scheduler.register_route_commits(_routes())
    scheduler.commit_group(
        [_terminal(0), _terminal(1, "rejected")], observed_duration_seconds=1.0
    )
    scheduler.commit_group([_terminal(2, "quarantine")], observed_duration_seconds=1.0)
    replay = scheduler.final_full_replay()

    assert replay["accepted_count"] == 1
    assert replay["rejected_count"] == 1
    assert replay["quarantine_count"] == 1
    assert replay["terminal_object_count"] == 3
    assert replay["full_replay_count"] == 1


def test_complexity_model_is_explicitly_scheduler_only():
    model = complexity_model(object_count=22_200)
    assert model["online_delta_row_visits"] == 22_200
    assert model["final_full_replay_rows"] == 22_200
    assert model["legacy_per_append_full_replay_row_visits"] > 200_000_000
    assert "provider" not in model["online_work"]
