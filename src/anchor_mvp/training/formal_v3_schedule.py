"""Deterministic, snapshot-sized exposure plans for the formal-v3 A--F matrix.

The public SWE-bench train bank contains 19,008 candidate tasks and five work
orders per task.  Only rows accepted by the execution Gold gate may enter a
frozen training snapshot.  This module deliberately derives optimizer steps
from that frozen *train* count; the historic 256-row gate and 640-exposure
smoke experiment are not scale limits.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


EXPOSURE_MODE = "snapshot_train_epochs_v1"
SPLIT_SCHEMA = "anchor.formal-v3-gold-splits.v1"
CANDIDATE_TASKS_PER_STAGE = 19_008
WORK_ORDERS_PER_TASK = 5
CANDIDATE_WORK_ORDERS = CANDIDATE_TASKS_PER_STAGE * WORK_ORDERS_PER_TASK
FORMAL_ARMS = ("B", "C", "D", "E", "F")
SPECIALIST_COUNT = 5


class FormalV3ScheduleError(ValueError):
    """Raised when a frozen snapshot cannot define a fair A--F schedule."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FormalV3ScheduleError(f"{name} must be a positive integer")
    return value


def validate_exposure_control(value: object) -> Mapping[str, Any]:
    """Validate the checked-in, data-sized formal-v3 exposure contract."""

    if not isinstance(value, Mapping):
        raise FormalV3ScheduleError("training.exposure_control must be a mapping")
    if value.get("mode") != EXPOSURE_MODE:
        raise FormalV3ScheduleError(
            f"training.exposure_control.mode must be {EXPOSURE_MODE!r}"
        )
    epochs = _positive_int(value.get("epochs"), "training.exposure_control.epochs")
    if epochs != 1:
        raise FormalV3ScheduleError(
            "formal-v3 control runs currently require exactly one train epoch"
        )
    if value.get("candidate_tasks_per_stage") != CANDIDATE_TASKS_PER_STAGE:
        raise FormalV3ScheduleError(
            "formal-v3 candidate_tasks_per_stage must be 19008"
        )
    if value.get("work_orders_per_task") != WORK_ORDERS_PER_TASK:
        raise FormalV3ScheduleError("formal-v3 work_orders_per_task must be 5")
    if value.get("candidate_work_orders") != CANDIDATE_WORK_ORDERS:
        raise FormalV3ScheduleError(
            "formal-v3 candidate_work_orders must be 95040"
        )
    if value.get("accepted_rows") != "all_frozen_train_gold":
        raise FormalV3ScheduleError(
            "formal-v3 accepted_rows must be all_frozen_train_gold"
        )
    if (
        value.get("accumulation_padding")
        != "deterministic_stage_stratified_epoch_prefix_v1"
    ):
        raise FormalV3ScheduleError(
            "formal-v3 accumulation padding must be "
            "deterministic_stage_stratified_epoch_prefix_v1"
        )
    if value.get("arm_total_exposure_control") != "equal_B_through_F":
        raise FormalV3ScheduleError(
            "formal-v3 must equalize total sample exposure across B--F"
        )
    return value


def derive_exposure_plan(
    *,
    arm: str,
    train_records_per_expert: Mapping[str, int],
    gradient_accumulation_steps: int,
    epochs: int = 1,
) -> dict[str, Any]:
    """Derive exact optimizer steps from a frozen balanced train split.

    Each routed arm trains five independent adapters.  B trains one mixed
    adapter but receives the same number of exposures from every stage. If a
    per-stage count is not divisible by gradient accumulation, every arm gets
    the same independently shuffled prefix padding (at most ``accumulation-1``
    rows per stage). The runtime interleaves those strata and reports the
    observed per-file exposures rather than inferring them from aggregate size.
    """

    if arm not in FORMAL_ARMS:
        raise FormalV3ScheduleError(f"arm must be one of {FORMAL_ARMS}")
    accumulation = _positive_int(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    epoch_count = _positive_int(epochs, "epochs")
    if len(train_records_per_expert) != SPECIALIST_COUNT:
        raise FormalV3ScheduleError(
            "train split must contain exactly five expert record counts"
        )
    counts = {
        str(expert): _positive_int(count, f"train_records_per_expert.{expert}")
        for expert, count in train_records_per_expert.items()
    }
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise FormalV3ScheduleError(
            "formal-v3 complete-chain training requires equal train rows per expert"
        )
    records_per_stage = next(iter(unique_counts))
    if records_per_stage > CANDIDATE_TASKS_PER_STAGE:
        raise FormalV3ScheduleError(
            "frozen train Gold count exceeds the 19008-task source population"
        )

    requested_per_stage = records_per_stage * epoch_count
    steps_per_stage = math.ceil(requested_per_stage / accumulation)
    padded_per_stage = steps_per_stage * accumulation
    padding_per_stage = padded_per_stage - requested_per_stage
    total_control_exposures = padded_per_stage * SPECIALIST_COUNT
    max_steps = (
        steps_per_stage * SPECIALIST_COUNT if arm == "B" else steps_per_stage
    )
    adapter_jobs = 1 if arm == "B" else SPECIALIST_COUNT

    return {
        "schema_version": "anchor.formal-v3-exposure-plan.v1",
        "mode": EXPOSURE_MODE,
        "arm": arm,
        "epochs": epoch_count,
        "train_records_per_expert": dict(sorted(counts.items())),
        "records_per_stage": records_per_stage,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps_per_stage": steps_per_stage,
        "max_steps_per_adapter_job": max_steps,
        "adapter_jobs": adapter_jobs,
        "requested_exposures_per_stage": requested_per_stage,
        "padded_exposures_per_stage": padded_per_stage,
        "planned_exposures_by_stage": {
            name: padded_per_stage for name in sorted(counts)
        },
        "padding_exposures_per_stage": padding_per_stage,
        "padding_exposures_by_stage": {
            name: padding_per_stage for name in sorted(counts)
        },
        "arm_total_sample_exposures": total_control_exposures,
        "padding_policy": "deterministic_stage_stratified_epoch_prefix_v1",
        "control_invariant": (
            "equal_total_and_per_stage_sample_exposure_B_through_F"
        ),
        "candidate_tasks_per_stage": CANDIDATE_TASKS_PER_STAGE,
        "candidate_work_orders": CANDIDATE_WORK_ORDERS,
    }
