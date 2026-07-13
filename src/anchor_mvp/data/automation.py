"""Unattended, gated concurrency ramp for defensive data distillation."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence, cast
from uuid import uuid4

from ..benchmark.heldout import (
    check_training_leakage,
    file_sha256,
    verify_heldout_manifest,
    verify_leak_audit,
)
from .artifact_validation import validate_tsx_fragment
from .cleaning import (
    build_inert_security_fixture,
    contains_secret_material,
    deterministic_security_fixture_oracle,
    sanitize_security_seed,
    validate_safe_payload,
)
from .coverage import detect_near_duplicate_seeds, evaluate_task_card_coverage
from .cli import _as_bool, _simple_config
from .mutator import mutate_frontend_code
from .pipeline import DistillationPipeline, PipelineReport
from .proposals import (
    PROPOSAL_GENERATOR_VERSION,
    deterministic_tool_policy_oracle,
)
from .provider import PRESETS, ProviderSelection, provider_spec, select_provider_model
from .schema import TASK_TYPES, SeedDemand, stable_id
from .storage import JsonlStore, SeedStore
from .task_cards import (
    CardAssignment,
    assignment_for_seed,
    axis_from_tags,
    load_task_card_catalog,
)
from .teacher import (
    BudgetExceeded,
    ClientDeadlineExceeded,
    CompatibleTeacher,
    MockTeacher,
    ProviderQuotaExhausted,
    RateLimitError,
    Teacher,
)


AUTOMATION_SCHEMA_VERSION = "2.0"
LEGACY_AUTOMATION_SCHEMA_VERSION = "1.0"
_USAGE_CHECKPOINT_REQUEST_INTERVAL = 8
_USAGE_CHECKPOINT_OUTPUT_TOKEN_INTERVAL = 4096
_USAGE_CHECKPOINT_MAX_SECONDS = 5.0
DEFAULT_CONCURRENCY_STAGES = (1,)
COLLECTION_POLICIES = frozenset({"gated", "collect_then_partition"})
QUALITY_STAGING_SCHEMA_VERSION = "anchor.automation-quality-staging.v1"
PARTITION_REJECT_SCHEMA_VERSION = "anchor.automation-partition-reject.v1"
NON_CHARGEABLE_FAILURE_CLASSES = frozenset(
    {
        "BudgetExceeded",
        "ClientDeadlineExceeded",
        "ProviderQuotaExhausted",
        "RateLimitError",
        "UpstreamDependencyError",
    }
)
_LABELS_BY_TASK: dict[str, frozenset[str]] = {
    "tool_policy": frozenset({"APPROVE", "BLOCK", "ESCALATE"}),
    "security": frozenset({"PASS", "BLOCK"}),
}
_LINEAGE_EDGES: tuple[tuple[str, str, str, str], ...] = (
    ("plan->tool_policy", "plan", "tool_policy", "source_plan_record_id"),
    ("plan->frontend", "plan", "frontend", "source_plan_record_id"),
    (
        "tool_policy->frontend",
        "tool_policy",
        "frontend",
        "source_tool_policy_record_id",
    ),
    ("frontend->review", "frontend", "review", "source_frontend_record_id"),
    ("review->security", "review", "security", "source_review_record_id"),
)


def _int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    raw: list[Any]
    if isinstance(value, int) and not isinstance(value, bool):
        # Older operator configs used a YAML scalar for a one-stage ramp. Keep
        # those files loadable, while canonical configs use a YAML list.
        raw = [value]
    elif isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(
            f"{name} must be a positive integer, list, or comma-separated list"
        )
    normalized: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain positive integers")
        if isinstance(item, int):
            candidate = item
        elif isinstance(item, str) and item.isdecimal():
            candidate = int(item)
        else:
            raise ValueError(f"{name} must contain positive integers")
        if candidate < 1:
            raise ValueError(f"{name} must contain positive integers")
        normalized.append(candidate)
    if not normalized:
        raise ValueError(f"{name} must contain at least one positive integer")
    return tuple(normalized)


def chargeable_failure_count(errors: Sequence[str]) -> int:
    """Count originating failures without charging dependency cascades."""

    return sum("UpstreamDependencyError" not in error for error in errors)


def _minimum_label_counts(value: Any) -> dict[str, dict[str, int]]:
    """Normalize optional per-expert label floors from an operator config."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("minimum_label_counts must be a mapping")
    normalized: dict[str, dict[str, int]] = {}
    for raw_task, raw_counts in value.items():
        task = str(raw_task)
        if task not in _LABELS_BY_TASK:
            raise ValueError(f"minimum_label_counts has unsupported task: {task}")
        if not isinstance(raw_counts, Mapping) or not raw_counts:
            raise ValueError(f"minimum_label_counts.{task} must be a non-empty mapping")
        counts: dict[str, int] = {}
        for raw_label, raw_count in raw_counts.items():
            label = str(raw_label)
            if label not in _LABELS_BY_TASK[task]:
                raise ValueError(
                    f"minimum_label_counts.{task} has unsupported label: {label}"
                )
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 1
            ):
                raise ValueError(
                    f"minimum_label_counts.{task}.{label} must be a positive integer"
                )
            counts[label] = raw_count
        normalized[task] = counts
    return normalized


def _minimum_gold_records(value: Any) -> dict[str, int]:
    """Normalize an explicit gold floor that is separate from raw collection size.

    A scalar applies to every task. A mapping must name every task so a typo or
    omitted expert cannot silently weaken the training-readiness contract.
    ``None`` preserves the legacy strict contract where the raw target is also
    the gold target.
    """

    if value is None:
        return {}
    if isinstance(value, bool):
        raise ValueError("minimum_gold_records_per_task must contain positive integers")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(
                "minimum_gold_records_per_task must contain positive integers"
            )
        return {task: value for task in TASK_TYPES}
    if isinstance(value, Mapping) and not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "minimum_gold_records_per_task must be a positive integer or complete task mapping"
        )
    unknown = {str(task) for task in value}.difference(TASK_TYPES)
    missing = set(TASK_TYPES).difference(str(task) for task in value)
    if unknown:
        raise ValueError(
            "minimum_gold_records_per_task has unsupported tasks: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "minimum_gold_records_per_task is missing tasks: "
            + ", ".join(sorted(missing))
        )
    normalized: dict[str, int] = {}
    for raw_task, raw_count in value.items():
        task = str(raw_task)
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
        ):
            raise ValueError(
                f"minimum_gold_records_per_task.{task} must be a positive integer"
            )
        normalized[task] = raw_count
    return normalized


def _expansion_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(
            f"monotonic_expansion_from.{name} must be a {qualifier} integer"
        )
    return value


@dataclass(frozen=True)
class MonotonicExpansionSource:
    """The exact prior collection contract accepted by an explicit expansion."""

    concurrency_stages: tuple[int, ...]
    stage_seed_counts: tuple[int, ...]
    raw_collection_target: int
    minimum_gold_records_per_task: dict[str, int]
    max_requests: int
    max_output_tokens_total: int
    max_failures: int

    def __post_init__(self) -> None:
        _expansion_integer(
            self.raw_collection_target,
            name="raw_collection_target",
            minimum=1,
        )
        _expansion_integer(self.max_requests, name="max_requests", minimum=1)
        _expansion_integer(
            self.max_output_tokens_total,
            name="max_output_tokens_total",
            minimum=1,
        )
        _expansion_integer(self.max_failures, name="max_failures", minimum=0)
        if (
            not self.concurrency_stages
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.concurrency_stages
            )
            or len(self.concurrency_stages) != len(self.stage_seed_counts)
        ):
            raise ValueError(
                "monotonic_expansion_from schedules must have matching positive values"
            )
        if (
            not self.stage_seed_counts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.stage_seed_counts
            )
            or tuple(sorted(set(self.stage_seed_counts))) != self.stage_seed_counts
        ):
            raise ValueError(
                "monotonic_expansion_from stage targets must be strictly increasing"
            )
        if self.raw_collection_target < self.stage_seed_counts[-1]:
            raise ValueError(
                "monotonic_expansion_from raw target must cover its final seed target"
            )
        normalized_gold = _minimum_gold_records(self.minimum_gold_records_per_task)
        if normalized_gold != self.minimum_gold_records_per_task:
            raise ValueError(
                "monotonic_expansion_from minimum gold mapping must be normalized"
            )
        if any(
            count > self.raw_collection_target for count in normalized_gold.values()
        ):
            raise ValueError(
                "monotonic_expansion_from gold floors cannot exceed its raw target"
            )

    def binding_overrides(self) -> dict[str, Any]:
        return {
            "concurrency_stages": self.concurrency_stages,
            "stage_seed_counts": self.stage_seed_counts,
            "raw_collection_target": self.raw_collection_target,
            "minimum_gold_records_per_task": self.minimum_gold_records_per_task,
            "max_requests": self.max_requests,
            "max_output_tokens_total": self.max_output_tokens_total,
            "max_failures": self.max_failures,
        }


def _monotonic_expansion_source(value: Any) -> MonotonicExpansionSource | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("monotonic_expansion_from must be a mapping")
    required = {
        "concurrency_stages",
        "stage_seed_counts",
        "raw_collection_target",
        "minimum_gold_records_per_task",
        "max_requests",
        "max_output_tokens_total",
        "max_failures",
    }
    observed = {str(key) for key in value}
    missing = required.difference(observed)
    unknown = observed.difference(required)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise ValueError(
            "monotonic_expansion_from requires one exact prior contract ("
            + "; ".join(details)
            + ")"
        )
    return MonotonicExpansionSource(
        concurrency_stages=_int_tuple(
            value["concurrency_stages"],
            name="monotonic_expansion_from.concurrency_stages",
        ),
        stage_seed_counts=_int_tuple(
            value["stage_seed_counts"],
            name="monotonic_expansion_from.stage_seed_counts",
        ),
        raw_collection_target=_expansion_integer(
            value["raw_collection_target"],
            name="raw_collection_target",
            minimum=1,
        ),
        minimum_gold_records_per_task=_minimum_gold_records(
            value["minimum_gold_records_per_task"]
        ),
        max_requests=_expansion_integer(
            value["max_requests"], name="max_requests", minimum=1
        ),
        max_output_tokens_total=_expansion_integer(
            value["max_output_tokens_total"],
            name="max_output_tokens_total",
            minimum=1,
        ),
        max_failures=_expansion_integer(
            value["max_failures"], name="max_failures", minimum=0
        ),
    )


@dataclass(frozen=True)
class AutomationConfig:
    sop_dir: Path
    output_dir: Path
    heldout_cases: Path | None = None
    heldout_fixtures_root: Path | None = None
    heldout_manifest: Path | None = None
    heldout_leak_audit: Path | None = None
    minimum_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    minimum_gold_records_per_task: dict[str, int] = field(default_factory=dict)
    artifact_validation_fixture: Path | None = None
    artifact_validation_workspace_root: Path | None = None
    artifact_validation_timeout_seconds: float = 30.0
    concurrency_stages: tuple[int, ...] = DEFAULT_CONCURRENCY_STAGES
    stage_seed_counts: tuple[int, ...] = (3,)
    raw_collection_target: int | None = None
    min_success_rate: float = 1.0
    max_duplicate_rate: float = 0.0
    max_safety_violations: int = 0
    max_failures: int = 8
    max_requests: int = 200
    max_output_tokens_total: int = 1_000_000
    quota_epoch_id: str = "default"
    max_failure_retries: int = 2
    cooldown_seconds: int = 18_000
    cooldown_poll_seconds: int = 60
    max_stagnant_gate_rounds: int = 5
    collection_policy: str = "gated"
    monotonic_expansion_from: MonotonicExpansionSource | None = None
    task_card_config: Path | None = None
    seed_index_offset: int = 0

    def __post_init__(self) -> None:
        if not self.concurrency_stages or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.concurrency_stages
        ):
            raise ValueError("concurrency_stages must contain positive integers")
        if len(self.stage_seed_counts) != len(self.concurrency_stages):
            raise ValueError(
                "stage_seed_counts must have one target per concurrency stage"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.stage_seed_counts
        ):
            raise ValueError("stage seed targets must be positive integers")
        if tuple(sorted(set(self.stage_seed_counts))) != self.stage_seed_counts:
            raise ValueError("stage seed targets must be strictly increasing")
        normalized_gold = _minimum_gold_records(self.minimum_gold_records_per_task)
        if normalized_gold != self.minimum_gold_records_per_task:
            raise ValueError(
                "minimum_gold_records_per_task must be normalized task counts"
            )
        if self.raw_collection_target is not None and (
            isinstance(self.raw_collection_target, bool)
            or not isinstance(self.raw_collection_target, int)
            or self.raw_collection_target < self.stage_seed_counts[-1]
        ):
            raise ValueError(
                "raw_collection_target must be an integer at least as large as "
                "the final stage seed target"
            )
        raw_target = self.raw_records_per_task
        over_target = {
            task: count for task, count in normalized_gold.items() if count > raw_target
        }
        if over_target:
            details = ", ".join(
                f"{task}={count}" for task, count in sorted(over_target.items())
            )
            raise ValueError(
                "minimum gold records cannot exceed the raw collection target "
                f"({raw_target}): {details}"
            )
        if not 0 <= self.min_success_rate <= 1:
            raise ValueError("min_success_rate must be between 0 and 1")
        if not 0 <= self.max_duplicate_rate <= 1:
            raise ValueError("max_duplicate_rate must be between 0 and 1")
        if self.max_safety_violations < 0 or self.max_failures < 0:
            raise ValueError("failure and safety budgets cannot be negative")
        if self.max_requests < 1 or self.max_output_tokens_total < 1:
            raise ValueError("request and output-token budgets must be positive")
        if not self.quota_epoch_id.strip():
            raise ValueError("quota_epoch_id cannot be empty")
        if self.max_failure_retries < 0:
            raise ValueError("max_failure_retries cannot be negative")
        if self.cooldown_seconds < 1 or self.cooldown_poll_seconds < 1:
            raise ValueError("cooldown values must be positive")
        if self.max_stagnant_gate_rounds < 1:
            raise ValueError("max_stagnant_gate_rounds must be positive")
        if self.collection_policy not in COLLECTION_POLICIES:
            raise ValueError(
                "collection_policy must be gated or collect_then_partition"
            )
        if (
            isinstance(self.seed_index_offset, bool)
            or not isinstance(self.seed_index_offset, int)
            or self.seed_index_offset < 0
        ):
            raise ValueError("seed_index_offset must be a non-negative integer")
        load_task_card_catalog(self.task_card_config)
        source = self.monotonic_expansion_from
        if source is not None:
            if (
                self.raw_collection_target is None
                or not self.minimum_gold_records_per_task
            ):
                raise ValueError(
                    "monotonic expansion requires explicit target raw/gold contracts"
                )
            if self.raw_records_per_task < source.raw_collection_target:
                raise ValueError("monotonic expansion cannot reduce the raw target")
            if self.stage_seed_counts[-1] < source.stage_seed_counts[-1]:
                raise ValueError("monotonic expansion cannot reduce the seed target")
            if any(
                self.minimum_gold_records_by_task[task]
                < source.minimum_gold_records_per_task[task]
                for task in TASK_TYPES
            ):
                raise ValueError("monotonic expansion cannot reduce a gold floor")
            if self.max_requests < source.max_requests:
                raise ValueError("monotonic expansion cannot reduce the request budget")
            if self.max_output_tokens_total < source.max_output_tokens_total:
                raise ValueError(
                    "monotonic expansion cannot reduce the output-token budget"
                )
            if self.max_failures < source.max_failures:
                raise ValueError("monotonic expansion cannot reduce the failure budget")
            expanded = (
                self.raw_records_per_task > source.raw_collection_target
                or self.stage_seed_counts[-1] > source.stage_seed_counts[-1]
                or any(
                    self.minimum_gold_records_by_task[task]
                    > source.minimum_gold_records_per_task[task]
                    for task in TASK_TYPES
                )
            )
            if not expanded:
                raise ValueError(
                    "monotonic expansion must increase seed, raw, or gold targets"
                )
        _minimum_label_counts(self.minimum_label_counts)
        artifact_validation_paths = (
            self.artifact_validation_fixture,
            self.artifact_validation_workspace_root,
        )
        if any(path is not None for path in artifact_validation_paths) and not all(
            path is not None for path in artifact_validation_paths
        ):
            raise ValueError(
                "artifact validation fixture and workspace root must be configured together"
            )
        if self.artifact_validation_timeout_seconds <= 0:
            raise ValueError("artifact validation timeout must be positive")
        heldout_paths = (
            self.heldout_cases,
            self.heldout_fixtures_root,
            self.heldout_manifest,
            self.heldout_leak_audit,
        )
        if any(path is not None for path in heldout_paths) and not all(
            path is not None for path in heldout_paths
        ):
            raise ValueError(
                "held-out automation gate paths must be configured together"
            )

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "automation"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def events_path(self) -> Path:
        return self.state_dir / "events.jsonl"

    @property
    def attempts_path(self) -> Path:
        return self.state_dir / "attempts.jsonl"

    @property
    def quality_staging_path(self) -> Path:
        return self.state_dir / "quality_staging.jsonl"

    @property
    def partition_dir(self) -> Path:
        return self.output_dir / "partitions"

    @property
    def minimum_gold_records_by_task(self) -> dict[str, int]:
        """Return explicit gold floors or the backward-compatible strict floor."""

        if self.minimum_gold_records_per_task:
            return dict(self.minimum_gold_records_per_task)
        return {task: self.raw_records_per_task for task in TASK_TYPES}

    @property
    def raw_records_per_task(self) -> int:
        """Final raw rows requested per expert, including overcollection headroom."""

        return self.raw_collection_target or self.stage_seed_counts[-1]

    def _status_binding_payload(self, *, include_gold_contract: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "anchor.automation-status-binding.v1",
            "sop_dir": str(self.sop_dir),
            "output_dir": str(self.output_dir),
            "heldout_cases": str(self.heldout_cases) if self.heldout_cases else None,
            "heldout_fixtures_root": (
                str(self.heldout_fixtures_root) if self.heldout_fixtures_root else None
            ),
            "heldout_manifest": str(self.heldout_manifest)
            if self.heldout_manifest
            else None,
            "heldout_leak_audit": (
                str(self.heldout_leak_audit) if self.heldout_leak_audit else None
            ),
            "concurrency_stages": self.concurrency_stages,
            "stage_seed_counts": self.stage_seed_counts,
            "min_success_rate": self.min_success_rate,
            "max_duplicate_rate": self.max_duplicate_rate,
            "max_safety_violations": self.max_safety_violations,
            "minimum_label_counts": self.minimum_label_counts,
            "artifact_validation_fixture": (
                str(self.artifact_validation_fixture)
                if self.artifact_validation_fixture
                else None
            ),
            "artifact_validation_workspace_root": (
                str(self.artifact_validation_workspace_root)
                if self.artifact_validation_workspace_root
                else None
            ),
            "artifact_validation_timeout_seconds": self.artifact_validation_timeout_seconds,
            "max_failures": self.max_failures,
            "max_requests": self.max_requests,
            "max_output_tokens_total": self.max_output_tokens_total,
            "max_failure_retries": self.max_failure_retries,
            "cooldown_seconds": self.cooldown_seconds,
            "cooldown_poll_seconds": self.cooldown_poll_seconds,
            "max_stagnant_gate_rounds": self.max_stagnant_gate_rounds,
            "collection_policy": self.collection_policy,
            "task_card_config": str(
                load_task_card_catalog(self.task_card_config).source
            ),
            "task_card_catalog_sha256": load_task_card_catalog(
                self.task_card_config
            ).sha256,
            "seed_index_offset": self.seed_index_offset,
        }
        if include_gold_contract:
            payload["schema"] = "anchor.automation-status-binding.v2"
            payload["raw_collection_target"] = self.raw_records_per_task
            payload["minimum_gold_records_per_task"] = self.minimum_gold_records_by_task
        return payload

    @staticmethod
    def _binding_sha256(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def status_binding_sha256(self) -> str:
        """Bind one output/state directory to its immutable corpus contract.

        A quota epoch is deliberately excluded: an operator may start a new
        provider quota window without changing the corpus definition. Ramp,
        quality-gate, and workspace settings are included so an opt-in fast
        profile cannot silently resume a serialized profile in the same state.
        """

        return self._binding_sha256(
            self._status_binding_payload(
                include_gold_contract=(
                    self.raw_collection_target is not None
                    or bool(self.minimum_gold_records_per_task)
                )
            )
        )

    @property
    def legacy_status_binding_sha256(self) -> str:
        """Hash the pre-gold-floor corpus contract for one explicit migration."""

        return self._binding_sha256(
            self._status_binding_payload(include_gold_contract=False)
        )

    @property
    def monotonic_expansion_source_binding_sha256(self) -> str | None:
        """Hash the declared source without weakening ordinary binding checks."""

        source = self.monotonic_expansion_from
        if source is None:
            return None
        payload = self._status_binding_payload(include_gold_contract=True)
        payload.update(source.binding_overrides())
        return self._binding_sha256(payload)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, repo_root: Path
    ) -> "AutomationConfig":
        def path_setting(name: str, default: str) -> Path:
            path = Path(str(value.get(name, default)))
            return (
                (repo_root / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )

        def optional_path(name: str) -> Path | None:
            raw = value.get(name)
            if raw is None or not str(raw).strip():
                return None
            path = Path(str(raw))
            return (
                (repo_root / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )

        return cls(
            sop_dir=path_setting("sop_dir", "skills"),
            output_dir=path_setting("output_dir", "data/automation-run"),
            heldout_cases=optional_path("heldout_cases"),
            heldout_fixtures_root=optional_path("heldout_fixtures_root"),
            heldout_manifest=optional_path("heldout_manifest"),
            heldout_leak_audit=optional_path("heldout_leak_audit"),
            minimum_label_counts=_minimum_label_counts(
                value.get("minimum_label_counts")
            ),
            minimum_gold_records_per_task=_minimum_gold_records(
                value.get("minimum_gold_records_per_task")
            ),
            artifact_validation_fixture=optional_path("artifact_validation_fixture"),
            artifact_validation_workspace_root=optional_path(
                "artifact_validation_workspace_root"
            ),
            artifact_validation_timeout_seconds=float(
                value.get("artifact_validation_timeout_seconds", 30.0)
            ),
            concurrency_stages=_int_tuple(
                value.get("concurrency_stages", "1"), name="concurrency_stages"
            ),
            stage_seed_counts=_int_tuple(
                value.get("stage_seed_counts", "3"), name="stage_seed_counts"
            ),
            raw_collection_target=(
                int(value["raw_collection_target"])
                if value.get("raw_collection_target") is not None
                else None
            ),
            min_success_rate=float(value.get("min_success_rate", 1.0)),
            max_duplicate_rate=float(value.get("max_duplicate_rate", 0.0)),
            max_safety_violations=int(value.get("max_safety_violations", 0)),
            max_failures=int(value.get("max_failures", 8)),
            max_requests=int(value.get("max_requests", 200)),
            max_output_tokens_total=int(
                value.get("max_output_tokens_total", 1_000_000)
            ),
            quota_epoch_id=str(value.get("quota_epoch_id", "default")),
            max_failure_retries=int(value.get("max_failure_retries", 2)),
            cooldown_seconds=int(value.get("cooldown_seconds", 18_000)),
            cooldown_poll_seconds=int(value.get("cooldown_poll_seconds", 60)),
            max_stagnant_gate_rounds=int(value.get("max_stagnant_gate_rounds", 5)),
            collection_policy=str(value.get("collection_policy", "gated")),
            monotonic_expansion_from=_monotonic_expansion_source(
                value.get("monotonic_expansion_from")
            ),
            task_card_config=optional_path("task_card_config"),
            seed_index_offset=int(value.get("seed_index_offset", 0)),
        )


@dataclass
class _LogicalUsageTracker:
    max_requests: int
    max_output_tokens: int
    requests: int = 0
    output_tokens: int = 0


class _TrackedTeacher:
    """Logical usage fallback for mocks; real clients expose wire-attempt usage."""

    def __init__(
        self,
        teacher: Teacher,
        *,
        max_requests: int,
        max_output_tokens: int,
        usage_progress_callback: Callable[[], Awaitable[None]] | None = None,
        logical_usage: _LogicalUsageTracker | None = None,
    ) -> None:
        self.inner = teacher
        self.model = teacher.model
        self.base_url = teacher.base_url
        self.protocol = teacher.protocol
        self.generation_params = teacher.generation_params
        self.max_requests = max_requests
        self.max_output_tokens = max_output_tokens
        self.logical_usage = logical_usage or _LogicalUsageTracker(
            max_requests=max_requests,
            max_output_tokens=max_output_tokens,
        )
        self.usage_progress_callback = usage_progress_callback

        limiter = getattr(teacher, "limit_remaining_budget", None)
        if limiter is not None:
            limiter(max_requests=max_requests, max_output_tokens=max_output_tokens)

    async def complete(self, *, system: str, user: str) -> str:
        if self.logical_usage.requests >= self.logical_usage.max_requests:
            raise BudgetExceeded("automation request budget exhausted")
        self.logical_usage.requests += 1
        if self.usage_progress_callback is not None:
            # Persist a conservative logical reservation before entering a
            # potentially long, uncancellable to_thread-backed wire request.
            await self.usage_progress_callback()
        try:
            result = await self.inner.complete(system=system, user=user)
            self.logical_usage.output_tokens += max(1, len(result) // 4)
            if self.logical_usage.output_tokens > self.logical_usage.max_output_tokens:
                raise BudgetExceeded("automation output-token budget exhausted")
            self.protocol = self.inner.protocol
            self.base_url = self.inner.base_url
            return result
        finally:
            if self.usage_progress_callback is not None:
                # Real teachers expose exact wire attempts/output usage only
                # after the request (including retries) returns or fails.
                await self.usage_progress_callback()

    @property
    def usage_snapshot(self) -> dict[str, int]:
        snapshot = getattr(self.inner, "usage_snapshot", None)
        if snapshot is not None:
            observed = dict(snapshot)
            return {
                # The logical reservation closes the crash window before the
                # real client's worker thread calls reserve_request().
                "requests": max(
                    self.logical_usage.requests,
                    int(observed.get("requests", 0)),
                ),
                "output_tokens": int(observed.get("output_tokens", 0)),
            }
        return {
            "requests": self.logical_usage.requests,
            "output_tokens": self.logical_usage.output_tokens,
        }

    @property
    def provider_provenance(self) -> dict[str, Any]:
        return dict(self.inner.provider_provenance)

    @property
    def usage_budget_id(self) -> int:
        return int(getattr(self.inner, "usage_budget_id", id(self)))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _new_quota_epoch(config: AutomationConfig) -> dict[str, Any]:
    return {
        "epoch_id": config.quota_epoch_id,
        "started_at": _iso(),
        "requests_used": 0,
        "output_tokens_used": 0,
        "failures_used": 0,
        "charged_failure_keys": [],
        "max_requests": config.max_requests,
        "max_output_tokens_total": config.max_output_tokens_total,
        "max_failures": config.max_failures,
    }


def _usage_checkpoint_policy(
    config: AutomationConfig,
) -> dict[str, int | float | str]:
    worst_case_requests = max(config.concurrency_stages) - 1
    return {
        "mode": "bounded_group_commit",
        "request_interval": _USAGE_CHECKPOINT_REQUEST_INTERVAL,
        "worst_case_requests": worst_case_requests,
        "maximum_unpersisted_requests": min(
            _USAGE_CHECKPOINT_REQUEST_INTERVAL - 1,
            worst_case_requests,
        ),
        "output_token_interval": _USAGE_CHECKPOINT_OUTPUT_TOKEN_INTERVAL,
        "maximum_unpersisted_output_tokens": (
            _USAGE_CHECKPOINT_OUTPUT_TOKEN_INTERVAL - 1
        ),
        "maximum_seconds": _USAGE_CHECKPOINT_MAX_SECONDS,
    }


def _legacy_binding_contract_proof(
    config: AutomationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove that a binding-v1 status is only being re-encoded, not expanded.

    Binding v1 omitted separate raw/gold fields, so its only defensible corpus
    contract is the final stage target for both. Any larger target must use the
    explicit monotonic-expansion command with an operator-declared source; any
    smaller gold floor is a contraction and is always refused.
    """

    implicit_target = config.stage_seed_counts[-1]
    source_contract = {
        "raw_collection_target": implicit_target,
        "minimum_gold_records_per_task": {task: implicit_target for task in TASK_TYPES},
    }
    target_contract = {
        "raw_collection_target": config.raw_records_per_task,
        "minimum_gold_records_per_task": config.minimum_gold_records_by_task,
    }
    if config.raw_records_per_task < implicit_target or any(
        config.minimum_gold_records_by_task[task] < implicit_target
        for task in TASK_TYPES
    ):
        raise ValueError(
            "legacy binding migration cannot reduce the implicit raw/gold floor"
        )
    if config.raw_records_per_task > implicit_target or any(
        config.minimum_gold_records_by_task[task] > implicit_target
        for task in TASK_TYPES
    ):
        raise ValueError(
            "legacy binding expansion requires explicit monotonic expansion"
        )
    return source_contract, target_contract


def _migrate_legacy_binding_status(
    config: AutomationConfig,
    status: dict[str, Any],
    *,
    trigger: str,
) -> dict[str, Any]:
    """Atomically reset executable state for the proven binding-v1 re-encode."""

    observed = status.get("config_binding_sha256")
    if observed != config.legacy_status_binding_sha256:
        raise ValueError("legacy automation status binding mismatch")
    source_contract, target_contract = _legacy_binding_contract_proof(config)
    if status.get("state") == "running" or status.get("current_worker") not in (
        None,
        "",
    ):
        raise ValueError(
            "automation status is active; stop the runner before contract migration"
        )
    run_history = status.setdefault("run_history", [])
    migration_history = status.setdefault("migration_history", [])
    if not isinstance(run_history, list) or not isinstance(migration_history, list):
        raise ValueError("automation status migration history must be a list")

    migrated_at = _iso()
    previous_partition = status.pop("partition", None)
    previous_run: dict[str, Any] = {
        "run_id": status.get("run_id"),
        "state": status.get("state"),
        "stage_index": status.get("stage_index"),
        "started_at": status.get("started_at"),
        "completed_at": status.get("completed_at"),
        "stages": status.get("stages", []),
        "metrics": status.get("metrics"),
        "last_gate": status.get("last_gate"),
        "heldout_gate": status.get("heldout_gate"),
    }
    if previous_partition is not None:
        previous_run["partition"] = previous_partition
    run_history.append(previous_run)
    migration_history.append(
        {
            "migration_type": "raw_gold_contract_v1_to_v2",
            "migrated_at": migrated_at,
            "previous_config_binding_sha256": observed,
            "new_config_binding_sha256": config.status_binding_sha256,
            "source_contract": source_contract,
            "target_contract": target_contract,
            "migration_trigger": trigger,
            "resume_policy": "fresh_run_stage_zero_after_equivalent_contract_proof",
            "partition_policy": "mark_stale_until_explicit_offline_refresh",
        }
    )
    status.update(
        {
            "config_binding_sha256": config.status_binding_sha256,
            "run_id": uuid4().hex,
            "state": "ready",
            "stage_index": 0,
            "current_concurrency": 0,
            "current_worker": None,
            "cooldown_until": None,
            "started_at": migrated_at,
            "updated_at": migrated_at,
            "completed_at": None,
            "event_sequence": 0,
            "stages": [],
            "metrics": {
                "records": 0,
                "elapsed_seconds": 0.0,
                "throughput_records_per_second": 0.0,
                "eta_seconds": None,
            },
            "last_gate": None,
            "heldout_gate": None,
            "collection_retry": None,
            "usage_checkpoint_policy": _usage_checkpoint_policy(config),
            "partition_stale_reason": "binding_v2_migration_pending_refresh",
        }
    )
    status.pop("partition_refreshed_at", None)
    return status


def _verify_status_config_binding(
    config: AutomationConfig, status: Mapping[str, Any]
) -> None:
    observed = status.get("config_binding_sha256")
    if observed == config.status_binding_sha256:
        return
    if (
        config.status_binding_sha256 != config.legacy_status_binding_sha256
        and observed == config.legacy_status_binding_sha256
    ):
        _legacy_binding_contract_proof(config)
        return
    raise ValueError(
        "automation status config binding mismatch; use a separate output_dir "
        "or remove only an intentionally discarded state directory"
    )


def _target_expansion_contract(config: AutomationConfig) -> dict[str, Any]:
    return {
        "concurrency_stages": config.concurrency_stages,
        "stage_seed_counts": config.stage_seed_counts,
        "raw_collection_target": config.raw_records_per_task,
        "minimum_gold_records_per_task": config.minimum_gold_records_by_task,
        "max_requests": config.max_requests,
        "max_output_tokens_total": config.max_output_tokens_total,
        "max_failures": config.max_failures,
    }


def migrate_monotonic_expansion_status(
    config: AutomationConfig,
) -> dict[str, Any]:
    """Explicitly rebind one quiescent v2 status to a proven larger contract."""

    source = config.monotonic_expansion_from
    source_binding = config.monotonic_expansion_source_binding_sha256
    if source is None or source_binding is None:
        raise ValueError(
            "config does not declare monotonic_expansion_from; migration refused"
        )
    path = config.status_path
    if not path.is_file():
        raise ValueError(
            "automation status does not exist; start the new profile directly"
        )
    status = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(status, dict):
        raise ValueError("automation status must be a JSON object")
    if status.get("schema_version") != AUTOMATION_SCHEMA_VERSION:
        raise ValueError(
            "monotonic expansion requires an existing v2 status; migrate legacy status first"
        )
    observed = status.get("config_binding_sha256")
    target_binding = config.status_binding_sha256
    if observed == target_binding:
        return {
            "status": "already_current",
            "config_binding_sha256": target_binding,
            "target_contract": _target_expansion_contract(config),
        }
    if not isinstance(observed, str) or observed != source_binding:
        raise ValueError(
            "monotonic expansion source binding mismatch; no status changes were made"
        )
    if status.get("state") == "running" or status.get("current_worker") not in (
        None,
        "",
    ):
        raise ValueError(
            "automation status is active; stop the runner before explicit migration"
        )
    run_history = status.setdefault("run_history", [])
    migration_history = status.setdefault("migration_history", [])
    if not isinstance(run_history, list) or not isinstance(migration_history, list):
        raise ValueError("automation status migration history must be a list")

    migrated_at = _iso()
    previous_partition = status.pop("partition", None)
    previous_run = {
        "run_id": status.get("run_id"),
        "state": status.get("state"),
        "stage_index": status.get("stage_index"),
        "started_at": status.get("started_at"),
        "completed_at": status.get("completed_at"),
        "stages": status.get("stages", []),
        "metrics": status.get("metrics"),
        "last_gate": status.get("last_gate"),
        "heldout_gate": status.get("heldout_gate"),
    }
    if previous_partition is not None:
        previous_run["partition"] = previous_partition
    run_history.append(previous_run)
    migration_history.append(
        {
            "migration_type": "monotonic_collection_expansion",
            "migrated_at": migrated_at,
            "previous_config_binding_sha256": observed,
            "new_config_binding_sha256": target_binding,
            "source_contract": source.binding_overrides(),
            "target_contract": _target_expansion_contract(config),
            "resume_policy": (
                "preserve_append_only_rows_reset_stage_zero_and_collect_missing_rows"
            ),
            "partition_policy": "mark_stale_until_explicit_offline_refresh",
        }
    )
    status.update(
        {
            "config_binding_sha256": target_binding,
            "run_id": uuid4().hex,
            "state": "ready",
            "stage_index": 0,
            "current_concurrency": 0,
            "current_worker": None,
            "cooldown_until": None,
            "started_at": migrated_at,
            "updated_at": migrated_at,
            "completed_at": None,
            "stages": [],
            "metrics": {
                "records": 0,
                "elapsed_seconds": 0.0,
                "throughput_records_per_second": 0.0,
                "eta_seconds": None,
            },
            "last_gate": None,
            "heldout_gate": None,
            "collection_retry": None,
            "usage_checkpoint_policy": _usage_checkpoint_policy(config),
            "partition_stale_reason": "monotonic_expansion_pending_refresh",
        }
    )
    status.pop("partition_refreshed_at", None)
    _atomic_write_json(path, status)
    return {
        "status": "migrated",
        "previous_config_binding_sha256": observed,
        "new_config_binding_sha256": target_binding,
        "source_contract": source.binding_overrides(),
        "target_contract": _target_expansion_contract(config),
        "next_state": "ready",
        "partition_status": "stale_until_explicit_offline_refresh",
    }


def _failure_identity(error: str) -> tuple[str, str, str] | None:
    """Return the stable seed/task/error-class identity emitted by the pipeline."""

    parts = [part.strip() for part in error.split(":", 3)]
    if len(parts) < 3 or not all(parts[:3]):
        return None
    task, seed_id, error_class = parts[:3]
    return seed_id, task, error_class


def _failure_key(seed_id: str, task: str, error_class: str) -> str:
    encoded = json.dumps(
        [seed_id, task, error_class], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AutomationRunner:
    def __init__(
        self,
        *,
        config: AutomationConfig,
        teacher: Teacher | None = None,
        teachers: Mapping[str, Teacher] | None = None,
    ) -> None:
        self.config = config
        self._pending_status_events: list[tuple[str, dict[str, Any]]] = []
        self.status = self._load_status()
        self._usage_checkpoint_lock = asyncio.Lock()
        self._usage_unsaved_requests = 0
        self._usage_unsaved_output_tokens = 0
        self._usage_activity_checkpointed = False
        self._last_usage_checkpoint_monotonic = time.monotonic()
        epoch = self.status["quota_epoch"]
        remaining_requests = config.max_requests - int(epoch["requests_used"])
        remaining_tokens = config.max_output_tokens_total - int(
            epoch["output_tokens_used"]
        )
        if teachers is None:
            if teacher is None:
                raise ValueError(
                    "automation requires a teacher or task teacher mapping"
                )
            teachers = {name: teacher for name in ("seed", *TASK_TYPES)}
        missing_workers = set(("seed", *TASK_TYPES)).difference(teachers)
        if missing_workers:
            raise ValueError(f"missing automation workers: {sorted(missing_workers)}")
        wrapper_by_teacher: dict[int, _TrackedTeacher] = {}
        logical_usage_by_budget: dict[int, _LogicalUsageTracker] = {}
        self.workers: dict[str, _TrackedTeacher] = {}
        for name in ("seed", *TASK_TYPES):
            raw_teacher = teachers[name]
            wrapper = wrapper_by_teacher.get(id(raw_teacher))
            if wrapper is None:
                usage_budget_id = int(
                    getattr(raw_teacher, "usage_budget_id", id(raw_teacher))
                )
                logical_usage = logical_usage_by_budget.setdefault(
                    usage_budget_id,
                    _LogicalUsageTracker(
                        max_requests=max(0, remaining_requests),
                        max_output_tokens=max(0, remaining_tokens),
                    ),
                )
                wrapper = _TrackedTeacher(
                    raw_teacher,
                    max_requests=max(0, remaining_requests),
                    max_output_tokens=max(0, remaining_tokens),
                    usage_progress_callback=self._checkpoint_usage,
                    logical_usage=logical_usage,
                )
                wrapper_by_teacher[id(raw_teacher)] = wrapper
            self.workers[name] = wrapper
        self._usage_sources = {
            worker.usage_budget_id: worker for worker in self.workers.values()
        }
        self._usage_baseline = self._aggregate_usage()
        self.events = JsonlStore(config.events_path)
        for event_type, data in self._pending_status_events:
            self._event(event_type, **data)

    def _load_status(self) -> dict[str, Any]:
        path = self.config.status_path
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))
            schema_version = status.get("schema_version")
            if schema_version == LEGACY_AUTOMATION_SCHEMA_VERSION:
                return self._migrate_legacy_status(status)
            if schema_version != AUTOMATION_SCHEMA_VERSION:
                raise ValueError("unsupported automation status schema")
            observed_binding = status.get("config_binding_sha256")
            if observed_binding == self.config.status_binding_sha256:
                pass
            elif (
                self.config.status_binding_sha256
                != self.config.legacy_status_binding_sha256
                and observed_binding == self.config.legacy_status_binding_sha256
            ):
                status = _migrate_legacy_binding_status(
                    self.config,
                    status,
                    trigger="runner_start",
                )
                self._pending_status_events.append(
                    (
                        "collection_contract_migrated",
                        {
                            "from_binding_schema": "v1",
                            "to_binding_schema": "v2",
                            "raw_collection_target": self.config.raw_records_per_task,
                            "minimum_gold_records_per_task": (
                                self.config.minimum_gold_records_by_task
                            ),
                            "resume_policy": "fresh_run_stage_zero_after_equivalent_contract_proof",
                        },
                    )
                )
            else:
                _verify_status_config_binding(self.config, status)
            self._normalize_v2_status(status)
            epoch = status["quota_epoch"]
            if str(epoch.get("epoch_id")) != self.config.quota_epoch_id:
                archived = deepcopy(epoch)
                archived["closed_at"] = _iso()
                archived["close_reason"] = "quota_epoch_changed"
                status["quota_history"].append(archived)
                previous_epoch_id = str(epoch.get("epoch_id", "unknown"))
                status["quota_epoch"] = _new_quota_epoch(self.config)
                status["cooldown_until"] = None
                if status.get("state") != "complete":
                    status["state"] = "running"
                self._pending_status_events.append(
                    (
                        "quota_epoch_started",
                        {
                            "previous_epoch_id": previous_epoch_id,
                            "quota_epoch_id": self.config.quota_epoch_id,
                            "reason": "configured_epoch_changed",
                        },
                    )
                )
            else:
                epoch["max_requests"] = self.config.max_requests
                epoch["max_output_tokens_total"] = self.config.max_output_tokens_total
                epoch["max_failures"] = self.config.max_failures
            return status
        return {
            "schema_version": AUTOMATION_SCHEMA_VERSION,
            "config_binding_sha256": self.config.status_binding_sha256,
            "run_id": uuid4().hex,
            "state": "ready",
            "stage_index": 0,
            "current_concurrency": 0,
            "current_worker": None,
            "cooldown_until": None,
            "started_at": _iso(),
            "updated_at": _iso(),
            "completed_at": None,
            "event_sequence": 0,
            "usage_checkpoint_policy": _usage_checkpoint_policy(self.config),
            "quota_epoch": _new_quota_epoch(self.config),
            "quota_history": [],
            "audit_ledger": {
                "requests_total": 0,
                "output_tokens_total": 0,
                "failure_observations_total": 0,
                "failure_entries": {},
            },
            "stages": [],
            "metrics": {
                "records": 0,
                "elapsed_seconds": 0.0,
                "throughput_records_per_second": 0.0,
                "eta_seconds": None,
            },
            "last_gate": None,
            "heldout_gate": None,
            "collection_retry": None,
        }

    def _normalize_v2_status(self, status: dict[str, Any]) -> None:
        status["usage_checkpoint_policy"] = _usage_checkpoint_policy(self.config)
        status.setdefault("collection_retry", None)
        status.setdefault("quota_history", [])
        ledger = status.setdefault("audit_ledger", {})
        ledger.setdefault("requests_total", 0)
        ledger.setdefault("output_tokens_total", 0)
        ledger.setdefault("failure_observations_total", 0)
        ledger.setdefault("failure_entries", {})
        epoch = status.setdefault("quota_epoch", _new_quota_epoch(self.config))
        epoch.setdefault("charged_failure_keys", [])
        epoch.setdefault("requests_used", 0)
        epoch.setdefault("output_tokens_used", 0)
        epoch.setdefault("failures_used", len(epoch["charged_failure_keys"]))

    def _migrate_legacy_status(self, legacy: dict[str, Any]) -> dict[str, Any]:
        """Start a fresh v2 epoch rather than guessing legacy stage semantics.

        v1 status did not persist the stage schedule. Retaining a non-terminal
        v1 ``stage_index`` under a changed schedule can skip every configured
        stage and incorrectly mark the run complete. Preserve the old status
        and budgets as audit history, but reset the executable run cursor.
        """

        previous_state = str(legacy.get("state", "unknown"))
        if previous_state == "complete" or legacy.get("partition") is not None:
            raise ValueError(
                "schema-v1 completed or partitioned status lacks a provable corpus "
                "contract; explicit source proof is required"
            )
        status = deepcopy(legacy)
        old_budgets = deepcopy(status.pop("budgets", {}))
        migrated_at = _iso()
        old_budgets.update(
            {
                "epoch_id": "legacy-v1",
                "started_at": status.get("started_at"),
                "closed_at": migrated_at,
                "close_reason": "schema_v1_migration",
                "charged_failure_keys": [],
            }
        )
        status["schema_version"] = AUTOMATION_SCHEMA_VERSION
        status["config_binding_sha256"] = self.config.status_binding_sha256
        status["quota_epoch"] = _new_quota_epoch(self.config)
        status["usage_checkpoint_policy"] = _usage_checkpoint_policy(self.config)
        status["collection_retry"] = None
        status["quota_history"] = [old_budgets]
        status["audit_ledger"] = {
            "requests_total": int(old_budgets.get("requests_used", 0)),
            "output_tokens_total": int(old_budgets.get("output_tokens_used", 0)),
            "failure_observations_total": int(old_budgets.get("failures_used", 0)),
            "failure_entries": {},
            "legacy_unkeyed_failures": int(old_budgets.get("failures_used", 0)),
        }
        previous_stage_index = status.get("stage_index")
        migration: dict[str, Any] = {
            "from_schema": LEGACY_AUTOMATION_SCHEMA_VERSION,
            "to_schema": AUTOMATION_SCHEMA_VERSION,
            "migrated_at": migrated_at,
            "legacy_status": deepcopy(legacy),
        }
        migration["resume_policy"] = "fresh_epoch_stage_zero"
        migration["previous_stage_index"] = previous_stage_index
        # A legacy v1 stage count cannot be proven compatible with the current
        # operator config. A clean v2 epoch avoids a false completion while
        # keeping old details in migration history. Completed/partitioned v1
        # statuses are rejected above because resetting them would otherwise
        # silently bind an unprovable historical corpus to the current floor.
        status.update(
            {
                "run_id": uuid4().hex,
                "state": "ready",
                "stage_index": 0,
                "current_concurrency": 0,
                "current_worker": None,
                "cooldown_until": None,
                "started_at": migrated_at,
                "updated_at": migrated_at,
                "completed_at": None,
                "event_sequence": 0,
                "stages": [],
                "metrics": {
                    "records": 0,
                    "elapsed_seconds": 0.0,
                    "throughput_records_per_second": 0.0,
                    "eta_seconds": None,
                },
                "last_gate": None,
                "heldout_gate": None,
                "collection_retry": None,
                "partition_stale_reason": "schema_v1_migration_pending_refresh",
            }
        )
        status.pop("partition_refreshed_at", None)
        status.setdefault("migration_history", []).append(migration)
        self._pending_status_events.append(
            (
                "status_migrated",
                {
                    "from_schema": LEGACY_AUTOMATION_SCHEMA_VERSION,
                    "to_schema": AUTOMATION_SCHEMA_VERSION,
                    "quota_epoch_id": self.config.quota_epoch_id,
                    "resume_policy": migration["resume_policy"],
                    "previous_stage_index": previous_stage_index,
                },
            )
        )
        return status

    def _save_status(self) -> None:
        self.status["updated_at"] = _iso()
        path = self.config.status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        had_usage_progress = bool(
            getattr(self, "_usage_unsaved_requests", 0)
            or getattr(self, "_usage_unsaved_output_tokens", 0)
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.status, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(self, "_usage_unsaved_requests"):
            self._usage_unsaved_requests = 0
            self._usage_unsaved_output_tokens = 0
            if had_usage_progress:
                self._usage_activity_checkpointed = True
            self._last_usage_checkpoint_monotonic = time.monotonic()

    def _event(self, event_type: str, **data: Any) -> None:
        self.status["event_sequence"] = int(self.status["event_sequence"]) + 1
        sequence = int(self.status["event_sequence"])
        event = {
            "id": f"event_{self.status['run_id']}_{sequence:08d}",
            "time": _iso(),
            "type": event_type,
            "run_id": self.status["run_id"],
            "stage_index": self.status["stage_index"],
            "data": data,
        }
        self.events.append(event)
        self._save_status()

    def _sync_usage(self) -> tuple[int, int]:
        current = self._aggregate_usage()
        requests_delta = max(0, current["requests"] - self._usage_baseline["requests"])
        tokens_delta = max(
            0, current["output_tokens"] - self._usage_baseline["output_tokens"]
        )
        epoch = self.status["quota_epoch"]
        epoch["requests_used"] += requests_delta
        epoch["output_tokens_used"] += tokens_delta
        ledger = self.status["audit_ledger"]
        ledger["requests_total"] += requests_delta
        ledger["output_tokens_total"] += tokens_delta
        self._usage_baseline = current
        self._usage_unsaved_requests += requests_delta
        self._usage_unsaved_output_tokens += tokens_delta
        return requests_delta, tokens_delta

    async def _checkpoint_usage(self, *, force: bool = False) -> None:
        """Serialize and bound durable usage checkpoints during long stages."""

        async with self._usage_checkpoint_lock:
            self._sync_usage()
            pending = bool(
                self._usage_unsaved_requests or self._usage_unsaved_output_tokens
            )
            if not pending:
                return
            elapsed = time.monotonic() - self._last_usage_checkpoint_monotonic
            due = (
                force
                or not self._usage_activity_checkpointed
                or self._usage_unsaved_requests >= _USAGE_CHECKPOINT_REQUEST_INTERVAL
                or self._usage_unsaved_output_tokens
                >= _USAGE_CHECKPOINT_OUTPUT_TOKEN_INTERVAL
                or elapsed >= _USAGE_CHECKPOINT_MAX_SECONDS
            )
            if due:
                self._save_status()

    def _aggregate_usage(self) -> dict[str, int]:
        snapshots = [worker.usage_snapshot for worker in self._usage_sources.values()]
        return {
            "requests": sum(item["requests"] for item in snapshots),
            "output_tokens": sum(item["output_tokens"] for item in snapshots),
        }

    def _budget_exhausted(self) -> str | None:
        epoch = self.status["quota_epoch"]
        if epoch["requests_used"] >= self.config.max_requests:
            return "request_budget"
        if epoch["output_tokens_used"] >= self.config.max_output_tokens_total:
            return "output_token_budget"
        if (
            epoch["failures_used"] > 0
            and epoch["failures_used"] >= self.config.max_failures
        ):
            return "failure_budget"
        return None

    def _record_report_failures(self, errors: Sequence[str]) -> int:
        """Audit failures and charge each stable identity once per quota epoch."""

        ledger = self.status["audit_ledger"]
        entries: dict[str, dict[str, Any]] = ledger["failure_entries"]
        epoch = self.status["quota_epoch"]
        charged = set(str(key) for key in epoch["charged_failure_keys"])
        new_charges = 0
        seen_this_report: set[str] = set()
        quarantine_events: list[dict[str, Any]] = []
        for error in errors:
            identity = _failure_identity(error)
            if identity is None:
                continue
            seed_id, task, error_class = identity
            if error_class in NON_CHARGEABLE_FAILURE_CLASSES:
                continue
            key = _failure_key(seed_id, task, error_class)
            if key in seen_this_report:
                continue
            seen_this_report.add(key)
            now = _iso()
            entry = entries.setdefault(
                key,
                {
                    "seed_id": seed_id,
                    "task": task,
                    "error_class": error_class,
                    "attempts_total": 0,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "quarantined": False,
                    "quarantined_at": None,
                },
            )
            entry["attempts_total"] = int(entry["attempts_total"]) + 1
            entry["last_seen_at"] = now
            ledger["failure_observations_total"] += 1
            if key not in charged:
                charged.add(key)
                new_charges += 1
            if (
                not entry["quarantined"]
                and int(entry["attempts_total"]) > self.config.max_failure_retries
            ):
                entry["quarantined"] = True
                entry["quarantined_at"] = now
                entry["quarantine_reason"] = "retry_limit_exceeded"
                quarantine_events.append(
                    {
                        "failure_key": key,
                        "seed_id": seed_id,
                        "task": task,
                        "error_class": error_class,
                        "attempts_total": entry["attempts_total"],
                        "max_failure_retries": self.config.max_failure_retries,
                    }
                )
        epoch["charged_failure_keys"] = sorted(charged)
        epoch["failures_used"] = len(charged)
        for event in quarantine_events:
            self._event("failure_quarantined", **event)
        return new_charges

    def _append_failure_attempts(self, errors: Sequence[str]) -> None:
        """Persist content-free failure observations for offline accounting.

        Teacher text and exception messages are deliberately excluded: malformed
        responses, request-structure failures, and credentials remain hard rejects.
        The audit ledger retains the repeat count while this JSONL provides a
        stable, append-only attempt index for later partition reports.
        """

        store = JsonlStore(self.config.attempts_path)
        entries = self.status["audit_ledger"]["failure_entries"]
        seen: set[str] = set()
        for error in errors:
            identity = _failure_identity(error)
            if identity is None:
                continue
            seed_id, task, error_class = identity
            if error_class in NON_CHARGEABLE_FAILURE_CLASSES:
                continue
            key = _failure_key(seed_id, task, error_class)
            if key in seen:
                continue
            seen.add(key)
            entry = entries.get(key, {})
            attempt_number = int(entry.get("attempts_total", 1))
            store.append(
                {
                    "id": stable_id(
                        "attempt", f"{key}:{attempt_number}:{self.status['run_id']}"
                    ),
                    "schema_version": "anchor.automation-attempt.v1",
                    "run_id": self.status["run_id"],
                    "seed_id": seed_id,
                    "task_type": task,
                    "outcome": "hard_reject",
                    "error_class": error_class,
                    "attempt_number": attempt_number,
                    "observed_at": _iso(),
                    "teacher_content_retained": False,
                }
            )

    def _quarantined_seed_ids_for_task(self, task_type: str) -> frozenset[str]:
        task_index = TASK_TYPES.index(task_type)  # type: ignore[arg-type]
        blocked_tasks = set(TASK_TYPES[: task_index + 1])
        entries = self.status["audit_ledger"]["failure_entries"].values()
        return frozenset(
            str(entry["seed_id"])
            for entry in entries
            if entry.get("quarantined") and entry.get("task") in blocked_tasks
        )

    def _partition_terminal_collection(
        self, *, seed_target: int, terminal_state: str, reason: str
    ) -> None:
        """Partition safe partial output when a provider/budget window closes."""

        for task_type in TASK_TYPES:
            path = self.config.output_dir / f"data_{task_type}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        heldout_gate = evaluate_heldout_scale_gate(self.config)
        self.status["heldout_gate"] = heldout_gate
        self._event("heldout_leakage_gate", **heldout_gate)
        if not heldout_gate["passed"]:
            self.status["state"] = "gate_blocked"
            self._event(
                "collection_hard_blocked",
                reason="heldout_or_integrity_gate",
                heldout_leakage=heldout_gate,
            )
            return
        try:
            partition = partition_collected_records(self.config, seed_target)
        except (OSError, ValueError) as error:
            self.status["state"] = "failed"
            self._event(
                "collection_hard_blocked",
                reason="malformed_or_unsafe_staging",
                error_type=type(error).__name__,
            )
            return
        self.status["partition"] = partition
        self.status.pop("partition_stale_reason", None)
        self.status["state"] = terminal_state
        self._event(
            "collection_partitioned",
            seed_target=seed_target,
            terminal_reason=reason,
            partition=partition,
        )

    async def run(self, *, wait_for_cooldown: bool = False) -> dict[str, Any]:
        if self.status["state"] in {
            "complete",
            "provider_quota_exhausted",
            "budget_exhausted",
            "client_deadline",
            "failed",
            "gate_blocked",
        }:
            return self.status
        if self.status["state"] == "ready":
            self.status["state"] = "running"
            self._event(
                "automation_started", stages=list(self.config.concurrency_stages)
            )

        retry_stage_index: int | None = None
        previous_gate_records: int | None = None
        stagnant_gate_rounds = 0
        while int(self.status["stage_index"]) < len(self.config.concurrency_stages):
            if await self._cooldown_gate(wait_for_cooldown=wait_for_cooldown):
                return self.status
            exhausted = self._budget_exhausted()
            if exhausted:
                await self._checkpoint_usage(force=True)
                self.status["state"] = "budget_exhausted"
                self._event("budget_exhausted", budget=exhausted)
                return self.status

            stage_index = int(self.status["stage_index"])
            if retry_stage_index != stage_index:
                retry_stage_index = stage_index
                previous_gate_records = None
                stagnant_gate_rounds = 0
            concurrency = self.config.concurrency_stages[stage_index]
            target = (
                self.config.raw_records_per_task
                if stage_index == len(self.config.concurrency_stages) - 1
                else self.config.stage_seed_counts[stage_index]
            )
            self.status["state"] = "running"
            self.status["current_concurrency"] = concurrency
            self._event("stage_started", concurrency=concurrency, seed_target=target)
            started = time.monotonic()
            try:
                report = await self._run_stage(
                    seed_target=target, concurrency=concurrency
                )
            except ProviderQuotaExhausted as error:
                await self._checkpoint_usage(force=True)
                self.status["state"] = "provider_quota_exhausted"
                self.status["quota_epoch"]["closed_at"] = _iso()
                self.status["quota_epoch"]["close_reason"] = "provider_quota_exhausted"
                self._event(
                    "provider_quota_exhausted",
                    classification="explicit_provider_quota",
                    retry_after_seconds=error.retry_after_seconds,
                )
                if self.config.collection_policy == "collect_then_partition":
                    self._partition_terminal_collection(
                        seed_target=target,
                        terminal_state="provider_quota_exhausted",
                        reason="explicit_provider_quota",
                    )
                return self.status
            except RateLimitError as error:
                await self._checkpoint_usage(force=True)
                self._set_cooldown(error.retry_after_seconds)
                if not wait_for_cooldown:
                    return self.status
                continue
            except ClientDeadlineExceeded as error:
                await self._checkpoint_usage(force=True)
                self.status["state"] = "client_deadline"
                self.status["last_client_deadline"] = {
                    "worker": "seed",
                    "seconds": error.seconds,
                    "time": _iso(),
                }
                self._event(
                    "client_deadline",
                    worker="seed",
                    seconds=error.seconds,
                    classification="client_deadline",
                )
                return self.status
            except (BudgetExceeded, RuntimeError, ValueError, OSError) as error:
                await self._checkpoint_usage(force=True)
                worker = str(self.status.get("current_worker") or "seed")
                self._record_report_failures(
                    [f"{worker}:__stage__: {type(error).__name__}: {error}"]
                )
                self.status["state"] = "failed"
                self._event(
                    "stage_failed",
                    error_type=type(error).__name__,
                    message=str(error)[:240],
                )
                return self.status

            elapsed = max(0.000001, time.monotonic() - started)
            await self._checkpoint_usage(force=True)
            self._record_report_failures(report.errors)
            self._append_failure_attempts(report.errors)
            if report.rate_limited:
                if report.provider_quota_exhausted:
                    self.status["state"] = "provider_quota_exhausted"
                    self.status["quota_epoch"]["closed_at"] = _iso()
                    self.status["quota_epoch"]["close_reason"] = (
                        "provider_quota_exhausted"
                    )
                    self._event(
                        "provider_quota_exhausted",
                        classification="explicit_provider_quota",
                        retry_after_seconds=report.retry_after_seconds,
                    )
                    if self.config.collection_policy == "collect_then_partition":
                        self._partition_terminal_collection(
                            seed_target=target,
                            terminal_state="provider_quota_exhausted",
                            reason="explicit_provider_quota",
                        )
                    return self.status
                self._set_cooldown(report.retry_after_seconds)
                if not wait_for_cooldown:
                    return self.status
                continue
            if report.client_deadline:
                self.status["state"] = "client_deadline"
                deadline_error = next(
                    (
                        item
                        for item in report.errors
                        if "ClientDeadlineExceeded" in item
                    ),
                    "task worker exceeded client wall-clock deadline",
                )
                worker = deadline_error.split(":", 1)[0]
                self.status["last_client_deadline"] = {
                    "worker": worker,
                    "message": deadline_error[:240],
                    "time": _iso(),
                }
                self._event(
                    "client_deadline",
                    worker=worker,
                    message=deadline_error[:240],
                    classification="client_deadline",
                )
                return self.status
            exhausted = self._budget_exhausted()
            if exhausted:
                if self.config.collection_policy == "collect_then_partition":
                    self._partition_terminal_collection(
                        seed_target=target,
                        terminal_state="budget_exhausted",
                        reason=exhausted,
                    )
                    return self.status
                self.status["state"] = "budget_exhausted"
                self._event("budget_exhausted", budget=exhausted)
                return self.status

            gate = evaluate_gate(self.config, target)
            heldout_gate = evaluate_heldout_scale_gate(self.config)
            gate["heldout_leakage"] = heldout_gate
            gate["passed"] = bool(gate["passed"] and heldout_gate["passed"])
            self.status["last_gate"] = gate
            self.status["heldout_gate"] = heldout_gate
            self._event("heldout_leakage_gate", **heldout_gate)
            self._update_metrics(gate["records"], elapsed)
            stage_result = {
                "index": stage_index,
                "concurrency": concurrency,
                "seed_target": target,
                "elapsed_seconds": elapsed,
                "gate": gate,
                "report": asdict(report),
            }
            self.status["stages"].append(stage_result)
            if self.config.collection_policy == "collect_then_partition":
                # Structural parsing, secret/safety filtering, and held-out
                # isolation remain fail-closed. Ordinary model quality is
                # retained and classified only after the collection stage.
                if not heldout_gate["passed"]:
                    self.status["state"] = "gate_blocked"
                    self._event(
                        "collection_hard_blocked",
                        reason="heldout_or_integrity_gate",
                        heldout_leakage=heldout_gate,
                    )
                    return self.status
                try:
                    partition = partition_collected_records(self.config, target)
                except (OSError, ValueError) as error:
                    self.status["state"] = "failed"
                    self._event(
                        "collection_hard_blocked",
                        reason="malformed_or_unsafe_staging",
                        error_type=type(error).__name__,
                    )
                    return self.status
                self.status["partition"] = partition
                self.status.pop("partition_stale_reason", None)
                if partition.get("training_ready") is True:
                    self.status["collection_retry"] = None
                    self.status["stage_index"] = stage_index + 1
                    self._event(
                        "collection_partitioned",
                        concurrency=concurrency,
                        seed_target=target,
                        partition=partition,
                        post_collection_gate_passed=gate["passed"],
                        stage_advanced=True,
                    )
                    continue

                raw_by_task = partition.get("raw_by_task", {})
                gold_by_task = partition.get("gold_by_task", {})
                progress = {
                    "raw_records": sum(int(value) for value in raw_by_task.values())
                    if isinstance(raw_by_task, Mapping)
                    else 0,
                    "gold_records": sum(int(value) for value in gold_by_task.values())
                    if isinstance(gold_by_task, Mapping)
                    else 0,
                    "complete_chains": int(partition.get("complete_chain_count", 0)),
                }
                retry = self.status.get("collection_retry")
                if (
                    not isinstance(retry, dict)
                    or int(retry.get("stage_index", -1)) != stage_index
                ):
                    retry = {
                        "stage_index": stage_index,
                        "rounds": 0,
                        "stagnant_rounds": 0,
                        "last_progress": None,
                    }
                previous_progress = retry.get("last_progress")
                progressed = not isinstance(previous_progress, Mapping) or any(
                    progress[key] > int(previous_progress.get(key, 0))
                    for key in progress
                )
                retry["rounds"] = int(retry.get("rounds", 0)) + 1
                retry["stagnant_rounds"] = (
                    0 if progressed else int(retry.get("stagnant_rounds", 0)) + 1
                )
                retry["last_progress"] = progress
                retry["raw_collection_shortfalls"] = dict(
                    partition.get("raw_collection_shortfalls", {})
                )
                retry["coverage_shortfalls"] = dict(
                    partition.get("coverage_shortfalls", {})
                )
                self.status["collection_retry"] = retry
                self._event(
                    "collection_partitioned",
                    concurrency=concurrency,
                    seed_target=target,
                    partition=partition,
                    post_collection_gate_passed=gate["passed"],
                    stage_advanced=False,
                    retry=retry,
                )

                quarantined = [
                    entry
                    for entry in self.status["audit_ledger"]["failure_entries"].values()
                    if entry.get("quarantined")
                ]
                if quarantined and partition.get("raw_collection_shortfalls"):
                    self.status["state"] = "gate_blocked"
                    self._event(
                        "collection_gate_blocked",
                        reason="failure_retry_limit_exhausted",
                        stagnant_rounds=retry["stagnant_rounds"],
                        quarantined_failure_count=len(quarantined),
                        partition=partition,
                    )
                    return self.status
                if (
                    int(retry["stagnant_rounds"])
                    >= self.config.max_stagnant_gate_rounds
                ):
                    self.status["state"] = "gate_blocked"
                    self._event(
                        "collection_gate_blocked",
                        reason="stagnant_collection_rounds",
                        stagnant_rounds=retry["stagnant_rounds"],
                        max_stagnant_gate_rounds=(self.config.max_stagnant_gate_rounds),
                        partition=partition,
                    )
                    return self.status
                self.status["state"] = "running"
                self._event(
                    "collection_retry_scheduled",
                    retry=retry,
                    partition=partition,
                    max_failure_retries=self.config.max_failure_retries,
                    max_stagnant_gate_rounds=(self.config.max_stagnant_gate_rounds),
                )
                continue
            if not gate["passed"]:
                records = int(gate["records"])
                if (
                    previous_gate_records is not None
                    and records <= previous_gate_records
                ):
                    stagnant_gate_rounds += 1
                else:
                    stagnant_gate_rounds = 0
                previous_gate_records = records
                if stagnant_gate_rounds >= self.config.max_stagnant_gate_rounds:
                    self.status["state"] = "gate_blocked"
                    self._event(
                        "gate_blocked",
                        gate=gate,
                        reason="stagnant_gate_rounds",
                        stagnant_gate_rounds=stagnant_gate_rounds,
                    )
                    return self.status
                self.status["state"] = "running"
                self._event(
                    "gate_retry_scheduled",
                    gate=gate,
                    stagnant_gate_rounds=stagnant_gate_rounds,
                    max_stagnant_gate_rounds=self.config.max_stagnant_gate_rounds,
                )
                continue
            self.status["stage_index"] = stage_index + 1
            self._event(
                "gate_passed",
                concurrency=concurrency,
                seed_target=target,
                gate=gate,
                metrics=self.status["metrics"],
            )

        await self._checkpoint_usage(force=True)
        self.status["state"] = "complete"
        self.status["current_concurrency"] = 0
        self.status["current_worker"] = None
        self.status["completed_at"] = _iso()
        self._event("automation_completed", metrics=self.status["metrics"])
        return self.status

    async def _run_stage(self, *, seed_target: int, concurrency: int) -> PipelineReport:
        self.status["current_worker"] = "seed"
        self._event("worker_started", worker="seed", concurrency=concurrency)
        seed_pipeline = DistillationPipeline(
            teacher=self.workers["seed"],
            sop_dir=self.config.sop_dir,
            output_dir=self.config.output_dir,
            concurrency=concurrency,
            seed_index_offset=self.config.seed_index_offset,
            task_card_config=self.config.task_card_config,
            progress_callback=self._checkpoint_usage,
        )
        await seed_pipeline.generate_seeds(seed_target)
        self._event("worker_completed", worker="seed", seed_target=seed_target)
        written: dict[str, int] = {}
        skipped: dict[str, int] = {}
        errors: list[str] = []
        rate_limited = False
        retry_after: float | None = None
        client_deadline = False
        provider_quota_exhausted = False
        for task_type in TASK_TYPES:
            self.status["current_worker"] = task_type
            self._event("worker_started", worker=task_type, concurrency=concurrency)
            worker_pipeline = DistillationPipeline(
                teacher=self.workers[task_type],
                sop_dir=self.config.sop_dir,
                output_dir=self.config.output_dir,
                concurrency=concurrency,
                seed_index_offset=self.config.seed_index_offset,
                task_card_config=self.config.task_card_config,
                progress_callback=self._checkpoint_usage,
            )
            excluded_seed_ids = self._quarantined_seed_ids_for_task(task_type)
            report = await worker_pipeline.run(
                seed_count=seed_target,
                tasks=[task_type],
                excluded_seed_ids=excluded_seed_ids,
            )
            # A structurally valid empty file is meaningful in collect-first
            # mode: it records zero accepted responses without confusing the
            # held-out scanner with a missing source.
            if self.config.collection_policy == "collect_then_partition":
                output_path = self.config.output_dir / f"data_{task_type}.jsonl"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch(exist_ok=True)
            written.update(report.written_by_task)
            skipped.update(report.skipped_by_task)
            errors.extend(report.errors)
            rate_limited = rate_limited or report.rate_limited
            client_deadline = client_deadline or report.client_deadline
            provider_quota_exhausted = (
                provider_quota_exhausted or report.provider_quota_exhausted
            )
            if report.retry_after_seconds is not None:
                retry_after = max(retry_after or 0.0, report.retry_after_seconds)
            self._event(
                "worker_completed",
                worker=task_type,
                written=report.written_by_task.get(task_type, 0),
                skipped=report.skipped_by_task.get(task_type, 0),
                errors=len(report.errors),
                rate_limited=report.rate_limited,
                client_deadline=report.client_deadline,
                quarantined_skipped=len(excluded_seed_ids),
            )
            if rate_limited or client_deadline:
                break
        self.status["current_worker"] = None
        return PipelineReport(
            requested_seeds=seed_target,
            available_seeds=seed_target,
            written_by_task=written,
            skipped_by_task=skipped,
            errors=tuple(errors),
            rate_limited=rate_limited,
            retry_after_seconds=retry_after,
            client_deadline=client_deadline,
            provider_quota_exhausted=provider_quota_exhausted,
        )

    async def _cooldown_gate(self, *, wait_for_cooldown: bool) -> bool:
        while True:
            raw = self.status.get("cooldown_until")
            if not raw:
                return False
            until = datetime.fromisoformat(str(raw))
            remaining = (until - _now()).total_seconds()
            if remaining <= 0:
                self.status["cooldown_until"] = None
                self.status["state"] = "running"
                self._event("cooldown_completed")
                return False
            self.status["state"] = "cooldown"
            self._save_status()
            if not wait_for_cooldown:
                return True
            await asyncio.sleep(
                min(float(self.config.cooldown_poll_seconds), remaining)
            )

    def _set_cooldown(self, retry_after_seconds: float | None) -> None:
        seconds = max(float(self.config.cooldown_seconds), retry_after_seconds or 0.0)
        until = _now() + timedelta(seconds=seconds)
        self.status["state"] = "cooldown"
        self.status["cooldown_until"] = _iso(until)
        self._event(
            "rate_limit_cooldown",
            retry_after_seconds=retry_after_seconds,
            cooldown_seconds=seconds,
            cooldown_until=self.status["cooldown_until"],
        )

    def _update_metrics(self, records: int, stage_elapsed: float) -> None:
        started = datetime.fromisoformat(str(self.status["started_at"]))
        elapsed = max(0.000001, (_now() - started).total_seconds())
        throughput = records / elapsed
        final_records = self.config.raw_records_per_task * len(TASK_TYPES)
        remaining = max(0, final_records - records)
        self.status["metrics"] = {
            "records": records,
            "elapsed_seconds": elapsed,
            "last_stage_seconds": stage_elapsed,
            "throughput_records_per_second": throughput,
            "eta_seconds": remaining / throughput if throughput > 0 else None,
        }


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("id", "missing-record-id"))


def _tool_policy_oracle_error(record: Mapping[str, Any]) -> str | None:
    """Return a public error when a tool-policy row is not local-oracle gold."""

    raw_input = record.get("input")
    raw_output = record.get("output")
    provenance = record.get("provenance")
    if not isinstance(raw_input, Mapping) or not isinstance(raw_output, Mapping):
        return "missing canonical input or output"
    proposals_raw = raw_input.get("tool_proposals")
    if not isinstance(proposals_raw, list) or not proposals_raw:
        return "missing inert tool proposals"
    proposals: list[dict[str, str]] = []
    for proposal in proposals_raw:
        if not isinstance(proposal, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in proposal.items()
        ):
            return "tool proposals are not canonical string mappings"
        proposals.append(dict(proposal))
    expected_output, expected_oracle = deterministic_tool_policy_oracle(proposals)
    if dict(raw_output) != expected_output:
        return "output differs from deterministic tool-policy oracle"
    if not isinstance(provenance, Mapping):
        return "missing provenance"
    if provenance.get("label_oracle") != expected_oracle:
        return "label_oracle differs from deterministic tool-policy oracle"
    proposal_manifest = provenance.get("tool_proposals")
    if not isinstance(proposal_manifest, Mapping):
        return "missing inert tool-proposal provenance"
    canonical = json.dumps(
        proposals, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    expected_proposal_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (
        proposal_manifest.get("generator") != PROPOSAL_GENERATOR_VERSION
        or proposal_manifest.get("executed") is not False
        or proposal_manifest.get("count") != len(proposals)
        or proposal_manifest.get("sha256") != expected_proposal_hash
    ):
        return "tool-proposal provenance is not deterministic inert metadata"
    return None


def _security_oracle_error(record: Mapping[str, Any]) -> str | None:
    """Return a public error when a security row is not local-fixture gold."""

    raw_input = record.get("input")
    raw_output = record.get("output")
    provenance = record.get("provenance")
    if not isinstance(raw_input, Mapping) or not isinstance(raw_output, Mapping):
        return "missing canonical input or output"
    reviewed_code = raw_input.get("reviewed_code")
    if not isinstance(reviewed_code, str):
        return "missing reviewed_code"
    try:
        expected_output, fixture_manifest = deterministic_security_fixture_oracle(
            reviewed_code
        )
    except ValueError as error:
        return f"security fixture is not canonical: {str(error)[:120]}"
    expected_oracle = {
        "oracle": "anchor-security-fixture-gold-v1",
        "decision": expected_output["decision"],
        "sha256": fixture_manifest["gold_sha256"],
    }
    if dict(raw_output) != expected_output:
        return "output differs from deterministic security fixture"
    if not isinstance(provenance, Mapping):
        return "missing provenance"
    if provenance.get("security_fixture") != fixture_manifest:
        return "security_fixture provenance differs from deterministic fixture"
    if provenance.get("label_oracle") != expected_oracle:
        return "label_oracle differs from deterministic security fixture"
    return None


def _oracle_normalized_disagreement_is_training_safe(
    task_type: str, record: Mapping[str, Any]
) -> bool:
    """Prove that no contrary teacher rationale remains in an oracle target.

    New rows carry explicit provenance. Pre-v6 rows are accepted only when
    their trace exactly matches the historical deterministic replacement and
    their assistant target is the canonical oracle label. Anything ambiguous
    remains a quality negative.
    """

    provenance = record.get("provenance")
    output = record.get("output")
    messages = record.get("messages")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(messages, list)
        or not messages
        or not isinstance(messages[-1], Mapping)
    ):
        return False
    decision = output.get("decision")
    if not isinstance(decision, str):
        return False
    expected_target = decision if task_type == "tool_policy" else f"[{decision}]"
    if str(messages[-1].get("content", "")).strip() != expected_target:
        return False
    label_oracle = provenance.get("label_oracle")
    if (
        not isinstance(label_oracle, Mapping)
        or label_oracle.get("decision") != decision
    ):
        return False

    explicit_source = provenance.get("supervision_source")
    if explicit_source is not None:
        return bool(
            explicit_source == "deterministic_oracle"
            and provenance.get("oracle_normalized") is True
            and provenance.get("teacher_decision_agrees_with_oracle") is False
            and provenance.get("decision_trace_source") == "deterministic_oracle"
        )

    trace = record.get("decision_trace")
    if (
        not isinstance(trace, list)
        or len(trace) != 1
        or not isinstance(trace[0], Mapping)
    ):
        return False
    return bool(
        trace[0].get("check") == "deterministic label oracle"
        and trace[0].get("evidence")
        == "The inert fixture or proposal manifest defines the gold class."
        and trace[0].get("action")
        == f"Emit {decision} without executing or reconstructing payloads."
    )


def _artifact_code(record: Mapping[str, Any]) -> tuple[str, str] | None:
    output = record.get("output")
    if not isinstance(output, Mapping):
        return None
    language = str(output.get("language", "")).casefold()
    code = output.get("code")
    if language not in {"tsx", "jsx"} or not isinstance(code, str) or not code.strip():
        return None
    return language, code


def _evaluate_artifact_gate(
    config: AutomationConfig,
    records_by_task: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build/test generated TSX data in isolated fixture workspaces.

    The data schema already binds a review row to its frontend source and a
    deterministic mutation manifest, so this gate can verify the full same-seed
    DAG without inventing a repository scaffold or executing generated code.
    """

    if config.artifact_validation_fixture is None:
        return {"enabled": False, "passed": True, "status": "DISABLED"}
    assert config.artifact_validation_workspace_root is not None
    fixture_root = config.artifact_validation_fixture
    workspace_root = config.artifact_validation_workspace_root
    frontend_by_id = {
        _record_id(record): record for record in records_by_task.get("frontend", [])
    }
    cache: dict[str, bool] = {}
    errors: list[str] = []
    checked = {"frontend": 0, "review": 0}

    def validate_record(task_type: str, record: Mapping[str, Any]) -> None:
        artifact = _artifact_code(record)
        record_id = _record_id(record)
        checked[task_type] += 1
        if artifact is None:
            errors.append(f"{task_type}:{record_id}:missing_tsx_artifact")
            return
        _, code = artifact
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if digest not in cache:
            try:
                report = validate_tsx_fragment(
                    code,
                    fixture_root=fixture_root,
                    workspace_root=workspace_root,
                    timeout_seconds=config.artifact_validation_timeout_seconds,
                )
                cache[digest] = bool(report.get("passed"))
            except (OSError, ValueError) as error:
                cache[digest] = False
                errors.append(
                    f"{task_type}:{record_id}:validator_error:{type(error).__name__}"
                )
        if not cache[digest]:
            errors.append(f"{task_type}:{record_id}:build_or_test_failed")

    for record in records_by_task.get("frontend", []):
        validate_record("frontend", record)

    for record in records_by_task.get("review", []):
        record_id = _record_id(record)
        provenance = record.get("provenance")
        raw_input = record.get("input")
        if not isinstance(provenance, Mapping) or not isinstance(raw_input, Mapping):
            errors.append(f"review:{record_id}:missing_dag_provenance")
            validate_record("review", record)
            continue
        source_id = str(provenance.get("source_frontend_record_id", ""))
        source = frontend_by_id.get(source_id)
        if source is None:
            errors.append(f"review:{record_id}:source_frontend_missing")
            validate_record("review", record)
            continue
        source_provenance = source.get("provenance")
        if not isinstance(source_provenance, Mapping) or (
            provenance.get("seed_id") != source_provenance.get("seed_id")
        ):
            errors.append(f"review:{record_id}:source_frontend_seed_mismatch")
        source_artifact = _artifact_code(source)
        candidate = raw_input.get("candidate_code")
        if source_artifact is None or not isinstance(candidate, str):
            errors.append(f"review:{record_id}:source_or_candidate_invalid")
            validate_record("review", record)
            continue
        try:
            expected_candidate, expected_manifest = mutate_frontend_code(
                source_artifact[1],
                source_record_id=source_id,
                preferred_rule=axis_from_tags(
                    provenance.get("card_tags", ()), "review_defect"
                ),
            )
        except ValueError as error:
            errors.append(
                f"review:{record_id}:mutation_recompute_failed:{type(error).__name__}"
            )
            validate_record("review", record)
            continue
        if (
            candidate != expected_candidate
            or provenance.get("mutation") != expected_manifest.to_dict()
        ):
            errors.append(f"review:{record_id}:candidate_or_mutation_not_canonical")
        review_artifact = _artifact_code(record)
        if review_artifact is None or review_artifact[1] != source_artifact[1]:
            errors.append(f"review:{record_id}:repair_does_not_restore_frontend_source")
        validate_record("review", record)

    record_failures: dict[str, list[str]] = {}
    for error in errors:
        parts = error.split(":", 2)
        if len(parts) == 3 and parts[0] in {"frontend", "review"}:
            record_failures.setdefault(f"{parts[0]}:{parts[1]}", []).append(parts[2])
    return {
        "enabled": True,
        "passed": not errors,
        "validator": "anchor-tsx-fragment-build-test-v1",
        "checked": checked,
        "unique_artifacts_built": len(cache),
        "error_count": len(errors),
        "record_failures": record_failures,
        "errors": errors[:40],
    }


def evaluate_gate(config: AutomationConfig, seed_target: int) -> dict[str, Any]:
    total_records = 0
    successful = 0
    duplicate_count = 0
    safety_violations = 0
    schema_errors: list[str] = []
    oracle_errors: list[str] = []
    label_counts: dict[str, Counter[str]] = {
        task_type: Counter() for task_type in _LABELS_BY_TASK
    }
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for task_type in TASK_TYPES:
        path = config.output_dir / f"data_{task_type}.jsonl"
        if not path.is_file():
            schema_errors.append(f"missing {path.name}")
            continue
        records: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                records.append(value)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            schema_errors.append(f"{path.name}: {error}")
            continue
        records_by_task[task_type] = records
        total_records += len(records)
        ids = [str(record.get("id", "")) for record in records]
        users = [
            str(record.get("messages", [{}])[0].get("content", ""))
            if isinstance(record.get("messages"), list) and record.get("messages")
            else ""
            for record in records
        ]
        duplicate_count += max(len(ids) - len(set(ids)), len(users) - len(set(users)))
        seed_ids = {
            str(record.get("provenance", {}).get("seed_id", ""))
            for record in records
            if isinstance(record.get("provenance"), dict)
        }
        seed_ids.discard("")
        successful += min(seed_target, len(seed_ids))

        if task_type in ("review", "security"):
            for record in records:
                try:
                    validate_safe_payload(
                        task_type,
                        {
                            "input": record.get("input", {}),
                            "output": record.get("output", {}),
                        },
                    )
                except ValueError:
                    safety_violations += 1
        if task_type == "tool_policy":
            for record in records:
                decision = (
                    record.get("output", {}).get("decision")
                    if isinstance(record.get("output"), Mapping)
                    else None
                )
                if isinstance(decision, str):
                    label_counts[task_type][decision] += 1
                oracle_error = _tool_policy_oracle_error(record)
                if oracle_error is not None:
                    oracle_errors.append(
                        f"tool_policy:{_record_id(record)}:{oracle_error}"
                    )
        if task_type == "security":
            for record in records:
                decision = (
                    record.get("output", {}).get("decision")
                    if isinstance(record.get("output"), Mapping)
                    else None
                )
                if isinstance(decision, str):
                    label_counts[task_type][decision] += 1
                oracle_error = _security_oracle_error(record)
                if oracle_error is not None:
                    oracle_errors.append(
                        f"security:{_record_id(record)}:{oracle_error}"
                    )
        try:
            from anchor_mvp.training.schema import validate_jsonl

            expert = {
                "plan": "planner",
                "tool_policy": "tool_policy",
                "frontend": "frontend_gen",
                "review": "frontend_review",
                "security": "security_gate",
            }[task_type]
            validate_jsonl(path, allowed_experts=[expert])
        except (ImportError, ValueError) as error:
            schema_errors.append(f"{path.name}: training validation failed: {error}")

    label_quota_errors: list[str] = []
    for quota_task, minimums in config.minimum_label_counts.items():
        for label, required in minimums.items():
            observed = label_counts[quota_task][label]
            if observed < required:
                label_quota_errors.append(
                    f"{quota_task}:{label}: observed={observed}, required={required}"
                )
    artifact_gate = _evaluate_artifact_gate(config, records_by_task)

    expected = seed_target * len(TASK_TYPES)
    success_rate = successful / expected if expected else 0.0
    duplicate_rate = duplicate_count / total_records if total_records else 0.0
    passed = (
        success_rate >= config.min_success_rate
        and duplicate_rate <= config.max_duplicate_rate
        and safety_violations <= config.max_safety_violations
        and not schema_errors
        and not oracle_errors
        and not label_quota_errors
        and bool(artifact_gate["passed"])
    )
    return {
        "passed": passed,
        "records": total_records,
        "expected_records": expected,
        "success_rate": success_rate,
        "min_success_rate": config.min_success_rate,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "max_duplicate_rate": config.max_duplicate_rate,
        "safety_violations": safety_violations,
        "max_safety_violations": config.max_safety_violations,
        "training_schema_ok": not schema_errors,
        "schema_errors": schema_errors[:20],
        "deterministic_oracle_ok": not oracle_errors,
        "oracle_errors": oracle_errors[:20],
        "label_counts": {
            task_type: dict(counts) for task_type, counts in label_counts.items()
        },
        "label_quota_ok": not label_quota_errors,
        "label_quota_errors": label_quota_errors[:20],
        "artifact_validation": artifact_gate,
    }


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _load_collection_records(
    config: AutomationConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Read only complete JSON objects; malformed collection files fail closed."""

    from anchor_mvp.training.schema import DatasetValidationError, validate_record

    expected_experts = {
        "plan": "planner",
        "tool_policy": "tool_policy",
        "frontend": "frontend_gen",
        "review": "frontend_review",
        "security": "security_gate",
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for task_type in TASK_TYPES:
        path = config.output_dir / f"data_{task_type}.jsonl"
        records: list[dict[str, Any]] = []
        if path.is_file():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"malformed collection JSONL: {path.name}:{line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"collection record is not an object: {path.name}:{line_number}"
                    )
                try:
                    expert = validate_record(value, source=f"{path.name}:{line_number}")
                except DatasetValidationError as error:
                    raise ValueError(
                        f"malformed collection record: {path.name}:{line_number}: "
                        f"{str(error).split(':')[-1].strip()}"
                    ) from error
                if expert != expected_experts[task_type]:
                    raise ValueError(
                        f"wrong expert in collection: {path.name}:{line_number}"
                    )
                records.append(value)
        loaded[task_type] = records
    return loaded


def _gold_seed_id(record: Mapping[str, Any]) -> str | None:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    value = provenance.get("seed_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _canonical_lineage_edge_error(
    edge: str,
    upstream: Mapping[str, Any],
    downstream: Mapping[str, Any],
) -> str | None:
    """Return a content-free contract code when one stored DAG edge is stale.

    Record IDs and seed IDs prove identity, but not that downstream canonical
    inputs still contain the exact upstream artifact. Recompute every controlled
    transform used by the pipeline so replacing content without changing IDs is
    rejected before a row can contribute to a complete chain.
    """

    upstream_output = upstream.get("output")
    downstream_input = downstream.get("input")
    if not isinstance(upstream_output, Mapping) or not isinstance(
        downstream_input, Mapping
    ):
        return "canonical_mapping_missing"

    if edge == "plan->tool_policy":
        upstream_input = upstream.get("input")
        if not isinstance(upstream_input, Mapping):
            return "plan_input_missing"
        if set(upstream_input) != {"requirement"}:
            return "plan_input_shape"
        if set(downstream_input) != {"requirement", "plan", "tool_proposals"}:
            return "tool_policy_input_shape"
        if downstream_input.get("requirement") != upstream_input.get("requirement"):
            return "tool_policy_requirement_content"
        if downstream_input.get("plan") != upstream_output:
            return "tool_policy_plan_content"
        return None

    if edge == "plan->frontend":
        upstream_input = upstream.get("input")
        if not isinstance(upstream_input, Mapping):
            return "plan_input_missing"
        if set(upstream_input) != {"requirement"}:
            return "plan_input_shape"
        if set(downstream_input) != {"requirement", "plan", "tool_policy"}:
            return "frontend_input_shape"
        if downstream_input.get("requirement") != upstream_input.get("requirement"):
            return "frontend_requirement_content"
        if downstream_input.get("plan") != upstream_output:
            return "frontend_plan_content"
        return None

    if edge == "tool_policy->frontend":
        upstream_input = upstream.get("input")
        if not isinstance(upstream_input, Mapping):
            return "tool_policy_input_missing"
        if set(downstream_input) != {"requirement", "plan", "tool_policy"}:
            return "frontend_input_shape"
        if downstream_input.get("requirement") != upstream_input.get("requirement"):
            return "frontend_requirement_content"
        if downstream_input.get("tool_policy") != upstream_output:
            return "frontend_tool_policy_content"
        return None

    if edge == "frontend->review":
        source_code = upstream_output.get("code")
        upstream_input = upstream.get("input")
        provenance = downstream.get("provenance")
        if (
            not isinstance(source_code, str)
            or not isinstance(upstream_input, Mapping)
            or not isinstance(provenance, Mapping)
        ):
            return "review_source_or_provenance"
        try:
            candidate, mutation = mutate_frontend_code(
                source_code,
                source_record_id=_record_id(upstream),
                preferred_rule=axis_from_tags(
                    provenance.get("card_tags", ()), "review_defect"
                ),
            )
        except ValueError:
            return "review_mutation_unavailable"
        expected_input = {
            "requirement": upstream_input.get("requirement"),
            "candidate_code": candidate.strip(),
            "known_benign_defect": mutation.known_benign_defect.strip(),
        }
        if dict(downstream_input) != expected_input:
            return "review_candidate_content"
        if provenance.get("mutation") != mutation.to_dict():
            return "review_mutation_manifest"
        downstream_output = downstream.get("output")
        if (
            not isinstance(downstream_output, Mapping)
            or downstream_output.get("code") != source_code
        ):
            return "review_repair_content"
        return None

    if edge == "review->security":
        source_code = upstream_output.get("code")
        upstream_input = upstream.get("input")
        provenance = downstream.get("provenance")
        downstream_output = downstream.get("output")
        if (
            not isinstance(source_code, str)
            or not isinstance(upstream_input, Mapping)
            or not isinstance(provenance, Mapping)
            or not isinstance(downstream_output, Mapping)
        ):
            return "security_source_or_provenance"
        upstream_requirement = upstream_input.get("requirement")
        if not isinstance(upstream_requirement, str) or not upstream_requirement:
            return "security_requirement_content"
        expected_requirement = sanitize_security_seed(
            SeedDemand(
                seed_id=_gold_seed_id(downstream) or "lineage-verifier",
                title="lineage verifier",
                request=upstream_requirement,
            )
        ).request
        observed_fixture = provenance.get("security_fixture")
        if not isinstance(observed_fixture, Mapping):
            return "security_fixture_manifest"
        seen_fixture_ids: set[str] = set()
        for index in range(64):
            candidate, expected_output, fixture = build_inert_security_fixture(
                source_code, index
            )
            fixture_id = str(fixture.get("fixture_id", ""))
            if fixture_id in seen_fixture_ids:
                break
            seen_fixture_ids.add(fixture_id)
            if dict(observed_fixture) != fixture:
                continue
            if dict(downstream_input) != {
                "requirement": expected_requirement,
                "reviewed_code": candidate.strip(),
            }:
                return "security_reviewed_code_content"
            if dict(downstream_output) != expected_output:
                return "security_fixture_output"
            return None
        return "security_fixture_manifest"

    return "unsupported_lineage_edge"


def _evaluate_gold_lineage(
    gold_by_task: Mapping[str, list[dict[str, Any]]],
    minimum_gold: Mapping[str, int],
) -> dict[str, Any]:
    """Validate the strict-gold DAG and count unique complete five-stage chains.

    Every downstream reference must resolve inside the corresponding strict-gold
    partition, not merely in the append-only raw collection. A valid edge also
    requires identical non-empty seed IDs. Complete chains are counted once per
    seed and must converge on the same planner record through both frontend
    inputs. Only content-free IDs and error codes are emitted in the manifest.
    """

    indexes: dict[str, dict[str, dict[str, Any]]] = {
        task: {_record_id(record): record for record in gold_by_task.get(task, [])}
        for task in TASK_TYPES
    }
    valid_sources: dict[tuple[str, str], dict[str, str]] = {}
    edge_errors: list[dict[str, Any]] = []

    for edge, upstream_task, downstream_task, source_field in _LINEAGE_EDGES:
        resolved: dict[str, str] = {}
        for downstream in gold_by_task.get(downstream_task, []):
            downstream_id = _record_id(downstream)
            provenance = downstream.get("provenance")
            source_value = (
                provenance.get(source_field)
                if isinstance(provenance, Mapping)
                else None
            )
            source_id = (
                source_value.strip()
                if isinstance(source_value, str) and source_value.strip()
                else None
            )
            error: dict[str, Any] = {
                "edge": edge,
                "upstream_task": upstream_task,
                "downstream_task": downstream_task,
                "source_field": source_field,
                "downstream_record_id": downstream_id,
            }
            if source_id is None:
                error["code"] = "missing_source_record_id"
                edge_errors.append(error)
                continue
            error["source_record_id"] = source_id
            upstream = indexes[upstream_task].get(source_id)
            if upstream is None:
                error["code"] = "source_not_strict_gold"
                edge_errors.append(error)
                continue
            downstream_seed = _gold_seed_id(downstream)
            upstream_seed = _gold_seed_id(upstream)
            if downstream_seed is None or upstream_seed is None:
                error["code"] = "missing_seed_id"
                edge_errors.append(error)
                continue
            if downstream_seed != upstream_seed:
                error["code"] = "seed_id_mismatch"
                edge_errors.append(error)
                continue
            downstream_provenance = downstream.get("provenance")
            upstream_provenance = upstream.get("provenance")
            assert isinstance(downstream_provenance, Mapping)
            assert isinstance(upstream_provenance, Mapping)
            downstream_alignment = downstream_provenance.get("alignment_id")
            upstream_alignment = upstream_provenance.get("alignment_id")
            if downstream_alignment is not None or upstream_alignment is not None:
                downstream_card = downstream_provenance.get("card_id")
                upstream_card = upstream_provenance.get("card_id")
                downstream_tags = downstream_provenance.get("card_tags")
                upstream_tags = upstream_provenance.get("card_tags")
                expected_alignment = (
                    stable_id("alignment", f"{downstream_seed}\n{downstream_card}")
                    if isinstance(downstream_card, str) and downstream_card
                    else None
                )
                if not (
                    isinstance(downstream_alignment, str)
                    and downstream_alignment == upstream_alignment
                    and downstream_alignment == expected_alignment
                    and downstream_card == upstream_card
                    and isinstance(downstream_tags, list)
                    and downstream_tags == upstream_tags
                ):
                    error["code"] = "alignment_id_mismatch"
                    edge_errors.append(error)
                    continue
            canonical_error = _canonical_lineage_edge_error(edge, upstream, downstream)
            if canonical_error is not None:
                error["code"] = "canonical_input_mismatch"
                error["contract"] = canonical_error
                edge_errors.append(error)
                continue
            resolved[downstream_id] = source_id
        valid_sources[(downstream_task, source_field)] = resolved

    chain_errors: list[dict[str, Any]] = []
    complete_seed_ids: set[str] = set()
    security_to_review = valid_sources[("security", "source_review_record_id")]
    review_to_frontend = valid_sources[("review", "source_frontend_record_id")]
    frontend_to_tool = valid_sources[("frontend", "source_tool_policy_record_id")]
    frontend_to_plan = valid_sources[("frontend", "source_plan_record_id")]
    tool_to_plan = valid_sources[("tool_policy", "source_plan_record_id")]

    for security in gold_by_task.get("security", []):
        security_id = _record_id(security)
        review_id = security_to_review.get(security_id)
        if review_id is None:
            continue
        frontend_id = review_to_frontend.get(review_id)
        if frontend_id is None:
            continue
        tool_id = frontend_to_tool.get(frontend_id)
        direct_plan_id = frontend_to_plan.get(frontend_id)
        if tool_id is None or direct_plan_id is None:
            continue
        tool_plan_id = tool_to_plan.get(tool_id)
        if tool_plan_id is None:
            continue
        if direct_plan_id != tool_plan_id:
            chain_errors.append(
                {
                    "code": "planner_reference_fork",
                    "security_record_id": security_id,
                    "review_record_id": review_id,
                    "frontend_record_id": frontend_id,
                    "tool_policy_record_id": tool_id,
                    "frontend_plan_record_id": direct_plan_id,
                    "tool_policy_plan_record_id": tool_plan_id,
                }
            )
            continue
        seed_id = _gold_seed_id(security)
        if seed_id is None:
            # A valid security edge already requires a seed. Keep this branch
            # fail-closed if that invariant changes in a future schema.
            chain_errors.append(
                {
                    "code": "missing_complete_chain_seed_id",
                    "security_record_id": security_id,
                }
            )
            continue
        if seed_id in complete_seed_ids:
            chain_errors.append(
                {
                    "code": "duplicate_complete_chain_seed",
                    "security_record_id": security_id,
                }
            )
            continue
        complete_seed_ids.add(seed_id)

    edge_errors_by_edge = Counter(str(error["edge"]) for error in edge_errors)
    chain_errors_by_code = Counter(str(error["code"]) for error in chain_errors)
    minimum_complete_chain_count = max(minimum_gold.values(), default=0)
    complete_chain_count = len(complete_seed_ids)
    lineage_complete = not edge_errors and not chain_errors
    return {
        "lineage_complete": lineage_complete,
        "complete_chain_count": complete_chain_count,
        "minimum_complete_chain_count": minimum_complete_chain_count,
        "complete_chain_count_sufficient": (
            complete_chain_count >= minimum_complete_chain_count
        ),
        "lineage_edge_error_count": len(edge_errors),
        "lineage_edge_errors_by_edge": dict(sorted(edge_errors_by_edge.items())),
        "lineage_edge_errors": edge_errors,
        "lineage_chain_error_count": len(chain_errors),
        "lineage_chain_errors_by_code": dict(sorted(chain_errors_by_code.items())),
        "lineage_chain_errors": chain_errors,
        "_complete_chain_seed_ids": sorted(complete_seed_ids),
    }


def partition_collected_records(
    config: AutomationConfig, seed_target: int | None = None
) -> dict[str, Any]:
    """Partition a completed collection without deleting ordinary model failures.

    Structurally accepted records first enter quality staging. Deterministic label,
    duplicate, and executable-artifact failures are retained as negatives. Unsafe
    or secret-bearing records become content-free rejects. The raw per-task files
    remain the append-only collection source; only ``partitions/gold`` is eligible
    to become a training snapshot.
    """

    target = seed_target or config.raw_records_per_task
    minimum_gold = config.minimum_gold_records_by_task
    minimum_complete_chain_count = max(minimum_gold.values(), default=0)
    try:
        records_by_task = _load_collection_records(config)
    except (OSError, ValueError) as error:
        # Replace any stale ready manifest with a content-free corpus blocker.
        # The malformed row itself is never copied into a partition artifact.
        _atomic_write_json(
            config.partition_dir / "manifest.json",
            {
                "schema_version": "anchor.automation-partition-manifest.v2",
                "collection_policy": config.collection_policy,
                "seed_target": target,
                "raw_collection_target": target,
                "minimum_gold_records_per_task": minimum_gold,
                "partition_complete": False,
                "rejects_quarantined": False,
                "coverage_complete": False,
                "lineage_complete": False,
                "complete_chain_count": 0,
                "minimum_complete_chain_count": minimum_complete_chain_count,
                "complete_chain_count_sufficient": False,
                "lineage_edge_error_count": 0,
                "lineage_edge_errors_by_edge": {},
                "lineage_edge_errors": [],
                "lineage_chain_error_count": 0,
                "lineage_chain_errors_by_code": {},
                "lineage_chain_errors": [],
                "gold_files": {},
                "training_ready": False,
                "corpus_blocker": "malformed_collection",
                "error_type": type(error).__name__,
                "content_emitted": False,
            },
        )
        raise
    empty_digest = hashlib.sha256(b"").hexdigest()
    source_digests = {
        task_type: (
            file_sha256(config.output_dir / f"data_{task_type}.jsonl")
            if (config.output_dir / f"data_{task_type}.jsonl").is_file()
            else empty_digest
        )
        for task_type in TASK_TYPES
    }
    artifact_gate = _evaluate_artifact_gate(config, records_by_task)
    artifact_failures = artifact_gate.get("record_failures", {})
    if not isinstance(artifact_failures, Mapping):
        artifact_failures = {}
    task_card_catalog = load_task_card_catalog(config.task_card_config)
    seed_assignments: dict[str, CardAssignment] = {}
    seed_assignment_errors: set[str] = set()
    for raw_seed in SeedStore(config.output_dir / "seeds.jsonl").records:
        try:
            seed = SeedDemand.from_mapping(raw_seed)
            seed_assignments[seed.seed_id] = assignment_for_seed(
                seed, task_card_catalog
            )
        except ValueError:
            raw_seed_id = raw_seed.get("seed_id")
            if isinstance(raw_seed_id, str) and raw_seed_id:
                seed_assignment_errors.add(raw_seed_id)

    id_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    for records in records_by_task.values():
        for record in records:
            id_counts[_record_id(record)] += 1
            messages = record.get("messages")
            prompt = ""
            if (
                isinstance(messages, list)
                and messages
                and isinstance(messages[0], Mapping)
            ):
                prompt = " ".join(
                    str(messages[0].get("content", "")).split()
                ).casefold()
            if prompt:
                prompt_counts[prompt] += 1

    staged: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    oracle_label_only: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    gold_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASK_TYPES}
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    gold_label_counts: dict[str, Counter[str]] = {
        task: Counter() for task in _LABELS_BY_TASK
    }
    audit_label_counts: dict[str, Counter[str]] = {
        task: Counter() for task in TASK_TYPES
    }

    for task_type in TASK_TYPES:
        for index, record in enumerate(records_by_task[task_type]):
            record_id = _record_id(record)
            quality_labels: set[str] = set()
            hard_labels: set[str] = set()
            audit_labels: set[str] = set()
            partition_record = record
            oracle_label_only_eligible = False
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if contains_secret_material(record):
                hard_labels.add("secret_detected")
            try:
                validate_safe_payload(
                    cast(Any, task_type),
                    {
                        "input": record.get("input", {}),
                        "output": record.get("output", {}),
                    },
                )
            except ValueError:
                hard_labels.add("unsafe_payload")

            messages = record.get("messages")
            prompt = ""
            if (
                isinstance(messages, list)
                and messages
                and isinstance(messages[0], Mapping)
            ):
                prompt = " ".join(
                    str(messages[0].get("content", "")).split()
                ).casefold()
            if record_id in seen_ids or id_counts[record_id] > 1:
                quality_labels.add("duplicate_record_id")
            if prompt and (prompt in seen_prompts or prompt_counts[prompt] > 1):
                quality_labels.add("duplicate_prompt")
            seen_ids.add(record_id)
            if prompt:
                seen_prompts.add(prompt)

            provenance = record.get("provenance")
            record_seed_id = (
                provenance.get("seed_id") if isinstance(provenance, Mapping) else None
            )
            assignment = (
                seed_assignments.get(record_seed_id)
                if isinstance(record_seed_id, str)
                else None
            )
            if (
                isinstance(record_seed_id, str)
                and record_seed_id in seed_assignment_errors
            ):
                quality_labels.add("task_card_assignment_invalid")
            elif assignment is not None and isinstance(provenance, Mapping):
                expected_card = assignment.provenance(record_seed_id)
                card_fields = {
                    "card_id",
                    "card_tags",
                    "alignment_id",
                    "task_card_legacy",
                    "task_card_catalog_sha256",
                    "seed_index",
                    "template_id",
                    "source_kind",
                    "source_digest",
                }
                observed_card = {
                    key: provenance[key] for key in card_fields if key in provenance
                }
                if assignment.legacy:
                    # Preserve accepted pre-card and short-lived slot-card
                    # samples without inventing nine-axis labels.  Raw files
                    # remain append-only; the partition view receives the
                    # unique requirement-bound legacy card identity.
                    partition_record = deepcopy(partition_record)
                    normalized_provenance = dict(partition_record["provenance"])
                    for key in card_fields:
                        normalized_provenance.pop(key, None)
                    normalized_provenance.update(expected_card)
                    partition_record["provenance"] = normalized_provenance
                elif observed_card != expected_card:
                    quality_labels.add("task_card_alignment_invalid")
                if assignment.source_kind == "swebench_heldout":
                    quality_labels.add("heldout_source_excluded")
                if assignment.axes is not None:
                    if task_type == "tool_policy":
                        proposal_manifest = provenance.get("tool_proposals")
                        if (
                            not isinstance(proposal_manifest, Mapping)
                            or proposal_manifest.get("variant")
                            != assignment.axes["tool_posture"]
                        ):
                            quality_labels.add("task_card_axis_mismatch")
                    elif task_type == "review":
                        mutation = provenance.get("mutation")
                        if (
                            not isinstance(mutation, Mapping)
                            or mutation.get("rule") != assignment.axes["review_defect"]
                        ):
                            quality_labels.add("task_card_axis_mismatch")
                    elif task_type == "security":
                        fixture = provenance.get("security_fixture")
                        if (
                            not isinstance(fixture, Mapping)
                            or fixture.get("kind") != assignment.axes["security_class"]
                        ):
                            quality_labels.add("task_card_axis_mismatch")

            oracle_error: str | None = None
            if task_type == "tool_policy":
                oracle_error = _tool_policy_oracle_error(record)
            elif task_type == "security":
                oracle_error = _security_oracle_error(record)
            if oracle_error is not None:
                quality_labels.add("deterministic_oracle_mismatch")
            if task_type in _LABELS_BY_TASK:
                provenance = record.get("provenance")
                output = record.get("output")
                observed = (
                    provenance.get("teacher_observed_decision")
                    if isinstance(provenance, Mapping)
                    else None
                )
                authoritative = (
                    output.get("decision") if isinstance(output, Mapping) else None
                )
                if isinstance(observed, str) and observed != authoritative:
                    audit_labels.add("teacher_label_disagreement")
                    if (
                        oracle_error is None
                        and _oracle_normalized_disagreement_is_training_safe(
                            task_type, record
                        )
                    ):
                        audit_labels.add("teacher_label_disagreement_oracle_normalized")
                        quality_labels.add(
                            "teacher_label_disagreement_oracle_label_only"
                        )
                        oracle_label_only_eligible = True
                        # Raw collection remains append-only. The partitioned
                        # training view makes the legacy normalization proof
                        # explicit so downstream consumers never mistake this
                        # row for teacher agreement.
                        partition_record = deepcopy(partition_record)
                        normalized_provenance = dict(partition_record["provenance"])
                        normalized_provenance.update(
                            {
                                "teacher_decision_agrees_with_oracle": False,
                                "supervision_source": "deterministic_oracle",
                                "oracle_normalized": True,
                                "decision_trace_source": "deterministic_oracle",
                            }
                        )
                        partition_record["provenance"] = normalized_provenance
                    else:
                        quality_labels.add("teacher_label_disagreement")
                        audit_labels.add("teacher_label_disagreement_unresolved")
            record_artifact_failures = artifact_failures.get(
                f"{task_type}:{record_id}", []
            )
            if isinstance(record_artifact_failures, list) and record_artifact_failures:
                quality_labels.add("artifact_validation_failed")

            disposition = (
                "reject" if hard_labels else "negative" if quality_labels else "gold"
            )
            labels = sorted(hard_labels | quality_labels)
            for audit_label in audit_labels:
                audit_label_counts[task_type][audit_label] += 1
            staging_id = stable_id(
                "quality",
                f"{task_type}:{index}:{record_id}:{source_digests[task_type]}",
            )
            staged_record: dict[str, Any] = {
                "id": staging_id,
                "schema_version": QUALITY_STAGING_SCHEMA_VERSION,
                "task_type": task_type,
                "source_record_id": record_id,
                "disposition": disposition,
                "quality": {
                    "labels": labels,
                    "audit_labels": sorted(audit_labels),
                    "strict_gold_eligible": disposition == "gold",
                },
            }
            if disposition == "reject":
                staged_record["content_retained"] = False
                rejects.append(
                    {
                        "id": staging_id,
                        "schema_version": PARTITION_REJECT_SCHEMA_VERSION,
                        "task_type": task_type,
                        "source_record_sha256": hashlib.sha256(
                            encoded.encode("utf-8")
                        ).hexdigest(),
                        "reason_codes": labels,
                        "content_retained": False,
                    }
                )
            else:
                staged_record["content_retained"] = True
                staged_record["record"] = partition_record
                if disposition == "negative":
                    negatives.append(staged_record)
                    if oracle_label_only_eligible:
                        weak_input = partition_record.get("input")
                        weak_output = partition_record.get("output")
                        weak_provenance = partition_record.get("provenance")
                        authoritative_decision = (
                            weak_output.get("decision")
                            if isinstance(weak_output, Mapping)
                            else None
                        )
                        label_oracle = (
                            weak_provenance.get("label_oracle")
                            if isinstance(weak_provenance, Mapping)
                            else None
                        )
                        if (
                            isinstance(weak_input, Mapping)
                            and isinstance(authoritative_decision, str)
                            and isinstance(label_oracle, Mapping)
                        ):
                            oracle_label_only.append(
                                {
                                    "id": stable_id(
                                        "oracle-label",
                                        f"{task_type}:{record_id}:{authoritative_decision}",
                                    ),
                                    "schema_version": "anchor.oracle-label-only.v1",
                                    "task_type": task_type,
                                    "source_record_id": record_id,
                                    "input": dict(weak_input),
                                    "output": {"decision": authoritative_decision},
                                    "provenance": {
                                        "supervision_source": "deterministic_oracle",
                                        "teacher_trace_included": False,
                                        "label_oracle": dict(label_oracle),
                                    },
                                }
                            )
                else:
                    gold_by_task[task_type].append(partition_record)
                    if task_type in _LABELS_BY_TASK:
                        output = partition_record.get("output")
                        decision = (
                            output.get("decision")
                            if isinstance(output, Mapping)
                            else None
                        )
                        if isinstance(decision, str):
                            gold_label_counts[task_type][decision] += 1
            staged.append(staged_record)

    # Near-duplicate comparison is seed-level and only considers complete
    # base-gold chains. This prevents the same seed's five expert prompts from
    # colliding and prevents an earlier invalid chain from occupying the stable
    # representative slot. A loser moves as one whole five-stage chain.
    base_lineage = _evaluate_gold_lineage(gold_by_task, minimum_gold)
    base_complete_seed_ids = set(
        cast(list[str], base_lineage.pop("_complete_chain_seed_ids"))
    )
    plan_by_seed = {_gold_seed_id(record): record for record in gold_by_task["plan"]}
    near_duplicate_candidates: list[dict[str, Any]] = []
    for seed_id in sorted(base_complete_seed_ids):
        plan = plan_by_seed.get(seed_id)
        raw_input = plan.get("input") if isinstance(plan, Mapping) else None
        requirement = (
            raw_input.get("requirement") if isinstance(raw_input, Mapping) else None
        )
        assignment = seed_assignments.get(seed_id)
        near_duplicate_candidates.append(
            {
                "seed_id": seed_id,
                "seed_index": assignment.seed_index if assignment is not None else None,
                "requirement": requirement,
            }
        )
    near_duplicate_seeds, near_duplicate_gate = detect_near_duplicate_seeds(
        near_duplicate_candidates, task_card_catalog
    )
    staged_by_record = {
        (str(item.get("task_type")), str(item.get("source_record_id"))): item
        for item in staged
    }
    if near_duplicate_seeds:
        for task_type in TASK_TYPES:
            retained: list[dict[str, Any]] = []
            for record in gold_by_task[task_type]:
                seed_id = _gold_seed_id(record)
                evidence = near_duplicate_seeds.get(seed_id or "")
                if evidence is None:
                    retained.append(record)
                    continue
                staged_record = staged_by_record[(task_type, _record_id(record))]
                staged_record["disposition"] = "negative"
                quality = cast(dict[str, Any], staged_record["quality"])
                quality["labels"] = sorted(
                    {*cast(list[str], quality["labels"]), "near_duplicate_requirement"}
                )
                quality["strict_gold_eligible"] = False
                quality["near_duplicate"] = {
                    "policy_id": near_duplicate_gate["policy_id"],
                    **evidence,
                }
                negatives.append(staged_record)
                if task_type in _LABELS_BY_TASK:
                    output = record.get("output")
                    decision = (
                        output.get("decision") if isinstance(output, Mapping) else None
                    )
                    if isinstance(decision, str):
                        gold_label_counts[task_type][decision] -= 1
            gold_by_task[task_type] = retained

    lineage = _evaluate_gold_lineage(gold_by_task, minimum_gold)
    complete_seed_ids = cast(list[str], lineage.pop("_complete_chain_seed_ids"))
    final_plan_by_seed = {
        _gold_seed_id(record): record for record in gold_by_task["plan"]
    }
    complete_assignments: list[tuple[str, CardAssignment]] = []
    task_bank: list[dict[str, Any]] = []
    for seed_id in complete_seed_ids:
        plan = final_plan_by_seed.get(seed_id)
        plan_input = plan.get("input") if isinstance(plan, Mapping) else None
        requirement = (
            plan_input.get("requirement") if isinstance(plan_input, Mapping) else None
        )
        if not isinstance(requirement, str) or not requirement.strip():
            # Lineage evaluation will already block this chain.  Keep a
            # deterministic, content-free fallback so cardinality also fails
            # closed instead of crashing partition audit.
            requirement = f"missing requirement for {seed_id}"
        assignment = seed_assignments.get(seed_id)
        if assignment is None:
            assignment = assignment_for_seed(
                SeedDemand(
                    seed_id=seed_id,
                    title="legacy collected task",
                    request=requirement,
                ),
                task_card_catalog,
            )
        complete_assignments.append((seed_id, assignment))
        alignment_id = assignment.alignment_for_seed(seed_id)
        task_bank_row: dict[str, Any] = {
            "schema_version": "anchor.task-bank-card.v1",
            "card_id": assignment.card_id,
            "alignment_id": alignment_id,
            "seed_id": seed_id,
            "requirement": requirement,
            "card_tags": list(assignment.tags),
            "source_kind": assignment.source_kind,
            "task_card_legacy": assignment.legacy,
        }
        if assignment.template_id is not None:
            task_bank_row["template_id"] = assignment.template_id
        if assignment.seed_index is not None:
            task_bank_row["seed_index"] = assignment.seed_index
        if assignment.axes is not None:
            task_bank_row["axes"] = dict(assignment.axes)
        if assignment.catalog_sha256 is not None:
            task_bank_row["task_card_catalog_sha256"] = assignment.catalog_sha256
        if assignment.source_digest is not None:
            task_bank_row["source_digest"] = assignment.source_digest
        task_bank.append(task_bank_row)

    task_bank.sort(
        key=lambda item: (
            item.get("seed_index")
            if isinstance(item.get("seed_index"), int)
            else 2**63 - 1,
            str(item["seed_id"]),
        )
    )

    _atomic_write_jsonl(config.quality_staging_path, staged)
    _atomic_write_jsonl(config.partition_dir / "negative.jsonl", negatives)
    _atomic_write_jsonl(
        config.partition_dir / "oracle_label_only.jsonl", oracle_label_only
    )
    _atomic_write_jsonl(config.partition_dir / "reject.jsonl", rejects)
    for task_type, records in gold_by_task.items():
        _atomic_write_jsonl(
            config.partition_dir / "gold" / f"data_{task_type}.jsonl", records
        )
    task_bank_path = config.partition_dir / "task_bank.jsonl"
    _atomic_write_jsonl(task_bank_path, task_bank)

    gold_files: dict[str, dict[str, Any]] = {}
    for task_type in TASK_TYPES:
        filename = f"data_{task_type}.jsonl"
        path = config.partition_dir / "gold" / filename
        gold_files[task_type] = {
            "path": filename,
            "records": len(gold_by_task[task_type]),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    coverage = {task: len(records) for task, records in gold_by_task.items()}
    task_card_coverage = evaluate_task_card_coverage(
        complete_assignments,
        task_card_catalog,
        minimum_complete_chain_count=minimum_complete_chain_count,
        task_bank_count=len(task_bank),
        stage_counts=coverage,
    )
    task_bank_file = {
        "path": "task_bank.jsonl",
        "records": len(task_bank),
        "bytes": task_bank_path.stat().st_size,
        "sha256": file_sha256(task_bank_path),
    }
    coverage_shortfalls = {
        task: minimum_gold[task] - count
        for task, count in coverage.items()
        if count < minimum_gold[task]
    }
    coverage_complete = not coverage_shortfalls
    raw_by_task = {task: len(records) for task, records in records_by_task.items()}
    raw_collection_shortfalls = {
        task: target - count for task, count in raw_by_task.items() if count < target
    }
    quota_errors: list[str] = []
    for task_type, minimums in config.minimum_label_counts.items():
        for label, required in minimums.items():
            observed = gold_label_counts[task_type][label]
            if observed < required:
                quota_errors.append(
                    f"{task_type}:{label}: observed={observed}, required={required}"
                )
    gold_count = sum(coverage.values())
    raw_count = sum(raw_by_task.values())
    partition_complete = raw_count == gold_count + len(negatives) + len(rejects)
    rejects_quarantined = all(
        reject.get("content_retained") is False
        and "record" not in reject
        and isinstance(reject.get("source_record_sha256"), str)
        for reject in rejects
    )
    reject_reason_counts: Counter[str] = Counter(
        reason
        for reject in rejects
        for reason in cast(list[str], reject.get("reason_codes", []))
    )
    gold_integrity_ok = all(
        not contains_secret_material(record)
        for records in gold_by_task.values()
        for record in records
    )
    heldout_gate = evaluate_heldout_scale_gate(config)
    manifest: dict[str, Any] = {
        "schema_version": "anchor.automation-partition-manifest.v2",
        "collection_policy": config.collection_policy,
        # seed_target remains as a compatibility alias. It is a raw capacity
        # target and must never be interpreted as the gold floor.
        "seed_target": target,
        "raw_collection_target": target,
        "minimum_gold_records_per_task": minimum_gold,
        "raw_by_task": raw_by_task,
        "raw_collection_complete": not raw_collection_shortfalls,
        "raw_collection_shortfalls": raw_collection_shortfalls,
        "staged_count": len(staged),
        "gold_count": gold_count,
        "negative_count": len(negatives),
        "oracle_label_only_count": len(oracle_label_only),
        "reject_count": len(rejects),
        "partition_complete": partition_complete,
        "rejects_quarantined": rejects_quarantined,
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "reject_rate": len(rejects) / raw_count if raw_count else 0.0,
        "gold_by_task": coverage,
        "gold_files": gold_files,
        "gold_label_counts": {
            task: dict(counts) for task, counts in gold_label_counts.items()
        },
        "label_quota_errors": quota_errors,
        "coverage_complete": coverage_complete,
        "coverage_shortfalls": coverage_shortfalls,
        "audit_label_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in audit_label_counts.items()
            if counts
        },
        "teacher_label_disagreements_by_task": {
            task: counts["teacher_label_disagreement"]
            for task, counts in audit_label_counts.items()
            if counts["teacher_label_disagreement"]
        },
        "oracle_normalized_disagreements_by_task": {
            task: counts["teacher_label_disagreement_oracle_normalized"]
            for task, counts in audit_label_counts.items()
            if counts["teacher_label_disagreement_oracle_normalized"]
        },
        "unresolved_disagreements_by_task": {
            task: counts["teacher_label_disagreement_unresolved"]
            for task, counts in audit_label_counts.items()
            if counts["teacher_label_disagreement_unresolved"]
        },
        "gold_integrity_ok": gold_integrity_ok,
        "near_duplicate_gate": near_duplicate_gate,
        "task_card_coverage": task_card_coverage,
        "task_bank_file": task_bank_file,
        **lineage,
        "heldout_gate": heldout_gate,
        "training_ready": (
            partition_complete
            and rejects_quarantined
            and gold_integrity_ok
            and coverage_complete
            and not quota_errors
            and bool(lineage["lineage_complete"])
            and bool(lineage["complete_chain_count_sufficient"])
            and bool(near_duplicate_gate["passed"])
            and bool(task_card_coverage["passed"])
            and bool(heldout_gate.get("passed"))
        ),
        "quality_staging_sha256": file_sha256(config.quality_staging_path),
        "negative_sha256": file_sha256(config.partition_dir / "negative.jsonl"),
        "oracle_label_only_sha256": file_sha256(
            config.partition_dir / "oracle_label_only.jsonl"
        ),
        "reject_sha256": file_sha256(config.partition_dir / "reject.jsonl"),
    }
    _atomic_write_json(config.partition_dir / "manifest.json", manifest)
    return manifest


def evaluate_heldout_scale_gate(config: AutomationConfig) -> dict[str, Any]:
    """Re-scan the current five-task corpus before increasing concurrency."""

    if config.heldout_cases is None:
        return {"enabled": False, "passed": True, "status": "DISABLED"}
    assert config.heldout_fixtures_root is not None
    assert config.heldout_manifest is not None
    assert config.heldout_leak_audit is not None
    try:
        manifest_digest = verify_heldout_manifest(
            config.heldout_cases,
            config.heldout_fixtures_root,
            config.heldout_manifest,
        )
        verify_leak_audit(config.heldout_leak_audit, manifest_digest)
        training_sources = [
            config.output_dir / f"data_{task_type}.jsonl" for task_type in TASK_TYPES
        ]
        sop_sources = sorted(
            path
            for path in config.sop_dir.iterdir()
            if path.suffix.casefold() in {".md", ".yaml", ".yml"}
        )
        report = check_training_leakage(
            config.heldout_cases,
            config.heldout_fixtures_root,
            config.heldout_manifest,
            training_sources,
            sop_sources,
        )
        return {
            "enabled": True,
            "passed": report["status"] == "PASS",
            "status": report["status"],
            "manifest_sha256": manifest_digest,
            "prebulk_audit_sha256": file_sha256(config.heldout_leak_audit),
            "collision_count": report["collision_count"],
            "case_count": report["case_count"],
            "training_source_count": report["training_source_count"],
            "sop_source_count": report["sop_source_count"],
            "similarity_threshold": report["similarity_threshold"],
            "content_emitted": report["content_emitted"],
            "collisions": report["collisions"][:20],
        }
    except (OSError, ValueError) as error:
        return {
            "enabled": True,
            "passed": False,
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error)[:240],
            "content_emitted": False,
        }


def _build_teacher(
    value: Mapping[str, Any],
    *,
    thinking_effort: str,
    selection: ProviderSelection | None = None,
) -> CompatibleTeacher:
    if selection is None:
        spec = provider_spec(value)
        legacy_model = (
            os.environ.get("KIMI_MODEL_ID") if "provider" not in value else None
        )
        selection = select_provider_model(
            spec,
            requested_model=str(value.get("model") or legacy_model or "") or None,
            discover=_as_bool(value.get("discover_models", False)),
            force_model=_as_bool(value.get("force_model", False)),
            model_index=int(value["model_index"])
            if value.get("model_index") is not None
            else None,
            timeout_seconds=float(value.get("discovery_timeout_seconds", 20)),
        )
    spec = selection.spec
    configured_fallback = value.get("fallback_protocol")
    fallback_protocol = (
        str(configured_fallback)
        if configured_fallback is not None
        else "openai"
        if spec.preset == "kimi-code-anthropic"
        else None
    )
    fallback_base = str(
        value.get(
            "fallback_base_url",
            PRESETS["kimi-code-openai"].base_url
            if spec.preset == "kimi-code-anthropic"
            else spec.base_url,
        )
    )
    return CompatibleTeacher(
        base_url=spec.base_url,
        fallback_base_url=fallback_base,
        model=selection.model,
        protocol=spec.protocol,
        fallback_protocol=fallback_protocol,  # type: ignore[arg-type]
        api_key_env=spec.api_key_env,
        anthropic_version=str(value.get("anthropic_version", "2023-06-01")),
        user_agent=str(value.get("user_agent", "anchor-moe-lora/0.1")),
        timeout_seconds=float(value.get("timeout_seconds", 600)),
        max_retries=int(value.get("max_retries", 1)),
        wall_clock_deadline_seconds=float(
            value.get("wall_clock_deadline_seconds", 900)
        ),
        temperature=float(value.get("temperature", 0.2)),
        max_tokens=int(value.get("max_tokens", 16384)),
        max_requests=int(value.get("max_requests", 200)),
        max_output_tokens_total=int(value.get("max_output_tokens_total", 1_000_000)),
        thinking_enabled=_as_bool(value.get("thinking_enabled", True)),
        thinking_effort=thinking_effort,
        thinking_budget_tokens=int(value.get("thinking_budget_tokens", 4096)),
        stream_openai=_as_bool(value.get("stream_openai", True)),
        stream_options_include_usage=_as_bool(
            value.get("stream_options_include_usage", False)
        ),
        provider_preset=spec.preset,
        model_source=selection.model_source,
        discovery_status=selection.discovery.status,
        discovery_model_count=len(selection.discovery.models),
    )


def _build_teachers(value: Mapping[str, Any], *, dry_run: bool) -> dict[str, Teacher]:
    if dry_run:
        mock = MockTeacher()
        return {name: cast(Teacher, mock) for name in ("seed", *TASK_TYPES)}
    spec = provider_spec(value)
    legacy_model = os.environ.get("KIMI_MODEL_ID") if "provider" not in value else None
    selection = select_provider_model(
        spec,
        requested_model=str(value.get("model") or legacy_model or "") or None,
        discover=_as_bool(value.get("discover_models", False)),
        force_model=_as_bool(value.get("force_model", False)),
        model_index=int(value["model_index"])
        if value.get("model_index") is not None
        else None,
        timeout_seconds=float(value.get("discovery_timeout_seconds", 20)),
    )
    default_effort = str(value.get("thinking_effort", "medium"))
    efforts = {
        "seed": str(value.get("thinking_effort_seed", default_effort)),
        "plan": str(value.get("thinking_effort_plan", default_effort)),
        "tool_policy": str(value.get("thinking_effort_tool_policy", "low")),
        "frontend": str(value.get("thinking_effort_frontend", default_effort)),
        "review": str(value.get("thinking_effort_review", default_effort)),
        "security": str(value.get("thinking_effort_security", "low")),
    }
    workers = {
        name: _build_teacher(value, thinking_effort=effort, selection=selection)
        for name, effort in efforts.items()
    }
    owner = workers["seed"]
    for name, worker in workers.items():
        if name != "seed":
            worker.share_usage_budget(owner)
    return cast(dict[str, Teacher], workers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gated unattended Anchor-MoE-LoRA distillation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="run the deterministic mock E2E"
    )
    parser.add_argument(
        "--wait-cooldown",
        action="store_true",
        help="remain visible and resume after persisted 429 cooldowns",
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--status-only", action="store_true")
    operation.add_argument(
        "--partition-only",
        action="store_true",
        help="recompute offline quality staging and gold/negative/reject partitions",
    )
    operation.add_argument(
        "--migrate-monotonic-expansion",
        action="store_true",
        help=(
            "explicitly rebind a stopped v2 run to the declared larger collection "
            "contract; performs no provider request"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    raw = _simple_config(args.config.resolve())
    config = AutomationConfig.from_mapping(raw, repo_root=repo_root)
    if args.migrate_monotonic_expansion:
        if args.dry_run or args.wait_cooldown:
            print(
                "anchor-automation: migration cannot be combined with runtime flags",
                file=sys.stderr,
            )
            return 2
        try:
            migration = migrate_monotonic_expansion_status(config)
        except (OSError, ValueError) as error:
            print(
                f"anchor-automation: {type(error).__name__}: {str(error)[:240]}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(migration, ensure_ascii=False, indent=2))
        return 0
    if args.status_only:
        if not config.status_path.exists():
            print(json.dumps({"state": "not_started"}, indent=2))
            return 0
        status = json.loads(config.status_path.read_text(encoding="utf-8"))
        if status.get("schema_version") == AUTOMATION_SCHEMA_VERSION:
            _verify_status_config_binding(config, status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if args.partition_only:
        persisted_status: dict[str, Any] | None = None
        try:
            if config.status_path.exists():
                loaded_status = json.loads(
                    config.status_path.read_text(encoding="utf-8")
                )
                if not isinstance(loaded_status, dict):
                    raise ValueError("automation status must be a JSON object")
                if loaded_status.get("schema_version") == AUTOMATION_SCHEMA_VERSION:
                    _verify_status_config_binding(config, loaded_status)
                    if (
                        loaded_status.get("config_binding_sha256")
                        != config.status_binding_sha256
                    ):
                        loaded_status = _migrate_legacy_binding_status(
                            config,
                            loaded_status,
                            trigger="offline_partition_refresh",
                        )
                    persisted_status = loaded_status
            manifest = partition_collected_records(config)
            if persisted_status is not None:
                persisted_status["partition"] = manifest
                persisted_status.pop("partition_stale_reason", None)
                persisted_status["partition_refreshed_at"] = _iso()
                persisted_status["updated_at"] = _iso()
                _atomic_write_json(config.status_path, persisted_status)
        except (OSError, ValueError) as error:
            print(
                f"anchor-automation: {type(error).__name__}: {str(error)[:240]}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0 if manifest["training_ready"] else 3
    credential_env = provider_spec(raw).api_key_env
    if not args.dry_run and not os.environ.get(credential_env):
        print(
            "anchor-automation: credential environment variable is not set",
            file=sys.stderr,
        )
        return 2
    try:
        runner = AutomationRunner(
            config=config,
            teachers=_build_teachers(raw, dry_run=args.dry_run),
        )
        status = asyncio.run(runner.run(wait_for_cooldown=args.wait_cooldown))
    except (OSError, ValueError, RuntimeError) as error:
        print(
            f"anchor-automation: {type(error).__name__}: {str(error)[:240]}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["state"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
