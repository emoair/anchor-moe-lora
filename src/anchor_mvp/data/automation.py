"""Unattended, gated concurrency ramp for defensive data distillation."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from ..benchmark.heldout import (
    check_training_leakage,
    file_sha256,
    verify_heldout_manifest,
    verify_leak_audit,
)
from .cleaning import validate_safe_payload
from .cli import _as_bool, _simple_config
from .pipeline import DistillationPipeline, PipelineReport
from .schema import TASK_TYPES
from .storage import JsonlStore
from .teacher import (
    BudgetExceeded,
    ClientDeadlineExceeded,
    CompatibleTeacher,
    MockTeacher,
    RateLimitError,
    Teacher,
)


AUTOMATION_SCHEMA_VERSION = "1.0"
REQUIRED_STAGES = (1, 2, 4, 8)


def _int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated list")
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integers") from error


def chargeable_failure_count(errors: Sequence[str]) -> int:
    """Count originating failures without charging dependency cascades."""

    return sum("UpstreamDependencyError" not in error for error in errors)


@dataclass(frozen=True)
class AutomationConfig:
    sop_dir: Path
    output_dir: Path
    heldout_cases: Path | None = None
    heldout_fixtures_root: Path | None = None
    heldout_manifest: Path | None = None
    heldout_leak_audit: Path | None = None
    concurrency_stages: tuple[int, ...] = REQUIRED_STAGES
    stage_seed_counts: tuple[int, ...] = (3, 6, 12, 24)
    min_success_rate: float = 1.0
    max_duplicate_rate: float = 0.0
    max_safety_violations: int = 0
    max_failures: int = 8
    max_requests: int = 200
    max_output_tokens_total: int = 1_000_000
    cooldown_seconds: int = 18_000
    cooldown_poll_seconds: int = 60
    max_stagnant_gate_rounds: int = 5

    def __post_init__(self) -> None:
        if self.concurrency_stages != REQUIRED_STAGES:
            raise ValueError("concurrency_stages must be exactly 1,2,4,8")
        if max(self.concurrency_stages) > 8:
            raise ValueError("automation concurrency hard limit is 8")
        if len(self.stage_seed_counts) != len(self.concurrency_stages):
            raise ValueError("stage_seed_counts must have one target per concurrency stage")
        if any(value < 1 for value in self.stage_seed_counts):
            raise ValueError("stage seed targets must be positive")
        if tuple(sorted(set(self.stage_seed_counts))) != self.stage_seed_counts:
            raise ValueError("stage seed targets must be strictly increasing")
        if not 0 <= self.min_success_rate <= 1:
            raise ValueError("min_success_rate must be between 0 and 1")
        if not 0 <= self.max_duplicate_rate <= 1:
            raise ValueError("max_duplicate_rate must be between 0 and 1")
        if self.max_safety_violations < 0 or self.max_failures < 0:
            raise ValueError("failure and safety budgets cannot be negative")
        if self.max_requests < 1 or self.max_output_tokens_total < 1:
            raise ValueError("request and output-token budgets must be positive")
        if self.cooldown_seconds < 1 or self.cooldown_poll_seconds < 1:
            raise ValueError("cooldown values must be positive")
        if self.max_stagnant_gate_rounds < 1:
            raise ValueError("max_stagnant_gate_rounds must be positive")
        heldout_paths = (
            self.heldout_cases,
            self.heldout_fixtures_root,
            self.heldout_manifest,
            self.heldout_leak_audit,
        )
        if any(path is not None for path in heldout_paths) and not all(
            path is not None for path in heldout_paths
        ):
            raise ValueError("held-out automation gate paths must be configured together")

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "automation"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def events_path(self) -> Path:
        return self.state_dir / "events.jsonl"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, repo_root: Path) -> "AutomationConfig":
        def path_setting(name: str, default: str) -> Path:
            path = Path(str(value.get(name, default)))
            return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()

        def optional_path(name: str) -> Path | None:
            raw = value.get(name)
            if raw is None or not str(raw).strip():
                return None
            path = Path(str(raw))
            return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(
            sop_dir=path_setting("sop_dir", "skills"),
            output_dir=path_setting("output_dir", "data/automation-run"),
            heldout_cases=optional_path("heldout_cases"),
            heldout_fixtures_root=optional_path("heldout_fixtures_root"),
            heldout_manifest=optional_path("heldout_manifest"),
            heldout_leak_audit=optional_path("heldout_leak_audit"),
            concurrency_stages=_int_tuple(
                value.get("concurrency_stages", "1,2,4,8"), name="concurrency_stages"
            ),
            stage_seed_counts=_int_tuple(
                value.get("stage_seed_counts", "3,6,12,24"), name="stage_seed_counts"
            ),
            min_success_rate=float(value.get("min_success_rate", 1.0)),
            max_duplicate_rate=float(value.get("max_duplicate_rate", 0.0)),
            max_safety_violations=int(value.get("max_safety_violations", 0)),
            max_failures=int(value.get("max_failures", 8)),
            max_requests=int(value.get("max_requests", 200)),
            max_output_tokens_total=int(value.get("max_output_tokens_total", 1_000_000)),
            cooldown_seconds=int(value.get("cooldown_seconds", 18_000)),
            cooldown_poll_seconds=int(value.get("cooldown_poll_seconds", 60)),
            max_stagnant_gate_rounds=int(value.get("max_stagnant_gate_rounds", 5)),
        )


class _TrackedTeacher:
    """Logical usage fallback for mocks; real clients expose wire-attempt usage."""

    def __init__(self, teacher: Teacher, *, max_requests: int, max_output_tokens: int) -> None:
        self.inner = teacher
        self.model = teacher.model
        self.base_url = teacher.base_url
        self.protocol = teacher.protocol
        self.generation_params = teacher.generation_params
        self.max_requests = max_requests
        self.max_output_tokens = max_output_tokens
        self.logical_requests = 0
        self.logical_output_tokens = 0

        limiter = getattr(teacher, "limit_remaining_budget", None)
        if limiter is not None:
            limiter(max_requests=max_requests, max_output_tokens=max_output_tokens)

    async def complete(self, *, system: str, user: str) -> str:
        if self.logical_requests >= self.max_requests:
            raise BudgetExceeded("automation request budget exhausted")
        self.logical_requests += 1
        result = await self.inner.complete(system=system, user=user)
        self.logical_output_tokens += max(1, len(result) // 4)
        if self.logical_output_tokens > self.max_output_tokens:
            raise BudgetExceeded("automation output-token budget exhausted")
        self.protocol = self.inner.protocol
        self.base_url = self.inner.base_url
        return result

    @property
    def usage_snapshot(self) -> dict[str, int]:
        snapshot = getattr(self.inner, "usage_snapshot", None)
        if snapshot is not None:
            return dict(snapshot)
        return {
            "requests": self.logical_requests,
            "output_tokens": self.logical_output_tokens,
        }

    @property
    def usage_budget_id(self) -> int:
        return int(getattr(self.inner, "usage_budget_id", id(self)))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


class AutomationRunner:
    def __init__(
        self,
        *,
        config: AutomationConfig,
        teacher: Teacher | None = None,
        teachers: Mapping[str, Teacher] | None = None,
    ) -> None:
        self.config = config
        self.status = self._load_status()
        remaining_requests = config.max_requests - int(self.status["budgets"]["requests_used"])
        remaining_tokens = config.max_output_tokens_total - int(
            self.status["budgets"]["output_tokens_used"]
        )
        if teachers is None:
            if teacher is None:
                raise ValueError("automation requires a teacher or task teacher mapping")
            teachers = {name: teacher for name in ("seed", *TASK_TYPES)}
        missing_workers = set(("seed", *TASK_TYPES)).difference(teachers)
        if missing_workers:
            raise ValueError(f"missing automation workers: {sorted(missing_workers)}")
        wrapper_by_teacher: dict[int, _TrackedTeacher] = {}
        self.workers: dict[str, _TrackedTeacher] = {}
        for name in ("seed", *TASK_TYPES):
            raw_teacher = teachers[name]
            wrapper = wrapper_by_teacher.get(id(raw_teacher))
            if wrapper is None:
                wrapper = _TrackedTeacher(
                    raw_teacher,
                    max_requests=max(0, remaining_requests),
                    max_output_tokens=max(0, remaining_tokens),
                )
                wrapper_by_teacher[id(raw_teacher)] = wrapper
            self.workers[name] = wrapper
        self._usage_sources = {
            worker.usage_budget_id: worker for worker in self.workers.values()
        }
        self._usage_baseline = self._aggregate_usage()
        self.events = JsonlStore(config.events_path)

    def _load_status(self) -> dict[str, Any]:
        path = self.config.status_path
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))
            if status.get("schema_version") != AUTOMATION_SCHEMA_VERSION:
                raise ValueError("unsupported automation status schema")
            budgets = status.setdefault("budgets", {})
            budgets["max_requests"] = self.config.max_requests
            budgets["max_output_tokens_total"] = self.config.max_output_tokens_total
            budgets["max_failures"] = self.config.max_failures
            return status
        return {
            "schema_version": AUTOMATION_SCHEMA_VERSION,
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
            "budgets": {
                "requests_used": 0,
                "output_tokens_used": 0,
                "failures_used": 0,
                "max_requests": self.config.max_requests,
                "max_output_tokens_total": self.config.max_output_tokens_total,
                "max_failures": self.config.max_failures,
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
        }

    def _save_status(self) -> None:
        self.status["updated_at"] = _iso()
        path = self.config.status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

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

    def _sync_usage(self) -> None:
        current = self._aggregate_usage()
        requests_delta = max(0, current["requests"] - self._usage_baseline["requests"])
        tokens_delta = max(0, current["output_tokens"] - self._usage_baseline["output_tokens"])
        budgets = self.status["budgets"]
        budgets["requests_used"] += requests_delta
        budgets["output_tokens_used"] += tokens_delta
        self._usage_baseline = current

    def _aggregate_usage(self) -> dict[str, int]:
        snapshots = [worker.usage_snapshot for worker in self._usage_sources.values()]
        return {
            "requests": sum(item["requests"] for item in snapshots),
            "output_tokens": sum(item["output_tokens"] for item in snapshots),
        }

    def _budget_exhausted(self) -> str | None:
        budgets = self.status["budgets"]
        if budgets["requests_used"] >= self.config.max_requests:
            return "request_budget"
        if budgets["output_tokens_used"] >= self.config.max_output_tokens_total:
            return "output_token_budget"
        if budgets["failures_used"] > 0 and budgets["failures_used"] >= self.config.max_failures:
            return "failure_budget"
        return None

    async def run(self, *, wait_for_cooldown: bool = False) -> dict[str, Any]:
        if self.status["state"] == "complete":
            return self.status
        if self.status["state"] == "ready":
            self.status["state"] = "running"
            self._event("automation_started", stages=list(self.config.concurrency_stages))

        retry_stage_index: int | None = None
        previous_gate_records: int | None = None
        stagnant_gate_rounds = 0
        while int(self.status["stage_index"]) < len(self.config.concurrency_stages):
            if await self._cooldown_gate(wait_for_cooldown=wait_for_cooldown):
                return self.status
            exhausted = self._budget_exhausted()
            if exhausted:
                self.status["state"] = "budget_exhausted"
                self._event("budget_exhausted", budget=exhausted)
                return self.status

            stage_index = int(self.status["stage_index"])
            if retry_stage_index != stage_index:
                retry_stage_index = stage_index
                previous_gate_records = None
                stagnant_gate_rounds = 0
            concurrency = self.config.concurrency_stages[stage_index]
            target = self.config.stage_seed_counts[stage_index]
            self.status["state"] = "running"
            self.status["current_concurrency"] = concurrency
            self._event("stage_started", concurrency=concurrency, seed_target=target)
            started = time.monotonic()
            try:
                report = await self._run_stage(seed_target=target, concurrency=concurrency)
            except RateLimitError as error:
                self._sync_usage()
                self._set_cooldown(error.retry_after_seconds)
                if not wait_for_cooldown:
                    return self.status
                continue
            except ClientDeadlineExceeded as error:
                self._sync_usage()
                self.status["budgets"]["failures_used"] += 1
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
                self._sync_usage()
                self.status["budgets"]["failures_used"] += 1
                self.status["state"] = "failed"
                self._event("stage_failed", error_type=type(error).__name__, message=str(error)[:240])
                return self.status

            elapsed = max(0.000001, time.monotonic() - started)
            self._sync_usage()
            # One rejected upstream sample can legitimately suppress several
            # downstream tasks. Charge only the originating generation/validation
            # error so a dependency cascade cannot exhaust the unattended budget.
            self.status["budgets"]["failures_used"] += chargeable_failure_count(
                report.errors
            )
            if report.rate_limited:
                self._set_cooldown(report.retry_after_seconds)
                if not wait_for_cooldown:
                    return self.status
                continue
            if report.client_deadline:
                self.status["state"] = "client_deadline"
                deadline_error = next(
                    (item for item in report.errors if "ClientDeadlineExceeded" in item),
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
            if not gate["passed"]:
                records = int(gate["records"])
                if previous_gate_records is not None and records <= previous_gate_records:
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
        )
        await seed_pipeline.generate_seeds(seed_target)
        self._event("worker_completed", worker="seed", seed_target=seed_target)
        written: dict[str, int] = {}
        skipped: dict[str, int] = {}
        errors: list[str] = []
        rate_limited = False
        retry_after: float | None = None
        client_deadline = False
        for task_type in TASK_TYPES:
            self.status["current_worker"] = task_type
            self._event("worker_started", worker=task_type, concurrency=concurrency)
            worker_pipeline = DistillationPipeline(
                teacher=self.workers[task_type],
                sop_dir=self.config.sop_dir,
                output_dir=self.config.output_dir,
                concurrency=concurrency,
            )
            report = await worker_pipeline.run(seed_count=seed_target, tasks=[task_type])
            written.update(report.written_by_task)
            skipped.update(report.skipped_by_task)
            errors.extend(report.errors)
            rate_limited = rate_limited or report.rate_limited
            client_deadline = client_deadline or report.client_deadline
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
            await asyncio.sleep(min(float(self.config.cooldown_poll_seconds), remaining))

    def _set_cooldown(self, retry_after_seconds: float | None) -> None:
        seconds = max(float(self.config.cooldown_seconds), retry_after_seconds or 0.0)
        until = _now() + timedelta(seconds=seconds)
        self.status["state"] = "cooldown"
        self.status["cooldown_until"] = _iso(until)
        self.status["budgets"]["failures_used"] += 1
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
        final_records = self.config.stage_seed_counts[-1] * len(TASK_TYPES)
        remaining = max(0, final_records - records)
        self.status["metrics"] = {
            "records": records,
            "elapsed_seconds": elapsed,
            "last_stage_seconds": stage_elapsed,
            "throughput_records_per_second": throughput,
            "eta_seconds": remaining / throughput if throughput > 0 else None,
        }


def evaluate_gate(config: AutomationConfig, seed_target: int) -> dict[str, Any]:
    total_records = 0
    successful = 0
    duplicate_count = 0
    safety_violations = 0
    schema_errors: list[str] = []
    for task_type in TASK_TYPES:
        path = config.output_dir / f"data_{task_type}.jsonl"
        if not path.is_file():
            schema_errors.append(f"missing {path.name}")
            continue
        records: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                records.append(value)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            schema_errors.append(f"{path.name}: {error}")
            continue
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
                        {"input": record.get("input", {}), "output": record.get("output", {})},
                    )
                except ValueError:
                    safety_violations += 1
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

    expected = seed_target * len(TASK_TYPES)
    success_rate = successful / expected if expected else 0.0
    duplicate_rate = duplicate_count / total_records if total_records else 0.0
    passed = (
        success_rate >= config.min_success_rate
        and duplicate_rate <= config.max_duplicate_rate
        and safety_violations <= config.max_safety_violations
        and not schema_errors
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
    }


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
) -> CompatibleTeacher:
    protocol = str(value.get("protocol", "anthropic"))
    default_base = (
        "https://api.kimi.com/coding/"
        if protocol == "anthropic"
        else "https://api.kimi.com/coding/v1"
    )
    return CompatibleTeacher(
        base_url=str(value.get("base_url", default_base)),
        fallback_base_url=str(value.get("fallback_base_url", "https://api.kimi.com/coding/v1")),
        model=str(os.environ.get("KIMI_MODEL_ID") or value.get("model", "kimi-for-coding")),
        protocol=protocol,  # type: ignore[arg-type]
        fallback_protocol=str(value.get("fallback_protocol", "openai")),  # type: ignore[arg-type]
        api_key_env=str(value.get("api_key_env", "KIMI_API_KEY")),
        anthropic_version=str(value.get("anthropic_version", "2023-06-01")),
        user_agent=str(value.get("user_agent", "anchor-moe-lora/0.1")),
        timeout_seconds=float(value.get("timeout_seconds", 600)),
        max_retries=int(value.get("max_retries", 1)),
        wall_clock_deadline_seconds=float(value.get("wall_clock_deadline_seconds", 900)),
        temperature=float(value.get("temperature", 0.2)),
        max_tokens=int(value.get("max_tokens", 16384)),
        max_requests=int(value.get("max_requests", 200)),
        max_output_tokens_total=int(value.get("max_output_tokens_total", 1_000_000)),
        thinking_enabled=_as_bool(value.get("thinking_enabled", True)),
        thinking_effort=thinking_effort,
        thinking_budget_tokens=int(value.get("thinking_budget_tokens", 4096)),
        stream_openai=_as_bool(value.get("stream_openai", True)),
        stream_options_include_usage=_as_bool(value.get("stream_options_include_usage", False)),
    )


def _build_teachers(value: Mapping[str, Any], *, dry_run: bool) -> dict[str, Teacher]:
    if dry_run:
        mock = MockTeacher()
        return {name: cast(Teacher, mock) for name in ("seed", *TASK_TYPES)}
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
        name: _build_teacher(value, thinking_effort=effort)
        for name, effort in efforts.items()
    }
    owner = workers["seed"]
    for name, worker in workers.items():
        if name != "seed":
            worker.share_usage_budget(owner)
    return cast(dict[str, Teacher], workers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gated unattended Anchor-MoE-LoRA distillation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="run the deterministic mock E2E")
    parser.add_argument(
        "--wait-cooldown",
        action="store_true",
        help="remain visible and resume after persisted 429 cooldowns",
    )
    parser.add_argument("--status-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    raw = _simple_config(args.config.resolve())
    config = AutomationConfig.from_mapping(raw, repo_root=repo_root)
    if args.status_only:
        if not config.status_path.exists():
            print(json.dumps({"state": "not_started"}, indent=2))
            return 0
        print(config.status_path.read_text(encoding="utf-8"))
        return 0
    if not args.dry_run and not os.environ.get(str(raw.get("api_key_env", "KIMI_API_KEY"))):
        print("anchor-automation: credential environment variable is not set", file=sys.stderr)
        return 2
    try:
        runner = AutomationRunner(
            config=config,
            teachers=_build_teachers(raw, dry_run=args.dry_run),
        )
        status = asyncio.run(runner.run(wait_for_cooldown=args.wait_cooldown))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"anchor-automation: {type(error).__name__}: {str(error)[:240]}", file=sys.stderr)
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["state"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
