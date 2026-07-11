from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from .gold import merge_gold_jsonl
from .harness import ToolingHarness
from .models import GoldRecord, SampleSpec
from .policy import ToolPolicy
from .runner import AgentExecutor
from .skills import SkillSourceRegistry
from .trace import digest_text


RAMP_STAGES = (1, 2, 4, 8)


def _project_path(project_root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{label} must be project-relative")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root") from exc
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class LiveBatchConfig:
    candidate_manifest: Path
    split_policy: Path
    skill_registry: Path
    workspace_root: Path
    gold_output: Path
    concurrency_stages: tuple[int, ...] = RAMP_STAGES
    samples_per_stage: tuple[int, ...] = RAMP_STAGES
    minimum_stage_success_rate: float = 1.0
    max_iterations: int = 8
    timeout_seconds: float = 900.0

    @classmethod
    def load(cls, project_root: str | Path, path: str | Path) -> "LiveBatchConfig":
        root = Path(project_root).resolve()
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or loaded.get("schema_version") != (
            "anchor.opencode-live-batch.v1"
        ):
            raise ValueError("unsupported OpenCode live batch schema")

        def integers(name: str) -> tuple[int, ...]:
            value = loaded.get(name)
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            try:
                return tuple(int(item) for item in value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain integers") from exc

        concurrency = integers("concurrency_stages")
        samples = integers("samples_per_stage")
        if concurrency != RAMP_STAGES:
            raise ValueError("concurrency_stages must be exactly 1,2,4,8")
        if len(samples) != len(concurrency) or any(value < 1 for value in samples):
            raise ValueError("samples_per_stage must have four positive values")
        success_rate = float(loaded.get("minimum_stage_success_rate", 1.0))
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("minimum_stage_success_rate must be between 0 and 1")
        return cls(
            candidate_manifest=_project_path(root, loaded.get("candidate_manifest"), "candidate_manifest"),
            split_policy=_project_path(root, loaded.get("split_policy"), "split_policy"),
            skill_registry=_project_path(root, loaded.get("skill_registry"), "skill_registry"),
            workspace_root=_project_path(root, loaded.get("workspace_root"), "workspace_root"),
            gold_output=_project_path(root, loaded.get("gold_output"), "gold_output"),
            concurrency_stages=concurrency,
            samples_per_stage=samples,
            minimum_stage_success_rate=success_rate,
            max_iterations=int(loaded.get("max_iterations", 8)),
            timeout_seconds=float(loaded.get("timeout_seconds", 900.0)),
        )


def _heldout_values(path: Path) -> tuple[set[str], set[str]]:
    identifiers: set[str] = set()
    requirements: set[str] = set()
    if path.suffix.casefold() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"held-out JSONL record is not an object: {path}")
            for name in ("case_id", "seed_id", "task_id", "sample_id"):
                if value.get(name):
                    identifiers.add(str(value[name]).casefold())
            if value.get("requirement"):
                requirements.add(" ".join(str(value["requirement"]).casefold().split()))
    return identifiers, requirements


def verify_execution_split(
    project_root: str | Path,
    split_policy_path: str | Path,
    candidate_manifest: str | Path,
) -> tuple[set[str], set[str]]:
    """Verify the independent held-out input list before loading candidates."""

    root = Path(project_root).resolve()
    loaded = yaml.safe_load(Path(split_policy_path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping) or loaded.get("schema_version") != (
        "anchor.execution-split-policy.v1"
    ):
        raise ValueError("unsupported execution split policy schema")
    candidate_paths = {
        _project_path(root, item, "candidate input")
        for item in loaded.get("candidate_inputs", [])
    }
    requested = Path(candidate_manifest).resolve()
    if requested not in candidate_paths:
        raise ValueError("candidate manifest is absent from the audited input list")
    identifiers: set[str] = set()
    requirements: set[str] = set()
    heldout_paths: set[Path] = set()
    for index, item in enumerate(loaded.get("heldout_inputs", [])):
        if not isinstance(item, Mapping):
            raise ValueError(f"heldout_inputs[{index}] must be an object")
        path = _project_path(root, item.get("path"), f"heldout_inputs[{index}].path")
        if path in candidate_paths:
            raise ValueError("candidate and held-out input paths overlap")
        expected = str(item.get("sha256", "")).casefold()
        if len(expected) != 64 or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"held-out input hash mismatch: {path}")
        heldout_paths.add(path)
        found_ids, found_requirements = _heldout_values(path)
        identifiers.update(found_ids)
        requirements.update(found_requirements)
    if not heldout_paths:
        raise ValueError("at least one independent held-out input is required")
    return identifiers, requirements


def load_candidate_samples(
    project_root: str | Path,
    manifest_path: str | Path,
    registry: SkillSourceRegistry,
    *,
    heldout_identifiers: set[str],
    heldout_requirements: set[str],
) -> tuple[SampleSpec, ...]:
    root = Path(project_root).resolve()
    loaded = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping) or loaded.get("schema_version") != (
        "anchor.execution-candidates.v1"
    ):
        raise ValueError("unsupported execution candidate schema")
    raw_tasks = loaded.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("candidate manifest needs non-empty tasks")
    samples: list[SampleSpec] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_tasks):
        if not isinstance(value, Mapping):
            raise ValueError(f"tasks[{index}] must be an object")
        sample_id = str(value.get("task_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate candidate task id: {sample_id!r}")
        if sample_id.casefold() in heldout_identifiers:
            raise ValueError(f"candidate task id collides with held-out: {sample_id}")
        seen.add(sample_id)
        source = _project_path(root, value.get("source_dir"), f"tasks[{index}].source_dir")
        if not source.is_dir():
            raise ValueError(f"candidate source directory is missing: {source}")
        prompt = str(value.get("task", "")).strip()
        if not prompt and value.get("requirement_file"):
            requirement = _project_path(
                root, value.get("requirement_file"), f"tasks[{index}].requirement_file"
            )
            prompt = requirement.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"candidate task is empty: {sample_id}")
        normalized = " ".join(prompt.casefold().split())
        if normalized in heldout_requirements or any(
            identifier in normalized for identifier in heldout_identifiers
        ):
            raise ValueError(f"candidate prompt leaks a held-out input: {sample_id}")
        source_ids = tuple(str(item) for item in value.get("skill_sources", []))
        composed, provenance = registry.compose_execution_prompt(prompt, source_ids)
        required = tuple(str(item) for item in value.get("required_validations", ["build"]))
        if not required or any(item not in {"build", "test", "lint"} for item in required):
            raise ValueError(f"invalid required validations: {sample_id}")
        samples.append(SampleSpec(sample_id, composed, source, required, provenance))
    return tuple(samples)


@dataclass(frozen=True)
class BatchStageResult:
    concurrency: int
    records: tuple[GoldRecord, ...]
    passed_gate: bool


def batch_run_succeeded(
    stages: Sequence[BatchStageResult], requested_stages: int
) -> bool:
    """Judge only the explicitly requested stage slice, never the full configured ramp."""

    return (
        requested_stages >= 1
        and len(stages) == requested_stages
        and all(stage.passed_gate for stage in stages)
    )


def _isolated_failure_record(
    sample: SampleSpec, executor: AgentExecutor, policy: ToolPolicy
) -> GoldRecord:
    return GoldRecord(
        sample_id=sample.sample_id,
        backend=executor.backend_name,
        success=False,
        workspace_id="not-created",
        max_iterations=policy.max_iterations,
        timeout_seconds=policy.timeout_seconds,
        agent_exit_code=125,
        timed_out=False,
        duration_ms=0.0,
        validations=(),
        tool_trace=(),
        changed_files=(),
        task_bundle_sha256=digest_text(sample.prompt),
        agent_stdout_sha256=None,
        agent_stderr_sha256=None,
        skill_provenance=sample.skill_provenance,
        public_outcome=None,
        error_codes=("isolated_sample_exception", "public_outcome_missing"),
    )


def run_live_batch(
    *,
    samples: Sequence[SampleSpec],
    config: LiveBatchConfig,
    executor: AgentExecutor,
    max_stages: int = 1,
    on_stage: Callable[[tuple[GoldRecord, ...]], None] | None = None,
) -> tuple[BatchStageResult, ...]:
    """Run isolated stages; one sample exception never aborts its siblings."""

    if not 1 <= max_stages <= len(config.concurrency_stages):
        raise ValueError(
            f"max_stages must be between 1 and {len(config.concurrency_stages)}"
        )
    selected_concurrency = config.concurrency_stages[:max_stages]
    selected_sample_counts = config.samples_per_stage[:max_stages]
    required_count = sum(selected_sample_counts)
    if len(samples) < required_count:
        raise ValueError(f"batch needs {required_count} candidates, found {len(samples)}")
    policy = ToolPolicy(
        max_iterations=config.max_iterations,
        timeout_seconds=config.timeout_seconds,
    )
    harness = ToolingHarness(config.workspace_root, executor, policy=policy)
    stages: list[BatchStageResult] = []
    offset = 0
    for concurrency, count in zip(selected_concurrency, selected_sample_counts):
        stage_samples = samples[offset : offset + count]
        offset += count
        completed: list[GoldRecord] = []
        # Harness failures are isolated at the future boundary and reduced to a
        # content-free failure record, so sibling samples continue without leaking
        # exception strings or partially collected model output.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(harness.run_sample, sample): sample for sample in stage_samples}
            failures = 0
            for future in as_completed(futures):
                try:
                    completed.append(future.result())
                except Exception:  # noqa: BLE001 - per-sample isolation boundary
                    failures += 1
                    completed.append(_isolated_failure_record(futures[future], executor, policy))
        records = tuple(sorted(completed, key=lambda item: item.sample_id))
        if on_stage is not None and records:
            on_stage(records)
        success_count = sum(record.success for record in records)
        rate = success_count / count
        passed = failures == 0 and rate >= config.minimum_stage_success_rate
        stages.append(BatchStageResult(concurrency, records, passed))
        if not passed:
            break
    return tuple(stages)


def merge_stage_into_gold(records: tuple[GoldRecord, ...], path: Path) -> None:
    merge_gold_jsonl(records, path)
