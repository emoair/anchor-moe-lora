from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from .config import DEFAULT_PROVIDER, OpenCodeProvider
from .gold import merge_gold_jsonl
from .harness import ToolingHarness
from .models import GoldRecord, SampleSpec, sample_contract_sha256
from .policy import ToolPolicy
from .runner import AgentExecutor, AnchorSandboxOptions, ControlledSessionCapture
from .skills import SkillSourceRegistry


DEFAULT_CONCURRENCY_STAGES = (1,)


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


def _positive_integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in value
    ):
        raise ValueError(f"{name} must contain only positive integers")
    return tuple(value)


def _load_protected_files(
    source: Path, value: object, *, label: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must pin at least one fixture file")
    protected: list[tuple[str, str]] = []
    for raw_path, raw_digest in value.items():
        relative = Path(str(raw_path))
        if relative.is_absolute():
            raise ValueError(f"{label} paths must be fixture-relative")
        path = (source / relative).resolve()
        try:
            normalized = path.relative_to(source).as_posix()
        except ValueError as exc:
            raise ValueError(f"{label} path escapes fixture: {relative}") from exc
        expected = str(raw_digest).casefold()
        if len(expected) != 64 or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"protected fixture hash mismatch: {path}")
        protected.append((normalized, expected))
    return tuple(sorted(protected))


@dataclass(frozen=True)
class LiveBatchConfig:
    candidate_manifest: Path
    split_policy: Path
    skill_registry: Path
    workspace_root: Path
    gold_output: Path
    provider: OpenCodeProvider = DEFAULT_PROVIDER
    opencode_executable: Path | None = None
    attempts_output: Path | None = None
    session_candidates: Path | None = None
    session_staging: Path | None = None
    session_quarantine: Path | None = None
    heldout_cases: Path | None = None
    heldout_fixtures_root: Path | None = None
    heldout_manifest: Path | None = None
    sandbox_linux_executable: Path | None = None
    sandbox_wsl_distro: str | None = None
    sandbox_supervisor: str | None = None
    sandbox_memory: str | None = None
    sandbox_cpus: str | None = None
    sandbox_pids: int | None = None
    sandbox_timeout_seconds: int | None = None
    retain_workspace: bool = False
    concurrency_stages: tuple[int, ...] = DEFAULT_CONCURRENCY_STAGES
    samples_per_stage: tuple[int, ...] = DEFAULT_CONCURRENCY_STAGES
    minimum_stage_success_rate: float = 1.0
    max_iterations: int | None = None
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if not isinstance(self.provider, OpenCodeProvider):
            raise ValueError("provider must be an audited OpenCodeProvider")
        concurrency = _positive_integer_sequence(
            self.concurrency_stages, name="concurrency_stages"
        )
        samples = _positive_integer_sequence(
            self.samples_per_stage, name="samples_per_stage"
        )
        if len(samples) != len(concurrency):
            raise ValueError(
                "samples_per_stage must have one positive value per concurrency stage"
            )
        AnchorSandboxOptions(
            linux_executable=self.sandbox_linux_executable,
            wsl_distro=self.sandbox_wsl_distro,
            supervisor=self.sandbox_supervisor,
            memory=self.sandbox_memory,
            cpus=self.sandbox_cpus,
            pids=self.sandbox_pids,
            timeout_seconds=self.sandbox_timeout_seconds,
        )
        if not isinstance(self.retain_workspace, bool):
            raise ValueError("retain_workspace must be a boolean")
        if self.max_iterations is not None and (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError(
                "max_iterations must be a positive integer when configured"
            )

    @classmethod
    def load(cls, project_root: str | Path, path: str | Path) -> "LiveBatchConfig":
        root = Path(project_root).resolve()
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or loaded.get("schema_version") != (
            "anchor.opencode-live-batch.v1"
        ):
            raise ValueError("unsupported OpenCode live batch schema")

        concurrency = _positive_integer_sequence(
            loaded.get("concurrency_stages", DEFAULT_CONCURRENCY_STAGES),
            name="concurrency_stages",
        )
        samples = _positive_integer_sequence(
            loaded.get("samples_per_stage", (1,) * len(concurrency)),
            name="samples_per_stage",
        )
        if len(samples) != len(concurrency):
            raise ValueError(
                "samples_per_stage must have one positive value per concurrency stage"
            )
        success_rate = float(loaded.get("minimum_stage_success_rate", 1.0))
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("minimum_stage_success_rate must be between 0 and 1")
        if (
            not isinstance(loaded.get("attempts_output"), str)
            or not str(loaded["attempts_output"]).strip()
        ):
            raise ValueError("attempts_output must be a project-relative path")
        capture_names = (
            "session_candidates",
            "session_quarantine",
            "heldout_cases",
            "heldout_fixtures_root",
            "heldout_manifest",
        )
        if not all(
            isinstance(loaded.get(name), str) and str(loaded[name]).strip()
            for name in capture_names
        ):
            raise ValueError(
                "batch config requires complete controlled session capture paths"
            )
        sandbox = loaded.get("anchor_sandbox", {})
        if not isinstance(sandbox, Mapping):
            raise ValueError("anchor_sandbox must be an object when configured")
        linux_executable: Path | None = None
        if sandbox.get("linux_executable") is not None:
            linux_executable = _project_path(
                root, sandbox["linux_executable"], "anchor_sandbox.linux_executable"
            )

        def optional_text(name: str) -> str | None:
            value = sandbox.get(name)
            return None if value is None else str(value)

        def optional_positive_int(name: str) -> int | None:
            value = sandbox.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"anchor_sandbox.{name} must be a positive integer")
            return value

        retain_workspace = loaded.get("retain_workspace", False)
        if not isinstance(retain_workspace, bool):
            raise ValueError("retain_workspace must be a boolean")
        provider = (
            DEFAULT_PROVIDER
            if loaded.get("provider") is None
            else OpenCodeProvider.from_mapping(loaded["provider"])
        )
        return cls(
            candidate_manifest=_project_path(
                root, loaded.get("candidate_manifest"), "candidate_manifest"
            ),
            split_policy=_project_path(
                root, loaded.get("split_policy"), "split_policy"
            ),
            skill_registry=_project_path(
                root, loaded.get("skill_registry"), "skill_registry"
            ),
            opencode_executable=_project_path(
                root, loaded.get("opencode_executable"), "opencode_executable"
            ),
            workspace_root=_project_path(
                root, loaded.get("workspace_root"), "workspace_root"
            ),
            gold_output=_project_path(root, loaded.get("gold_output"), "gold_output"),
            provider=provider,
            attempts_output=_project_path(
                root, loaded.get("attempts_output"), "attempts_output"
            ),
            session_candidates=_project_path(
                root, loaded["session_candidates"], "session_candidates"
            ),
            session_staging=_project_path(
                root,
                loaded.get(
                    "session_staging", "artifacts/tooling/session_staging.raw.jsonl"
                ),
                "session_staging",
            ),
            session_quarantine=_project_path(
                root, loaded["session_quarantine"], "session_quarantine"
            ),
            heldout_cases=_project_path(root, loaded["heldout_cases"], "heldout_cases"),
            heldout_fixtures_root=_project_path(
                root, loaded["heldout_fixtures_root"], "heldout_fixtures_root"
            ),
            heldout_manifest=_project_path(
                root, loaded["heldout_manifest"], "heldout_manifest"
            ),
            sandbox_linux_executable=linux_executable,
            sandbox_wsl_distro=optional_text("wsl_distro"),
            sandbox_supervisor=optional_text("supervisor"),
            sandbox_memory=optional_text("memory"),
            sandbox_cpus=optional_text("cpus"),
            sandbox_pids=optional_positive_int("pids"),
            sandbox_timeout_seconds=optional_positive_int("timeout_seconds"),
            retain_workspace=retain_workspace,
            concurrency_stages=concurrency,
            samples_per_stage=samples,
            minimum_stage_success_rate=success_rate,
            max_iterations=loaded.get("max_iterations"),
            timeout_seconds=float(loaded.get("timeout_seconds", 900.0)),
        )

    def controlled_capture(self, *, mode: str = "strict") -> ControlledSessionCapture:
        values = (
            self.session_candidates,
            self.session_quarantine,
            self.heldout_cases,
            self.heldout_fixtures_root,
            self.heldout_manifest,
        )
        if any(value is None for value in values):
            raise ValueError("controlled session capture is incomplete")
        return ControlledSessionCapture(
            *values,  # type: ignore[arg-type]
            staging_path=self.session_staging,
            mode=mode,
        )

    def anchor_sandbox_options(self) -> AnchorSandboxOptions:
        return AnchorSandboxOptions(
            linux_executable=self.sandbox_linux_executable,
            wsl_distro=self.sandbox_wsl_distro,
            supervisor=self.sandbox_supervisor,
            memory=self.sandbox_memory,
            cpus=self.sandbox_cpus,
            pids=self.sandbox_pids,
            timeout_seconds=self.sandbox_timeout_seconds,
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
        source = _project_path(
            root, value.get("source_dir"), f"tasks[{index}].source_dir"
        )
        if not source.is_dir():
            raise ValueError(f"candidate source directory is missing: {source}")
        prompt = str(value.get("task", "")).strip()
        requirement_relative: str | None = None
        if value.get("requirement_file"):
            requirement = _project_path(
                root, value.get("requirement_file"), f"tasks[{index}].requirement_file"
            )
            try:
                requirement_relative = requirement.relative_to(source).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"candidate requirement must be inside its fixture: {sample_id}"
                ) from exc
            requirement_prompt = requirement.read_text(encoding="utf-8").strip()
            if prompt and " ".join(prompt.split()) != " ".join(
                requirement_prompt.split()
            ):
                raise ValueError(
                    f"candidate task disagrees with fixture requirement: {sample_id}"
                )
            prompt = requirement_prompt
        if not prompt:
            raise ValueError(f"candidate task is empty: {sample_id}")
        normalized = " ".join(prompt.casefold().split())
        if normalized in heldout_requirements or any(
            identifier in normalized for identifier in heldout_identifiers
        ):
            raise ValueError(f"candidate prompt leaks a held-out input: {sample_id}")
        source_ids = tuple(str(item) for item in value.get("skill_sources", []))
        composed, provenance = registry.compose_execution_prompt(prompt, source_ids)
        required = tuple(
            str(item) for item in value.get("required_validations", ["build"])
        )
        if not required or any(
            item not in {"build", "test", "lint"} for item in required
        ):
            raise ValueError(f"invalid required validations: {sample_id}")
        protected = _load_protected_files(
            source,
            value.get("protected_files"),
            label=f"tasks[{index}].protected_files",
        )
        input_files = _load_protected_files(
            source, value.get("input_files"), label=f"tasks[{index}].input_files"
        )
        protected_paths = {path for path, _ in protected}
        if protected_paths.intersection(path for path, _ in input_files):
            raise ValueError(
                f"candidate input and protected paths overlap: {sample_id}"
            )
        if "package.json" not in protected_paths:
            raise ValueError(f"candidate must protect package.json: {sample_id}")
        if (
            requirement_relative is not None
            and requirement_relative not in protected_paths
        ):
            raise ValueError(
                f"candidate must protect its requirement file: {sample_id}"
            )
        if "test" in required and not any(
            path.startswith("test/") for path in protected_paths
        ):
            raise ValueError(
                f"candidate must protect at least one test file: {sample_id}"
            )
        requires_changes = value.get("requires_changes", False)
        if not isinstance(requires_changes, bool):
            raise ValueError(f"candidate requires_changes must be boolean: {sample_id}")
        samples.append(
            SampleSpec(
                sample_id,
                composed,
                source,
                required,
                provenance,
                protected,
                input_files,
                requires_changes,
            )
        )
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
        task_bundle_sha256=sample_contract_sha256(sample),
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
    collection_mode: bool = False,
    on_stage: Callable[[tuple[GoldRecord, ...]], None] | None = None,
) -> tuple[BatchStageResult, ...]:
    """Run isolated stages; one sample exception never aborts its siblings."""

    if collection_mode:
        capture = getattr(executor, "session_capture", None)
        if (
            not isinstance(capture, ControlledSessionCapture)
            or capture.mode != "collect"
            or not callable(getattr(executor, "finalize_capture", None))
        ):
            raise ValueError(
                "collection_mode requires collect-mode controlled session capture"
            )
    if not 1 <= max_stages <= len(config.concurrency_stages):
        raise ValueError(
            f"max_stages must be between 1 and {len(config.concurrency_stages)}"
        )
    selected_concurrency = config.concurrency_stages[:max_stages]
    selected_sample_counts = config.samples_per_stage[:max_stages]
    required_count = sum(selected_sample_counts)
    if len(samples) < required_count:
        raise ValueError(
            f"batch needs {required_count} candidates, found {len(samples)}"
        )
    policy = ToolPolicy(
        max_iterations=config.max_iterations,
        timeout_seconds=config.timeout_seconds,
    )
    harness = ToolingHarness(
        config.workspace_root,
        executor,
        policy=policy,
        retain_workspace=config.retain_workspace,
    )
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
            futures = {
                pool.submit(harness.run_sample, sample): sample
                for sample in stage_samples
            }
            failures = 0
            for future in as_completed(futures):
                try:
                    completed.append(future.result())
                except Exception:  # noqa: BLE001 - per-sample isolation boundary
                    failures += 1
                    completed.append(
                        _isolated_failure_record(futures[future], executor, policy)
                    )
        records = tuple(sorted(completed, key=lambda item: item.sample_id))
        if on_stage is not None and records:
            on_stage(records)
        success_count = sum(record.success for record in records)
        rate = success_count / count
        if collection_mode:
            hard_rejects = sum(
                any(
                    code.startswith("session_hard_reject_")
                    for code in record.error_codes
                )
                for record in records
            )
            passed = failures == 0 and hard_rejects == 0
        else:
            passed = failures == 0 and rate >= config.minimum_stage_success_rate
        stages.append(BatchStageResult(concurrency, records, passed))
        if not passed:
            break
    return tuple(stages)


def merge_stage_into_gold(records: tuple[GoldRecord, ...], path: Path) -> None:
    merge_gold_jsonl(records, path)
