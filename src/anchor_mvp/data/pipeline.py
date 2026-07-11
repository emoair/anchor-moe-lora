"""Asynchronous, resumable seed generation and task distillation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .cleaning import (
    build_inert_security_fixture,
    contains_active_payload,
    extract_frontend_payload,
    extract_json_object,
    sanitize_security_seed,
    validate_safe_payload,
)
from .mutator import mutate_frontend_code
from .prompts import seed_prompt, task_prompt, template_sha256
from .proposals import deterministic_tool_policy_oracle, generate_inert_tool_proposals
from .schema import DistilledRecord, SeedDemand, TASK_TYPES, TaskType, normalized_text, stable_id
from .sops import load_sop_directory
from .storage import JsonlStore, SeedStore, completed_seed_ids
from .teacher import ClientDeadlineExceeded, RateLimitError, Teacher


class UpstreamDependencyError(RuntimeError):
    """A task cannot run until its same-seed upstream record exists."""


def _ensure_frontend_public_trace(payload: dict[str, object]) -> str:
    """Add an explicitly attributed audit trace when the teacher omits one.

    This is contract evidence produced by the pipeline, not reconstructed model
    reasoning. The caller records the returned source label in provenance.
    """

    trace = payload.get("decision_trace")
    if isinstance(trace, list) and trace:
        return "teacher"
    output = payload.get("output")
    code = output.get("code", "") if isinstance(output, dict) else ""
    if not isinstance(code, str) or not code.strip():
        return "teacher_missing_invalid"
    payload["decision_trace"] = [
        {
            "check": "upstream contract",
            "evidence": "same-seed plan and tool-policy records were present",
            "action": "implement one bounded frontend component",
        },
        {
            "check": "tool boundary",
            "evidence": "tool proposals were inert and executed=false",
            "action": "produce code without executing tools",
        },
        {
            "check": "output contract",
            "evidence": f"tsx code length={len(code)} characters",
            "action": "send the artifact to the domain reviewer",
        },
    ]
    return "pipeline_contract_fallback"


def _normalize_frontend_payload(payload: dict[str, object]) -> str:
    """Normalize a small allowlist of code-only response shapes."""

    output = payload.get("output")
    if isinstance(output, dict) and isinstance(output.get("code"), str):
        return "canonical"
    language = str(payload.get("language", "tsx"))
    candidate: object = None
    source = "unrecognized"
    if isinstance(output, str):
        candidate, source = output, "output_string"
    elif isinstance(payload.get("code"), str):
        candidate, source = payload["code"], "top_level_code"
    elif isinstance(output, dict):
        for key in ("artifact", "implementation", "content"):
            if isinstance(output.get(key), str):
                candidate, source = output[key], f"output_{key}"
                break
    if isinstance(candidate, str) and candidate.strip():
        payload["output"] = {"language": language, "code": candidate}
    return source


@dataclass(frozen=True)
class PipelineReport:
    requested_seeds: int
    available_seeds: int
    written_by_task: dict[str, int]
    skipped_by_task: dict[str, int]
    errors: tuple[str, ...] = field(default_factory=tuple)
    rate_limited: bool = False
    retry_after_seconds: float | None = None
    client_deadline: bool = False


class DistillationPipeline:
    def __init__(
        self,
        *,
        teacher: Teacher,
        sop_dir: str | Path,
        output_dir: str | Path,
        concurrency: int = 8,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.teacher = teacher
        self.sops = load_sop_directory(sop_dir)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _complete(self, *, system: str, user: str) -> str:
        async with self.semaphore:
            return await self.teacher.complete(system=system, user=user)

    async def generate_seeds(self, count: int) -> list[SeedDemand]:
        if count < 0:
            raise ValueError("seed count cannot be negative")
        store = SeedStore(self.output_dir / "seeds.jsonl")
        seeds = [SeedDemand.from_mapping(item) for item in store.records]
        fingerprints = set(store.request_fingerprints)
        next_index = len(seeds)
        rounds = 0
        while len(seeds) < count and rounds < 5:
            missing = count - len(seeds)
            indices = range(next_index, next_index + missing)
            next_index += missing

            async def create(index: int) -> tuple[int, SeedDemand]:
                system, user = seed_prompt(index)
                payload = extract_json_object(await self._complete(system=system, user=user))
                if contains_active_payload(payload):
                    raise ValueError("seed contains active payload material")
                return index, SeedDemand.from_mapping(payload)

            results = await asyncio.gather(*(create(index) for index in indices), return_exceptions=True)
            rate_limits = [result for result in results if isinstance(result, RateLimitError)]
            if rate_limits:
                retry_values = [
                    item.retry_after_seconds
                    for item in rate_limits
                    if item.retry_after_seconds is not None
                ]
                raise RateLimitError(max(retry_values) if retry_values else None)
            deadlines = [result for result in results if isinstance(result, ClientDeadlineExceeded)]
            if deadlines:
                raise deadlines[0]
            for result in results:
                if isinstance(result, BaseException):
                    continue
                _, seed = result
                fingerprint = stable_id("request", normalized_text(seed.request))
                if fingerprint in fingerprints:
                    continue
                if store.append(seed.to_dict()):
                    fingerprints.add(fingerprint)
                    seeds.append(seed)
            rounds += 1
        if len(seeds) < count:
            raise RuntimeError(f"teacher produced only {len(seeds)} unique seeds after retries")
        return seeds[:count]

    async def run(
        self,
        *,
        seed_count: int,
        tasks: Iterable[TaskType] = TASK_TYPES,
        excluded_seed_ids: Iterable[str] = (),
    ) -> PipelineReport:
        requested = tuple(tasks)
        unknown = set(requested).difference(TASK_TYPES)
        if unknown:
            raise ValueError(f"unknown tasks: {', '.join(sorted(unknown))}")
        # User ordering cannot bypass same-seed dependencies.
        selected = tuple(task for task in TASK_TYPES if task in requested)
        excluded = frozenset(str(seed_id) for seed_id in excluded_seed_ids)
        seeds = await self.generate_seeds(seed_count)
        written: dict[str, int] = {}
        skipped: dict[str, int] = {}
        errors: list[str] = []
        rate_limited = False
        retry_after_seconds: float | None = None
        client_deadline = False
        for task_type in selected:
            store = JsonlStore(self.output_dir / f"data_{task_type}.jsonl")
            completed = completed_seed_ids(store.records)
            pending = [
                (index, seed)
                for index, seed in enumerate(seeds)
                if seed.seed_id not in completed and seed.seed_id not in excluded
            ]
            skipped[task_type] = len(seeds) - len(pending)

            async def distill(index: int, original_seed: SeedDemand) -> DistilledRecord:
                seed = sanitize_security_seed(original_seed) if task_type == "security" else original_seed
                task_input: dict[str, Any] | None = None
                provenance_extra: dict[str, object] | None = None
                known_benign_defect: str | None = None
                authoritative_output: dict[str, Any] | None = None
                if task_type == "tool_policy":
                    plan_source = self._upstream_record("plan", seed.seed_id)
                    proposals, proposal_manifest = generate_inert_tool_proposals(seed, index)
                    authoritative_output, oracle_manifest = deterministic_tool_policy_oracle(proposals)
                    task_input = {
                        "plan": dict(plan_source["output"]),
                        "tool_proposals": proposals,
                    }
                    provenance_extra = {
                        "source_plan_record_id": str(plan_source["id"]),
                        "tool_proposals": proposal_manifest,
                        "label_oracle": oracle_manifest,
                    }
                elif task_type == "frontend":
                    plan_source = self._upstream_record("plan", seed.seed_id)
                    policy_source = self._upstream_record("tool_policy", seed.seed_id)
                    task_input = {
                        "plan": dict(plan_source["output"]),
                        "tool_policy": dict(policy_source["output"]),
                    }
                    provenance_extra = {
                        "source_plan_record_id": str(plan_source["id"]),
                        "source_tool_policy_record_id": str(policy_source["id"]),
                    }
                elif task_type == "review":
                    source = self._upstream_record("frontend", seed.seed_id)
                    candidate, manifest = mutate_frontend_code(
                        str(source["output"]["code"]),
                        source_record_id=str(source["id"]),
                    )
                    known_benign_defect = manifest.known_benign_defect
                    task_input = {
                        "candidate_code": candidate,
                        "known_benign_defect": known_benign_defect,
                    }
                    provenance_extra = {
                        "source_frontend_record_id": str(source["id"]),
                        "mutation": manifest.to_dict(),
                    }
                elif task_type == "security":
                    source = self._upstream_record("review", seed.seed_id)
                    reviewed_code, authoritative_output, fixture_manifest = build_inert_security_fixture(
                        str(source["output"]["code"]), index
                    )
                    task_input = {"reviewed_code": reviewed_code}
                    provenance_extra = {
                        "source_review_record_id": str(source["id"]),
                        "security_fixture": fixture_manifest,
                        "label_oracle": {
                            "oracle": "anchor-security-fixture-gold-v1",
                            "decision": authoritative_output["decision"],
                            "sha256": fixture_manifest["gold_sha256"],
                        },
                    }
                system, user = task_prompt(
                    task_type,
                    seed,
                    self.sops[task_type],
                    index,
                    task_input=task_input,
                    known_benign_defect=known_benign_defect,
                )
                raw_response = await self._complete(system=system, user=user)
                if task_type == "frontend":
                    payload, extraction = extract_frontend_payload(raw_response)
                    provenance_extra = dict(provenance_extra or {})
                    provenance_extra["payload_extraction"] = extraction
                else:
                    payload = extract_json_object(raw_response)
                if authoritative_output is not None:
                    payload["output"] = authoritative_output
                    decision = str(authoritative_output["decision"])
                    payload["decision_trace"] = [
                        {
                            "check": "deterministic label oracle",
                            "evidence": "The inert fixture or proposal manifest defines the gold class.",
                            "action": f"Emit {decision} without executing or reconstructing payloads.",
                        }
                    ]
                if task_type == "frontend":
                    normalization = _normalize_frontend_payload(payload)
                    provenance_extra["payload_normalization"] = normalization
                validate_safe_payload(task_type, payload)
                if task_type == "frontend":
                    trace_source = _ensure_frontend_public_trace(payload)
                    provenance_extra["decision_trace_source"] = trace_source
                if task_input is not None:
                    validate_safe_payload(
                        task_type,
                        {"input": task_input, "output": payload.get("output", {})},
                    )
                return DistilledRecord.from_teacher_payload(
                    payload=payload,
                    task_type=task_type,
                    seed=seed,
                    sop=self.sops[task_type],
                    teacher_model=self.teacher.model,
                    teacher_base_url=self.teacher.base_url,
                    teacher_protocol=self.teacher.protocol,
                    generation_params=self.teacher.generation_params,
                    template_sha256=template_sha256(task_type),
                    canonical_task_input=task_input,
                    provenance_extra=provenance_extra,
                )

            results = await asyncio.gather(
                *(distill(index, seed) for index, seed in pending),
                return_exceptions=True,
            )
            task_written = 0
            for (_, seed), result in zip(pending, results):
                if isinstance(result, BaseException):
                    errors.append(f"{task_type}:{seed.seed_id}: {type(result).__name__}: {result}")
                    if isinstance(result, RateLimitError):
                        rate_limited = True
                        if result.retry_after_seconds is not None:
                            retry_after_seconds = max(
                                retry_after_seconds or 0.0,
                                result.retry_after_seconds,
                            )
                    if isinstance(result, ClientDeadlineExceeded):
                        client_deadline = True
                    continue
                if store.append(result.to_dict()):
                    task_written += 1
            written[task_type] = task_written
        return PipelineReport(
            requested_seeds=seed_count,
            available_seeds=len(seeds),
            written_by_task=written,
            skipped_by_task=skipped,
            errors=tuple(errors),
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
            client_deadline=client_deadline,
        )

    def _upstream_record(self, task_type: TaskType, seed_id: str) -> dict[str, object]:
        store = JsonlStore(self.output_dir / f"data_{task_type}.jsonl")
        matches = [
            record
            for record in store.records
            if isinstance(record.get("provenance"), dict)
            and str(record["provenance"].get("seed_id", "")) == seed_id
        ]
        if not matches:
            raise UpstreamDependencyError(
                f"{task_type} record for seed {seed_id} is required before downstream distillation"
            )
        if len(matches) > 1:
            raise UpstreamDependencyError(
                f"multiple {task_type} records found for seed {seed_id}; dependency is ambiguous"
            )
        record = matches[0]
        output = record.get("output")
        if not isinstance(output, dict):
            raise UpstreamDependencyError(
                f"{task_type} record for seed {seed_id} has no successful output"
            )
        if task_type in ("frontend", "review") and not isinstance(output.get("code"), str):
            raise UpstreamDependencyError(
                f"{task_type} record for seed {seed_id} has no successful output.code"
            )
        if task_type == "plan" and (
            not isinstance(output.get("summary"), str)
            or not isinstance(output.get("steps"), list)
        ):
            raise UpstreamDependencyError(
                f"plan record for seed {seed_id} has no validated plan output"
            )
        if task_type == "tool_policy" and output.get("decision") not in (
            "APPROVE",
            "BLOCK",
            "ESCALATE",
        ):
            raise UpstreamDependencyError(
                f"tool_policy record for seed {seed_id} has no validated advisory decision"
            )
        return record
