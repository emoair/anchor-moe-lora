"""Asynchronous, resumable seed generation and task distillation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from .cleaning import (
    build_inert_security_fixture,
    contains_active_payload,
    contains_secret_material,
    extract_frontend_payload,
    extract_json_object,
    redact_active_payload_material,
    sanitize_security_seed,
    validate_safe_payload,
)
from .mutator import mutate_frontend_code
from .prompts import seed_prompt, task_prompt, template_sha256
from .proposals import deterministic_tool_policy_oracle, generate_inert_tool_proposals
from .schema import (
    DistilledRecord,
    SeedDemand,
    TASK_TYPES,
    TaskType,
    normalized_text,
    stable_id,
    utc_now,
    validate_output,
)
from .sops import load_sop_directory
from .storage import JsonlStore, SeedStore, completed_seed_ids
from .teacher import (
    ClientDeadlineExceeded,
    ProviderQuotaExhausted,
    RateLimitError,
    Teacher,
    TeacherError,
)
from .task_cards import (
    TaskCardCatalog,
    assignment_for_card,
    assignment_for_seed,
    load_task_card_catalog,
)


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
    provider_quota_exhausted: bool = False


class DistillationPipeline:
    def __init__(
        self,
        *,
        teacher: Teacher,
        sop_dir: str | Path,
        output_dir: str | Path,
        concurrency: int = 8,
        seed_index_offset: int = 0,
        task_card_config: str | Path | None = None,
        progress_callback: Callable[[], Awaitable[None]] | None = None,
        quarantine_invalid_seeds: bool = False,
        quality_feedback: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if seed_index_offset < 0:
            raise ValueError("seed_index_offset cannot be negative")
        self.teacher = teacher
        self.sops = load_sop_directory(sop_dir)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.concurrency = concurrency
        self.seed_index_offset = seed_index_offset
        self.task_cards: TaskCardCatalog = load_task_card_catalog(task_card_config)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.progress_callback = progress_callback
        self.quarantine_invalid_seeds = quarantine_invalid_seeds
        self.quality_feedback = {
            str(seed_id): dict(feedback)
            for seed_id, feedback in (quality_feedback or {}).items()
        }

    async def _complete(self, *, system: str, user: str) -> str:
        async with self.semaphore:
            return await self.teacher.complete(system=system, user=user)

    async def _checkpoint_progress(self) -> None:
        if self.progress_callback is not None:
            await self.progress_callback()

    async def generate_seeds(self, count: int) -> list[SeedDemand]:
        if count < 0:
            raise ValueError("seed count cannot be negative")
        store = SeedStore(self.output_dir / "seeds.jsonl")
        rejection_store = JsonlStore(self.output_dir / "seed_rejections.jsonl")
        seeds = [SeedDemand.from_mapping(item) for item in store.records]
        fingerprints = set(store.request_fingerprints)
        # Persisted global indices make crash/resume and out-of-order async
        # completion independent of JSONL append order. Legacy rows reserve the
        # historical prefix but never receive fabricated nine-axis metadata.
        used_indices = {
            seed.seed_index for seed in seeds if seed.seed_index is not None
        }
        legacy_count = sum(seed.seed_index is None for seed in seeds)
        used_indices.update(
            range(self.seed_index_offset, self.seed_index_offset + legacy_count)
        )
        rejected_indices = {
            int(record["seed_index"])
            for record in rejection_store.records
            if isinstance(record.get("seed_index"), int)
            and int(record["seed_index"]) >= self.seed_index_offset
        }
        used_indices.update(rejected_indices)
        next_index = self.seed_index_offset

        def take_indices(amount: int) -> tuple[int, ...]:
            nonlocal next_index
            selected: list[int] = []
            while len(selected) < amount:
                if next_index not in used_indices:
                    selected.append(next_index)
                    used_indices.add(next_index)
                next_index += 1
            return tuple(selected)

        pending_indices = list(take_indices(max(count - len(seeds), 0)))
        rounds = 0
        while pending_indices and rounds < 5:
            retry_indices: list[int] = []

            async def create(
                index: int,
            ) -> tuple[int, SeedDemand | None, dict[str, Any] | None]:
                template = self.task_cards.template_for_index(index)
                system, user = seed_prompt(index, card=template)
                raw_response = await self._complete(system=system, user=user)
                try:
                    payload = extract_json_object(raw_response)
                    if contains_active_payload(payload):
                        raise ValueError("seed contains active payload material")
                    if contains_secret_material(payload):
                        raise ValueError("seed contains credential-like material")
                    candidate = SeedDemand.from_mapping(payload)
                    assignment = assignment_for_card(
                        template,
                        self.task_cards,
                        seed_index=index,
                        requirement=candidate.request,
                    )
                except ValueError as error:
                    if not self.quarantine_invalid_seeds:
                        raise
                    response_sha256 = sha256(raw_response.encode("utf-8")).hexdigest()
                    rejection = {
                        "id": stable_id("seed-rejection", f"{index}:{response_sha256}"),
                        "schema_version": "anchor.seed-rejection.v1",
                        "seed_index": index,
                        "template_id": template.template_id,
                        "error_class": type(error).__name__,
                        "reason": str(error)[:160],
                        "raw_response_sha256": response_sha256,
                        "content_retained": False,
                        "observed_at": utc_now(),
                    }
                    return index, None, rejection
                return (
                    index,
                    SeedDemand(
                        seed_id=candidate.seed_id,
                        title=candidate.title,
                        request=candidate.request,
                        category=candidate.category,
                        tags=assignment.tags,
                        card_id=assignment.card_id,
                        seed_index=index,
                        template_id=assignment.template_id,
                        source_kind=assignment.source_kind,
                        source_digest=assignment.source_digest,
                    ),
                    None,
                )

            remaining_indices = iter(pending_indices)
            quota_errors: list[ProviderQuotaExhausted] = []
            rate_limits: list[RateLimitError] = []
            deadlines: list[ClientDeadlineExceeded] = []
            terminal_errors: list[Exception] = []
            terminal_seen = asyncio.Event()
            replacement_count = 0

            async def seed_worker() -> None:
                nonlocal replacement_count
                while not terminal_seen.is_set():
                    try:
                        index = next(remaining_indices)
                    except StopIteration:
                        return
                    try:
                        _, seed, rejection = await create(index)
                    except ProviderQuotaExhausted as error:
                        quota_errors.append(error)
                        terminal_seen.set()
                    except RateLimitError as error:
                        rate_limits.append(error)
                        terminal_seen.set()
                    except ClientDeadlineExceeded as error:
                        deadlines.append(error)
                        terminal_seen.set()
                    except (TeacherError, ValueError) as error:
                        # The teacher owns the only transport retry loop.  Once
                        # it exhausts that bounded request-local allowance, or
                        # the returned seed fails schema/safety validation, do
                        # not silently replay the prompt in this outer loop.
                        terminal_errors.append(error)
                        terminal_seen.set()
                    except Exception as error:
                        # Unknown implementation/runtime failures are not a
                        # reason to replay a paid teacher request. Duplicate
                        # successful questions are handled separately below.
                        terminal_errors.append(error)
                        terminal_seen.set()
                    else:
                        if rejection is not None:
                            rejection_store.append(rejection)
                            rejected_indices.add(index)
                            replacement_count += 1
                            await self._checkpoint_progress()
                            continue
                        if seed is None:  # defensive union narrowing
                            terminal_errors.append(
                                RuntimeError("seed result omitted without rejection")
                            )
                            terminal_seen.set()
                            continue
                        fingerprint = stable_id(
                            "request", normalized_text(seed.request)
                        )
                        if fingerprint in fingerprints:
                            retry_indices.append(index)
                            continue
                        try:
                            appended = store.append(seed.to_dict())
                        except BaseException:
                            terminal_seen.set()
                            raise
                        if appended:
                            fingerprints.add(fingerprint)
                            seeds.append(seed)
                            await self._checkpoint_progress()
                        else:
                            retry_indices.append(index)

            worker_tasks = [
                asyncio.create_task(seed_worker())
                for _ in range(min(self.concurrency, len(pending_indices)))
            ]
            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            fatal = next(
                (
                    result
                    for result in worker_results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            if fatal is not None:
                raise fatal

            if quota_errors:
                retry_values = [
                    item.retry_after_seconds
                    for item in quota_errors
                    if item.retry_after_seconds is not None
                ]
                raise ProviderQuotaExhausted(
                    max(retry_values) if retry_values else None
                )
            if rate_limits:
                retry_values = [
                    item.retry_after_seconds
                    for item in rate_limits
                    if item.retry_after_seconds is not None
                ]
                raise RateLimitError(max(retry_values) if retry_values else None)
            if deadlines:
                raise deadlines[0]
            if terminal_errors:
                raise terminal_errors[0]
            pending_indices = sorted(set(retry_indices))
            if replacement_count:
                pending_indices.extend(take_indices(replacement_count))
            rounds += 1
        if len(seeds) < count:
            raise RuntimeError(
                f"teacher produced only {len(seeds)} unique seeds after retries"
            )
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
        planned = frozenset(self.quality_feedback)
        # An explicit quality plan is a scoped override of legacy quarantine.
        excluded = frozenset(str(seed_id) for seed_id in excluded_seed_ids) - planned
        seeds = await self.generate_seeds(seed_count)
        written: dict[str, int] = {}
        skipped: dict[str, int] = {}
        errors: list[str] = []
        rate_limited = False
        retry_after_seconds: float | None = None
        client_deadline = False
        provider_quota_exhausted = False
        terminal_task_index: int | None = None
        for task_index, task_type in enumerate(selected):
            store = JsonlStore(self.output_dir / f"data_{task_type}.jsonl")
            completed = completed_seed_ids(store.records)
            completed_quality_generation: set[str] = set()
            for record in store.records:
                provenance = record.get("provenance")
                retry = (
                    provenance.get("quality_retry")
                    if isinstance(provenance, Mapping)
                    else None
                )
                seed_id = (
                    str(provenance.get("seed_id", ""))
                    if isinstance(provenance, Mapping)
                    else ""
                )
                expected = self.quality_feedback.get(seed_id)
                if (
                    seed_id
                    and isinstance(retry, Mapping)
                    and isinstance(expected, Mapping)
                    and retry.get("generation") == expected.get("generation")
                ):
                    completed_quality_generation.add(seed_id)
            pending = [
                (index, seed)
                for index, seed in enumerate(seeds)
                if (
                    seed.seed_id not in completed
                    if seed.seed_id not in planned
                    else seed.seed_id not in completed_quality_generation
                )
                and seed.seed_id not in excluded
            ]
            skipped[task_type] = len(seeds) - len(pending)

            async def distill(index: int, original_seed: SeedDemand) -> DistilledRecord:
                seed = (
                    sanitize_security_seed(original_seed)
                    if task_type == "security"
                    else original_seed
                )
                canonical_index = (
                    seed.seed_index if seed.seed_index is not None else index
                )
                quality_retry = self.quality_feedback.get(seed.seed_id)
                task_input: dict[str, Any] | None = None
                card_assignment = assignment_for_seed(seed, self.task_cards)
                provenance_extra: dict[str, object] = card_assignment.provenance(
                    seed.seed_id
                )
                known_benign_defect: str | None = None
                authoritative_output: dict[str, Any] | None = None
                if task_type == "tool_policy":
                    plan_source = self._upstream_record("plan", seed.seed_id)
                    posture = (
                        card_assignment.axes.get("tool_posture")
                        if card_assignment.axes is not None
                        else None
                    )
                    proposals, proposal_manifest = generate_inert_tool_proposals(
                        seed, canonical_index, variant=posture
                    )
                    authoritative_output, oracle_manifest = (
                        deterministic_tool_policy_oracle(proposals)
                    )
                    task_input = {
                        "plan": dict(plan_source["output"]),
                        "tool_proposals": proposals,
                    }
                    provenance_extra.update(
                        {
                            "source_plan_record_id": str(plan_source["id"]),
                            "tool_proposals": proposal_manifest,
                            "label_oracle": oracle_manifest,
                        }
                    )
                elif task_type == "frontend":
                    plan_source = self._upstream_record("plan", seed.seed_id)
                    policy_source = self._upstream_record("tool_policy", seed.seed_id)
                    task_input = {
                        "plan": dict(plan_source["output"]),
                        "tool_policy": dict(policy_source["output"]),
                    }
                    provenance_extra.update(
                        {
                            "source_plan_record_id": str(plan_source["id"]),
                            "source_tool_policy_record_id": str(policy_source["id"]),
                        }
                    )
                elif task_type == "review":
                    source = self._upstream_record("frontend", seed.seed_id)
                    preferred_rule = (
                        card_assignment.axes.get("review_defect")
                        if card_assignment.axes is not None
                        else None
                    )
                    candidate, manifest = mutate_frontend_code(
                        str(source["output"]["code"]),
                        source_record_id=str(source["id"]),
                        preferred_rule=preferred_rule,
                    )
                    known_benign_defect = manifest.known_benign_defect
                    task_input = {
                        "candidate_code": candidate,
                        "known_benign_defect": known_benign_defect,
                    }
                    provenance_extra.update(
                        {
                            "source_frontend_record_id": str(source["id"]),
                            "mutation": manifest.to_dict(),
                        }
                    )
                elif task_type == "security":
                    source = self._upstream_record("review", seed.seed_id)
                    security_class = (
                        card_assignment.axes.get("security_class")
                        if card_assignment.axes is not None
                        else None
                    )
                    fixture_index = {"benign_boundary": 0, "safe_negative": 1}.get(
                        str(security_class), canonical_index
                    )
                    reviewed_code, authoritative_output, fixture_manifest = (
                        build_inert_security_fixture(
                            str(source["output"]["code"]), fixture_index
                        )
                    )
                    task_input = {"reviewed_code": reviewed_code}
                    provenance_extra.update(
                        {
                            "source_review_record_id": str(source["id"]),
                            "security_fixture": fixture_manifest,
                            "label_oracle": {
                                "oracle": "anchor-security-fixture-gold-v1",
                                "decision": authoritative_output["decision"],
                                "sha256": fixture_manifest["gold_sha256"],
                            },
                        }
                    )
                system, user = task_prompt(
                    task_type,
                    seed,
                    self.sops[task_type],
                    canonical_index,
                    task_input=task_input,
                    known_benign_defect=known_benign_defect,
                    quality_retry=quality_retry,
                )
                raw_response = await self._complete(system=system, user=user)
                # Snapshot request-local route metadata immediately after the
                # completion. CompatibleTeacher stores this in a ContextVar;
                # its shared default route may be changed concurrently by a
                # successful compatibility probe.
                provider_provenance = dict(self.teacher.provider_provenance)
                route_base_url = provider_provenance.get("base_url")
                route_protocol = provider_provenance.get("protocol")
                teacher_base_url = (
                    route_base_url
                    if isinstance(route_base_url, str) and route_base_url
                    else self.teacher.base_url
                )
                teacher_protocol = (
                    route_protocol
                    if isinstance(route_protocol, str) and route_protocol
                    else self.teacher.protocol
                )
                if task_type == "frontend":
                    payload, extraction = extract_frontend_payload(raw_response)
                    provenance_extra = dict(provenance_extra)
                    provenance_extra["payload_extraction"] = extraction
                else:
                    payload = extract_json_object(raw_response)
                if task_type in ("plan", "tool_policy"):
                    sanitized_payload, redaction_count = redact_active_payload_material(
                        payload
                    )
                    if not isinstance(sanitized_payload, dict):
                        raise ValueError(
                            "sanitized teacher payload must remain an object"
                        )
                    payload = sanitized_payload
                    if redaction_count:
                        provenance_extra = dict(provenance_extra or {})
                        provenance_extra["active_payload_redaction"] = {
                            "count": redaction_count,
                            "replacement": "DEFENSIVE_ACTIVE_CONTENT_PLACEHOLDER",
                            "raw_response_sha256": sha256(
                                raw_response.encode("utf-8")
                            ).hexdigest(),
                            "content_retained": False,
                        }
                if authoritative_output is not None:
                    # The local oracle defines the training target, but a valid
                    # teacher disagreement is still useful negative evidence.
                    # Reject malformed/unsafe teacher structures before replacing
                    # the target; never turn a corrupt response into apparent gold.
                    observed_output = payload.get("output")
                    if not isinstance(observed_output, dict):
                        raise ValueError(
                            "teacher classification output must be an object"
                        )
                    validate_output(task_type, observed_output)
                    validate_safe_payload(task_type, payload)
                    provenance_extra = dict(provenance_extra)
                    observed_decision = str(observed_output["decision"])
                    decision = str(authoritative_output["decision"])
                    teacher_agrees = observed_decision == decision
                    provenance_extra.update(
                        {
                            "teacher_observed_decision": observed_decision,
                            "teacher_decision_agrees_with_oracle": teacher_agrees,
                            "supervision_source": "deterministic_oracle",
                            "oracle_normalized": True,
                            "decision_trace_source": (
                                "teacher_agreement"
                                if teacher_agrees
                                else "deterministic_oracle"
                            ),
                        }
                    )
                    payload["output"] = authoritative_output
                    if not teacher_agrees:
                        # Never retain a teacher trace that argues for the
                        # opposite label. The disagreement remains explicit in
                        # provenance while the SFT target is entirely oracle
                        # normalized.
                        payload["decision_trace"] = [
                            {
                                "check": "deterministic label oracle",
                                "evidence": (
                                    "The inert fixture or proposal manifest defines the gold class."
                                ),
                                "action": (
                                    f"Emit {decision} without executing or reconstructing payloads."
                                ),
                            }
                        ]
                if task_type == "frontend":
                    normalization = _normalize_frontend_payload(payload)
                    frontend_provenance: dict[str, object] = provenance_extra
                    frontend_provenance["payload_normalization"] = normalization
                    provenance_extra = frontend_provenance
                validate_safe_payload(task_type, payload)
                if task_type == "frontend":
                    trace_source = _ensure_frontend_public_trace(payload)
                    frontend_provenance = provenance_extra
                    frontend_provenance["decision_trace_source"] = trace_source
                    provenance_extra = frontend_provenance
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
                    teacher_base_url=teacher_base_url,
                    teacher_protocol=teacher_protocol,
                    generation_params=self.teacher.generation_params,
                    provider_provenance=provider_provenance,
                    template_sha256=template_sha256(task_type),
                    canonical_task_input=task_input,
                    provenance_extra=provenance_extra,
                    quality_retry=quality_retry,
                )

            remaining_jobs = iter(enumerate(pending))
            task_written = 0
            task_errors: dict[int, str] = {}
            stage_rate_limits: list[RateLimitError] = []
            stage_deadlines: list[ClientDeadlineExceeded] = []
            terminal_seen = asyncio.Event()

            async def task_worker() -> None:
                nonlocal task_written
                while not terminal_seen.is_set():
                    try:
                        position, (index, seed) = next(remaining_jobs)
                    except StopIteration:
                        return
                    try:
                        result: DistilledRecord | Exception = await distill(index, seed)
                    except Exception as error:
                        result = error
                    if isinstance(result, BaseException):
                        task_errors[position] = (
                            f"{task_type}:{seed.seed_id}: "
                            f"{type(result).__name__}: {result}"
                        )
                        if isinstance(result, RateLimitError):
                            stage_rate_limits.append(result)
                            terminal_seen.set()
                        if isinstance(result, ClientDeadlineExceeded):
                            stage_deadlines.append(result)
                            terminal_seen.set()
                        continue
                    try:
                        appended = store.append(result.to_dict())
                    except BaseException:
                        terminal_seen.set()
                        raise
                    if appended:
                        # Count only durable, deduplicated appends.  Updating as
                        # each future completes makes long batches observable
                        # and crash/resume safe without weakening stage order.
                        task_written += 1
                        await self._checkpoint_progress()

            worker_tasks = [
                asyncio.create_task(task_worker())
                for _ in range(min(self.concurrency, len(pending)))
            ]
            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            fatal = next(
                (
                    result
                    for result in worker_results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            if fatal is not None:
                raise fatal

            if stage_rate_limits:
                rate_limited = True
                provider_quota_exhausted = provider_quota_exhausted or any(
                    isinstance(error, ProviderQuotaExhausted)
                    for error in stage_rate_limits
                )
                retry_values = [
                    error.retry_after_seconds
                    for error in stage_rate_limits
                    if error.retry_after_seconds is not None
                ]
                if retry_values:
                    retry_after_seconds = max(retry_after_seconds or 0.0, *retry_values)
            if stage_deadlines:
                client_deadline = True

            errors.extend(task_errors[position] for position in sorted(task_errors))
            written[task_type] = task_written
            if rate_limited or client_deadline:
                terminal_task_index = task_index
                break

        if terminal_task_index is not None:
            # Keep the report shape stable while making it explicit that no
            # downstream stage was attempted after the terminal signal.
            for task_type in selected[terminal_task_index + 1 :]:
                written[task_type] = 0
                skipped[task_type] = 0
        return PipelineReport(
            requested_seeds=seed_count,
            available_seeds=len(seeds),
            written_by_task=written,
            skipped_by_task=skipped,
            errors=tuple(errors),
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
            client_deadline=client_deadline,
            provider_quota_exhausted=provider_quota_exhausted,
        )

    def _upstream_record(self, task_type: TaskType, seed_id: str) -> dict[str, Any]:
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
        if task_type in ("frontend", "review") and not isinstance(
            output.get("code"), str
        ):
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
