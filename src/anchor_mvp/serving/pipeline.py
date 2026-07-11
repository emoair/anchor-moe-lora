from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from ..review_contract import (
    REVIEW_VERDICT_SCHEMA_VERSION,
    ReviewVerdict,
    parse_review_verdict,
    revision_issues_json,
)
from .types import CompletionBackend, CompletionRequest, Message, TokenUsage


class StageStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class AdapterSelection:
    base: str
    frontend: str
    review: str
    security: str
    planner: str | None = None
    tool_policy: str | None = None
    mixed: str | None = None
    review_verdict: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    adapters: AdapterSelection
    timeout_seconds: float = 120.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.1
    max_tokens_per_stage: int = 1024
    temperature: float = 0.0
    max_review_cycles: int = 2

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("pipeline timeout and attempts must be positive")
        if self.max_tokens_per_stage < 1 or self.max_review_cycles < 1:
            raise ValueError("pipeline token and review-cycle limits must be positive")


@dataclass
class StageArtifact:
    stage: str
    model: str
    input_text: str
    output_text: str = ""
    status: StageStatus = StageStatus.FAILED
    attempts: int = 0
    backend_attempts: int = 0
    latency_ms: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None
    cycle: int | None = None
    contract_version: str | None = None


@dataclass
class PipelineResult:
    requirement: str
    decision: str
    final_code: str | None
    artifacts: list[StageArtifact]
    success: bool
    fail_closed: bool = False
    errors: list[str] = field(default_factory=list)
    tool_policy_decision: str | None = None
    deterministic_tool_policy_decision: str | None = None
    review_cycles: int = 0
    review_verdict: str | None = None

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=sum(item.usage.prompt_tokens for item in self.artifacts),
            completion_tokens=sum(item.usage.completion_tokens for item in self.artifacts),
            total_tokens=sum(item.usage.total_tokens for item in self.artifacts),
        )


class PipelineRouter:
    """Application-level task pipeline, not a neural mixture-of-experts layer."""

    def __init__(self, backend: CompletionBackend, config: PipelineConfig) -> None:
        self.backend = backend
        self.config = config

    async def run(self, requirement: str) -> PipelineResult:
        artifacts: list[StageArtifact] = []
        frontend = await self._run_stage(
            "frontend",
            self.config.adapters.frontend,
            requirement,
            "Produce a complete implementation for the website requirement. Return only the code.",
        )
        artifacts.append(frontend)
        if frontend.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, frontend)

        review_input = (
            f"REQUIREMENT:\n{requirement}\n\nCANDIDATE CODE:\n{frontend.output_text}"
        )
        review = await self._run_stage(
            "review",
            self.config.adapters.review,
            review_input,
            "Review and repair the candidate. Return the complete corrected code only.",
        )
        artifacts.append(review)
        if review.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, review)

        security_input = (
            f"REQUIREMENT:\n{requirement}\n\nREVIEWED CODE:\n{review.output_text}"
        )
        security = await self._run_stage(
            "security",
            self.config.adapters.security,
            security_input,
            "Audit intent and code. Return exactly one label [PASS] or [BLOCK], no other text.",
        )
        artifacts.append(security)
        if security.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, security)

        decision = parse_security_decision(security.output_text)
        if decision is None:
            security.status = StageStatus.FAILED
            security.error = "ambiguous security verdict"
            return self._failed_closed(requirement, artifacts, security)
        return PipelineResult(
            requirement=requirement,
            decision=decision,
            final_code=review.output_text if decision == "PASS" else None,
            artifacts=artifacts,
            success=True,
            fail_closed=False,
        )

    async def run_five_stage(
        self,
        requirement: str,
        *,
        tool_proposal_labels: tuple[str, ...],
    ) -> PipelineResult:
        """Run the primary five-stage route with a deterministic policy authority.

        Tool proposals are inert labels only. The model's policy verdict is advisory;
        both it and the local allowlist must approve before the coder is called.
        """

        artifacts: list[StageArtifact] = []
        review_model = self.config.adapters.review_verdict
        if not review_model:
            return PipelineResult(
                requirement=requirement,
                decision="BLOCK",
                final_code=None,
                artifacts=artifacts,
                success=False,
                fail_closed=True,
                errors=["review: verdict-v2 adapter is not configured"],
            )
        planner = await self._run_stage(
            "planner",
            self.config.adapters.planner or self.config.adapters.base,
            requirement,
            "Return a concise public implementation plan with testable deliverables; do not write code.",
        )
        artifacts.append(planner)
        if planner.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, planner)

        labels = "\n".join(tool_proposal_labels) or "[NO_PROPOSALS]"
        policy_input = (
            f"REQUIREMENT:\n{requirement}\n\nPLAN:\n{planner.output_text}\n\n"
            f"INERT TOOL PROPOSALS:\n{labels}"
        )
        policy = await self._run_stage(
            "tool_policy",
            self.config.adapters.tool_policy or self.config.adapters.base,
            policy_input,
            "Classify inert tool labels. Return exactly APPROVE, BLOCK, or ESCALATE. "
            "This verdict never grants runtime authority.",
        )
        artifacts.append(policy)
        if policy.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, policy)
        model_policy = parse_tool_policy_decision(policy.output_text)
        enforced_policy = deterministic_tool_policy(tool_proposal_labels)
        if model_policy != "APPROVE" or enforced_policy != "APPROVE":
            policy.status = StageStatus.FAILED
            policy.error = (
                f"policy stopped route: model={model_policy or 'AMBIGUOUS'}, "
                f"deterministic={enforced_policy}"
            )
            result = self._failed_closed(requirement, artifacts, policy)
            result.tool_policy_decision = model_policy
            result.deterministic_tool_policy_decision = enforced_policy
            return result

        frontend_input = (
            f"REQUIREMENT:\n{requirement}\n\nAPPROVED PUBLIC PLAN:\n{planner.output_text}"
        )
        frontend = await self._run_stage(
            "frontend",
            self.config.adapters.frontend,
            frontend_input,
            "Produce a complete implementation for the website requirement. Return only the code.",
        )
        frontend.cycle = 0
        artifacts.append(frontend)
        if frontend.status is not StageStatus.SUCCEEDED:
            return self._failed_closed(requirement, artifacts, frontend)

        current_code = frontend.output_text
        verdict_history: list[ReviewVerdict] = []
        passed_verdict: ReviewVerdict | None = None
        for cycle in range(1, self.config.max_review_cycles + 1):
            review_input = (
                f"REQUIREMENT:\n{requirement}\n\nCANDIDATE CODE:\n{current_code}\n\n"
                f"REVIEW CYCLE:\n{cycle} of {self.config.max_review_cycles}"
            )
            review = await self._run_stage(
                "review",
                review_model,
                review_input,
                "Return only the public anchor.domain-review-verdict.v2 JSON contract. "
                "Use verdict PASS with an empty issues list, or REVISE with one or more "
                "concise issues containing code, severity, summary, and required_change. "
                "Do not return repaired code, markdown, private reasoning, or extra keys.",
            )
            review.cycle = cycle
            review.contract_version = REVIEW_VERDICT_SCHEMA_VERSION
            artifacts.append(review)
            if review.status is not StageStatus.SUCCEEDED:
                result = self._failed_closed(requirement, artifacts, review)
                result.tool_policy_decision = model_policy
                result.deterministic_tool_policy_decision = enforced_policy
                result.review_cycles = cycle
                return result
            verdict = parse_review_verdict(review.output_text)
            if verdict is None:
                review.status = StageStatus.FAILED
                review.error = "ambiguous or invalid public review verdict"
                result = self._failed_closed(requirement, artifacts, review)
                result.tool_policy_decision = model_policy
                result.deterministic_tool_policy_decision = enforced_policy
                result.review_cycles = cycle
                return result
            verdict_history.append(verdict)
            if verdict.verdict == "PASS":
                passed_verdict = verdict
                break
            if cycle >= self.config.max_review_cycles:
                review.status = StageStatus.FAILED
                review.error = "review cycle limit exhausted without PASS"
                result = self._failed_closed(requirement, artifacts, review)
                result.tool_policy_decision = model_policy
                result.deterministic_tool_policy_decision = enforced_policy
                result.review_cycles = cycle
                result.review_verdict = verdict.verdict
                return result
            revision_input = (
                f"REQUIREMENT:\n{requirement}\n\nCURRENT CODE:\n{current_code}\n\n"
                f"PUBLIC REVIEW ISSUES:\n{revision_issues_json(verdict)}"
            )
            revision = await self._run_stage(
                "frontend",
                self.config.adapters.frontend,
                revision_input,
                "Revise the complete implementation to address every public review issue. "
                "Return only the complete revised code; do not discuss the review.",
            )
            revision.cycle = cycle
            revision.contract_version = REVIEW_VERDICT_SCHEMA_VERSION
            artifacts.append(revision)
            if revision.status is not StageStatus.SUCCEEDED:
                result = self._failed_closed(requirement, artifacts, revision)
                result.tool_policy_decision = model_policy
                result.deterministic_tool_policy_decision = enforced_policy
                result.review_cycles = cycle
                result.review_verdict = verdict.verdict
                return result
            current_code = revision.output_text

        assert passed_verdict is not None
        tool_trace_summary = {
            "proposal_labels": list(tool_proposal_labels),
            "model_policy": model_policy,
            "deterministic_policy": enforced_policy,
            "builder_calls": sum(item.stage == "frontend" for item in artifacts),
            "review_cycles": [
                {
                    "cycle": index,
                    "verdict": verdict.verdict,
                    "issue_codes": [issue.code for issue in verdict.issues],
                }
                for index, verdict in enumerate(verdict_history, start=1)
            ],
        }
        security_input = (
            f"REQUIREMENT:\n{requirement}\n\nFINAL REVIEW-PASSED CODE:\n{current_code}\n\n"
            "PUBLIC TOOL TRACE SUMMARY:\n"
            f"{json.dumps(tool_trace_summary, ensure_ascii=False, sort_keys=True)}"
        )
        security = await self._run_stage(
            "security",
            self.config.adapters.security,
            security_input,
            "Audit intent and code. Return exactly one label [PASS] or [BLOCK], no other text.",
        )
        artifacts.append(security)
        if security.status is not StageStatus.SUCCEEDED:
            result = self._failed_closed(requirement, artifacts, security)
            result.tool_policy_decision = model_policy
            result.deterministic_tool_policy_decision = enforced_policy
            result.review_cycles = len(verdict_history)
            result.review_verdict = passed_verdict.verdict
            return result
        decision = parse_security_decision(security.output_text)
        if decision is None:
            security.status = StageStatus.FAILED
            security.error = "ambiguous security verdict"
            result = self._failed_closed(requirement, artifacts, security)
            result.tool_policy_decision = model_policy
            result.deterministic_tool_policy_decision = enforced_policy
            result.review_cycles = len(verdict_history)
            result.review_verdict = passed_verdict.verdict
            return result
        return PipelineResult(
            requirement=requirement,
            decision=decision,
            final_code=current_code if decision == "PASS" else None,
            artifacts=artifacts,
            success=True,
            fail_closed=False,
            tool_policy_decision=model_policy,
            deterministic_tool_policy_decision=enforced_policy,
            review_cycles=len(verdict_history),
            review_verdict=passed_verdict.verdict,
        )

    async def _run_stage(
        self, stage: str, model: str, input_text: str, system_prompt: str
    ) -> StageArtifact:
        artifact = StageArtifact(stage=stage, model=model, input_text=input_text)
        started = time.perf_counter()
        for attempt in range(1, self.config.max_attempts + 1):
            artifact.attempts = attempt
            try:
                response = await asyncio.wait_for(
                    self.backend.complete(
                        CompletionRequest(
                            model=model,
                            messages=(
                                Message("system", system_prompt),
                                Message("user", input_text),
                            ),
                            max_tokens=self.config.max_tokens_per_stage,
                            temperature=self.config.temperature,
                        )
                    ),
                    timeout=self.config.timeout_seconds,
                )
                artifact.output_text = response.content
                artifact.usage = response.usage
                artifact.status = StageStatus.SUCCEEDED
                artifact.backend_attempts += response.attempts
                artifact.latency_ms = (time.perf_counter() - started) * 1000
                return artifact
            except asyncio.TimeoutError:
                artifact.backend_attempts += 1
                artifact.status = StageStatus.TIMED_OUT
                artifact.error = f"stage exceeded {self.config.timeout_seconds:.3f}s"
            except Exception as exc:
                artifact.backend_attempts += int(getattr(exc, "attempts", 1))
                artifact.status = StageStatus.FAILED
                artifact.error = f"{type(exc).__name__}: {exc}"
            if attempt < self.config.max_attempts:
                await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))
        artifact.latency_ms = (time.perf_counter() - started) * 1000
        return artifact

    @staticmethod
    def _failed_closed(
        requirement: str, artifacts: list[StageArtifact], failed: StageArtifact
    ) -> PipelineResult:
        message = f"{failed.stage}: {failed.error or failed.status.value}"
        return PipelineResult(
            requirement=requirement,
            decision="BLOCK",
            final_code=None,
            artifacts=artifacts,
            success=False,
            fail_closed=True,
            errors=[message],
        )


def parse_security_decision(text: str) -> str | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        decision = str(payload.get("decision", "")).upper()
        return decision if decision in {"PASS", "BLOCK"} else None
    except (json.JSONDecodeError, AttributeError):
        pass
    matches = {item.upper() for item in re.findall(r"\[(PASS|BLOCK)\]", stripped, re.IGNORECASE)}
    return matches.pop() if len(matches) == 1 else None


_APPROVED_TOOL_LABELS = {
    "INERT_TOOL_READ_WORKSPACE",
    "INERT_TOOL_EDIT_SUBMISSION",
    "INERT_TOOL_NPM_BUILD",
    "INERT_TOOL_NPM_TEST",
}
_ESCALATE_TOOL_LABELS = {
    "INERT_TOOL_ADD_DEPENDENCY_APPROVAL",
    "INERT_TOOL_EXTERNAL_ASSET_APPROVAL",
}


def parse_tool_policy_decision(text: str) -> str | None:
    matches = {
        item.upper()
        for item in re.findall(r"\b(APPROVE|BLOCK|ESCALATE)\b", text, re.IGNORECASE)
    }
    return matches.pop() if len(matches) == 1 else None


def deterministic_tool_policy(labels: tuple[str, ...]) -> str:
    proposed = set(labels)
    if not proposed:
        return "BLOCK"
    if proposed <= _APPROVED_TOOL_LABELS:
        return "APPROVE"
    if proposed <= (_APPROVED_TOOL_LABELS | _ESCALATE_TOOL_LABELS) and proposed.intersection(
        _ESCALATE_TOOL_LABELS
    ):
        return "ESCALATE"
    return "BLOCK"
