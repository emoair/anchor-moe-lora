from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum

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
    mixed: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    adapters: AdapterSelection
    timeout_seconds: float = 120.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.1
    max_tokens_per_stage: int = 1024
    temperature: float = 0.0


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


@dataclass
class PipelineResult:
    requirement: str
    decision: str
    final_code: str | None
    artifacts: list[StageArtifact]
    success: bool
    fail_closed: bool = False
    errors: list[str] = field(default_factory=list)

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
